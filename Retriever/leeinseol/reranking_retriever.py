import json

import numpy as np
from datasets import load_from_disk
from transformers import HfArgumentParser

from arguments_retriever import (
    RetrieverArguments,
    DataArguments,
    MetricArguments,
    TrainingArguments,
    SparseArguments,
    DenseArguments,
    QueryEncoderArguments,
    PassageEncoderArguments
)
from DenseRetriever import DenseRetriever, PretrainedRetriever
from Sparse_retriever import TFIDFRetriever, BM25Retriever
from metrics_retriever import mrr_k, recall_k
from utils_retriever import get_docs_id, set_seed, setup_logger, timer

def get_retriever(retriever_type, logger, sparse_args, dense_args, query_args, passage_args) :
    if retriever_type == "TF-IDF" : 
        logger.info(f"SparseArguments: \n{sparse_args}")
        retriever = TFIDFRetriever(sparse_args, logger)
    
    elif retriever_type == "BM-25" :
        logger.info(f"SparseArguments: \n{sparse_args}")
        retriever = BM25Retriever(sparse_args, logger)
    
    elif retriever_type == "Dense" :
        logger.info(f"DenseArguments: \n{dense_args}")
        logger.info(f"QueryArguments: \n{query_args}")
        logger.info(f"PassageArguments: \n{passage_args}")
        retriever = DenseRetriever(dense_args, query_args, passage_args, logger)
    
    elif retriever_type == "Pre-trained" :
        logger.info(f"DenseArguments: \n{dense_args}")
        retriever = PretrainedRetriever(dense_args, logger)
    
    else :
        raise ValueError(f"Unknown '{retriever_type}' retriever type. Setting another Retriever.")
    
    return retriever

def fit_retriever(retriever_type, retriever, dataset, corpus, trainig_args) :
    if retriever_type == "Dense" :
        retriever.fit(dataset, corpus, trainig_args)
    else :
        retriever.fit(corpus, trainig_args)

def main() :
    parser = HfArgumentParser(
        (RetrieverArguments, DataArguments, MetricArguments, TrainingArguments,
         SparseArguments, DenseArguments, QueryEncoderArguments, PassageEncoderArguments)
    )
    args = parser.parse_args_into_dataclasses()
    retriever_args, data_args, metric_args, training_args = args[:4]
    sparse_args, dense_args, query_args, passage_args = args[4:]

    set_seed(seed=training_args.seed, deterministic=False)

    logger = setup_logger(training_args.output_dir)
    logger.info(f"RetrieverArguments: \n{retriever_args}")
    logger.info(f"DataArguments: \n{data_args}")
    logger.info(f"MetricArguments: \n{metric_args}")
    logger.info(f"TrainingArguments: \n{training_args}")

    dataset = load_from_disk(data_args.train_data_path)
    with open(data_args.all_context_path, "r") as file :
        corpus = json.load(file)
    
    first_retriever_type = retriever_args.first_retriever_type
    second_retriever_type = retriever_args.second_retriever_type

    first_retriever = get_retriever(first_retriever_type, logger, 
                                    sparse_args, dense_args, 
                                    query_args, passage_args)
    second_retriever = get_retriever(second_retriever_type, logger, 
                                     sparse_args, dense_args, 
                                     query_args, passage_args)

    if first_retriever_type != "Pre-trained" :
        fit_retriever(first_retriever_type, first_retriever, 
                      dataset, corpus, training_args)
    if second_retriever_type != "Pre-trained" : 
        fit_retriever(second_retriever_type, second_retriever,
                    dataset, corpus, training_args)
    
    val_queries = dataset['validation']['question']
    val_context = dataset['validation']['context']
    ground_truth_ids = get_docs_id(corpus, val_context)

    with timer(logger, "Retrieving with two retriever by reranking") : 
        if retriever_args.first_task_description :
            first_scores, first_indices = first_retriever.retrieve(val_queries, retriever_args.first_k,
                                                                training_args.per_device_eval_batch_size,
                                                                retriever_args.first_task_description)
        else :
            first_scores, first_indices = first_retriever.retrieve(val_queries, retriever_args.first_k,
                                                                training_args.per_device_eval_batch_size)

        first_contexts = []
        for i in range(len(first_indices)) :
            first_contexts.append([corpus[key] for key in first_indices[i]])

        if retriever_args.second_task_description :
            second_scores , second_indices = second_retriever.reranking(first_indices, first_contexts, val_queries,
                                                                        retriever_args.second_k, 
                                                                        retriever_args.second_task_description)
        else :
            second_scores , second_indices = second_retriever.reranking(first_indices, first_contexts, val_queries,
                                                                        retriever_args.second_k)
    
    for k in metric_args.recall_k : 
        recalls = recall_k(ground_truth_ids, first_indices, k)
        logger.info(f"Recall_{k} - Mean : {np.mean(recalls):.4f}, Std : {np.std(recalls):.4f}")

    for k in metric_args.mrr_k : 
        mrrs = mrr_k(ground_truth_ids, first_indices, k)
        logger.info(f"MRR_{k} - Mean : {np.mean(mrrs):.4f}, Std : {np.std(mrrs):.4f}")

    
    for k in metric_args.recall_k : 
        recalls = recall_k(ground_truth_ids, second_indices, k)
        logger.info(f"Recall_{k} - Mean : {np.mean(recalls):.4f}, Std : {np.std(recalls):.4f}")

    for k in metric_args.mrr_k : 
        mrrs = mrr_k(ground_truth_ids, second_indices, k)
        logger.info(f"MRR_{k} - Mean : {np.mean(mrrs):.4f}, Std : {np.std(mrrs):.4f}")
    
if __name__ == "__main__":
    main()