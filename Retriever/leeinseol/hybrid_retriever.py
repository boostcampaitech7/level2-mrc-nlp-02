import json

import numpy as np
from datasets import load_from_disk
from transformers import HfArgumentParser
import torch
import torch.nn.functional as F

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

    fit_retriever(first_retriever_type, first_retriever, 
                      dataset, corpus, training_args)
    fit_retriever(second_retriever_type, second_retriever,
                    dataset, corpus, training_args)
    
    val_queries = dataset['validation']['question']
    val_context = dataset['validation']['context']
    ground_truth_ids = get_docs_id(corpus, val_context)

    with timer(logger, "Retrieving with two retriever by hibrid") : 
        if first_retriever_type == "Pre-trained" :
            first_scores, first_indices = first_retriever.retrieve(val_queries, retriever_args.top_k, 
                                                 training_args.per_device_eval_batch_size,
                                                 dense_args.task_description, hybrid = True)
        else :
            first_scores, first_indices = first_retriever.retrieve(val_queries, retriever_args.top_k,
                                                 training_args.per_device_eval_batch_size, hybrid = True)            

        if second_retriever_type == "Pre-trained" :
            second_scores, second_indices = second_retriever.retrieve(val_queries, retriever_args.top_k, 
                                                 training_args.per_device_eval_batch_size,
                                                 dense_args.task_description, hybrid = True)
        else :
            second_scores, second_indices = second_retriever.retrieve(val_queries, retriever_args.top_k,
                                                 training_args.per_device_eval_batch_size, hybrid = True)            

        assert len(first_indices) == len(second_indices), "Two retrievers return different length."
        assert first_indices == second_indices, "Two retrievers return different document ids"
        first_scores, second_scores = torch.tensor(first_scores), torch.tensor(second_scores)

    # Check
    first_min, first_max = first_scores.min(), first_scores.max()
    second_min, second_max = second_scores.min(), second_scores.max() 
    logger.info(f"First min/max : {first_min}, {first_max}")
    logger.info(f"Second min/max : {second_min}, {second_max}")

    # Min-Max Scaling
    first_scores_min_max = (first_scores - first_min) / (first_max - first_min)
    second_scores_min_max = (second_scores - second_min) / (second_max - second_min)
    logger.info(f"After min_max scaling First min/max : {first_scores_min_max.min()}, {first_scores_min_max.max()}")
    logger.info(f"After min_max scaling Second min/max : {second_scores_min_max.min()}, {second_scores_min_max.max()}")

    # F.normalize
    first_scores_normalize = F.normalize(first_scores, p = 2, dim = 1)
    second_scores_normalize = F.normalize(second_scores, p = 2, dim = 1)
    logger.info(f"After normalize scaling First min/max : {first_scores_normalize.min()}, {first_scores_normalize.max()}")
    logger.info(f"After normalize scaling Second min/max : {second_scores_normalize.min()}, {second_scores_normalize.max()}")

    weight = 0.45
    while weight < 0.66 : 
        first_weight = np.round(weight, 2)
        second_weight = 1 - first_weight
        weight += 0.01

        # Min-max
        final_scores_min_max = (first_weight * first_scores_min_max) + (second_weight * second_scores_min_max)
        final_scores_min_max, final_indices_min_max = final_scores_min_max.topk(retriever_args.top_k, dim = 1)

        ids_min_max = []
        for i in range(final_indices_min_max.shape[0]) : 
            ids_min_max.append([first_indices[idx.item()] for idx in final_indices_min_max[i].reshape(-1)])

        logger.info(f"Hybird retriever-min_max : {first_weight}, {second_weight}")
        for k in metric_args.recall_k : 
            recalls = recall_k(ground_truth_ids, ids_min_max, k)
            logger.info(f"Recall_{k} - Mean : {np.mean(recalls):.4f}, Std : {np.std(recalls):.4f}")

        for k in metric_args.mrr_k : 
            mrrs = mrr_k(ground_truth_ids, ids_min_max, k)
            logger.info(f"MRR_{k} - Mean : {np.mean(mrrs):.4f}, Std : {np.std(mrrs):.4f}")
        

        # F.normalize
        final_scores_normalize = (first_weight * first_scores_normalize) + (second_weight * second_scores_normalize)
        final_scores_normalize, final_indices_normalize = final_scores_normalize.topk(retriever_args.top_k, dim = 1)

        ids_normalize = []
        for i in range(final_scores_normalize.shape[0]) : 
            ids_normalize.append([first_indices[idx.item()] for idx in final_indices_normalize[i].reshape(-1)])

        logger.info(f"Hybird retriever-normalize : {first_weight}, {second_weight}")
        for k in metric_args.recall_k : 
            recalls = recall_k(ground_truth_ids, ids_normalize, k)
            logger.info(f"Recall_{k} - Mean : {np.mean(recalls):.4f}, Std : {np.std(recalls):.4f}")

        for k in metric_args.mrr_k : 
            mrrs = mrr_k(ground_truth_ids, ids_normalize, k)
            logger.info(f"MRR_{k} - Mean : {np.mean(mrrs):.4f}, Std : {np.std(mrrs):.4f}")


if __name__ == "__main__":
    main()