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
)
from DenseRetriever import DenseRetriever
from Sparse_retriever import TFIDFRetriever, BM25Retriever
from metrics_retriever import mrr_k, recall_k
from utils_retriever import get_docs_id, set_seed, setup_logger


def main() :
    parser = HfArgumentParser(
        (RetrieverArguments, DataArguments, MetricArguments, TrainingArguments,
         SparseArguments, DenseArguments)
    )
    retriever_args, data_args, metric_args, training_args, sparse_args, dense_args = parser.parse_args_into_dataclasses()

    set_seed(seed=training_args.seed, deterministic=False)

    logger = setup_logger(training_args.output_dir)
    logger.info(f"RetrieverArguments: \n{retriever_args}")
    logger.info(f"DataArguments: \n{data_args}")
    logger.info(f"MetricArguments: \n{metric_args}")
    logger.info(f"TrainingArguments: \n{training_args}")

    dataset = load_from_disk(data_args.train_data_path)
    with open(data_args.all_context_path, "r") as file :
        corpus = json.load(file)
    
    # 깔끔한 코드는 나중에 정리...
    retriever_type = retriever_args.retriever_type
    if retriever_type == "TF-IDF" : 
        logger.info(f"SparseArguments: \n{sparse_args}")
        retriever = TFIDFRetriever(sparse_args, logger)
        retriever.fit(corpus, training_args)
    
    elif retriever_type == "BM-25" :
        logger.info(f"SparseArguments: \n{sparse_args}")
        retriever = BM25Retriever(sparse_args, logger)
        retriever.fit(corpus, training_args)
    
    elif retriever_type == "Dense" :
        logger.info(f"DenseArguments: \n{dense_args}")
        retriever = DenseRetriever(dense_args, logger)
        retriever.fit(dataset, corpus, training_args)
    
    else :
        raise ValueError(f"Unknown '{retriever_type}' retriever type. Setting another Retriever.")
    
    val_queries = dataset['validation']['question']
    val_context = dataset['validation']['context']

    scores, indices = retriever.retrieve(val_queries, retriever_args.top_k, training_args.per_device_eval_batch_size)
    # scores : 2-d List, (num_query, top_k)
    # indices : 2-d List, (num_query, top_k)

    ground_truth_ids = get_docs_id(corpus, val_context)
    # 1-d List (num_query,)
    recalls = recall_k(ground_truth_ids, indices, metric_args.recall_k)
    mrrs = mrr_k(ground_truth_ids, indices, metric_args.mrr_k)

    logger.info(f"Recall_{metric_args.recall_k} - Mean : {np.mean(recalls):.4f}, Std : {np.std(recalls):.4f}")
    logger.info(f"MRR_{metric_args.mrr_k} - Mean : {np.mean(mrrs):.4f}, Std : {np.std(mrrs):.4f}")


if __name__ == "__main__":
    main()