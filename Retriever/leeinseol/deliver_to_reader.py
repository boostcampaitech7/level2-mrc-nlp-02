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

def get_retriever(retriever_type, corpus, load_dir) :
    training_args = TrainingArguments
    sparse_args = SparseArguments
    dense_args = DenseArguments    
    training_args.output_dir = load_dir
    logger = setup_logger(training_args.output_dir, "dummy.txt")
    training_args.do_train = False

    if retriever_type =="TF-IDF" :
        logger.info(f"TF-IDF arguments: \n{sparse_args}")
        retriever = TFIDFRetriever(sparse_args, logger)
        retriever.fit(corpus, training_args)
    elif retriever_type == "BM-25" :
        logger.info(f"BM-25 arguments: \n{sparse_args}")
        retriever = BM25Retriever(sparse_args, logger)
        retriever.fit(corpus, training_args)
    # elif retriever_type == "Dense" :
    #     logger.info(f"Dense arguments: \n{dense_args}")
    #     logger.info(f"Query arguments: \n{query_args}")
    #     logger.info(f"Passage arguments: \n{passage_args}")
    #     retriever = DenseRetriever(dense_args, query_args, passage_args, logger)
    #     retriever.fit(dataset, corpus, training_args)
    elif retriever_type == "Pre-trained" :
        logger.info(f"Pre-trained arguments: \n{dense_args}")
        retriever = PretrainedRetriever(dense_args, logger)
        retriever.fit(corpus, training_args)
    else :
        raise ValueError(f"Unknown '{retriever_type}' retriever type. Setting another retriever_type in <TF-IDF, BM-25, Pre-trained>")

    return retriever