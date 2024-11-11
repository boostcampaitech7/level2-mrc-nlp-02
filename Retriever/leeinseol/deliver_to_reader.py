import torch
import torch.nn.functional as F

from arguments_retriever import (
    TrainingArguments,
    SparseArguments,
    DenseArguments,
)
from DenseRetriever import PretrainedRetriever
from Sparse_retriever import TFIDFRetriever, BM25Retriever
from utils_retriever import setup_logger

def get_retriever(retriever_type, corpus, load_dir) :
    training_args = TrainingArguments
    sparse_args = SparseArguments
    dense_args = DenseArguments    
    training_args.output_dir = load_dir
    logger = setup_logger(training_args.output_dir, "dummy.txt", False, False)
    training_args.do_train = False

    if retriever_type =="TF-IDF" :
        retriever = TFIDFRetriever(sparse_args, logger)
        retriever.fit(corpus, training_args)
    elif retriever_type == "BM-25" :
        retriever = BM25Retriever(sparse_args, logger)
        retriever.fit(corpus, training_args)
    elif retriever_type == "Pre-trained" :
        retriever = PretrainedRetriever(dense_args, logger)
        retriever.fit(corpus, training_args)
    else :
        raise ValueError(f"Unknown '{retriever_type}' retriever type. Setting another retriever_type in <TF-IDF, BM-25, Pre-trained>")

    return retriever

def get_hybrid_retriever(first_retriever_type, second_retriever_type, corpus, load_dir,
                         queries, top_k, first_weight, scaling, batch_size = 4) :
    dense_args = DenseArguments  

    first_retriever = get_retriever(first_retriever_type, corpus, load_dir)
    second_retriever = get_retriever(second_retriever_type, corpus, load_dir)

    if first_retriever_type == "Pre-trained" :
        first_scores, first_indices = first_retriever.retrieve(queries, top_k, batch_size,
                                                               dense_args.task_description, 
                                                               hybrid = True)
    else : 
        first_scores, first_indices = first_retriever.retrieve(queries, top_k, batch_size,
                                                               hybrid = True)
    
    if second_retriever_type == "Pre-trained" :
        second_scores, second_indices = second_retriever.retrieve(queries, top_k, batch_size,
                                                                  dense_args.task_description, 
                                                                  hybrid = True)
    else :
        second_scores, second_indices = second_retriever.retrieve(queries, top_k, batch_size,
                                                                  hybrid = True)

    assert len(first_indices) == len(second_indices), "Two retrievers return different length."
    assert first_indices == second_indices, "Two retrievers return different document ids"

    first_scores, second_scores = torch.tensor(first_scores), torch.tensor(second_scores)
    
    if scaling == "Min-Max" :
        first_scores = (first_scores - first_scores.min()) / (first_scores.max() - first_scores.min())
        second_scores = (second_scores - second_scores.min()) / (second_scores.max() - second_scores.min())
    elif scaling == "L2" :
        first_scores = F.normalize(first_scores, p = 2, dim = 1)
        second_scores = F.normalize(second_scores, p = 2, dim = 1)
    else :
        raise ValueError(f"Unknown {scaling} scaling type. Select in <'Min-Max', 'L2'>")

    second_weight = 1 - first_weight

    final_scores = (first_weight * first_scores) + (second_weight * second_scores)
    final_scores, final_indices = final_scores.topk(top_k, dim = 1)

    final_id = []
    for i in range(final_indices.shape[0]) : 
        final_id.append([first_indices[idx.item()] for idx in final_indices[i].reshape(-1)])
    
    return final_scores.tolist(), final_id
