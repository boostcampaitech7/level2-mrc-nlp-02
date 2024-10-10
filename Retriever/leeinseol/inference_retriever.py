import os 
import numpy as np
import json
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments
)

from arguments_retriever import RetrieverArguments, DataTrainingArguments_Retriever
from DenseRetriever import Retriever
from utils_retriever import setup_logger, set_seed

seed = 2024
set_seed(seed=seed, deterministic=False)

def main() :
    parser = HfArgumentParser(
        (RetrieverArguments, DataTrainingArguments_Retriever, TrainingArguments)
    )

    retriever_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logger = setup_logger(training_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        retriever_args.tokenizer_name
        if retriever_args.tokenizer_name is not None
        else retriever_args.model_name_or_path
    ) 

    retriever = Retriever(retriever_args, data_args, tokenizer, logger)
    best_model_path = os.path.join(training_args.output_dir, "best_model.pth")
    retriever.load_model(best_model_path)
    retriever.load_embedding(training_args.output_dir)

    dataset = load_from_disk(data_args.train_dataset_path)
    _, val_dataset = dataset['train'], dataset['validation']
    sample_idx = np.random.choice(range(len(val_dataset)), data_args.num_sample)
    val_dataset = val_dataset[sample_idx]
    query = val_dataset['question']
    ground_truth = val_dataset['context']
    
    retriever.build_faiss()
    scores, indices = retriever.retrieve(query, data_args.top_k_retrieval)
    print(indices)
    with open(data_args.all_context_path, 'r') as file :
        all_context = json.load(file)
    
    for i in range(data_args.num_sample) :
        print(f"{i}'s Query : {query[i]}")
        print(f"{i}'s ground truth : {ground_truth[i]}")
        for j in range(data_args.top_k_retrieval) :
            print(f"Top-{j+1} passage with score {scores[i][j]:.4f}")
            print(all_context[indices[i][j]]['context'])

if __name__ == "__main__":
    main()