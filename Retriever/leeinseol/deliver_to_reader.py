import os
import warnings
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments
)
from arguments_retriever import RetrieverArguments, DataTrainingArguments_Retriever
from DenseRetriever import Retriever

warnings.filterwarnings("ignore")

def get_retriever(use_faiss, model_saved_dir, emb_saved_dir, model_file_name = "best_model.pth", emb_file_name = "embeddings.pt") :
    parser = HfArgumentParser(
        (RetrieverArguments, DataTrainingArguments_Retriever)
        )
    retriever_args, data_args = parser.parse_args_into_dataclasses()
    retriever_args.use_faiss = use_faiss

    tokenizer = AutoTokenizer.from_pretrained(
        retriever_args.tokenizer_name
        if retriever_args.tokenizer_name is not None
        else retriever_args.model_name_or_path
        ) 

    retriever = Retriever(retriever_args, data_args, tokenizer, logger = None)
    model_path = os.path.join(model_saved_dir, model_file_name)
    retriever.load_model(model_path)
    retriever.load_embedding(emb_saved_dir, emb_file_name)
    retriever.build_faiss()

    return retriever