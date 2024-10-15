from dataclasses import dataclass, field
from typing import Optional, Tuple, Union


@dataclass
class RetrieverArguments :
    retriever_type : str = field(
        default= "Dense",  # "TF-IDF", "BM-25"
        metadata= {
            "help": ""
        }
    )
    top_k : int = field(
        default= 5, 
        metadata= {
            "help" : ""
        }
    )


@dataclass
class DataArguments :
    train_data_path : str = field(
        default= "../../data/train_dataset",
        metadata= {
            "help" : ""
        }
    )
    test_data_path : str = field(
        default= "../../data/test_dataset",
        metadata= {
            "help" : ""
        }
    )
    all_context_path : str = field(
        default= "../../data/all_context.json",
        metadata= {
            "help" : ""
        }
    )


@dataclass
class MetricArguments :
    recall_k : int = field(
        default= 1,
        metadata={
            "help" : ""
        }
    )
    mrr_k : int = field(
        default= 3,
        metadata={
            "help" : ""
        }
    )
    num_sample : int = field(
        default= 5,
        metadata={
            "help" : ""
        }
    )


@dataclass
class TrainingArguments :
    seed : int = field(
        default= 2024,
        metadata={
            "help" : ""
        }
    )
    project_name : str = field(
        default= "Level2_MRC",
        metadata={
            "help" : ""
        }
    )
    output_dir : str = field(
        default= "results/",
        metadata={
            "help" : ""
        }
    )
    model_file : str = field(
        default= "best_model.pth",
        metadata={
            "help" : ""
        }
    )
    embeddings_file : str = field(
        default= "embeddings.pt",
        metadata={
            "help" : ""
        }
    )
    indexer_file : str = field(
        default= "indexer.index",
        metadata={
            "help" : ""
        }
    )
    # for only TF-IDF
    tfidf_file : str = field(
        default = "tfidf_model.pkl",
        metadata= {
            "help" : ""
        }
    )
    bm25_file : str = field(
        default= "bm25_model.pkl",
        metadata= {
            "help" : ""
        }
    )
    do_train : bool = field(
        default= True,
        metadata={
            "help" : ""
        }
    )
    use_k_fold : bool = field(
        default= True,
        metadata={
            "help" : ""
        }
    )
    n_splits : int = field(
        default= 16,
        metadata={
            "help" : ""
        }
    )
    per_device_train_batch_size : int = field(
        default= 16,
        metadata={
            "help" : ""
        }
    )
    per_device_eval_batch_size : int = field(
        default= 16,
        metadata={
            "help" : ""
        }
    )
    weight_decay : float = field(
        default= 0.01,
        metadata={
            "help" : ""
        }
    )
    learning_rate : float = field(
        default= 2e-5,
        metadata={
            "help" : ""
        }
    )
    adam_epsilon : float = field(
        default= 1e-8,
        metadata={
            "help" : ""
        }
    )
    gradient_accumulation_steps : int = field(
        default= 1,
        metadata={
            "help" : ""
        }
    )
    num_train_epochs : int = field(
        default= 10,
        metadata= {
            "help" : ""
        }
    )
    warmup_steps : int = field(
        default= 250,
        metadata= {
            "help" : ""
        }
    )
    logging_step : int = field(
        default= 10,
        metadata= {
            "help" : ""
        }
    )


@dataclass
class SparseArguments :
    sparse_tokenizer_name : str = field(
        default= "simple", # "huggingface tokenizer"
        metadata={
            "help" : ""
        }
    )
    # TF-IDF
    n_gram : Tuple[int, int] = field(
        default= (1,2),
        metadata={
            "help" : "--ngram 1,2"
        }
    )
    # BM-25
    k1 : float = field(
        default= 1.5,
        metadata={
            "help" : ""
        }
    )
    b : float = field(
        default= 0.75,
        metadata={
            "help" : ""
        }
    )


@dataclass
class EncoderArguments :
    encoder_type : str = field(
        default= "Bert",
        metadata={
            "help" : ""
        }
    )
    model_name_or_path : str = field(
        default= 'klue/bert-base',
        metadata={
            "help" : ""
        }
    )
    embedding_dim : Optional[str] = field(
        default= None,
        metadata={
            "help" : ""
        }
    )
    padding : Union[bool, str] = field(
        default= "max_length", 
        metadata= {
            "help" : ""
        }
    )
    truncation : bool = field(
        default= True,
        metadata= {
            "help" : ""
        }
    )
    max_length : int = field(
        default= 128,
        metadata= {
            "help" : ""
        }
    )
    stride : int = field(
        default= 64,
        metadata= {
            "help" : ""
        }
    )


@dataclass
class DenseArguments :
    query_encoder : EncoderArguments = field(
        default_factory= EncoderArguments,
        metadata={
            "help" : "--query_encoder.padding True "
        }
    )
    passage_encoder : EncoderArguments = field(
        default_factory= EncoderArguments,
        metadata={
            "help" : ""
        }
    )
    use_overflow : bool = field(
        default= False,
        metadata= {
            "help" : ""
        }
    )
    # faiss
    use_faiss : bool = field(
        default= False,
        metadata={
            "help" : ""
        }
    )
    num_clusters : int = field(
        default= 16,
        metadata={
            "help" : ""
        }
    )
    n_iter : int = field(
        default= 25,
        metadata={
            "help" : ""
        }
    )
    indexer_nprobe : int = field(
        default= 3,
        metadata={
            "help" : ""
        }
    )


