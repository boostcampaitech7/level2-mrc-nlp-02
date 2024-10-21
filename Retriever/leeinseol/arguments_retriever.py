from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List


@dataclass
class RetrieverArguments :
    retriever_type : str = field(
        default= "Dense",  # "TF-IDF", "BM-25", # "Pre-trained"
        metadata= {
            "help": ""
        }
    )
    top_k : int = field(
        default= 100, 
        metadata= {
            "help" : ""
        }
    )
    first_retriever_type : str = field(
        default="BM-25", 
        metadata={
            "help" : ""
        }
    )
    second_retriever_type : str = field(
        default="Pre-trained",
        metadata= {
            "help" : ""
        }
    )
    first_k : int = field(
        default=25,
        metadata={
            "help" : ""
        }
    )
    second_k : int = field(
        default= 5,
        metadata= {
            "help" : ""
        }
    )
    first_task_description : str = field(
        default= "",
        metadata= {
            "help" : ""
        }
    )
    second_task_description : str = field(
        default= 'Given a web search query, retrieve relevant passages that answer the query',
        metadata={
            "help" : ""
        }
    )



@dataclass
class DataArguments :
    train_data_path : str = field(
        default= "../../data/no_duplicate/dataset/train_dataset", # Naver_KorQuADv1
        metadata= {
            "help" : ""
        }
    )
    test_data_path : str = field(
        default= "../../data/no_duplicate/dataset/test_dataset",
        metadata= {
            "help" : ""
        }
    )
    all_context_path : str = field(
        default= "../../data/no_duplicate/dataset/wikipedia_documents.json",
        metadata= {
            "help" : ""
        }
    )


@dataclass
class MetricArguments :
    recall_k : List[int] = field(
        default_factory= lambda: [1,3,5,10,25], # ,10,25
        metadata={
            "help" : ""
        }
    )
    mrr_k : List[int] = field(
        default_factory= lambda: [1,3,5,10,25], #,10,25
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
    do_train : bool = field(
        default= True,
        metadata={
            "help" : ""
        }
    )
    use_k_fold : bool = field(
        default= False,
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
        default= 32,
        metadata={
            "help" : ""
        }
    )
    per_device_eval_batch_size : int = field(
        default= 32,
        metadata={
            "help" : ""
        }
    )
    gradient_accumulation_steps : int = field(
        default= 4,
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
    weight_decay : float = field(
        default= 0.01,
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
    num_train_epochs : int = field(
        default= 5,
        metadata= {
            "help" : ""
        }
    )
    logging_step : int = field(
        default= 50,
        metadata= {
            "help" : ""
        }
    )
    # for only Sparse Retriever
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
class DenseArguments :
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
    # For using pre-trained only
    pre_trained_model_name_or_path : str = field(
        default = "intfloat/multilingual-e5-large-instruct",
        metadata={
            "help" : ""
        }
    )
    pre_trained_split : bool = field(
        default= True,
        metadata={
            "help" : ""
        }
    )
    task_description : str = field(
        default= 'Given a web search query, retrieve relevant passages that answer the query', # ""
        metadata={
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
class QueryEncoderArguments :
    query_type : str = field(
        default= "Bert",
        metadata={
            "help" : ""
        }
    )
    query_model_name_or_path : str = field(
        default= 'bert-base-multilingual-cased',
        metadata={
            "help" : ""
        }
    )


@dataclass
class PassageEncoderArguments :
    passage_type : str = field(
        default="Bert",
        metadata={
            "help" : ""
        }
    )
    passage_model_name_or_path : str = field(
        default='bert-base-multilingual-cased',
        metadata={
            "help" : ""
        }
    )
    passage_stride : int = field(
        default= 64,
        metadata= {
            "help" : ""
        }
    )