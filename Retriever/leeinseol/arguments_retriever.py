from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RetrieverArguments:
    ## 리트리버 관련 인자들
    project_name: str = field(
        default="Level2_MRC_Retriever",
        metadata={
            "help": "Setting wandb project name."  
            # wandb 프로젝트 이름을 설정합니다.
        },
    )
    encoder_name: str = field(
        default="Bert",
        metadata={
            "help" : "Choice which encoder use." 
        }
    )
    model_name_or_path: str = field(
        default="klue/bert-base", # "bert-base-multilingual-cased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"  
            # 사전 학습된 모델의 경로 또는 모델 식별자를 지정합니다.
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"  
            # 토크나이저의 경로 또는 이름을 지정합니다. (model_name과 다를 경우)
        },
    )
    use_faiss: bool = field(
        default=True,
        metadata={
            "help": "Flag whether use faiss or not"  
            # faiss 사용 여부를 지정하는 플래그입니다.
        }
    )
    num_clusters: int = field(
        default=16,
        metadata={
            "help": "The number of clusters using in faiss."  
            # faiss 클러스터링에 사용할 클러스터 수를 지정합니다.
        }
    )
    n_iter: int = field(
        default=25,
        metadata={
            "help": "Number of iterations for clustering with faiss."  
            # faiss 클러스터링 과정에서 반복할 횟수를 지정합니다.
        }
    )


@dataclass 
class DataTrainingArguments_Retriever:
    # 데이터 로드 및 전처리 관련 인자들
    n_splits: int = field(
        default=5,
        metadata={
            "help": "Decide how many folds to split into for cross validation."  
            # cross validation을 위해 데이터를 몇 개의 폴드로 분할할지 설정합니다.
        }
    )
    train_dataset_path: str = field(
        default="../../data/train_dataset",
        metadata={
            "help": "Path to the dataset to be used for training and evaluation."  
            # 학습 및 평가에 사용할 데이터셋의 경로를 지정합니다.
        }
    )
    test_dataset_path: str = field(
        default="../../data/test_dataset",
        metadata={
            "help": "Path to the dataset to be used for testing."
        }
    )
    all_context_path: str = field(
        default="../../data/all_context.json",
        metadata={
            "help" : "Path to the all context path to be used for ranking."
        }
    )
    padding: str = field(
        default="max_length",
        metadata={
            "help": "Padding strategy to use. Choices are 'max_length' or 'longest'."  
            # 패딩 전략을 지정합니다. ('max_length' 또는 'longest' 중 선택)
        }
    )
    truncation: bool = field(
        default=True,
        metadata={
            "help": "Whether to truncate sequences longer than the max sequence length."  
            # 최대 시퀀스 길이를 초과하는 경우 시퀀스를 잘라내는지 여부를 지정합니다.
        }
    )
    return_tensors: str = field(
        default="pt",
        metadata={
            "help": "Format of the returned tensors. Choices are 'pt' (PyTorch) or 'tf' (TensorFlow)."  
            # 반환된 텐서 형식을 지정합니다. ('pt'는 PyTorch, 'tf'는 TensorFlow)
        }
    )
    top_k_retrieval: int = field(
        default=5,
        metadata={
            "help": "Number of top-k passages to retrieve based on similarity."  
            # 유사도에 따라 검색할 상위 패시지 개수를 지정합니다.
        }
    )
    num_sample: int = field(
        default = 5,
        metadata={
            "help" : "Number of validation dataset selected for testing."
        }
    )


    

