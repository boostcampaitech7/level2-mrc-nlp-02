"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""
import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from retrieval import SparseRetrieval
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed,
)

from Reader.utils.run_mrc import run_mrc
from Reader.utils.utils_qa import check_no_error, postprocess_qa_predictions
from Reader.train.trainers import QuestionAnsweringTrainer

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    model_args = ModelArguments()
    data_args = DataTrainingArguments()
    training_args = TrainingArguments(
        output_dir="./result",
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        do_train = False,
        do_eval = True,
        do_predict = False
    )

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.infer_dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)
    datasets = load_from_disk(data_args.infer_dataset_name)
    print(datasets)

    ########### Load Saved Reader ########################
    model_path = "./Reader/logs/model_name_settings"
    config = AutoConfig.from_pretrained(
        model_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    ######################################################

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize, datasets, training_args, data_args,
        )


    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


if __name__ == "__main__":
    main()
