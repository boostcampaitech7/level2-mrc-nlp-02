import logging
import os
import sys
import random
import numpy as np
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from typing import NoReturn

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_metric


from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed,
)

from Reader.train.trainers import QuestionAnsweringTrainer
from Reader.utils.utils_qa import check_no_error, postprocess_qa_predictions
from Reader.utils.run_mrc import run_mrc


seed = 2024
deterministic = False

random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)
if deterministic: # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다. 
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    ############## hyperparameter tuning #################
    model_args = ModelArguments()
    data_args = DataTrainingArguments()
    training_args = TrainingArguments(
        output_dir="../logs/klue_bert/",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        gradient_accumulation_steps=1,
        do_train = True,
        do_eval = True,
        do_predict = False,
    )

    print(model_args.model_name_or_path)

    """
    training_args.output_dir="./results"  # 모델 출력 디렉토리

    training_args.num_train_epochs = 3  # 학습 에폭 수
    training_args.per_device_train_batch_size = 8  # 디바이스당 학습 배치 크기
    training_args.per_device_eval_batch_size = 16  # 디바이스당 평가 배치 크기
    training_args.learning_rate = 2e-5  # 학습률

    training_args.warmup_ratio = 0.1  # 웜업 스텝 수
    training_args.weight_decay = 0.01  # 가중치 감쇠
    training_args.logging_dir = "./logs"  # 로깅 디렉토리
    training_args.logging_steps = 10  # 로깅 간격
    training_args.eval_strategy = "steps"  # 평가 전략
    training_args.eval_steps = 500  # 평가 간격
    training_args.save_steps = 1000  # 모델 저장 간격
    training_args.gradient_accumulation_steps = 1 # gradient_acc (batch 늘리기)

    training_args.do_train = True # train을 할 것인지 설정
    training_args.do_eval = True # eval을 할 것인지 설정
    training_args.do_precit = False # inference 시에 활용
    """
    ####################################################### 

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


if __name__ == "__main__":
    main()
