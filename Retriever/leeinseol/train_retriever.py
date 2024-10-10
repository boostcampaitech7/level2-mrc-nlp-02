import os
import wandb
import json
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_metric
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments
)

from arguments_retriever import RetrieverArguments, DataTrainingArguments_Retriever
from utils_retriever import set_seed, setup_logger, timer
from dataset_retriever import get_dataset_to_tensor, get_all_context_to_tensor
from DenseRetriever import Retriever

seed = 2024
set_seed(seed=seed, deterministic=False)

def main() :
    parser = HfArgumentParser(
        (RetrieverArguments, DataTrainingArguments_Retriever, TrainingArguments)
    )

    retriever_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # output_dir -> cli
    training_args.logging_steps = 10
    training_args.num_train_epochs = 10
    training_args.per_device_train_batch_size = 16
    training_args.per_device_eval_batch_size = 16
    training_args.learning_rate = 2e-5
    training_args.warmup_steps = 250
    training_args.weight_decay = 0.01

    logger = setup_logger(training_args.output_dir)
    logger.info(f"RetrieverArguments {retriever_args}")
    logger.info(f"DataTrainingArguments {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # WandB 프로젝트 시작
    wandb.init(project=retriever_args.project_name,
               config={"k_folds": data_args.n_splits, 
                       "epochs": training_args.num_train_epochs})

    # dataset, tokenizer 로드 
    tokenizer = AutoTokenizer.from_pretrained(
        retriever_args.tokenizer_name
        if retriever_args.tokenizer_name is not None
        else retriever_args.model_name_or_path
    )  
    dataset = load_from_disk(data_args.train_dataset_path)
    train_dataset, val_dataset = dataset['train'], dataset['validation']

    # k-fold 설정 
    best_val_loss = float('inf')
    best_model_path = os.path.join(training_args.output_dir, "best_model.pth")
    best_fold = 0
    num_rows = train_dataset.num_rows
    indices = list(range(num_rows))
    kf = KFold(n_splits=data_args.n_splits, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)) :
        train_subset = train_dataset.select(train_idx)
        val_subset = train_dataset.select(val_idx)

        train_subset = get_dataset_to_tensor(train_subset, tokenizer, data_args, True)
        val_subset = get_dataset_to_tensor(val_subset, tokenizer, data_args, True)

        subtrain_loader = DataLoader(train_subset,
                                  batch_size=training_args.per_device_train_batch_size,
                                  shuffle=True)
        subval_loader = DataLoader(val_subset,
                                batch_size=training_args.per_device_eval_batch_size)

        retriever = Retriever(retriever_args, data_args, tokenizer, logger)
        
        # 학습 및 평가
        with timer(logger, f"Training fold {fold + 1}") :
            retriever.train(training_args, subtrain_loader, subval_loader, wandb, fold)
        
        val_loss = retriever.evaluate(subval_loader)
        logger.info(f"Fold {fold+1} final validation loss : {val_loss}")

        if val_loss < best_val_loss : 
            best_val_loss = val_loss
            best_fold = fold + 1
            retriever.save_model(best_model_path)
            logger.info(f"New best model saved at fold {fold+1} with validation loss {val_loss}")
    logger.info(f"Best model saved at fold {best_fold} with validation loss {best_val_loss}")
    
    wandb.finish()

    # Best model 로드 및 평가 
    logger.info("Loading best model...")
    retriever.load_model(best_model_path)
    val_dataset_tensor = get_dataset_to_tensor(val_dataset, tokenizer, data_args, True)
    val_dataloader = DataLoader(val_dataset_tensor,
                                batch_size=training_args.per_device_eval_batch_size, shuffle=False)
    final_val_loss = retriever.evaluate(val_dataloader)
    logger.info(f"Best model's validation loss(fold {best_fold}) : {final_val_loss}")


    # 모든 문서에 대한 임베딩 계산 
    context_path = data_args.all_context_path
    with open(context_path, "r") as file :
        contexts = json.load(file)
    contexts_dataset = get_all_context_to_tensor(contexts, tokenizer, data_args)
    contexts_loader = DataLoader(contexts_dataset,
                                 batch_size=training_args.per_device_train_batch_size, 
                                 shuffle=False)    
    logger.info("Calculate embedding...")
    with timer(logger, f"Calculate embedding") :
        retriever.calculate_embedding(contexts_loader)
    retriever.save_embedding(training_args.output_dir)

if __name__ == "__main__":
    main()