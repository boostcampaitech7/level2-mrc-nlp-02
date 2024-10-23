import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, NoReturn, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)
from Retriever.chw.utils.util import timer


class DenseRetrieval:

    def __init__(self, args, dataset, data_path, context_path, num_neg, tokenizer, p_encoder, q_encoder):

        self.args = args
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.dataset = dataset
        self.train_dataset = None
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.prepare_in_batch_negative(num_neg=num_neg)

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer
        # print(self.dataset.keys())
        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset["context"]])))

        p_with_neg = []

        for c in dataset["context"]:

            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset["question"], padding="max_length", truncation=True, return_tensors="pt")
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors="pt")

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg + 1, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg + 1, max_len)

        train_dataset = TensorDataset(p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"])

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        valid_seqs = tokenizer(dataset["context"], padding="max_length", truncation=True, return_tensors="pt")
        passage_dataset = TensorDataset(valid_seqs["input_ids"], valid_seqs["attention_mask"], valid_seqs["token_type_ids"])
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)

    def save_model(self, save_path, optimizer, scheduler):
        model_path = os.path.join(self.data_path, save_path)
        torch.save(
            {
                "p_encoder_state_dict": self.p_encoder.state_dict(),
                "q_encoder_state_dict": self.q_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            model_path,
        )

    def load_model(self, load_path="save_model.pt", args=None):
        model_path = os.path.join(self.data_path, load_path)
        if os.path.isfile(model_path):

            if args is None:
                args = self.args

            # 저장된 체크포인트 불러오기
            checkpoint = torch.load(model_path)

            # 모델 상태 복원
            self.p_encoder.load_state_dict(checkpoint["p_encoder_state_dict"])
            self.q_encoder.load_state_dict(checkpoint["q_encoder_state_dict"])

            # Optimizer 및 Scheduler 복원
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": args.weight_decay},
                {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": 0.0},
                {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": args.weight_decay},
                {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": 0.0},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
            )

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            print("모델과 Optimizer, Scheduler 상태를 성공적으로 불러왔습니다.")
        else:
            print("저장된 모델이 없습니다.")

    def train(self, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(batch_size).long()  # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                    }

                    q_inputs = {"input_ids": batch[3].to(args.device), "attention_mask": batch[4].to(args.device), "token_type_ids": batch[5].to(args.device)}

                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  # (batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

        self.save_model("save_model.pt", optimizer, scheduler)

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        if isinstance(query_or_dataset, str):

            results = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")
            indices = results.tolist()
            print(results)
            for i, idx in enumerate(indices):
                print(f"Top-{i+1} passage with index {idx}")
                print(self.dataset["context"][idx])

            return [self.dataset["context"][idx] for idx in indices]

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("bulk query exhaustive search"):
                results = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk)
                indices = results
                print("indices length", len(indices))
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                print("idx - ", idx)
                print("example - ", example)
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join([self.contexts[pid] for pid in indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_passage_embedding(self, args=None, p_encoder=None, mode="train"):
        pickle_name = self.context_path + "_dense_emb.bin"
        emb_path = os.path.join(self.data_path, pickle_name)

        if args is None:
            args = self.args

        if os.path.isfile(emb_path) and mode != "train":
            with open(emb_path, "rb") as f:
                p_embs = pickle.load(f)
                self.p_embs = p_embs
        else:
            if p_encoder is None:
                p_encoder = self.p_encoder

            with torch.no_grad():
                p_encoder.eval()

                p_embs = []
                for batch in tqdm(self.passage_dataloader, desc="passage embedding"):

                    batch = tuple(t.to(args.device) for t in batch)
                    p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                    p_emb = p_encoder(**p_inputs).to("cpu")
                    p_embs.append(p_emb)
            # print([p.shape for p in p_embs])
            p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)
            self.p_embs = p_embs

            with open(emb_path, "wb") as f:
                pickle.dump(p_embs, f)

    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()
            print(type(query))
            print(query)
            q_seqs_val = self.tokenizer([str(query)], padding="max_length", truncation=True, return_tensors="pt").to(args.device)
            q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)

        ## get_passage_embedding()로 미리 생성한 passage embedding(p_embs)를 활용해서 내적곱을 합니다.
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:k]

    def get_relevant_doc_bulk(self, queries, k=1, args=None, p_encoder=None, q_encoder=None):
        print("bulk queries : ", queries)
        print("len(queries)", len(queries))

        if args is None:
            args = self.args

        if q_encoder is None:
            q_encoder = self.q_encoder

        results = []
        with torch.no_grad():
            q_encoder.eval()

            for query in tqdm(queries, desc="query embedding and get relevant topk doc"):
                q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors="pt").to(args.device)
                q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)
                ## get_passage_embedding()로 미리 생성한 passage embedding(p_embs)를 활용해서 내적곱을 합니다.
                dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embs, 0, 1))
                rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
                results.append(rank[:k])

        print("bulk results : ", results)
        print("bulk results len : ", len(results))
        return results
