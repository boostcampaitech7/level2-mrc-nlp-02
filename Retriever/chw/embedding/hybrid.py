import os
import json
from tqdm.auto import tqdm
import pickle
import os.path as p
import pandas as pd

from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from datasets import load_from_disk, concatenate_datasets, Dataset
from Retriever.chw.utils.util import timer


class HybridLogisticRetrieval:
    def __init__(self, args, train_dataset, data_path, context_path, sparse_retriever, dense_retriever):

        self.data_path = data_path
        self.context_path = context_path
        self.train_dataset = train_dataset
        self.logistic = None

        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

    def logistic_train_with_ditribution(self):
        # train_dataset = concatenate_datasets([datasets["train"], datasets["validation"]])
        """_summary_
            이 훈련 방법은 검색 결과의 분포를 특징으로 삼아 분류기를 학습합니다.
            상위 64개의 문서를 뽑아내여 각 문서를 2^i 승으로 6까지
        Returns:
            _type_: _description_
        """
        queries = self.train_dataset["question"]

        queries = self.train_dataset["question"]
        org_contexts = self.train_dataset["context"]

        # doc_scores, doc_indices = np.array(sparse_scores), np.array(sparse_indices)
        top_k_limit = 5
        X, Y = [], []

        for query, org_context in tqdm(zip(queries, org_contexts), desc="hybrid logistic train"):
            sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc(query, k=5, method="bm25")

            feature_vector = [sparse_scores[: min(pow(2, i), len(sparse_scores))] for i in range(1, 6)]
            feature_vector = F.softmax(torch.tensor(list(map(lambda x: x.mean(), feature_vector))), dim=-1)

            label = 0
            org_context_idx = -1
            print("sparse_indices", sparse_indices)
            context_list = [self.contexts[i] for i in sparse_indices]
            if org_context in context_list:
                org_context_idx = context_list.index(org_context)
            print("org_context top index in wiki(y) : ", org_context_idx)
            if org_context_idx != -1 and org_context_idx <= top_k_limit:
                label = 1

            X.append(feature_vector)
            Y.append(label)

        logistic = LogisticRegression()
        logistic.fit(X, Y)

        return logistic

    def logistic_train_ver2(self, topk=1):

        queries = self.train_dataset["question"]
        org_contexts = self.train_dataset["context"]

        sparse_features = []
        dense_features = []
        labels = []
        top_k_limit = 5

        for query, org_context in tqdm(zip(queries, org_contexts), desc="hybrid logistic train"):
            sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc(query, k=topk, method="bm25")
            dense_scores, dense_indices = self.dense_retriever.get_relevant_doc(query, k=topk)
            print("sparse_scores", sparse_scores)
            print("len(sparse_scores)", len(sparse_scores))
            print("dense_scores", dense_scores)
            print("len(dense_scores)", dense_scores.shape)
            sparse_features.append(np.array(sparse_scores).mean())
            dense_features.append(dense_scores.numpy().mean())

            label = 0
            org_context_idx = -1
            print("sparse_indices", sparse_indices)
            context_list = [self.contexts[i] for i in sparse_indices]
            if org_context in context_list:
                org_context_idx = context_list.index(org_context)
            print("org_context top index in wiki(org_context_idx) : ", org_context_idx)
            if org_context_idx != -1 and org_context_idx < top_k_limit:
                label = 1

            labels.append(label)

        print(sparse_features, dense_features)
        X = np.stack([sparse_features, dense_features], axis=-1)
        logistic = LogisticRegression()
        logistic.fit(X, labels)

        return logistic

    def get_logistic_regression(self, save_name=None, labels=[0, 1], topk=1):
        pickle_name = save_name
        model_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                self.logistic = pickle.load(f)
        else:
            self.logistic = self.logistic_train_with_ditribution()
            with open(model_path, "wb") as f:
                pickle.dump(self.logistic, f)

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        if isinstance(query_or_dataset, str):
            raise NotImplementedError
            # scores, results = self.get_relevant_doc(query_or_dataset, k=topk)
            # print("[Search query]\n", query_or_dataset, "\n")
            # indices = results.tolist()
            # print(results)
            # for i, idx in enumerate(indices):
            #     print(f"Top-{i+1} passage with index {idx}")
            #     print(self.contexts[idx])

            # return [self.contexts[idx] for idx in indices]

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("bulk query exhaustive search"):
                label_indices, sparse_indices, dense_indices = self.get_relevant_doc_bulk_with_distribution(query_or_dataset["question"], topk=topk)

                print("indices length", len(label_indices), len(sparse_indices), len(dense_indices))
                print("label indices  ---------", label_indices)
                assert len(label_indices) == len(sparse_indices), "레이블과 임베딩 인덱스 길이가 다릅니다."
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Hybrid retrieval: ")):
                # print("idx - ", idx)
                # print("label_indices[idx] - ", label_indices[idx])
                indices = sparse_indices[idx] if label_indices[idx] == 1 else dense_indices[idx]
                # print("indices - ", indices)

                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": [self.contexts[pid] for pid in indices],
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                # print("origin context - ", tmp["original_context"])
                # print("topk context", tmp["context"])
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_bulk_ver2(self, queries, topk=1):

        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc_bulk(queries, k=topk, method="bm25")
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(queries, k=topk)
        sparse_convert = [np.array(score).mean() for score in sparse_scores]
        dense_convert = [score.numpy().mean() for score in dense_scores]
        print(sparse_scores[:2], sparse_convert[:2])
        print(dense_scores[:2], dense_convert[:2])
        feature_vectors = []
        for sparse_score, dense_score in zip(sparse_convert, dense_convert):
            features = np.stack([sparse_score, dense_score], axis=-1)
            feature_vectors.append(features)
        # features = np.stack([sparse_convert, dense_convert], axis=-1)
        # hybrid_scores = self.logistic.predict_proba(features)[:, 1]

        # label_indices = np.argsort(hybrid_scores)[::-1
        label_list = self.logistic.predict(feature_vectors)
        # print("*" * 20, "hybrid_scores", "*" * 20)
        # print(hybrid_scores)
        # print(len(hybrid_scores))
        print("*" * 20, "label_indices", "*" * 20)
        print(label_list)
        print(len(label_list))
        return label_list, sparse_indices, dense_indices

    def get_relevant_doc_bulk_with_distribution(self, queries, topk=1):
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc_bulk(queries, k=topk, method="bm25")
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(queries, k=topk)
        # print(sparse_scores[:2])
        feature_vectors = []
        for sparse_score in sparse_scores:
            feature_vector = [sparse_score[: min(pow(2, i), len(sparse_score))] for i in range(1, 6)]
            feature_vector = F.softmax(torch.tensor(list(map(lambda x: x.mean(), feature_vector))), dim=-1)

            feature_vectors.append(feature_vector)

        label_list = self.logistic.predict(feature_vectors)
        # print("*" * 20, "hybrid_scores", "*" * 20)
        # print(hybrid_scores)
        # print(len(hybrid_scores))
        print("*" * 20, "label_indices", "*" * 20)
        print(label_list)
        print(len(label_list))
        return label_list, sparse_indices, dense_indices
