import os
import json
from tqdm.auto import tqdm
import pickle
import os.path as p
import pandas as pd

from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from datasets import load_from_disk, concatenate_datasets, Dataset
from Retriever.chw.utils.util import timer


class HybridLogisticRetrieval:
    def __init__(self, args, train_dataset, data_path, context_path, sparse_retriever, dense_retriever):

        self.data_path = data_path
        self.context_path = context_path
        self.train_dataset = train_dataset
        self.logistic = None
        self.num_features = 3
        self.kbound = 3

        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

    def logistic_train(self):
        # train_dataset = concatenate_datasets([datasets["train"], datasets["validation"]])

        queries = self.train_dataset["question"]
        doc_scores, doc_indices = self.sparse_retriever.get_relevant_doc_bulk(queries, topk=10)
        doc_scores, doc_indices = np.array(doc_scores), np.array(doc_indices)

        contexts = np.array(self.sparse_retriever.contexts)

        train_x, train_y = [], []

        for idx in tqdm(range(len(doc_scores))):
            doc_index = doc_indices[idx]
            org_context = self.train_dataset["context"][idx]

            feature_vector = [doc_scores[idx][: pow(2, i)] for i in range(1, self.num_features + 1)]
            feature_vector = list(map(lambda x: x.mean(), feature_vector))
            feature_vector = softmax(feature_vector)

            label = 0
            y = -1
            if org_context in contexts[doc_index]:
                y = list(contexts[doc_index]).index(org_context)
            if y != -1 and y < self.kbound:
                label = 1

            train_x.append(feature_vector)
            train_y.append(label)

        logistic = LogisticRegression()
        logistic.fit(train_x, train_y)

        return logistic

    def logistic_train_ver2(self, labels, topk=1):

        queries = self.train_dataset["question"]
        sparse_features = []
        dense_features = []
        for query in tqdm(queries, desc="hybrid logistic train"):
            sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc(query, k=topk, method="bm25")
            dense_scores, dense_indices = self.dense_retriever.get_relevant_doc(query, k=topk)
            print("sparse_scores", sparse_scores)
            print("len(sparse_scores)", len(sparse_scores))
            print("dense_scores", dense_scores)
            print("len(dense_scores)", dense_scores.shape)
            sparse_features.append(sparse_scores)
            dense_features.append(dense_scores)

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
            self.logistic = self.logistic_train_ver2(labels=labels, topk=topk)
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
                scores, results = self.get_relevant_doc_bulk_ver2(query_or_dataset["question"], topk=topk)
                indices = results
                print("indices length", len(indices))
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                print("idx - ", idx)

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

                # print("origin context - ", tmp["original_context"])
                # print("topk context", tmp["context"])
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_bulk_ver2(self, queries, topk=1):

        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc_bulk(queries, k=topk, method="bm25")
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(queries, k=topk)

        features = np.stack([sparse_scores, dense_scores], axis=-1)
        hybrid_scores = self.logistic.predict_proba(features)[:, 1][:topk]

        combined_indices = np.argsort(hybrid_scores)[::-1][:topk]

        return combined_indices, hybrid_scores
