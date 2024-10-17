import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

import argparse
from transformers import AutoTokenizer

# 커스텀
from Retriever.chw.embedding.sparse import SparseRetrieval

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", metavar="../../data/train_dataset", type=str, help="")
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="../../data", type=str, help="")
    parser.add_argument("--context_path", metavar="wikipedia_documents", type=str, help="")
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")
    parser.add_argument("--topk", metavar=3, type=int, help="")
    parser.add_argument("--method", metavar="tfidf", type=str, help="임베딩 방법을 인수로 전달합니다. ex) tfidf, bm25")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)

    ## 현재 validation 데이터셋에 대한 테스트, train 데이터셋도 포함시키고 싶을 경우 밑의 org_dataset["train"].flatten_indices() 주석 해제
    full_ds = concatenate_datasets(
        [
            # org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds[:2])
    print("dataset size", len(full_ds))
    ## full_ds의 컬럼 ['title','context','question','id','answers - ['answer_start', 'text']','document_id']
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )
    if args.method == "bm25":
        retriever.get_bm25_embedding()
    else:
        retriever.get_sparse_embedding()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    print(args.topk)
    if args.use_faiss:

        num_clusters = 64
        retriever.build_faiss(num_clusters=num_clusters)
        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query, topk=args.topk)
            # print("single with no faiss - query scores, indices", scores, indices)
        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds, topk=args.topk)
            df["correct"] = df.apply(lambda row: row["context"].find(row["original_context"]) != -1, axis=1)

            print("idx < 10 context compare", df[:10]["original_context"], df[:10]["context"])
            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query, topk=args.topk, method=args.method)
        # print("single with no faiss - query scores, indices", scores, indices)
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=args.topk, method=args.method)
            df["correct"] = df.apply(lambda row: row["context"].find(row["original_context"]) != -1, axis=1)

            print("idx < 10 context compare", df[:10]["original_context"], df[:10]["context"])

            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )
