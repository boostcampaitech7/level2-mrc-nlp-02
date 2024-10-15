import os
import pickle

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from utils_retriever import timer


class TFIDFRetriever :
    def __init__(self, sparse_args, logger) :
        tokenizer_func = get_tokenizer_func(sparse_args) 
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=sparse_args.n_gram, token_pattern = None)
        self.sparse_matrix = None
        self.logger = logger

    def fit(self, corpus, training_args) :
        corpus_list = []
        for i in range(len(corpus)) :
            corpus_list.append(corpus[i]['context'])

        model_path = os.path.join(training_args.output_dir, training_args.tfidf_file)
        if training_args.do_train :
            with timer(self.logger, "Fitting Sparse Matrix") :
                self.vectorizer.fit(corpus_list)
            with timer(self.logger, "Transforming corpus") :
                self.sparse_matrix = self.vectorizer.transform(corpus_list)
            self.save_model(model_path)
        else :
            self.load_model(model_path)

    def retrieve(self, queries, top_k, batch_size) :
        # queries : 1-dim list 
        assert self.sparse_matrix is not None, "Sparse matrix hasn't been calculated yet."
        assert batch_size <= len(queries), "Batch size must smaller then length of queries."        

        scores = []
        indices = []
        for i in range(0, len(queries), batch_size) : # consider when queries too many 
            batch_queries = queries[i:i+batch_size]
            query_vec = self.vectorizer.transform(batch_queries)

            results = query_vec * self.sparse_matrix.T # (batch_size, num_corpus)
            for result in results :
                sorted_result = np.argsort(-result.data)
                doc_scores = result.data[sorted_result][:top_k].tolist()
                doc_ids = result.indices[sorted_result][:top_k].tolist()
                scores.append(doc_scores)
                indices.append(doc_ids)

        return scores, indices

    def save_model(self, save_path) :
        with open(save_path, "wb") as file :
            pickle.dump({
                "vectorizer" : self.vectorizer,
                "sparse_matrix" : self.sparse_matrix
            }, file)
        self.logger.info(f"TF-IDF model and Sparse matrix are saved in {save_path}")

    def load_model(self, load_path) :
        with open(load_path, "rb") as file :
            data = pickle.load(file)
            self.vectorizer = data['vectorizer']
            self.sparse_matrix = data['sparse_matrix']
        self.logger.info(f"TF-IDF model and Sparse matrix are loaded at {load_path}")


# BM-25
class BM25Retriever :
    def __init__(self, sparse_args, logger) :
        self.tokenizer_func = get_tokenizer_func(sparse_args)
        self.logger = logger
        self.bm25 = None
        self.tokenized_corpus = None
        self.k1 = sparse_args.k1
        self.b = sparse_args.b
    
    def fit(self, corpus, training_args) :
        corpus_list = []
        for i in range(len(corpus)) :
            corpus_list.append(corpus[i]['context'])

        model_path = os.path.join(training_args.output_dir, training_args.bm25_file)
        if training_args.do_train :
            with timer(self.logger, "Tokenizing Corpus") :
                self.tokenized_corpus = [self.tokenizer_func(doc) for doc in corpus_list]

            with timer(self.logger, "Initializing BM25") :
                self.bm25 = BM25Okapi(self.tokenized_corpus, k1 = self.k1, b = self.b)
            self.save_model(model_path)
        else :
            self.load_model(model_path)

    def retrieve(self, queries, top_k, batch_size) :
        assert self.bm25 is not None, "BM25 model hasn't been fitted yet."
        assert batch_size <= len(queries), "Batch size must smaller then length of queries."

        scores = []
        indices = []
        for i in range(0, len(queries), batch_size) :
            batch_queries = queries[i:i+batch_size]
            tokenized_queries = [self.tokenizer_func(query) for query in batch_queries]

            for query in tokenized_queries :
                query_scores = self.bm25.get_scores(query)

                sorted_indices = np.argsort(query_scores)[-top_k:][::-1]
                doc_scores = [query_scores[idx] for idx in sorted_indices]
                scores.append(doc_scores)
                indices.append(sorted_indices.tolist())
        
        return scores, indices
    def save_model(self, save_path) :
        with open(save_path, "wb") as file :
            pickle.dump(self.bm25, file)
    
    def load_model(self, load_path) :
        with open(load_path, "rb") as file :
            self.bm25 = pickle.load(file)



def simple_tokenizer(text) :
    return text.lower().split()

def huggingface_tokenizer(tokenizer) :
    def tokenize(text) : 
        return tokenizer.tokenize(text, add_special_tokens = False)
    return tokenize

def get_tokenizer_func(sparse_args) :
    if sparse_args.sparse_tokenizer_name == "simple" :
        return simple_tokenizer
    else :
        tokenizer = AutoTokenizer.from_pretrained(sparse_args.sparse_tokenizer_name)
        return huggingface_tokenizer(tokenizer)