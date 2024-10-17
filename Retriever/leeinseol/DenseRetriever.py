import os 

import faiss
import torch
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    get_linear_schedule_with_warmup
)

from utils_retriever import timer


# 인코더 정의
class Encoder:
    def __new__(cls, encoder_type, model_name_or_path, embedding_dim = None) :
        if encoder_type == "Bert" :
            config = BertConfig.from_pretrained(model_name_or_path)
            if embedding_dim :
                config.hidden_size = embedding_dim
            return BertEncoder(config).from_pretrained(model_name_or_path, config = config)
        else :
            raise ValueError(f"Unknown '{encoder_type}' encoder type. Setting another Encoder.")


# 각 인코더 정의
class BertEncoder(BertPreTrainedModel) :
    def __init__(self, config) :
        super(BertEncoder, self).__init__(config) 

        self.bert = BertModel(config)
        self.init_weights() 
    
    def forward(self, input_ids,
                attention_mask = None, token_type_ids = None) :
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1] # CLS 토큰에 해당하는 임베딩
    
        return pooled_output


class DenseRetriever :
    def __init__(self, dense_args, logger) :
        self.dense_args = dense_args

        assert dense_args.query_encoder.embedding_dim == dense_args.passage_encoder.embedding_dim, \
        "Query, Passage Encoder have different embedding dimention."
        self.query_tokenizer = AutoTokenizer.from_pretrained(dense_args.query_encoder.model_name_or_path)
        self.query_encoder = Encoder(dense_args.query_encoder.encoder_type,
                                     dense_args.query_encoder.model_name_or_path,
                                     dense_args.query_encoder.embedding_dim)
        
        self.passage_tokenizer = AutoTokenizer.from_pretrained(dense_args.passage_encoder.model_name_or_path)
        self.passage_encoder = Encoder(dense_args.passage_encoder.encoder_type,
                                       dense_args.passage_encoder.model_name_or_path,
                                       dense_args.passage_encoder.embedding_dim)
        
        self.passage_embeddings = None
        self.document_mappings = None
        self.indexer = None 

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = logger 
        
    def fit(self, dataset, corpus, training_args) :
        output_dir = training_args.output_dir
        model_path = os.path.join(output_dir, training_args.model_file)
        embeddings_path = os.path.join(output_dir, training_args.embeddings_file)
        indexer_path = os.path.join(output_dir, training_args.indexer_file)
    
        if training_args.do_train : 
            wandb.init(project = training_args.project_name,
                       config = {
                           "dense_args" : vars(self.dense_args),
                           "training_args" : vars(training_args)
                       })
            
            train_dataset, val_dataset = dataset['train'], dataset['validation']
            if training_args.use_k_fold :
                with timer(self.logger, "Total training") :
                    self.k_fold_train(train_dataset, val_dataset, training_args, model_path)
            else :
                train_tensor = self.get_dataset_to_tensor(train_dataset, True, self.dense_args.use_overflow)
                val_tensor = self.get_dataset_to_tensor(val_dataset, True, self.dense_args.use_overflow)
                train_loader = DataLoader(train_tensor, 
                                          batch_size = training_args.per_device_train_batch_size,
                                          shuffle = True)
                val_loader = DataLoader(val_tensor, 
                                        batch_size = training_args.per_device_eval_batch_size)
                with timer(self.logger, "Total training") : 
                    self.train(train_loader, val_loader, training_args)
                self.save_model(model_path)
                final_val_loss = self.evaluate(val_loader) 
                self.logger.info(f"Model saved at {model_path} with validation loss {final_val_loss:.4f}")
                
            with timer(self.logger, "Calculating all passage embedding") :
                self.calculate_embeddings(corpus, self.dense_args.use_overflow, 
                                          training_args.per_device_eval_batch_size, embeddings_path) 
            self.build_faiss(indexer_path)
            wandb.finish()

        else :
            self.load_model(model_path)
            self.load_embeddings(embeddings_path)
            if self.dense_args.use_faiss :
                self.load_indexer(indexer_path)

    def retrieve(self, queries, top_k, batch_size) :
        # dataset : .arrow 파일 형태로 된 데이터셋 
        assert self.passage_embeddings is not None, "There is no passage embeddings."
        assert self.document_mappings is not None, "There is no document mappings."
        assert batch_size <= len(queries), "Batch size must smaller then length of query set." 
        if self.dense_args.use_faiss :
            assert self.indexer is not None, "There is no indexer."

        query_tensor = self.get_dataset_to_tensor(queries, False, self.dense_args.use_overflow)
        query_loader = DataLoader(query_tensor, batch_size=batch_size, shuffle = False)
        
        result_scores = []
        result_indices = []
        self.query_encoder.eval()
        self.query_encoder.to(self.device)
        with torch.no_grad() :
            for batch in query_loader :
                batch = tuple(t.to(self.device) for t in batch)
                query_inputs = {
                    'input_ids' : batch[0],
                    'attention_mask' : batch[1],
                    'token_type_ids' : batch[2],
                }
                query_outputs = self.query_encoder(**query_inputs)
                query_embeddings = F.normalize(query_outputs, p = 2, dim = 1)
                if self.dense_args.use_faiss : 
                    query_embeddings = query_embeddings.cpu().numpy()
                    scores, indices = self.indexer.search(query_embeddings, top_k * 5)
                    # type : numpy.ndarray, numpy.ndarray
                else :
                    passage_embeddings = F.normalize(self.passage_embeddings.to(self.device), p = 2, dim = 1)
                    similarity_scores = torch.matmul(query_embeddings, passage_embeddings.T) # (batch_size, num_passages)
                    scores, indices = similarity_scores.topk(top_k * 5, dim = 1)
                    # type : torch.Tensor, torch.Tensor

                for i, index_list in enumerate(indices) : 
                    if torch.is_tensor(index_list) : 
                        index_list = index_list.cpu().tolist()
                        scores_list = scores[i].cpu().tolist()
                    else :
                        scores_list = scores[i].tolist()
                    
                    seen_docs = set()
                    unique_doc_ids = []
                    unique_scores = []

                    for idx, score in zip(index_list, scores_list) :
                        full_id = self.document_mappings[idx]
                        doc_id = full_id.split('_')[0] # 문서아이디_단락아이디
                        if doc_id not in seen_docs :
                            seen_docs.add(doc_id)
                            unique_doc_ids.append(doc_id)
                            unique_scores.append(score)
                        
                        if len(unique_doc_ids) == top_k :
                            break
                    
                    result_scores.append(unique_scores)
                    result_indices.append(unique_doc_ids)
                
        torch.cuda.empty_cache()

        return result_scores, result_indices       
        
    def k_fold_train(self, train_dataset, val_dataset, training_args, best_model_path) :
        def initialize_encoders() :
            self.query_encoder = Encoder(self.dense_args.query_encoder.encoder_type,
                                         self.dense_args.query_encoder.model_name_or_path,
                                         self.dense_args.query_encoder.embedding_dim)
            self.passage_encoder = Encoder(self.dense_args.passage_encoder.encoder_type,
                                           self.dense_args.passage_encoder.model_name_or_path,
                                           self.dense_args.passage_encoder.embedding_dim)

        best_val_loss, best_fold = float('inf'), 0
        num_rows = train_dataset.num_rows
        indices = list(range(num_rows))
        kf = KFold(n_splits=training_args.n_splits, shuffle=True, random_state=training_args.seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)) :
            initialize_encoders() # for new fold train
            train_subset = train_dataset.select(train_idx)
            val_subset = train_dataset.select(val_idx)

            train_subtensor = self.get_dataset_to_tensor(train_subset, True, self.dense_args.use_overflow) 
            val_subtensor = self.get_dataset_to_tensor(val_subset, True, self.dense_args.use_overflow)

            train_subloader = DataLoader(train_subtensor, 
                                         batch_size = training_args.per_device_train_batch_size,
                                         shuffle = True)
            val_subloader = DataLoader(val_subtensor,
                                       batch_size = training_args.per_device_eval_batch_size)
            
            with timer(self.logger, f"Training fold {fold+1}") :
                self.train(train_subloader, val_subloader, training_args, fold)

            val_loss = self.evaluate(val_subloader)
            self.logger.info(f"Fold {fold+1} final validation loss : {val_loss:.4f}")

            if val_loss < best_val_loss :
                best_val_loss = val_loss
                best_fold = fold + 1
                self.save_model(best_model_path)
                self.logger.info(f"New best model saved at fold {fold+1} with validation loss {val_loss:.4f}")
        self.logger.info(f"Best model saved at fold {best_fold} with validation loss {best_val_loss:.4f}")

        self.load_model(best_model_path)
        val_tensor = self.get_dataset_to_tensor(val_dataset, True, self.dense_args.use_overflow)
        val_loader = DataLoader(val_tensor, 
                                batch_size = training_args.per_device_eval_batch_size)
        final_val_loss = self.evaluate(val_loader)
        self.logger.info(f"Best model's total validation loss(fold {best_fold}) : {final_val_loss:.4f}")

    def train(self, train_loader, val_loader, training_args, fold = None) : 
        self.query_encoder.to(self.device)
        self.passage_encoder.to(self.device)

        no_decay = ['bias','LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params' : [p for n, p in self.passage_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay' : training_args.weight_decay
            },
            {
                'params' : [p for n, p in self.passage_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay' : 0.0
            },
            {
                'params' : [p for n, p in self.query_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay' : training_args.weight_decay
            },
            {
                'params' : [p for n, p in self.query_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay' : 0.0
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                                      lr = training_args.learning_rate, 
                                      eps = training_args.adam_epsilon)
        t_total = len(train_loader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=training_args.warmup_steps,
                                                    num_training_steps=t_total)
        global_step = 0
        optimizer.zero_grad() # 한번에 encoder들 초기화 
        torch.cuda.empty_cache()

        self.logger.info("Start Training...")
        for epoch in range(training_args.num_train_epochs) :
            for step, batch in enumerate(train_loader) :
                self.query_encoder.train()
                self.passage_encoder.train()

                batch = tuple(t.to(self.device) for t in batch)
                query_inputs = {
                    'input_ids' : batch[0],
                    'attention_mask' : batch[1],
                    'token_type_ids' : batch[2]
                }
                passage_inputs = {
                    'input_ids' : batch[3],
                    'attention_mask' : batch[4],
                    'token_type_ids' : batch[5],
                }
                document_ids = batch[6] # 1-d

                query_outputs = self.query_encoder(**query_inputs) # (batch_size, emb_dim)
                passage_outputs = self.passage_encoder(**passage_inputs) # (batch_size, emb_dim)

                # cosine similarity : dot product after l2 normalization
                query_outputs = F.normalize(query_outputs, p = 2, dim = 1)
                passage_outputs = F.normalize(passage_outputs, p = 2, dim = 1)
                similarity_matrix = torch.matmul(query_outputs, passage_outputs.T)
                # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

                # In-batch Negative sampling
                # 같은 문서에 속하는 단락들끼리는 1, 아니면 0으로 타켓을 만듬
                positive_mask = (document_ids.unsqueeze(1) == document_ids.unsqueeze(0)).float()
                # using broadcasting :
                # (batch_size, 1) == (1, batch_size) -> (batch_size, batch_size) == (batch_size, batch_size)
                targets = positive_mask.to(self.device)
                
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(similarity_matrix, targets)

                # 한 번의 epoch에서 지정된 step마다 train loss 기록 
                if step % training_args.logging_step == 0:
                    if fold :
                        self.logger.info(f"Fold {fold+1} Epoch [{epoch}/{training_args.num_train_epochs}] Step [{step}/{len(train_loader)}] train loss : {loss.item():.4f}")
                        wandb.log({
                            f'train_loss_fold{fold+1}' : loss.item()
                        })
                        wandb.log({
                            f'learning_rate_fold_{fold+1}' : scheduler.get_last_lr()[0]
                        })
                    else :
                        self.logger.info(f"Epoch [{epoch}/{training_args.num_train_epochs}] Step [{step}/{len(train_loader)}] train loss : {loss.item():.4f}")
                        wandb.log({
                            'train_loss' : loss.item()
                        })
                        wandb.log({
                            'learning_rate' : scheduler.get_last_lr()[0]
                        })

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() # 한번에 encoder 초기화 
                global_step += 1
                torch.cuda.empty_cache()
            
            # epoch이 끝날 때마다 val loss 기록
            val_loss = self.evaluate(val_loader)
            if fold :
                self.logger.info(f"Fold {fold+1} Epoch [{epoch}/{training_args.num_train_epochs}] validation loss : {val_loss:.4f}")
                wandb.log({
                    f'val_loss_fold_{fold+1}' : val_loss
                })
            else :
                self.logger.info(f"Epoch [{epoch}/{training_args.num_train_epochs}] validation loss : {val_loss:.4f}")
                wandb.log({
                    f'val_loss' : val_loss
                })

    def evaluate(self, val_loader) :
        self.query_encoder.to(self.device)
        self.passage_encoder.to(self.device)

        self.query_encoder.eval()
        self.passage_encoder.eval()

        val_loss = 0
        with torch.no_grad() :
            for step, batch in enumerate(val_loader) :
                batch = tuple(t.to(self.device) for t in batch)
                query_inputs = {
                    'input_ids' : batch[0],
                    'attention_mask' : batch[1],
                    'token_type_ids' : batch[2]
                }
                passage_inputs = {
                    'input_ids' : batch[3],
                    'attention_mask' : batch[4],
                    'token_type_ids' : batch[5],
                }
                document_ids = batch[6]

                query_outputs = self.query_encoder(**query_inputs) # (batch_size, emb_dim)
                passage_outputs = self.passage_encoder(**passage_inputs) # (batch_size, emb_dim)
                
                query_outputs = F.normalize(query_outputs, p = 2, dim = 1)
                passage_outputs = F.normalize(passage_outputs, p = 2, dim = 1)
                similarity_matrix = torch.matmul(query_outputs, passage_outputs.T)
                
                positive_mask = (document_ids.unsqueeze(1) == document_ids.unsqueeze(0)).float()
                targets = positive_mask.to(self.device)
                
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(similarity_matrix, targets)

                val_loss += loss.item()
        
        return val_loss / len(val_loader)

    def calculate_embeddings(self, corpus, use_overflow, batch_size, embeddings_path) :
        all_context_tensor = self.get_all_context_to_tensor(corpus, use_overflow)
        all_context_loader = DataLoader(all_context_tensor, batch_size=batch_size, shuffle=False)
        
        embeddings = []
        self.passage_encoder.to(self.device)
        
        with torch.no_grad() :
            for batch in all_context_loader :
                input_ids, attention_mask, token_type_ids = [tensor.to(self.device) for tensor in batch]

                passage_inputs = {
                    'input_ids' : input_ids,
                    'attention_mask' : attention_mask,
                    'token_type_ids' : token_type_ids,
                }
                batch_embeddings = self.passage_encoder(**passage_inputs) # (batch_size, emb_dim)
                embeddings.append(batch_embeddings.cpu())

        self.passage_embeddings = torch.cat(embeddings, dim = 0)

        self.save_embeddings(embeddings_path)

    def build_faiss(self, indexer_path) : 
        if self.dense_args.use_faiss :
            with timer(self.logger, "Building faiss") :
                num_clusters = self.dense_args.num_clusters
                n_iter = self.dense_args.n_iter
                emb_dim = self.passage_embeddings.shape[-1]

                # normalizing vector to use inner-product -> cosine similarity
                normalized_embeddings = F.normalize(self.passage_embeddings, p = 2, dim = 1).numpy()


                # Clustering
                index_flat = faiss.IndexFlatIp(emb_dim) # inner-product 
                # index_flat = faiss.IndexFlatL2(emb_dim) # L2-distance
                clus = faiss.Clustering(emb_dim, num_clusters)
                clus.verbose = True
                clus.niter = n_iter
                clus.train(normalized_embeddings, index_flat)
                
                # Retrieve cluster centroids 
                centroids = faiss.vector_float_to_array(clus.centroids).reshape(num_clusters, emb_dim)
                
                # Quantizer setup with centroids
                quantizer = faiss.IndexFlatIP(emb_dim) # inner-product 
                # quantizer = faiss.IndexFlatL2(emb_dim) # L2-distance
                quantizer.add(centroids)

                # IVF + Scalar Quantizer indexer setup 
                self.indexer = faiss.IndexIVFScalarQuantizer(
                    quantizer,
                    emb_dim,
                    num_clusters,
                    faiss.ScalarQuantizer.QT_8bit,
                    # faiss.METRIC_L2, # when using L2-distance
                    faiss.METRIC_INNER_PRODUCT, # when using cosine similarity
                )

                # Training and adding embeddings
                self.indexer.train(self.passage_embeddings.numpy())
                self.indexer.add(self.passage_embeddings.numpy())

                # 설정에 따라 비활성화된 경우 추가로 설정 가능(default = 1)
                self.indexer.nprobe = self.dense_args.indexer_nprobe

                self.save_indexer(indexer_path)

    def get_dataset_to_tensor(self, dataset, train, use_overflow) :
        # 추후에 정답이 있는 부분만 골라내고, positive_mask에서 가중치를 주는 방식으로 해도 될듯? 
        # 약간 나눠진게 4개다
        # 정답 있는 부분에 1/2 주고 나머지 가중치 나눠 갖고 
        # 아이디에 소수점 부분 더해서 관리하면 될듯?? 
        query_args = self.dense_args.query_encoder

        if isinstance(dataset, list) :
            q_seqs = self.query_tokenizer(dataset,
                                          padding = query_args.padding,
                                          truncation = query_args.truncation,
                                          max_length = query_args.max_length,
                                          return_tensors = 'pt',
                                          )
        else :
            q_seqs = self.query_tokenizer(dataset['question'],
                                          padding = query_args.padding,
                                          truncation = query_args.truncation,
                                          max_length = query_args.max_length,
                                          return_tensors = 'pt',
                                          )
        if train : 
            passage_args = self.dense_args.passage_encoder
            p_seqs = self.passage_tokenizer(dataset['context'],
                                            padding = passage_args.padding,
                                            truncation = passage_args.truncation,
                                            max_length = passage_args.max_length,
                                            return_tensors = 'pt',
                                            return_overflowing_tokens = use_overflow,
                                            stride = passage_args.stride if use_overflow else 0,
                                            )
            
            document_ids = dataset['document_id']

            if use_overflow :
                sample_mapping = p_seqs.pop("overflow_to_sample_mapping")
                q_seqs = {
                    key : torch.cat([q_seqs[key][sample_mapping] for key in q_seqs], dim = 0)
                    for key in q_seqs
                }
                # overflow에 따라 document_id 업데이트
                document_ids = [document_ids[idx] for idx in sample_mapping]
            
            # token_type_ids가 없는 경우 처리(e.g. RoBERTa, ...)
            token_type_ids = p_seqs.get("token_type_ids", torch.zeros_like(p_seqs['input_ids']))
            
            return TensorDataset(
                q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs.get('token_type_ids', torch.zeros_like(q_seqs['input_ids'])),
                p_seqs['input_ids'], p_seqs['attention_mask'], token_type_ids,
                torch.tensor(document_ids)
            )
        
        return TensorDataset(
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs.get('token_type_ids', torch.zeros_like(q_seqs['input_ids']))
        )

    def get_all_context_to_tensor(self, corpus, use_overflow) :
        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_segment_ids = []

        passage_args = self.dense_args.passage_encoder
        for doc in corpus :
            doc_id = doc['id']
            context = doc['context']

            p_seqs = self.passage_tokenizer(
                context,
                padding = passage_args.padding,
                truncation = passage_args.truncation,
                max_length = passage_args.max_length,
                return_tensors = 'pt',
                return_overflowing_tokens = use_overflow,
                stride = passage_args.stride if use_overflow else 0,
            )

            if use_overflow :
                sample_mapping = p_seqs.pop("overflow_to_sample_mapping")
                segment_ids = [f"{doc_id}_{i}" for i in range(len(sample_mapping))]
            else :
                segment_ids = [f"{doc_id}_0"]
            
            all_input_ids.append(p_seqs['input_ids'])
            all_attention_mask.append(p_seqs['attention_mask'])
            all_token_type_ids.append(p_seqs.get("token_type_ids", torch.zeros_like(p_seqs['input_ids'])))
            all_segment_ids.extend(segment_ids)
        
        all_input_ids = torch.cat(all_input_ids, dim = 0)
        all_attention_mask = torch.cat(all_attention_mask, dim = 0)
        all_token_type_ids = torch.cat(all_token_type_ids, dim = 0)
        
        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        self.document_mappings = all_segment_ids

        return tensor_dataset

    def save_embeddings(self, embeddings_path) :
        assert self.passage_embeddings is not None, "Passage embeddings are None. You must calculate passage embeddings."
        assert self.document_mappings is not None, "Document mappings are None. You must calculate passage embeddings."
        torch.save({
            'embeddings' : self.passage_embeddings, 
            'document_mappings' : self.document_mappings
            }, embeddings_path)
    
    def load_embeddings(self, embeddings_path) :
        assert os.path.exists(embeddings_path), f"Passage embeddings are not in {embeddings_path}."
        embeddings = torch.load(embeddings_path)

        self.passage_embeddings = embeddings['embeddings']
        self.document_mappings = embeddings['document_mappings']
    
    def save_model(self, model_path) :
        torch.save({
            'query_encoder' : self.query_encoder.state_dict(),
            'passage_encoder' : self.passage_encoder.state_dict()
        }, model_path)
    
    def load_model(self, model_path) :
        assert os.path.exists(model_path), f"Saved model is not in {model_path}."
        checkpoint = torch.load(model_path)
        self.query_encoder.load_state_dict(checkpoint['query_encoder'])
        self.passage_encoder.load_state_dict(checkpoint['passage_encoder'])

    def save_indexer(self, indexer_path) :
        assert self.indexer is not None, "Indexer is None. You must build faiss."
        faiss.write_index(self.indexer, indexer_path) 
    
    def load_indexer(self, indexer_path) :
        assert os.path.exists(indexer_path), f"Indexer is not in {indexer_path}."
        self.indexer = faiss.read_index(indexer_path)
