import os 

import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

from utils_retriever import timer


# 인코더 정의
class Encoder:
    def __new__(cls, encoder_type, model_name_or_path) :
        if encoder_type == "Bert" :
            config = BertConfig.from_pretrained(model_name_or_path)
            return BertEncoder(config).from_pretrained(model_name_or_path, config = config)
        if encoder_type == "Use_mean_pooling" :
            return MeanPoolingEncoder(model_name_or_path)
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


class MeanPoolingEncoder(nn.Module) : 
    def __init__(self, model_name_or_path) :
        super(MeanPoolingEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask = None, token_type_ids = None) :
        outputs = self.model(input_ids = input_ids,
                             attention_mask = attention_mask,
                             token_type_ids = token_type_ids)
        sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        return sentence_embeddings
    
    def mean_pooling(self, model_output, attention_mask) :
        token_embedings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedings.size()).float()
        return torch.sum(token_embedings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min = 1e-9)


class DenseRetriever :
    def __init__(self, dense_args, query_args, passage_args, logger) :
        self.dense_args = dense_args
        self.query_args = query_args
        self.passage_args = passage_args

        self.query_tokenizer = AutoTokenizer.from_pretrained(query_args.query_model_name_or_path)
        self.query_encoder = Encoder(query_args.query_type,
                                     query_args.query_model_name_or_path
                                    )
        
        self.passage_tokenizer = AutoTokenizer.from_pretrained(passage_args.passage_model_name_or_path)
        self.passage_encoder = Encoder(passage_args.passage_type,
                                       passage_args.passage_model_name_or_path
                                       )
        
        query_emb_dim = self.query_encoder.config.hidden_size
        passage_emb_dim = self.passage_encoder.config.hidden_size
        assert query_emb_dim == passage_emb_dim, "Query, Passage Encoder have different embedding dimention."
        
        self.passage_embeddings = None
        self.document_mappings = None # document ids that map to passage_embedding.
        self.indexer = None # use when use_faiss=True

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = logger 
        
    def fit(self, dataset, corpus, training_args) :
        output_dir  = training_args.output_dir
        model_path = os.path.join(output_dir, training_args.model_file)
        embeddings_path = os.path.join(output_dir, training_args.embeddings_file)
        indexer_path = os.path.join(output_dir, training_args.indexer_file)
    
        if training_args.do_train : 
            wandb.init(project = training_args.project_name,
                       config = {
                           "dense_args" : vars(self.dense_args),
                           "query_args" : vars(self.query_args),
                           "passage_args" : vars(self.passage_args),
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
            self.query_encoder = Encoder(self.query_args.query_type,
                                         self.query_args.query_model_name_or_path
                                         )
            self.passage_encoder = Encoder(self.passage_args.passage_type,
                                           self.passage_args.passage_model_name_or_path
                                           )

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
                                                    num_warmup_steps=int(ｔ_total*0.1),
                                                    num_training_steps=t_total)
        global_step = 0
        optimizer.zero_grad() # 한번에 encoder들 초기화 
        torch.cuda.empty_cache()

        total_steps = training_args.num_train_epochs * len(train_loader)
        progress_bar = tqdm(total = total_steps, bar_format='{l_bar} | Remaining: {remaining}', ncols=80)
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
                
                loss = self.contrastive_loss(similarity_matrix, targets)

                # loss_fn = torch.nn.BCEWithLogitsLoss()
                # loss = loss_fn(similarity_matrix, targets)
                
                # batch_size = similarity_matrix.size(0)
                # targets = torch.arange(0, batch_size).long().to(self.device)
                
                # similarity_matrix = F.log_softmax(similarity_matrix, dim = 1)
                
                # loss = F.nll_loss(similarity_matrix, targets)
                
                # 한 번의 epoch에서 지정된 step마다 train loss 기록 
                if step % training_args.logging_step == 0:
                    val_loss = self.evaluate(val_loader)
                    if fold :
                        self.logger.info(f"Fold {fold+1} Epoch [{epoch}/{training_args.num_train_epochs}] Step [{step}/{len(train_loader)}] train loss : {loss.item():.4f}")
                        self.logger.info(f"Fold {fold+1} Epoch [{epoch}/{training_args.num_train_epochs}] Step [{step}/{len(train_loader)}] val loss : {val_loss:.4f}")
                        wandb.log({
                            f'train_loss_fold{fold+1}' : loss.item(),
                            f'val_loss_fold_{fold+1}' : val_loss,
                            f'learning_rate_fold_{fold+1}' : scheduler.get_last_lr()[0]
                        })
                    else :
                        self.logger.info(f"Epoch [{epoch}/{training_args.num_train_epochs}] Step [{step}/{len(train_loader)}] train loss : {loss.item():.4f}")
                        self.logger.info(f"Epoch [{epoch}/{training_args.num_train_epochs}] Step [{step}/{len(train_loader)}] val loss : {val_loss:.4f}")
                        wandb.log({
                            'train_loss' : loss.item(),
                            f'val_loss' : val_loss,
                            'learning_rate' : scheduler.get_last_lr()[0]
                        })

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() # 한번에 encoder 초기화 
                global_step += 1
                progress_bar.update(1)
                
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
                
                loss = self.contrastive_loss(similarity_matrix, targets)

                # loss_fn = torch.nn.BCEWithLogitsLoss()
                # loss = loss_fn(similarity_matrix, targets)
                
                # batch_size = similarity_matrix.size(0)
                # targets = torch.arange(0, batch_size).long().to(self.device)
                
                # similarity_matrix = F.log_softmax(similarity_matrix, dim = 1)
                
                # loss = F.nll_loss(similarity_matrix, targets)
                    
                val_loss += loss.item()
        
        return val_loss / len(val_loader)

    def contrastive_loss(self, similarity_matrix, targets, margin = 1.0) :
        # Convert cosine similarity to distance
        distances = (1 - similarity_matrix) / 2
        # cosin similarity : -1 ~ 1 -> distances : 0 ~ 1(1 -> 0, -1 -> 0) 

        # Positive pairs should have smaller distances
        positive_loss = targets * distances.pow(2)
        # target이 1인 경우만 거리 계산 

        # Negative pairs should have larger distances with margin
        negative_loss = (1 - targets) * F.relu(margin - distances).pow(2)
        # target이 0인 경우만 거리 계산 
        # large margin : 더 멀리 떨어뜨리려고 함.
        # small margin : 음성 쌍 간의 거리 제한이 줄어듬.(수렴 속도 상승, 음성 쌍 간의 구분력 하락)

        # Mean loss over all pairs
        loss = (positive_loss + negative_loss).mean()

        return loss 

    def calculate_embeddings(self, corpus, use_overflow, batch_size, embeddings_path) :
        all_context_tensor = self.get_all_context_to_tensor(corpus, use_overflow)
        all_context_loader = DataLoader(all_context_tensor, batch_size=batch_size, shuffle=False)
        
        embeddings = []
        self.passage_encoder.to(self.device)
        
        with torch.no_grad() :
            total_steps = len(all_context_loader)
            progress_bar = tqdm(total = total_steps, bar_format='{l_bar} | Remaining: {remaining}', ncols=80)
            for batch in all_context_loader :
                input_ids, attention_mask, token_type_ids = [tensor.to(self.device) for tensor in batch]

                passage_inputs = {
                    'input_ids' : input_ids,
                    'attention_mask' : attention_mask,
                    'token_type_ids' : token_type_ids,
                }
                batch_embeddings = self.passage_encoder(**passage_inputs) # (batch_size, emb_dim)
                embeddings.append(batch_embeddings.cpu())
                progress_bar.update(1)

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
        # dataset : 모든 데이터 다 들어옴(배치단위 x)
        # 추후에 정답이 있는 부분만 골라내고, positive_mask에서 가중치를 주는 방식으로 해도 될듯? 
        # 약간 나눠진게 4개다
        # 정답 있는 부분에 1/2 주고 나머지 가중치 나눠 갖고 
        # 아이디에 소수점 부분 더해서 관리하면 될듯?? 

        if isinstance(dataset, list) :
            q_seqs = self.query_tokenizer(dataset,
                                          padding = "longest",
                                          truncation=True,
                                          return_tensors = 'pt',
                                          )
        else :
            q_seqs = self.query_tokenizer(dataset['question'],
                                          padding = "longest",
                                          truncation=True,
                                          return_tensors = 'pt',
                                          )
        if train : 
            p_seqs = self.passage_tokenizer(dataset['context'],
                                            padding = "longest",
                                            truncation=True,
                                            return_tensors = 'pt',
                                            return_overflowing_tokens = use_overflow,
                                            stride = self.passage_args.passage_stride if use_overflow else 0,
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

        for doc_id, context in corpus.items() :
            p_seqs = self.passage_tokenizer(
                context,
                padding = "max_length",
                truncation =True,
                return_tensors = 'pt',
                return_overflowing_tokens = use_overflow,
                stride = self.passage_args.passage_stride if use_overflow else 0,
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


class PretrainedRetriever : 
    def __init__(self, dense_args, logger) :
        self.dense_args = dense_args

        self.tokenizer = AutoTokenizer.from_pretrained(dense_args.pre_trained_model_name_or_path)
        self.model = AutoModel.from_pretrained(dense_args.pre_trained_model_name_or_path)
        
        self.embeddings = None
        self.document_mappings = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger

    def fit(self, corpus, training_args) :
        embeddings_path = os.path.join(training_args.output_dir, training_args.embeddings_file)
        
        if self.dense_args.pre_trained_split :
            if training_args.do_train : 
                corpus_list = [corpus[key] for key in corpus.keys()]
                id_list = [key for key in corpus.keys()]
                self.model.to(self.device)

                all_embeddings = []
                all_ids = []
                batch_size = training_args.per_device_train_batch_size
                with timer(self.logger, "Calculating embeddings") : 
                    for i in tqdm(range(0, len(corpus_list), batch_size)) : 
                        sub_doc = corpus_list[i:i+batch_size]
                        sub_id = id_list[i:i+batch_size]
                        batch, sub_id = self.get_document_to_tensor(sub_doc, sub_id, self.dense_args.use_overflow)
                        all_ids.extend(sub_id)
                        batch = {key:val.to(self.device) for key, val in batch.items()}
                        with torch.no_grad() :
                            sub_outputs = self.model(**batch)
                            sub_embeddings = self.average_pool(sub_outputs.last_hidden_state, batch['attention_mask'])
                            sub_embeddings = F.normalize(sub_embeddings, p = 2, dim = 1)
                            all_embeddings.append(sub_embeddings.cpu())
                    self.embeddings = torch.cat(all_embeddings, dim = 0)
                    self.document_mappings = all_ids
                    self.save_embeddings(embeddings_path)
            else :
                self.load_embeddings(embeddings_path)
        else :
            # 쿼리와 문서가 합쳐져서 들어갈 경우 사용 
            raise NotImplementedError("Using query+passage model is not implement yet.")

    def retrieve(self, queries, top_k, batch_size, task_description, hybrid = False) : 
        """
        Arguments
        - queries : List[str]
        """
        assert self.embeddings is not None, "Embeddings are None. You must calculate embeddings."
        assert self.document_mappings is not None, "Document mappings are None. You must calculate embeddings."
        assert batch_size <= len(queries), "Batch size must smaller then length of query set." 
        
        if self.dense_args.use_faiss :
            raise NotImplementedError("Using Faiss is not implement yet.")
        else :
            all_scores = []
            self.model.to(self.device)
            self.embeddings = self.embeddings.to(self.device)
            for i in tqdm(range(0,len(queries), batch_size)) :
                sub_queries = queries[i:i+batch_size]
                sub_queries = [self.get_detailed_instruct(q, task_description) for q in sub_queries]

                sub_batch = self.tokenizer(sub_queries, max_length=self.tokenizer.model_max_length,
                                           padding=True, truncation=True, return_tensors='pt')
                sub_batch = {key:val.to(self.device) for key, val in sub_batch.items()}
                with torch.no_grad() :
                    sub_query_outputs = self.model(**sub_batch)
                    sub_query_embeddings = self.average_pool(sub_query_outputs.last_hidden_state, sub_batch['attention_mask'])

                    sub_query_embeddings = F.normalize(sub_query_embeddings, p = 2, dim = 1)
                    sub_scores = (sub_query_embeddings @ self.embeddings.T)
                    all_scores.append(sub_scores)
            all_scores = torch.cat(all_scores, dim= 0)
            if hybrid :
                return all_scores.tolist(), self.document_mappings

            scores, indices = all_scores.topk(top_k, dim = 1)
        
        ids = []
        for i in range(indices.shape[0]) :
            ids.append([self.document_mappings[idx.item()] for idx in indices[i].reshape(-1)])

        return scores.tolist(), ids

    def reranking(self, doc_ids, documents, queries, top_k, task_description) :
        """
        Arguments
        - doc_ids : List[List[str]]
        - documents : List[List[str]]
        - queries : List[str]
        """
        if self.dense_args.use_faiss :
            raise NotImplementedError("Using Faiss is not implement yet.")
        else :
            # 일단 문서 아이디에 매핑되는 임베딩 가져와야함.
            self.model.to(self.device)
            all_scores = []
            all_indices = [] 
            for i in range(len(queries)) : 
                query = [self.get_detailed_instruct(queries[i], task_description)]
                query_seq = self.tokenizer(query, max_length=self.tokenizer.model_max_length,
                                            padding=True, truncation=True, return_tensors='pt')
                query_seq = {key:val.to(self.device) for key, val in query_seq.items()}
        
                candidate_documents = documents[i]
                candidate_doc_ids = doc_ids[i]
                doc_seqs, new_doc_ids = self.get_document_to_tensor(candidate_documents, 
                                            candidate_doc_ids,
                                            self.dense_args.use_overflow)
                doc_seqs = {key:val.to(self.device) for key, val in doc_seqs.items()}
                
                with torch.no_grad() :
                    query_output = self.model(**query_seq)
                    query_emb = self.average_pool(query_output.last_hidden_state, query_seq['attention_mask'])
                    query_emb = F.normalize(query_emb, p = 2, dim = 1)

                    doc_outputs = self.model(**doc_seqs)
                    doc_embs = self.average_pool(doc_outputs.last_hidden_state, doc_seqs['attention_mask'])
                    doc_embs = F.normalize(doc_embs, p = 2, dim = 1)
                    
                    result = (query_emb @ doc_embs.T)
                
                scores, indices = result.topk(top_k, dim = 1)
                indices = [new_doc_ids[idx.item()] for idx in indices.reshape(-1)]
                all_scores.append(scores.tolist())
                all_indices.append(indices)
            
            return all_scores, all_indices

    def get_document_to_tensor(self, documents, ids, use_overflow) :
        doc_seqs = self.tokenizer(documents, max_length=self.tokenizer.model_max_length, 
                                  padding = True, truncation=True, return_tensors='pt',
                                  return_overflowing_tokens= use_overflow,
                                  stride = self.dense_args.stride if use_overflow else 0,
                                  )
        if use_overflow :
            sample_mapping = doc_seqs.pop("overflow_to_sample_mapping")
            doc_seqs = {
                key : torch.cat([doc_seqs[key][sample_mapping] for key in doc_seqs], dim = 0)
                for key in doc_seqs
            }
            ids = [ids[idx] for idx in sample_mapping]
        
        return doc_seqs, ids

    def average_pool(self, last_hidden_states, attention_mask) : 
        # 모두 텐서
        # 리턴도 텐서
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_detailed_instruct(self, query, task_description) : 
        # 모두 str, 리턴도 str
        return f'Instruct: {task_description}\nQuery: {query}'

    def save_embeddings(self, embeddings_path) : 
        assert self.embeddings is not None, "Embeddings are None. You must calculate embeddings."
        assert self.document_mappings is not None, "Document mappings are None. You must calculate embeddings."
        torch.save({
            'embeddings' : self.embeddings,
            'document_mappings' : self.document_mappings
            }, embeddings_path)
    
    def load_embeddings(self, embeddings_path) :
        assert os.path.exists(embeddings_path), f"Embeddings are not in {embeddings_path}."
        embeddings = torch.load(embeddings_path)

        self.embeddings = embeddings['embeddings']
        self.document_mappings = embeddings['document_mappings']
