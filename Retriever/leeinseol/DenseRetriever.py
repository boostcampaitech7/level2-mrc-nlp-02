import os 
from transformers import (
    BertModel, BertPreTrainedModel, 
    get_linear_schedule_with_warmup)
import torch.nn.functional as F
import torch
from tqdm import tqdm, trange
import faiss

# Retriever 정의 
class Retriever :
    def __init__(self, retriever_args, data_args, tokenizer, logger) :
        self.retriever_args = retriever_args
        self.data_args = data_args
        self.query_encoder = Encoder(retriever_args.encoder_name, 
                                     retriever_args.model_name_or_path)
        self.passage_encoder = Encoder(retriever_args.encoder_name, 
                                       retriever_args.model_name_or_path)
        self.tokenizer = tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = logger
        self.passage_embeddings = None
        self.indexer = None

    def train(self, training_args, train_loader, val_loader, wandb, fold) :
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
        self.query_encoder.zero_grad()
        self.passage_encoder.zero_grad()
        torch.cuda.empty_cache()

        self.logger.info("Start Training")
        # train_iterator = trange(int(training_args.num_train_epochs), desc = "Epoch")
        for epoch in range(training_args.num_train_epochs) :
            # epoch_iterator = tqdm(train_loader, desc="Iteration")

            for step, batch in enumerate(train_loader) :
                self.query_encoder.train()
                self.passage_encoder.train()

                batch = tuple(t.to(self.device) for t in batch)
                
                p_inputs = {'input_ids' : batch[0],
                            'attention_mask' : batch[1],
                            'token_type_ids' : batch[2]
                            }
                q_inputs = {'input_ids' : batch[3],
                            'attention_mask' : batch[4],
                            'token_type_ids' : batch[5]
                            }
                
                p_outputs = self.passage_encoder(**p_inputs)
                q_outputs = self.query_encoder(**q_inputs)

                # In-batch Negative 
                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                # (batch, emb_dim) x (emb_dim, batch) = (batch, batch)

                batch_size = sim_scores.size(0)
                targets = torch.arange(0, batch_size).long()
                targets = targets.to(self.device)

                sim_scores = F.log_softmax(sim_scores, dim = 1)

                loss = F.nll_loss(sim_scores, targets)
                # loss 시각화하기 위한 것 필요할 듯 

                # 한 번의 epoch에서 지정된 step마다 train loss 기록
                if step % training_args.logging_steps == 0:
                    self.logger.info(f"Fold {fold+1} Epoch [{epoch}/{training_args.num_train_epochs}] Step [{step}/{len(train_loader)}] train loss : {loss.item():.4f}")
                wandb.log({
                    f'train_loss_fold_{fold+1}' : loss.item()
                })
                wandb.log({
                    f'learning_rate_fold_{fold+1}' : scheduler.get_last_lr()[0]
                })

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.query_encoder.zero_grad()
                self.passage_encoder.zero_grad()
                global_step += 1
                torch.cuda.empty_cache()                
                
            # epoch이 끝날 때마다 val loss 기록 
            val_loss = self.evaluate(val_loader)
            self.logger.info(f"Fold {fold+1} Epoch [{epoch}/{training_args.num_train_epochs}] validation loss : {val_loss:.4f}")
            wandb.log({
                f'val_loss_fold_{fold+1}' : val_loss
            })

    def evaluate(self, val_loader) :
        self.passage_encoder.to(self.device)
        self.query_encoder.to(self.device)
        
        self.passage_encoder.eval()
        self.query_encoder.eval() 
        
        val_loss = 0
        with torch.no_grad() :    
            for step, batch in enumerate(val_loader) :
                
                batch = tuple(t.to(self.device) for t in batch)

                p_inputs = {'input_ids' : batch[0],
                            'attention_mask' : batch[1],
                            'token_type_ids' : batch[2]
                            }
                q_inputs = {'input_ids' : batch[3],
                            'attention_mask' : batch[4],
                            'token_type_ids' : batch[5]
                            }
                p_outputs = self.passage_encoder(**p_inputs)
                q_outputs = self.query_encoder(**q_inputs)

                # In-batch Negative 
                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                # (batch, emb_dim) x (emb_dim, batch) = (batch, batch)

                batch_size = batch[0].size(0)
                targets = torch.arange(0, batch_size).long()
                targets = targets.to(self.device)

                sim_scores = F.log_softmax(sim_scores, dim = 1)

                loss = F.nll_loss(sim_scores, targets)
                val_loss += loss.item()

        return val_loss / len(val_loader) 

    def calculate_embedding(self, loader) :
        embeddings = []
        self.passage_encoder.to(self.device)
        self.passage_encoder.eval()
        with torch.no_grad() :
            for step, batch in enumerate(loader) :
                inputs = {
                    'input_ids' : batch[0].to(self.device),
                    'attention_mask' : batch[1].to(self.device),
                    'token_type_ids' : batch[2].to(self.device)
                }
                outputs = self.passage_encoder(**inputs)
                embeddings.append(outputs.cpu())
                if step % 100 == 0 :
                    self.logger.info(f"Step [{step}/{len(loader)}] done.")
        
        embeddings = torch.cat(embeddings, dim = 0)
        self.passage_embeddings = embeddings

    def save_embedding(self, output_dir, file_name = "embeddings.pt") :
        file_path = os.path.join(output_dir, file_name)
        torch.save(self.passage_embeddings, file_path)

    def load_embedding(self, output_dir, file_name = "embeddings.pt") :
        file_path = os.path.join(output_dir, file_name)
        self.passage_embeddings = torch.load(file_path)
        
    def save_model(self, path) :
        torch.save({
            'query_encoder' : self.query_encoder.state_dict(),
            'passage_encoder' : self.passage_encoder.state_dict()
        }, path)
    
    def load_model(self, path) :
        checkpoint = torch.load(path)
        self.query_encoder.load_state_dict(checkpoint['query_encoder'])
        self.passage_encoder.load_state_dict(checkpoint['passage_encoder'])

    def build_faiss(self) :
        if self.retriever_args.use_faiss :
            num_clusters = self.retriever_args.num_clusters
            n_iter = self.retriever_args.n_iter

            emb_dim = self.passage_embeddings.shape[-1]
            index_flat = faiss.IndexFlatL2(emb_dim)

            # Clustering
            clus = faiss.Clustering(emb_dim, num_clusters)
            clus.verbose = True
            clus.niter = n_iter
            clus.train(self.passage_embeddings.numpy(), index_flat)
            
            # Retrieve cluster centroids 
            centroids = faiss.vector_float_to_array(clus.centroids)
            centroids = centroids.reshape(num_clusters, emb_dim)

            # Quantizer setup with centroids
            quantizer = faiss.IndexFlatL2(emb_dim)
            quantizer.add(centroids)

            # IVF + Scalar Quantizer indexer setup 
            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer,
                quantizer.d,
                num_clusters,
                faiss.METRIC_L2
            )

            # Training and adding embeddings
            self.indexer.train(self.passage_embeddings.numpy())
            self.indexer.add(self.passage_embeddings.numpy())

    def retrieve(self, queries, k) :
        assert self.passage_embeddings is not None, "There is no passage embeddings."

        q_seqs = self.tokenizer([query for query in queries], 
                               padding = self.data_args.padding,
                               truncation = self.data_args.truncation,
                               return_tensors = self.data_args.return_tensors).to(self.device)

        with torch.no_grad() :
            self.query_encoder.eval()
            self.query_encoder.to(self.device)
            q_embs = self.query_encoder(input_ids = q_seqs['input_ids'],
                                        attention_mask = q_seqs['attention_mask'],
                                        token_type_ids = q_seqs['token_type_ids']).to('cpu')
                                        
        torch.cuda.empty_cache()
        
        if self.retriever_args.use_faiss :
            q_embs = q_embs.numpy()
            D, I = self.indexer.search(q_embs, k)
            scores, indices = torch.tensor(D), torch.tensor(I)
        else :
            self.passage_embeddings = self.passage_embeddings.to(self.device)
            dot_prod_scores = torch.matmul(q_embs.to(self.device), self.passage_embeddings.T).to('cpu')
            scores, indices = torch.topk(dot_prod_scores, k, dim = 1)
            scores, indices = scores.squeeze(), indices.squeeze()

        return scores.tolist(), indices.tolist()
        

# 인코더 정의
class Encoder:
    def __new__(cls, encoder_name, config) :
        if encoder_name == "Bert" :
            return BertEncoder.from_pretrained(config)
        else :
            raise ValueError(f'Unknown {encoder_name} encoder. Setting another Retriever encoder.')


# 각 인코더 정의
class BertEncoder(BertPreTrainedModel) :
    def __init__(self, config) :
        super(BertEncoder, self).__init__(config) 

        self.bert = BertModel(config)
        # self.init_weights() # 가중치 초기화
        # 명시적으로 할 필요는 없어보임
    
    def forward(self, input_ids,
                attention_mask = None, token_type_ids = None) :
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1] # CLS 토큰에 해당하는 임베딩
    
        return pooled_output