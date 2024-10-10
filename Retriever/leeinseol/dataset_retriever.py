from transformers import AutoTokenizer 
from torch.utils.data import TensorDataset

def get_dataset_to_tensor(dataset, tokenizer, data_args, train = True) :
    q_seqs = tokenizer(dataset['question'],
                       padding = data_args.padding,
                       truncation = data_args.truncation,
                       return_tensors = data_args.return_tensors)
    
    if train :
        p_seqs = tokenizer(dataset['context'],
                           padding = data_args.padding,
                           truncation = data_args.truncation,
                           return_tensors = data_args.return_tensors)
        return TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
                             q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
    else :
        return TensorDataset(q_seqs['input_ids'], 
                             q_seqs['attention_mask'], 
                             q_seqs['token_type_ids'])

def get_all_context_to_tensor(documents, tokenizer, data_args) :
    p_seqs = tokenizer([doc['context'] for doc in documents],
                       padding = data_args.padding,
                       truncation = data_args.truncation,
                       return_tensors = data_args.return_tensors)
    return TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'])

    # token_type_ids를 생성하지 않는 토크나이저는 이걸로 해야함. 
    # return TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'],
    #                      p_seqs.get('token_type_ids', torch.zeros_like(p_seqs['input_ids'])))