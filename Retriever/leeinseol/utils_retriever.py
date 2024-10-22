import logging
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
from transformers import is_torch_available


def set_seed(seed: int = 42, deterministic = False) :
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available() :
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic : 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def setup_logger(output_dir, log_file_name = "train.log", print_cli = True, print_file = True) :
    os.makedirs(output_dir, exist_ok=True) # 폴더가 없으면 생성
    log_file = os.path.join(output_dir, log_file_name)

    # logger 생성
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # format 설정
    formatter = logging.Formatter("%(asctime)s - %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S")

    # handler 생성
    if not logger.hasHandlers() :
        if print_cli : 
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        
        if print_file : 
            file_handler = logging.FileHandler(log_file, mode = 'w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
    
@contextmanager # with 구문과 함께 사용 가능
def timer(logger, name) :
    start = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - start:.2f} seconds')

def get_docs_id(corpus, ground_context) :
    context_to_id = {context: id for id, context in corpus.items()}

    doc_ids = []
    for context in ground_context :
        doc_id = context_to_id.get(context, None)
        if doc_id is not None :
            doc_ids.append(doc_id)
        else :
            doc_ids.append("Not Found")
    return doc_ids