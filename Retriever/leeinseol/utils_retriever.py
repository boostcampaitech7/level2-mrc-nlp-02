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

def setup_logger(output_dir, log_file_name = "train.log", print_cli = True) :
    os.makedirs(output_dir, exist_ok=True) # 폴더가 없으면 생성
    log_file = os.path.join(output_dir, log_file_name)

    # logger 생성
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # handler 생성
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file, mode = 'w')

    # format 설정
    formatter = logging.Formatter("%(asctime)s - %(message)s",
                                  datefmt="%m/%d/%Y %H:%M:%S")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # logger에 handler 추가
    if print_cli :
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
    
@contextmanager # with 구문과 함께 사용 가능
def timer(logger, name) :
    start = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - start:.2f} seconds')

def get_docs_id(all_context, ground_context) :
    context_to_id = {doc['context'] : doc['id'] for doc in all_context}

    doc_ids = []
    for context in ground_context :
        doc_id = context_to_id.get(context, None)
        if doc_id is not None :
            doc_ids.append(doc_id)
        else :
            doc_ids.append("Not Found")
    return doc_ids