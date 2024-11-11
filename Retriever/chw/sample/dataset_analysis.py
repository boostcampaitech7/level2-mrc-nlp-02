import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, NoReturn, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)

dataset_name = "./data/train_dataset"
data_path = "./data"
context_path = "wikipedia_documents.json"

with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
    wiki = json.load(f)

contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

values = [v["text"] for v in wiki.values()]
print(wiki.keys())

# print(values)
print(wiki["0"].keys())
