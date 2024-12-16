from copy import deepcopy
import math, os, deepspeed
from tokenize import TokenError
import torch
import random
import torch.distributed as dist
from transformers import set_seed, get_constant_schedule_with_warmup, PreTrainedTokenizerBase
from peft import PeftModel
import numpy as np
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader

def get_grouped_parameters(model: torch.nn.Module, weight_decay):
    """
    Group model parameters for optimization based on weight decay settings.
    Args:
        model (torch.nn.Module): The model containing parameters to group.
        args: Argument namespace containing weight decay configuration.
    Returns:
        List[Dict]: A list of dictionaries with grouped parameters.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
def set_all_seed(seed:int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model_tokenizer(model:DeepSpeedEngine, tokenizer:PreTrainedTokenizerBase, output_dir:str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(model, DeepSpeedEngine):
        model.module.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        raise NotImplementedError

def all_reduce_mean(item:torch.Tensor):
    item_copy = deepcopy(item.detach())
    dist.all_reduce(item_copy, op=dist.ReduceOp.AVG)

