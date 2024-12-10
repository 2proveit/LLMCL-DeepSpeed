from copy import deepcopy
import dis
import math, os
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


def set_all_seed(seed:int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model_tokenizer(model:DeepSpeedEngine, tokenizer:PreTrainedTokenizerBase, output_dir:str):
    model_save_group = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    # output_dir = os.path.join(output_dir, f"{task_name}_epoch_{epoch}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(model_save_group, os.path.join(output_dir, f"pytorch_model.bin"))
    model_save_group.config.json_file(os.path.join(output_dir, f"config.json"))
    tokenizer.save_pretrained(output_dir)

def all_reduce_mean(item:torch.Tensor):
    item_copy = deepcopy(item.detach())
    dist.all_reduce(item_copy, op=dist.ReduceOp.AVG)

