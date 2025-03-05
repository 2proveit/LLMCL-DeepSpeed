#!/usr/bin/env python
from typing import Dict
import torch.distributed
import torch, logging, os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Union
from transformers import AutoModel, PreTrainedTokenizerBase
from .vanilla import VanillaTrainer
from datasets import Dataset, concatenate_datasets
from llmcl.train.get_args import CLTrainingArguments
from llmcl.get_dataset import Collector
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiTaskTrainer(VanillaTrainer):
    def __init__(self, model:Union[torch.nn.Module, AutoModel],
                 datasets:Dict[str, Dataset], 
                 args:CLTrainingArguments, 
                 tokenizer:PreTrainedTokenizerBase,
                 **kwargs
                ):
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.eval_datasets = kwargs.get("eval_datasets", None)
        self.args = args
        self.optimizer = None
        self.lr_scheduler = None
        self.train_loader = None
        self.dataloaders:Dict[str, DataLoader] = {}
        self.eval_dataloaders:Dict[str, DataLoader] = {}
        self.update_steps:int=-1
        self.global_steps:int=-1
        
         
    def _init_train_dataloader(self) -> None:
        """process dataloader for each task"""
        self.writer = SummaryWriter(log_dir=self.args.logging_dir)
        mtl_dataset = concatenate_datasets(self.datasets.values())
        sampler = DistributedSampler(mtl_dataset)

        self.dataloaders['MTL'] = DataLoader(
            dataset=mtl_dataset,
            sampler=sampler,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=Collector(self.tokenizer)
            )

        if self.eval_datasets:
            mtl_eval_dataset = concatenate_datasets(self.eval_datasets.values())
            eval_sampler = DistributedSampler(mtl_eval_dataset)
            self.eval_dataloaders['MTL'] = DataLoader(
                dataset=mtl_eval_dataset,
                sampler=eval_sampler,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=Collector(self.tokenizer)
            )
            self.args.do_eval=(True and self.args.do_eval)
        else:
            self.args.do_eval=False