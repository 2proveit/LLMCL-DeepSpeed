#!/usr/bin/env python
from ast import Dict, arg
from re import escape
import deepspeed
from regex import D
from sympy import im
import torch, logging, tqdm, os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, DataLoader
from llmcl.utils import save_model_tokenizer
from typing import Union
from transformers import AutoModel, DataCollatorForSeq2Seq
from datasets import Dataset
from ..get_args import CLTrainingArguments
from peft import get_peft_model
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VanillaTrainer:
    def __init__(self, model:Union[torch.nn.Module, AutoModel], optimizer, lr_scheduler, datasets:Dict[str, Dataset], args:CLTrainingArguments):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.datasets = datasets
        self.args = args
        
         
    def _process_train_dataloader(self, dataset:Dataset) -> DataLoader:
        """process dataloader for each task"""
        if self.args.world_size > 1:
            sampler = DistributedSampler(dataset)
        else:
            sampler = RandomSampler(dataset)
        return DataLoader(sampler=sampler, batch_size=self.args.per_device_train_batch_size, collate_fn=DataCollatorForSeq2Seq(tokenizer=self.tokenizer))
    
    def _process_model(self, task:str, task_idx:int) -> None:
        if task_idx == 0:
            self.model =  get_peft_model(self.model, peft_config=self.args.lora_config)

        deepspeed.init_distributed()
        
        
    def _at_task_begin(self, task:str): 
        pass

    def _at_task_end(self, task:str):
        pass

    def _at_back_propagation(self, task:str):
        pass
             
    def save_model(self, task:str, epoch:int):
        if self.args.output_dir and self.args.local_rank in [-1, 0]:
            self.args.output_dir = os.path.join(self.args.output_dir, f"{task}_round_{epoch}")
            logger.info(f"Saving to: {self.args.local_rank}")
        if self.args.local_rank in [-1, 0]:
           save_model_tokenizer(self.model, self.tokenizer, self.args) 
        # TODO: check if it is zero-3 stage 
     
    def train_task(self, task_name:str, dataloader:DataLoader):
        update_steps = len(dataloader) * self.args.num_train_epoch
        tqdm_bar = tqdm.tqdm(update_steps, desc=f"Training {task_name}", disable=not self.args.local_rank in [-1, 0])  

        for epoch in range(self.args.num_train_epoch):
            if self.args.local_rank in [-1, 0]:
                logger.info(f"** Train on {task_name}")

            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                model_outputs = self.model(**batch)
                loss = model_outputs.loss

                if self.args.global_rank == 0:
                    tqdm_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    tqdm_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                self.model.step()
            self.save_model(task_name, epoch)
            
             
    def continual_learning(self):
        for i, (task_name, dataset) in enumerate(self.datasets.items()):
            self._process_model(task_name, task_idx=i)
            train_loader = self._process_train_dataloader(dataset)
            self.train_task(task_name, train_loader)