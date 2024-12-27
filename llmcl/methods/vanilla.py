#!/usr/bin/env python
import deepspeed, math, json
from typing import Dict
import torch.distributed
import torch, logging, tqdm, os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from llmcl.utils import save_model_tokenizer, get_grouped_parameters
from typing import Union
from transformers import AutoModel, get_cosine_schedule_with_warmup,get_constant_schedule_with_warmup, PreTrainedTokenizerBase
from deepspeed.ops.adam import FusedAdam
from datasets import Dataset
from llmcl.train.get_args import CLTrainingArguments
from llmcl.get_dataset import Collector
from peft import get_peft_model, PeftModel
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VanillaTrainer:
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
        for task, dataset in self.datasets.items():
            sampler = DistributedSampler(dataset)
            self.dataloaders[task] = DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=Collector()
            )
        if self.eval_datasets:
            for task, eval_dataset in self.eval_datasets.items():
                eval_sampler = DistributedSampler(eval_dataset)
                self.eval_dataloaders[task] = DataLoader(
                    dataset=eval_dataset,
                    sampler=eval_sampler,
                    batch_size=self.args.per_device_eval_batch_size,
                    collate_fn=Collector()
                )
            self.args.do_eval=True
        else:
            self.args.do_eval=False
    
    
    def _get_optim_lr_scheduler(self) -> tuple:
        """make sure you initilized the train loader and model for your current task"""
        optimize_grouped_params = self.model.parameters()
        optimizer = FusedAdam(
            optimize_grouped_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        assert self.dataloaders, "self.datloaders is null"
        self.update_steps = math.ceil(sum(len(train_loader) * self.args.num_train_epochs / self.args.gradient_accumulation_steps for train_loader in self.dataloaders.values()))
        lr_scheduler =  get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=max(10, int(self.update_steps * self.args.warmup_ratio)),
            num_training_steps=self.update_steps
        )
        return optimizer, lr_scheduler
        
        
    def _init_model(self) -> None:
        self.model = get_peft_model(self.model, peft_config=self.args.lora_config)
        logger.info("** model initilized!")
        if isinstance(self.model, PeftModel):
            self.model.print_trainable_parameters()
    
    def _log_hparams(self):
        if not hasattr(self, "writer"):
            self.writer = SummaryWriter(log_dir=self.args.logging_dir)
        self.writer.add_hparams(
            hparam_dict=dict(
                train_micro_batch_size_per_gpu = self.args.per_device_train_batch_size,
                gradient_accumulation_steps = self.args.gradient_accumulation_steps,
                train_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size,
                warmup_ratio = self.args.warmup_ratio,
                learning_rate = self.args.learning_rate,
                num_train_epochs = self.args.num_train_epochs,
                update_steps = self.update_steps,
                num_tasks = len(self.dataloaders),
            ),
            metric_dict=dict(loss=0.0),
        )

    
    def _initilize_deepspeed(self) -> None:
        optimizer, lr_scheduler = self._get_optim_lr_scheduler()

        with open(self.args.deepspeed_config, 'r') as f:
            ds_config = json.loads(f.read())
        ds_config["train_micro_batch_size_per_gpu"]= self.args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"]= self.args.gradient_accumulation_steps
        ds_config['train_batch_size'] = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        ds_config['steps_per_print'] = self.args.logging_steps

        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config,
            dist_init_required=True,
        )
        logger.info("** deepspeed initilized!")
        
    def _at_task_begin(self, task:str): 
        pass

    def _at_task_end(self, task:str):
        pass

    def _at_back_propagation(self, task:str):
        pass
             
    def save_model(self, task:str, epoch:int, desc:str=''):
        save_dir = os.path.join(self.args.output_dir, f"{self.args.cl_method}_{task}_round_{epoch}_desc_{desc}")
        logger.info(f"Saving to: {save_dir}")
        if self.args.local_rank in [-1, 0]:
           save_model_tokenizer(model=self.model, tokenizer=self.tokenizer, output_dir=save_dir) 
        # TODO: check if it is zero-3 stage 
    
    @torch.no_grad()
    def eval_step(self, eval_dataloader:DataLoader):
        eval_tqdm_bar = tqdm.tqdm(self.update_steps, desc=f"Evalating...", disable=not self.args.local_rank in [-1, 0]) 
        eval_loss  = 0
        for eval_step, eval_batch in enumerate(eval_dataloader):
            eval_batch = {k:v.to(self.args.device) for k, v in eval_batch.items()}
            eval_model_outputs = self.model(**eval_batch)
            eval_loss += eval_model_outputs.loss
            if self.args.local_rank == 0:
                eval_tqdm_bar.update(1)

        eval_loss /= len(eval_dataloader)
        torch.distributed.all_reduce(eval_loss, torch.distributed.ReduceOp.SUM)
        eval_loss /= torch.distributed.get_world_size()
        return eval_loss.item()


    def train_task(self, task_name:str, dataloader:DataLoader):
        tqdm_bar = tqdm.tqdm(self.update_steps, desc=f"Training {task_name}", disable=not self.args.local_rank in [-1, 0])  
        task_step = -1
        for epoch in range(int(self.args.num_train_epochs)):
            if self.args.local_rank in [-1, 0]:
                logger.info(f"** Train on {task_name}")

            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                model_outputs = self.model(**batch)
                # print(f"rank: {self.args.local_rank}: logits: {model_outputs.logits}")
                # print(f"rank: {self.args.local_rank}, labels: {batch['labels']}")
                loss = model_outputs.loss
                task_step += 1
                self.global_steps += 1
                self.writer.add_scalar(f'Train/Loss/{task_name}', loss.item(), task_step)
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_steps)
                self.writer.add_scalar(f'Lr', self.lr_scheduler.get_lr()[0], self.global_steps)

                if self.args.do_eval and (step+1)%self.args.eval_steps == 0:
                    eval_loss = self.eval_step(self.eval_dataloaders[task_name])
                    if hasattr(self, "writer"):
                        self.writer.add_scalar("Eval/Loss", eval_loss, self.global_steps)

                if self.args.global_rank == 0:
                    tqdm_bar.update(1)
                    description = f"Epoch {epoch+1}/{int(self.args.num_train_epochs)}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}, lr: {self.lr_scheduler.get_lr()}"
                    tqdm_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                self.model.step()
            self.save_model(task_name, epoch)
            
             
    def continual_learning(self):
        self._init_train_dataloader()
        self._init_model()
        self._initilize_deepspeed()
        for i, (task_name, train_loader) in enumerate(self.dataloaders.items()):
            self.train_task(task_name, train_loader)
        self._log_hparams()
        self.writer.close()