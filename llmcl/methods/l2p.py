#!/usr/bin/env python
import deepspeed, math, json
from typing import Dict
import torch.distributed
import torch.nn as nn
import numpy as np
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
from copy import deepcopy
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def l2_normalize(x, dim=None, epsilon=1e-12):
    square_norm = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_norm, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm

def convert_L2P_model(model):
    def init_new_prompt(pool_size,prompt_len):
        embed_tokens = model.model.embed_tokens
        N = embed_tokens.weight.shape[0]
        prompt_weigths = []
        for t in range(pool_size):
            prompt_weight = []
            for i in range(prompt_len):
                with torch.no_grad():
                    j = np.random.randint(N) # random token
                    w = deepcopy(embed_tokens.weight[j].detach().cpu().numpy())
                    prompt_weight.append(w)
            prompt_weigths.append(prompt_weight)

        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths
    model.model.prompt = nn.Parameter(torch.tensor(init_new_prompt(400, 500),requires_grad=True))
    return model

class L2PTrainer:
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

        self.top_k:int=5
        self.pool_size:int=30
        self.pull_constraint_coeff:float=0.5
        self.batch_wise=False

         
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
        self.model = convert_L2P_model(self.model)
        if not hasattr(self, "prompt_mean"):
            self.prompt_mean = torch.mean(self.model.model.prompt, dim=1).to(f"cuda:{torch.distributed.get_rank()}") # bs, h_dim
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
                attn_masks = batch['attention_mask']
                labels = batch['labels']
                """
                self.model: DeepSpeedEngine
                self.model.module: PeftModel
                self.model.module.model: Qwen2ForCasualLM
                self.model.module.model.model: Qwen2Model
                """

                embed_tokens = self.model.module.model.model.embed_tokens 
                input_embeddings = embed_tokens(batch['input_ids'])
                input_embeddings_mean = input_embeddings.mean(dim=1) # bs, h_dim

                prompt_norm = l2_normalize(self.prompt_mean, dim=1) # pool, h_dim
                input_embeddings_norm = l2_normalize(input_embeddings_mean,dim=1) # bs, h_dim
                prompt_norm = prompt_norm.to(input_embeddings_norm.dtype)
                similarity = torch.matmul(input_embeddings_norm, prompt_norm.t()) # bs, pool
                
                _, idx = torch.topk(similarity, k=self.top_k)
                if self.batch_wise:
                    idx = idx.to(torch.float32)
                    prompt_id, id_counts = torch.unique(idx, return_counts=True)
                    idx = idx.to(torch.bfloat16)
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([
                            prompt_id,
                            torch.full((self.pool_size - prompt_id.shape[0],),
                            torch.min(idx.flatten()).item(), device=prompt_id.device
                        )])
                        id_counts = torch.cat([
                            id_counts,
                            torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device
                        )])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(input_embeddings.shape[0], -1) # B, top_k
                
                batched_prompt_raw = self.model.module.model.model.prompt[idx] # B, top_k, length, C
                batch_size, top_k, length, c = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C
                inputs_embeds = torch.cat([batched_prompt, input_embeddings],axis=1)
                
                prefix_length = batched_prompt.shape[1]
                attn_masks = torch.concat((torch.tensor(1).to(attn_masks).repeat(batch_size,prefix_length),attn_masks), axis=1)
                labels = torch.concat((labels[0][0].repeat(batch_size,inputs_embeds.shape[1]-labels.shape[1]),labels),axis=1)
                outputs = self.model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attn_masks, use_cache=False)
                loss = outputs[0]
                
                batched_key_norm = prompt_norm[idx]
                input_embeddings_norm = input_embeddings_norm.unsqueeze(1) # B, 1, C
                sim = batched_key_norm * input_embeddings_norm # B, top_k, C
                reduce_sim = torch.sum(sim) / inputs_embeds.shape[0] # Scalar

                loss -= reduce_sim * self.pull_constraint_coeff

                
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