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


class ResMLP(torch.nn.Module):
    def __init__(self, 
                 bottleneck_size,
                 module_type='MLP2',
                 emb_dimension=512,
                 residual=True,
                 ):
        """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
        Args:
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used. 
                Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer'). 
                Defaults to 'MLP1'.
            emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 512 (T5-small embedding dimension).
            residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
        """
        super().__init__()
        if module_type=='MLP1':
            # if layer_norm:
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size, bias=False),
                nn.ReLU(),
                nn.Linear(bottleneck_size, emb_dimension, bias=False),
                nn.LayerNorm(emb_dimension),
            )
        elif module_type=='MLP2':
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, bottleneck_size // 2),
                nn.Tanh(),
                nn.Linear(bottleneck_size // 2, emb_dimension),
            )

        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs:torch.Tensor):
        if self.residual:
            return self.module(inputs).to(inputs.dtype) + inputs
        else:
            return self.module(inputs).to(inputs.dtype)



class PPTrainer:
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

        self.num_task = len(datasets)
        self.mlps = nn.ModuleList([ResMLP(128, 'MLP1', model.config.hidden_size, residual=True) for _ in range(self.num_task)]).to("cuda")
        self.freeze_weights = True
        self.seq_len = 512
        self.prefix_len = 512
        self.embed_tokens = self.model.model.embed_tokens
        self.embed_tokens_dim = self.model.model.embed_tokens.weight.shape[0]
        self.embed_tokens_length = self.model.model.embed_tokens.weight.shape[1]
        self.prompt = nn.Parameter(
            torch.tensor(self.init_prompt(), requires_grad=True)
        ).to('cuda')
        self.previous_prompts = torch.zeros([0, self.prompt.shape[1]], requires_grad=False, dtype=torch.bfloat16).to('cuda') # [0, embed_dim] 
        self.mlps = nn.ModuleList(
            [ResMLP(128, 'MLP1', model.config.hidden_size, residual=True) for _ in range(self.num_task)]
            ).to('cuda')
    
    def init_prompt(self):
        prompt_weights = []
        for i in range(self.prefix_len):
            with torch.no_grad():
                j = np.random.randint(self.embed_tokens_length)
                w = deepcopy(self.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weights.append(w / 100)
        return np.array(prompt_weights)     
    
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
        eval_tqdm_bar = tqdm.tqdm(total=len(eval_dataloader), desc=f"Evalating...", disable=not self.args.local_rank in [-1, 0]) 
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

    @torch.no_grad()
    def update_prompt(self, task_index:int):
        new_prompt = self.mlps[task_index](self.prompt)
        self.previous_prompts = torch.cat((self.previous_prompts, new_prompt), dim=0)
        logger.info("update prompt!")
    
    def freeze_mlps(self, task_idx:int,):
        for i, mlp in enumerate(self.mlps):
            if i != task_idx:
                for p in mlp.parameters():
                    p.requires_grad = False
            else:
                for p in mlp.parameters():
                    p.requires_grad = True
                    

    def train_task(self, task_name:str, dataloader:DataLoader):
        tqdm_bar = tqdm.tqdm(self.update_steps, desc=f"Training {task_name}", disable=not self.args.local_rank in [-1, 0])  
        task_step = -1
        task_inex = list(self.datasets.keys()).index(task_name)
        for epoch in range(int(self.args.num_train_epochs)):
            if self.args.local_rank in [-1, 0]:
                logger.info(f"** Train on {task_name}")

            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                attn_mask = batch['attention_mask']
                labels = batch['labels']
                """
                self.model: DeepSpeedEngine
                self.model.module: PeftModel
                self.model.module.model: Qwen2ForCasualLM
                self.model.module.model.model: Qwen2Model
                """
                
                input_embeddings = self.embed_tokens(batch['input_ids'])
                batch_size = input_embeddings.shape[0] # batch_size

                mlp = self.mlps[task_inex]
                prompt = mlp(self.prompt)  #[prefix_len, embed_dim]
                
                input_embeddings = torch.cat(
                    [prompt.unsqueeze(0).repeat(batch_size, 1, 1), # [batch_size, prefix_len, embed_dim]
                        self.previous_prompts.unsqueeze(0).repeat(batch_size, 1, 1), # [batch_size, len_of_learned_tasks, embed_dim]
                        input_embeddings
                    ], axis=1)# [batch_size, seq_len, embed_dim]
                full_prefix_len = prompt.shape[0] + self.previous_prompts.shape[0] # prefix_len + len_of_learned_tasks
                
                attn_mask = torch.cat(
                    (torch.tensor(1).to('cuda').repeat(batch_size, full_prefix_len), attn_mask),
                    axis=1) # [batch_size, prefix_len + learned_tasks_len]
                labels = torch.concat((labels[0][0].repeat(batch_size, input_embeddings.shape[1] - labels.shape[1]), labels),axis=1).detach()#[batch_size, prefix_len + learned_tasks_len, embed_dim]

                input_embeddings = input_embeddings.to(torch.bfloat16)
                outputs = self.model(inputs_embeds=input_embeddings, labels=labels, attention_mask=attn_mask, use_cache=False)
                loss = outputs[0]
                
                
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
            self.freeze_mlps(i)
            self.train_task(task_name, train_loader)
            self.update_prompt(i)
        self._log_hparams()
        self.writer.close()