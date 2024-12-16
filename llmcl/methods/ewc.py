from typing import Union, Dict
import tqdm, logging, torch
import torch.distributed as dist
from .vanilla import VanillaTrainer
from typing_extensions import override
from peft import get_peft_model, PeftModel
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, PreTrainedTokenizerBase
from llmcl.train.get_args import CLTrainingArguments
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EWCTrainer(VanillaTrainer):
    def __init__(self, model:Union[torch.nn.Module, AutoModel],
                 datasets:Dict[str, Dataset],
                 args:CLTrainingArguments,
                 tokenizer:PreTrainedTokenizerBase
                ):
        super().__init__(model, datasets, args, tokenizer)
        self.ewc_lambda = self.args.ewc_lambda
        self.prior = {}
        self.fisher = {}
        self.ewc_loss:float = -1

    def _init_model(self):
        assert not isinstance(self.model, PeftModel)
        self.model = get_peft_model(self.model, peft_config=self.args.lora_config)
        logger.info("** model initilized!")
        if isinstance(self.model, PeftModel):
            self.model.print_trainable_parameters()
        self.prior = {n: p.detach().clone().to(self.args.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p.data.detach().clone(), device=self.args.device, dtype=p.dtype) for n, p in self.model.named_parameters() if p.requires_grad}
    
    def compute_ewc_reg_loss(self):
        ewc_reg_loss = 0
        for n, p in self.model.module.named_parameters():
            if n in self.fisher:
                ewc_reg_loss += (self.fisher[n] * (safe_get_full_fp32_param(p) - self.prior[n]).pow(2)).sum() * self.ewc_lambda / 2
        self.ewc_loss = ewc_reg_loss.item()
        return ewc_reg_loss 
    
    @override 
    def _at_task_end(self):
        for n, p in self.model.module.named_parameters():
            if p.requires_grad:
                self.prior[n] = p.detach().clone().data.to(self.args.device)
     
    @override
    def _at_back_propagation(self, task:str):
        for n, p in self.model.module.named_parameters():
            if n in self.fisher and p.requires_grad:
                self.fisher[n] += safe_get_full_grad(p).detach().clone().data.to(self.args.device) / len(self.dataloaders[task])

    @override
    def train_task(self, task_name:str, dataloader:DataLoader):
        tqdm_bar = tqdm.tqdm(self.update_steps, desc=f"Training {task_name}", disable=not self.args.local_rank in [-1, 0])  

        for epoch in range(int(self.args.num_train_epochs)):
            if self.args.local_rank in [-1, 0]:
                logger.info(f"** Train on {task_name}")

            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                model_outputs = self.model(**batch)
                loss = model_outputs.loss
                loss += self.compute_ewc_reg_loss()

                if self.args.global_rank == 0:
                    tqdm_bar.update(1)
                    description = f"Epoch {epoch+1}/{int(self.args.num_train_epochs)}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}, EWC Reg Loss: {self.ewc_loss:.4f} lr: {self.lr_scheduler.get_lr()}"
                    tqdm_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                self._at_back_propagation(task_name)
                self.model.step()
                
            self.save_model(task_name, epoch)
        self._at_task_end()
            
    @override 
    def continual_learning(self):
        self._init_train_dataloader()
        self._init_model()
        self._initilize_deepspeed()
        for i, (task_name, train_loader) in enumerate(self.dataloaders.items()):
            self.train_task(task_name, train_loader)