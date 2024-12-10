from curses.panel import update_panels
import dis
from re import S
from turtle import up
import tqdm, logging, torch
import torch.distributed as dist
from .vanilla import VanillaTrainer
from typing_extensions import override
from datasets import Dataset
from torch.utils.data import DataLoader
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EWCTrainer(VanillaTrainer):
    def __init__(self, model, optimizer, lr_scheduler, datasets, args):
        super().__init__(model, optimizer, lr_scheduler, datasets, args)
        self.ewc_lambda = args.ewc_lambda
        self.ewc_task_name

        self.fisher = {}
        self.prior = {}
        self.gloabl_step = 0

    def compute_ewc_reg_loss(self):
        ewc_reg_loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                ewc_reg_loss += (self.fisher[n].cuda() * (p - self.prior[n].cuda()).pow(2)).sum() * self.ewc_lambda / 2

        dist.all_reduce(ewc_reg_loss, op=dist.ReduceOp.SUM) 
        ewc_reg_loss /= dist.get_world_size()
        if ewc_reg_loss.item()>0 and dist.get_rank() == 0:
            logger.info(f"EWC Loss: {ewc_reg_loss.item()}")
        return ewc_reg_loss 
    
    @override
    def _at_task_begin(self, task):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.prior[n] = p.detach().clone().data.cpu()
     
    @override
    def _at_back_propagation(self, task, **kwargs):
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] += p.grad.detach().clone().data.cpu() / (self.gloabl_step * dist.get_world_size())

    @override
    def train_task(self, task_name:str, dataloader:DataLoader):
        update_steps = len(dataloader) * self.args.num_train_epoch
        tqdm_bar = tqdm.tqdm(update_steps, desc=f"Training {task_name}", disable=not self.args.local_rank in [-1, 0])  

        self._at_task_begin(task_name)
        
        for epoch in range(self.args.num_train_epoch):
            if self.args.local_rank in [-1, 0]:
                logger.info(f"** Train on {task_name}")

            for step, batch in enumerate(dataloader):
                self.gloabl_step += 1
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                model_outputs = self.model(**batch)
                loss = model_outputs.loss + self.compute_ewc_reg_loss()

                if self.args.global_rank == 0:
                    tqdm_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    tqdm_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                self._at_back_propagation(task_name)
                self.model.step()
            self.save_model(task_name, epoch)
        
        self._at_task_end(task_name)
            
             
    def continual_learning(self):
        self.fisher = {n: p.detach().clone().data.zero_() for n, p in self.model.named_parameters() if p.requires_grad}
        for i, (task_name, dataset) in enumerate(self.datasets.items()):
            self._process_model(task_name, task_idx=i)
            train_loader = self._process_train_dataloader(dataset)
            self.train_task(task_name, train_loader)