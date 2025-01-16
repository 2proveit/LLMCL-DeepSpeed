from typing import Union, Dict
import torch.distributed
import tqdm, logging, torch
import torch.distributed as dist
from .vanilla import VanillaTrainer
from qpth.qp import QPFunction
from typing_extensions import override
from peft import get_peft_model, PeftModel
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, PreTrainedTokenizerBase
from llmcl.train.get_args import CLTrainingArguments
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AGEMTrainer(VanillaTrainer):
    def __init__(self, model: Union[torch.nn.Module, AutoModel],
                 datasets: Dict[str, Dataset],
                 args: CLTrainingArguments,
                 tokenizer: PreTrainedTokenizerBase,
                 **kwargs):
        super().__init__(model, datasets, args, tokenizer, **kwargs)
        self.all_grads = {} 
        self.task_idx:int = -1
        self.task_name:str = ''
        
    
    def grad_proj(self, name, grad, margin=1.0, eps=1e-4):
        ori_shape = grad.shape
        grad_dtype = grad.dtype
        grad = grad.view(-1).to(torch.float32)
        pre_grad = self.all_grads[name].cuda().to(torch.float32)
        grad, pre_grad = grad.unsqueeze(1), pre_grad.unsqueeze(1)
        dot_product = torch.mm(grad.t(), pre_grad)
        
        if (dot_product < 0):
            new_grad = grad - (torch.mm(grad.t(), pre_grad) + eps) / (torch.mm(pre_grad.t(), pre_grad) + eps) * pre_grad
            grad.copy_(new_grad)
            
        return grad.view(ori_shape).to(grad_dtype)
                
    def save_grad(self, name):
        def hook(grad):
            grad = torch.nan_to_num(grad, nan=0.0) 
            self.all_grads[name] += grad.detach().clone().view(-1).to(self.args.device)
            return self.grad_proj(name, grad)
        return hook
    
    def retain_grad(self):
        for name, param in self.model.named_parameters():
            if name in self.all_grads and param.requires_grad:
                param.register_hook(self.save_grad(name))
    
    def _init_model(self):
        assert not isinstance(self.model, PeftModel)
        self.model = get_peft_model(self.model, peft_config=self.args.lora_config)
        logger.info("** model initialized!")
        if isinstance(self.model, PeftModel):
            self.model.print_trainable_parameters()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.all_grads[n] = torch.zeros([p.data.numel()], dtype=torch.float16, device=self.args.device)
        self.retain_grad()


    @override
    def train_task(self, task_name:str, dataloader:DataLoader):
        tqdm_bar = tqdm.tqdm(self.update_steps, desc=f"Training {task_name}", disable=not self.args.local_rank in [-1, 0])  
        task_step = -1

        for epoch in range(int(self.args.num_train_epochs)):
            if self.args.local_rank in [-1, 0]:
                logger.info(f"** Train on {task_name}")

            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                model_outputs = self.model(**batch)
                loss = model_outputs.loss

                task_step += 1
                self.global_steps += 1
                self.writer.add_scalar(f'Train/Loss/{task_name}', loss.item(), task_step)
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_steps)
                self.writer.add_scalar('Lr', self.lr_scheduler.get_lr()[0], self.global_steps)

                if self.args.do_eval and (step+1)%self.args.eval_steps==0:
                    eval_loss = self.eval_step(self.eval_dataloaders[task_name])
                    if hasattr(self, "wrter"):
                        self.writer.add_scalar("Eval/Loss", eval_loss, self.global_steps)

                if self.args.global_rank == 0:
                    tqdm_bar.update(1)
                    description = f"Epoch {epoch+1}/{int(self.args.num_train_epochs)}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}, lr: {self.lr_scheduler.get_lr()}"
                    tqdm_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                self.model.step()

            self.save_model(task_name, epoch)
            
    @override 
    def continual_learning(self):
        self._init_train_dataloader()
        self._init_model()
        self._initilize_deepspeed()
        for i, (task_name, train_loader) in enumerate(self.dataloaders.items()):
            self.task_idx = i
            self.task_name= task_name
            self.train_task(task_name, train_loader)
        self._log_hparams()
        self.writer.close()
