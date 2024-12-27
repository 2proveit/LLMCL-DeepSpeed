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


class GEMTrainer(VanillaTrainer):
    def __init__(self, model: Union[torch.nn.Module, AutoModel],
                 datasets: Dict[str, Dataset],
                 args: CLTrainingArguments,
                 tokenizer: PreTrainedTokenizerBase,
                 **kwargs):
        super().__init__(model, datasets, args, tokenizer, **kwargs)
        self.all_grads = {} 
        self.task_idx:int = -1
        self.task_name:str = ''
        
    
    def grad_proj(self, name, grad, idx, margin=1.0, eps=1e-4):
        ori_shape = grad.shape # None
        grad = grad.view(-1)
        grad_dtype = grad.dtype
        grad = grad.to(torch.float32) # torch.linalg not support bf16

        pre_grad = (self.all_grads[name].to(self.args.device)[:, :idx+1].to(torch.float32) / len(self.dataloaders[self.task_name])).clone()
        # torch.distributed.all_reduce(pre_grad, op=torch.distributed.ReduceOp.SUM) # TODO no need for deepspeed zero-2, only for zero-1
        # pre_grad /= torch.distributed.get_world_size()
        dot_product = torch.mm(grad.unsqueeze(0), pre_grad)
        if (dot_product < 0).sum() != 0:
            pre_grad_cuda = pre_grad.t()
            grad_cuda = grad.contiguous().view(-1)
            t = pre_grad_cuda.shape[0]
            P = torch.matmul(pre_grad_cuda, pre_grad_cuda.t())
            P = 0.5 * (P + P.t())
            
            P[torch.isnan(P)] = 0.0
            eigenvalues = torch.linalg.eigvals(P)
            if not (eigenvalues.real > 0).all(): # due to the grad clip happens after the projection, the grad could be huge, refactor eps=1.0 is reasonable
                P += torch.eye(t).cuda() * eps
            
            q = torch.matmul(pre_grad_cuda, grad_cuda).t() * -1

            P = P.to(torch.float32)
            q = q.to(torch.float32)
            G = torch.eye(t).cuda() * -1
            h = torch.zeros(t).cuda() - margin
            e = torch.Tensor().cuda()
            v = QPFunction(verbose=False)(P, q, G, h, e, e)[0]
            v = v.to(torch.float32)
            x = torch.matmul(v, pre_grad_cuda) + grad_cuda
            grad.copy_(x.view(-1))
        return grad.view(ori_shape).to(grad_dtype)
        
    def save_grad(self, name):
        def hook(grad):
            grad = torch.nan_to_num(grad, nan=0.0) 
            self.all_grads[name][:,self.task_idx] += grad.detach().clone().view(-1).to(self.args.device)
            return self.grad_proj(name, grad, idx=self.task_idx)
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
                self.all_grads[n] = torch.zeros([p.data.numel(), len(self.dataloaders)], dtype=torch.float16, device=self.args.device)
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
