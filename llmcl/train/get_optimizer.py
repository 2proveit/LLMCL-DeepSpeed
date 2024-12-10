import torch, math
from torch.utils.data import DataLoader
from deepspeed import DeepSpeedEngine, PipelineEngine
from torch.optim import AdamW, Adam
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup
from typing import Dict

def get_grouped_parameters(model: torch.nn.Module, args):
    """
    Group model parameters for optimization based on weight decay settings.

    Args:
        model (torch.nn.Module): The model containing parameters to group.
        args: Argument namespace containing weight decay configuration.

    Returns:
        List[Dict]: A list of dictionaries with grouped parameters.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

def get_optimizer(model:torch.nn.Module, args, dataloaders:Dict[str, DataLoader]):
    grouped_optimizer_parameters = get_grouped_parameters(model, args)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(
        grouped_optimizer_parameters,
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    # lr_scheduler 
    train_loader_total_len = sum(len(dl) for dl in dataloaders.values())
    num_update_steps_per_epoch = math.ceil(train_loader_total_len / args.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    return optimizer, lr_scheduler