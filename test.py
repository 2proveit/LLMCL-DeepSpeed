import os
import torch
import deepspeed
from deepspeed.utils import safe_get_full_grad
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 配置模型和数据
MODEL_NAME = "/ssd/xumingzhou/checkpoints/official_model/Qwen2.5-7B-Instruct"
BATCH_SIZE = 1
SEQ_LEN = 128

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=SEQ_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

# 准备数据
texts = [
    "DeepSpeed makes training large models easy!",
    "LoRA reduces the number of trainable parameters effectively.",
]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
dataset = CustomDataset(tokenizer, texts)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 配置 LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 修改为 Qwen 的具体模块
    lora_dropout=0.1,
    bias="none",
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu"
)

model = get_peft_model(model, lora_config)

# DeepSpeed 配置
ds_config = {
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "gradient_accumulation_steps": 1,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2},  # ZeRO stage 2
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    },
}

param_names = [n for n, p in model.named_parameters() if p.requires_grad]
print(param_names)
# 初始化 DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)
# 训练循环
model_engine.train()
for step, batch in enumerate(dataloader):
    # 将数据移动到模型所在设备
    inputs = {key: value.to(model_engine.local_rank) for key, value in batch.items()}
    
    # 前向传播
    outputs = model_engine(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    # 反向传播和参数更新

    model_engine.backward(loss)
    model_engine.step()
    for n, p in model_engine.module.named_parameters():
        if p.requires_grad:
            p_grad = safe_get_full_grad(p)
            print(p_grad)
    print(f"Step {step}: Loss = {loss.item()}")
    
    # 只训练一步
    if step == 0:
        break