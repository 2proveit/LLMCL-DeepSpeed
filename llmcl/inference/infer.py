import os
import argparse
import json
import logging
from typing import Dict, List
import aiohttp
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from llmcl.get_dataset import get_datasets, Collector
from peft import get_peft_model
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, lora_adapter_path, tokenizer_path):
    lora_config_path = os.path.join(lora_adapter_path, 'adapter_config.json')
    if os.path.exists(lora_config_path):
        with open(lora_config_path, 'r') as f:
            lora_config = json.loads(f.read())
        base_model = AutoModel.from_pretrained(lora_config['base_model_name_or_path'])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        raise NotImplementedError
    lora_model = get_peft_model(base_model, peft_config=lora_config)

    return lora_model, tokenizer

    
async def async_chat(message: List[Dict], base_url:str, api_key:str, model_name:str, **generate_params):
    if isinstance(message, str):
        message = [{'role': 'user', 'content': message}]
    if isinstance(message, dict):
        message = [message]

    url = f"{base_url}chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": message,
        **generate_params
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response_data = await response.json()
            output = response_data["choices"][0]["message"]["content"]
            return output

def infer_task(model:AutoModelForCausalLM, dataset, task_name, save_path):
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=Collector())
    task_results = []
    for batch in dataloader:
        results = model.generate(**batch)
        task_results.extend(results)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{task_name}".jsonl), 'w') as f:
        for res in task_results:
            f.write(json.dumps(res, ensure_ascii=False))
    return task_results


def evaluate_result():
    pass


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--max_len')
    parser.add_argument('--lora_adapter_path')
    parser.add_argument('--tokenizer_path')
    
    return parser.parse_args()

def main():
    adapter_weight_paths = {
        
    }
    test_datasets = get_datasets(args, tokenizer=tokenizer)
    result_metrics = {}
    args = args_parse()
    for task_name, adapter_path in adapter_weight_paths.items():
        result_metrics[task_name] = {}
        logger.info(f"using {task_name} adapter weight!!")
        model, tokenizer = load_model(model_path='', lora_adapter_path=adapter_path, tokenizer_path=adapter_path)
        for dataset_name, dataset in test_datasets.items():
            task_results = infer_task(model=model, dataset=dataset, task_name=dataset_name, save_path=f"output/{task_name}_inference")

            metrics:float= evaluate_result(task_results)
            result_metrics[task_name][dataset_name] = metrics
            logger.info(f"dataset: {dataset_name}inference finished")
    