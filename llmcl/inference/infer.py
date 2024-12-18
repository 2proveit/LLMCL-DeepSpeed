import os
import argparse
import json
import logging
import torch
import asyncio
from typing import Dict, List
import aiohttp
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, GenerationConfig
from peft import get_peft_model, PeftConfig, set_peft_model_state_dict, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GENERATION_CONFIG = GenerationConfig.from_dict({
    "max_new_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.75,
})


def collector(tokenizer: PreTrainedTokenizerBase, max_len: int = 2048):
    tokenizer.padding_side = 'left'
    def collect(batch: List[dict]):
        prompts = ["Input\n" + data.get('prompt') + "\nOutput:\n" for data in batch]
        tokenized = tokenizer(prompts, return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'prompts': prompts,
            'answers': [a.get('answer') for a in batch]
        }
    
    return collect

def load_dataset(data_files:str):
    with open(data_files, 'r') as f:
        data = json.loads(f.read())
    data = Dataset.from_list(data)
    return data

def load_model(model_path, lora_adapter_path, tokenizer_path):
    lora_config_path = os.path.join(lora_adapter_path, 'adapter_config.json')
    if os.path.exists(lora_config_path):
        lora_config = PeftConfig.from_pretrained(lora_adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(lora_config.base_model_name_or_path, device_map="balanced", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        raise NotImplementedError
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

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

@torch.no_grad()
def infer_task(model:AutoModelForCausalLM, dataset, task_name, save_path, tokenizer) -> List[Dict[str, str]]:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collector(tokenizer))
    task_results = []
    for batch in dataloader:
        prompts = batch.pop("prompts")
        answers = batch.pop("answers")
        input_ids_shape = batch['input_ids'].shape
        batch = {k:v.to(model.device) for k, v in batch.items()}
        results = model.generate(**batch, generation_config=GENERATION_CONFIG)
        mask = torch.cat(
            (torch.zeros(input_ids_shape), torch.ones(input_ids_shape[0], results.shape[1] - input_ids_shape[1])),
            dim=-1
        ).to(torch.int64).to(results.device)
        results= (results* mask).cpu().numpy().tolist()
        gened_texts = tokenizer.batch_decode(results, skip_special_tokens=True)
        
        for i in range(len(gened_texts)):
            task_results.append(dict(
                prompt=prompts[i],
                response=gened_texts[i],
                answer=answers[i]
            ))

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{task_name}.jsonl"), 'w') as f:
        for res in task_results:
            f.write(json.dumps(res, ensure_ascii=False)+'\n')
    return task_results

async def remote_infer_task(model, dataset, task_name, save_path, tokenizer):
    tasks = []
    for data in dataset:
        tasks.append(async_chat(data['prompt']))
    result = await asyncio.gather(**tasks)
    task_results = [
        dict(
            prompt=dataset[i]['prompt'],
            response=result[i],
            answer=dataset[i]['answer']
        )
        for i in range(len(dataset))
    ]
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{task_name}.jsonl"), 'w') as f:
        for res in task_results:
            f.write(json.dumps(res, ensure_ascii=False)+'\n')
    return task_results
        

def evaluate_result(task_results, dataset_name) -> float:
    return 1.0

def main():
    METHOD = "vanilla"
    ADAPTER_WEIGHT_PATHS = {
        "C-STANCE": "output/vanilla/vanilla_C-STANCE_round_2_desc_loss_0.6080",
        "FOMC": "output/vanilla/vanilla_FOMC_round_2_desc_loss_0.0913",
        "MeetingBank": "output/vanilla/vanilla_MeetingBank_round_2_desc_loss_1.7218",
        "Py150": "output/vanilla/vanilla_Py150_round_2_desc_loss_0.1856",
        "ScienceQA": "output/vanilla/vanilla_ScienceQA_round_2_desc_loss_0.3663",
        "NumGLUE-cm": "output/vanilla/vanilla_NumGLUE-cm_round_2_desc_loss_0.0888",
        "NumGLUE-ds": "output/vanilla/vanilla_NumGLUE-ds_round_2_desc_loss_0.0284",
        "20Minuten": "output/vanilla/vanilla_20Minuten_round_2_desc_loss_1.8274"
    }
    DATA_PATH = "data/TRACE-Benchmark/LLM-CL-Benchmark_5000"
    OUTPUT_PATH = f'infer_output/{METHOD}'

    result_metrics = {}
    for task_name, adapter_path in ADAPTER_WEIGHT_PATHS.items():
        result_metrics[task_name] = {}
        logger.info(f"using {task_name} adapter weight!!")
        model, tokenizer = load_model(model_path='', lora_adapter_path=adapter_path, tokenizer_path=adapter_path)
        for dataset_name in ADAPTER_WEIGHT_PATHS.keys():
            dataset = load_dataset(os.path.join(DATA_PATH, dataset_name, 'test.json'))
            task_results = asyncio.run(remote_infer_task(model=model, dataset=dataset, task_name=dataset_name, save_path=f"{OUTPUT_PATH}/train_{task_name}", tokenizer=tokenizer))
            metrics:float= evaluate_result(task_results, dataset_name)
            result_metrics[task_name][dataset_name] = metrics
            logger.info(f"dataset: {dataset_name}inference finished")

    with open(os.path.join(OUTPUT_PATH, 'result.json'), 'w') as f:
        f.write(json.dumps(result_metrics, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()