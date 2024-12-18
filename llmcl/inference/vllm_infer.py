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
from vllm import LLM, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def load_dataset(data_files:str):
    with open(data_files, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    data = Dataset.from_list(data)
    return data

def load_model(model_path):
    llm = LLM(model=model_path, enable_lora=True, tensor_parallel_size=2)
    return llm
    

@torch.no_grad()
def infer_task(model:LLM, dataset, task_name, save_path, lora_adapter_path) -> List[Dict[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    prompts = []
    answers = []
    sample_params = SamplingParams(
        max_tokens=512,
        temperature=0,
        repetition_penalty=1.15,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    sample_params.stop = [tokenizer.eos_token]
    for data in dataset:
        prompts.append("Input\n" + data.get('prompt') + "\nOutput:\n")
        answers.append(data.pop("answer"))
    
    responses = model.generate(
        prompts=prompts,
        sampling_params=sample_params,
        lora_request=LoRARequest(
            'lora', 1, lora_adapter_path
        )
    )
    task_results = [] 
    for i in range(len(responses)):
        task_results.append(dict(
            prompt=prompts[i],
            response=responses[i].outputs[0].text,
            answer=answers[i]
        ))

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{task_name}.jsonl"), 'w', encoding='utf-8') as f:
        for res in task_results:
            f.write(json.dumps(res, ensure_ascii=False)+'\n')
    return task_results


        

def evaluate_result(task_results, dataset_name) -> float:
    return 1.0

def main():
    METHOD = "ewc"
    ADAPTER_WEIGHT_PATHS = {
        "C-STANCE": "output/ewc/ewc_C-STANCE_round_2_desc_loss_0.6089",
        "FOMC": "output/ewc/ewc_FOMC_round_2_desc_loss_0.0927",
        "MeetingBank": "output/ewc/ewc_MeetingBank_round_2_desc_loss_1.7228",
        "Py150": "output/ewc/ewc_Py150_round_2_desc_loss_0.1869",
        "ScienceQA": "output/ewc/ewc_ScienceQA_round_2_desc_loss_0.3688",
        "NumGLUE-cm": "output/ewc/ewc_NumGLUE-cm_round_2_desc_loss_0.0905",
        "NumGLUE-ds": "output/ewc/ewc_NumGLUE-ds_round_2_desc_loss_0.0306",
        "20Minuten": "output/ewc/ewc_20Minuten_round_2_desc_loss_1.8288"
    }
    DATA_PATH = "data/TRACE-Benchmark/LLM-CL-Benchmark_5000"
    OUTPUT_PATH = f'infer_output/{METHOD}'
    MODEL_PATH = '/ssd/xumingzhou/checkpoints/official_model/Qwen2.5-7B-Instruct'


    model = load_model(model_path=MODEL_PATH)
    result_metrics = {}
    for task_name, adapter_path in ADAPTER_WEIGHT_PATHS.items():
        result_metrics[task_name] = {}
        for dataset_name in ADAPTER_WEIGHT_PATHS.keys():
            dataset = load_dataset(os.path.join(DATA_PATH, dataset_name, 'test.json'))
            if os.path.exists(os.path.join("{OUTPUT_PATH}/train_{task_name}", f"{dataset_name}.jsonl")):
                logger.info(f"skip train: {task_name}, infer: {dataset_name}")
                continue
            logger.info(f"method: {METHOD}, train: {task_name}, infer: {dataset_name} inference .......") 
            task_results = infer_task(model, dataset, dataset_name, save_path=f"{OUTPUT_PATH}/train_{task_name}", lora_adapter_path=adapter_path)
            metrics:float= evaluate_result(task_results, dataset_name)
            result_metrics[task_name][dataset_name] = metrics
            logger.info(f"method: {METHOD}, train: {task_name}, infer: {dataset_name} inference finished")

    with open(os.path.join(OUTPUT_PATH, 'result.json'), 'w') as f:
        f.write(json.dumps(result_metrics, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()