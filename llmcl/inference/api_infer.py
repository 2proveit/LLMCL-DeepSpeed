import datetime
import concurrent
import json
import os
import argparse
import tqdm
from typing import List
import requests
import logging
from datasets import Dataset
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, url=None, api_key=None, model_name=None):
        self.base_url = "http://localhost:8000/v1" if not url else url
        self.api_key = "API KEY" if not api_key else api_key
        self.model_name = model_name if model_name else "sql-lora"


    def send_request(self, prompt, max_tokens=7, temperature=0):
        url = f"{self.base_url}/completions"
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()

        return response_data

    def send_concurrent_requests(self, prompts: List[str], max_tokens=7, temperature=0):
        """
        并发发送多个请求。
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.send_request, prompt, max_tokens, temperature
                )
                for prompt in prompts
            ]
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"请求出错: {e}")
        return results


def load_dataset(data_files:str):
    with open(data_files, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    data = Dataset.from_list(data)
    return data

def infer_task(client:APIClient, dataset,task_name, save_path:str):
    prompts = []
    answers = []
    responses = []
    for data in dataset:
        prompts.append("Input\n" + data.get('prompt') + "\nOutput:\n")
        answers.append(data.pop("answer"))
    responses = client.send_concurrent_requests(prompts=prompts, max_tokens=512)
    
    task_results = []
    for i in range(len(prompts)):
        try:
            task_results.append(dict(
                prompt=prompts[i],
                response=responses[i]['choices'][0]['text'],
                answer=answers[i]
            ))
        except Exception as e:
            logger.warning(e)
    
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{task_name}.jsonl"), 'w', encoding='utf-8') as f:
        for res in task_results:
            f.write(json.dumps(res, ensure_ascii=False)+'\n')

def arg_parse():
    parser = argparse.ArgumentParser("API inference")
    parser.add_argument('--method')
    parser.add_argument('--port')
    return parser.parse_args()

def main():
    args = arg_parse()
    METHOD=args.method
    ADAPTER_WEIGHT_PATHS = {
            "C-STANCE": f"output2/{METHOD}/{METHOD}_C-STANCE_round_2_desc_",
            "FOMC": "output2/{METHOD}/{METHOD}_FOMC_round_2_desc_",
            "MeetingBank": f"output2/{METHOD}/{METHOD}_MeetingBank_round_2_desc_",
            "Py150": f"output2/{METHOD}/{METHOD}_Py150_round_2_desc_",
            "ScienceQA": f"output2/{METHOD}/{METHOD}_ScienceQA_round_2_desc_",
            "NumGLUE-cm": f"output2/{METHOD}/{METHOD}_NumGLUE-cm_round_2_desc_",
            "NumGLUE-ds": f"output2/{METHOD}/{METHOD}_NumGLUE-ds_round_2_desc_",
            "20Minuten": f"output2/{METHOD}/{METHOD}_20Minuten_round_2_desc_",
        }
    DATA_PATH = "data/TRACE-Benchmark/LLM-CL-Benchmark_5000"

    OUTPUT_PATH = f'infer_output_14b/{METHOD}'
    BASE_URL = f'http://localhost:{args.port}/v1'
    model_name=f'{METHOD}_' 


    for task_name, adapter_path in ADAPTER_WEIGHT_PATHS.items():
        client = APIClient(url=BASE_URL, model_name=model_name+task_name)
        for dataset_name in ADAPTER_WEIGHT_PATHS.keys():
            dataset = load_dataset(os.path.join(DATA_PATH, dataset_name, 'test.json'))
            if os.path.exists(os.path.join("{OUTPUT_PATH}/train_{task_name}", f"{dataset_name}.jsonl")):
                logger.info(f"skip train: {task_name}, infer: {dataset_name}")
                continue
            logger.info(f"method: {METHOD}, train: {task_name}, infer: {dataset_name} inference .......") 
            task_results = infer_task(client=client, dataset=dataset, task_name=dataset_name, save_path=f"{OUTPUT_PATH}/train_{task_name}")
            logger.info(f"method: {METHOD}, train: {task_name}, infer: {dataset_name} inference finished")

if __name__ == "__main__":
    main()