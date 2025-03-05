import argparse
import re
import json
import logging
from typing import List
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

supprted_methods=['vanilla', 'ewc', 'gem', 'agem', 'l2p', 'pp', 'mtl', 'one']

def parse_args():
    parser = argparse.ArgumentParser("inference for multi lora modules")
    parser.add_argument('--adapter_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--tp', type=int, default=4, help='tensor paralle size for vllm')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--dataset_names', nargs='+', default=None)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--base_model_path', type=str, default=None)

    return parser.parse_args()

def get_method(path:Path, methods:list):
    path = str(path)
    pattern = r'\b(' + '|'.join(re.escape(mtd) for mtd in methods) + r')\b'
    match = re.search(pattern, path)
    if match:
        return match.group(0).strip()
    else:
        raise ValueError(f"Can not match any of {methods} for {path}")

def get_train_dataset_name(path:Path, dataset_names:list):
    path = str(path.parts[-1])
    pattern = r'(' + '|'.join(re.escape(data_name) for data_name in dataset_names+['MTL']) + r')'
    match = re.search(pattern, path)
    if match:
        return match.group(0).strip()
    else:
        raise ValueError(f"Can not match any of {dataset_names} for {path}")

def prepare_dataset(example):
    input_text = example.get('prompt', '')
    example['input_prompt'] = input_text
    example['message'] = [{"role": 'user', "content": input_text}]
    return example


def main():
    args:argparse.Namespace = parse_args()
    assert Path(args.adapter_path).exists(), "Adater path not found"

    adapters = []
    for folder in Path(args.adapter_path).rglob("*"):
        if folder.is_dir() and folder.joinpath("adapter_config.json").is_file():
            try:
                method = get_method(folder, supprted_methods)
                train_dataset_name = get_train_dataset_name(folder, args.dataset_names)
            except:
                logger.warning("get method and train dataset name failed")
                continue
            logger.info(f"path: {str(folder)}\t method: {method}\t train_dataset: {train_dataset_name}")
            adapters.append(dict(
                adapter_path=folder,
                method=method,
                train_dataset_name=train_dataset_name
            ))
    
    adapters = [adapter_info for adapter_info in adapters if 'round_2' in str(adapter_info['adapter_path'])]
    logger.info(f"Num adapters: {len(adapters)}")
    
    if len(adapters) == 0:
        raise FileNotFoundError("No adapters was found!")
    with adapters[0]['adapter_path'].joinpath("adapter_config.json").open('r') as f:
        base_model_path = json.loads(f.read())['base_model_name_or_path']

    if not Path(base_model_path).exists():
        if args.base_model_path and Path(args.base_model_path).exists():
            base_model_path = args.base_model_path
        else:
            raise ValueError("base model path in config file not exists, please specify a base model path in CLI")
    
    vllm_model = LLM(model=base_model_path, tensor_parallel_size=args.tp, enable_lora=True, gpu_memory_utilization=args.gpu_memory_utilization)
    
    for i, adapter_info in enumerate(adapters):
        train_dataset_name = adapter_info.get("train_dataset_name")
        adapter_path = adapter_info.get("adapter_path")
        method = adapter_info.get("method")
        Path(args.save_path).joinpath(method).mkdir(exist_ok=True, parents=True)

        for infer_dataset_name in args.dataset_names:
            save_file_dir = Path(args.save_path).joinpath(method).joinpath(f"train_{train_dataset_name}_infer_{infer_dataset_name}.jsonl")
            if save_file_dir.exists():
                logger.info(f"File Exsists for inference for mathod {method}, train on {train_dataset_name}, inference on {infer_dataset_name}")
                continue
            # load test dataset
            data_file_dir = Path(args.data_path).joinpath(infer_dataset_name).joinpath("test.json")
            data = Dataset.from_list(json.loads(data_file_dir.open('r', encoding='utf-8').read()))
            data = data.map(prepare_dataset, num_proc=32)

            sample_params = SamplingParams(
                temperature=0
            )
            responses = vllm_model.generate(
                [d['input_prompt'] for d in data],
                sampling_params=sample_params,
                lora_request=LoRARequest(
                    lora_name=method, lora_int_id=i, lora_path=str(adapter_path)
                )
            )
            with save_file_dir.open("w", encoding='utf-8') as f:
                for idx, resp in enumerate(responses):
                    try:
                        item = dict(
                            prompt=data[idx]['input_prompt'],
                            response=resp.outputs[0].text,
                            answer=data[idx]['answer']
                        )
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        f.flush()
                    except Exception as e:
                        logger.error(f"Can not write in to file : {e}")


if __name__ == "__main__":
    main()