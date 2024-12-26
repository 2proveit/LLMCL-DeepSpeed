import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from peft import PeftModel


def get_args():
    parser = argparse.ArgumentParser("praser for merge a lora module")
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--save_path",type=str, default="./merged_ckpt")
    parser.add_argument("--base_model_path", type=str, default=None)
    return parser.parse_args()



def main():
    args = get_args()
    lora_config_path = Path(args.lora_path).joinpath('adapter_config.json')
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    assert lora_config_path.exists()

    with open(lora_config_path, 'r') as f:
        lora_config = json.loads(f.read())

    if Path(lora_config['base_model_name_or_path']).exists():
        base_path = lora_config['base_model_name_or_path']
    
    if args.base_model_path and Path(args.base_model_path).exists():
        base_path = args.base_model_path

    base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16)
    lora_model = PeftModel.from_pretrained(base_model, args.lora_path)
    print(type(lora_model))
    merged_model = lora_model.merge_and_unload()
    print(type(merged_model))
    merged_model.save_pretrained(args.save_path)
    # lora_model = PeftModel.from_pretrained(args.lora_path)
    # print(type(lora_model))
    # lora_model.merge_and_upload()
    # print(type(lora_model))

if __name__ == "__main__":
    main()

    