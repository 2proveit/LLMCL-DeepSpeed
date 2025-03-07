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
    from peft import PeftModel, PeftConfig

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # config = PeftConfig.from_pretrained(args.lora_path)
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained("path/to/save/merged_model")
    
if __name__ == "__main__":
    main()

    