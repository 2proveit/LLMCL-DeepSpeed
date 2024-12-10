import argparse
from transformers import TrainingArguments, HfArgumentParser
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import dataclass


class CLTrainingArguments(TrainingArguments):
    data_dir: str = "data"
    max_length: int = 1024
    train_on_inputs: bool = False
    dataset_names: str = "20Minuten,FOMC,MeetingBank,NumGLUE-ds,ScienceQA,C-STANCE,NumGLUE-cm"
    cl_method: str = "vanilla"  # 添加 cl_method 参数

def get_train_args():
    parser = HfArgumentParser(CLTrainingArguments)
    args = parser.parse_args_into_dataclasses()
    return args

if __name__ == "__main__":
    args = get_train_args()
    print(args.data_dir)
    print(args.max_length)
    print(args.train_on_inputs)
    print(args.dataset_names)
    print(args.cl_method)