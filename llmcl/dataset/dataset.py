import torch
from pathlib import Path
from typing import Any, List, Dict, Tuple, Callable, Optional, Union
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, Dataset


def format_data(data: Dict, chat: bool = False) -> Tuple[str, str]:
	if not chat:
		if data.get('input'):
			prompt = "Instruction:\n" + data.get('instruction') + "\n" + "Input:\n" + data.get('input') + "\nOutput:\n"
		else:
			prompt = "Input\n" + data.get('instruction') + "\nOutput:\n"
		return prompt, data.get('output')
	else:
		raise NotImplementedError("Chat format not implemented yet.")


def tokenize_data(tokenizer: PreTrainedTokenizerBase, max_length: int = 2048) -> Callable[
	[Dict], Dict[str, torch.Tensor]]:
	def tokenize(data: Dict) -> Dict[str, torch.Tensor]:
		prompt, output = format_data(data)
		tokenized = tokenizer(prompt + output, return_tensors="pt", padding=True, max_length=max_length)
		labels = tokenized['input_ids'].clone()
		prompt_len = len(tokenizer(prompt)['input_ids'])
		labels[:prompt_len] = -100  # ignore loss for prompt tokens
		return dict(
			input_ids=tokenized['input_ids'].squeeze(),
			attention_mask=tokenized['attention_mask'].squeeze(),
			labels=labels.squeeze()
		)
	return tokenize


def get_dataset(data_dir: Union[Path, str], tokenizer: PreTrainedTokenizerBase, max_length: int = 2048) -> Dataset:
	if not data_dir.exists():
		raise FileNotFoundError(f"{data_dir} does not exist.")
	data: Dataset = load_dataset("json", data_files=[data_dir], field="data", split="train")
	tokenized_data = data.map(tokenize_data(tokenizer=tokenizer, max_length=max_length), batched=True)
	return tokenized_data


def get_datasets(args, tokenizer) -> Dict[str, Dataset]:
	task_datasets = {}
	for name, path in args.data_dir.items():
		task_datasets[name] = get_dataset(path, tokenizer, args.max_length)
	return task_datasets
