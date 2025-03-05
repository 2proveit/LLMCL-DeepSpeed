import torch, os, json
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Any, List, Dict, Tuple, Callable, Optional, Union
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, Dataset

class Collector:
	def __init__(self, tokenizer:PreTrainedTokenizerBase=None):
		if not tokenizer:
			raise ValueError
		self.tokenizer = tokenizer
	def __call__(self, batch):
		return {
			"input_ids": pad_sequence([torch.tensor(item['input_ids'], dtype=torch.int64) for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id),
			"attention_mask": pad_sequence([torch.tensor(item['attention_mask'], dtype=torch.int64) for item in batch], batch_first=True, padding_value=0),
			"labels": pad_sequence([torch.tensor(item['labels'], dtype=torch.int64) for item in batch], batch_first=True, padding_value=-100)
		}

def format_data(data: Dict, tokenizer:PreTrainedTokenizerBase, chat: bool = False) -> Tuple[str, str]:

	return data['prompt'], data['answer']
 
def tokenize_data(tokenizer: PreTrainedTokenizerBase, max_length: int = 1024) -> Callable[[Dict], Dict[str, torch.Tensor]]:
	def tokenize(data: Dict[str, Union[str, List]]) -> Dict[str, torch.Tensor]:
		# format data
		prompt, output = format_data(data, tokenizer)
		# Tokenize prompt and output separately
		prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=False)
		output_tokens = tokenizer(output, add_special_tokens=False, return_tensors="pt", padding=False)
		# Concatenate tokens
		input_ids = torch.cat((prompt_tokens['input_ids'], output_tokens['input_ids']), dim=-1).squeeze(0)
		attention_mask = torch.cat([prompt_tokens['attention_mask'], output_tokens['attention_mask']], dim=-1).squeeze(0)
		# Create labels with -100 for prompt tokens
		labels = torch.cat([torch.full_like(prompt_tokens['input_ids'][-1], -100), output_tokens['input_ids'][-1]])

		input_ids = input_ids[:max_length]
		attention_mask = input_ids[:max_length]
		labels = labels[:max_length]

		return dict(
			input_ids=input_ids,
			attention_mask=attention_mask,
			labels=labels
		)
	return tokenize


def get_datasets(args, tokenizer, split='train') -> Dict[str, Dataset]:
	""" make sure your dataset look like:
		```args.data_path/args.dataset_names[i]/{split}.json```

	Args:
		args: args.data_path, get the root path of datasets
		tokenizer: tokenizer
		split (str, optional): get the `train` dataset. Defaults to 'train'.

	Returns:
		Dict[str, Dataset]: dict of datasets
	"""
	task_datasets = {}
	for name in args.dataset_names.split(','):
		dataset = load_dataset("json", data_files=os.path.join(args.data_path, name, f"{split}.json"), split='train')
		dataset = dataset.map(tokenize_data(tokenizer))
		task_datasets[name] = dataset
	return task_datasets