import torch, os, json
from pathlib import Path
from typing import Any, List, Dict, Tuple, Callable, Optional, Union
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, Dataset

class Collector:
    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        return dict(
            input_ids=torch.tensor(input_ids, dtype=torch.int64),
            attention_mask=torch.tensor(attention_mask, dtype=torch.int64),
            labels=torch.tensor(labels, dtype=torch.int64)
		)
            

def format_data(data: Dict, chat: bool = False) -> Tuple[str, str]:
	"""turn your custom dataset in to the formated dataset.
		you may implement this for your own behavior.

	Args:
		data (Dict): one piece of data
		chat (bool, optional): if is the chat mode. Defaults to False.

	Raises:
		NotImplementedError: need more about chat format dataset, like:
  		```[{'role': 'system', 'content': 'you are a helpful AI'}]```

	Returns:
		Tuple[str, str]: {prompt:str, answer:str}
	"""
	if not chat:
		instruction = data.get('instruction', '')
		input_text = data.get('input')
		prompt = (
      		f"Instruction:\n{instruction}\nInput:\n{input_text}\nOutput:\n"
			if input_text else f"Input:\n{input_text}\nOutput:\n"
        )
		return prompt, data.get('answer')
	else: # TODO
		raise NotImplementedError("Chat format not implemented yet.")


def tokenize_data(tokenizer: PreTrainedTokenizerBase, max_length: int = 2048) -> Callable[[Dict], Dict[str, torch.Tensor]]:
    def tokenize(data: Dict) -> Dict[str, torch.Tensor]:
        prompt, output = format_data(data)
        
        # Tokenize prompt and output separately
        prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=False, truncation=True, max_length=max_length)
        output_tokens = tokenizer(output, add_special_tokens=False, return_tensors="pt", padding=False, truncation=True, max_length=max_length - len(prompt_tokens['input_ids'][0]))
        
        # Concatenate tokens
        input_ids = torch.cat([prompt_tokens['input_ids'][0], output_tokens['input_ids'][0]])
        attention_mask = torch.cat([prompt_tokens['attention_mask'][0], output_tokens['attention_mask'][0]])
        
        # Create labels with -100 for prompt tokens
        labels = torch.cat([torch.full_like(prompt_tokens['input_ids'][0], -100), output_tokens['input_ids'][0]])
        
        # Truncate to max_length
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    return tokenize


def get_dataset(data_dir: Union[Path, str], tokenizer: PreTrainedTokenizerBase, max_length: int = 2048) -> Dataset:
	if not os.path.exists(data_dir):
		raise FileNotFoundError(f"{data_dir} does not exist.")
	data = Dataset.from_list(json.loads(open(data_dir, 'r').read()))
	# data: Dataset = load_dataset("json", data_files=[data_dir], field="data", split="train")
	tokenized_data = data.map(tokenize_data(tokenizer=tokenizer, max_length=max_length), batched=False, num_proc=32)
	return tokenized_data


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
		task_datasets[name] = get_dataset(os.path.join(args.data_path, name, f'{split}.json'), tokenizer, args.max_length)
	return task_datasets

