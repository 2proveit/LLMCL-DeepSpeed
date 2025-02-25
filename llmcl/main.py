import os, torch, deepspeed, logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import set_all_seed
import torch.distributed
from methods import TRAINERS
from train import get_train_args
from get_dataset import get_datasets
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
	args = get_train_args()

	if args.local_rank == -1:
		device = 'cuda'
	else:
		torch.cuda.set_device(args.local_rank)
		args.device = torch.device('cuda', args.local_rank)
		deepspeed.init_distributed()

	args.global_rank = torch.distributed.get_rank()
	args.world_size = torch.distributed.get_world_size()
	set_all_seed(args.seed)

	model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
	tokenizer.padding_side = "left"
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token
		model.resize_token_embeddings(len(tokenizer))
	datasets: dict = get_datasets(args, tokenizer)
	eval_datasets:dict = get_datasets(args, tokenizer, "eval")
	
	if args.cl_method == 'one':
		from llmcl.methods import VanillaTrainer
		for name, dataset in datasets.items():
			trainer = VanillaTrainer(model=model, datasets={name:dataset}, args=args, tokenizer=tokenizer, eval_datasets={name:eval_datasets[name]})
			trainer.continual_learning()
	else:
		trainer = TRAINERS[args.cl_method](model, datasets, args, tokenizer, eval_datasets=eval_datasets)
		trainer.continual_learning()

if __name__ == "__main__":
	main()
