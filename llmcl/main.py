import os, torch, deepspeed, logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import set_all_seed
import torch.distributed
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
		device = torch.device('cuda', args.local_rank)
		deepspeed.init_distributed()

	args.global_rank = torch.distributed.get_rank()
	args.world_size = torch.distributed.get_world_size()
	set_all_seed(args.seed)

	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
	tokenizer.padding_side = "left"
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	datasets: dict = get_datasets(args, tokenizer)
	model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
	# trainer = TRAINERS[args.cl_method](model, datasets, args)
	# trainer.continual_learning()

if __name__ == "__main__":
	main()
