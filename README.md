![log](assets/logo.png)
# LLMCL-DeepSpeed

LLMCL-DeepSpeed is a framewrok that uses DeepSpeed to implement several classical continual learning algorithms of large language models.

## Features

- **Classical incremental learning methods**: we implement some of the classical continual learning algorithms.
- **Simple-to-use code**: code are simple, *NOT* use [Huggingface Trainer](https://github.com/huggingface/transformers) API, you are in-charge of all the training process!
- **DeepSpeed integration for dsitributed**: we use [DeppSpeed](https://github.com/microsoft/DeepSpeed) to support milti-GPU training.

## Quick Start
### Installation
```bash
conda create -n llmcl python==3.10
pip install -r requirements.txt
```

### Prepare Data
refer [llmcl/get_dataset.py](llmcl/get_dataset.py) to prepare your own data.\
- make sure you dataset locates like:
    ```
    data_name
    ├── test.json
    └── train.json
    ```
- you may need customize your own data_process code ``llmcl/get_dataset.py `format_data` ``to deal with your own dataset. 

### Training
LLMCL-DeepSpeed implements deepspeed zero-2 training, config files can be found [llmcl/ds_config/ds_zero2.json](llmcl/ds_config/ds_zero2.json)

```bash
port=$(shuf -i25000-30000 -n1)
export PYTHONPATH=$(pwd)/llmcl:$PYTHONPATH
cl_method=gem
output_dir=output
deepspeed --include=localhost:0,1 --master_port $port llmcl/main.py --deepspeed \
   --seed 42 \
   --data_path data/LLM-CL-Benchmark_5000 \
   --dataset_names C-STANCE,FOMC \
   --model_name_or_path ./Qwen2.5-7B-Instruct \
   --logging_steps 100 \
   --eval_steps 50 \
   --per_device_train_batch_size 6 \
   --gradient_accumulation_steps 8 \
   --per_device_eval_batch_size 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 3 \
   --warmup_ratio 0.03 \
   --ewc_lambda 100 \
   --cl_method $cl_method \
   --logging_dir $output_dir/${cl_method}/tensorboard_logs \
   --output_dir $output_dir/${cl_method} \
   --deepspeed_config llmcl/ds_config/ds_zero2.json > logs/train_${cl_method}.log 2>&1 &
```

> NOTE \
> currently not support deepspeed zero3 training and multi-node training.

