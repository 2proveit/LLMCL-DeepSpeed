#!bin/bash
port=$(shuf -i25000-30000 -n1)
export PYTHONPATH=$(pwd)/llmcl:$PYTHONPATH
deepspeed --include=localhost:0,1 --master_port $port llmcl/main.py \
   --deepspeed llmcl/ds_config/ds_zero2.json \
   --seed 42 \
   --data_path data/TRACE-Benchmark/LLM-CL-Benchmark_5000 \
   --dataset_names C-STANCE,FOMC \
   --model_name_or_path /hhd/lixinlong/model_base/official_model/Qwen2.5-7B-Instruct \
   --logging_steps 100 \
   --per_device_train_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 3 \
   --warmup_ratio 0.03 \
   --cl_method vanilla \
   --output_dir output/ > logs/train.log 2>&1 &
