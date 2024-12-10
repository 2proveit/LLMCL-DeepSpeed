#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0,1 --master_port $port llmcl/main.py \
   --deepspeed llmcl/ds_config/ds_zero2.json \
   --seed 42 \
   --data_path /hhd/lixinlong/code/LLMCL-DeepSpeed/data/TRACE-Benchmark/LLM-CL-Benchmark_5000 \
   --dataset_names C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
   --model_name_or_path /hhd/lixinlong/model_base/official_model/Qwen2.5-7B-Instruct \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 16 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 5 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --warmup_steps 0 \
   --cl_method lora \
   --output_dir /hhd/lixinlong/code/LLMCL-DeepSpeed/output/ > /hhd/lixinlong/code/LLMCL-DeepSpeed/logs/train.log 2>&1 &
