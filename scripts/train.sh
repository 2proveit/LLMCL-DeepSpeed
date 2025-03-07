set -x
port=$(shuf -i25000-30000 -n1)
export PYTHONPATH=$(pwd)/llmcl:$PYTHONPATH
model_name_or_path=$1
output_dir=$2
lr=1e-4
num_train_epochs=3
batch_size=4
per_device_eval_batch_size=16
grad_acc=8
# vanilla ewc gem mtl agem l2p pp ilora one

for cl_method in vanilla ewc gem mtl agem l2p pp ilora one; do
   deepspeed --include=localhost:0,1,2,3 --master_port $port llmcl/main.py --deepspeed \
      --seed 42 \
      --data_path data/TRACE-Benchmark/LLM-CL-Benchmark_5000 \
      --dataset_names C-STANCE,FOMC,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
      --model_name_or_path $model_name_or_path \
      --logging_steps 100 \
      --do_eval false \
      --per_device_train_batch_size $batch_size \
      --per_device_eval_batch_size $per_device_eval_batch_size \
      --gradient_accumulation_steps $grad_acc \
      --learning_rate $lr \
      --num_train_epochs $num_train_epochs \
      --warmup_ratio 0.03 \
      --eval_steps 100 \
      --cl_method $cl_method \
      --logging_dir $output_dir/${cl_method}/tensorboard_logs \
      --output_dir $output_dir/${cl_method} \
      --deepspeed_config llmcl/ds_config/ds_zero2.json > logs/train_${cl_method}.log 2>&1
done
