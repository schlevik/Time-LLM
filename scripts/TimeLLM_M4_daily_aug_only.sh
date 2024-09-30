model_name=TimeLLM

train_epochs=50
llama_layers=32
batch_size=16
learning_rate=0.001
d_model=8
d_ff=32

master_port=12345
num_process=2

comment='TimeLLM-M4'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --llm_layers $llama_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len 1 \
  --stride 1 \
  --batch_size $batch_size \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --loss 'SMAPE' \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --aug m4-Daily-train_168_gen_50ksteps_10k.npy \
  --aug_only 1 \
  --percent_aug 100
