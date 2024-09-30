model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

master_port=12345
num_process=1
batch_size=24
d_model=16
d_ff=32

comment='TimeLLM-illness'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path illness.csv \
  --model_id weather_512_96 \
  --model $model_name \
  --data Illness \
  --features S \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
