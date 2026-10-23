export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer_DINOv3
des='TimeXer-DINOv3-Exogenous'
patch_len=24

# Chạy thử nghiệm TimeXer với Backbone DINO-v3 và DINO-v3 Composite Loss Function trên EPF
python3 -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path NP.csv \
  --model_id NP_168_24_DINOv3 \
  --model $model_name \
  --encoder_type dino_v3 \
  --loss dino_v3 \
  --dino_temp_s 0.1 \
  --dino_temp_t 0.04 \
  --dino_lambda_distill 0.1 \
  --dino_lambda_gram 0.05 \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len $patch_len \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --itr 1
