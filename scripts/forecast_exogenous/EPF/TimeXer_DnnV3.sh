export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer_DnnV3
des='TimeXer-DnnV3-MS'
patch_len=24

# Chạy thử nghiệm TimeXer với Backbone DNN-v3 trên tập dữ liệu EPF (NP)
python3 -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path NP.csv \
  --model_id NP_168_24_DnnV3 \
  --model $model_name \
  --encoder_type dnn_v3 \
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
