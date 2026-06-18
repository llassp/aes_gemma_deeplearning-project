#!/bin/bash
set -euo pipefail
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=0

cd /home/public/new_dl/aes2_gemma

for FOLD in 1 2 3 4; do
  echo "===== V8 Training Fold $FOLD / 5 ====="
  venv/bin/torchrun --nproc_per_node=4 stage2_train/sft_ordinal_v8.py       --config config/base_config_ordinal_v8.yaml       --fold "$FOLD"       --all_folds data/processed/train_full.jsonl       --output_dir "outputs/v8_kfold/fold_$FOLD"       --num_train_epochs 2 --bf16 --flash_attention_2       --optim adamw_torch_fused       --per_device_train_batch_size 2       --gradient_accumulation_steps 4       --dataloader_num_workers 4

  echo "===== OOF Inference Fold $FOLD ====="
  venv/bin/python stage2_train/predict_oof.py       --model_name_or_path "outputs/v8_kfold/fold_$FOLD/final_regression"       --base_model_id /home/public/new_dl/gemma-4-E4B-it-local       --all_folds data/processed/train_full.jsonl       --fold "$FOLD"       --out_csv outputs/oof/v8_oof.csv       --batch_size 8
done

echo "===== Threshold Optimization ====="
venv/bin/python optimize_thresholds_v8.py     --oof_csv outputs/oof/v8_oof.csv     --output outputs/oof/v8_thresholds.json

echo "===== ALL DONE ====="
