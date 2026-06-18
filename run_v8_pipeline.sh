#!/bin/bash
# run_v8_pipeline.sh — Full V8 Two-Stage Training Pipeline
#
# Stage 2a: Train on Persuade corpus (broader domain)
# Stage 2b: Fine-tune on Kaggle-Only data (in-domain)
#
# Prerequisites:
#   1. data/processed/train_persuade.jsonl and train_kaggle_only.jsonl exist
#      (run scripts/split_persuade.py if not)
#   2. venv/ is set up with all dependencies
#   3. 4 × NVIDIA A100 80GB available

set -euo pipefail

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=0

PROJECT_ROOT="/home/public/new_dl/aes2_gemma"
cd "$PROJECT_ROOT"

# ── Phase 3.1: Split data if not already done ──
if [ ! -f data/processed/train_persuade.jsonl ] || [ ! -f data/processed/train_kaggle_only.jsonl ]; then
  echo "=== Splitting Persuade vs Kaggle-only data ==="
  venv/bin/python scripts/split_persuade.py \
      --input data/processed/train_full.jsonl \
      --persuade_out data/processed/train_persuade.jsonl \
      --kaggle_out   data/processed/train_kaggle_only.jsonl
fi

STAGE2A_OUT="outputs/v8_stage2a_persuade"
FINAL_OUT="outputs/final_ordinal_v8"

# ── Phase 3.2 Stage 2a: Train on Persuade data ──
echo "========================================="
echo "=== Stage 2a: Training on Persuade Data ==="
echo "========================================="

venv/bin/torchrun --nproc_per_node=4 stage2_train/sft_ordinal_v8.py \
    --config config/base_config_ordinal_v8.yaml \
    --train_data_override data/processed/train_persuade.jsonl \
    --val_data data/processed/val_sft_nocot.jsonl \
    --output_dir "$STAGE2A_OUT" \
    --num_train_epochs 2 \
    --bf16 \
    --flash_attention_2 \
    --optim adamw_torch_fused \
    --torch_compile \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 4

# ── Phase 3.2 Stage 2b: Fine-tune on Kaggle-Only data ──
echo "========================================="
echo "=== Stage 2b: Fine-tuning on Kaggle-Only Data ==="
echo "========================================="

venv/bin/torchrun --nproc_per_node=4 stage2_train/sft_ordinal_v8.py \
    --config config/base_config_ordinal_v8.yaml \
    --model_name_or_path "$STAGE2A_OUT/final_regression" \
    --train_data_override data/processed/train_kaggle_only.jsonl \
    --val_data data/processed/val_sft_nocot.jsonl \
    --output_dir "$FINAL_OUT" \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --bf16 \
    --flash_attention_2 \
    --optim adamw_torch_fused \
    --torch_compile \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 4

echo "========================================="
echo "=== V8 Two-Stage Pipeline Complete ==="
echo "=== Final model at $FINAL_OUT ==="
echo "========================================="
