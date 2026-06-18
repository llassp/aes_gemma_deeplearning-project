#!/bin/bash
# run_v8_kfold.sh — Train 5 folds sequentially with V8 Ordinal Regression,
# then dump OOF predictions and optimize thresholds.
#
# Usage:
#   bash run_v8_kfold.sh
#
# Prerequisites:
#   1. data/processed/train_full.jsonl exists (run scripts/split_kfold.py first)
#   2. venv/ is set up with all dependencies
#   3. 4 × NVIDIA A100 80GB available

set -euo pipefail

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=0

PROJECT_ROOT="/home/public/new_dl/aes2_gemma"
cd "$PROJECT_ROOT"

ALL_FOLDS="data/processed/train_full.jsonl"
OUT_ROOT="outputs/v8_kfold"
OOF_DIR="outputs/oof"
OOF_OUT="$OOF_DIR/v8_oof.csv"

mkdir -p "$OUT_ROOT" "$OOF_DIR"

# Clear previous OOF file
rm -f "$OOF_OUT"

for FOLD in 0 1 2 3 4; do
  echo "========================================="
  echo "=== V8 Training Fold $FOLD / 5 ==="
  echo "========================================="

  venv/bin/torchrun --nproc_per_node=4 stage2_train/sft_ordinal_v8.py \
      --config config/base_config_ordinal_v8.yaml \
      --fold "$FOLD" \
      --all_folds "$ALL_FOLDS" \
      --output_dir "$OUT_ROOT/fold_$FOLD" \
      --num_train_epochs 2 \
      --bf16 \
      --flash_attention_2 \
      --optim adamw_torch_fused \
      --torch_compile \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --dataloader_num_workers 4

  echo "=== V8 OOF Inference Fold $FOLD ==="
  venv/bin/python stage2_train/predict_oof.py \
      --model_name_or_path "$OUT_ROOT/fold_$FOLD/final_regression" \
      --base_model_id "/home/public/new_dl/gemma-4-E4B-it-local" \
      --all_folds "$ALL_FOLDS" \
      --fold "$FOLD" \
      --out_csv "$OOF_OUT" \
      --batch_size 8
done

echo "========================================="
echo "=== All 5 folds complete ==="
echo "=== OOF predictions at $OOF_OUT ==="
echo "========================================="

echo ""
echo "=== Optimizing thresholds ==="
venv/bin/python optimize_thresholds_v8.py \
    --oof_csv "$OOF_OUT" \
    --output "$OOF_DIR/v8_thresholds.json"

echo "=== Done ==="
