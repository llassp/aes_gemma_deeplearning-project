#!/bin/bash
set -euo pipefail
export WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=0

cd /home/public/new_dl/aes2_gemma
ALL_FOLDS="data/processed/train_full.jsonl"
OUT_ROOT="outputs/ablation/exp2_kfold"
OOF_OUT="outputs/oof/exp2_huber_oof.csv"
mkdir -p "$OUT_ROOT" "$(dirname $OOF_OUT)"
rm -f "$OOF_OUT"

for FOLD in 0 1 2 3 4; do
  echo "===== Exp2 Huber+Attn Fold $FOLD / 5 ====="
  venv/bin/torchrun --nproc_per_node=4 stage2_train/sft_ordinal_v8.py     --config config/base_config_ordinal_v8.yaml     --fold "$FOLD" --all_folds "$ALL_FOLDS"     --output_dir "$OUT_ROOT/fold_$FOLD"     --num_train_epochs 2 --bf16 --flash_attention_2     --optim adamw_torch_fused     --per_device_train_batch_size 2 --gradient_accumulation_steps 4     --dataloader_num_workers 4     --no_ordinal

  echo "===== OOF Inference Fold $FOLD ====="
  venv/bin/python stage2_train/predict_oof.py     --model_name_or_path "$OUT_ROOT/fold_$FOLD/final_regression"     --base_model_id /home/public/new_dl/gemma-4-E4B-it-local     --all_folds "$ALL_FOLDS" --fold "$FOLD"     --out_csv "$OOF_OUT" --batch_size 8
done

echo "===== Threshold Optimization ====="
venv/bin/python -c "
import csv, json
with open('data/processed/train_full.jsonl') as f:
    id2score = {json.loads(l)['essay_id']: json.loads(l)['score'] for l in f}
rows = []
with open('$OOF_OUT') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames + ['true_score'] if 'true_score' not in reader.fieldnames else reader.fieldnames
    for r in reader:
        r['true_score'] = id2score.get(r['essay_id'], '')
        rows.append(r)
with open('$OOF_OUT', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
print(f'Added true_score to {len(rows)} rows')
"
venv/bin/python optimize_thresholds_v8.py     --oof_csv "$OOF_OUT"     --output outputs/oof/exp2_huber_thresholds.json

echo "===== EXP2 HUBER 5-FOLD OOF DONE ====="
