#!/bin/bash
# Data preparation for AES 2.0 V8
# Converts raw CSV files to JSONL format with fold assignments
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo " Data Preparation"
echo "============================================"

# ── Step 1: Check raw data exists ──
if [ ! -f "data/train_old.csv" ] || [ ! -f "data/train_new.csv" ]; then
    echo "ERROR: Raw data files not found."
    echo "  data/train_old.csv  — Persuade corpus (from Kaggle AES 2.0)"
    echo "  data/train_new.csv  — Kaggle-only essays (from Kaggle AES 2.0)"
    echo ""
    echo "Download from:"
    echo "  https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data"
    exit 1
fi

# ── Step 2: Convert CSV → JSONL ──
echo "[1] Converting CSV to JSONL..."
python -c "
import csv, json, os

for src, dst, is_persuade in [
    ('data/train_old.csv', 'data/processed/train_persuade.jsonl', True),
    ('data/train_new.csv', 'data/processed/train_kaggle_only.jsonl', False),
]:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src) as fin, open(dst, 'w') as fout:
        reader = csv.DictReader(fin)
        n = 0
        for row in reader:
            rec = {
                'essay_id': row['essay_id'],
                'essay': row['full_text'],
                'score': int(float(row['score'])),
                'is_persuade': is_persuade,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
        print(f'  {src} -> {dst}: {n} rows')

# Merge into train_full.jsonl
with open('data/processed/train_persuade.jsonl') as f:
    all_rows = [json.loads(l) for l in f]
with open('data/processed/train_kaggle_only.jsonl') as f:
    all_rows += [json.loads(l) for l in f]
with open('data/processed/train_full.jsonl', 'w') as f:
    for r in all_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'  Combined: data/processed/train_full.jsonl: {len(all_rows)} rows')
"

# ── Step 3: Create 5-fold splits ──
echo ""
echo "[2] Creating 5-fold stratified splits..."
python scripts/split_kfold.py \
    --input data/processed/train_full.jsonl \
    --out_dir data/processed/

echo ""
echo "============================================"
echo " Data preparation complete"
echo "============================================"
echo ""
echo "Files created:"
echo "  data/processed/train_full.jsonl         (17,307 samples)"
echo "  data/processed/train_persuade.jsonl      (12,874 samples)"
echo "  data/processed/train_kaggle_only.jsonl   (4,433 samples)"
echo "  data/processed/fold_0~4.jsonl           (5 fold files)"
echo "  data/processed/train_full.jsonl         (master file with fold assignments)"
echo ""
echo "Ready to train: bash run_v8_kfold.sh"
