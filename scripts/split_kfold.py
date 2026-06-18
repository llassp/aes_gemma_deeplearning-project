"""Stratified 5-fold split for Kaggle AES 2.0.

Reads a JSONL file and emits 5 fold files:
    {out_dir}/fold_{0..4}.jsonl

Each line keeps the original fields plus a 'fold' integer in [0, 4].
"""
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser(
        description="Stratified 5-fold split for AES 2.0 essays"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input JSONL file"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for fold files",
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of folds"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    scores = np.array([r["score"] for r in rows])
    skf = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )

    # Assign fold labels to all rows
    fold_labels = np.full(len(rows), -1, dtype=int)
    for fold_id, (_, val_idx) in enumerate(skf.split(rows, scores)):
        fold_labels[val_idx] = fold_id
        out_path = args.out_dir / f"fold_{fold_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for i in val_idx:
                r = dict(rows[i])
                r["fold"] = fold_id
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(val_idx)} rows to {out_path}")

    # Also write a single master file with all fold assignments
    master_path = args.out_dir / "train_full.jsonl"
    with master_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            r = dict(r)
            r["fold"] = int(fold_labels[i])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to master file: {master_path}")


if __name__ == "__main__":
    main()
