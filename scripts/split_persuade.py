"""Split the full training JSONL into persuade-only and kaggle-only files.

Filters on the 'is_persuade' boolean field in each record.
"""
import json
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Split AES 2.0 data into Persuade and Kaggle-only sets"
    )
    p.add_argument(
        "--input", type=Path, required=True, help="Input JSONL file"
    )
    p.add_argument(
        "--persuade_out",
        type=Path,
        required=True,
        help="Output path for Persuade-only data",
    )
    p.add_argument(
        "--kaggle_out",
        type=Path,
        required=True,
        help="Output path for Kaggle-only data",
    )
    args = p.parse_args()

    n_p, n_k = 0, 0
    with args.input.open(encoding="utf-8") as fin, \
         args.persuade_out.open("w", encoding="utf-8") as fp, \
         args.kaggle_out.open("w", encoding="utf-8") as fk:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_persuade", False):
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_p += 1
            else:
                fk.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_k += 1

    print(f"Persuade: {n_p}  Kaggle-only: {n_k}")


if __name__ == "__main__":
    main()
