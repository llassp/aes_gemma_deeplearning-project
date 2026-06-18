# Kaggle AES 2.0 V6 → V7 Upgrade Execution Guide (Claude Code)

> ## Language Directive (READ FIRST)
>
> **All reasoning, code, comments, log messages, documentation, commit messages, and user-facing reports produced during this upgrade MUST be in English.** The Kaggle AES 2.0 task operates on English student essays; the base model (Gemma) is English-dominant. Mixing languages increases tokenization overhead, hurts instruction following, and causes train/inference distribution drift. This rule applies to:
> - Every Python/Bash file you create or modify
> - Every docstring, inline comment, and print/log statement
> - Every WandB run name, config key, and argparse `--help` text
> - Every status report back to the user
>
> Do NOT translate the task description into Chinese even if the user writes in Chinese.

## Context for AI Assistant

- **Project Root**: `/home/public/new_dl/aes2_gemma/`
- **Current Script**: `stage2_train/sft_regression_v6.py`
- **Current Config**: `config/base_config_regression_v6.yaml`
- **Hardware**: 4 × NVIDIA A100 80GB (BF16-ready, FlashAttention-2 capable)
- **Eval Metric**: Quadratic Weighted Kappa (QWK)
- **Score Range**: 1–6 (5 ordinal cut points)

## Goal

Upgrade the current Huber-based continuous regression pipeline to an **Ordinal Regression + 5-Fold OOF + Data-Domain Two-Stage** pipeline, fully optimized for 4× A100, based on the Kaggle 2nd place solution.

> Please execute the following upgrade tasks **in order** (Phases 0 → 5). If dependencies are missing or file paths do not match, search the project tree and adapt automatically.

---

## Phase 0: English Prompt Template Standard

**Objective**: Lock down a single English prompt template for both training and inference. This eliminates train/inference prompt drift and keeps the tokenizer's BPE working in its high-frequency region.

### Action 0.1: Define the unified prompt builder

**Target**: `stage2_train/prompt.py` (new file)

```python
"""Unified English prompt template for Kaggle AES 2.0 essay scoring.

This module is the SINGLE SOURCE OF TRUTH for how an essay is wrapped
before being fed to Gemma. The template must be byte-identical between
training and inference to avoid distribution shift.
"""
from __future__ import annotations

# Special tokens (Gemma uses <bos><start_of_turn>...<end_of_turn>)
BOS = "<bos>"
START_TURN = "<start_of_turn>"
END_TURN = "<end_of_turn>"

SYSTEM_INSTRUCTION = (
    "You are an expert essay grader. "
    "Read the student's essay and assign a holistic score on a scale of 1 to 6, "
    "where 1 = very poor and 6 = excellent. "
    "Consider organization, development, sentence structure, word choice, "
    "voice, and conventions."
)


def format_prompt(essay: str) -> str:
    """Wrap a raw essay string into the canonical Gemma chat template.

    The template intentionally uses \\n\\n separators (Gemma's default)
    and never inserts a trailing space before <end_of_turn>.

    Returns the full prompt INCLUDING the assistant turn header,
    so the model learns to produce the score right after it.
    """
    user_turn = f"{START_TURN}user\n{SYSTEM_INSTRUCTION}\n\nEssay:\n{essay}{END_TURN}\n"
    assistant_turn = f"{START_TURN}model\n"
    return f"{BOS}{user_turn}{assistant_turn}"


def format_for_inference(essay: str) -> str:
    """Inference variant. Same as format_prompt; provided for symmetry
    and to allow future divergence (e.g. adding few-shot exemplars)."""
    return format_prompt(essay)
```

### Action 0.2: Wire the prompt into the Data Collator

**Target**: `sft_regression_v6.py` → Data Collator (and the v7 ordinal variant)

```python
from prompt import format_prompt  # from Action 0.1

class EssayCollator:
    def __call__(self, batch: list[dict]) -> dict:
        # ALWAYS go through format_prompt, never concatenate raw text
        texts = [format_prompt(item["essay"]) for item in batch]
        labels = torch.tensor([item["score"] for item in batch], dtype=torch.long)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1536,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc
```

> ⚠️ **Non-negotiable rules**:
> 1. Every place that builds a model input string MUST call `format_prompt()`.
> 2. Do not localize the system instruction. The English wording above is part of the model contract.
> 3. If you change the template, you must re-train from scratch — thresholds and OOF become invalid.

### Action 0.3: Sanity-check tokenizer behavior

Run a one-off script to confirm BPE efficiency stays > 95%:

```python
from transformers import AutoTokenizer
from prompt import format_prompt

tok = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
essay = "School uniforms should be required. " * 200  # ~600 tokens
ids = tok(format_prompt(essay), return_tensors="pt")["input_ids"]
# Count byte-level / unknown-ish fragments
n_special = sum(1 for i in ids[0] if i in tok.all_special_ids)
print(f"Total tokens: {ids.shape[1]}, special: {n_special}")
# Expectation: special tokens < 5, total tokens close to len(words)
```

---

## Phase 1: Convert Huber Regression to Ordinal Regression

**Objective**: Solve the severe 1→2 score drift. Change the output head from dim=1 to dim=5 and switch to `BCEWithLogitsLoss`.

### Action 1.1: Modify the model head and output dimension

In the model definition file (e.g. `sft_regression_v6.py` or related model utils):

- Find `nn.Linear(hidden_size, 1)` or `num_labels=1`.
- Replace it with `num_labels=5` (1-6 score range has 5 cut points).

**Target**: `Gemma4ForScoreRegression.__init__`

Before:

```python
self.score = nn.Linear(config.hidden_size, 1, bias=False)
```

After:

```python
self.score = nn.Linear(config.hidden_size, 5, bias=False)
```

### Action 1.2: Add the ordinal label conversion utility

Convert scalar scores (1–6) into 5-dim multi-label binary targets.

**Target**: `sft_regression_v6.py` — Data Collator or Model Forward

```python
import torch

def score_to_ordinal_target(score_tensor: torch.Tensor, num_classes: int = 6) -> torch.Tensor:
    """Convert a 1-6 score into an Ordinal Target.

    Example: score=4 -> [1, 1, 1, 0, 0]
    """
    batch_size = score_tensor.size(0)  # assumed score range 1-6
    levels = torch.arange(1, num_classes, device=score_tensor.device).expand(
        batch_size, num_classes - 1
    )
    targets = (score_tensor.unsqueeze(1) > levels).float()
    return targets
```

### Action 1.3: Replace the loss function

Remove the Huber loss and switch to weighted BCE loss.

**Target**: `sft_regression_v6.py` — loss computation in Model Forward

```python
import torch.nn.functional as F

# Assume labels are still the raw score (1-6)
ordinal_labels = score_to_ordinal_target(labels)  # shape: [B, 5]

# Optional per-sample weighting (shape [B, 1])
if use_weighted_loss and sample_weights is not None:
    loss = F.binary_cross_entropy_with_logits(logits, ordinal_labels, reduction='none')
    loss = (loss * sample_weights.unsqueeze(1)).mean()
else:
    loss = F.binary_cross_entropy_with_logits(logits, ordinal_labels, reduction='mean')
```

> ⚠️ **Class Imbalance reminder (critical for high-score recall)**: Scores 5 and 6 are rare, so the `[..., 1]` (high-score boundary) cells of the ordinal target are far fewer than `[..., 0]`. Plain BCE will bias the model toward low scores.
>
> Strongly recommend adding `pos_weight` to the BCE loss (computed from the training-set class distribution) or switching to **Focal Loss**. This is the single biggest lever for recalling high-score essays (5/6).

Example (compute `pos_weight` from training distribution; shape `[5]`, weights grow toward the high end):

```python
import torch.nn.functional as F

# Example values: pos_weight ≈ neg_count / pos_count
# MUST be recomputed from your actual training set distribution
pos_weight = torch.tensor([1.2, 1.5, 1.8, 3.0, 5.0], device=logits.device)

if use_weighted_loss and sample_weights is not None:
    loss = F.binary_cross_entropy_with_logits(
        logits, ordinal_labels,
        pos_weight=pos_weight,      # KEY: up-weight positives (high-score boundaries)
        reduction='none'
    )
    loss = (loss * sample_weights.unsqueeze(1)).mean()
else:
    loss = F.binary_cross_entropy_with_logits(
        logits, ordinal_labels,
        pos_weight=pos_weight,
        reduction='mean'
    )
```

### Action 1.4: Modify the inference path

Convert the 5 logits into a 1-6 integer score.

**Target**: `sft_regression_v6.py` — inference / OOF saving

```python
def logits_to_continuous_score(logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid-sum to a continuous score in [1.0, 6.0] for OOF threshold search."""
    probs = torch.sigmoid(logits)  # shape: [B, 5]
    # Base = 1, then accumulate probabilities to obtain a continuous score
    continuous_scores = probs.sum(dim=1) + 1.0
    return continuous_scores


def logits_to_discrete_score(logits: torch.Tensor) -> torch.Tensor:
    """Optional: round to an integer in 1-6 for direct QWK evaluation."""
    return torch.round(logits_to_continuous_score(logits)).long()
```

---

## Phase 2: Introduce Attention Pooling

**Objective**: Replace last-token pooling so the model can pick up dispersed error cues across 1536-token essays.

### Action 2.1: Implement the Attention Pooling module

**Target**: `sft_regression_v6.py` — Model Definition

```python
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: [B, L, H]
        # attention_mask:    [B, L]
        w = self.attention(last_hidden_state).squeeze(-1)  # [B, L]
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, dim=1)  # [B, L]
        pooled_output = torch.sum(w.unsqueeze(-1) * last_hidden_state, dim=1)  # [B, H]
        return pooled_output
```

> ⚠️ **Initialization detail (critical for training stability)**: With default PyTorch init, the attention weights are extremely biased toward a single token (usually the first or last), which causes the loss to spike or diverge at step 0.
>
> Initialize the final `nn.Linear(hidden_size, 1)` weights and bias to **0 or a very small value** (e.g. `1e-3`). This makes the attention weights nearly uniform at start, so the model **gracefully degrades to Mean Pooling** and gradually learns meaningful token weights.

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        # KEY: start near uniform attention -> degrades to mean pooling
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)
        # If you don't want a hard zero, use a very small std instead:
        # nn.init.normal_(self.attention[-1].weight, std=1e-3)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: [B, L, H]
        # attention_mask:    [B, L]
        w = self.attention(last_hidden_state).squeeze(-1)  # [B, L]
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, dim=1)  # [B, L]
        pooled_output = torch.sum(w.unsqueeze(-1) * last_hidden_state, dim=1)  # [B, H]
        return pooled_output
```

### Action 2.2: Replace the pooling logic in the model forward

**Target**: `Gemma4ForScoreRegression.forward`

Before (Last Token):

```python
seq_lens = attention_mask.sum(dim=1) - 1
pooled = last_hidden[batch_idx, seq_lens]
```

After (Attention Pooling):

In `__init__`, declare:

```python
self.pooler = AttentionPooling(config.hidden_size)
```

In `forward`:

```python
pooled_output = self.pooler(last_hidden_state, attention_mask)
logits = self.score(pooled_output)  # shape: [B, 5]
```

> ⚠️ **Gemma padding reminder (MUST verify)**: Gemma uses **RoPE + Causal Mask** like Llama. The tokenizer's `padding_side` directly affects training and inference behavior.
>
> - **Training** (variable-length batches): use **Right Padding** (`padding_side='right'`), and mask pad positions in `labels` with `-100`.
> - **Inference / batched generation**: switch to **Left Padding** (`padding_side='left'`) so the last valid token position is stable.
>
> **Core principle**: With Attention Pooling, the model is **less sensitive to padding direction** (it does global weighted sum rather than reading the last token), but you MUST still ensure `attention_mask` correctly masks all `<pad>` tokens. A broken mask will let `<pad>` embeddings leak into the pooled output and silently shift scores.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Training
tokenizer.padding_side = 'right'

# Inference / OOF generation
tokenizer.padding_side = 'left'
```

---

## Phase 2.5: 5-Fold StratifiedKFold OOF Pipeline

**Objective**: Generate out-of-fold predictions for every training essay. This is the foundation for unbiased threshold search (Phase 4) and model selection. The Kaggle AES 2.0 dataset is small (~18,000 essays), so 5-fold is the standard.

### Action 2.5.1: Create the fold-split script

**Target**: `scripts/split_kfold.py` (new file)

```python
"""Stratified 5-fold split for Kaggle AES 2.0.

Reads data/processed/train_full.jsonl and emits 5 fold files:
    data/processed/fold_{i}.jsonl
Each line keeps the original fields plus a 'fold' integer in [0, 4].
"""
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.input.open() as f:
        for line in f:
            rows.append(json.loads(line))

    scores = np.array([r["score"] for r in rows])
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for fold_id, (_, val_idx) in enumerate(skf.split(rows, scores)):
        out_path = args.out_dir / f"fold_{fold_id}.jsonl"
        with out_path.open("w") as f:
            for i in val_idx:
                r = dict(rows[i])
                r["fold"] = fold_id
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(val_idx)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

### Action 2.5.2: Modify the training script to honor `fold`

**Target**: `stage2_train/sft_ordinal_v7.py`

```python
# In the training entry point
import argparse

parser.add_argument("--fold", type=int, required=True, help="Validation fold id in [0, 4]")
parser.add_argument("--all_folds", type=str, required=True, help="Path to the master JSONL with 'fold' field")
```

```python
# In the dataset construction
all_rows = [json.loads(l) for l in open(args.all_folds)]
train_rows = [r for r in all_rows if r["fold"] != args.fold]
val_rows   = [r for r in all_rows if r["fold"] == args.fold]

train_ds = EssayDataset(train_rows, tokenizer)
val_ds   = EssayDataset(val_rows, tokenizer)
```

### Action 2.5.3: K-Fold OOF driver shell

**Target**: `run_v7_kfold.sh` (new file)

```bash
#!/bin/bash
# run_v7_kfold.sh — train 5 folds sequentially, dump OOF predictions to outputs/oof/

set -euo pipefail

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=0

ALL_FOLDS="data/processed/train_full.jsonl"
OUT_ROOT="outputs/v7_kfold"
OOF_OUT="outputs/oof/v7_oof.csv"

mkdir -p "$OUT_ROOT" "$OOF_OUT"

for FOLD in 0 1 2 3 4; do
  echo "============================="
  echo "=== Training fold $FOLD ==="
  echo "============================="

  venv/bin/torchrun --nproc_per_node=4 stage2_train/sft_ordinal_v7.py \
      --config config/base_config_ordinal_v7.yaml \
      --fold "$FOLD" \
      --all_folds "$ALL_FOLDS" \
      --output_dir "$OUT_ROOT/fold_$FOLD" \
      --num_train_epochs 2 \
      --bf16 \
      --flash_attention_2

  echo "=== Inferring OOF for fold $FOLD ==="
  venv/bin/torchrun --nproc_per_node=4 stage2_train/predict_oof.py \
      --model_name_or_path "$OUT_ROOT/fold_$FOLD" \
      --all_folds "$ALL_FOLDS" \
      --fold "$FOLD" \
      --out_csv "$OOF_OUT"
done

echo "All 5 folds done. OOF at $OOF_OUT"
```

---

## Phase 3: Data-Domain Two-Stage Training

**Objective**: Train first on the broader Kaggle + Persuade corpus, then fine-tune on Kaggle-Only data.

### Action 3.1: New data-split script

Create `scripts/split_persuade.py` to separate the official train data based on an external Persuade corpus.

> **Note**: If the local Persuade corpus is not yet available, prompt the user to download it. We assume the data source has gained an `is_persuade` boolean field.

```python
"""Split the full training JSONL into persuade-only and kaggle-only files."""
import json
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--persuade_out", type=Path, required=True)
    p.add_argument("--kaggle_out",   type=Path, required=True)
    args = p.parse_args()

    n_p, n_k = 0, 0
    with args.input.open() as fin, \
         args.persuade_out.open("w") as fp, \
         args.kaggle_out.open("w") as fk:
        for line in fin:
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
```

### Action 3.2: Multi-stage training shell

Create `run_v7_pipeline.sh`.

```bash
#!/bin/bash
# run_v7_pipeline.sh — full V7 upgrade pipeline

set -euo pipefail

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=0

# Phase 3.1: split data if not already done
if [ ! -f data/processed/train_persuade.jsonl ] || [ ! -f data/processed/train_kaggle_only.jsonl ]; then
  echo "=== Splitting persuade vs kaggle-only ==="
  venv/bin/python scripts/split_persuade.py \
      --input data/processed/train_full.jsonl \
      --persuade_out data/processed/train_persuade.jsonl \
      --kaggle_out   data/processed/train_kaggle_only.jsonl
fi

echo "=== Stage 2a: Train on Persuade Data ==="
venv/bin/torchrun --nproc_per_node=4 stage2_train/sft_ordinal_v7.py \
    --config config/base_config_ordinal_v7.yaml \
    --train_data data/processed/train_persuade.jsonl \
    --output_dir outputs/v7_stage2a_persuade \
    --num_train_epochs 2 \
    --bf16 \
    --flash_attention_2

echo "=== Stage 2b: Fine-tune on Kaggle-Only Data ==="
venv/bin/torchrun --nproc_per_node=4 stage2_train/sft_ordinal_v7.py \
    --config config/base_config_ordinal_v7.yaml \
    --train_data data/processed/train_kaggle_only.jsonl \
    --model_name_or_path outputs/v7_stage2a_persuade \
    --output_dir outputs/final_ordinal_v7 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --bf16 \
    --flash_attention_2
```

---

## Phase 4: OOF Threshold Brute-Force

**Objective**: Solve the 5→6 score collapse. Exhausively sweep the last cut point on the OOF predictions produced by Phase 2.5.

### Action 4.1: Create `optimize_thresholds_v7.py`

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize


def qwk(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def optimize_and_bruteforce(oof_predictions: np.ndarray, y_true: np.ndarray) -> list[float]:
    """Run Nelder-Mead then brute-force the 5->6 boundary.

    Args:
        oof_predictions: continuous scores from the model, e.g. 1.2, 4.8
        y_true: ground-truth integer scores in 1-6
    """
    # 5 cut points for the 1-6 score range
    initial_thresholds = [1.5, 2.5, 3.5, 4.5]

    def loss_func(thresholds):
        # Enforce monotonicity
        if not all(thresholds[i] < thresholds[i + 1] for i in range(len(thresholds) - 1)):
            return 1.0
        preds = np.digitize(oof_predictions, thresholds) + 1
        return -qwk(y_true, preds)

    print("Running Nelder-Mead optimization...")
    res = minimize(loss_func, initial_thresholds, method='Nelder-Mead')
    best_th = res.x.tolist()

    current_best_qwk = -res.fun
    print(f"Nelder-Mead Best QWK: {current_best_qwk:.5f}")
    print(f"Base Thresholds: {best_th}")

    # --- Brute-force the 5->6 boundary ---
    print("\nBrute-forcing the 5->6 boundary (threshold[3])...")
    best_th_56 = best_th[3]

    for t in np.arange(3.80, 5.00, 0.01):
        test_th = best_th.copy()
        test_th[3] = t
        if test_th[2] >= test_th[3]:
            continue

        preds = np.digitize(oof_predictions, test_th) + 1
        score = qwk(y_true, preds)

        if score > current_best_qwk:
            current_best_qwk = score
            best_th_56 = t
            print(f"  [Improved] th_56={t:.3f} -> QWK={score:.5f}")

    best_th[3] = best_th_56
    print(f"\nFinal Optimized Thresholds: {best_th}")
    return best_th
```

> **Claude**: According to the project's data loading convention, add code to load the OOF results and invoke the function above.

---

## Phase 5: A100 Optimization

**Objective**: Fully exploit 4 × A100 80GB. Without these flags the model is leaving 3-5× training throughput on the table.

### Action 5.1: Mandatory training flags

Add the following to every `torchrun` command in `run_v7_pipeline.sh` and `run_v7_kfold.sh`:

| Flag | Value | Why |
|------|-------|-----|
| `--bf16` | bool | A100 BF16 is ~15× faster than FP32; same convergence on this task |
| `--flash_attention_2` | bool | Cuts memory ~2× and speeds up 1536-token attention ~2.5× |
| `--gradient_checkpointing` | bool | Off if your batch fits; on only if you bump `--per_device_train_batch_size` |
| `--dataloader_num_workers` | 4 | Overlap data loading with GPU compute |
| `--dataloader_pin_memory` | True | Faster CPU→GPU H2D copies |
| `--optim` | `adamw_torch_fused` | Fused AdamW is ~20% faster on A100 |
| `--torch_compile` | bool | `torch.compile(mode="reduce-overhead")` for 1.3-1.5× speedup |

Example final training command:

```bash
venv/bin/torchrun --nproc_per_node=4 stage2_train/sft_ordinal_v7.py \
    --config config/base_config_ordinal_v7.yaml \
    --fold 0 \
    --all_folds data/processed/train_full.jsonl \
    --output_dir outputs/v7_kfold/fold_0 \
    --num_train_epochs 2 \
    --bf16 \
    --flash_attention_2 \
    --optim adamw_torch_fused \
    --torch_compile \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4   # effective batch = 4 GPUs × 4 × 4 = 64
```

### Action 5.2: DDP tuning for 4 × A100

**Target**: `sft_ordinal_v7.py` — DDP init

```python
import torch.distributed as dist

# Pick the fastest bucket size for A100's 600 GB/s NVLink
dist.init_process_group(
    backend="nccl",
    init_method="env://",
)
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Larger buckets = fewer all-reduce calls, but more memory
# 25 MB is a good default for 9B-class models on A100
os.environ.setdefault("NCCL_BUCKET_SIZE", "26214400")  # bytes
os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "simple")
os.environ.setdefault("NCCL_IB_HCA", "mlx5")
```

### Action 5.3: Memory & speed budget

With the flags above on Gemma-2 9B + 1536 tokens + 4 × A100 80GB:

- Per-GPU peak memory: ~38 GB (with gradient checkpointing off, BF16)
- Effective batch: 64
- Tokens/sec/GPU: ~3,800 (vs ~900 in FP32 + eager attention — about 4.2× speedup)
- One full 2-epoch fold run: ~35 min (vs ~2.5 h without A100 flags)

If you hit OOM, prefer reducing `--per_device_train_batch_size` from 4→2 and doubling `--gradient_accumulation_steps` rather than turning off BF16/FlashAttn.

### Action 5.4: Inference optimization

**Target**: `stage2_train/predict_oof.py`

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).eval().cuda()

# torch.compile gives a big win for repeated forward passes (OOF, threshold search)
model = torch.compile(model, mode="reduce-overhead")

# Left padding is required for stable last-token logits under causal mask
tokenizer.padding_side = 'left'
```

---

## Execution Feedback Requirements (For Claude Code)

After completing the modifications and file creations, report back to the user:

1. Confirm `sft_ordinal_v7.py` was saved with the key changes (Loss in Phase 1, Attention Pooling in Phase 2, A100 flags in Phase 5).
2. Verify `data/processed/` contains `train_persuade.jsonl`, `train_kaggle_only.jsonl`, and the 5 `fold_*.jsonl` files. If not, ask the user whether the data preparation scripts should be authored.
3. Run `run_v7_kfold.sh` first to produce OOF, then `run_v7_pipeline.sh` for the final two-stage training. Monitor both runs end-to-end and report the per-fold QWK and final continuous QWK.
4. All status reports, log messages, and WandB run names must be in English.
