"""Out-of-Fold (OOF) Inference Script for V8 Ordinal Model.

Loads the custom Gemma4ForScoreRegression class (same as training),
runs inference on the validation fold, and outputs continuous scores + predictions.

IMPORTANT (Phase 5.4 correction): Uses Gemma4ForScoreRegression, NOT
AutoModelForSequenceClassification, to correctly load Attention Pooling weights.
torch.compile uses dynamic=True for variable-length inputs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Import from same package (predict_oof.py lives in stage2_train/) ──
from prompt import format_prompt, extract_essay
from sft_ordinal_v8 import (
    Gemma4ForScoreRegression,
    AttentionPooling,
    logits_to_continuous_score,
    logits_to_discrete_score,
    patch_gemma4_clippable_layers,
)


def load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class EssayInferenceDataset(Dataset):
    """Dataset for OOF inference — returns essay_id, input_ids, attention_mask."""

    def __init__(self, records, tokenizer, max_length: int = 1536):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.texts = []
        self.ids = []
        for r in records:
            essay = extract_essay(r)
            self.texts.append(format_prompt(essay))
            self.ids.append(r.get("essay_id", ""))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "essay_id": self.ids[idx],
        }


def collate_fn(batch: list) -> dict:
    """Left-padding collator for inference (stable under causal mask)."""
    from transformers import DataCollatorWithPadding

    # Use the tokenizer from the first item's context — set in main
    pass  # Will use HF default collator via DataLoader


def main():
    parser = argparse.ArgumentParser(
        description="V8 OOF Inference — generate predictions for a fold"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (adapter_model.safetensors dir)",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="/home/public/new_dl/gemma-4-E4B-it-local",
        help="Base Gemma4 model ID (for loading transformer backbone)",
    )
    parser.add_argument(
        "--all_folds",
        type=str,
        required=True,
        help="Path to master JSONL with 'fold' field",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Which fold to run inference on (0-4)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV path for OOF predictions (appends if exists)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Inference batch size"
    )
    parser.add_argument(
        "--max_length", type=int, default=1536, help="Max token length"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    is_main = local_rank == 0

    if is_main:
        print(f"[INFO] Loading model from: {args.model_name_or_path}")
        print(f"[INFO] Base model: {args.base_model_id}")

    # Load tokenizer — LEFT padding for inference stability
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id, local_files_only=True
    )
    tokenizer.padding_side = "left"  # Critical for causal LM inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 5.4: Load custom Gemma4ForScoreRegression (NOT AutoModelForSequenceClassification)
    # This ensures Attention Pooling weights are loaded correctly.
    model = Gemma4ForScoreRegression(
        args.base_model_id,
        num_labels=5,
        device_map={"": device} if "cuda" in str(device) else None,
    )

    # Load the adapter weights from the checkpoint
    from peft import PeftModel

    try:
        model = PeftModel.from_pretrained(model, args.model_name_or_path)
        if is_main:
            print("[INFO] Loaded PEFT adapter from checkpoint")
    except Exception as e:
        print(f"[WARN] PEFT load failed: {e}")
        # Fallback: try loading safetensors directly
        import safetensors.torch as sf

        adapter_path = os.path.join(
            args.model_name_or_path, "adapter_model.safetensors"
        )
        if os.path.exists(adapter_path):
            state_dict = sf.load_file(adapter_path)
            model.load_state_dict(state_dict, strict=False)
            if is_main:
                print(f"[INFO] Loaded {len(state_dict)} keys from safetensors")

    model.eval()
    model.to(device)

    # Phase 5.4: torch.compile with dynamic=True for variable-length inputs
    try:
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        if is_main:
            print("[INFO] torch.compile applied (dynamic=True)")
    except Exception as e:
        if is_main:
            print(f"[WARN] torch.compile failed: {e}. Continuing without.")

    # Load data for this fold
    all_rows = load_jsonl(args.all_folds)
    fold_rows = [r for r in all_rows if r.get("fold") == args.fold]
    if is_main:
        print(
            f"[INFO] Fold {args.fold}: {len(fold_rows)} samples for OOF inference"
        )

    dataset = EssayInferenceDataset(fold_rows, tokenizer, args.max_length)

    # Manual DataLoader with left-padding collator
    def left_pad_collate(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        essay_ids = [item["essay_id"] for item in batch]
        scores = [item.get("score", None) for item in batch]

        # Left padding: pad on the left side
        max_len = max(ids.size(0) for ids in input_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(input_ids, attention_masks):
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                pad_token_id = tokenizer.pad_token_id or 0
                ids = torch.cat(
                    [torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids]
                )
                mask = torch.cat(
                    [torch.zeros(pad_len, dtype=mask.dtype), mask]
                )
            padded_ids.append(ids)
            padded_masks.append(mask)

        result = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_masks),
            "essay_id": essay_ids,
        }
        if all(s is not None for s in scores):
            result["score"] = torch.tensor(scores, dtype=torch.float)
        return result

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=left_pad_collate,
        num_workers=0,  # 0 for inference to avoid CUDA fork issues
    )

    # Run inference
    all_continuous = []
    all_ids = []
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs["logits"]
            # Handle both ordinal [B, 5] and regression [B, 1] outputs
            if logits.shape[-1] == 1:
                # Huber regression: value is 0-5, shift back to 1-6
                continuous = logits.view(-1).float() + 1.0
            else:
                # Ordinal BCE: 5-dim logits → sigmoid sum → continuous
                continuous = logits_to_continuous_score(logits)

            all_continuous.extend(continuous.cpu().tolist())
            all_ids.extend(batch["essay_id"])
            if "score" in batch:
                all_scores.extend(batch["score"].tolist())

    # Write OOF CSV
    import csv

    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["essay_id", "fold", "continuous_score"]
            if all_scores:
                header.append("true_score")
            writer.writerow(header)

        for i in range(len(all_ids)):
            row = [all_ids[i], args.fold, all_continuous[i]]
            if all_scores:
                row.append(all_scores[i])
            writer.writerow(row)

    if is_main:
        print(
            f"[INFO] Wrote {len(all_ids)} predictions to {args.out_csv}"
        )
        if all_continuous:
            arr = np.array(all_continuous)
            print(
                f"[INFO] Score stats: "
                f"mean={arr.mean():.3f}, std={arr.std():.3f}, "
                f"min={arr.min():.3f}, max={arr.max():.3f}"
            )


if __name__ == "__main__":
    main()
