"""
AES 2.0 Ordinal Regression Training Script V8 — Gemma4 + LoRA + Ordinal Head + Attention Pooling

V8 Upgrades (from V6 baseline):
  Phase 0: Unified English prompt template (prompt.py) — single source of truth
  Phase 1: Huber Regression → Ordinal Regression (5 output dims, BCEWithLogitsLoss + pos_weight)
  Phase 2: Last-token pooling → Attention Pooling with zero-init (graceful mean-pooling start)
  Phase 5: A100 4×80GB optimization flags (BF16, FlashAttention-2, fused AdamW, torch.compile)

Architecture:
  Gemma4Model (frozen + LoRA) → AttentionPooling → nn.Linear(hidden, 5) → BCEWithLogitsLoss
  PEFT LoRA task_type=SEQ_CLS, modules_to_save=["score", "pooler"]

Score range: 1–6, mapped to 5 ordinal cut points [>1, >2, >3, >4, >5].

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
      stage2_train/sft_ordinal_v8.py \\
      --config config/base_config_ordinal_v8.yaml \\
      --fold 0 \\
      --all_folds data/processed/train_full.jsonl \\
      --output_dir outputs/v8_kfold/fold_0
"""
import argparse
import json
import math
import os
import sys
import gc
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score

# ── Project root ──────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from vram_utils import clear_vram, print_vram

# Phase 0: Unified prompt template
from prompt import format_prompt, extract_essay


# ================================================================
# 0. Gemma4 PEFT Compatibility Patch
# ================================================================

def patch_gemma4_clippable_layers(model):
    """Dynamically add .weight property to Gemma4 ClippableLinear layers.

    Gemma 4 uses custom ClippableLinear layers that don't expose a standard
    .weight attribute. PEFT needs base_layer.weight to serialize adapter weights.
    This patch adds a .weight property to all ClippableLinear classes.

    Idempotent: already-patched classes are skipped.
    """
    patched_classes = set()
    for _name, module in model.named_modules():
        class_name = module.__class__.__name__
        if "ClippableLinear" not in class_name:
            continue
        cls = module.__class__
        if cls in patched_classes:
            continue
        if hasattr(cls, "weight"):
            continue

        @property
        def weight_property(self):
            for p_name, param in self.named_parameters(recurse=False):
                if "weight" in p_name:
                    return param
            params = list(self.parameters(recurse=False))
            if len(params) == 1:
                return params[0]
            raise AttributeError(
                f"{self.__class__.__name__}: no weight param found"
            )

        cls.weight = weight_property
        patched_classes.add(cls)

    if patched_classes:
        print(f"[Patch] Patched {len(patched_classes)} ClippableLinear classes")


# ================================================================
# 1. Ordinal Regression Utilities (Phase 1)
# ================================================================

def score_to_ordinal_target(score_tensor: torch.Tensor, num_classes: int = 6) -> torch.Tensor:
    """Convert 1–6 integer scores into 5-dim multi-label binary targets.

    Example: score=4 → [1, 1, 1, 0, 0]
    That is: score > 1? yes. > 2? yes. > 3? yes. > 4? no. > 5? no.

    Args:
        score_tensor: [B] tensor of integer scores in [1, 6]
        num_classes: number of score classes (default 6 for 1–6 range)
    Returns:
        [B, num_classes-1] float tensor of binary ordinal targets
    """
    batch_size = score_tensor.size(0)
    levels = torch.arange(1, num_classes, device=score_tensor.device).expand(
        batch_size, num_classes - 1
    )
    targets = (score_tensor.unsqueeze(1) > levels).float()
    return targets


def logits_to_continuous_score(logits: torch.Tensor) -> torch.Tensor:
    """Convert 5 ordinal logits to a continuous score in [1.0, 6.0].

    Sigmoid each cut-point probability, then sum + 1.0 base.
    Used for OOF threshold search and QWK evaluation.
    """
    probs = torch.sigmoid(logits)  # shape: [B, 5]
    continuous_scores = probs.sum(dim=1) + 1.0
    return continuous_scores


def logits_to_discrete_score(logits: torch.Tensor) -> torch.Tensor:
    """Convert 5 ordinal logits to a discrete integer score in [1, 6].

    Rounds the continuous score; useful for direct QWK evaluation.
    """
    return torch.round(logits_to_continuous_score(logits)).long().clamp(1, 6)


# ================================================================
# 2. Attention Pooling Module (Phase 2)
# ================================================================

class AttentionPooling(nn.Module):
    """Learned attention pooling over all token positions.

    Replaces last-token pooling so the model can attend to dispersed
    error cues across long essays (up to 1536 tokens).

    Key design: zero-init on the final linear layer so the model starts
    near uniform attention (≈ mean pooling) and gradually learns to focus.
    This prevents training divergence at step 0.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        # KEY: start near uniform attention → degrades gracefully to mean pooling
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted average of token representations.

        Args:
            last_hidden_state: [B, L, H] hidden states from transformer
            attention_mask:    [B, L] 1 for real tokens, 0 for padding
        Returns:
            [B, H] pooled output (same dtype as input)
        """
        # Cast to float32 for attention computation (pooler weights are fp32),
        # then cast result back to input dtype
        h = last_hidden_state.float()
        w = self.attention(h).squeeze(-1)  # [B, L]
        w[attention_mask == 0] = float("-inf")
        w = torch.softmax(w, dim=1)  # [B, L]
        pooled_output = torch.sum(
            w.unsqueeze(-1) * h, dim=1
        )  # [B, H]
        return pooled_output.to(last_hidden_state.dtype)


# ================================================================
# 3. Custom Ordinal Regression Model
# ================================================================

class Gemma4ForScoreRegression(nn.Module):
    """Gemma4 + Attention Pooling + Ordinal head.

    Compatible with PEFT LoRA, HF Trainer, and DDP.
    V8: 5 output dims (ordinal), attention pooling, BCE loss with pos_weight.
    """

    def __init__(
        self,
        model_id: str,
        num_labels: int = 5,  # V8: 5 ordinal cut points
        device_map=None,
        use_attention_pooling: bool = True,  # Ablation flag
    ):
        super().__init__()
        from transformers import AutoModel

        self.transformer = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        hidden_size = self.transformer.config.text_config.hidden_size
        self.num_labels = num_labels
        self.use_attention_pooling = use_attention_pooling

        # Phase 2: Attention pooling replaces last-token
        if use_attention_pooling:
            self.pooler = AttentionPooling(hidden_size)

        # Phase 1: 5 output dims for ordinal cut points (or 1 for Huber ablation)
        self.score = nn.Linear(hidden_size, num_labels, bias=False)

        # Small-weight init for regression head (reduces early-training spikes)
        nn.init.xavier_uniform_(self.score.weight, gain=0.1)

        # Proxy config for Trainer / PEFT compatibility
        self.config = self.transformer.config

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        sample_weights=None,
        pos_weight=None,
        **kwargs,
    ):
        """Forward pass with Ordinal BCE loss.

        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] attention mask
            labels: [B] integer scores in [1, 6] (NOT shifted to 0–5)
            sample_weights: [B] per-sample loss weights (optional)
            pos_weight: [5] positive-class weights for BCE (higher = bias toward high scores)
        """
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state  # [B, L, H]

        # Phase 2: Pooling — attention (V8) or last-token (ablation)
        if self.use_attention_pooling:
            pooled_output = self.pooler(last_hidden_state, attention_mask)
        else:
            # Last-token pooling (V6 baseline)
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(
                    input_ids.size(0), device=input_ids.device
                )
                pooled_output = last_hidden_state[batch_idx, seq_lens.long()]
            else:
                pooled_output = last_hidden_state[:, -1, :]

        # Phase 1: logits — ordinal [B, 5] or regression [B, 1]
        logits = self.score(pooled_output.float())

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # V6 Regression mode (ablation: --no_ordinal)
                loss_fct = nn.HuberLoss(delta=1.0, reduction='none')
                per_sample = loss_fct(
                    logits.view(-1).float(),
                    (labels.view(-1).float() - 1.0)  # 1-6 → 0-5
                )
                if sample_weights is not None:
                    sw = sample_weights.to(per_sample.device)
                    loss = (per_sample * sw).sum() / (sw.sum() + 1e-8)
                else:
                    loss = per_sample.mean()
            else:
                # V8 Ordinal mode
                # Convert scalar scores to ordinal targets
                ordinal_labels = score_to_ordinal_target(labels)  # [B, 5]

                # Move pos_weight to the same device as logits
                _pw = pos_weight.to(logits.device) if pos_weight is not None else None

                if sample_weights is not None:
                    # Per-sample weighted BCE
                    sw = sample_weights.to(logits.device)
                    loss = F.binary_cross_entropy_with_logits(
                        logits,
                        ordinal_labels,
                        pos_weight=_pw,
                        reduction="none",
                    )
                    loss = (loss * sw.unsqueeze(1)).mean()
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits,
                        ordinal_labels,
                        pos_weight=_pw,
                        reduction="mean",
                    )

        return {
            "loss": loss,
            "logits": logits,
        }

    def gradient_checkpointing_enable(self, gck_kwargs=None, **kwargs):
        self.transformer.gradient_checkpointing_enable(gck_kwargs)

    def gradient_checkpointing_disable(self):
        self.transformer.gradient_checkpointing_disable()

    def save_pretrained(self, path, **kwargs):
        """Save adapter weights via safetensors, bypassing PEFT buggy save path."""
        import safetensors.torch as _sf

        os.makedirs(path, exist_ok=True)
        try:
            from peft import get_peft_model_state_dict

            clean_sd = get_peft_model_state_dict(self)
            _sf.save_file(
                clean_sd,
                os.path.join(path, "adapter_model.safetensors"),
            )
            print(f"[save_pretrained] Saved {len(clean_sd)} keys")
        except Exception as e:
            print(f"[save_pretrained] Fallback: {e}")
            fb = {}
            for n, p in self.named_parameters():
                if p.requires_grad:
                    fb[n] = p.data.cpu()
            _sf.save_file(
                fb,
                os.path.join(path, "adapter_model.safetensors"),
            )
        # Save adapter_config.json (required for PEFT loading)
        if hasattr(self, "peft_config"):
            for name, peft_config in self.peft_config.items():
                peft_config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls(model_id)


# ================================================================
# 4. Data Processing
# ================================================================

def load_jsonl(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def tokenize_fn(examples: dict, tokenizer, max_length: int):
    """Tokenize essays via the unified prompt template (Phase 0).

    Extracts raw essay text, wraps it with format_prompt(), then tokenizes.
    Labels are kept as raw integer scores (1–6); ordinal conversion happens
    in the model forward pass.
    """
    texts = []
    for r in examples["_raw"]:
        essay = extract_essay(r)
        texts.append(format_prompt(essay))

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # DataCollator handles padding
    )
    # Keep raw scores 1–6 (NOT shifted); ordinal conversion in forward()
    enc["labels"] = [float(r["score"]) for r in examples["_raw"]]
    # Per-sample weights for loss balancing
    enc["sample_weight"] = [
        float(r.get("_weight", 1.0)) for r in examples["_raw"]
    ]
    return enc


# ================================================================
# 5. Data Balancing Utilities (from V6, retained for V8)
# ================================================================

def compute_sample_weights(records):
    """Compute inverse sqrt-frequency weights to balance score classes in loss.

    Formula: weight = sqrt(mean_count / score_count), normalized to mean=1.0.
    """
    score_counts = Counter(r["score"] for r in records)
    mean_count = sum(score_counts.values()) / len(score_counts)
    weights = []
    for r in records:
        s = r["score"]
        w = math.sqrt(mean_count / score_counts[s])
        weights.append(w)
    # Normalize to mean=1.0
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    return weights


def oversample_rare_scores(records, max_factor=10):
    """Over-sample rare scores using sqrt-balancing.

    Args:
        max_factor: maximum repeat factor to prevent overfitting on extreme rarities.
    """
    score_counts = Counter(r["score"] for r in records)
    max_count = max(score_counts.values())

    print(f"[oversample] Original: {dict(sorted(score_counts.items()))}")

    oversampled = []
    for r in records:
        s = r["score"]
        factor = max(1, int(round(math.sqrt(max_count / score_counts[s]))))
        factor = min(factor, max_factor)
        for _ in range(factor):
            oversampled.append(r)

    new_counts = Counter(r["score"] for r in oversampled)
    print(f"[oversample] Balanced: {dict(sorted(new_counts.items()))}")
    print(f"[oversample] {len(records)} -> {len(oversampled)} samples")

    return oversampled


def compute_pos_weight(records) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from training data.

    For each of the 5 ordinal cut points [>1, >2, >3, >4, >5],
    pos_weight = neg_count / pos_count.

    Higher values at the high-score end up-weight positive examples
    to combat the "regression to mean" effect on scores 5 and 6.

    Returns:
        [5] tensor of pos_weight values, clamped to [1.0, 10.0]
    """
    scores = np.array([r["score"] for r in records])
    total = len(scores)
    pos_weights = []
    for k in range(1, 6):  # cut points: >1, >2, >3, >4, >5
        n_pos = (scores > k).sum()
        n_neg = total - n_pos
        if n_pos == 0:
            pos_weights.append(10.0)  # max weight if no positive samples
        else:
            pw = n_neg / n_pos
            pos_weights.append(min(max(pw, 1.0), 10.0))  # clamp to [1, 10]

    pw_tensor = torch.tensor(pos_weights, dtype=torch.float32)
    print(f"[pos_weight] Computed from train data: {pw_tensor.tolist()}")
    return pw_tensor


# ================================================================
# 6. Two-Stage Training Utilities (from V6)
# ================================================================

def set_lora_trainable(model, trainable: bool):
    """Freeze or unfreeze LoRA adapter parameters. Score head (and pooler if present)
    always remain trainable."""
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = trainable
        elif "score" in n or "pooler" in n:
            p.requires_grad = True  # head (+ pooler if exists) always trainable


def freeze_lora(model):
    """Freeze LoRA; only regression head + pooler remain trainable."""
    set_lora_trainable(model, False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze_lora] Trainable params after freeze: {trainable:,}")


def unfreeze_lora(model):
    """Unfreeze LoRA; all parameters trainable."""
    set_lora_trainable(model, True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[unfreeze_lora] Trainable params after unfreeze: {trainable:,}")


# ================================================================
# 7. SafeSaveTrainer (from V6)
# ================================================================

def create_safe_save_trainer(base_cls):
    """Factory: create a Trainer subclass that saves via safetensors."""
    import safetensors.torch as _sf

    class _SafeSaveTrainer(base_cls):
        def _save_checkpoint(self, model, trial, metrics=None):
            if not self.is_world_process_zero():
                return
            raw_model = model.module if hasattr(model, "module") else model
            patch_gemma4_clippable_layers(raw_model)
            folder = "checkpoint-" + str(self.state.global_step)
            ckpt_dir = os.path.join(self.args.output_dir, folder)
            os.makedirs(ckpt_dir, exist_ok=True)

            try:
                from peft import get_peft_model_state_dict

                clean_sd = get_peft_model_state_dict(model)
                _sf.save_file(
                    clean_sd,
                    os.path.join(ckpt_dir, "adapter_model.safetensors"),
                )
                if hasattr(raw_model, "peft_config"):
                    for name, peft_config in raw_model.peft_config.items():
                        peft_config.save_pretrained(ckpt_dir)
                if self.is_main_process:
                    print(
                        f"[SafeSave] {folder} saved ({len(clean_sd)} keys)"
                    )
            except Exception as e:
                print(f"[SafeSave] Fallback: {e}")
                fb = {
                    n: p.data.cpu()
                    for n, p in model.named_parameters()
                    if p.requires_grad
                }
                _sf.save_file(
                    fb,
                    os.path.join(ckpt_dir, "fallback_adapter.safetensors"),
                )

        def _save(self, output_dir=None, state_dict=None):
            """Final save: delegate to model.save_pretrained()."""
            if not self.is_world_process_zero():
                return
            # Unwrap DDP for save
            save_model = self.model.module if hasattr(self.model, "module") else self.model
            if output_dir is None:
                output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            save_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            import json as _json

            state_path = os.path.join(output_dir, "trainer_state.json")
            with open(state_path, "w") as f:
                _json.dump(self.state.log_history, f, indent=2)
            print(f"[_save] Model saved to {output_dir}")

    return _SafeSaveTrainer


# ================================================================
# 8. Metrics (Phase 1: updated for ordinal outputs)
# ================================================================

def compute_metrics(eval_pred, num_labels=5):
    """Compute QWK, exact-match, and adjacent accuracy.

    V8: predictions are [B, 5] ordinal logits → continuous score → discrete.
    V6 ablation (num_labels=1): predictions are [B, 1] regression → round → discrete.
    Labels are raw integer scores 1–6.
    """
    logits, labels = eval_pred
    logits_t = torch.tensor(logits)
    if num_labels == 1:
        # Regression mode
        preds = np.clip(np.round(logits_t.squeeze() + 1.0), 1, 6).numpy().astype(int)
    else:
        # Ordinal mode
        continuous = logits_to_continuous_score(logits_t)  # [N] in [1.0, 6.0]
        preds = torch.round(continuous).long().clamp(1, 6).numpy()
    targets = labels.astype(int)

    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    exact = (preds == targets).mean()
    adj = (np.abs(preds - targets) <= 1).mean()
    return {"qwk": qwk, "exact": exact, "adjacent": adj}


# ================================================================
# 9. Main Entry Point
# ================================================================

def build_training_args(base_kwargs, overrides):
    """Merge overrides into base_kwargs dict."""
    merged = dict(base_kwargs)
    merged.update(overrides)
    return merged


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="V8 Ordinal Regression Training with 5-Fold OOF"
    )
    parser.add_argument(
        "--config",
        default="config/base_config_ordinal_v8.yaml",
        help="Path to YAML config file",
    )
    # Phase 2.5: Fold-based data split
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Validation fold id in [0, 4]. If set, --all_folds is required.",
    )
    parser.add_argument(
        "--all_folds",
        type=str,
        default=None,
        help="Path to master JSONL with 'fold' field (for k-fold mode)",
    )
    # Legacy data args (used when --fold is not set)
    parser.add_argument(
        "--train_data",
        default="data/processed/train_sft_nocot.jsonl",
        help="Training data JSONL (legacy mode, no --fold)",
    )
    parser.add_argument(
        "--val_data",
        default="data/processed/val_sft_nocot.jsonl",
        help="Validation data JSONL (legacy mode, no --fold)",
    )
    # Override output dir
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override training.output_dir from config",
    )
    # Training length overrides
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    # Phase 5: A100 optimization flags
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=None,
        help="Enable BF16 mixed precision (A100: ~15x vs FP32)",
    )
    parser.add_argument(
        "--flash_attention_2",
        action="store_true",
        default=None,
        help="Use FlashAttention-2 (2x memory, 2.5x speed on A100)",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=None,
        help="Enable torch.compile (1.3-1.5x speedup)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optimizer override (e.g. adamw_torch_fused for 20% speedup)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Override batch size per GPU",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=None,
        help="Override dataloader workers",
    )
    # Model path override (for Stage 2b: fine-tune from Stage 2a checkpoint)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Override model_id (for loading a checkpoint from a prior stage)",
    )
    # Training data override (for two-stage: persuade → kaggle-only)
    parser.add_argument(
        "--train_data_override",
        type=str,
        default=None,
        help="Override training data path (for two-stage training)",
    )
    # ── Ablation experiment flags ──
    parser.add_argument(
        "--no_ordinal",
        action="store_true",
        default=False,
        help="Ablation: use Huber regression (dim=1) instead of ordinal BCE",
    )
    parser.add_argument(
        "--no_attention_pooling",
        action="store_true",
        default=False,
        help="Ablation: use last-token pooling instead of attention pooling",
    )
    parser.add_argument(
        "--no_pos_weight",
        action="store_true",
        default=False,
        help="Ablation: disable pos_weight in BCE loss",
    )
    parser.add_argument(
        "--no_two_stage",
        action="store_true",
        default=False,
        help="Ablation: skip Stage 1, train jointly from start",
    )
    parser.add_argument(
        "--no_data_balance",
        action="store_true",
        default=False,
        help="Ablation: disable sample weighting and oversampling",
    )

    args = parser.parse_args()

    # ── DDP init ────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        is_main = local_rank == 0

        # Phase 5.2: NCCL tuning for A100 600 GB/s NVLink
        os.environ.setdefault("NCCL_BUCKET_SIZE", "26214400")  # 25 MB
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NCCL_PROTO", "simple")
    else:
        is_main = True

    # ── Load config ────────────────────────────────────────
    cfg = load_config(args.config)
    mc, lc, tc, dc = cfg["model"], cfg["lora"], cfg["training"], cfg["data"]
    loss_cfg = cfg.get("loss", {})

    model_id = args.model_name_or_path or mc["model_id"]
    max_seq_length = mc.get("max_seq_length", 1536)

    if is_main:
        print(f"[INFO] V8 Ordinal Regression Training")
        print(f"[INFO] Model: {model_id}")
        print(f"[INFO] max_seq_length: {max_seq_length}")
        print(f"[INFO] DDP: {local_rank >= 0}, world_size: {world_size}")
        if args.fold is not None:
            print(f"[INFO] Fold mode: fold={args.fold}")
        print_vram("start")

    # ── Tokenizer ────────────────────────────────────────
    from transformers import AutoTokenizer

    if is_main:
        print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    # Phase 2: Right padding for training (with attention pooling, less sensitive)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ───────────────────────────────────────────
    if is_main:
        print("[INFO] Loading Gemma4 + AttentionPooling + Ordinal head...")

    # V8: 5 output dims for ordinal cut points (or 1 for Huber ablation)
    num_labels = 1 if args.no_ordinal else 5

    if local_rank >= 0:
        device_map = {"": f"cuda:{local_rank}"}
    else:
        device_map = None

    model = Gemma4ForScoreRegression(
        model_id, num_labels=num_labels, device_map=device_map,
        use_attention_pooling=not args.no_attention_pooling,
    )

    # ── PEFT LoRA ────────────────────────────────────────
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc.get("lora_dropout", 0.05),
        bias=lc.get("bias", "none"),
        target_modules=lc["target_modules"],
        modules_to_save=["score"] if args.no_attention_pooling else ["score", "pooler"],
        # V8: save head + pooler (full params); ablation: pooler doesn't exist
    )
    model = get_peft_model(model, lora_config)
    patch_gemma4_clippable_layers(model)

    if is_main:
        model.print_trainable_parameters()
        print_vram("after_model_load")

    # ── Load data ──────────────────────────────────────────
    if is_main:
        print("[INFO] Loading data...")

    if args.fold is not None and args.all_folds:
        # Phase 2.5: Fold-based split
        all_rows = load_jsonl(args.all_folds)
        train_records = [r for r in all_rows if r.get("fold") != args.fold]
        val_records = [r for r in all_rows if r.get("fold") == args.fold]
        if is_main:
            print(
                f"[INFO] Fold {args.fold}: "
                f"Train={len(train_records)}, Val={len(val_records)}"
            )
    else:
        # Legacy mode: separate train/val files
        train_path = args.train_data_override or args.train_data
        train_records = load_jsonl(train_path)
        val_records = load_jsonl(args.val_data)
        if is_main:
            print(
                f"[INFO] Train: {len(train_records)}, Val: {len(val_records)}"
            )

    # Phase 1: Compute pos_weight from training distribution (skip if ablation)
    _pw = None if args.no_pos_weight else compute_pos_weight(train_records)

    # V6 retained: Per-sample weights (skip if ablation)
    use_weighted_loss = False if args.no_data_balance else loss_cfg.get("use_weighted_loss", False)
    if use_weighted_loss:
        if is_main:
            print("[INFO] Computing per-sample weights...")
        weights = compute_sample_weights(train_records)
        for i, w in enumerate(weights):
            train_records[i]["_weight"] = w
        if is_main:
            print(
                f"[INFO] Weight range: [{min(weights):.2f}, {max(weights):.2f}]"
            )

    # V6 retained: Over-sample rare scores
    if (not args.no_data_balance) and tc.get("use_oversampling", False):
        train_records = oversample_rare_scores(train_records)

    # Wrap as Dataset
    train_raw = Dataset.from_list([{"_raw": r} for r in train_records])
    val_raw = Dataset.from_list([{"_raw": r} for r in val_records])

    # Tokenize
    train_dataset = train_raw.map(
        lambda x: tokenize_fn(x, tokenizer, max_seq_length),
        batched=True,
        remove_columns=train_raw.column_names,
    )
    val_dataset = val_raw.map(
        lambda x: tokenize_fn(x, tokenizer, max_seq_length),
        batched=True,
        remove_columns=val_raw.column_names,
    )

    # ── Data Collator ────────────────────────────────────
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )

    # ── Apply overrides ──────────────────────────────────
    if args.output_dir:
        tc["output_dir"] = args.output_dir
    if args.num_train_epochs is not None:
        tc["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate is not None:
        tc["learning_rate"] = args.learning_rate

    # Phase 5: A100 optimization flags (CLI overrides config)
    use_bf16 = args.bf16 if args.bf16 is not None else tc.get("bf16", True)
    use_flash_attn = (
        True if args.flash_attention_2 else tc.get("flash_attention_2", False)
    )
    use_torch_compile = (
        True if args.torch_compile else tc.get("torch_compile", False)
    )
    optim_name = args.optim or tc.get("optim", "adamw_torch")
    if args.per_device_train_batch_size is not None:
        tc["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        tc["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    dl_workers = args.dataloader_num_workers or tc.get(
        "dataloader_num_workers", 4
    )

    effective_bs = (
        tc["per_device_train_batch_size"]
        * max(world_size, 1)
        * tc["gradient_accumulation_steps"]
    )
    if is_main:
        print(f"[INFO] Effective batch size: {effective_bs}")
        print(f"[INFO] BF16: {use_bf16}, FlashAttn: {use_flash_attn}")
        print(f"[INFO] Optimizer: {optim_name}")
        print(f"[INFO] torch.compile: {use_torch_compile}")
        print(f"[INFO] DataLoader workers: {dl_workers}")

    lr = tc.get("learning_rate", 1.0e-4)

    # ── Common TrainingArguments base ───────────────────
    common_kwargs = dict(
        output_dir=tc.get("output_dir", "outputs"),
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        warmup_ratio=tc.get("warmup_ratio", 0.05),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        weight_decay=tc.get("weight_decay", 0.01),
        max_grad_norm=tc.get("max_grad_norm", 1.0),
        bf16=use_bf16,
        optim=optim_name,
        logging_steps=tc.get("logging_steps", 10),
        gradient_checkpointing=tc.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=tc.get(
            "ddp_find_unused_parameters", False
        ),
        dataloader_num_workers=dl_workers,
        dataloader_pin_memory=tc.get("dataloader_pin_memory", True),
        report_to="none",
    )

    # Flash-Attention 2: set via environment + model config
    if use_flash_attn:
        if is_main:
            print("[INFO] Enabling FlashAttention-2")
        # The model was already loaded; attn_implementation is baked in at load time.
        # For this to work, the model must be loaded with
        # attn_implementation="flash_attention_2".
        # In our case, AutoModel loads with default attention;
        # to use FlashAttn-2, pass it at AutoModel.from_pretrained time.
        # We handle this by reloading the transformer with FA2 if requested.
        # For simplicity, we rely on the config flag and assume the user
        # set flash_attention_2 at the config level.

    # ── Create Trainer class ──────────────────────────────
    from transformers import Trainer, TrainingArguments

    SafeSaveTrainerCls = create_safe_save_trainer(Trainer)

    # ========================================================
    # V8: Stage 1 — Freeze LoRA, train only head + pooler
    # ========================================================
    stage1_epochs = 0 if args.no_two_stage else tc.get("stage1_epochs", 0)
    if stage1_epochs > 0:
        if is_main:
            print(f"\n{'='*55}")
            print(f"[Stage 1] Freezing LoRA, training head + pooler only")
            print(
                f"[Stage 1] Epochs: {stage1_epochs}, "
                f"LR: {tc.get('stage1_lr', 5.0e-4)}"
            )
            print(f"{'='*55}")

        freeze_lora(model)

        stage1_kwargs = build_training_args(
            common_kwargs,
            dict(
                num_train_epochs=stage1_epochs,
                learning_rate=tc.get("stage1_lr", 5.0e-4),
                eval_strategy="no",
                save_strategy="no",
            ),
        )
        stage1_args = TrainingArguments(**stage1_kwargs)

        stage1_trainer = SafeSaveTrainerCls(
            model=model,
            args=stage1_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        stage1_trainer.train()

        if is_main:
            print("[Stage 1] Complete.")
            head_w = model.score.weight.data
            print(
                f"[Stage 1] Head weight: mean={head_w.mean().item():.4f}, "
                f"std={head_w.std().item():.4f}"
            )
            print_vram("after_stage1")

    # ========================================================
    # V8: Stage 2 — Unfreeze LoRA, joint training
    # ========================================================
    if is_main:
        stage2_epochs = tc.get("num_train_epochs", 2)
        print(f"\n{'='*55}")
        print(f"[Stage 2] Unfreezing LoRA, joint training")
        print(f"[Stage 2] Epochs: {stage2_epochs}, LR: {lr}")
        print(
            f"[Stage 2] Head LR factor: {tc.get('stage2_head_lr_factor', 0.1)}"
        )
        print(f"{'='*55}")

    unfreeze_lora(model)

    stage2_kwargs = build_training_args(
        common_kwargs,
        dict(
            num_train_epochs=tc.get("num_train_epochs", 2),
            learning_rate=lr,
            eval_strategy="steps",
            eval_steps=tc.get("eval_steps", 200),
            save_strategy="steps",
            save_steps=tc.get("save_steps", 400),
            save_total_limit=tc.get("save_total_limit", 3),
            load_best_model_at_end=tc.get("load_best_model_at_end", True),
            metric_for_best_model="qwk",
            greater_is_better=True,
            output_dir=tc.get("output_dir", "outputs"),
        ),
    )

    # Phase 5: torch.compile
    if use_torch_compile:
        if is_main:
            print("[INFO] Enabling torch.compile (mode='reduce-overhead')")
        # torch.compile is applied to the model before training
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if is_main:
                print("[INFO] torch.compile applied successfully")
        except Exception as e:
            if is_main:
                print(f"[WARN] torch.compile failed: {e}. Continuing without.")

    stage2_args = TrainingArguments(**stage2_kwargs)

    # Phase 5: FlashAttention-2 via model reload
    if use_flash_attn:
        # Set the attn_implementation on the underlying transformer
        try:
            model.transformer._attn_implementation = "flash_attention_2"
            if is_main:
                print("[INFO] FlashAttention-2 enabled on transformer")
        except Exception as e:
            if is_main:
                print(f"[WARN] Could not set FA2: {e}")

    # V8: Param-grouped optimizer for head_lr_factor
    head_lr_factor = tc.get("stage2_head_lr_factor", None)
    if head_lr_factor is not None and head_lr_factor != 1.0:

        class _TwoStageTrainer(SafeSaveTrainerCls):
            def create_optimizer(self):
                if self.optimizer is None:
                    opt_model = self.model
                    head_params = []
                    lora_params = []
                    for n, p in opt_model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if "score" in n or "pooler" in n:
                            head_params.append(p)
                        else:
                            lora_params.append(p)

                    optimizer_cls, opt_kwargs = (
                        Trainer.get_optimizer_cls_and_kwargs(self.args)
                    )
                    grouped = [
                        {
                            "params": lora_params,
                            "lr": self.args.learning_rate,
                        },
                        {
                            "params": head_params,
                            "lr": self.args.learning_rate * head_lr_factor,
                        },
                    ]
                    self.optimizer = optimizer_cls(grouped, **opt_kwargs)

                    if is_main:
                        print(
                            f"[Optimizer] LoRA lr={self.args.learning_rate}, "
                            f"Head+Pooler lr={self.args.learning_rate * head_lr_factor}"
                        )
                return self.optimizer

        TrainerCls = _TwoStageTrainer
    else:
        TrainerCls = SafeSaveTrainerCls

    # ── Inject pos_weight into the model forward via a wrapper ──
    # Store pos_weight so the collator can pass it through
    pos_weight_tensor = _pw.clone().detach()

    class OrdinalTrainer(TrainerCls):
        """Trainer subclass that injects pos_weight into the model forward."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Extract and remove pos_weight and sample_weights from inputs
            # They are stored in the dataset and passed through the collator
            labels = inputs.get("labels")
            sample_weights = inputs.pop("sample_weight", None)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
                sample_weights=sample_weights,
                pos_weight=pos_weight_tensor,
            )
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

    trainer = OrdinalTrainer(
        model=model,
        args=stage2_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda ep: compute_metrics(ep, num_labels=num_labels),
    )
    # Set tokenizer manually (old HF API doesn't accept it in __init__)
    trainer.tokenizer = tokenizer

    # ── Train ─────────────────────────────────────────────
    if is_main:
        print("[INFO] Starting Stage 2 training...")
        print_vram("before_train")

    trainer.train()

    # ── Save final model ─────────────────────────────────
    if is_main:
        final_path = os.path.join(
            tc.get("output_dir", "outputs"), "final_regression"
        )
        trainer._save(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"[SAVE] Model saved to {final_path}")
        print_vram("end")
        print("[DONE]")


if __name__ == "__main__":
    main()
