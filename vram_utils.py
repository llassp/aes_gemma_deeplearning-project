"""Shared VRAM and GPU utilities for AES 2.0 pipeline."""
import gc
import torch


def clear_vram() -> None:
    """Force release GPU memory between pipeline phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_vram(label: str = "") -> None:
    """Print current and peak VRAM usage across all visible GPUs."""
    if not torch.cuda.is_available():
        print(f"[VRAM {label}] CUDA not available")
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        peak = torch.cuda.max_memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"[VRAM {label}] GPU {i}: alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")
