"""Auto-pick a torch device: cuda → cpu.

MPS is skipped by default. Triplet loss uses `_cdist_backward`, which MPS
doesn't implement natively — you can still opt in via `MTG_DEVICE=mps` plus
`PYTORCH_ENABLE_MPS_FALLBACK=1`, but training will fall back to CPU for that
op anyway, so cuda/cpu are the supported fast paths.
"""
import os
import torch


def pick_device() -> str:
    override = os.environ.get("MTG_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
