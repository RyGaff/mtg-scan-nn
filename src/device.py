"""Auto-pick the best available torch device (cuda → mps → cpu)."""
import os
import torch


def pick_device() -> str:
    override = os.environ.get("MTG_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
