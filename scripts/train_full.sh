#!/usr/bin/env bash
# Full training: 30 epochs, batch 64, AdamW + cosine, BatchHard triplet.
# Requires ~35k images downloaded via scripts/download_data.sh first.
# Checkpoints saved to artifacts/ckpt_epoch*.pt and artifacts/card_encoder.pt.
#
# Device auto-selects cuda → mps → cpu. Override with MTG_DEVICE env var:
#   MTG_DEVICE=cuda:1 bash scripts/train_full.sh
set -euo pipefail
python -m src.train "$@"
