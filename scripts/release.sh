#!/usr/bin/env bash
# After training completes: evaluate on held-out set, then export all artifacts.
set -euo pipefail

CKPT="${CKPT:-artifacts/card_encoder.pt}"
if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "== evaluating $CKPT =="
python -m src.evaluate --ckpt "$CKPT" --augs-per-eval 5

echo "== exporting artifacts =="
python -m src.export --ckpt "$CKPT"

echo "== artifacts =="
ls -lh artifacts/card_encoder.mlmodel artifacts/card_encoder.tflite \
       artifacts/card_embeds_v2.bin artifacts/manifest.json
