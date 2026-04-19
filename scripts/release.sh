#!/usr/bin/env bash
# After training completes: evaluate on held-out set, benchmark latency,
# then export all artifacts. All stdout is tee'd to artifacts/release.log.
set -euo pipefail

CKPT="${CKPT:-artifacts/card_encoder.pt}"
if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 1
fi

mkdir -p artifacts
LOG="artifacts/release.log"
exec > >(tee -a "$LOG") 2>&1

echo "== release $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
echo "ckpt: $CKPT"

echo "== evaluating =="
python -m src.evaluate --ckpt "$CKPT" --augs-per-eval 5

echo "== exporting artifacts =="
# Export first so the .mlmodel + .tflite exist before we benchmark them.
python -m src.export --ckpt "$CKPT"

echo "== benchmarking =="
python -m src.benchmark --n 100

echo "== re-exporting to fold benchmark results into manifest =="
python -m src.export --ckpt "$CKPT"

echo "== artifacts =="
ls -lh artifacts/card_encoder.mlmodel artifacts/card_encoder.tflite \
       artifacts/card_embeds_v2.bin   artifacts/manifest.json \
       artifacts/eval_results.json    artifacts/benchmark_results.json \
       artifacts/train_log.json       2>/dev/null || true
