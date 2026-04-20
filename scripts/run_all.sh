#!/usr/bin/env bash
# End-to-end pipeline: setup → download → train → release.
#
# Prereqs: conda/miniconda installed. Run from repo root.
# Resumable: data download skips existing files, training can resume via
# MTG_RESUME=<ckpt path>, release re-runs cheap.
#
# Usage:
#   bash scripts/run_all.sh              # full run
#   bash scripts/run_all.sh --smoke      # 1 epoch on 50 cards (~3 min)
#   MTG_DEVICE=cuda:1 bash scripts/run_all.sh
#   SKIP_DOWNLOAD=1 bash scripts/run_all.sh   # images already on disk
#   SKIP_TRAIN=1    bash scripts/run_all.sh   # re-export from existing ckpt

set -euo pipefail

ENV_NAME="${ENV_NAME:-mtg-encoder}"
CKPT="${CKPT:-artifacts/card_encoder.pt}"
SMOKE_FLAG=""
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE_FLAG="--smoke"
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf "\n== %s ==\n" "$*"; }

log "phase 1/5: env"
if ! command -v conda >/dev/null; then
  echo "conda not found on PATH" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  bash scripts/setup.sh
fi
conda activate "$ENV_NAME"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

log "phase 2/5: data"
if [[ -n "${SKIP_DOWNLOAD:-}" ]]; then
  echo "SKIP_DOWNLOAD set, skipping"
elif [[ -n "$SMOKE_FLAG" ]]; then
  python -m src.download_images --limit 50
else
  python -m src.download_images
fi

log "phase 3/5: train"
if [[ -n "${SKIP_TRAIN:-}" && -f "$CKPT" ]]; then
  echo "SKIP_TRAIN set and $CKPT exists, skipping"
else
  TRAIN_ARGS=()
  [[ -n "$SMOKE_FLAG" ]] && TRAIN_ARGS+=("$SMOKE_FLAG")
  if [[ -n "${MTG_RESUME:-}" ]]; then
    TRAIN_ARGS+=(--resume "$MTG_RESUME")
  fi
  python -m src.train "${TRAIN_ARGS[@]}"
fi

log "phase 4/5: evaluate"
python -m src.evaluate --ckpt "$CKPT" --augs-per-eval 5

log "phase 5/5: release (export + benchmark + manifest)"
bash scripts/release.sh

log "done"
echo "final manifest:"
python -c "import json; m=json.load(open('artifacts/manifest.json')); \
           print(f'  version        : {m[\"version\"]}'); \
           print(f'  coreml         : {m[\"encoder\"][\"coreml\"][\"path\"]}'); \
           print(f'  tflite         : {m[\"encoder\"][\"tflite\"][\"path\"]}'); \
           print(f'  embeddings     : {m[\"embeddings\"][\"count\"]} × {m[\"embeddings\"][\"dim\"]}-d'); \
           e=m.get('eval',{}); print(f'  eval top-1     : {e.get(\"top1\",0)*100:.2f}%'); \
           print(f'  eval top-3     : {e.get(\"top3\",0)*100:.2f}%'); \
           b=m.get('benchmark',{}); print(f'  coreml latency : {b.get(\"coreml_ms_per_image\",0):.2f} ms/img'); \
           print(f'  tflite latency : {b.get(\"tflite_ms_per_image\",0):.2f} ms/img')"
echo ""
echo "artifacts to ship:"
ls -lh artifacts/card_encoder.mlmodel artifacts/card_encoder.tflite \
       artifacts/card_embeds_v2.bin   artifacts/manifest.json
