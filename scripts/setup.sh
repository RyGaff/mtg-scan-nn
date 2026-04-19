#!/usr/bin/env bash
# Bootstrap a fresh GPU machine for mtg-card-encoder training.
#
# Assumes: conda/miniconda installed, NVIDIA drivers + CUDA toolkit present
# (for CUDA wheels) OR Apple Silicon (for MPS). CPU-only works too, just slow.
#
# Run from the repo root:
#   bash scripts/setup.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-mtg-encoder}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "== creating conda env '$ENV_NAME' (python $PYTHON_VERSION) =="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env '$ENV_NAME' already exists — skipping create"
else
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

# Activate via conda shell hook so this works in non-interactive bash.
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "== installing project (editable) + dev deps =="
pip install --upgrade pip
pip install -e ".[dev]"

echo "== verifying imports =="
python -c "
import torch, timm, pytorch_metric_learning, albumentations, coremltools, tensorflow, onnx, onnx2tf
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(),
      'mps?', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
print('all imports OK')
"

echo "== next steps =="
echo "  conda activate $ENV_NAME"
echo "  bash scripts/download_data.sh     # ~35k images, ~15 min"
echo "  bash scripts/train_full.sh        # 30 epochs (GPU recommended)"
echo "  bash scripts/release.sh           # evaluate + export artifacts"
