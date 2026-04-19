# MTG Card Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a 256-dim image-embedding model that identifies any Magic: The Gathering card from a rectified 400×560 RGB photo, exporting CoreML + TFLite artifacts (≤5 MB each) and a precomputed embedding binary for ~35k unique cards.

**Architecture:** MobileNetV3-Small (timm, ImageNet pretrained) → GAP → Linear(576→256) → L2Normalize. BatchHard triplet loss with online hard positive/negative mining on same-card augmentations. Preprocessing (mean/std normalize + resize 224×224) is baked into Layer 0 of exported models so the mobile consumer feeds raw uint8/255 RGB in `[0,1]`.

**Tech Stack:** Python 3.10+, PyTorch 2.x, `timm`, `pytorch_metric_learning`, `albumentations` (augmentations), `coremltools`, `tensorflow` + `tf.lite` (via `onnx2tf` or direct), `pillow`, `numpy`, `requests`, `tqdm`, `pytest`.

---

## File Structure

```
mtg-card-encoder/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py              # hyperparameters, paths, constants
│   ├── scryfall.py            # bulk data fetch
│   ├── download_images.py     # CLI: fetch unique_artwork + images
│   ├── augment.py             # albumentations pipeline
│   ├── dataset.py             # CardDataset (anchor+positive pairs)
│   ├── model.py               # CardEncoder module
│   ├── losses.py              # BatchHard triplet wrapper
│   ├── train.py               # CLI: training loop
│   ├── evaluate.py            # CLI: top-1/top-3 on held-out
│   ├── embed_binary.py        # read/write card_embeds_v2.bin
│   ├── export.py              # CLI: CoreML + TFLite + embeddings + manifest
│   └── benchmark.py           # CoreML/TFLite latency
├── tests/
│   ├── __init__.py
│   ├── test_scryfall.py
│   ├── test_augment.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_losses.py
│   ├── test_embed_binary.py
│   ├── test_export.py
│   └── fixtures/
│       └── sample_card.jpg
├── data/                       # .gitignored
│   ├── unique_artwork.json
│   └── images/
└── artifacts/                  # .gitignored, output of export.py
    ├── card_encoder.mlmodel
    ├── card_encoder.tflite
    ├── card_embeds_v2.bin
    └── manifest.json
```

---

## Task 1: Repo Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `src/config.py`

- [ ] **Step 1.1: Init git + create directory tree**

```bash
cd /Users/rgaffney/sources/lotusfield/mtg-scan-nn
git init
mkdir -p src tests tests/fixtures data/images artifacts docs/superpowers/plans
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 1.2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
venv/
.env
data/
artifacts/
*.mlmodel
*.tflite
*.bin
*.pt
*.ckpt
.DS_Store
.pytest_cache/
.ruff_cache/
*.egg-info/
wandb/
```

- [ ] **Step 1.3: Write `pyproject.toml`**

```toml
[project]
name = "mtg-card-encoder"
version = "0.1.0"
description = "Train and export a 256-dim MTG card image embedding model"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.2",
  "torchvision>=0.17",
  "timm>=1.0",
  "pytorch-metric-learning>=2.5",
  "albumentations>=1.4",
  "opencv-python-headless>=4.9",
  "pillow>=10.0",
  "numpy>=1.26",
  "requests>=2.31",
  "tqdm>=4.66",
  "coremltools>=7.2",
  "tensorflow>=2.15",
  "onnx>=1.16",
  "onnx2tf>=1.22",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 1.4: Write `src/config.py`**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
ARTIFACTS_DIR = ROOT / "artifacts"
UNIQUE_ARTWORK_JSON = DATA_DIR / "unique_artwork.json"

EMBED_DIM = 256
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
TRIPLET_MARGIN = 0.2

MANIFEST_VERSION = 2
BIN_MAGIC = 0x4547544D  # little-endian bytes spell ASCII 'MTGE'

SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data"
# Scryfall asks for 50-100ms between requests and a descriptive User-Agent.
# https://scryfall.com/docs/api#rate-limits-and-good-citizenship
SCRYFALL_USER_AGENT = "mtg-card-encoder/0.1 (+https://github.com/lotusfield)"
SCRYFALL_REQUEST_DELAY_MS = 100  # per-worker throttle on image fetches
SCRYFALL_IMAGE_WORKERS = 4       # → ~40 req/s aggregate, well under limit
```

- [ ] **Step 1.5: Install + verify**

Run: `pip install -e ".[dev]"`
Expected: successful install.

Run: `python -c "import torch, timm, pytorch_metric_learning, albumentations, coremltools; print('ok')"`
Expected: `ok`

- [ ] **Step 1.6: Commit**

```bash
git add pyproject.toml .gitignore src/__init__.py tests/__init__.py src/config.py
git commit -m "chore: scaffold mtg-card-encoder project"
```

---

## Task 2: README with Contract

**Files:**
- Create: `README.md`

- [ ] **Step 2.1: Write `README.md`**

```markdown
# mtg-card-encoder

Train a compact on-device image-embedding model that identifies any Magic: The Gathering card from a rectified photograph.

## I/O Contract (non-negotiable)

### Encoder
- **Input:** `1×224×224×3` RGB tensor, values in `[0, 1]` (uint8 / 255). ImageNet mean/std normalization is baked into Layer 0.
- **Output:** `1×256` Float32, L2-normalized. Dot-product = cosine similarity.
- **Latency target:** ≤100 ms on iPhone 12 / Pixel 6.

### Embeddings binary (`card_embeds_v2.bin`, little-endian)
Header:
| offset | size | field |
|--------|------|-------|
| 0      | 4    | ASCII magic = `"MTGE"` (bytes `0x4D 0x54 0x47 0x45`; equals uint32 LE `0x4547544D`) |
| 4      | 4    | uint32 version = 2 |
| 8      | 4    | uint32 count N |
| 12     | 4    | uint32 dim D = 256 |
| 16     | 4    | uint32 model_hash (first 32 bits of SHA-256 of encoder file) |

Records (N times):
- 36 bytes: ASCII UUID `scryfall_id`
- D×4 bytes: float32 L2-normalized embedding

### Manifest (`manifest.json`)
```json
{
  "version": 2,
  "encoder": {
    "coreml": {"path": "card_encoder.mlmodel", "sha256": "..."},
    "tflite": {"path": "card_encoder.tflite", "sha256": "..."}
  },
  "embeddings": {"path": "card_embeds_v2.bin", "sha256": "...", "count": 35123}
}
```

Encoder `version` MUST match embeddings `version` — they are bound.

## Pipeline

1. `src/download_images.py` — fetch Scryfall `unique_artwork` + card images
2. `src/train.py` — BatchHard triplet training (MobileNetV3-Small → 256-d)
3. `src/evaluate.py` — top-1/top-3 on held-out augmented eval set
4. `src/export.py` — write CoreML, TFLite, embeddings, manifest
5. `src/benchmark.py` — measure inference latency

## Targets

- Top-1 ≥98% on held-out augmented cards
- Top-1 ≥95% on real-world phone photo test set
- Encoder file size ≤5 MB each (CoreML, TFLite)

## Out of scope

- Mobile app integration (consumer repo)
- Card detection / rectification (already solved)
- OCR

## Scryfall data usage

- Metadata: one-shot download of `unique_artwork` via `https://api.scryfall.com/bulk-data` — never loop queries against `/cards/...` endpoints.
- Images: per-card fetches from `cards.scryfall.io` CDN, rate-limited to ~40 req/s aggregate (4 workers × 100 ms per-worker delay, honors 429 `Retry-After`, descriptive User-Agent). Resumable — re-running skips already-downloaded files.
- Attribution per Scryfall guidelines: card images and data © Wizards of the Coast, served by Scryfall. This project redistributes only the derived 256-d embeddings, not the source images.
```

- [ ] **Step 2.2: Commit**

```bash
git add README.md
git commit -m "docs: add README with I/O contract"
```

---

## Task 3: Scryfall Bulk Data Fetch (TDD)

**Files:**
- Create: `src/scryfall.py`
- Test: `tests/test_scryfall.py`

- [ ] **Step 3.1: Write failing test**

```python
# tests/test_scryfall.py
from unittest.mock import patch, MagicMock
from src.scryfall import get_unique_artwork_url, download_unique_artwork

def test_get_unique_artwork_url_filters_for_unique_artwork():
    bulk_response = {
        "data": [
            {"type": "default_cards", "download_uri": "https://x/default.json"},
            {"type": "unique_artwork", "download_uri": "https://x/unique.json"},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = bulk_response
    mock_resp.raise_for_status = MagicMock()
    with patch("src.scryfall.requests.get", return_value=mock_resp):
        url = get_unique_artwork_url()
    assert url == "https://x/unique.json"

def test_download_unique_artwork_writes_json(tmp_path):
    cards = [{"id": "abc", "name": "Lightning Bolt"}]
    mock_bulk = MagicMock()
    mock_bulk.json.return_value = {"data": [
        {"type": "unique_artwork", "download_uri": "https://x/u.json"}]}
    mock_bulk.raise_for_status = MagicMock()
    mock_data = MagicMock()
    mock_data.json.return_value = cards
    mock_data.raise_for_status = MagicMock()
    dest = tmp_path / "u.json"
    with patch("src.scryfall.requests.get", side_effect=[mock_bulk, mock_data]):
        download_unique_artwork(dest)
    import json
    assert json.loads(dest.read_text()) == cards
```

- [ ] **Step 3.2: Run test, expect FAIL**

Run: `pytest tests/test_scryfall.py -v`
Expected: `ImportError` / `ModuleNotFoundError` for `src.scryfall`.

- [ ] **Step 3.3: Implement `src/scryfall.py`**

```python
import json
from pathlib import Path
import requests
from src.config import SCRYFALL_BULK_URL, SCRYFALL_USER_AGENT

HEADERS = {"User-Agent": SCRYFALL_USER_AGENT, "Accept": "application/json"}

def get_unique_artwork_url() -> str:
    resp = requests.get(SCRYFALL_BULK_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    for entry in resp.json()["data"]:
        if entry["type"] == "unique_artwork":
            return entry["download_uri"]
    raise RuntimeError("unique_artwork bulk entry not found")

def download_unique_artwork(dest: Path) -> None:
    """Bulk JSON (~100 MB) — one request, not per-card."""
    url = get_unique_artwork_url()
    resp = requests.get(url, headers=HEADERS, timeout=300)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(resp.json()))
```

- [ ] **Step 3.4: Run test, expect PASS**

Run: `pytest tests/test_scryfall.py -v`
Expected: 2 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/scryfall.py tests/test_scryfall.py
git commit -m "feat: Scryfall unique_artwork bulk fetch"
```

---

## Task 4: Image Downloader CLI

**Files:**
- Create: `src/download_images.py`

Note: no unit test — this is an I/O driver. We smoke-test with 3 cards.

- [ ] **Step 4.1: Implement `src/download_images.py`**

**Rate-limit notes:** Scryfall bulk JSON is a single request (already done in Task 3). Card images live on `cards.scryfall.io` CDN but Scryfall still requests citizenship: 50–100 ms between requests per client, descriptive User-Agent, honor `Retry-After` on 429. Config defaults to 4 workers × 100 ms = ~40 req/s aggregate. Resumable: pre-existing files are skipped so re-running after a 429 burst is free.

```python
"""Fetch Scryfall unique_artwork + download each card's normal-res image.

Usage:
  python -m src.download_images              # fetch all
  python -m src.download_images --limit 10   # smoke test
  python -m src.download_images --refresh-bulk  # re-pull unique_artwork.json

Rate-limited per Scryfall good-citizenship guidelines. Resumable.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from tqdm import tqdm
from src.config import (IMAGES_DIR, UNIQUE_ARTWORK_JSON, SCRYFALL_USER_AGENT,
                        SCRYFALL_REQUEST_DELAY_MS, SCRYFALL_IMAGE_WORKERS)
from src.scryfall import download_unique_artwork

HEADERS = {"User-Agent": SCRYFALL_USER_AGENT}

def _image_uri(card: dict) -> str | None:
    if "image_uris" in card:
        return card["image_uris"].get("normal")
    # double-faced: take first face with a normal URI
    for face in card.get("card_faces") or []:
        uris = face.get("image_uris") or {}
        if "normal" in uris:
            return uris["normal"]
    return None

def _get_with_retry(url: str, session: requests.Session, max_retries: int = 5):
    """Retry on 429 w/ Retry-After, exponential backoff on 5xx / network errs."""
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", backoff))
                time.sleep(wait)
                backoff = min(backoff * 2, 30.0)
                continue
            if 500 <= r.status_code < 600:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    raise RuntimeError(f"exhausted retries for {url}")

def _download_one(card: dict, dest_dir: Path, session: requests.Session,
                  delay_s: float) -> str | None:
    scryfall_id = card["id"]
    out = dest_dir / f"{scryfall_id}.jpg"
    if out.exists():
        return scryfall_id  # resumable skip
    uri = _image_uri(card)
    if uri is None:
        return None
    try:
        r = _get_with_retry(uri, session)
        out.write_bytes(r.content)
        time.sleep(delay_s)  # per-worker throttle
        return scryfall_id
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=SCRYFALL_IMAGE_WORKERS)
    p.add_argument("--delay-ms", type=int, default=SCRYFALL_REQUEST_DELAY_MS)
    p.add_argument("--refresh-bulk", action="store_true")
    args = p.parse_args()

    if args.refresh_bulk or not UNIQUE_ARTWORK_JSON.exists():
        print(f"Downloading bulk unique_artwork to {UNIQUE_ARTWORK_JSON}")
        download_unique_artwork(UNIQUE_ARTWORK_JSON)

    cards = json.loads(UNIQUE_ARTWORK_JSON.read_text())
    if args.limit:
        cards = cards[: args.limit]
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    delay_s = args.delay_ms / 1000.0
    print(f"workers={args.workers}  delay={args.delay_ms}ms  "
          f"(~{args.workers/delay_s:.0f} req/s aggregate)")

    ok = fail = 0
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_download_one, c, IMAGES_DIR, session, delay_s)
                   for c in cards]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            if fut.result() is not None:
                ok += 1
            else:
                fail += 1
    print(f"done: {ok} ok, {fail} fail")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Smoke test**

Run: `python -m src.download_images --limit 3`
Expected: `data/unique_artwork.json` populated, 3 JPEGs in `data/images/`, `done: 3 ok, 0 fail`.

- [ ] **Step 4.3: Commit**

```bash
git add src/download_images.py
git commit -m "feat: parallel Scryfall image downloader"
```

---

## Task 5: Augmentation Pipeline (TDD)

**Files:**
- Create: `src/augment.py`
- Test: `tests/test_augment.py`
- Create: `tests/fixtures/sample_card.jpg` (small placeholder — copy one from `data/images/`)

- [ ] **Step 5.1: Add fixture**

```bash
cp data/images/$(ls data/images/ | head -1) tests/fixtures/sample_card.jpg
```

- [ ] **Step 5.2: Write failing test**

```python
# tests/test_augment.py
import numpy as np
from PIL import Image
from pathlib import Path
from src.augment import build_train_transform, build_eval_transform
from src.config import INPUT_SIZE

FIXTURE = Path(__file__).parent / "fixtures" / "sample_card.jpg"

def test_train_transform_outputs_correct_shape():
    img = np.array(Image.open(FIXTURE).convert("RGB"))
    t = build_train_transform()
    out = t(image=img)["image"]
    assert out.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert out.dtype == np.float32 or str(out.dtype).startswith("torch.float")

def test_train_transform_is_stochastic():
    img = np.array(Image.open(FIXTURE).convert("RGB"))
    t = build_train_transform()
    a = t(image=img)["image"]
    b = t(image=img)["image"]
    # at least one pixel differs
    import numpy as np
    a_np = a.numpy() if hasattr(a, "numpy") else a
    b_np = b.numpy() if hasattr(b, "numpy") else b
    assert not np.array_equal(a_np, b_np)

def test_eval_transform_is_deterministic():
    img = np.array(Image.open(FIXTURE).convert("RGB"))
    t = build_eval_transform()
    a = t(image=img)["image"]
    b = t(image=img)["image"]
    import numpy as np
    a_np = a.numpy() if hasattr(a, "numpy") else a
    b_np = b.numpy() if hasattr(b, "numpy") else b
    assert np.array_equal(a_np, b_np)
```

- [ ] **Step 5.3: Run test, expect FAIL**

Run: `pytest tests/test_augment.py -v`
Expected: ImportError.

- [ ] **Step 5.4: Implement `src/augment.py`**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from src.config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD

def build_train_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=int(INPUT_SIZE * 1.15)),
        A.PadIfNeeded(min_height=int(INPUT_SIZE*1.15), min_width=int(INPUT_SIZE*1.15),
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Perspective(scale=(0.02, 0.08), p=0.7),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
        A.RandomResizedCrop(height=INPUT_SIZE, width=INPUT_SIZE,
                            scale=(0.8, 1.0), ratio=(0.95, 1.05), p=1.0),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15,
                      hue=0.03, p=0.8),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.ImageCompression(quality_lower=40, quality_upper=95, p=0.5),
        # specular glare
        A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9), angle_lower=0,
                         src_radius=40, num_flare_circles_lower=1,
                         num_flare_circles_upper=3, p=0.25),
        # color temperature
        A.RGBShift(r_shift_limit=15, g_shift_limit=10, b_shift_limit=15, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def build_eval_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=INPUT_SIZE),
        A.PadIfNeeded(min_height=INPUT_SIZE, min_width=INPUT_SIZE,
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

- [ ] **Step 5.5: Run tests, expect PASS**

Run: `pytest tests/test_augment.py -v`
Expected: 3 passed.

- [ ] **Step 5.6: Commit**

```bash
git add src/augment.py tests/test_augment.py tests/fixtures/sample_card.jpg
git commit -m "feat: training/eval augmentation pipelines"
```

---

## Task 6: Dataset (TDD)

**Files:**
- Create: `src/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 6.1: Write failing test**

```python
# tests/test_dataset.py
import numpy as np
from PIL import Image
from pathlib import Path
import pytest
from src.dataset import CardDataset

@pytest.fixture
def tiny_image_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    ids = []
    for i in range(4):
        sid = f"00000000-0000-0000-0000-00000000000{i}"
        Image.new("RGB", (488, 680), color=(i*50, 100, 150)).save(d / f"{sid}.jpg")
        ids.append(sid)
    return d, ids

def test_dataset_length(tiny_image_dir):
    d, ids = tiny_image_dir
    ds = CardDataset(d, scryfall_ids=ids, samples_per_card=2, train=True)
    assert len(ds) == 8  # 4 cards * 2 samples

def test_dataset_returns_tensor_and_label(tiny_image_dir):
    import torch
    d, ids = tiny_image_dir
    ds = CardDataset(d, scryfall_ids=ids, samples_per_card=2, train=True)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < 4

def test_dataset_labels_stable_per_card(tiny_image_dir):
    d, ids = tiny_image_dir
    ds = CardDataset(d, scryfall_ids=ids, samples_per_card=3, train=True)
    label_of_id = {}
    for i in range(len(ds)):
        _, lbl = ds[i]
        sid = ds.scryfall_id_for_index(i)
        label_of_id.setdefault(sid, lbl)
        assert label_of_id[sid] == lbl
```

- [ ] **Step 6.2: Run tests, expect FAIL**

Run: `pytest tests/test_dataset.py -v`
Expected: ImportError.

- [ ] **Step 6.3: Implement `src/dataset.py`**

```python
from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from src.augment import build_train_transform, build_eval_transform

class CardDataset(Dataset):
    """Yields (image, label) where label = card index.

    For BatchHard triplet mining, each batch should contain multiple samples
    per card. We emit `samples_per_card` virtual samples per physical card;
    the DataLoader sampler (or default shuffle) plus batch size ensures
    hard triplets form naturally.
    """
    def __init__(self, images_dir: Path, scryfall_ids: Sequence[str],
                 samples_per_card: int = 4, train: bool = True):
        self.images_dir = Path(images_dir)
        self.scryfall_ids = list(scryfall_ids)
        self.samples_per_card = samples_per_card
        self.transform = build_train_transform() if train else build_eval_transform()

    def __len__(self) -> int:
        return len(self.scryfall_ids) * self.samples_per_card

    def scryfall_id_for_index(self, idx: int) -> str:
        return self.scryfall_ids[idx // self.samples_per_card]

    def __getitem__(self, idx: int):
        card_idx = idx // self.samples_per_card
        sid = self.scryfall_ids[card_idx]
        path = self.images_dir / f"{sid}.jpg"
        img = np.array(Image.open(path).convert("RGB"))
        out = self.transform(image=img)["image"]
        return out, card_idx
```

- [ ] **Step 6.4: Run tests, expect PASS**

Run: `pytest tests/test_dataset.py -v`
Expected: 3 passed.

- [ ] **Step 6.5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: CardDataset with per-card label emission"
```

---

## Task 7: Model (TDD)

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 7.1: Write failing test**

```python
# tests/test_model.py
import torch
from src.model import CardEncoder
from src.config import EMBED_DIM, INPUT_SIZE

def test_encoder_output_shape():
    m = CardEncoder()
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    y = m(x)
    assert y.shape == (2, EMBED_DIM)

def test_encoder_output_is_l2_normalized():
    m = CardEncoder()
    m.eval()
    x = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        y = m(x)
    norms = y.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_encoder_param_count_under_3m():
    m = CardEncoder()
    n = sum(p.numel() for p in m.parameters())
    # MobileNetV3-Small ~2.5M + 576*256 head
    assert n < 3_000_000, f"got {n}"
```

- [ ] **Step 7.2: Run tests, expect FAIL**

Run: `pytest tests/test_model.py -v`
Expected: ImportError.

- [ ] **Step 7.3: Implement `src/model.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.config import EMBED_DIM

class CardEncoder(nn.Module):
    def __init__(self, backbone: str = "mobilenetv3_small_100", embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features  # 576 for mobilenetv3_small_100
        self.head = nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        embed = self.head(feats)
        return F.normalize(embed, p=2, dim=1)
```

- [ ] **Step 7.4: Run tests, expect PASS**

Run: `pytest tests/test_model.py -v`
Expected: 3 passed.

- [ ] **Step 7.5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: CardEncoder MobileNetV3-Small + 256-d L2-normalized head"
```

---

## Task 8: Triplet Loss Wrapper (TDD)

**Files:**
- Create: `src/losses.py`
- Test: `tests/test_losses.py`

- [ ] **Step 8.1: Write failing test**

```python
# tests/test_losses.py
import torch
from src.losses import build_triplet_loss

def test_loss_returns_scalar():
    loss_fn = build_triplet_loss(margin=0.2)
    embeds = torch.randn(8, 256)
    embeds = torch.nn.functional.normalize(embeds, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(embeds, labels)
    assert loss.ndim == 0
    assert loss.item() >= 0

def test_loss_zero_when_perfect_separation():
    loss_fn = build_triplet_loss(margin=0.1)
    # two classes, orthogonal unit vectors → perfect separation
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    embeds = torch.cat([a, a, b, b], dim=0)
    labels = torch.tensor([0, 0, 1, 1])
    loss = loss_fn(embeds, labels)
    assert loss.item() == 0.0
```

- [ ] **Step 8.2: Run tests, expect FAIL**

Run: `pytest tests/test_losses.py -v`
Expected: ImportError.

- [ ] **Step 8.3: Implement `src/losses.py`**

```python
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner

class BatchHardTripletLoss:
    def __init__(self, margin: float):
        self.loss_fn = TripletMarginLoss(margin=margin)
        self.miner = BatchHardMiner()

    def __call__(self, embeddings, labels):
        triplets = self.miner(embeddings, labels)
        return self.loss_fn(embeddings, labels, triplets)

def build_triplet_loss(margin: float):
    return BatchHardTripletLoss(margin=margin)
```

- [ ] **Step 8.4: Run tests, expect PASS**

Run: `pytest tests/test_losses.py -v`
Expected: 2 passed.

- [ ] **Step 8.5: Commit**

```bash
git add src/losses.py tests/test_losses.py
git commit -m "feat: BatchHard triplet loss wrapper"
```

---

## Task 9: Training Loop

**Files:**
- Create: `src/train.py`

Note: no unit tests on training itself (wall-clock) — we smoke-test `--smoke` path (1 epoch, 50 cards).

- [ ] **Step 9.1: Implement `src/train.py`**

```python
"""Training loop for CardEncoder.

Usage:
  python -m src.train                     # full training
  python -m src.train --smoke             # 1 epoch, 50 cards, verify wiring
  python -m src.train --resume <ckpt>
"""
import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from src.config import (BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, TRIPLET_MARGIN,
                        IMAGES_DIR, UNIQUE_ARTWORK_JSON, ARTIFACTS_DIR)
from src.dataset import CardDataset
from src.model import CardEncoder
from src.losses import build_triplet_loss

def load_scryfall_ids(limit: int | None = None) -> list[str]:
    cards = json.loads(UNIQUE_ARTWORK_JSON.read_text())
    ids = [c["id"] for c in cards if (IMAGES_DIR / f"{c['id']}.jpg").exists()]
    if limit:
        ids = ids[:limit]
    return ids

def split_ids(ids: list[str], eval_frac: float = 0.1, seed: int = 42):
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * eval_frac)
    return shuffled[cut:], shuffled[:cut]

def train(smoke: bool = False, resume: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    limit = 50 if smoke else None
    epochs = 1 if smoke else EPOCHS
    batch = 16 if smoke else BATCH_SIZE

    all_ids = load_scryfall_ids(limit=limit)
    train_ids, eval_ids = split_ids(all_ids)
    print(f"train: {len(train_ids)} cards, eval: {len(eval_ids)} cards")

    train_ds = CardDataset(IMAGES_DIR, train_ids, samples_per_card=4, train=True)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    model = CardEncoder().to(device)
    loss_fn = build_triplet_loss(margin=TRIPLET_MARGIN)
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(optim, T_max=epochs * len(train_loader))

    start_epoch = 0
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            embeds = model(imgs)
            loss = loss_fn(embeds, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg = total_loss / max(len(train_loader), 1)
        print(f"epoch {epoch} avg loss: {avg:.4f}")

        ckpt = ARTIFACTS_DIR / f"ckpt_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "epoch": epoch,
                    "train_ids": train_ids,
                    "eval_ids": eval_ids}, ckpt)
        print(f"saved {ckpt}")

    final = ARTIFACTS_DIR / "card_encoder.pt"
    torch.save({"model": model.state_dict(),
                "train_ids": train_ids, "eval_ids": eval_ids}, final)
    print(f"saved {final}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(smoke=args.smoke, resume=args.resume)

if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Smoke run**

Prereq: run `python -m src.download_images --limit 50` first if not done.
Run: `python -m src.train --smoke`
Expected: 1 epoch completes without error, loss decreases or stays finite, `artifacts/card_encoder.pt` written.

- [ ] **Step 9.3: Commit**

```bash
git add src/train.py
git commit -m "feat: training loop (AdamW + cosine, BatchHard triplet)"
```

---

## Task 10: Embedding Binary Format (TDD)

**Files:**
- Create: `src/embed_binary.py`
- Test: `tests/test_embed_binary.py`

- [ ] **Step 10.1: Write failing test**

```python
# tests/test_embed_binary.py
import numpy as np
from pathlib import Path
from src.embed_binary import write_embeddings, read_embeddings

def test_roundtrip(tmp_path):
    ids = [f"{i:08x}-0000-0000-0000-000000000000" for i in range(3)]
    embeds = np.random.randn(3, 256).astype(np.float32)
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
    model_hash = 0xDEADBEEF
    path = tmp_path / "embeds.bin"
    write_embeddings(path, ids, embeds, model_hash=model_hash, version=2)
    read_ids, read_embeds, header = read_embeddings(path)
    assert read_ids == ids
    np.testing.assert_allclose(read_embeds, embeds, atol=1e-6)
    assert header["magic"] == 0x4547544D
    assert header["version"] == 2
    assert header["count"] == 3
    assert header["dim"] == 256
    assert header["model_hash"] == 0xDEADBEEF

def test_rejects_non_l2_normalized(tmp_path):
    import pytest
    ids = ["00000000-0000-0000-0000-000000000000"]
    bad = np.array([[1.0, 2.0] + [0.0]*254], dtype=np.float32)  # norm != 1
    with pytest.raises(ValueError, match="L2-normalized"):
        write_embeddings(tmp_path / "x.bin", ids, bad, model_hash=0, version=2)

def test_rejects_wrong_id_length(tmp_path):
    import pytest
    ids = ["short"]
    embeds = np.zeros((1, 256), dtype=np.float32)
    embeds[0, 0] = 1.0
    with pytest.raises(ValueError, match="36 char"):
        write_embeddings(tmp_path / "x.bin", ids, embeds, model_hash=0, version=2)
```

- [ ] **Step 10.2: Run tests, expect FAIL**

Run: `pytest tests/test_embed_binary.py -v`
Expected: ImportError.

- [ ] **Step 10.3: Implement `src/embed_binary.py`**

```python
import struct
from pathlib import Path
import numpy as np
from src.config import BIN_MAGIC

HEADER_FMT = "<IIIII"   # magic, version, count, dim, model_hash
HEADER_SIZE = struct.calcsize(HEADER_FMT)
ID_BYTES = 36

def write_embeddings(path: Path, scryfall_ids, embeds: np.ndarray,
                     model_hash: int, version: int) -> None:
    if embeds.ndim != 2 or embeds.dtype != np.float32:
        raise ValueError("embeds must be 2-D float32")
    n, d = embeds.shape
    if n != len(scryfall_ids):
        raise ValueError("id count != embedding count")
    for sid in scryfall_ids:
        if len(sid) != ID_BYTES:
            raise ValueError(f"expected 36 char UUID, got {len(sid)} for {sid!r}")
    norms = np.linalg.norm(embeds, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("embeddings must be L2-normalized")

    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, BIN_MAGIC, version, n, d, model_hash & 0xFFFFFFFF))
        for sid, emb in zip(scryfall_ids, embeds):
            f.write(sid.encode("ascii"))
            f.write(emb.tobytes())

def read_embeddings(path: Path):
    with open(path, "rb") as f:
        header_raw = f.read(HEADER_SIZE)
        magic, version, count, dim, model_hash = struct.unpack(HEADER_FMT, header_raw)
        if magic != BIN_MAGIC:
            raise ValueError(f"bad magic 0x{magic:08x}")
        ids = []
        embeds = np.empty((count, dim), dtype=np.float32)
        for i in range(count):
            ids.append(f.read(ID_BYTES).decode("ascii"))
            embeds[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return ids, embeds, {"magic": magic, "version": version, "count": count,
                         "dim": dim, "model_hash": model_hash}
```

- [ ] **Step 10.4: Run tests, expect PASS**

Run: `pytest tests/test_embed_binary.py -v`
Expected: 3 passed.

- [ ] **Step 10.5: Commit**

```bash
git add src/embed_binary.py tests/test_embed_binary.py
git commit -m "feat: card_embeds_v2.bin read/write"
```

---

## Task 11: Evaluation CLI

**Files:**
- Create: `src/evaluate.py`

- [ ] **Step 11.1: Implement `src/evaluate.py`**

```python
"""Evaluate top-1/top-3 on held-out cards w/ augmentation.

Usage:
  python -m src.evaluate --ckpt artifacts/card_encoder.pt
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from src.augment import build_train_transform, build_eval_transform
from src.config import IMAGES_DIR
from src.model import CardEncoder

@torch.no_grad()
def compute_embeddings(model, ids, images_dir, transform, device, augs_per_card=1):
    model.eval()
    embeds = []
    labels = []
    for label, sid in enumerate(tqdm(ids, desc="embed")):
        img = np.array(Image.open(images_dir / f"{sid}.jpg").convert("RGB"))
        for _ in range(augs_per_card):
            t = transform(image=img)["image"]
            e = model(t.unsqueeze(0).to(device)).cpu().numpy()[0]
            embeds.append(e)
            labels.append(label)
    return np.stack(embeds), np.array(labels)

def topk_accuracy(query, query_labels, gallery, gallery_labels, k=1):
    sims = query @ gallery.T  # cosine since normalized
    topk = np.argsort(-sims, axis=1)[:, :k]
    pred_labels = gallery_labels[topk]
    correct = (pred_labels == query_labels[:, None]).any(axis=1)
    return correct.mean()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--augs-per-eval", type=int, default=5)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = CardEncoder().to(device)
    model.load_state_dict(ck["model"])
    eval_ids = ck["eval_ids"]
    print(f"evaluating on {len(eval_ids)} held-out cards")

    gallery, g_labels = compute_embeddings(
        model, eval_ids, IMAGES_DIR, build_eval_transform(), device, augs_per_card=1)
    query, q_labels = compute_embeddings(
        model, eval_ids, IMAGES_DIR, build_train_transform(), device,
        augs_per_card=args.augs_per_eval)

    top1 = topk_accuracy(query, q_labels, gallery, g_labels, k=1)
    top3 = topk_accuracy(query, q_labels, gallery, g_labels, k=3)
    print(f"top-1: {top1*100:.2f}%  top-3: {top3*100:.2f}%")

if __name__ == "__main__":
    main()
```

- [ ] **Step 11.2: Smoke run (after smoke train)**

Run: `python -m src.evaluate --ckpt artifacts/card_encoder.pt --augs-per-eval 2`
Expected: prints top-1 and top-3 percentages (may be low on smoke-trained model).

- [ ] **Step 11.3: Commit**

```bash
git add src/evaluate.py
git commit -m "feat: top-k evaluation on augmented held-out set"
```

---

## Task 12: Export CoreML + TFLite (TDD for determinism)

**Files:**
- Create: `src/export.py`
- Test: `tests/test_export.py`

Note: We bake preprocessing into an export-time wrapper module: input is uint8/255 RGB `[0,1]` (NHWC for CoreML/TFLite semantics). Internally we transpose and apply ImageNet normalization.

- [ ] **Step 12.1: Write failing test**

```python
# tests/test_export.py
import numpy as np
import torch
from src.export import ExportWrapper
from src.model import CardEncoder

def test_export_wrapper_expects_nchw_zero_one_input():
    """The wrapper applies ImageNet normalize internally and expects inputs in [0,1]."""
    m = CardEncoder()
    wrap = ExportWrapper(m)
    x = torch.rand(1, 3, 224, 224)  # NCHW in [0,1]
    y = wrap(x)
    assert y.shape == (1, 256)
    norms = y.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

def test_export_wrapper_matches_model_after_manual_normalize():
    m = CardEncoder()
    m.eval()
    wrap = ExportWrapper(m).eval()
    x = torch.rand(1, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        direct = m((x - mean) / std)
        wrapped = wrap(x)
    torch.testing.assert_close(direct, wrapped, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 12.2: Run tests, expect FAIL**

Run: `pytest tests/test_export.py -v`
Expected: ImportError.

- [ ] **Step 12.3: Implement `src/export.py`**

```python
"""Export CoreML + TFLite artifacts, generate embeddings binary, write manifest.

Usage:
  python -m src.export --ckpt artifacts/card_encoder.pt
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from src.config import (ARTIFACTS_DIR, EMBED_DIM, IMAGENET_MEAN, IMAGENET_STD,
                        IMAGES_DIR, INPUT_SIZE, MANIFEST_VERSION,
                        UNIQUE_ARTWORK_JSON)
from src.augment import build_eval_transform
from src.embed_binary import write_embeddings
from src.model import CardEncoder

class ExportWrapper(nn.Module):
    """Model + baked-in ImageNet normalization.

    Input:  NCHW float32 in [0,1]
    Output: NxD L2-normalized float32
    """
    def __init__(self, model: CardEncoder):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def export_coreml(wrapper: nn.Module, out_path: Path):
    import coremltools as ct
    wrapper.eval()
    example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    traced = torch.jit.trace(wrapper, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=example.shape)],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS15,
    )
    mlmodel.save(str(out_path))

def export_tflite(wrapper: nn.Module, out_path: Path, tmp_dir: Path):
    """PyTorch → ONNX → TF SavedModel (onnx2tf) → TFLite."""
    import onnx
    import onnx2tf
    import tensorflow as tf

    tmp_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = tmp_dir / "encoder.onnx"
    tf_dir = tmp_dir / "tf_saved"
    wrapper.eval()
    example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        wrapper, example, onnx_path,
        input_names=["image"], output_names=["embedding"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes=None,
    )
    onnx.checker.check_model(str(onnx_path))
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(tf_dir),
        non_verbose=True,
    )
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tfl = converter.convert()
    out_path.write_bytes(tfl)

@torch.no_grad()
def compute_all_embeddings(model: CardEncoder, scryfall_ids, device):
    transform = build_eval_transform()
    model.eval()
    embeds = np.empty((len(scryfall_ids), EMBED_DIM), dtype=np.float32)
    for i, sid in enumerate(tqdm(scryfall_ids, desc="embed all")):
        img = np.array(Image.open(IMAGES_DIR / f"{sid}.jpg").convert("RGB"))
        t = transform(image=img)["image"].unsqueeze(0).to(device)
        embeds[i] = model(t).cpu().numpy()[0]
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    embeds = embeds / np.clip(norms, 1e-8, None)
    return embeds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = CardEncoder().to(device)
    model.load_state_dict(ck["model"])
    wrapper = ExportWrapper(model).to(device).eval()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    coreml_path = ARTIFACTS_DIR / "card_encoder.mlmodel"
    tflite_path = ARTIFACTS_DIR / "card_encoder.tflite"
    embeds_path = ARTIFACTS_DIR / "card_embeds_v2.bin"
    manifest_path = ARTIFACTS_DIR / "manifest.json"

    print("exporting CoreML...")
    export_coreml(wrapper.cpu(), coreml_path)
    print("exporting TFLite...")
    export_tflite(wrapper.cpu(), tflite_path, ARTIFACTS_DIR / "_tmp_tflite")

    # size enforcement
    for pth in (coreml_path, tflite_path):
        sz_mb = pth.stat().st_size / (1024 * 1024)
        print(f"{pth.name}: {sz_mb:.2f} MB")
        if sz_mb > 5.0:
            raise SystemExit(f"{pth.name} exceeds 5 MB (got {sz_mb:.2f})")

    # model_hash: first 32 bits of sha256 of the CoreML file
    encoder_sha = sha256_file(coreml_path)
    model_hash = int(encoder_sha[:8], 16)

    # generate embeddings for all downloaded cards
    cards = json.loads(UNIQUE_ARTWORK_JSON.read_text())
    ids = [c["id"] for c in cards if (IMAGES_DIR / f"{c['id']}.jpg").exists()]
    print(f"embedding {len(ids)} cards...")
    model = model.to(device)  # raw encoder (eval transform already normalizes)
    embeds = compute_all_embeddings(model, ids, device)
    write_embeddings(embeds_path, ids, embeds, model_hash=model_hash,
                     version=MANIFEST_VERSION)

    manifest = {
        "version": MANIFEST_VERSION,
        "encoder": {
            "coreml": {"path": coreml_path.name, "sha256": encoder_sha},
            "tflite": {"path": tflite_path.name, "sha256": sha256_file(tflite_path)},
        },
        "embeddings": {
            "path": embeds_path.name,
            "sha256": sha256_file(embeds_path),
            "count": len(ids),
            "dim": EMBED_DIM,
        },
        "model_hash": f"0x{model_hash:08x}",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 12.4: Run wrapper tests, expect PASS**

Run: `pytest tests/test_export.py -v`
Expected: 2 passed.

- [ ] **Step 12.5: Smoke run export on smoke checkpoint**

Run: `python -m src.export --ckpt artifacts/card_encoder.pt`
Expected: 4 artifacts in `artifacts/`, all sizes reported, `card_encoder.mlmodel` and `card_encoder.tflite` both ≤5 MB, `card_embeds_v2.bin` exists, `manifest.json` valid JSON.

- [ ] **Step 12.6: Verify binary matches spec**

```bash
python -c "
from src.embed_binary import read_embeddings
ids, embeds, h = read_embeddings('artifacts/card_embeds_v2.bin')
print('header:', h)
print('N ids:', len(ids), 'shape:', embeds.shape)
assert h['magic'] == 0x4547544D
assert h['version'] == 2
assert h['dim'] == 256
print('OK')
"
```
Expected: `OK`, header values correct.

- [ ] **Step 12.7: Commit**

```bash
git add src/export.py tests/test_export.py
git commit -m "feat: CoreML + TFLite export, embeddings binary, manifest"
```

---

## Task 13: Inference Benchmark

**Files:**
- Create: `src/benchmark.py`

- [ ] **Step 13.1: Implement `src/benchmark.py`**

```python
"""Measure CoreML + TFLite single-image latency.

Usage:
  python -m src.benchmark
"""
import argparse
import time
import numpy as np
from pathlib import Path
from src.config import ARTIFACTS_DIR, INPUT_SIZE

def bench_coreml(path: Path, n: int = 100):
    import coremltools as ct
    model = ct.models.MLModel(str(path))
    x = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    # warm
    for _ in range(5):
        model.predict({"image": x})
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict({"image": x})
    dt = (time.perf_counter() - t0) * 1000 / n
    print(f"CoreML (host CPU, sim proxy): {dt:.2f} ms/image")

def bench_tflite(path: Path, n: int = 100):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = np.random.rand(*inp["shape"]).astype(np.float32)
    for _ in range(5):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        interp.get_tensor(out["index"])
    t0 = time.perf_counter()
    for _ in range(n):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        interp.get_tensor(out["index"])
    dt = (time.perf_counter() - t0) * 1000 / n
    print(f"TFLite (host CPU): {dt:.2f} ms/image")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    bench_coreml(ARTIFACTS_DIR / "card_encoder.mlmodel", args.n)
    bench_tflite(ARTIFACTS_DIR / "card_encoder.tflite", args.n)

if __name__ == "__main__":
    main()
```

- [ ] **Step 13.2: Smoke run**

Run: `python -m src.benchmark --n 20`
Expected: two latency numbers printed (host CPU — a proxy; actual iPhone 12 / Pixel 6 numbers must be measured in the consumer app).

- [ ] **Step 13.3: Commit**

```bash
git add src/benchmark.py
git commit -m "feat: CoreML + TFLite latency benchmark"
```

---

## Task 14: Real-World Test Harness

**Files:**
- Create: `tests/test_real_world_photos.py` (skipped unless fixtures present)
- Create: `tests/fixtures/real_photos/README.md`

- [ ] **Step 14.1: Write harness**

```python
# tests/test_real_world_photos.py
"""Run the trained encoder against phone photos with ground-truth scryfall_ids.

Fixtures expected at tests/fixtures/real_photos/<scryfall_id>.jpg
(filename == ground-truth id). Skip if none present.

Requires:
  artifacts/card_encoder.pt
  artifacts/card_embeds_v2.bin
"""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.augment import build_eval_transform
from src.embed_binary import read_embeddings
from src.model import CardEncoder

FIXTURES = Path(__file__).parent / "fixtures" / "real_photos"

def _photos():
    if not FIXTURES.exists():
        return []
    return sorted(FIXTURES.glob("*.jpg"))

@pytest.mark.skipif(not _photos(), reason="no real-world photo fixtures")
def test_real_world_top1_top3():
    ckpt = Path("artifacts/card_encoder.pt")
    embeds_path = Path("artifacts/card_embeds_v2.bin")
    assert ckpt.exists() and embeds_path.exists()

    ids, gallery, _ = read_embeddings(embeds_path)
    id_to_idx = {sid: i for i, sid in enumerate(ids)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device)
    model = CardEncoder().to(device).eval()
    model.load_state_dict(ck["model"])
    transform = build_eval_transform()

    top1 = top3 = total = 0
    for photo in _photos():
        gt = photo.stem
        if gt not in id_to_idx:
            continue
        img = np.array(Image.open(photo).convert("RGB"))
        t = transform(image=img)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            q = model(t).cpu().numpy()[0]
        sims = gallery @ q
        order = np.argsort(-sims)
        top_ids = [ids[i] for i in order[:3]]
        total += 1
        if top_ids[0] == gt:
            top1 += 1
        if gt in top_ids:
            top3 += 1
    print(f"\nreal-world: top-1={top1}/{total}  top-3={top3}/{total}")
    assert total > 0
    assert top1 / total >= 0.95, f"top-1 {top1/total:.2%} below 95% target"
```

- [ ] **Step 14.2: Write README for fixtures**

```markdown
# Real-world photo fixtures

Place JPEG photos of real MTG cards here, named `<scryfall_id>.jpg`.
Example: `f2b9a1d4-5e7a-4c89-9d6c-abcd12345678.jpg`

Photos should be already-rectified 400×560 RGB (matching the mobile consumer's
output). The encoder itself is size-agnostic — augmentation will resize.

Target: ≥95% top-1 on this set.
```

- [ ] **Step 14.3: Commit**

```bash
git add tests/test_real_world_photos.py tests/fixtures/real_photos/README.md
git commit -m "test: real-world photo harness (skipped w/o fixtures)"
```

---

## Task 15: Full Training + Final Export

This is the long-running wall-clock task — no TDD applies to actual convergence. Execute once the code-level plan above is green.

- [ ] **Step 15.1: Download full dataset**

Run: `python -m src.download_images`
Expected: ~35k JPEGs in `data/images/` (takes hours, ≥8 GB disk).

- [ ] **Step 15.2: Full training**

Run: `python -m src.train`
Expected: 30 epochs, GPU required. Per-epoch checkpoint. Loss trends down. Final `artifacts/card_encoder.pt`.

- [ ] **Step 15.3: Evaluate**

Run: `python -m src.evaluate --ckpt artifacts/card_encoder.pt --augs-per-eval 5`
Expected: top-1 ≥98%. If below target, tune (more epochs, harder miner, higher margin) and re-train.

- [ ] **Step 15.4: Export all artifacts**

Run: `python -m src.export --ckpt artifacts/card_encoder.pt`
Expected: `card_encoder.mlmodel` ≤5 MB, `card_encoder.tflite` ≤5 MB, `card_embeds_v2.bin` populated for all ~35k cards, `manifest.json` written.

- [ ] **Step 15.5: Benchmark**

Run: `python -m src.benchmark`
Expected: host-CPU proxy numbers printed. Real ≤100 ms target verified in consumer app on device.

- [ ] **Step 15.6: Run full test suite**

Run: `pytest -v`
Expected: all unit tests pass; real-world test passes if fixtures supplied.

- [ ] **Step 15.7: Tag release**

```bash
git add artifacts/manifest.json  # commit manifest only; artifacts are .gitignored binaries
git commit -m "release: v2 encoder + embeddings"
git tag v2.0.0
```

---

## Self-Review Checklist (done during planning, not execution)

- [x] Spec coverage: README contract, download, augmentation, model, triplet loss, training, eval, CoreML, TFLite, embeddings binary, manifest, benchmark, real-world harness — all mapped to tasks.
- [x] No placeholders: every code block is full implementation.
- [x] Type consistency: `ExportWrapper`, `CardEncoder`, `CardDataset`, `BatchHardTripletLoss`, `write_embeddings`/`read_embeddings` names used consistently across tasks.
- [x] I/O contract: preprocessing baked in Layer 0 via `ExportWrapper`; 256-d L2-normalized output verified by test; binary format little-endian with exact header/record layout verified by roundtrip test.
- [x] Size constraint enforced in `export.py` Step 12.5.
- [x] Version binding: `MANIFEST_VERSION=2` flows into both `manifest.json` and binary header.
