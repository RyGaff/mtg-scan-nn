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
| 0      | 4    | ASCII magic `"MTGE"` (bytes `0x4D 0x54 0x47 0x45`; equals uint32 LE `0x4547544D`) |
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

## Setup on a fresh GPU machine

Prerequisites: `conda` or `miniconda`, NVIDIA driver + CUDA runtime (for CUDA wheels) OR Apple Silicon (for MPS). Python 3.11 env is created automatically.

```bash
git clone <repo> mtg-card-encoder && cd mtg-card-encoder
bash scripts/setup.sh                 # creates conda env `mtg-encoder`, installs deps
conda activate mtg-encoder

bash scripts/download_data.sh         # ~35k images, ~15 min, resumable
bash scripts/train_full.sh            # 30 epochs — GPU strongly recommended
bash scripts/release.sh               # evaluate + export .mlmodel/.tflite/.bin/manifest
```

Device auto-selects: `cuda` → `mps` → `cpu`. Override via `MTG_DEVICE=cuda:1 bash scripts/train_full.sh`.

Smoke check (runs locally on CPU in ~1 min after dataset download):

```bash
python -m src.download_images --limit 50
python -m src.train --smoke
python -m src.evaluate --ckpt artifacts/card_encoder.pt --augs-per-eval 2
pytest -v
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/setup.sh` | Create conda env, install editable package + deps, verify imports |
| `scripts/download_data.sh` | Fetch Scryfall bulk + images (rate-limited, resumable) |
| `scripts/train_full.sh` | 30-epoch training run, emits per-epoch + final checkpoints |
| `scripts/release.sh` | Evaluate + export CoreML / TFLite / embeddings binary / manifest |

## Targets

- Top-1 ≥98% on held-out augmented cards
- Top-1 ≥95% on real-world phone photo test set
- Encoder file size ≤8 MB each (CoreML `.mlmodel`, TFLite)

## Out of scope

- Mobile app integration (consumer repo)
- Card detection / rectification (already solved)
- OCR

## Scryfall data usage

- Metadata: one-shot download of `unique_artwork` via `https://api.scryfall.com/bulk-data` — never loop queries against `/cards/...` endpoints.
- Images: per-card fetches from `cards.scryfall.io` CDN, rate-limited to ~40 req/s aggregate (4 workers × 100 ms per-worker delay, honors 429 `Retry-After`, descriptive User-Agent). Resumable — re-running skips already-downloaded files.
- Attribution per Scryfall guidelines: card images and data © Wizards of the Coast, served by Scryfall. This project redistributes only the derived 256-d embeddings, not the source images.
