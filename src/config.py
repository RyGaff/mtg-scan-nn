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
# uint32 magic — packed little-endian so on-disk bytes spell ASCII "MTGE"
# (0x4745544D LE → 0x4D 0x54 0x47 0x45 = 'M' 'T' 'G' 'E')
BIN_MAGIC = 0x4745544D

SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data"
# Scryfall asks for 50-100ms between requests and a descriptive User-Agent.
# https://scryfall.com/docs/api#rate-limits-and-good-citizenship
SCRYFALL_USER_AGENT = "mtg-card-encoder/0.1 (+https://github.com/lotusfield)"
SCRYFALL_REQUEST_DELAY_MS = 100  # per-worker throttle on image fetches
SCRYFALL_IMAGE_WORKERS = 4       # → ~40 req/s aggregate, well under limit
