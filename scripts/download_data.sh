#!/usr/bin/env bash
# Download Scryfall unique_artwork + all ~35k card images.
# Rate-limited to ~40 req/s aggregate. Resumable.
set -euo pipefail
python -m src.download_images "$@"
