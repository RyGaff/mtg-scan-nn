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
