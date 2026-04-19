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
