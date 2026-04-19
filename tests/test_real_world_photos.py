"""Run the trained encoder against phone photos with ground-truth scryfall_ids.

Fixtures expected at tests/fixtures/real_photos/<scryfall_id>.jpg
(filename == ground-truth id). Skip if none present.

Requires:
  artifacts/card_encoder.pt
  artifacts/card_embeds_v2.bin
"""
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
    ck = torch.load(ckpt, map_location=device, weights_only=False)
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
