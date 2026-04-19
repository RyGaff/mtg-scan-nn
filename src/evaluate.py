"""Evaluate top-1/top-3 on held-out cards w/ augmentation.

Usage:
  python -m src.evaluate --ckpt artifacts/card_encoder.pt
"""
import argparse
import hashlib
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from src.augment import build_train_transform, build_eval_transform
from src.config import ARTIFACTS_DIR, IMAGES_DIR
from src.model import CardEncoder


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

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

    from src.device import pick_device
    device = pick_device()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
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

    results_path = ARTIFACTS_DIR / "eval_results.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "ckpt_path": str(Path(args.ckpt).resolve()),
        "ckpt_sha256": _sha256(Path(args.ckpt)),
        "eval_count": len(eval_ids),
        "augs_per_eval": args.augs_per_eval,
        "top1": float(top1),
        "top3": float(top3),
    }, indent=2))
    print(f"wrote {results_path}")

if __name__ == "__main__":
    main()
