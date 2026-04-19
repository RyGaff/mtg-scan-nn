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

if __name__ == "__main__":
    main()
