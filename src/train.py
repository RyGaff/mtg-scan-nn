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
from src.device import pick_device
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
    device = pick_device()
    print(f"device: {device}")

    limit = 50 if smoke else None
    epochs = 1 if smoke else EPOCHS
    batch = 16 if smoke else BATCH_SIZE

    all_ids = load_scryfall_ids(limit=limit)
    train_ids, eval_ids = split_ids(all_ids)
    print(f"train: {len(train_ids)} cards, eval: {len(eval_ids)} cards")

    train_ds = CardDataset(IMAGES_DIR, train_ids, samples_per_card=4, train=True)

    # On macOS, multiprocessing fork can cause issues; use num_workers=0 for
    # smoke runs. pin_memory is only useful with CUDA.
    num_workers = 0 if smoke else 4
    pin_memory = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=True)

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
