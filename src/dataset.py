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
