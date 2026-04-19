import numpy as np
from PIL import Image
from pathlib import Path
import pytest
from src.dataset import CardDataset


@pytest.fixture
def tiny_image_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    ids = []
    for i in range(4):
        sid = f"00000000-0000-0000-0000-00000000000{i}"
        Image.new("RGB", (488, 680), color=(i*50, 100, 150)).save(d / f"{sid}.jpg")
        ids.append(sid)
    return d, ids


def test_dataset_length(tiny_image_dir):
    d, ids = tiny_image_dir
    ds = CardDataset(d, scryfall_ids=ids, samples_per_card=2, train=True)
    assert len(ds) == 8  # 4 cards * 2 samples


def test_dataset_returns_tensor_and_label(tiny_image_dir):
    import torch
    d, ids = tiny_image_dir
    ds = CardDataset(d, scryfall_ids=ids, samples_per_card=2, train=True)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < 4


def test_dataset_labels_stable_per_card(tiny_image_dir):
    d, ids = tiny_image_dir
    ds = CardDataset(d, scryfall_ids=ids, samples_per_card=3, train=True)
    label_of_id = {}
    for i in range(len(ds)):
        _, lbl = ds[i]
        sid = ds.scryfall_id_for_index(i)
        label_of_id.setdefault(sid, lbl)
        assert label_of_id[sid] == lbl
