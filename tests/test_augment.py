import numpy as np
from PIL import Image
from pathlib import Path
from src.augment import build_train_transform, build_eval_transform
from src.config import INPUT_SIZE

FIXTURE = Path(__file__).parent / "fixtures" / "sample_card.jpg"

def test_train_transform_outputs_correct_shape():
    img = np.array(Image.open(FIXTURE).convert("RGB"))
    t = build_train_transform()
    out = t(image=img)["image"]
    assert out.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert out.dtype == np.float32 or str(out.dtype).startswith("torch.float")

def test_train_transform_is_stochastic():
    img = np.array(Image.open(FIXTURE).convert("RGB"))
    t = build_train_transform()
    a = t(image=img)["image"]
    b = t(image=img)["image"]
    a_np = a.numpy() if hasattr(a, "numpy") else a
    b_np = b.numpy() if hasattr(b, "numpy") else b
    assert not np.array_equal(a_np, b_np)

def test_eval_transform_is_deterministic():
    img = np.array(Image.open(FIXTURE).convert("RGB"))
    t = build_eval_transform()
    a = t(image=img)["image"]
    b = t(image=img)["image"]
    a_np = a.numpy() if hasattr(a, "numpy") else a
    b_np = b.numpy() if hasattr(b, "numpy") else b
    assert np.array_equal(a_np, b_np)
