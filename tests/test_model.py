import torch
from src.model import CardEncoder
from src.config import EMBED_DIM, INPUT_SIZE

def test_encoder_output_shape():
    m = CardEncoder()
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    y = m(x)
    assert y.shape == (2, EMBED_DIM)

def test_encoder_output_is_l2_normalized():
    m = CardEncoder()
    m.eval()
    x = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        y = m(x)
    norms = y.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_encoder_param_count_under_3m():
    m = CardEncoder()
    n = sum(p.numel() for p in m.parameters())
    # MobileNetV3-Small ~2.5M + 576*256 head
    assert n < 3_000_000, f"got {n}"
