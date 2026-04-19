import torch
from src.losses import build_triplet_loss

def test_loss_returns_scalar():
    loss_fn = build_triplet_loss(margin=0.2)
    embeds = torch.randn(8, 256)
    embeds = torch.nn.functional.normalize(embeds, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(embeds, labels)
    assert loss.ndim == 0
    assert loss.item() >= 0

def test_loss_zero_when_perfect_separation():
    loss_fn = build_triplet_loss(margin=0.1)
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    embeds = torch.cat([a, a, b, b], dim=0)
    labels = torch.tensor([0, 0, 1, 1])
    loss = loss_fn(embeds, labels)
    assert loss.item() == 0.0
