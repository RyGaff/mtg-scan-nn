import numpy as np
import torch
from src.export import ExportWrapper
from src.model import CardEncoder

def test_export_wrapper_expects_nchw_zero_one_input():
    """Wrapper applies ImageNet normalize internally; expects inputs in [0,1]."""
    m = CardEncoder()
    wrap = ExportWrapper(m)
    x = torch.rand(1, 3, 224, 224)  # NCHW in [0,1]
    y = wrap(x)
    assert y.shape == (1, 256)
    norms = y.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

def test_export_wrapper_matches_model_after_manual_normalize():
    m = CardEncoder()
    m.eval()
    wrap = ExportWrapper(m).eval()
    x = torch.rand(1, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        direct = m((x - mean) / std)
        wrapped = wrap(x)
    torch.testing.assert_close(direct, wrapped, atol=1e-5, rtol=1e-5)
