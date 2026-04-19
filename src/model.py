import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.config import EMBED_DIM

class CardEncoder(nn.Module):
    def __init__(self, backbone: str = "mobilenetv3_small_100", embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool="avg"
        )
        # timm's MobileNetV3 keeps its pre-logits conv_head when num_classes=0,
        # so the post-pool feature width (1024) differs from backbone.num_features
        # (576). Probe the real width with a dummy forward rather than trusting
        # an attribute name that varies across timm versions.
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        self.head = nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        embed = self.head(feats)
        return F.normalize(embed, p=2, dim=1)
