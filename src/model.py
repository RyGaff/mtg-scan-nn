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
        feat_dim = self.backbone.num_features  # 576 for mobilenetv3_small_100
        self.head = nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        embed = self.head(feats)
        return F.normalize(embed, p=2, dim=1)
