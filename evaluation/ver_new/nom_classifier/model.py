"""Nôm glyph embedding model: ResNet-18 backbone + ArcFace margin head.

The BACKBONE produces an L2-normalized embedding used at inference for S3
(cosine of crop-embedding vs candidate FD-glyph-embedding). The ArcFace head is
training-only: it pushes same-character embeddings together and different ones
apart with an angular margin, which is what DINOv2 zero-shot fails to do
(REPORT_dinov2_unsuitable.md). Training on BOTH real crops and FD glyphs aligns
the two domains so a woodblock crop and the clean FD glyph of the same char land
close.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


_BACKBONES = {
    "resnet18": (torchvision.models.resnet18, "ResNet18_Weights"),
    "resnet34": (torchvision.models.resnet34, "ResNet34_Weights"),
    "resnet50": (torchvision.models.resnet50, "ResNet50_Weights"),
}


class NomEmbedder(nn.Module):
    """Image -> L2-normalized embedding. `arch` picks the backbone capacity."""

    def __init__(self, embed_dim: int = 256, pretrained: bool = True,
                 arch: str = "resnet18"):
        super().__init__()
        ctor, wname = _BACKBONES[arch]
        weights = getattr(torchvision.models, wname).IMAGENET1K_V1 if pretrained else None
        bb = ctor(weights=weights)
        in_feats = bb.fc.in_features
        bb.fc = nn.Identity()
        self.arch = arch
        self.backbone = bb
        self.proj = nn.Linear(in_feats, embed_dim)

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=1)


class ArcMargin(nn.Module):
    """ArcFace head (training only). cosine logits with additive angular margin."""

    def __init__(self, embed_dim: int, n_classes: int, s: float = 30.0, m: float = 0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, emb, labels=None):
        cos = emb @ F.normalize(self.W, dim=1).t()          # emb is already L2-norm
        if labels is None:
            return cos * self.s
        cos = cos.clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, cos.size(1)).to(cos.dtype)
        return (onehot * target + (1 - onehot) * cos) * self.s
