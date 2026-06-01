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


class NomEmbedder(nn.Module):
    """Image -> L2-normalized embedding."""

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = torchvision.models.resnet18(weights=weights)
        in_feats = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.proj = nn.Linear(in_feats, embed_dim)

    def forward(self, x):
        f = self.backbone(x)
        e = self.proj(f)
        return F.normalize(e, dim=1)


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
