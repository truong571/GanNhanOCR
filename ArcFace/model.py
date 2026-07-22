"""Model: NomEmbedder backbone (checkpoint-compatible) + Sub-center ArcFace head.

The BACKBONE is byte-identical in structure to
pipeline/align_engine/nom_classifier/model.py so a checkpoint exported here is a
drop-in for the repo's NomEncoder (infer.py loads ck["backbone"] into this exact
module). The HEAD is upgraded to **Sub-center ArcFace** (Deng et al., ECCV 2020,
"Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces"):
K sub-centers per class let a clean dominant sub-center form while noisy
woodblock variants / mis-cut crops land on the OTHER sub-centers — directly
targeting the noisy-auto-label origin of the current 0.57 error-AUC (roadmap P2).

At export time (export_checkpoint.py) the K sub-centers are collapsed to ONE
vector per class → head["W"] of shape (n_classes, embed_dim), which is what
infer.py multiplies for the Max-Logit / head-logit gate (roadmap P0).
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
    """Image -> L2-normalized embedding. IDENTICAL state_dict layout to the repo
    (backbone.* + proj.*) so the exported `backbone` loads into NomEncoder."""

    def __init__(self, embed_dim: int = 256, pretrained: bool = True, arch: str = "resnet18"):
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


class SubCenterArcMargin(nn.Module):
    """Sub-center ArcFace head (training only).

    W: (n_classes * K, embed_dim). For each sample we take the MAX cosine over the
    K sub-centers of each class (the sample is judged by its nearest sub-center of
    a class), then apply the additive angular margin on the target class only.

    K=1 reduces to vanilla ArcFace. K=3 is the ECCV'20 default for noisy labels.
    """

    def __init__(self, embed_dim: int, n_classes: int, k: int = 3,
                 s: float = 30.0, m: float = 0.30):
        super().__init__()
        self.n_classes, self.k, self.s, self.m = n_classes, k, s, m
        self.W = nn.Parameter(torch.randn(n_classes * k, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def sub_cosines(self, emb):
        """(B, n_classes) max-over-subcenter cosine. emb is already L2-normalized."""
        cos = emb @ F.normalize(self.W, dim=1).t()               # (B, n_cls*K)
        cos = cos.view(-1, self.n_classes, self.k)               # (B, n_cls, K)
        return cos.max(dim=2).values                             # (B, n_cls)

    def forward(self, emb, labels=None):
        cos = self.sub_cosines(emb)
        if labels is None:
            return cos * self.s
        cos = cos.clamp(-1 + 1e-6, 1 - 1e-6)
        target = torch.cos(torch.acos(cos) + self.m)
        onehot = F.one_hot(labels, self.n_classes).to(cos.dtype)
        return (onehot * target + (1 - onehot) * cos) * self.s
