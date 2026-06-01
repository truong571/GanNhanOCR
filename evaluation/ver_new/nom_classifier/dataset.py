"""Dataset of Nôm glyph images (real crops + FD glyphs) -> (tensor, class_id).

Augmentation simulates woodblock variation (rotation, scale, erosion/dilation,
ink noise, random threshold) so the embedding is invariant to print style and
keys only on character identity. Applied to BOTH crops and FD glyphs so the two
domains are pulled together.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def _augment(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    # affine: rotate + scale + small translate
    ang = random.uniform(-7, 7)
    sc = random.uniform(0.88, 1.12)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 2] += random.uniform(-0.06, 0.06) * w
    M[1, 2] += random.uniform(-0.06, 0.06) * h
    gray = cv2.warpAffine(gray, M, (w, h), borderValue=255)
    # stroke thickness: erode/dilate
    if random.random() < 0.5:
        k = np.ones((random.choice([2, 3]),) * 2, np.uint8)
        gray = (cv2.erode if random.random() < 0.5 else cv2.dilate)(gray, k)
    # ink noise
    if random.random() < 0.6:
        gray = np.clip(gray.astype(np.float32) +
                       np.random.normal(0, random.uniform(4, 14), gray.shape), 0, 255).astype(np.uint8)
    # random binarize (woodblock is high-contrast)
    if random.random() < 0.3:
        gray = ((gray > random.randint(110, 150)) * 255).astype(np.uint8)
    return gray


class NomDataset(Dataset):
    def __init__(self, index_csv: str, root: str, classes: dict[str, int],
                 split: str, img_size: int = 128, train: bool = True):
        self.root = Path(root)
        self.classes = classes
        self.size = img_size
        self.train = train
        rows = list(csv.DictReader(open(index_csv, encoding="utf-8")))
        self.rows = [r for r in rows if r["split"] == split and r["label"] in classes]

    def __len__(self):
        return len(self.rows)

    def _load(self, path: str) -> np.ndarray:
        g = cv2.imread(str(self.root / path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.full((self.size, self.size), 255, np.uint8)
        return g

    def __getitem__(self, i):
        r = self.rows[i]
        g = self._load(r["path"])
        if self.train:
            g = _augment(g)
        # pad to square on white, resize
        h, w = g.shape
        s = max(h, w)
        canvas = np.full((s, s), 255, np.uint8)
        canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = g
        g = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = np.repeat(g[None].astype(np.float32) / 255.0, 3, axis=0)   # 3xHxW
        for c in range(3):
            x[c] = (x[c] - MEAN[c]) / STD[c]
        return torch.from_numpy(x), self.classes[r["label"]]
