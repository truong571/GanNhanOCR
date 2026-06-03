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


def _elastic(gray, alpha=9.0, sigma=4.0):
    h, w = gray.shape
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(gray, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=255)


def _perspective(gray, mag=0.08):
    h, w = gray.shape
    d = mag * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.random.uniform(-d, d, src.shape).astype(np.float32)
    return cv2.warpPerspective(gray, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=255)


def _augment(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    # affine: rotate + scale + translate
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-9, 9), random.uniform(0.85, 1.15))
    M[0, 2] += random.uniform(-0.07, 0.07) * w
    M[1, 2] += random.uniform(-0.07, 0.07) * h
    gray = cv2.warpAffine(gray, M, (w, h), borderValue=255)
    if random.random() < 0.45:                       # warp cong (mộc bản)
        gray = _elastic(gray)
    if random.random() < 0.30:                       # nghiêng trang
        gray = _perspective(gray)
    if random.random() < 0.55:                       # nét đậm/mảnh
        k = np.ones((random.choice([2, 3]),) * 2, np.uint8)
        gray = (cv2.erode if random.random() < 0.5 else cv2.dilate)(gray, k)
    if random.random() < 0.6:                        # nhiễu mực
        gray = np.clip(gray.astype(np.float32) +
                       np.random.normal(0, random.uniform(5, 16), gray.shape), 0, 255).astype(np.uint8)
    if random.random() < 0.3:                         # nhị phân ngẫu nhiên
        gray = ((gray > random.randint(105, 155)) * 255).astype(np.uint8)
    if random.random() < 0.25:                        # cutout (đứt nét/che)
        s = int(min(h, w) * random.uniform(0.12, 0.30))
        cy, cx = random.randint(0, h), random.randint(0, w)
        gray[max(0, cy - s // 2):cy + s // 2, max(0, cx - s // 2):cx + s // 2] = 255
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
