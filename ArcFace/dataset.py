"""Dataset + splits + samplers.

Fixes the three data-side defects the audit measured on the OLD encoder:
  * P2 leakage — the old split was RANDOM crop-level, so 86% of pages had crops
    in >1 split → val/test were contaminated by same-scan neighbours.
    `assign_splits(mode="page_disjoint")` splits by (book,page); mode="lobo"
    holds out a whole book (leave-one-book-out) for an honest cross-book number.
  * long tail — 522 classes have 1 crop, 1017 have <8. `class_balanced_weights`
    feeds a WeightedRandomSampler so rare glyphs are seen as often as 麻 (1644).
  * confusable pairs (㝵/người-type) — `ConfusionBatchSampler` co-locates a class
    with its SinoNom-similar chars in the same batch so ArcFace's margin is forced
    to separate the exact lookalikes that drive the errors (hard-negative mining).

Image framing is byte-identical to infer.NomEncoder._prep (square white-pad →
resize → /255 → mean/std 0.5, 3ch) so train and inference see the same pixels.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.5, 0.5, 0.5], np.float32)


# --------------------------------------------------------------------------- #
# Splits — page-disjoint (default) or leave-one-book-out
# --------------------------------------------------------------------------- #
def assign_splits(df, mode="page_disjoint", holdout_book="", val_frac=0.1,
                  test_frac=0.1, seed=42):
    """Return a list of 'train'/'val'/'test' aligned to df rows.

    page_disjoint: hash (book,page) → a whole page lands entirely in one split, so
      no crop of a train page can appear in val/test.
    lobo: every crop of `holdout_book` → test; the rest split page-disjoint into
      train/val — the honest "does it generalise to an unseen book" setting.
    FD synthetic glyphs (source=='fd') ALWAYS go to train (they are references,
    never an evaluation target).
    """
    out = []
    for _, r in df.iterrows():
        if str(r.get("source")) == "fd":
            out.append("train"); continue
        book, page = str(r["book"]), str(r["page"])
        if mode == "lobo" and holdout_book and book == holdout_book:
            out.append("test"); continue
        h = int(hashlib.md5(f"{seed}|{book}|{page}".encode()).hexdigest(), 16) % 1000 / 1000.0
        if mode == "lobo":
            out.append("val" if h < val_frac / (1 - test_frac) else "train")
        else:
            out.append("test" if h < test_frac else ("val" if h < test_frac + val_frac else "train"))
    return out


def class_balanced_weights(labels_idx, n_classes):
    """Inverse-frequency weight per sample for WeightedRandomSampler (long tail)."""
    freq = np.bincount(labels_idx, minlength=n_classes).astype(np.float64)
    freq[freq == 0] = 1.0
    w = 1.0 / freq[labels_idx]
    return torch.as_tensor(w, dtype=torch.double)


# --------------------------------------------------------------------------- #
# Confusion-aware batches (hard-negative mining)
# --------------------------------------------------------------------------- #
class ConfusionBatchSampler(Sampler):
    """Build batches that deliberately mix a class with its visual lookalikes.

    similar_map: {class_idx: [similar_class_idx, ...]} built from the SinoNom
    similarity dict (only pairs whose BOTH chars exist in this training set).
    Each batch: pick a seed class, add its present similars, then fill with random
    classes; draw `per_class` crops for every chosen class. Forces the margin to
    separate the exact confusions that produce wrong labels.
    """

    def __init__(self, labels_idx, similar_map, batch_size=128, per_class=4,
                 seed=42, length=None):
        self.by_class = defaultdict(list)
        for i, c in enumerate(labels_idx):
            self.by_class[int(c)].append(i)
        self.classes = [c for c, v in self.by_class.items() if v]
        self.similar_map = similar_map
        self.batch_size = batch_size
        self.per_class = max(1, per_class)
        self.rng = random.Random(seed)
        self.length = length or (len(labels_idx) // batch_size)

    def __len__(self):
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            chosen, seen = [], set()
            while len(chosen) * self.per_class < self.batch_size:
                seed_c = self.rng.choice(self.classes)
                group = [seed_c] + [s for s in self.similar_map.get(seed_c, []) if s in self.by_class]
                for c in group:
                    if c in seen:
                        continue
                    seen.add(c); chosen.append(c)
                    if len(chosen) * self.per_class >= self.batch_size:
                        break
            batch = []
            for c in chosen:
                pool = self.by_class[c]
                batch += (self.rng.choices(pool, k=self.per_class) if len(pool) < self.per_class
                          else self.rng.sample(pool, self.per_class))
            yield batch[:self.batch_size]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class NomCropDataset(Dataset):
    def __init__(self, paths, labels_idx, img=128, train=True, aug=True, weights=None):
        self.paths = list(paths)
        self.y = list(labels_idx)
        self.img = img
        self.train = train and aug
        # per-sample loss weight (tier: GOLD 1.0 > SILVER 0.5 > FD 0.4) — lets the
        # trainer trust clean GOLD more than noisy AI-audited SILVER (de-circular).
        self.w = list(weights) if weights is not None else [1.0] * len(self.paths)

    def __len__(self):
        return len(self.paths)

    def _square(self, g):
        h, w = g.shape
        s = max(h, w)
        canvas = np.full((s, s), 255, np.uint8)
        canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = g
        return cv2.resize(canvas, (self.img, self.img), interpolation=cv2.INTER_AREA)

    def _augment(self, g):
        # geometry only — NO horizontal flip (glyphs are not mirror-invariant)
        a = self.img
        ang = np.random.uniform(-8, 8)
        tx, ty = np.random.uniform(-0.06, 0.06, 2) * a
        sc = np.random.uniform(0.90, 1.10)
        M = cv2.getRotationMatrix2D((a / 2, a / 2), ang, sc)
        M[:, 2] += (tx, ty)
        g = cv2.warpAffine(g, M, (a, a), borderValue=255, flags=cv2.INTER_LINEAR)
        if np.random.rand() < 0.3:                       # ink thickness jitter
            k = np.ones((2, 2), np.uint8)
            g = cv2.erode(g, k) if np.random.rand() < 0.5 else cv2.dilate(g, k)
        if np.random.rand() < 0.25:                      # random-erasing (occlusion)
            eh, ew = np.random.randint(a // 8, a // 3, 2)
            y0, x0 = np.random.randint(0, a - eh), np.random.randint(0, a - ew)
            g[y0:y0 + eh, x0:x0 + ew] = 255
        return g

    def __getitem__(self, i):
        g = cv2.imread(str(self.paths[i]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.full((self.img, self.img), 255, np.uint8)
        else:
            g = self._square(g)
        if self.train:
            g = self._augment(g)
        x = np.repeat((g[None].astype(np.float32) / 255.0), 3, axis=0)
        x = (x - MEAN[:, None, None]) / STD[:, None, None]
        return torch.from_numpy(x), int(self.y[i]), float(self.w[i])
