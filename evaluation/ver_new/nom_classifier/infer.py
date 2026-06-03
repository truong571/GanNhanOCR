"""Load a trained Nôm embedder and embed images — the drop-in for S3.

Replaces DINOv2 in evaluation/ver_new/visual_signal.py: build a NomEncoder once,
embed the crop and each candidate FD glyph, rank by cosine. The trained embedder
IS discriminative on Nôm (see eval_discrim.py), unlike DINOv2.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from model import NomEmbedder            # noqa: E402

MEAN = (0.5, 0.5, 0.5); STD = (0.5, 0.5, 0.5)


class NomEncoder:
    def __init__(self, ckpt: str = str(HERE / "checkpoints" / "best.pt"), device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                   else ("mps" if torch.backends.mps.is_available() else "cpu")))
        ck = torch.load(ckpt, map_location=self.device)
        self.size = ck.get("img", 128)
        self.net = NomEmbedder(ck.get("embed_dim", 256), pretrained=False,
                               arch=ck.get("arch", "resnet18")).to(self.device)
        self.net.load_state_dict(ck["backbone"]); self.net.eval()
        self._cache: dict[str, np.ndarray] = {}

    def _prep(self, gray: np.ndarray) -> torch.Tensor:
        h, w = gray.shape; s = max(h, w)
        canvas = np.full((s, s), 255, np.uint8)
        canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = gray
        g = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = np.repeat(g[None].astype(np.float32) / 255.0, 3, axis=0)
        for c in range(3):
            x[c] = (x[c] - MEAN[c]) / STD[c]
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_gray(self, gray: np.ndarray) -> np.ndarray:
        return self.net(self._prep(gray)).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_path(self, path: str) -> np.ndarray | None:
        if path in self._cache:
            return self._cache[path]
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            return None
        e = self.embed_gray(g); self._cache[path] = e
        return e

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return max(0.0, (float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) + 1) / 2)


if __name__ == "__main__":
    enc = NomEncoder()
    print("NomEncoder loaded on", enc.device, "img", enc.size)
