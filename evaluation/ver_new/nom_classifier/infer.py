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
                               arch=ck.get("arch") or "resnet18").to(self.device)
        self.net.load_state_dict(ck["backbone"]); self.net.eval()
        self._cache: dict[str, np.ndarray] = {}
        # ---- optional ArcFace head -> Max-Logit open-set score (roadmap #7) ----
        # The trainer (kaggle_train.save_ck) DOES save the head + class map; infer
        # historically ignored them. We load them when present so a crop can be
        # scored against ALL trained classes (Max-Logit-Score, Vaze et al. ICLR'22)
        # — a candidate-INDEPENDENT "is this a known glyph at all?" gate that catches
        # miscut/garbage crops before they become a wrong SILVER label. Absent head
        # -> mls() returns None and callers fall back to the cosine/kNN gate.
        self._Wn = None
        self.classes = None         # idx -> label
        head = ck.get("head")
        cl = ck.get("classes")
        if head is not None and "W" in head:
            import torch.nn.functional as _F
            W = head["W"].to(self.device).float()          # (n_cls, embed)
            self._Wn = _F.normalize(W, dim=1)
            if isinstance(cl, dict):                        # label -> idx
                self.classes = {i: lab for lab, i in cl.items()}
            elif isinstance(cl, (list, tuple)):
                self.classes = {i: lab for i, lab in enumerate(cl)}
        self.has_head = self._Wn is not None

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

    @torch.no_grad()
    def logits(self, emb: np.ndarray) -> np.ndarray | None:
        """Cosine logits of an embedding against EVERY trained class (needs head)."""
        if self._Wn is None:
            return None
        e = torch.from_numpy(np.asarray(emb, np.float32)).to(self.device)
        e = e / (e.norm() + 1e-9)
        return (self._Wn @ e).cpu().numpy()                 # (n_cls,) cosines in [-1,1]

    def mls(self, emb: np.ndarray) -> float | None:
        """Max-Logit-Score: max cosine to any trained class — an open-set / OOD
        confidence for the crop (candidate-independent). Higher = more glyph-like.
        None if the checkpoint has no head. (Vaze et al., ICLR 2022.)"""
        lg = self.logits(emb)
        return float(lg.max()) if lg is not None else None

    def predict_topk(self, emb: np.ndarray, k: int = 5):
        """[(label, cosine)] over ALL trained classes, for diagnostics. [] if no head."""
        lg = self.logits(emb)
        if lg is None or self.classes is None:
            return []
        idx = np.argsort(lg)[::-1][:k]
        return [(self.classes.get(int(i), str(int(i))), float(lg[i])) for i in idx]

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return max(0.0, (float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) + 1) / 2)

    @staticmethod
    def cosine_raw(a: np.ndarray, b: np.ndarray) -> float:
        """True cosine in [-1, 1] (no (cos+1)/2 remap) — for calibration and for
        reporting interpretable same/diff numbers (Bước 2)."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


if __name__ == "__main__":
    enc = NomEncoder()
    print("NomEncoder loaded on", enc.device, "img", enc.size)
