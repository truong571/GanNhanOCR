"""encoders.py — encoder Nôm (ResNet18 + ArcFace, checkpoint có sẵn trong repo) + glyph font/FD.

Bản chép NGUYÊN hành vi gold_img_audit/visual_verifier/score_visual.py (Enc, MultiEnc, Glyphs) mà vòng 4 dùng
(a04_units, a04b_glyphs, a08_crops). Chỉ ĐỌC checkpoint; không huấn luyện, không tải gì.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .params import ENCODERS, FD_DIRS, FONTS, REPO

MEAN = 0.5


def device(req: str | None = None) -> str:
    import torch
    if req and req != "auto":
        return req
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Enc:
    def __init__(self, which, dev, norm):
        import torch
        import torch.nn.functional as F
        sys.path.insert(0, str(REPO / "pipeline" / "align_engine" / "nom_classifier"))
        from model import NomEmbedder   # noqa: E402 (chỉ import)
        self.torch = torch
        ck = torch.load(REPO / ENCODERS[which], map_location="cpu", weights_only=False)
        self.size = ck.get("img", 128)
        self.dev = torch.device(dev)
        self.net = NomEmbedder(ck.get("embed_dim", 256), pretrained=False, arch=ck.get("arch") or "resnet18")
        self.net.load_state_dict(ck["backbone"])
        self.net.eval().to(self.dev)
        W = ck["head"]["W"].float()
        self.Wn = F.normalize(W, dim=1).numpy()
        cl = ck["classes"]
        self.lab2idx = dict(cl) if isinstance(cl, dict) else {c: i for i, c in enumerate(cl)}
        self.norm = norm

    def prep(self, gray):
        if self.norm:   # kéo giãn tương phản theo phân vị
            lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
            if hi - lo > 10:
                gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        h, w = gray.shape
        s = max(h, w)
        can = np.full((s, s), 255, np.uint8)
        can[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = gray
        g = cv2.resize(can, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return (g.astype(np.float32) / 255.0 - MEAN) / 0.5

    def embed(self, grays, bs=256):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(grays), bs):
                x = np.stack([self.prep(g) for g in grays[i:i + bs]])
                x = torch.from_numpy(np.repeat(x[:, None], 3, axis=1)).to(self.dev)
                out.append(self.net(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 256), np.float32)


class MultiEnc:
    """emb = [e1, e2, ...]/sqrt(k) ⇒ tích vô hướng = TRUNG BÌNH cosine các encoder."""

    def __init__(self, names, dev, norm):
        self.encs = [Enc(n, dev, norm) for n in names]
        self.k = len(self.encs)

    def embed(self, grays):
        outs = [e.embed(grays) for e in self.encs]
        return (np.concatenate(outs, 1) / np.sqrt(self.k)).astype(np.float32)


class Glyphs:
    def __init__(self):
        from fontTools.ttLib import TTFont
        from PIL import ImageFont
        self.fonts = []
        for f in FONTS:
            f = REPO / f
            if f.exists():
                self.fonts.append((ImageFont.truetype(str(f), 84), set(TTFont(str(f)).getBestCmap())))
        self.fd = {}
        for d in FD_DIRS:
            d = REPO / d
            if not d.exists():
                continue
            for root, _, fs in os.walk(d):
                for fn in fs:
                    if fn.startswith("U+") and fn.endswith(".png"):
                        p = Path(root) / fn
                        if p.stat().st_size < 1024:      # con trỏ git-LFS chưa tải
                            continue
                        try:
                            ch = chr(int(fn[2:-4], 16))
                        except ValueError:
                            continue
                        self.fd.setdefault(ch, str(p))

    def font(self, ch):
        from PIL import Image, ImageDraw
        for f, cmap in self.fonts:
            if ord(ch) in cmap:
                im = Image.new("L", (112, 112), 255)
                ImageDraw.Draw(im).text((56, 56), ch, font=f, fill=0, anchor="mm")
                return np.array(im)
        return None

    def fdimg(self, ch):
        p = self.fd.get(ch)
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE) if p else None
