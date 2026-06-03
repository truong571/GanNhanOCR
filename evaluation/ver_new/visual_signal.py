"""S3 — visual glyph-match signal (TRAINED Nôm embedder + FontDiffusion glyphs).

The third independent signal. For a Nôm crop it ranks candidate characters by
cosine of a NÔM-TRAINED embedding (evaluation/ver_new/nom_classifier, ResNet-18
+ ArcFace) against the FontDiffusion reference glyph of each candidate. This
REPLACES DINOv2, which was proven non-discriminative on chữ-Nôm
(REPORT_dinov2_unsuitable.md: cosine 0.91 between different chars, retrieval 0%).
The trained encoder: T2 separation +0.29, T3 retrieval 76.5% (DINOv2: +0.01, 0%).

Returns consensus.S3 (top_char / cosine / margin / top_in_dict) -> decide_label
populates the SILVER tier. Checkpoint auto-found at nom-embed/best.pt (train via
nom_classifier/ on Kaggle, download best.pt there).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.ver_new.consensus import S3
from evaluation.ver_new.nom_classifier.infer import NomEncoder


def _is_cjk(ch: str) -> bool:
    if not ch or len(ch) != 1:
        return False
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2EBEF
            or 0xF900 <= o <= 0xFAFF)


def _find_ckpt(repo: Path) -> str:
    for c in [repo / "nom-embed" / "best.pt",
              repo / "evaluation" / "ver_new" / "nom_classifier" / "checkpoints" / "best.pt",
              repo / "nom-embed" / "last.pt"]:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "Nôm embedder checkpoint not found (nom-embed/best.pt). Train it first via "
        "evaluation/ver_new/nom_classifier (Kaggle) and place best.pt at nom-embed/.")


class VisualS3:
    def __init__(self, repo: Path, font_path: str | None = None, fd_dir: str = "",
                 cache_dir: str | None = None, ckpt: str | None = None):
        self.enc = NomEncoder(ckpt or _find_ckpt(Path(repo)))
        self.fd_index = self._build_fd_index(Path(fd_dir))
        self._page_cache: dict[str, Image.Image] = {}
        self._ref_cache: dict[str, np.ndarray] = {}
        self.n_fd = 0
        self.n_font = 0   # no font fallback with the trained encoder
        print(f"  S3 = trained Nôm embedder on {self.enc.device} | FD glyphs {len(self.fd_index)}",
              flush=True)

    @staticmethod
    def _build_fd_index(fd_dir: Path) -> dict[str, str]:
        idx: dict[str, str] = {}
        if not fd_dir.exists():
            return idx
        for png in fd_dir.rglob("U+*.png"):
            try:
                idx[chr(int(png.stem.replace("U+", ""), 16))] = str(png)
            except ValueError:
                pass
        return idx

    def _page(self, page_png: str) -> Image.Image:
        img = self._page_cache.get(page_png)
        if img is None:
            img = Image.open(page_png).convert("RGB")
            self._page_cache[page_png] = img
        return img

    def _ref_emb(self, char: str):
        """Reference embedding = trained-encoder embedding of the FD glyph."""
        if char in self._ref_cache:
            return self._ref_cache[char]
        p = self.fd_index.get(char)
        if not p:
            return None
        e = self.enc.embed_path(p)
        if e is not None:
            self.n_fd += 1
            self._ref_cache[char] = e
        return e

    def compute(self, page_png: str, bbox, ocr_char: str | None,
                s2_candidates: list[str]) -> S3 | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = self._page(page_png).crop((x1, y1, x2, y2))
        gray = np.asarray(crop.convert("L"))
        ink = (gray < 128).mean()                # reject blank / solid crops
        if ink < 0.03 or ink > 0.97:
            return None
        crop_emb = self.enc.embed_gray(gray)

        cands: list[str] = []
        for c in ([ocr_char] if ocr_char else []) + list(s2_candidates):
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if not cands:
            return None

        scored: dict[str, float] = {}
        for c in cands:
            e = self._ref_emb(c)
            scored[c] = self.enc.cosine(crop_emb, e) if e is not None else 0.0

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        top_char, top = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        return S3(top_char=top_char, cosine=top, margin=top - runner,
                  top_in_dict=top_char in set(s2_candidates))
