"""S3 — visual glyph-match signal (DINOv2 + FontDiffusion glyph cache).

The third independent signal. For a Nôm crop it ranks candidate characters by
DINOv2 cosine similarity, using the FontDiffusion-generated glyph
(gannhanocr-fd/, ~90k woodblock-style images — closest to the scan) when
available, otherwise a font-rendered glyph. Reuses the project's DINOv2Ranker
(core/ranking/dinov2_ranker.py) so the model + embedding cache are identical to
production step3.

Returns consensus.S3 so it feeds straight into decide_label:
  - cosine        = visual score of `ocr_char` itself (used by GOLD-sanity and
                    by the SILVER cosine gate)
  - top_char      = the visually best candidate (argmax)
  - margin        = ocr_char score − best competing candidate
  - beats_all_s2  = ocr_char out-scores every S2 dictionary candidate

This is the heaviest, least-provisioned signal — see FLOW.md §9. It is OPTIONAL:
without it GOLD (dictionary floor) still stands; with it SILVER (out-of-dict
Ext-B glyphs) becomes recoverable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch.nn.functional as F
from PIL import Image


def _is_cjk(ch: str) -> bool:
    if not ch or len(ch) != 1:
        return False
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2EBEF
            or 0xF900 <= o <= 0xFAFF)

from core.ranking.dinov2_ranker import DINOv2Ranker
from evaluation.ver_new.consensus import S3


class VisualS3:
    def __init__(self, repo: Path, font_path: str, fd_dir: str,
                 cache_dir: str | None = None):
        self.ranker = DINOv2Ranker(font_path=font_path,
                                   embedding_cache_dir=cache_dir)
        self.fd_index = self._build_fd_index(Path(fd_dir))
        self._page_cache: dict[str, Image.Image] = {}
        self.n_fd = 0
        self.n_font = 0

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
        """Reference embedding: FontDiffusion glyph if cached, else font render."""
        p = self.fd_index.get(char)
        if p:
            e = self.ranker._embed_crop(p)
            if e is not None:
                self.n_fd += 1
                return e
        self.n_font += 1
        return self.ranker._embed_char(char)

    def compute(self, page_png: str, bbox, ocr_char: str | None,
                s2_candidates: list[str]) -> S3 | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if x2 - x1 < 8 or y2 - y1 < 8:          # too small to be a real glyph
            return None
        crop = self._page(page_png).crop((x1, y1, x2, y2))
        # Reject blank / solid crops (page-edge whitespace, ink bleed): these
        # give spurious cosine≈1.0 against blank glyph renders -> SILVER garbage.
        ink = (np.asarray(crop.convert("L")) < 128).mean()
        if ink < 0.03 or ink > 0.97:
            return None
        crop_emb = self.ranker._embed(crop)

        # Only real single CJK ideographs are valid candidates (drops "" and junk).
        cands: list[str] = []
        for c in ([ocr_char] if ocr_char else []) + list(s2_candidates):
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if not cands:
            return None

        scored: dict[str, float] = {}
        for c in cands:
            e = self._ref_emb(c)
            if e is None:
                scored[c] = 0.0
                continue
            sim = float(F.cosine_similarity(crop_emb.unsqueeze(0), e.unsqueeze(0)))
            scored[c] = max(0.0, (sim + 1.0) / 2.0)

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        top_char, top = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        return S3(top_char=top_char, cosine=top, margin=top - runner,
                  top_in_dict=top_char in set(s2_candidates))
