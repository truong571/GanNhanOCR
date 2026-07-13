"""Giai đoạn 2 — Nâng chuẩn SOTA (phiếu bầu độc lập thật).

The vote-fusion framework that combines correlated OCR votes (kim, qwen, nomnaocr) with
truly-independent channels (S3 glyph verifier, QN dictionary) the RIGHT way:

  independence  Kish n_eff — how many *effective* independent votes there really are
  fusion        calibrated stacked logistic combiner fit on the Phase-0 human audit
  gating        asymmetric promote/demote (promotion hard, demotion easy)
  qwen_verifier blind multiple-choice verifier (sycophancy + position-bias guards)

The heavy independent-vote MODELS (kraken CTC forced-align, NomNaOCR-LOBO, MegaHan97K,
Qwen-VL) are data producers that emit a per-crop channel CSV and plug into this
framework — see drivers in README. This package (numpy + scipy + pandas only) is the
stable, fully-tested core.
"""
from __future__ import annotations

from . import channels, fusion, gating, independence, qwen_verifier

__all__ = ["channels", "fusion", "gating", "independence", "qwen_verifier"]
__version__ = "1.0.0"
