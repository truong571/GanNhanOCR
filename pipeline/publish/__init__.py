"""Giai đoạn 3 — Công bố đạt chuẩn quốc tế (main-pipeline stage).

Turns the remediated dataset into an internationally-publishable release:
  splits     page-disjoint + leave-one-book-out, with cross-split leakage audit
  metadata   Frictionless Data Package + Croissant JSON-LD (real sha256, valid PK)
  datasheet  Gebru + JOHD cultural-heritage datasheet
  export     HuggingFace Parquet with typed Features (Image + ClassLabel)
  validate   CI gate — the invariants the standards require, fails loud

Heavy external baselines (kNN/ResNet/ViT on the official splits) are documented drivers,
not run here. numpy + pandas + PIL only; `datasets` used for the parquet export.
"""
from __future__ import annotations

from . import datasheet, export, hashing, metadata, splits, validate

__all__ = ["splits", "metadata", "datasheet", "export", "validate", "hashing"]
__version__ = "1.0.0"
