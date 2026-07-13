"""Giai đoạn 0 — Tạo ground truth (main-pipeline stage).

A self-contained, human-in-the-loop ground-truth pipeline for the Hán-Nôm auto-labeling
project. Produces the one number the pipeline cannot produce on its own: a statistically
defensible measurement of GOLD/usable label precision against the original scans.

Flow:
    rank      score every usable crop by error suspicion  -> labels_ranked.csv
    plan      compute the acceptance / CI sample-size plan  (pure statistics)
    sample    draw a stratified (or SRS) audit sample        -> sample.csv
    grid      render a blinded HTML audit tool + manifest     -> audit.html + manifest.jsonl
    (human)   auditor labels each crop; exports              -> verdicts.jsonl
    estimate  precision + Wilson/CP CI + acceptance + PPI     -> report.json

Modules: stats, suspicion, sampling, audit_grid, estimate. Depends only on
numpy + scipy + pandas + PIL (no sklearn / cleanlab).
"""
from __future__ import annotations

from . import audit_grid, estimate, sampling, stats, suspicion

__all__ = ["stats", "suspicion", "sampling", "audit_grid", "estimate"]
__version__ = "1.0.0"
