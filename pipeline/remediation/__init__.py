"""Giai đoạn 1 — Sửa lỗi đã chứng minh (main-pipeline stage).

Deterministic, post-hoc remediation of the committed dataset for the errors proven in
the 3-round evaluation, applied in the mandatory order (quarantine -> demote ->
dedup-split). Does not re-run the pipeline; every change is auditable and asserted.

    census    report the AE-1 ∪ F1 duplicate-crop census
    apply     write labels_remediated.csv + remediation_report.json

The source-level fixes that PREVENT these errors from recurring on future builds live
next to the code they fix (align_engine._pick_reseg monotone assignment, the lowercase
SYLLABLE gate, OCR retry/backoff, the --strict S3 flag).
"""
from __future__ import annotations

from . import census, remediate

__all__ = ["census", "remediate"]
__version__ = "1.0.0"
