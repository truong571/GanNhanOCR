"""Tập công bố theo GOLD chính xác (`python -m pipeline.publish gold-exact`, run_pipeline.sh --publish).

Tách bộ gộp `dataset/_ALL/` thành tập ẢNH (ô gold_exact = ok, ảnh = crop chuẩn) và tập VĂN BẢN/NHÃN; KHÔNG chia tập
(quyết định A-10). Công cụ công bố cũ tháng 7 (split/metadata/datasheet/export/validate) đã xoá 30/09.
"""
from __future__ import annotations

__all__ = ["gold_exact_release"]
__version__ = "2.0.0"
