"""Kimhannom adapter.

Kimhannom already runs during Step 1 and writes its recognition into
`detection.json -> columns -> chars -> ocr_char`. This engine does not
call the API again at crop-time; it just reads that field so the
consensus layer can treat it uniformly alongside the other engines.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocr_engines.base import OCREngine, RecognitionResult


class KimhannomEngine(OCREngine):
    name = "kimhannom"

    def recognize_crop(
        self,
        crop_path: str,
        context: dict | None = None,
    ) -> RecognitionResult:
        # Not used directly — Kimhannom operates at page level upstream.
        # Kept for interface compatibility.
        if context and context.get("ocr_char"):
            return RecognitionResult(char=context["ocr_char"], confidence=1.0)
        return RecognitionResult(char=None, confidence=0.0)

    def recognize_page(
        self,
        detection: dict,
        crops_base: Path,
        cache_path: Path | None = None,
        verbose: bool = False,
    ) -> dict:
        if cache_path and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        page_name = detection.get("book_page") or detection.get("page") or "unknown"
        out = {"page": page_name, "engine": self.name, "columns": []}

        for col in detection.get("columns", []):
            col_out = {"column": col["column"], "chars": []}
            for ch in col.get("chars", []):
                ocr_char = ch.get("ocr_char")
                col_out["chars"].append({
                    "char_idx": ch["char_idx"],
                    "char": ocr_char if ocr_char else None,
                    "confidence": 1.0 if ocr_char else 0.0,
                })
            out["columns"].append(col_out)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        if verbose:
            n = sum(len(c["chars"]) for c in out["columns"])
            filled = sum(
                1 for c in out["columns"]
                for ch in c["chars"] if ch["char"]
            )
            print(f"    [kimhannom] {filled}/{n} chars from Step 1 cache")

        return out
