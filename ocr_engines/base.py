"""Shared interface for all OCR engines.

Each engine recognizes a single character crop and returns
(char, confidence). Engines may return None when they cannot recognize.

Output file format (per page, per engine):
    prepared/<book>/ocr_results/<engine_name>/page_XXXX.json
    {
        "page": "page_0012",
        "engine": "paddleocrv5_nom",
        "columns": [
            {"column": 1, "chars": [
                {"char_idx": 0, "char": "經", "confidence": 0.95}
            ]}
        ]
    }
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RecognitionResult:
    char: str | None
    confidence: float  # 0.0 - 1.0; use 1.0 when the engine has no score


class OCREngine(ABC):
    name: str

    @abstractmethod
    def recognize_crop(
        self,
        crop_path: str,
        context: dict | None = None,
    ) -> RecognitionResult:
        """Recognize one character crop.

        Args:
            crop_path: Absolute path to a character crop PNG.
            context: Optional extras, e.g. {"column_image": <path>,
                "column_chars_expected": 6}. Engines may ignore.
        """
        raise NotImplementedError

    def recognize_page(
        self,
        detection: dict,
        crops_base: Path,
        cache_path: Path | None = None,
        verbose: bool = False,
    ) -> dict:
        """Run recognition on every char in a page detection.json.

        Reads the `columns -> chars -> crop_file` structure written by
        `pipeline/step2_align.py` and produces a parallel structure with
        per-engine recognition output. Caches to `cache_path` if given.
        """
        if cache_path and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        page_name = detection.get("book_page") or detection.get("page") or "unknown"
        out = {
            "page": page_name,
            "engine": self.name,
            "columns": [],
        }

        for col in detection.get("columns", []):
            col_out = {"column": col["column"], "chars": []}
            for ch in col.get("chars", []):
                crop_file = ch.get("crop_file")
                if not crop_file:
                    col_out["chars"].append({
                        "char_idx": ch["char_idx"],
                        "char": None,
                        "confidence": 0.0,
                    })
                    continue
                crop_path = crops_base / crop_file
                if not crop_path.exists():
                    col_out["chars"].append({
                        "char_idx": ch["char_idx"],
                        "char": None,
                        "confidence": 0.0,
                    })
                    continue

                res = self.recognize_crop(str(crop_path))
                col_out["chars"].append({
                    "char_idx": ch["char_idx"],
                    "char": res.char,
                    "confidence": float(res.confidence),
                })

                if verbose:
                    marker = res.char if res.char else "∅"
                    print(f"    [{self.name}] col{col['column']:02d}"
                          f"_char{ch['char_idx']:03d} → {marker} "
                          f"({res.confidence:.2f})")

            out["columns"].append(col_out)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        return out
