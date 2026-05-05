"""Data helpers for reading GanNhanOCR prepared outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CropRecord:
    page: str
    column: int
    char_idx: int
    crop_file: str
    cleaned_file: str | None
    crop_path: Path
    bbox: list[int] | None
    kimhannom_char: str | None


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def iter_detection_files(book_dir: Path, pages: set[str] | None = None) -> list[Path]:
    detected_dir = book_dir / "detected"
    files = sorted(detected_dir.glob("page_*_detection.json"))
    if pages:
        files = [
            p for p in files
            if p.stem.replace("_detection", "") in pages
        ]
    return files


def iter_crop_records(detection: dict, crops_base: Path) -> Iterable[CropRecord]:
    page = detection.get("book_page") or detection.get("page") or "unknown"
    for col in detection.get("columns", []):
        column = int(col.get("column", 0))
        for ch in col.get("chars", []):
            crop_file = ch.get("crop_file") or ""
            if not crop_file:
                continue
            yield CropRecord(
                page=page,
                column=column,
                char_idx=int(ch.get("char_idx", 0)),
                crop_file=crop_file,
                cleaned_file=ch.get("cleaned_file"),
                crop_path=crops_base / crop_file,
                bbox=ch.get("bbox"),
                kimhannom_char=ch.get("ocr_char"),
            )


def load_reference_labels(book_dir: Path, dataset_book_dir: Path | None = None) -> dict:
    """Load current labels as comparison reference.

    Prefer `prepared/<book>/labeled/labels.csv`; fall back to
    `dataset/<book>/labels.csv` when needed.
    """
    candidates = [book_dir / "labeled" / "labels.csv"]
    if dataset_book_dir:
        candidates.append(dataset_book_dir / "labels.csv")

    label_path = next((p for p in candidates if p.exists()), None)
    if label_path is None:
        return {}

    refs: dict[tuple[str, str], dict] = {}
    with open(label_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            page = row.get("page") or ""
            crop_file = row.get("crop_file") or ""
            if not page or not crop_file:
                continue
            refs[(page, crop_file)] = {
                "reference_char": row.get("nom_char") or "",
                "syllable": row.get("syllable") or "",
                "tier": row.get("tier") or "",
                "matched": row.get("matched") or "",
                "source_file": str(label_path),
            }
    return refs


def parse_pages(value: str | None) -> set[str] | None:
    if not value:
        return None
    pages: set[str] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            pages.add(f"page_{int(item):04d}")
        else:
            pages.add(item)
    return pages
