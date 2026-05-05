#!/usr/bin/env python3
"""Run DeepSeek-OCR against GanNhanOCR prepared data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from deepseek_ocr_flow.data import (
    iter_crop_records,
    iter_detection_files,
    load_json,
    load_reference_labels,
    parse_pages,
    write_json,
)
from deepseek_ocr_flow.evaluate import write_csv, write_evaluation
from deepseek_ocr_flow.recognizer import (
    DeepSeekOCRConfig,
    DeepSeekOCRRecognizer,
    stable_key,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def main() -> None:
    args = parse_args()
    set_offline_mode(args.offline)
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    paths = config["paths"]

    book = args.book
    book_dir = resolve_path(paths["data_dir"]) / book
    dataset_book_dir = resolve_path(paths.get("output_dir", "dataset")) / book

    pages = parse_pages(args.pages)
    maybe_prepare(args.prepare, config_path, book, book_dir, pages)

    detection_files = iter_detection_files(book_dir, pages)
    if args.limit_pages:
        detection_files = detection_files[: args.limit_pages]
    if not detection_files:
        raise SystemExit(f"No detection files found for {book} in {book_dir / 'detected'}")

    result_root = resolve_path(args.output_dir) / book
    engine_dir = result_root / "deepseek_ocr"
    engine_dir.mkdir(parents=True, exist_ok=True)

    dry_manifest = {
        "book": book,
        "pages": [p.stem.replace("_detection", "") for p in detection_files],
        "mode": args.mode,
        "max_crops": args.max_crops,
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        dry_manifest["crop_count"] = count_crops(detection_files, book_dir)
        dry_manifest["label_reference"] = bool(
            load_reference_labels(book_dir, dataset_book_dir)
        )
        write_json(engine_dir / "dry_run.json", dry_manifest)
        print(f"Dry-run OK. Manifest: {engine_dir / 'dry_run.json'}")
        return

    recognizer = DeepSeekOCRRecognizer(
        DeepSeekOCRConfig(
            model_name=args.model,
            device=args.device,
            dtype=args.dtype,
            attention=args.attention,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=not args.no_crop_mode,
            test_compress=not args.no_test_compress,
            save_results=args.save_model_results,
            crop_canvas=args.crop_canvas,
            crop_prompt=args.crop_prompt,
            page_prompt=args.page_prompt,
        ),
        work_dir=engine_dir,
    )

    rows: list[dict] = []
    if args.mode in {"crops", "both"}:
        rows = run_crop_flow(
            detection_files=detection_files,
            book=book,
            book_dir=book_dir,
            out_dir=engine_dir,
            recognizer=recognizer,
            max_crops=args.max_crops,
            force=args.force,
        )
        write_csv(engine_dir / "labels_rec.csv", rows)
        refs = load_reference_labels(book_dir, dataset_book_dir)
        if rows:
            summary = write_evaluation(engine_dir, rows, refs)
            print_summary(summary)

    if args.mode in {"pages", "both"}:
        run_page_flow(
            detection_files=detection_files,
            book_dir=book_dir,
            out_dir=result_root / "page_ocr",
            recognizer=recognizer,
            force=args.force,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR full-flow experiment for GanNhanOCR",
    )
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--book", required=True)
    parser.add_argument("--pages", default=None, help="Comma-separated page_0012 or 12 values")
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--max-crops", type=int, default=None)
    parser.add_argument(
        "--prepare",
        choices=["auto", "always", "never"],
        default="auto",
        help="Run pipeline step1+step2 when detection files are missing.",
    )
    parser.add_argument("--mode", choices=["crops", "pages", "both"], default="crops")
    parser.add_argument("--output-dir", default="DeepSeek-OCR/results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use local Hugging Face cache only; do not attempt model download.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing cached result files")

    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention", default="auto")
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-canvas", type=int, default=384)
    parser.add_argument("--no-crop-mode", action="store_true")
    parser.add_argument("--no-test-compress", action="store_true")
    parser.add_argument("--save-model-results", action="store_true")
    parser.add_argument("--crop-prompt", default="<image>\nFree OCR.")
    parser.add_argument("--page-prompt", default="<image>\n<|grounding|>OCR this image.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def set_offline_mode(enabled: bool) -> None:
    if not enabled:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def maybe_prepare(
    mode: str,
    config_path: Path,
    book: str,
    book_dir: Path,
    pages: set[str] | None,
) -> None:
    existing = iter_detection_files(book_dir, pages)
    if mode == "never" or (mode == "auto" and existing):
        return
    if mode == "auto":
        print("No detection files found; running pipeline step1 and step2 first.")
    run_module("pipeline.step1_extract", config_path, book)
    run_module("pipeline.step2_align", config_path, book)


def run_module(module: str, config_path: Path, book: str) -> None:
    cmd = [sys.executable, "-m", module, str(config_path), book]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_crop_flow(
    detection_files: list[Path],
    book: str,
    book_dir: Path,
    out_dir: Path,
    recognizer: DeepSeekOCRRecognizer | None,
    max_crops: int | None,
    force: bool,
) -> list[dict]:
    rows: list[dict] = []
    processed = 0
    crops_base = book_dir / "detected"

    for det_path in detection_files:
        detection = load_json(det_path)
        page_name = det_path.stem.replace("_detection", "")
        cache_suffix = f".limit{max_crops}" if max_crops is not None else ""
        page_out = {
            "page": page_name,
            "engine": "deepseek_ocr",
            "partial": max_crops is not None,
            "columns": [],
        }
        cached_page = out_dir / f"{page_name}{cache_suffix}.json"
        if cached_page.exists() and not force:
            cached = load_json(cached_page)
            rows.extend(flatten_page_rows(cached, book))
            continue

        current_col = None
        col_out = None
        for record in iter_crop_records(detection, crops_base):
            if max_crops is not None and processed >= max_crops:
                break
            if not record.crop_path.exists():
                continue

            if current_col != record.column:
                if col_out is not None:
                    page_out["columns"].append(col_out)
                current_col = record.column
                col_out = {"column": record.column, "chars": []}

            if recognizer is None:
                result = {
                    "char": None,
                    "confidence": 0.0,
                    "raw_text": "",
                    "prepared_image": "",
                }
            else:
                key = stable_key(book, page_name, record.column, record.char_idx, record.crop_file)
                result = recognizer.recognize_crop(record.crop_path, key)

            char_row = {
                "char_idx": record.char_idx,
                "char": result["char"],
                "confidence": float(result["confidence"]),
                "raw_text": result["raw_text"],
                "crop_file": record.crop_file,
                "kimhannom_char": record.kimhannom_char,
                "bbox": record.bbox,
                "prepared_image": result.get("prepared_image", ""),
            }
            col_out["chars"].append(char_row)
            rows.append({
                "source": book,
                "page": page_name,
                "column": record.column,
                "char_idx": record.char_idx,
                "crop_file": record.crop_file,
                "char": result["char"] or "",
                "confidence": f"{float(result['confidence']):.3f}",
                "kimhannom_char": record.kimhannom_char or "",
                "bbox": record.bbox or "",
                "raw_text": result["raw_text"],
            })
            processed += 1

        if col_out is not None:
            page_out["columns"].append(col_out)
        write_json(cached_page, page_out)

        if max_crops is not None and processed >= max_crops:
            break

    return rows


def flatten_page_rows(page: dict, book: str) -> list[dict]:
    rows: list[dict] = []
    page_name = page.get("page", "")
    for col in page.get("columns", []):
        for ch in col.get("chars", []):
            rows.append({
                "source": book,
                "page": page_name,
                "column": col.get("column", ""),
                "char_idx": ch.get("char_idx", ""),
                "crop_file": ch.get("crop_file", ""),
                "char": ch.get("char") or "",
                "confidence": f"{float(ch.get('confidence') or 0):.3f}",
                "kimhannom_char": ch.get("kimhannom_char") or "",
                "bbox": ch.get("bbox") or "",
                "raw_text": ch.get("raw_text") or "",
            })
    return rows


def run_page_flow(
    detection_files: list[Path],
    book_dir: Path,
    out_dir: Path,
    recognizer: DeepSeekOCRRecognizer | None,
    force: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for det_path in detection_files:
        page_name = det_path.stem.replace("_detection", "")
        out_path = out_dir / f"{page_name}.txt"
        if out_path.exists() and not force:
            continue
        page_img = book_dir / "pages" / f"{page_name}.png"
        if not page_img.exists():
            continue
        text = "" if recognizer is None else recognizer.recognize_page(page_img, page_name)
        out_path.write_text(text, encoding="utf-8")


def count_crops(detection_files: list[Path], book_dir: Path) -> int:
    total = 0
    crops_base = book_dir / "detected"
    for det_path in detection_files:
        detection = load_json(det_path)
        total += sum(1 for _ in iter_crop_records(detection, crops_base))
    return total


def print_summary(summary: dict) -> None:
    print("DeepSeek-OCR evaluation")
    print("=" * 32)
    print(f"Processed:          {summary['processed']}")
    print(f"Predicted:          {summary['predicted']}")
    print(f"With reference:     {summary['with_reference']}")
    print(f"Exact reference:    {summary['exact_reference']}")
    print(f"Accuracy reference: {summary['accuracy_reference']:.3f}")
    print(f"Agree Kimhannom:    {summary['exact_kimhannom']}")
    print(f"Agreement rate:     {summary['agreement_kimhannom']:.3f}")
    if summary.get("reference_source"):
        print(f"Reference:          {summary['reference_source']}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"[DeepSeek-OCR] {exc}") from exc
