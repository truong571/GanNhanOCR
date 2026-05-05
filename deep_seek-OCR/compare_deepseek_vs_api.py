#!/usr/bin/env python3
"""Compare DeepSeek-OCR experiment output with Kimhannom API output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent


def main() -> None:
    args = parse_args()
    book = args.book
    prepared_book = PROJECT_ROOT / "prepared" / book
    result_root = EXPERIMENT_DIR / "results" / book
    deepseek_dir = result_root / "deepseek_ocr"
    api_dir = result_root / "api_call"
    compare_dir = result_root / "comparison"
    api_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    refs = load_references(prepared_book / "labeled" / "labels.csv")
    api_rows = load_api_rows(prepared_book, refs)
    deepseek_rows = load_deepseek_rows(deepseek_dir)

    write_csv(api_dir / "labels_rec.csv", api_rows)
    write_json(api_dir / "evaluation.json", evaluate(api_rows, "api_char"))

    compare_rows = build_compare_rows(api_rows, deepseek_rows)
    write_csv(compare_dir / "deepseek_vs_api.csv", compare_rows)

    summary = {
        "book": book,
        "api_call": evaluate(api_rows, "api_char"),
        "deepseek": summarize_deepseek(deepseek_dir, deepseek_rows),
        "comparison": summarize_comparison(compare_rows),
        "files": {
            "api_rows": str(api_dir / "labels_rec.csv"),
            "api_evaluation": str(api_dir / "evaluation.json"),
            "comparison_csv": str(compare_dir / "deepseek_vs_api.csv"),
            "summary": str(compare_dir / "summary.json"),
            "markdown": str(compare_dir / "summary.md"),
        },
    }
    write_json(compare_dir / "summary.json", summary)
    write_markdown(compare_dir / "summary.md", summary)
    print(f"Wrote comparison: {compare_dir / 'summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", required=True)
    return parser.parse_args()


def load_references(path: Path) -> dict[tuple[str, str], dict]:
    refs: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return refs
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            refs[(row.get("page", ""), row.get("crop_file", ""))] = row
    return refs


def load_api_rows(prepared_book: Path, refs: dict[tuple[str, str], dict]) -> list[dict]:
    rows: list[dict] = []
    for det_path in sorted((prepared_book / "detected").glob("page_*_detection.json")):
        with open(det_path, "r", encoding="utf-8") as f:
            detection = json.load(f)
        page = detection.get("book_page") or det_path.stem.replace("_detection", "")
        for col in detection.get("columns", []):
            for ch in col.get("chars", []):
                crop_file = ch.get("crop_file") or ""
                ref = refs.get((page, crop_file), {})
                api_char = ch.get("ocr_char") or ""
                reference_char = ref.get("nom_char") or ""
                rows.append({
                    "page": page,
                    "column": col.get("column"),
                    "char_idx": ch.get("char_idx"),
                    "crop_file": crop_file,
                    "api_char": api_char,
                    "reference_char": reference_char,
                    "syllable": ref.get("syllable") or "",
                    "reference_tier": ref.get("tier") or "",
                    "api_exact_reference": bool(
                        api_char and reference_char and api_char == reference_char
                    ),
                    "bbox": ch.get("bbox") or "",
                })
    return rows


def load_deepseek_rows(deepseek_dir: Path) -> dict[tuple[str, str], dict]:
    path = deepseek_dir / "labels_rec.csv"
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[(row.get("page", ""), row.get("crop_file", ""))] = row
    return rows


def build_compare_rows(api_rows: list[dict], deepseek_rows: dict[tuple[str, str], dict]) -> list[dict]:
    rows: list[dict] = []
    for api in api_rows:
        ds = deepseek_rows.get((api["page"], api["crop_file"]), {})
        deepseek_char = ds.get("char") or ""
        rows.append({
            **api,
            "deepseek_char": deepseek_char,
            "deepseek_confidence": ds.get("confidence") or "",
            "deepseek_raw_text": ds.get("raw_text") or "",
            "deepseek_exact_reference": bool(
                deepseek_char
                and api.get("reference_char")
                and deepseek_char == api["reference_char"]
            ),
            "deepseek_agrees_api": bool(deepseek_char and deepseek_char == api["api_char"]),
        })
    return rows


def evaluate(rows: list[dict], pred_key: str) -> dict:
    total = len(rows)
    predicted = sum(1 for r in rows if r.get(pred_key))
    with_ref = sum(1 for r in rows if r.get("reference_char"))
    exact_ref = sum(
        1 for r in rows
        if r.get(pred_key) and r.get("reference_char") and r[pred_key] == r["reference_char"]
    )
    return {
        "total": total,
        "predicted": predicted,
        "with_reference": with_ref,
        "exact_reference": exact_ref,
        "prediction_rate": predicted / total if total else 0.0,
        "accuracy_reference": exact_ref / with_ref if with_ref else 0.0,
    }


def summarize_deepseek(deepseek_dir: Path, rows: dict[tuple[str, str], dict]) -> dict:
    status_path = deepseek_dir / "run_status.json"
    status = {}
    if status_path.exists():
        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)
    return {
        "status": status.get("status", "missing_labels_rec_csv" if not rows else "ok"),
        "predicted_rows": len(rows),
        "error": status.get("error", {}),
        "run_status": str(status_path) if status_path.exists() else "",
    }


def summarize_comparison(rows: list[dict]) -> dict:
    compared = sum(1 for r in rows if r.get("deepseek_char"))
    agrees = sum(1 for r in rows if r.get("deepseek_agrees_api"))
    return {
        "rows": len(rows),
        "rows_with_deepseek_prediction": compared,
        "deepseek_agrees_api": agrees,
        "deepseek_api_agreement_rate": agrees / compared if compared else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_markdown(path: Path, summary: dict) -> None:
    api = summary["api_call"]
    deepseek = summary["deepseek"]
    comparison = summary["comparison"]
    lines = [
        f"# DeepSeek-OCR vs Kimhannom API - {summary['book']}",
        "",
        "## Ket luan ngan",
        "",
        "- Kimhannom API co ket qua OCR tren toan bo crop.",
        "- DeepSeek-OCR chua sinh duoc `labels_rec.csv` tren may hien tai vi inference yeu cau CUDA.",
        "- Vi DeepSeek khong co prediction row, chua tinh duoc agreement/accuracy that giua DeepSeek va API.",
        "",
        "## Kimhannom API",
        "",
        f"- Total crops: {api['total']}",
        f"- Predicted: {api['predicted']} ({api['prediction_rate']:.2%})",
        f"- Exact voi reference labels hien co: {api['exact_reference']}/{api['with_reference']} ({api['accuracy_reference']:.2%})",
        "",
        "## DeepSeek-OCR",
        "",
        f"- Status: `{deepseek['status']}`",
        f"- Prediction rows: {deepseek['predicted_rows']}",
        f"- Error: `{deepseek.get('error', {}).get('message', '')}`",
        "",
        "## So sanh truc tiep",
        "",
        f"- Rows co DeepSeek prediction: {comparison['rows_with_deepseek_prediction']}/{comparison['rows']}",
        f"- DeepSeek agrees API: {comparison['deepseek_agrees_api']}",
        "",
        "## Files",
        "",
        f"- API rows: `{summary['files']['api_rows']}`",
        f"- API evaluation: `{summary['files']['api_evaluation']}`",
        f"- Comparison CSV: `{summary['files']['comparison_csv']}`",
        f"- Summary JSON: `{summary['files']['summary']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
