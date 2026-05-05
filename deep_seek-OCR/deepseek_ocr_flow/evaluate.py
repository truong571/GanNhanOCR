"""Evaluation helpers for DeepSeek-OCR experiment outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from .data import write_json


def evaluate_rows(rows: list[dict], refs: dict[tuple[str, str], dict]) -> tuple[list[dict], dict]:
    eval_rows: list[dict] = []
    totals = {
        "processed": 0,
        "predicted": 0,
        "with_reference": 0,
        "exact_reference": 0,
        "exact_kimhannom": 0,
        "empty": 0,
    }

    for row in rows:
        totals["processed"] += 1
        key = (row.get("page", ""), row.get("crop_file", ""))
        ref = refs.get(key, {})
        pred = row.get("char") or ""
        kim = row.get("kimhannom_char") or ""
        reference = ref.get("reference_char", "")

        if pred:
            totals["predicted"] += 1
        else:
            totals["empty"] += 1

        exact_ref = bool(reference and pred == reference)
        exact_kim = bool(kim and pred == kim)
        if reference:
            totals["with_reference"] += 1
        if exact_ref:
            totals["exact_reference"] += 1
        if exact_kim:
            totals["exact_kimhannom"] += 1

        eval_rows.append({
            **row,
            "syllable": ref.get("syllable", ""),
            "reference_char": reference,
            "reference_tier": ref.get("tier", ""),
            "reference_matched": ref.get("matched", ""),
            "exact_reference": exact_ref,
            "exact_kimhannom": exact_kim,
        })

    denom_ref = totals["with_reference"] or 1
    denom_processed = totals["processed"] or 1
    summary = {
        **totals,
        "accuracy_reference": totals["exact_reference"] / denom_ref,
        "agreement_kimhannom": totals["exact_kimhannom"] / denom_processed,
        "prediction_rate": totals["predicted"] / denom_processed,
        "reference_source": next(
            (v.get("source_file") for v in refs.values() if v.get("source_file")),
            "",
        ),
    }
    return eval_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation(out_dir: Path, rows: list[dict], refs: dict[tuple[str, str], dict]) -> dict:
    eval_rows, summary = evaluate_rows(rows, refs)
    write_csv(out_dir / "evaluation.csv", eval_rows)
    write_json(out_dir / "evaluation.json", summary)
    return summary
