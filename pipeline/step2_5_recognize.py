"""Step 2.5: Cross-recognize crops with multiple OCR engines, then vote.

Runs after `step2_align` has produced `detected/page_XXXX_detection.json`
and the corresponding crops. Writes:

    prepared/<book>/ocr_results/<engine>/page_XXXX.json
    prepared/<book>/ocr_results/consensus/page_XXXX.json
    dataset/<book>/<engine>/labels_rec.csv          (per-engine audit)

The consensus JSON is consumed by step3 to add a strong "tier 0"
agreement signal before dictionary lookup.

Config (pipeline.yaml -> step2_5):
    engines:                 # list of engine names to run
        - kimhannom
        - paddleocrv5_nom
        - nomna_ocr
    weights:                 # optional override of consensus.DEFAULT_WEIGHTS
        kimhannom: 1.2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ocr_engines.base import OCREngine
from ocr_engines.consensus import (
    DEFAULT_WEIGHTS, merge_page, summarize, write_consensus,
)
from ocr_engines.kimhannom import KimhannomEngine
from ocr_engines.paddleocrv5_nom import PaddleOCRv5NomEngine
from ocr_engines.nomna_ocr import NomNaOCREngine

from pipeline.step0_setup import load_config


ENGINE_REGISTRY: dict[str, type[OCREngine]] = {
    "kimhannom": KimhannomEngine,
    "paddleocrv5_nom": PaddleOCRv5NomEngine,
    "nomna_ocr": NomNaOCREngine,
}


def build_engines(names: list[str]) -> list[OCREngine]:
    engines: list[OCREngine] = []
    for n in names:
        cls = ENGINE_REGISTRY.get(n)
        if cls is None:
            print(f"[step2_5] Unknown engine: {n}", file=sys.stderr)
            continue
        try:
            engines.append(cls())
        except Exception as e:
            print(f"[step2_5] Failed to init {n}: {e}", file=sys.stderr)
    return engines


def process_book(
    config: dict,
    book_name: str,
    engines: list[OCREngine],
    weights: dict | None,
    verbose: bool = True,
) -> None:
    paths = config["paths"]
    data_dir = Path(paths["data_dir"]) / book_name
    detected_dir = data_dir / "detected"
    crops_base = detected_dir  # crop_file is relative to this
    results_dir = data_dir / "ocr_results"
    consensus_dir = results_dir / "consensus"

    detection_files = sorted(detected_dir.glob("page_*_detection.json"))
    if not detection_files:
        print(f"[step2_5] No detection files in {detected_dir}", file=sys.stderr)
        return

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step 2.5: Recognize + Consensus — {book_name}")
        print(f"  Engines: {[e.name for e in engines]}")
        print(f"  Pages:   {len(detection_files)}")
        print(f"{'='*60}")

    audit_rows: dict[str, list[dict]] = {e.name: [] for e in engines}
    audit_rows["consensus"] = []
    broken: set[str] = set()

    for det_path in detection_files:
        with open(det_path, "r", encoding="utf-8") as f:
            detection = json.load(f)
        page_name = det_path.stem.replace("_detection", "")
        detection["book_page"] = page_name

        # Per-engine recognition (cached). If an engine fails (e.g. missing
        # deps or weights), log once and skip it for remaining pages so
        # consensus still produces useful output from the rest.
        engine_pages: dict = {}
        for engine in engines:
            if engine.name in broken:
                continue
            cache_path = results_dir / engine.name / f"{page_name}.json"
            try:
                page_result = engine.recognize_page(
                    detection=detection,
                    crops_base=crops_base,
                    cache_path=cache_path,
                    verbose=verbose,
                )
            except Exception as e:
                print(
                    f"[step2_5] Engine '{engine.name}' failed on {page_name}: "
                    f"{type(e).__name__}: {e}. Disabling for remaining pages.",
                    file=sys.stderr,
                )
                broken.add(engine.name)
                continue
            engine_pages[engine.name] = page_result
            _append_audit_rows(audit_rows[engine.name], page_result, book_name)

        if not engine_pages:
            print(f"[step2_5] No engine succeeded on {page_name}; skipping.",
                  file=sys.stderr)
            continue

        # Consensus merge
        consensus = merge_page(engine_pages, weights=weights)
        consensus_path = consensus_dir / f"{page_name}.json"
        write_consensus(consensus, consensus_path)
        _append_audit_rows(
            audit_rows["consensus"], consensus, book_name,
            include_agreement=True,
        )

        if verbose:
            stats = summarize(consensus)
            print(f"  {page_name}: {stats}")

    # Write per-engine audit CSVs under dataset/<book>/<engine>/
    out_root = Path(paths["output_dir"]) / book_name
    for engine_name, rows in audit_rows.items():
        out_dir = out_root / engine_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_audit_csv(out_dir / "labels_rec.csv", rows)

    if verbose:
        print(f"\n  Audit CSVs: {out_root}/<engine>/labels_rec.csv")


def _append_audit_rows(
    rows: list[dict],
    page: dict,
    book: str,
    include_agreement: bool = False,
) -> None:
    for col in page.get("columns", []):
        for ch in col.get("chars", []):
            row = {
                "source": book,
                "page": page.get("page"),
                "column": col["column"],
                "char_idx": ch["char_idx"],
                "char": ch.get("char") or "",
                "confidence": f"{ch.get('confidence', 0.0):.3f}",
            }
            if include_agreement:
                row["agreement"] = ch.get("agreement", "")
            rows.append(row)


def _write_audit_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Step 2.5: Multi-engine OCR + consensus voting",
    )
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    parser.add_argument("book", type=str, help="Book name")
    parser.add_argument(
        "--engines", type=str, default=None,
        help="Comma-separated engine names (override config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    step_cfg = config.get("step2_5", {})

    engine_names = (
        args.engines.split(",") if args.engines
        else step_cfg.get("engines", list(ENGINE_REGISTRY.keys()))
    )
    weights = step_cfg.get("weights", DEFAULT_WEIGHTS)

    engines = build_engines(engine_names)
    if not engines:
        print("[step2_5] No engines available. Aborting.", file=sys.stderr)
        sys.exit(1)

    process_book(config, args.book, engines, weights, verbose=True)


if __name__ == "__main__":
    main()
