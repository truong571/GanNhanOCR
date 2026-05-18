"""Step 2: Structure-driven alignment (dataset_v5 logic).

Default method (`structural`): the validated pipeline from
evaluation/test_column_align/. Achieves 437/445 pages structurally OK and
41,824 gold pairs across 3 books (vs. 4,133 in the legacy Levenshtein
approach).

Pipeline:
  parser_v5 (QN 1-9, 98.9%)
    → nom_detect_v3 (hybrid → close-merge → projection fallback, 100%)
    → marker strip if Kim count > expected
    → projection re-segment if Kim count < expected
    → 1-1 pair by index in column
    → emit aligned/<page>_aligned.json with crops + detection JSON.

Legacy method (`levenshtein`): the original alignment is preserved in
`pipeline/step2_align_legacy.py`. Select via `step2.method` in config or
`--legacy` CLI flag.

Output schema (aligned/<page>_aligned.json) is backward-compatible with
step3_label.py — each pair has {type, column, syllable, char: {ocr_char,
bbox, crop_file, cleaned_file}}.
"""

import argparse
import json
import sys
from pathlib import Path

# Make evaluation/test_column_align/ importable so we reuse VALIDATED logic
# without copy-pasting (which would create drift). The eval folder is the
# source of truth — when fixing a parser/detector bug, edit there and step2
# picks it up automatically.
_EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "test_column_align"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.image.char_segmenter import segment_characters_in_column  # noqa: E402
from core.image.crop_cleaner import CharacterCleaner  # noqa: E402
from core.image.image_processing import load_and_binarize  # noqa: E402
from core.text.dictionary import load_qn_to_nom, load_similarity_dict  # noqa: E402

# Validated modules from evaluation/test_column_align/
from parser_v5 import parse_v5  # noqa: E402
from parser_v2 import load_v1_transcription  # noqa: E402
from nom_detect_v3 import detect_nom_columns_v3  # noqa: E402
from export_dataset_v4 import resegment_col  # noqa: E402

from pipeline.step0_setup import load_config  # noqa: E402


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _get_qn_lines(book_dir: Path, page_name: str,
                  qn_dict: set | None) -> tuple[dict, str]:
    """Load QN lines via parser_v5 from VietOCR cache, fall back to v1 txt."""
    cache = book_dir / "transcriptions" / f"{page_name}_qn_ocr_cache.json"
    if cache.exists():
        try:
            text = json.load(open(cache, "r", encoding="utf-8")).get("text", "")
            if text:
                v5, _ = parse_v5(text, qn_dict=qn_dict)
                if v5:
                    return v5, "v5"
        except Exception:
            pass
    v1_path = book_dir / "transcriptions" / f"{page_name}.txt"
    return (load_v1_transcription(str(v1_path)) if v1_path.exists() else {}), "v1"


def process_page_structural(
    page_name: str,
    data_dir: Path,
    qn_dict_set: set,
    step1_cfg: dict,
    step2_cfg: dict | None = None,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Structural alignment for one page. Returns (alignment, stats).

    Same output contract as the legacy `process_page`:
      - aligned list of {type, column, syllable, char: {ocr_char, bbox,
        crop_file, cleaned_file}} dicts.
      - stats {matches, gaps, chars}.
    Plus writes detection JSON and crop PNGs into detected/.
    """
    pages_dir = data_dir / "pages"
    denoised_dir = data_dir / "pages_denoised"
    crops_dir = data_dir / "detected" / "crops"
    cleaned_dir = data_dir / "detected" / "crops_cleaned"

    img_path = pages_dir / f"{page_name}.png"
    if not img_path.exists():
        return [], {}

    color_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if color_img is None:
        return [], {}
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    H_img, W_img = gray_img.shape

    # Load Kimhannom OCR cache
    ocr_path = data_dir / "detected" / f"{page_name}_ocr_cache.json"
    if not ocr_path.exists():
        return [], {}
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    ocr_columns = ocr_data.get("columns", [])

    # Parse QN lines (v5)
    qn_lines, qn_src = _get_qn_lines(data_dir, page_name, qn_dict_set)
    qn_keys = sorted(qn_lines.keys())

    # Load binary (for nom_detect_v3 projection fallback + reseg)
    bin_src = denoised_dir / f"{page_name}.png"
    if not bin_src.exists():
        bin_src = img_path
    try:
        _, binary = load_and_binarize(str(bin_src))
    except Exception:
        binary = None

    # Detect 9 Nôm columns (hybrid → close-merge → projection fallback).
    if binary is not None:
        cols, col_method = detect_nom_columns_v3(binary, ocr_columns, 9)
    else:
        from run_full import nom_cols_hybrid
        cols = nom_cols_hybrid(ocr_columns, min_len=4)
        col_method = "hybrid_no_image"

    page_col_match = (len(cols) == len(qn_keys))
    qn_parse_ok = (len(qn_lines) == 9)
    nom_suspect = (col_method == "suspect")

    # PARTIAL-ALIGNMENT recovery: if QN parsed < 9 but Nôm has enough cols to
    # cover the highest QN line id, pair by line_id (nom_cols[k-1] ↔ line k).
    # This handles pages where VietOCR dropped a leading or trailing line
    # marker but the Nôm image still has all 9 cols.
    max_qn = max(qn_keys) if qn_keys else 0
    partial_recovery = (not qn_parse_ok and not nom_suspect
                        and len(cols) >= max_qn and max_qn > 0)
    page_ok = ((page_col_match and qn_parse_ok and not nom_suspect)
               or partial_recovery)

    # Iteration plan: list of (nom_idx, qn_line_id) pairs.
    if partial_recovery:
        iter_pairs = [(line_id - 1, line_id) for line_id in qn_keys
                      if (line_id - 1) < len(cols)]
    else:
        n_align = min(len(cols), len(qn_keys))
        iter_pairs = [(i, qn_keys[i]) for i in range(n_align)]

    # Build per-col char lists (apply marker strip / reseg).
    col_chars: list[list[dict]] = []
    col_ok_flags: list[bool] = []
    col_ids: list[int] = []

    for nom_idx, line_id in iter_pairs:
        cluster = cols[nom_idx]
        qn_line = qn_lines[line_id]
        actual = len(cluster["chars"])
        expected = len(qn_line)
        count_ok = True
        chars_used: list[dict]

        if actual > expected:
            chars_used = [
                {"bbox": [int(b) for b in c["bbox"]],
                 "ocr_char": c.get("char")}
                for c in cluster["chars"][actual - expected:]
            ]
        elif actual < expected:
            chars_used = None  # type: ignore
            if binary is not None and cluster["chars"]:
                res = resegment_col(binary, cluster, expected)
                if res:
                    chars_used = [
                        {"bbox": [int(b) for b in r["bbox"]],
                         "ocr_char": r.get("char")} for r in res
                    ]
            if chars_used is None and binary is not None and \
                    cluster.get("bbox"):
                try:
                    bboxes = segment_characters_in_column(
                        binary, cluster["bbox"], expected_count=expected)
                    if len(bboxes) == expected:
                        chars_used = [
                            {"bbox": [int(x1), int(y1), int(x2), int(y2)],
                             "ocr_char": None}
                            for (x1, y1, x2, y2) in bboxes
                        ]
                except Exception:
                    pass
            if chars_used is None:
                chars_used = [
                    {"bbox": [int(b) for b in c["bbox"]],
                     "ocr_char": c.get("char")} for c in cluster["chars"]
                ]
                count_ok = False
        else:
            chars_used = [
                {"bbox": [int(b) for b in c["bbox"]],
                 "ocr_char": c.get("char")} for c in cluster["chars"]
            ]

        col_chars.append(chars_used)
        col_ok_flags.append(count_ok)
        col_ids.append(line_id)

    # ── Crop generation + detection JSON ──
    crop_size = step1_cfg.get("crop_size", 64)
    cleaner = CharacterCleaner(
        target_size=crop_size,
        sauvola_k=step1_cfg.get("sauvola_k", 0.2),
        sauvola_window=step1_cfg.get("sauvola_window", 25),
        min_stroke=step1_cfg.get("min_stroke", 2),
    )
    pad_frac = (step2_cfg or {}).get("crop_pad_frac", 0.0)

    detection_columns = []
    aligned: list[dict] = []
    total_chars = 0
    total_matches = 0

    page_crops_dir = crops_dir / page_name
    page_cleaned_dir = cleaned_dir / page_name

    for i, col_num in enumerate(col_ids):
        chars = col_chars[i]
        syllables = qn_lines[col_num]

        if chars:
            page_crops_dir.mkdir(parents=True, exist_ok=True)
            page_cleaned_dir.mkdir(parents=True, exist_ok=True)

        col_data = {
            "column": col_num,
            "num_chars": len(chars),
            "count_ok": col_ok_flags[i],
            "chars": [],
        }

        for char_idx, char_info in enumerate(chars):
            if char_idx >= len(syllables):
                break
            cx1, cy1, cx2, cy2 = char_info["bbox"]
            # Clip + pad
            if pad_frac > 0.0:
                w = cx2 - cx1
                h = cy2 - cy1
                px = int(round(w * pad_frac))
                py = int(round(h * pad_frac))
                cx1 = max(0, cx1 - px)
                cy1 = max(0, cy1 - py)
                cx2 = min(W_img, cx2 + px)
                cy2 = min(H_img, cy2 + py)
            cx1 = max(0, min(cx1, W_img - 1))
            cx2 = max(cx1 + 1, min(cx2, W_img))
            cy1 = max(0, min(cy1, H_img - 1))
            cy2 = max(cy1 + 1, min(cy2, H_img))

            crop_file = f"crops/{page_name}/col{col_num:02d}_char{char_idx:03d}.png"
            cleaned_file = f"crops_cleaned/{page_name}/col{col_num:02d}_char{char_idx:03d}.png"

            crop_color = color_img[cy1:cy2, cx1:cx2]
            if crop_color.size > 0:
                cv2.imwrite(str(data_dir / "detected" / crop_file), crop_color)
                gray_crop = gray_img[cy1:cy2, cx1:cx2]
                try:
                    cleaned, _ = cleaner.clean(gray_crop)
                    if cleaned is not None:
                        cv2.imwrite(str(data_dir / "detected" / cleaned_file),
                                    cleaned)
                except Exception:
                    cleaned_file = ""

            char_record = {
                "char_idx": char_idx,
                "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "width": int(cx2 - cx1),
                "height": int(cy2 - cy1),
                "crop_file": crop_file,
                "cleaned_file": cleaned_file,
                "ocr_char": char_info.get("ocr_char"),
            }
            col_data["chars"].append(char_record)

            aligned.append({
                "type": "match",
                "column": col_num,
                "syllable": syllables[char_idx],
                "char": char_record,
                "alignment_ok": bool(col_ok_flags[i] and page_ok),
            })
            total_matches += 1
            total_chars += 1

        detection_columns.append(col_data)

    # Save detection JSON
    detection_data = {
        "book_page": page_name,
        "image_size": [int(W_img), int(H_img)],
        "num_columns": len(detection_columns),
        "total_chars": total_chars,
        "method": col_method,
        "qn_src": qn_src,
        "qn_parse_ok": qn_parse_ok,
        "page_col_match": page_col_match,
        "partial_recovery": partial_recovery,
        "alignment_ok_all": page_ok and all(col_ok_flags),
        "columns": detection_columns,
    }
    det_path = data_dir / "detected" / f"{page_name}_detection.json"
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump(detection_data, f, ensure_ascii=False, indent=2,
                  cls=NumpyEncoder)

    stats = {
        "matches": total_matches,
        "gaps": 0,  # structural method emits no gaps; suspect rows still have type=match
        "chars": total_chars,
        "page_ok": detection_data["alignment_ok_all"],
        "method": col_method,
    }
    return aligned, stats


def align_book_structural(config: dict, book_name: str, verbose: bool = True):
    paths = config["paths"]
    step1_cfg = config.get("step1", {})
    step2_cfg = config.get("step2", {})
    data_dir = Path(paths["data_dir"]) / book_name

    qn_to_nom = load_qn_to_nom(paths["qn_to_nom_dict"])
    qn_dict_set = set(qn_to_nom.keys())

    trans_dir = data_dir / "transcriptions"
    aligned_dir = data_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    trans_files = sorted(trans_dir.glob("page_*.txt"))
    if not trans_files:
        print(f"[ERROR] No transcription files in {trans_dir}", file=sys.stderr)
        return

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step 2: Structural Align — {book_name}")
        print(f"  Method: structural (parser_v5 + nom_detect_v3 + reseg)")
        print(f"  Pages: {len(trans_files)}")
        print(f"{'='*60}")

    total_matches = 0
    total_chars = 0
    pages_ok = 0
    method_counts: dict[str, int] = {}

    for trans_path in trans_files:
        page_name = trans_path.stem
        if page_name.endswith("_qn_tmp") or page_name.endswith("_qn_ocr_cache"):
            continue

        alignment, stats = process_page_structural(
            page_name, data_dir, qn_dict_set, step1_cfg, step2_cfg,
            verbose=verbose,
        )
        if not alignment:
            if verbose:
                print(f"  [SKIP] {page_name}")
            continue

        out_path = aligned_dir / f"{page_name}_aligned.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(alignment, f, ensure_ascii=False, indent=2,
                      cls=NumpyEncoder)

        total_matches += stats["matches"]
        total_chars += stats["chars"]
        if stats.get("page_ok"):
            pages_ok += 1
        m = stats.get("method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1

        if verbose:
            print(f"  {page_name}: {stats['matches']} matches, "
                  f"{stats['chars']} chars, method={stats['method']}, "
                  f"page_ok={stats.get('page_ok')}")

    if verbose:
        print(f"\n  Totals: {total_matches} matches, {total_chars} chars cropped")
        print(f"  Pages structurally OK: {pages_ok}/{len(trans_files)}")
        print(f"  Method distribution: {method_counts}")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Structure-driven Align")
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    parser.add_argument("book", type=str, help="Book name")
    parser.add_argument("--legacy", action="store_true",
                        help="Fall back to the original Levenshtein alignment "
                             "(pipeline/step2_align_legacy.py).")
    args = parser.parse_args()

    config = load_config(args.config)
    method = config.get("step2", {}).get("method", "structural")
    if args.legacy or method == "levenshtein":
        from pipeline.step2_align_legacy import align_book
        align_book(config, args.book)
    else:
        align_book_structural(config, args.book)


if __name__ == "__main__":
    main()
