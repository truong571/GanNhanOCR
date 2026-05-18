"""Final export with parser_v4 + projection re-segmentation for underflow cols.

Improvements over export_dataset.py:
  1. parser_v4 (95.5% pages 9-line vs v3's 81.3%).
  2. For cols where hybrid.actual < qn.expected (underflow), run
     segment_characters_in_column on the Nôm image to derive new bboxes.
     Kimhannom OCR identity is then matched by y-overlap to the new bboxes;
     unmatched new bboxes keep ocr_char=None (still align-ok, just no Tier-1
     hit — falls into Silver/Review automatically).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.image.char_segmenter import segment_characters_in_column
from core.image.image_processing import load_and_binarize

from parser_v4 import parse_v4  # noqa: F401
from parser_v5 import parse_v5
from parser_v2 import load_v1_transcription
from probe import load_qn_to_nom
from run_full import nom_cols_hybrid, load_similar


def get_qn_lines_v4(book_dir: Path, page_name: str, qn_dict):
    cache = book_dir / "transcriptions" / f"{page_name}_qn_ocr_cache.json"
    if cache.exists():
        try:
            text = json.load(open(cache)).get("text", "")
            if text:
                # Prefer v5 (fixes 15 extra pages over v4), fall back to v4.
                v, _ = parse_v5(text, qn_dict=qn_dict)
                if len(v) == 9:
                    return v, "v5"
                v4, _ = parse_v4(text, qn_dict=qn_dict)
                if v4 and len(v4) >= len(v):
                    return v4, "v4"
                if v:
                    return v, "v5"
        except Exception:
            pass
    v1_path = book_dir / "transcriptions" / f"{page_name}.txt"
    return (load_v1_transcription(str(v1_path)) if v1_path.exists() else {}), "v1"


def resegment_col(binary, cluster, expected: int):
    """Project-segment col body and assign Kimhannom chars by y-overlap.

    Returns: list of dicts {bbox, ocr_char (or None)} with len == expected.
    """
    chars = cluster["chars"]
    x_lo = min(c["bbox"][0] for c in chars)
    x_hi = max(c["bbox"][2] for c in chars)
    y_lo = min(c["bbox"][1] for c in chars)
    y_hi = max(c["bbox"][3] for c in chars)
    H = binary.shape[0]
    pad = 12
    bbox = (x_lo, max(0, y_lo - pad), x_hi, min(H, y_hi + pad))
    new_bboxes = segment_characters_in_column(binary, bbox,
                                              expected_count=expected)
    if len(new_bboxes) != expected:
        return None

    # Map old kimhannom chars to new bboxes by max y-overlap.
    out = []
    used = set()
    for (nx1, ny1, nx2, ny2) in new_bboxes:
        best = None
        best_ov = 0
        for i, c in enumerate(chars):
            if i in used:
                continue
            cy1, cy2 = c["bbox"][1], c["bbox"][3]
            ov = max(0, min(ny2, cy2) - max(ny1, cy1))
            if ov > best_ov:
                best_ov = ov
                best = i
        ocr_char = None
        if best is not None and best_ov > 0:
            ocr_char = chars[best].get("char")
            used.add(best)
        out.append({
            "bbox": [int(nx1), int(ny1), int(nx2), int(ny2)],
            "char": ocr_char,
        })
    return out


def export_book(book_dir, book_name, qn_to_nom, similar, qn_dict,
                gw, sw, rw, counts):
    aligned_files = sorted((book_dir / "aligned").glob("page_*_aligned.json"))
    binary_cache = {}

    for af in aligned_files:
        page = af.stem.replace("_aligned", "")
        ocr_path = book_dir / "detected" / f"{page}_ocr_cache.json"
        if not ocr_path.exists():
            continue
        ocr_data = json.load(open(ocr_path))
        ocr_columns = ocr_data.get("columns", [])
        qn_lines, qn_src = get_qn_lines_v4(book_dir, page, qn_dict)
        qn_keys = sorted(qn_lines.keys())

        cols = nom_cols_hybrid(ocr_columns, min_len=4)
        n_align = min(len(cols), len(qn_keys))
        page_col_match = (len(cols) == len(qn_keys))
        qn_parse_ok = (len(qn_lines) == 9)

        # Lazy-load binary if any col needs reseg.
        binary = None

        for i in range(n_align):
            cluster = cols[i]
            qn_line = qn_lines[qn_keys[i]]
            actual = len(cluster["chars"])
            expected = len(qn_line)
            count_ok = True
            chars_used = None
            reseg = False

            if actual > expected:
                chars_used = [{"bbox": c["bbox"], "char": c.get("char")}
                              for c in cluster["chars"][actual - expected:]]
            elif actual < expected:
                # Try projection re-segmentation
                if binary is None:
                    img_path = book_dir / "pages_denoised" / f"{page}.png"
                    if not img_path.exists():
                        img_path = book_dir / "pages" / f"{page}.png"
                    if img_path.exists():
                        _, binary = load_and_binarize(str(img_path))
                if binary is not None:
                    reseg_result = resegment_col(binary, cluster, expected)
                    if reseg_result:
                        chars_used = reseg_result
                        reseg = True
                if chars_used is None:
                    chars_used = [{"bbox": c["bbox"], "char": c.get("char")}
                                  for c in cluster["chars"]]
                    count_ok = False
            else:
                chars_used = [{"bbox": c["bbox"], "char": c.get("char")}
                              for c in cluster["chars"]]

            for j in range(min(len(chars_used), len(qn_line))):
                ch = chars_used[j]
                oc = ch.get("char")
                qn = qn_line[j]
                tier = 3
                sim_inter = []
                if oc and qn:
                    cands = qn_to_nom.get(qn.strip().lower(), [])
                    if cands and oc in cands:
                        tier = 1
                    elif cands and oc in similar:
                        inter = [s for s in similar[oc] if s in cands]
                        if inter:
                            tier = 2
                            sim_inter = inter[:5]
                else:
                    tier = 0 if not oc else 3

                suspect = []
                if not qn_parse_ok:
                    suspect.append("qn_parse_lt9")
                if not page_col_match:
                    suspect.append("col_count_mismatch")
                if not count_ok:
                    suspect.append("col_underflow")
                if not oc:
                    suspect.append("no_ocr_char_at_pos")

                alignment_ok = (count_ok and page_col_match and qn_parse_ok)
                row = {
                    "source": book_name,
                    "page": page,
                    "column": qn_keys[i],
                    "char_idx": j,
                    "syllable": qn,
                    "ocr_char": oc or "",
                    "tier": tier,
                    "similar_intersect": "|".join(sim_inter),
                    "alignment_ok": int(alignment_ok),
                    "reseg": int(reseg),
                    "suspect_reason": ";".join(suspect),
                    "bbox": json.dumps(ch.get("bbox")) if ch.get("bbox") else "",
                    "qn_src": qn_src,
                }

                if alignment_ok and oc and qn and tier in (1, 2):
                    gw.writerow(row)
                    counts["gold"] += 1
                elif alignment_ok and oc and qn and tier == 3:
                    sw.writerow(row)
                    counts["silver"] += 1
                else:
                    rw.writerow(row)
                    counts["review"] += 1
                counts[f"t{tier}"] += 1
                if reseg:
                    counts["reseg_pairs"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+",
                    default=["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    repo = Path(args.repo)
    qn_to_nom = load_qn_to_nom(str(repo / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"))
    qn_dict = set(qn_to_nom.keys())
    similar = load_similar(str(repo / "Dict" / "SinoNom_Similar_Dic_v2.csv"))

    out_dir = Path(__file__).parent / "out" / "dataset_v4"
    (out_dir / "gold").mkdir(parents=True, exist_ok=True)
    (out_dir / "silver").mkdir(parents=True, exist_ok=True)
    (out_dir / "review").mkdir(parents=True, exist_ok=True)

    fieldnames = ["source", "page", "column", "char_idx", "syllable",
                  "ocr_char", "tier", "similar_intersect", "alignment_ok",
                  "reseg", "suspect_reason", "bbox", "qn_src"]

    counts = {"gold": 0, "silver": 0, "review": 0,
              "t0": 0, "t1": 0, "t2": 0, "t3": 0, "reseg_pairs": 0}

    with open(out_dir / "gold" / "labels.csv", "w", encoding="utf-8", newline="") as g, \
         open(out_dir / "silver" / "labels.csv", "w", encoding="utf-8", newline="") as s, \
         open(out_dir / "review" / "labels.csv", "w", encoding="utf-8", newline="") as r:
        gw = csv.DictWriter(g, fieldnames=fieldnames); gw.writeheader()
        sw = csv.DictWriter(s, fieldnames=fieldnames); sw.writeheader()
        rw = csv.DictWriter(r, fieldnames=fieldnames); rw.writeheader()
        for book in args.books:
            book_dir = repo / "prepared" / book
            if not book_dir.exists():
                continue
            print(f"  export {book}...", flush=True)
            export_book(book_dir, book, qn_to_nom, similar, qn_dict,
                        gw, sw, rw, counts)

    summary = {
        "books": args.books,
        "gold": counts["gold"],
        "silver": counts["silver"],
        "review": counts["review"],
        "tier1": counts["t1"],
        "tier2": counts["t2"],
        "tier3": counts["t3"],
        "tier0": counts["t0"],
        "reseg_pairs": counts["reseg_pairs"],
        "total": counts["gold"] + counts["silver"] + counts["review"],
    }
    json.dump(summary, open(out_dir / "summary.json", "w"),
              ensure_ascii=False, indent=2)
    print("\n=== v4 export summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
