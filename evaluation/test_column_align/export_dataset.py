"""Export Gold / Silver / Review datasets per Final Plan Step 6.

Writes:
  out/dataset/gold/labels.csv
  out/dataset/silver/labels.csv
  out/dataset/review/labels.csv
  out/dataset/summary.json

Sample shape (CSV columns):
  source, page, column, char_idx, syllable, ocr_char, tier,
  similar_candidates, alignment_ok, suspect_reason, bbox

Note: bbox here is the Kimhannom char bbox (best available without running
image projection per pair). For Step 2B-quality bboxes, pair this export with
`test_projection_seg.py` output and re-derive bboxes from the projection
segmenter when promoting silver→gold.
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from probe import load_qn_to_nom
from parser_v3 import parse_v3
from parser_v2 import load_v1_transcription
from run_full import (
    nom_cols_hybrid, get_qn_lines, load_similar,
)


def export_book(book_dir: Path, book_name: str, qn_to_nom, similar, qn_dict,
                gold_w, silver_w, review_w, counts):
    aligned_files = sorted((book_dir / "aligned").glob("page_*_aligned.json"))

    for af in aligned_files:
        page = af.stem.replace("_aligned", "")
        ocr_path = book_dir / "detected" / f"{page}_ocr_cache.json"
        if not ocr_path.exists():
            continue
        ocr_data = json.load(open(ocr_path))
        ocr_columns = ocr_data.get("columns", [])
        qn_lines, qn_src = get_qn_lines(book_dir, page, qn_dict)

        qn_keys = sorted(qn_lines.keys())
        cols = nom_cols_hybrid(ocr_columns, min_len=4)
        n_align = min(len(cols), len(qn_keys))

        page_col_match = (len(cols) == len(qn_keys))
        qn_parse_ok = (len(qn_lines) == 9)

        for i in range(n_align):
            cluster = cols[i]
            qn_line = qn_lines[qn_keys[i]]
            actual = len(cluster["chars"])
            expected = len(qn_line)
            if actual > expected:
                chars_used = cluster["chars"][actual - expected:]
                count_ok = True
            elif actual < expected:
                chars_used = cluster["chars"]
                count_ok = False
            else:
                chars_used = cluster["chars"]
                count_ok = True

            for j in range(min(len(chars_used), len(qn_line))):
                ch = chars_used[j]
                oc = ch.get("char")
                qn = qn_line[j]
                bbox = ch.get("bbox")
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
                            tier = 3
                else:
                    tier = 0

                suspect = []
                if not qn_parse_ok:
                    suspect.append("qn_parse_lt9")
                if not page_col_match:
                    suspect.append("col_count_mismatch")
                if not count_ok:
                    suspect.append("col_underflow")
                if not oc:
                    suspect.append("no_ocr_char")

                row = {
                    "source": book_name,
                    "page": page,
                    "column": qn_keys[i],
                    "char_idx": j,
                    "syllable": qn,
                    "ocr_char": oc or "",
                    "tier": tier,
                    "similar_intersect": "|".join(sim_inter),
                    "alignment_ok": int(count_ok and page_col_match
                                        and qn_parse_ok),
                    "suspect_reason": ";".join(suspect),
                    "bbox": json.dumps(bbox) if bbox else "",
                    "qn_src": qn_src,
                }

                alignment_ok = (count_ok and page_col_match and qn_parse_ok
                                and oc and qn)
                if alignment_ok and tier in (1, 2):
                    gold_w.writerow(row)
                    counts["gold"] += 1
                elif alignment_ok and tier == 3:
                    silver_w.writerow(row)
                    counts["silver"] += 1
                else:
                    review_w.writerow(row)
                    counts["review"] += 1
                counts[f"t{tier}"] += 1


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

    out_dir = Path(__file__).parent / "out" / "dataset"
    (out_dir / "gold").mkdir(parents=True, exist_ok=True)
    (out_dir / "silver").mkdir(parents=True, exist_ok=True)
    (out_dir / "review").mkdir(parents=True, exist_ok=True)

    fieldnames = ["source", "page", "column", "char_idx", "syllable",
                  "ocr_char", "tier", "similar_intersect", "alignment_ok",
                  "suspect_reason", "bbox", "qn_src"]

    counts = {"gold": 0, "silver": 0, "review": 0,
              "t0": 0, "t1": 0, "t2": 0, "t3": 0}

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
            print(f"  export {book}...")
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
        "total": counts["gold"] + counts["silver"] + counts["review"],
    }
    json.dump(summary, open(out_dir / "summary.json", "w"),
              ensure_ascii=False, indent=2)
    print("\n=== Export summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
