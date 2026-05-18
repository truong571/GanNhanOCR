"""v5 export: parser_v5 + nom_detect_v3 (hybrid + close-merge + projection FB)
            + projection re-segmentation for underflow.

This is the cumulative best of ALL strengths from 7 prior phương án.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.image.image_processing import load_and_binarize

from parser_v5 import parse_v5
from parser_v2 import load_v1_transcription
from probe import load_qn_to_nom
from run_full import load_similar
from export_dataset_v4 import resegment_col
from nom_detect_v3 import detect_nom_columns_v3


def get_qn_lines_v5(book_dir, page_name, qn_dict):
    cache = book_dir / "transcriptions" / f"{page_name}_qn_ocr_cache.json"
    if cache.exists():
        try:
            text = json.load(open(cache)).get("text", "")
            if text:
                v, _ = parse_v5(text, qn_dict=qn_dict)
                if v:
                    return v, "v5"
        except Exception:
            pass
    v1_path = book_dir / "transcriptions" / f"{page_name}.txt"
    return (load_v1_transcription(str(v1_path)) if v1_path.exists() else {}), "v1"


def export_book(book_dir, book_name, qn_to_nom, similar, qn_dict,
                gw, sw, rw, counts):
    aligned = sorted((book_dir / "aligned").glob("page_*_aligned.json"))
    for af in aligned:
        page = af.stem.replace("_aligned", "")
        ocr_path = book_dir / "detected" / f"{page}_ocr_cache.json"
        if not ocr_path.exists():
            continue
        ocr_data = json.load(open(ocr_path))
        ocr_columns = ocr_data.get("columns", [])
        qn_lines, qn_src = get_qn_lines_v5(book_dir, page, qn_dict)
        qn_keys = sorted(qn_lines.keys())

        # Lazy binary load — only if we need projection fallback or reseg.
        binary = None

        def get_binary():
            nonlocal binary
            if binary is None:
                img = book_dir / "pages_denoised" / f"{page}.png"
                if not img.exists():
                    img = book_dir / "pages" / f"{page}.png"
                if img.exists():
                    _, binary = load_and_binarize(str(img))
            return binary

        # Try hybrid first; if fails, get binary and try projection.
        bin_for_detect = get_binary()
        if bin_for_detect is None:
            # No image — pure hybrid only
            from run_full import nom_cols_hybrid as _h
            cols = _h(ocr_columns, 4)
            method = "hybrid_no_image"
        else:
            cols, method = detect_nom_columns_v3(bin_for_detect, ocr_columns, 9)
        counts.setdefault(f"method_{method}", 0)
        counts[f"method_{method}"] += 1

        n_align = min(len(cols), len(qn_keys))
        page_col_match = (len(cols) == len(qn_keys))
        qn_parse_ok = (len(qn_lines) == 9)
        nom_suspect = (method == "suspect")

        for i in range(n_align):
            cluster = cols[i]
            qn_line = qn_lines[qn_keys[i]]
            actual = len(cluster["chars"])
            expected = len(qn_line)
            count_ok = True
            reseg_used = False

            if actual > expected:
                chars_used = [{"bbox": c["bbox"], "char": c.get("char")}
                              for c in cluster["chars"][actual - expected:]]
            elif actual < expected:
                bin_local = get_binary()
                chars_used = None
                if bin_local is not None and cluster["chars"]:
                    res = resegment_col(bin_local, cluster, expected)
                    if res:
                        chars_used = res
                        reseg_used = True
                if chars_used is None and bin_local is not None and \
                        "bbox" in cluster:
                    # Pure projection seg in cluster's bbox
                    from core.image.char_segmenter import segment_characters_in_column
                    try:
                        bboxes = segment_characters_in_column(
                            bin_local, cluster["bbox"], expected_count=expected)
                        if len(bboxes) == expected:
                            chars_used = [{"bbox": [int(b[0]), int(b[1]),
                                                    int(b[2]), int(b[3])],
                                           "char": None} for b in bboxes]
                            reseg_used = True
                    except Exception:
                        pass
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
                if not qn_parse_ok: suspect.append("qn_parse_lt9")
                if not page_col_match: suspect.append("col_count_mismatch")
                if not count_ok: suspect.append("col_underflow")
                if nom_suspect: suspect.append("nom_column_suspect")
                if not oc: suspect.append("no_ocr_char_at_pos")

                alignment_ok = (count_ok and page_col_match and qn_parse_ok
                                and not nom_suspect)
                row = {
                    "source": book_name, "page": page, "column": qn_keys[i],
                    "char_idx": j, "syllable": qn, "ocr_char": oc or "",
                    "tier": tier, "similar_intersect": "|".join(sim_inter),
                    "alignment_ok": int(alignment_ok),
                    "reseg": int(reseg_used),
                    "nom_col_method": method,
                    "suspect_reason": ";".join(suspect),
                    "bbox": json.dumps(ch.get("bbox")) if ch.get("bbox") else "",
                    "qn_src": qn_src,
                }
                if alignment_ok and oc and qn and tier in (1, 2):
                    gw.writerow(row); counts["gold"] += 1
                elif alignment_ok and oc and qn and tier == 3:
                    sw.writerow(row); counts["silver"] += 1
                else:
                    rw.writerow(row); counts["review"] += 1
                counts[f"t{tier}"] += 1
                if reseg_used:
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

    out_dir = Path(__file__).parent / "out" / "dataset_v5"
    for sub in ("gold", "silver", "review"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    fieldnames = ["source", "page", "column", "char_idx", "syllable",
                  "ocr_char", "tier", "similar_intersect", "alignment_ok",
                  "reseg", "nom_col_method", "suspect_reason", "bbox", "qn_src"]
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
        "tier1": counts["t1"], "tier2": counts["t2"],
        "tier3": counts["t3"], "tier0": counts["t0"],
        "reseg_pairs": counts["reseg_pairs"],
        "col_methods": {k: v for k, v in counts.items() if k.startswith("method_")},
        "total": counts["gold"] + counts["silver"] + counts["review"],
    }
    json.dump(summary, open(out_dir / "summary.json", "w"),
              ensure_ascii=False, indent=2)
    print("\n=== v5 export summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
