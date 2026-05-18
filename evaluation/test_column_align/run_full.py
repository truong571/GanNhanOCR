"""Full evaluation across all 3 books with 3-tier dataset classification.

For each pair (ocr_char, qn_syllable, alignment_ok):
  tier 1 = ocr_char ∈ qn_to_nom[qn_syl]
  tier 2 = similar_dict[ocr_char] ∩ qn_to_nom[qn_syl] ≠ ∅
  tier 3 = otherwise (candidate — would need visual rank to confirm)

Dataset 3-tier (without running FontDiffusion+DINOv2 here):
  gold   = alignment_ok AND tier ∈ {1, 2}
  silver = alignment_ok AND tier == 3 (would be ranked visually downstream)
  review = NOT alignment_ok OR no ocr_char

Also compares two Nôm column detection methods:
  A. cluster: x-overlap clustering of Kimhannom cols (current B++ approach)
  B. filter:  drop Kimhannom cols with len ≤ 3, keep rest right-to-left
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nom_column_cluster import cluster_columns
from parser_v3 import parse_v3
from parser_v2 import load_v1_transcription
from probe import load_qn_to_nom


def load_similar(path: str) -> dict[str, set]:
    d: dict[str, set] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    sims = ast.literal_eval(row[1])
                    if isinstance(sims, list):
                        d[row[0].strip()] = set(sims)
                except (ValueError, SyntaxError):
                    pass
    return d


def get_qn_lines(book_dir: Path, page_name: str,
                 qn_dict: set | None) -> tuple[dict, str]:
    cache = book_dir / "transcriptions" / f"{page_name}_qn_ocr_cache.json"
    if cache.exists():
        try:
            text = json.load(open(cache)).get("text", "")
            if text:
                v3, _ = parse_v3(text, qn_dict=qn_dict)
                if v3:
                    return v3, "v3"
        except Exception:
            pass
    v1_path = book_dir / "transcriptions" / f"{page_name}.txt"
    return (load_v1_transcription(str(v1_path)) if v1_path.exists() else {}), "v1"


def nom_cols_cluster(ocr_columns, t):
    return cluster_columns(ocr_columns, overlap_threshold=t)


def nom_cols_hybrid(ocr_columns, min_len=4):
    """Filter cols with len ≥ min_len as 'body' cols, then re-attach short cols
    to the nearest body col by x-distance. Preserves marker chars that filter
    alone would drop, while keeping filter's better col identification.
    """
    body = []
    shorts = []
    for col in ocr_columns:
        if not col:
            continue
        xs = [c["bbox"][0] for c in col] + [c["bbox"][2] for c in col]
        cx = (min(xs) + max(xs)) / 2
        rec = {"x_center": cx, "x_range": (min(xs), max(xs)),
               "chars": list(col)}
        if len(col) >= min_len:
            body.append(rec)
        else:
            shorts.append(rec)

    # Attach each short col to nearest body by x-distance (only if x-overlap > 0
    # OR distance < 80px — to avoid stretching across the page).
    for sh in shorts:
        if not body:
            break
        best = None
        best_d = float("inf")
        for b in body:
            # Compute x-distance between centers
            d = abs(sh["x_center"] - b["x_center"])
            # Require some overlap or close proximity
            overlap = max(0, min(sh["x_range"][1], b["x_range"][1])
                          - max(sh["x_range"][0], b["x_range"][0]))
            if overlap > 0 or d < 80:
                if d < best_d:
                    best_d = d
                    best = b
        if best is not None:
            best["chars"].extend(sh["chars"])

    for b in body:
        b["chars"].sort(key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
    body.sort(key=lambda m: -m["x_center"])
    return body


def nom_cols_filter(ocr_columns, min_len=4):
    """Drop OCR cols with len < min_len, then sort right-to-left.

    Each remaining 'col' is wrapped to match cluster_columns output shape:
    {x_center, chars}.
    """
    out = []
    for col in ocr_columns:
        if not col or len(col) < min_len:
            continue
        xs = [c["bbox"][0] for c in col] + [c["bbox"][2] for c in col]
        cx = (min(xs) + max(xs)) / 2
        # sort chars within col top-to-bottom
        chars_sorted = sorted(
            col, key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
        out.append({"x_center": cx, "chars": chars_sorted})
    out.sort(key=lambda m: -m["x_center"])
    return out


def align_and_tier(nom_cols, qn_lines, qn_to_nom, similar_dict):
    """Align by index; per cluster: strip top markers if actual > expected.
    Returns: (per_pair_records, stats).
    """
    qn_keys = sorted(qn_lines.keys())
    n_align = min(len(nom_cols), len(qn_keys))
    pairs = []
    col_align_ok = []
    for i in range(n_align):
        cluster = nom_cols[i]
        qn_line = qn_lines[qn_keys[i]]
        actual = len(cluster["chars"])
        expected = len(qn_line)
        if actual > expected:
            chars_used = cluster["chars"][actual - expected:]
            align_ok = True
            count_ok = True
        elif actual < expected:
            chars_used = cluster["chars"]
            align_ok = False
            count_ok = False
        else:
            chars_used = cluster["chars"]
            align_ok = True
            count_ok = True
        col_align_ok.append(count_ok)
        for j in range(min(len(chars_used), len(qn_line))):
            oc = chars_used[j].get("char")
            qn = qn_line[j]
            tier = 0  # 0=no probe possible
            if oc and qn:
                cands = qn_to_nom.get(qn.strip().lower(), [])
                if cands and oc in cands:
                    tier = 1
                elif cands and oc in similar_dict:
                    if any(s in cands for s in similar_dict[oc]):
                        tier = 2
                    else:
                        tier = 3
                else:
                    tier = 3
            pairs.append({
                "ocr_char": oc, "qn_syl": qn, "tier": tier,
                "align_ok": align_ok, "count_ok": count_ok,
                "col_qn_id": qn_keys[i], "char_idx": j,
            })
    n_clusters = len(nom_cols)
    n_qn = len(qn_keys)
    page_ok = (n_clusters == n_qn) and all(col_align_ok)
    return pairs, {
        "n_clusters": n_clusters,
        "n_qn": n_qn,
        "col_count_match": n_clusters == n_qn,
        "n_cols_ok": sum(col_align_ok),
        "page_ok": page_ok,
    }


def classify_tier3(pairs):
    """Bucket pairs by tier and gold/silver/review."""
    counts = {"t1": 0, "t2": 0, "t3": 0, "t0": 0,
              "gold": 0, "silver": 0, "review": 0}
    for p in pairs:
        t = p["tier"]
        counts[f"t{t}"] += 1
        if not p["ocr_char"] or not p["qn_syl"]:
            counts["review"] += 1
        elif p["align_ok"] and t in (1, 2):
            counts["gold"] += 1
        elif p["align_ok"] and t == 3:
            counts["silver"] += 1
        else:
            counts["review"] += 1
    return counts


def run_one_book(book_dir: Path, qn_to_nom, similar_dict, qn_dict_set,
                 method: str, t: float = 0.3, min_len: int = 4):
    aligned_files = sorted((book_dir / "aligned").glob("page_*_aligned.json"))
    book_pairs = []
    book_pages_ok = 0
    book_pages_col_match = 0
    book_pages_qn9 = 0
    n_pages = 0

    for af in aligned_files:
        page_name = af.stem.replace("_aligned", "")
        ocr_path = book_dir / "detected" / f"{page_name}_ocr_cache.json"
        if not ocr_path.exists():
            continue
        n_pages += 1
        ocr_data = json.load(open(ocr_path))
        ocr_columns = ocr_data.get("columns", [])
        qn_lines, _ = get_qn_lines(book_dir, page_name, qn_dict_set)
        if len(qn_lines) == 9:
            book_pages_qn9 += 1

        if method == "cluster":
            cols = nom_cols_cluster(ocr_columns, t)
        elif method == "filter":
            cols = nom_cols_filter(ocr_columns, min_len=min_len)
        elif method == "hybrid":
            cols = nom_cols_hybrid(ocr_columns, min_len=min_len)
        else:
            raise ValueError(method)

        pairs, stats = align_and_tier(cols, qn_lines, qn_to_nom, similar_dict)
        if stats["page_ok"]:
            book_pages_ok += 1
        if stats["col_count_match"]:
            book_pages_col_match += 1
        book_pairs.extend(pairs)

    return {
        "n_pages": n_pages,
        "pages_qn9": book_pages_qn9,
        "pages_col_match": book_pages_col_match,
        "pages_ok": book_pages_ok,
        "tier_counts": classify_tier3(book_pairs),
        "pairs_total": len(book_pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+",
                    default=["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    repo = Path(args.repo)
    qn_to_nom = load_qn_to_nom(str(repo / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"))
    qn_dict_set = set(qn_to_nom.keys())
    similar = load_similar(str(repo / "Dict" / "SinoNom_Similar_Dic_v2.csv"))
    print(f"[load] qn={len(qn_to_nom)} similar={len(similar)}")

    methods = [
        ("cluster", {"t": 0.3}),
        ("filter", {"min_len": 4}),
        ("hybrid", {"min_len": 4}),
    ]
    grand_results = {}

    for method_name, kwargs in methods:
        per_book = {}
        for book in args.books:
            book_dir = repo / "prepared" / book
            if not book_dir.exists():
                print(f"  [skip] {book} not found")
                continue
            res = run_one_book(book_dir, qn_to_nom, similar, qn_dict_set,
                               method=method_name, **kwargs)
            per_book[book] = res
            tc = res["tier_counts"]
            print(f"  {method_name} {book}: pages={res['n_pages']} "
                  f"pages_ok={res['pages_ok']} "
                  f"pairs={res['pairs_total']} "
                  f"t1={tc['t1']} t2={tc['t2']} t3={tc['t3']} "
                  f"gold={tc['gold']} silver={tc['silver']} review={tc['review']}")
        grand_results[method_name] = per_book

    # Write summary
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(grand_results, open(out_dir / "full_results.json", "w"),
              ensure_ascii=False, indent=2)

    md = []
    md.append("# Full evaluation — 3 books × 2 nôm-col methods\n")
    md.append("Methods:\n")
    md.append("- **cluster**: x-overlap clustering of Kimhannom cols (threshold=0.3).")
    md.append("- **filter**: drop Kimhannom cols with `len ≤ 3`, keep rest right-to-left.")
    md.append("- **hybrid**: filter as anchors + re-attach short cols by nearest x-distance.\n")
    md.append("Tiers (without visual rank):\n")
    md.append("- Tier 1: `ocr_char ∈ qn_to_nom[qn_syl]`.")
    md.append("- Tier 2: `similar_dict[ocr_char] ∩ qn_to_nom[qn_syl] ≠ ∅`.")
    md.append("- Tier 3: otherwise (would need FontDiffusion+DINOv2 to confirm).\n")
    md.append("3-tier dataset:\n")
    md.append("- **Gold**: `alignment_ok` AND tier ∈ {1, 2}.")
    md.append("- **Silver**: `alignment_ok` AND tier 3 (visual-rank downstream).")
    md.append("- **Review**: alignment suspect / missing OCR / no QN.\n")

    for method_name, per_book in grand_results.items():
        md.append(f"\n## Method: `{method_name}`\n")
        md.append("| Book | Pages | QN=9 | col_match | page_ok | "
                  "Pairs | Gold | Silver | Review | T1 | T2 | T3 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        tot = {k: 0 for k in
               ["n_pages", "pages_qn9", "pages_col_match", "pages_ok",
                "pairs_total", "gold", "silver", "review", "t1", "t2", "t3"]}
        for book, r in per_book.items():
            tc = r["tier_counts"]
            md.append(f"| {book} | {r['n_pages']} | {r['pages_qn9']} "
                      f"| {r['pages_col_match']} | {r['pages_ok']} "
                      f"| {r['pairs_total']} | {tc['gold']} | {tc['silver']} "
                      f"| {tc['review']} | {tc['t1']} | {tc['t2']} | {tc['t3']} |")
            tot["n_pages"] += r["n_pages"]
            tot["pages_qn9"] += r["pages_qn9"]
            tot["pages_col_match"] += r["pages_col_match"]
            tot["pages_ok"] += r["pages_ok"]
            tot["pairs_total"] += r["pairs_total"]
            for k in ("gold", "silver", "review", "t1", "t2", "t3"):
                tot[k] += tc[k]
        md.append(f"| **TOTAL** | **{tot['n_pages']}** "
                  f"| **{tot['pages_qn9']}** | **{tot['pages_col_match']}** "
                  f"| **{tot['pages_ok']}** | **{tot['pairs_total']}** "
                  f"| **{tot['gold']}** | **{tot['silver']}** "
                  f"| **{tot['review']}** | **{tot['t1']}** | **{tot['t2']}** "
                  f"| **{tot['t3']}** |")

        md.append("")
        md.append(f"Gold % of pairs: "
                  f"**{tot['gold']*100/max(1,tot['pairs_total']):.2f}%**")
        md.append(f"Tier1+2 hit rate (across all pairs, NOT just gold): "
                  f"**{(tot['t1']+tot['t2'])*100/max(1,tot['pairs_total']):.2f}%**")
        md.append(f"Pages structurally OK: "
                  f"**{tot['pages_ok']}/{tot['n_pages']} "
                  f"({tot['pages_ok']*100/max(1,tot['n_pages']):.1f}%)**")

    (out_dir / "RESULTS_full.md").write_text("\n".join(md) + "\n")
    print(f"\n[done] wrote {out_dir/'RESULTS_full.md'}")


if __name__ == "__main__":
    main()
