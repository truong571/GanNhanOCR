"""Run Strategy A vs Strategy B on SachThanhTruyen2 and emit RESULTS.md.

Strategy A = existing pipeline output (prepared/.../aligned/*.json).
Strategy B = align_v2 (this dir).

Both probed with the same tier-1 dict (ocr_char ∈ qn_to_nom[qn_syl]).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from align_v2 import run_page
from probe import load_qn_to_nom, probe_pairs
from strategy_a_probe import collect_pairs_a


def evaluate_book(book_dir: Path, qn_to_nom: dict, out_dir: Path):
    aligned_dir = book_dir / "aligned"
    aligned_files = sorted(aligned_dir.glob("page_*_aligned.json"))

    per_page = []  # rows: {page, a_hits, a_total, b_hits, b_total, ...}

    for af in aligned_files:
        page_name = af.stem.replace("_aligned", "")

        # Strategy A
        pairs_a = collect_pairs_a(af)
        a_hits, a_total = probe_pairs(pairs_a, qn_to_nom)

        # Strategy B
        result = run_page(page_name, book_dir)
        if result is None:
            continue
        pairs_b_raw, stats_b = result
        pairs_b = [(p["ocr_char"], p["qn_syl"]) for p in pairs_b_raw]
        b_hits, b_total = probe_pairs(pairs_b, qn_to_nom)

        # B variants: align_ok-only
        pairs_b_ok = [(p["ocr_char"], p["qn_syl"])
                      for p in pairs_b_raw if p["col_align_ok"]]
        b_ok_hits, b_ok_total = probe_pairs(pairs_b_ok, qn_to_nom)

        per_page.append({
            "page": page_name,
            "a_hits": a_hits, "a_total": a_total,
            "a_rate": (a_hits / a_total) if a_total else 0.0,
            "b_hits": b_hits, "b_total": b_total,
            "b_rate": (b_hits / b_total) if b_total else 0.0,
            "b_ok_hits": b_ok_hits, "b_ok_total": b_ok_total,
            "b_ok_rate": (b_ok_hits / b_ok_total) if b_ok_total else 0.0,
            "n_ocr_cols": stats_b["n_ocr_cols"],
            "n_clusters": stats_b["n_clusters"],
            "n_qn_lines": stats_b["n_qn_lines"],
            "col_count_match": stats_b["col_count_match"],
            "n_cols_align_ok": stats_b["n_cols_align_ok"],
            "alignment_ok_all": stats_b["alignment_ok_all"],
        })

    return per_page


def aggregate(per_page: list[dict]) -> dict:
    sum_a_hits = sum(p["a_hits"] for p in per_page)
    sum_a_total = sum(p["a_total"] for p in per_page)
    sum_b_hits = sum(p["b_hits"] for p in per_page)
    sum_b_total = sum(p["b_total"] for p in per_page)
    sum_b_ok_hits = sum(p["b_ok_hits"] for p in per_page)
    sum_b_ok_total = sum(p["b_ok_total"] for p in per_page)
    pages = len(per_page)
    pages_col_match = sum(1 for p in per_page if p["col_count_match"])
    pages_align_ok = sum(1 for p in per_page if p["alignment_ok_all"])

    def safe(n, d): return (n / d) if d else 0.0

    return {
        "n_pages": pages,
        "A_rate": safe(sum_a_hits, sum_a_total),
        "A_hits": sum_a_hits, "A_total": sum_a_total,
        "B_rate": safe(sum_b_hits, sum_b_total),
        "B_hits": sum_b_hits, "B_total": sum_b_total,
        "B_ok_rate": safe(sum_b_ok_hits, sum_b_ok_total),
        "B_ok_hits": sum_b_ok_hits, "B_ok_total": sum_b_ok_total,
        "pages_col_match": pages_col_match,
        "pages_col_match_pct": safe(pages_col_match, pages),
        "pages_align_ok": pages_align_ok,
        "pages_align_ok_pct": safe(pages_align_ok, pages),
    }


def write_results(agg: dict, per_page: list[dict], out_dir: Path, book_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save per-page json + csv-ish summary
    with open(out_dir / "per_page.json", "w", encoding="utf-8") as f:
        json.dump(per_page, f, ensure_ascii=False, indent=2)

    # Distribution buckets
    def bucket(r):
        if r < 0.1: return "<10%"
        if r < 0.2: return "10-20%"
        if r < 0.3: return "20-30%"
        if r < 0.4: return "30-40%"
        if r < 0.5: return "40-50%"
        if r < 0.6: return "50-60%"
        if r < 0.7: return "60-70%"
        if r < 0.8: return "70-80%"
        return "80%+"

    from collections import Counter
    dist_a = Counter(bucket(p["a_rate"]) for p in per_page)
    dist_b = Counter(bucket(p["b_rate"]) for p in per_page)
    dist_b_ok = Counter(bucket(p["b_ok_rate"]) for p in per_page if p["b_ok_total"])

    order = ["<10%", "10-20%", "20-30%", "30-40%", "40-50%",
             "50-60%", "60-70%", "70-80%", "80%+"]

    def fmt_dist(d):
        return " | ".join(f"{k}:{d.get(k,0)}" for k in order)

    md = []
    md.append(f"# Column Align Strategy A vs B — {book_name}\n")
    md.append("Metric: tier-1 probe rate `ocr_char ∈ qn_to_nom[qn_syllable]`.\n")
    md.append("Strategy A = existing pipeline `aligned/*_aligned.json` matches.\n")
    md.append("Strategy B = align_v2 (cluster OCR cols by x-overlap → "
              "strip top marker → 1-1 pair by index).\n")
    md.append("Strategy B_ok = subset of B pairs where the column is "
              "`col_align_ok=True` (no underflow).\n")
    md.append("\n## Headline\n")
    md.append(f"- Pages evaluated: **{agg['n_pages']}**")
    md.append(f"- Strategy A probe rate: **{agg['A_rate']*100:.2f}%** "
              f"({agg['A_hits']}/{agg['A_total']})")
    md.append(f"- Strategy B probe rate: **{agg['B_rate']*100:.2f}%** "
              f"({agg['B_hits']}/{agg['B_total']})")
    md.append(f"- Strategy B_ok probe rate: **{agg['B_ok_rate']*100:.2f}%** "
              f"({agg['B_ok_hits']}/{agg['B_ok_total']})")
    md.append(f"- Pages with cluster count == QN line count: "
              f"**{agg['pages_col_match']}/{agg['n_pages']}** "
              f"({agg['pages_col_match_pct']*100:.1f}%)")
    md.append(f"- Pages fully `alignment_ok` (all cols clean): "
              f"**{agg['pages_align_ok']}/{agg['n_pages']}** "
              f"({agg['pages_align_ok_pct']*100:.1f}%)")
    md.append("\n## Per-page rate distribution\n")
    md.append(f"Strategy A:    {fmt_dist(dist_a)}")
    md.append(f"Strategy B:    {fmt_dist(dist_b)}")
    md.append(f"Strategy B_ok: {fmt_dist(dist_b_ok)}")
    md.append("\n## Top 10 pages where B beats A (delta)\n")
    deltas = sorted(per_page, key=lambda p: -(p["b_rate"] - p["a_rate"]))[:10]
    md.append("| page | A% | B% | delta | clusters/qn | align_ok |")
    md.append("|---|---:|---:|---:|---|:-:|")
    for p in deltas:
        md.append(f"| {p['page']} | {p['a_rate']*100:.1f} | {p['b_rate']*100:.1f} "
                  f"| +{(p['b_rate']-p['a_rate'])*100:.1f} "
                  f"| {p['n_clusters']}/{p['n_qn_lines']} "
                  f"| {'Y' if p['alignment_ok_all'] else 'N'} |")
    md.append("\n## Top 10 pages where B regresses vs A\n")
    regress = sorted(per_page, key=lambda p: (p["b_rate"] - p["a_rate"]))[:10]
    md.append("| page | A% | B% | delta | clusters/qn | align_ok |")
    md.append("|---|---:|---:|---:|---|:-:|")
    for p in regress:
        md.append(f"| {p['page']} | {p['a_rate']*100:.1f} | {p['b_rate']*100:.1f} "
                  f"| {(p['b_rate']-p['a_rate'])*100:+.1f} "
                  f"| {p['n_clusters']}/{p['n_qn_lines']} "
                  f"| {'Y' if p['alignment_ok_all'] else 'N'} |")

    md.append("\n## Notes & next steps\n")
    md.append("- B_ok is the headline rate to compare. It restricts to columns "
              "where strategy B produced no underflow (i.e. clusters where Kimhannom "
              "had ≥ expected_count chars, so marker-strip is valid).")
    md.append("- Pages where `clusters > qn_lines` after x-overlap clustering "
              "indicate Kimhannom still over-split (marker x-shift > 50% overlap "
              "threshold). Tune `overlap_threshold` in nom_column_cluster.py.")
    md.append("- Pages where `clusters < qn_lines` indicate two real columns "
              "got merged (rare). Inspect manually.")
    md.append("- Underflow columns (cluster.actual < qn.expected) are NOT included "
              "in B_ok — these are candidates for projection-based re-segmentation "
              "on the original image (next iteration).")
    md.append("- Merge into pipeline only after B_ok ≥ A + 10pp AND no critical "
              "regression (>5pp drop) on ≥ 5% of pages.")

    with open(out_dir / "RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen2")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--dict", default="Dict/QuocNgu_SinoNom_TongHop3.csv")
    args = ap.parse_args()

    repo = Path(args.repo)
    book_dir = repo / "prepared" / args.book
    dict_path = repo / args.dict

    print(f"[load] dict: {dict_path}", flush=True)
    qn_to_nom = load_qn_to_nom(str(dict_path))
    print(f"[load] qn_to_nom entries: {len(qn_to_nom)}", flush=True)

    print(f"[run] evaluate {args.book}", flush=True)
    per_page = evaluate_book(book_dir, qn_to_nom, Path(__file__).parent / "out")
    print(f"[run] pages processed: {len(per_page)}", flush=True)

    agg = aggregate(per_page)
    out_dir = Path(__file__).parent / "out"
    write_results(agg, per_page, out_dir, args.book)
    print(f"[done] A={agg['A_rate']*100:.2f}%  B={agg['B_rate']*100:.2f}%  "
          f"B_ok={agg['B_ok_rate']*100:.2f}%", flush=True)
    print(f"[done] wrote {out_dir/'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
