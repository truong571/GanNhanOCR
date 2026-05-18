"""Strategy B+ evaluation: parser_v2 fallback + overlap_threshold sweep.

For each page:
  qn_lines = v1; if len(v1) < 9 and raw txt exists, try v2 and adopt if v2 has more lines.

Sweep overlap_threshold ∈ {0.3, 0.4, 0.5, 0.6} on cluster step.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from align_v2 import align_page_b
from nom_column_cluster import cluster_columns
from parser_v2 import load_v1_transcription, parse_v2
from probe import load_qn_to_nom, probe_pairs


def get_qn_lines(book_dir: Path, page_name: str) -> tuple[dict, str]:
    """Returns (qn_lines, source) where source ∈ {'v1','v2'}."""
    v1_path = book_dir / "transcriptions" / f"{page_name}.txt"
    v1 = load_v1_transcription(str(v1_path)) if v1_path.exists() else {}
    if len(v1) >= 9:
        return v1, "v1"
    raw_path = book_dir / "transcriptions_raw" / f"{page_name}.txt"
    if raw_path.exists():
        v2, _ = parse_v2(raw_path.read_text(encoding="utf-8"))
        if len(v2) > len(v1):
            return v2, "v2"
    return v1, "v1"


def align_with_threshold(ocr_columns, qn_lines, overlap_threshold):
    # Override cluster_columns default by monkey-patching call
    clusters = cluster_columns(ocr_columns, overlap_threshold=overlap_threshold)
    # Reuse align_page_b logic by recomputing here (simple inline)
    qn_keys = sorted(qn_lines.keys())
    n_align = min(len(clusters), len(qn_keys))
    pairs = []
    col_ok = []
    for i in range(n_align):
        cluster = clusters[i]
        qn_line = qn_lines[qn_keys[i]]
        actual = len(cluster["chars"])
        expected = len(qn_line)
        if actual > expected:
            chars_used = cluster["chars"][actual - expected:]
            align_ok = True
        elif actual < expected:
            chars_used = cluster["chars"]
            align_ok = False
        else:
            chars_used = cluster["chars"]
            align_ok = True
        for j in range(min(len(chars_used), len(qn_line))):
            pairs.append((chars_used[j].get("char"), qn_line[j], align_ok))
        col_ok.append(align_ok)
    return pairs, {
        "n_clusters": len(clusters),
        "n_qn": len(qn_keys),
        "col_count_match": len(clusters) == len(qn_keys),
        "n_cols_ok": sum(col_ok),
        "all_ok": (len(clusters) == len(qn_keys)) and all(col_ok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen2")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--dict", default="Dict/QuocNgu_SinoNom_TongHop3.csv")
    args = ap.parse_args()

    repo = Path(args.repo)
    book_dir = repo / "prepared" / args.book
    qn_to_nom = load_qn_to_nom(str(repo / args.dict))

    thresholds = [0.3, 0.4, 0.5, 0.6]
    aligned_files = sorted((book_dir / "aligned").glob("page_*_aligned.json"))
    page_names = [af.stem.replace("_aligned", "") for af in aligned_files]

    # Stats: per threshold {hits, total, hits_ok, total_ok, pages_all_ok, pages_col_match}
    summary = {t: {"hits": 0, "total": 0, "hits_ok": 0, "total_ok": 0,
                   "pages_all_ok": 0, "pages_col_match": 0} for t in thresholds}
    v2_used = 0
    per_page_rows = []

    for page_name in page_names:
        ocr_path = book_dir / "detected" / f"{page_name}_ocr_cache.json"
        if not ocr_path.exists():
            continue
        ocr_data = json.load(open(ocr_path))
        ocr_columns = ocr_data.get("columns", [])
        qn_lines, src = get_qn_lines(book_dir, page_name)
        if src == "v2":
            v2_used += 1

        row = {"page": page_name, "qn_src": src, "n_qn": len(qn_lines)}
        for t in thresholds:
            pairs_t, stats_t = align_with_threshold(ocr_columns, qn_lines, t)
            # probe
            hits = total = hits_ok = total_ok = 0
            for oc, qn, ok in pairs_t:
                if oc and qn:
                    total += 1
                    hit = oc in qn_to_nom.get(qn.strip().lower(), [])
                    if hit:
                        hits += 1
                    if ok:
                        total_ok += 1
                        if hit:
                            hits_ok += 1
            summary[t]["hits"] += hits
            summary[t]["total"] += total
            summary[t]["hits_ok"] += hits_ok
            summary[t]["total_ok"] += total_ok
            if stats_t["all_ok"]:
                summary[t]["pages_all_ok"] += 1
            if stats_t["col_count_match"]:
                summary[t]["pages_col_match"] += 1
            row[f"t{t}_rate"] = (hits / total) if total else 0.0
            row[f"t{t}_ok_rate"] = (hits_ok / total_ok) if total_ok else 0.0
            row[f"t{t}_clusters"] = stats_t["n_clusters"]
            row[f"t{t}_all_ok"] = stats_t["all_ok"]
        per_page_rows.append(row)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(per_page_rows, open(out_dir / "per_page_v2.json", "w"),
              ensure_ascii=False, indent=2)

    # Write markdown
    md = []
    md.append(f"# Strategy B+ — parser_v2 fallback + overlap_threshold sweep\n")
    md.append(f"Book: **{args.book}**, pages: **{len(page_names)}**, "
              f"parser_v2 adopted on: **{v2_used}** pages\n")
    md.append("\n## Sweep over overlap_threshold\n")
    md.append("| threshold | rate (all pairs) | rate (col_ok) | "
              "pages col_count_match | pages all_ok |")
    md.append("|---:|---:|---:|---:|---:|")
    for t in thresholds:
        s = summary[t]
        r = (s["hits"] / s["total"]) if s["total"] else 0
        r_ok = (s["hits_ok"] / s["total_ok"]) if s["total_ok"] else 0
        md.append(f"| {t:.1f} | {r*100:.2f}% ({s['hits']}/{s['total']}) "
                  f"| {r_ok*100:.2f}% ({s['hits_ok']}/{s['total_ok']}) "
                  f"| {s['pages_col_match']}/{len(page_names)} "
                  f"({s['pages_col_match']*100/len(page_names):.1f}%) "
                  f"| {s['pages_all_ok']}/{len(page_names)} "
                  f"({s['pages_all_ok']*100/len(page_names):.1f}%) |")

    md.append("\n## Comparison vs baselines\n")
    md.append("| Variant | rate | pages all_ok |")
    md.append("|---|---:|---:|")
    md.append("| A (current pipeline) | 6.39% | — |")
    md.append("| B baseline (t=0.5, v1 only) | 42.17% | 100/159 |")
    best_t = max(thresholds, key=lambda t: summary[t]["hits_ok"] / max(1, summary[t]["total_ok"]))
    s = summary[best_t]
    md.append(f"| **B+ best (t={best_t}, v2 fallback)** | "
              f"**{(s['hits_ok']/max(1,s['total_ok']))*100:.2f}%** "
              f"| **{s['pages_all_ok']}/{len(page_names)}** |")

    md.append("\n## Conclusion\n")
    md.append("- parser_v2 fallback adopted on pages where v1 returned < 9 lines.")
    md.append("- overlap_threshold sweep verifies merge behaviour on cases where "
              "Kimhannom split marker stack into a separate column.")
    md.append(f"- Best threshold: **{best_t}**.")

    (out_dir / "RESULTS_v2.md").write_text("\n".join(md) + "\n")
    print(f"[done] wrote {out_dir/'RESULTS_v2.md'}")
    for t in thresholds:
        s = summary[t]
        r_ok = (s["hits_ok"] / s["total_ok"]) * 100 if s["total_ok"] else 0
        print(f"  t={t}  ok_rate={r_ok:.2f}%  "
              f"pages_all_ok={s['pages_all_ok']}/{len(page_names)}")


if __name__ == "__main__":
    main()
