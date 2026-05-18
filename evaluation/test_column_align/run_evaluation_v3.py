"""Strategy B++ : parser_v3 + overlap_threshold sweep.

parser_v3 brings 159/159 pages to exactly 9 QN lines (vs v1: 137/159).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nom_column_cluster import cluster_columns
from parser_v3 import parse_v3
from parser_v2 import load_v1_transcription
from probe import load_qn_to_nom


def get_qn_lines_v3(book_dir: Path, page_name: str,
                    qn_dict: set[str] | None) -> tuple[dict, str]:
    """Use VietOCR cache (clean re-OCR) as primary source for parser_v3.
    Fall back to raw PDF text, then to existing v1 transcription.
    """
    vietocr_cache = book_dir / "transcriptions" / f"{page_name}_qn_ocr_cache.json"
    if vietocr_cache.exists():
        try:
            data = json.load(open(vietocr_cache))
            text = data.get("text", "")
            if text:
                v3, _ = parse_v3(text, qn_dict=qn_dict)
                if v3:
                    return v3, "v3_vietocr"
        except Exception:
            pass
    raw_path = book_dir / "transcriptions_raw" / f"{page_name}.txt"
    if raw_path.exists():
        v3, _ = parse_v3(raw_path.read_text(encoding="utf-8"), qn_dict=qn_dict)
        if v3:
            return v3, "v3_pdf"
    v1_path = book_dir / "transcriptions" / f"{page_name}.txt"
    return (load_v1_transcription(str(v1_path)) if v1_path.exists() else {}), "v1"


def align_with(ocr_columns, qn_lines, overlap_threshold):
    clusters = cluster_columns(ocr_columns, overlap_threshold=overlap_threshold)
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
    qn_dict_set = set(qn_to_nom.keys())

    thresholds = [0.3, 0.4, 0.5, 0.6]
    aligned_files = sorted((book_dir / "aligned").glob("page_*_aligned.json"))
    page_names = [af.stem.replace("_aligned", "") for af in aligned_files]

    summary = {t: {"hits": 0, "total": 0, "hits_ok": 0, "total_ok": 0,
                   "pages_all_ok": 0, "pages_col_match": 0} for t in thresholds}
    n_v3 = 0
    n_v3_pdf = 0
    n_v1 = 0
    n_9 = 0
    per_page = []

    for page_name in page_names:
        ocr_path = book_dir / "detected" / f"{page_name}_ocr_cache.json"
        if not ocr_path.exists():
            continue
        ocr_data = json.load(open(ocr_path))
        ocr_columns = ocr_data.get("columns", [])
        qn_lines, src = get_qn_lines_v3(book_dir, page_name, qn_dict_set)
        if src == "v3_vietocr":
            n_v3 += 1
        elif src == "v3_pdf":
            n_v3_pdf += 1
        else:
            n_v1 += 1
        if len(qn_lines) == 9:
            n_9 += 1

        row = {"page": page_name, "src": src, "n_qn": len(qn_lines)}
        for t in thresholds:
            pairs_t, st = align_with(ocr_columns, qn_lines, t)
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
            if st["all_ok"]:
                summary[t]["pages_all_ok"] += 1
            if st["col_count_match"]:
                summary[t]["pages_col_match"] += 1
            row[f"t{t}_rate"] = (hits / total) if total else 0.0
            row[f"t{t}_ok_rate"] = (hits_ok / total_ok) if total_ok else 0.0
            row[f"t{t}_all_ok"] = st["all_ok"]
            row[f"t{t}_clusters"] = st["n_clusters"]
        per_page.append(row)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(per_page, open(out_dir / "per_page_v3.json", "w"),
              ensure_ascii=False, indent=2)

    n_pages = len(page_names)
    md = []
    md.append(f"# Strategy B++ — parser_v3 + overlap_threshold sweep\n")
    md.append(f"Book: **{args.book}**, pages: **{n_pages}**, "
              f"parser_v3 used on: **{n_v3}** pages, "
              f"pages with QN=9: **{n_9}/{n_pages}**\n")
    md.append("\n## Sweep over overlap_threshold\n")
    md.append("| t | rate (all pairs) | rate (col_ok) | "
              "pages col_count_match | pages all_ok |")
    md.append("|---:|---:|---:|---:|---:|")
    for t in thresholds:
        s = summary[t]
        r = (s["hits"] / s["total"]) if s["total"] else 0
        r_ok = (s["hits_ok"] / s["total_ok"]) if s["total_ok"] else 0
        md.append(f"| {t:.1f} | {r*100:.2f}% ({s['hits']}/{s['total']}) "
                  f"| {r_ok*100:.2f}% ({s['hits_ok']}/{s['total_ok']}) "
                  f"| {s['pages_col_match']}/{n_pages} "
                  f"({s['pages_col_match']*100/n_pages:.1f}%) "
                  f"| {s['pages_all_ok']}/{n_pages} "
                  f"({s['pages_all_ok']*100/n_pages:.1f}%) |")

    best_t = max(thresholds, key=lambda t: summary[t]["pages_all_ok"])
    s = summary[best_t]

    md.append("\n## Comparison vs baselines\n")
    md.append("| Variant | rate (col_ok) | pages all_ok | pages col_match |")
    md.append("|---|---:|---:|---:|")
    md.append("| A (current pipeline) | 6.39% | — | — |")
    md.append("| B (v1 parser, t=0.5) | 43.56% | 100/159 (62.9%) | 117/159 (73.6%) |")
    md.append("| B+ (v2 parser, t=0.3) | 41.96% | 128/159 (80.5%) | 151/159 (95.0%) |")
    md.append(f"| **B++ (v3 parser, t={best_t})** | "
              f"**{(s['hits_ok']/max(1,s['total_ok']))*100:.2f}%** "
              f"| **{s['pages_all_ok']}/{n_pages} "
              f"({s['pages_all_ok']*100/n_pages:.1f}%)** "
              f"| **{s['pages_col_match']}/{n_pages} "
              f"({s['pages_col_match']*100/n_pages:.1f}%)** |")

    (out_dir / "RESULTS_v3.md").write_text("\n".join(md) + "\n")
    print(f"[done] wrote {out_dir/'RESULTS_v3.md'}")
    for t in thresholds:
        s = summary[t]
        r_ok = (s["hits_ok"] / max(1, s["total_ok"])) * 100
        print(f"  t={t}  ok_rate={r_ok:.2f}%  "
              f"pages_all_ok={s['pages_all_ok']}/{n_pages}  "
              f"col_match={s['pages_col_match']}/{n_pages}")


if __name__ == "__main__":
    main()
