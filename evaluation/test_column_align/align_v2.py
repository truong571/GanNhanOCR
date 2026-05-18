"""Strategy B alignment: QN = source of truth for column count and char count.

For each page:
  1. qn_lines = parse QN (use existing transcriptions/*.txt = parser_v1 output).
  2. Cluster Kimhannom OCR cols by x-overlap → real Nôm cols (right-to-left).
  3. Align cluster k ↔ qn_lines[k+1] by index.
  4. Within each cluster, len(cluster.chars) vs len(qn_line):
     - actual == expected     → direct 1-1 pair.
     - actual >  expected     → strip top (actual - expected) chars as marker.
     - actual <  expected     → underflow: pad with None, flag suspect.
  5. Compute probe + per-column / per-page diagnostics.

Output per page: list of pairs (ocr_char, qn_syllable, col_id, alignment_ok, note)
plus a stats dict.
"""

import json
from pathlib import Path

from nom_column_cluster import cluster_columns
from parser_v2 import load_v1_transcription, parse_v2


def align_page_b(ocr_columns: list[list[dict]],
                 qn_lines: dict[int, list[str]]) -> tuple[list[dict], dict]:
    """Strategy B alignment. Returns (pairs, stats)."""
    clusters = cluster_columns(ocr_columns)
    qn_keys_sorted = sorted(qn_lines.keys())
    n_clusters = len(clusters)
    n_qn = len(qn_keys_sorted)

    page_col_match = (n_clusters == n_qn)
    pairs: list[dict] = []
    col_stats = []

    n_align = min(n_clusters, n_qn)
    for i in range(n_align):
        cluster = clusters[i]
        qn_id = qn_keys_sorted[i]
        qn_line = qn_lines[qn_id]
        actual = len(cluster["chars"])
        expected = len(qn_line)

        col_note = "ok"
        chars_used = cluster["chars"]
        col_align_ok = True

        if actual > expected:
            stripped = actual - expected
            chars_used = cluster["chars"][stripped:]  # drop top marker chars
            col_note = f"strip_top:{stripped}"
        elif actual < expected:
            col_note = f"underflow:{expected - actual}"
            col_align_ok = False

        # Pair by index up to common length
        n_pair = min(len(chars_used), len(qn_line))
        for j in range(n_pair):
            ocr_char = chars_used[j].get("char")
            qn_syl = qn_line[j]
            pairs.append({
                "col_qn_id": qn_id,
                "cluster_idx": i,
                "char_idx": j,
                "ocr_char": ocr_char,
                "qn_syl": qn_syl,
                "col_align_ok": col_align_ok,
                "note": col_note,
            })

        col_stats.append({
            "qn_id": qn_id,
            "expected": expected,
            "actual": actual,
            "stripped": max(0, actual - expected),
            "underflow": max(0, expected - actual),
            "align_ok": col_align_ok,
            "note": col_note,
        })

    # Account for unmatched clusters / unmatched QN lines
    unmatched_clusters = n_clusters - n_align
    unmatched_qn = n_qn - n_align

    stats = {
        "n_ocr_cols": len(ocr_columns),
        "n_clusters": n_clusters,
        "n_qn_lines": n_qn,
        "col_count_match": page_col_match,
        "unmatched_clusters": unmatched_clusters,
        "unmatched_qn": unmatched_qn,
        "col_stats": col_stats,
        "n_cols_align_ok": sum(1 for cs in col_stats if cs["align_ok"]),
        "alignment_ok_all": (page_col_match and all(cs["align_ok"] for cs in col_stats)),
    }
    return pairs, stats


def run_page(page_name: str, prepared_dir: Path) -> tuple[list[dict], dict] | None:
    ocr_path = prepared_dir / "detected" / f"{page_name}_ocr_cache.json"
    trans_path = prepared_dir / "transcriptions" / f"{page_name}.txt"
    if not ocr_path.exists() or not trans_path.exists():
        return None
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    ocr_columns = ocr_data.get("columns", [])
    qn_lines = load_v1_transcription(str(trans_path))
    return align_page_b(ocr_columns, qn_lines)
