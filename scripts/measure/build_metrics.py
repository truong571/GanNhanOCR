#!/usr/bin/env python
"""build_metrics.py — chỉ số một lần build (dataset_out_<X>/) để so trước/sau (B1', 2026-09-22; 0 token).

Đọc labels.csv (+ labels_final.csv nếu có), summary.json (layout_gate), remediation_report.json, confusion_fix_report.json
và (tuỳ chọn) prepared*/<book>/transcriptions/page_*.json để gắn nguồn QN (qn_source: ocr | <ref>_exact | <ref>_fuzzy)
cho từng ô theo (page, column, syl_idx < len_odd → câu lục, else câu bát).

Chạy:
  .venv/bin/python scripts/measure/build_metrics.py --book KimVanKieu1884 \
      dataset_out_KimVanKieu1884:prepared/KimVanKieu1884/transcriptions \
      dataset_out_KimVanKieu1884_b1:prepared/KimVanKieu1884/transcriptions [--out measure_out/.../build_metrics.json]
Mỗi tham số vị trí = <dataset_out dir>[:<transcriptions dir>]. In JSON; --out ghi tệp. Chỉ đọc.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def load_qn_source(trans_dir: Path) -> dict[tuple[str, int], tuple[int, str, str]]:
    """(page, column) → (len_odd, qn_source_odd, qn_source_even)."""
    out = {}
    for tf in sorted(glob.glob(str(trans_dir / "page_*.json"))):
        t = json.load(open(tf, encoding="utf-8"))
        for c in t["columns"]:
            vo, ve = c.get("verse_odd") or {}, c.get("verse_even") or {}
            out[(Path(tf).stem, int(c["column"]))] = (int(c.get("len_odd") or 0), vo.get("qn_source") or "ocr",
                                                        ve.get("qn_source") or "ocr")
    return out


def metrics(d: Path, book: str, trans_dir: Path | None) -> dict:
    rows = list(csv.DictReader(open(d / "labels.csv", encoding="utf-8", newline="")))
    cols = {}
    for r in rows:
        cols.setdefault((r["page"], _int(r["column"])), r)
    n_cols = len(cols)
    det_eq = sum(1 for r in cols.values() if _int(r.get("n_det")) == _int(r.get("n_qn")))
    m_eq = sum(1 for r in cols.values() if _int(r.get("n_ocr")) == _int(r.get("n_qn")))
    n14 = sum(1 for r in cols.values() if _int(r.get("n_qn")) == 14)
    det_eq14 = sum(1 for r in cols.values() if _int(r.get("n_qn")) == 14 and _int(r.get("n_det")) == 14)
    out = dict(dir=str(d), n_cells=len(rows), n_cols=n_cols,
               tier_raw=dict(Counter(r["tier"] for r in rows)),
               tier_v3=dict(Counter(r.get("tier_v3", "") for r in rows)),
               n_det_eq_N=det_eq, n_det_eq_N_pct=round(100 * det_eq / n_cols, 1),
               n_det_eq_N_in_qn14=f"{det_eq14}/{n14}",
               M_eq_N=m_eq, M_eq_N_pct=round(100 * m_eq / n_cols, 1),
               n_qn_eq_14_cols=n14,
               ocr_char_pct=round(100 * sum(1 for r in rows if r.get("ocr_char")) / len(rows), 1),
               p_register_ge_08=sum(1 for r in rows if (float(r["p_register"]) if r.get("p_register") else 0) >= 0.8),
               box_source=dict(Counter(r.get("box_source", "") for r in rows)),
               count_source=dict(Counter(r.get("count_source", "") for r in rows)),
               rule_top=dict(Counter(r.get("rule", "") for r in rows).most_common(10)),
               crop_quality_flag=dict(Counter(r.get("crop_quality_flag", "") for r in rows)))
    sp = d / "summary.json"
    if sp.exists():
        S = json.load(open(sp, encoding="utf-8"))
        lg = (S.get("layout_gate") or {}).get(book) or {}
        out["page_ok"] = f"{lg.get('page_ok')}/{lg.get('pages')}"
        out["pages_not_ok"] = lg.get("pages_not_ok")
        out["n_anchor_pairs"] = S.get("n_anchor_pairs")
        out["detector_params_by_book"] = S.get("detector_params_by_book")
    rp = d / "remediation_report.json"
    if rp.exists():
        R = json.load(open(rp, encoding="utf-8"))
        out["remediation"] = {k: R[k] for k in R if k in ("tier_before", "tier_after", "n_quarantined", "n_demoted",
                                                            "census", "quarantine_reasons")}
    cp = d / "confusion_fix_report.json"
    if cp.exists():
        C = json.load(open(cp, encoding="utf-8"))
        out["confusion_fix"] = {k: C[k] for k in C if k in ("n_fixes", "total_demoted", "tier_after")}
    fp = d / "labels_final.csv"
    if fp.exists():
        fin = list(csv.DictReader(open(fp, encoding="utf-8", newline="")))
        out["tier_final"] = dict(Counter(r["tier"] for r in fin))
        if trans_dir is not None:
            qs = load_qn_source(trans_dir)
            by = {}
            for r in fin:
                lo, so, se = qs.get((r["page"], _int(r["column"])), (0, "ocr", "ocr"))
                src = so if (_int(r.get("syl_idx")) or 0) < lo else se
                by.setdefault(src, Counter())[r["tier"]] += 1
            out["tier_final_by_qn_source"] = {k: dict(v) for k, v in sorted(by.items())}
            out["tier_final_by_qn_source_pct_GOLD"] = {k: round(100 * v.get("GOLD", 0) / sum(v.values()), 1)
                                                       for k, v in sorted(by.items())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="<dataset_out dir>[:<transcriptions dir>]")
    ap.add_argument("--book", default="KimVanKieu1884")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    for spec in a.dirs:
        d, _, t = spec.partition(":")
        res[Path(d).name] = metrics(REPO / d if not Path(d).is_absolute() else Path(d), a.book,
                                    (REPO / t) if t else None)
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt, encoding="utf-8")
        print("→", a.out)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
