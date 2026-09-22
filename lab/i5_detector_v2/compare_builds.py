#!/usr/bin/env python
"""So sánh 2 bản build sách mới (v1 ↔ v2): tier, GOLD ảnh, box_source, I5 n_det==N, khớp dị bản (cross).

  compare_builds.py --books LucVanTien1883 KimVanKieu1884 Chrestomathie1872 --suffix _v2 [--base-suffix ""]
Đọc dataset_out_<Book><suffix>/{summary.json, labels.csv, mechanism_gates_report.json, auto_precision/SUMMARY.json}.
Chỉ đọc; in bảng ≤ 40 dòng. Số là proxy như mọi số của dự án (không GT người).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def stats(book: str, suffix: str) -> dict | None:
    d = REPO / f"dataset_out_{book}{suffix}"
    if not (d / "labels.csv").exists():
        return None
    out: dict = {"dir": d.name}
    rows = list(csv.DictReader(open(d / "labels.csv", encoding="utf-8")))
    out["n_cells"] = len(rows)
    out["seg_backend"] = ",".join(sorted({r.get("seg_backend", "") for r in rows}))[:60]
    out["box_source"] = dict(Counter(r.get("box_source", "") for r in rows))
    cols = {}
    for r in rows:
        k = (r.get("page"), r.get("column"))
        if k not in cols and r.get("n_det") not in (None, "") and r.get("n_qn") not in (None, ""):
            try:
                cols[k] = (int(float(r["n_det"])), int(float(r["n_qn"])))
            except ValueError:
                pass
    if cols:
        out["I5_n_det_eq_N_pct"] = round(100 * sum(1 for a, b in cols.values() if a == b) / len(cols), 1)
        out["n_cols"] = len(cols)
    for name in ("labels_gated.csv", "labels_final.csv"):
        if (d / name).exists():
            t = Counter(r.get("tier", "") for r in csv.DictReader(open(d / name, encoding="utf-8")))
            out[f"tier[{name}]"] = dict(sorted(t.items()))
            break
    mg = d / "mechanism_gates_report.json"
    if mg.exists():
        r = json.load(open(mg, encoding="utf-8"))
        out["gold_image"] = (r.get("gold_image") or {}).get("after")
        out["gold_text_only"] = r.get("gold_text_only")
        out["images_to_export"] = r.get("images_to_export")
        out["gates_decided"] = r.get("gates_decided")
    ap_ = d / "auto_precision" / "SUMMARY.json"
    if ap_.exists():
        s = json.load(open(ap_, encoding="utf-8"))
        b = ((s.get("cross") or {}).get("books") or {}).get(book) or {}
        for ref, v in (b.get("refs") or {}).items():
            at = v.get("all_tiers") or {}
            out[f"cross[{ref}] GOLD_eq_pct"] = at.get("GOLD_eq_pct")
            out[f"cross[{ref}] GOLD_eq_or_di_the_pct"] = at.get("GOLD_eq_or_di_the_pct")
            out[f"cross[{ref}] n_GOLD"] = at.get("n_GOLD")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--books", nargs="+", default=["LucVanTien1883", "KimVanKieu1884", "Chrestomathie1872"])
    ap.add_argument("--suffix", default="_v2")
    ap.add_argument("--base-suffix", default="")
    a = ap.parse_args()
    for book in a.books:
        base, new = stats(book, a.base_suffix), stats(book, a.suffix)
        print(f"\n== {book}: {base['dir'] if base else '(không có bản gốc)'}  ↔  {new['dir'] if new else '(chưa build _v2)'}")
        if not base or not new:
            continue
        keys = [k for k in list(base) + [k for k in new if k not in base] if k != "dir"]
        for k in keys:
            b, n = base.get(k), new.get(k)
            flag = "" if b == n else "  <-- khác"
            print(f"  {k:42s} {str(b)[:60]:60s} | {str(n)[:60]}{flag}")
    print("\nQuyết định (README §5): giữ v2 khi I5 n_det==N tăng, GOLD ảnh không giảm đáng kể, cross GOLD_eq_pct không giảm > 0,5 điểm, "
          "box_source ink_cut/detector_low giảm; ngược lại: set_book_keys.py --remove detector_ckpt (giữ detector_resize: area nếu "
          "box_ref_v1_area đã tốt hơn v1).")


if __name__ == "__main__":
    main()
