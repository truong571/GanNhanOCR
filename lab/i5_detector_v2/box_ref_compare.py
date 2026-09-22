#!/usr/bin/env python
"""In bảng so sánh các lần chạy scripts/measure/box_ref_eval.py (v1 linear · v1 area · v2 area) từ summary.json.

  box_ref_compare.py measure_out/box_ref_v1 measure_out/box_ref_v1_area measure_out/box_ref_v2
Cột: I5 n_det==N @0,15 (cột), tầng trong-cửa-sổ đúng N, ok50 / miss / extra / |dy| / cắt-thân-chữ của legacy@0,15
trên ô tham chiếu verified (ảnh prepared) và của pitch (honest, chỉ ô nguồn detector). ≤ 40 dòng.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(d: Path):
    p = d / "summary.json"
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))


def main():
    dirs = [Path(x) for x in sys.argv[1:]] or [Path("measure_out/box_ref_v1"), Path("measure_out/box_ref_v1_area"),
                                               Path("measure_out/box_ref_v2")]
    runs = [(d.name, load(d)) for d in dirs]
    runs = [(n, s) for n, s in runs if s]
    if not runs:
        print("không có summary.json nào"); return
    books = sorted({b for _, s in runs for b in s.get("per_book", {})})
    hdr = f"{'sách / chỉ số':44s}" + "".join(f"{n[:18]:>19s}" for n, _ in runs)
    print(hdr); print("-" * len(hdr))
    for b in books:
        def col(s, key):
            e = (s.get("per_book") or {}).get(b) or {}
            c15 = (e.get("columns") or {}).get("prepared@0.15") or {}
            lg = (e.get("cells_verified") or {}).get("legacy@0.15_prepared") or {}
            ph = (e.get("cells_verified") or {}).get("pitch_prepared_honest(det_src_only)") or {}
            return {"I5 n_det==N @0,15 (cột) %": c15.get("I5_n_det_eq_N_pct"),
                    "tầng trong cửa sổ đúng N %": c15.get("n_det_in_tiers_eq_N_pct"),
                    "legacy@0,15 ok50 %": lg.get("ok_iou50_pct"), "legacy@0,15 miss %": lg.get("miss_pct"),
                    "legacy@0,15 extra/100": lg.get("extra_per_100_cells"), "legacy@0,15 |dy| %bước": lg.get("abs_dy_pct_pitch_med"),
                    "legacy@0,15 cắt thân chữ %": lg.get("cut_glyph_pct"),
                    "pitch honest ok50 %": ph.get("ok_iou50_pct"), "pitch cắt thân chữ %": ph.get("cut_glyph_pct")}.get(key)
        keys = ["I5 n_det==N @0,15 (cột) %", "tầng trong cửa sổ đúng N %", "legacy@0,15 ok50 %", "legacy@0,15 miss %",
                "legacy@0,15 extra/100", "legacy@0,15 |dy| %bước", "legacy@0,15 cắt thân chữ %", "pitch honest ok50 %",
                "pitch cắt thân chữ %"]
        for k in keys:
            print(f"{(b + ' ' + k)[:44]:44s}" + "".join(f"{str(col(s, k)):>19s}" for _, s in runs))
    print("\ndetector:", "; ".join(f"{n}: {(s.get('detector') or {}).get('ckpt', 'v1')} / {(s.get('detector') or {}).get('resize', 'linear')}"
                               for n, s in runs))
    print("Mọi số là proxy (ô tham chiếu kim + chiếu mực); chỉ 'cắt thân chữ' không phụ thuộc tham chiếu. CI mức cột ≈ ±3,5 điểm/27 trang.")


if __name__ == "__main__":
    main()
