"""v_pb_nguon_goc_16 (04/10) — PHẢN BIỆN: đối chiếu từng hàng bảng gốc p05 với lần chạy lại HÀM GỐC ở hạt giống đã phục hồi (cả HQC3, cả mean_margin).

Đọc: ket_qua_full_10_bo.json (gốc), tim_seed_*.json (script 02: hạt giống khớp dấu vân tay HQC1+HQC2),
      chay_goc_rec_*.json (script 01 --pairs: hàm gốc p05 chạy lại với đúng hạt giống đó).
Ra: measure_out/_tn11/verify/pb_nguon_goc/doi_chieu_hat_giong.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V = REPO / "measure_out/_tn11/verify/pb_nguon_goc"
orig = {r["book"]: r for r in json.loads((REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
found = {}
for fp in glob.glob(str(V / "tim_seed_*.json")):
    for b, v in json.loads(Path(fp).read_text(encoding="utf-8")).items():
        found[b] = v
rerun = {}
for fp in glob.glob(str(V / "chay_goc_rec_*.json")):
    for r in json.loads(Path(fp).read_text(encoding="utf-8"))["runs"]:
        rerun[(r["book"], r["seed"])] = r["res"]
KEYS = ("hqc1_white_canvas", "hqc2_book_paper_canvas", "hqc3_canonical_normalized")
out = {}
for b in ("stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK", "B18", "B34"):
    if b not in found:
        out[b] = {"trang_thai": "chua_tim"}
        continue
    f = found[b]
    seeds = [h["seed"] for h in f["hits"] if h["khop_cos_margin"]]
    row = {"n_hat_giong_cung_n": f["seeds_cung_n"], "n_khop_top1": f["ung_vien_khop_top1"], "seeds_khop_day_du": seeds, "tu_kiem_bang_tra": f.get("tu_kiem")}
    for s in seeds:
        rr = rerun.get((b, s))
        if rr is None:
            row[f"chay_lai_{s}"] = "chua_chay"
            continue
        o = orig[b]
        same = (rr["n_cells"] == o["n_cells"])
        det = {}
        for k in KEYS:
            for m in ("top1", "mean_cos", "mean_margin"):
                ok = abs(rr[k][m] - o[k][m]) <= (0.05 if m == "top1" else 1.1e-4)
                det[f"{k[:4]}_{m}"] = bool(ok)
                same &= ok
        row[f"chay_lai_{s}"] = {"khop_ca_9_so_+n": bool(same), "goc": [o["n_cells"]] + [o[k]["top1"] for k in KEYS], "chay_lai": [rr["n_cells"]] + [rr[k]["top1"] for k in KEYS]}
    out[b] = row
(V / "doi_chieu_hat_giong.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for b, r in out.items():
    print(b, json.dumps(r, ensure_ascii=False)[:420])
