"""v_pb_nguon_goc_10 (04/10) — PHẢN BIỆN: dòng TỔNG của Bảng IV ("~54,3 % -> 84,7 %, +30,4") được tính thế nào?

Đầu vào: 10 cặp trước/sau của báo cáo (chép từ đề bài K12) và số ô mỗi bộ (độ dài measure_out/_tn8/base/<bộ>.pkl).
Tính: TB không trọng số, TB trọng số theo số ô của bộ, và mọi cách trộn (trước có/không trọng số x sau có/không trọng số).
Ra: measure_out/_tn11/verify/pb_nguon_goc/k12_trong_so.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
rows = [("stt2", 60.4, 93.8), ("stt4", 58.1, 83.3), ("stt11", 61.0, 84.2), ("Chr", 74.2, 90.0), ("L83", 75.4, 85.0), ("KVK", 88.9, 94.4),
        ("L16", 49.8, 94.7), ("TK", 48.2, 89.5), ("B18", 43.0, 70.3), ("B34", 34.6, 61.9)]
n = np.array([len(T.load_base(b)) for b, _, _ in rows], float)
bf = np.array([r[1] for r in rows]); af = np.array([r[2] for r in rows])
res = {"n_o_moi_bo": {r[0]: int(x) for r, x in zip(rows, n)}, "tong_n": int(n.sum()),
       "khong_trong_so": {"truoc": round(float(bf.mean()), 2), "sau": round(float(af.mean()), 2), "tang": round(float((af - bf).mean()), 2)},
       "trong_so_theo_o": {"truoc": round(float((bf * n).sum() / n.sum()), 2), "sau": round(float((af * n).sum() / n.sum()), 2), "tang": round(float(((af - bf) * n).sum() / n.sum()), 2)},
       "bao_cao_neu": {"truoc": 54.3, "sau": 84.7, "tang": 30.4}}
res["tang_khong_trong_so_tb_cac_dong_bao_cao"] = round(float(np.mean([27.3, 27.3, 44.9, 41.3, 33.4, 25.2, 23.2, 15.8, 9.6, 5.5])), 2)
# 'sau' của 8 dòng đầu có bằng max(HQC1,HQC2,HQC3) của p05 không; 'trước' của 4 bộ nhãn người = min(f_font,f_fd) của p01
p05 = {r["book"]: r for r in json.loads((REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
chk = {}
for b, bef, aft in rows[:8]:
    h = [p05[b][k]["top1"] for k in ("hqc1_white_canvas", "hqc2_book_paper_canvas", "hqc3_canonical_normalized")]
    chk[b] = {"HQC": h, "sau_bao_cao": aft, "sau_bang_max_HQC": bool(abs(max(h) - aft) < 0.05), "HQC1_da_la_max": bool(h[0] >= max(h) - 1e-9)}
res["sau_la_max_HQC_o_8_bo"] = {"so_bo_khop": int(sum(v["sau_bang_max_HQC"] for v in chk.values())), "so_bo_HQC1_da_la_max": int(sum(v["HQC1_da_la_max"] for v in chk.values())), "chi_tiet": chk}
p01 = {b: json.loads((REPO / f"measure_out/_tn11/p01_{b}.json").read_text(encoding="utf-8"))["top1"] for b in ("B18", "B34", "L16", "TK")}
res["truoc_4_bo_la_min_font_fd"] = {b: {"min(f_font,f_fd)": min(p01[b]["f_font"]["tat_ca"], p01[b]["f_fd"]["tat_ca"]), "truoc_bao_cao": bef}
                                    for b, bef, _ in rows if b in p01}
print(json.dumps(res, ensure_ascii=False, indent=1))
(OUT / "k12_trong_so.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
