"""v_pb_truy_nguon_09 (04/10) — PHẢN BIỆN: mốc 'chọn đúng chữ kim' (is_kim) trên cùng tập ô/giao thức p01/TN8 (ô có chữ đúng trong ứng viên),
để đặt các số 'Trước/Sau' của báo cáo cạnh mốc mà pipeline thực sự đang có. 0 API, CPU, chỉ đọc bảng ứng viên.
Ra: measure_out/_tn11/verify/pb_truy_nguon/kim_baseline.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"
res = {}
for b in ("B18", "B34", "L16", "TK"):
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "is_kim", "f_vW"]]
    hy = F.groupby("i").y.max(); cells = hy[hy == 1].index.values
    F = F[F.i.isin(cells)]
    k = F[F.is_kim == 1].groupby("i").y.max().reindex(cells).fillna(0)       # chữ kim đúng chữ người? (ô không có ứng viên kim -> 0)
    has_kim = F.groupby("i").is_kim.max().reindex(cells).fillna(0)
    res[b] = dict(n=int(len(cells)), kim_dung_pct=round(100 * float(k.mean()), 2), o_co_ung_vien_kim_pct=round(100 * float(has_kim.mean()), 2),
                  kim_dung_tren_o_co_kim_pct=round(100 * float(k[has_kim == 1].mean()), 2))
    print(b, res[b], flush=True)
(OUT / "kim_baseline.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
