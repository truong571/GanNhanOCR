"""v_pb_truy_nguon_12 (04/10) — PHẢN BIỆN: ở 6 bộ không nhãn người p05 lấy sự thật := D.label của pipeline. Đo: nhãn pipeline có CHÍNH LÀ chữ kim (is_kim) không?
Top-1 của luật 'chọn chữ kim' so với nhãn pipeline trên đúng tập ô p05 (ô có nhãn pipeline trong ứng viên). 0 API, CPU, đọc bảng ứng viên.
Ra: measure_out/_tn11/verify/pb_truy_nguon/kim_la_nhan.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"; res = {}
for b in ("stt2", "stt4", "stt11", "Chr", "L83", "KVK"):
    D = T.load_base(b); F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "is_kim"]]
    lbl = D.label.values[F.i.values]; F["yl"] = ((F.c.values == lbl) & (lbl != "")).astype(int)
    h = F.groupby("i").yl.max(); cells = h[h == 1].index.values; G = F[F.i.isin(cells)]
    k = G[G.is_kim == 1].groupby("i").yl.max().reindex(cells).fillna(0)
    lab_eq_ocr = float(((D.label.values[cells] == D.ocr_char.values[cells])).mean())
    res[b] = dict(n=int(len(cells)), kim_trung_nhan_pct=round(100 * float(k.mean()), 2), nhan_bang_ocr_char_pct=round(100 * lab_eq_ocr, 2))
    print(b, res[b], flush=True)
(OUT / "kim_la_nhan.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
