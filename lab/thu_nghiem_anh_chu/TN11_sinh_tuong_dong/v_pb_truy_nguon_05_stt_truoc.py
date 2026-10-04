"""v_pb_truy_nguon_05 (04/10) — PHẢN BIỆN: tìm nguồn của 'Trước' STT (60,4 / 58,1 / 61,0) bằng liệt kê TỔ HỢP tầng/rule/cổng
(labels_final.csv, TN8 base) — mở rộng hơn 27 định nghĩa của người kiểm chứng. 0 API, CPU.
Ra: measure_out/_tn11/verify/pb_truy_nguon/stt_truoc.json
"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"
TARGET = {"stt2": 60.4, "stt4": 58.1, "stt11": 61.0}
L = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
print(L.columns.tolist()[:40])
res = {}
cands = {}  # tên định nghĩa -> {bộ: giá trị %}
for b in TARGET:
    X = L[L.book == b]
    n = len(X)
    # tổ hợp tầng
    tiers = sorted(set(X.tier))
    for k in range(1, len(tiers) + 1):
        for sub in itertools.combinations(tiers, k):
            cands.setdefault("labels_final tier in " + "+".join(sub), {})[b] = round(100 * float(X.tier.isin(sub).mean()), 2)
    # theo rule (tiền tố s1_inter_s2_direct, v.v.)
    for pre in sorted({r.split(":")[0] for r in X.rule if r}):
        cands.setdefault(f"labels_final rule == {pre}", {})[b] = round(100 * float((X.rule.str.split(":").str[0] == pre).mean()), 2)
    # tầng GOLD theo rule direct*
    g = X[X.tier == "GOLD"]
    for pre in sorted({r.split(":")[0] for r in g.rule if r}):
        cands.setdefault(f"labels_final GOLD & rule {pre}", {})[b] = round(100 * float(((X.tier == "GOLD") & (X.rule.str.split(":").str[0] == pre)).mean()), 2)
    # đo trên mẫu số khác: chỉ ô có nhãn/ocr_char
    for dn, dm in (("ocr_char!=''", X.ocr_char != ""), ("label!=''", X.label != "")):
        den = float(dm.sum())
        cands.setdefault(f"GOLD / {dn}", {})[b] = round(100 * float(((X.tier == "GOLD") & dm).sum() / max(den, 1)), 2)
    D = T.load_base(b)
    for t in sorted(set(D.tier)):
        cands.setdefault(f"tn8 base tier=={t}", {})[b] = round(100 * float((D.tier == t).mean()), 2)
    cands.setdefault("tn8 base GOLD ∪ GOLD_text_only", {})[b] = round(100 * float(D.tier.isin(["GOLD", "GOLD_text_only"]).mean()), 2)
    # tỉ lệ kim==nhãn, kim∈R...
    cands.setdefault("label==ocr_char / mọi ô (tn8 base)", {})[b] = round(100 * float(((D.label == D.ocr_char) & (D.label != "")).mean()), 2)
hit3 = {k: v for k, v in cands.items() if len(v) == 3 and all(abs(v[b] - TARGET[b]) <= 0.051 for b in TARGET)}
hit_any = {k: {b: v[b] for b in v if abs(v[b] - TARGET[b]) <= 0.051} for k, v in cands.items() if any(abs(v[b] - TARGET[b]) <= 0.051 for b in v)}
res["so_dinh_nghia_da_thu"] = len(cands); res["khop_ca_ba"] = hit3; res["khop_tung_bo"] = hit_any
print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])
(OUT / "stt_truoc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
