"""v_pb_nguon_goc_11 (04/10) — PHẢN BIỆN: crop mà p05 dùng cho HQC3 (page_crops chỉ với ô hợp lệ của mẫu) có KHÁC crop sản xuất (E_full) không, trên CHÍNH MẪU GỐC?

Mẫu gốc = ô chọn bởi hạt giống đã phục hồi (B34 40704, L16 87120). Với mỗi ô hợp lệ (đúng như p05:195-201):
  - crop_p05  = page_crops(recs = chỉ các ô hợp lệ trên trang)      (p05:209-215)
  - crop_full = page_crops(recs = MỌI ô trên trang)                 (cách t02_embed dựng E_full)
So: crop giống từng điểm ảnh? cos(E_full[ô], embed(crop_p05, norm mặc định)) và cos(E_full[ô], embed(crop_full)).
Ra: measure_out/_tn11/verify/pb_nguon_goc/crop_ngu_canh.json   (CPU, 0 API)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
spec = importlib.util.spec_from_file_location("p05_goc", str(HERE / "p05_full_corpus_he_quy_chieu.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
T = M.T
from t02_embed import page_crops  # noqa: E402
from pipeline.gold_exact import signals_img as SI  # noqa: E402
from pipeline.gold_exact.common import Assets  # noqa: E402

enc = SI.Scorers(Assets(), "cpu", M.EMB_DIR, False, lambda m: None).enc()
res = {}
for book, seed in (("B34", 40704), ("L16", 87120)):
    D = T.load_base(book)
    E = M.nrm(np.load(M.EMB_DIR / f"{book}_enc.npy").astype(np.float32))
    F = pd.read_pickle(M.CAND_DIR / f"{book}.pkl")
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    target = list(tru.index)
    sel = np.sort(np.random.default_rng(seed).choice(target, size=20, replace=False))
    S = F[F.i.isin(sel)].copy()
    S["rk"] = S.f_vW.fillna(-9)
    S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(3)
    ht = S.groupby("i").y.max()
    valid = sorted(ht[ht == 1].index)
    cfg = T.BOOKS[book]
    prep = T.REPO / cfg["prep"]
    vset = set(valid)
    same = 0
    cos_p05, cos_full = [], []
    n = 0
    for pg in sorted(set(D.loc[valid, "page"])):
        allp = D[D.page == pg]
        r1 = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox) if ii in vset]
        r2 = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox)]
        c1 = {ii: g for ii, g, _ in page_crops((book, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", r1))}
        c2 = {ii: g for ii, g, _ in page_crops((book, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", r2))}
        ids = [r[0] for r in r1 if c1.get(r[0]) is not None and c2.get(r[0]) is not None]
        for i in ids:
            a, b = c1[i], c2[i]
            same += int(a.shape == b.shape and int(np.abs(a.astype(int) - b.astype(int)).max()) == 0)
            va = M.nrm(np.asarray(enc.embed([a]), np.float32))[0]
            vb = M.nrm(np.asarray(enc.embed([b]), np.float32))[0]
            cos_p05.append(float(E[i] @ va)); cos_full.append(float(E[i] @ vb)); n += 1
    res[book] = {"seed": seed, "n_valid": len(valid), "n_so_sanh": n, "crop_giong_het_tung_diem": same,
                 "cos_E_vs_crop_p05_tb": round(float(np.mean(cos_p05)), 4), "cos_E_vs_crop_p05_min": round(float(np.min(cos_p05)), 4),
                 "cos_E_vs_crop_full_tb": round(float(np.mean(cos_full)), 4), "cos_E_vs_crop_full_min": round(float(np.min(cos_full)), 4)}
    print(book, json.dumps(res[book], ensure_ascii=False), flush=True)
(OUT / "crop_ngu_canh.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
