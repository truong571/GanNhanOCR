"""v_pb_nguon_goc_13 (04/10) — PHẢN BIỆN: người kiểm chứng chỉ đo HQC3 với crop cắt lại SAI ngữ cảnh (p05:212). HQC3 với crop ĐÚNG ngữ cảnh có giúp không?

Mẫu lớn (N ô hợp lệ/bộ, ngẫu nhiên cố định seed 12345, giao thức p05: top-3 theo f_vW, đúng nếu top-1 == chữ y==1 đầu tiên):
  HQC1  = E_full(crop sản xuất) · glyph nền trắng (norm=False)            [như p05:259]
  HQC2  = E_full · glyph nhúng patch giấy (norm=False)                     [như p05:260]
  HQC3f = nhị phân hoá(crop cắt với recs = MỌI ô trên trang) · glyph nhị phân hoá   [HQC3 nhưng crop khớp sản xuất; p05 dùng recs chỉ ô mẫu]
  + tham chiếu trong cùng top-3: f_font sản xuất, f_vW hạng 1.
McNemar chính xác theo ô. CPU, 0 API. Hàm trợ giúp lấy từ p05 (import).
Ra: measure_out/_tn11/verify/pb_nguon_goc/hqc3_dung_ngu_canh_<tag>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
spec = importlib.util.spec_from_file_location("p05_goc", str(HERE / "p05_full_corpus_he_quy_chieu.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
T = M.T


def emb(enc, imgs, bs=256):
    out = [np.asarray(enc.embed(imgs[i:i + bs], norm=False), np.float32) for i in range(0, len(imgs), bs)]
    return M.nrm(np.concatenate(out))


def mc(x, y):
    x = np.asarray(x) == 1; y = np.asarray(y) == 1
    b = int((x & ~y).sum()); c = int((~x & y).sum())
    p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return {"diff_pct": round(float((x.mean() - y.mean()) * 100), 1), "a_dung_b_sai": b, "a_sai_b_dung": c, "p": round(p, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from t02_embed import page_crops
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    enc = SI.Scorers(Assets(), "cpu", M.EMB_DIR, False, lambda m: None).enc()
    res = {}
    for book in a.books:
        t0 = time.time()
        D = T.load_base(book)
        E = M.nrm(np.load(M.EMB_DIR / f"{book}_enc.npy").astype(np.float32))
        F = pd.read_pickle(M.CAND_DIR / f"{book}.pkl")
        if "y" in F.columns and F.y.sum() > 0:
            tru = F[F.y == 1].drop_duplicates("i").set_index("i"); target = list(tru.index); human = True
        else:
            lbl = D.label.values; tgt = lbl[F.i.values]; vm = (F.c.values == tgt) & (tgt != "")
            target = sorted(set(F.i.values[vm])); F["y"] = np.where(vm, 1, 0); human = False
        Ft = F[F.i.isin(set(target))].copy()
        Ft["rk"] = Ft.f_vW.fillna(-9) if "f_vW" in Ft.columns else Ft.f_font.fillna(-9)
        S_all = Ft.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(3)
        ht = S_all.groupby("i").y.max()
        valid = np.array(sorted(ht[ht == 1].index))
        rng = np.random.default_rng(12345)
        sample = np.sort(rng.choice(valid, size=min(a.n, len(valid)), replace=False))
        S = S_all[S_all.i.isin(set(sample))]
        paper = M.extract_book_paper_model(book)
        chars = sorted(set(S.c))
        raw = [M.render_font_raw(c, 128) for c in chars]
        V1 = dict(zip(chars, emb(enc, raw)))
        V2 = dict(zip(chars, emb(enc, [M.blend_on_paper(r, paper) for r in raw])))
        V3 = dict(zip(chars, emb(enc, [M.binarize_canonical(r) for r in raw])))
        cfg = T.BOOKS[book]; prep = T.REPO / cfg["prep"]
        sset = set(int(x) for x in sample)
        crops = {}
        for pg in sorted(set(D.loc[sample, "page"])):
            allp = D[D.page == pg]
            recs = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox)]   # MỌI ô trên trang (đúng t02_embed)
            for ii, g, _ in page_crops((book, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs)):
                if ii in sset and g is not None:
                    crops[ii] = g
        ids = [int(i) for i in sample if int(i) in crops]
        C3 = dict(zip(ids, emb(enc, [M.binarize_canonical(cv2.resize(crops[i], (128, 128))) for i in ids])))
        rows = []
        grp = {i: g for i, g in S.groupby("i")}
        for i in ids:
            sub = grp[i]
            cs = list(sub.c); ys = list(sub.y)
            gt = cs[[k for k, y in enumerate(ys) if y == 1][0]]

            def top(sc):
                t = max(sc, key=sc.get)
                return int(t == gt)

            s1 = {c: float(E[i] @ V1[c]) for c in cs}; s2 = {c: float(E[i] @ V2[c]) for c in cs}; s3 = {c: float(C3[i] @ V3[c]) for c in cs}
            fF = sub.f_font.fillna(-9).to_numpy()
            rows.append({"i": i, "a1": top(s1), "a2": top(s2), "a3f": top(s3), "aF": int(sub.iloc[int(np.argmax(fF))].y == 1), "aV": int(sub.iloc[0].y == 1)})
        df = pd.DataFrame(rows)
        r = {"human": human, "n_valid_pop": int(len(valid)), "n_mau": int(len(df)), "top1": {k: round(float(df[k].mean()) * 100, 1) for k in ("a1", "a2", "a3f", "aF", "aV")},
             "HQC3f_vs_HQC1": mc(df.a3f, df.a1), "HQC3f_vs_HQC2": mc(df.a3f, df.a2), "HQC2_vs_HQC1": mc(df.a2, df.a1), "HQC3f_vs_fFont": mc(df.a3f, df.aF), "HQC1_vs_fFont": mc(df.a1, df.aF),
             "sec": round(time.time() - t0, 1)}
        res[book] = r
        print(book, json.dumps(r, ensure_ascii=False), flush=True)
        (OUT / f"hqc3_dung_ngu_canh_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
