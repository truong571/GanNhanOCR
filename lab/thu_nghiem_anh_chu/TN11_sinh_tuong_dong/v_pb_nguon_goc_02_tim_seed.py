"""v_pb_nguon_goc_02 (04/10) — PHẢN BIỆN: có PHỤC HỒI được hạt giống của bảng gốc p05 không? (người kiểm chứng trước nói "không thể").

Ý tưởng: `hash(book) % 100000` chỉ nhận 100 000 giá trị. Với mỗi hạt giống s: ô được chọn = sort(default_rng(s).choice(target_cells, 20)).
Kết quả HQC1/HQC2 của MỘT ô (Top-1, cos chữ đúng, margin) chỉ phụ thuộc ô + chữ ứng viên (nhúng glyph không phụ thuộc mẫu) nên tính
sẵn cho MỌI ô đích một lần, rồi quét 100 000 hạt giống bằng tra bảng và so với dấu vân tay trong ket_qua_full_10_bo.json
(n_cells, top1/mean_cos/mean_margin của HQC1 và HQC2). Hạt giống khớp được kiểm lại bằng hàm GỐC p05 (script 01) để thêm HQC3.
CPU, 0 API, không sửa p05 (import mô-đun để dùng lại hàm trợ giúp; OUT_DIR không dùng).

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .../v_pb_nguon_goc_02_tim_seed.py --books B34 L16 ...
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
OUT.mkdir(parents=True, exist_ok=True)
P05 = HERE / "p05_full_corpus_he_quy_chieu.py"
ORIG = REPO / "measure_out" / "_tn11" / "p05_full_corpus" / "ket_qua_full_10_bo.json"


def load_p05():
    spec = importlib.util.spec_from_file_location("p05_goc", str(P05))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def embed_chunks(enc, imgs, bs=256):
    out = []
    for i in range(0, len(imgs), bs):
        out.append(np.asarray(enc.embed(imgs[i:i + bs], norm=False), np.float32))
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def per_cell_tables(M, book, encoder, log):
    T = M.T
    D = T.load_base(book)
    E_full = M.nrm(np.load(M.EMB_DIR / f"{book}_enc.npy").astype(np.float32))
    F = pd.read_pickle(M.CAND_DIR / f"{book}.pkl")
    if "y" in F.columns and F.y.sum() > 0:
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        target_cells = list(tru.index)
        is_human = True
    else:
        lbl = D.label.values
        tgt = lbl[F.i.values]
        valid_mask = (F.c.values == tgt) & (tgt != "")
        target_cells = sorted(set(F.i.values[valid_mask]))
        F["y"] = np.where(valid_mask, 1, 0)
        is_human = False
    tset = set(target_cells)
    Ft = F[F.i.isin(tset)].copy()
    Ft["rk"] = Ft.f_vW.fillna(-9) if "f_vW" in Ft.columns else Ft.f_font.fillna(-9)
    S = Ft.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(3)
    has_truth = S.groupby("i").y.max()
    valid_cells = sorted(has_truth[has_truth == 1].index)
    S = S[S.i.isin(valid_cells)]
    log(f"[{book}] target={len(target_cells)} valid(top3 chứa đúng)={len(valid_cells)} human={is_human}")

    chars = sorted(set(S.c))
    cid = {c: j for j, c in enumerate(chars)}
    paper = M.extract_book_paper_model(book)
    raw = [M.render_font_raw(c, 128) for c in chars]
    V1 = M.nrm(embed_chunks(encoder, raw))
    V2 = M.nrm(embed_chunks(encoder, [M.blend_on_paper(r, paper) for r in raw]))
    log(f"[{book}] đã nhúng {len(chars)} chữ ứng viên (HQC1, HQC2)")

    rows = {}
    for i, g in S.groupby("i", sort=True):
        rows[i] = (list(g.c), list(g.y))
    acc1 = {}; cos1 = {}; m1 = {}; acc2 = {}; cos2 = {}; m2 = {}
    ndup = 0
    for i in valid_cells:
        cs, ys = rows[i]
        if len(set(cs)) != len(cs):
            ndup += 1
        gt = cs[[k for k, y in enumerate(ys) if y == 1][0]]
        for V, A, Cc, Mm in ((V1, acc1, cos1, m1), (V2, acc2, cos2, m2)):
            s = {c: float(E_full[i] @ V[cid[c]]) for c in cs}
            top = max(s, key=s.get)
            wrong = [c for c in cs if c != gt]
            A[i] = int(top == gt)
            Cc[i] = s[gt]
            Mm[i] = s[gt] - max(s[w] for w in wrong) if wrong else 0.0
    log(f"[{book}] ô có ứng viên trùng chữ trong top-3: {ndup}")
    return target_cells, valid_cells, (acc1, cos1, m1, acc2, cos2, m2), is_human


def build_arrays(target_cells, valid_cells, tabs):
    acc1, cos1, m1, acc2, cos2, m2 = tabs
    tarr = np.asarray(target_cells)
    mx = int(tarr.max()) + 1
    row_of = np.full(mx, -1, np.int64)
    row_of[tarr] = np.arange(len(tarr))
    vflag = np.zeros(len(tarr), bool)
    arr = {k: np.zeros(len(tarr)) for k in ("A1", "C1", "M1", "A2", "C2", "M2")}
    for i in valid_cells:
        r = row_of[i]
        vflag[r] = True
        arr["A1"][r] = acc1[i]; arr["C1"][r] = cos1[i]; arr["M1"][r] = m1[i]
        arr["A2"][r] = acc2[i]; arr["C2"][r] = cos2[i]; arr["M2"][r] = m2[i]
    return tarr, row_of, vflag, arr


def stats_for_seed(seed, tarr, row_of, vflag, arr, n_sample=20):
    size = min(n_sample, len(tarr))
    sel = np.sort(np.random.default_rng(seed).choice(tarr, size=size, replace=False))
    r = row_of[sel]
    v = vflag[r]
    rv = r[v]
    n = int(v.sum())
    if n == 0:
        return n, None, sel
    return n, dict(a1=round(float(arr["A1"][rv].mean() * 100), 1), a2=round(float(arr["A2"][rv].mean() * 100), 1),
                   c1=float(arr["C1"][rv].mean()), c2=float(arr["C2"][rv].mean()),
                   g1=float(arr["M1"][rv].mean()), g2=float(arr["M2"][rv].mean())), sel


def search(target_cells, valid_cells, tabs, orig, n_sample=20, tol=1.6e-4):
    tarr, row_of, vflag, arr = build_arrays(target_cells, valid_cells, tabs)
    n0 = orig["n_cells"]
    t1_0 = orig["hqc1_white_canvas"]; t2_0 = orig["hqc2_book_paper_canvas"]
    hits = []
    n_pass_n = 0
    size = min(n_sample, len(tarr))
    for seed in range(100000):
        sel = np.sort(np.random.default_rng(seed).choice(tarr, size=size, replace=False))
        r = row_of[sel]
        v = vflag[r]
        n = int(v.sum())
        if n != n0:
            continue
        n_pass_n += 1
        rv = r[v]
        a1 = round(float(arr["A1"][rv].mean() * 100), 1); a2 = round(float(arr["A2"][rv].mean() * 100), 1)
        if a1 != t1_0["top1"] or a2 != t2_0["top1"]:
            continue
        c1 = float(arr["C1"][rv].mean()); c2 = float(arr["C2"][rv].mean()); g1 = float(arr["M1"][rv].mean()); g2 = float(arr["M2"][rv].mean())
        ok = (abs(c1 - t1_0["mean_cos"]) <= tol and abs(c2 - t2_0["mean_cos"]) <= tol
              and abs(g1 - t1_0["mean_margin"]) <= tol and abs(g2 - t2_0["mean_margin"]) <= tol)
        hits.append({"seed": seed, "n": n, "top1": [a1, a2], "cos": [round(c1, 4), round(c2, 4)],
                     "margin": [round(g1, 4), round(g2, 4)], "khop_cos_margin": bool(ok),
                     "cells": [int(x) for x in np.sort(sel)]})
    return hits, n_pass_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    M = load_p05()
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    enc = SI.Scorers(Assets(), "cpu", M.EMB_DIR, False, lambda m: None).enc()
    orig = {r["book"]: r for r in json.loads(ORIG.read_text(encoding="utf-8"))}

    def log(s):
        print(s, flush=True)

    res = {}
    for b in a.books:
        t0 = time.time()
        tc, vc, tabs, human = per_cell_tables(M, b, enc, log)
        # tự kiểm: bảng tra phải tái hiện đúng hàm GỐC ở các hạt giống đã chạy bằng script 01 (hs0 và các tệp chay_goc_*.json)
        tarr, row_of, vflag, arr = build_arrays(tc, vc, tabs)
        selfchk = []
        for fp in sorted(OUT.glob("chay_goc_*.json")):
            for run in json.loads(fp.read_text(encoding="utf-8"))["runs"]:
                if run["book"] != b:
                    continue
                n, st, _ = stats_for_seed(run["seed"], tarr, row_of, vflag, arr)
                rr = run["res"]
                same = (st is not None and n == rr["n_cells"] and st["a1"] == rr["hqc1_white_canvas"]["top1"] and st["a2"] == rr["hqc2_book_paper_canvas"]["top1"]
                        and abs(st["c1"] - rr["hqc1_white_canvas"]["mean_cos"]) < 1.6e-4 and abs(st["c2"] - rr["hqc2_book_paper_canvas"]["mean_cos"]) < 1.6e-4
                        and abs(st["g1"] - rr["hqc1_white_canvas"]["mean_margin"]) < 1.6e-4 and abs(st["g2"] - rr["hqc2_book_paper_canvas"]["mean_margin"]) < 1.6e-4)
                selfchk.append({"tep": fp.name, "seed": run["seed"], "giong_ham_goc": bool(same)})
        log(f"[{b}] tự kiểm bảng tra vs hàm gốc: {sum(x['giong_ham_goc'] for x in selfchk)}/{len(selfchk)} lượt giống")
        hits, npn = search(tc, vc, tabs, orig[b])
        n_ok = sum(h["khop_cos_margin"] for h in hits)
        res[b] = {"is_human": human, "n_target": len(tc), "n_valid_top3": len(vc), "orig": {k: orig[b][k] for k in ("n_cells", "hqc1_white_canvas", "hqc2_book_paper_canvas", "hqc3_canonical_normalized")},
                  "seeds_cung_n": npn, "ung_vien_khop_top1": len(hits), "khop_day_du_hqc1_hqc2": n_ok, "tu_kiem": selfchk, "hits": hits, "sec": round(time.time() - t0, 1)}
        log(f"[{b}] n_goc={orig[b]['n_cells']} | hạt giống cùng n: {npn}/100000 | khớp Top-1 HQC1+HQC2: {len(hits)} | khớp thêm cos+margin: {n_ok} | seeds khớp đủ: {[h['seed'] for h in hits if h['khop_cos_margin']]} [{time.time() - t0:.0f}s]")
        (OUT / f"tim_seed_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
