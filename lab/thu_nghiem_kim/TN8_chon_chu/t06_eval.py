"""t06_eval.py — học bộ chọn chữ hai tầng trên sự thật của bộ HỌC, đo trên bộ THỬ (LOBO), 0 API.

Họ chữ viết tay: B34 -> B18 và B18 -> B34 (mô hình KHÔNG thấy nhãn người của bộ thử). Họ in: TK -> L16, L16 -> TK.
Mỗi bộ học: dự đoán ngoài-khối (5 khối trang) để chọn ngưỡng (LOBO) — không dùng điểm trong mẫu.
Ra: measure_out/_tn8/pred/<bộ thử>__from_<bộ học>.pkl (một hàng/ô: phần, top1, P, đúng chữ, đúng vị trí, ...)
    + pred/<bộ học>__oof.pkl (ngoài-khối của bộ học).
  .venv/bin/python lab/thu_nghiem_kim/TN8_chon_chu/t06_eval.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import tn8model as M  # noqa: E402

PRED = T.OUT / "pred"
WEAK_BOX = ("vdp_low", "vdp_virtual", "vdp_fallback")
GROUPS = ["GOLD", "direct_qn", "direct_crop", "direct_other", "similar", "syl", "syl_bridge", "nocontext", "lowpost",
          "syl_cropbad", "textonly", "other"]
MAX_TRAIN_CELLS = 40000
CAND_TAG = os.environ.get("TN8_TAG", "")          # "" = lượt 1; "_p2" = nguyên mẫu lượt 2 (t05b)
TRAIN_TAG = os.environ.get("TN8_TAG_TRAIN", CAND_TAG)   # bộ HỌC: lượt 2 dựng từ dự đoán ngoài-khối của chính nó


def num(s, default=np.nan):
    return pd.to_numeric(pd.Series(s), errors="coerce").fillna(default).values


def load_book(b: str, kim_mode: str = "kim", tag: str | None = None):
    """-> dict(D, F sắp theo i, Xc, names, cells, grp, part). kim_mode: 'kim' (ocr_char) | 'lt2' (STT: kim := lt2)."""
    D = T.load_base(b)
    F = pd.read_pickle(T.OUT / "cand" / f"{b}{CAND_TAG if tag is None else tag}.pkl").sort_values(["i", "c"], kind="stable").reset_index(drop=True)
    cells = pd.read_pickle(T.OUT / "cand" / f"{b}_cells.pkl")
    kcol = "is_kim"
    if kim_mode in ("lt2", "union"):                 # STT: lt2 (Nôm) đóng vai kim như Borg; union = lt1 hoặc lt2
        F = F.copy()
        F["is_kim"] = F.is_lt2.values if kim_mode == "lt2" else np.maximum(F.is_kim.values, F.is_lt2.values)
        F["is_lt2"] = 0
    Xc, names = M.cand_matrix(F, kcol)
    grp = M.group_of(D)
    kk = pd.Series(((F[kcol].values == 1) & (F.inR.values == 1)).astype(np.int8)).groupby(F.i.values).max()
    kim_inR = np.zeros(len(D), bool); kim_inR[kk.index.values] = kk.values > 0
    part = np.where(kim_inR, "in", "out").astype(object)
    part[np.isin(grp, ["notplaus", "quarantine"])] = "none"
    return dict(b=b, D=D, F=F, Xc=Xc, names=names, cells=cells, grp=grp, part=part, kim_inR=kim_inR)


def cell_X(Bk: dict, Tt: pd.DataFrame, idx: np.ndarray):
    """Đặc trưng tầng 2 cho các ô idx (Tt = top_table theo i)."""
    D, cells, grp = Bk["D"], Bk["cells"], Bk["grp"]
    t = Tt.reindex(idx)
    cols = [t.p1.values, np.log(np.maximum(t.p1.values, 1e-6)), t.p1.values - t.p2.values, np.log(t.n_cand.values)]
    names = ["p1", "lp1", "gap", "ln_cand"]
    for nm in ("is_kim", "inR", "is_lt2", "f_self", "d_self", "has_self", "ln_self", "f_font", "d_font", "f_fd", "d_fd",
               "f_head", "d_head", "f_vW", "d_vW", "f_vP", "d_vP", "has_vP", "f_hum", "d_hum", "has_hum", "lp_syl", "lp_glob"):
        cols.append(t["t_" + nm].values); names.append("t_" + nm)
    w = cells.crop_w.values[idx]; h = cells.crop_h.values[idx]
    mw = np.median(cells.crop_w.values[cells.crop_ok.values > 0]); mh = np.median(cells.crop_h.values[cells.crop_ok.values > 0])
    cols += [np.log(np.maximum(w, 1) / mw), np.log(np.maximum(h, 1) / mh), cells.ink.values[idx], cells.crop_ok.values[idx]]
    names += ["lw", "lh", "ink", "crop_ok"]
    bs = D.box_source.values[idx].astype(str)
    cols.append(np.array([x in WEAK_BOX or "ink_cut" in x or "detector_low" in x for x in bs], float)); names.append("box_weak")
    nq = num(D.n_qn.values[idx], 0); no = num(D.n_ocr.values[idx], 0); nd = num(D.n_det.values[idx], 0)
    cols += [np.minimum(np.abs(no - nq), 5), np.minimum(np.abs(nd - nq), 5)]; names += ["d_ocr_qn", "d_det_qn"]
    for gname in GROUPS:
        cols.append((grp[idx] == gname).astype(float)); names.append("g_" + gname)
    X = np.stack(cols, 1).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    return X, names


def target(Bk, Tt, idx, pos_needed):
    D = Bk["D"]
    t = Tt.reindex(idx)
    y = t.y1.fillna(0).values.astype(bool)
    if pos_needed:
        y = y & D.pos_ok.values[idx]
    return y


def fit_models(train: list[dict], part: str, blocks_excl: dict | None = None, seed=0):
    """Tầng 1 + tầng 2 trên các bộ học (bỏ các khối trong blocks_excl[b] nếu có). Trả (m1, m2)."""
    rng = np.random.default_rng(seed)
    X1s, i1s, y1s = [], [], []
    off = 0
    for Bk in train:
        D, F = Bk["D"], Bk["F"]
        blk = Bk["cells"].blk.values
        ok = (Bk["part"] == part) & (D.gt_char.values != "")
        if blocks_excl and Bk["b"] in blocks_excl:
            ok &= ~np.isin(blk, blocks_excl[Bk["b"]])
        anyy = F.groupby("i").y.max().reindex(np.arange(len(D))).fillna(0).values > 0
        ncand = F.groupby("i").size().reindex(np.arange(len(D))).fillna(0).values
        cand_cells = np.nonzero(ok & anyy & (ncand >= 2))[0]
        if len(cand_cells) > MAX_TRAIN_CELLS:
            cand_cells = np.sort(rng.choice(cand_cells, MAX_TRAIN_CELLS, replace=False))
        sel = np.isin(F.i.values, cand_cells)
        X1s.append(Bk["Xc"][sel]); i1s.append(F.i.values[sel] + off); y1s.append(F.y.values[sel])
        off += len(D) + 1
    m1 = M.CLogit().fit(np.concatenate(X1s), np.concatenate(i1s), np.concatenate(y1s))
    X2s, y2s = [], []
    for Bk in train:
        D, F = Bk["D"], Bk["F"]
        blk = Bk["cells"].blk.values
        p = m1.predict(Bk["Xc"], F.i.values)
        Tt = M.top_table(F, p, Bk["Xc"], Bk["names"])
        pos_needed = Bk["b"] in ("L16", "TK")
        ok = (Bk["part"] == part) & (D.gt_char.values != "")
        if blocks_excl and Bk["b"] in blocks_excl:
            ok &= ~np.isin(blk, blocks_excl[Bk["b"]])
        idx = np.nonzero(ok & np.isin(np.arange(len(D)), Tt.index.values))[0]
        X2, n2 = cell_X(Bk, Tt, idx)
        X2s.append(X2); y2s.append(target(Bk, Tt, idx, pos_needed))
    m2 = M.Logit().fit(np.concatenate(X2s), np.concatenate(y2s))
    return m1, m2


def predict(Bk: dict, m1, m2, part: str, rows: np.ndarray | None = None) -> pd.DataFrame:
    D, F = Bk["D"], Bk["F"]
    p = m1.predict(Bk["Xc"], F.i.values)
    Tt = M.top_table(F, p, Bk["Xc"], Bk["names"])
    idx = np.nonzero(Bk["part"] == part)[0]
    idx = idx[np.isin(idx, Tt.index.values)]
    if rows is not None:
        idx = idx[np.isin(idx, rows)]
    X2, _ = cell_X(Bk, Tt, idx)
    P = m2.predict(X2)
    t = Tt.reindex(idx)
    return pd.DataFrame(dict(i=idx, part=part, top1=t.top1.values, p1=t.p1.values, P=P, y1=t.y1.values,
                             yref1=t.yref1.values, any_y=t.any_y.values))


def assemble(Bk: dict, preds: list[pd.DataFrame]) -> pd.DataFrame:
    D = Bk["D"]
    out = pd.DataFrame(dict(i=np.arange(len(D)), page=D.page.values, grp=Bk["grp"], part=Bk["part"], tier=D.tier.values,
                            label=D.label.values, kim=D.ocr_char.values, gtc=D.gt_char.values, ref=D.ref.values,
                            pos_known=D.pos_known.values, pos_ok=D.pos_ok.values, blk=Bk["cells"].blk.values,
                            gate=D.gate_reason.astype(str).str.split(":").str[0].values))
    P = pd.concat(preds).set_index("i") if preds else pd.DataFrame()
    for c in ("top1", "p1", "P", "y1", "yref1", "any_y"):
        out[c] = P[c].reindex(out.i).values if len(P) else np.nan
    C = T.lex()
    out["lab_ok"] = [bool(l) and bool(g) and C.var_eq_plus(l, g) for l, g in zip(out.label, out.gtc)]
    return out


def run_pair(train_codes: list[str], test_code: str, kim_mode_test="kim", tag=None):
    t0 = time.time()
    train = [load_book(b, tag=TRAIN_TAG) for b in train_codes]
    test = load_book(test_code, kim_mode_test)
    preds_test, preds_oof = [], {b: [] for b in train_codes}
    for part in ("in", "out"):
        m1, m2 = fit_models(train, part)
        preds_test.append(predict(test, m1, m2, part))
        w = dict(zip(m2_names(test), m2.w[:-1] / m2.sd))
        print(f"  [{test_code} <- {'+'.join(train_codes)}] phần {part}: tầng1 w(top) "
              + ", ".join(f"{n}={v:+.2f}" for n, v in sorted(zip(train[0]['names'], m1.w / m1.sd), key=lambda kv: -abs(kv[1]))[:6]),
              flush=True)
        # ngoài-khối trên bộ học (chọn ngưỡng LOBO)
        for k in range(5):
            excl = {b: [k] for b in train_codes}
            m1k, m2k = fit_models(train, part, blocks_excl=excl)
            for Bk in train:
                rows = np.nonzero(Bk["cells"].blk.values == k)[0]
                preds_oof[Bk["b"]].append(predict(Bk, m1k, m2k, part, rows))
    PRED.mkdir(parents=True, exist_ok=True)
    tag = tag or f"{test_code}{CAND_TAG}__from_{'+'.join(train_codes)}" + ("" if kim_mode_test == "kim" else f"__{kim_mode_test}")
    assemble(test, preds_test).to_pickle(PRED / f"{tag}.pkl")
    for Bk in train:
        assemble(Bk, preds_oof[Bk["b"]]).to_pickle(PRED / f"{Bk['b']}{CAND_TAG}__oof_{'+'.join(train_codes)}.pkl")
    print(f"[t06] {tag} xong [{time.time() - t0:.0f}s]", flush=True)


def m2_names(Bk):
    return []


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["B34>B18", "B18>B34", "TK>L16", "L16>TK"]
    for j in jobs:
        tr, te = j.split(">")
        mode = "kim"
        if ":" in te:
            te, mode = te.split(":")
        run_pair(tr.split("+"), te, mode)
