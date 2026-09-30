"""model.py — bộ chọn chữ hai tầng (bản chép lab/thu_nghiem_kim/TN8_chon_chu/tn8model.py + t06_eval.cell_X, chỉ đổi nơi đặt).

  tầng 1  logit có điều kiện (softmax trong ô) trên ứng viên R(âm) ∪ {kim}(∪ {lt2}) — 35 đặc trưng ứng viên;
  tầng 2  hồi quy logistic mức ô — 46 đặc trưng — P(nhãn top-1 đúng [∧ khe đúng ở sách in]).
Hai mô hình theo phần: 'in' (kim ∈ R(âm)) / 'out' (kim ∉ R / không kim). Tham số lưu trong models/chon_chu/chooser_<tên>.npz.
Mọi phép tính numpy/pandas (tầng 1 học bằng torch LBFGS — chỉ ở export_assets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIMS = ["f_font", "f_fd", "f_head", "f_self", "f_hum", "f_vW", "f_vP", "f_humk", "f_selfk"]
KIM_COLS = ("is_kim", "kim_inR", "is_lt2")
WEAK_BOX = ("vdp_low", "vdp_virtual", "vdp_fallback")
GROUPS = ["GOLD", "direct_qn", "direct_crop", "direct_other", "similar", "syl", "syl_bridge", "nocontext", "lowpost",
          "syl_cropbad", "textonly", "other"]
CELL_T_FEATS = ("is_kim", "inR", "is_lt2", "f_self", "d_self", "has_self", "ln_self", "f_font", "d_font", "f_fd", "d_fd",
                "f_head", "d_head", "f_vW", "d_vW", "f_vP", "d_vP", "has_vP", "f_hum", "d_hum", "has_hum", "lp_syl", "lp_glob")


def _s(v) -> str:
    return "" if v is None else str(v)


def group_of(D: pd.DataFrame) -> np.ndarray:
    """Nhóm ô theo (tier, rule, gate_reason) — == tn8model.group_of."""
    t = D.tier.astype(str).values
    r = D.rule.astype(str).values
    g = (D.gate_reason.astype(str) if "gate_reason" in D.columns else pd.Series([""] * len(D))).values
    lab = D.label.astype(str).values
    out = np.array(["other"] * len(D), dtype=object)
    direct = np.array([x.startswith("s1_inter_s2_direct") for x in r]) & (lab != "")
    out[direct] = "direct_other"
    out[direct & np.array([x.startswith("qn_count_unfixed") for x in g])] = "direct_qn"
    out[direct & np.array([x.startswith("crop_bad") for x in g])] = "direct_crop"
    out[np.array([x.startswith("s1_inter_s2_similar") for x in r])] = "similar"
    out[t == "SYLLABLE"] = "syl"
    out[(t == "SYLLABLE") & np.array([x.startswith("s1_inter_s2") for x in r])] = "syl_bridge"
    out[(t == "REVIEW") & (r == "no_context")] = "nocontext"
    out[(t == "REVIEW") & (r == "low_posterior")] = "lowpost"
    out[(t == "REVIEW") & np.array([x.startswith("syl_ctx") for x in r])] = "syl_cropbad"
    out[(t == "REVIEW") & (r == "not_plausible")] = "notplaus"
    out[t == "GOLD_text_only"] = "textonly"
    out[t == "QUARANTINE"] = "quarantine"
    out[t == "GOLD"] = "GOLD"
    return out


def cand_matrix(F: pd.DataFrame, kim_col: str = "is_kim", with_kim: bool = True) -> tuple[np.ndarray, list[str]]:
    """35 đặc trưng ứng viên (tầng 1). F sắp theo i (ô liên tục). with_kim=False -> bỏ 3 cột kim (bộ chọn CHỈ ẢNH, 32)."""
    i = F.i.values
    cols, names = [], []
    first = np.r_[True, i[1:] != i[:-1]]
    gid = np.cumsum(first) - 1
    for s in SIMS:
        v = F[s].values.astype(np.float64)
        has = ~np.isnan(v)
        vv = np.where(has, v, -1.0)
        o = np.lexsort((vv, gid))
        gs, vs = gid[o], vv[o]
        last = np.r_[gs[1:] != gs[:-1], True]
        mx = vs[last]
        size = np.bincount(gid)
        pos_last = np.nonzero(last)[0]
        s2 = np.where(size > 1, vs[np.maximum(pos_last - 1, 0)], -1.0)
        mxc, s2c = mx[gid], s2[gid]
        d = vv - np.where(vv >= mxc, s2c, mxc)
        d = np.where(has, np.clip(d, -0.5, 0.5), 0.0)
        cmin = pd.Series(np.where(has, v, np.inf)).groupby(gid).transform("min").values
        v0 = np.where(has, v, np.where(np.isfinite(cmin), cmin, 0.0))
        cols += [v0, d, has.astype(float)]; names += [s, "d_" + s[2:], "has_" + s[2:]]
    kim = F[kim_col].values.astype(float)
    inR = F.inR.values.astype(float)
    cols += [F.lp_syl.values, F.lp_glob.values, np.log1p(F.n_self.values), np.log1p(F.n_hum.values), kim, kim * inR,
             F.is_lt2.values.astype(float), inR]
    names += ["lp_syl", "lp_glob", "ln_self", "ln_hum", "is_kim", "kim_inR", "is_lt2", "inR"]
    X = np.stack(cols, 1).astype(np.float32)
    if not with_kim:
        keep = [j for j, n in enumerate(names) if n not in KIM_COLS]
        return X[:, keep], [names[j] for j in keep]
    return X, names


class CLogit:
    """Softmax trong từng ô; mất mát = −log Σ_{ứng viên đúng} p; L2 trên w. predict: numpy."""

    def __init__(self, l2=1e-3, mu=None, sd=None, w=None):
        self.l2, self.mu, self.sd, self.w = l2, mu, sd, w

    def fit(self, X, i, y, iters=300):
        import torch
        seg = pd.factorize(i)[0]
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float64)
        st = torch.tensor(seg); yt = torch.tensor(y.astype(bool))
        ns = int(seg.max()) + 1
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xt @ w
            mx = torch.full((ns,), -1e30, dtype=torch.float64).scatter_reduce(0, st, z, "amax")
            e = torch.exp(z - mx[st])
            den = torch.zeros(ns, dtype=torch.float64).index_add(0, st, e)
            num = torch.zeros(ns, dtype=torch.float64).index_add(0, st, e * yt)
            loss = -(torch.log(num + 1e-300) - torch.log(den)).mean() + self.l2 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.w = w.detach().numpy()
        return self

    def predict(self, X, i):
        seg = pd.factorize(i)[0]
        z = ((X - self.mu) / self.sd) @ self.w
        mx = pd.Series(z).groupby(seg).transform("max").values
        e = np.exp(z - mx)
        return e / pd.Series(e).groupby(seg).transform("sum").values


class Logit:
    def __init__(self, l2=1e-3, mu=None, sd=None, w=None):
        self.l2, self.mu, self.sd, self.w = l2, mu, sd, w

    def fit(self, X, y, iters=500):
        import torch
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xt = torch.tensor(np.c_[(X - self.mu) / self.sd, np.ones(len(X))], dtype=torch.float64)
        yt = torch.tensor(y.astype(float))
        w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xt @ w
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + self.l2 * (w[:-1] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.w = w.detach().numpy()
        return self

    def predict(self, X):
        z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))] @ self.w
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def top_table(F: pd.DataFrame, p: np.ndarray, Xc: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Mỗi ô: top-1 (chữ, p1, đặc trưng của top-1), p2, số ứng viên. (== tn8model.top_table, không cột sự thật.)"""
    G = pd.DataFrame({"i": F.i.values, "c": F.c.values, "p": p, "r": np.arange(len(F))})
    G = G.sort_values(["i", "p"], ascending=[True, False])
    first = G.groupby("i").head(1).set_index("i")
    second = G.groupby("i").nth(1).set_index("i") if len(G) else G
    T = pd.DataFrame(index=first.index)
    T["top1"] = first.c; T["p1"] = first.p
    T["p2"] = second.p.reindex(T.index).fillna(0.0)
    T["n_cand"] = G.groupby("i").size()
    X1 = Xc[first.r.values]
    for j, nm in enumerate(names):
        T["t_" + nm] = X1[:, j]
    return T


def num(s, default=np.nan):
    return pd.to_numeric(pd.Series(s), errors="coerce").fillna(default).values


def cell_X(D: pd.DataFrame, cells: pd.DataFrame, grp: np.ndarray, Tt: pd.DataFrame, idx: np.ndarray):
    """46 đặc trưng tầng 2 cho các ô idx (== t06_eval.cell_X)."""
    t = Tt.reindex(idx)
    cols = [t.p1.values, np.log(np.maximum(t.p1.values, 1e-6)), t.p1.values - t.p2.values, np.log(t.n_cand.values)]
    names = ["p1", "lp1", "gap", "ln_cand"]
    for nm in CELL_T_FEATS:
        cols.append(t["t_" + nm].values); names.append("t_" + nm)
    w = cells.crop_w.values[idx]; h = cells.crop_h.values[idx]
    okc = cells.crop_ok.values > 0
    mw = np.median(cells.crop_w.values[okc]); mh = np.median(cells.crop_h.values[okc])
    cols += [np.log(np.maximum(w, 1) / mw), np.log(np.maximum(h, 1) / mh), cells.ink.values[idx], cells.crop_ok.values[idx]]
    names += ["lw", "lh", "ink", "crop_ok"]
    bs = D.box_source.astype(str).values[idx]
    cols.append(np.array([x in WEAK_BOX or "ink_cut" in x or "detector_low" in x for x in bs], float)); names.append("box_weak")
    nq = num(D.n_qn.values[idx], 0); no = num(D.n_ocr.values[idx], 0); nd = num(D.n_det.values[idx], 0)
    cols += [np.minimum(np.abs(no - nq), 5), np.minimum(np.abs(nd - nq), 5)]; names += ["d_ocr_qn", "d_det_qn"]
    for gname in GROUPS:
        cols.append((grp[idx] == gname).astype(float)); names.append("g_" + gname)
    X = np.nan_to_num(np.stack(cols, 1).astype(np.float64), nan=0.0)
    return X, names


# ------------------------------------------------------------------------------------------------ lưu / nạp tham số
def pack(models: dict, meta: dict) -> dict:
    """models = {'in': (m1, m2), 'out': (m1, m2)} hoặc {'vis': m1} -> dict mảng cho np.savez."""
    z = {"meta_json": np.array(__import__("json").dumps(meta, ensure_ascii=False))}
    for part, mm in models.items():
        seq = mm if isinstance(mm, tuple) else (mm,)
        for k, m in zip(("m1", "m2"), seq):
            for a in ("mu", "sd", "w"):
                z[f"{part}_{k}_{a}"] = np.asarray(getattr(m, a), np.float64)
    return z


def unpack(z) -> tuple[dict, dict]:
    import json
    meta = json.loads(str(z["meta_json"]))
    out = {}
    for part in ("in", "out", "vis"):
        if f"{part}_m1_w" not in z:
            continue
        m1 = CLogit(mu=z[f"{part}_m1_mu"], sd=z[f"{part}_m1_sd"], w=z[f"{part}_m1_w"])
        if f"{part}_m2_w" in z:
            m2 = Logit(mu=z[f"{part}_m2_mu"], sd=z[f"{part}_m2_sd"], w=z[f"{part}_m2_w"])
            out[part] = (m1, m2)
        else:
            out[part] = m1
    return out, meta
