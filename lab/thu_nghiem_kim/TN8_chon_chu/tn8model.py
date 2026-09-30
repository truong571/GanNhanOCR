"""tn8model.py — bộ chọn chữ hai tầng (0 API):
  tầng 1  logit có điều kiện (softmax trong ô) trên ứng viên R(âm) ∪ {kim}(∪ {lt2}) — đặc trưng ảnh + tiên nghiệm;
  tầng 2  hồi quy logistic mức ô: P(nhãn top-1 đúng [∧ vị trí đúng]) — để chấp nhận GOLD theo ngưỡng.
Hai mô hình riêng theo phần: 'in' (kim ∈ R(âm)) và 'out' (kim ∉ R / không kim). Học trên sự thật của bộ HỌC, áp cho bộ THỬ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

SIMS = ["f_font", "f_fd", "f_head", "f_self", "f_hum", "f_vW", "f_vP", "f_humk", "f_selfk"]


def group_of(D: pd.DataFrame) -> np.ndarray:
    t, r, g = D.tier.values, D.rule.astype(str).values, D.gate_reason.astype(str).values
    lab = D.label.values
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


def cand_matrix(F: pd.DataFrame, kim_col: str = "is_kim") -> tuple[np.ndarray, list[str]]:
    """Đặc trưng ứng viên (tầng 1). F phải sắp theo i (ô liên tục)."""
    i = F.i.values
    cols, names = [], []
    first = np.r_[True, i[1:] != i[:-1]]
    gid = np.cumsum(first) - 1
    for s in SIMS:
        v = F[s].values.astype(np.float64)
        has = ~np.isnan(v)
        vv = np.where(has, v, -1.0)
        # max và max thứ hai theo ô
        o = np.lexsort((vv, gid))
        gs, vs = gid[o], vv[o]
        last = np.r_[gs[1:] != gs[:-1], True]
        mx = vs[last]
        size = np.bincount(gid)
        pos_last = np.nonzero(last)[0]
        s2 = np.where(size > 1, vs[np.maximum(pos_last - 1, 0)], -1.0)
        inv = np.empty_like(o); inv[o] = np.arange(len(o))
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
    return np.stack(cols, 1).astype(np.float32), names


class CLogit:
    """Softmax trong từng ô; mất mát = −log Σ_{ứng viên đúng} p. L2 trên w."""

    def __init__(self, l2=1e-3):
        self.l2 = l2

    def fit(self, X, i, y, iters=300):
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
    def __init__(self, l2=1e-3):
        self.l2 = l2

    def fit(self, X, y, iters=500):
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
    """Mỗi ô: top-1 (chữ, p1, y1, đặc trưng của top-1), p2."""
    G = pd.DataFrame({"i": F.i.values, "c": F.c.values, "p": p, "y": F.y.values, "y_ref": F.y_ref.values, "r": np.arange(len(F))})
    G = G.sort_values(["i", "p"], ascending=[True, False])
    first = G.groupby("i").head(1).set_index("i")
    second = G.groupby("i").nth(1).set_index("i") if len(G) else G
    T = pd.DataFrame(index=first.index)
    T["top1"] = first.c; T["p1"] = first.p; T["y1"] = first.y; T["yref1"] = first.y_ref
    T["p2"] = second.p.reindex(T.index).fillna(0.0)
    T["n_cand"] = G.groupby("i").size()
    T["any_y"] = G.groupby("i").y.max()
    X1 = Xc[first.r.values]
    for j, nm in enumerate(names):
        T["t_" + nm] = X1[:, j]
    return T
