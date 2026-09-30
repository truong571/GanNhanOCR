"""t07_policy.py — chính sách GOLD mới trên bộ có sự thật: tỉ lệ GOLD (trên MỌI ô) và độ đúng (hai vế / chữ), ngưỡng chọn
(a) LOBO: trên dự đoán NGOÀI-KHỐI của bộ học; (b) CV cụm trang: 5 khối trang của chính bộ thử (chọn trên 4, báo khối còn lại).
CI 95 % bootstrap cụm trang. Ablation đòn bẩy L1 (crop_bad), L2 (qn_count_unfixed), L2b (kim∈R khác), L4 (bộ chọn, kim∉R).
0 API. Ra: measure_out/_tn8/policy_<bộ thử>.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402

GRID = np.round(np.r_[np.arange(0.05, 0.95, 0.025), np.arange(0.95, 0.996, 0.005)], 4)
# cổng VĂN BẢN không bao giờ gỡ bằng ảnh: dị bản chống (cross_similar), âm đã sửa dấu (am_sua_dau), chữ cầu gần hình
EXCL_GATES = ("cross_similar", "am_sua_dau", "bridge_similar")
IN_GROUPS = ("direct_qn", "direct_crop", "direct_other", "similar", "textonly", "other")
L_OF = {"direct_crop": "L1", "direct_qn": "L2", "direct_other": "L2b", "similar": "L2b", "textonly": "L2b", "other": "L2b",
        "syl": "L4", "syl_bridge": "L4", "nocontext": "L4", "lowpost": "L4", "syl_cropbad": "L4"}


def correct_cols(o: pd.DataFrame, new: np.ndarray):
    """(đúng chữ, đúng hai vế, có gt, có vị trí) theo ô cho tập = GOLD hiện tại ∪ new (nhãn mới = top1)."""
    gold = (o.tier == "GOLD").values
    txt = np.where(gold, o.lab_ok.values, o.y1.fillna(0).values.astype(bool))
    has = o.gtc.values != ""
    both = txt & o.pos_ok.values
    return txt, both, has, o.pos_known.values & has


def metrics(o: pd.DataFrame, acc: np.ndarray, B=True) -> dict:
    txt, both, has, pk = correct_cols(o, acc)
    N = len(o)
    r = dict(n=int(acc.sum()), share=round(acc.sum() / N, 4))
    m = acc & pk
    if m.sum():
        pt, lo, hi = T.boot_ci_pages(both[m], o.page.values[m]) if B else (both[m].mean(), np.nan, np.nan)
        r.update(both=round(pt, 4), both_ci=[round(lo, 4), round(hi, 4)], n_both=int(m.sum()))
    m = acc & has
    if m.sum():
        pt, lo, hi = T.boot_ci_pages(txt[m], o.page.values[m]) if B else (txt[m].mean(), np.nan, np.nan)
        r.update(text=round(pt, 4), text_ci=[round(lo, 4), round(hi, 4)], n_text=int(m.sum()))
    return r


def added_prec(o: pd.DataFrame, sel: np.ndarray) -> tuple[float, int]:
    """Độ đúng (hai vế nếu có vị trí, không thì chữ) của các ô MỚI sel."""
    pk = o.pos_known.values & (o.gtc.values != "")
    y = o.y1.fillna(0).values.astype(bool) & o.pos_ok.values
    m = sel & pk
    if m.sum() >= 30:
        return float(y[m].mean()), int(m.sum())
    has = o.gtc.values != ""
    m = sel & has
    return (float(o.y1.fillna(0).values.astype(bool)[m].mean()) if m.sum() else np.nan), int(m.sum())


def pick_tau(o: pd.DataFrame, part: str, target: float, groups=None, mode: str = "marginal") -> float:
    """Ngưỡng trên tập chọn o cho ô MỚI (không GOLD) của phần `part`.
    marginal (mặc định): xếp ô theo P giảm dần; τ = P tại vị trí cuối cùng mà độ đúng CỬA SỔ W ô liền trước (theo P) còn
    ≥ target — tức ô biên (ô tệ nhất được nhận) vẫn đúng ≥ target, không để nhóm xấu nấp sau nhóm tốt.
    average: τ nhỏ nhất mà độ đúng TRUNG BÌNH của mọi ô mới ≥ target."""
    base = (o.tier.values != "GOLD") & (o.part.values == part) & ~np.isnan(o.P.values)
    if "gate" in o:
        base &= ~np.isin(o.gate.values, EXCL_GATES)
    if groups is not None:
        base &= np.isin(o.grp.values, groups)
    if mode == "average":
        for t in GRID:
            p, n = added_prec(o, base & (o.P.values >= t))
            if n >= 30 and p >= target:
                return float(t)
        return 1.01
    pk = o.pos_known.values & (o.gtc.values != "")
    has = o.gtc.values != ""
    use_pos = (base & pk).sum() >= 200
    ev = base & (pk if use_pos else has)
    y = o.y1.fillna(0).values.astype(bool) & (o.pos_ok.values if use_pos else True)
    P = o.P.values[ev]; yy = y[ev].astype(float)
    if len(P) < 30:
        return 1.01
    order = np.argsort(-P, kind="stable")
    P, yy = P[order], yy[order]
    W = int(max(50, min(300, 0.05 * len(P))))
    cs = np.r_[0, np.cumsum(yy)]
    k = np.arange(1, len(P) + 1)
    lo = np.maximum(0, k - W)
    win = (cs[k] - cs[lo]) / (k - lo)
    good = np.nonzero((win >= target) & (k >= min(W, len(P))))[0]
    if not len(good):
        return 1.01
    return float(P[good[-1]])


def accept(o: pd.DataFrame, tau_in: float, tau_out: float, groups=None) -> np.ndarray:
    ng = (o.tier.values != "GOLD") & ~np.isin(o.gate.values if "gate" in o else np.array([""] * len(o)), EXCL_GATES)
    P = np.nan_to_num(o.P.values, nan=-1)
    a = ng & (((o.part.values == "in") & (P >= tau_in)) | ((o.part.values == "out") & (P >= tau_out)))
    if groups is not None:
        a &= np.isin(o.grp.values, groups)
    return a


def evaluate(test: str, train: str, target=0.92) -> dict:
    o = pd.read_pickle(T.OUT / "pred" / f"{test}__from_{train}.pkl")
    oof = pd.concat([pd.read_pickle(T.OUT / "pred" / f"{b}__oof_{train}.pkl") for b in train.split("+")])
    gold = o.tier.values == "GOLD"
    res = dict(test=test, train=train, target_sel=target, N=len(o), current=metrics(o, gold))
    # ---- (a) LOBO: ngưỡng từ ngoài-khối bộ học
    ti, to = pick_tau(oof, "in", target), pick_tau(oof, "out", target)
    res["lobo"] = dict(tau_in=ti, tau_out=to, all=metrics(o, gold | accept(o, ti, to)))
    # ---- (b) CV cụm trang trên bộ thử
    acc = np.zeros(len(o), bool); taus = []
    for k in range(5):
        tr = o[o.blk != k]
        a, b_ = pick_tau(tr, "in", target), pick_tau(tr, "out", target)
        taus.append((a, b_))
        acc |= (o.blk.values == k) & accept(o, a, b_)
    res["cv"] = dict(taus=taus, all=metrics(o, gold | acc))
    # ---- ablation theo đòn bẩy (ngưỡng LOBO)
    abl = {}
    for L in ("L1", "L2", "L2b", "L4"):
        gs = [g for g, l in L_OF.items() if l == L]
        m = np.isin(o.grp.values, gs) & ~gold
        abl[L] = dict(n_nhom=int(m.sum()),
                      tat_ca_nhan_kim=metrics(o, gold | (m & (o.part.values == "in"))) if L != "L4" else None,
                      bo_chon=metrics(o, gold | accept(o, ti, to, gs)),
                      them_prec=added_prec(o, accept(o, ti, to, gs)))
    res["ablation_lobo"] = abl
    # ---- đường cong (tham khảo) theo phần
    curves = {}
    for part in ("in", "out"):
        rows = []
        base = ~gold & (o.part.values == part) & ~np.isnan(o.P.values)
        for t in GRID[::2]:
            s = base & (o.P.values >= t)
            p, n = added_prec(o, s)
            rows.append(dict(tau=float(t), n_add=int(s.sum()), prec_add=None if np.isnan(p) else round(p, 4), n_eval=n))
        curves[part] = rows
    res["curves"] = curves
    # ---- nhóm: số ô và tỉ lệ nhận (LOBO)
    a = accept(o, ti, to)
    grp = {}
    for g_, gg in o.groupby("grp"):
        idx = gg.index.values
        grp[g_] = dict(n=int(len(gg)), nhan=int(a[idx].sum()), prec_add=added_prec(o, a & o.index.isin(idx))[0])
    res["groups_lobo"] = grp
    return res


def frontier(o: pd.DataFrame, target: float, metric: str = "both") -> dict:
    """Nhận ô mới theo P giảm dần (ngưỡng CHUNG hai phần): tỉ lệ GOLD và độ đúng TỔNG (GOLD hiện tại ∪ ô mới).
    Trả τ = P nhỏ nhất còn giữ độ đúng tổng ≥ target (trên chính o) + vài điểm đường cong."""
    gold = o.tier.values == "GOLD"
    elig = ~gold & np.isin(o.part.values, ["in", "out"]) & ~np.isnan(o.P.values)
    txt, both, has, pk = correct_cols(o, elig)
    ev = pk if metric == "both" else has
    ok = both if metric == "both" else txt
    cG, nG = float(ok[gold & ev].sum()), float((gold & ev).sum())
    idx = np.nonzero(elig)[0]
    idx = idx[np.argsort(-o.P.values[idx], kind="stable")]
    ce = np.cumsum(ok[idx] & ev[idx]); ne = np.cumsum(ev[idx])
    prec = (cG + ce) / np.maximum(nG + ne, 1)
    share = (gold.sum() + np.arange(1, len(idx) + 1)) / len(o)
    good = np.nonzero(prec >= target)[0]
    kmax = int(good[-1]) if len(good) else -1
    tau = float(o.P.values[idx[kmax]]) if kmax >= 0 else 1.01
    pts = {}
    for sh in (0.80, 0.85, 0.90, 0.92, 0.94):
        j = np.searchsorted(share, sh)
        if j < len(share):
            pts[str(sh)] = dict(prec=round(float(prec[j]), 4), tau=round(float(o.P.values[idx[j]]), 4))
    return dict(tau=tau, share_max=round(float(share[kmax]), 4) if kmax >= 0 else round(gold.mean(), 4),
                prec_at_max=round(float(prec[kmax]), 4) if kmax >= 0 else None, points=pts,
                share_all=round(float(share[-1]), 4) if len(share) else None,
                prec_all=round(float(prec[-1]), 4) if len(prec) else None)


def evaluate_overall(test: str, train: str, target=0.92, metric="both") -> dict:
    """Chế độ 'tổng': ngưỡng CHUNG trên P để độ đúng TỔNG ≥ target (LOBO từ ngoài-khối bộ học; CV cụm trang bộ thử)."""
    o = pd.read_pickle(T.OUT / "pred" / f"{test}__from_{train}.pkl")
    oof = pd.concat([pd.read_pickle(T.OUT / "pred" / f"{b}__oof_{train}.pkl") for b in train.split("+")])
    gold = o.tier.values == "GOLD"
    elig = ~gold & np.isin(o.part.values, ["in", "out"])
    P = np.nan_to_num(o.P.values, nan=-1)
    f_oof = frontier(oof, target, metric)
    res = dict(test=test, train=train, target=target, metric=metric, oof_frontier=f_oof,
               test_frontier_in_sample=frontier(o, target, metric))
    res["lobo"] = dict(tau=f_oof["tau"], all=metrics(o, gold | (elig & (P >= f_oof["tau"]))))
    acc = np.zeros(len(o), bool); taus = []
    for k in range(5):
        fk = frontier(o[o.blk != k].reset_index(drop=True), target, metric)
        taus.append(fk["tau"])
        acc |= (o.blk.values == k) & elig & (P >= fk["tau"])
    res["cv"] = dict(taus=taus, all=metrics(o, gold | acc))
    return res


def main():
    pairs = [a.split(">") for a in (sys.argv[1:] or ["B34>B18", "B18>B34", "TK>L16", "L16>TK"])]
    for tr, te in pairs:
        for target in (0.80, 0.90):
            r = evaluate(te, tr, target)
            T.jdump(r, T.OUT / f"policy_{te}__from_{tr}__t{int(target * 100)}.json")
            c, l, v = r["current"], r["lobo"], r["cv"]
            print(f"[t07] {te} <- {tr} (đích chọn {target}): hiện tại GOLD {c['share']:.3f} hai vế {c.get('both')} | "
                  f"LOBO τ=({l['tau_in']},{l['tau_out']}) GOLD {l['all']['share']:.3f} hai vế {l['all'].get('both')} "
                  f"{l['all'].get('both_ci')} chữ {l['all'].get('text')} | CV GOLD {v['all']['share']:.3f} hai vế "
                  f"{v['all'].get('both')} {v['all'].get('both_ci')}", flush=True)
            if target == 0.90:
                for met in ("both", "text"):
                    q = evaluate_overall(te, tr, 0.92, met)
                    T.jdump(q, T.OUT / f"overall_{te}__from_{tr}__{met}.json")
                    print(f"     [tổng ≥0,92 theo {met}] LOBO τ={q['lobo']['tau']:.3f} GOLD {q['lobo']['all']['share']:.3f} hai vế "
                          f"{q['lobo']['all'].get('both')} {q['lobo']['all'].get('both_ci')} chữ {q['lobo']['all'].get('text')} "
                          f"{q['lobo']['all'].get('text_ci')} | CV GOLD {q['cv']['all']['share']:.3f} hai vế {q['cv']['all'].get('both')} "
                          f"chữ {q['cv']['all'].get('text')} {q['cv']['all'].get('text_ci')} | trần trong mẫu {q['test_frontier_in_sample']['share_max']}", flush=True)
            for L, a in r["ablation_lobo"].items():
                print(f"     {L}: nhóm {a['n_nhom']}, bộ chọn -> GOLD {a['bo_chon']['share']:.3f} (thêm prec {a['them_prec'][0]:.3f} "
                      f"n {a['them_prec'][1]})" + (f"; nhận hết nhãn kim -> GOLD {a['tat_ca_nhan_kim']['share']:.3f} hai vế "
                                                    f"{a['tat_ca_nhan_kim'].get('both')}" if a["tat_ca_nhan_kim"] else ""))


if __name__ == "__main__":
    main()


# ================================================================================================ ngưỡng THEO NHÓM
PROMO_GROUPS = ("direct_qn", "direct_crop", "direct_other", "similar", "textonly", "other", "syl", "syl_bridge", "nocontext",
                "lowpost", "syl_cropbad")
MIN_SEL = 50


def pick_tau_groups(sel: pd.DataFrame, target: float) -> dict:
    """τ riêng từng nhóm (tiêu chí biên trên tập chọn, nhóm đó); nhóm có < MIN_SEL ô đo được trên tập chọn -> KHÔNG nâng."""
    out = {}
    for g in PROMO_GROUPS:
        base = (sel.tier.values != "GOLD") & (sel.grp.values == g) & np.isin(sel.part.values, ["in", "out"])
        base &= ~np.isin(sel.gate.values, EXCL_GATES) if "gate" in sel else True
        n_ev = int((base & (sel.gtc.values != "")).sum())
        if n_ev < MIN_SEL:
            out[g] = 1.01
            continue
        taus = []
        for part in ("in", "out"):
            taus.append(pick_tau(sel[sel.grp.values == g], part, target))
        out[g] = dict(zip(("in", "out"), taus))
    return out


def accept_groups(o: pd.DataFrame, taus: dict) -> np.ndarray:
    ng = (o.tier.values != "GOLD") & ~np.isin(o.gate.values, EXCL_GATES)
    P = np.nan_to_num(o.P.values, nan=-1)
    a = np.zeros(len(o), bool)
    for g, t in taus.items():
        if not isinstance(t, dict):
            continue
        m = ng & (o.grp.values == g)
        a |= m & (((o.part.values == "in") & (P >= t["in"])) | ((o.part.values == "out") & (P >= t["out"])))
    return a


def evaluate_groups(test: str, train: str, target: float, tag: str = "") -> dict:
    o = pd.read_pickle(T.OUT / "pred" / f"{test}{tag}__from_{train}.pkl")
    oof = pd.concat([pd.read_pickle(T.OUT / "pred" / f"{b}{tag}__oof_{train}.pkl") for b in train.split("+")])
    gold = o.tier.values == "GOLD"
    tl = pick_tau_groups(oof, target)
    aL = accept_groups(o, tl)
    acc = np.zeros(len(o), bool); tcv = []
    for k in range(5):
        tk = pick_tau_groups(o[o.blk != k], target)
        tcv.append(tk)
        acc |= (o.blk.values == k) & accept_groups(o, tk)
    per = {}
    for g in PROMO_GROUPS:
        m = o.grp.values == g
        if m.sum():
            per[g] = dict(n=int(m.sum()), nhan_lobo=int((aL & m).sum()), prec_lobo=added_prec(o, aL & m),
                          nhan_cv=int((acc & m).sum()), prec_cv=added_prec(o, acc & m))
    return dict(test=test, train=train, target=target, taus_lobo=tl, lobo=metrics(o, gold | aL), cv=metrics(o, gold | acc),
                per_group=per, current=metrics(o, gold), accepted_lobo=aL, accepted_cv=acc)
