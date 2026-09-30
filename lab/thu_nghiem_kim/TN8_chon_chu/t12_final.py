"""t12_final.py — CHÍNH SÁCH CUỐI TN8 + bảng số cho 10 bộ -> measure_out/_tn8/summary.json. 0 API.

Chính sách (quyết định từ số LOBO của t07; không dò thêm trên bộ thử):
  GOLD hiện tại giữ nguyên (nhãn không đổi).
  Ô KHÔNG GOLD được nâng GOLD với nhãn = top-1 của bộ chọn hai tầng khi:
    (1) nhóm ô thuộc danh sách NÂNG ĐƯỢC của họ chữ:
          viết tay: direct_qn (qn_count_unfixed), direct_crop (crop_bad), direct_other, similar, other (confusion_fix),
                    syl (SYLLABLE), nocontext, lowpost, syl_cropbad
          in/khắc : direct_crop (crop_bad), other (confusion_fix)       [L2/L4 trên sách in: đo được là KHÔNG đạt]
        không bao giờ: GOLD_text_only (hộp không tự tin), cổng văn bản cross_similar / am_sua_dau / bridge_similar,
        not_plausible (âm giữ chỗ / không hợp lệ), QUARANTINE;
    (2) P (tầng 2) ≥ τ_phần, τ chọn theo TIÊU CHÍ BIÊN 0,80 (ô biên — ô tệ nhất được nhận — vẫn đúng ≥ 80 % trên tập chọn)
        trên dự đoán ngoài-khối của bộ HỌC (LOBO) — CV cụm trang của chính bộ thử báo kèm.
STT (không sự thật): (a) BẢO THỦ = GOLD ∪ {lt1 ∉ R, lt2 ∈ R, top-1 chỉ-ảnh == lt2}; (b) MẠNH = chính sách viết tay ở chế
độ union (lt1 ∪ lt2) — độ đúng chỉ SUY ĐOÁN.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import t07_policy as P  # noqa: E402

PROMO = {"hand": ("direct_qn", "direct_crop", "direct_other", "similar", "other", "syl", "nocontext", "lowpost", "syl_cropbad"),
         "print": ("direct_crop", "other")}
LEVER = {"direct_crop": "L1", "direct_qn": "L2", "direct_other": "L2b", "similar": "L2b", "other": "L2b",
         "syl": "L4", "nocontext": "L4", "lowpost": "L4", "syl_cropbad": "L4"}
TARGET = 0.80


def restrict(o: pd.DataFrame, fam: str, groups=None) -> pd.DataFrame:
    """Tập chọn/đánh giá: chỉ ô thuộc nhóm nâng được (ô khác coi như P = nan)."""
    o = o.copy()
    g = PROMO[fam] if groups is None else groups
    bad = ~np.isin(o.grp.values, g) | np.isin(o.gate.values, P.EXCL_GATES)
    o.loc[bad & (o.tier.values != "GOLD"), "P"] = np.nan
    return o


def taus_of(sel: pd.DataFrame, fam: str) -> tuple[float, float]:
    s = restrict(sel, fam)
    return P.pick_tau(s, "in", TARGET), P.pick_tau(s, "out", TARGET)


def acc_of(o: pd.DataFrame, fam: str, ti: float, to: float, groups=None) -> np.ndarray:
    return P.accept(restrict(o, fam, groups), ti, to)


def pos_rate(o, acc):
    m = acc & o.pos_known.values
    return round(float(o.pos_ok.values[m].mean()), 4) if m.sum() else None


TAU_RELABEL = 0.80


def relabel_mask(o: pd.DataFrame, fam: str) -> np.ndarray:
    """L5 (chỉ viết tay): ô GOLD hiện tại, phần 'in', top-1 ≠ nhãn (V1+) và P ≥ TAU_RELABEL -> nhãn := top-1."""
    if fam != "hand":
        return np.zeros(len(o), bool)
    C = T.lex()
    top = o.top1.fillna("").values
    m = (o.tier.values == "GOLD") & (o.part.values == "in") & (np.nan_to_num(o.P.values, nan=-1) >= TAU_RELABEL) & (top != "")
    dis = np.array([bool(t) and not C.var_eq_plus(t, l) for t, l in zip(top, o.label.values)])
    return m & dis


def metrics_final(o: pd.DataFrame, acc: np.ndarray, relab: np.ndarray) -> dict:
    """Như P.metrics nhưng nhãn cuối: GOLD giữ nhãn (trừ ô L5 -> top-1), ô mới -> top-1."""
    C = T.lex()
    gold = o.tier.values == "GOLD"
    lab = np.where(gold & ~relab, o.label.values, o.top1.fillna("").values)
    ok = np.array([bool(l) and bool(g) and C.var_eq_plus(l, g) for l, g in zip(lab, o.gtc.values)])
    has = o.gtc.values != ""
    r = dict(n=int(acc.sum()), share=round(acc.sum() / len(o), 4), n_relabel=int((relab & acc).sum()))
    m = acc & has & o.pos_known.values
    if m.sum():
        pt, lo, hi = T.boot_ci_pages((ok & o.pos_ok.values)[m], o.page.values[m])
        r.update(both=round(pt, 4), both_ci=[round(lo, 4), round(hi, 4)], n_both=int(m.sum()))
    m = acc & has
    if m.sum():
        pt, lo, hi = T.boot_ci_pages(ok[m], o.page.values[m])
        r.update(text=round(pt, 4), text_ci=[round(lo, 4), round(hi, 4)], n_text=int(m.sum()))
    return r


def truth_book(te: str, tr: str, fam: str) -> dict:
    o = pd.read_pickle(T.OUT / "pred" / f"{te}__from_{tr}.pkl")
    oof = pd.read_pickle(T.OUT / "pred" / f"{tr}__oof_{tr}.pkl")
    gold = o.tier.values == "GOLD"
    ti, to = taus_of(oof, fam)
    aL = acc_of(o, fam, ti, to)
    acc = np.zeros(len(o), bool); tcv = []
    for k in range(5):
        a, b_ = taus_of(o[o.blk != k], fam)
        tcv.append([a, b_])
        acc |= (o.blk.values == k) & acc_of(o, fam, a, b_)
    RL = relabel_mask(o, fam)
    r = dict(book=te, train=tr, fam=fam, N=len(o), current=P.metrics(o, gold), tau_lobo=[ti, to],
             lobo_L5=metrics_final(o, gold | aL, RL), cv_L5=metrics_final(o, gold | acc, RL),
             L5_mot_minh=metrics_final(o, gold, RL),
             lobo=P.metrics(o, gold | aL), cv=dict(taus=tcv, **P.metrics(o, gold | acc)),
             pos_rate_new_lobo=pos_rate(o, aL & ~gold), n_new_no_gt_lobo=int((aL & (o.gtc.values == "")).sum()))
    # ablation theo đòn bẩy (mỗi đòn một mình, cùng ngưỡng LOBO) + biến thể không bộ chọn cho L1/L2
    abl = {}
    for L in ("L1", "L2", "L2b", "L4"):
        gs = [g for g, l in LEVER.items() if l == L]
        a = acc_of(o, "hand", ti, to, gs) if fam == "hand" else acc_of(o, fam, ti, to, [g for g in gs if g in PROMO[fam]])
        m_all = np.isin(o.grp.values, gs) & ~gold & ~np.isin(o.gate.values, P.EXCL_GATES)
        blk = dict(n_nhom=int(m_all.sum()), bo_chon=P.metrics(o, gold | a), them=P.added_prec(o, a))
        if L in ("L1", "L2"):
            nk = m_all & (o.part.values == "in")
            blk["nhan_kim_khong_loc"] = P.metrics(o, gold | nk)
            blk["nhan_kim_khong_loc_them"] = P.added_prec(o, nk)
        if fam == "print" and L in ("L2", "L4"):            # đo để CHỨNG MINH không nâng được (ngưỡng họ viết tay không áp)
            s = restrict(oof, fam, gs)
            a2 = P.accept(restrict(o, fam, gs), P.pick_tau(s, "in", TARGET), P.pick_tau(s, "out", TARGET))
            blk["neu_cho_phep"] = dict(them=P.added_prec(o, a2), n=int(a2.sum()))
        abl[L] = blk
    r["ablation"] = abl
    # theo nhóm
    grp = {}
    for g in sorted(set(o.grp.values)):
        m = o.grp.values == g
        grp[g] = dict(n=int(m.sum()), nhan=int((aL & m).sum()), prec_them=P.added_prec(o, aL & m))
    r["groups"] = grp
    # trần: nhận MỌI ô nâng được (không ngưỡng)
    aA = acc_of(o, fam, 0.0, 0.0)
    r["tran_nhan_het_nhom_nang_duoc"] = P.metrics(o, gold | aA)
    return r


def stt_book(b: str) -> dict:
    st = json.load(open(T.OUT / "stt_rules.json", encoding="utf-8"))[b]
    M = pickle.load(open(T.OUT / "models" / "hand.pkl", "rb"))
    oof = M["oof"]
    ti, to = taus_of(oof, "hand")
    out = dict(book=b, N=st["N"], current_share=st["share_now"], bao_thu=dict(n_new=st["L3_lt2_vis"], share=st["share_L3_lt2_vis"]),
               lt2_khong_anh=dict(n_new=st["L3_lt2"], share=st["share_L3_lt2"]))
    for mode in ("union", "kim"):
        o = pd.read_pickle(T.OUT / "pred" / f"{b}__from_hand__{mode}.pkl")
        gold = o.tier.values == "GOLD"
        a = acc_of(o, "hand", ti, to)
        out[f"manh_{mode}"] = dict(tau=[ti, to], n_new=int(a.sum()), share=round((gold.sum() + a.sum()) / len(o), 4),
                                   mean_P_new=round(float(np.nanmean(o.P.values[a])), 4))
    # kiểm chéo độc lập (chế độ kim = lt1: lt2 không vào mô hình)
    o = pd.read_pickle(T.OUT / "pred" / f"{b}__from_hand__kim.pkl")
    a = acc_of(o, "hand", ti, to)
    l2 = pd.read_pickle(T.OUT / "stt_lt2" / f"{b}.pkl").lt2.fillna("").values
    D = T.load_base(b)
    C = T.lex()
    R = {s: C.R_of(s) for s in set(D.syllable)}
    l2in = np.array([bool(x) and x in R[s] for x, s in zip(l2, D.syllable.values)])
    m = a & l2in
    agree = np.array([C.var_eq_plus(t, x) for t, x in zip(o.top1.fillna("").values[m], l2[m])])
    out["kiem_cheo_lt2_o_moi"] = dict(n=int(m.sum()), top1_eq_lt2=round(float(agree.mean()), 4) if m.sum() else None,
                                      o_moi_co_lt2_inR=round(float(m.sum() / max(1, a.sum())), 4))
    return out


def print_book(b: str) -> dict:
    M = pickle.load(open(T.OUT / "models" / "print.pkl", "rb"))
    ti, to = taus_of(M["oof"], "print")
    o = pd.read_pickle(T.OUT / "pred" / f"{b}__from_print__kim.pkl")
    gold = o.tier.values == "GOLD"
    a = acc_of(o, "print", ti, to)
    r = dict(book=b, N=len(o), current_share=round(gold.mean(), 4), tau=[ti, to], n_new=int(a.sum()),
             share=round((gold.sum() + a.sum()) / len(o), 4), mean_P_new=round(float(np.nanmean(o.P.values[a])), 4) if a.sum() else None,
             groups={g: dict(n=int((o.grp.values == g).sum()), nhan=int((a & (o.grp.values == g)).sum()))
                     for g in sorted(set(o.grp.values))})
    if b in ("L83", "KVK"):
        C = T.lex()
        hasref = o.ref.values != ""
        m = a & hasref
        r["di_ban_trung_o_moi"] = [round(float(o.yref1.values[m].mean()), 4) if m.sum() else None, int(m.sum())]
        g = gold & hasref
        r["di_ban_trung_GOLD"] = [round(float(np.mean([any(C.var_eq_plus(l, x) for x in rf.split("|"))
                                                       for l, rf in zip(o.label.values[g], o.ref.values[g])])), 4), int(g.sum())]
    return r


def main():
    global TARGET
    if len(sys.argv) > 1:
        TARGET = float(sys.argv[1])
    S = dict(policy=__doc__, target_bien=TARGET, promo=PROMO, excl_gates=P.EXCL_GATES, books={})
    for te, tr, fam in (("B18", "B34", "hand"), ("B34", "B18", "hand"), ("L16", "TK", "print"), ("TK", "L16", "print")):
        r = truth_book(te, tr, fam)
        S["books"][te] = r
        c, l, v = r["current"], r["lobo"], r["cv"]
        q = r["lobo_L5"]; q2 = r["cv_L5"]; q3 = r["L5_mot_minh"]
        print(f"[t12] {te} +L5 (sửa nhãn GOLD {q['n_relabel']}): LOBO hai vế {q.get('both')} {q.get('both_ci')} chữ {q.get('text')} "
              f"{q.get('text_ci')} | CV hai vế {q2.get('both')} chữ {q2.get('text')} {q2.get('text_ci')} | L5 một mình: GOLD hai vế "
              f"{q3.get('both')} chữ {q3.get('text')} {q3.get('text_ci')}", flush=True)
        print(f"[t12] {te}: hiện tại {c['share']:.3f} (hai vế {c.get('both')}, chữ {c.get('text')}) -> LOBO {l['share']:.3f} hai vế "
              f"{l.get('both')} {l.get('both_ci')} chữ {l.get('text')} {l.get('text_ci')} | CV {v['share']:.3f} hai vế {v.get('both')} "
              f"{v.get('both_ci')} chữ {v.get('text')} {v.get('text_ci')} | vị trí ô mới {r['pos_rate_new_lobo']}", flush=True)
        for L, a in r["ablation"].items():
            ex = ""
            if "nhan_kim_khong_loc" in a:
                ex = f"; nhận hết nhãn kim: {a['nhan_kim_khong_loc']['share']:.3f} (thêm {a['nhan_kim_khong_loc_them']})"
            if "neu_cho_phep" in a:
                ex += f"; nếu cho phép: {a['neu_cho_phep']}"
            print(f"      {L}: nhóm {a['n_nhom']} -> {a['bo_chon']['share']:.3f} (thêm {a['them']}){ex}", flush=True)
    for b in ("stt2", "stt4", "stt11"):
        r = stt_book(b)
        S["books"][b] = r
        print(f"[t12] {b}: {r['current_share']} -> bảo thủ {r['bao_thu']['share']} | mạnh union {r['manh_union']['share']} "
              f"(mean_P {r['manh_union']['mean_P_new']}) | kiểm chéo lt2 {r['kiem_cheo_lt2_o_moi']}", flush=True)
    for b in ("Chr", "L83", "KVK"):
        r = print_book(b)
        S["books"][b] = r
        print(f"[t12] {b}: {r['current_share']} -> {r['share']} (+{r['n_new']}, mean_P {r['mean_P_new']}) "
              f"{r.get('di_ban_trung_o_moi', '')} {r.get('di_ban_trung_GOLD', '')}", flush=True)
    fam_tau = {}
    for fam in ("hand", "print"):
        M = pickle.load(open(T.OUT / "models" / f"{fam}.pkl", "rb"))
        fam_tau[fam] = taus_of(M["oof"], fam)
    S["tau_trien_khai_ho"] = fam_tau
    print(f"[t12] ngưỡng triển khai (ngoài-khối gộp hai bộ học, biên {TARGET}): {fam_tau}", flush=True)
    T.jdump(S, T.OUT / f"summary_t12_b{int(round(TARGET * 100))}.json")


if __name__ == "__main__":
    main()
