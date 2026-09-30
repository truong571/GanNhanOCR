"""t11_stt_rules.py — đòn bẩy L3 cho STT (lt1 / lt2 / union / + phiếu ảnh) và phép đo TƯƠNG TỰ trên Borg. 0 API.

Bộ chọn CHỈ-ẢNH (t10: tầng 1 không đặc trưng kim) học trên Borg; top-1 ảnh = vtop, xác suất softmax vp.
STT (không sự thật) — quy tắc thêm GOLD cho ô KHÔNG GOLD:
  L3_lt2      lt1 ∉ R(âm) ∧ lt2 ∈ R(âm)                      nhãn = lt2
  L3_lt2_vis  L3_lt2 ∧ vtop == lt2 (V1+)                      hai tín hiệu độc lập một phần đồng ý
  L4_vis      lt1, lt2 ∉ R ∧ vp ≥ τ                           chỉ ảnh (không kiểm được)
Tương tự đo được trên Borg (kim Borg = lt2): ô KHÔNG GOLD có kim ∈ R — độ đúng nhãn kim khi vtop == kim và khi ≠;
ô kim ∉ R — độ đúng vtop theo vp (đường cong) — đều LOBO (mô hình học trên sách Borg kia).
Ra: measure_out/_tn8/stt_rules.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import t06_eval as E  # noqa: E402
import t10_visual_only as V  # noqa: E402


def vis_top(Bk, m):
    F = Bk["F"]
    p = m.predict(V.strip(Bk), F.i.values)
    G = pd.DataFrame({"i": F.i.values, "c": F.c.values, "p": p}).sort_values(["i", "p"], ascending=[True, False])
    t = G.groupby("i").head(1).set_index("i")
    return t.c.reindex(np.arange(len(Bk["D"]))).fillna("").values, t.p.reindex(np.arange(len(Bk["D"]))).fillna(0).values


def borg_analog(C):
    B = {b: E.load_book(b) for b in ("B18", "B34")}
    out = {}
    for te, tr in (("B18", "B34"), ("B34", "B18")):
        m = V.fit([B[tr]])
        vt, vp = vis_top(B[te], m)
        D = B[te]["D"]
        R = {s: C.R_of(s) for s in set(D.syllable)}
        kin = np.array([bool(k) and k in R[s] for k, s in zip(D.ocr_char.values, D.syllable.values)])
        has = D.gt_char.values != ""
        ng = (D.tier.values != "GOLD") & ~np.isin(B[te]["grp"], ["notplaus", "quarantine"])
        kim_ok = np.array([bool(k) and bool(g) and C.var_eq_plus(k, g) for k, g in zip(D.ocr_char.values, D.gt_char.values)])
        vt_ok = np.array([bool(a) and bool(g) and C.var_eq_plus(a, g) for a, g in zip(vt, D.gt_char.values)])
        agree = np.array([bool(a) and bool(k) and C.var_eq_plus(a, k) for a, k in zip(vt, D.ocr_char.values)])
        m1 = ng & kin & has & agree; m2 = ng & kin & has & ~agree
        r = dict(kimR_vis_agree=[round(float(kim_ok[m1].mean()), 4), int(m1.sum()), T.boot_ci_pages(kim_ok[m1], D.page.values[m1])],
                 kimR_vis_disagree_kim_ok=[round(float(kim_ok[m2].mean()), 4), int(m2.sum())],
                 kimR_vis_disagree_vis_ok=[round(float(vt_ok[m2].mean()), 4), int(m2.sum())],
                 kimR_share_agree=round(float(m1.sum() / max(1, (ng & kin & has).sum())), 4))
        m3 = ng & ~kin & has
        curve = []
        for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
            s = m3 & (vp >= t)
            curve.append([t, int(s.sum()), round(float(vt_ok[s].mean()), 4) if s.sum() else None,
                          round(float(s.sum() / max(1, m3.sum())), 4)])
        r["kim_notR_vis_curve[tau,n,prec,frac]"] = curve
        out[f"{te}<-{tr}"] = r
        print(f"[t11] Borg {te}<-{tr}: kim∈R & ảnh đồng ý -> nhãn đúng {r['kimR_vis_agree'][:2]} (tỉ lệ đồng ý "
              f"{r['kimR_share_agree']}); bất đồng: kim đúng {r['kimR_vis_disagree_kim_ok']}, ảnh đúng "
              f"{r['kimR_vis_disagree_vis_ok']}; kim∉R chỉ ảnh {curve}", flush=True)
    return out


def main():
    C = T.lex()
    res = dict(borg=borg_analog(C))
    m = V.fit([E.load_book("B18"), E.load_book("B34")])
    for b in ("stt2", "stt4", "stt11"):
        Bk = E.load_book(b)
        D = Bk["D"]
        vt, vp = vis_top(Bk, m)
        l2 = pd.read_pickle(T.OUT / "stt_lt2" / f"{b}.pkl").lt2.fillna("").values
        R = {s: C.R_of(s) for s in set(D.syllable)}
        l1in = np.array([bool(k) and k in R[s] for k, s in zip(D.ocr_char.values, D.syllable.values)])
        l2in = np.array([bool(x) and x in R[s] for x, s in zip(l2, D.syllable.values)])
        gold = D.tier.values == "GOLD"
        elig = ~gold & ~np.isin(Bk["grp"], ["notplaus", "quarantine"])
        agree2 = np.array([bool(a) and bool(x) and C.var_eq_plus(a, x) for a, x in zip(vt, l2)])
        r = dict(N=len(D), gold=int(gold.sum()), share_now=round(gold.mean(), 4))
        L3 = elig & ~l1in & l2in
        L3v = L3 & agree2
        r["L3_lt2"] = int(L3.sum()); r["L3_lt2_vis"] = int(L3v.sum())
        r["share_L3_lt2"] = round((gold.sum() + L3.sum()) / len(D), 4)
        r["share_L3_lt2_vis"] = round((gold.sum() + L3v.sum()) / len(D), 4)
        rest = elig & ~l1in & ~l2in
        r["neither_in_R"] = int(rest.sum())
        r["L4_vis_by_tau"] = {str(t): int((rest & (vp >= t)).sum()) for t in (0.8, 0.9, 0.95, 0.98)}
        r["share_L3v_L4_0.95"] = round((gold.sum() + L3v.sum() + (rest & (vp >= 0.95)).sum()) / len(D), 4)
        # ô không GOLD mà lt1 ∈ R (nhóm lop_nham…): lt1 == lt2?
        k = elig & l1in
        r["nonGOLD_lt1_inR"] = int(k.sum())
        res[b] = r
        print(f"[t11] {b}: GOLD {r['share_now']} | +L3_lt2 {r['L3_lt2']} -> {r['share_L3_lt2']} | +L3_lt2∧ảnh {r['L3_lt2_vis']} -> "
              f"{r['share_L3_lt2_vis']} | cả hai ∉ R {r['neither_in_R']}, ảnh ≥ τ {r['L4_vis_by_tau']} | L3v+L4(0,95) -> "
              f"{r['share_L3v_L4_0.95']}", flush=True)
    T.jdump(res, T.OUT / "stt_rules.json")


if __name__ == "__main__":
    main()
