"""t11_print_kimverify.py — "tin chữ kim" cho ô KHÔNG GOLD của sách in, kiểm bằng ảnh, hiệu chuẩn trên IHR (LOBO). 0 API.

Câu hỏi: crop có đúng là chữ kim (nội dung) không, bất kể âm QN? Trên IHR (L16, TK) sự thật = chữ người tại tâm crop (t10).
Mô hình: hồi quy logistic trên đặc trưng ẢNH của chữ kim (không đặc trưng nhóm/âm để khỏi học "no_context = sai" của IHR):
  f_vW, f_vP (CNN vft sạch với bộ đang chấm, như TN8), d_vW (= f_vW(kim) − max f_vW ứng viên khác), f_self, f_selfk (nguyên mẫu
  cùng sách, khối trang khác), f_font, f_fd, f_head (MultiEnc: học STT, ngoài mẫu với sách in), ln(1+n_self), geo_ok (tâm crop
  nằm trong hộp chữ kim đã ghép), kim ∈ R(âm) (chỉ để mô hình biết ô "dễ"; bản KHÔNG có cờ này báo song song).
LOBO: học L16 -> thử TK và ngược lại (ngưỡng chọn trên dự đoán NGOÀI-KHỐI 5 khối trang của bộ học, tiêu chí biên như TN8:
  τ = P của ô cuối mà cửa sổ ~200 ô liền trước còn đúng ≥ 0,90). Sau đó học L16+TK -> áp L83/KVK/Chr.
Ra measure_out/_tn9/print_kimverify.json + print/<bộ>_kv.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
import tn8model as M8  # noqa: E402

FEATS = ["f_vW", "f_vP", "d_vW", "f_self", "f_selfk", "f_font", "f_fd", "f_head", "ln_self", "geo_ok", "has_vP", "has_self"]
GROUPS_ELIG = ["direct_qn", "nocontext", "syl", "syl_bridge", "notplaus", "textonly", "direct_other", "similar", "syl_cropbad",
               "lowpost", "direct_crop", "other"]


def feats(b: str) -> pd.DataFrame:
    D = pd.read_pickle(T.OUT / "print" / f"{b}.pkl")
    F = pd.read_pickle(T.T8 / "cand" / f"{b}.pkl")
    k = D.ocr_char.values
    F = F[F.i.values < len(D)]
    is_k = F.c.values == k[F.i.values]
    Fk = F[is_k].drop_duplicates("i").set_index("i")
    other = F[~is_k].groupby("i").f_vW.max()
    X = pd.DataFrame(index=D.index)
    for c in ("f_vW", "f_vP", "f_self", "f_selfk", "f_font", "f_fd", "f_head"):
        X[c] = Fk[c].reindex(D.index).values
    X["d_vW"] = X.f_vW - other.reindex(D.index).fillna(-1.0).values
    X["ln_self"] = np.log1p(Fk.n_self.reindex(D.index).fillna(0).values)
    X["has_vP"] = X.f_vP.notna().astype(float)
    X["has_self"] = X.f_self.notna().astype(float)
    X["geo_ok"] = D.geo_ok.astype(float).values
    X["kim_inR"] = D.kim_inR.astype(float).values
    for c in X.columns:
        med = np.nanmedian(X[c].values) if X[c].notna().any() else 0.0
        X[c] = X[c].fillna(med if c not in ("f_self", "f_selfk", "f_vP") else -1.0)
    return D, X


def tau_margin(p, y, target=0.90, W=200):
    o = np.argsort(-p)
    ys, ps = y[o].astype(float), p[o]
    cs = np.cumsum(ys)
    tau = 1.01
    for j in range(len(ys)):
        lo = max(0, j - W + 1)
        win = (cs[j] - (cs[lo - 1] if lo > 0 else 0)) / (j - lo + 1)
        if j >= W - 1 and win < target:
            break
        tau = ps[j]
    return float(tau)


def oof_pred(X, y, pages, cols):
    blk = T.page_blocks(pages, 5)
    p = np.zeros(len(y))
    for k in range(5):
        m = M8.Logit().fit(X[cols].values[blk != k], y[blk != k])
        p[blk == k] = m.predict(X[cols].values[blk == k])
    return p


def main():
    C = T.lex()
    out = {}
    data = {b: feats(b) for b in ("L16", "TK", "L83", "KVK", "Chr")}
    for cols_name, cols in (("img+inR", FEATS + ["kim_inR"]), ("img", FEATS)):
        res = {}
        for te, tr in (("TK", "L16"), ("L16", "TK")):
            Dtr, Xtr = data[tr]; Dte, Xte = data[te]
            mtr = (Dtr.content != "").values & (Dtr.ocr_char != "").values
            ytr = Dtr.kim_ok_content.values.astype(bool)
            p_oof = oof_pred(Xtr[mtr], ytr[mtr], Dtr.page.values[mtr], cols)
            # ngưỡng: tiêu chí biên trên ô KHÔNG GOLD của bộ học (ngoài-khối)
            ng = (Dtr.tier.values[mtr] != "GOLD")
            tau = tau_margin(p_oof[ng], ytr[mtr][ng]) if ng.sum() > 50 else 0.9
            m = M8.Logit().fit(Xtr[cols].values[mtr], ytr[mtr])
            pte = m.predict(Xte[cols].values)
            Dte = Dte.assign(pkv=pte)
            mte = (Dte.content != "").values
            r = dict(tau=round(tau, 4))
            for g in ["ALL_nonGOLD"] + GROUPS_ELIG + ["GOLD"]:
                if g == "ALL_nonGOLD":
                    sel = mte & (Dte.tier.values != "GOLD")
                else:
                    sel = mte & (Dte.grp.values == g)
                if sel.sum() == 0:
                    continue
                acc = sel & (pte >= tau)
                yk = Dte.kim_ok_content.values.astype(bool)
                yt = Dte.kim_triple_ok.values.astype(bool)
                r[g] = dict(n=int(sel.sum()), accept=int(acc.sum()),
                            prec_content=T.boot_ci_pages(yk[acc], Dte.page.values[acc]) if acc.sum() else None,
                            prec_triple=round(float(yt[acc].mean()), 4) if acc.sum() else None,
                            base_content=round(float(yk[sel].mean()), 4))
            res[f"{te}<-{tr}"] = r
            print(f"[t11] {cols_name} {te}<-{tr} τ={tau:.3f}: " + "; ".join(
                f"{g} {v['accept']}/{v['n']} P={v['prec_content'][0] if v['prec_content'] else None}"
                for g, v in r.items() if isinstance(v, dict)), flush=True)
            data[te][0][f"pkv_{cols_name}_lobo"] = pte
        out[cols_name] = res
        # học cả hai -> áp in không sự thật
        Xall = pd.concat([data["L16"][1], data["TK"][1]])
        Dall = pd.concat([data["L16"][0], data["TK"][0]])
        mall = (Dall.content != "").values & (Dall.ocr_char != "").values
        yall = Dall.kim_ok_content.values.astype(bool)
        pages = (Dall.book_col.astype(str) + "/" + Dall.page.astype(str)).values
        p_oof = oof_pred(Xall[mall], yall[mall], pages[mall], cols)
        ng = Dall.tier.values[mall] != "GOLD"
        tau = tau_margin(p_oof[ng], yall[mall][ng])
        m = M8.Logit().fit(Xall[cols].values[mall], yall[mall])
        out[cols_name]["tau_both"] = round(tau, 4)
        for b in ("L83", "KVK", "Chr"):
            D, X = data[b]
            p = m.predict(X[cols].values)
            D[f"pkv_{cols_name}"] = p
            rr = {}
            for g in GROUPS_ELIG:
                sel = (D.grp.values == g)
                if sel.sum() == 0:
                    continue
                acc = sel & (p >= tau)
                v = dict(n=int(sel.sum()), accept=int(acc.sum()), p_mean=round(float(p[acc].mean()), 4) if acc.sum() else None)
                if "ref" in D and (D.ref.values[acc] != "").sum():
                    hr = acc & (D.ref.values != "")
                    v["kim_eq_ref"] = [round(float(D.kim_eq_ref.values[hr].mean()), 4), int(hr.sum())]
                rr[g] = v
            gold = (D.tier.values == "GOLD")
            if "ref" in D:
                hr = gold & (D.ref.values != "")
                rr["GOLD_kim_eq_ref"] = [round(float(D.kim_eq_ref.values[hr].mean()), 4), int(hr.sum())]
            out[cols_name][b] = rr
            print(f"[t11] {cols_name} {b} (τ {tau:.3f}): " + "; ".join(f"{g} {v['accept']}/{v['n']}" + (f" ref {v['kim_eq_ref']}" if isinstance(v, dict) and 'kim_eq_ref' in v else "")
                                                           for g, v in rr.items() if isinstance(v, dict)), flush=True)
    for b, (D, X) in data.items():
        D.to_pickle(T.OUT / "print" / f"{b}_kv.pkl")
    T.jdump(out, T.OUT / "print_kimverify.json")


if __name__ == "__main__":
    main()
