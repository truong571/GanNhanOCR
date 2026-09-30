"""t04_selftrain_diag.py — bộ chọn CHỈ-ẢNH TỰ HỌC trong sách (giả thuyết 3) + chẩn đoán vì sao chuyển giao Borg→STT kém. 0 API.

Cùng đặc trưng tầng 1 của TN8 (bỏ cờ kim: is_kim, kim_inR, is_lt2), cùng CLogit; KHÁC: nhãn học = NHÃN TỰ ĐỘNG của chính
sách (không người) trên ô "chắc", học ngoài-khối (5 khối trang liền; khối thử không bao giờ góp ô học).
  STT  ô chắc = GOLD, nhãn ∈ R(âm), lt1 == lt2 == nhãn (V1+)                    (hai lần đọc kim đồng ý)
  Borg ô chắc = GOLD rule direct, nhãn = kim ∈ R(âm)                             (một lần đọc; nhãn người CHỈ để ĐO)
Đo:
  (a) top-1 ngoài-khối trên ô chắc (STT: = nhãn; Borg: = nhãn VÀ = chữ người) — so với bộ chọn học từ Borg (TN8 t10).
  (b) Borg: độ đúng thật (chữ người) theo nhóm ô khó khi học từ nhãn tự động — đối chứng LOBO-người của TN8.
  (c) STT: nhóm ô mới (lt1 ∉ R, lt2 ∈ R): top-1 == lt2 ?
Ra measure_out/_tn9/selftrain/<bộ>.pkl (top1, p1, p2, fold) + selftrain_diag.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
import tn8model as M  # noqa: E402
import t06_eval as E  # noqa: E402
import t10_visual_only as V  # noqa: E402

OD = T.OUT / "selftrain"
MAXC = 60000


def sure_mask(Bk, C, stt: bool):
    D = Bk["D"]
    lab = D.label.values
    R = {s: C.R_of(s) for s in set(D.syllable)}
    inR = np.array([bool(l) and l in R[s] for l, s in zip(lab, D.syllable.values)])
    gold = D.tier.values == "GOLD"
    direct = D.rule.astype(str).str.startswith("s1_inter_s2_direct").values
    kim_eq = np.array([bool(l) and l == k for l, k in zip(lab, D.ocr_char.values)])
    m = gold & inR & direct & kim_eq
    if stt:
        l2 = Bk["cells"].lt2.fillna("").values
        m &= np.array([bool(x) and C.var_eq_plus(l, x) for l, x in zip(lab, l2)])
    return m


def y_self(Bk, mask):
    """y theo nhãn tự động (chỉ ô trong mask)."""
    F, D = Bk["F"], Bk["D"]
    C = T.lex()
    lab = D.label.values
    ok = mask[F.i.values]
    y = np.zeros(len(F), np.int8)
    idx = np.nonzero(ok)[0]
    y[idx] = [int(c == lab[i] or C.var_eq_plus(c, lab[i])) for i, c in zip(F.i.values[idx], F.c.values[idx])]
    return y


def fit_cells(parts, seed=0):
    """parts: list of (Bk, cell_mask, y_vec). CLogit trên đặc trưng chỉ-ảnh."""
    rng = np.random.default_rng(seed)
    Xs, Is, Ys = [], [], []
    off = 0
    for Bk, cm, y in parts:
        F, D = Bk["F"], Bk["D"]
        nc = F.groupby("i").size().reindex(np.arange(len(D))).fillna(0).values
        anyy = pd.Series(y).groupby(F.i.values).max().reindex(np.arange(len(D))).fillna(0).values > 0
        cells = np.nonzero(cm & anyy & (nc >= 2))[0]
        if len(cells) > MAXC:
            cells = np.sort(rng.choice(cells, MAXC, replace=False))
        sel = np.isin(F.i.values, cells)
        Xs.append(V.strip(Bk)[sel]); Is.append(F.i.values[sel] + off); Ys.append(y[sel])
        off += len(D) + 1
    return M.CLogit().fit(np.concatenate(Xs), np.concatenate(Is), np.concatenate(Ys))


def top_of(Bk, m):
    F = Bk["F"]
    p = m.predict(V.strip(Bk), F.i.values)
    G = pd.DataFrame({"i": F.i.values, "c": F.c.values, "p": p}).sort_values(["i", "p"], ascending=[True, False])
    t1 = G.groupby("i").head(1).set_index("i")
    t2 = G.groupby("i").nth(1).set_index("i")
    out = pd.DataFrame(index=np.arange(len(Bk["D"])))
    out["top1"] = t1.c.reindex(out.index).fillna("")
    out["p1"] = t1.p.reindex(out.index).fillna(0.0)
    out["p2"] = t2.p.reindex(out.index).fillna(0.0)
    return out


def oof(books: dict, masks: dict, ys: dict, extra_train: list | None = None):
    """Dự đoán ngoài-khối cho mỗi bộ trong books: khối k của bộ b thử bằng mô hình học trên (mọi bộ, bỏ khối k của b)."""
    res = {}
    for b, Bk in books.items():
        blk = Bk["cells"].blk.values
        parts_pred = []
        for k in range(5):
            parts = []
            for b2, Bk2 in books.items():
                cm = masks[b2].copy()
                if b2 == b:
                    cm &= blk != k
                parts.append((Bk2, cm, ys[b2]))
            if extra_train:
                parts += extra_train
            m = fit_cells(parts, seed=k)
            t = top_of(Bk, m)
            t["fold"] = k
            parts_pred.append(t[blk == k])
        res[b] = pd.concat(parts_pred).sort_index()
        print(f"[t04] {b}: ngoài-khối xong", flush=True)
    return res


def main():
    C = T.lex()
    OD.mkdir(parents=True, exist_ok=True)
    out = {}
    # ---------------- Borg: tự học từ nhãn tự động, ĐO bằng người
    borg = {b: E.load_book(b) for b in ("B18", "B34")}
    for b, Bk in borg.items():
        Bk["cells"]["lt2"] = ""
    bm = {b: sure_mask(Bk, C, False) for b, Bk in borg.items()}
    by = {b: y_self(Bk, bm[b]) for b, Bk in borg.items()}
    # mỗi sách Borg tự học RIÊNG (như STT: trong sách) — không trộn sách kia để đối chứng "tự học một sách"
    for b, Bk in borg.items():
        r = oof({b: Bk}, {b: bm[b]}, {b: by[b]})[b]
        r.to_pickle(OD / f"{b}.pkl")
        D = Bk["D"]; grp = Bk["grp"]
        gt = D.gt_char.values
        ok = np.array([bool(g) and bool(t) and C.var_eq_plus(t, g) for t, g in zip(r.top1.values, gt)])
        has = gt != ""
        rr = dict(n_sure=int(bm[b].sum()), sure_label_ok=round(float(np.mean([C.var_eq_plus(l, g) for l, g in
                                                                              zip(D.label.values[bm[b] & has], gt[bm[b] & has])])), 4))
        s = bm[b] & has
        rr["sure_top1_eq_label"] = round(float(np.mean([C.var_eq_plus(t, l) for t, l in zip(r.top1.values[bm[b]], D.label.values[bm[b]])])), 4)
        rr["sure_top1_ok"] = round(float(ok[s].mean()), 4)
        for g in ("GOLD", "direct_qn", "syl", "nocontext", "similar", "syl_bridge"):
            mm = has & (grp == g)
            if mm.sum():
                rr[f"{g}_top1_ok"] = [round(float(ok[mm].mean()), 4), int(mm.sum())]
                # đường cong theo p1
                cur = []
                for t in (0.5, 0.7, 0.8, 0.9, 0.95):
                    q = mm & (r.p1.values >= t)
                    cur.append([t, round(float(q.sum() / mm.sum()), 3), round(float(ok[q].mean()), 4) if q.sum() else None])
                rr[f"{g}_curve[t,frac,prec]"] = cur
        out[b] = rr
        print(f"[t04] Borg {b} tự học: {rr}", flush=True)
    # ---------------- STT
    stt = {b: E.load_book(b) for b in T.STT}
    sm = {b: sure_mask(Bk, C, True) for b, Bk in stt.items()}
    sy = {b: y_self(Bk, sm[b]) for b, Bk in stt.items()}
    R = oof(stt, sm, sy)
    for b, Bk in stt.items():
        r = R[b]; r.to_pickle(OD / f"{b}.pkl")
        D = Bk["D"]
        l2 = Bk["cells"].lt2.fillna("").values
        Rm = {s: C.R_of(s) for s in set(D.syllable)}
        l1in = np.array([bool(k) and k in Rm[s] for k, s in zip(D.ocr_char.values, D.syllable.values)])
        l2in = np.array([bool(x) and x in Rm[s] for x, s in zip(l2, D.syllable.values)])
        gold = D.tier.values == "GOLD"
        rr = dict(n_sure=int(sm[b].sum()))
        rr["sure_top1_eq_label"] = round(float(np.mean([C.var_eq_plus(t, l) for t, l in zip(r.top1.values[sm[b]], D.label.values[sm[b]])])), 4)
        j = ~gold & ~l1in & l2in
        rr["new_l2only_top1_eq_lt2"] = [round(float(np.mean([C.var_eq_plus(t, x) for t, x in zip(r.top1.values[j], l2[j])])), 4), int(j.sum())]
        k = gold & l1in & l2in & ~np.array([C.var_eq_plus(a, x) for a, x in zip(D.label.values, l2)])
        rr["gold_conflict_top1_eq_lt1"] = round(float(np.mean([C.var_eq_plus(t, x) for t, x in zip(r.top1.values[k], D.label.values[k])])), 4)
        rr["gold_conflict_top1_eq_lt2"] = round(float(np.mean([C.var_eq_plus(t, x) for t, x in zip(r.top1.values[k], l2[k])])), 4)
        g2 = gold & ~sm[b]
        rr["gold_notsure_top1_eq_label"] = [round(float(np.mean([C.var_eq_plus(t, x) for t, x in zip(r.top1.values[g2], D.label.values[g2])])), 4), int(g2.sum())]
        out[b] = rr
        print(f"[t04] STT {b} tự học: {rr}", flush=True)
    T.jdump(out, T.OUT / "selftrain_diag.json")


if __name__ == "__main__":
    main()
