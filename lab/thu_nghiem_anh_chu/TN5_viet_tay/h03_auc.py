#!/usr/bin/env python3
"""TN5 h03 — tín hiệu nào PHÂN BIỆT trên chữ viết tay? AUC + precision theo cờ, trên Borg (nhãn người), hai vế:
  y_lab  = V1+(nhãn, chữ người)                        (ô có gt)
  y_slot = tâm hộp mực crop chuẩn ∈ hộp người          (ô có hộp người mức keep_high trở lên — trượt hộp người ≤ 1,9 %)
  y_both = y_lab ∧ y_slot                              (ô có cả hai)
AUC theo hướng "điểm cao = đúng" (AUC < 0,5 -> tín hiệu ngược). Thêm AUC TRONG simg = 1 (bộ kiểm viết tay đã chứng):
tín hiệu còn giúp gì sau cổng chính. So sánh phân bố tín hiệu Borg ↔ STT (median) để thấy chuyển được hay không.

Ra: measure_out/_thu_nghiem_anh_chu/TN5/auc.csv, auc_summary.json; in bảng ngắn.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "measure_out/_thu_nghiem_anh_chu/TN5"
KEEP = ("keep_v5", "keep", "keep_high")


def auc(score, y):
    s = np.asarray(score, float); y = np.asarray(y, bool)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    if y.all() or (~y).all() or len(y) < 20:
        return np.nan, int(m.sum())
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), int(m.sum())


def load():
    import sys as _s
    _s.path.insert(0, str(Path(__file__).parent))
    from h02_features import BORG_ONLY, COMMON
    F = pd.read_pickle(OUT / "features.pkl")
    T = pd.read_csv(OUT / "truth.csv", dtype={"gt": str, "h_char": str}, keep_default_na=False)
    T = T[["cell_uid", "gt", "v1p", "strict", "p4", "slot_new", "slot_old", "keep_level", "has_hbox", "d_new", "iou_old", "h_idx"]]
    B = F[F.set8.isin(["B18", "B34"])].merge(T, on="cell_uid", how="left", validate="1:1")
    B["has_gt"] = B["gt"].fillna("") != ""
    B["keepbox"] = B.keep_level.isin(KEEP) & B.has_hbox.astype(bool)
    B["y_lab"] = B.v1p.astype(bool)
    B["y_slot"] = B.slot_new.astype(bool)
    B["y_both"] = B.y_lab & B.y_slot
    return F, B, COMMON, BORG_ONLY


def main():
    F, B, COMMON, BORG_ONLY = load()
    rows = []
    for f in COMMON + BORG_ONLY:
        r = dict(signal=f, borg_only=f in BORG_ONLY)
        for tgt, mk in (("lab", B.has_gt), ("slot", B.keepbox), ("both", B.has_gt & B.keepbox)):
            y = B["y_" + tgt].values
            a, _ = auc(B[f].values[mk.values], y[mk.values])
            r[f"auc_{tgt}"] = round(a, 3) if a == a else None
            ms = mk.values & (B.simg.values == 1)
            a2, _ = auc(B[f].values[ms], y[ms])
            r[f"auc_{tgt}_in_simg"] = round(a2, 3) if a2 == a2 else None
        vals = pd.Series(B[f].values).dropna().unique()
        if len(vals) <= 2 and set(vals) <= {0, 1}:
            for tgt, mk in (("lab", B.has_gt), ("both", B.has_gt & B.keepbox)):
                y = B["y_" + tgt].values; m = mk.values
                for v in (0, 1):
                    mm = m & (B[f].values == v)
                    r[f"{tgt}_prec_if_{v}"] = round(float(y[mm].mean()), 4) if mm.any() else None
                    r[f"n_{tgt}_if_{v}"] = int(mm.sum())
        r["med_borg"] = float(np.nanmedian(B[f].values)) if np.isfinite(B[f].values.astype(float)).any() else None
        s = F[F.set8.str.startswith("stt")][f].values.astype(float)
        r["med_stt"] = float(np.nanmedian(s)) if np.isfinite(s).any() else None
        rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "auc.csv", index=False)
    base = dict(n_gold=int(len(B)), n_gt=int(B.has_gt.sum()), n_keepbox=int(B.keepbox.sum()),
                n_both_eval=int((B.has_gt & B.keepbox).sum()),
                p_lab=round(float(B.y_lab[B.has_gt].mean()), 4), p_slot=round(float(B.y_slot[B.keepbox].mean()), 4),
                p_both=round(float(B.y_both[B.has_gt & B.keepbox].mean()), 4),
                simg_n=int(B.simg.sum()),
                simg_p_lab=round(float(B.y_lab[B.has_gt & (B.simg == 1)].mean()), 4),
                simg_p_slot=round(float(B.y_slot[B.keepbox & (B.simg == 1)].mean()), 4),
                simg_p_both=round(float(B.y_both[B.has_gt & B.keepbox & (B.simg == 1)].mean()), 4))
    json.dump(dict(base=base), open(OUT / "auc_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(base, ensure_ascii=False))
    pd.set_option("display.width", 250)
    print(A[["signal", "auc_lab", "auc_slot", "auc_both", "auc_lab_in_simg", "auc_slot_in_simg", "auc_both_in_simg", "med_borg",
             "med_stt"]].to_string(index=False))


if __name__ == "__main__":
    main()
