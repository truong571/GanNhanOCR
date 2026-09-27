"""compare_ref.py — đối chiếu tín hiệu gói tính lại với số tham chiếu của các phép đo gốc (chỉ dùng khi kiểm, không cần lúc chạy).

Tham chiếu: base_v2.pkl (chính sách v3, 118.954 ô: p_wood, viss, cert STT, ady/adx/vis_z, cờ hộp, luật A, r2_reason, ta),
masks_new.pkl (TN4 v2: H1n, SIMGo, TA_OK, slot_new -> "lai" = H1n ∧ SIMGo ∧ TA_OK), gold_tn4.pkl (one_char_ok, tall_new, ...).
Khác biệt nhỏ ở điểm liên tục là do nhúng lại trên MPS (thứ tự/lô khác) — báo số ô ĐỔI QUYẾT ĐỊNH để thấy tác động.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _num(a, b, m=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if m is not None:
        a, b = a[m], b[m]
    both = ~np.isnan(a) & ~np.isnan(b)
    d = np.abs(a[both] - b[both])
    return dict(n=int(len(a)), n_both=int(both.sum()), nan_mismatch=int((np.isnan(a) != np.isnan(b)).sum()),
                maxabs=float(d.max()) if len(d) else 0.0, frac_le_1e4=float((d <= 1e-4).mean()) if len(d) else 1.0)


def _bool(a, b, m=None):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    if m is not None:
        a, b = a[m], b[m]
    return dict(n=int(len(a)), mine=int(a.sum()), ref=int(b.sum()), only_mine=int((a & ~b).sum()), only_ref=int((~a & b).sum()))


def compare(X: pd.DataFrame, base_pkl, masks_pkl=None, tn4_pkl=None) -> dict:
    B = pd.read_pickle(base_pkl).set_index("cell_uid").reindex(X.cell_uid)
    R = {"n_ref_missing": int(B.book_set.isna().sum())}
    S8 = X.set8.values
    bb = lambda c: B[c].fillna(False).astype(bool).values
    R["p_wood_T"] = _num(X.p_wood_T, B.p_wood_T); R["p_wood_L"] = _num(X.p_wood_L, B.p_wood_L)
    R["viss_T_L16"] = _num(X.viss_T, B.viss_T, S8 == "L16")
    R["viss_L_TK"] = _num(X.viss_L, B.viss_L, S8 == "TK")
    R["viss_X_litho"] = _num(X.viss_X, B.viss_X, np.isin(S8, ["L83", "KVK"]))
    R["vis_m_win_glyph"] = _num(X.vis_m_win_glyph, B.vis_m_win_glyph)
    R["ady"] = _num(X.ady, B.ady); R["adx"] = _num(X.adx, B.adx)
    stt = np.isin(S8, ["stt2", "stt4", "stt11"])
    R["lobo_pT"] = _num(X.lobo_pT, B.lobo_pT, stt); R["lobo_pL"] = _num(X.lobo_pL, B.lobo_pL, stt)
    R["lobo_cert"] = _bool(X.lobo_cert.fillna(0).astype(int) == 1, B.lobo_cert_00015.fillna(0).astype(int).values == 1, stt)
    R["lobo_nh_eq"] = float((X.lobo_nh.fillna(-1).values[stt] == B.lobo_nh.fillna(-1).values[stt]).mean())
    mm = np.isin(X.book_set.values, ["SachThanhTruyen", "LucVanTien1883", "KimVanKieu1884"])
    R["m_hom"] = _num(X.m_hom, B.m_hom, mm)
    for mine, ref in (("int_foreign", "int_foreign"), ("rescue", "rescue"), ("similar", "similar"), ("weak_text", "weak_text"),
                      ("dup_bbox", "geo_f_dup_bbox"), ("ov_heavy", "geo_f_ov_heavy"), ("cnt", "geo_f_cnt_ocr_ne_qn")):
        R[mine] = _bool(X[mine].values, bb(ref))
    bcref = np.where(S8 == "L16", bb("bc_k06v1"), bb("bc_k05dxv05"))
    R["bc"] = _bool(X.bc.values, bcref)
    R["mocr_cons"] = _bool(X.mocr_cons.values[mm], bb("mocr_cons")[mm])
    R["mocr_gate"] = _bool(X.mocr.values, B.r2_reason.fillna("").values == "M_ocr_t50")
    R["ta_eq"] = float((X.ta.values == B.ta.fillna("na").values).mean())
    R["ta_diff_by_set"] = {s: int(((X.ta.values != B.ta.fillna("na").values) & (S8 == s)).sum()) for s in ("TK", "KVK", "L83")}
    if tn4_pkl is not None:
        T = pd.read_pickle(tn4_pkl).set_index("cell_uid").reindex(X.cell_uid)
        for c in ("one_char_ok", "tall_new", "f_blank", "f_cut", "f_two", "f_ink", "bleed_new", "trunc_new"):
            R[f"tn4_{c}"] = _bool(X[c].values, T[c].fillna(False).astype(bool).values)
    if masks_pkl is not None:
        M = pd.read_pickle(masks_pkl).set_index("cell_uid").reindex(X.cell_uid)
        for mine, ref in (("H1_new", "H1n"), ("SIMG", "SIMGo"), ("TA_OK", "TA_OK")):
            R[f"mask_{mine}"] = _bool(X[mine].values, M[ref].fillna(False).astype(bool).values)
        lai_ref = (M.H1n & M.SIMGo & M.TA_OK).fillna(False).astype(bool).values
        R["lai"] = _bool(X.lai.values, lai_ref)
        R["lai_by_set"] = {s: dict(mine=int((X.lai.values & (S8 == s)).sum()), tn4=int((lai_ref & (S8 == s)).sum()),
                                   only_mine=int((X.lai.values & ~lai_ref & (S8 == s)).sum()),
                                   only_tn4=int((~X.lai.values & lai_ref & (S8 == s)).sum()))
                           for s in ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK"]}
        R["_lai_ref_array"] = lai_ref
        R["_slot_new_ref"] = M.slot_new.fillna("").astype(str).values
    return R
