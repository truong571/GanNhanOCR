"""TN6 x10 — ước lượng (SUY ĐOÁN) lợi/hại vị trí của visual_dp trên STT bằng mô hình HIỆU CHUẨN trên Borg DungLy (có hộp người):
trên các ô mà hai bộ giải LỆCH hộp (IoU < 0,5), P(bộ giải A đúng | dấu của m_win_A − m_win_B) học trên DungLy (pitch vs visual_dp),
áp sang ô lệch của STT (legacy vs visual_dp, chỉ ô GOLD có m_win). Ra measure_out/_tn6/STT/x10_calib.json. 0 API.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[3]
D = REPO / "measure_out/_tn6"
K = ["page", "column", "nom_idx", "syl_idx"]


def iou(a, b):
    a, b = json.loads(a), json.loads(b)
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I)


# --- hiệu chuẩn trên DungLy (+ Kinh nếu có indirect)
cal = {}
rows = []
for book in ("SachDungLyHoThan", "SachKinhThayCaBinh"):
    fp, fv = D / book / "current/eval/indirect.pkl", D / book / "visual_dp/eval/indirect.pkl"
    if not (fp.exists() and fv.exists()):
        continue
    P, V = pd.read_pickle(fp), pd.read_pickle(fv)
    JP, JV = pd.read_pickle(D / book / "current/eval/join.pkl"), pd.read_pickle(D / book / "visual_dp/eval/join.pkl")
    for X in (P, V, JP, JV):
        for c in K:
            X[c] = X[c].astype(str)
    M = P.merge(V, on=K, suffixes=("_a", "_b"))
    M = M.merge(JP[K + ["slot_ok", "has_h", "keep_level"]], on=K).merge(JV[K + ["slot_ok"]], on=K, suffixes=("_a", "_b"))
    M = M[M.has_h & M.keep_level.isin(["keep_v5", "keep", "keep_high", "khong"]) & M.m_win_a.notna() & M.m_win_b.notna()]
    M["dis"] = [iou(x, y) < 0.5 for x, y in zip(M.bbox_a, M.bbox_b)]
    d = M[M.dis]
    rows.append(d.assign(book=book))
C = pd.concat(rows)
bwin = C.m_win_b > C.m_win_a
cal = dict(n=int(len(C)), p_b_right_given_bwin=round(float(C.slot_ok_b[bwin].mean()), 4),
           p_a_right_given_bwin=round(float(C.slot_ok_a[bwin].mean()), 4),
           p_b_right_given_awin=round(float(C.slot_ok_b[~bwin].mean()), 4),
           p_a_right_given_awin=round(float(C.slot_ok_a[~bwin].mean()), 4), frac_bwin=round(float(bwin.mean()), 4))
out = dict(calib_borg=cal, stt={})
for s in ("stt2", "stt4", "stt11"):
    P = pd.read_pickle(D / f"STT/current/eval/indirect_{s}.pkl"); V = pd.read_pickle(D / f"STT/visual_dp/eval/indirect_{s}.pkl")
    for X in (P, V):
        for c in K:
            X[c] = X[c].astype(str)
    M = P.merge(V, on=K, suffixes=("_a", "_b"))
    M = M[(M.tier_a == "GOLD") & M.m_win_a.notna() & M.m_win_b.notna()]
    M["dis"] = [iou(x, y) < 0.5 for x, y in zip(M.bbox_a, M.bbox_b)]
    d = M[M.dis]; bw = d.m_win_b > d.m_win_a
    # kỳ vọng số ô đúng trên phần LỆCH theo mô hình Borg
    e_b = bw.sum() * cal["p_b_right_given_bwin"] + (~bw).sum() * cal["p_b_right_given_awin"]
    e_a = bw.sum() * cal["p_a_right_given_bwin"] + (~bw).sum() * cal["p_a_right_given_awin"]
    out["stt"][s] = dict(n_gold=int(len(M)), frac_dis=round(float(M.dis.mean()), 4), n_dis=int(len(d)),
                         frac_vdp_win_in_dis=round(float(bw.mean()), 4) if len(d) else None,
                         exp_net_gain_pts=round(float((e_b - e_a) / max(1, len(M)) * 100), 3))
json.dump(out, open(D / "STT/x10_calib.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
