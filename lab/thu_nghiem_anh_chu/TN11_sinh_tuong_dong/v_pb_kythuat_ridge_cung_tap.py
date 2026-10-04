"""v_pb_kythuat (04/10) — PHẢN BIỆN độc lập: so f_ridge với f_fd và f_font trên CÙNG TẬP ứng viên (CPU, đọc pickle, 0 API, không sửa tệp).

Người kiểm chứng đã chỉ ra Top-1 f_fd (p01, điền -9) bị độ phủ làm sai lệch, NHƯNG khi bác "chữ hiếm 55–75 %" của báo cáo TN11 họ lại dùng chính số f_fd kiểu p01
cho chữ hiếm (B18 53,4; B34 45,7) để so với f_ridge (55,4; 40,9) — tức so Ridge với một con số đã bị "xổ số độ phủ" thổi lên. Script này làm phép so ĐÚNG:
chỉ ứng viên có đủ f_font, f_fd, f_ridge; ô mà chữ đúng có FD và >= 3 ứng viên như vậy; xếp hạng trong cùng danh sách; Top-1 một tín hiệu + CI bootstrap 95 % của chênh
f_ridge − f_fd và f_ridge − f_font; tách "tất cả" / "chữ hiếm" (n_self < 2, ngoài tập học của Ridge).
Ra: measure_out/_tn11/verify/pb_kythuat/ridge_cung_tap.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_pb_kythuat_ridge_cung_tap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CAND = REPO / "measure_out" / "_tn8" / "cand"
P01 = REPO / "measure_out" / "_tn11"
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_kythuat"
OUT.mkdir(parents=True, exist_ok=True)


def ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(n)]
    return [round(float(d.mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]


def pc(H, col):
    return H.loc[H.groupby("i")[col].idxmax()].set_index("i").y


def main():
    res = {}
    for b in ("B18", "B34", "L16", "TK"):
        F = pd.read_pickle(CAND / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font", "f_fd"]]
        S = pd.read_pickle(P01 / f"p01_{b}_scores.pkl")[["i", "c", "f_ridge", "f_shift"]]
        F = F.merge(S, on=["i", "c"], how="left")
        hy = F.groupby("i").y.max()
        F = F[F.i.isin(hy[hy == 1].index)]
        ns = F[F.y == 1].drop_duplicates("i").set_index("i").n_self.fillna(0)
        rare = set(ns[ns < 2].index)
        G = F[F.f_fd.notna() & F.f_font.notna() & F.f_ridge.notna()]
        cnt = G.groupby("i").size()
        ok = set(cnt[cnt >= 3].index) & set(G[G.y == 1].i)
        d = {}
        for tag, ids in (("tat_ca", ok), ("chu_hiem", ok & rare), ("chu_co_mau", ok - rare)):
            H = G[G.i.isin(ids)]
            if H.i.nunique() < 50:
                d[tag] = dict(o=int(H.i.nunique()), ghi_chu="quá ít ô")
                continue
            t = {c: pc(H, c) for c in ("f_font", "f_fd", "f_ridge", "f_shift")}
            r = dict(o=int(H.i.nunique()), ngau_nhien=round(float((1.0 / H.groupby("i").size()).mean()) * 100, 1))
            for c, v in t.items():
                r[c] = round(float(v.mean()) * 100, 1)
            r["ridge_tru_fd_CI95"] = ci((t["f_ridge"] - t["f_fd"]).to_numpy(float))
            r["ridge_tru_font_CI95"] = ci((t["f_ridge"] - t["f_font"]).to_numpy(float))
            d[tag] = r
        res[b] = d
        print(b, json.dumps(d, ensure_ascii=False), flush=True)
    (OUT / "ridge_cung_tap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
