"""w_hien_trang_du_lieu_02b_do_neo_nhan_nguoi.py — ĐỘ ĐÚNG của ô neo tự động, ĐO bằng nhãn người (B18, B34, L16, TK). NHÃN NGƯỜI CHỈ ĐỂ CHẤM.

Không dùng kết quả này để CHỌN từng ô/ảnh phong cách; chỉ để biết nguồn ô neo nào đáng tin cho cặp tinh chỉnh (mức bộ, không mức ô).
Cách đo: khoá ô (page, cột, nom_idx, syl_idx) của ô neo hiện tại nối với bảng TN8 (gt_char / pos_known / pos_ok dựng 30/09). Hợp lệ vì ở 4 bộ này
`w_hien_trang_du_lieu_03_tn8_cu.py` cho thấy KHOÁ Ô, THỨ TỰ HÀNG và HỘP không đổi (gt phụ thuộc khoá + hộp), nhãn ô neo giữ nguyên.
Đúng = V1+ (pipeline.gold_exact.common.var_eq_plus) giữa nhãn và chữ người cùng vị trí. Hai quy ước phủ GT: (a) pos_known (có hộp/GT người tại ô; B18/B34 chỉ ~57 %/41 % ô neo) -> dung_chu_pct_CI,
"hai vế" = đúng ∧ pos_ok; (b) chữ người theo vị trí câu (gt_char ≠ '', quy ước của TN8 `y` và của số "Borg 88–91 %" của báo cáo cũ) -> dung_chu_theo_cau_pct_CI.
CI 95 % = bootstrap CỤM TRANG (B = 1000).
Ra: measure_out/_tn11/full/hien_trang_du_lieu/do_neo_nhan_nguoi.json.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_02b_do_neo_nhan_nguoi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import w_hien_trang_du_lieu_lib as L  # noqa: E402

sys.path.insert(0, str(L.REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402  (chỉ để lấy lex() = Assets + set_lexicon; không ghi gì)

BOOKS = ["B18", "B34", "L16", "TK"]
GEO = ["f_blank", "f_cut", "f_two", "bleed_new", "trunc_new", "tall_new", "dup_bbox", "ov_heavy", "int_foreign"]


def boot(ok, pages, B=1000, seed=20261004):
    ok = np.asarray(ok, float); pages = np.asarray(pages)
    if len(ok) == 0:
        return [None, None, None]
    up, inv = np.unique(pages, return_inverse=True)
    k = np.bincount(inv, weights=ok, minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(up), size=(B, len(up)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return [round(float(ok.mean()) * 100, 2), round(float(np.quantile(r, 0.025)) * 100, 2), round(float(np.quantile(r, 0.975)) * 100, 2)]


def main():
    C = T.lex()
    out = {}
    for ma in BOOKS:
        N = pd.read_csv(L.OUT / "neo" / f"{ma}.csv", dtype=str, keep_default_na=False)
        G = L.load_ge(ma).set_index("image")
        b = pd.read_pickle(L.TN8 / "base" / f"{ma}.pkl").reset_index(drop=True)
        b["_k"] = (b.page + "|" + b.column.astype(int).astype(str) + "|" + b.nom_idx.astype(str) + "|" + b.syl_idx.astype(str)).to_numpy()
        N["_k"] = (N.page + "|" + N.column.astype(int).astype(str) + "|" + N.nom_idx.astype(str) + "|" + N.syl_idx.astype(str)).to_numpy()
        B = b.set_index("_k")
        inv = L.Inv()
        inv.check("khoa_o_neo_nam_trong_base_TN8", N._k.isin(B.index).all(), f"{(~N._k.isin(B.index)).sum()} ngoài")
        N["gt_nguoi"] = N._k.map(B.gt_char).fillna("")
        N["pos_known"] = N._k.map(B.pos_known).fillna(False).astype(bool)
        N["pos_ok"] = N._k.map(B.pos_ok).fillna(False).astype(bool)
        inv.check("nhan_neo_hien_tai_bang_nhan_base", (N.label == N._k.map(B.label)).all(), f"{(N.label != N._k.map(B.label)).sum()} khác")
        N["gold_exact"] = N.image.map(G.gold_exact).fillna("(khong_co)")
        geo = np.zeros(len(N), bool)
        for f in GEO:
            geo |= (N.image.map(G[f]).fillna("0") == "1").to_numpy()
        geo |= (N.image.map(G.one_char_ok).fillna("0") != "1").to_numpy()
        geo |= (N.image.map(G.crop_status).fillna("") != "ok").to_numpy()
        N["sach_hh"] = (~geo) & (N.tier == "GOLD")
        N["dung"] = [bool(g) and (l == g or C.var_eq_plus(g, l)) for l, g in zip(N.label, N.gt_nguoi)]
        N["hai_ve"] = N.dung & N.pos_ok
        subsets = {
            "tat_ca_neo_theo_dinh_nghia": np.ones(len(N), bool),
            "neo_tier_GOLD": (N.tier == "GOLD").to_numpy(),
            "neo_GOLD_sach_hinh_hoc": N.sach_hh.to_numpy(),
            "neo_GOLD_khong_qua_chon_chu": ((N.tier == "GOLD") & ~N.rule.str.contains("chon_chu:")).to_numpy(),
            "neo_GOLD_luat_thuan_khong_hau_to": ((N.tier == "GOLD") & ~N.rule.str.contains(r"\|")).to_numpy(),
            "neo_GOLD_qua_chon_chu": ((N.tier == "GOLD") & N.rule.str.contains("chon_chu:")).to_numpy(),
            "neo_gold_exact_ok": (N.gold_exact == "ok").to_numpy(),
            "neo_GOLD_khong_ok": ((N.tier == "GOLD") & (N.gold_exact != "ok")).to_numpy(),
            "neo_GOLD_text_only_tier": (N.tier == "GOLD_text_only").to_numpy(),
            "neo_tier_khac_GOLD_REVIEW_QUARANTINE": N.tier.isin(["REVIEW", "QUARANTINE"]).to_numpy(),
        }
        R = dict(ma=ma, n_neo=int(len(N)), n_neo_co_gt=int(N.pos_known.sum()))
        N["co_gt"] = N.gt_nguoi != ""
        for name, m in subsets.items():
            g = N[m]
            gk = g[g.pos_known]
            ga = g[g.co_gt]
            R[name] = dict(n=int(len(g)), n_co_gt=int(len(gk)), n_co_chu_nguoi_theo_cau=int(len(ga)),
                           dung_chu_pct_CI=boot(gk.dung, gk.page) if len(gk) else None,
                           dung_hai_ve_pct_CI=boot(gk.hai_ve, gk.page) if len(gk) else None,
                           dung_chu_theo_cau_pct_CI=boot(ga.dung, ga.page) if len(ga) else None,
                           chu_khac_nhau=int(g.label.nunique()), chu_ge5=int((g.label.value_counts() >= 5).sum()))
        R["bat_bien"] = inv.summary(); R["bat_bien_chi_tiet"] = inv.rows
        out[ma] = R
        a = R["tat_ca_neo_theo_dinh_nghia"]; g = R["neo_tier_GOLD"]; s = R["neo_GOLD_sach_hinh_hoc"]; k = R["neo_gold_exact_ok"]
        print(f"[02b] {ma}: neo {len(N)} có GT {R['n_neo_co_gt']} | đúng chữ: tất cả {a['dung_chu_pct_CI']}, GOLD {g['dung_chu_pct_CI']}, GOLD sạch hình học {s['dung_chu_pct_CI']}, ok {k['dung_chu_pct_CI']} "
              f"| bất biến {R['bat_bien']['dat']}/{R['bat_bien']['tong']}", flush=True)
    L.jdump(out, L.OUT / "do_neo_nhan_nguoi.json")
    print("->", L.rel(L.OUT / "do_neo_nhan_nguoi.json"))


if __name__ == "__main__":
    main()
