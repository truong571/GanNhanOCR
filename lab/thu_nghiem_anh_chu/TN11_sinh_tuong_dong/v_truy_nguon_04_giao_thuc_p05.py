"""v_truy_nguon_04 (03/10) — MÔ PHỎNG GIAO THỨC p05 (bảng 10 bộ, Top-1 trong top-3, n ~ 20) BẰNG ĐẶC TRƯNG CÓ SẴN, trên TOÀN BỘ ô. 0 API, CPU, chỉ ĐỌC.

Mục đích: tách "đo cái gì" khỏi "đo bao nhiêu". p05_full_corpus_he_quy_chieu.py (bản của phiên khác) làm:
  1. chọn ngẫu nhiên n=20 ô (rng = default_rng(hash(book) % 100000)) trong các ô có "sự thật" (người: y==1; 6 bộ còn lại: sự thật := D.label
     của chính pipeline, mọi tầng có nhãn khác rỗng);
  2. giữ top-3 ứng viên theo f_vW (CNN kiểm, học nhãn người sách khác), bỏ ô mà chữ đúng không nằm trong top-3;
  3. glyph = render PIL của NomNaTong-Regular.ttf (KHÔNG có phông dự phòng Plangothic như pipeline) -> chữ thiếu glyph = ẢNH TRẮNG;
  4. Top-1 = ứng viên có cos(crop, glyph) lớn nhất (hòa: ứng viên đứng trước theo f_vW, vì dict chèn theo thứ tự đã sắp).
Script này mô phỏng các bước 1-3 bằng bảng ứng viên đã lưu (measure_out/_tn8/cand) và đo:
  - sự thật của 6 bộ không nhãn người gồm tầng nào;  - độ giữ lại của bước 2;  - tỉ lệ ứng viên thiếu glyph NomNaTong và các tình huống
    (a) chữ đúng có glyph + 2 nhiễu trắng, (b) 1 nhiễu trắng, (c) không nhiễu trắng, (d) chữ đúng thiếu glyph;
  - mức "đoán ngẫu nhiên" thực của giao thức (1/3 chỉ đúng khi cả 3 ứng viên có glyph) và "luật ngốc" chỉ biết ứng viên nào CÓ glyph;
  - Top-1 của f_font (có phông dự phòng) trong top-3 vs trong ~26 ứng viên trên CÙNG các ô; độ rộng phân bố mẫu n=20;
  - tính tái lập: hash(book) % 100000 qua 3 tiến trình Python.
Ra: measure_out/_tn11/verify/truy_nguon/giao_thuc_p05.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "truy_nguon"
OUT.mkdir(parents=True, exist_ok=True)
HUMAN = ("B18", "B34", "L16", "TK")
K = 3


def rate(ok, pages):
    m, lo, hi = T.boot_ci_pages(np.asarray(ok, float), pages)
    return dict(p=round(100 * m, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2), n=int(len(ok)))


def main():
    cm = set(TTFont(str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")).getBestCmap())
    has_glyph = lambda c: len(c) == 1 and ord(c) in cm          # noqa: E731
    p05 = {r["book"]: r for r in json.loads((REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
    res = {}
    for b in T.ORDER:
        D = T.load_base(b)
        F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font", "f_fd", "f_self", "f_vW"]]
        human = b in HUMAN
        if human:
            F["ytruth"] = F.y.values.astype(int)
        else:
            lbl = D.label.values[F.i.values]
            F["ytruth"] = ((F.c.values == lbl) & (lbl != "")).astype(int)
        hy = F.groupby("i").ytruth.max()
        tcells = hy[hy == 1].index.values                               # "target_cells" của p05
        F = F[F.i.isin(tcells)].copy()
        F["has"] = F.c.map(has_glyph)
        # top-3 theo f_vW (thứ tự p05: sort_values(["i","rk"], [True, False]) rồi head(3); f_vW NaN -> -9)
        F["rk"] = F.f_vW.fillna(-9)
        S = F.sort_values(["i", "rk"], ascending=[True, False], kind="stable").groupby("i").head(K)
        has_truth = S.groupby("i").ytruth.max()
        valid = has_truth[has_truth == 1].index.values
        S = S[S.i.isin(valid)].copy()
        r = dict(bo=b, sach_nhan_nguoi=human, o_co_su_that=int(len(tcells)), o_giu_lai_sau_top3=int(len(valid)),
                 giu_lai_pct=round(100 * len(valid) / max(1, len(tcells)), 2), n_p05=p05[b]["n_cells"])
        if not human:
            tier = D.tier.values[valid]
            r["tang_cua_o_su_that_p05"] = {t: int((tier == t).sum()) for t in sorted(set(tier))}
            r["ty_le_GOLD_trong_o_su_that_pct"] = round(100 * float((tier == "GOLD").mean()), 2)
        # tình huống glyph trong top-3
        S = S.sort_values(["i", "rk"], ascending=[True, False], kind="stable")
        g = S.groupby("i")
        ncand = g.size()
        n_blank = g.has.apply(lambda s: int((~s).sum()))
        truth_has = S[S.ytruth == 1].drop_duplicates("i").set_index("i").has
        sit = pd.DataFrame(dict(nc=ncand, nb=n_blank, th=truth_has)).dropna()
        nb_dis = np.where(sit.th, sit.nb, sit.nb - 1)                # số nhiễu trắng
        a = sit.th & (nb_dis == sit.nc - 1)                          # chữ đúng có glyph, MỌI nhiễu trắng
        bsit = sit.th & (nb_dis > 0) & ~a
        c_ = sit.th & (nb_dis == 0)
        d_ = ~sit.th
        r["tinh_huong_top3_pct"] = dict(a_moi_nhieu_trang=round(100 * float(a.mean()), 2), b_mot_phan_nhieu_trang=round(100 * float(bsit.mean()), 2),
                                        c_khong_nhieu_trang=round(100 * float(c_.mean()), 2), d_chu_dung_thieu_glyph=round(100 * float(d_.mean()), 2))
        r["ty_le_ung_vien_top3_thieu_glyph_pct"] = round(100 * float((~S.has).mean()), 2)
        r["ty_le_ung_vien_nhieu_thieu_glyph_pct"] = round(100 * float((~S[S.ytruth == 0].has).mean()), 2)
        # mức đoán ngẫu nhiên: đều (1/n) vs "luật ngốc" (chọn ngẫu nhiên trong ứng viên CÓ glyph; nếu không có thì ứng viên đầu theo f_vW)
        chance_u = (1.0 / sit.nc).mean()
        dumb = []
        for (i, grp) in S.groupby("i"):
            gh = grp[grp.has]
            if len(gh):
                dumb.append(float(gh.ytruth.mean()))
            else:
                dumb.append(float(grp.ytruth.iloc[0]))
        r["doan_ngau_nhien_deu_pct"] = round(100 * float(chance_u), 2)
        r["luat_ngoc_chi_biet_co_glyph_pct"] = round(100 * float(np.mean(dumb)), 2)
        # Top-1 của f_font (phông dự phòng) trong top-3 vs trong mọi ứng viên, trên CÙNG ô giữ lại
        pg = D.page.values[valid]
        for m in ("f_font", "f_fd", "f_self", "f_vW"):
            x = S[["i", m, "ytruth"]].copy(); x[m] = x[m].fillna(-9)
            t3 = x.loc[x.groupby("i")[m].idxmax()].set_index("i").ytruth.reindex(valid).values
            Fa = F[F.i.isin(valid)][["i", m, "ytruth"]].copy(); Fa[m] = Fa[m].fillna(-9)
            ta = Fa.loc[Fa.groupby("i")[m].idxmax()].set_index("i").ytruth.reindex(valid).values
            r[f"top1_{m}"] = dict(top3=rate(t3, pg), tat_ca_ung_vien=rate(ta, pg))
        # phân bố mẫu n = 20 (giao thức p05) cho f_font trong top-3
        x = S[["i", "f_font", "ytruth"]].copy(); x["f_font"] = x.f_font.fillna(-9)
        t3 = x.loc[x.groupby("i").f_font.idxmax()].set_index("i").ytruth.reindex(valid).values.astype(float)
        rng = np.random.default_rng(20261003)
        draws = np.array([t3[rng.choice(len(t3), size=min(20, len(t3)), replace=False)].mean() for _ in range(5000)]) * 100
        r["mau_n20_f_font_top3"] = dict(trung_binh=round(float(draws.mean()), 1), sd=round(float(draws.std()), 1),
                                        p2_5=round(float(np.quantile(draws, .025)), 1), p97_5=round(float(np.quantile(draws, .975)), 1),
                                        min=round(float(draws.min()), 1), max=round(float(draws.max()), 1))
        # (chỉ bộ người) so với sự thật người vs nhãn pipeline trên cùng ô
        if human:
            lbl = D.label.values
            yl = np.array([lbl[i] == c for i, c in zip(S.i.values, S.c.values)])
            S["ylabel"] = yl.astype(int)
            both = S.groupby("i").ylabel.max()
            cells_l = both[both == 1].index.values
            xs = S[S.i.isin(cells_l)][["i", "f_font", "ytruth", "ylabel"]].copy(); xs["f_font"] = xs.f_font.fillna(-9)
            top = xs.loc[xs.groupby("i").f_font.idxmax()].set_index("i")
            r["tren_o_co_nhan_pipeline_trong_top3"] = dict(
                n=int(len(top)), top1_f_font_vs_nhan_pipeline_pct=round(100 * float(top.ylabel.mean()), 2),
                top1_f_font_vs_chu_nguoi_pct=round(100 * float(top.ytruth.mean()), 2))
        res[b] = r
        print(f"\n== {b}{' [nhãn người]' if human else ' [sự thật := nhãn pipeline]'}: ô có sự thật {r['o_co_su_that']}; giữ sau top-3 {r['o_giu_lai_sau_top3']} ({r['giu_lai_pct']} %); n(p05)={r['n_p05']}")
        if not human:
            print(f"   tầng của ô 'sự thật' p05: {r['tang_cua_o_su_that_p05']}  (GOLD {r['ty_le_GOLD_trong_o_su_that_pct']} %)")
        print(f"   ứng viên top-3 thiếu glyph NomNaTong {r['ty_le_ung_vien_top3_thieu_glyph_pct']} % (nhiễu {r['ty_le_ung_vien_nhieu_thieu_glyph_pct']} %); tình huống {r['tinh_huong_top3_pct']}")
        print(f"   đoán ngẫu nhiên đều {r['doan_ngau_nhien_deu_pct']} % | luật ngốc (chỉ biết có glyph) {r['luat_ngoc_chi_biet_co_glyph_pct']} %")
        for m in ("f_font", "f_fd", "f_self", "f_vW"):
            v = r[f"top1_{m}"]
            print(f"   Top-1 {m:7s} trong top-3 {v['top3']['p']:6.2f} [{v['top3']['lo']:5.2f},{v['top3']['hi']:5.2f}]  | trong mọi ứng viên (cùng ô) {v['tat_ca_ung_vien']['p']:6.2f}")
        print(f"   mẫu n=20 (f_font top-3): {r['mau_n20_f_font_top3']}")
        if human:
            print(f"   trên ô có nhãn pipeline trong top-3: {r['tren_o_co_nhan_pipeline_trong_top3']}")
    # tái lập hash
    hs = []
    for _ in range(3):
        o = subprocess.run([sys.executable, "-c", "print(hash('B18') % 100000, hash('TK') % 100000)"], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "random"})
        hs.append(o.stdout.strip())
    res["_hash_book_mod_100000_ba_tien_trinh_B18_TK"] = hs
    print("\nhash(book)%100000 (B18, TK) qua 3 tiến trình:", hs)
    (OUT / "giao_thuc_p05.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
