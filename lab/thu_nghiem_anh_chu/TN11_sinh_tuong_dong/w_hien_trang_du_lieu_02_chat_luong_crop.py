"""w_hien_trang_du_lieu_02_chat_luong_crop.py — CHẤT LƯỢNG NGUỒN CROP của ô neo: crop GỐC (dataset/<Bộ>/gold) so với crop CHUẨN (dataset/_ALL/crops_chuan, chỉ ô gold_exact=ok).

A. Toàn quần thể ô neo (cờ do pipeline ghi, không nhãn người): crop_quality_flag / stray_ink / border_ink / ink_pct (labels_trace.csv) và cờ hình học
   của gold_exact (f_blank f_cut f_two bleed_new trunc_new tall_new dup_bbox ov_heavy int_foreign one_char_ok) — tỉ lệ theo ô neo ok / không ok;
   "neo sạch hình học" = GOLD ∧ không cờ hình học nào ∧ crop_status ok (KHÔNG dùng gold_exact=ok, vì ok phụ thuộc bộ kiểm học từ nhãn người của sách khác).
B. Mẫu điểm ảnh (seed cố định): mô tả ảnh (chế độ, số mức xám, nền, kênh), và với ô neo ok: cùng một ô, crop gốc vs crop chuẩn — vuông?, kích thước, tỉ lệ mực,
   mực chạm mép, tỉ lệ mực của khối lớn nhất, lệch tâm mực, mực ngoài cửa sổ giữa (sau cùng bước style_image của p02).
C. Bất biến: crop_w/crop_h của trace == kích thước tệp; ink_pct ≈ tỉ lệ mực Otsu; md5 crop chuẩn == gold_exact.crop_chuan_md5; crop chuẩn 128 đúng 128×128.
Ra: measure_out/_tn11/full/hien_trang_du_lieu/chat_luong_crop.json. 0 API/GPU, CPU. Không mở ảnh bằng mắt/LLM.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_02_chat_luong_crop.py [--books B34 ...] [--n 300]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)
sys.path.insert(0, str(Path(__file__).parent))
import w_hien_trang_du_lieu_lib as L  # noqa: E402

GEO_FLAGS = ["f_blank", "f_cut", "f_two", "bleed_new", "trunc_new", "tall_new", "dup_bbox", "ov_heavy", "int_foreign"]
SEED = 20261004


def pmap(fn, items, w=8):
    with ThreadPoolExecutor(w) as ex:
        return list(ex.map(fn, items))


def style_image(g):
    """== p02_fd_phong_cach_sach.style_image: Otsu (mực đen / nền trắng) trong ô vuông lề 10 %, 128×128."""
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape
    s = int(max(h, w) * 1.2)
    out = np.full((s, s), 255, np.uint8)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = bw
    return cv2.resize(out, (128, 128), interpolation=cv2.INTER_AREA)


def ink_mask(g):
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = g <= t                  # ảnh 2 mức (STT: 0/255) cho t = 0 -> mực = mức 0
    if ink.mean() > 0.5:           # cực tính ngược (nền tối)
        ink = ~ink
    return ink


def metrics(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None or g.size == 0:
        return None
    h, w = g.shape
    ink = ink_mask(g)
    fi = float(ink.mean())
    n_ink = int(ink.sum())
    if n_ink == 0:
        return dict(w=w, h=h, fi=0.0, touch=0, big=np.nan, cx=np.nan, cy=np.nan, multi15=0, b2_tb=-1, b2_lr=-1)
    touch = int(ink[0, :].any()) + int(ink[-1, :].any()) + int(ink[:, 0].any()) + int(ink[:, -1].any())
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    ys, xs = np.nonzero(ink)
    multi15 = int((areas >= 0.15 * areas.sum()).sum() >= 2)     # >= 2 khối mực lớn (>= 15 % mực) — gợi ý có chữ/mảnh chữ khác lẫn vào (hoặc nét rời tự nhiên)
    b2_tb = b2_lr = -1
    if multi15:
        o = np.argsort(-areas)[:2] + 1
        t, hh_, l_, ww_ = (st[o[1], cv2.CC_STAT_TOP], st[o[1], cv2.CC_STAT_HEIGHT], st[o[1], cv2.CC_STAT_LEFT], st[o[1], cv2.CC_STAT_WIDTH])
        b2_tb = int(t == 0 or t + hh_ >= h); b2_lr = int(l_ == 0 or l_ + ww_ >= w)      # khối thứ hai chạm mép trên/dưới (mảnh chữ kề cùng cột) / trái-phải
    return dict(w=w, h=h, fi=fi, touch=touch, big=float(areas.max() / areas.sum()), cx=float(xs.mean() / w - 0.5), cy=float(ys.mean() / h - 0.5), multi15=multi15, b2_tb=b2_tb, b2_lr=b2_lr)


def describe(path):
    try:
        from PIL import Image
        im = Image.open(path)
        a = np.asarray(im)
    except Exception:  # noqa: BLE001
        return None
    gray = a if a.ndim == 2 else np.asarray(im.convert("L"))
    same = True if a.ndim == 2 else bool((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all())
    vals, cnt = np.unique(gray, return_counts=True)
    return dict(mode=im.mode, kenh_giong_nhau=same, so_muc_xam=int(len(vals)), nen=int(vals[np.argmax(cnt)]), min=int(gray.min()), max=int(gray.max()),
                tl_nen_pho_bien=float(cnt.max() / gray.size))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def agg(rows, keys):
    out = {}
    for k in keys:
        x = np.array([r[k] for r in rows if r and r.get(k) is not None], float)
        x = x[~np.isnan(x)]
        out[k] = dict(n=int(len(x)), mean=round(float(x.mean()), 4), median=round(float(np.median(x)), 4), p10=round(float(np.percentile(x, 10)), 4),
                      p90=round(float(np.percentile(x, 90)), 4)) if len(x) else {}
    return out


def book(ma, nsamp):
    t0 = time.time()
    C = L.CFG[ma]
    root_pub = L.REPO / "dataset" / C["ds"]
    inv = L.Inv()
    N = pd.read_csv(L.OUT / "neo" / f"{ma}.csv", dtype=str, keep_default_na=False)
    N = N[N.tier == "GOLD"].copy()                      # ô neo GOLD (có crop ở dataset/<Bộ>/)
    G = L.load_ge(ma)
    T = L.load_trace(ma)
    T = T[T.image.isin(set(N.image))].set_index("image")
    G = G.set_index("image")
    R = dict(ma=ma, n_neo_GOLD=int(len(N)))
    # ---- A: cờ pipeline
    N["gold_exact"] = N.image.map(G.gold_exact)
    for c in ["crop_quality_flag", "stray_ink", "border_ink", "ink_pct"]:
        N["tr_" + c] = N.image.map(T[c]) if c in T.columns else ""
    for f in GEO_FLAGS + ["one_char_ok", "crop_status"]:
        N["g_" + f] = N.image.map(G[f])
    geo = np.zeros(len(N), bool)
    for f in GEO_FLAGS:
        geo |= (N["g_" + f] == "1").to_numpy()
    geo |= (N.g_one_char_ok != "1").to_numpy()
    geo |= (N.g_crop_status != "ok").to_numpy()
    N["hinh_hoc_sach"] = ~geo
    R["A"] = dict(
        co_trong_trace=int(N.tr_crop_quality_flag.ne("").sum()),
        crop_quality_flag=N.tr_crop_quality_flag.value_counts().to_dict(),
        ti_le_co_hinh_hoc_theo_co={f: round(float((N["g_" + f] == "1").mean()), 4) for f in GEO_FLAGS},
        one_char_ok_0=round(float((N.g_one_char_ok != "1").mean()), 4), crop_status=N.g_crop_status.value_counts().to_dict(),
        n_hinh_hoc_sach=int(N.hinh_hoc_sach.sum()), ti_le_hinh_hoc_sach=round(float(N.hinh_hoc_sach.mean()), 4))
    for name, m in (("ok", N.gold_exact == "ok"), ("khong_ok", N.gold_exact != "ok"), ("tat_ca", np.ones(len(N), bool))):
        g = N[m]
        d = dict(n=int(len(g)), n_hinh_hoc_sach=int(g.hinh_hoc_sach.sum()), ti_le_hinh_hoc_sach=round(float(g.hinh_hoc_sach.mean()), 4) if len(g) else None)
        for c in ("stray_ink", "border_ink", "ink_pct"):
            x = pd.to_numeric(g["tr_" + c], errors="coerce").dropna()
            d[c] = dict(median=round(float(x.median()), 4), p90=round(float(x.quantile(0.9)), 4)) if len(x) else {}
        R["A_theo_nhom_" + name] = d
    old_bad = N.tr_crop_quality_flag.isin(["bleed", "truncated", "blank"]).to_numpy()
    new_bad = np.zeros(len(N), bool)
    for f in ["bleed_new", "trunc_new", "f_blank", "f_cut", "f_two", "tall_new"]:
        new_bad |= (N["g_" + f] == "1").to_numpy()
    R["A_co_cu_vs_moi"] = dict(co_cu_bleed_truncated_blank=int(old_bad.sum()), ti_le_cu=round(float(old_bad.mean()), 4), co_moi_bleed_trunc_blank_cut_two_tall=int(new_bad.sum()),
                               ti_le_moi=round(float(new_bad.mean()), 4), chi_cu=int((old_bad & ~new_bad).sum()), chi_moi=int((~old_bad & new_bad).sum()), ca_hai=int((old_bad & new_bad).sum()),
                               ghi_chu="cờ cũ = crop_quality_flag của labels_trace (crop gốc); cờ mới = cờ hình học gold_exact trên crop chuẩn dựng cho MỌI ô GOLD (không chỉ ok)")
    R["A_neo_sach_hinh_hoc_theo_gold_exact"] = N[N.hinh_hoc_sach].gold_exact.value_counts().to_dict()
    cs = N[N.hinh_hoc_sach].label.value_counts()
    R["A_neo_sach_hinh_hoc_chu"] = dict(so_chu=int(len(cs)), ge3=int((cs >= 3).sum()), ge5=int((cs >= 5).sum()))
    # ---- B: mẫu điểm ảnh
    rng = np.random.default_rng(SEED + sum(map(ord, ma)))
    pick = N.iloc[np.sort(rng.choice(len(N), size=min(nsamp, len(N)), replace=False))] if len(N) else N
    desc = pmap(describe, [str(L.REPO / p) for p in pick.duong_crop])
    ds_ = [d for d in desc if d]
    R["B_mo_ta_anh_gold"] = dict(n=len(ds_), mode=pd.Series([d["mode"] for d in ds_]).value_counts().to_dict(),
                                  kenh_RGB_giong_nhau=round(float(np.mean([d["kenh_giong_nhau"] for d in ds_])), 4),
                                  so_muc_xam_median=float(np.median([d["so_muc_xam"] for d in ds_])), nen_median=float(np.median([d["nen"] for d in ds_])),
                                  nen_p10_p90=[float(np.percentile([d["nen"] for d in ds_], 10)), float(np.percentile([d["nen"] for d in ds_], 90))],
                                  tl_nen_pho_bien_median=round(float(np.median([d["tl_nen_pho_bien"] for d in ds_])), 3),
                                  max_xam_median=float(np.median([d["max"] for d in ds_])), min_xam_median=float(np.median([d["min"] for d in ds_])))
    mg = pmap(metrics, [str(L.REPO / p) for p in pick.duong_crop])
    R["B_chi_so_crop_goc_mau_neo_GOLD"] = dict(n=len([m for m in mg if m]), **agg(mg, ["fi", "touch", "big", "cx", "cy", "multi15"]),
                                                 ti_le_cham_mep_ge1=round(float(np.mean([m["touch"] >= 1 for m in mg if m])), 4),
                                                 ti_le_cham_mep_ge3=round(float(np.mean([m["touch"] >= 3 for m in mg if m])), 4))
    # ---- B2: ô neo ok: gold vs chuẩn
    ok = N[(N.gold_exact == "ok") & (N.crop_chuan_ton_tai == "True")]
    R["B2_n_neo_ok_co_crop_chuan"] = int(len(ok))
    if len(ok):
        okp = ok.iloc[np.sort(rng.choice(len(ok), size=min(nsamp, len(ok)), replace=False))]
        pg = [str(L.REPO / p) for p in okp.duong_crop]
        pc = [str((root_pub / c).resolve()) for c in okp.crop_chuan_rel]
        mg_ = pmap(metrics, pg); mc_ = pmap(metrics, pc)
        dc = pmap(describe, pc)
        pairs = [(a, b) for a, b in zip(mg_, mc_) if a and b]
        gsq = np.array([abs(np.log(a["w"] / a["h"])) for a, _ in pairs]); csq = np.array([abs(np.log(b["w"] / b["h"])) for _, b in pairs])
        R["B2_gold_vs_chuan_cung_o"] = dict(
            n_cap=len(pairs),
            gold=dict(**agg([a for a, _ in pairs], ["w", "h", "fi", "touch", "big", "cx", "cy", "multi15"]), vuong_ti_le=round(float((gsq < 0.02).mean()), 4)),
            chuan=dict(**agg([b for _, b in pairs], ["w", "h", "fi", "touch", "big", "cx", "cy", "multi15"]), vuong_ti_le=round(float((csq < 0.02).mean()), 4)),
            ghi_chu_cham_mep="chạm mép của crop chuẩn = 0 THEO THIẾT KẾ (lề 10 % quanh hộp mực); crop gốc cắt sát mực (tighten_box) nên chạm mép là bình thường — không phải thước đo chất lượng",
            multi15_chi_gold=int(sum(a["multi15"] == 1 and b["multi15"] == 0 for a, b in pairs)), multi15_chi_chuan=int(sum(a["multi15"] == 0 and b["multi15"] == 1 for a, b in pairs)),
            multi15_ca_hai=int(sum(a["multi15"] == 1 and b["multi15"] == 1 for a, b in pairs)),
            multi15_chi_gold_khoi2_cham_mep_tren_duoi=int(sum(a["multi15"] == 1 and b["multi15"] == 0 and a["b2_tb"] == 1 for a, b in pairs)),
            multi15_chi_gold_khoi2_cham_mep_trai_phai=int(sum(a["multi15"] == 1 and b["multi15"] == 0 and a["b2_lr"] == 1 for a, b in pairs)),
            ti_le_cham_mep_ge1=dict(gold=round(float(np.mean([a["touch"] >= 1 for a, _ in pairs])), 4), chuan=round(float(np.mean([b["touch"] >= 1 for _, b in pairs])), 4)),
            ti_le_cham_mep_ge3=dict(gold=round(float(np.mean([a["touch"] >= 3 for a, _ in pairs])), 4), chuan=round(float(np.mean([b["touch"] >= 3 for _, b in pairs])), 4)),
            lech_tam_muc_abs_median=dict(gold=round(float(np.median([np.hypot(a["cx"], a["cy"]) for a, _ in pairs])), 4),
                                         chuan=round(float(np.median([np.hypot(b["cx"], b["cy"]) for _, b in pairs])), 4)),
            mo_ta_chuan=dict(mode=pd.Series([d["mode"] for d in dc if d]).value_counts().to_dict(), nen_median=float(np.median([d["nen"] for d in dc if d])),
                             so_muc_xam_median=float(np.median([d["so_muc_xam"] for d in dc if d]))))
        # C: md5 + 128
        sub = okp.iloc[:min(100, len(okp))]
        m5 = pmap(md5, [str((root_pub / c).resolve()) for c in sub.crop_chuan_rel])
        exp = [G.crop_chuan_md5.get(i, "") for i in sub.image]
        inv.check("md5_crop_chuan_khop_gold_exact_mau100", all(a == b for a, b in zip(m5, exp)), f"{sum(a != b for a, b in zip(m5, exp))} lệch/{len(sub)}")
        p128 = [str((root_pub / G.crop_chuan_128.get(i, "")).resolve()) for i in sub.image]
        sz = pmap(lambda p: (lambda g: None if g is None else g.shape)(cv2.imread(p, cv2.IMREAD_GRAYSCALE)), p128)
        inv.check("crop_chuan_128_dung_128x128_mau100", all(s == (128, 128) for s in sz), f"{sum(s != (128, 128) for s in sz)} sai/{len(sz)}")
    # C: trace crop_w/crop_h == tệp
    ch = pick.iloc[:min(300, len(pick))]
    wh = pmap(lambda p: (lambda g: None if g is None else (g.shape[1], g.shape[0]))(cv2.imread(str(L.REPO / p), cv2.IMREAD_GRAYSCALE)), list(ch.duong_crop))
    tw = [(int(float(T.crop_w.get(i, "nan"))), int(float(T.crop_h.get(i, "nan")))) if T.crop_w.get(i, "") not in ("", None) else None for i in ch.image]
    okc = [a == b for a, b in zip(wh, tw) if a is not None and b is not None]
    inv.check("trace_crop_w_h_khop_tep_mau300", len(okc) > 0 and np.mean(okc) >= 0.999, f"khớp {np.mean(okc):.4f} trên {len(okc)}")
    if len(pick):
        fi_me = np.array([m["fi"] for m in mg if m]); fi_tr = pd.to_numeric(pick.image.map(T.ink_pct), errors="coerce").to_numpy()[:len(fi_me)]
        okv = ~np.isnan(fi_tr)
        corr = float(np.corrcoef(fi_me[okv], fi_tr[okv])[0, 1]) if okv.sum() > 5 else float("nan")
        R["C_tuong_quan_ink_pct_trace_vs_Otsu"] = round(corr, 3)
        inv.check("ink_pct_trace_tuong_quan_voi_Otsu_ge_0.6", corr >= 0.6, f"r={corr:.3f}")
    R["bat_bien"] = inv.summary(); R["bat_bien_chi_tiet"] = inv.rows
    R["giay"] = round(time.time() - t0, 1)
    a2 = R.get("B2_gold_vs_chuan_cung_o")
    print(f"[02] {ma}: neo GOLD {len(N)} sạch hình học {R['A']['n_hinh_hoc_sach']} ({R['A']['ti_le_hinh_hoc_sach']}) | ok∩chuẩn {R['B2_n_neo_ok_co_crop_chuan']} | "
          + (f"chạm mép≥1 gold {a2['ti_le_cham_mep_ge1']['gold']} chuẩn {a2['ti_le_cham_mep_ge1']['chuan']}; multi15 chỉ-gold {a2['multi15_chi_gold']} chỉ-chuẩn {a2['multi15_chi_chuan']} cả hai {a2['multi15_ca_hai']} | " if a2 else "") + f"bất biến {R['bat_bien']['dat']}/{R['bat_bien']['tong']} [{R['giay']}s]", flush=True)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", default=L.ORDER)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    out = {}
    for ma in a.books:
        out[ma] = book(ma, a.n)
    p = L.OUT / ("chat_luong_crop.json" if a.books == L.ORDER else "chat_luong_crop_" + "_".join(a.books) + ".json")
    L.jdump(out, p)
    print("->", L.rel(p))


if __name__ == "__main__":
    main()
