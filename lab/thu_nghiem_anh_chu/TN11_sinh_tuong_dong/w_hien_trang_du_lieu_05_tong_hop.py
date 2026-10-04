"""w_hien_trang_du_lieu_05_tong_hop.py — GỘP kết quả 01–04 thành ket_qua.json + dựng ảnh phong cách ứng viên (chọn TỰ ĐỘNG, không nhãn người, không trạng thái gold_exact).

Tập ô neo KHUYẾN NGHỊ cho gói Kaggle = ô neo (định nghĩa đề bài) ∧ tier == GOLD ∧ không cờ hình học (f_blank f_cut f_two bleed_new trunc_new tall_new dup_bbox ov_heavy int_foreign,
one_char_ok == 1 [signals_geom.py:262: crop_status ok ∧ ¬f_blank ∧ ¬f_cut ∧ ¬f_two ∧ ¬f_ink ∧ ¬bleed_new ∧ ¬trunc_new], crop_status == ok) ∧ cạnh ngắn crop ≥ 24 px. Ảnh phong cách ứng viên: trong tập đó, mỗi chữ một ô, kích thước gần trung vị của sách (±25 %), tỉ lệ mực trong
[p25, p75], mực lạc ≤ trung vị, chữ có ≥ 5 ô neo sạch; xếp theo khoảng cách chuẩn hoá tới (rộng, cao, tỉ lệ mực) trung vị; lấy 12.
Ra: measure_out/_tn11/full/hien_trang_du_lieu/{ket_qua.json, style_ung_vien/<ma>.csv, neo_khuyen_nghi/<ma>.csv}. 0 API/GPU.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_05_tong_hop.py
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import w_hien_trang_du_lieu_lib as L  # noqa: E402

GEO = ["f_blank", "f_cut", "f_two", "bleed_new", "trunc_new", "tall_new", "dup_bbox", "ov_heavy", "int_foreign"]
NSTYLE = 12
MIN_SIDE = 24


def J(name):
    return json.load(open(L.OUT / name, encoding="utf-8"))


CACHE = L.REPO / "prepared" / "_gold_exact" / "crop_chuan"


def uid_path(uid):          # == pipeline.gold_exact.common.uid_path
    a = uid.split("/")
    return "/".join(a[:3]) + "/" + "_".join(a[3:]) + ".png"


def clean_flags(N, G):
    geo = np.zeros(len(N), bool)
    for f in GEO:
        geo |= (N.image.map(G[f]).fillna("0") == "1").to_numpy()
    geo |= (N.image.map(G.one_char_ok).fillna("0") != "1").to_numpy()
    geo |= (N.image.map(G.crop_status).fillna("") != "ok").to_numpy()
    return ~geo


def cnt_stats(labels):
    c = collections.Counter(labels)
    return dict(so_chu=len(c), chu_ge3=int(sum(v >= 3 for v in c.values())), chu_ge5=int(sum(v >= 5 for v in c.values())),
                ti_le_o_thuoc_chu_lt2=round(float(np.mean([c[x] < 2 for x in labels])), 4) if len(labels) else None)


def main():
    t0 = time.time()
    K, Q, H, T8, P3, RR = J("kiem_ke.json"), J("chat_luong_crop.json"), J("do_neo_nhan_nguoi.json"), J("tn8_cu.json"), J("neo_truoc_chon_chu.json"), J("rui_ro.json")
    CCH = J("cache_crop_chuan.json")
    CCV = J("cache_crop_chuan_xac_minh.json") if (L.OUT / "cache_crop_chuan_xac_minh.json").exists() else None
    inv = L.Inv()
    (L.OUT / "style_ung_vien").mkdir(exist_ok=True); (L.OUT / "neo_khuyen_nghi").mkdir(exist_ok=True)
    tong = collections.Counter()
    sach = {}
    pooled = []
    for ma in L.ORDER:
        C = L.CFG[ma]
        N = pd.read_csv(L.OUT / "neo" / f"{ma}.csv", dtype=str, keep_default_na=False)
        G = L.load_ge(ma).set_index("image")
        Ng = N[(N.tier == "GOLD") & (N.trong_pub == "True")].copy()
        Ng["w"] = Ng.w.astype(float); Ng["h"] = Ng.h.astype(float)
        for c in ("ink_pct", "stray_ink", "border_ink"):
            Ng[c] = pd.to_numeric(Ng[c], errors="coerce")
        cl = clean_flags(Ng, G)
        Ng["sach_hinh_hoc"] = cl
        Ng["canh_ngan"] = np.minimum(Ng.w, Ng.h)
        Rk = Ng[Ng.sach_hinh_hoc & (Ng.canh_ngan >= MIN_SIDE)].copy()
        Rk["cell_uid"] = Rk.image.map(G.cell_uid)
        Rk["crop_chuan_cache"] = [L.rel(CACHE / "sq" / uid_path(u)) for u in Rk.cell_uid]
        Rk["crop_chuan_cache_128"] = [L.rel(CACHE / "sq128" / uid_path(u)) for u in Rk.cell_uid]
        ex_c = np.array([os.path.isfile(L.REPO / p) for p in Rk.crop_chuan_cache], bool)
        Rk["cache_ton_tai"] = ex_c
        Rk.drop(columns=["sach_hinh_hoc"]).to_csv(L.OUT / "neo_khuyen_nghi" / f"{ma}.csv", index=False)
        st = cnt_stats(list(Rk.label))
        pooled.append(Rk.assign(ma=ma))
        # ảnh phong cách ứng viên
        cc = collections.Counter(Rk.label)
        hm, wm = Rk.h.median(), Rk.w.median()
        q1, q3 = Rk.ink_pct.quantile(0.25), Rk.ink_pct.quantile(0.75)
        sm = Rk.stray_ink.median()
        S = Rk[(Rk.h.between(0.8 * hm, 1.25 * hm)) & (Rk.w.between(0.8 * wm, 1.25 * wm)) & (Rk.ink_pct.between(q1, q3)) & ((Rk.stray_ink <= sm) | Rk.stray_ink.isna())
               & (Rk.label.map(cc) >= 5)].copy()
        iqr = max(q3 - q1, 1e-6)
        S["diem"] = (S.h / hm - 1).abs() + (S.w / wm - 1).abs() + (S.ink_pct - Rk.ink_pct.median()).abs() / iqr
        S = S.sort_values(["diem", "image"]).drop_duplicates("label").head(NSTYLE).copy()
        S.insert(0, "hang", range(1, len(S) + 1))
        S["co_crop_chuan"] = S.image.map(G.crop_chuan).fillna("") != ""
        S["crop_chuan"] = [L.rel((L.REPO / "dataset" / C["ds"] / c).resolve()) if c else "" for c in S.image.map(G.crop_chuan).fillna("")]
        S["crop_chuan_cache_128"] = S.crop_chuan_cache_128
        S[["hang", "ma", "label", "duong_crop", "crop_chuan_cache", "crop_chuan_cache_128", "cache_ton_tai", "crop_chuan", "co_crop_chuan", "w", "h", "ink_pct", "stray_ink", "border_ink", "page", "column", "nom_idx", "syl_idx", "diem"]].to_csv(
            L.OUT / "style_ung_vien" / f"{ma}.csv", index=False)
        inv.check(f"{ma}_du_{NSTYLE}_anh_phong_cach_ung_vien", len(S) == NSTYLE, f"{len(S)}")
        inv.check(f"{ma}_anh_phong_cach_ton_tai", all((L.REPO / p).is_file() for p in S.duong_crop) and bool(S.cache_ton_tai.all()), "")
        inv.check(f"{ma}_neo_khuyen_nghi_co_du_crop_chuan_nho_dem", int(ex_c.sum()) == len(Rk), f"{int(ex_c.sum())}/{len(Rk)}")
        # thống kê theo tầng/định nghĩa
        k = K[ma]; n = k["neo"]; ge = k["gold_exact"]; cu = k["chu_ung_vien"]
        kt = k["kich_thuoc_crop"]["GOLD"]; mo = Q[ma]["B_mo_ta_anh_gold"]
        small = int((Ng.canh_ngan < MIN_SIDE).sum())
        extra_bad = {t: int(c) for t, c in n["co_tep_crop_theo_tang"].items() if t != "GOLD"}
        by_gold = int(sum(os.path.getsize(L.REPO / p) for p in Rk.duong_crop))
        by_cache = int(sum(os.path.getsize(L.REPO / p) for p, e in zip(Rk.crop_chuan_cache, ex_c) if e))
        by_cache128 = int(sum(os.path.getsize(L.REPO / p) for p, e in zip(Rk.crop_chuan_cache_128, ex_c) if e))
        okr = Rk[Rk.image.map(G.crop_chuan).fillna("") != ""]
        by_chuan = int(sum(os.path.getsize((L.REPO / "dataset" / C["ds"] / G.crop_chuan[i]).resolve()) for i in okr.image))
        rec = dict(
            n=int(len(Rk)), byte_crop_goc=by_gold, byte_crop_chuan_cua_o_ok_trong_tap=by_chuan, co_crop_chuan_nho_dem=int(ex_c.sum()), byte_crop_chuan_nho_dem=by_cache,
            byte_crop_chuan_nho_dem_128=by_cache128, **st, loai_do_hinh_hoc=int((~Ng.sach_hinh_hoc).sum()), loai_do_canh_ngan_lt_24=int(small),
            ok_trong_tap_khuyen_nghi=int((Rk.image.map(G.gold_exact) == "ok").sum()), co_crop_chuan_trong_tap_khuyen_nghi=int((Rk.image.map(G.crop_chuan).fillna("") != "").sum()),
            file=f"measure_out/_tn11/full/hien_trang_du_lieu/neo_khuyen_nghi/{ma}.csv", style_file=f"measure_out/_tn11/full/hien_trang_du_lieu/style_ung_vien/{ma}.csv",
            style_tieu_chi=f"mỗi chữ 1 ô, rộng/cao trong ±25 % trung vị ({wm:.0f}×{hm:.0f}), ink_pct trong [p25,p75]=[{q1:.3f},{q3:.3f}], stray_ink ≤ trung vị, chữ ≥ 5 ô neo sạch; lấy {NSTYLE}")
        tr = Q[ma]["A"]
        sach[ma] = dict(
            ma=ma, ten=C["name"], loai=C["kind"], crop_source=C["crop_source"],
            nguon=k["nguon"], mtime=k["mtime"], so_trang=k["so_trang_png"],
            so_dong_pub=k["so_dong_pub"], so_dong_all=k["so_dong_all"], tang_pub=k["tang_pub"], tang_all=k["tang_all"],
            so_o_co_anh_pub=k["so_o_co_anh_pub"], so_o_co_anh_all=k["so_o_co_anh_all"], crop_thieu_pub=k["crop_thieu_pub"], crop_thieu_pub_theo_tang=k["crop_thieu_pub_theo_tang"],
            kich_thuoc_crop_GOLD=dict(w=kt["w"], h=kt["h"], ti_le_w_tren_h_median=kt["ti_le_w_tren_h_median"], mode=kt["mode"]),
            mo_ta_anh=dict(mode=mo["mode"], kenh_RGB_giong_nhau=mo["kenh_RGB_giong_nhau"], so_muc_xam_median=mo["so_muc_xam_median"], nen_median=mo["nen_median"],
                           nen_p10_p90=mo["nen_p10_p90"], max_xam_median=mo["max_xam_median"], min_xam_median=mo["min_xam_median"]),
            neo=dict(dinh_nghia=n["dinh_nghia"], tong_theo_dinh_nghia=n["tong_theo_dinh_nghia"], co_tep_crop=n["co_tep_crop"], theo_tang=n["co_tep_crop_theo_tang"],
                     so_chu=n["so_chu_khac_nhau"], chu_ge3=n["chu_ge3"], chu_ge5=n["chu_ge5"], ti_le_o_GOLD_thuoc_chu_hiem_lt2=n["ti_le_o_GOLD_thuoc_chu_hiem_lt2"],
                     ti_le_o_moi_tang_theo_chu_kim_hiem_lt2=n["ti_le_o_moi_tang_theo_chu_kim_hiem_lt2"], hau_to_luat=n["hau_to_luat"], thuan_khong_hau_to=n["thuan_khong_hau_to"],
                     qua_chon_chu=n["qua_chon_chu"], qua_gate_qn_count_unfixed=n["qua_gate_qn_count_unfixed"], qua_gate_crop_bad=n["qua_gate_crop_bad"]),
            gold_exact=dict(file=k["nguon"]["gold_exact"], trang_thai=ge["trang_thai"], evidence_level=ge["evidence_level"], o_neo_theo_trang_thai=ge["o_neo_theo_trang_thai"],
                            neo_ok_n=ge["neo_ok_n"], neo_ok_so_chu=ge["neo_ok_so_chu"], neo_ok_chu_ge3=ge["neo_ok_chu_ge3"], neo_ok_chu_ge5=ge["neo_ok_chu_ge5"],
                            o_ok_co_crop_chuan=ge["o_ok_co_crop_chuan_ton_tai"]),
            chu_ung_vien=cu, chu_ung_vien_file=f"measure_out/_tn11/full/hien_trang_du_lieu/chu_ung_vien/{ma}.txt (+ .tsv: số ô trong R, số ô kim đọc, số ô neo)",
            chat_luong=dict(neo_GOLD=tr and Q[ma]["n_neo_GOLD"], hinh_hoc_sach=tr["n_hinh_hoc_sach"], ti_le_hinh_hoc_sach=tr["ti_le_hinh_hoc_sach"],
                            co_cu_vs_moi=Q[ma]["A_co_cu_vs_moi"], gold_vs_chuan_cung_o=Q[ma].get("B2_gold_vs_chuan_cung_o")),
            do_bang_nhan_nguoi_CHI_CHAM=H.get(ma), khuyen_nghi_tap_neo=rec,
            neo_truoc_chon_chu=P3.get(ma), tn8=T8[ma], rui_ro=RR["sach"][ma],
            evaluation_only=RR["evaluation_only"].get(ma), neo_khac_GOLD=extra_bad)
        tong["so_dong_pub"] += k["so_dong_pub"]; tong["o_co_anh_pub"] += k["so_o_co_anh_pub"]; tong["neo_co_crop"] += n["co_tep_crop"]
        tong["neo_khuyen_nghi"] += len(Rk); tong["chu_ung_vien_theo_sach"] += cu["so_chu"]; tong["chu_ung_vien_neo_lt2"] += cu["so_chu_neo_lt2"]
        tong["chu_ung_vien_neo_lt5"] += cu["so_chu"] - cu["so_chu_neo_ge5"]; tong["neo_ok"] += ge["neo_ok_n"]
        print(f"[05] {ma}: khuyến nghị {len(Rk)} ô neo, {st['so_chu']} chữ (≥3 {st['chu_ge3']}, ≥5 {st['chu_ge5']}), loại hình học {int((~Ng.sach_hinh_hoc).sum())}, cạnh<24 {small}; style {len(S)}", flush=True)
    # ---- STT: gộp 3 quyển (so sánh)
    PL = pd.concat(pooled)
    stt = PL[PL.ma.isin(["stt2", "stt4", "stt11"])]
    per = {m: cnt_stats(list(stt[stt.ma == m].label)) for m in ["stt2", "stt4", "stt11"]}
    pool_c = collections.Counter(stt.label)
    stt_info = dict(
        thu_muc=dict(
            gop_hien_hanh="dataset/SachThanhTruyen/ — bộ GỘP 3 quyển (labels.csv 62.108 dòng = GOLD 51.891 + SYLLABLE 10.217; gold_exact.csv 51.891; labels_trace.csv; crop gold/ và syllable/); cột `book` ∈ {stt2, stt4, stt11} TÁCH quyển; cột `page` KHÔNG tách (trang trùng tên giữa quyển)",
            mot_luot_day_du="dataset_out/labels_final.csv (83.542 dòng MỌI tầng; cột book) + dataset_out/{gold,syllable}/ — bản dựng đầy đủ 01/10 19:27, nguồn của dataset/SachThanhTruyen",
            ban_tach_cu="dataset/SachThanhTruyen2/, SachThanhTruyen4/, SachThanhTruyen11/ — bản TÁCH CŨ 23/09 (trước visual_dp + bản chặt TN9): không gold_exact.csv, không labels_trace.csv; KHÔNG dùng",
            trang="prepared/SachThanhTruyen{2,4,11}/pages (nhị phân 2 mức, nguồn crop) + pages_denoised; kim: prepared/SachThanhTruyen{2,4,11}/kim_l1skel_l2 (lt1+lt2)"),
        cach_tach_quyen=("lọc cột `book` (stt2/stt4/stt11) của labels.csv/labels_final.csv, hoặc `set8` của gold_exact.csv; đường ảnh có tiền tố quyển (gold/stt2_page_0012_c01_000.png) nên đường ảnh duy nhất; "
                         "KHÔNG khoá theo (page, cột, nom_idx, syl_idx) vì lẫn quyển"),
        so_dong_pub={m: K[m]["so_dong_pub"] for m in ["stt2", "stt4", "stt11"]}, so_dong_all={m: K[m]["so_dong_all"] for m in ["stt2", "stt4", "stt11"]},
        tang_pub={m: K[m]["tang_pub"] for m in ["stt2", "stt4", "stt11"]},
        rui_ro_lan_quyen=RR["STT"], neo_khuyen_nghi_tung_quyen=per,
        bang_TN8_STT_la_ban_cu_23_09=dict((q, dict(gold_base_tn8=T8[q]["tang_base"].get("GOLD"), gold_thu_muc_cu_23_09=RR["STT"]["thu_muc_STT_cu_23_09"][q]["tang_cu"].get("GOLD"),
                                                  gold_hien_tai=K[q]["tang_pub"].get("GOLD"))) for q in ["stt2", "stt4", "stt11"]),
        neo_khuyen_nghi_neu_GOP_3_quyen=dict(so_chu=len(pool_c), chu_ge3=int(sum(v >= 3 for v in pool_c.values())), chu_ge5=int(sum(v >= 5 for v in pool_c.values())), n=int(len(stt)),
                                              ghi_chu="chỉ để so sánh; phong cách từng quyển khác nhau (kích thước crop median 117×144 / 74×92 / 98×134 px) nên KHÔNG gộp ảnh phong cách"))
    # ---- khuyến nghị nguồn ô neo + ảnh phong cách theo cuốn (số lấy từ dữ liệu vừa đo)
    kn = {}
    for ma in L.ORDER:
        r = sach[ma]; k = r["khuyen_nghi_tap_neo"]; C = L.CFG[ma]
        sv = pd.read_csv(L.OUT / "style_ung_vien" / f"{ma}.csv", dtype=str, keep_default_na=False)
        top3 = [dict(chu=a, crop_goc=b, crop_chuan_128=c2, crop_chuan_ok_xuat_ban=c or None, wxh=f"{float(w):.0f}x{float(h):.0f}") for a, b, c, c2, w, h in zip(sv.label[:3], sv.duong_crop[:3], sv.crop_chuan[:3], sv.crop_chuan_cache_128[:3], sv.w[:3], sv.h[:3])]
        ge = r["gold_exact"]; q = Q[ma]; kt = r["kich_thuoc_crop_GOLD"]
        nct = r["neo"]["qua_chon_chu"]
        luu = []
        if ma.startswith("stt"):
            luu.append(f"CHỈ lọc book=={C['book_val']} (cột book của dataset/SachThanhTruyen/labels.csv, gold_exact.csv: set8); không gộp 3 quyển, không dùng dataset/SachThanhTruyen{{2,4,11}}/ (23/09)")
            b2 = q["B2_gold_vs_chuan_cung_o"]
            luu.append("crop gốc là ảnh NHỊ PHÂN 2 mức (0/255) cắt từ trang đã xử lý; trên mẫu ô ok: "
                       f"{b2['multi15_chi_gold']}/{b2['n_cap']} crop gốc có ≥ 2 khối mực lớn (≥ 15 % mực) mà crop chuẩn cùng ô chỉ 1 khối (ngược lại {b2['multi15_chi_chuan']}); "
                       f"trong {b2['multi15_chi_gold']} ô đó khối thứ hai chạm mép trên/dưới ở {b2['multi15_chi_gold_khoi2_cham_mep_tren_duoi']} (gợi ý mảnh chữ kề cùng cột) → "
                       "ảnh phong cách nên lấy từ crop chuẩn (ô co_crop_chuan) hoặc chọn thêm bước loại mảnh kề")
        if ma in ("B18", "B34", "L16", "TK"):
            luu.append("bộ evaluation_only (TAP_DANH_GIA.md cấm đưa dòng labels.csv vào train/val của mô hình): chỉ dùng ô neo cho TINH CHỈNH khi người dùng quyết định; dùng làm ảnh phong cách khi SINH (suy luận) nhẹ hơn nhưng vẫn lấy crop của bộ")
            luu.append(f"crop rất nhỏ (rộng median {kt['w']['median']:.0f} px, cao {kt['h']['median']:.0f}) — ảnh phong cách 128×128 là phóng ≈{128 / max(kt['h']['median'], 1):.1f}×; "
                       f"loại {k['loai_do_canh_ngan_lt_24']} ô cạnh ngắn < 24 px")
        if ma in ("B18", "B34"):
            luu.append(f"{nct}/{r['neo']['co_tep_crop']} ô neo đi qua chon_chu và {r['neo']['qua_gate_qn_count_unfixed']} có cột âm QN lệch số (gate:qn_count_unfixed); độ đúng ô neo hiện tại đo bằng nhãn người (CHỈ chấm): "
                       f"{H[ma]['neo_tier_GOLD']['dung_chu_theo_cau_pct_CI'][0]} % (theo câu, n={H[ma]['neo_tier_GOLD']['n_co_chu_nguoi_theo_cau']}) / {H[ma]['neo_tier_GOLD']['dung_chu_pct_CI'][0]} % (có hộp người, n={H[ma]['neo_tier_GOLD']['n_co_gt']}); "
                       f"bản TN8 cũ chỉ {P3[ma]['do_bang_nhan_nguoi_CHI_CHAM']['neo_tn8']['dung_chu_pct']} %")
            luu.append(f"mất {P3[ma]['chi_truoc']} ô neo so với bảng TN8 vì chon_chu L5 đổi nhãn (nhãn kim đúng chỉ {P3[ma]['do_bang_nhan_nguoi_o_chi_truoc']['nhan_truoc_dung_pct']} %): dùng nhãn HIỆN TẠI, đừng dùng neo TN8")
            luu.append(f"{(q['A']['ti_le_co_hinh_hoc_theo_co']['f_two'] * 100):.1f} % ô GOLD neo bị cờ f_two (crop chứa 2 chữ) và {(q['A']['ti_le_co_hinh_hoc_theo_co']['tall_new'] * 100):.1f} % tall_new → bắt buộc lọc cờ hình học")
        if ma in ("Chr", "L83", "KVK"):
            luu.append("crop gốc có NỀN XÁM PHẲNG ≈128 và mực 0 (thạch bản, đã chuẩn hoá tương phản) — style_image() (Otsu) xử lý được; không phải giấy tự nhiên")
        if ma in ("Chr", "L83"):
            luu.append(f"ô neo ok (có crop chuẩn) chỉ {ge['neo_ok_n']} ({ge['neo_ok_so_chu']} chữ, ≥5 ô: {ge['neo_ok_chu_ge5']} chữ): crop chuẩn đủ cho ảnh phong cách nhưng KHÔNG đủ cho cặp tinh chỉnh → dùng crop gốc đã lọc ({k['n']} ô)")
        if ma in ("L16", "TK"):
            luu.append(f"ô neo ok {ge['neo_ok_n']} (đo trên nhãn người: đúng chữ {H[ma]['neo_gold_exact_ok']['dung_chu_pct_CI'][0]} % so với {H[ma]['neo_GOLD_khong_ok']['dung_chu_pct_CI'][0]} % ở ô GOLD không ok) — nhưng ok phụ thuộc bộ kiểm học từ nhãn người sách kia nên KHÔNG dùng ok làm tiêu chí chọn")
        if ma == "KVK":
            luu.append(f"{r['neo']['theo_tang'].get('REVIEW', 0)} ô neo tầng REVIEW và {r['neo']['theo_tang'].get('QUARANTINE', 0)} QUARANTINE có crop ở thư mục dựng (không có ở dataset/KimVanKieu1884/): loại")
        luu.append("lọc tier == GOLD: ô neo tầng GOLD_text_only/REVIEW/QUARANTINE (crop chỉ ở thư mục dựng) bị loại" + (f" — đo bằng nhãn người: đúng chữ ở ô neo tầng khác GOLD thấp ({H[ma]['neo_tier_khac_GOLD_REVIEW_QUARANTINE']['dung_chu_theo_cau_pct_CI'][0]} %)" if ma in H else ""))
        kn[ma] = dict(
            nguon_neo=(f"dataset/{C['ds']}/labels.csv" + (f" lọc book=={C['book_val']}" if ma.startswith("stt") else "") + " (crop: dataset/" + C["ds"] + "/<cột image>); bảng đã lọc sẵn: " + k["file"]),
            bo_loc=("tier=='GOLD' ∧ rule bắt đầu s1_inter_s2_direct ∧ label==ocr_char ∧ label∈R(âm) ∧ cờ hình học gold_exact sạch (one_char_ok=1 [= crop_status ok ∧ ¬f_blank ∧ ¬f_cut ∧ ¬f_two ∧ ¬f_ink ∧ ¬bleed_new ∧ ¬trunc_new, "
                    "pipeline/gold_exact/signals_geom.py:262] ∧ ¬tall_new ∧ ¬dup_bbox ∧ ¬ov_heavy ∧ ¬int_foreign) ∧ cạnh ngắn ≥ 24 px"),
            so_o=k["n"], so_chu=k["so_chu"], chu_ge3=k["chu_ge3"], chu_ge5=k["chu_ge5"],
            anh_phong_cach=dict(file=k["style_file"], tieu_chi=k["style_tieu_chi"], top3=top3,
                                 nguon_crop=("crop CHUẨN từ nhớ đệm gold_exact (cột crop_chuan_cache_128: 128×128; crop_chuan_cache: cạnh gốc) — có cho 100 % ô neo khuyến nghị, KHÔNG cần ô ok; "
                                             "phương án thay: crop gốc (duong_crop) qua style_image() của p02")),
            nguon_cap_tinh_chinh=f"{k['file']} — ảnh: cột crop_chuan_cache(_128) (prepared/_gold_exact/crop_chuan/{{sq,sq128}}/<uid_path>) hoặc crop gốc dataset/{C['ds']}/gold/ (cột duong_crop)",
            danh_sach_chu_can_sinh=r["chu_ung_vien_file"].split(" ")[0], so_chu_can_sinh=r["chu_ung_vien"]["so_chu"],
            chu_khong_co_o_neo_ge5=r["chu_ung_vien"]["so_chu"] - r["chu_ung_vien"]["so_chu_neo_ge5"],
            luu_y=luu)
    # ---- bảng TN8 cũ
    tn8 = {}
    for ma in L.ORDER:
        r = T8[ma]; p3 = P3.get(ma)
        tn8[ma] = dict(cu=r["bang_tn8_cu"], ly_do=r["ly_do_cu"], mtime_base=r["mtime_base_tn8"], mtime_labels=r["mtime_labels_hien_tai"], n_base=r["n_base"], n_hien_tai=r["n_hien_tai"],
                       khoa_o_chung=r["khoa_chung"], cung_thu_tu_hang=r["cung_thu_tu_hang"], o_doi_tang=r["so_khac_nhau_tren_khoa_chung"]["tier"], o_doi_nhan=r["so_khac_nhau_tren_khoa_chung"]["label"],
                       o_doi_hop=r["so_khac_nhau_tren_khoa_chung"]["bbox"], iou_hop=r["iou_hop"], neo=r["neo"], chu_ung_vien=r.get("chu_ung_vien"),
                       neo_truoc_chon_chu_bang_neo_tn8=(p3 or {}).get("neo_truoc_bang_neo_tn8"))
    # ---- tuổi dữ liệu: tệp mới nhất dưới dataset/ và mtime nhãn từng cuốn
    newest = (0.0, "")
    n_files = 0
    for dp, _dn, fn in os.walk(L.REPO / "dataset"):
        for f in fn:
            fp = os.path.join(dp, f)
            n_files += 1
            m = os.path.getmtime(fp)
            if m > newest[0]:
                newest = (m, fp)
    tuoi = dict(
        tep_moi_nhat_duoi_dataset=dict(duong=L.rel(newest[1]), mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(newest[0])), so_tep_quet=n_files),
        labels_pub_mtime={m: K[m]["mtime"]["labels_pub"] for m in L.ORDER}, labels_all_mtime={m: K[m]["mtime"]["labels_all"] for m in L.ORDER},
        gold_exact_mtime={m: K[m]["mtime"]["gold_exact"] for m in L.ORDER}, bang_tn8_base_mtime={m: T8[m]["mtime_base_tn8"] for m in L.ORDER},
        dataset_ngoai_git=subprocess.run(["git", "-C", str(L.REPO), "ls-files", "dataset", "prepared", "measure_out"], capture_output=True, text=True).stdout.count("\n") == 0)
    out = dict(
        tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"), git_head=subprocess.run(["git", "-C", str(L.REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        script=sorted(p.name for p in Path(__file__).parent.glob("w_hien_trang_du_lieu_*.py")),
        dinh_nghia=dict(
            o_neo_tu_dong="rule.startswith('s1_inter_s2_direct') ∧ label ≠ '' ∧ label == ocr_char ∧ label ∈ R(âm) (R = Dict/QuocNgu_SinoNom.csv qua pipeline.gold_exact.common.R_of); đếm trên bảng MỌI tầng; 'có crop' = tệp tồn tại thật",
            o_neo_khuyen_nghi="o_neo ∧ tier == GOLD ∧ không cờ hình học gold_exact (one_char_ok = 1 ∧ ¬tall_new ∧ ¬dup_bbox ∧ ¬ov_heavy ∧ ¬int_foreign) ∧ cạnh ngắn ≥ 24 px (không dùng trạng thái gold_exact ok; không nhãn người)",
            chu_hiem="chữ có < 2 ô neo cùng sách (ti_le_o_GOLD_thuoc_chu_hiem_lt2: ô GOLD có nhãn thuộc chữ hiếm; ti_le_o_moi_tang_theo_chu_kim_hiem_lt2: mọi ô theo chữ kim)",
            chu_ung_vien="R(âm của ô) ∪ chữ kim (ocr_char) [∪ mọi chữ lt1/lt2 trong kim_l1skel_l2 với STT], mọi tầng"),
        cache_crop_chuan=dict(tong_hop=CCH, xac_minh_dung_lai_bang_ham_pipeline=CCV, duong="prepared/_gold_exact/crop_chuan/{sq,sq128}/<book_set>/<book>/<page>/c<cột>_n<nom_idx>_s<syl_idx>.png",
                              ghi_chu="clean_rebuild_all.sh GIỮ prepared/_gold_exact (chỉ --deep mới xoá: scripts/clean_rebuild_all.sh:35,195); prepared/ ngoài git"),
        tong=dict(tong), tuoi_du_lieu=tuoi, sach=sach, stt=stt_info, tn8_cu=tn8, khuyen_nghi=kn,
        cach_chay_lai=["export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0   # chỉ CPU, 0 API",
                       ".venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_01_kiem_ke.py        # ~1 phút: kiểm kê, ô neo, chữ ứng viên, neo/<ma>.csv",
                       ".venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_02_chat_luong_crop.py  # ~10 giây",
                       ".venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_02b_do_neo_nhan_nguoi.py  # nhãn người CHỈ để chấm",
                       ".venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_03_tn8_cu.py ; ..._03b_neo_truoc_chon_chu.py",
                       ".venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_04_rui_ro.py          # ~35 giây (md5 toàn bộ tệp)",
                       ".venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_05_tong_hop.py        # gộp -> ket_qua.json, style_ung_vien/, neo_khuyen_nghi/"],
        luu_y_bang_cong_viec=("neo/<ma>.csv và neo_khuyen_nghi/<ma>.csv là bảng LÀM VIỆC cục bộ: có cột gold_exact, crop_chuan_rel/ton_tai, gate_reason, chon_chu (suy ra từ bộ kiểm học trên nhãn/văn bản người) — "
                              "phải bỏ trước khi đóng gói Kaggle (giữ crop + label + md5)"),
        rui_ro=dict(trung_md5_cheo_sach=RR["trung_md5_cheo_sach"], trung_md5_cheo_quyen_STT=RR["trung_md5_cheo_quyen_STT"], tep_mo_coi_dataset=RR["tep_mo_coi_dataset"],
                    ro_nhan_nguoi=RR["ro_nhan_nguoi"], evaluation_only=RR["evaluation_only"]),
        bat_bien_tung_buoc=dict(kiem_ke={m: K[m]["bat_bien"] for m in L.ORDER}, chat_luong_crop={m: Q[m]["bat_bien"] for m in L.ORDER}, do_neo_nhan_nguoi={m: H[m]["bat_bien"] for m in H},
                                 tn8_cu={m: T8[m]["bat_bien"] for m in L.ORDER}, neo_truoc_chon_chu={m: P3[m]["bat_bien"] for m in P3}, rui_ro=RR["bat_bien"]))
    # bất biến tổng
    for name, d in out["bat_bien_tung_buoc"].items():
        tot = sum(v["tong"] for v in d.values()) if name != "rui_ro" else d["tong"]
        dat = sum(v["dat"] for v in d.values()) if name != "rui_ro" else d["dat"]
        inv.check(f"{name}_dat_het", tot == dat, f"{dat}/{tot}")
    out["bat_bien"] = inv.summary(); out["bat_bien_chi_tiet"] = inv.rows
    out["giay"] = round(time.time() - t0, 1)
    L.jdump(out, L.OUT / "ket_qua.json")
    print(f"[05] ket_qua.json: bất biến gộp {out['bat_bien']['dat']}/{out['bat_bien']['tong']} {[r['ten'] for r in out['bat_bien']['rot']]} | tổng {dict(tong)}")
    print("->", L.rel(L.OUT / "ket_qua.json"))


if __name__ == "__main__":
    main()
