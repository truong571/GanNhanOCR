"""v_thu_hqc_07_tonghop.py — gộp hiệu chuẩn + mô hình giấy + tuning + phép đo (danhgia.json) + chẩn đoán lệch nền + kiểm mã K09 thành
measure_out/_tn11/verify/thu_hqc/ket_qua.json, đánh giá TIÊU CHÍ ĐĂNG KÝ TRƯỚC và in bảng gọn.
  .venv/bin/python .../v_thu_hqc_07_tonghop.py [--danhgia danhgia.json]
Tiêu chí (đăng ký trước): "co_ich" = tồn tại bộ có nhãn người và tín hiệu (f_font | f_fd) với Δ Top-1 (C1-C0, mọi ô) >= +2,0 và CI95 % cận dưới > 0;
"ap_dung" = một tín hiệu có Δ >= +1,0 và CI cận dưới > 0 ở CẢ 4 bộ; "khong_ap_dung" = không bộ/tín hiệu nào dương có ý nghĩa (âm hoặc không khác biệt);
còn lại = "hon_hop_can_them_du_lieu".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402

OUT = H.OUT


def jl(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pick_set(dg, b, kind):
    """tập ngẫu nhiên LỚN NHẤT có phủ đầy đủ cho (bộ, tín hiệu): all (full_coverage) > s3000 > s1000 (phu_day_du). Trả (tên, dict d_C1_C0 tất cả)."""
    sets = ((dg.get(b) or {}).get("sets", {}) or {})
    s = (sets.get("all") or {}).get(kind)
    if s and s.get("full_coverage"):
        return "all", s["d_C1_C0"].get("tat_ca"), s
    for nm in ("s3000", "s1000"):
        s = (sets.get(nm) or {}).get(kind)
        if s and s.get("phu_day_du") and "C1-C0" in s.get("delta", {}):
            return nm, s["delta"]["C1-C0"].get("tat_ca"), s
    return None, None, None


def criteria(dg):
    rows = []
    for b in H.BOOKS:
        for kind in ("font", "fd"):
            nm, d, _ = pick_set(dg, b, kind)
            if d:
                rows.append(dict(bo=b, tin_hieu=f"f_{kind}", tap=nm, **d))
    out = dict(bang=rows)
    out["co_ich"] = any(r["delta"] >= 2.0 and r["ci_thap"] > 0 for r in rows)
    ap = {}
    for kind in ("f_font", "f_fd"):
        rr = [r for r in rows if r["tin_hieu"] == kind]
        ap[kind] = (len(rr) == 4) and all(r["delta"] >= 1.0 and r["ci_thap"] > 0 for r in rr)
    out["ap_dung_theo_tin_hieu"] = ap
    out["ap_dung"] = any(ap.values())
    out["co_bo_duong_co_y_nghia"] = any(r["ci_thap"] > 0 for r in rows)
    out["co_bo_am_co_y_nghia"] = any(r["ci_cao"] < 0 for r in rows)
    if out["ap_dung"]:
        out["khuyen_nghi"] = "ap_dung"
    elif not out["co_bo_duong_co_y_nghia"]:
        out["khuyen_nghi"] = "khong_ap_dung"
    else:
        out["khuyen_nghi"] = "hon_hop_can_them_du_lieu"
    return out


def k09_kiem_ma():
    """Kiểm từng khẳng định mã/giao thức của K09 (p05_full_corpus_he_quy_chieu.py) bằng số đo của chính phép thử này."""
    P05 = "lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p05_full_corpus_he_quy_chieu.py"
    gy = {b: jl(OUT / "giay" / f"{b}.json") or {} for b in H.BOOKS}
    hc = jl(OUT / "hieuchuan.json") or {}
    gt = jl(OUT / "giao_thuc_k09.json") or {}
    k09 = jl(H.REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json") or []
    cmp = lambda a, b_: "tang" if a > b_ else ("giam" if a < b_ else "bang")
    h2 = [cmp(r["hqc2_book_paper_canvas"]["top1"], r["hqc1_white_canvas"]["top1"]) for r in k09]
    h3 = [cmp(r["hqc3_canonical_normalized"]["top1"], r["hqc1_white_canvas"]["top1"]) for r in k09]
    out = []
    out.append(dict(khang_dinh="HQC1 = crop thô so với glyph nền trắng (baseline)", ket_qua="khong_phai_baseline_san_xuat",
                    bang_chung=f"{P05}:58-70 dựng glyph riêng (khung 128, cỡ 0,75x128, chỉ NomNaTong) != san xuất pipeline/gold_exact/signals_img.py:314-324 (khung 112, cỡ 84, 3 phông dự phòng). "
                               f"Glyph sản xuất tái tạo trên CPU: max|Δ|={hc.get('max_abs_all')} (hieuchuan.json), nên baseline đúng là cột f_font/f_fd, không phải 'HQC1' của p05."))
    out.append(dict(khang_dinh="HQC2 = alpha-blend glyph lên paper_patch + ink_median (mã 100-155)", ket_qua="mo_ta_dung_ma_nhung_mau_giay_la_tong_hop_o_3_tren_4_bo",
                    bang_chung=f"{P05}:127-132 đúng công thức; nhưng mẫu giấy từ MỘT trang đã xử lý (dòng 81 sample_pages[min(10,..)]), mảng ít mực nhất, nếu min_ink>200 thì thay bằng nhiễu Gauss (dòng 112-114). "
                               "Chạy lại đúng vòng quét: min_ink = " + ", ".join(f"{b}:{(gy[b].get('k09_paper_stats') or {}).get('min_ink_patch')}" for b in H.BOOKS)
                               + " -> giấy tổng hợp: " + ", ".join(f"{b}:{(gy[b].get('k09_paper_stats') or {}).get('fallback_synthetic_gauss')}" for b in H.BOOKS) + " (giay/<bộ>.json)."))
    out.append(dict(khang_dinh="Thống kê giấy: nền/mực/tương phản của sách (vd nền 179 vs 255)", ket_qua="lech_so_do_16_trang",
                    bang_chung="K09 (nền/mực/tương phản) vs đo 16 trang ảnh gốc (giấy p90/mực p3/tương phản): "
                               + "; ".join(f"{b}: {(gy[b].get('k09_paper_stats') or {}).get('bg_median')}/{(gy[b].get('k09_paper_stats') or {}).get('ink_median')}/{(gy[b].get('k09_paper_stats') or {}).get('contrast')} vs "
                                           f"{gy[b].get('paper_level_p90_median', float('nan')):.0f}/{gy[b].get('ink_level_p3_median', float('nan')):.0f}/{gy[b].get('contrast', float('nan')):.0f}" for b in H.BOOKS)))
    out.append(dict(khang_dinh="Lý do: lớp đầu CNN nhạy độ sáng nền/tương phản; nền 179 vs 255 làm vector lệch pha", ket_qua="tien_de_sai_o_loi_vao_encoder",
                    bang_chung="Crop sản xuất được giãn p2/p98 trước encoder (signals_img.py:241-251; t02_embed.py:156 enc.embed(buf_g) norm=True mặc định, signals_img.py:266,385): "
                               + "; ".join(f"{b}: p98 thô {((hc.get('books') or {}).get(b) or {}).get('crop_recut', {}).get('raw_p98_median')}->{((hc.get('books') or {}).get(b) or {}).get('crop_recut', {}).get('after_stretch_p98_median')} sau giãn" for b in H.BOOKS)
                               + ". K09 nhúng glyph HQC2 với norm=False (p05:224-226) nên TẠO lệch glyph(không giãn) vs crop(có giãn)."))
    out.append(dict(khang_dinh="HQC3 = nhị phân Otsu cả crop và glyph, lọc đốm <6, căn giữa 128 lề 10 %", ket_qua="dung_ma_nhung_crop_bi_ep_vuong",
                    bang_chung=f"{P05}:135-163 đúng mô tả; nhưng crop bị cv2.resize((128,128)) trước khi nhị phân (dòng 236-239) làm méo tỉ lệ; bản C2 ở đây giữ tỉ lệ."))
    out.append(dict(khang_dinh="Bảng Top-1 trước->sau 10 bộ chứng minh rõ ràng", ket_qua="khong_du_suc_thuyet_phuc",
                    bang_chung=f"n=16-20 ô/bộ ({P05}:356 n_sample=20; ket_qua_full_10_bo.json n_cells); top-3 theo f_vW rồi giữ ô có chữ đúng trong top-3 ({P05}:196-200): "
                               + "; ".join(f"{b}: {gt[b]['ti_le_o_co_chu_dung_trong_top3_theo_f_vW']}% ô còn lại, Top-1 f_font trong 3 ƯV {gt[b]['top1_trong_3_ung_vien_f_font']} vs toàn bộ ƯV {gt[b]['top1_toan_bo_ung_vien_f_font']}" for b in H.BOOKS if b in gt)
                               + f"; hạt giống hash(book) ({P05}:191) không tái lập; 6/10 bộ dùng nhãn pipeline ({P05}:177-188); n=20 -> SD Top-1 ~9-11 điểm (giao_thuc_k09.json n20_*). "
                                 f"Chính ket_qua_full_10_bo.json: HQC2 so HQC1: {h2.count('tang')} tăng/{h2.count('bang')} bằng/{h2.count('giam')} giảm; HQC3 so HQC1: {h3.count('tang')} tăng/{h3.count('bang')} bằng/{h3.count('giam')} giảm."))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--danhgia", default="danhgia.json")
    a = ap.parse_args()
    dg = jl(OUT / a.danhgia) or {}
    res = dict(ngay="2026-10-04", seed=H.SEED, thiet_bi="cpu (không MPS)", workers=H.NWORK,
               hieu_chuan=jl(OUT / "hieuchuan.json"),
               mo_hinh_giay={b: {k: v for k, v in (jl(OUT / "giay" / f"{b}.json") or {}).items() if k not in ("pages_used",)} for b in H.BOOKS},
               kiem_nhanh=jl(OUT / "kiem_nhanh.json"),
               tuning={b: {k: v for k, v in (jl(OUT / "tune.json") or {}).get(b, {}).items() if k not in ("tune_cells", "tune_cells_fd")} for b in H.BOOKS},
               do=dg)
    res["do_k3_s500"] = jl(OUT / "danhgia_k3.json")
    tn8 = {}
    for nm in ("tn8_hqc/ket_qua_tn8_hqc_truth_BB.json", "tn8_hqc/ket_qua_tn8_hqc_truth_TL.json", "tn8_hqc/ket_qua_tn8_hqc.json"):
        j = jl(OUT / nm)
        if j:
            tn8[nm.split("/")[-1]] = j
    res["bo_chon_tn8"] = tn8
    res["phu_luc"] = {nm: jl(OUT / f"danhgia_{nm}.json") for nm in ("k3a", "p1only", "p2only", "c2g")}
    res["ket_qua_gon"] = jl(OUT / "ket_qua_gon.json")
    res["k09_kiem_ma"] = k09_kiem_ma()
    res["giao_thuc_k09"] = jl(OUT / "giao_thuc_k09.json")
    res["tieu_chi"] = criteria(dg)
    H.jdump(res, OUT / "ket_qua.json")
    c = res["tieu_chi"]
    print(f"== tiêu chí đăng ký trước: có_ích={c['co_ich']} áp_dụng={c['ap_dung']} -> {c['khuyen_nghi']}")
    for r in c["bang"]:
        print(f"   {r['bo']:4s} {r['tin_hieu']:7s} [{r['tap']}] Δ(C1-C0)={r['delta']:+6.2f} [{r['ci_thap']:+6.2f}, {r['ci_cao']:+6.2f}] n={r['n']}")


if __name__ == "__main__":
    main()
