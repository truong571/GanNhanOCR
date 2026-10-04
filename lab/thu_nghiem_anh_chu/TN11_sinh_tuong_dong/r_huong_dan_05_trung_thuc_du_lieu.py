"""r_huong_dan_05_trung_thuc_du_lieu.py — REVIEW hướng "huong_dan" (5/5): tính trung thực của hướng dẫn và các khẳng định về dữ liệu:
  A. gói có chứa tệp nhãn không; danh sách chữ lấy từ cột nào (có cột nhãn người không)
  B. "mỗi cuốn một phong cách": ảnh phong cách đúng cuốn/quyển, khác nhau, tái dựng được từ crop nguồn nêu trong báo cáo gói
  C. "metadata mỗi tar ghi chữ nào dùng phông nào" có đúng với mã/tar thật không
  D. "Kỳ vọng" so với số đã đo (SINH_ANH_SO_SANH_2026-10-03.md) — trích đúng số, không diễn giải
  E. "dữ liệu mới nhất" (mtime), tệp TAP_DANH_GIA
  F. phông dự phòng: có từng được chạy chưa (CPU kiểm thật có dùng phông dự phòng không) + vẽ thử ảnh nội dung bằng chính mã của gói
0 GPU, 0 API, 0 mạng. Chỉ ĐỌC repo.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_huong_dan_05_trung_thuc_du_lieu.py
Ra: measure_out/_tn11/full/review/huong_dan/05_trung_thuc_du_lieu.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections
import csv
import hashlib
import io
import json
import os
import re
import tarfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL = REPO / "measure_out" / "_tn11" / "full"
OUT = FULL / "review" / "huong_dan"
PACK = FULL / "kaggle_pack_20261004"
ZIP = FULL / "kaggle_pack_20261004.zip"
GUIDE = REPO / "docs" / "KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md"
SINH = REPO / "docs" / "SINH_ANH_SO_SANH_2026-10-03.md"

R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "invariants": {}}


def inv(name, ok, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:300], flush=True)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def style_image(g: np.ndarray) -> np.ndarray:
    """CHÉP p11_dung_goi_toan_bo.style_image (== p02.style_image) để kiểm nguồn gốc ảnh phong cách."""
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw < 128).mean() > 0.5:
        bw = 255 - bw
    h, w = bw.shape
    s = int(max(h, w) * 1.2)
    out = np.full((s, s), 255, np.uint8)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = bw
    return cv2.resize(out, (128, 128), interpolation=cv2.INTER_AREA)


def main():
    g = GUIDE.read_text(encoding="utf-8")
    plan = json.loads((PACK / "plan.json").read_text(encoding="utf-8"))
    rep = json.loads((FULL / "kaggle_pack_20261004_bao_cao.json").read_text(encoding="utf-8"))
    books = plan["books"]

    # ============================================================ A. không có nhãn trong gói
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    ext = collections.Counter(Path(n).suffix.lower() for n in names)
    R["goi_dinh_dang_tep"] = dict(ext)
    bad = [n for n in names if re.search(r"(nhan_nguoi|human|truth|ground|gt_|label|manifest\.tsv|\.csv$|\.tsv$|\.pkl$|\.xlsx$)", Path(n).name, re.I) and Path(n).name != "manifest.json"]
    inv("A1.zip_khong_co_tep_ten_nhan_csv_tsv_pkl_xlsx", not bad and set(ext) <= {".py", ".json", ".png", ".safetensors", ".ttf", ".otf"}, dict(bad=bad, ext=dict(ext)))
    inv("A2.plan_json_khoa_chi_gom_danh_sach_chu_phong_lo", set(plan) == {"tool", "shard", "passes", "font_names", "books", "tasks", "sha"} and all(set(v) == {"chars", "font_idx", "uncovered", "styles"} for v in books.values()), sorted(plan))
    heads = {}
    for b, rel in (("B18", "prepared/_auto/SachKinhThayCaBinh/dataset_out/labels_gated.csv"), ("B34", "prepared/_auto/SachDungLyHoThan/dataset_out/labels_gated.csv"),
                   ("L16", "prepared/LucVanTien1916/dataset_out/labels_gated.csv"), ("TK", "prepared/TruyenKieu1872/dataset_out/labels_gated.csv")):
        with open(REPO / rel, encoding="utf-8", newline="") as f:
            heads[b] = next(csv.reader(f))
    pat = re.compile(r"(^gt|human|nguoi|ihr|truth|sinonom)", re.I)
    sus = {b: [c for c in h if pat.search(c)] for b, h in heads.items()}
    inv("A3.tep_nhan_nguon_cua_4_bo_nhan_nguoi_khong_co_cot_nhan_nguoi", all(not v for v in sus.values()) and all("chon_chu_truoc" in h for h in heads.values()), dict(cot_nghi_van=sus, co_chon_chu_truoc={b: "chon_chu_truoc" in h for b, h in heads.items()}))
    kq = json.loads((FULL / "quy_mo_va_ngan_sach" / "ket_qua.json").read_text(encoding="utf-8"))
    R["nguon_nhan_dung_dung_de_dung_danh_sach"] = {b: v["tep"] for b, v in kq["meta"]["nguon_nhan"].items()}
    inv("A4.nguon_nhan_la_labels_gated_hoac_labels_final_cua_pipeline", all(Path(v["tep"]).name in ("labels_gated.csv", "labels_final.csv") for v in kq["meta"]["nguon_nhan"].values()), R["nguon_nhan_dung_dung_de_dung_danh_sach"])
    lib = (HERE / "w_quy_mo_va_ngan_sach_lib.py").read_text(encoding="utf-8")
    inv("A5.tai_lieu_noi_ban_sach_dung_nhan_truoc_chon_chu_va_B18_B34_khac_ban_literal", "bản \"sach\" là bản DÙNG ĐỂ DỰNG GÓI" in lib and kq["theo_sach"]["B18"]["S0_literal"] != kq["theo_sach"]["B18"]["S0_sach"] and kq["theo_sach"]["B34"]["S0_literal"] != kq["theo_sach"]["B34"]["S0_sach"],
        dict(B18=[kq["theo_sach"]["B18"]["S0_literal"], kq["theo_sach"]["B18"]["S0_sach"]], B34=[kq["theo_sach"]["B34"]["S0_literal"], kq["theo_sach"]["B34"]["S0_sach"]]))
    # "5 ứng viên đầu MỖI Ô" (hướng dẫn) so với định nghĩa trong mã: top-k CỦA MỖI ÂM
    inv("A6.dinh_nghia_S1_trong_ma_la_top_k_cua_MOI_AM_khong_phai_moi_o", "top-k ứng viên R(âm) CỦA MỖI ÂM trong cuốn" in lib and "5 ứng viên đầu mỗi ô" in g, "hướng dẫn mục 0 nói 'mỗi ô'; mã nói 'mỗi âm trong cuốn'")

    # ============================================================ B. mỗi cuốn một phong cách
    exp = {"stt2": "SachThanhTruyen/stt2/", "stt4": "SachThanhTruyen/stt4/", "stt11": "SachThanhTruyen/stt11/", "Chr": "Chrestomathie1872/chrestomathie1872/", "L83": "LucVanTien1883/lucvantien1883/",
           "KVK": "KimVanKieu1884/kimvankieu1884/", "L16": "LucVanTien1916/lucvantien1916/", "TK": "TruyenKieu1872/truyenkieu1872/", "B18": "SachKinhThayCaBinh/sachkinhthaycabinh/", "B34": "SachDungLyHoThan/sachdunglyhothan/"}
    dung_cuon = {b: all(s["nguon"].split("crop_chuan/sq128/")[1].startswith(exp[b]) for s in rep["sach"][b]["phong_cach"]) for b in exp}
    inv("B1.anh_phong_cach_cua_moi_cuon_lay_tu_crop_cua_chinh_cuon_hoac_quyen_do", all(dung_cuon.values()) and len(dung_cuon) == 10, dung_cuon)
    shas, dims, graypct = {}, {}, {}
    for b in exp:
        for j in (0, 1):
            raw = (PACK / "data" / b / f"style_{j}.png").read_bytes()
            shas[f"{b}_s{j}"] = sha(raw)
            im = cv2.imread(str(PACK / "data" / b / f"style_{j}.png"), cv2.IMREAD_GRAYSCALE)
            dims[f"{b}_s{j}"] = im.shape
            graypct[f"{b}_s{j}"] = round(float(((im > 0) & (im < 255)).mean()) * 100, 1)
    inv("B2.20_anh_phong_cach_khac_nhau_tung_doi_va_deu_128x128", len(set(shas.values())) == 20 and all(d == (128, 128) for d in dims.values()), dict(khac_nhau=len(set(shas.values()))))
    inv("B3.ba_quyen_STT_co_ba_anh_phong_cach_va_ba_danh_sach_chu_khac_nhau", len({shas["stt2_s0"], shas["stt4_s0"], shas["stt11_s0"]}) == 3 and len({books["stt2"]["chars"], books["stt4"]["chars"], books["stt11"]["chars"]}) == 3, None)
    tai_dung = {}
    for b in exp:
        s0 = rep["sach"][b]["phong_cach"][0]
        src = cv2.imread(str(REPO / s0["nguon"]), cv2.IMREAD_GRAYSCALE)
        pack0 = cv2.imread(str(PACK / "data" / b / "style_0.png"), cv2.IMREAD_GRAYSCALE)
        tai_dung[b] = bool(src is not None and np.array_equal(style_image(src), pack0))
    inv("B4.style_0_tai_dung_chinh_xac_tu_crop_nguon_trong_bao_cao_(Otsu->vuong_1.2->128)", all(tai_dung.values()), tai_dung)
    R["ty_le_pixel_xam_trong_anh_phong_cach_pct"] = graypct
    inv("B5.anh_phong_cach_co_pixel_xam_do_noi_suy_INTER_AREA_nen_'nhi_phan_Otsu'_chi_dung_truoc_khi_co_ve_128", max(graypct.values()) > 5, dict(max_pct=max(graypct.values()), min_pct=min(graypct.values())))

    # ============================================================ C. _meta.json của tar
    tars = sorted((FULL / "runner" / "real_cpu" / "out" / "shards").rglob("*.tar"))
    meta_keys = None
    if tars:
        with tarfile.open(tars[0]) as tf:
            meta = json.loads(tf.extractfile("_meta.json").read())
        meta_keys = sorted(meta)
    src = (HERE / "w_runner_khung_chay.py").read_text(encoding="utf-8")
    m = re.search(r"meta = dict\((.*?)\)\n", src, re.S)
    inv("C1._meta.json_cua_tar_khong_co_thong_tin_phong", meta_keys is not None and not any("font" in k for k in meta_keys) and "font" not in (m.group(1) if m else ""), dict(khoa_meta=meta_keys, mau=str(tars[0].relative_to(REPO)) if tars else None))
    inv("C2.hướng_dẫn_noi_metadata_moi_tar_ghi_phong", "metadata mỗi tar ghi chữ nào dùng phông nào" in g, g[g.index("metadata mỗi tar"):][:60])
    inv("C3.phong_theo_chu_nam_o_plan_json_font_idx_va_font_names", all(len(v["font_idx"]) == len(v["chars"]) for v in books.values()) and plan["font_names"][0] == "NomNaTong", plan["font_names"])

    # ============================================================ D. "Kỳ vọng" so với số đã đo
    doc = SINH.read_text(encoding="utf-8")
    row = next(ln for ln in doc.splitlines() if ln.startswith("| p02 chuẩn |"))
    m = re.search(r"Top-1 ([\d,]+) % \(nét thường\) / ([\d,]+) % \(nét đậm\) so với phông ([\d,]+) % và FD ([\d,]+) %; Δ so với phông ([\d,]+) \[([−\-+\d,; ]+)\] / (−[\d,]+) \[([−\-+\d,; ]+)\]\. Trên 188 ô top-8: (\+[\d,]+) \[([^\]]+)\] / (\+[\d,]+) \[([^\]]+)\]", row)
    R["so_do_B34_p02_chuan"] = dict(trich=row[:60], nhom=m.groups() if m else None)
    inv("D1.so_B34_chenh_0_0_chi_la_nhanh_net_thuong_con_net_dam_la_-5,4_va_CI_rong_+-8,1", bool(m) and m.group(5) == "0,0" and m.group(7) == "−5,4" and "−8,1; +8,1" in m.group(6), m.groups() if m else row[:200])
    ev = {}
    for key, rx in (("Ridge_vao_bo_chon", r"\| p05 \|[^\n]*Không đổi[^\n]*"), ("HQC_vao_bo_chon", r"\| HQC \|[^\n]*Thêm vào bộ chọn TN8: Δ ≤ \+0,09[^\n]*"), ("TN12_kho_FD", r"\| TN12 \|[^\n]*không đổi\*\*"), ("p10_ablation", r"\| p10 \|[^\n]*bỏ cả hai −0,01…\+0,05[^\n]*"),
                    ("ket_luan_bao_hoa", r"\*\*Phía so ảnh của bộ chọn chữ đã bão hoà\*\*[^\n]*"), ("ket_luan_chi_nen_chay_neu_muon_bang_chung_am", r"chỉ nên chạy nếu muốn có\s+bằng chứng âm[^\n]*"), ("tinh_chinh_xau_di", r"\| p04 \|[^\n]*\*\*Xấu đi\*\*[^\n]*")):
        mm = re.search(rx, doc)
        ev[key] = (mm.group(0)[:230] if mm else None)
    R["bang_chung_da_do_truoc"] = ev
    inv("D2.tai_lieu_da_do_ghi_phia_so_anh_bao_hoa_va_chi_nen_chay_de_lay_bang_chung_am", all(v for v in ev.values()), {k: bool(v) for k, v in ev.items()})
    exp_sec = g[g.index("- **Kỳ vọng**"):g.index("## 1.")]
    inv("D3.muc_ky_vong_trong_huong_dan_khong_nhac_bao_hoa_hay_CI_hay_ngưỡng_+0,3_kho_dat", not re.search(r"bão hoà|CI|\[−8|khó đạt|bằng chứng âm", exp_sec), exp_sec[:200])
    inv("D4.huong_dan_co_nhac_ket_qua_am_o_muc_3", "ngược lại ghi kết quả âm" in g, None)
    m_sect = re.search(r"Khung chưa chạy trên GPU Kaggle thật và HF thật", g)
    inv("D5.huong_dan_noi_ro_khung_chua_chay_GPU_Kaggle_va_HF_that", bool(m_sect), dict(dong=g[: m_sect.start()].count("\n") + 1 if m_sect else None, vi_tri="cuối mục 5 (không có ở mục 0)"))
    inv("D6.muc_0_khong_canh_bao_chua_chay_that", "chưa chạy" not in g[: g.index("## 1.")], None)

    # ============================================================ E. dữ liệu mới nhất + TAP_DANH_GIA
    newest = (0.0, "")
    n = 0
    for r, _, fs in os.walk(REPO / "dataset"):
        for f in fs:
            p = os.path.join(r, f)
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            n += 1
            if mt > newest[0]:
                newest = (mt, p)
    R["dataset_tep_moi_nhat"] = dict(so_tep=n, mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest[0])), tep=os.path.relpath(newest[1], REPO))
    inv("E1.khong_tep_nao_duoi_dataset_moi_hon_01/10_20:40:30", time.strftime("%m-%d %H:%M", time.localtime(newest[0])) <= "10-01 20:40", R["dataset_tep_moi_nhat"])
    srcm = {b: v["mtime"] for b, v in kq["meta"]["nguon_nhan"].items()}
    inv("E2.tep_nhan_dung_dung_goi_co_mtime_01/10_19:27_den_20:17", min(srcm.values()) >= "2026-10-01 19:27:00" and max(srcm.values()) <= "2026-10-01 20:17:59", dict(min=min(srcm.values()), max=max(srcm.values())))
    ge = [time.strftime("%H:%M", time.localtime(p.stat().st_mtime)) for p in (REPO / "dataset").glob("*/gold_exact.csv")]
    inv("E3.gold_exact_20:26", set(ge) == {"20:26"}, sorted(set(ge)))
    tap = (REPO / "dataset" / "SachDungLyHoThan" / "TAP_DANH_GIA.md").read_text(encoding="utf-8")
    tall = (REPO / "dataset" / "_ALL" / "TAP_DANH_GIA.md").read_text(encoding="utf-8")
    inv("E4.TAP_DANH_GIA_B34_cam_dua_dong_labels_csv_vao_tap_train_val_cua_mo_hinh_nao", "KHÔNG** đưa bất kỳ dòng nào của `labels.csv` này vào tập train/val của mô hình nào" in tap, None)
    inv("E5.TAP_DANH_GIA_B34_noi_adapter_khong_doc_cot_Nom_nguoi", "không đọc cột chữ Nôm người" in tap, None)
    inv("E6.LOI_o_dataset/_ALL/TAP_DANH_GIA.md:_goi_ca_4_bo_la_'ban_in'_trong_khi_dataset/SachDungLyHoThan/TAP_DANH_GIA.md_noi_B34_la_'CHEP_TAY'",
        "bản in" in tall and "CHÉP TAY" in tap and "SachDungLyHoThan" in tall, "xác nhận mâu thuẫn giữa hai tệp (không nằm trong hướng dẫn Kaggle)")
    m_ev = re.search(r"132\.916 trong 235\.012", tall)
    inv("E7._ALL_TAP_DANH_GIA_con_so_132.916/235.012", bool(m_ev), None)
    # đếm thật evaluation_only trong labels.csv gộp nếu có cột
    try:
        import pandas as pd
        hdr = pd.read_csv(REPO / "dataset" / "_ALL" / "labels.csv", nrows=2)
        cols = list(hdr.columns)
        R["_ALL_cot"] = [c for c in cols if "eval" in c.lower() or "split" in c.lower() or "lobo" in c.lower()]
        if "evaluation_only" in cols:
            ev_n = pd.read_csv(REPO / "dataset" / "_ALL" / "labels.csv", usecols=["evaluation_only"])["evaluation_only"]
            R["_ALL_evaluation_only"] = dict(so_dong=int(len(ev_n)), evaluation_only_1=int((ev_n == 1).sum()))
            inv("E8._ALL_evaluation_only_khop_132.916_tren_235.012", int((ev_n == 1).sum()) == 132916 and len(ev_n) == 235012, R["_ALL_evaluation_only"])
        else:
            inv("E8._ALL_co_cot_evaluation_only", False, cols[:20])
    except Exception as e:  # noqa: BLE001
        inv("E8._ALL_evaluation_only", False, f"{type(e).__name__}: {e}")

    # ============================================================ F. phông dự phòng
    t = (HERE / "w_runner_kiem_that_cpu.py").read_text(encoding="utf-8")
    from fontTools.ttLib import TTFont
    f_nn = TTFont(str(PACK / "code" / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"), lazy=True)
    cm = set()
    for tb in f_nn["cmap"].tables:
        cm |= set(tb.cmap)
    chars_cpu = "孟恒固南"
    inv("F1.kiem_that_CPU_chi_dung_4_chu_deu_co_trong_NomNaTong_nen_phong_du_phong_chua_tung_chay_voi_mo_hinh", all(ord(c) in cm for c in chars_cpu) and all(c in t for c in chars_cpu), dict(chu=chars_cpu))
    fn = plan["font_names"]
    use = collections.Counter()
    pick = collections.defaultdict(list)
    for b, v in books.items():
        for c, i in zip(v["chars"], v["font_idx"]):
            use[fn[i]] += 1
            if len(pick[fn[i]]) < 60:
                pick[fn[i]].append(c)
    R["so_chu_theo_phong"] = dict(use)
    sys.path.insert(0, str(PACK / "code")); sys.path.insert(0, str(PACK / "code" / "font_diffusion"))
    fonts_json = json.loads((PACK / "fonts.json").read_text(encoding="utf-8"))
    try:
        from inference.sample_optimized import FontManager
        from src.tools.utils import ttf2im
        res = {}
        for name in fn:
            p = PACK / "code" / "font_diffusion" / "fonts" / fonts_json[name]
            fm = FontManager(str(p))
            font = fm.get_font(p.stem)
            nd = ttf2im(font, "͸")                               # ký tự chưa gán -> hình .notdef của chính phông
            nd_arr = np.array(nd) if nd is not None else None
            n = n_none = n_tofu = n_blank = 0
            for c in pick[name]:
                im = ttf2im(font, c)
                n += 1
                if im is None:
                    n_none += 1
                    continue
                a = np.array(im)
                ink = float((255 - a.mean()) / 255 * 100)
                if ink < 0.5:
                    n_blank += 1
                if nd_arr is not None and a.shape == nd_arr.shape and np.array_equal(a, nd_arr):
                    n_tofu += 1
            res[name] = dict(thu=n, none=n_none, trang=n_blank, giong_notdef=n_tofu, tong_chu_trong_ke_hoach=use[name])
        R["ve_thu_anh_noi_dung_bang_ma_cua_goi_CPU"] = res
        inv("F2.moi_phong_trong_ke_hoach_nap_duoc_bang_ma_goi_va_ve_ra_anh_khong_trang_khong_notdef", all(v["none"] == 0 and v["trang"] == 0 and v["giong_notdef"] == 0 for v in res.values()), res)
    except Exception as e:  # noqa: BLE001
        inv("F2.ve_thu_phong_du_phong", False, f"{type(e).__name__}: {str(e)[:200]}")

    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]),
                     fail_ten=[k for k, v in R["invariants"].items() if not v["ok"]])
    (OUT / "05_trung_thuc_du_lieu.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
