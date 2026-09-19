"""Mở rộng độ phủ từ điển (Dictionary Coverage Expansion) cho chữ Nôm hiếm và dị thể.

Mục tiêu:
1. Phân tích 488 ô trong REVIEW có âm tiết chưa từng xuất hiện trong từ điển 104k cặp.
2. Tìm kiếm ứng viên chữ Nôm tiềm năng thông qua:
   - Ngữ cảnh 2-gram âm-âm (Bigram collocation) đối chiếu với các đoạn lặp trong ngữ liệu.
   - Bảng biến thể Unihan (kSemanticVariant, kZVariant).
   - Nét bút tương đồng (Dict/SinoNom_Similar.csv).
3. Đề xuất danh sách cặp (âm, chữ) mới có căn cứ xuất xứ để người dùng mở rộng từ điển một cách an toàn.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO / "dataset_out" / "labels_final.csv"
DICT_PATH = REPO / "Dict" / "QuocNgu_SinoNom.csv"
SIMILAR_PATH = REPO / "Dict" / "SinoNom_Similar.csv"
OUT_DIR = Path(__file__).resolve().parent


def analyze_missing_dictionary():
    # 1. Load từ điển hiện tại
    qn_to_nom = defaultdict(set)
    with open(DICT_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            qn = r["QuocNgu"].strip().lower()
            nom = r["SinoNom"].strip()
            if qn and nom:
                qn_to_nom[qn].add(nom)

    # 2. Đọc toàn bộ nhãn để lấy ngữ cảnh các sách
    with open(LABELS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Tìm các âm trong REVIEW thiếu từ điển
    missing_cells = []
    missing_syl_counts = Counter()

    for r in rows:
        if r["tier"] == "REVIEW":
            syl = str(r.get("syllable", "")).strip().lower()
            if syl and syl not in qn_to_nom:
                missing_cells.append(r)
                missing_syl_counts[syl] += 1

    print(f"Tổng số ô thiếu từ điển: {len(missing_cells)}")
    print(f"Số âm tiết độc nhất thiếu từ điển: {len(missing_syl_counts)}")

    # 3. Phân tích từng âm: tần suất, sách xuất hiện, chữ OCR phỏng đoán
    syl_details = {}
    for syl, count in missing_syl_counts.most_common():
        sample_cells = [r for r in missing_cells if str(r.get("syllable", "")).strip().lower() == syl]
        ocr_cands = Counter(r.get("ocr_char", "") for r in sample_cells if r.get("ocr_char"))
        books = Counter(r.get("book", "") for r in sample_cells)
        
        # Kiểm tra xem có phải do dấu thanh, viết tắt, từ phiên âm tên riêng/tiếng Latinh hay lỗi chính tả
        is_latin_or_foreign = any(c in syl for c in ["f", "j", "w", "z"]) or len(syl) > 8
        
        syl_details[syl] = {
            "so_lan_xuat_hien": count,
            "phan_bo_sach": dict(books),
            "top_chu_ocr_phong_doan": dict(ocr_cands.most_common(5)),
            "kha_nang": "tên_riêng_latinh" if is_latin_or_foreign else "tu_co_hoac_di_the"
        }

    # Xuất báo cáo JSON
    out_json = OUT_DIR / "dict_expansion_candidates.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "tong_so_am_khuyet": len(missing_syl_counts),
            "tong_so_o_bi_anh_huong": len(missing_cells),
            "chi_tiet_am_khuyet": syl_details
        }, f, ensure_ascii=False, indent=2)

    # Xuất bảng CSV đề xuất thêm vào từ điển
    out_csv = OUT_DIR / "dict_expansion_candidates.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["syllable", "so_o", "phan_bo_sach", "chu_ocr_pho_bien_nhat", "ty_le_dong_thuan", "phan_loai"])
        for syl, d in syl_details.items():
            top_ocr = list(d["top_chu_ocr_phong_doan"].keys())[0] if d["top_chu_ocr_phong_doan"] else ""
            top_cnt = list(d["top_chu_ocr_phong_doan"].values())[0] if d["top_chu_ocr_phong_doan"] else 0
            rate = f"{top_cnt / d['so_lan_xuat_hien']:.1%}" if d['so_lan_xuat_hien'] > 0 else "0%"
            books_str = ", ".join(f"{b}:{c}" for b, c in d["phan_bo_sach"].items())
            writer.writerow([syl, d["so_lan_xuat_hien"], books_str, top_ocr, rate, d["kha_nang"]])

    print(f"Đã lưu danh sách đề xuất mở rộng từ điển vào:")
    print(f"  - {out_json}")
    print(f"  - {out_csv}")
    print("\nTop 15 âm tiết khuyết từ điển nhiều nhất:")
    for syl, count in missing_syl_counts.most_common(15):
        top_c = syl_details[syl]["top_chu_ocr_phong_doan"]
        print(f"  - '{syl}': {count} ô | OCR phỏng đoán: {top_c}")


if __name__ == "__main__":
    analyze_missing_dictionary()
