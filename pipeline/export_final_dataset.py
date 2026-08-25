"""Xuất bộ dataset CUỐI CÙNG (usable: GOLD+SILVER+SYLLABLE) từ dataset_out/ ra
một thư mục TỰ CHỨA (labels.csv + ảnh crop copy hẳn, không phụ thuộc dataset_out/).

Gọi bởi run_pipeline.sh sau bước remediate. XOÁ SẠCH --out trước khi ghi, nên
thư mục đó LUÔN LÀ bản mới nhất của lần chạy gần nhất — không cộng dồn qua các
lần chạy trước.

Usage:
    python3 pipeline/export_final_dataset.py \
        --labels dataset_out/labels_remediated.csv --src-root dataset_out --out dataset
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

USABLE_TIERS = {"GOLD", "SILVER", "SYLLABLE"}


def export_dataset(labels_path: Path, src_root: Path, out_root: Path) -> int:
    if not labels_path.exists():
        print(f"[export] không thấy {labels_path}", file=sys.stderr)
        return 1

    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)
        rows = [r for r in all_rows if r.get("tier") in USABLE_TIERS]

    # LOẠI TRỪ PHẢI ỒN ÀO. Từ 2026-08-19 tier SILVER_uncalibrated (10.890 ô do S3
    # quyết, 0 verdict người) nằm ngoài USABLE_TIERS nên tự rơi khỏi bộ giao nộp —
    # nếu im lặng thì một hôm nào đó 10.890 ô biến mất mà không ai biết vì sao.
    excluded = Counter(r.get("tier") for r in all_rows if r.get("tier") not in USABLE_TIERS)
    for tier, n in sorted(excluded.items(), key=lambda kv: -kv[1]):
        print(f"[export] LOẠI khỏi bộ giao nộp: {tier or '(trống)':22} {n:>7,} dòng")

    if not rows:
        print("[export] 0 dòng usable (GOLD/SILVER/SYLLABLE) trong "
              f"{labels_path} — không ghi gì vào {out_root}.", file=sys.stderr)
        return 1

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    n_copied = 0
    n_missing = 0
    for r in rows:
        rel = r["image"]
        src = src_root / rel
        if not src.exists():
            n_missing += 1
            continue
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1

    # GẮN CỜ ẢNH DÙNG ĐƯỢC — PHẢI ĐẶT TRƯỚC KHI GHI.
    # Lỗi thật 2026-08-25: khối này từng nằm SAU w.writerows(), nên CSV ra cột RỖNG
    # trong khi log vẫn báo "424 ô" — log TRẤN AN SAI, còn đội chấm thì không nhận được
    # cảnh báo nào. Đó là cả mục đích của cột này.
    for r in rows:
        r["usable_image"] = "0" if r.get("crop_quality_flag") in ("blank", "truncated") else "1"

    # GẮN CỜ TRANG KHÔNG ĐỦ 9 CỘT. Bố cục ván khắc LUÔN 9 cột, nên thiếu cột là dấu hiệu
    # phép ghép cột Nôm<->Quốc ngữ trên trang đó có thể đã trượt.
    # Cờ này chỉ nói MỘT sự việc đo được (trang không đủ 9 cột), KHÔNG kết luận nhãn sai —
    # trong 3 trang bắt được, stt4/page_0252 có REVIEW 19% ≈ mức chung 17,9% nên nhiều khả
    # năng lành, còn stt4/page_0110 (68%) và stt11/page_0010_p0028 (70%) thì hỏng thật.
    # Tỷ lệ REVIEW in kèm bên dưới để người đọc tự phân định.
    # Đáng chú ý: phép kiểm "9 cột" chạy sau extract ĐẾM CỘT OCR NÔM nên nó cho
    # stt4/page_0110 đi qua (Nôm đủ 9) dù chỉ 8 cột sống sót tới bước ghép. Đây là chỗ
    # duy nhất bắt được cả hai phía.
    # Đếm trên TOÀN BỘ hàng (all_rows) chứ không phải hàng đã lọc — bộ giao nộp bỏ REVIEW
    # nên một trang lành mà cả cột rơi vào REVIEW sẽ bị đếm hụt và mang tiếng oan.
    _cot_theo_trang: dict[tuple, set] = {}
    for r in all_rows:
        _cot_theo_trang.setdefault((r.get("book"), r.get("page")), set()).add(r.get("column"))
    _trang_lech = {k for k, v in _cot_theo_trang.items() if len(v) != 9}
    for r in rows:
        r["page_cot_lech"] = "1" if (r.get("book"), r.get("page")) in _trang_lech else "0"
    _n_lech = sum(1 for r in rows if r["page_cot_lech"] == "1")

    # TÍNH LẠI label_in_train TRÊN ĐÚNG BỘ ĐƯỢC CÔNG BỐ.
    # build_dataset tính cột này ở Bước 3 trên TOÀN BỘ 82k hàng — GOLD + SILVER + REVIEW
    # gộp lại — rồi đóng băng. Nhưng bộ giao nộp chỉ có GOLD + SYLLABLE, và giữa hai mốc
    # đó confusion-fix còn hạ 1.988 hàng GOLD xuống REVIEW. Hệ quả: một lớp chữ chỉ còn
    # sống trong SILVER/REVIEW của phía train vẫn bị ghi là "có trong train", nên ô val/test
    # mang lớp đó KHÔNG được cảnh báo. Đo được 31 hàng sai (cột ghi 128, số thật 159).
    # Đây là cột người ta lọc để đánh giá, nên sai ở đây là chỉ số sai mà không ai biết.
    _train_lop = {r["label"] for r in rows
                  if r.get("split") == "train" and r.get("label_level") == "char" and r.get("label")}
    _sua = 0
    for r in rows:
        moi = ("1" if r["label"] in _train_lop else "0") \
            if (r.get("label_level") == "char" and r.get("label")) else ""
        if moi != r.get("label_in_train"):
            _sua += 1
        r["label_in_train"] = moi
    _unseen = sum(1 for r in rows if r["label_in_train"] == "0")

    with open(out_root / "labels.csv", "w", encoding="utf-8", newline="") as f:
        for _c in ("usable_image", "page_cot_lech"):
            if _c not in fieldnames:
                fieldnames = list(fieldnames) + [_c]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    tiers = Counter(r["tier"] for r in rows)
    # TÁCH BẠCH HAI LOẠI NHÃN (2026-08-25, sau phản biện hội đồng). Gộp GOLD+SYLLABLE
    # thành một con số là THỔI SỐ: tầng SYLLABLE có cột `label` RỖNG — nó chỉ ghi ÂM
    # Quốc ngữ, KHÔNG gán chữ Nôm nào. Sản phẩm gán nhãn CẤP KÝ TỰ chỉ là phần GOLD.
    # GẮN CỜ ẢNH DÙNG ĐƯỢC (2026-08-25). Đo được 424 ô có ảnh hỏng (34+330 GOLD, 6+54
    # SYLLABLE) vẫn nằm trong bộ giao nộp. KHÔNG loại chúng: nhãn có thể vẫn ĐÚNG dù
    # ảnh hỏng, và loại đi sẽ đổi số + phá chuỗi băm. Thay vào đó gắn cờ để (a) người
    # chấm biết bỏ qua chiều ẢNH thay vì chấm nhầm thành "nhãn sai", (b) người huấn
    # luyện mô hình lọc được.
    _nbad = sum(1 for r in rows if r["usable_image"] == "0")
    _nchar = sum(1 for r in rows if (r.get("label") or "").strip())
    _nsyl = len(rows) - _nchar
    print(f"[export] {out_root}/labels.csv — {_nchar:,} nhãn CẤP KÝ TỰ "
          f"+ {_nsyl:,} chú giải CẤP ÂM TIẾT = {len(rows):,} dòng "
          f"(GOLD {tiers.get('GOLD', 0)}, SILVER {tiers.get('SILVER', 0)}, "
          f"SYLLABLE {tiers.get('SYLLABLE', 0)})")
    print(f"[export] ⚠️ con số đem so với bộ dữ liệu Hán Nôm khác là {_nchar:,}, "
          f"KHÔNG phải {len(rows):,}")
    if _n_lech:
        _rv_all = sum(1 for r in all_rows if r.get("tier") == "REVIEW") / max(1, len(all_rows))
        print(f"[export] 🔴 page_cot_lech=1: {_n_lech:,} ô trên {len(_trang_lech)} trang "
              f"KHÔNG đủ 9 cột (mức REVIEW chung {100*_rv_all:.1f}%):")
        for b, pg in sorted(_trang_lech):
            _pr = [r for r in all_rows if r.get("book") == b and r.get("page") == pg]
            _rv = sum(1 for r in _pr if r.get("tier") == "REVIEW") / max(1, len(_pr))
            _co = sorted({int(r["column"]) for r in _pr})
            print(f"           {b}/{pg}: cột {_co} · REVIEW {100*_rv:.0f}%"
                  f"{'  <- lệch xa mức chung, ghép cột nhiều khả năng đã trượt' if _rv > 2*_rv_all else ''}")
    print(f"[export] label_in_train: {_unseen:,} ô có lớp chữ KHÔNG có trong train "
          f"của chính bộ này (tính lại tại bước xuất, sửa {_sua:,} ô so với Bước 3)")
    print(f"[export] usable_image=0 (ảnh trắng/cụt, ĐỪNG chấm chiều ảnh): {_nbad:,} ô")
    print(f"[export] ảnh: {n_copied} đã copy, {n_missing} thiếu trên đĩa")
    # export XOÁ SẠCH thư mục đích rồi ghi lại, nên tài liệu đi kèm biến mất theo.
    # run_pipeline gọi make_dataset_docs ngay sau đây; chạy TAY thì dễ quên, và bộ giao
    # bộ giao ra sẽ thiếu README/DATASHEET — không ai biết cột nào nghĩa là gì.
    if not (out_root / "README.md").exists():
        print(f"[export] ⚠️ {out_root}/ CHƯA có tài liệu đi kèm. Chạy tiếp:\n"
              f"    python -m pipeline.tools.make_dataset_docs --dataset {out_root}")
    if n_missing:
        print(f"[export] CẢNH BÁO: {n_missing} ảnh có trong {labels_path.name} "
              f"nhưng KHÔNG có file thật trong {src_root}/ — kiểm tra lại bước build.",
              file=sys.stderr)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="dataset_out/labels_remediated.csv")
    ap.add_argument("--src-root", default="dataset_out")
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()
    sys.exit(export_dataset(Path(args.labels), Path(args.src_root), Path(args.out)))


if __name__ == "__main__":
    main()
