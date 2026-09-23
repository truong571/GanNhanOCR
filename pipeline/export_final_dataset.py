"""Xuất bộ dataset CUỐI CÙNG (usable: GOLD+SILVER+SYLLABLE) từ dataset_out/ ra
một thư mục TỰ CHỨA (labels.csv + ảnh crop copy hẳn, không phụ thuộc dataset_out/).

Gọi bởi run_pipeline.sh sau bước remediate. XOÁ SẠCH --out trước khi ghi, nên
thư mục đó LUÔN LÀ bản mới nhất của lần chạy gần nhất — không cộng dồn qua các
lần chạy trước.

SCHEMA GIAO NỘP (A-9/A-10, DANH_MUC_SUA_DOI_CUOI_2026-09-16 §2-§3, từ 16/09):
    labels.csv        12 cột CỐ ĐỊNH (whitelist GIAO_NOP) — không split/label_in_train,
                      không cột chẩn đoán. Nội bộ `dataset_out/labels_final.csv` giữ đủ cột.
    labels_trace.csv  sidecar chẩn đoán, CÙNG số dòng/thứ tự, khoá `image`
                      (chỉ ghi cột CÓ trong nguồn — thế hệ cũ thiếu tier_v3/p_register…).
    columns.csv       một dòng mỗi cột (book,page,column): n_ocr, n_qn, n_det, count_source.
    GOLD_text_only    (chỉ sách lithograph đã qua B4' mechanism_gates) dòng vào labels.csv với tier
                      này, ảnh crop KHÔNG copy; labels không có tier này -> hành vi y như trước.

Usage:
    python3 pipeline/export_final_dataset.py \
        --labels dataset_out/labels_final.csv --src-root dataset_out --out dataset
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

# Tầng GOLD_text_only (B4' pipeline/remediation/mechanism_gates.py, thạch bản 2026-09-22): nhãn
# văn bản GOLD nhưng hộp ảnh nghi lệch (n_det ≠ N / midpoint / split) → vào labels.csv, KHÔNG copy
# ảnh crop. Tầng chỉ xuất hiện khi cổng cơ chế chạy; labels STT không có → hành vi export như cũ.
TIER_TEXT_ONLY = "GOLD_text_only"
IMAGE_TIERS = {"GOLD", "SILVER", "SYLLABLE"}
USABLE_TIERS = IMAGE_TIERS | {TIER_TEXT_ONLY}

# 12 cột giao nộp — thứ tự cố định. Bỏ hẳn (hàm thuần của cột khác, xem §2): label_level
# (= f(tier)), usable_image (= f(crop_quality_flag)), page_cot_lech (suy từ columns.csv),
# split/split_group/label_in_train (bỏ chia train/val/test — README ghi công thức hash cũ),
# seg_backend (1 giá trị -> summary.json), readmitted_from_s3_demotion (rỗng toàn bộ).
GIAO_NOP = ["image", "book", "page", "column", "ocr_char", "syllable", "label",
            "unicode", "tier", "rule", "bbox", "image_md5"]
# Sidecar labels_trace.csv: chỉ những cột có mặt trong nguồn được ghi (không bịa cột rỗng).
TRACE = ["image", "nom_idx", "syl_idx", "syllable_ocr", "syllable_raw", "qn_fix_kind", "tier_v3",
         "tier_goc", "rule_goc", "p_register", "dict_support", "context_evidence",
         "l1_support", "l1_tie", "flank_gold", "box_source", "qd01_locked", "qd01_excluded",
         "label_canonical", "crop_quality_flag", "stray_ink", "border_ink", "ink_pct",
         "crop_w", "crop_h", "seg_flag", "s3_cosine",
         # B4' cổng cơ chế (mechanism_gates): chỉ có ở sách lithograph đã qua cổng;
         # n_det_mismatch chỉ có khi cổng chạy ở chế độ pitch (luật a', 2026-09-22)
         "gate_reason", "di_ban_khac", "n_det_mismatch",
         # B-2 (--visual-emission): chỉ có khi build bật cờ; tắt cờ -> không ghi (không bịa cột)
         "p_visual_syl", "visual_fold", "visual_argmax", "visual_max_p"]
# columns.csv: khoá (book,page,column) + các đại lượng cấp CỘT (giá trị đầu tiên gặp).
COT_KHOA = ["book", "page", "column"]
COT_CSV = ["n_ocr", "n_qn", "n_det", "count_source"]


def export_dataset(labels_path: Path, src_root: Path, out_root: Path, n_columns: int = 9) -> int:
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
    # Cột giao nộp thiếu trong nguồn -> LỖI trước khi xoá gì, không âm thầm ghi rỗng:
    # schema cố định 12 cột là cả mục đích của whitelist.
    _thieu = [c for c in GIAO_NOP if c not in fieldnames]
    if _thieu:
        print(f"[export] {labels_path} THIẾU cột giao nộp bắt buộc: {_thieu}", file=sys.stderr)
        return 1

    # XOÁ SẠCH RỒI GHI LẠI — nhưng CHỈ xoá thứ do chính bước này sinh ra.
    # Sự cố thật 2026-09-09: re-dataset/check/ (bộ kiểm độc lập do người dùng viết, CÓ
    # trong git) bị rmtree nuốt mất khi chạy lại export. Thư mục đích không phải của
    # riêng bước xuất: người ta đặt thêm việc của họ vào đó, và một lệnh dọn dẹp không
    # có quyền xoá thứ mình không tạo ra.
    SINH_BOI_BUOC_NAY = {"gold", "silver", "syllable", "review"}
    if out_root.exists():
        for m in out_root.iterdir():
            if m.is_dir():
                if m.name in SINH_BOI_BUOC_NAY:
                    shutil.rmtree(m)
                else:
                    print(f"[export] GIỮ LẠI thư mục không do bước này sinh: {m.name}/")
            elif m.suffix in {".csv", ".md", ".xlsx"}:
                m.unlink()          # labels.csv + tài liệu đi kèm: sinh lại ngay dưới đây
    out_root.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing = 0
    n_text_only = 0
    for r in rows:
        if r.get("tier") == TIER_TEXT_ONLY:
            n_text_only += 1          # nhãn văn bản vào labels.csv; ảnh KHÔNG giao (hộp nghi lệch)
            continue
        rel = r["image"]
        src = src_root / rel
        if not src.exists():
            n_missing += 1
            continue
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1

    # CỜ ẢNH DÙNG ĐƯỢC: KHÔNG còn là cột (usable_image = f(crop_quality_flag), §2). Vẫn
    # đếm để log — lỗi thật 2026-08-25 là cột rỗng mà log vẫn trấn an "424 ô"; nay cờ sống
    # ở labels_trace.csv (crop_quality_flag), người chấm lọc `blank`/`truncated` ở đó.
    _nbad = sum(1 for r in rows if r.get("crop_quality_flag") in ("blank", "truncated"))

    # TRANG KHÔNG ĐỦ 9 CỘT. Bố cục ván khắc LUÔN 9 cột, nên thiếu cột là dấu hiệu phép ghép
    # cột Nôm<->Quốc ngữ trên trang đó có thể đã trượt. Cờ chỉ nói MỘT sự việc đo được, KHÔNG
    # kết luận nhãn sai — trong 3 trang bắt được ở lần 25/08, stt4/page_0252 có REVIEW 19% ≈
    # mức chung 17,9% nên nhiều khả năng lành, còn stt4/page_0110 (68%) và
    # stt11/page_0010_p0028 (70%) thì hỏng thật. Tỷ lệ REVIEW in kèm để người đọc tự phân định.
    # Phép kiểm "9 cột" chạy sau extract ĐẾM CỘT OCR NÔM nên cho stt4/page_0110 đi qua (Nôm đủ
    # 9) dù chỉ 8 cột sống sót tới bước ghép — đây là chỗ duy nhất bắt được cả hai phía.
    # Đếm trên TOÀN BỘ hàng (all_rows), không phải hàng đã lọc — bộ giao nộp bỏ REVIEW nên
    # một trang lành mà cả cột rơi vào REVIEW sẽ bị đếm hụt và mang tiếng oan.
    # Từ 16/09 KHÔNG ghi cột page_cot_lech nữa: columns.csv (dựng từ all_rows) đủ để
    # make_dataset_docs suy lại danh sách trang; ở đây chỉ log.
    _cot_theo_trang: dict[tuple, set] = {}
    for r in all_rows:
        _cot_theo_trang.setdefault((r.get("book"), r.get("page")), set()).add(r.get("column"))
    # n_columns: 9 (STT, mặc định) | 10 (thạch bản LVT1883/KVK1884, --n-columns); chỉ đổi LOG, không đổi tệp.
    _trang_lech = {k for k, v in _cot_theo_trang.items() if len(v) != n_columns}
    _n_lech = sum(1 for r in rows if (r.get("book"), r.get("page")) in _trang_lech)

    # BỎ split/split_group/label_in_train (A-10 giai đoạn 1): ba cột là hàm thuần của
    # (book,page) — split_group == book|page, split == int(md5(book|page),16)%100 — nên
    # không mất thông tin; ai cần tự chia theo trang bằng công thức ghi trong README.
    # Khối "tính lại label_in_train trên đúng bộ được công bố" (25/08) xoá theo.

    # labels.csv — ĐÚNG 12 cột, đúng thứ tự GIAO_NOP (cột thiếu đã chặn ở trên, trước khi xoá).
    with open(out_root / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GIAO_NOP, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # labels_trace.csv — cùng số dòng/thứ tự, khoá `image`; chỉ cột có trong nguồn.
    _trace_cols = [c for c in TRACE if c in fieldnames]
    with open(out_root / "labels_trace.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_trace_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # columns.csv — một dòng mỗi cột (book,page,column) trên TOÀN BỘ hàng (kể cả cột chỉ
    # có REVIEW): n_ocr/n_qn/n_det là sự thật cấp cột, không phụ thuộc tier; và nhờ đủ cột
    # mới suy được trang thiếu cột. Giá trị = dòng ĐẦU TIÊN gặp của cột đó.
    _cot_csv = [c for c in COT_CSV if c in fieldnames]
    _cot_seen: dict[tuple, dict] = {}
    for r in all_rows:
        k = tuple(r.get(c, "") for c in COT_KHOA)
        if k not in _cot_seen:
            _cot_seen[k] = {**{c: r.get(c, "") for c in COT_KHOA},
                            **{c: r.get(c, "") for c in _cot_csv}}
    with open(out_root / "columns.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COT_KHOA + _cot_csv)
        w.writeheader()
        w.writerows(_cot_seen.values())

    tiers = Counter(r["tier"] for r in rows)
    # TÁCH BẠCH HAI LOẠI NHÃN (2026-08-25, sau phản biện hội đồng). Gộp GOLD+SYLLABLE
    # thành một con số là THỔI SỐ: tầng SYLLABLE có cột `label` RỖNG — nó chỉ ghi ÂM
    # Quốc ngữ, KHÔNG gán chữ Nôm nào. Sản phẩm gán nhãn CẤP KÝ TỰ chỉ là phần GOLD.
    # ẢNH HỎNG (2026-08-25). Đo được 424 ô có ảnh hỏng (34+330 GOLD, 6+54 SYLLABLE) vẫn
    # nằm trong bộ giao nộp. KHÔNG loại chúng: nhãn có thể vẫn ĐÚNG dù ảnh hỏng, và loại
    # đi sẽ đổi số + phá chuỗi băm. Cờ crop_quality_flag ở labels_trace.csv để (a) người
    # chấm bỏ qua chiều ẢNH thay vì chấm nhầm thành "nhãn sai", (b) người huấn luyện lọc.
    _nchar = sum(1 for r in rows if (r.get("label") or "").strip())
    _nsyl = len(rows) - _nchar
    print(f"[export] {out_root}/labels.csv — {_nchar:,} nhãn CẤP KÝ TỰ "
          f"+ {_nsyl:,} chú giải CẤP ÂM TIẾT = {len(rows):,} dòng "
          f"(GOLD {tiers.get('GOLD', 0)}, SILVER {tiers.get('SILVER', 0)}, "
          f"SYLLABLE {tiers.get('SYLLABLE', 0)})")
    print(f"[export] ⚠️ con số đem so với bộ dữ liệu Hán Nôm khác là {_nchar:,}, "
          f"KHÔNG phải {len(rows):,}")
    if n_text_only:
        print(f"[export] tầng {TIER_TEXT_ONLY}: {n_text_only:,} ô có nhãn ký tự trong labels.csv "
              f"nhưng KHÔNG giao ảnh crop (cổng cơ chế B4': hộp nghi lệch — gate_reason ở labels_trace.csv); "
              f"GOLD có ảnh = {tiers.get('GOLD', 0):,}")
    print(f"[export] labels.csv {len(GIAO_NOP)} cột · labels_trace.csv {len(_trace_cols)} cột "
          f"(thiếu trong nguồn: {[c for c in TRACE if c not in fieldnames] or 'không'}) · "
          f"columns.csv {len(_cot_seen):,} cột trang (cột đo: {_cot_csv or 'chưa có'})")
    if _n_lech:
        _rv_all = sum(1 for r in all_rows if r.get("tier") == "REVIEW") / max(1, len(all_rows))
        print(f"[export] 🔴 trang KHÔNG đủ {n_columns} cột: {_n_lech:,} ô trên {len(_trang_lech)} trang "
              f"(mức REVIEW chung {100*_rv_all:.1f}%; suy lại được từ columns.csv):")
        for b, pg in sorted(_trang_lech):
            _pr = [r for r in all_rows if r.get("book") == b and r.get("page") == pg]
            _rv = sum(1 for r in _pr if r.get("tier") == "REVIEW") / max(1, len(_pr))
            _co = sorted({int(r["column"]) for r in _pr})
            print(f"           {b}/{pg}: cột {_co} · REVIEW {100*_rv:.0f}%"
                  f"{'  <- lệch xa mức chung, ghép cột nhiều khả năng đã trượt' if _rv > 2*_rv_all else ''}")
    print(f"[export] ảnh trắng/cụt (crop_quality_flag blank/truncated ở labels_trace.csv, "
          f"ĐỪNG chấm chiều ảnh): {_nbad:,} ô")
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
    ap.add_argument("--n-columns", type=int, default=9,
                    help="số cột kỳ vọng mỗi trang cho LOG trang thiếu cột (STT 9 mặc định; thạch bản 10)")
    args = ap.parse_args()
    sys.exit(export_dataset(Path(args.labels), Path(args.src_root), Path(args.out), args.n_columns))


if __name__ == "__main__":
    main()
