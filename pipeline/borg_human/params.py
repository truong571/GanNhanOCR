"""params.py — THAM SỐ ĐÓNG BĂNG của pipeline.borg_human (không chọn lại; đổi một giá trị = phải đo lại vòng 4/5).

Mỗi khối ghi NGUỒN (script gốc của vòng 4 `r4/borg_align/scripts/*` và vòng 5 `r5/borg_quality/*`, bản lưu bền ở
measure_out/_audit_2026-09-26/). REF = số đo gốc dùng để kiểm tái lập (`python -m pipeline.borg_human --stage repro`).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "measure_out" / "_audit_2026-09-26"
R4 = AUDIT / "r4" / "borg_align"                   # a01..a14 + blib.py (vòng 4, 26/09)
R5 = AUDIT / "r5" / "borg_quality"                 # q01..q14 + qlib.py (vòng 5, 26/09)
SRC = REPO / "measure_out" / "_borg_human_src"      # đầu vào đóng băng (hồ sơ Paddle) + SHA256SUMS
WORK = REPO / "measure_out" / "_borg_human"         # trung gian (hộp, đơn vị, nhúng, căn) — không commit
OUT = REPO / "dataset" / "_BORG_NHAN_NGUOI"         # bản giao (TÁCH khỏi GOLD tự động và dataset/_ALL)

# thứ tự sách = thứ tự dict BOOKS của blib.py (quyết định thứ tự đơn vị -> uid khớp hồ sơ Paddle)
BOOKS = {
    "SachKinhThayCaBinh": dict(data="data/SachKinhThayCaBinh", shelfmark="Borg.tonch.18",
                               iiif="https://digi.vatlib.it/iiif/MSS_Borg.tonch.18/manifest.json",
                               title="Sách kinh Thầy cả Bỉnh"),
    "SachDungLyHoThan": dict(data="data/SachDungLyHoThan", shelfmark="Borg.tonch.34",
                             iiif="https://digi.vatlib.it/iiif/MSS_Borg.tonch.34/manifest.json",
                             title="Sách Dũng Lý Hộ Thần"),
}

# ---- a01_detect.py: CenterNet v1 (STT) trên prepared/<b>/pages (= stretch(gray(jpg)), mad = 0), hộp thô thr >= 0.05
DETECTOR = dict(ckpt="train_crop/detector_r34.best.pt", img=1024, thr=0.05, resize="linear", stretch_lo=1.0, stretch_hi=99.0)

# ---- blib.columns / nms_vertical: cột = cụm tâm-x hộp tin cậy, hộp yếu gán về cột gần nhất, xếp PHẢI -> TRÁI
COLUMNS = dict(thr_core=0.2, thr_min=0.08, gap_factor=0.6, nms_iou=0.45, nms_xo=0.5, weak_dx=0.5, fit_min=5)

# ---- a04_units.py: đơn vị = hộp thật + ô ẢO ở khe trống lớn; crop nhúng pad 0.12 + tighten_box trên ảnh xám prepared
UNITS = dict(pad=0.12, virt_h=0.85, gap_split=1.6, edge_open=0.6, line_min_boxes=8, core_thr=0.2)

# ---- encoder (score_visual.py của gold_img_audit/visual_verifier): v1 = nom-embed/best.pt, v2 = ArcFace/checkpoints/best.pt,
#      kéo giãn tương phản p2–p98 (norm=True), nhúng ghép [v1, v2]/sqrt(2); glyph = font NomNaTong/Plangothic + ảnh FD
ENCODERS = {"v1": "nom-embed/best.pt", "v2": "ArcFace/checkpoints/best.pt"}
FONTS = ("fonts/NomNaTong-Regular.ttf", "fonts/PlangothicP1-Regular.ttf", "fonts/PlangothicP2-Regular.ttf")
FD_DIRS = ("ArcFace/data/glyphs", "gannhanocr-fd")

# ---- a05_align.py: DP đơn điệu (Viterbi + forward-backward) chữ NGƯỜI ↔ đơn vị; phát xạ = cos chuẩn hoá theo hàng
DP = dict(b_real=1.5, b_virt=0.2, b_merge=0.3, skipc=-2.5, skipv=-0.05, skipr0=-0.3, skipr1=-3.0, merge_max=1.7, alpha=1.0)
PROTO = dict(n_folds=5, post_min=0.9, n_min=2, shrink=3.0)   # nguyên mẫu từ ô post>=0.9 của lượt trước, CHỈ từ khối trang KHÁC
# chuỗi lượt (BAO_CAO_ANH_CHU_CHINH_XAC_2026-09-26.md §9): font -> proto -> proto, cho từng encoder
PASSES = [("f_v1", "font", "v1", None), ("q1_v1", "proto", "v1", "f_v1"), ("q2_v1", "proto", "v1", "q1_v1"),
          ("f_v2", "font", "v2", None), ("q1_v2", "proto", "v2", "f_v2"), ("q2_v2", "proto", "v2", "q1_v2")]
FINAL, CHECK = "q2_v1", "q2_v2"                              # bộ căn cuối (v1) và bộ căn kiểm chéo (v2)

# ---- a12_final.py: luật keep (ưu tiên precision)
KEEP = dict(det_min=0.2, page_skip_max=0.1, post_v1=0.999, post_v2=0.999, post_v2_high=0.99)

# ---- a08_crops.py: crop giao = pipeline.align_engine.build_dataset.save_crop (pad 0.12 = step2.crop_pad_frac, carve láng
#      giềng prev/next cùng cột theo y, tighten; hình học trên prepared, điểm ảnh jpg gốc)
SAVE_CROP_PAD = 0.12

# ---- vòng 5 (r5/borg_quality): lọc bằng bộ đọc ĐỘC LẬP Paddle PP-OCRv6 (không học trên STT/Borg), hồ sơ ĐÓNG BĂNG
PADDLE = dict(profile="prof_borgx.pkl",
              sha256="53d3c64e1fd0a97e69699b6f91e7bbafd2317e90caa8b8bd4fabe786e27db330",
              delay_h=0.0946,     # q06_borg_slip.json 'C|ihr_delay_h' (trễ đỉnh CTC đo trên nhãn người IHR, slot_ok=1)
              D=3,                # cửa sổ 7 chữ c_{j-3..j+3}
              tau_norm=0.5,       # q13/q14: Paddle "thấy" dạng Hán L khi s(u, L) >= 0,5
              boot_B=300, boot_seed=0)
# q13/q14: 11 chữ Nôm N đã thăm dò -> dạng Hán L mà người phiên có thể đã chuẩn hoá
PROBE_PAIRS = {"𢧚": "年", "𠸜": "先", "𠀧": "巴", "𧘇": "意", "𠰺": "代", "𧶮": "古", "𠫾": "多", "𢚸": "弄", "𩈘": "末",
               "𡗊": "饒", "𨕭": "連"}
# q13.json t0.5 pi_LB (cận dưới tỉ lệ hộp gõ N mà hình là L) + CI 95 % bootstrap cụm trang — 4 cặp có cận dưới > 0
KNOWN_NORMALISED = {"𠸜": ("先", 0.7158, (0.6208, 0.8023)), "𢧚": ("年", 0.1812, (0.1387, 0.2228)),
                    "𠀧": ("巴", 0.0619, (0.0366, 0.0899)), "𧘇": ("意", 0.0504, (0.0384, 0.0645))}

# ---- xuất: split theo TRANG (sha256("book/page") mod 20: 0-1 test, 2-3 val, còn lại train) — không phụ thuộc thứ tự
SPLIT = dict(mod=20, test=(0, 1), val=(2, 3))
KEEP_LEVELS = ("keep_v5", "keep", "keep_high", "khong")      # mức cao nhất đạt được (keep_v5 ⊂ keep ⊂ keep_high)
CROP_LEVELS = ("keep_v5", "keep")    # mức có ảnh crop + crop chuẩn (keep_high thêm ≈ 16 k ô ⇒ vượt ~1 GB: không kèm ảnh)

# ---- số đo GỐC để kiểm tái lập (r4 summary.json / borg_cells.csv, r5 q14.json / q06_borg_slip.json / q07)
REF = dict(
    cells=143043, cells_book={"SachKinhThayCaBinh": 118935, "SachDungLyHoThan": 24108},
    keep=56261, keep_high=71951, keep_book={"SachKinhThayCaBinh": 50740, "SachDungLyHoThan": 5521},
    kinds={"real": 138885, "skip": 3603, "virtual": 405, "merge": 150},
    keep_paddle_flagged=1315, keep_paddle_ok=54946, removed_normalised=62, keep_v5=54884,
    removed_by_char={"𠸜": 57, "𢚸": 3, "𠀧": 2},
    theta_paddle_keep=dict(n=15383, theta=0.0132, ci=(0.0094, 0.0169)),     # q06 'D|borg_keep|mid+delay'
    theta_T1_v2_keep=dict(n=50155, theta=0.0127, ci=(0.0073, 0.0181)),      # r4 validate_q2_v1_final_keep 'T1_v2|keep'
    theta_T2_v2_keep=dict(n=43357, theta=0.0012, ci=(0.0005, 0.0019)),      # (v2 tham gia chọn keep -> lệch lạc quan)
)
