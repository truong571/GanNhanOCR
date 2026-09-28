#!/usr/bin/env python3
"""
web/server.py — máy chủ trình diễn GanNhanOCR (luận văn Thạc sĩ: gán nhãn tự động văn bản Hán Nôm cổ).

CHỈ dùng thư viện chuẩn Python (ThreadingHTTPServer). CHỈ ĐỌC dữ liệu dưới gốc dự án, không bao giờ ghi:

  dataset/<Bộ>/labels.csv (+ labels_trace.csv -> khoá cell_uid y hệt pipeline/tools/merge_datasets._uid)
  dataset/_ALL/gold_exact.csv (GOLD chính xác, khoá cell_uid; vắng -> dataset/<Bộ>/gold_exact.csv), dataset/_ALL/crops_chuan/
  dataset/_ALL/{labels.csv,SOURCES.json} · dataset/_BORG_NHAN_NGUOI/{labels.csv,crops/,crops_chuan/,BUILD_INFO.json}
  prepared/<Bộ>/pages/ (Borg tự động: prepared/_auto/<Sách>/pages/) · measure_out/**/summary.json · measure_out/SUMMARY.json
  prepared/<LVT1883|KVK1884>/dataset_out/auto_precision_{verify,gated}/SUMMARY.json · Dict/QuocNgu_SinoNom.csv

Mọi con số trả về đều ĐỌC từ tệp; tệp thiếu -> "chưa có" (không bịa). Mức chắc của số độ chính xác ghi rõ:
ĐO trên nhãn người · ƯỚC LƯỢNG (qua dị bản người) · SUY ĐOÁN (không có sự thật người).

Gốc dữ liệu: --root <thư mục> > biến môi trường GANNHANOCR_ROOT > thư mục cha của web/.

Chạy:   .venv/bin/python web/server.py [cổng] [--root DIR] [--host 0.0.0.0]
API:    /api/stats · /api/books · /api/page?book=&page= · /api/search?q=&book=&tier=&gx=&limit=
        /api/pipeline_flow · /api/benchmarks · /api/reload (nạp lại dữ liệu sau khi pipeline chạy xong)
Tệp:    /crops/<sách>/<ảnh> · /crops_chuan/<đường> · /borg_human/<đường> · /page_scans/<sách>/<tệp>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

csv.field_size_limit(1 << 30)

WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parent
NA = "chưa có"


def default_root() -> Path:
    env = os.environ.get("GANNHANOCR_ROOT", "").strip()
    return Path(env).expanduser().resolve() if env else REPO_ROOT


# =====================================================================================================================
# Danh mục 10 bộ + 2 mục xem nhãn người (chỉ là SIÊU DỮ LIỆU mô tả — mọi con số đọc từ tệp)
# =====================================================================================================================
# Mức chắc của độ chính xác ô ok theo bộ = bản chép pipeline/gold_exact/common.EVIDENCE (chỉ dùng khi gold_exact.csv vắng).
EVIDENCE_OF = {"L16": "do_tren_nhan_nguoi", "TK": "do_tren_nhan_nguoi", "B18": "do_tren_nhan_nguoi",
               "B34": "do_tren_nhan_nguoi", "KVK": "uoc_luong", "L83": "uoc_luong",
               "stt2": "suy_doan", "stt4": "suy_doan", "stt11": "suy_doan", "Chr": "suy_doan"}
EVIDENCE_VI = {"do_tren_nhan_nguoi": "ĐO trên nhãn người", "uoc_luong": "ƯỚC LƯỢNG (qua dị bản người)",
               "suy_doan": "SUY ĐOÁN (không có sự thật người)"}
LEVEL_OF_EVIDENCE = {"do_tren_nhan_nguoi": "do", "uoc_luong": "uoc_luong", "suy_doan": "suy_doan"}
LEVEL_VI = {"do": "ĐO trên nhãn người", "uoc_luong": "ƯỚC LƯỢNG (qua dị bản người)",
            "suy_doan": "SUY ĐOÁN (không có sự thật người)", "do_tu_dong": "ĐO trên tham chiếu tự động"}
ROLE_VI = {"giao_nop": "Giao nộp", "danh_gia_ihr": "Đánh giá (IHR-NomDB)", "danh_gia_borg": "Đánh giá (Borg, Vatican)",
           "nhan_nguoi": "Bộ crop nhãn người (Borg)"}
STATUS_VI = {"ok": "Có dữ liệu", "ban_cu": "Bản xuất cũ (23/09) — chưa có bản dựng mới",
             "tu_bo_gop": "Đọc từ bộ gộp dataset/_ALL",
             "chua_co": "Chưa có dữ liệu (đang dựng lại hoặc chưa chạy pipeline)", "loi": "Lỗi đọc dữ liệu"}

GX_STATES = ("ok", "text_only", "uncertified", "review")
GX_VI = {"ok": "GOLD chính xác (ok)", "text_only": "Chỉ tin chữ (text_only)", "uncertified": "Chưa chứng nhận (uncertified)",
         "review": "Cần xem lại (review)", "chua_co": "GOLD nhưng chưa có gold_exact"}
GX_DESC = {
    "ok": "Ảnh và chữ cùng qua mọi cổng tự động; ảnh giao là crop chuẩn (khung vuông đúng một chữ).",
    "text_only": "Giữ nhãn chữ, nhưng ảnh chưa chắc là đúng MỘT chữ của đúng ô.",
    "uncertified": "Chưa đủ bằng chứng ảnh↔chữ (bộ kiểm dưới ngưỡng) hoặc dị bản người không chứng nhãn.",
    "review": "Luật A (ảnh của ô khác, nhãn cứu, cầu tự dạng, văn bản yếu) hoặc M-OCR bất đồng — cần xem lại.",
}
# Lý do (cột `reason` của gold_exact.csv, pipeline/gold_exact/policy.decide) -> lời giải thích tiếng Việt dễ hiểu.
REASON_VI = {
    "": "Qua mọi cổng tự động: ảnh là đúng một chữ của đúng ô và chữ được bộ kiểm chứng nhận.",
    "A0a_anh_cua_o_khac": "Ảnh lấy từ ô khác (ô được cứu dùng tệp crop của ô bên cạnh).",
    "A1_rescue_nhan_khong_do_OCR_doc": "Nhãn được \"cứu\" — không do OCR đọc ra trực tiếp.",
    "A2a_cau_tu_dang": "Nhãn chỉ nối được qua tự dạng gần giống (cầu tự dạng), chưa chắc.",
    "A2b_van_ban_yeu": "Chứng cứ văn bản yếu (âm sửa dấu, xác suất thấp hoặc đọc theo ngữ liệu).",
    "M_ocr_t50": "Các bộ OCR bất đồng về chữ này (M-OCR t50) — cần xem lại.",
    "M_ocr_t50_STT": "Các bộ OCR bất đồng về chữ này (M-OCR t50, Sách Thánh Truyện) — cần xem lại.",
    "A0_crop_chuan_trang": "Crop chuẩn gần như trắng — không thấy nét mực.",
    "A0_crop_chuan_cat_net": "Crop chuẩn cắt vào nét chữ.",
    "A0b_mot_hop_hai_cot": "Một hộp chữ bị gán cho hai cột.",
    "B0_hai_hop_chong_nang": "Hai hộp chữ chồng lấn nặng lên nhau.",
    "B0_hai_chu": "Crop chuẩn có vẻ chứa hai chữ.",
    "B0_muc_bat_thuong_so_voi_nhan": "Lượng mực bất thường so với chữ của nhãn.",
    "B0_muc_la_bleed_crop_moi": "Mực lạ (của chữ bên cạnh) tràn vào crop.",
    "B0_truncated_crop_moi": "Chữ bị cụt ở mép crop.",
    "B0_tall_crop_moi": "Crop quá cao — có thể dính chữ kề.",
    "CNT_so_chu_OCR_cot_khac_so_am_QN": "Số chữ OCR của cột khác số âm Quốc ngữ — dễ trượt ô.",
    "BC_truot_theo_hop_kim_vis": "Hộp lệch so với hộp chữ của OCR (kim) / tín hiệu thị giác — nghi trượt ô.",
    "HW_vis_z_duoi_0": "Chữ viết tay: tín hiệu thị giác của khe dưới 0 — nghi trượt ô.",
    "core_loss_crop_chuan_cat_vao_chu": "Crop chuẩn cắt vào lõi chữ.",
    "U_di_ban_nguoi_chong_nhan": "Dị bản do người số hoá mâu thuẫn với nhãn.",
    "U_STT_thieu_nguyen_mau_nguoi": "Sách Thánh Truyện: thiếu mẫu chữ người (< ngưỡng) để bộ kiểm chữ viết tay đối chiếu.",
    "U_STT_bo_kiem_viet_tay_khong_chung_nhan": "Sách Thánh Truyện: bộ kiểm chữ viết tay (LOBO) không chứng nhận ảnh↔chữ.",
    "U_Borg_thieu_nguyen_mau_nguoi_LOBO": "Borg: thiếu mẫu chữ người (học chéo sách) để đối chiếu.",
    "U_Borg_bo_kiem_viet_tay_LOBO_khong_chung_nhan": "Borg: bộ kiểm chữ viết tay học chéo sách (LOBO) không chứng nhận.",
    "U_bo_kiem_anh_duoi_nguong_TN1": "Bộ kiểm ảnh↔chữ dưới ngưỡng tin cậy (TN1).",
    "U_khong_co_van_ban_nguoi_chung": "Không có văn bản người (dị bản) chứng cho nhãn.",
}
KEEP_VI = {"keep_v5": "keep_v5 (khuyên dùng)", "keep": "keep", "keep_high": "keep_high (chỉ có hộp)", "khong": "không giữ"}


def reason_vi(code: str) -> str:
    return REASON_VI.get(code or "", f"Luật {code}" if code else REASON_VI[""])


def _book(id_, set8, book_set, title, subtitle, author, script, genre, n_columns, role, pages, default_page=None,
          book_key=None, legacy=None, ds_out=None, kind="auto"):
    return dict(id=id_, set8=set8, book_set=book_set, title=title, subtitle=subtitle, author=author, script=script,
                genre=genre, n_columns=n_columns, role=role, pages_dirs=pages, default_page=default_page,
                book_key=book_key, legacy=legacy, ds_out=ds_out, kind=kind,
                layout_desc=f"{genre} · {script} · {n_columns} cột/trang" if isinstance(n_columns, int)
                else f"{genre} · {script} · số cột theo trang")


STT_AUTHOR = "Khuyết danh (văn bản Công giáo)"
BOOKS = [
    _book("SachThanhTruyen2", "stt2", "SachThanhTruyen", "Sách Thánh Truyện (Quyển 2)",
          "Bản chép tay chữ Nôm Công giáo — quyển 2", STT_AUTHOR, "viết tay", "Văn xuôi", 9, "giao_nop",
          ["SachThanhTruyen2/pages"], "page_0012", book_key="stt2", legacy="SachThanhTruyen2"),
    _book("SachThanhTruyen4", "stt4", "SachThanhTruyen", "Sách Thánh Truyện (Quyển 4)",
          "Bản chép tay chữ Nôm Công giáo — quyển 4", STT_AUTHOR, "viết tay", "Văn xuôi", 9, "giao_nop",
          ["SachThanhTruyen4/pages"], "page_0012", book_key="stt4", legacy="SachThanhTruyen4"),
    _book("SachThanhTruyen11", "stt11", "SachThanhTruyen", "Sách Thánh Truyện (Quyển 11)",
          "Bản chép tay chữ Nôm Công giáo — quyển 11", STT_AUTHOR, "viết tay", "Văn xuôi", 9, "giao_nop",
          ["SachThanhTruyen11/pages"], "page_0010", book_key="stt11", legacy="SachThanhTruyen11"),
    _book("LucVanTien1883", "L83", "LucVanTien1883", "Lục Vân Tiên (1883)",
          "Bản thạch bản năm Quý Mùi (1883)", "Nguyễn Đình Chiểu", "thạch bản", "Thơ lục bát", 10, "giao_nop",
          ["LucVanTien1883/pages"], "page_0002", ds_out="prepared/LucVanTien1883/dataset_out"),
    _book("KimVanKieu1884", "KVK", "KimVanKieu1884", "Kim Vân Kiều (1884)",
          "Bản thạch bản Kim Vân Kiều tân truyện (Giáp Thân 1884)", "Nguyễn Du", "thạch bản", "Thơ lục bát", 10,
          "giao_nop", ["KimVanKieu1884/pages"], "page_0002", ds_out="prepared/KimVanKieu1884/dataset_out"),
    _book("Chrestomathie1872", "Chr", "Chrestomathie1872", "Chrestomathie Annamite (1872)",
          "Giáo trình văn xuôi tiếng An Nam (1872)", "Trương Vĩnh Ký / E. Luro", "sách in", "Văn xuôi", 7,
          "giao_nop", ["Chrestomathie1872/pages"], "page_0002", ds_out="prepared/Chrestomathie1872/dataset_out"),
    _book("LucVanTien1916", "L16", "LucVanTien1916", "Lục Vân Tiên (1916)",
          "Mộc bản 1916 — tập đánh giá IHR-NomDB (có nhãn người)", "Nguyễn Đình Chiểu", "mộc bản", "Thơ lục bát", 10,
          "danh_gia_ihr", ["LucVanTien1916/pages"], "page_0002", ds_out="prepared/LucVanTien1916/dataset_out"),
    _book("TruyenKieu1872", "TK", "TruyenKieu1872", "Truyện Kiều (1872)",
          "Mộc bản Duy Minh Thị (1872) — tập đánh giá IHR-NomDB (có nhãn người)", "Nguyễn Du", "mộc bản",
          "Thơ lục bát", 10, "danh_gia_ihr", ["TruyenKieu1872/pages"], "page_0002",
          ds_out="prepared/TruyenKieu1872/dataset_out"),
    _book("SachKinhThayCaBinh", "B18", "SachKinhThayCaBinh", "Sách kinh Thầy cả Bỉnh (Borg.tonch.18)",
          "Bản chép tay Vatican — chữ viết tay Công giáo, tập đánh giá (có nhãn người)", "Philiphê Bỉnh",
          "viết tay", "Văn xuôi", "auto", "danh_gia_borg",
          ["_auto/SachKinhThayCaBinh/pages", "SachKinhThayCaBinh/pages"],
          ds_out="prepared/_auto/SachKinhThayCaBinh/dataset_out"),
    _book("SachDungLyHoThan", "B34", "SachDungLyHoThan", "Sách Dũng Lý Hộ Thần (Borg.tonch.34)",
          "Bản chép tay Vatican — chữ viết tay Công giáo, tập đánh giá (có nhãn người)", "Khuyết danh",
          "viết tay", "Văn xuôi", "auto", "danh_gia_borg",
          ["_auto/SachDungLyHoThan/pages", "SachDungLyHoThan/pages"],
          ds_out="prepared/_auto/SachDungLyHoThan/dataset_out"),
    _book("BorgNguoi_SachKinhThayCaBinh", "B18", "SachKinhThayCaBinh", "Borg.tonch.18 — nhãn người",
          "Hộp + chữ do NGƯỜI phiên (dataset/_BORG_NHAN_NGUOI) — tách khỏi GOLD tự động", "Philiphê Bỉnh",
          "viết tay", "Văn xuôi", "auto", "nhan_nguoi",
          ["SachKinhThayCaBinh/pages", "_auto/SachKinhThayCaBinh/pages"], kind="human"),
    _book("BorgNguoi_SachDungLyHoThan", "B34", "SachDungLyHoThan", "Borg.tonch.34 — nhãn người",
          "Hộp + chữ do NGƯỜI phiên (dataset/_BORG_NHAN_NGUOI) — tách khỏi GOLD tự động", "Khuyết danh",
          "viết tay", "Văn xuôi", "auto", "nhan_nguoi",
          ["SachDungLyHoThan/pages", "_auto/SachDungLyHoThan/pages"], kind="human"),
]
BOOK_BY_ID = {b["id"]: b for b in BOOKS}
AUTO_IDS = [b["id"] for b in BOOKS if b["kind"] == "auto"]
BOOK_SETS = list(dict.fromkeys(b["book_set"] for b in BOOKS if b["kind"] == "auto"))
BOOK_GROUPS = {
    "all": AUTO_IDS,
    "giao_nop": [b["id"] for b in BOOKS if b["role"] == "giao_nop"],
    "danh_gia": [b["id"] for b in BOOKS if b["role"] in ("danh_gia_ihr", "danh_gia_borg")],
    "nhan_nguoi": [b["id"] for b in BOOKS if b["kind"] == "human"],
}
ALIASES = {
    "lvt1883": "LucVanTien1883", "l83": "LucVanTien1883", "lvt1916": "LucVanTien1916", "l16": "LucVanTien1916",
    "kieu1884": "KimVanKieu1884", "kvk": "KimVanKieu1884", "kieu1872": "TruyenKieu1872", "tk": "TruyenKieu1872",
    "chresto1872": "Chrestomathie1872", "chr": "Chrestomathie1872", "stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4",
    "stt11": "SachThanhTruyen11", "sachthanhtruyen": "SachThanhTruyen11", "b18": "SachKinhThayCaBinh",
    "borg18": "SachKinhThayCaBinh", "b34": "SachDungLyHoThan", "borg34": "SachDungLyHoThan",
    "borgnguoi18": "BorgNguoi_SachKinhThayCaBinh", "borgnguoi34": "BorgNguoi_SachDungLyHoThan",
}


def resolve_book_id(bid: str | None) -> str:
    """Chuẩn hoá mã sách (không phân biệt hoa thường, hỗ trợ bí danh)."""
    if not bid:
        return "LucVanTien1883"
    b = bid.strip()
    if b in BOOK_BY_ID or b in BOOK_GROUPS:
        return b
    low = b.lower()
    for k in BOOK_BY_ID:
        if k.lower() == low:
            return k
    return ALIASES.get(low, b)


def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt (tra cứu gõ không dấu)."""
    if not s:
        return ""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


# =====================================================================================================================
# Định dạng số kiểu Việt Nam
# =====================================================================================================================
def fmt_int(x) -> str:
    if x is None:
        return NA
    try:
        return f"{int(x):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(x)


def fmt_pct(x, d: int = 2, scale: float = 100.0) -> str:
    """tỉ lệ 0..1 (scale 100) hoặc đã là % (scale 1) -> '98,05 %'."""
    if x is None:
        return NA
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NA
    if math.isnan(v):
        return NA
    return f"{scale * v:.{d}f} %".replace(".", ",")


def fmt_ci(ci, d: int = 2, scale: float = 100.0) -> str:
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return ""
    if math.isnan(lo) or math.isnan(hi):
        return ""
    return f"[{scale * lo:.{d}f}–{scale * hi:.{d}f}]".replace(".", ",")


def dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# =====================================================================================================================
# Đọc tệp an toàn
# =====================================================================================================================
def rel_to(p: Path, root: Path) -> str:
    try:
        return str(Path(p).relative_to(root))
    except ValueError:
        return str(p)


def read_json(p: Path):
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — tệp đang ghi dở / hỏng -> coi như chưa có
        return None
    return None


def read_csv_cols(p: Path, cols: list[str]) -> tuple[list[tuple], list[str]]:
    """Đọc CHỈ các cột cần (nhanh, ít RAM). Cột thiếu -> ''. Trả (danh sách tuple, danh sách cột có mặt)."""
    with open(p, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd, None) or []
        idx = {c: i for i, c in enumerate(hdr)}
        pos = [idx.get(c) for c in cols]
        out = []
        for row in rd:
            n = len(row)
            out.append(tuple((row[i] if i is not None and i < n else "") for i in pos))
    return out, [c for c in cols if c in idx]


def image_size(p: Path) -> tuple[int, int] | None:
    """(rộng, cao) từ header PNG/JPEG — không cần thư viện ảnh."""
    try:
        with open(p, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":
                        b = f.read(1)
                    while b == b"\xff":
                        b = f.read(1)
                    if not b:
                        return None
                    marker = b[0]
                    if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                        continue
                    ln = int.from_bytes(f.read(2), "big")
                    if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                        f.read(1)
                        h = int.from_bytes(f.read(2), "big")
                        w = int.from_bytes(f.read(2), "big")
                        return w, h
                    f.seek(ln - 2, 1)
    except OSError:
        return None
    return None


def safe_join(base: Path, rel: str) -> Path | None:
    """base/rel nếu (sau khi resolve symlink) vẫn nằm trong base và là tệp; ngược lại None."""
    rel = unquote(rel or "").lstrip("/")
    if not rel or "\x00" in rel:
        return None
    parts = Path(rel).parts
    if any(p == ".." for p in parts):
        return None
    try:
        b = base.resolve()
        p = (b / rel).resolve()
    except (OSError, RuntimeError):
        return None
    if (p == b or b in p.parents) and p.is_file():
        return p
    return None


def parse_bbox(s: str):
    try:
        v = json.loads(s) if s else None
        if isinstance(v, list) and len(v) == 4:
            return [float(x) for x in v]
    except (ValueError, TypeError):
        pass
    return None


# =====================================================================================================================
# Kho dữ liệu nạp vào RAM (ảnh chụp bất biến; /api/reload thay nguyên khối)
# =====================================================================================================================
class Row:
    __slots__ = ("uid", "image", "page", "column", "ocr_char", "syllable", "syl_norm", "label", "unicode", "tier",
                 "rule", "bbox", "ge", "extra")

    def __init__(self, uid, image, page, column, ocr_char, syllable, label, unicode_, tier, rule, bbox, ge=None,
                 extra=None):
        self.uid, self.image, self.page, self.column = uid, image, page, column
        self.ocr_char, self.syllable, self.label, self.unicode = ocr_char, syllable, label, unicode_
        self.syl_norm = strip_accents(syllable)
        self.tier, self.rule, self.bbox, self.ge, self.extra = tier, rule, bbox, ge, extra


class GoldExact:
    """dataset/_ALL/gold_exact.csv (hoặc bản theo bộ) -> tra theo cell_uid và theo (book_set, ảnh tương đối bộ)."""

    COLS = ["cell_uid", "book_set", "book", "set8", "gold_exact", "reason", "evidence_level", "policy_version",
            "config_sha16", "image", "crop_chuan", "core_loss", "core_loss_flag", "one_char_ok", "crop_status"]

    def __init__(self):
        self.by_uid: dict[str, tuple] = {}
        self.by_img: dict[tuple, tuple] = {}
        self.counts: dict[str, Counter] = {}
        self.reasons: dict[str, Counter] = {}
        self.evidence: dict[str, str] = {}
        self.policy_version = self.config_sha16 = None
        self.sources: list[str] = []
        self.mode = None        # "all" | "theo_bo" | None

    @property
    def available(self) -> bool:
        return bool(self.by_uid)


def norm_crop_chuan(s: str) -> str:
    """'crops_chuan/…' (bộ gộp) hoặc '../_ALL/crops_chuan/…' (bản theo bộ) -> 'crops_chuan/…'."""
    if not s:
        return ""
    i = s.find("crops_chuan/")
    return s[i:] if i >= 0 else s


def load_gold_exact(root: Path, log=print) -> GoldExact:
    G = GoldExact()
    ds = root / "dataset"
    f_all = ds / "_ALL" / "gold_exact.csv"
    files = []
    if f_all.is_file():
        files, G.mode = [f_all], "all"
    else:
        files = [ds / bs / "gold_exact.csv" for bs in BOOK_SETS if (ds / bs / "gold_exact.csv").is_file()]
        G.mode = "theo_bo" if files else None
    for f in files:
        try:
            rows, _ = read_csv_cols(f, GoldExact.COLS)
        except Exception as e:  # noqa: BLE001
            log(f"  [!] không đọc được {f}: {e}")
            continue
        G.sources.append(rel_to(f, root))
        for (uid, bs, _bk, s8, st, why, ev, pv, cfg, img, cc, cl, clf, oc, cst) in rows:
            if not uid:
                continue
            rec = (st, why, ev, norm_crop_chuan(cc), cl, clf, oc, cst)
            G.by_uid[uid] = rec
            pre = f"crops/{bs}/"
            G.by_img[(bs, img[len(pre):] if img.startswith(pre) else img)] = rec
            G.counts.setdefault(s8, Counter())[st] += 1
            G.reasons.setdefault(s8, Counter())[why] += 1
            if ev:
                G.evidence[s8] = ev
            G.policy_version = G.policy_version or pv or None
            G.config_sha16 = G.config_sha16 or cfg or None
    if G.available:
        log(f"  [OK] gold_exact: {len(G.by_uid):,} ô GOLD ({G.mode}: {', '.join(G.sources)})")
    else:
        log("  [!] chưa có gold_exact.csv (dataset/_ALL hoặc dataset/<Bộ>) — trạng thái GOLD chính xác = chưa có")
    return G


class BookData:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rows: list[Row] = []
        self.pages: dict[str, list[int]] = {}
        self.status = "chua_co"
        self.source = None
        self.note = ""
        self.tiers: Counter = Counter()
        self.gx: Counter = Counter()           # trạng thái gold_exact của các dòng đã ghép (ô GOLD)
        self.n_gold = 0
        self.n_joined = 0
        self.crop_bases: list[Path] = []
        self.eval_marked = False
        self.scan_pages: list[str] = []
        self.pages_dir: Path | None = None
        self.extra_counts: dict = {}
        self.best_page: str | None = None     # trang minh hoạ tốt nhất (nhiều ô ok / GOLD / keep_v5 nhất)


def make_uid(bs: str, r: dict, ni: str, si: str, i: int) -> str:
    """Bản chép ĐÚNG pipeline/tools/merge_datasets._uid (khoá chính cell_uid của bộ gộp)."""
    if ni == "" and si == "":
        return f"{bs}/{r.get('book', '')}/{r.get('page', '')}/c{r.get('column', '')}/#{i}"
    return f"{bs}/{r.get('book', '')}/{r.get('page', '')}/c{r.get('column', '')}/n{ni}/s{si}"


def _scan_pages(root: Path, cfg: dict) -> tuple[Path | None, list[str]]:
    for rel in cfg["pages_dirs"]:
        d = root / "prepared" / rel
        if d.is_dir():
            try:
                names = sorted(p.stem for p in d.iterdir()
                               if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.stem.startswith("page_"))
            except OSError:
                names = []
            if names:
                return d, names
    return None, []


def _finish(bd: BookData, root: Path):
    good = "keep_v5" if bd.cfg["kind"] == "human" else "GOLD"
    score: dict[str, list] = {}
    for i, r in enumerate(bd.rows):
        bd.pages.setdefault(r.page, []).append(i)
        bd.tiers[r.tier] += 1
        s = score.setdefault(r.page, [0, 0, 0])
        s[0] += bool(r.ge and r.ge[0] == "ok")
        s[1] += r.tier == good
        s[2] += 1
    if score:
        bd.best_page = max(sorted(score), key=lambda pg: tuple(score[pg]))
    bd.pages_dir, bd.scan_pages = _scan_pages(root, bd.cfg)


LABEL_COLS = ["image", "book", "page", "column", "ocr_char", "syllable", "label", "unicode", "tier", "rule", "bbox"]
ALL_COLS = LABEL_COLS + ["cell_uid", "book_set", "evaluation_only"]


def load_auto_book(root: Path, cfg: dict, G: GoldExact, cache: dict, log=print) -> BookData:
    bd = BookData(cfg)
    ds = root / "dataset"
    bs, key = cfg["book_set"], cfg.get("book_key")
    cands = [("ok", ds / bs, key)]
    if cfg.get("legacy"):
        cands.append(("ban_cu", ds / cfg["legacy"], None))
    for status, d, k in cands:
        lp = d / "labels.csv"
        if not (lp.is_file() and lp.stat().st_size > 0):
            continue
        ck = str(d)
        if ck not in cache:
            L = read_csv_cols(lp, LABEL_COLS)[0]          # tuple theo LABEL_COLS (ít RAM hơn dict từng dòng)
            T = None
            tp = d / "labels_trace.csv"
            if tp.is_file():
                try:
                    T, _ = read_csv_cols(tp, ["nom_idx", "syl_idx"])
                    if len(T) != len(L):
                        T = None
                except Exception:  # noqa: BLE001
                    T = None
            cache[ck] = (L, T)
        L, T = cache[ck]
        if k and not any(r[1] == k for r in L):
            continue                      # bản gộp 3 quyển chưa có quyển này -> thử bản theo quyển
        join = status == "ok" and G.available
        for i, r in enumerate(L):
            img, book, page, col, ocr, syl, lab, uni, tier, rule, bbox = r
            if k and book != k:
                continue
            ni, si = T[i] if T else ("", "")
            uid = make_uid(bs, dict(book=book, page=page, column=col), ni, si, i)
            ge = None
            if tier == "GOLD":
                bd.n_gold += 1
                if join:
                    ge = G.by_uid.get(uid) or G.by_img.get((bs, img))
                    if ge:
                        bd.n_joined += 1
                        bd.gx[ge[0]] += 1
            bd.rows.append(Row(uid, img, page.strip(), col, ocr, syl, lab, uni, tier, rule, bbox, ge))
        bd.status, bd.source = status, rel_to(lp, root)
        bd.crop_bases = [d, ds / "_ALL" / "crops" / bs]
        bd.eval_marked = (d / "evaluation_only.json").is_file()
        if status == "ban_cu":
            bd.note = ("Đang hiển thị bản xuất cũ theo quyển (23/09); gold_exact không ghép với bản này — "
                       f"bản dựng mới nằm ở dataset/{bs}/ (chưa có).")
        elif bd.n_gold and not bd.n_joined:
            bd.note = "Chưa có gold_exact cho bộ này (bước B8 chạy sau khi gộp)." if not G.available else \
                "gold_exact.csv không khớp khoá cell_uid với labels.csv (khác lượt dựng) — chạy lại ./run_pipeline.sh --merge."
        if not key:
            cache.pop(ck, None)           # bộ một sách: giải phóng ngay (STT giữ cho 3 quyển)
        break
    else:
        # dự phòng: bộ gộp dataset/_ALL/labels.csv (ảnh 'crops/<bs>/…' tương đối dataset/_ALL)
        la = ds / "_ALL" / "labels.csv"
        if la.is_file():
            ck = str(la) + "#cols"
            if ck not in cache:
                cache[ck] = read_csv_cols(la, ALL_COLS)[0]
            for (img, book, page, col, ocr, syl, lab, uni, tier, rule, bbox, uid, rbs, ev) in cache[ck]:
                if rbs != bs or (key and book != key):
                    continue
                ge = G.by_uid.get(uid) if tier == "GOLD" else None
                if tier == "GOLD":
                    bd.n_gold += 1
                    if ge:
                        bd.n_joined += 1
                        bd.gx[ge[0]] += 1
                bd.rows.append(Row(uid, img, page.strip(), col, ocr, syl, lab, uni, tier, rule, bbox, ge))
                bd.eval_marked = bd.eval_marked or ev == "1"
            if bd.rows:
                bd.status, bd.source = "tu_bo_gop", rel_to(la, root)
                bd.crop_bases = [ds / "_ALL"]
    _finish(bd, root)
    return bd


HUMAN_COLS = ["cell_uid", "book", "page", "column", "char", "codepoint", "syllable", "kind", "bbox", "image",
              "crop_chuan", "keep_level", "align_conf", "one_char_ok", "split_hint", "folio", "paddle_test"]
HUMAN_EXTRA = ("crop_chuan", "keep_level", "align_conf", "one_char_ok", "kind", "split_hint", "folio", "paddle_test")


def load_human_book(root: Path, cfg: dict, cache: dict) -> BookData:
    """dataset/_BORG_NHAN_NGUOI/labels.csv — đọc CHỈ các cột cần (≈ 143 nghìn dòng × 31 cột: tránh dict từng dòng)."""
    bd = BookData(cfg)
    d = root / "dataset" / "_BORG_NHAN_NGUOI"
    lp = d / "labels.csv"
    if lp.is_file() and lp.stat().st_size > 0:
        ck = str(lp) + "#cols"
        if ck not in cache:
            cache[ck] = read_csv_cols(lp, HUMAN_COLS)[0]
        oc = Counter()
        for (uid, book, page, col, ch, cp, syl, kind, bbox, img, cc, lv, conf, one, split, folio, pt) in cache[ck]:
            if book != cfg["book_set"]:
                continue
            oc["one_char_ok"] += one == "1"
            oc["with_image"] += bool(img)
            bd.rows.append(Row(uid, img, page.strip(), col, "", syl, ch, cp, lv or "khong", kind, bbox, None,
                               (cc, lv, conf, one, kind, split, folio, pt)))
        bd.extra_counts = dict(oc)
        if bd.rows:
            bd.status, bd.source = "ok", rel_to(lp, root)
            bd.crop_bases = [d]
            bd.eval_marked = True
    _finish(bd, root)
    return bd


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.books: dict[str, BookData] = {}
        self.ge = GoldExact()
        self.dict_entries = None
        self.dims: dict[str, tuple] = {}
        self.dims_lock = threading.Lock()


def code_file(root: Path, rel: str) -> Path:
    """Tệp MÃ/CẤU HÌNH (config/, Dict/): lấy dưới gốc dữ liệu nếu có, không thì dưới repo."""
    p = root / rel
    return p if p.exists() else REPO_ROOT / rel


def load_store(root: Path, log=print) -> Store:
    t0 = time.time()
    st = Store(root)
    log(f"[server] Nạp dữ liệu từ {root} ...")
    st.ge = load_gold_exact(root, log)
    cache: dict = {}
    for cfg in BOOKS:
        try:
            bd = load_human_book(root, cfg, cache) if cfg["kind"] == "human" else load_auto_book(root, cfg, st.ge, cache, log)
        except Exception as e:  # noqa: BLE001 — một bộ hỏng không được làm sập máy chủ
            bd = BookData(cfg)
            bd.status, bd.note = "loi", f"Lỗi đọc dữ liệu: {type(e).__name__}: {e}"
            _finish(bd, root)
        st.books[cfg["id"]] = bd
        mark = "OK" if bd.rows else "!!"
        extra = f" · gold_exact ghép {bd.n_joined:,}/{bd.n_gold:,} ô GOLD" if bd.n_gold else ""
        log(f"  [{mark}] {cfg['id']}: {len(bd.rows):,} dòng · {len(bd.pages)} trang nhãn · "
            f"{len(bd.scan_pages)} ảnh trang · {STATUS_VI[bd.status]}{extra}")
    cache.clear()
    try:
        dpath = code_file(root, "Dict/QuocNgu_SinoNom.csv")
        if dpath.is_file():
            with open(dpath, encoding="utf-8", newline="") as f:
                st.dict_entries = max(0, sum(1 for _ in f) - 1)
    except OSError:
        pass
    log(f"[server] Nạp xong trong {time.time() - t0:.1f} s")
    return st


STORE: Store | None = None
RELOAD_LOCK = threading.Lock()


# =====================================================================================================================
# Đường dẫn URL <-> tệp
# =====================================================================================================================
def crop_url(bid: str, image: str) -> str | None:
    if not image:
        return None
    return f"/crops/{bid}/{quote(image)}"


def crop_chuan_url(rel: str) -> str | None:
    rel = norm_crop_chuan(rel)
    if not rel.startswith("crops_chuan/"):
        return None
    return "/crops_chuan/" + quote(rel[len("crops_chuan/"):])


def human_url(rel: str) -> str | None:
    return f"/borg_human/{quote(rel)}" if rel else None


def file_for_url(st: Store, url: str) -> Path | None:
    """URL tệp của máy chủ -> tệp trên đĩa (cùng luật với bộ phục vụ; None nếu không có / ngoài vùng cho phép)."""
    path = urlparse(url).path
    parts = path.strip("/").split("/", 2)
    if len(parts) < 2:
        return None
    kind = parts[0]
    if kind == "crops" and len(parts) == 3:
        bd = st.books.get(resolve_book_id(unquote(parts[1])))
        if not bd:
            return None
        for base in bd.crop_bases:
            p = safe_join(base, parts[2])
            if p:
                return p
        return None
    if kind == "crops_chuan":
        return safe_join(st.root / "dataset" / "_ALL" / "crops_chuan", path.split("/crops_chuan/", 1)[1])
    if kind == "borg_human":
        return safe_join(st.root / "dataset" / "_BORG_NHAN_NGUOI", path.split("/borg_human/", 1)[1])
    if kind == "page_scans" and len(parts) == 3:
        cfg = BOOK_BY_ID.get(resolve_book_id(unquote(parts[1])))
        if not cfg:
            return None
        for rel in cfg["pages_dirs"]:
            p = safe_join(st.root / "prepared" / rel, parts[2])
            if p:
                return p
    return None


def scan_file(st: Store, bd: BookData, page: str) -> Path | None:
    for rel in bd.cfg["pages_dirs"]:
        for ext in (".png", ".jpg", ".jpeg"):
            p = st.root / "prepared" / rel / f"{page}{ext}"
            if p.is_file():
                return p
    return None


def page_dims(st: Store, p: Path) -> tuple[int, int] | None:
    k = str(p)
    with st.dims_lock:
        if k in st.dims:
            return st.dims[k]
    d = image_size(p)
    with st.dims_lock:
        st.dims[k] = d
    return d


# =====================================================================================================================
# Số đo từ measure_out (không bao giờ bịa: thiếu -> "chưa có")
# =====================================================================================================================
class Measures:
    def __init__(self, st: Store):
        self.st = st
        self.root = st.root
        self.mo = st.root / "measure_out"
        self.missing: list[str] = []
        self._cache: dict = {}

    def load(self, p: Path, what: str):
        k = str(p)
        if k not in self._cache:
            d = read_json(p)
            if d is None:
                self.missing.append(f"{what}: {rel_to(p, self.root)}")
            self._cache[k] = d
        return self._cache[k]

    def src(self, p: Path) -> str:
        return rel_to(p, self.root)

    def ihr(self, name):
        p = self.mo / name / "ihr_endtoend" / "summary.json"
        return self.load(p, f"ihr_endtoend {name}"), self.src(p)

    def borg(self, name):
        p = self.mo / name / "borg_endtoend" / "summary.json"
        return self.load(p, f"borg_endtoend {name}"), self.src(p)

    def gold_exact(self):
        p = self.mo / "gold_exact" / "summary.json"
        return self.load(p, "gold_exact_eval"), self.src(p)

    def summary(self):
        p = self.mo / "SUMMARY.json"
        return self.load(p, "bộ đo measure.py --all"), self.src(p)

    def box_ref(self):
        p = self.mo / "box_ref" / "summary.json"
        return self.load(p, "box_ref (hộp legacy vs pitch)"), self.src(p)

    def borg_human(self):
        p = self.mo / "borg_human" / "summary.json"
        return self.load(p, "borg_human_eval"), self.src(p)

    def cross(self, book_id):
        cfg = BOOK_BY_ID[book_id]
        ds = self.root / (cfg.get("ds_out") or f"prepared/{book_id}/dataset_out")
        for sub in ("auto_precision_verify", "auto_precision_gated", "auto_precision"):
            p = ds / sub / "SUMMARY.json"
            d = read_json(p)
            if d is not None:
                return dig(d, "cross", "books", book_id), self.src(p)
        self.missing.append(f"dị bản {book_id}: {rel_to(ds / 'auto_precision_verify' / 'SUMMARY.json', self.root)}")
        return None, None

    def sources_all(self):
        p = self.root / "dataset" / "_ALL" / "SOURCES.json"
        return self.load(p, "bộ gộp dataset/_ALL"), self.src(p)

    def borg_build_info(self):
        p = self.root / "dataset" / "_BORG_NHAN_NGUOI" / "BUILD_INFO.json"
        return self.load(p, "bộ nhãn người Borg"), self.src(p)


def acc_row(book_id, level, metric, value=None, value_text=None, ci="", n=None, source="", note=""):
    cfg = BOOK_BY_ID.get(book_id, {})
    vt = value_text if value_text is not None else fmt_pct(value)
    return dict(book=book_id, book_title=cfg.get("title", book_id), level=level, level_vi=LEVEL_VI.get(level, level),
                metric=metric, value=value, value_text=vt, ci_text=ci or "", n=n,
                n_text=fmt_int(n) if n is not None else "", source=source or "", note=note,
                available=vt not in (NA, "", None))


def accuracy_rows(M: Measures) -> list[dict]:
    """Bản chép logic docs/BAO_CAO_TONG_HOP (scripts/bao_cao_tong_hop.py §2)."""
    out = []
    ge, ge_src = M.gold_exact()
    for s8, name in (("L16", "LucVanTien1916"), ("TK", "TruyenKieu1872")):
        s, src = M.ihr(name)
        g = dig(s, "precision", "gold_anh")
        if g:
            out.append(acc_row(name, "do", "GOLD ảnh: nhãn = chữ người", g.get("precision"), ci=fmt_ci(g.get("ci95")),
                               n=g.get("with_gt"), source=src))
            out.append(acc_row(name, "do", "GOLD ảnh (bỏ PUA + dị thể)", g.get("precision_bo_pua_dithe"),
                               n=g.get("with_gt"), source=src))
        else:
            out.append(acc_row(name, "do", "GOLD ảnh: nhãn = chữ người", source=src))
        r = dig(ge, "ihr", s8)
        if r and "both_pt" in r:
            ci = [r.get("both_lo"), r.get("both_hi")] if r.get("both_lo") is not None else r.get("ci95")
            out.append(acc_row(name, "do", "GOLD chính xác (ok): đúng cả chữ lẫn ảnh", r.get("both_pt"), ci=fmt_ci(ci),
                               n=r.get("n_eval"), source=ge_src, note="chỉ báo, CI bootstrap cụm trang"))
        else:
            out.append(acc_row(name, "do", "GOLD chính xác (ok): đúng cả chữ lẫn ảnh", source=ge_src))
    for name in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
        s, src = M.borg(name)
        g = dig(s, "tiers", "GOLD")
        if g and g.get("v1p"):
            out.append(acc_row(name, "do", "GOLD: nhãn = chữ người (V1+ tính biến thể)", g["v1p"].get("pt"),
                               ci=fmt_ci(g["v1p"].get("ci_page")), n=g.get("n_eval"), source=src))
            if g.get("strict"):
                out.append(acc_row(name, "do", "GOLD: trùng hẳn chữ người (strict)", g["strict"].get("pt"),
                                   ci=fmt_ci(g["strict"].get("ci_page")), n=g.get("n_eval"), source=src))
        else:
            out.append(acc_row(name, "do", "GOLD: nhãn = chữ người (V1+ tính biến thể)", source=src,
                               note="chưa có nhãn máy" if s is not None else ""))
        ok = dig(s, "gold_exact", "ok")
        if ok and ok.get("v1p"):
            out.append(acc_row(name, "do", f"GOLD chính xác (ok, n ok {fmt_int(ok.get('n'))}): nhãn = chữ người",
                               ok["v1p"].get("pt"), ci=fmt_ci(ok["v1p"].get("wilson")) + " Wilson",
                               n=ok.get("n_eval"), source=src))
    for name in ("LucVanTien1883", "KimVanKieu1884"):
        c, src = M.cross(name)
        refs = (c or {}).get("refs") or {}
        got = False
        for ref, blk in refs.items():
            at = (blk or {}).get("all_tiers") or {}
            if "GOLD_eq_pct" not in at:
                continue
            got = True
            out.append(acc_row(name, "uoc_luong", f"GOLD = chữ dị bản {ref} (cận dưới)", at["GOLD_eq_pct"],
                               value_text=fmt_pct(at["GOLD_eq_pct"], 1, 1), ci=fmt_ci(at.get("GOLD_wilson95"), 1, 1),
                               n=at.get("n_GOLD"), source=src))
            if at.get("GOLD_eq_or_di_the_pct") is not None:
                out.append(acc_row(name, "uoc_luong", f"GOLD = chữ {ref} hoặc dị thể cùng âm",
                                   at["GOLD_eq_or_di_the_pct"], value_text=fmt_pct(at["GOLD_eq_or_di_the_pct"], 1, 1),
                                   n=at.get("n_GOLD"), source=src))
        if not got:
            out.append(acc_row(name, "uoc_luong", "GOLD = chữ dị bản người (cận dưới)", source=src or ""))
    est, est_src = None, ""
    for name in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
        s, src = M.borg(name)
        if s and s.get("stt_estimate_suy_doan"):
            est, est_src = s["stt_estimate_suy_doan"], src
            break
    for s8, bid in (("stt2", "SachThanhTruyen2"), ("stt4", "SachThanhTruyen4"), ("stt11", "SachThanhTruyen11")):
        v = (est or {}).get(s8)
        if v:
            val, _, ci = str(v).partition(" [")
            out.append(acc_row(bid, "suy_doan", "tập \"lai\" (ước lượng TN3, không có sự thật người)",
                               value_text=f"{val} %", ci=f"[{ci}" if ci else "", source=est_src))
        else:
            out.append(acc_row(bid, "suy_doan", "tập \"lai\" (ước lượng TN3, không có sự thật người)", source=est_src))
    chr_row = acc_row("Chrestomathie1872", "suy_doan", "không có nhãn người lẫn dị bản số hoá", value_text="không đo được")
    chr_row["available"] = False
    out.append(chr_row)
    return out


def kim_rows(M: Measures) -> list[dict]:
    out = []
    for name in ("LucVanTien1916", "TruyenKieu1872"):
        s, src = M.ihr(name)
        k = dig(s, "kim_raw")
        if k:
            out.append(acc_row(name, "do", "OCR kim ở ô (mọi ô có nhãn người)", k.get("precision"),
                               ci=fmt_ci(k.get("ci95")), n=k.get("n"), source=src, note="mộc bản"))
            out.append(acc_row(name, "do", "OCR kim ở ô GOLD", k.get("precision_tren_o_GOLD"), n=k.get("n_gold"),
                               source=src, note="mộc bản"))
    for name in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
        s, src = M.borg(name)
        k = (s or {}).get("kim_at_cells") or {}
        if k.get("strict"):
            out.append(acc_row(name, "do", "OCR kim ở ô (strict)", k["strict"].get("pt"),
                               ci=fmt_ci(k["strict"].get("ci_page")), n=k.get("n_eval"), source=src, note="viết tay"))
        if k.get("v1p"):
            out.append(acc_row(name, "do", "OCR kim ở ô (V1+)", k["v1p"].get("pt"), ci=fmt_ci(k["v1p"].get("ci_page")),
                               n=k.get("n_eval"), source=src, note="viết tay"))
    for name in ("LucVanTien1883", "KimVanKieu1884"):
        c, src = M.cross(name)
        for ref, blk in ((c or {}).get("refs") or {}).items():
            at = (blk or {}).get("all_tiers") or {}
            if "ocr_eq_ref_all_pct" in at:
                out.append(acc_row(name, "uoc_luong", f"OCR kim ở ô = chữ dị bản {ref} (mọi tầng)",
                                   at["ocr_eq_ref_all_pct"], value_text=fmt_pct(at["ocr_eq_ref_all_pct"], 1, 1),
                                   n=at.get("n_cells"), source=src, note="thạch bản"))
    return out


def gold_exact_table(st: Store, M: Measures) -> dict:
    G = st.ge
    ge_sum, ge_src = M.gold_exact()
    rows, tot = [], Counter()
    table_sum = {r.get("bo"): r for r in (ge_sum or {}).get("table") or []}
    for cfg in BOOKS:
        if cfg["kind"] != "auto":
            continue
        s8 = cfg["set8"]
        c = G.counts.get(s8)
        src = ", ".join(G.sources) if c else ""
        # bảng của gold_exact_eval chỉ dùng khi KHÔNG có gold_exact.csv nào (tránh trộn hai lượt dựng khác nhau)
        if c is None and not G.available and s8 in table_sum:
            t = table_sum[s8]
            c = Counter({k: int(t.get(k) or 0) for k in GX_STATES})
            src = ge_src
        ev = G.evidence.get(s8) or EVIDENCE_OF.get(s8, "suy_doan")
        if c is not None:
            gold = sum(c.values())
            tot.update(c)
        else:
            gold = None
        rows.append(dict(book=cfg["id"], book_title=cfg["title"], set8=s8, role=cfg["role"], role_vi=ROLE_VI[cfg["role"]],
                         evidence=ev, evidence_vi=EVIDENCE_VI.get(ev, ev), gold=gold,
                         **{k: (c.get(k, 0) if c is not None else None) for k in GX_STATES},
                         ok_pct=(c.get("ok", 0) / gold if c is not None and gold else None), source=src,
                         available=c is not None))
    gtot = sum(tot.values())
    return dict(available=any(r["available"] for r in rows), rows=rows,
                totals=dict(gold=gtot, **{k: tot.get(k, 0) for k in GX_STATES}, ok_pct=tot.get("ok", 0) / gtot if gtot else None),
                policy_version=G.policy_version or (ge_sum or {}).get("policy_version"),
                config_sha16=G.config_sha16 or (ge_sum or {}).get("config_sha16"),
                source=", ".join(G.sources) or (ge_src if ge_sum else ""),
                profile_handwriting=(ge_sum or {}).get("profile_handwriting") or profile_from_config(st.root),
                core_loss_gate=(ge_sum or {}).get("core_loss_gate"))


def profile_from_config(root: Path) -> dict | None:
    """Khối profiles.handwriting của config/gold_exact.yaml (đọc bằng regex — máy chủ không phụ thuộc PyYAML)."""
    p = code_file(root, "config/gold_exact.yaml")
    try:
        t = p.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^profiles:\s*\n\s+handwriting:\s*\n((?:\s{4,}.*\n?)+)", t, re.M)
    if not m:
        return None
    blk = m.group(1)

    def g(key):
        mm = re.search(rf"^\s+{key}:\s*([^#\n]+)", blk, re.M)
        return mm.group(1).strip() if mm else None

    lst = lambda s: [x.strip() for x in (s or "").strip("[] ").split(",") if x.strip()]
    return dict(sets=lst(g("sets")), drop=lst(g("drop_gates")), slot=g("slot_gate"), hand_q=g("hand_q"),
                n_hum_min=g("n_hum_min"), source=rel_to(p, root))


def comparisons(M: Measures) -> list[dict]:
    """Phương pháp đề xuất (pitch decoding) so với cơ sở (legacy @0,15) — measure_out/box_ref/summary.json."""
    s, src = M.box_ref()
    out = []
    specs = [("ok_iou50_pct", "Ô khớp hộp tham chiếu (IoU ≥ 0,5)", True),
             ("cut_glyph_pct", "Hộp cắt vào thân chữ", False),
             ("miss_pct", "Ô bị bỏ sót", False)]
    n_pages = dig(s, "pages", "per_book") or {}
    for bid in ("LucVanTien1883", "KimVanKieu1884"):
        cv = dig(s, "per_book", bid, "cells_verified") or {}
        base, prop = cv.get("legacy@0.15_prepared") or {}, cv.get("pitch_prepared") or {}
        for key, label, higher in specs:
            a, b = base.get(key), prop.get(key)
            if a is None and b is None:
                continue
            imp = ""
            if a is not None and b is not None:
                d = float(b) - float(a)
                good = d > 0 if higher else d < 0
                imp = (f"{d:+.1f} điểm %".replace(".", ",") + (" (tốt hơn)" if good else " (không tốt hơn)" if d else ""))
            out.append(dict(metric=f"{label} — {BOOK_BY_ID[bid]['title']}",
                            baseline=f"{fmt_pct(a, 1, 1)} (hộp legacy, ngưỡng 0,15)" if a is not None else NA,
                            proposed=f"{fmt_pct(b, 1, 1)} (CenterNet + Pitch Decoding)" if b is not None else NA,
                            improvement=imp or "—", level_vi=LEVEL_VI["do_tu_dong"],
                            impact=f"Đo trên {fmt_int(n_pages.get(bid))} trang có hộp tham chiếu tự động (box_ref).",
                            source=src))
    if not out:
        out.append(dict(metric="Hộp ký tự: pitch decoding so với hộp legacy", baseline=NA, proposed=NA, improvement="—",
                        level_vi=LEVEL_VI["do_tu_dong"], impact="Chưa có measure_out/box_ref/summary.json "
                        "(chạy scripts/measure/measure.py --all).", source=src))
    out.append(dict(metric="Can thiệp thủ công khi gán nhãn", baseline="Gán nhãn thủ công / bán tự động",
                    proposed="0 quyết định của người (mọi nhãn theo luật)", improvement="Tự động hoá hoàn toàn",
                    level_vi="Thuộc tính thiết kế", impact="Nhãn người (IHR, Borg) chỉ dùng để ĐO, không vào quyết định.",
                    source="pipeline/"))
    return out


def invariants_block(M: Measures) -> dict:
    s, src = M.summary()
    out = dict(available=s is not None, source=src, evals=[])
    if s:
        t = s.get("totals") or {}
        out.update(generated_at=s.get("generated_at"), cli=s.get("cli"), totals=t,
                   text=f"{fmt_int(t.get('inv_pass'))} PASS · {fmt_int(t.get('inv_fail'))} FAIL",
                   steps=[dict(step=r.get("step"), book=r.get("book"), rc=r.get("rc"), n_pass=r.get("n_pass"),
                               n_fail=r.get("n_fail"), n_soft_fail=r.get("n_soft_fail"), n_skip=r.get("n_skip"))
                          for r in (s.get("results") or [])])
    else:
        out.update(text=NA, totals={}, steps=[])
    evs = [("GOLD chính xác (gold_exact_eval)", M.gold_exact())]
    for name in ("LucVanTien1916", "TruyenKieu1872"):
        evs.append((f"Đầu–cuối IHR {name}", M.ihr(name)))
    for name in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
        evs.append((f"Đầu–cuối Borg {name}", M.borg(name)))
    evs.append(("Bộ crop nhãn người Borg (borg_human_eval)", M.borg_human()))
    for label, (d, src_) in evs:
        inv = (d or {}).get("invariants") or []
        out["evals"].append(dict(name=label, source=src_, available=d is not None, total=len(inv),
                                 n_pass=sum(1 for i in inv if isinstance(i, dict) and i.get("pass") is True),
                                 n_fail=sum(1 for i in inv if isinstance(i, dict) and i.get("pass") is False)))
    return out


def borg_human_block(st: Store, M: Measures) -> dict:
    s, src = M.borg_human()
    bi, bi_src = M.borg_build_info()
    per_book = {}
    for bid in BOOK_GROUPS["nhan_nguoi"]:
        bd = st.books.get(bid)
        if bd and bd.rows:
            per_book[bd.cfg["book_set"]] = dict(cells=len(bd.rows), levels=dict(bd.tiers), pages=len(bd.pages),
                                                **bd.extra_counts)
    out = dict(available=bool(per_book) or s is not None or bi is not None, per_book=per_book, source=src)
    th = (s or {}).get("theta_paddle_keep") or dig(bi, "theta", "paddle", "keep")
    if th:
        out["theta"] = dict(value=th.get("theta"), text=fmt_pct(th.get("theta")),
                            ci_text=fmt_ci(th.get("ci95") or th.get("theta_ci")), n=th.get("n"),
                            level_vi="ƯỚC LƯỢNG bằng bộ đọc độc lập (Paddle)",
                            meaning="Tỉ lệ ô keep bị trượt ±1 chữ so với hộp người",
                            source=src if s and s.get("theta_paddle_keep") else bi_src)
    if s:
        out.update(n_cells=s.get("n_cells"), cumulative=s.get("cumulative"), with_image=s.get("with_image"),
                   one_char_ok=s.get("one_char_ok"), size_mb=s.get("size_mb"))
    elif bi:
        c = dig(bi, "counts", "cumulative", "tong") or {}
        out.update(n_cells=c.get("cells"), cumulative=dict(keep_v5=c.get("keep_v5"), keep=c.get("keep"),
                                                            keep_high=c.get("keep_high")),
                   with_image=c.get("with_image"), one_char_ok=c.get("one_char_ok"), source=bi_src)
    return out


# =====================================================================================================================
# Tải trọng API (hàm thuần — build_sample_data.py dùng lại)
# =====================================================================================================================
def book_summary(st: Store, bd: BookData) -> dict:
    cfg = bd.cfg
    total = len(bd.rows)
    t = bd.tiers
    gx = st.ge.counts.get(cfg["set8"]) if cfg["kind"] == "auto" else None
    ev = st.ge.evidence.get(cfg["set8"]) or EVIDENCE_OF.get(cfg["set8"], "suy_doan")
    gold = t.get("GOLD", 0)
    d = dict(id=cfg["id"], title=cfg["title"], subtitle=cfg["subtitle"], author=cfg["author"], layout=cfg["layout_desc"],
             kind=cfg["kind"], role=cfg["role"], role_vi=ROLE_VI[cfg["role"]], script=cfg["script"], genre=cfg["genre"],
             set8=cfg["set8"], book_set=cfg["book_set"], status=bd.status, status_vi=STATUS_VI[bd.status], note=bd.note,
             source=bd.source, total=total, tiers=dict(t), gold=gold, syllable=t.get("SYLLABLE", 0),
             gold_text_only=t.get("GOLD_text_only", 0), gold_pct=round(gold / total * 100, 1) if total else 0,
             pages=len(bd.pages), n_scans=len(bd.scan_pages),
             evaluation_only=bd.eval_marked or cfg["role"] != "giao_nop",
             evidence_level=ev, evidence_vi=EVIDENCE_VI.get(ev, ev))
    if cfg["kind"] == "auto":
        # số trạng thái theo bộ = đếm trên gold_exact.csv (nguồn của bước B8); bộ đang hiển thị bản cũ -> không gắn
        use = gx if (gx and bd.status in ("ok", "tu_bo_gop")) else None
        d["gold_exact"] = {k: (use.get(k, 0) if use else None) for k in GX_STATES}
        d["gold_exact_available"] = use is not None
        d["gold_exact_joined"] = bd.n_joined
        d["gold_exact_ok_pct"] = round(use.get("ok", 0) / sum(use.values()) * 100, 1) if use and sum(use.values()) else None
    else:
        d["keep_levels"] = dict(t)
        d.update(bd.extra_counts)
    return d


def api_stats(st: Store) -> dict:
    M = Measures(st)
    books, tot = {}, Counter()
    by_role = {"giao_nop": Counter(), "danh_gia": Counter()}
    n_with = 0
    for bid in AUTO_IDS:
        bd = st.books.get(bid)
        if not bd:
            continue
        b = book_summary(st, bd)
        books[bid] = b
        if b["total"]:
            n_with += 1
        c = Counter(total=b["total"], gold=b["gold"], syllable=b["syllable"], gold_text_only=b["gold_text_only"])
        tot.update(c)
        by_role["giao_nop" if b["role"] == "giao_nop" else "danh_gia"].update(c)
    human = {bid: book_summary(st, st.books[bid]) for bid in BOOK_GROUPS["nhan_nguoi"] if bid in st.books}
    gxt = gold_exact_table(st, M)
    inv = invariants_block(M)
    src_all, src_all_path = M.sources_all()
    total = tot.get("total", 0)
    gt = gxt["totals"]
    im = dict(
        total_characters=total, total_gold=tot.get("gold", 0), total_syllable=tot.get("syllable", 0),
        total_gold_text_only=tot.get("gold_text_only", 0),
        gold_rate=round(tot.get("gold", 0) / total * 100, 1) if total else None,
        gold_rate_overall=round((tot.get("gold", 0) + tot.get("gold_text_only", 0)) / total * 100, 1) if total else None,
        books_count=len(AUTO_IDS), books_with_data=n_with,
        deliver_characters=by_role["giao_nop"].get("total", 0), eval_characters=by_role["danh_gia"].get("total", 0),
        gold_exact_available=gxt["available"], gold_exact_ok=gt.get("ok") if gxt["available"] else None,
        gold_exact_gold=gt.get("gold") if gxt["available"] else None,
        gold_exact_ok_pct=round(gt["ok_pct"] * 100, 1) if gxt["available"] and gt.get("ok_pct") is not None else None,
        gold_exact_counts={k: gt.get(k) for k in GX_STATES} if gxt["available"] else None,
        invariants_available=inv["available"], invariants_text=inv.get("text", NA),
        invariants_pass=(inv.get("totals") or {}).get("inv_pass"), invariants_fail=(inv.get("totals") or {}).get("inv_fail"),
        invariants_generated_at=inv.get("generated_at"),
        dict_entries=st.dict_entries, human_intervention=0, automation_rate="Tự động theo quy tắc",
    )
    all_info = None
    if src_all:
        bb = src_all.get("bat_bien") or {}
        all_info = dict(n_dong=src_all.get("n_dong"), n_tep_crop=src_all.get("n_tep_crop"),
                        eval_only_dong=src_all.get("eval_only_dong"), n_bo=src_all.get("n_bo"), bo=src_all.get("bo"),
                        bat_bien_pass=sum(1 for v in bb.values() if v is True), bat_bien_total=len(bb),
                        ngay=src_all.get("ngay"), source=src_all_path)
    return dict(
        title="Hệ thống hỗ trợ gán nhãn tự động văn bản Hán Nôm cổ (GanNhanOCR)",
        subtitle="Báo cáo thực nghiệm & trực quan hoá dữ liệu — luận văn Thạc sĩ",
        data_root=str(st.root), loaded_at=st.loaded_at, impact_metrics=im, books=books, human_books=human,
        gold_exact=dict(available=gxt["available"], totals=gt, policy_version=gxt["policy_version"],
                        config_sha16=gxt["config_sha16"], source=gxt["source"],
                        states={k: dict(label=GX_VI[k], desc=GX_DESC[k]) for k in GX_STATES}),
        all_dataset=all_info, evidence_levels=EVIDENCE_VI,
        missing_sources=list(dict.fromkeys(M.missing)),
    )


def api_books(st: Store) -> list[dict]:
    out = []
    for cfg in BOOKS:
        bd = st.books.get(cfg["id"])
        if not bd:
            continue
        pages = sorted(bd.pages) if bd.pages else list(bd.scan_pages)
        default = bd.best_page if bd.best_page in pages else cfg.get("default_page")   # trang nhiều ô ok nhất
        if default not in pages:
            default = pages[0] if pages else None
        b = book_summary(st, bd)
        b.update(n_columns=cfg["n_columns"], layout_desc=cfg["layout_desc"], available_pages=pages,
                 sample_pages=pages[:5], default_page=default, total_chars=len(bd.rows),
                 has_gold_exact=bool(bd.n_joined))
        out.append(b)
    return out


def char_payload(st: Store, bd: BookData, r: Row) -> dict:
    cfg = bd.cfg
    human = cfg["kind"] == "human"
    d = dict(cell_uid=r.uid, image_rel=r.image, column=r.column, ocr_char=r.ocr_char, syllable=r.syllable,
             label=r.label, unicode=r.unicode, tier=r.tier, rule=r.rule, page=r.page)
    if human:
        x = dict(zip(HUMAN_EXTRA, r.extra)) if r.extra else {}
        d.update(crop_url=human_url(r.image), crop_chuan_url=human_url(x.get("crop_chuan", "")), gold_exact=None,
                 keep_level=x.get("keep_level", ""), keep_vi=KEEP_VI.get(x.get("keep_level", ""), x.get("keep_level", "")),
                 align_conf=x.get("align_conf", ""), one_char_ok=x.get("one_char_ok", ""), kind=x.get("kind", ""),
                 split_hint=x.get("split_hint", ""), folio=x.get("folio", ""), evidence_level="do_tren_nhan_nguoi",
                 evidence_vi="Nhãn do người phiên (Excel) — dùng làm sự thật để đo")
        return d
    d["crop_url"] = crop_url(cfg["id"], r.image) if r.tier != "GOLD_text_only" else None   # text_only: không xuất crop
    ge = r.ge
    if ge:
        stt, why, ev, cc, cl, clf, oc, cst = ge
        d.update(gold_exact=stt, gold_exact_vi=GX_VI.get(stt, stt), ge_reason=why, ge_reason_vi=reason_vi(why),
                 evidence_level=ev, evidence_vi=EVIDENCE_VI.get(ev, ev), crop_chuan_url=crop_chuan_url(cc) if cc else None,
                 core_loss=cl, core_loss_flag=clf, one_char_ok=oc, crop_status=cst)
    elif r.tier == "GOLD" and bd.status in ("ok", "tu_bo_gop"):
        d.update(gold_exact="chua_co", gold_exact_vi=GX_VI["chua_co"], ge_reason="",
                 ge_reason_vi="Chưa có gold_exact.csv khớp ô này (bước B8 chưa chạy hoặc khác lượt dựng).",
                 crop_chuan_url=None)
    else:
        d.update(gold_exact=None, crop_chuan_url=None)
    return d


def api_page(st: Store, query: dict) -> tuple[dict, int]:
    bid = resolve_book_id((query.get("book") or ["LucVanTien1883"])[0])
    bd = st.books.get(bid)
    if not bd:
        return {"error": f"Không tìm thấy sách: {bid}"}, 404
    cfg = bd.cfg
    pages = sorted(bd.pages) if bd.pages else list(bd.scan_pages)
    page = (query.get("page") or [None])[0] or (cfg.get("default_page") if cfg.get("default_page") in bd.pages else None) \
        or bd.best_page or (pages[0] if pages else "page_0001")
    idx = bd.pages.get(page, [])
    sf = scan_file(st, bd, page)
    dims = page_dims(st, sf) if sf else None
    chars = []
    rows = [bd.rows[i] for i in idx]
    boxes = [(r, parse_bbox(r.bbox)) for r in rows]
    boxes = [(r, b) for r, b in boxes if b]
    if not dims:
        mx = max((b[2] for _, b in boxes), default=0)
        my = max((b[3] for _, b in boxes), default=0)
        dims = (int(mx * 1.04) or 1896, int(my * 1.03) or 3212)
    W, H = dims

    def colkey(r):
        try:
            return int(float(r.column))
        except (TypeError, ValueError):
            return 9999

    boxes.sort(key=lambda rb: (colkey(rb[0]), rb[1][1]))
    for n, (r, (x0, y0, x1, y1)) in enumerate(boxes, 1):
        c = char_payload(st, bd, r)
        c.update(index=n, bbox=[x0, y0, x1, y1],
                 rect=dict(left_pct=round(x0 / W * 100, 3), top_pct=round(y0 / H * 100, 3),
                           width_pct=round((x1 - x0) / W * 100, 3), height_pct=round((y1 - y0) / H * 100, 3)))
        try:
            c["column"] = int(float(r.column))
        except (TypeError, ValueError):
            pass
        chars.append(c)
    tc = Counter(c["tier"] for c in chars)
    gc = Counter(c.get("gold_exact") or "khong_ap_dung" for c in chars)
    msg = ""
    if not bd.rows:
        msg = STATUS_VI[bd.status] + (f" — {bd.note}" if bd.note else "")
    elif not idx:
        msg = "Trang này không có ô chữ nào trong bảng nhãn."
    return dict(book=bid, book_title=cfg["title"], kind=cfg["kind"], page=page, status=bd.status,
                status_vi=STATUS_VI[bd.status], note=bd.note, message=msg,
                scan_url=f"/page_scans/{bid}/{quote(sf.name)}" if sf else None, has_scan=bool(sf),
                dimensions=dict(width=W, height=H), char_count=len(chars), n_rows_page=len(idx),
                tier_counts=dict(tc), gold_exact_counts=dict(gc), characters=chars,
                has_gold_exact=bool(bd.n_joined)), 200


def api_search(st: Store, query: dict) -> dict:
    q = (query.get("q") or [""])[0].strip().lower()
    qn = strip_accents(q)
    bsel = resolve_book_id((query.get("book") or ["all"])[0])
    tier = (query.get("tier") or ["all"])[0]
    gx = (query.get("gx") or ["all"])[0]
    try:
        limit = max(1, min(int((query.get("limit") or [60])[0]), 300))
    except ValueError:
        limit = 60
    ids = BOOK_GROUPS.get(bsel) or ([bsel] if bsel in st.books else AUTO_IDS)
    tier_all = tier.lower() == "all"
    per_book: list[tuple[BookData, list[Row]]] = []
    n_total = 0
    for bid in ids:
        bd = st.books.get(bid)
        if not bd or not bd.rows:
            continue
        hits = []
        for r in bd.rows:
            if not tier_all and r.tier != tier and r.tier.upper() != tier.upper():
                continue
            if gx != "all":
                g = r.ge[0] if r.ge else ("chua_co" if r.tier == "GOLD" and bd.status in ("ok", "tu_bo_gop")
                                         and bd.cfg["kind"] == "auto" else "")
                if g != gx:
                    continue
            if q and not ((q in r.syllable.lower()) or (qn and qn in r.syl_norm) or (q in r.label.lower())
                          or (q in r.ocr_char.lower()) or (q in r.unicode.lower())):
                continue
            hits.append(r)
        n_total += len(hits)
        if hits:
            per_book.append((bd, hits))
    # chọn đều giữa các sách và rải đều trong mỗi sách (không chỉ lấy đầu bảng)
    picked: list[tuple[BookData, Row]] = []
    if per_book:
        quota = max(1, math.ceil(limit / len(per_book)))
        seen = set()
        for bd, hits in per_book:
            step = max(1, len(hits) // quota)
            for r in hits[::step][:quota]:
                picked.append((bd, r))
                seen.add(id(r))
        for bd, hits in per_book:              # còn chỗ -> lấp bằng kết quả chưa chọn
            if len(picked) >= limit:
                break
            for r in hits:
                if len(picked) >= limit:
                    break
                if id(r) not in seen:
                    picked.append((bd, r))
                    seen.add(id(r))
        picked = picked[:limit]
    results = []
    for bd, r in picked:
        c = char_payload(st, bd, r)
        c.update(book=bd.cfg["id"], book_title=bd.cfg["title"], image=r.image)
        results.append(c)
    return dict(query=q, book=bsel, tier=tier, gx=gx, count=len(results), total_matches=n_total, results=results)


def api_pipeline_flow(st: Store) -> list[dict]:
    M = Measures(st)
    box, box_src = M.box_ref()
    gxt = gold_exact_table(st, M)
    ge_sum, ge_src = M.gold_exact()
    src_all, src_all_path = M.sources_all()
    de = f"{fmt_int(st.dict_entries)} mục từ" if st.dict_entries else "từ điển Quốc ngữ ↔ Hán Nôm"

    def m(label, value, source="", level=""):
        return dict(label=label, value=value, source=source, level_vi=level)

    det_metrics = []
    for bid in ("LucVanTien1883", "KimVanKieu1884"):
        cv = dig(box, "per_book", bid, "cells_verified") or {}
        a, b = (cv.get("legacy@0.15_prepared") or {}), (cv.get("pitch_prepared") or {})
        if a.get("cut_glyph_pct") is not None or b.get("cut_glyph_pct") is not None:
            det_metrics.append(m(f"{BOOK_BY_ID[bid]['title']}: hộp cắt vào thân chữ (legacy → pitch)",
                                 f"{fmt_pct(a.get('cut_glyph_pct'), 1, 1)} → {fmt_pct(b.get('cut_glyph_pct'), 1, 1)}",
                                 box_src, LEVEL_VI["do_tu_dong"]))
        if a.get("ok_iou50_pct") is not None or b.get("ok_iou50_pct") is not None:
            det_metrics.append(m(f"{BOOK_BY_ID[bid]['title']}: ô khớp hộp tham chiếu IoU ≥ 0,5 (legacy → pitch)",
                                 f"{fmt_pct(a.get('ok_iou50_pct'), 1, 1)} → {fmt_pct(b.get('ok_iou50_pct'), 1, 1)}",
                                 box_src, LEVEL_VI["do_tu_dong"]))
    if not det_metrics:
        det_metrics.append(m("Hộp ký tự so với tham chiếu (box_ref)", NA, box_src))

    gate_metrics = []
    for bid in ("LucVanTien1883", "KimVanKieu1884"):
        c, src = M.cross(bid)
        for ref, blk in ((c or {}).get("refs") or {}).items():
            at = (blk or {}).get("all_tiers") or {}
            if "GOLD_eq_pct" in at:
                gate_metrics.append(m(f"{BOOK_BY_ID[bid]['title']}: GOLD = chữ dị bản {ref}",
                                      fmt_pct(at["GOLD_eq_pct"], 1, 1) + f" (n {fmt_int(at.get('n_GOLD'))})", src,
                                      LEVEL_VI["uoc_luong"]))
    if not gate_metrics:
        gate_metrics.append(m("Đối soát dị bản LVT1883/KVK1884", NA))

    stats = api_stats(st)
    exp_metrics = [m(b["title"], f"{fmt_int(b['total'])} dòng · GOLD {fmt_int(b['gold'])}" if b["total"] else
                     STATUS_VI[b["status"]], b.get("source") or "") for b in stats["books"].values()]
    merge_metrics = []
    if src_all:
        bb = src_all.get("bat_bien") or {}
        merge_metrics = [m("Dòng trong dataset/_ALL/labels.csv", fmt_int(src_all.get("n_dong")), src_all_path),
                         m("Dòng evaluation_only (tập đánh giá)", fmt_int(src_all.get("eval_only_dong")), src_all_path),
                         m("Bất biến bước gộp", f"{sum(1 for v in bb.values() if v is True)}/{len(bb)} PASS", src_all_path)]
    else:
        merge_metrics = [m("Bộ gộp dataset/_ALL", NA, src_all_path)]

    ge_metrics = []
    if gxt["available"]:
        t = gxt["totals"]
        ge_metrics.append(m("Ô GOLD được gắn trạng thái", fmt_int(t.get("gold")), gxt["source"]))
        ge_metrics.append(m("ok · text_only · uncertified · review",
                            " · ".join(fmt_int(t.get(k)) for k in GX_STATES), gxt["source"]))
    for s8, name in (("L16", "LucVanTien1916"), ("TK", "TruyenKieu1872")):
        r = dig(ge_sum, "ihr", s8)
        if r and "both_pt" in r:
            ge_metrics.append(m(f"{BOOK_BY_ID[name]['title']}: ô ok đúng cả chữ lẫn ảnh",
                                f"{fmt_pct(r['both_pt'])} {fmt_ci(r.get('ci95') or [r.get('both_lo'), r.get('both_hi')])}"
                                f" (n {fmt_int(r.get('n_eval'))})", ge_src, LEVEL_VI["do"]))
    for name in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
        s, src = M.borg(name)
        ok = dig(s, "gold_exact", "ok")
        if ok and ok.get("v1p"):
            ge_metrics.append(m(f"{BOOK_BY_ID[name]['title']}: ô ok có nhãn = chữ người",
                                f"{fmt_pct(ok['v1p'].get('pt'))} (n {fmt_int(ok.get('n_eval'))})", src, LEVEL_VI["do"]))
    if not ge_metrics:
        ge_metrics.append(m("GOLD chính xác", NA, ge_src))
    prof = gxt.get("profile_handwriting")
    prof_details = []
    if prof:
        prof_details = [
            "Áp cho các bộ CHỮ VIẾT TAY: " + ", ".join(prof.get("sets") or []) + " (6 bộ in/khắc giữ nguyên chính sách cũ).",
            "Bỏ cổng: " + (", ".join(prof.get("drop") or []) or "không") +
            " — CNT (số chữ OCR của cột ≠ số âm) không phân biệt đúng/sai trên chữ viết tay nên không còn hạ ô.",
            f"Cổng khe: {prof.get('slot')} · mức q của bộ kiểm viết tay: {prof.get('hand_q')} · "
            f"tối thiểu {prof.get('n_hum_min')} mẫu chữ người.",
            "Bộ kiểm viết tay chạy LOBO theo sách: sách Borg đang đánh giá luôn dùng mô hình học trên sách Borg kia.",
        ]
    policy = gxt.get("policy_version")
    return [
        dict(step=1, id="ingest", name="Tiền xử lý ảnh & OCR trang", tag="Chuẩn hoá ảnh · OCR có bộ đệm",
             input="Ảnh quét trang gốc và bản phiên âm Quốc ngữ (Borg: phiên âm người theo trang).",
             model="Kéo giãn tương phản / Otsu; OCR chữ Nôm (kim) và OCR Quốc ngữ, mọi lượt gọi lưu bộ đệm theo md5 ảnh.",
             process="Chuẩn hoá nền giấy, tách biên trang, dựng bộ nạp theo loại sách (STT, thạch bản, văn xuôi, IHR, Borg). "
                     "Chạy lại toàn bộ chỉ dùng bộ đệm: không gọi API.",
             output="prepared/<Bộ>/pages, detected/*_ocr_cache.json, transcriptions/",
             evidence="Chốt chặn chạy lại: mọi yêu cầu API bị chặn, bộ đệm gốc phải trùng sha256 trước/sau.",
             metrics=[]),
        dict(step=2, id="detect", name="Phát hiện vị trí ký tự", tag="CenterNet & Pitch Decoding",
             input="Ảnh cột chữ dọc tách từ trang.",
             model="CenterNet (ResNet-34) + giải mã nhịp ký tự (Pitch Decoding).",
             process="Dự đoán tâm chữ qua bản đồ nhiệt, rồi giải mã khoảng cách nhịp đều để tách chữ dính, hạn chế hộp rỗng "
                     "và hộp cắt vào nét.",
             output="Hộp bao [xmin, ymin, xmax, ymax] cho từng chữ trong cột.",
             evidence="So với hộp legacy trên trang có hộp tham chiếu tự động (box_ref).", metrics=det_metrics),
        dict(step=3, id="align", name="Gióng hàng song ngữ", tag="Banded Dynamic Programming",
             input="Chuỗi hộp chữ và chuỗi âm Quốc ngữ đối ứng.",
             model=f"Quy hoạch động dải hẹp (Banded DP) + từ điển Quốc ngữ ↔ Hán Nôm ({de}).",
             process="Tìm đường ghép tối ưu giữa hộp ảnh và âm tiết; ràng buộc nhịp thơ lục bát (6/8) với thơ; "
                     "chặn trôi lệch xuyên cột.",
             output="Nhãn sơ bộ cho từng hộp + mã luật ghép.",
             evidence="Gán nhãn hoàn toàn theo luật, không có quyết định của người.", metrics=[]),
        dict(step=4, id="remediate", name="Kiểm kê & hiệu chỉnh lỗi", tag="Rà soát nhầm lẫn hệ thống",
             input="Bảng nhãn sơ bộ.",
             model="Kiểm kê trùng ảnh/chữ + bảng sửa nhầm lẫn có hệ thống (confusion_fix).",
             process="Phát hiện chữ nhầm phổ biến (đồng âm, tự dạng gần), xếp tầng GOLD / SYLLABLE / REVIEW / QUARANTINE.",
             output="Bảng nhãn đã chuẩn hoá (labels_remediated / labels_final).",
             evidence="Lưu vết sha256 sau mỗi bước biến đổi (CHECKSUMS.txt).", metrics=[]),
        dict(step=5, id="gates", name="Cổng cơ chế & đối soát dị bản", tag="Cổng (a') · dị bản 1871/1916",
             input="Ô nghi vấn: lệch số chữ, lệch nhịp, hộp tràn biên.",
             model="Cổng cơ chế + so chéo với bản khắc độc lập do người số hoá.",
             process="Ô qua cổng giữ GOLD; ô không chắc ảnh nhưng chắc chữ -> GOLD_text_only; đo tỉ lệ khớp dị bản sau cổng.",
             output="labels_gated.csv", evidence="Tỉ lệ GOLD trùng chữ dị bản là CẬN DƯỚI (dị bản khác chữ hợp lệ bị tính sai).",
             metrics=gate_metrics),
        dict(step=6, id="export", name="Đóng gói bộ theo sách", tag="dataset/<Bộ>/ · 12 cột",
             input="Bảng nhãn đã qua cổng và ảnh crop.",
             model="export_final_dataset + tài liệu tự sinh (README, DATASHEET, xlsx).",
             process="Xuất crop theo tầng (gold/, syllable/), bảng 12 cột labels.csv + labels_trace.csv; bộ IHR/Borg được "
                     "đóng dấu evaluation_only (tập đánh giá, không trộn vào huấn luyện).",
             output="dataset/<Bộ>/labels.csv, gold/, syllable/, README.md, DATASHEET.md",
             evidence="Số dòng theo bộ đọc trực tiếp từ labels.csv.", metrics=exp_metrics),
        dict(step=7, id="merge", name="Gộp bộ dataset/_ALL", tag="cell_uid · evaluation_only",
             input="10 bộ dataset/<Bộ>/ (3 quyển STT trong một bộ).",
             model="merge_datasets: khoá chính cell_uid = (bộ, sách, trang, cột, nom_idx, syl_idx).",
             process="Gộp mọi bộ, gắn book_set / evaluation_only / split_hint, kiểm bất biến (khoá duy nhất, ảnh tồn tại, "
                     "cờ đánh giá đúng bộ).",
             output="dataset/_ALL/labels.csv, crops/, SOURCES.json, CHECKSUMS.txt",
             evidence="Bất biến bước gộp ghi trong SOURCES.json.", metrics=merge_metrics),
        dict(step=8, id="gold_exact", name="GOLD chính xác (B8)", tag="Ảnh + chữ · 4 trạng thái",
             input="Mọi ô GOLD của dataset/_ALL (không đổi labels.csv).",
             model="Luật đầu tiên khớp thắng: review → text_only → uncertified → ok; crop chuẩn v2 (khung vuông một chữ); "
                   "bộ kiểm ảnh↔chữ (ngưỡng TN1) và bộ kiểm chữ viết tay LOBO"
                   + (f"; chính sách {policy}" if policy else "") + ".",
             process="Gắn cho mỗi ô GOLD đúng một trạng thái ok / text_only / uncertified / review kèm lý do và mức chứng cứ "
                     "(ĐO ở LVT1916, TK1872, Borg; ƯỚC LƯỢNG ở LVT1883, KVK1884; SUY ĐOÁN ở STT, Chrestomathie). "
                     "Profile CHỮ VIẾT TAY: chính sách riêng cho STT + 2 bản Borg (xem chi tiết).",
             output="dataset/_ALL/gold_exact.csv, crops_chuan/ (ô ok), dataset/<Bộ>/gold_exact.csv",
             evidence="Độ chính xác chỉ ĐO được trên bộ có nhãn người; nơi khác là ước lượng hoặc suy đoán.",
             metrics=ge_metrics, details=prof_details, details_title="Profile chữ viết tay (handwriting)"),
    ]


def api_benchmarks(st: Store) -> dict:
    M = Measures(st)
    acc = accuracy_rows(M)
    kim = kim_rows(M)
    gxt = gold_exact_table(st, M)
    comp = comparisons(M)
    inv = invariants_block(M)
    bh = borg_human_block(st, M)
    return dict(
        title="Đánh giá thực nghiệm — số đo đọc từ measure_out/", data_root=str(st.root), loaded_at=st.loaded_at,
        accuracy=acc,
        accuracy_note=("CI: IHR = Wilson; Borg = bootstrap cụm trang trừ khi ghi Wilson; dị bản = Wilson (điểm %). "
                       "ĐO = so với nhãn người từng chữ · ƯỚC LƯỢNG = so dị bản do người số hoá (cận dưới) · "
                       "SUY ĐOÁN = không có sự thật người."),
        kim=kim, gold_exact_table=gxt, comparisons=comp, invariants=inv, borg_human=bh,
        missing_sources=list(dict.fromkeys(M.missing)),
    )


# =====================================================================================================================
# HTTP
# =====================================================================================================================
class GanNhanHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def log_message(self, fmt, *args):  # gọn log: bỏ ảnh
        if "/crops" in self.path or "/page_scans/" in self.path or "/borg_human/" in self.path:
            return
        super().log_message(fmt, *args)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path.startswith(("/api/", "/crops/", "/crops_chuan/", "/borg_human/", "/page_scans/")):
            self.do_GET()
        else:
            super().do_HEAD()

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        try:
            if path.startswith("/api/"):
                self.handle_api(path, parse_qs(url.query))
            elif path.startswith(("/crops/", "/crops_chuan/", "/borg_human/", "/page_scans/")):
                self.serve_data_file(path)
            else:
                super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001 — không để một yêu cầu làm sập máy chủ
            try:
                self.send_json({"error": f"{type(e).__name__}: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception:  # noqa: BLE001
                pass

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_text(self, status, msg: str):
        """Như send_error nhưng thông điệp UTF-8 nằm trong thân (send_error mã hoá latin-1 -> lỗi với tiếng Việt)."""
        body = msg.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def handle_api(self, path: str, query: dict):
        st = STORE
        if st is None:
            self.send_json({"error": "Máy chủ đang nạp dữ liệu"}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if path == "/api/stats":
            self.send_json(api_stats(st))
        elif path == "/api/books":
            self.send_json(api_books(st))
        elif path == "/api/page":
            data, code = api_page(st, query)
            self.send_json(data, status=code)
        elif path == "/api/search":
            self.send_json(api_search(st, query))
        elif path == "/api/pipeline_flow":
            self.send_json(api_pipeline_flow(st))
        elif path == "/api/benchmarks":
            self.send_json(api_benchmarks(st))
        elif path == "/api/reload":
            self.send_json(reload_store())
        else:
            self.send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    def serve_data_file(self, path: str):
        st = STORE
        f = file_for_url(st, path) if st else None
        if f and f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"):
            f = None                      # vùng dữ liệu chỉ phục vụ ẢNH (không lộ labels.csv, cache…)
        if not f:
            self.send_text(HTTPStatus.NOT_FOUND, "Không tìm thấy tệp (hoặc đường dẫn ngoài vùng cho phép).")
            return
        mime, _ = mimetypes.guess_type(str(f))
        try:
            size = f.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            if self.command != "HEAD":
                with open(f, "rb") as h:
                    while chunk := h.read(65536):
                        self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def reload_store() -> dict:
    global STORE
    if not RELOAD_LOCK.acquire(blocking=False):
        return {"ok": False, "message": "Đang nạp lại, thử lại sau."}
    try:
        root = STORE.root if STORE else default_root()
        new = load_store(root)
        STORE = new
        return {"ok": True, "loaded_at": new.loaded_at,
                "books": {k: len(v.rows) for k, v in new.books.items()}, "gold_exact": len(new.ge.by_uid)}
    finally:
        RELOAD_LOCK.release()


def run_server(port: int = 8080, host: str = "", root: Path | None = None):
    global STORE
    STORE = load_store(root or default_root())
    try:
        httpd = ThreadingHTTPServer((host, port), GanNhanHandler)
    except OSError as e:
        if port == 8080:
            print(f"[server] Cổng 8080 đang bận ({e}), thử cổng 8088...")
            httpd = ThreadingHTTPServer((host, 8088), GanNhanHandler)
            port = 8088
        else:
            raise
    httpd.daemon_threads = True
    print("\n" + "=" * 65)
    print("  GanNhanOCR — máy chủ trình diễn đang chạy")
    print(f"  Gốc dữ liệu: {STORE.root}")
    print(f"  Truy cập: http://localhost:{port}   (nạp lại dữ liệu: /api/reload)")
    print("=" * 65 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Đã dừng máy chủ.")
    finally:
        httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Máy chủ trình diễn GanNhanOCR (chỉ đọc dữ liệu).")
    ap.add_argument("port", nargs="?", type=int, default=8080)
    ap.add_argument("--root", default=None, help="gốc dữ liệu (mặc định: $GANNHANOCR_ROOT hoặc thư mục cha của web/)")
    ap.add_argument("--host", default="", help="địa chỉ lắng nghe (mặc định mọi giao diện; 127.0.0.1 = chỉ máy này)")
    a = ap.parse_args(argv)
    root = Path(a.root).expanduser().resolve() if a.root else default_root()
    run_server(a.port, a.host, root)


if __name__ == "__main__":
    main()
