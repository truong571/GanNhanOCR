"""r_goi_du_lieu_00_lib.py — thư viện dùng chung của hướng "KIỂM GÓI VÀ DỮ LIỆU" (thẩm định gói Kaggle sinh ảnh theo sách, TN11 full).

Chỉ CPU, 0 API, 0 GPU. KHÔNG sửa tệp có sẵn của repo/gói; mọi đầu ra ở measure_out/_tn11/full/review/goi_du_lieu/.
Chạy mọi script r_goi_du_lieu_* với PYTHONDONTWRITEBYTECODE=1 và cwd = thư mục đầu ra (để tệp log phụ của thư viện thứ ba không rơi ra ngoài):

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 ../../../../../.venv/bin/python -B <script>
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FULL = REPO / "measure_out" / "_tn11" / "full"
PACK = FULL / "kaggle_pack_20261004"
ZIP = FULL / "kaggle_pack_20261004.zip"
REPORT = FULL / "kaggle_pack_20261004_bao_cao.json"
SPEC = FULL / "kaggle_spec"
OUT = FULL / "review" / "goi_du_lieu"
HTD = FULL / "hien_trang_du_lieu"

ORDER = ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK", "B18", "B34"]
HUMAN = ("B18", "B34", "L16", "TK")          # evaluation_only: có nhãn người, chỉ để CHẤM
STT = ("stt2", "stt4", "stt11")
# bảng nhãn SAU chon_chu (đường mà khâu quy mô đọc: T.BOOKS[b]["labels"]) và bảng TRƯỚC chon_chu
POST = {"stt2": "dataset_out/labels_final.csv", "stt4": "dataset_out/labels_final.csv", "stt11": "dataset_out/labels_final.csv",
        "Chr": "prepared/Chrestomathie1872/dataset_out/labels_gated.csv", "L83": "prepared/LucVanTien1883/dataset_out/labels_gated.csv",
        "KVK": "prepared/KimVanKieu1884/dataset_out/labels_gated.csv", "L16": "prepared/LucVanTien1916/dataset_out/labels_gated.csv",
        "TK": "prepared/TruyenKieu1872/dataset_out/labels_gated.csv",
        "B18": "prepared/_auto/SachKinhThayCaBinh/dataset_out/labels_gated.csv", "B34": "prepared/_auto/SachDungLyHoThan/dataset_out/labels_gated.csv"}
PREF = {k: (v if k in STT else v.replace("labels_gated.csv", "labels_gated_truoc_chon_chu.csv")) for k, v in POST.items()}
BOOK_VAL = {"stt2": "stt2", "stt4": "stt4", "stt11": "stt11", "Chr": "chrestomathie1872", "L83": "lucvantien1883", "KVK": "kimvankieu1884",
            "L16": "lucvantien1916", "TK": "truyenkieu1872", "B18": "sachkinhthaycabinh", "B34": "sachdunglyhothan"}
DS = {"stt2": "SachThanhTruyen", "stt4": "SachThanhTruyen", "stt11": "SachThanhTruyen", "Chr": "Chrestomathie1872", "L83": "LucVanTien1883",
      "KVK": "KimVanKieu1884", "L16": "LucVanTien1916", "TK": "TruyenKieu1872", "B18": "SachKinhThayCaBinh", "B34": "SachDungLyHoThan"}
PLAN_COLS = ["book", "page", "column", "syllable", "ocr_char", "label", "tier", "rule"]   # cột tối thiểu để dựng danh sách chữ (không cột chon_chu*)
ANCHOR_PREFIX = "s1_inter_s2_direct"
K = 5


def rd(p, **k) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, **k)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj) -> str:
    """== w_quy_mo_va_ngan_sach_lib.sha256_of (để đối chiếu mã băm kế hoạch)."""
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def jdump(obj, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_jd), encoding="utf-8")
    return p


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Inv:
    """Bất biến PASS/FAIL thay cho agent phản biện (quy ước CLAUDE.md)."""

    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append(dict(ten=name, dat=bool(ok), chi_tiet=str(detail)[:300]))
        print(f"[inv] {'PASS' if ok else 'FAIL'} {name} {str(detail)[:200]}", flush=True)
        return bool(ok)

    def summary(self):
        return dict(tong=len(self.rows), dat=sum(r["dat"] for r in self.rows), rot=[r["ten"] for r in self.rows if not r["dat"]])


# ---------------------------------------------------------------------------------------------- Unicode
def is_pua(c: str) -> bool:
    o = ord(c)
    return 0xE000 <= o <= 0xF8FF or 0xF0000 <= o <= 0xFFFFD or 0x100000 <= o <= 0x10FFFD


def block_of(c: str) -> str:
    o = ord(c)
    if 0x4E00 <= o <= 0x9FFF:
        return "URO"
    if 0x3400 <= o <= 0x4DBF:
        return "ExtA"
    if 0x20000 <= o <= 0x2A6DF:
        return "ExtB"
    if 0x2A700 <= o <= 0x2EBEF:
        return "ExtC-F"
    if 0x30000 <= o <= 0x323AF:
        return "ExtG-H"
    if 0xF900 <= o <= 0xFAFF or 0x2F800 <= o <= 0x2FA1F:
        return "Compat"
    if 0x2E80 <= o <= 0x2FDF:
        return "Kangxi"
    if 0xE000 <= o <= 0xF8FF:
        return "PUA-BMP"
    if o >= 0xF0000:
        return "PUA-sup"
    return "other"


# ---------------------------------------------------------------------------------------------- phông
def cmap_union(path) -> set:
    """Hợp mọi bảng con cmap (đúng cách FontDiffuser is_char_in_font và khâu quy mô)."""
    from fontTools.ttLib import TTFont
    f = TTFont(str(path), fontNumber=0, lazy=True)
    s: set = set()
    for tb in f["cmap"].tables:
        s |= set(tb.cmap)
    f.close()
    return s


def cmap_freetype(path) -> tuple[set, tuple]:
    """Bảng cmap mà FreeType sẽ CHỌN cho FT_ENCODING_UNICODE: ưu tiên (3,10) rồi (0,4)/(0,6)/(3,1)/(0,*) — xấp xỉ quy tắc FT_Select_Charmap."""
    from fontTools.ttLib import TTFont
    f = TTFont(str(path), fontNumber=0, lazy=True)
    tabs = {(t.platformID, t.platEncID): set(t.cmap) for t in f["cmap"].tables}
    f.close()
    for key in ((3, 10), (0, 6), (0, 4), (3, 1), (0, 3), (0, 2), (0, 1), (0, 0)):
        if key in tabs:
            return tabs[key], key
    return set(), (-1, -1)
