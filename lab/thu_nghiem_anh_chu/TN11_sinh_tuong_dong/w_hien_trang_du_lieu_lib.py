"""w_hien_trang_du_lieu_lib.py — thư viện dùng chung cho hướng "HIỆN TRẠNG DỮ LIỆU MỚI NHẤT" của gói Kaggle sinh ảnh theo sách (TN11 full).

Chỉ ĐỌC dataset/, prepared/, dataset_out/, config/, measure_out/_tn8/; mọi đầu ra ở measure_out/_tn11/full/hien_trang_du_lieu/.
0 API, 0 GPU, không mở ảnh bằng mắt/LLM (mọi số do mã sinh). Chạy với PYTHONDONTWRITEBYTECODE=1 (không để lại .pyc ngoài thư mục đầu ra).

10 "cuốn" của pipeline (mã TN8): stt2 stt4 stt11 (ba quyển Sách Các Thánh Truyện) · Chr · L83 · KVK · L16 · TK · B18 · B34.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = REPO / "measure_out" / "_tn11" / "full" / "hien_trang_du_lieu"
OUT.mkdir(parents=True, exist_ok=True)
TN8 = REPO / "measure_out" / "_tn8"

ORDER = ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK", "B18", "B34"]
ANCHOR_PREFIX = "s1_inter_s2_direct"      # định nghĩa ô neo tự động (TN8 t05_feats.ANCHOR_RULES[0]; yêu cầu của đề bài)

# ma -> cấu hình nguồn. all_file = bảng MỌI tầng mới nhất (sau chon_chu với 7 bộ không phải STT); all_root = thư mục `image` của bảng đó tương đối với.
# ds = thư mục dataset/<Bộ>/ chứa bản ĐÃ GIAO (labels.csv: GOLD + GOLD_text_only + SYLLABLE; crop ở dataset/<Bộ>/{gold,syllable}/).
_STT = dict(ds="SachThanhTruyen", all_file="dataset_out/labels_final.csv", all_root="dataset_out", crop_source="processed",
            kind="hand", truth=None, yaml="config/pipeline.yaml")
CFG = {
    "stt2": dict(_STT, name="SachThanhTruyen2", book_val="stt2", prep="prepared/SachThanhTruyen2", old_ds="SachThanhTruyen2"),
    "stt4": dict(_STT, name="SachThanhTruyen4", book_val="stt4", prep="prepared/SachThanhTruyen4", old_ds="SachThanhTruyen4"),
    "stt11": dict(_STT, name="SachThanhTruyen11", book_val="stt11", prep="prepared/SachThanhTruyen11", old_ds="SachThanhTruyen11"),
    "Chr": dict(name="Chrestomathie1872", ds="Chrestomathie1872", book_val="chrestomathie1872", prep="prepared/Chrestomathie1872",
                all_file="prepared/Chrestomathie1872/dataset_out/labels_gated.csv", all_root="prepared/Chrestomathie1872/dataset_out",
                crop_source="original", kind="print", truth=None, yaml="config/pipeline_Chrestomathie1872.yaml"),
    "L83": dict(name="LucVanTien1883", ds="LucVanTien1883", book_val="lucvantien1883", prep="prepared/LucVanTien1883",
                all_file="prepared/LucVanTien1883/dataset_out/labels_gated.csv", all_root="prepared/LucVanTien1883/dataset_out",
                crop_source="original", kind="print", truth="diban", yaml="config/pipeline_LucVanTien1883.yaml"),
    "KVK": dict(name="KimVanKieu1884", ds="KimVanKieu1884", book_val="kimvankieu1884", prep="prepared/KimVanKieu1884",
                all_file="prepared/KimVanKieu1884/dataset_out/labels_gated.csv", all_root="prepared/KimVanKieu1884/dataset_out",
                crop_source="original", kind="print", truth="diban", yaml="config/pipeline_KimVanKieu1884.yaml"),
    "L16": dict(name="LucVanTien1916", ds="LucVanTien1916", book_val="lucvantien1916", prep="prepared/LucVanTien1916",
                all_file="prepared/LucVanTien1916/dataset_out/labels_gated.csv", all_root="prepared/LucVanTien1916/dataset_out",
                crop_source="original", kind="print", truth="ihr", yaml="config/pipeline_LucVanTien1916.yaml"),
    "TK": dict(name="TruyenKieu1872", ds="TruyenKieu1872", book_val="truyenkieu1872", prep="prepared/TruyenKieu1872",
               all_file="prepared/TruyenKieu1872/dataset_out/labels_gated.csv", all_root="prepared/TruyenKieu1872/dataset_out",
               crop_source="original", kind="print", truth="ihr", yaml="config/pipeline_TruyenKieu1872.yaml"),
    "B18": dict(name="SachKinhThayCaBinh", ds="SachKinhThayCaBinh", book_val="sachkinhthaycabinh", prep="prepared/_auto/SachKinhThayCaBinh",
                all_file="prepared/_auto/SachKinhThayCaBinh/dataset_out/labels_gated.csv",
                all_root="prepared/_auto/SachKinhThayCaBinh/dataset_out", crop_source="original", kind="hand", truth="borg",
                yaml="config/pipeline_SachKinhThayCaBinh.yaml"),
    "B34": dict(name="SachDungLyHoThan", ds="SachDungLyHoThan", book_val="sachdunglyhothan", prep="prepared/_auto/SachDungLyHoThan",
                all_file="prepared/_auto/SachDungLyHoThan/dataset_out/labels_gated.csv",
                all_root="prepared/_auto/SachDungLyHoThan/dataset_out", crop_source="original", kind="hand", truth="borg",
                yaml="config/pipeline_SachDungLyHoThan.yaml"),
}
for _m, _c in CFG.items():
    _c["ma"] = _m

# Cột/đường có nhãn người hoặc suy ra từ nhãn người — KHÔNG được đưa vào gói Kaggle (xem w_hien_trang_du_lieu_04_rui_ro.py).
HUMAN_COLS_TN8_BASE = {
    "gt_char": "chữ người (Borg B18/B34: bản dịch Nôm người; L16/TK: GT IHR-NomDB) — chỉ để chấm",
    "slot": "L16/TK: khe người (Slot của eval_ihr) suy ra từ GT IHR",
    "pos_known": "có hộp/GT người tại vị trí ô (Borg: hộp người keep_high+; IHR: GT khác rỗng)",
    "pos_ok": "ô khớp khe người (suy ra từ hộp người + chữ người)",
    "keep_level": "mức tin hộp người của Borg (borg_human_eval/tn6lib.join)",
    "ref": "chữ DỊ BẢN người (L83/KVK) cùng vị trí âm — văn bản người của ấn bản khác, không phải nhãn của chính cuốn",
}
HUMAN_COLS_TN8_CAND = {
    "y": "ứng viên == gt_char (hoặc dị thể V1+) — nhãn đúng/sai suy từ chữ người",
    "y_ref": "ứng viên == chữ dị bản người",
    "f_hum": "cos(crop, nguyên mẫu ảnh NGƯỜI của Borg sách kia) — đặc trưng dựng từ nhãn người của sách kia (LOBO)",
    "n_hum": "số mẫu người của chữ (sách kia)",
    "f_humk": "biến thể f_hum cho kim",
    "f_vW": "CNN kiểm ảnh↔chữ: bảng W học từ glyph + crop người (hand_*: Borg keep; vft_*: IHR phần A + Borg) — LOBO, vẫn là suy ra từ nhãn người",
    "f_vP": "CNN kiểm: nguyên mẫu người P (chỉ khi nh ≥ 3) — dựng từ nhãn người",
}

# cột công khai của dataset/<Bộ>/labels.csv (12 cột cố định)
PUB_COLS = ["image", "book", "page", "column", "ocr_char", "syllable", "label", "unicode", "tier", "rule", "bbox", "image_md5"]


def rd(p, **k) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, **k)


def mt(p) -> str:
    p = Path(p)
    if not p.exists():
        return "THIEU"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except Exception:  # noqa: BLE001
        return str(p)


def load_all(ma: str) -> pd.DataFrame:
    """Bảng MỌI tầng mới nhất của cuốn (STT: dataset_out/labels_final.csv lọc theo cột book; còn lại labels_gated.csv)."""
    c = CFG[ma]
    D = rd(REPO / c["all_file"])
    if ma.startswith("stt"):
        D = D[D.book == c["book_val"]]
    return D.reset_index(drop=True)


def load_pub(ma: str) -> pd.DataFrame:
    """dataset/<Bộ>/labels.csv (bản giao: GOLD + GOLD_text_only + SYLLABLE); STT lọc theo cột book."""
    c = CFG[ma]
    D = rd(REPO / "dataset" / c["ds"] / "labels.csv")
    if ma.startswith("stt"):
        D = D[D.book == c["book_val"]]
    return D.reset_index(drop=True)


def load_ge(ma: str) -> pd.DataFrame:
    c = CFG[ma]
    G = rd(REPO / "dataset" / c["ds"] / "gold_exact.csv")
    if ma.startswith("stt"):
        G = G[G.set8 == ma]
    return G.reset_index(drop=True)


def load_trace(ma: str) -> pd.DataFrame:
    c = CFG[ma]
    return rd(REPO / "dataset" / c["ds"] / "labels_trace.csv")


_R = {}


def R_of(s: str) -> set:
    if "f" not in _R:
        from pipeline.gold_exact import common as C   # chỉ dùng R_of (Dict/QuocNgu_SinoNom.csv), không cần Assets
        _R["f"] = C.R_of
    return _R["f"](s)


def anchor_mask(D: pd.DataFrame) -> np.ndarray:
    """Ô neo tự động: rule bắt đầu bằng s1_inter_s2_direct ∧ label ≠ '' ∧ label == ocr_char ∧ label ∈ R(âm). (t05_feats.build; bỏ điều kiện crop_ok — đếm tệp riêng)"""
    a = (D.rule.str.startswith(ANCHOR_PREFIX) & (D.label != "") & (D.label == D.ocr_char)).to_numpy()
    Rm = {s: R_of(s) for s in set(D.syllable[a])}
    return a & np.array([(l in Rm[s]) if x else False for l, s, x in zip(D.label, D.syllable, a)], bool)


def jdump(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_jd), encoding="utf-8")


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


def stats(x, nd=1) -> dict:
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return {}
    return dict(n=int(len(x)), mean=round(float(x.mean()), nd), median=round(float(np.median(x)), nd),
                p10=round(float(np.percentile(x, 10)), nd), p90=round(float(np.percentile(x, 90)), nd),
                min=round(float(x.min()), nd), max=round(float(x.max()), nd))


class Inv:
    """Bất biến (PASS/FAIL) thay cho agent phản biện — quy ước CLAUDE.md."""

    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append(dict(ten=name, dat=bool(ok), chi_tiet=str(detail)[:240]))
        return bool(ok)

    def summary(self):
        return dict(tong=len(self.rows), dat=sum(r["dat"] for r in self.rows), rot=[r for r in self.rows if not r["dat"]])
