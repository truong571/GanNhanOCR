"""tn8lib.py — TN8 "chọn chữ" (0 API): tiện ích chung — danh mục bộ, đường dẫn, đọc bảng, CI bootstrap cụm trang.

Quy ước: chỉ ĐỌC pipeline/, core/, config/, data/, dataset/, prepared/; mọi đầu ra ở measure_out/_tn8/.
Không mở ảnh bằng mắt/LLM — mọi xử lý ảnh bằng mã.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "scripts/measure"), str(REPO / "lab/thu_nghiem_anh_chu/TN6_hop_anh")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = REPO / "measure_out" / "_tn8"
OUT.mkdir(parents=True, exist_ok=True)

# bộ -> (tệp nhãn đầy đủ, thư mục prepared, lọc cột book, crop_source, loại chữ, sự thật)
BOOKS = {
    "stt2": dict(labels="dataset_out/labels_final.csv", prep="prepared/SachThanhTruyen2", book_filter="stt2",
                 crop="processed", kind="hand", truth=None, full="SachThanhTruyen2"),
    "stt4": dict(labels="dataset_out/labels_final.csv", prep="prepared/SachThanhTruyen4", book_filter="stt4",
                 crop="processed", kind="hand", truth=None, full="SachThanhTruyen4"),
    "stt11": dict(labels="dataset_out/labels_final.csv", prep="prepared/SachThanhTruyen11", book_filter="stt11",
                  crop="processed", kind="hand", truth=None, full="SachThanhTruyen11"),
    "Chr": dict(labels="prepared/Chrestomathie1872/dataset_out/labels_gated.csv", prep="prepared/Chrestomathie1872",
                crop="original", kind="print", truth=None, full="Chrestomathie1872"),
    "L83": dict(labels="prepared/LucVanTien1883/dataset_out/labels_gated.csv", prep="prepared/LucVanTien1883",
                crop="original", kind="print", truth="diban", full="LucVanTien1883"),
    "KVK": dict(labels="prepared/KimVanKieu1884/dataset_out/labels_gated.csv", prep="prepared/KimVanKieu1884",
                crop="original", kind="print", truth="diban", full="KimVanKieu1884"),
    "L16": dict(labels="prepared/LucVanTien1916/dataset_out/labels_gated.csv", prep="prepared/LucVanTien1916",
                crop="original", kind="print", truth="ihr", full="LucVanTien1916"),
    "TK": dict(labels="prepared/TruyenKieu1872/dataset_out/labels_gated.csv", prep="prepared/TruyenKieu1872",
               crop="original", kind="print", truth="ihr", full="TruyenKieu1872"),
    "B18": dict(labels="prepared/_auto/SachKinhThayCaBinh/dataset_out/labels_gated.csv",
                prep="prepared/_auto/SachKinhThayCaBinh", crop="original", kind="hand", truth="borg",
                full="SachKinhThayCaBinh"),
    "B34": dict(labels="prepared/_auto/SachDungLyHoThan/dataset_out/labels_gated.csv",
                prep="prepared/_auto/SachDungLyHoThan", crop="original", kind="hand", truth="borg",
                full="SachDungLyHoThan"),
}
ORDER = ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK", "B18", "B34"]
BOOT_B, BOOT_SEED = 2000, 20260930


def rd(p, **k) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, **k)


def base_path(b: str) -> Path:
    return OUT / "base" / f"{b}.pkl"


def load_base(b: str) -> pd.DataFrame:
    return pd.read_pickle(base_path(b))


def bbox_of(s):
    try:
        v = json.loads(s) if isinstance(s, str) else s
        return [float(x) for x in v[:4]] if v is not None and len(v) >= 4 else None
    except Exception:  # noqa: BLE001
        return None


def boot_ci_pages(ok: np.ndarray, pages: np.ndarray, B=BOOT_B, seed=BOOT_SEED):
    """Tỉ lệ ok.mean() + CI 95 % bootstrap CỤM TRANG."""
    ok = np.asarray(ok, float); pages = np.asarray(pages)
    if len(ok) == 0:
        return (float("nan"), float("nan"), float("nan"))
    up, inv = np.unique(pages, return_inverse=True)
    k = np.bincount(inv, weights=ok, minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(up), size=(B, len(up)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return (float(ok.mean()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))


def page_blocks(pages: np.ndarray, k: int = 5) -> np.ndarray:
    """Khối trang liền (k khối theo thứ tự tên trang) — dùng cho CV cụm trang."""
    up = np.array(sorted(set(pages)))
    blk = {p: min(k - 1, i * k // max(1, len(up))) for i, p in enumerate(up)}
    return np.array([blk[p] for p in pages])


_LEX = {}


def lex():
    """R_of / var_eq_plus / var_in của gold_exact.common (sau set_lexicon)."""
    if not _LEX:
        from pipeline.gold_exact import common as C
        C.set_lexicon(C.Assets())
        _LEX["C"] = C
    return _LEX["C"]


def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
