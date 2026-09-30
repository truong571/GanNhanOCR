"""tn9lib.py — TN9 (STT + sách in tới 90/90): tiện ích chung. 0 API.

Quy ước: chỉ ĐỌC pipeline/, core/, config/, data/, dataset/, prepared/, dataset_out/; mọi đầu ra ở measure_out/_tn9/.
Sách in / Borg / IHR: đọc từ ẢNH CHỤP measure_out/_tn9/snap/ (agent khác đang dựng lại prepared/<Bộ>/dataset_out).
STT không bị dựng lại -> đọc thẳng dataset_out/ + prepared/SachThanhTruyen*/ (chỉ đọc).
Tái dùng bảng TN8 (measure_out/_tn8/{base,cand,emb,stt_lt2}) — sinh từ CÙNG nhãn trước khi dựng lại (đã kiểm số dòng).
Không mở ảnh/CSV bằng mắt: mọi xử lý bằng mã.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TN8 = REPO / "lab/thu_nghiem_kim/TN8_chon_chu"
for p in (str(REPO), str(REPO / "scripts/measure"), str(REPO / "lab/thu_nghiem_anh_chu/TN6_hop_anh"), str(TN8)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = REPO / "measure_out" / "_tn9"
OUT.mkdir(parents=True, exist_ok=True)
SNAP = OUT / "snap"
T8 = REPO / "measure_out" / "_tn8"
STT = ["stt2", "stt4", "stt11"]
STT_FULL = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}
BOOT_B, BOOT_SEED = 2000, 20260930


def rd(p, **k) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, **k)


def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def boot_ci_pages(ok, pages, B=BOOT_B, seed=BOOT_SEED):
    ok = np.asarray(ok, float); pages = np.asarray(pages)
    if len(ok) == 0:
        return (float("nan"), float("nan"), float("nan"))
    up, inv = np.unique(pages, return_inverse=True)
    k = np.bincount(inv, weights=ok, minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(up), size=(B, len(up)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return (float(ok.mean()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))


def page_blocks(pages, k: int = 5) -> np.ndarray:
    up = np.array(sorted(set(pages)))
    blk = {p: min(k - 1, i * k // max(1, len(up))) for i, p in enumerate(up)}
    return np.array([blk[p] for p in pages])


_C = {}


def lex():
    if not _C:
        from pipeline.gold_exact import common as C
        C.set_lexicon(C.Assets())
        _C["C"] = C
    return _C["C"]


# ------------------------------------------------------------------ âm tiết mờ (lỗi dấu của OCR QN)
def strip_tone(s: str) -> str:
    """Bỏ DẤU THANH (giữ â ê ô ơ ư ă đ)."""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch) == "Mn" and ch in "̣̀́̃̉":
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def fold(s: str) -> str:
    """Bỏ MỌI dấu (a-z) + đ -> d."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


@lru_cache(maxsize=1)
def fuzzy_index():
    C = lex()
    R = C._dicts()[0]
    by_tone, by_fold = {}, {}
    for k, v in R.items():
        by_tone.setdefault(strip_tone(k), set()).update(v)
        by_fold.setdefault(fold(k), set()).update(v)
    return by_tone, by_fold


def R_tone(s: str) -> set:
    C = lex()
    return fuzzy_index()[0].get(strip_tone(C.R_key(s)), set())


def R_fold(s: str) -> set:
    C = lex()
    return fuzzy_index()[1].get(fold(C.R_key(s)), set())


def load_stt(b: str) -> pd.DataFrame:
    """Bảng ô TN8 của STT + lt2 (TN8 t03) — một hàng/ô, cùng thứ tự base."""
    D = pd.read_pickle(T8 / "base" / f"{b}.pkl")
    L2 = pd.read_pickle(T8 / "stt_lt2" / f"{b}.pkl")
    D["lt2"] = L2.lt2.fillna("").values
    D["lt2_status"] = L2.lt2_status.fillna("").values
    D["lt2_geom"] = L2.lt2_geom.fillna("").values if "lt2_geom" in L2 else ""
    return D
