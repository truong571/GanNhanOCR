"""tn6lib.py — TN6 (hộp ảnh): ghép ô nhãn máy Borg ↔ HỘP NGƯỜI (dataset/_BORG_NHAN_NGUOI) để đo "crop đúng vị trí".

Cùng chuỗi ghép với TN5 h01_truth.py (0 API, CPU, không mở ảnh):
  ô (trang, cột, syl_idx) -> J = syl_span[0](cột) + syl_idx (adapter prepared/_auto/<S>/transcriptions; = borg_endtoend_eval)
  -> (câu, vị trí k) theo số âm câu (chỉ câu "đếm bằng") -> chỉ số chữ người của trang h_idx (human_seq, kiểm chuỗi Lo == Excel)
  -> hộp người (bbox, kind, keep_level, column vật lý).
slot_ok = tâm hộp máy (bbox labels) ∈ hộp người của khe; d = (chỉ số chữ người có hộp chứa tâm, gần h_idx nhất) − h_idx.
same_col = hộp người chứa tâm máy nằm CÙNG cột vật lý với hộp người của khe (phân rã lỗi: trong cột vs sai cột).
Nhãn người CHỈ dùng để ĐO (không đi vào quyết định của pipeline).
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "scripts/measure")):
    if p not in sys.path:
        sys.path.insert(0, p)

BOOKS = ("SachKinhThayCaBinh", "SachDungLyHoThan")
KEEP_OK = ("keep_v5", "keep", "keep_high")        # hộp người tin được (trượt ±1 ≤ 1,9 %)


def rd(p, **k):
    return pd.read_csv(p, dtype=str, keep_default_na=False, **k)


def bb(s):
    try:
        v = json.loads(s) if isinstance(s, str) else s
        return [float(x) for x in v[:4]] if v is not None and len(v) >= 4 else None
    except Exception:  # noqa: BLE001
        return None


def human_index(book: str) -> dict:
    out = {}
    for f in sorted((REPO / "prepared" / book / "transcriptions").glob("page_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        off = 0
        for s in d["sentences"]:
            cs = "".join(c for c in s["nom_clean"] if unicodedata.category(c) == "Lo")
            out[(f.stem, s["sentence_id"])] = (off, cs)
            off += len(cs)
    return out


_CACHE: dict = {}


def truth_tables(book: str):
    """(J->h_idx theo trang, hộp người theo trang) — cache theo sách."""
    if book in _CACHE:
        return _CACHE[book]
    import borg_endtoend_eval as BE
    H = BE.human_pages(book)
    hidx = human_index(book)
    j2h = {}
    for pg, h in H.items():
        acc, m = 0, {}
        for s in h["sents"]:
            if s["eq"]:
                off, cs = hidx.get((pg, s["sid"]), (None, ""))
                if off is not None and cs == s["nom"]:
                    for k in range(len(s["syl"])):
                        m[acc + k] = off + k
            acc += len(s["syl"])
        j2h[pg] = m
    HB = rd(REPO / "dataset/_BORG_NHAN_NGUOI/labels.csv",
            usecols=["book", "page", "idx", "column", "kind", "bbox", "keep_level"])
    HB = HB[HB.book == book]
    pages = {}
    for p, g in HB.groupby("page"):
        B = np.array([bb(b) or [np.nan] * 4 for b in g.bbox], float)
        pages[p] = dict(ids=g.idx.astype(int).values, B=B, keep=g.keep_level.values,
                        col=pd.to_numeric(g.column, errors="coerce").fillna(-1).astype(int).values, kind=g.kind.values)
    spans = BE.load_spans(book)
    _CACHE[book] = (j2h, pages, spans)
    return _CACHE[book]


def join(book: str, L: pd.DataFrame) -> pd.DataFrame:
    """L: nhãn máy (page, column, syl_idx, bbox, ...). Trả L + h_idx, keep_level, slot_ok, d, same_col, has_h."""
    j2h, pages, spans = truth_tables(book)
    n = len(L)
    h_idx = np.full(n, -1); keep = np.array([""] * n, dtype=object)
    slot = np.zeros(n, bool); d = np.full(n, np.nan); same = np.full(n, np.nan); has_h = np.zeros(n, bool)
    iou = np.full(n, np.nan)
    for t, (pg, col, si, bs) in enumerate(zip(L.page, L.column, L.syl_idx, L.bbox)):
        try:
            col, si = int(float(col)), int(float(si))
        except (TypeError, ValueError):
            continue
        j0 = spans.get((pg, col))
        if j0 is None or si < 0:
            continue
        hi = j2h.get(pg, {}).get(j0 + si)
        if hi is None:
            continue
        h_idx[t] = hi
        P = pages.get(pg)
        if P is None:
            continue
        m = P["ids"] == hi
        if not m.any():
            continue
        k = int(np.argmax(m))
        keep[t] = P["keep"][k]
        hb = P["B"][k]
        a = bb(bs)
        if a is None or np.isnan(hb[0]):
            continue
        has_h[t] = True
        cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        slot[t] = hb[0] <= cx <= hb[2] and hb[1] <= cy <= hb[3]
        ix = max(0.0, min(a[2], hb[2]) - max(a[0], hb[0])); iy = max(0.0, min(a[3], hb[3]) - max(a[1], hb[1]))
        inter = ix * iy
        iou[t] = inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (hb[2] - hb[0]) * (hb[3] - hb[1]) - inter)
        B = P["B"]
        ins = (cx >= B[:, 0]) & (cx <= B[:, 2]) & (cy >= B[:, 1]) & (cy <= B[:, 3])
        if ins.any():
            ids = P["ids"][ins]
            q = int(np.argmin(np.abs(ids - hi)))
            d[t] = ids[q] - hi
            same[t] = float(P["col"][ins][q] == P["col"][k])
    out = L.copy()
    out["h_idx"] = h_idx; out["keep_level"] = keep; out["has_h"] = has_h
    out["slot_ok"] = slot; out["d"] = d; out["same_col"] = same; out["iou_h"] = iou
    return out


def slot_block(T: pd.DataFrame, mask=None) -> dict:
    """Tỉ lệ đúng vị trí trên hộp người tin được (keep_high+) và keep_v5 (lọc Paddle)."""
    if mask is not None:
        T = T[mask]
    k = T[T.has_h & T.keep_level.isin(KEEP_OK)]
    k5 = T[T.has_h & (T.keep_level == "keep_v5")]
    wrong = k[~k.slot_ok]
    return dict(n=int(len(k)), slot=round(float(k.slot_ok.mean()), 4) if len(k) else None,
                n_v5=int(len(k5)), slot_v5=round(float(k5.slot_ok.mean()), 4) if len(k5) else None,
                wrong_same_col=round(float((wrong.same_col == 1).mean()), 4) if len(wrong) else None,
                wrong_d1=round(float(wrong.d.abs().eq(1).mean()), 4) if len(wrong) else None,
                wrong_nobox=round(float(wrong.d.isna().mean()), 4) if len(wrong) else None)
