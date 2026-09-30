"""stt.py — chữ kim lt2 (lang_type 2) của từng ô STT (đường STT của bước chọn chữ — TẮT mặc định).

Cùng phương pháp 'line' của scripts/measure/stt_lt2_eval.py (bản chép trong pipeline.gold_exact.signals_lt2): hộp chữ lt1 của ô
= cột kim dựng lại (cache prepared/_gold_exact/kim_pages/<Sách>/<trang>.pkl, thiếu thì _detect) tại nom_idx, chữ phải == ocr_char;
dòng lt1 chứa ô -> dòng lt2 IoU ≥ 0,5 -> căn chuỗi -> chữ lt2. Chỉ ĐỌC prepared/. 0 API.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
S8 = {"SachThanhTruyen2": "stt2", "SachThanhTruyen4": "stt4", "SachThanhTruyen11": "stt11"}


def lt2_reads(D, book: str, bdict: dict, log=print) -> np.ndarray:
    from pipeline.gold_exact import signals_geom as SG
    from pipeline.gold_exact import signals_lt2 as SL
    from pipeline.step0_setup import load_config
    s8 = S8[book]
    pages = SL.load_pages(s8)
    base = load_config(str(REPO / "config/pipeline.yaml"))
    SG._kim_init(str(REPO / base["paths"]["qn_to_nom_dict"]))
    out = np.array([""] * len(D), dtype=object)
    n_ok = n_pg = 0
    for pg, g in D.groupby("page"):
        pair = pages["lt2"].get(pg)
        if pair is None:
            continue
        r1, r2 = pair
        pp = SL.PagePair(r1.get("boxes_raw"), r2.get("boxes_raw"))
        pf = REPO / "prepared/_gold_exact/kim_pages" / book / f"{pg}.pkl"
        cols = None
        if pf.exists():
            try:
                z = pickle.load(open(pf, "rb"))
                cols = z.get("out") if isinstance(z, dict) else None
            except Exception:  # noqa: BLE001
                cols = None
        if cols is None:
            cols = SG._kim_page((book, pg, bdict)) or {}
        n_pg += 1
        for i, col, ni, oc in zip(g.index, g.column_i, g.nom_i, g.ocr_char):
            chars = cols.get(int(col))
            if not chars or ni < 0 or ni >= len(chars) or not chars[ni][1] or chars[ni][0] != oc:
                continue
            d = pp.by_line(chars[ni][0], [float(v) for v in chars[ni][1][:4]])
            if d.get("st_line") == "ok":
                out[i] = d.get("oth_line", "")
                n_ok += 1
    log(f"[chon_chu] lt2 STT {book}: {n_pg} trang có lt2, ghép được {n_ok:,}/{len(D):,} ô")
    return out
