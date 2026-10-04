#!/usr/bin/env python3
"""Đọc cache Google Vision ĐÃ CHẠY (vision/cache/<cfg>/) — KHÔNG gọi mạng, không tốn hạn mức. Dùng khi nghiên cứu.

    import sys; sys.path.insert(0, "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/vision")
    import vision_data as vd
    vd.status()                                   # số trang đã có theo cuốn (+ số ký hiệu)
    syms = vd.load_page("KVK", "page_0075")       # list[dict]: ch, conf, x0, y0, x1, y1, ang  — toạ độ TRANG GỐC (px của prepared/<Bộ>/pages_denoised)
    df = vd.symbols_df(["KVK", "L83"])            # pandas.DataFrame mọi ký hiệu: book, page, ch, conf, x0..y1, ang
    for book, page, syms in vd.iter_pages("TK"): ...
    raw = vd.load_raw("KVK", "page_0075")         # phản hồi THÔ của Vision (responses[0]) của ảnh ghép chứa trang này + bố cục ghép

Khoá sách: KVK, L83, TK, L16, Chr, stt11, stt2, stt4, B34, B18 (thư mục sách: xem BOOK_DIRS). Cấu hình chính: vd.main_cfg() (n=2 ghép lưới, gợi ý zh-Hant, cao ô 80, trần 10,5 MP).
Bố cục ghép và hệ số co giãn từng trang nằm trong vision/cache/<cfg>/layout/<gid>.json; mỗi tệp trang chứa sx, sy, canvas_w/h, vision_w/h, frame, gid.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VISION_CACHE_ROOT", HERE / "cache"))
BOOK_DIRS = {"KVK": "KimVanKieu1884", "L83": "LucVanTien1883", "TK": "TruyenKieu1872", "L16": "LucVanTien1916", "Chr": "Chrestomathie1872",
             "stt11": "SachThanhTruyen11", "stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "B34": "SachDungLyHoThan", "B18": "SachKinhThayCaBinh"}
COLS = ["ch", "conf", "x0", "y0", "x1", "y1", "ang"]


def main_cfg() -> str:
    """Cấu hình có nhiều trang nhất (bỏ _pilot và legacy); đặt VISION_MAIN_CFG để ép."""
    if os.environ.get("VISION_MAIN_CFG"):
        return os.environ["VISION_MAIN_CFG"]
    best, bn = None, -1
    for d in ROOT.iterdir():
        if d.name.startswith("_") or d.name.startswith("legacy") or not (d / "pages").is_dir():
            continue
        n = sum(1 for _ in (d / "pages").glob("*/*.json"))
        if n > bn:
            best, bn = d.name, n
    if best is None:
        raise FileNotFoundError(f"chưa có cấu hình nào trong {ROOT}")
    return best


def _pages_dir(cfg: str | None) -> Path:
    return ROOT / (cfg or main_cfg()) / "pages"


def pages(book: str | None = None, cfg: str | None = None):
    """[(book, page)] đã có trong cache."""
    base = _pages_dir(cfg)
    out = []
    for bd in sorted(base.iterdir()):
        if book and bd.name != book:
            continue
        out += [(bd.name, f.stem) for f in sorted(bd.glob("*.json"))]
    return out


def _page_json(book: str, page: str, cfg: str | None) -> dict:
    return json.loads((_pages_dir(cfg) / book / f"{page}.json").read_text(encoding="utf-8"))


def load_page(book: str, page: str, cfg: str | None = None) -> list:
    """Ký hiệu của một trang: list[dict(ch, conf, x0, y0, x1, y1, ang)] — toạ độ trang gốc (px)."""
    j = _page_json(book, page, cfg)
    cols = j.get("cols", COLS)
    return [dict(zip(cols, s)) for s in j["sym"]]


def iter_pages(book: str | None = None, cfg: str | None = None):
    for b, p in pages(book, cfg):
        yield b, p, load_page(b, p, cfg)


def symbols_df(books=None, cfg: str | None = None):
    """pandas.DataFrame mọi ký hiệu (book, page, ch, conf, x0, y0, x1, y1, ang)."""
    import pandas as pd
    rows = []
    for b in (books or list(BOOK_DIRS)):
        for bk, pg in pages(b, cfg):
            j = _page_json(bk, pg, cfg)
            cols = j.get("cols", COLS)
            for s in j["sym"]:
                d = dict(zip(cols, s))
                d.update(book=bk, page=pg)
                rows.append(d)
    return pd.DataFrame(rows, columns=["book", "page"] + COLS)


def load_raw(book: str, page: str, cfg: str | None = None) -> dict:
    """Phản hồi THÔ (responses[0]) của ảnh ghép chứa trang này + 'layout' của ảnh ghép."""
    c = cfg or main_cfg()
    j = _page_json(book, page, c)
    gid = j["gid"]
    raw = json.loads(gzip.decompress((ROOT / c / "raw" / f"{gid}.json.gz").read_bytes()).decode("utf-8"))["responses"][0]
    lay = ROOT / c / "layout" / f"{gid}.json"
    return dict(response=raw, layout=json.loads(lay.read_text(encoding="utf-8")) if lay.exists() else None, page_meta={k: v for k, v in j.items() if k != "sym"})


def status(cfg: str | None = None, show=True) -> dict:
    c = cfg or main_cfg()
    res = {}
    for b in BOOK_DIRS:
        ps = pages(b, c)
        ns = 0
        for _, p in ps:
            ns += len(_page_json(b, p, c)["sym"])
        res[b] = dict(pages=len(ps), symbols=ns)
    if show:
        print(f"cấu hình {c}")
        for b, v in res.items():
            print(f"  {b:6s} {v['pages']:4d} trang  {v['symbols']:7,d} ký hiệu")
        print(f"  TỔNG {sum(v['pages'] for v in res.values())} trang, {sum(v['symbols'] for v in res.values()):,} ký hiệu")
    return res


if __name__ == "__main__":
    status()
