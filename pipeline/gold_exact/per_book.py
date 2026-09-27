"""per_book.py — bản lọc theo bộ `dataset/<Bộ>/gold_exact.csv` (dọn + khối tài liệu). Không nạp numpy/pandas/torch.

Bản lọc theo bộ trỏ crop chuẩn của bộ gộp (`crop_chuan` = `../_ALL/crops_chuan/…`). Hễ bộ gộp bị gộp lại (crops_chuan/ bị xoá)
hoặc bước gold_exact dọn đầu ra cũ, MỌI bản lọc theo bộ phải bị xoá theo — kẻo còn tệp trỏ tới crop đã xoá / labels.csv cũ (N2).
Dùng bởi pipeline/tools/merge_datasets.py (lúc dọn) và pipeline/gold_exact/publish.py (clean/publish).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from . import doc_text as DT

NAME = "gold_exact.csv"


def _refers_to(f: Path, all_dir: Path) -> bool:
    """Tệp gold_exact.csv theo bộ có trỏ crop chuẩn của all_dir không (dòng ok đầu tiên)."""
    rel = os.path.relpath(all_dir, f.parent).replace(os.sep, "/") + "/"
    try:
        with open(f, encoding="utf-8", newline="") as h:
            r = csv.DictReader(h)
            if "crop_chuan" not in (r.fieldnames or []):
                return False
            for row in r:
                c = row.get("crop_chuan") or ""
                if c:
                    return c.startswith(rel)
    except (OSError, UnicodeDecodeError, csv.Error):
        return True                      # tệp hỏng cạnh bộ gộp -> coi như đầu ra cũ của bước này
    return False


def clean_per_book(root: Path, all_dir: Path, books=None) -> list[str]:
    """Xoá `<root>/<Bộ>/gold_exact.csv` của mọi bộ trong `books` + mọi thư mục con của root có gold_exact.csv trỏ crop của
    all_dir; đưa khối gold_exact trong README/DATASHEET của bộ về trạng thái 'không có tệp'. Trả danh sách đã xoá."""
    root, all_dir = Path(root).resolve(), Path(all_dir).resolve()
    gone = []
    if not root.is_dir():
        return gone
    books = set(books or ())
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.resolve() == all_dir:
            continue
        f = d / NAME
        if not f.exists():
            continue
        if d.name in books or _refers_to(f, all_dir):
            f.unlink()
            gone.append(f"{d.name}/{NAME}")
            DT.refresh_book_docs(d, present=False)
    return gone


def gold_images(labels_csv: Path) -> list[str]:
    """Danh sách `image` của các dòng tier GOLD trong labels.csv của bộ nguồn (đã sắp) — cho invariant N3."""
    with open(labels_csv, encoding="utf-8", newline="") as h:
        return sorted(r["image"] for r in csv.DictReader(h) if r.get("tier") == "GOLD")
