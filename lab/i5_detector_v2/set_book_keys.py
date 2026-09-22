#!/usr/bin/env python
"""Ghi/xoá khoá `detector_ckpt` / `detector_resize` cho mục sách trong config/pipeline_<Book>.yaml (giữ nguyên comment).

  set_book_keys.py config/pipeline_LucVanTien1883.yaml --book LucVanTien1883 \\
      --set detector_ckpt=train_crop/detector_r34_v2_litho.pt --set detector_resize=area
  set_book_keys.py <cfg> --book <Book> --remove detector_ckpt --remove detector_resize   # hoàn nguyên v1
Chỉ sửa trong khối `  - name: <Book>` (đến mục sách kế / khoá cấp 1 kế); khoá đã có -> thay dòng; chưa có -> chèn
sau `box_decoder:` (hoặc `det_thr:` / `layout:`). Idempotent. In diff 1 dòng/khoá.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ANCHORS = ("box_decoder", "det_thr", "det_xmargin", "qn_syllables_per_column", "n_columns", "layout")


def _book_block(lines: list[str], book: str):
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^\s*-\s*name:\s*{re.escape(book)}\s*(#.*)?$", ln):
            start = i
            break
    if start is None:
        return None, None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^\s*-\s*name:", lines[j]) or re.match(r"^[A-Za-z_]\w*\s*:", lines[j]):
            end = j
            break
    return start, end


def apply(path: Path, book: str, sets: dict[str, str], removes: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    s, e = _book_block(lines, book)
    if s is None:
        raise SystemExit(f"{path}: không thấy mục sách '- name: {book}'")
    indent = re.match(r"^(\s*)-", lines[s]).group(1) + "  "
    log = []
    for key in removes:
        for j in range(e - 1, s, -1):
            if re.match(rf"^{indent}{key}\s*:", lines[j]):
                log.append(f"- {lines[j].rstrip()}"); del lines[j]; e -= 1
    for key, val in reversed(list(sets.items())):     # chèn ngược để thứ tự cuối = thứ tự --set
        new_line = f"{indent}{key}: {val}\n"
        hit = None
        for j in range(s + 1, e):
            if re.match(rf"^{indent}{key}\s*:", lines[j]):
                hit = j; break
        if hit is not None:
            if lines[hit] != new_line:
                log.append(f"~ {lines[hit].rstrip()}  ->  {new_line.rstrip()}")
                lines[hit] = new_line
            continue
        pos = None
        for anchor in ANCHORS:
            for j in range(s + 1, e):
                if re.match(rf"^{indent}{anchor}\s*:", lines[j]):
                    pos = j + 1
            if pos is not None:
                break
        pos = pos if pos is not None else s + 1
        lines.insert(pos, new_line); e += 1
        log.append(f"+ {new_line.rstrip()}")
    path.write_text("".join(lines), encoding="utf-8")
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--book", required=True)
    ap.add_argument("--set", action="append", default=[], help="key=value")
    ap.add_argument("--remove", action="append", default=[])
    a = ap.parse_args()
    sets = dict(kv.split("=", 1) for kv in a.set)
    log = apply(Path(a.config), a.book, sets, a.remove)
    print(f"{a.config} [{a.book}]: " + ("; ".join(log) if log else "không đổi"))


if __name__ == "__main__":
    main()
