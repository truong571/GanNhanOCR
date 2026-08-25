"""Kiểm — và nếu được phép thì vá — môi trường thư viện, TRƯỚC khi pipeline chạy.

VÌ SAO ĐỨNG RIÊNG THÀNH MỘT BƯỚC
--------------------------------
Thiếu một gói giữa chừng thì pipeline chết ở phút thứ 20, sau khi đã xoá `dataset_out/`.
Kiểm ở preflight tốn 1 giây và chặn được cả lần chạy hỏng.

BA CẠM BẪY CỦA MÔI TRƯỜNG NÀY (chép từ đầu requirements.txt — ĐỪNG tự ý "sửa cho gọn")
-------------------------------------------------------------------------------------
* `pygame` KHÔNG có wheel cho Python 3.14 -> dùng `pygame-ce` (drop-in, import vẫn là
  `import pygame`). Cài `pygame` sẽ build từ nguồn và hỏng.
* `vietocr` PHẢI cài `--no-deps`: metadata của nó pin albumentations/imgaug, kéo theo
  Pillow 10.2.0 vốn không có wheel cho 3.14. Các deps runtime nó cần đều đã có sẵn.
* `safetensors` ghim bản `0.8.0rc0` vì đó là bản có wheel cho 3.14.

Vì ba lẽ đó, `--fix` KHÔNG chạy `pip install -r requirements.txt` một phát rồi thôi:
nó cài phần thường trước, rồi cài `vietocr --no-deps` riêng, đúng thứ tự tài liệu.

    python -m pipeline.tools.check_deps           # chỉ kiểm, exit 1 nếu thiếu
    python -m pipeline.tools.check_deps --fix     # kiểm rồi cài phần còn thiếu
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQ = REPO / "requirements.txt"
# gói phải cài RIÊNG với --no-deps (xem đầu requirements.txt)
NO_DEPS = {"vietocr"}


def parse_req(path: Path = REQ) -> list[tuple[str, str, str]]:
    """-> [(tên, phiên_bản_ghim hoặc '', dòng gốc)]"""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!\[;]", line)[0].strip()
        if not name:
            continue
        m = re.search(r"==\s*([^\s;]+)", line)
        out.append((name, m.group(1) if m else "", line))
    return out


def audit() -> dict:
    import importlib.metadata as md
    thieu, lech, ok = [], [], []
    for name, pin, line in parse_req():
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            thieu.append((name, pin, line))
            continue
        if pin and got != pin:
            lech.append((name, pin, got, line))
        else:
            ok.append(name)
    return {"ok": ok, "thieu": thieu, "lech": lech}


def fix(a: dict, py: str) -> int:
    """Cài phần còn thiếu, TÔN TRỌNG thứ tự và cờ đặc biệt trong requirements.txt."""
    thuong = [l for n, p, l in a["thieu"] if n.lower() not in NO_DEPS]
    thuong += [l for n, p, g, l in a["lech"] if n.lower() not in NO_DEPS]
    rieng = [l for n, p, l in a["thieu"] if n.lower() in NO_DEPS]
    rieng += [l for n, p, g, l in a["lech"] if n.lower() in NO_DEPS]

    rc = 0
    if thuong:
        print(f"  [deps] cài {len(thuong)} gói thường ...", flush=True)
        rc |= subprocess.run([py, "-m", "pip", "install", *thuong]).returncode
    for line in rieng:
        # --no-deps: metadata của vietocr kéo theo Pillow 10.2.0 (không có wheel 3.14)
        print(f"  [deps] cài RIÊNG --no-deps: {line}", flush=True)
        rc |= subprocess.run([py, "-m", "pip", "install", "--no-deps", line]).returncode
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.check_deps")
    ap.add_argument("--fix", action="store_true", help="cài phần còn thiếu / lệch phiên bản")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    if not REQ.exists():
        print(f"[deps] không thấy {REQ}", file=sys.stderr)
        return 1

    a = audit()
    tong = len(a["ok"]) + len(a["thieu"]) + len(a["lech"])
    if not a["thieu"] and not a["lech"]:
        print(f"  [deps] {len(a['ok'])}/{tong} gói ĐỦ và ĐÚNG phiên bản ghim")
        return 0

    for n, p, l in a["thieu"]:
        print(f"  [deps] 🔴 THIẾU        {n}{'==' + p if p else ''}")
    for n, p, g, l in a["lech"]:
        print(f"  [deps] ⚠️  LỆCH        {n}: ghim {p}, đang có {g}")

    if not args.fix:
        print(f"  [deps] {len(a['thieu'])} thiếu · {len(a['lech'])} lệch. Vá bằng:\n"
              f"      {args.python} -m pipeline.tools.check_deps --fix", file=sys.stderr)
        return 1

    if fix(a, args.python) != 0:
        print("  [deps] 🔴 pip báo lỗi — xem log ở trên", file=sys.stderr)
        return 1
    b = audit()
    if b["thieu"] or b["lech"]:
        print(f"  [deps] 🔴 VẪN còn {len(b['thieu'])} thiếu / {len(b['lech'])} lệch sau khi cài",
              file=sys.stderr)
        return 1
    print(f"  [deps] ✅ đã vá xong — {len(b['ok'])}/{tong} gói đủ và đúng phiên bản")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
