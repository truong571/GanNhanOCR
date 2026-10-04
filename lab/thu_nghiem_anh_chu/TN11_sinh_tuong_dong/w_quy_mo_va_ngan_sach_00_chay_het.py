"""w_quy_mo_va_ngan_sach_00_chay_het.py — chạy trọn hướng "quy mô danh sách chữ + ngân sách GPU" (5 bước, ≈ 2 phút CPU, 0 API, 0 GPU), rồi so mã băm với lần chạy trước.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_quy_mo_va_ngan_sach_00_chay_het.py

Thứ tự: 01_tap_chu (S0/S1/S2 + xếp hạng) → 02_phong (cmap, chuỗi phông, kiểm vẽ) → 03_phu (đường cong phủ) → 04_ngan_sach (giờ GPU, ca, ket_qua.json) → 05_kiem (kiểm độc lập).
Khi nhãn mới (chạy lại ./run_pipeline.sh --book all --yes) chỉ cần chạy lại lệnh này: danh sách được dựng lại từ labels_gated.csv / labels_final.csv mới nhất.
Mã băm in ra là của JSON đã bỏ khoá thời gian ('thoi_gian_giay') và danh sách script ('script'); hai lần chạy liên tiếp phải trùng (tất định).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "full" / "quy_mo_va_ngan_sach"
STEPS = ["w_quy_mo_va_ngan_sach_01_tap_chu.py", "w_quy_mo_va_ngan_sach_02_phong.py", "w_quy_mo_va_ngan_sach_03_phu.py",
         "w_quy_mo_va_ngan_sach_04_ngan_sach.py", "w_quy_mo_va_ngan_sach_05_kiem.py"]
FILES = ["buoc1_tap_chu.json", "phong.json", "phu.json", "ket_qua.json", "chu_uu_tien_B34.tsv", "chu_uu_tien_B34_chars.txt"]


def norm_hash(p: Path) -> str:
    if not p.exists():
        return "-"
    if p.suffix != ".json":
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("thoi_gian_giay", "script")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    return hashlib.sha256(json.dumps(strip(json.load(open(p, encoding="utf-8"))), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def main():
    before = {f: norm_hash(OUT / f) for f in FILES}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTORCH_ENABLE_MPS_FALLBACK="0", CUDA_VISIBLE_DEVICES="")
    t0 = time.time()
    for st in STEPS:
        t = time.time()
        r = subprocess.run([sys.executable, str(HERE / st)], cwd=REPO, env=env, capture_output=True, text=True)
        tail = [x for x in (r.stdout + r.stderr).splitlines() if x.startswith("[b")][-1:]
        print(f"{st}: mã thoát {r.returncode} [{time.time() - t:.0f}s] {tail[0] if tail else ''}", flush=True)
        (OUT / ("log_" + re.search(r"_(\d\d)_", st).group(1) + ".txt")).write_text("\n".join(x for x in (r.stdout + r.stderr).splitlines() if not x.startswith("objc[")), encoding="utf-8")
        if r.returncode:
            sys.exit(r.returncode)
    after = {f: norm_hash(OUT / f) for f in FILES}
    for f in FILES:
        print(f"{f}: {before[f]} -> {after[f]} {'GIỐNG' if before[f] == after[f] else 'KHÁC (lần đầu hoặc nhãn đã đổi)'}")
    print(f"tổng [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
