"""Tương thích phiên bản Python của bộ chạy trong GÓI (phần điều khiển chỉ cần thư viện chuẩn): biên dịch + chạy --stub trọn 410 việc bằng Python 3.10 (/opt/homebrew/bin/python3.10)
và so băm 410 tar với lần chạy bằng Python 3.14 (stub_out do script 01 tạo). Kaggle hiện chạy Python 3.13.15 (log_B34.txt của lần chạy thật trước) — nằm giữa 3.10 và 3.14.
Ra: .../review/chay_thu_dau_cuoi/11_py310.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
import os
import shutil
import subprocess

import r_chay_thu_dau_cuoi_lib as L

rep = L.Rep("11_py310")
P10 = "/opt/homebrew/bin/python3.10"
R, PACK = L.R, L.PACK
if not os.path.exists(P10):
    rep.data("bo_qua", "không có python3.10"); rep.save(); sys.exit(0)
src = f"import sys; [compile(open(f, encoding='utf-8').read(), f, 'exec') for f in sys.argv[1:]]; print(sys.version.split()[0])"
p = subprocess.run([P10, "-B", "-c", src, str(PACK / "w_runner_khung_chay.py"), str(PACK / "code" / "fd_wrapper_fix.py")], capture_output=True, text=True, env=L.env())
rep.inv("Y1.bien_dich_runner_va_wrapper_bang_python3.10", p.returncode == 0, dict(version=p.stdout.strip(), loi=p.stderr[-200:]))
out = R / "stub_py310"
if out.exists():
    shutil.rmtree(out)
p = subprocess.run([P10, "-B", str(PACK / "w_runner_khung_chay.py"), "run", "--stub", "--stub-ms", "0.2", "--workers", "2", "--pack", str(PACK), "--out", str(out)], cwd=R / "stub_cwd", env=L.env(), capture_output=True, text=True)
st = L.status_of(out)
ref = L.shard_hashes(R / "stub_out")
cur = L.shard_hashes(out)
rep.inv("Y2.stub_410_viec_python3.10_hoan_tat_va_tar_byte_giong_python3.14", p.returncode == 0 and st["complete"] and len(cur) == 410 and cur == ref, dict(code=p.returncode, complete=st["complete"], so_tar=len(cur), byte_giong=cur == ref))
rep.save()
