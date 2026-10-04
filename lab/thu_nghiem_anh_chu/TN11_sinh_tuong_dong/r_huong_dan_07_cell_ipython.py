"""r_huong_dan_07_cell_ipython.py — REVIEW hướng "huong_dan" (7/7): ô code Kaggle trong hướng dẫn (`!python $(find ...) run \\ ...`) chạy đúng không
trong IPython thật (cùng bộ chuyển `!` mà notebook Kaggle dùng)? Dựng một venv RIÊNG trong thư mục review (pip --no-cache-dir, không đụng .venv),
cài IPython, rồi chạy CHÍNH văn bản ô lấy từ tệp hướng dẫn với một "bộ chạy giả" chỉ in argv. Kiểm:
  (1) ô nguyên văn (sau khi thay <hf_user>) -> argv nhận đủ, `$(find ...)` và dấu \\ xuống dòng hoạt động
  (2) quên thay <hf_user> (còn dấu < >) -> lỗi shell, KHÔNG tới được bộ chạy
  (3) biến môi trường đặt trong ô Python (os.environ["HF_TOKEN"]) có tới tiến trình con `!python` không
  (4) biến thể %%bash
Cần mạng (PyPI) chỉ để cài IPython; 0 GPU/0 API.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_huong_dan_07_cell_ipython.py
Ra: measure_out/_tn11/full/review/huong_dan/07_cell_ipython.json (venv tạm ipy_venv/ bị xoá khi xong)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "full" / "review" / "huong_dan"
T = OUT / "ipy_test"
VENV = OUT / "ipy_venv"
GUIDE = REPO / "docs" / "KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md"
R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "invariants": {}}


def inv(name, ok, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:400], flush=True)


DRIVER = r'''
import sys, json
from IPython.core.interactiveshell import InteractiveShell
ip = InteractiveShell.instance()
cell = open(sys.argv[1], encoding="utf-8").read()
res = ip.run_cell(cell, store_history=False)
print("CELL_OK=" + str(res.success))
'''


def run_cell(py: Path, cell: str, tag: str, env_extra=None):
    cf = T / f"cell_{tag}.txt"
    cf.write_text(cell, encoding="utf-8")
    df = T / "driver.py"
    df.write_text(DRIVER, encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PATH=f"{VENV / 'bin'}:{os.environ['PATH']}")
    env.pop("HF_TOKEN", None)
    if env_extra:
        env.update(env_extra)
    p = subprocess.run([str(py), str(df), str(cf)], capture_output=True, text=True, env=env, cwd=str(T))
    out = p.stdout + p.stderr
    m = re.search(r"ARGV=(\[.*\])", out)
    return dict(exit=p.returncode, argv=json.loads(m.group(1)) if m else None, hf_token=("HF_TOKEN_SET=True" in out), cell_ok=("CELL_OK=True" in out), cuoi=out.strip()[-300:])


def main():
    g = GUIDE.read_text(encoding="utf-8")
    shutil.rmtree(T, ignore_errors=True)
    T.mkdir(parents=True)
    # --- venv + IPython
    if not (VENV / "bin" / "python").exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True, capture_output=True)
    pr = subprocess.run([str(VENV / "bin" / "python"), "-m", "pip", "install", "--no-cache-dir", "-q", "--disable-pip-version-check", "ipython"], capture_output=True, text=True)
    py = VENV / "bin" / "python"
    ver = subprocess.run([str(py), "-c", "import IPython;print(IPython.__version__)"], capture_output=True, text=True).stdout.strip()
    R["ipython"] = dict(version=ver or None, pip_exit=pr.returncode, pip_loi=(pr.stderr or "")[-200:] if pr.returncode else "")
    if not ver:
        inv("I0.cai_duoc_IPython_trong_venv_rieng", False, R["ipython"])
        (OUT / "07_cell_ipython.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
        shutil.rmtree(VENV, ignore_errors=True)
        return
    inv("I0.cai_duoc_IPython_trong_venv_rieng", True, ver)

    # --- bộ chạy giả + thư mục /kaggle/input giả
    fake_in = T / "input" / "tn11-full-pack"
    fake_in.mkdir(parents=True)
    (fake_in / "w_runner_khung_chay.py").write_text('import sys, json, os\nprint("ARGV=" + json.dumps(sys.argv[1:], ensure_ascii=False))\nprint("HF_TOKEN_SET=" + str(bool(os.environ.get("HF_TOKEN"))))\n', encoding="utf-8")

    # --- ô nguyên văn trong hướng dẫn
    blocks, cur, inb = [], [], False
    for ln in g.split("\n"):
        if ln.strip().startswith("```"):
            if inb:
                blocks.append(cur); cur = []
            inb = not inb
            continue
        if inb:
            cur.append(ln)
    kb = [b for b in blocks if b and b[0].lstrip().startswith("!python")]
    R["so_o_kaggle_trong_huong_dan"] = len(kb)
    pilot = "\n".join(ln[3:] if ln.startswith("   ") else ln for ln in kb[0])      # bỏ thụt 3 dấu cách của danh sách markdown (người dùng dán bản đã bỏ thụt)
    R["o_thu_nguyen_van"] = pilot
    cell_fake = pilot.replace("<hf_user>", "toi").replace("/kaggle/input", str(T / "input"))
    r1 = run_cell(py, cell_fake, "pilot")
    R["ket_qua_o_nguyen_van"] = r1
    want = ["run", "--out", "/kaggle/tmp/tn11_full", "--tmp", "/kaggle/tmp/tn11", "--bundle-dir", "/kaggle/working/tn11_bundles", "--remote", "hf:toi/gannhanocr-tn11-full", "--max-tasks", "2", "--budget-hours", "0.5"]
    inv("I1.o_nguyen_van_trong_IPython:_find_va_dau_\\_xuong_dong_hoat_dong_argv_day_du", r1["cell_ok"] and r1["argv"] == want, dict(argv=r1["argv"]))

    # --- dán nguyên văn từ tệp .md thô (còn thụt 3 dấu cách của danh sách markdown)
    raw = "\n".join(kb[0]).replace("<hf_user>", "toi").replace("/kaggle/input", str(T / "input"))
    r1c = run_cell(py, raw, "pilot_con_thut")
    R["ket_qua_dan_tu_md_tho_con_thut_3_dau_cach"] = r1c
    inv("I1c.dan_tu_md_tho_(con_thut_3_dau_cach)_van_chay_dung_trong_IPython", r1c["argv"] == want, dict(argv=r1c["argv"]))

    # --- thụt đầu dòng 4 dấu cách ở dòng tiếp theo (đúng như hướng dẫn) đã nằm trong pilot; thêm biến thể KHÔNG có `\` cuối dòng để thấy hậu quả
    broken = re.sub(r" \\\n", "\n", cell_fake)
    r1b = run_cell(py, broken, "pilot_thieu_backslash")
    R["ket_qua_thieu_dau_backslash"] = r1b
    inv("I1b.neu_nguoi_dung_xoa_dau_\\_cuoi_dong_thi_chi_dong_dau_chay_va_cac_dong_sau_thanh_lenh_rieng_gay_loi", r1b["argv"] == ["run"] or r1b["argv"] is None or r1b["argv"] != want, dict(argv=r1b["argv"], cuoi=r1b["cuoi"][-160:]))

    # --- quên thay <hf_user>
    forget = pilot.replace("/kaggle/input", str(T / "input"))
    r2 = run_cell(py, forget, "quen_thay")
    R["ket_qua_quen_thay_hf_user"] = r2
    inv("I2.quen_thay_<hf_user>:_shell_bao_loi_va_bo_chay_KHONG_duoc_goi", r2["argv"] is None, dict(cuoi=r2["cuoi"][-220:]))

    # --- biến môi trường từ ô Python tới tiến trình con
    pre = 'import os\nos.environ["HF_TOKEN"] = "abc"\n'
    r3 = run_cell(py, f'{pre}\n!python {fake_in}/w_runner_khung_chay.py run --x 1', "env")
    R["ket_qua_env"] = r3
    inv("I3.os.environ[HF_TOKEN]_dat_trong_o_Python_toi_duoc_tien_trinh_con_`!python`", r3["hf_token"], dict(argv=r3["argv"]))
    r3b = run_cell(py, f"!python {fake_in}/w_runner_khung_chay.py run --x 1", "env_khong_dat")
    inv("I3b.doi_chung:_khong_dat_thi_tien_trinh_con_khong_co_HF_TOKEN", not r3b["hf_token"], dict(argv=r3b["argv"]))

    # --- biến thể %%bash
    bash_cell = "%%bash\n" + cell_fake.replace("!python $(find", "python $(find", 1)
    r4 = run_cell(py, bash_cell, "bash")
    R["ket_qua_bash"] = r4
    inv("I4.bien_the_%%bash_cung_cho_argv_day_du", r4["argv"] == want, dict(argv=r4["argv"], cuoi=r4["cuoi"][-120:]))

    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]),
                     fail_ten=[k for k, v in R["invariants"].items() if not v["ok"]])
    (OUT / "07_cell_ipython.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    shutil.rmtree(VENV, ignore_errors=True)
    shutil.rmtree(T, ignore_errors=True)
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
