"""r_goi_du_lieu_09_tong_hop.py — gộp bất biến PASS/FAIL của r_goi_du_lieu_01..08 (chỉ đọc JSON đầu ra của chính hướng này) -> 00_tong_hop.json.

Quy ước: một bất biến FAIL ở script 01–05, 07, 08 là LỖI của gói/dữ liệu; ở script 06 mỗi FAIL là một PHÁT HIỆN (kiểm được đặt để lộ khoảng hở của ca thử/bộ chạy/tài liệu).

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_09_tong_hop.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402

FILES = ["01_ro_nhan_nguoi", "02_anh_phong_cach", "02b_dung_lai_crop_chuan", "03_chu_va_phong", "04_toan_ven_goi", "05_cascade_ttf2im", "06_phu_ca_thu_va_bo_cuc", "07_chay_thu_cpu_cascade", "08_kiem_bang_PUA"]


def main():
    out = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"), script={}, tong=dict(bat_bien=0, dat=0))
    for n in FILES:
        p = Lb.OUT / f"{n}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        bb = d["bat_bien"]
        rows = d.get("bat_bien_chi_tiet", [])
        out["script"][n] = dict(tong=bb["tong"], dat=bb["dat"], rot=bb["rot"], tao_luc=d.get("tao_luc"), giay=d.get("giay"),
                                phat_hien_khong_phai_loi=("06" in n))
        out["tong"]["bat_bien"] += bb["tong"]; out["tong"]["dat"] += bb["dat"]
    lines = {n: f"{v['dat']}/{v['tong']}" + (f" rớt={v['rot']}" if v["rot"] else "") for n, v in out["script"].items()}
    out["tom_tat"] = lines
    Lb.jdump(out, "00_tong_hop.json")
    print(json.dumps(out["tom_tat"], ensure_ascii=False, indent=1))
    print("TỔNG", out["tong"])


if __name__ == "__main__":
    main()
