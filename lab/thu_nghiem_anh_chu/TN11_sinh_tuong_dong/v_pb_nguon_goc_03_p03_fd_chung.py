"""v_pb_nguon_goc_03 (04/10) — PHẢN BIỆN: baseline 2 của p03 ("FD phong cách chung STT2") có thật sự là FD không?

Kiểm bằng ĐỌC DỮ LIỆU (không suy đoán):
  1. ket_qua_p03.json gốc: từng hàng có cos/margin/top1 của 'font' và 'fd' trùng nhau không.
  2. Với mọi chữ ứng viên của p03: đường dẫn p03 dùng (p03:168) có tồn tại không, kích thước, đầu tệp, cv2.imread có None không
     (=> nhánh rơi về phông p03:169-175).
  3. Toàn kho gannhanocr-fd: số PNG, số tệp < 1 KB (con trỏ Git-LFS?), số tệp đọc được.
  4. Kiểm .gitattributes / .gitmodules để biết kho có phải LFS.
Ra: measure_out/_tn11/verify/pb_nguon_goc/p03_fd_chung.json
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
OUT.mkdir(parents=True, exist_ok=True)
J = json.loads((REPO / "measure_out/_tn11/p03_direct/ket_qua_p03.json").read_text(encoding="utf-8"))

res = {}
rows = J["details"]
same_cos = sum(1 for r in rows if r["cos_font"] == r["cos_fd"])
same_mar = sum(1 for r in rows if r["margin_font"] == r["margin_fd"])
same_top = sum(1 for r in rows if r["top1_font"] == r["top1_fd"])
res["1_json"] = {"n_hang": len(rows), "cos_font==cos_fd": same_cos, "margin_font==margin_fd": same_mar, "top1_font==top1_fd": same_top,
                 "metrics": J["metrics"]}

chars = sorted({c for r in rows for c in r["candidates"]})
det = []
for c in chars:
    p = REPO / "gannhanocr-fd" / f"{ord(c):02X}"[:2] / f"U+{ord(c):04X}.png"   # đúng biểu thức p03:168
    d = {"char": c, "cp": f"U+{ord(c):04X}", "path": str(p.relative_to(REPO)), "exists": p.exists()}
    if p.exists():
        d["size"] = p.stat().st_size
        d["head"] = open(p, "rb").read(24).decode("latin1")
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        d["imread_None"] = im is None
    det.append(d)
res["2_chu_ung_vien"] = {"n_chu": len(chars), "n_ton_tai": sum(d["exists"] for d in det),
                         "n_imread_None_trong_so_ton_tai": sum(d.get("imread_None", False) for d in det),
                         "n_roi_ve_phong": sum((not d["exists"]) or d.get("imread_None", False) for d in det),
                         "chi_tiet_mau": det[:4], "tep_khong_phai_LFS": [d["cp"] for d in det if d.get("exists") and not d["head"].startswith("version https://git-lfs")]}

tot = small = readable = 0
big_nonlfs = []
for root, _, fs in os.walk(REPO / "gannhanocr-fd"):
    for fn in fs:
        if not fn.endswith(".png"):
            continue
        tot += 1
        p = Path(root) / fn
        sz = p.stat().st_size
        if sz < 1024:
            small += 1
        else:
            readable += 1
            if len(big_nonlfs) < 3:
                big_nonlfs.append(str(p.relative_to(REPO)))
res["3_kho_gannhanocr_fd"] = {"tong_png": tot, "duoi_1KB": small, "tu_1KB": readable, "vi_du_tu_1KB": big_nonlfs}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO, timeout=60).stdout.strip()[:400]
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


res["4_git"] = {"gitattributes_lfs": sh("grep -n 'lfs' .gitattributes | head -5"), "gitmodules": sh("cat .gitmodules 2>/dev/null | head -12"),
                "git_lfs_cli": sh("command -v git-lfs || echo 'khong co git-lfs'")}
(OUT / "p03_fd_chung.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "2_chu_ung_vien"}, ensure_ascii=False)[:1800])
print(json.dumps({k: v for k, v in res["2_chu_ung_vien"].items() if k != "chi_tiet_mau"}, ensure_ascii=False))
print(json.dumps(res["2_chu_ung_vien"]["chi_tiet_mau"][:2], ensure_ascii=False))
