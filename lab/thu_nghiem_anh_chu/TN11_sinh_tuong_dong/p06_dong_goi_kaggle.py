"""TN11 p06 — đóng gói dữ liệu + mã cho chạy TN11 trên Kaggle (tn11_kaggle.py). 0 API, không nhãn người trong gói.

Gói = measure_out/_tn11/kaggle/tn11_kaggle_data.zip:
  tn11_manifest.json · tn11_kaggle.py · code/core/ranking/fontdiffusion_gen.py · code/font_diffusion/{src,inference,fonts/NomNaTong}
  · ckpt/*.safetensors (checkpoint FontDiffuser của dự án, kèm sha256) · data/<sách>/{pairs.npz (crop ô neo tự động + nhãn kim),
  chars.txt (chữ ứng viên cần sinh), style_0.png, style_1.png}
Cần có trước: p02 --stage plan và p04 --stage data cho từng sách.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p06_dong_goi_kaggle.py [--books B34 B18]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
T11 = REPO / "measure_out" / "_tn11"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", default=["B34", "B18"])
    a = ap.parse_args(argv)
    out = T11 / "kaggle" / "tn11_kaggle_data.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    ck = REPO / "font_diffusion" / "ckpt" / "PROD"
    man = dict(tao="TN11 p06", ckpt_sha256={}, books={})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__).parent / "kaggle" / "tn11_kaggle.py", "tn11_kaggle.py")
        z.writestr("code/core/__init__.py", ""); z.writestr("code/core/ranking/__init__.py", "")
        z.write(REPO / "core/ranking/fontdiffusion_gen.py", "code/core/ranking/fontdiffusion_gen.py")
        z.write(Path(__file__).parent / "fd_wrapper_fix.py", "code/fd_wrapper_fix.py")     # use_fst=False (lỗi gốc của wrapper)
        fd = REPO / "font_diffusion"
        for sub in ("src", "inference"):
            for f in sorted((fd / sub).rglob("*.py")):
                z.write(f, f"code/font_diffusion/{f.relative_to(fd)}")
        z.write(fd / "fonts/NomNaTong-Regular.ttf", "code/font_diffusion/fonts/NomNaTong-Regular.ttf")
        for f in sorted(ck.glob("*.safetensors")):
            man["ckpt_sha256"][f.name] = sha256(f)
            z.write(f, f"ckpt/{f.name}", compress_type=zipfile.ZIP_STORED)
        for b in a.books:
            P = pd.read_pickle(T11 / f"p02_{b}" / "plan.pkl")
            z.write(T11 / f"p04_{b}" / "pairs.npz", f"data/{b}/pairs.npz", compress_type=zipfile.ZIP_STORED)
            z.writestr(f"data/{b}/chars.txt", "\n".join(P["chars"]))
            for j in range(len(P["styles"])):
                z.write(T11 / f"p02_{b}" / f"style_{j}.png", f"data/{b}/style_{j}.png")
            man["books"][b] = dict(chars=len(P["chars"]), styles=P["style_chars"])
        z.writestr("tn11_manifest.json", json.dumps(man, ensure_ascii=False, indent=1))
    print(f"[p06] {out} ({out.stat().st_size / 1e6:.0f} MB): {json.dumps(man['books'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
