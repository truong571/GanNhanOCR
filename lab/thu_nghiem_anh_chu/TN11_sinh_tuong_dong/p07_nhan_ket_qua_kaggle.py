"""TN11 p07 — nhận tn11_ket_qua.zip từ Kaggle, chấm bằng p02 --stage eval (encoder + nhãn người CHỈ trên máy), in bảng. 0 API.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p07_nhan_ket_qua_kaggle.py ~/Downloads/tn11_ket_qua.zip
Giải nén vào measure_out/_tn11/kaggle_out/, nối vào p02_<sách>/kg_<mô hình>/s<j>, chấm từng mô hình (base / ft1000 / ft3000 …),
ghi measure_out/_tn11/p07_ket_qua.json.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
T11 = REPO / "measure_out" / "_tn11"
KO = T11 / "kaggle_out"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("cần đường dẫn tn11_ket_qua.zip")
    with zipfile.ZipFile(sys.argv[1]) as z:
        z.extractall(KO)
    res = {}
    for bdir in sorted(p for p in KO.iterdir() if p.is_dir()):
        b = bdir.name
        if not (T11 / f"p02_{b}" / "plan.pkl").exists():
            continue
        models = sorted({m.group(1) for d in bdir.iterdir() if (m := re.match(r"(.+)_s(\d+)$", d.name))})
        res[b] = {"train_log": json.loads((bdir / "train_log.json").read_text()) if (bdir / "train_log.json").exists() else None}
        for mdl in models:
            sub = f"kg_{mdl}"
            for d in bdir.glob(f"{mdl}_s*"):
                link = T11 / f"p02_{b}" / sub / f"s{d.name.rsplit('_s', 1)[1]}"
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    os.symlink(d, link)
            subprocess.run([sys.executable, str(Path(__file__).parent / "p02_fd_phong_cach_sach.py"), "--book", b, "--stage", "eval",
                            "--gen-sub", sub], check=True, capture_output=True)
            r = json.loads((T11 / f"p02_{b}" / f"ket_qua_{sub}.json").read_text())
            res[b][mdl] = r
            t = r["top1_trong_top_k"]
            gen = {k: v for k, v in t.items() if k.startswith("f_gen_s")}
            print(f"{b} {mdl:8s} f_fd {t['f_fd']:5.1f} | " + " ".join(f"{k[6:]} {v:5.1f}" for k, v in gen.items())
                  + f" | TB t0 {t['f_gen_t0']:5.1f} t2 {t['f_gen_t2']:5.1f} | vW {t['f_vW']:5.1f} vW+fd {t['vW+fd']:5.1f} "
                  f"vW+gen {max(t['vW+gen_t0'], t['vW+gen_t2']):5.1f} | người {t['f_hum']:5.1f} | chênh {r['chenh_lech_CI95']}", flush=True)
    (T11 / "p07_ket_qua.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
