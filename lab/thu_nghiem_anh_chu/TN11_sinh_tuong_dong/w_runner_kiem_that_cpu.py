"""TN11 "full" — hướng RUNNER/KAGGLE: chạy THẬT đầu-cuối trên CPU (không GPU/MPS, không mạng) con đường mô hình thật của w_runner_khung_chay.py:
pack (có ckpt, kiểm băm) -> run (setup_real chép code+ckpt sang --tmp, cổng use_fst, RealGen) -> tar lô -> bundle -> validate -> unbundle.
Kế hoạch cực nhỏ (2 cuốn × 2 chữ × 1 phong cách) để CPU làm trong vài phút. Chỉ ĐỌC repo; ghi vào measure_out/_tn11/full/runner/real_cpu/.

    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_runner_kiem_that_cpu.py
Ra: measure_out/_tn11/full/runner/kiem_that_cpu.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "full" / "runner"
W = OUT / "real_cpu"
RUNNER = HERE / "w_runner_khung_chay.py"
ENV = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0")
R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "invariants": {}}


def inv(name, ok, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:400], flush=True)


def sh(args, cwd):
    p = subprocess.run([sys.executable, str(RUNNER), *map(str, args)], capture_output=True, text=True, cwd=cwd, env=ENV)
    return p.returncode, p.stdout + p.stderr


def main():
    if W.exists():
        shutil.rmtree(W)
    (W / "work").mkdir(parents=True)
    chain = {"NomNaTong": "font_diffusion/fonts/NomNaTong-Regular.ttf", "HanNomA": "font_diffusion/fonts/HAN NOM A.ttf", "HanNomB": "font_diffusion/fonts/HAN NOM B.ttf",
             "HanaMinA": "font_diffusion/fonts/HanaMinA.ttf", "HanaMinB": "font_diffusion/fonts/HanaMinB.ttf", "HanaMinC": "font_diffusion/fonts/HanaMinC.otf"}
    spec = {"fonts": chain, "books": {}}
    for b, chars in (("B34", "孟恒"), ("B18", "固南")):
        (W / f"chars_{b}.txt").write_text("\n".join(chars), encoding="utf-8")
        spec["books"][b] = dict(chars_file=str(W / f"chars_{b}.txt"), style_pngs=[str(REPO / "measure_out/_tn11" / f"p02_{b}" / "style_0.png")])
    (W / "spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    code, log = sh(["pack", "--spec", W / "spec.json", "--out", W / "pack", "--shard", "2", "--passes", "base:0", "--repo", REPO], W / "work")
    man = json.loads((W / "pack" / "manifest.json").read_text(encoding="utf-8")) if (W / "pack" / "manifest.json").exists() else {}
    inv("R1.pack_co_ckpt_va_kiem_goi", code == 0 and set(man.get("ckpt_sha256", {})) == {"unet.safetensors", "style_encoder.safetensors", "content_encoder.safetensors"}, dict(giay=round(time.time() - t0, 1), log=log.strip()[-250:]))
    t0 = time.time()
    code, log = sh(["run", "--pack", W / "pack", "--out", W / "out", "--tmp", W / "tmp", "--workers", "1", "--budget-hours", "1", "--safety-s", "0", "--gen-bs", "2",
                    "--device", "cpu", "--est-s", "60", "--bundle-dir", W / "bundles", "--bundle-every", "3600"], W / "work")
    st = json.loads((W / "out" / "status" / "status.json").read_text(encoding="utf-8")) if (W / "out" / "status" / "status.json").exists() else {}
    inv("R2.run_that_cpu_hoan_tat", code == 0 and st.get("complete") and st.get("tasks_done") == 2, dict(code=code, status=st, giay=round(time.time() - t0, 1), log=log.strip()[-300:] if code else ""))
    code, vlog = sh(["validate", "--pack", W / "pack", "--out", W / "out"], W / "work")
    try:
        vr = json.loads(vlog.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        vr = vlog[-300:]
    inv("R3.validate_sach_4_anh_khong_bo_qua", code == 0 and isinstance(vr, dict) and vr["images"] == 4 and not vr["skipped_by_reason"] and vr["n_errors"] == 0, vr)
    env = json.loads((W / "out" / "status" / "env.json").read_text(encoding="utf-8")) if (W / "out" / "status" / "env.json").exists() else {}
    inv("R4.ghi_moi_truong", env.get("torch") and env.get("diffusers") and env.get("gpus") == [], dict(torch=env.get("torch"), diffusers=env.get("diffusers"), python=env.get("python")))
    ixs = sorted((W / "bundles").glob("bundle_*.index.json"))
    tars = sorted((W / "bundles").glob("bundle_*.tar"))
    inv("R5.bundle_dau_ra", len(ixs) == 1 and len(tars) == 1 and len(json.loads(ixs[0].read_text())["ids"]) == 2, dict(files=[p.name for p in (W / "bundles").iterdir()]))
    # ảnh thật là PNG 96×96 RGB không rỗng
    from PIL import Image
    import io
    ok_img, info = True, []
    for tp in (W / "out" / "shards").rglob("*.tar"):
        with tarfile.open(tp) as tf:
            for m in tf.getmembers():
                if m.name.startswith("U+"):
                    im = Image.open(io.BytesIO(tf.extractfile(m).read()))
                    ink = 255 - sum(im.convert("L").getdata()) / (im.size[0] * im.size[1])
                    info.append((tp.parent.parent.name, m.name, im.size, im.mode, round(ink, 1)))
                    ok_img &= im.size == (96, 96) and im.mode == "RGB" and ink > 5
    inv("R6.anh_that_96x96_RGB_co_muc", ok_img and len(info) == 4, info)
    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]))
    (OUT / "kiem_that_cpu.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
