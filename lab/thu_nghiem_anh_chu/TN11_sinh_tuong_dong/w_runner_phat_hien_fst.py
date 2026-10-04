"""TN11 "full" — hướng RUNNER/KAGGLE: kiểm trên MÔ HÌNH THẬT (CPU) rằng
  (1) cổng assert_standard_path() BẮT được lỗi use_fst của wrapper gốc (core/ranking/fontdiffusion_gen.py) lúc nạp mô hình và cho QUA bản sửa;
  (2) bộ sinh RealGen (trong bộ nhớ, hạt giống theo lô) cho ảnh GIỐNG HỆT wrapper chuẩn (fd_wrapper_fix) cùng hạt giống — tức khung mới không đổi kết quả sinh.
Không GPU/MPS (device cpu, PYTORCH_ENABLE_MPS_FALLBACK=0), không API. Chỉ ĐỌC repo; ghi vào measure_out/_tn11/full/runner/fst/ (và đổi cwd vào đó
để các tệp .log của font_diffusion không rơi vào repo).

    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_runner_phat_hien_fst.py [--keep-going]
Ra: measure_out/_tn11/full/runner/phat_hien_fst.json
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "full" / "runner"
FST_DIR = OUT / "fst"
FST_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(FST_DIR)                                  # .log của font_diffusion (setup_logger/FileHandler) sẽ nằm ở đây
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "font_diffusion"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import w_runner_khung_chay as K  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
CK = REPO / "font_diffusion" / "ckpt" / "PROD"
FONT = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
STYLE = REPO / "measure_out" / "_tn11" / "p02_B34" / "style_0.png"
R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "thiet_bi": "cpu", "torch": torch.__version__, "invariants": {}}


def inv(name, ok, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:400], flush=True)


def pick_chars(n=2):
    from fontTools.ttLib import TTFont
    f = TTFont(str(FONT), lazy=True); cm = set()
    for st in f["cmap"].tables:
        cm |= set(st.cmap)
    pref = ["孟", "恒", "固", "南", "罪"]
    return [c for c in pref if ord(c) in cm][:n]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    chars = pick_chars(2)
    R["chu_thu"] = chars
    task = dict(book="B34", model="base", style=0, id="B34/base_s0/00000")

    # ---------------- (1b) bản sửa: RealGen qua cổng
    t0 = time.time()
    rg = K.RealGen(REPO, CK, {"NomNaTong": str(FONT)}, "cpu", batch=2, cache_dir=FST_DIR / "cache_fixed")
    R["nap_ban_sua_giay"] = round(time.time() - t0, 1)
    inv("F1.ban_sua_qua_cong", rg.guard["ok"] and rg.guard["model_class"] == "FontDiffuserModelDPM", dict(guard=rg.guard, tham_so_M=round(n_params(rg.g.pipe.model) / 1e6, 1)))

    # ---------------- (2) tương đương với wrapper chuẩn cùng hạt giống
    S = 12345
    torch.manual_seed(S)
    rg.g.cache_dir = FST_DIR / "ref"
    t0 = time.time()
    rg.g.generate(chars, str(STYLE), style_name="ref")                         # đường wrapper (fd_wrapper_fix), lô 2
    t_ref = time.time() - t0
    t0 = time.time()
    imgs, sk = rg.generate(task, chars, ["NomNaTong"] * len(chars), STYLE, S)  # đường RealGen
    t_new = time.time() - t0
    diffs = {}
    for c in chars:
        a = np.asarray(Image.open(FST_DIR / "ref" / "ref" / f"U+{ord(c):04X}.png").convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(io.BytesIO(imgs[c])).convert("RGB"), dtype=np.int16)
        diffs[c] = dict(shape=list(a.shape), max_abs_diff=int(np.abs(a - b).max()), mean_ink=round(float(255 - a.mean()), 2))
    inv("F2.RealGen_giong_wrapper_cung_hat_giong", not sk and all(v["max_abs_diff"] == 0 for v in diffs.values()),
        dict(diffs=diffs, giay_wrapper=round(t_ref, 1), giay_RealGen=round(t_new, 1)))
    # hạt giống tất định: sinh lại cho đúng byte
    imgs2, _ = rg.generate(task, chars, ["NomNaTong"] * len(chars), STYLE, S)
    inv("F3.sinh_lai_cung_hat_giong_dung_byte", imgs2 == imgs, dict(bytes=[len(v) for v in imgs.values()]))
    imgs3, _ = rg.generate(task, chars, ["NomNaTong"] * len(chars), STYLE, S + 1)
    inv("F3b.doi_hat_giong_doi_anh", imgs3 != imgs, None)

    # ---------------- (1a) bản lỗi: gắn lại đường nạp của wrapper gốc vào RealGen -> cổng phải ném FstGuardError
    import fd_wrapper_fix as F
    from core.ranking.fontdiffusion_gen import FontDiffusionGenerator as Orig
    saved = F.FixedFontDiffusionGenerator._load_pipeline
    F.FixedFontDiffusionGenerator._load_pipeline = Orig._load_pipeline         # CHỈ trong tiến trình thử này (không sửa tệp nào)
    t0 = time.time()
    try:
        K.RealGen(REPO, CK, {"NomNaTong": str(FONT)}, "cpu", batch=2, cache_dir=FST_DIR / "cache_buggy")
        raised, msg = False, ""
    except K.FstGuardError as e:
        raised, msg = True, str(e)
    finally:
        F.FixedFontDiffusionGenerator._load_pipeline = saved
    R["nap_ban_loi_giay"] = round(time.time() - t0, 1)
    inv("F4.ban_loi_bi_cong_bat_ngay_luc_nap", raised and "FontDiffuserModelDPMWithFST" in msg and "not found" in msg, dict(thong_bao=msg[:700]))

    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]))
    (OUT / "phat_hien_fst.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
