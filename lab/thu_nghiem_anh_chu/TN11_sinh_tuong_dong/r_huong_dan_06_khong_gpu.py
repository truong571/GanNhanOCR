"""r_huong_dan_06_khong_gpu.py — REVIEW hướng "huong_dan" (6/6): lệnh trong hướng dẫn KHÔNG truyền --device. Nếu phiên Save & Run All không có GPU
(quên chọn Accelerator, tài khoản chưa xác minh điện thoại -> hộp chọn hiện "Requires phone verification", Kaggle đổi cấu hình...), bộ chạy làm gì?
Chạy CHÍNH `run` (đường mô hình thật, thiết bị không phải CUDA — máy này không có CUDA) trên một gói NHỎ do script này dựng trong thư mục review (2 cuốn x 2 chữ, mã/ckpt/phông/ảnh phong cách liên kết tới gói thật, chỉ ĐỌC):
  (a) như hướng dẫn: không --device           -> kỳ vọng: im lặng chạy bằng CPU, mã thoát 0, "complete"
  (b) đề xuất:       --device cuda            -> kỳ vọng: dừng nhanh, traceback trong log, không có ảnh
  (c) phông dự phòng: 1 chữ cho MỖI trong 6 phông của kế hoạch thật (gán phông thật), sinh bằng mô hình thật trên CPU -> không lỗi, ảnh không trống
0 GPU, 0 API, 0 mạng. Thư mục tạm (≈ 440 MB chép mã+ckpt) nằm trong thư mục review và bị xoá khi xong.

    PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_huong_dan_06_khong_gpu.py
Ra: measure_out/_tn11/full/review/huong_dan/06_khong_gpu.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib
import importlib.util

import numpy as np
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL = REPO / "measure_out" / "_tn11" / "full"
OUT = FULL / "review" / "huong_dan"
W = OUT / "nogpu"
PACK_REAL = FULL / "kaggle_pack_20261004"
PACK_TINY = W / "pack"
RUNNER = HERE / "w_runner_khung_chay.py"

R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "invariants": {}}


def inv(name, ok, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:400], flush=True)


def run(tag, extra, pack=None):
    out, tmp = W / f"{tag}_out", W / "tmp"
    cmd = [sys.executable, str(RUNNER), "run", "--pack", str(pack or PACK_TINY), "--out", str(out), "--tmp", str(tmp), "--workers", "1", "--budget-hours", "1", "--safety-s", "0",
           "--gen-bs", "2", "--est-s", "60", *extra]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTORCH_ENABLE_MPS_FALLBACK="0"))
    st_p = out / "status" / "status.json"
    st = json.loads(st_p.read_text(encoding="utf-8")) if st_p.exists() else None
    env_p = out / "status" / "env.json"
    env = json.loads(env_p.read_text(encoding="utf-8")) if env_p.exists() else {}
    log = (p.stdout + "\n" + p.stderr)
    n_tar = len(list((out / "shards").rglob("*.tar"))) if (out / "shards").exists() else 0
    loading = [ln.strip() for ln in log.splitlines() if "Loading FontDiffusion pipeline" in ln]
    tb = [ln for ln in log.splitlines() if ln.startswith(("RuntimeError", "AssertionError", "ValueError", "AttributeError", "torch.", "Traceback"))]
    res = dict(lenh=" ".join(["run", *extra]) or "run", exit=p.returncode, giay=round(time.time() - t0, 1), tar=n_tar, status={k: (st or {}).get(k) for k in ("complete", "worker_exit_codes", "worker_stop", "stop_reason", "tasks_done")},
               gpus_trong_env_json=env.get("gpus"), dong_log_thiet_bi=loading[:2], dong_loi=tb[-3:], duoi_log=log.strip()[-380:])
    return res


def build_tiny_pack(K):
    """Gói nhỏ: kế hoạch 2 cuốn (B34, B18) x 2 chữ phổ biến (孟恒 / 固南, đều có trong NomNaTong), lô 2; mã, ckpt, phông, ảnh phong cách là LIÊN KẾT tới gói thật."""
    from fontTools.ttLib import TTFont
    f = TTFont(str(PACK_REAL / "code" / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"), lazy=True)
    cm = set()
    for tb in f["cmap"].tables:
        cm |= set(tb.cmap)
    plan = K.make_plan({"B34": dict(chars=list("孟恒"), styles=1), "B18": dict(chars=list("固南"), styles=1)}, [("base", 0)], 2, {"NomNaTong": cm})
    PACK_TINY.mkdir(parents=True)
    (PACK_TINY / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (PACK_TINY / "fonts.json").write_text(json.dumps({"NomNaTong": "NomNaTong-Regular.ttf"}), encoding="utf-8")
    real_man = json.loads((PACK_REAL / "manifest.json").read_text(encoding="utf-8"))
    (PACK_TINY / "manifest.json").write_text(json.dumps(dict(tool=K.TOOL, plan_sha=plan["sha"], ckpt_sha256=real_man["ckpt_sha256"], books={})), encoding="utf-8")
    (PACK_TINY / "ckpt").mkdir()
    for fn in real_man["ckpt_sha256"]:
        os.symlink(PACK_REAL / "ckpt" / fn, PACK_TINY / "ckpt" / fn)
    os.symlink(PACK_REAL / "code", PACK_TINY / "code")
    for b in ("B34", "B18"):
        (PACK_TINY / "data" / b).mkdir(parents=True)
        shutil.copyfile(PACK_REAL / "data" / b / "style_0.png", PACK_TINY / "data" / b / "style_0.png")
    return plan


def build_font_pack(K):
    """Gói nhỏ thứ hai: 1 chữ cho mỗi phông (gán phông THẬT lấy từ plan.json của gói), cuốn B34, một lô 6 chữ."""
    from fontTools.ttLib import TTFont
    real = json.loads((PACK_REAL / "plan.json").read_text(encoding="utf-8"))
    fj = json.loads((PACK_REAL / "fonts.json").read_text(encoding="utf-8"))
    first = {}
    for b in ["B34", *[x for x in real["books"] if x != "B34"]]:
        v = real["books"][b]
        for c, i in zip(v["chars"], v["font_idx"]):
            first.setdefault(real["font_names"][i], c)
    cms = {}
    for name in real["font_names"]:
        f = TTFont(str(PACK_REAL / "code" / "font_diffusion" / "fonts" / fj[name]), lazy=True)
        cm = set()
        for tb in f["cmap"].tables:
            cm |= set(tb.cmap)
        cms[name] = cm
    chars = [first[n] for n in real["font_names"] if n in first]
    plan = K.make_plan({"B34": dict(chars=chars, styles=1, font_of={first[n]: n for n in first})}, [("base", 0)], 6, cms)
    pk = W / "pack_phong"
    pk.mkdir(parents=True)
    (pk / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(PACK_REAL / "fonts.json", pk / "fonts.json")
    real_man = json.loads((PACK_REAL / "manifest.json").read_text(encoding="utf-8"))
    (pk / "manifest.json").write_text(json.dumps(dict(tool=K.TOOL, plan_sha=plan["sha"], ckpt_sha256=real_man["ckpt_sha256"], books={})), encoding="utf-8")
    (pk / "ckpt").mkdir()
    for fn in real_man["ckpt_sha256"]:
        os.symlink(PACK_REAL / "ckpt" / fn, pk / "ckpt" / fn)
    os.symlink(PACK_REAL / "code", pk / "code")
    (pk / "data" / "B34").mkdir(parents=True)
    shutil.copyfile(PACK_REAL / "data" / "B34" / "style_0.png", pk / "data" / "B34" / "style_0.png")
    return pk, plan, {n: first[n] for n in first}


def main():
    miss = []
    spec = importlib.util.spec_from_file_location("K", RUNNER)
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    for mod in K.PIP:
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            miss.append(mod)
    if miss or not PACK_REAL.exists():
        print("BỎ QUA: thiếu mô-đun (sẽ khiến ensure_deps pip install vào .venv) hoặc thiếu gói thật:", miss, PACK_REAL.exists())
        R["bo_qua"] = dict(thieu=miss, goi=PACK_REAL.exists())
        (OUT / "06_khong_gpu.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    import torch
    R["may_nay"] = dict(cuda=torch.cuda.is_available(), torch=torch.__version__, so_gpu=torch.cuda.device_count())
    inv("N0.may_nay_khong_co_CUDA_giong_phien_Kaggle_khong_bat_Accelerator", not torch.cuda.is_available(), R["may_nay"])
    shutil.rmtree(W, ignore_errors=True)
    W.mkdir(parents=True)
    build_tiny_pack(K)
    a = run("a_khong_device", [])
    R["a_nhu_huong_dan_khong_device"] = a
    inv("N1.khong_--device:_im_lang_chay_bang_CPU_hoan_tat_ma_thoat_0_(ca_Kaggle_se_'thanh_cong'_nhung_tra_gia_mot_ca_CPU)", a["exit"] == 0 and a["status"]["complete"] and a["tar"] == 2 and a["gpus_trong_env_json"] == [],
        dict(exit=a["exit"], tar=a["tar"], stop_reason=a["status"]["stop_reason"], gpus=a["gpus_trong_env_json"], dong_log=a["dong_log_thiet_bi"]))
    b = run("b_device_cuda", ["--device", "cuda"])
    R["b_de_xuat_device_cuda"] = b
    inv("N2.--device_cuda:_dung_nhanh_khong_tao_tar_stop_reason_crash_or_error_va_co_loi_trong_log", b["tar"] == 0 and b["status"]["stop_reason"] == "crash_or_error" and bool(b["dong_loi"]), dict(exit=b["exit"], tar=b["tar"], stop_reason=b["status"]["stop_reason"], giay=b["giay"], dong_loi=b["dong_loi"]))
    inv("N3.--device_cuda_khong_pha_thanh_cong_gia:_run_van_exit_0_nen_notebook_hien_'xong'_trong_khi_0_anh", b["exit"] == 0 and b["tar"] == 0, dict(exit=b["exit"], tar=b["tar"]))

    # (c) 6 phông thật
    import io, tarfile
    from PIL import Image
    pk, plan2, per_font = build_font_pack(K)
    c = run("c_6_phong", ["--device", "cpu"], pack=pk)
    ink = {}
    for tp in (W / "c_6_phong_out" / "shards").rglob("*.tar"):
        with tarfile.open(tp) as tf:
            for m in tf.getmembers():
                if m.name.startswith("U+"):
                    im = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("L")
                    a = np.asarray(im, dtype=np.float32)
                    ink[m.name] = round(float((255 - a).mean() / 255 * 100), 1)
    ph_of = {f"U+{ord(ch):04X}.png": n for n, ch in per_font.items()}
    R["c_6_phong_thuc_su"] = dict(run=c, chu_theo_phong=per_font, ink_pct={ph_of.get(k, k): v for k, v in ink.items()})
    inv("N4.6_phong_cua_ke_hoach_thuc_deu_sinh_duoc_anh_bang_mo_hinh_that_tren_CPU_khong_bo_qua_khong_trong", c["exit"] == 0 and c["status"]["complete"] and len(ink) == len(per_font) == 6 and min(ink.values()) > 3.0,
        dict(tar=c["tar"], so_anh=len(ink), ink_pct=R["c_6_phong_thuc_su"]["ink_pct"], stop_reason=c["status"]["stop_reason"]))
    shutil.rmtree(W / "tmp", ignore_errors=True)
    shutil.rmtree(W / "pack", ignore_errors=True)
    shutil.rmtree(W / "pack_phong", ignore_errors=True)
    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]),
                     fail_ten=[k for k, v in R["invariants"].items() if not v["ok"]])
    (OUT / "06_khong_gpu.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
