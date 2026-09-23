#!/usr/bin/env python
"""Sinh kaggle_train_i5_v2.ipynb (nbformat 4) — notebook "Run All" gọi train_kaggle.py trong bundle.
Chạy: .venv/bin/python lab/i5_detector_v2/build_notebook.py   (make_bundle.py copy notebook vào bundle/)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MD_INTRO = """# Detector v2 (CenterNet r34, thạch bản + STT) — huấn luyện trên Kaggle · Run All · reset-safe

**Trước khi chạy:** Settings → Accelerator **GPU T4** (x2 cũng được; P100 có thể lỗi với torch mới) · Persistence **Files only**
(giữ `/kaggle/working/last.pt` khi restart) · Add Input → dataset dựng từ `i5v2_bundle.zip` (`make_bundle.py`).
Internet chỉ cần nếu đẩy lên HF (`HF_REPO` + Secret `HF_TOKEN`).

Mỗi epoch: train cân bằng 50/50 thạch bản/STT (aug nền xám · otsu/stretch · kéo dọc ±10 % · nét · blur · crop cột) →
đánh giá trên **val page-disjoint**: ok50 / miss / extra / |dy| / **% tầng n==N** (hộp thô @0,15 & 0,2) / cắt thân chữ,
**STT F1** (guard ≥ v1 − 0,01), Chrestomathie (sách không train). Best theo ok50_litho qua guard; dừng sớm 4 epoch.
Kết quả: `/kaggle/working/{best.pt, best_litho.pt, last.pt, metrics.csv, report.md}`. **Reset?** Run All lại → tự resume từ `last.pt`."""

CELL_CFG = '''# ---- 1) Cấu hình — chỉ sửa ở đây -------------------------------------------------
EPOCHS   = 20        # ≈ 2–3 phút/epoch trên T4 ở img 1024 (20 epoch ≈ 1 giờ + eval)
BATCH    = 4         # hết RAM GPU -> 2
IMG      = 1024      # hoặc 1280 (chậm ~1,6×, batch 2)
LR       = 2e-4      # cosine + warmup 1 epoch
SEED     = 0
EVAL_CHRESTO = 20    # số trang Chrestomathie (held-out, không train) đo mỗi epoch; 0 = bỏ
HF_REPO  = "mdnt571/nom-char-det-v2"   # ""=local; vd "mdnt571/nom-char-det-v2": đẩy last/best/best_litho mỗi epoch + resume từ hub (cần Secret HF_TOKEN). "" = local
# GUARD STT (2026-09-23, vòng 7). Lần chạy 22/09 cho thấy v2 TỐT HƠN HẲN trên thạch bản
# (ok50 98,71 → 100,0 · % tầng n==N 94,7 → 98,7 · cắt thân chữ 11,32 → 0,22) nhưng STT F1@0,2
# tụt 0,8772 → 0,858–0,864 (quên miền), nên guard chặn và KHÔNG ckpt nào được ghi ra.
# Repo ĐÃ CÓ `books[].detector_ckpt` THEO SÁCH -> ckpt thạch bản không bao giờ chạy trên STT,
# guard chỉ cần khi một mô hình phải phục vụ cả hai miền. "none" = tắt guard (best.pt chọn
# theo riêng tiêu chí thạch bản); "0.01" = như cũ. DÙ THẾ NÀO trainer cũng LUÔN ghi thêm
# best_litho.pt (+ cảnh báo trong report.md), nên một lần chạy không bao giờ về tay trắng.
GUARD_STT = "none"   # "none" | số, vd "0.01"
EXTRA    = ""        # tham số thêm cho train_kaggle.py, vd "--patience 6 --p-gray 0.5"
SMOKE    = False     # True: chạy thử 4 trang/miền, 1 epoch (kiểm tra đường ống ~2 phút)'''

CELL_SETUP = '''# ---- 2) Tìm bundle, chép mã ra /kaggle/working, kiểm GPU ---------------------------
import os, sys, glob, shutil, subprocess
hits = sorted(glob.glob("/kaggle/input/**/train_kaggle.py", recursive=True)) or sorted(glob.glob("./**/train_kaggle.py", recursive=True))
assert hits, "Không thấy train_kaggle.py — Add Input: gắn dataset dựng từ i5v2_bundle.zip"
dirs = [os.path.dirname(h) for h in hits]
# DATA = thư mục CÓ ảnh (bundle_stats.json). SRC = thư mục mã MỚI NHẤT: nếu có input chỉ chứa
# mã (i5v2_code.zip, ~50 KB) thì ưu tiên nó, để sửa mã không phải đẩy lại 243 MB ảnh.
DATA = next((d for d in dirs if os.path.exists(os.path.join(d, "bundle_stats.json"))), dirs[0])
SRC  = next((d for d in dirs if not os.path.exists(os.path.join(d, "bundle_stats.json"))), DATA)
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work")
CODE = os.path.join(WORK, "i5v2_code"); os.makedirs(CODE, exist_ok=True)
shutil.copytree(os.path.join(SRC, "i5v2"), os.path.join(CODE, "i5v2"), dirs_exist_ok=True)
shutil.copy2(os.path.join(SRC, "train_kaggle.py"), os.path.join(CODE, "train_kaggle.py"))
for f in ("manifest_train.json", "manifest_val.json", "v1/detector_r34.best.pt", "bundle_stats.json"):
    assert os.path.exists(os.path.join(DATA, f)), f"bundle thiếu {f}"
tok = ""
if HF_REPO:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "huggingface_hub"], check=False)
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        tok = os.environ.get("HF_TOKEN", "")
    os.environ["HF_TOKEN"] = tok or ""
import torch, json
print("data :", DATA); print("mã   :", SRC, "(gói mã riêng)" if SRC != DATA else "(trong bundle)")
print("code :", CODE); print("out  :", WORK)
print("GPU  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (bật GPU T4!)", "| torch", torch.__version__)
print("bundle:", json.load(open(os.path.join(DATA, "bundle_stats.json")))["by_domain_split"])
print("HF   :", HF_REPO or "(local-only)", "| token", "có" if tok else "không")
print("resume:", "có last.pt → tiếp tục" if os.path.exists(os.path.join(WORK, "last.pt")) else "chưa có last.pt → từ đầu")'''

CELL_TRAIN = '''# ---- 3) Train — reset-safe (chạy lại cell này/Run All sau reset để tiếp tục) --------
cmd = [sys.executable, os.path.join(CODE, "train_kaggle.py"), "--data", DATA, "--out", WORK,
       "--epochs", str(EPOCHS), "--batch", str(BATCH), "--img", str(IMG), "--lr", str(LR), "--seed", str(SEED),
       "--eval-chresto", str(EVAL_CHRESTO), "--workers", "2"]
if HF_REPO: cmd += ["--hf-repo", HF_REPO]
if GUARD_STT: cmd += ["--guard-stt-f1", str(GUARD_STT)]
if SMOKE:   cmd += ["--smoke"]
if EXTRA:   cmd += EXTRA.split()
print("$", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)'''

CELL_RESULT = '''# ---- 4) Kết quả: bảng v1 vs v2 + theo epoch ------------------------------------------
import csv
print(open(os.path.join(WORK, "report.md"), encoding="utf-8").read())
rows = list(csv.DictReader(open(os.path.join(WORK, "metrics.csv"), encoding="utf-8")))
keys = ["epoch", "loss", "litho_ok50", "litho_tiers_eq_015", "litho_tiers_eq_020", "litho_cut", "stt_F1_020", "chresto_tiers_eq_015", "is_best"]
print(" | ".join(keys))
for r in rows:
    print(" | ".join(str(r.get(k, "")) for k in keys))
for nm in ("best.pt", "best_litho.pt"):
    bp = os.path.join(WORK, nm)
    if not os.path.exists(bp):
        print("\\nKHÔNG có", nm)
        continue
    d = torch.load(bp, map_location="cpu", weights_only=False)
    print("\\n" + nm + ": epoch", d["epoch"], "| img", d["img"], "| use_dcn", d["use_dcn"], "|", os.path.getsize(bp) // 2**20, "MB")
    print("val:", {k: d["val"].get(k) for k in ("litho_ok50", "litho_tiers_eq_015", "litho_cut", "stt_F1_020")})
    print("v1 :", {k: d["base_v1"].get(k) for k in ("litho_ok50", "litho_tiers_eq_015", "litho_cut", "stt_F1_020")})
    if d.get("warning"): print("⚠️", d["warning"])'''

MD_END = """### Xong
- Tải **`best.pt`** *hoặc* **`best_litho.pt`** (tab Output / `/kaggle/working/`) + `metrics.csv` + `report.md` về máy.
  `best_litho.pt` luôn được ghi (chọn theo RIÊNG tiêu chí thạch bản, không xét guard STT) — dùng nó khi guard chặn `best.pt`.
  ⚠️ ckpt chọn theo tiêu chí thạch bản CHỈ được khai qua `books[].detector_ckpt` cho sách thạch/mộc bản, KHÔNG đặt làm detector toàn cục (STT sẽ tụt).
- Ở repo: `lab/i5_detector_v2/apply_v2.sh <đường dẫn ckpt>` → đo `box_ref_eval` v1↔v2 trên ảnh gốc, build `--book all-new --suffix _v2`.
- Reset / hết giờ: **Run All** lại — cell (3) tự resume từ `last.pt` (Persistence Files only) hoặc từ HF nếu khai `HF_REPO`."""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True,
                                  "language": "python", "sourceType": "notebook"}},
          "cells": [cell("markdown", MD_INTRO), cell("code", CELL_CFG), cell("code", CELL_SETUP),
                    cell("code", CELL_TRAIN), cell("code", CELL_RESULT), cell("markdown", MD_END)]}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i}"
    out = HERE / "kaggle_train_i5_v2.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
