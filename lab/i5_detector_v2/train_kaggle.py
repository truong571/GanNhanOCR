#!/usr/bin/env python
"""Huấn luyện detector CenterNet v2 (thạch bản + STT) từ bundle — chạy được ở Kaggle lẫn máy.

Kaggle (notebook kaggle_train_i5_v2.ipynb gọi đúng lệnh này):
    python train_kaggle.py --data /kaggle/input/<dataset> --out /kaggle/working
Máy — chứng minh chạy (≤ 8 ảnh, 1 epoch, img 512, CPU vài phút):
    .venv/bin/python lab/i5_detector_v2/train_kaggle.py --data lab/i5_detector_v2/bundle --smoke --out <scratch>/smoke
Chỉ đánh giá 1 ckpt trên val bundle (mốc v1 / so v2), cùng mã với trainer:
    .venv/bin/python lab/i5_detector_v2/train_kaggle.py --data lab/i5_detector_v2/bundle --eval-only --ckpt <pt> [--out-json f]
Resume: có <out>/last.pt (hoặc --hf-repo có last.pt) → tự tiếp tục epoch còn thiếu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from i5v2.trainer import Cfg, eval_only, run   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="thư mục bundle (có manifest_*.json, images/, v1/)")
    ap.add_argument("--out", default="out", help="thư mục ghi best.pt/last.pt/metrics.csv/report.md")
    ap.add_argument("--init", default="", help="ckpt khởi tạo (mặc định <data>/v1/detector_r34.best.pt)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img", type=int, default=0, help="train: 1024 mặc định (T4 ~2–3 phút/epoch) hoặc 1280; eval-only: 0 = theo ckpt")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.997)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = mặc định)")
    ap.add_argument("--device", default="", help="cuda | cpu | mps (mặc định tự chọn)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="0 = 2 × min(số trang miền)")
    ap.add_argument("--limit", type=int, default=0, help="số trang mỗi miền (thử)")
    ap.add_argument("--eval-pages", type=int, default=0, help="số trang val mỗi miền (0 = tất cả)")
    ap.add_argument("--eval-chresto", type=int, default=20, help="số trang Chrestomathie held-out đo mỗi epoch")
    ap.add_argument("--stt-tol", type=float, default=0.01,
                    help="(cũ, giữ tương thích) dung sai guard STT; bị --guard-stt-f1 ghi đè nếu khai")
    ap.add_argument("--guard-stt-f1", default="",
                    help="dung sai guard STT F1 (số, vd 0.01 = mặc định như cũ) hoặc 'none' = TẮT guard. "
                         "Tắt khi ckpt v2 chỉ phục vụ sách thạch/mộc bản qua books[].detector_ckpt theo sách "
                         "(STT không bao giờ chạy qua nó). Dù bật hay tắt, trainer LUÔN ghi thêm best_litho.pt "
                         "= epoch tốt nhất theo riêng tiêu chí thạch bản, kèm cảnh báo trong report.md.")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-gain", type=float, default=0.2)
    ap.add_argument("--p-crop", type=float, default=0.35)
    ap.add_argument("--ystretch", type=float, default=0.10)
    ap.add_argument("--p-gray", type=float, default=0.4)
    ap.add_argument("--p-otsu", type=float, default=0.15)
    ap.add_argument("--p-stretch", type=float, default=0.15)
    ap.add_argument("--p-stroke", type=float, default=0.3)
    ap.add_argument("--resize", default="area", choices=["area", "linear"],
                    help="letterbox khi đo: area (khử răng cưa, = books[].detector_resize: area) | linear (v1 pipeline cũ)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--hf-repo", default="", help="vd user/nom-char-det-v2 (đẩy last/best mỗi epoch, kéo về khi resume)")
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="4 trang/miền, 1 epoch, img 512, batch 2, eval 2 trang")
    ap.add_argument("--eval-only", action="store_true", help="chỉ đánh giá --ckpt trên val bundle")
    ap.add_argument("--ckpt", default="", help="(eval-only) ckpt cần đo")
    ap.add_argument("--out-json", default="", help="(eval-only) ghi kết quả JSON")
    a = ap.parse_args()
    if a.guard_stt_f1:
        g = a.guard_stt_f1.strip().lower()
        if g in ("none", "off", "tat", "tắt", "0none"):
            a.stt_tol = None
        else:
            try:
                a.stt_tol = float(g)
            except ValueError:
                ap.error(f"--guard-stt-f1 = {a.guard_stt_f1!r}; cần một số (vd 0.01) hoặc 'none'")

    if a.eval_only:
        ck = a.ckpt or a.init or str(Path(a.data) / "v1" / "detector_r34.best.pt")
        if a.threads:
            import torch
            torch.set_num_threads(a.threads)
        r = eval_only(ck, a.data, img=(a.img or (512 if a.smoke else 0)), eval_pages=a.eval_pages or (2 if a.smoke else 0),
                      eval_chresto=a.eval_chresto, device=a.device, limit=a.limit, out_json=a.out_json, resize=a.resize)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return

    cfg = Cfg(data=a.data, out=a.out, init=a.init, epochs=a.epochs, batch=a.batch, img=a.img or 1024, lr=a.lr, wd=a.wd,
              warmup_epochs=a.warmup_epochs, ema=a.ema, seed=a.seed, workers=a.workers, threads=a.threads,
              device=a.device, amp=not a.no_amp, samples_per_epoch=a.samples_per_epoch, limit=a.limit,
              eval_pages=a.eval_pages, eval_chresto=a.eval_chresto, stt_tol=a.stt_tol, patience=a.patience,
              min_gain=a.min_gain, p_crop=a.p_crop, ystretch=a.ystretch, p_gray=a.p_gray, p_otsu=a.p_otsu,
              p_stretch=a.p_stretch, p_stroke=a.p_stroke, log_every=a.log_every, hf_repo=a.hf_repo,
              hf_token=a.hf_token, resume=not a.no_resume, smoke=a.smoke, resize=a.resize)
    run(cfg)


if __name__ == "__main__":
    main()
