"""Detector v2 — fine-tune CenterNet r34 (v1, học STT) trên NHÃN YẾU thạch bản + STT (kế hoạch B).

Khác train_centernet.py (v1):
  • Dữ liệu: build_lithograph_manifest.py → manifest_train/val.json (2 miền `domain` litho|stt,
    `ignore_boxes` tô trắng trước khi học: tầng không verified, ô xấu, REVIEW STT).
  • Lấy mẫu CÂN BẰNG MIỀN 50/50 mỗi epoch (WeightedRandomSampler; STT 445 trang vs thạch bản
    ~268 trang) — tránh quên STT (hồi quy cột đúng số hộp 90,1 %).
  • Augmentation thêm cho thạch bản (LithoDataset): NỀN XÁM (nén dải xám: mực → lo∈[20,80], nền
    → hi∈[100,210], mô phỏng ảnh JPG gốc nền ~128 mà CenterNet v1 "mù"), KÉO DỌC ngẫu nhiên
    ±10 % (jitter bước cột), rồi affine/blur/noise của v1.
  • Khởi tạo từ v1 (--init train_crop/detector_r34.best.pt), LR nhỏ (1e-4), cosine.
  • Đánh giá mỗi epoch: (a) thạch bản val: ok50 (IoU ≥ 0,5), |dy| % bước, % tầng n == N, cắt vào
    thân chữ (eval_boxes_ref.evaluate); (b) STT val: F1 @IoU 0,5 (hồi quy). Lưu best theo
    ok50_litho VỚI ĐIỀU KIỆN F1_stt ≥ F1_stt(v1) − 0,01 (không hi sinh STT).
  • Tiêu chí dừng: 3 epoch liên tiếp ok50_litho không tăng ≥ 0,2 điểm, hoặc F1_stt tụt > 0,01.

Thời gian đo (22/09, Mac 10 nhân, CPU, --threads 8): img 1024 batch 2 ≈ 2–3 s/ảnh (4 ảnh + eval 2 trang
= 11,8 s) → 1 epoch cân bằng (2×215 trang train) ≈ 15–25 phút + eval 60 trang ≈ 40 s; 20 epoch ≈ 5–8 h CPU.
Kaggle T4 ≈ 0,3 s/ảnh → ≈ 2–3 phút/epoch, 20 epoch ≈ 1 giờ. Chứng minh chạy được (đã chạy, 5–12 s):
  .venv/bin/python train_crop/train_v2_lithograph.py --limit 3 --img 512 --epochs 1 --batch 2 \\
      --workers 0 --threads 6 --device cpu --eval-pages 2 --out /tmp/v2_smoke.pt
Chạy thật (Kaggle): xem docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md §B.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_centernet import CenterNetDataset, STRIDE, build_targets, _augment, _letterbox   # noqa: E402
from model_centernet import build_model                                                   # noqa: E402
from train_centernet import CenterNetLoss, train_one_epoch, validate, _pick_device        # noqa: E402
from infer_centernet import CenterNetDetector                                             # noqa: E402
import eval_boxes_ref                                                                     # noqa: E402


class LithoDataset(CenterNetDataset):
    """CenterNetDataset + tô trắng ignore_boxes + augment nền xám / kéo dọc (chỉ khi train)."""

    def __init__(self, items, img=1024, train=True, gray_bg_p=0.5, ystretch=0.10, seed=0):
        super().__init__(items, img=img, train=train, seed=seed)
        self.gray_bg_p, self.ystretch = gray_bg_p, ystretch

    def _load(self, path):
        import cv2
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        return im

    def __getitem__(self, i):
        import cv2
        it = self.items[i]
        im = self._load(it["image"])
        rng = np.random.default_rng(None if self.train else self._seed + i)
        if im is None:
            im = np.full((self.img, self.img, 3), 255, np.uint8); boxes = np.zeros((0, 4), np.float32)
        else:
            boxes = np.array(it["boxes"], np.float32).reshape(-1, 4)
            for g in it.get("ignore_boxes") or []:                 # tô trắng vùng không nhãn
                x1, y1, x2, y2 = [int(v) for v in g[:4]]
                im[max(0, y1):y2, max(0, x1):x2] = 255
        if self.train and im.shape[0] > 1:
            if self.ystretch > 0:                                  # kéo dọc: jitter bước cột
                sy = float(rng.uniform(1 - self.ystretch, 1 + self.ystretch))
                H, W = im.shape[:2]
                im = cv2.resize(im, (W, max(1, int(H * sy))), interpolation=cv2.INTER_LINEAR)
                if len(boxes):
                    boxes[:, [1, 3]] *= sy
            if rng.random() < self.gray_bg_p:                      # nền xám: nén dải xám
                lo, hi = float(rng.uniform(20, 80)), float(rng.uniform(100, 210))
                im = np.clip(lo + im.astype(np.float32) * (hi - lo) / 255.0, 0, 255).astype(np.uint8)
            canvas, boxes = _augment(im, boxes, self.img, rng)
        else:
            canvas, s, _, _ = _letterbox(im, self.img)
            boxes = boxes * s if len(boxes) else boxes.reshape(-1, 4)
        oh = ow = self.img // STRIDE
        hm, wh, off, ind, mask = build_targets(boxes, oh, ow, self.max_obj)
        x = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
        return {"image": torch.from_numpy(x).permute(2, 0, 1).contiguous(), "hm": torch.from_numpy(hm),
                "wh": torch.from_numpy(wh), "off": torch.from_numpy(off), "ind": torch.from_numpy(ind),
                "mask": torch.from_numpy(mask), "n_boxes": int(mask.sum())}


def balanced_sampler(items, n_samples=None):
    """Trọng số sao cho mỗi miền được rút 50 % số mẫu (STT nhiều trang hơn thạch bản)."""
    dom = [it.get("domain", "litho") for it in items]
    cnt = {d: dom.count(d) for d in set(dom)}
    w = [1.0 / cnt[d] for d in dom]
    n = n_samples or 2 * min(cnt.values())
    return torch.utils.data.WeightedRandomSampler(w, num_samples=n, replacement=True), cnt


def _eval(net_ckpt_path, img, val_items, thr, eval_pages, workers):
    det = CenterNetDetector(net_ckpt_path, img=img, thr=0.05, device="cpu")
    det.img = img                                   # đánh giá ở đúng --img đang train (ckpt v1 ghi 1024)
    lit = [it for it in val_items if it.get("domain") == "litho"][:eval_pages]
    stt = [it for it in val_items if it.get("domain") == "stt"][:eval_pages]
    r = eval_boxes_ref.evaluate(det, lit + stt, [thr])
    L = (r.get("litho") or {}).get(str(thr), {})
    S = (r.get("stt") or {}).get(str(thr), {})
    return {"litho_ok50": L.get("ok_iou50_pct"), "litho_miss": L.get("miss_pct"), "litho_extra": L.get("extra_per_100"),
            "litho_dy": L.get("abs_dy_pct_pitch_med"), "litho_tiers_eq": L.get("tiers_n_eq_N_pct"),
            "litho_cut": L.get("cut_glyph_pct"), "stt_F1": S.get("F1"), "stt_ok50": S.get("ok_iou50_pct")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default=str(HERE / "data_lithograph"))
    ap.add_argument("--init", default=str(HERE / "detector_r34.best.pt"))
    ap.add_argument("--img", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (0 = trong tiến trình)")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--device", default="", help="cpu | mps | cuda (mặc định tự chọn)")
    ap.add_argument("--limit", type=int, default=0, help="số trang MỖI MIỀN (thử)")
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="0 = 2 × min(số trang miền)")
    ap.add_argument("--eval-pages", type=int, default=30, help="số trang val mỗi miền để đánh giá")
    ap.add_argument("--eval-thr", type=float, default=0.15)
    ap.add_argument("--no-gray-bg", action="store_true")
    ap.add_argument("--ystretch", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--stt-tol", type=float, default=0.01, help="F1 STT được tụt tối đa so với v1")
    ap.add_argument("--out", default=str(HERE / "detector_v2_litho.pt"))
    ap.add_argument("--log-every", type=int, default=20)
    a = ap.parse_args()

    md = Path(a.manifest_dir)
    train_items = json.load(open(md / "manifest_train.json", encoding="utf-8"))
    val_items = json.load(open(md / "manifest_val.json", encoding="utf-8"))
    if a.limit:
        def _lim(items):
            out = []
            for d in ("litho", "stt"):
                out += [it for it in items if it.get("domain") == d][:a.limit]
            return out
        train_items, val_items = _lim(train_items), _lim(val_items)
    device = a.device or _pick_device()
    torch.set_num_threads(max(1, a.threads)); torch.manual_seed(0); np.random.seed(0)
    ds = LithoDataset(train_items, img=a.img, train=True, gray_bg_p=0.0 if a.no_gray_bg else 0.5, ystretch=a.ystretch)
    sampler, cnt = balanced_sampler(train_items, a.samples_per_epoch or None)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                         drop_last=True, persistent_workers=(a.workers > 0))
    print(f"[v2] device={device} | train {cnt} → {len(sampler)} mẫu/epoch (cân bằng miền) | val {len(val_items)} "
          f"| img {a.img} | init {a.init}", flush=True)

    net = build_model(arch="resnet34_fpn", pretrained=False, use_dcn=False)
    use_dcn = False
    if a.init and Path(a.init).exists():
        d = torch.load(a.init, map_location="cpu")
        use_dcn = bool(d.get("use_dcn", False))
        net = build_model(arch=d.get("arch", "resnet34_fpn"), pretrained=False, use_dcn=use_dcn)
        net.load_state_dict(d.get("model", d), strict=False)
        print(f"  init từ {a.init} (img ckpt {d.get('img')}, dcn {use_dcn})", flush=True)
    net = net.to(device)
    crit = CenterNetLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs))
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # mốc v1 trên cùng tập val (hồi quy STT + xuất phát thạch bản)
    base = _eval(a.init, a.img, val_items, a.eval_thr, a.eval_pages, a.workers) if Path(a.init).exists() else {}
    print(f"  mốc v1: {base}", flush=True)
    best, best_ep, stall, hist = None, -1, 0, []
    last_path = str(Path(a.out).with_suffix(".last.pt"))
    best_path = str(Path(a.out).with_suffix(".best.pt"))
    for ep in range(a.epochs):
        t0 = time.time()
        tr = train_one_epoch(net, loader, crit, opt, scaler, device, use_amp, log_every=a.log_every)
        sched.step()
        ckpt = {"model": net.state_dict(), "arch": "resnet34_fpn", "stride": STRIDE, "img": a.img,
                "use_dcn": use_dcn, "epoch": ep + 1, "v2": True, "init": a.init}
        torch.save(ckpt, last_path)
        ev = _eval(last_path, a.img, val_items, a.eval_thr, a.eval_pages, a.workers)
        ev["epoch"] = ep + 1; ev["loss"] = round(tr["total"], 4); ev["sec"] = round(time.time() - t0, 1)
        hist.append(ev)
        ok_stt = (ev.get("stt_F1") is None or base.get("stt_F1") is None
                  or ev["stt_F1"] >= base["stt_F1"] - a.stt_tol)
        score = ev.get("litho_ok50") or 0.0
        print(f"epoch {ep+1:2d}/{a.epochs} {ev['sec']:6.1f}s loss {tr['total']:.4f} (hm {tr['hm']:.4f}) | litho ok50 "
              f"{ev['litho_ok50']} miss {ev['litho_miss']} extra {ev['litho_extra']} |dy| {ev['litho_dy']}%p "
              f"tiers== {ev['litho_tiers_eq']}% cut {ev['litho_cut']}% | STT F1 {ev['stt_F1']} ({'ok' if ok_stt else 'TỤT'})",
              flush=True)
        if ok_stt and (best is None or score >= best + 0.2):
            best, best_ep, stall = score, ep + 1, 0
            ckpt["val"] = ev; torch.save(ckpt, best_path)
            print(f"    ** best litho ok50 {best} -> {best_path}", flush=True)
        else:
            stall += 1
        if stall >= a.patience:
            print(f"  dừng sớm: {a.patience} epoch không cải thiện ≥ 0,2 điểm / STT tụt", flush=True)
            break
    json.dump({"base_v1": base, "history": hist, "best_epoch": best_ep, "best_litho_ok50": best,
               "args": vars(a)}, open(str(Path(a.out).with_suffix(".history.json")), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[done] best epoch {best_ep} litho ok50 {best} | last {last_path} | history {Path(a.out).with_suffix('.history.json')}")
    print("  Nối vào pipeline: NOM_DETECTOR_CKPT=<best.pt> (detector_infer._find_ckpt) rồi chạy lại "
          "scripts/measure/detector_transfer.py (STT ≥ 90,1 %) + box_ref_eval.py + build --limit.")


if __name__ == "__main__":
    main()
