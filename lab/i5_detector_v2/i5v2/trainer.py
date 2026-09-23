"""Vòng huấn luyện detector v2 (phương án B, docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md §3).

  • init từ v1 (bundle/v1/detector_r34.best.pt), AdamW lr 2e-4, warmup 1 epoch + cosine theo bước, AMP (CUDA),
    EMA 0,997, seed cố định; loader cân bằng miền 50/50 litho/STT (WeightedRandomSampler).
  • mỗi epoch: đánh giá (trọng số EMA) trên val page-disjoint: thạch bản ok50 / miss / extra / |dy| /
    % tầng n==N @0,15 & 0,2 / cắt thân chữ; STT F1 @0,2 và @0,15; Chrestomathie (sách KHÔNG train,
    chỉ đo): % cột n==N, cắt.
  • best = ok50_litho cao nhất VỚI ĐIỀU KIỆN STT F1 (cả 0,2 lẫn 0,15) ≥ v1 − stt_tol (0,01);
    `--guard-stt-f1 none` TẮT điều kiện đó (ckpt chỉ dùng cho sách thạch/mộc bản qua
    books[].detector_ckpt). NGOÀI RA luôn ghi `best_litho.pt` = epoch tốt nhất theo RIÊNG
    tiêu chí thạch bản, KHÔNG xét guard STT — để một lần chạy luôn để lại ckpt dùng được;
    dừng sớm khi `patience` (4) epoch liền không tăng ≥ 0,2 điểm (hoặc STT tụt).
  • ghi out/{last.pt (toàn trạng thái, resume), best.pt (định dạng pipeline), metrics.csv, report.md,
    base_v1.json, history.json}; resume tự động từ out/last.pt (hoặc kéo từ HF hub nếu khai --hf-repo).
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import hub
from .data import I5Dataset, balanced_sampler, limit_per_domain, load_split
from .decode import Detector, build_from_ckpt, load_ckpt
from .evalref import evaluate, flatten
from .loss import CenterNetLoss

STRIDE = 4
METRIC_KEYS = ["epoch", "loss", "loss_hm", "lr", "sec", "is_best", "stt_ok",
               "litho_ok50", "litho_miss", "litho_extra", "litho_dy_med", "litho_dy_p90", "litho_iou_med",
               "litho_tiers_eq_015", "litho_tiers_eq_020", "litho_tiers_minus_015", "litho_tiers_plus_015",
               "litho_cut", "litho_ok50_020", "stt_F1_020", "stt_P_020", "stt_R_020", "stt_F1_015", "stt_cut_020",
               "chresto_tiers_eq_015", "chresto_tiers_eq_020", "chresto_ok50", "chresto_cut", "eval_s"]


@dataclass
class Cfg:
    data: str = "bundle"
    out: str = "out"
    init: str = ""                 # "" -> <data>/v1/detector_r34.best.pt
    epochs: int = 20
    batch: int = 4
    img: int = 1024
    lr: float = 2e-4
    wd: float = 1e-4
    warmup_epochs: float = 1.0
    ema: float = 0.995             # cửa sổ ≈ 200 bước (~2 epoch); khởi động theo timm: min(ema, (1+t)/(10+t))
    resize: str = "area"          # letterbox khi ĐO: area (khử răng cưa; = books[].detector_resize: area) | linear (v1 cũ)
    seed: int = 0
    workers: int = 2
    threads: int = 0
    device: str = ""
    amp: bool = True
    samples_per_epoch: int = 0     # 0 = 2 × min(số trang miền)
    limit: int = 0                 # số trang mỗi miền (thử)
    eval_pages: int = 0            # 0 = toàn bộ val mỗi miền
    eval_chresto: int = 20         # số trang Chrestomathie (held-out) đo mỗi epoch; 0 = bỏ
    stt_tol: float = 0.01           # dung sai guard STT; None = TẮT guard (--guard-stt-f1 none)
    patience: int = 4
    min_gain: float = 0.2          # điểm ok50 tăng tối thiểu để tính "cải thiện"
    min_epochs: int = 6            # không dừng sớm trước epoch này (đầu fine-tune ok50 có thể tụt dưới v1 rồi mới vượt)
    p_crop: float = 0.35
    ystretch: float = 0.10
    p_gray: float = 0.4
    p_otsu: float = 0.15
    p_stretch: float = 0.15
    p_stroke: float = 0.3
    grad_clip: float = 10.0
    log_every: int = 20
    hf_repo: str = ""
    hf_token: str = ""
    resume: bool = True
    smoke: bool = False
    notes: dict = field(default_factory=dict)


def pick_device(pref: str = "") -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"          # MPS: DeformConv2d (DCN của v1) chưa có backward → không tự chọn; --device mps nếu cố ý


def seed_all(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """Trung bình mũ trọng số; buffer (BN running stats) sao chép trực tiếp."""

    def __init__(self, net, decay: float):
        self.decay = decay
        self.n_updates = 0
        self.module = copy.deepcopy(net).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net):
        self.n_updates += 1
        d = min(self.decay, (1 + self.n_updates) / (10 + self.n_updates))   # khởi động: bám sát mô hình thô lúc đầu
        msd = self.module.state_dict()
        for k, v in net.state_dict().items():
            if v.dtype.is_floating_point:
                msd[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                msd[k].copy_(v)


def _worker_init(_wid: int):
    """OpenCV trong worker fork: tắt đa luồng nội bộ (tránh treo/oversubscription trên Kaggle 4 CPU)."""
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass


def _move(batch, device):
    return {k: batch[k].to(device, non_blocking=True) for k in ("hm", "wh", "off", "ind", "mask")}


def lr_at(step: int, total: int, warm: int, base: float) -> float:
    if warm > 0 and step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return max(1e-6, 0.5 * base * (1 + math.cos(math.pi * min(1.0, t))))


def eval_net(net, cfg: Cfg, val_lit, val_stt, val_chr, device) -> dict:
    det = Detector(net=net, img=cfg.img, thr=0.05, device=device, resize=cfg.resize)
    res = evaluate(det, val_lit + val_stt + val_chr, thrs=(0.15, 0.2))
    flat = flatten(res)
    flat["_raw"] = res
    return flat


def _score(ev: dict) -> tuple:
    """Điểm chọn best: ok50 (làm tròn 0,1) rồi % tầng n==N @0,15 rồi −cắt thân chữ — so từ điển; khởi điểm = v1 cùng mã,
    nên best.pt CHỈ được ghi khi v2 vượt v1 (+area) trên chính tập val này."""
    return (round(ev.get("litho_ok50") or 0.0, 1), round(ev.get("litho_tiers_eq_015") or 0.0, 1),
            -round(ev.get("litho_cut") or 0.0, 2))


def _stt_ok(ev: dict, base: dict, tol: float | None) -> bool:
    """`tol=None` -> guard TẮT (luôn True). Dùng khi ckpt v2 chỉ phục vụ sách thạch/mộc
    bản (books[].detector_ckpt theo sách) nên STT không bao giờ chạy qua nó — guard chỉ
    cần khi MỘT mô hình phải phục vụ cả hai miền."""
    if tol is None:
        return True
    for k in ("stt_F1_020", "stt_F1_015"):
        if ev.get(k) is None or base.get(k) is None:
            continue
        if ev[k] < base[k] - tol:
            return False
    return True


def _pipeline_ckpt(state_dict, meta, cfg: Cfg, epoch: int, val: dict | None, extra: dict | None = None) -> dict:
    d = {"model": state_dict, "arch": meta["arch"], "stride": STRIDE, "img": cfg.img, "use_dcn": meta["use_dcn"],
         "epoch": epoch, "v2": True, "init": cfg.init, "val": {k: v for k, v in (val or {}).items() if k != "_raw"},
         "train_cfg": {k: v for k, v in asdict(cfg).items() if k not in ("hf_token",)}}
    if extra:
        d.update(extra)
    return d


def write_report(out: Path, base: dict, hist: list[dict], best_ep: int, cfg: Cfg,
                 best_litho_ep: int = -1):
    rows = [("thạch bản val: ô tham chiếu IoU ≥ 0,5 (%)", "litho_ok50", "↑"),
            ("thạch bản: miss (%)", "litho_miss", "↓"), ("thạch bản: extra / 100 ô", "litho_extra", "↓"),
            ("thạch bản: |dy| tâm trung vị (% bước)", "litho_dy_med", "↓"), ("thạch bản: |dy| p90 (% bước)", "litho_dy_p90", "↓"),
            ("thạch bản: % tầng n == N @0,15 (hộp thô)", "litho_tiers_eq_015", "↑"),
            ("thạch bản: % tầng n == N @0,2", "litho_tiers_eq_020", "↑"),
            ("thạch bản: % tầng n < N @0,15", "litho_tiers_minus_015", "↓"), ("thạch bản: % tầng n > N @0,15", "litho_tiers_plus_015", "↓"),
            ("thạch bản: cắt vào thân chữ (%; không phụ thuộc tham chiếu)", "litho_cut", "↓"),
            ("STT val: F1 @0,2 (ngưỡng production) — GUARD ≥ v1 − 0,01", "stt_F1_020", "↑"),
            ("STT val: F1 @0,15", "stt_F1_015", "↑"), ("STT val: cắt (%) @0,2", "stt_cut_020", "↓"),
            ("Chrestomathie (KHÔNG train): % cột n == N @0,15", "chresto_tiers_eq_015", "↑"),
            ("Chrestomathie: % cột n == N @0,2", "chresto_tiers_eq_020", "↑"),
            ("Chrestomathie: ô tham chiếu IoU ≥ 0,5 (%)", "chresto_ok50", "↑"), ("Chrestomathie: cắt (%)", "chresto_cut", "↓")]
    best = next((h for h in hist if h["epoch"] == best_ep), None)
    lines = [f"# Detector v2 — báo cáo huấn luyện ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"img {cfg.img} · batch {cfg.batch} · lr {cfg.lr} · epochs chạy {len(hist)}/{cfg.epochs} · best epoch **{best_ep}** "
             f"· init `{Path(cfg.init).name}` · seed {cfg.seed} · smoke {cfg.smoke}", "",
             "Val page-disjoint (trang 10k+3 mỗi sách). Mọi số là proxy (ô tham chiếu = kim + chiếu mực, không GT người);"
             " % tầng n==N đếm hộp THÔ (không ép N). Chrestomathie không có trang nào trong train.", "",
             "| chỉ số | v1 | v2 (best) | Δ | mục tiêu |", "|---|---|---|---|---|"]
    bl = next((h for h in hist if h["epoch"] == best_litho_ep), None)
    goals = {"litho_ok50": "≥ 98", "litho_tiers_eq_015": "≥ 90", "litho_cut": "giảm",
             "stt_F1_020": (f"≥ v1 − {cfg.stt_tol}" if cfg.stt_tol is not None else "GUARD TẮT")}
    for name, k, arrow in rows:
        b = base.get(k); v = best.get(k) if best else None
        d = (None if b is None or v is None else round(v - b, 2))
        lines.append(f"| {name} | {b} | {v} | {'' if d is None else ('+' if d > 0 else '') + str(d)} {arrow} | {goals.get(k, '')} |")
    if bl:
        lines += ["", "## `best_litho.pt` — chọn theo RIÊNG tiêu chí thạch bản (KHÔNG qua guard STT)", "",
                  f"Epoch **{best_litho_ep}**. Ckpt này tồn tại để một lần chạy luôn để lại tệp dùng được cho "
                  "sách thạch/mộc bản, kể cả khi guard STT chặn `best.pt`.", "",
                  "| chỉ số | v1 | best_litho | Δ |", "|---|---|---|---|"]
        for name, k, arrow in rows:
            b = base.get(k); v = bl.get(k)
            d = (None if b is None or v is None else round(v - b, 2))
            lines.append(f"| {name} | {b} | {v} | {'' if d is None else ('+' if d > 0 else '') + str(d)} {arrow} |")
        _ok = _stt_ok(bl, base, cfg.stt_tol if cfg.stt_tol is not None else 0.01)
        lines += ["", ("> ⚠️ **CẢNH BÁO.** Epoch này " + ("QUA" if _ok else "KHÔNG qua")
                       + f" guard STT (F1@0,2 {base.get('stt_F1_020')} → {bl.get('stt_F1_020')}"
                       + f", @0,15 {base.get('stt_F1_015')} → {bl.get('stt_F1_015')}). "
                       + ("Vì vậy CHỈ được khai qua `books[].detector_ckpt` cho các sách thạch/mộc bản; "
                          "đặt làm detector TOÀN CỤC (env `NOM_DETECTOR_CKPT`) sẽ làm STT tụt và phá "
                          "bất biến byte-identical của bộ STT." if not _ok
                          else "Dù vậy vẫn nên khai theo sách để giữ STT hoàn toàn không đổi byte."))]
    lines += ["", "## Theo epoch", "", "| ep | loss | ok50 | n==N@0,15 | cut | STT F1@0,2 | STT ok | best |", "|---|---|---|---|---|---|---|---|"]
    for h in hist:
        lines.append(f"| {h['epoch']} | {h.get('loss')} | {h.get('litho_ok50')} | {h.get('litho_tiers_eq_015')} | {h.get('litho_cut')} "
                     f"| {h.get('stt_F1_020')} | {'ok' if h.get('stt_ok') else 'TỤT'} | {'*' if h.get('is_best') else ''} |")
    verdict = []
    if best:
        verdict.append(f"- ok50_litho {base.get('litho_ok50')} → {best.get('litho_ok50')} (mục tiêu ≥ 98): "
                       f"{'ĐẠT' if (best.get('litho_ok50') or 0) >= 98 else 'chưa đạt'}")
        verdict.append(f"- % tầng n==N @0,15 {base.get('litho_tiers_eq_015')} → {best.get('litho_tiers_eq_015')} (mục tiêu ≥ 90): "
                       f"{'ĐẠT' if (best.get('litho_tiers_eq_015') or 0) >= 90 else 'chưa đạt'}")
        verdict.append(f"- STT F1@0,2 {base.get('stt_F1_020')} → {best.get('stt_F1_020')} (guard ≥ v1 − {cfg.stt_tol}): "
                       f"{'giữ' if _stt_ok(best, base, cfg.stt_tol) else 'TỤT'}")
        verdict.append(f"- cắt thân chữ {base.get('litho_cut')} → {best.get('litho_cut')}: "
                       f"{'giảm' if (best.get('litho_cut') or 0) < (base.get('litho_cut') or 0) else 'KHÔNG giảm'}")
    else:
        verdict.append("- KHÔNG có epoch nào vượt v1 (ok50 → n==N → cắt, cùng mã cùng val) qua guard STT → không có best.pt.")
        if bl:
            verdict.append(f"- **NHƯNG có `best_litho.pt` (epoch {best_litho_ep})**: trên thạch bản ok50 "
                           f"{base.get('litho_ok50')} → {bl.get('litho_ok50')}, % tầng n==N @0,15 "
                           f"{base.get('litho_tiers_eq_015')} → {bl.get('litho_tiers_eq_015')}, cắt thân chữ "
                           f"{base.get('litho_cut')} → {bl.get('litho_cut')}. Dùng ckpt này cho 5 sách thạch/mộc bản "
                           "qua `books[].detector_ckpt`; 3 sách STT KHÔNG khai (giữ v1) — xem cảnh báo ở trên.")
        else:
            verdict.append("- Cũng KHÔNG có `best_litho.pt` (không epoch nào vượt v1 ngay trên thạch bản) → "
                           "giữ v1 + `detector_resize: area`.")
    lines += ["", "## Kết luận", ""] + verdict + ["", "Bước tiếp: tải `best.pt` (hoặc `best_litho.pt` nếu guard STT chặn) "
                                                 "về máy → `lab/i5_detector_v2/apply_v2.sh <ckpt>` "
                                                 "(đo box_ref_eval v1↔v2 trên ảnh gốc + build all-new _v2)."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: Cfg) -> dict:
    t_start = time.time()
    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(cfg.data)
    if not cfg.init:
        cfg.init = str(data / "v1" / "detector_r34.best.pt")
    if cfg.smoke:
        cfg.epochs, cfg.img, cfg.batch = 1, min(cfg.img, 512), 2
        cfg.limit, cfg.eval_pages, cfg.eval_chresto = cfg.limit or 4, cfg.eval_pages or 2, min(cfg.eval_chresto, 2)
        cfg.workers, cfg.samples_per_epoch = 0, cfg.samples_per_epoch or 4
    device = pick_device(cfg.device)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    seed_all(cfg.seed)
    use_amp = bool(cfg.amp and device == "cuda")

    # ---- dữ liệu
    train_items = load_split(data, "train")
    val_items = load_split(data, "val")
    if cfg.limit:
        train_items = limit_per_domain(train_items, cfg.limit)
        val_items = limit_per_domain(val_items, cfg.limit)
    train_items = [it for it in train_items if it.get("domain") in ("litho", "stt") and it.get("boxes")]
    val_lit = [it for it in val_items if it.get("domain") == "litho"]
    val_stt = [it for it in val_items if it.get("domain") == "stt"]
    if cfg.eval_pages:
        val_lit, val_stt = val_lit[:cfg.eval_pages], val_stt[:cfg.eval_pages]
    val_chr = []
    if cfg.eval_chresto and (data / "manifest_chresto.json").exists():
        val_chr = load_split(data, "chresto")[:cfg.eval_chresto]
    ds = I5Dataset(train_items, img=cfg.img, train=True, seed=cfg.seed, p_crop=cfg.p_crop, ystretch=cfg.ystretch,
                   p_gray=cfg.p_gray, p_otsu=cfg.p_otsu, p_stretch=cfg.p_stretch, p_stroke=cfg.p_stroke)
    sampler, cnt = balanced_sampler(train_items, cfg.samples_per_epoch or None)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch, sampler=sampler, num_workers=cfg.workers,
                                         drop_last=True, pin_memory=(device == "cuda"),
                                         persistent_workers=(cfg.workers > 0), worker_init_fn=_worker_init)
    steps_per_epoch = len(loader)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warm_steps = int(cfg.warmup_epochs * steps_per_epoch)
    print(f"[v2] device={device} amp={use_amp} | train {cnt} → {len(sampler)} mẫu/epoch ({steps_per_epoch} bước × batch {cfg.batch}) "
          f"| val litho {len(val_lit)} stt {len(val_stt)} chresto {len(val_chr)} | img {cfg.img} | init {cfg.init}", flush=True)

    # ---- mô hình
    net, meta = build_from_ckpt(cfg.init)
    if meta["missing"] or meta["unexpected"]:
        print(f"  [init] thiếu {len(meta['missing'])} / thừa {len(meta['unexpected'])} khoá state_dict", flush=True)
    net = net.to(device)
    ema = EMA(net, cfg.ema)
    crit = CenterNetLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- mốc v1 (cùng mã, cùng tập val) — cache để resume không đo lại
    base_p = out / "base_v1.json"
    base = None
    if base_p.exists():
        try:
            b = json.load(open(base_p, encoding="utf-8"))
            if b.get("_key") == [cfg.img, len(val_lit), len(val_stt), len(val_chr)]:
                base = b
        except Exception:
            base = None
    if base is None:
        t0 = time.time()
        base = eval_net(ema.module, cfg, val_lit, val_stt, val_chr, device)
        base["_key"] = [cfg.img, len(val_lit), len(val_stt), len(val_chr)]
        json.dump(base, open(base_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  mốc v1 ({time.time() - t0:.0f}s): ok50 {base['litho_ok50']} | n==N@0,15 {base['litho_tiers_eq_015']} "
              f"| cut {base['litho_cut']} | STT F1@0,2 {base['stt_F1_020']} (@0,15 {base['stt_F1_015']})"
              + (f" | chresto n==N {base['chresto_tiers_eq_015']}" if val_chr else ""), flush=True)
    else:
        print(f"  mốc v1 (cache): ok50 {base['litho_ok50']} | STT F1@0,2 {base['stt_F1_020']}", flush=True)

    # ---- resume
    token = hub.resolve_token(cfg.hf_token) if cfg.hf_repo else None
    if cfg.hf_repo and token:
        hub.ensure_repo(cfg.hf_repo, token)
    last_p, best_p = out / "last.pt", out / "best.pt"
    # (2026-09-23) best_litho.pt = epoch tốt nhất theo RIÊNG tiêu chí thạch bản (ok50 → n==N
    # → −cắt), KHÔNG xét guard STT. Lý do: repo đã có books[].detector_ckpt THEO SÁCH, nên
    # ckpt thạch bản không bao giờ chạy trên STT; guard chỉ cần khi một mô hình phục vụ cả
    # hai miền. Lần chạy 22/09 mất trắng vì guard chặn mà không có tệp nào khác được ghi.
    litho_p = out / "best_litho.pt"
    # best khởi điểm = v1 (cùng mã, cùng val): epoch nào không vượt v1 thì không thành best.pt
    start_ep, best, best_ep, stall, hist, global_step = 0, _score(base), -1, 0, [], 0
    best_litho, best_litho_ep = _score(base), -1
    best_ok50_ref = base.get("litho_ok50") or 0.0
    if cfg.resume and not last_p.exists() and cfg.hf_repo:
        hub.pull(cfg.hf_repo, "last.pt", token, out)
        if not best_p.exists():
            hub.pull(cfg.hf_repo, "best.pt", token, out)
    if cfg.resume and last_p.exists():
        st = load_ckpt(last_p)
        if st.get("v2") and "model_raw" in st and st.get("img") == cfg.img:
            net.load_state_dict(st["model_raw"]); ema.module.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
            start_ep, global_step = int(st["epoch"]), int(st.get("global_step", 0))
            ema.n_updates = global_step
            best, best_ep, stall, hist = tuple(st.get("best") or _score(base)), int(st.get("best_epoch", -1)), int(st.get("stall", 0)), list(st.get("history", []))
            best_litho = tuple(st.get("best_litho") or best)
            best_litho_ep = int(st.get("best_litho_epoch", -1))
            best_ok50_ref = float(st.get("best_ok50_ref", best_ok50_ref))
            if "rng" in st:
                torch.set_rng_state(st["rng"]["torch"]); np.random.set_state(st["rng"]["numpy"])
            print(f"  [resume] {last_p}: đã xong {start_ep}/{cfg.epochs} epoch, best {best} @ep {best_ep}, stall {stall}", flush=True)
        else:
            print(f"  [resume] {last_p} không khớp (img/định dạng) → huấn luyện từ đầu", flush=True)

    mcsv = out / "metrics.csv"
    if start_ep == 0 or not mcsv.exists():
        with open(mcsv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_KEYS, extrasaction="ignore"); w.writeheader()
            w.writerow({"epoch": 0, "is_best": "", "stt_ok": True, **{k: base.get(k) for k in METRIC_KEYS if k in base}})

    # ---- vòng epoch
    for ep in range(start_ep, cfg.epochs):
        t0 = time.time()
        net.train()
        agg = {"total": 0.0, "hm": 0.0}; nb = 0
        for bi, batch in enumerate(loader):
            lr = lr_at(global_step, total_steps, warm_steps, cfg.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x = batch["image"].to(device, non_blocking=True)
            tg = _move(batch, device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    outp = net(x)
                loss, parts = crit(outp, tg)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                outp = net(x)
                loss, parts = crit(outp, tg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt.step()
            ema.update(net)
            global_step += 1
            agg["total"] += float(loss.detach()); agg["hm"] += parts["hm"]; nb += 1
            if cfg.log_every and bi % cfg.log_every == 0:
                print(f"    ep {ep + 1} bước {bi:4d}/{steps_per_epoch} lr {lr:.2e} loss {float(loss.detach()):.4f} "
                      f"(hm {parts['hm']:.4f} wh {parts['wh']:.4f} off {parts['off']:.4f})", flush=True)
        tr_loss = agg["total"] / max(1, nb)
        ev = eval_net(ema.module, cfg, val_lit, val_stt, val_chr, device)
        stt_ok = _stt_ok(ev, base, cfg.stt_tol)
        score = _score(ev)
        is_best = bool(stt_ok and ev.get("litho_ok50") is not None and score > best)
        improved = bool(stt_ok and ev.get("litho_ok50") is not None and ev["litho_ok50"] >= best_ok50_ref + cfg.min_gain)
        row = {"epoch": ep + 1, "loss": round(tr_loss, 4), "loss_hm": round(agg["hm"] / max(1, nb), 4),
               "lr": f"{lr:.2e}", "sec": round(time.time() - t0, 1), "is_best": is_best, "stt_ok": stt_ok,
               **{k: v for k, v in ev.items() if k != "_raw"}}
        hist.append(row)
        print(f"epoch {ep + 1:2d}/{cfg.epochs} {row['sec']:6.1f}s loss {tr_loss:.4f} | litho ok50 {ev['litho_ok50']} miss {ev['litho_miss']} "
              f"extra {ev['litho_extra']} n==N@0,15 {ev['litho_tiers_eq_015']} (@0,2 {ev['litho_tiers_eq_020']}) cut {ev['litho_cut']} "
              f"| STT F1@0,2 {ev['stt_F1_020']} ({'ok' if stt_ok else 'TỤT'})"
              + (f" | chresto n==N {ev['chresto_tiers_eq_015']}" if val_chr else "")
              + (" ** BEST" if is_best else ""), flush=True)
        if improved:
            stall = 0; best_ok50_ref = ev["litho_ok50"]
        else:
            stall += 1
        if is_best:
            best, best_ep = score, ep + 1
            torch.save(_pipeline_ckpt(ema.module.state_dict(), meta, cfg, ep + 1, ev, {"base_v1": {k: v for k, v in base.items() if k != "_raw"}}), best_p)
        is_best_litho = bool(ev.get("litho_ok50") is not None and score > best_litho)
        if is_best_litho:
            best_litho, best_litho_ep = score, ep + 1
            torch.save(_pipeline_ckpt(ema.module.state_dict(), meta, cfg, ep + 1, ev, {
                "base_v1": {k: v for k, v in base.items() if k != "_raw"},
                "selected_by": "litho_only",
                "stt_guard_passed": bool(stt_ok),
                # CẢNH BÁO đi THEO TỆP: ai nạp ckpt này cũng đọc được lý do và giới hạn.
                "warning": ("best_litho.pt chọn theo RIÊNG tiêu chí thạch bản (ok50 → % tầng n==N → −cắt), "
                            "KHÔNG qua guard STT F1. CHỈ dùng cho sách thạch/mộc bản qua books[].detector_ckpt "
                            f"theo sách; STT F1@0,2 của epoch này = {ev.get('stt_F1_020')} so v1 {base.get('stt_F1_020')}. "
                            "Đưa ckpt này thành detector TOÀN CỤC sẽ làm STT tụt."),
            }), litho_p)
        with open(mcsv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=METRIC_KEYS, extrasaction="ignore").writerow(row)
        torch.save(_pipeline_ckpt(ema.module.state_dict(), meta, cfg, ep + 1, ev, {
            "model_raw": net.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
            "global_step": global_step, "best": list(best), "best_epoch": best_ep, "stall": stall, "history": hist,
            "best_litho": list(best_litho), "best_litho_epoch": best_litho_ep,
            "best_ok50_ref": best_ok50_ref,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}}), last_p)
        json.dump({"base_v1": {k: v for k, v in base.items() if k != "_raw"}, "history": hist, "best_epoch": best_ep,
                   "best_score(ok50,tiers,-cut)": list(best), "cfg": {k: v for k, v in asdict(cfg).items() if k != "hf_token"}},
                  open(out / "history.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        write_report(out, base, hist, best_ep, cfg, best_litho_ep)
        if cfg.hf_repo and token:
            hub.push(last_p, cfg.hf_repo, token)
            if is_best:
                hub.push(best_p, cfg.hf_repo, token)
            if is_best_litho:
                hub.push(litho_p, cfg.hf_repo, token)
            hub.push(mcsv, cfg.hf_repo, token); hub.push(out / "report.md", cfg.hf_repo, token)
        if stall >= cfg.patience and ep + 1 >= cfg.min_epochs:
            print(f"  dừng sớm: {cfg.patience} epoch không cải thiện ≥ {cfg.min_gain} điểm ok50 (hoặc STT tụt)", flush=True)
            break
    print(f"[done] {time.time() - t_start:.0f}s | best epoch {best_ep} score(ok50,n==N,-cut) {best} (v1 {_score(base)}) "
          f"| {best_p if best_ep > 0 else 'KHÔNG có best.pt: không epoch nào vượt v1 qua guard STT'}"
          f" | best_litho epoch {best_litho_ep}: {litho_p if best_litho_ep > 0 else 'KHÔNG có (không epoch nào vượt v1 trên thạch bản)'}"
          f" | {out / 'report.md'}", flush=True)
    return {"best_epoch": best_ep, "best_score": best, "base_v1": base, "history": hist, "out": str(out)}


def eval_only(ckpt: str, data: str, img: int = 0, eval_pages: int = 0, eval_chresto: int = 0, device: str = "",
              limit: int = 0, out_json: str = "", resize: str = "area") -> dict:
    """Đánh giá 1 ckpt (v1 hoặc v2) trên val bundle bằng đúng mã của trainer → bảng phẳng."""
    device = pick_device(device)
    d = load_ckpt(ckpt)
    img = img or int(d.get("img", 1024))
    val_items = load_split(data, "val")
    if limit:
        val_items = limit_per_domain(val_items, limit)
    val_lit = [it for it in val_items if it.get("domain") == "litho"]
    val_stt = [it for it in val_items if it.get("domain") == "stt"]
    if eval_pages:
        val_lit, val_stt = val_lit[:eval_pages], val_stt[:eval_pages]
    val_chr = load_split(data, "chresto") if (Path(data) / "manifest_chresto.json").exists() else []
    if eval_chresto:
        val_chr = val_chr[:eval_chresto]
    net, meta = build_from_ckpt(ckpt)
    cfg = Cfg(data=data, img=img, resize=resize)
    t0 = time.time()
    ev = eval_net(net.to(device), cfg, val_lit, val_stt, val_chr, device)
    flat = {k: v for k, v in ev.items() if k != "_raw"}
    flat.update({"_ckpt": str(ckpt), "_img": img, "_resize": resize, "_pages": {"litho": len(val_lit), "stt": len(val_stt), "chresto": len(val_chr)},
                 "_device": device, "_sec": round(time.time() - t0, 1)})
    if out_json:
        json.dump({**flat, "_raw": ev["_raw"]}, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return flat
