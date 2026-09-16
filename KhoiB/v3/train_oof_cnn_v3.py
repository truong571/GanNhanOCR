"""HUẤN LUYỆN 5-FOLD CNN ÂM TIẾT OUT-OF-FOLD — THẾ HỆ NHÃN v3 (Khối B-1, sửa theo thẩm định K1 16/09).
Chạy trên Kaggle GPU (T4/P100, fp16) hoặc Mac (MPS/CPU, fp32). Kiến trúc + siêu tham số = KhoiB/train_oof_cnn.py.

Đầu vào (cùng thư mục hoặc --crops/--labels):
  crops_v3.npz       X (N,64,64) uint8 + ok + khoá (book,page,column,nom_idx,syl_idx,bbox) + labels_md5   [KhoiB/v3/make_crops_v3.py]
  labels_final.csv   bộ nhãn v3 (83.239 dòng, CÓ nom_idx/syl_idx/bbox/tier/tier_v3)         [dataset_out/labels_final.csv]

Khác bản cũ (4 điểm K1):
  1. KHOÁ = (book,page,column,nom_idx) đọc THẲNG từ csv v3 — không groupby().cumcount(); đối chiếu khoá csv ↔ npz từng dòng.
  2. LƯU models/fold{k}.pt (state + classes) → B-2 dùng được cho hộp bất kỳ (load_fold_model / predict_logprobs bên dưới).
  3. LƯU probs_oof.npz: LP (N×C float16, log-softmax; P = exp(LP)) → B-3 với âm bất kỳ, B-2 phát xạ đủ vector.
  4. Crop cắt từ bbox v3 (crops_v3.npz), fold = md5(f"{book}|{page}") % 5 (không phải "{book}_{page}").
  + p_syl_v3 để TRỐNG (không phải 0) khi âm ∉ lớp → phân biệt được với p≈0 thật; ghi '%.6g' không làm tròn về 0.

Lớp: âm có ≥ --min-count (5) ô tier ∈ {GOLD, SYLLABLE} trên TOÀN BỘ bộ nhãn (không dùng split), cố định cho 5 fold
     (--classes <json> để ép danh sách lớp, ví dụ 706 lớp cũ).
Huấn luyện fold k: ô tier ∈ {GOLD, SYLLABLE} ∧ fold ≠ k ∧ âm ∈ lớp; 15 epoch cố định, val chỉ để in (không chọn mô hình).
Dự đoán OOF: MỌI ô (mọi tier) của fold k.

Đầu ra (--out-dir):
  classes.json, models/fold{0..4}.pt, probs_oof.npz, p_visual_oof_v3.csv, summary.json

    python train_oof_cnn_v3.py --crops crops_v3.npz --labels labels_final.csv --out-dir output            # Kaggle ≈ 12–15 phút
    python train_oof_cnn_v3.py --smoke                                                                    # kiểm mã: 300 ô, 1 epoch, < 3 phút
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SZ = 64
N_FOLDS = 5
TRAIN_TIERS = ("GOLD", "SYLLABLE")
FOLD_FORMULA = 'int(hashlib.md5(f"{book}|{page}".encode()).hexdigest(), 16) % 5'
GENERATION = "v3"


# --------------------------------------------------------------------------- model (y hệt KhoiB/train_oof_cnn.py)
def build_net(n_cls: int, emb: int = 256) -> nn.Module:
    def blk(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.e = nn.Sequential(nn.Linear(256, emb), nn.BatchNorm1d(emb))
            self.c = nn.Linear(emb, n_cls)

        def forward(self, x):
            z = self.e(self.f(x))
            return self.c(z), z

    return Net()


def prep_batch(X: np.ndarray, idx: np.ndarray, dev: torch.device, aug: bool = False) -> torch.Tensor:
    x = X[idx].astype(np.float32) / 255.0
    if aug:
        out = np.empty_like(x)
        for k in range(len(idx)):
            s = 1.0 + random.uniform(-0.08, 0.08)
            tx = random.uniform(-3, 3) + (1.0 - s) * SZ / 2.0
            ty = random.uniform(-3, 3) + (1.0 - s) * SZ / 2.0
            M = np.float32([[s, 0, tx], [0, s, ty]])
            out[k] = cv2.warpAffine(x[k], M, (SZ, SZ), borderValue=1.0)
        x = out
    return torch.from_numpy(1.0 - x).unsqueeze(1).to(dev)      # mực = 1, nền = 0


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def page_fold(book: str, page: str) -> int:
    return int(hashlib.md5(f"{book}|{page}".encode()).hexdigest(), 16) % N_FOLDS


# --------------------------------------------------------------------------- API cho B-2/B-3 (dùng lại mô hình đã lưu)
def load_fold_model(path, dev: torch.device | None = None):
    """-> (net.eval(), classes, dev). Tệp models/fold{k}.pt = {"state", "classes", "fold", ...} (cùng khuôn lab model.pt)."""
    dev = dev or get_device()
    ck = torch.load(path, map_location=dev)
    net = build_net(len(ck["classes"])).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, list(ck["classes"]), dev


@torch.no_grad()
def predict_logprobs(net: nn.Module, X: np.ndarray, dev: torch.device, bs: int = 1024) -> np.ndarray:
    """X (M,64,64) uint8 (cắt bằng cut() của lab) -> log-softmax (M, C) float32."""
    net.eval()
    out = np.zeros((len(X), net.c.out_features), np.float32)
    for k in range(0, len(X), bs):
        idx = np.arange(k, min(len(X), k + bs))
        lg, _ = net(prep_batch(X, idx, dev))
        out[idx] = F.log_softmax(lg.float(), 1).cpu().numpy()
    return out


# --------------------------------------------------------------------------- tiện ích
def rankdata(x: np.ndarray) -> np.ndarray:
    """hạng trung bình khi hoà (như scipy.stats.rankdata)."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    _, inv, cnt = np.unique(sx, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt + 1
    avg = start + (cnt - 1) / 2.0
    r = np.empty(len(x), np.float64)
    r[order] = avg[inv]
    return r


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0,5·P(hoà) — Mann–Whitney."""
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smoke_subset(df: pd.DataFrame, n_cells: int, seed: int) -> np.ndarray:
    """Chọn nguyên CỘT ngẫu nhiên, xoay vòng qua 5 fold (mỗi fold có ô để train/val/OOF), cho tới ≥ n_cells ô
    (nguyên cột để p_left/p_right có hàng xóm)."""
    rng = random.Random(seed)
    by_fold = {f: [] for f in range(N_FOLDS)}
    for (b, p, c), idx in df.groupby(["book", "page", "column"]).indices.items():
        by_fold[page_fold(b, p)].append(idx)
    for f in by_fold:
        rng.shuffle(by_fold[f])
    keep = []
    while len(keep) < n_cells and any(by_fold.values()):
        for f in range(N_FOLDS):
            if by_fold[f]:
                keep.extend(by_fold[f].pop().tolist())
    return np.array(sorted(keep))


def clean_json(o):
    """NaN -> None để summary.json là JSON hợp lệ."""
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="5-fold CNN âm tiết OOF, thế hệ v3")
    ap.add_argument("--crops", default="crops_v3.npz")
    ap.add_argument("--labels", default="labels_final.csv")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-count", type=int, default=5, help="ngưỡng số ô GOLD+SYLLABLE để một âm thành lớp")
    ap.add_argument("--classes", default=None, help="json danh sách lớp cố định (ép), mặc định: rút theo --min-count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="kiểm mã: ~300 ô, 1 epoch, min-count 2")
    ap.add_argument("--smoke-cells", type=int, default=300)
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 1
        args.min_count = min(args.min_count, 2)
        args.bs = min(args.bs, 64)
        if args.out_dir == "output":
            args.out_dir = "_smoke_out"

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); (out_dir / "models").mkdir(parents=True, exist_ok=True)
    dev = get_device()
    use_amp = dev.type == "cuda"
    print("=" * 72)
    print(f"KHỐI B-1 (v3): 5-FOLD CNN OUT-OF-FOLD (ÂM TIẾT){'  [SMOKE]' if args.smoke else ''}")
    print(f"  Device: {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else dev.type}) | amp fp16: {use_amp} | torch {torch.__version__}")
    print(f"  Crops: {args.crops} | Labels: {args.labels} | Epochs/fold: {args.epochs} | bs {args.bs} | lr {args.lr}")
    print("=" * 72, flush=True)

    # 1. nạp + đối chiếu khoá
    t_all = time.time()
    df = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    need = ["book", "page", "column", "nom_idx", "syl_idx", "bbox", "syllable", "tier"]
    miss = [c for c in need if c not in df.columns]
    assert not miss, f"labels thiếu cột {miss} — cần bộ v3 (dataset_out/labels_final.csv 2026-09-16)"
    if "tier_v3" not in df.columns:
        df["tier_v3"] = ""
    z = np.load(args.crops)
    X = z["X"]; ok = z["ok"].astype(bool)
    assert len(df) == len(X) == len(ok), f"N lệch: labels {len(df)} · crops {len(X)}"
    for c in ("book", "page", "column", "nom_idx"):
        if c in z:
            bad = int((z[c].astype(str) != df[c].to_numpy(str)).sum())
            assert bad == 0, f"khoá '{c}' lệch giữa npz và csv ở {bad} dòng — sai thế hệ hoặc sai thứ tự"
    labels_md5 = md5_file(args.labels)
    npz_md5 = str(z["labels_md5"]) if "labels_md5" in z else ""
    if npz_md5 and npz_md5 != labels_md5:
        print(f"  [CẢNH BÁO] labels_md5 trong npz ({npz_md5[:8]}) ≠ md5 csv ({labels_md5[:8]}) — khoá vẫn khớp từng dòng, tiếp tục", flush=True)
    key = df.book + "|" + df.page + "|" + df.column + "|" + df.nom_idx
    assert key.is_unique, "khoá (book,page,column,nom_idx) không duy nhất"
    df["nom_idx_i"] = df.nom_idx.astype(int); df["syl_idx_i"] = df.syl_idx.astype(int)

    sub = None
    if args.smoke:
        sub = smoke_subset(df, args.smoke_cells, args.seed)
        df = df.iloc[sub].reset_index(drop=True); X = X[sub]; ok = ok[sub]
        print(f"  [SMOKE] {len(df)} ô / {df[['book','page','column']].drop_duplicates().shape[0]} cột / {df[['book','page']].drop_duplicates().shape[0]} trang")

    df["fold"] = [page_fold(b, p) for b, p in zip(df.book, df.page)]
    N = len(df)
    pages = df[["book", "page", "fold"]].drop_duplicates(["book", "page"])
    assert (df.groupby(["book", "page"]).fold.nunique() <= 1).all()

    # 2. lớp cố định
    reliable = df.tier.isin(TRAIN_TIERS) & ok
    if args.classes:
        classes = json.load(open(args.classes, encoding="utf-8"))
        src = f"ép từ {args.classes}"
    else:
        cnt = Counter(df.loc[reliable, "syllable"])
        classes = sorted(s for s, n in cnt.items() if n >= args.min_count)
        src = f"âm có ≥{args.min_count} ô tier∈{list(TRAIN_TIERS)} trên toàn bộ (không dùng split)"
    cid = {s: i for i, s in enumerate(classes)}; C = len(classes)
    in_cls = df.syllable.isin(cid).to_numpy()
    assert C >= 2, "quá ít lớp"
    json.dump(classes, open(out_dir / "classes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  Ô: {N:,} (ok {int(ok.sum()):,}) | trang: {len(pages)} | trang/fold: {pages.fold.value_counts().sort_index().to_dict()}")
    print(f"  Lớp: {C} ({src}) | ô GOLD+SYL: {int(reliable.sum()):,}, trong lớp: {int((reliable & in_cls).sum()):,} | âm ∉ lớp: {int((~in_cls).sum()):,}", flush=True)

    # 3. 5 fold
    LP = np.full((N, C), np.nan, np.float16)
    fold_metrics = []
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:                                   # torch cũ
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    y_all = np.array([cid.get(s, -1) for s in df.syllable], np.int64)
    fold_arr = df.fold.to_numpy()
    tier_arr = df.tier.to_numpy()
    for f in range(N_FOLDS):
        print("\n" + "-" * 72 + f"\n>>> FOLD {f} (giữ lại fold {f} · train fold ≠ {f})\n" + "-" * 72, flush=True)
        train_idx = np.where((fold_arr != f) & np.isin(tier_arr, TRAIN_TIERS) & ok & in_cls)[0]
        val_idx = np.where((fold_arr == f) & np.isin(tier_arr, TRAIN_TIERS) & ok & in_cls)[0]
        pred_idx = np.where((fold_arr == f) & ok)[0]
        train_y = y_all[train_idx]; val_y = y_all[val_idx]
        tr_cnt = np.bincount(train_y, minlength=C).astype(np.float64)
        w = 1.0 / np.sqrt(tr_cnt[train_y]); w /= w.sum()
        print(f"  train {len(train_idx):,} · val {len(val_idx):,} · dự đoán OOF {len(pred_idx):,}", flush=True)
        if len(train_idx) == 0 or len(val_idx) == 0:
            print("  [BỎ QUA] fold rỗng"); continue

        net = build_net(C).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        total_steps = args.epochs * math.ceil(len(train_idx) / args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr * 1.5, total_steps=total_steps)
        t0 = time.time(); val_top1 = val_top5 = float("nan")
        for ep in range(args.epochs):
            net.train()
            order = np.random.default_rng(ep + f * 100 + args.seed).choice(len(train_idx), size=len(train_idx), replace=True, p=w)
            tot = 0.0
            for k in range(0, len(order), args.bs):
                sel = order[k:k + args.bs]
                if len(sel) < 2:                        # BatchNorm cần ≥2 mẫu
                    continue
                xb = prep_batch(X, train_idx[sel], dev, aug=True)
                yb = torch.from_numpy(train_y[sel]).to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    logits, _ = net(xb)
                    loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item() * len(sel)
            # val (chỉ in — không chọn mô hình)
            lp_val = predict_logprobs(net, X[val_idx], dev)
            top5 = np.argsort(-lp_val, 1)[:, :5]
            val_top1 = float((top5[:, 0] == val_y).mean()); val_top5 = float((top5 == val_y[:, None]).any(1).mean())
            print(f"  Epoch {ep + 1:2d}/{args.epochs} | loss {tot / max(1, len(order)):.4f} | val top-1 {val_top1:.2%} top-5 {val_top5:.2%} ({len(val_idx)}) | {time.time() - t0:.0f}s", flush=True)

        torch.save({"state": net.state_dict(), "classes": classes, "fold": f, "fold_formula": FOLD_FORMULA, "generation": GENERATION,
                    "labels_md5": labels_md5, "arch": {"emb": 256, "sz": SZ, "input": "1 - x/255 (mực = 1)", "builder": "build_net"},
                    "epochs": args.epochs, "bs": args.bs, "lr": args.lr, "n_train": int(len(train_idx)), "val_top1": val_top1},
                   out_dir / "models" / f"fold{f}.pt")
        LP[pred_idx] = predict_logprobs(net, X[pred_idx], dev).astype(np.float16)
        fold_metrics.append({"fold": f, "pages": int((pages.fold == f).sum()), "train_cells": int(len(train_idx)), "val_cells": int(len(val_idx)),
                             "pred_cells": int(len(pred_idx)), "val_top1": round(val_top1, 4), "val_top5": round(val_top5, 4),
                             "time_sec": round(time.time() - t0, 1)})
        del net, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # 4. bảng OOF
    print("\n" + "=" * 72 + "\nXuất kết quả...", flush=True)
    LPf = LP.astype(np.float32)
    has = ~np.isnan(LPf[:, 0])
    P = np.where(has[:, None], np.exp(LPf), np.nan)
    ar = np.where(has, np.nanargmax(np.where(has[:, None], P, -1.0), 1), -1)
    argmax = np.array([classes[i] if i >= 0 else "" for i in ar], object)
    max_prob = np.where(has, P[np.arange(N), np.clip(ar, 0, C - 1)], np.nan)
    p_syl = np.full(N, np.nan, np.float32)
    m = has & in_cls
    p_syl[m] = P[np.where(m)[0], y_all[m]]
    top5_json = [""] * N
    for i in np.where(has)[0]:
        t5 = np.argsort(-P[i])[:5]
        top5_json[i] = json.dumps([[classes[j], round(float(P[i, j]), 5)] for j in t5], ensure_ascii=False)
    # hàng xóm theo syl_idx cùng cột
    pos = {(b, p, c, s): i for i, (b, p, c, s) in enumerate(zip(df.book, df.page, df.column, df.syl_idx_i))}
    p_left = np.full(N, np.nan, np.float32); p_right = np.full(N, np.nan, np.float32)
    nb_syl_l = np.full(N, "", object); nb_syl_r = np.full(N, "", object)
    for i, (b, p, c, s) in enumerate(zip(df.book, df.page, df.column, df.syl_idx_i)):
        if not has[i]:
            continue
        for d, arr, nb in ((-1, p_left, nb_syl_l), (1, p_right, nb_syl_r)):
            j = pos.get((b, p, c, s + d))
            if j is None:
                continue
            sj = df.syllable.iat[j]; nb[i] = sj
            if sj in cid:
                arr[i] = P[i, cid[sj]]
    out = pd.DataFrame({
        "book": df.book, "page": df.page, "column": df.column, "nom_idx": df.nom_idx, "syl_idx": df.syl_idx, "bbox": df.bbox,
        "fold": df.fold, "tier": df.tier, "tier_v3": df.tier_v3, "syllable_v3": df.syllable, "syl_in_classes": in_cls.astype(int),
        "ok": ok.astype(int), "p_syl_v3": p_syl, "argmax": argmax, "max_prob": max_prob, "top5": top5_json,
        "p_left": p_left, "p_right": p_right,
    })
    out_csv = out_dir / "p_visual_oof_v3.csv"
    out.to_csv(out_csv, index=False, float_format="%.6g")
    np.savez_compressed(out_dir / "probs_oof.npz", LP=LP, classes=np.array(classes), fold=df.fold.to_numpy(np.int8), ok=ok,
                        book=df.book.to_numpy(str), page=df.page.to_numpy(str), column=df.column.to_numpy(str), nom_idx=df.nom_idx.to_numpy(str),
                        labels_md5=np.array(labels_md5), note=np.array("LP = log-softmax float16 (N×C); P = np.exp(LP); NaN = không dự đoán (ok=0)"))
    print(f"  {out_csv} ({len(out):,} dòng, {os.path.getsize(out_csv) / 1e6:.1f} MB) · probs_oof.npz LP {LP.shape}")

    # 5. summary
    hit1 = has & in_cls & (argmax == df.syllable.to_numpy(object))
    top5_hit = np.zeros(N, bool)
    for i in np.where(has & in_cls)[0]:
        top5_hit[i] = y_all[i] in np.argsort(-P[i])[:5]

    def block(mask):
        mm = mask & has & in_cls
        n = int(mm.sum())
        if n == 0:
            return {"n_total": int(mask.sum()), "n_in_classes": 0}
        ps = p_syl[mm]
        return {"n_total": int(mask.sum()), "n_in_classes": n, "top1": round(float(hit1[mm].mean()), 4), "top5": round(float(top5_hit[mm].mean()), 4),
                "p_syl_p50": round(float(np.median(ps)), 4), "p_syl_p10": round(float(np.quantile(ps, .1)), 4),
                "pct_p_syl_lt_0_05": round(float((ps < 0.05).mean()) * 100, 2),
                "pct_argmax_dung_va_max_prob_ge_0_9": round(float((hit1[mm] & (max_prob[mm] >= 0.9)).mean()) * 100, 2)}

    tv3 = df.tier_v3.to_numpy(object); tier = df.tier.to_numpy(object)
    by_tier_v3 = {t: block(tv3 == t) for t in sorted(set(tv3)) if t != ""}
    by_tier = {t: block(tier == t) for t in sorted(set(tier))}
    by_fold = {int(f): block(fold_arr == f) for f in range(N_FOLDS)}
    # E2b: ô CHAR_A — p âm của mình vs p âm ô kề (syl_idx ± 1, cùng cột, âm kề ≠ âm mình, âm kề ∈ lớp)
    ca = (tv3 == "CHAR_A") if (tv3 == "CHAR_A").any() else ((tier == "GOLD") & (df.rule.to_numpy(object) == "s1_inter_s2_direct") if "rule" in df.columns else (tier == "GOLD"))
    right = p_syl[ca & has & in_cls]
    own = df.syllable.to_numpy(object)
    wl = ca & ~np.isnan(p_left) & (nb_syl_l != own); wr = ca & ~np.isnan(p_right) & (nb_syl_r != own)
    wrong = np.concatenate([p_left[wl], p_right[wr]])
    e2b = {"dinh_nghia": "AUC P(âm v3 | crop) của ô CHAR_A (đúng) vs P(âm ô kề syl_idx±1 | crop) của cùng ô (sai); loại kề cùng âm",
           "n_right": int(len(right)), "n_wrong": int(len(wrong)), "n_wrong_prev": int(wl.sum()), "n_wrong_next": int(wr.sum()),
           "n_ke_cung_am_bo": int((ca & ~np.isnan(p_left) & (nb_syl_l == own)).sum() + (ca & ~np.isnan(p_right) & (nb_syl_r == own)).sum()),
           "auc": round(auc(right, wrong), 4), "auc_prev": round(auc(right, p_left[wl]), 4), "auc_next": round(auc(right, p_right[wr]), 4),
           "pct_right_p_lt_0_05": round(float((right < 0.05).mean()) * 100, 2) if len(right) else None,
           "pct_wrong_p_lt_0_05": round(float((wrong < 0.05).mean()) * 100, 2) if len(wrong) else None,
           "theo_fold": {int(f): round(auc(p_syl[ca & has & in_cls & (fold_arr == f)],
                                          np.concatenate([p_left[wl & (fold_arr == f)], p_right[wr & (fold_arr == f)]])), 4) for f in range(N_FOLDS)}}
    rv = (tv3 == "REVIEW") & has
    b3 = {"review_co_p": int(rv.sum()), "argmax_bang_am_v3": int((rv & (argmax == own)).sum()),
          "qua_cong_0_9": int((rv & (argmax == own) & (max_prob >= 0.9)).sum()), "qua_cong_0_8": int((rv & (argmax == own) & (max_prob >= 0.8)).sum())}
    vt1 = [m_["val_top1"] for m_ in fold_metrics]; vt5 = [m_["val_top5"] for m_ in fold_metrics]
    summary = {
        "generation": GENERATION, "smoke": bool(args.smoke), "labels": Path(args.labels).name, "labels_md5": labels_md5, "crops": Path(args.crops).name,
        "n_cells": int(N), "n_ok": int(ok.sum()), "n_pages": int(len(pages)), "n_classes": C, "classes_source": src, "min_count": args.min_count,
        "fold_formula": FOLD_FORMULA, "pages_per_fold": {int(k): int(v) for k, v in pages.fold.value_counts().sort_index().items()},
        "train_tiers": list(TRAIN_TIERS), "epochs": args.epochs, "bs": args.bs, "lr": args.lr, "seed": args.seed,
        "device": str(dev), "amp_fp16": use_amp, "torch": torch.__version__, "total_time_min": round((time.time() - t_all) / 60, 2),
        "mean_val_top1": round(float(np.mean(vt1)), 4) if vt1 else None, "mean_val_top5": round(float(np.mean(vt5)), 4) if vt5 else None,
        "folds": fold_metrics, "by_fold_oof": by_fold, "by_tier_v3": by_tier_v3, "by_tier": by_tier, "e2b": e2b, "b3_gate_review": b3,
        "n_syl_ngoai_lop": int((~in_cls).sum()), "n_syl_ngoai_lop_theo_tier_v3": {t: int(((~in_cls) & (tv3 == t)).sum()) for t in sorted(set(tv3)) if t != ""},
        "outputs": ["classes.json", "models/fold{0..4}.pt", "probs_oof.npz", "p_visual_oof_v3.csv", "summary.json"],
    }
    json.dump(clean_json(summary), open(out_dir / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  mean val top-1 {summary['mean_val_top1']} · top-5 {summary['mean_val_top5']} · E2b AUC {e2b['auc']} "
          f"(right<0,05 {e2b['pct_right_p_lt_0_05']}% · wrong<0,05 {e2b['pct_wrong_p_lt_0_05']}%) · {summary['total_time_min']} phút")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
