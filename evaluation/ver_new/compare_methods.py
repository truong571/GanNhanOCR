"""So sánh các PHƯƠNG PHÁP so-khớp-ảnh thay cho DINOv2 — đo retrieval@1.

DINOv2 hỏng vì là encoder ảnh-tự-nhiên zero-shot (không hợp glyph: retrieval 0%).
Vì glyph tham chiếu nay CÙNG phong cách viết tay (gannhanocr-fd), nhiều cách KHÔNG
CẦN TRAIN trở nên khả thi. Script này, trên các crop VAL (held-out), với tập ứng
viên thật R = {ocr_char} ∪ dict-readings và tham chiếu = glyph tương đồng của mỗi
ứng viên, đo retrieval@1 (chữ true có được xếp #1 không) cho từng phương pháp:

  trained        encoder Nôm (ResNet+ArcFace) — baseline hiện tại (CẦN train)
  pixel_cos      cosine ảnh xám resize 96×96            (không train)
  template_ncc   tương quan chéo chuẩn hoá (NCC)        (không train)
  stroke_grid    lưới mật-độ-nét 12×12, cosine          (không train)
  proj_profile   hình chiếu ngang+dọc, cosine           (không train)
  dice           chồng nhị phân (Dice)                  (không train)
  hog            HOG (hướng gradient), cosine           (không train)
  dinov2         (tuỳ chọn --dinov2) encoder DINOv2 — để tái lập "sàn" 0%

Run:
  .venv/bin/python evaluation/ver_new/compare_methods.py --n 600      # [--dinov2]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
from evaluation.ver_new.bbox_fix import tighten_box                # noqa: E402

SZ = 96


def prep(gray):
    """-> (gray_float 96×96 in [0,1], binary bool 96×96, ink-tightened)."""
    tb = tighten_box(gray)
    if tb is not None:
        a, c, b, d = tb
        if b - a >= 6 and d - c >= 6:
            gray = gray[c:d, a:b]
    g = cv2.resize(gray, (SZ, SZ), interpolation=cv2.INTER_AREA)
    gf = g.astype(np.float32) / 255.0
    bw = g < 128
    return gf, bw


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0


# ---- non-learned feature vectors (higher cosine = more similar) ----
def f_pixel(gf, bw):
    return 1.0 - gf                                  # ink=1, paper=0

def f_grid(gf, bw, k=12):
    cell = SZ // k
    return bw[:k * cell, :k * cell].reshape(k, cell, k, cell).mean(axis=(1, 3)).ravel()

def f_proj(gf, bw):
    return np.concatenate([bw.mean(axis=1), bw.mean(axis=0)])

_hog = cv2.HOGDescriptor((SZ, SZ), (32, 32), (16, 16), (16, 16), 9)
def f_hog(gf, bw):
    return _hog.compute((gf * 255).astype(np.uint8)).ravel()

VEC = {"pixel_cos": f_pixel, "stroke_grid": f_grid, "proj_profile": f_proj, "hog": f_hog}


# ---- pairwise scores ----
def s_ncc(cg, gg):
    return float(cv2.matchTemplate(cg, gg, cv2.TM_CCOEFF_NORMED)[0, 0])

def s_dice(cb, gb):
    inter = np.logical_and(cb, gb).sum()
    tot = cb.sum() + gb.sum()
    return float(2 * inter / tot) if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "dataset_out"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--dinov2", action="store_true", help="cũng đo DINOv2 (tải model; có thể chậm)")
    args = ap.parse_args()
    random.seed(0)

    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]))
    fd_index = vs3.fd_index

    dino = None
    if args.dinov2:
        try:
            from core.ranking.dinov2_ranker import DINOv2Ranker
            dino = DINOv2Ranker()
            print("  DINOv2 loaded", flush=True)
        except Exception as e:
            print(f"  [DINOv2 OFF] {type(e).__name__}: {e}", flush=True)

    D = Path(args.dataset)
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == "GOLD" and r["split"] == "val" and r["image"] and r["syllable"]]
    random.shuffle(rows); rows = rows[:args.n]
    print(f"VAL crops: {len(rows)}\n", flush=True)

    methods = ["trained"] + list(VEC) + ["template_ncc", "dice"] + (["dinov2"] if dino else [])
    hit = {m: 0 for m in methods}
    n = 0
    gcache = {}                                       # char -> prepped glyph feats

    def glyph_feats(ch):
        if ch in gcache:
            return gcache[ch]
        p = fd_index.get(ch)
        if not p:
            gcache[ch] = None; return None
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            gcache[ch] = None; return None
        gf, bw = prep(g)
        feats = {k: fn(gf, bw) for k, fn in VEC.items()}
        feats["_gf"], feats["_bw"] = gf, bw
        feats["_emb"] = vs3.enc.embed_path(str(p))
        feats["_dino"] = dino.embed_path(str(p)) if dino else None
        gcache[ch] = feats
        return feats

    for r in rows:
        cg_full = cv2.imread(str(D / r["image"]), cv2.IMREAD_GRAYSCALE)
        if cg_full is None:
            continue
        cgf, cbw = prep(cg_full)
        cvecs = {k: fn(cgf, cbw) for k, fn in VEC.items()}
        cemb = vs3.enc.embed_gray((cgf * 255).astype(np.uint8))
        cdino = dino.embed_gray((cgf * 255).astype(np.uint8)) if dino else None
        true = r["label"]
        cands = []
        for c in ([r["ocr_char"]] if r["ocr_char"] else []) + qn.get((r["syllable"] or "").lower(), []):
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if true not in cands:
            cands.append(true)
        cands = [c for c in cands if glyph_feats(c) is not None]
        if len(cands) < 2 or true not in cands:
            continue
        n += 1
        for m in methods:
            best_c, best_s = None, -1e9
            for c in cands:
                gf = gcache[c]
                if m == "trained":
                    s = cos(cemb, gf["_emb"]) if gf["_emb"] is not None else -1
                elif m == "dinov2":
                    s = cos(cdino, gf["_dino"]) if gf["_dino"] is not None else -1
                elif m == "template_ncc":
                    s = s_ncc(cgf, gf["_gf"])
                elif m == "dice":
                    s = s_dice(cbw, gf["_bw"])
                else:
                    s = cos(cvecs[m], gf[m])
                if s > best_s:
                    best_s, best_c = s, c
            hit[m] += (best_c == true)
        if n % 150 == 0:
            print(f"  ... {n}", flush=True)

    print(f"\n=== retrieval@1 trên {n} quyết định VAL (tham chiếu = glyph tương đồng) ===")
    print(f"  {'phương pháp':16s} {'train?':8s} retrieval@1")
    need = {"trained": "có", "dinov2": "không"}
    for m in sorted(methods, key=lambda x: -hit[x]):
        tr = need.get(m, "không")
        print(f"  {m:16s} {tr:8s} {hit[m]/max(n,1):6.1%}  ({hit[m]}/{n})")
    print("\n  'trained' = encoder hiện tại (baseline). Các dòng 'không' = KHÔNG cần train,")
    print("  thử được ngay; nếu sát baseline -> có thể dùng làm fallback / ensemble không-train.")


if __name__ == "__main__":
    main()
