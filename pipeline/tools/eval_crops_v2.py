"""D-1 · ĐO crops_v2 (F3g) so với crop giao nộp v1 — KHÔNG CẦN NGƯỜI.

Đầu vào: labels_final.csv (v1: bbox, image, crop_quality_flag/stray/border), thư mục crop v1 (dataset_out hoặc
dataset_out_v3), sidecar labels_crops_v2.csv + thư mục crops_v2 (pipeline/tools/recrop_v2.py), CNN OOF Khối B-1'
(KhoiB/v3/p_visual_oof_v3_results/models/fold{0..4}.pt, fold = md5("book|page") % 5 -> mô hình KHÔNG học trang),
mini-GT hộp: config/qd01_cells.csv (2.012 ô bbox_cu người đã xem) + KhoiB/v3/b5/qd01_3nhanh_lech_117.csv.

Đo (ghi <out>/eval_crops_v2.json + in bảng):
 (i)   tỉ lệ f3g/fallback, phân bố |recenter_shift| (p50/p90, >0,15/0,25/0,40p), lý do fallback
 (ii)  cờ crop_quality v1 vs v2 (ok/bleed/truncated/blank), stray/border p95
 (iii) THƯỚC ĐO KHÔNG NGƯỜI: p_syl OOF của CNN trên 4 biến thể crop của CÙNG ô usable (GOLD+SYLLABLE, âm ∈ lớp):
         v1_page = cut(trang, bbox, pad 0,08)       (= crops_v3.npz đã huấn luyện/OOF — kiểm khớp p_visual_oof_v3.csv)
         v1_file = cut(tệp crop v1)                 (crop giao nộp: pad 0,12 + carve + tighten)
         v2_page = cut(trang, bbox_v2, pad 0,08)    (hộp tái định tâm, KHÔNG seam)
         v2_file = cut(tệp crop v2)                 (F3g đầy đủ: nới 0,10p + seam ±0,10p + tighten)
       top-1 (argmax == âm), trung vị p_syl, % p ≥ 0,9, mean log p; theo tầng shift (>0,25p; fallback) và tier.
       McNemar (b/c) v2_file vs v1_file trên top-1.
 (iv)  mini-GT hộp: 2.012 ô QĐ-01: IoU(bbox_v2, bbox_cu), IoU(bbox_v2_win, bbox_cu), |Δtâm|/pitch, % IoU<0,5;
       117 ô lệch: F3g(bbox_truoc_lock) có tiến về bbox_cu không (IoU/|Δtâm| trước vs sau).
 (v)   crop→crop 1-NN chéo trang (kiểu hướng C: 90 nhãn, ≤40 ô/nhãn, ≤3 ô/trang, seed) — pixel 48×48 và
       đặc trưng 576 chiều không huấn luyện (bóng mờ 16×16 σ1,5 + 8×8 σ4 + gradient 4 hướng lưới 8×8), cosine;
       so với kỳ vọng hướng C: 576-d 34,5→57,2 %, pixel 21,9→43,5 %; cờ ok 90,8→98,2 %; rơi về 5,2 %.
 (vi)  crop→GLYPH PHÔNG (máy đo không huấn luyện, KHÔNG vòng tròn với hộp/CNN): kho tham chiếu = mọi nhãn Nôm của ô
       usable render bằng 6 họ phông (HAN NOM A/B, Kai, Khai, Minh, HanaMin A/B, NomNaTong; loại tofu; ≥2 họ), cùng
       cut 64×64 + feat576, cosine max theo phông. (a) top-1/top-5/hạng trung vị nhãn của chính ô (hướng C: 1,06→5,87 %);
       (b) PHÉP THỬ NHẦM CHỮ: crop giống glyph nhãn CHÍNH NÓ hơn hay giống glyph nhãn HÀNG XÓM (trên/dưới) hơn —
       tỉ lệ "thua hàng xóm" v1 vs v2 theo tầng |shift|; nếu v2 thua hàng xóm nhiều hơn v1 ở ô shift lớn -> F3g khoá vào chữ kề.
 (vii) OOF ĐỐI XỨNG (tuỳ chọn --oof-v2 / --oof-v1file): CNN huấn luyện LẠI 5-fold trên chính crop v2 (và đối chứng crop
       tệp v1) bằng KhoiB/v3/train_oof_cnn_v3.py + make_crops_v2_npz.py, cùng seed/epoch; so OOF top-1/p50 trên cùng ô với
       bản v1 chính thức (crops_v3.npz). Cộng thêm tỉ lệ argmax OOF == ÂM HÀNG XÓM (trên/dưới) theo tầng shift — ước
       lượng "crop đang cho thấy chữ kề" không cần người (mô hình v1 nhìn crop v1; mô hình v2 nhìn crop v2).
 + HTML 20 crop mẫu v1|v2 cạnh nhau (+ 10 ô QĐ-01 IoU thấp nhất) -> <out>/crops_v2_sample.html

    .venv/bin/python -m pipeline.tools.eval_crops_v2 --labels dataset_out_v3/labels_final.csv --v1-root dataset_out_v3 \
        --v2-root <out của recrop_v2> --out dataset_out/khoi_d
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline.align_engine import recenter_f3g as rf                              # noqa: E402
from pipeline.align_engine.visual_emission import cut_box, page_fold, SZ, CUT_PAD  # noqa: E402

BOOKDIR = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}
USABLE = ("GOLD", "SYLLABLE")


def _bb(s):
    try:
        b = json.loads(s)
        return [int(v) for v in b] if b and len(b) == 4 else None
    except Exception:
        return None


def iou(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cy(b):
    return (b[1] + b[3]) / 2.0


def pct(x, q):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else float("nan")


def cut_file(path: Path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None or g.size == 0:
        return None
    h, w = g.shape[:2]
    return cut_box(g, [0, 0, w, h], pad=CUT_PAD)


# --------------------------------------------------------------------------- (iii) CNN OOF
def cnn_scores(df: pd.DataFrame, X: dict[str, np.ndarray], okm: dict[str, np.ndarray], models_dir: Path,
               bs: int = 1024):
    """df: các ô (có cột book,page,syllable). X[variant] (N,64,64). Trả dict variant -> (p_syl, argmax_is_syl,
    logp) với NaN khi ô không cắt được/âm ∉ lớp."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(REPO / "KhoiB/v3"))
    from train_oof_cnn_v3 import build_net, get_device
    dev = get_device()
    folds = np.array([page_fold(b, p) for b, p in zip(df.book, df.page)])
    out = {}
    nets = {}
    for k in range(5):
        ck = torch.load(models_dir / f"fold{k}.pt", map_location=dev)
        net = build_net(len(ck["classes"])).to(dev)
        net.load_state_dict(ck["state"]); net.eval()
        nets[k] = (net, list(ck["classes"]))
    classes = nets[0][1]
    cid = {s: i for i, s in enumerate(classes)}
    syl_idx = np.array([cid.get(str(s).lower(), -1) for s in df.syllable])
    for name, Xv in X.items():
        p_syl = np.full(len(df), np.nan, np.float32)
        top1 = np.full(len(df), np.nan, np.float32)
        logp = np.full(len(df), np.nan, np.float32)
        for k in range(5):
            idx = np.where((folds == k) & okm[name] & (syl_idx >= 0))[0]
            net, cls = nets[k]
            assert cls == classes
            with torch.no_grad():
                for s in range(0, len(idx), bs):
                    sel = idx[s:s + bs]
                    x = torch.from_numpy(1.0 - Xv[sel].astype(np.float32) / 255.0).unsqueeze(1).to(dev)
                    lg, _ = net(x)
                    LP = F.log_softmax(lg.float(), 1).cpu().numpy()
                    lp = LP[np.arange(len(sel)), syl_idx[sel]]
                    logp[sel] = lp
                    p_syl[sel] = np.exp(lp)
                    top1[sel] = (LP.argmax(1) == syl_idx[sel]).astype(np.float32)
        out[name] = {"p_syl": p_syl, "top1": top1, "logp": logp}
    return out, classes


def summarize(mask, S, names):
    r = {}
    for n in names:
        p, t, l = S[n]["p_syl"][mask], S[n]["top1"][mask], S[n]["logp"][mask]
        ok = ~np.isnan(p)
        r[n] = {"n": int(ok.sum()), "top1": float(np.nanmean(t)) if ok.any() else None,
                "p50": float(np.nanmedian(p)) if ok.any() else None,
                "p_ge_0.9": float(np.nanmean(p >= 0.9)) if ok.any() else None,
                "mean_logp": float(np.nanmean(l)) if ok.any() else None}
    return r


def mcnemar(a, b):
    """a, b: 0/1 (NaN bỏ). Trả (b_only_right, a_only_right, p hai phía nhị thức chính xác)."""
    from math import comb
    m = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[m] > 0.5, b[m] > 0.5
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p


# --------------------------------------------------------------------------- (v) crop→crop 1-NN
def feat576(X: np.ndarray) -> np.ndarray:
    """Đặc trưng không huấn luyện kiểu KB 0/hướng C: bóng mờ 16×16 (σ1,5) + 8×8 (σ4) + gradient 4 hướng lưới 8×8."""
    N = len(X)
    out = np.zeros((N, 256 + 64 + 256), np.float32)
    for i in range(N):
        g = 1.0 - X[i].astype(np.float32) / 255.0
        b1 = cv2.resize(cv2.GaussianBlur(g, (0, 0), 1.5), (16, 16), interpolation=cv2.INTER_AREA).ravel()
        b2 = cv2.resize(cv2.GaussianBlur(g, (0, 0), 4.0), (8, 8), interpolation=cv2.INTER_AREA).ravel()
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        dirs = [np.maximum(gx, 0), np.maximum(-gx, 0), np.maximum(gy, 0), np.maximum(-gy, 0)]
        gr = np.concatenate([cv2.resize(d, (8, 8), interpolation=cv2.INTER_AREA).ravel() for d in dirs])
        out[i] = np.concatenate([b1, b2, gr])
    return out


def nn_cross_page(F: np.ndarray, labels: np.ndarray, pages: np.ndarray) -> float:
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T
    same_page = pages[:, None] == pages[None, :]
    S[same_page] = -np.inf
    nn = S.argmax(1)
    return float((labels[nn] == labels).mean())


def sample_c2c(df: pd.DataFrame, seed: int, n_labels: int = 90, per_label: int = 40, per_page: int = 3):
    rng = np.random.default_rng(seed)
    d = df[(df.tier.isin(USABLE)) & (df.label != "")]
    cnt = d.label.value_counts()
    labs = cnt[cnt >= 20].index.tolist()
    labs = [labs[i] for i in rng.permutation(len(labs))[:n_labels]]
    rows = []
    for lb in labs:
        g = d[d.label == lb].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        per = Counter(); k = 0
        for r in g.itertuples():
            key = (r.book, r.page)
            if per[key] >= per_page:
                continue
            per[key] += 1; rows.append(r.Index); k += 1
            if k >= per_label:
                break
    return d.loc[rows]


# --------------------------------------------------------------------------- (vi) crop→glyph phông
FONTS = [("hannom", "HAN NOM A.ttf"), ("hannom", "HAN NOM B.ttf"), ("kai", "Han-Nom Kai 1.00.otf"),
         ("khai", "Han-Nom-Khai-Regular-300623.ttf"), ("minh", "Han-nom Minh 1.42.otf"),
         ("hanamin", "HanaMinA.ttf"), ("hanamin", "HanaMinB.ttf"), ("nomnatong", "NomNaTong-Regular2.otf")]


def glyph_bank(labels: list[str], font_dir: Path, min_families: int = 2):
    """Render mỗi nhãn bằng các phông (96×96, loại tofu = giống render U+0378 hoặc trắng) -> cut 64×64 -> feat576.
    Trả (F (M,576), owner (M,) chỉ số nhãn, kept_labels)."""
    from PIL import Image, ImageDraw, ImageFont
    fts = []
    for fam, fn in FONTS:
        p = font_dir / fn
        if p.exists():
            ft = ImageFont.truetype(str(p), 80)
            im = Image.new("L", (96, 96), 255); ImageDraw.Draw(im).text((48, 48), "\u0378", font=ft, fill=0, anchor="mm")
            fts.append((fam, ft, np.array(im)))
    feats, owner, fam_cnt = [], [], defaultdict(set)
    tiles = []
    for li, lb in enumerate(labels):
        for fam, ft, tofu in fts:
            im = Image.new("L", (96, 96), 255); ImageDraw.Draw(im).text((48, 48), lb, font=ft, fill=0, anchor="mm")
            g = np.array(im)
            if (g < 128).sum() < 20 or np.array_equal(g, tofu):
                continue
            c = cut_box(g, [0, 0, 96, 96], pad=CUT_PAD)
            if c is None:
                continue
            tiles.append(c); owner.append(li); fam_cnt[li].add(fam)
    if not tiles:
        return None, None, []
    F = feat576(np.stack(tiles)); owner = np.array(owner)
    keep_lab = {li for li, fams in fam_cnt.items() if len(fams) >= min_families}
    m = np.array([o in keep_lab for o in owner])
    return F[m], owner[m], sorted(keep_lab)


def glyph_eval(us: pd.DataFrame, X: dict, okm: dict, names: list[str], nb_label: dict, font_dir: Path,
               groups: dict):
    """groups: {tag: chỉ số vị trí trong us}. Mỗi tag: top-1/top-5/hạng nhãn chính ô + tỉ lệ thua glyph hàng xóm."""
    labels = sorted(set(us.label[us.label != ""]) | {l for v in nb_label.values() for l in v if l})
    F, owner, kept = glyph_bank(labels, font_dir)
    if F is None:
        return {"error": "không render được phông"}
    kept_set = {labels[i] for i in kept}          # kept = CHỈ SỐ nhãn (≥2 họ phông)
    kept_pos = {li: k for k, li in enumerate(kept)}
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    lid = {lb: i for i, lb in enumerate(labels)}
    R = {"n_labels_bank": len(kept), "n_font_vectors": int(len(F)), "families": sorted({f for f, _ in FONTS})}
    def sims(Xs):
        Q = feat576(Xs); Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        S = Q @ Fn.T                                       # (n, M)
        # max theo phông -> (n, n_labels)
        best = np.full((len(Xs), len(labels)), -1.0, np.float32)
        for j in range(len(owner)):
            np.maximum(best[:, owner[j]], S[:, j], out=best[:, owner[j]])
        return best
    for tag, idx in groups.items():
        idx = np.array([i for i in idx if us.label.iloc[i] in kept_set], dtype=int)
        R[tag] = {"n": int(len(idx))}
        if len(idx) == 0:
            continue
        own = np.array([lid[us.label.iloc[i]] for i in idx])
        keys = [(us.book.iloc[i], us.page.iloc[i], us.column.iloc[i], int(us.nom_idx.iloc[i])) for i in idx]
        prv = np.array([lid.get(nb_label.get(k, (None, None))[0] or "", -1) for k in keys])
        nxt = np.array([lid.get(nb_label.get(k, (None, None))[1] or "", -1) for k in keys])
        has_nb = (prv >= 0) | (nxt >= 0)
        for n in names:
            ok = okm[n][idx]
            B = sims(X[n][idx])
            B_kept = B[:, kept]                                # xếp hạng trong kho ≥2 họ phông
            own_kept = np.array([kept_pos[o] for o in own])
            s_own = B_kept[np.arange(len(idx)), own_kept]
            rank = (B_kept > s_own[:, None]).sum(1) + 1
            s_p = np.where(prv >= 0, B[np.arange(len(idx)), np.clip(prv, 0, None)], -9)
            s_n = np.where(nxt >= 0, B[np.arange(len(idx)), np.clip(nxt, 0, None)], -9)
            lose = (np.maximum(s_p, s_n) > s_own) & has_nb & ok
            R[tag][n] = {"top1": float((rank[ok] == 1).mean()), "top5": float((rank[ok] <= 5).mean()),
                         "rank_p50": float(np.median(rank[ok])),
                         "lose_to_neighbor": float(lose[has_nb & ok].mean()), "n_with_neighbor": int((has_nb & ok).sum())}
    return R


# --------------------------------------------------------------------------- HTML
def b64png(path: Path) -> str:
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return ""


def write_html(rows: list[dict], out: Path, title: str, sub: str):
    css = ("body{font-family:system-ui;margin:16px;background:#fafafa}table{border-collapse:collapse}"
           "td,th{border:1px solid #ccc;padding:4px 8px;text-align:center;font-size:13px;vertical-align:top}"
           "img{max-height:150px;image-rendering:pixelated;background:#fff}h1{font-size:18px}p{font-size:13px;color:#444}")
    h = [f"<!doctype html><meta charset=utf-8><title>{title}</title><style>{css}</style><h1>{title}</h1><p>{sub}</p>",
         "<table><tr><th>#</th><th>ô</th><th>nhãn / âm</th><th>crop v1 (giao nộp)</th><th>crop v2 (F3g)</th>"
         "<th>crop_mode / shift (pitch)</th><th>cờ v1 → v2</th><th>p_syl OOF v1_file → v2_file</th><th>ghi chú</th></tr>"]
    for i, r in enumerate(rows, 1):
        h.append(f"<tr><td>{i}</td><td>{r['image']}</td><td style='font-size:22px'>{r.get('label','')}<br>"
                 f"<span style='font-size:12px'>{r.get('syllable','')}</span></td>"
                 f"<td><img src='{r['v1']}'></td><td><img src='{r['v2']}'></td>"
                 f"<td>{r.get('crop_mode','')} / {r.get('shift','')}</td><td>{r.get('flag1','')} → {r.get('flag2','')}</td>"
                 f"<td>{r.get('p1','')} → {r.get('p2','')}</td><td>{r.get('note','')}</td></tr>")
    h.append("</table>")
    out.write_text("\n".join(h), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(REPO / "dataset_out/labels_final.csv"))
    ap.add_argument("--v1-root", default=str(REPO / "dataset_out"))
    ap.add_argument("--v2-root", default=str(REPO / "dataset_out/khoi_d/crops_v2"))
    ap.add_argument("--sidecar", default=None, help="mặc định <v2-root>/labels_crops_v2.csv")
    ap.add_argument("--oof-csv", default=str(REPO / "KhoiB/v3/p_visual_oof_v3_results/p_visual_oof_v3.csv"))
    ap.add_argument("--models", default=str(REPO / "KhoiB/v3/p_visual_oof_v3_results/models"))
    ap.add_argument("--qd01", default=str(REPO / "config/qd01_cells.csv"))
    ap.add_argument("--lech117", default=str(REPO / "KhoiB/v3/b5/qd01_3nhanh_lech_117.csv"))
    ap.add_argument("--pages-dir", default=str(REPO / "prepared"))
    ap.add_argument("--out", default=str(REPO / "dataset_out/khoi_d"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--n-sample", type=int, default=20)
    ap.add_argument("--oof-v2", default=None, help="thư mục kết quả train_oof_cnn_v3 trên crops_v2.npz (mục vii)")
    ap.add_argument("--oof-v1file", default=None, help="thư mục kết quả train_oof_cnn_v3 trên crops_v1file.npz (đối chứng)")
    ap.add_argument("--skip-glyph", action="store_true")
    ap.add_argument("--font-dir", default=str(REPO / "font_diffusion/fonts"))
    ap.add_argument("--n-glyph", type=int, default=6000, help="số ô mỗi tầng shift cho (vi)")
    args = ap.parse_args(argv)

    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    v1_root, v2_root = Path(args.v1_root), Path(args.v2_root)
    sidecar = Path(args.sidecar) if args.sidecar else v2_root / "labels_crops_v2.csv"
    lab_md5 = hashlib.md5(Path(args.labels).read_bytes()).hexdigest()
    df = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    sc = pd.read_csv(sidecar, dtype=str, keep_default_na=False)
    sc_md5 = hashlib.md5(sidecar.read_bytes()).hexdigest()
    m = df.merge(sc[["image", "bbox_v2", "bbox_v2_win", "crop_mode", "guard_reason", "prior_used", "recenter_shift",
                     "pitch", "image_md5_v2", "ink_pct_v2", "crop_quality_flag_v2", "stray_ink_v2", "border_ink_v2"]],
                 on="image", how="inner")
    m = m[m.image != ""].reset_index(drop=True)
    m["shift"] = pd.to_numeric(m.recenter_shift, errors="coerce")
    m["ashift"] = m["shift"].abs()
    us = m[m.tier.isin(USABLE)].reset_index(drop=True)
    R = {"labels": args.labels, "labels_md5": lab_md5, "sidecar": str(sidecar), "sidecar_md5": sc_md5,
         "n_with_crop": int(len(m)), "n_usable": int(len(us)), "seed": args.seed,
         "params": {k: getattr(rf, k) for k in ("H_RANGE", "H_TARGET", "TOP_SEARCH", "LAMBDA_H", "LAMBDA_C",
                                                 "GUARD_SHIFT", "EXT", "SEAM_BAND", "NEIGHBOR_OK")}}

    # (i) --------------------------------------------------------------
    modes = us.crop_mode.value_counts().to_dict()
    sh = us.ashift.to_numpy()
    R["i_mode"] = {"crop_mode": modes, "pct_f3g": round(100 * modes.get("f3g", 0) / len(us), 2),
                   "fallback_reason": us[us.crop_mode == "fallback"].guard_reason.value_counts().to_dict(),
                   "prior_used_pct": round(100 * float(pd.to_numeric(us.prior_used, errors="coerce").mean()), 2),
                   "abs_shift": {"p50": pct(sh, 50), "p90": pct(sh, 90), "gt_0.15": float(np.nanmean(sh > 0.15)),
                                 "gt_0.25": float(np.nanmean(sh > 0.25)), "gt_0.40": float(np.nanmean(sh > 0.40))},
                   "abs_shift_f3g_only": {"p50": pct(us[us.crop_mode == "f3g"].ashift, 50),
                                          "p90": pct(us[us.crop_mode == "f3g"].ashift, 90)},
                   "by_box_source": {k: {"n": int(len(g)), "pct_f3g": round(100 * float((g.crop_mode == "f3g").mean()), 2),
                                         "shift_gt_0.25": round(100 * float((g.ashift > 0.25).mean()), 2)}
                                     for k, g in us.groupby("box_source")},
                   "md5_changed_pct": round(100 * float((us.image_md5_v2 != us.image_md5).mean()), 2)}
    print(f"(i) usable {len(us):,}: f3g {R['i_mode']['pct_f3g']}% · fallback {modes.get('fallback',0):,} "
          f"{R['i_mode']['fallback_reason']} · |shift| p50 {R['i_mode']['abs_shift']['p50']:.3f} p90 "
          f"{R['i_mode']['abs_shift']['p90']:.3f} · >0,25p {100*R['i_mode']['abs_shift']['gt_0.25']:.1f}% · "
          f">0,40p {100*R['i_mode']['abs_shift']['gt_0.40']:.1f}% · md5 đổi {R['i_mode']['md5_changed_pct']}%", flush=True)

    # (ii) -------------------------------------------------------------
    f1 = us.crop_quality_flag.value_counts(normalize=True).round(4).to_dict()
    f2 = us.crop_quality_flag_v2.value_counts(normalize=True).round(4).to_dict()
    s1 = pd.to_numeric(us.stray_ink, errors="coerce"); s2 = pd.to_numeric(us.stray_ink_v2, errors="coerce")
    b1 = pd.to_numeric(us.border_ink, errors="coerce"); b2 = pd.to_numeric(us.border_ink_v2, errors="coerce")
    R["ii_quality"] = {"flag_v1": f1, "flag_v2": f2,
                       "flag_v2_f3g_only": us[us.crop_mode == "f3g"].crop_quality_flag_v2.value_counts(normalize=True).round(4).to_dict(),
                       "stray_p95": [pct(s1, 95), pct(s2, 95)], "border_p95": [pct(b1, 95), pct(b2, 95)],
                       "crop_h_px_p50": [pct(pd.to_numeric(us.crop_h, errors="coerce"), 50),
                                         pct(pd.to_numeric(sc.crop_h_v2, errors="coerce"), 50)],
                       "transition": pd.crosstab(us.crop_quality_flag, us.crop_quality_flag_v2).to_dict()}
    print(f"(ii) cờ ok v1 {100*f1.get('ok',0):.1f}% → v2 {100*f2.get('ok',0):.1f}% · bleed {100*f1.get('bleed',0):.1f}→"
          f"{100*f2.get('bleed',0):.1f} · truncated {100*f1.get('truncated',0):.2f}→{100*f2.get('truncated',0):.2f} · "
          f"stray p95 {pct(s1,95):.3f}→{pct(s2,95):.3f} · border p95 {pct(b1,95):.3f}→{pct(b2,95):.3f}", flush=True)

    # (iii) ------------------------------------------------------------
    if not args.skip_cnn:
        N = len(us)
        X = {k: np.zeros((N, SZ, SZ), np.uint8) for k in ("v1_page", "v1_file", "v2_page", "v2_file")}
        okm = {k: np.zeros(N, bool) for k in X}
        us["bb"] = us.bbox.apply(_bb); us["bb2"] = us.bbox_v2.apply(_bb)
        for (book, page), g in us.groupby(["book", "page"], sort=False):
            gray = cv2.imread(str(Path(args.pages_dir) / BOOKDIR[book] / "pages" / f"{page}.png"), cv2.IMREAD_GRAYSCALE)
            for i in g.index:
                r = us.loc[i]
                if gray is not None:
                    c = cut_box(gray, r.bb);
                    if c is not None: X["v1_page"][i] = c; okm["v1_page"][i] = True
                    c = cut_box(gray, r.bb2) if r.bb2 else None
                    if c is not None: X["v2_page"][i] = c; okm["v2_page"][i] = True
                c = cut_file(v1_root / r.image)
                if c is not None: X["v1_file"][i] = c; okm["v1_file"][i] = True
                c = cut_file(v2_root / r.image)
                if c is not None: X["v2_file"][i] = c; okm["v2_file"][i] = True
        print(f"(iii) cắt 4 biến thể × {N:,} ô: " + " · ".join(f"{k} {int(v.sum()):,}" for k, v in okm.items())
              + f" · {time.time()-t0:.0f}s", flush=True)
        S, classes = cnn_scores(us, X, okm, Path(args.models))
        names = list(X)
        # kiểm khớp v1_page với p_visual_oof_v3.csv (cùng crop, cùng mô hình; MPS fp32 vs CUDA fp16)
        oof = pd.read_csv(args.oof_csv, dtype=str, keep_default_na=False)
        oof = oof[["book", "page", "column", "nom_idx", "p_syl_v3"]]
        chk = us[["book", "page", "column", "nom_idx"]].copy(); chk["p_here"] = S["v1_page"]["p_syl"]
        chk = chk.merge(oof, on=["book", "page", "column", "nom_idx"], how="left")
        pv = pd.to_numeric(chk.p_syl_v3, errors="coerce")
        ok = ~np.isnan(chk.p_here) & ~pv.isna()
        diff = np.abs(chk.p_here[ok] - pv[ok])
        R["iii_check_vs_oof_csv"] = {"n": int(ok.sum()), "abs_diff_p50": float(diff.median()),
                                     "abs_diff_p99": float(diff.quantile(0.99)), "abs_diff_max": float(diff.max())}
        print(f"(iii) khớp p_visual_oof_v3.csv: n {int(ok.sum()):,} |Δp| p50 {diff.median():.2e} p99 {diff.quantile(0.99):.2e} max {diff.max():.3f}")
        allm = np.ones(N, bool)
        strata = {"all_usable": allm, "f3g": (us.crop_mode == "f3g").to_numpy(),
                  "fallback": (us.crop_mode == "fallback").to_numpy(),
                  "shift_gt_0.25": (us.ashift > 0.25).to_numpy() & (us.crop_mode == "f3g").to_numpy(),
                  "shift_le_0.10": (us.ashift <= 0.10).to_numpy() & (us.crop_mode == "f3g").to_numpy(),
                  "GOLD": (us.tier == "GOLD").to_numpy(), "SYLLABLE": (us.tier == "SYLLABLE").to_numpy(),
                  "v1_flag_bleed": (us.crop_quality_flag == "bleed").to_numpy(),
                  "v1_flag_truncated": (us.crop_quality_flag == "truncated").to_numpy()}
        R["iii_cnn"] = {k: summarize(v, S, names) for k, v in strata.items()}
        R["iii_cnn"]["n_classes"] = len(classes)
        for k in ("all_usable", "shift_gt_0.25", "fallback"):
            row = R["iii_cnn"][k]
            print(f"(iii) {k:14s} n={row['v1_file']['n']:,} top1 " + " ".join(f"{n} {100*row[n]['top1']:.2f}%" for n in names)
                  + " | p50 " + " ".join(f"{row[n]['p50']:.3f}" for n in names)
                  + " | p≥0,9 " + " ".join(f"{100*row[n]['p_ge_0.9']:.1f}%" for n in names), flush=True)
        R["iii_mcnemar_top1"] = {}
        for a, b in (("v1_file", "v2_file"), ("v1_page", "v2_page"), ("v1_page", "v2_file")):
            for k in ("all_usable", "shift_gt_0.25"):
                mk = strata[k]
                n01, n10, p = mcnemar(S[a]["top1"][mk], S[b]["top1"][mk])
                R["iii_mcnemar_top1"][f"{b}_vs_{a}@{k}"] = {"b_right_a_wrong": n01, "a_right_b_wrong": n10, "p": p}
                print(f"(iii) McNemar {b} vs {a} @{k}: +{n01} / −{n10}, p={p:.2e}")
        us["p1_file"] = S["v1_file"]["p_syl"]; us["p2_file"] = S["v2_file"]["p_syl"]
        us["p1_page"] = S["v1_page"]["p_syl"]; us["p2_page"] = S["v2_page"]["p_syl"]
        # (v) crop→crop 1-NN chéo trang
        smp = sample_c2c(us, args.seed)
        idx = smp.index.to_numpy()
        keep = okm["v1_file"][idx] & okm["v2_file"][idx]
        idx = idx[keep]
        labs = us.label.to_numpy()[idx]; pages = (us.book + "|" + us.page).to_numpy()[idx]
        R["v_c2c"] = {"n_cells": int(len(idx)), "n_labels": int(len(set(labs))), "chance": float(np.mean([
            (labs == l).mean() for l in set(labs)]))}
        for name in names:
            Xs = X[name][idx]
            px = np.stack([cv2.resize(x, (48, 48), interpolation=cv2.INTER_AREA).ravel() for x in Xs]).astype(np.float32)
            px = 1.0 - px / 255.0
            R["v_c2c"][name] = {"pixel48": nn_cross_page(px, labs, pages), "feat576": nn_cross_page(feat576(Xs), labs, pages)}
        R["v_c2c"]["expect_huong_c"] = {"feat576": [0.345, 0.572], "pixel48": [0.219, 0.435], "flag_ok": [0.908, 0.982],
                                        "fallback": 0.052}
        print(f"(v) crop→crop 1-NN chéo trang n={len(idx)} ({len(set(labs))} nhãn): " + " · ".join(
            f"{n} px48 {100*R['v_c2c'][n]['pixel48']:.1f}% f576 {100*R['v_c2c'][n]['feat576']:.1f}%" for n in names), flush=True)
        # (vi) crop→glyph phông + phép thử nhầm chữ (nhãn hàng xóm = nhãn bản ghi kề cùng cột, mọi tier)
        if not args.skip_glyph:
            nb_label = {}
            dfl = df[df.bbox != ""].copy(); dfl["ni"] = dfl.nom_idx.astype(int)
            for (b, pg, c), g in dfl.groupby(["book", "page", "column"]):
                g = g.sort_values("ni"); L = g.label.tolist(); NI = g.ni.tolist()
                for i, n_ in enumerate(NI):
                    nb_label[(b, pg, c, n_)] = (L[i - 1] if i > 0 else None, L[i + 1] if i < len(L) - 1 else None)
            rng2 = np.random.default_rng(args.seed + 1)
            def sub(mask):
                w = np.where(mask)[0]
                return w[rng2.permutation(len(w))[:args.n_glyph]]
            groups = {"sample_c2c": idx, "shift_gt_0.25": sub(strata["shift_gt_0.25"]),
                      "shift_le_0.10": sub(strata["shift_le_0.10"]),
                      "fallback": sub(strata["fallback"])}
            f3 = (us.crop_mode == "f3g").to_numpy(); a = us.ashift.to_numpy()
            for lo, hi in ((0.10, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40)):
                groups[f"bin_{lo:.2f}_{hi:.2f}"] = sub(f3 & (a > lo) & (a <= hi))
            R["vi_glyph"] = glyph_eval(us, X, okm, names, nb_label, Path(args.font_dir), groups)
            for tag in groups:
                g = R["vi_glyph"].get(tag, {})
                if g and g.get("n"):
                    print(f"(vi) crop→glyph {tag:14s} n={g['n']:,}: " + " · ".join(
                        f"{n} top1 {100*g[n]['top1']:.2f}% top5 {100*g[n]['top5']:.1f}% hạng {g[n]['rank_p50']:.0f} "
                        f"thua-hàng-xóm {100*g[n]['lose_to_neighbor']:.1f}%" for n in names), flush=True)
    else:
        us["p1_file"] = us["p2_file"] = np.nan

    # (iv) mini-GT ------------------------------------------------------
    cells = pd.read_csv(args.qd01, dtype=str, keep_default_na=False)
    q = us.merge(cells[["book", "page", "column", "nom_idx", "bbox_cu"]], on=["book", "page", "column", "nom_idx"], how="inner")
    q["bcu"] = q.bbox_cu.apply(_bb); q["b2"] = q.bbox_v2.apply(_bb); q["b2w"] = q.bbox_v2_win.apply(_bb); q["b1"] = q.bbox.apply(_bb)
    q["pitch_f"] = pd.to_numeric(q.pitch, errors="coerce")
    q["iou_v2"] = [iou(a, b) for a, b in zip(q.b2, q.bcu)]
    q["iou_v2w"] = [iou(a, b) for a, b in zip(q.b2w, q.bcu)]
    q["iou_v1"] = [iou(a, b) for a, b in zip(q.b1, q.bcu)]
    q["dc"] = [abs(cy(a) - cy(b)) / p if (a and b and p) else np.nan for a, b, p in zip(q.b2, q.bcu, q.pitch_f)]
    f3 = q[q.crop_mode == "f3g"]
    R["iv_qd01"] = {"n": int(len(q)), "bbox_eq_bbox_cu": int((q.iou_v1 >= 0.999).sum()),
                    "n_f3g": int(len(f3)), "n_fallback": int((q.crop_mode == "fallback").sum()),
                    "iou_v2_tight": {"p50": pct(q.iou_v2, 50), "p10": pct(q.iou_v2, 10), "lt_0.5": float((q.iou_v2 < 0.5).mean())},
                    "iou_v2_win": {"p50": pct(q.iou_v2w, 50), "p10": pct(q.iou_v2w, 10), "lt_0.5": float((q.iou_v2w < 0.5).mean())},
                    "center_dist_pitch": {"p50": pct(q.dc, 50), "p90": pct(q.dc, 90), "gt_0.25": float(np.nanmean(q.dc > 0.25))},
                    "iou_bound_note": "bbox_cu cao ≈1,22p, bbox_v2 cao ≈1,02p -> IoU tối đa ≈0,84 dù tâm trùng; dùng |Δtâm|/pitch"}
    print(f"(iv) QĐ-01 n={len(q)} (bbox==bbox_cu {R['iv_qd01']['bbox_eq_bbox_cu']}) f3g {len(f3)}: IoU(v2,cu) p50 {pct(q.iou_v2,50):.3f} "
          f"<0,5 {100*float((q.iou_v2<0.5).mean()):.1f}% · IoU(v2_win,cu) p50 {pct(q.iou_v2w,50):.3f} <0,5 "
          f"{100*float((q.iou_v2w<0.5).mean()):.1f}% · |Δtâm| p50 {pct(q.dc,50):.3f}p >0,25p {100*float(np.nanmean(q.dc>0.25)):.1f}%", flush=True)
    # 117 ô lệch: F3g từ bbox_truoc_lock có tiến về bbox_cu?
    l117 = pd.read_csv(args.lech117, dtype=str, keep_default_na=False)
    rows117 = []
    allbb = df[df.bbox != ""].copy(); allbb["bb"] = allbb.bbox.apply(_bb)
    for (book, page), g in l117.groupby(["book", "page"]):
        gray = cv2.imread(str(Path(args.pages_dir) / BOOKDIR[book] / "pages" / f"{page}.png"), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        pg = allbb[(allbb.book == book) & (allbb.page == page)]
        col_cys = {c: sorted(cy(b) for b in gg.bb) for c, gg in pg.groupby("column")}
        pp = rf.page_pitch(col_cys)
        for r in g.itertuples():
            bcu, btl = _bb(r.bbox_cu), _bb(r.bbox_truoc_lock)
            if not bcu or not btl:
                continue
            colr = pg[pg.column == r.column].copy()
            colr["cy"] = colr.bb.apply(cy); colr = colr.sort_values("cy")
            pitch = rf.column_pitch(col_cys.get(r.column, []), fallback=pp) or pp
            bbs = colr.bb.tolist(); c0 = cy(btl)
            # hàng xóm: bản ghi cùng cột gần nhất phía trên/dưới (loại chính ô này = bản ghi có bbox_cu)
            others = [b for b in bbs if b != bcu]
            pv = max((b for b in others if cy(b) < c0 - 0.3 * pitch), key=cy, default=None)
            nx = min((b for b in others if cy(b) > c0 + 0.3 * pitch), key=cy, default=None)
            b2, mode, shift, meta = rf.recenter_box(gray, btl, pv, nx, pitch)
            rows117.append({"key": f"{book}/{page}/c{r.column}/{r.nom_idx}", "iou_before": iou(btl, bcu),
                            "iou_after": iou(b2, bcu), "dc_before": abs(cy(btl) - cy(bcu)) / pitch,
                            "dc_after": abs(cy(b2) - cy(bcu)) / pitch, "mode": mode, "shift": shift})
    d117 = pd.DataFrame(rows117)
    if len(d117):
        f = d117[d117["mode"] == "f3g"]
        R["iv_lech117"] = {"n": int(len(d117)), "n_f3g": int(len(f)),
                           "iou_before_p50": pct(d117.iou_before, 50), "iou_after_p50": pct(d117.iou_after, 50),
                           "dc_before_p50": pct(d117.dc_before, 50), "dc_after_p50": pct(d117.dc_after, 50),
                           "dc_before_gt_0.25": float((d117.dc_before > 0.25).mean()),
                           "dc_after_gt_0.25": float((d117.dc_after > 0.25).mean()),
                           "closer_after": int((f.dc_after < f.dc_before - 0.02).sum()),
                           "farther_after": int((f.dc_after > f.dc_before + 0.02).sum())}
        print(f"(iv) 117 ô lệch: n={len(d117)} f3g {len(f)} · |Δtâm| vs bbox_cu p50 trước {pct(d117.dc_before,50):.3f}p → sau "
              f"{pct(d117.dc_after,50):.3f}p · >0,25p {100*float((d117.dc_before>0.25).mean()):.1f}% → "
              f"{100*float((d117.dc_after>0.25).mean()):.1f}% · gần hơn {R['iv_lech117']['closer_after']} / xa hơn "
              f"{R['iv_lech117']['farther_after']}", flush=True)

    # (vii) OOF đối xứng ------------------------------------------------
    oof_dirs = {"v1_page_official": Path(args.oof_csv).parent}
    if args.oof_v2:
        oof_dirs["v2_file_retrained"] = Path(args.oof_v2)
    if args.oof_v1file:
        oof_dirs["v1_file_retrained"] = Path(args.oof_v1file)
    if len(oof_dirs) > 1:
        key = ["book", "page", "column", "nom_idx"]
        # âm hàng xóm theo cột (mọi tier)
        nb_syl = {}
        dfl = df[df.bbox != ""].copy(); dfl["ni"] = dfl.nom_idx.astype(int)
        for (b, pg, c), g in dfl.groupby(["book", "page", "column"]):
            g = g.sort_values("ni"); L = g.syllable.str.lower().tolist(); NI = g.ni.tolist()
            for i, n_ in enumerate(NI):
                nb_syl[(b, pg, c, str(n_))] = (L[i - 1] if i > 0 else None, L[i + 1] if i < len(L) - 1 else None)
        base = us[key + ["syllable", "tier", "crop_mode", "ashift"]].copy()
        base["syl"] = base.syllable.str.lower()
        tabs = {}
        for name, d in oof_dirs.items():
            o = pd.read_csv(d / "p_visual_oof_v3.csv", dtype=str, keep_default_na=False)
            o = o[key + ["p_syl_v3", "argmax", "max_prob", "syl_in_classes"]].rename(
                columns={"p_syl_v3": f"p_{name}", "argmax": f"arg_{name}", "max_prob": f"pmax_{name}",
                         "syl_in_classes": f"inc_{name}"})
            base = base.merge(o, on=key, how="left")
            tabs[name] = json.loads((d / "summary.json").read_text()).get("mean_val_top1")
        names7 = list(oof_dirs)
        ok = np.ones(len(base), bool)
        for n in names7:
            ok &= (base[f"inc_{n}"] == "1").to_numpy() & (base[f"p_{n}"] != "").to_numpy()
        base = base[ok].reset_index(drop=True)
        R["vii_oof"] = {"n_common": int(len(base)), "mean_val_top1_by_run": tabs, "strata": {}}
        nbp = np.array([nb_syl.get((r.book, r.page, r.column, r.nom_idx), (None, None))[0] for r in base.itertuples()])
        nbn = np.array([nb_syl.get((r.book, r.page, r.column, r.nom_idx), (None, None))[1] for r in base.itertuples()])
        has_nb = (nbp != None) | (nbn != None)  # noqa: E711
        strata7 = {"all": np.ones(len(base), bool), "f3g": (base.crop_mode == "f3g").to_numpy(),
                   "fallback": (base.crop_mode == "fallback").to_numpy(),
                   "shift_gt_0.25": ((base.ashift > 0.25) & (base.crop_mode == "f3g")).to_numpy(),
                   "shift_le_0.10": ((base.ashift <= 0.10) & (base.crop_mode == "f3g")).to_numpy(),
                   # ô gần như KHÔNG tái định tâm: v1/v2 cùng chữ, chỉ khác cửa sổ (ngữ cảnh hàng xóm, seam, tighten)
                   # -> khoảng cách top-1 ở tầng này = lệch phân bố/ngữ cảnh, KHÔNG phải nhầm chữ
                   "shift_le_0.05": ((base.ashift <= 0.05) & (base.crop_mode == "f3g")).to_numpy()}
        for lo, hi in ((0.25, 0.30), (0.30, 0.35), (0.35, 0.40)):
            strata7[f"bin_{lo:.2f}_{hi:.2f}"] = ((base.ashift > lo) & (base.ashift <= hi) & (base.crop_mode == "f3g")).to_numpy()
        for tag, mk in strata7.items():
            row = {"n": int(mk.sum())}
            for n in names7:
                p = pd.to_numeric(base[f"p_{n}"], errors="coerce").to_numpy()[mk]
                t1 = (base[f"arg_{n}"] == base.syl).to_numpy()[mk]
                nbhit = ((base[f"arg_{n}"].to_numpy() == nbp) | (base[f"arg_{n}"].to_numpy() == nbn)) & (base[f"arg_{n}"] != base.syl).to_numpy()
                row[n] = {"top1": float(t1.mean()), "p50": float(np.nanmedian(p)), "p_ge_0.9": float(np.nanmean(p >= 0.9)),
                          "argmax_is_neighbor_syl": float(nbhit[mk & has_nb].mean()) if (mk & has_nb).any() else None}
            R["vii_oof"]["strata"][tag] = row
            print(f"(vii) OOF {tag:14s} n={row['n']:,}: " + " · ".join(
                f"{n} top1 {100*row[n]['top1']:.2f}% p50 {row[n]['p50']:.3f} argmax=âm-kề {100*row[n]['argmax_is_neighbor_syl']:.2f}%"
                for n in names7), flush=True)
        for a, b in (("v1_page_official", "v2_file_retrained"), ("v1_file_retrained", "v2_file_retrained")):
            if a in names7 and b in names7:
                ta = (base[f"arg_{a}"] == base.syl).to_numpy().astype(float); tb = (base[f"arg_{b}"] == base.syl).to_numpy().astype(float)
                n01, n10, pv = mcnemar(ta, tb)
                R["vii_oof"][f"mcnemar_{b}_vs_{a}"] = {"b_right_a_wrong": n01, "a_right_b_wrong": n10, "p": pv}
                print(f"(vii) McNemar {b} vs {a}: +{n01} / −{n10}, p={pv:.2e}")

    # HTML mẫu ---------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    def pick(d, n):
        return d.sample(min(n, len(d)), random_state=int(rng.integers(1 << 30)))
    n = args.n_sample
    parts = [pick(us[(us.crop_mode == "f3g") & (us.ashift > 0.25)], n * 12 // 20),
             pick(us[(us.crop_mode == "f3g") & (us.ashift <= 0.10)], n * 4 // 20),
             pick(us[us.crop_mode == "fallback"], n - n * 12 // 20 - n * 4 // 20)]
    smp = pd.concat(parts)
    qlow = q.sort_values("iou_v2").head(10)
    def row(r, note=""):
        return {"image": r.image, "label": r.label, "syllable": r.syllable, "v1": b64png(v1_root / r.image),
                "v2": b64png(v2_root / r.image), "crop_mode": r.crop_mode,
                "shift": f"{float(r.recenter_shift):+.2f}" if r.recenter_shift != "" else "",
                "flag1": r.crop_quality_flag, "flag2": r.crop_quality_flag_v2,
                "p1": "" if pd.isna(r.p1_file) else f"{r.p1_file:.3f}", "p2": "" if pd.isna(r.p2_file) else f"{r.p2_file:.3f}",
                "note": note}
    html_rows = [row(r, "f3g |shift|>0,25p" if abs(float(r.recenter_shift)) > 0.25 and r.crop_mode == "f3g"
                     else ("fallback (= v1)" if r.crop_mode == "fallback" else "f3g |shift|≤0,10p")) for r in smp.itertuples()]
    html_rows += [row(r, f"QĐ-01 IoU(v2,cu) thấp nhất: {r.iou_v2:.2f}, |Δtâm| {r.dc:.2f}p") for r in qlow.itertuples()]
    write_html(html_rows, out / "crops_v2_sample.html", "crops_v2 (F3g) — mẫu v1 | v2",
               f"labels {Path(args.labels).name} md5 {lab_md5[:12]} · sidecar md5 {sc_md5[:12]} · seed {args.seed} · "
               f"{n} ô mẫu (12 |shift|>0,25p, 4 ≤0,10p, 4 fallback) + 10 ô QĐ-01 IoU thấp nhất. "
               f"v2 fallback trùng byte v1. p_syl = CNN OOF fold theo trang (Khối B-1'), cut 64×64 từ tệp crop.")
    R["wall_s"] = round(time.time() - t0, 1)
    (out / "eval_crops_v2.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"-> {out / 'eval_crops_v2.json'} · {out / 'crops_v2_sample.html'} · {R['wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
