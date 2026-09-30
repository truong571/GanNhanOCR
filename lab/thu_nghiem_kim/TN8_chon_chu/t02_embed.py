"""t02_embed.py — cắt crop cho MỌI ô (mọi tầng) đúng hình học save_crop của pipeline (pad 0,12, carve chữ kề cùng cột,
tighten, điểm ảnh gốc với sách crop_source=original) rồi nhúng: MultiEnc v1+v2 (norm) 512-d + 6 CNN 64×64 (vft_T/L,
hand_T/L_Kinh, hand_T/L_DungLy). 0 API, MPS. Không ghi tệp ảnh (chỉ mảng nhúng).

Ra: measure_out/_tn8/emb/<bộ>_{enc,vft_T,vft_L,hand_T_Kinh,hand_L_Kinh,hand_T_DungLy,hand_L_DungLy}.npy (float16, theo
thứ tự hàng base/<bộ>.pkl) + <bộ>_crop_meta.npz (w, h, ink, ok). Kiểm: crop dựng lại == tệp crop giao nộp trên mẫu GOLD.
  .venv/bin/python lab/thu_nghiem_kim/TN8_chon_chu/t02_embed.py [bộ ...] [--check-only]
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
from pipeline.align_engine.bbox_fix import carve_neighbor_ink, tighten_box  # noqa: E402
from pipeline.align_engine.build_dataset import _paper_bg, load_original_page  # noqa: E402
from pipeline.gold_exact import signals_img as SI  # noqa: E402
from pipeline.gold_exact.common import Assets  # noqa: E402

PAD = 0.12
NETS = ["vft_T", "vft_L", "hand_T_Kinh", "hand_L_Kinh", "hand_T_DungLy", "hand_L_DungLy"]
EMB = T.OUT / "emb"


def cut(img, gray_full, bbox, prev_bbox, next_bbox, img_orig):
    """== build_dataset.save_crop (không ghi tệp) -> ảnh xám của tệp sẽ giao (gốc nếu có) + (w, h, ink trên bản xử lý)."""
    H, W = img.shape[:2]
    ox1, oy1, ox2, oy2 = (int(v) for v in bbox)
    pw, ph = int((ox2 - ox1) * PAD), int((oy2 - oy1) * PAD)
    x1, y1 = max(0, ox1 - pw), max(0, oy1 - ph)
    x2, y2 = min(W, ox2 + pw), min(H, oy2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    crop = crop.copy()
    crop_o = img_orig[y1:y2, x1:x2].copy() if img_orig is not None else None
    if gray_full is not None and (prev_bbox is not None or next_bbox is not None):
        crop = carve_neighbor_ink(crop, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox)
        if crop_o is not None:
            crop_o = carve_neighbor_ink(crop_o, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox,
                                        bg=_paper_bg(crop_o))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    tb = tighten_box(gray)
    if tb is not None:
        a, c, b, d = tb
        crop = crop[c:d, a:b]; gray = gray[c:d, a:b]
        if crop_o is not None:
            crop_o = crop_o[c:d, a:b]
    if crop.size == 0:
        return None, None
    out = crop_o if crop_o is not None else crop
    # tệp giao = PNG của `out`; scorer đọc lại IMREAD_GRAYSCALE -> mô phỏng bằng mã hoá/giải mã PNG
    ok, buf = cv2.imencode(".png", out)
    g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return g, (gray.shape[1], gray.shape[0], float((gray < 128).mean()))


def page_crops(args):
    b, png, prep, orig, recs = args
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if img is None:
        return [(i, None, None) for i, *_ in recs]
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_orig = None
    if orig:
        img_orig = load_original_page(prep, Path(png).stem, shape_like=img.shape[:2])
        if img_orig is None:
            raise FileNotFoundError(f"thiếu ảnh gốc {b} {png}")
    by_col = defaultdict(list)
    for i, col, bb in recs:
        if bb is not None:
            by_col[col].append((i, bb))
    nb = {}
    for col, L in by_col.items():
        L.sort(key=lambda t: (t[1][1] + t[1][3]) / 2.0)
        for k, (i, bb) in enumerate(L):
            nb[i] = (L[k - 1][1] if k > 0 else None, L[k + 1][1] if k < len(L) - 1 else None)
    out = []
    for i, col, bb in recs:
        if bb is None:
            out.append((i, None, None)); continue
        pv, nx = nb.get(i, (None, None))
        g, meta = cut(img, gray_full, bb, pv, nx, img_orig)
        out.append((i, g, meta))
    return out


def check(b, D, n=150):
    """crop dựng lại == tệp giao nộp (ô có image) trên mẫu — trả (n_so, n_trung)."""
    cfg = T.BOOKS[b]
    root = T.REPO / cfg["labels"].rsplit("/", 1)[0]
    S = D[(D.image != "")].sample(min(n, int((D.image != "").sum())), random_state=0)
    prep = T.REPO / cfg["prep"]
    same = tot = 0
    for pg, g in S.groupby("page"):
        allp = D[D.page == pg]
        recs = [(i, c, T.bbox_of(bb)) for i, c, bb in zip(allp.index, allp.column, allp.bbox)]
        res = {i: gg for i, gg, _ in page_crops((b, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs))}
        for i, im in zip(g.index, g.image):
            f = root / im
            if not f.exists():
                continue
            ref = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            mine = res.get(i)
            tot += 1
            same += mine is not None and mine.shape == ref.shape and int(np.abs(mine.astype(int) - ref.astype(int)).max()) == 0
    return tot, same


def main():
    books = [a for a in sys.argv[1:] if not a.startswith("--")] or T.ORDER
    check_only = "--check-only" in sys.argv
    EMB.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    A = Assets()
    nets = enc = None
    for b in books:
        D = T.load_base(b)
        tot, same = check(b, D)
        print(f"[t02] {b}: kiểm crop dựng lại == tệp giao {same}/{tot}", flush=True)
        if check_only:
            continue
        if all((EMB / f"{b}_{k}.npy").exists() for k in ["enc"] + NETS):
            print(f"[t02] {b}: đã có nhúng — bỏ qua", flush=True); continue
        if nets is None:
            nets = {k: SI.load_net(A.load(f"{k.replace('vft_', 'vft_model_').replace('hand_', 'hand_model_')}.pt"), dev)
                    for k in NETS}
            enc = SI.MultiEnc([A.ext(SI.ENC_FILES["v1"]), A.ext(SI.ENC_FILES["v2"])], dev.type, True)
        cfg = T.BOOKS[b]
        prep = T.REPO / cfg["prep"]
        t0 = time.time()
        jobs = []
        for pg, g in D.groupby("page", sort=True):
            recs = [(i, c, T.bbox_of(bb)) for i, c, bb in zip(g.index, g.column, g.bbox)]
            jobs.append((b, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs))
        N = len(D)
        E = {k: np.zeros((N, 256), np.float16) for k in NETS}
        Ee = np.zeros((N, 512), np.float16)
        meta = np.zeros((N, 4), np.float32)          # w, h, ink, ok
        buf_i, buf_g = [], []

        def flush():
            if not buf_i:
                return
            X64 = np.stack([SI.prep64(g) for g in buf_g])
            for k, net in nets.items():
                E[k][buf_i] = SI.net_embed(net, X64, dev).astype(np.float16)
            Ee[buf_i] = enc.embed(buf_g).astype(np.float16)
            buf_i.clear(); buf_g.clear()
        done = 0
        with ThreadPoolExecutor(6) as ex:
            for res in ex.map(page_crops, jobs):
                for i, g, m in res:
                    if g is None or g.size == 0:
                        continue
                    meta[i] = [m[0], m[1], m[2], 1]
                    buf_i.append(i); buf_g.append(g)
                done += 1
                if len(buf_i) >= 4096:
                    flush()
                    print(f"  {b}: {done}/{len(jobs)} trang [{time.time() - t0:.0f}s]", flush=True)
        flush()
        for k in NETS:
            np.save(EMB / f"{b}_{k}.npy", E[k])
        np.save(EMB / f"{b}_enc.npy", Ee)
        np.save(EMB / f"{b}_crop_meta.npy", meta)
        print(f"[t02] {b}: {N} ô, crop ok {int(meta[:, 3].sum())} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
