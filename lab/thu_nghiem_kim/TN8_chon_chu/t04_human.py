"""t04_human.py — nhúng crop NHÃN NGƯỜI Borg (dataset/_BORG_NHAN_NGUOI, hộp keep_high+ = keep/keep_high/keep_v5, kind real)
cắt LẠI từ trang prepared/_auto/<Sách>/pages (+ ảnh gốc) bằng CÙNG hình học save_crop như ô pipeline (t02.cut), chữ kề
= hộp người cùng cột. Dùng làm NGUYÊN MẪU NGƯỜI cho bộ chọn chữ — LOBO: ô B18 chỉ dùng mẫu B34 và ngược lại.
0 API. Ra: measure_out/_tn8/emb/hum_<B18|B34>_{enc,<net>}.npy + hum_<..>_meta.pkl
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import t02_embed as E2  # noqa: E402
from pipeline.align_engine.build_dataset import load_original_page  # noqa: E402
from pipeline.gold_exact import signals_img as SI  # noqa: E402
from pipeline.gold_exact.common import Assets  # noqa: E402

KEEP_OK = ("keep_v5", "keep", "keep_high")


def page_job(args):
    prep, pg, recs = args
    png = prep / "pages" / f"{pg}.png"
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if img is None:
        return [(i, None) for i, *_ in recs]
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_orig = load_original_page(prep, pg, shape_like=img.shape[:2])
    by_col = defaultdict(list)
    for i, col, bb, use in recs:
        by_col[col].append((i, bb, use))
    out = []
    for col, L in by_col.items():
        L.sort(key=lambda t: (t[1][1] + t[1][3]) / 2.0)
        for k, (i, bb, use) in enumerate(L):
            if not use:
                continue
            pv = L[k - 1][1] if k > 0 else None
            nx = L[k + 1][1] if k < len(L) - 1 else None
            g, _ = E2.cut(img, gray_full, bb, pv, nx, img_orig)
            out.append((i, g))
    return out


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    A = Assets()
    H = T.rd(T.REPO / "dataset/_BORG_NHAN_NGUOI/labels.csv")
    nets = {k: SI.load_net(A.load(f"{k.replace('vft_', 'vft_model_').replace('hand_', 'hand_model_')}.pt"), dev)
            for k in E2.NETS}
    enc = SI.MultiEnc([A.ext(SI.ENC_FILES["v1"]), A.ext(SI.ENC_FILES["v2"])], dev.type, True)
    for code, book in (("B34", "SachDungLyHoThan"), ("B18", "SachKinhThayCaBinh")):
        t0 = time.time()
        h = H[(H.book == book)].copy().reset_index(drop=True)
        h["bb"] = h.bbox.map(T.bbox_of)
        h["use"] = h.keep_level.isin(KEEP_OK) & (h.kind == "real") & h.bb.notna() & (h.char.str.len() == 1)
        prep = T.REPO / "prepared/_auto" / book
        jobs = []
        for pg, g in h[h.bb.notna()].groupby("page"):
            recs = [(i, c, bb, u) for i, c, bb, u in zip(g.index, g.column, g.bb, g.use)]
            jobs.append((prep, pg, recs))
        idx, grays = [], []
        with ThreadPoolExecutor(6) as ex:
            for res in ex.map(page_job, jobs):
                for i, gimg in res:
                    if gimg is not None and gimg.size:
                        idx.append(i); grays.append(gimg)
        M = h.loc[idx, ["page", "idx", "column", "char", "syllable", "keep_level", "khoi_trang", "cell_uid"]].reset_index(drop=True)
        out = {k: [] for k in ["enc"] + E2.NETS}
        for s in range(0, len(grays), 4096):
            part = grays[s:s + 4096]
            X64 = np.stack([SI.prep64(g) for g in part])
            for k, net in nets.items():
                out[k].append(SI.net_embed(net, X64, dev).astype(np.float16))
            out["enc"].append(enc.embed(part).astype(np.float16))
        for k, v in out.items():
            np.save(T.OUT / "emb" / f"hum_{code}_{k}.npy", np.concatenate(v))
        M.to_pickle(T.OUT / "emb" / f"hum_{code}_meta.pkl")
        print(f"[t04] {code}: {len(M)} crop người (keep_high+, real) [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
