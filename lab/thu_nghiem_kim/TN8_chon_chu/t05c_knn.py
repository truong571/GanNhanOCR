"""t05c_knn.py — thêm đặc trưng LÁNG GIỀNG GẦN (top-3 cos trung bình) vào bảng ứng viên:
  f_humk   crop · 3 crop NGƯỜI gần nhất có chữ = ứng viên (Borg SÁCH KIA — LOBO; STT: cả hai sách Borg)
  f_selfk  crop · 3 ô NEO cùng sách gần nhất có nhãn = ứng viên, CHỈ ở khối trang khác (tự động, không người)
(trung bình nguyên mẫu che mất biến thể nét chữ viết tay; thử nhanh: top-1 ô khó B18 82,4 → 84,9 %, B34 77,5 → 84,9 %).
Ghi thêm cột vào measure_out/_tn8/cand/<bộ>.pkl. 0 API.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import t05_feats as F5  # noqa: E402

K = 3
CH = 1500


def knn_col(E, ci, cc, cblk, Rm_emb, Rlab, Rblk, exclude_same_block: bool):
    """top-K cos trung bình của E[ci] với hàng Rm_emb có nhãn cc (và khối ≠ cblk nếu exclude)."""
    out = np.full(len(ci), np.nan, np.float32)
    by = {}
    for j, (l, k) in enumerate(zip(Rlab, Rblk)):
        by.setdefault(l, []).append(j)
    by = {l: np.array(v) for l, v in by.items()}
    order = np.argsort(ci, kind="stable")
    ci_s, cc_s, cb_s = ci[order], cc[order], cblk[order]
    cells = np.unique(ci_s)
    for s in range(0, len(cells), CH):
        cs = cells[s:s + CH]
        lo, hi = np.searchsorted(ci_s, cs[0]), np.searchsorted(ci_s, cs[-1], side="right")
        rowmap = {c: r for r, c in enumerate(cs)}
        S = E[cs] @ Rm_emb.T                                     # (CH, n_ref)
        seg_c, seg_i, seg_b = cc_s[lo:hi], ci_s[lo:hi], cb_s[lo:hi]
        df = pd.DataFrame({"c": seg_c, "r": [rowmap[x] for x in seg_i], "b": seg_b, "pos": np.arange(lo, hi)})
        keys = ["c", "b"] if exclude_same_block else ["c"]
        for key, g in df.groupby(keys, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            c, b = key[0], (key[1] if exclude_same_block else None)
            idx = by.get(c)
            if idx is None:
                continue
            if exclude_same_block:
                idx = idx[Rblk[idx] != b]
            if len(idx) == 0:
                continue
            sub = S[np.ix_(g.r.values, idx)]
            k = min(K, sub.shape[1])
            top = -np.partition(-sub, k - 1, axis=1)[:, :k] if sub.shape[1] > k else sub
            out[order[g.pos.values]] = top.mean(1)
    return out


def run(b: str):
    t0 = time.time()
    C = T.lex()
    D = T.load_base(b)
    E = np.load(F5.EMB / f"{b}_enc.npy").astype(np.float32)
    F = pd.read_pickle(F5.CAND / f"{b}.pkl")
    cells = pd.read_pickle(F5.CAND / f"{b}_cells.pkl")
    ci, cc = F.i.values, F.c.values.astype(object)
    blk = cells.blk.values
    # người (LOBO)
    if b in F5.HUM:
        Hs, Hl = [], []
        for hb in F5.HUM[b]:
            Hs.append(np.load(F5.EMB / f"hum_{hb}_enc.npy").astype(np.float32))
            Hl.append(pd.read_pickle(F5.EMB / f"hum_{hb}_meta.pkl").char.values)
        H = np.concatenate(Hs); Hl = np.concatenate(Hl).astype(object)
        F["f_humk"] = knn_col(E, ci, cc, blk[ci], H, Hl, np.zeros(len(Hl), int), False)
    else:
        F["f_humk"] = np.nan
    # neo cùng sách, khối khác
    anc = cells.anchor.values & (cells.crop_ok.values > 0)
    ai = np.nonzero(anc)[0]
    F["f_selfk"] = knn_col(E, ci, cc, blk[ci], E[ai], D.label.values[ai].astype(object), blk[ai], True)
    F.to_pickle(F5.CAND / f"{b}.pkl")
    print(f"[t05c] {b}: f_humk phủ {np.mean(~np.isnan(F.f_humk.values)):.3f}, f_selfk phủ {np.mean(~np.isnan(F.f_selfk.values)):.3f} "
          f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    for b in (sys.argv[1:] or T.ORDER):
        run(b)
