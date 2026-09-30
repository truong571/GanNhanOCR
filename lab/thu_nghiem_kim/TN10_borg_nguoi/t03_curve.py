"""TN10 t03 (30/09) — đường SỐ Ô GIỮ ↔ θ (trượt ±1, Paddle độc lập) cho bộ chữ người Borg, có/không mốc kim. 0 API.

So sánh công bằng = cùng θ. Với mỗi thư mục làm việc (beta mốc kim trong bước gióng: work_b0 = hiện tại, work_b1, work_b2…)
và mỗi luật giữ:
  P(t)       base & agree2 & post ≥ 0,999 & post2 ≥ t                      (luật hiện tại khi t = 0,999; keep_high khi t = 0,99)
  P(t)+K     P(t) ∪ (base & agree2 & kim xác nhận & min(post, post2) ≥ t_k) (thêm ô kim xác nhận vị trí, t_k ∈ lưới)
base = hộp thật, điểm detector ≥ 0,2, trang bỏ chữ < 10 % (như st_keep). θ + CI bootstrap cụm trang = paddle.summarize.
"kim xác nhận" = chữ kim (nguồn --src) có tâm trong hộp đơn vị (nới 25 %) trùng V1+ chữ người (như t01 / kim_anchor).

    .venv/bin/python lab/thu_nghiem_kim/TN10_borg_nguoi/t03_curve.py --works work_b0 work_b1 work_b2 [--src lt2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
T10 = REPO / "measure_out" / "_tn10"


def kim_confirm(C: pd.DataFrame, src: str, pad: float = 0.25) -> np.ndarray:
    from pipeline.borg_human.kim_anchor import kim_chars
    from pipeline.gold_exact.common import Assets, set_lexicon, var_eq_plus
    set_lexicon(Assets())
    out = np.zeros(len(C), bool)
    memo: dict = {}
    for (book, page), g in C.groupby(["book", "page"], sort=False):
        K = []
        for s in src.split("+"):
            k = kim_chars(book, page, s)
            K += k or []
        if not K:
            continue
        kc = np.array([k[0] for k in K], dtype=object); kx = np.array([k[1] for k in K]); ky = np.array([k[2] for k in K])
        pos = C.index.get_indexer(g.index)
        for i, r in zip(pos, g.itertuples()):
            if r.kind != "real" or not isinstance(r.char, str):
                continue
            w, h = r.x2 - r.x1, r.y2 - r.y1
            m = (kx >= r.x1 - pad * w) & (kx <= r.x2 + pad * w) & (ky >= r.y1 - pad * h) & (ky <= r.y2 + pad * h)
            for ch in kc[m]:
                v = memo.get((ch, r.char))
                if v is None:
                    v = memo[(ch, r.char)] = bool(var_eq_plus(ch, r.char))
                if v:
                    out[i] = True
                    break
    return out


def theta(C: pd.DataFrame, m: np.ndarray) -> dict:
    from pipeline.borg_human.paddle import summarize
    from pipeline.borg_human.params import PADDLE
    t = C.paddle_test.fillna("").to_numpy()
    key = (C.book + "/" + C.page).to_numpy()
    sel = m & (t != "")
    offs = np.array([0 if x == "ok" else int(x[4:]) for x in t[sel]])
    s = summarize(offs, key[sel], B=PADDLE["boot_B"], seed=PADDLE["boot_seed"])
    lech = int((m & np.char.startswith(t.astype(str), "lech")).sum())
    return dict(n=int(m.sum()), n_sau_loc_paddle=int(m.sum()) - lech, theta=s["theta"], theta_ci=s["theta_ci"], n_paddle=s["n"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--works", nargs="+", default=["work_b0", "work_b1", "work_b2"])
    ap.add_argument("--src", default="lt2")
    a = ap.parse_args(argv)
    rows = []
    for w in a.works:
        C = pd.read_csv(T10 / w / "cells.csv.gz")
        base = (C.kind == "real") & (C.det_score >= 0.2) & (C.page_skip < 0.1) & (C.agree2 == 1)
        K = kim_confirm(C, a.src)
        mn = np.minimum(C.post, C.post2)
        for t in (0.999, 0.995, 0.99, 0.98, 0.95):
            P = (base & (C.post >= 0.999) & (C.post2 >= t)).to_numpy()
            rows.append(dict(work=w, luat=f"P({t})", **theta(C, P)))
            for tk in (0.99, 0.95, 0.9):
                if tk > t:
                    continue
                PK = P | (base & K & (mn >= tk)).to_numpy()
                rows.append(dict(work=w, luat=f"P({t})+kim(min_post≥{tk})", **theta(C, PK)))
    R = pd.DataFrame(rows)
    R["theta_ci"] = R.theta_ci.astype(str)
    T10.mkdir(parents=True, exist_ok=True)
    R.to_csv(T10 / f"t03_curve_{a.src}.csv", index=False)
    print(R.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
