"""v_pb_nguon_goc_04 (04/10) — PHẢN BIỆN: độ chính xác của ô neo tự động ("> 99 %") so với nhãn người, DỰNG LẠI BẰNG MÃ MỚI.

Quy tắc ô neo copy từ p01:74-75 / p02:46-50 (tự viết lại ở đây, không import):
   rule bắt đầu 's1_inter_s2_direct' & label != '' & label == ocr_char & label ∈ R(âm tiết) & meta[:,3] > 0
meta = (w, h, ink, ok) theo t02_embed.py:147 -> meta[:,3] là cờ ok, KHÔNG phải mực; thêm biến thể meta[:,2] > 0 (mực thật).
Độ đúng đo trên 2 cách: (a) nghiêm ngặt label == D.gt_char (nhãn người/GT) ; (b) label ∈ {c : F.y == 1} (cho phép dị thể như TN8).
Ra: measure_out/_tn11/verify/pb_nguon_goc/o_neo.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
OUT.mkdir(parents=True, exist_ok=True)
C = T.lex()


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


res = {}
for b in ("B34", "B18", "L16", "TK"):
    D = T.load_base(b)
    meta = np.load(T.OUT / "emb" / f"{b}_crop_meta.npy")
    Rm = {s: C.R_of(s) for s in set(D.syllable)}
    base = np.array((D.rule.str.startswith("s1_inter_s2_direct") & (D.label != "") & (D.label == D.ocr_char)).values, bool)
    inR = np.array([l in Rm[s] for l, s in zip(D.label, D.syllable)])
    anc = base & inR & (meta[:, 3] > 0)
    anc_ink = base & inR & (meta[:, 2] > 0)
    lab = D.label.values
    gt = D.gt_char.values
    has_gt = np.array([g not in ("", None) for g in gt])
    m = anc & has_gt
    strict = float(np.mean([lab[i] == gt[i] for i in np.nonzero(m)[0]])) * 100 if m.any() else float("nan")
    k_s = int(sum(lab[i] == gt[i] for i in np.nonzero(m)[0]))
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y"]]
    truth = F[F.y == 1].groupby("i").c.apply(set).to_dict()
    ai = [i for i in np.nonzero(anc)[0] if i in truth]
    k_y = int(sum(lab[i] in truth[i] for i in ai))
    # chỉ tầng GOLD (nếu có cột tier)
    gold = np.array((D.tier == "GOLD").values, bool)
    mg = anc & has_gt & gold
    k_g = int(sum(lab[i] == gt[i] for i in np.nonzero(mg)[0]))
    res[b] = {"n_D": len(D), "n_anchor_meta3": int(anc.sum()), "n_anchor_ink_gt_0": int(anc_ink.sum()),
              "meta3_all_ones_on_anchor": bool((meta[anc, 3] == 1).all()),
              "strict": {"n": int(m.sum()), "k": k_s, "pct": round(strict, 2), "wilson95": wilson(k_s, int(m.sum()))},
              "theo_F_y": {"n": len(ai), "k": k_y, "pct": round(100 * k_y / max(1, len(ai)), 2), "wilson95": wilson(k_y, len(ai))},
              "strict_GOLD": {"n": int(mg.sum()), "k": k_g, "pct": round(100 * k_g / max(1, int(mg.sum())), 2)},
              "co_gt_tren_neo_pct": round(100 * m.sum() / max(1, anc.sum()), 1)}
    print(b, json.dumps(res[b], ensure_ascii=False), flush=True)
(OUT / "o_neo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
