"""t15_l1_recheck.py — L1: đo lại chất lượng crop trên ẢNH GỐC (tệp giao nộp, crop_source original) bằng ngưỡng Otsu
theo từng crop, cho ô bị cổng (c) crop_bad (cờ blank/truncated đo trên crops_bin = bản nhị phân) và một mẫu GOLD 'ok'.
Mục tiêu: thiết kế phép kiểm lại để cổng (c) chỉ hạ ô mà ảnh GỐC cũng trắng/cụt. Sự thật (L16/TK/B18/B34) chỉ để ĐO.
0 API. Ra: measure_out/_tn8/l1_recheck.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402


def orig_metrics(g):
    """(contrast, ink_otsu, border_otsu, ink_fixed128) trên ảnh xám gốc."""
    if g is None or g.size == 0:
        return (np.nan,) * 4
    p5, p95 = np.percentile(g, 5), np.percentile(g, 95)
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = g <= thr                       # = phần THRESH_BINARY đặt 0 (cùng quy ước mechanism_gates.recheck_original)
    ink = float(bw.mean())
    border = float(max(bw[:2, :].mean(), bw[-2:, :].mean())) if g.shape[0] >= 4 else 0.0
    return (float(p95 - p5), ink, border, float((g < 128).mean()))


def main():
    C = T.lex()
    res = {}
    for b in ("L16", "TK", "B18", "B34", "Chr", "L83", "KVK"):
        cfg = T.BOOKS[b]
        root = T.REPO / cfg["labels"].rsplit("/", 1)[0]
        D = T.load_base(b)
        bad = (D.gate_reason.str.startswith("crop_bad")).values
        gold_ok = (D.tier.values == "GOLD") & (D.crop_quality_flag.values == "ok")
        rng = np.random.default_rng(0)
        gi = np.nonzero(gold_ok)[0]
        gi = rng.choice(gi, min(1500, len(gi)), replace=False)
        idx = np.r_[np.nonzero(bad)[0], gi]
        rows = []
        for i in idx:
            f = root / D.image.values[i] if D.image.values[i] else None
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f is not None and f.exists() else None
            rows.append((i, *orig_metrics(g), g is not None))
        M = pd.DataFrame(rows, columns=["i", "contrast", "ink_o", "border_o", "ink128", "has_img"]).set_index("i")
        M["kind"] = np.where(bad[M.index.values], D.gate_reason.values[M.index.values], "GOLD_ok")
        M["kind"] = M["kind"] + np.where(D.label.values[M.index.values] != "", "", "|khong_nhan")
        M["lab_ok"] = [bool(l) and bool(g_) and C.var_eq_plus(l, g_) for l, g_ in zip(D.label.values[M.index.values],
                                                                                   D.gt_char.values[M.index.values])]
        M["has_gt"] = D.gt_char.values[M.index.values] != ""
        M["pos_known"] = D.pos_known.values[M.index.values]
        M["pos_ok"] = D.pos_ok.values[M.index.values]
        out = {}
        for k, g in M.groupby("kind"):
            q = g[g.has_img]
            out[k] = dict(n=int(len(g)), has_img=int(g.has_img.sum()),
                          contrast_p5_p50=[round(float(q.contrast.quantile(0.05)), 1), round(float(q.contrast.median()), 1)],
                          ink_o_p5_p50_p95=[round(float(q.ink_o.quantile(x)), 3) for x in (0.05, 0.5, 0.95)],
                          border_o_p50_p95=[round(float(q.border_o.quantile(x)), 3) for x in (0.5, 0.95)])
        # quy tắc kiểm lại (ứng viên): ảnh gốc KHÔNG trắng ⇔ contrast ≥ Cmin ∧ ink_o ≥ 0.03; KHÔNG cụt ⇔ border_o ≤ Bmax
        cands = {}
        for cmin in (20, 30, 40):
            for bmax in (0.20, 0.30, 0.40):
                ok_o = (M.contrast >= cmin) & (M.ink_o >= 0.03) & (M.border_o <= bmax) & M.has_img
                r = {}
                for k, g in M.groupby("kind"):
                    s = g[ok_o.loc[g.index]]
                    e = s[s.has_gt]
                    ep = s[s.pos_known & s.has_gt]
                    r[k] = dict(giu=int(len(s)), n=int(len(g)), chu=round(float(e.lab_ok.mean()), 4) if len(e) else None,
                                hai_ve=round(float((ep.lab_ok & ep.pos_ok).mean()), 4) if len(ep) else None)
                cands[f"C{cmin}_B{bmax}"] = r
        res[b] = dict(stats=out, rules=cands)
        print(f"[t15] {b}: " + " | ".join(f"{k}: n {v['n']} contrast p5/p50 {v['contrast_p5_p50']} ink_o {v['ink_o_p5_p50_p95']} "
                                          f"border_o {v['border_o_p50_p95']}" for k, v in out.items()), flush=True)
        r = cands["C30_B0.3"]
        print("     C30_B0.3: " + " | ".join(f"{k}: giữ {v['giu']}/{v['n']} chữ {v['chu']} hai vế {v['hai_ve']}" for k, v in r.items()),
              flush=True)
    T.jdump(res, T.OUT / "l1_recheck.json")


if __name__ == "__main__":
    main()
