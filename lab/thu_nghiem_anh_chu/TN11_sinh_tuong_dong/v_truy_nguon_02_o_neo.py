"""v_truy_nguon_02 (03/10) — ĐỘ CHÍNH XÁC Ô NEO (K02 nói "> 99 %") trên 4 bộ có nhãn người (B18, B34: Borg; L16, TK: IHR).
0 API, CPU, chỉ ĐỌC.

Tập ô neo = ĐÚNG tập mà bộ chọn chữ TN8 và p01/p02 dùng: cột `anchor` của measure_out/_tn8/cand/<bộ>_cells.pkl, được dựng ở
lab/thu_nghiem_kim/TN8_chon_chu/t05_feats.py:74-76 =  rule bắt đầu "s1_inter_s2_direct" ∧ label != "" ∧ label == ocr_char
∧ label ∈ R(âm) ∧ crop_ok (meta[:,3] > 0).  Script kiểm lại: dựng lại từ luật và so với cột `anchor` (phải trùng 100 %).
Sự thật = D.gt_char (chữ người; TN8 t00_base). Đo P(label đúng chữ người) trên các tập con:
  S1  neo có chữ người ("ô có sự thật" theo quy ước TN8)             — bảo thủ: lẫn ô lệch khe (Borg)
  S2  S1 ∧ pos_known (có hộp người tin được)                         — Borg: tập hộp dễ, lạc quan
  S3  S1 ∧ pos_ok    (tâm bbox ∈ hộp/khe người)                     — vị trí đã xác nhận
Đúng = label == gt  ("chặt")  hoặc label == gt ∨ var_eq_plus(label, gt)  ("dị thể", cách y của TN8 tính).  CI 95 % bootstrap CỤM TRANG (tn8lib).
Ra: measure_out/_tn11/verify/truy_nguon/o_neo.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "truy_nguon"
OUT.mkdir(parents=True, exist_ok=True)


def is_pua(ch: str) -> bool:
    return any(0xE000 <= ord(c) <= 0xF8FF or 0xF0000 <= ord(c) <= 0xFFFFD or 0x100000 <= ord(c) <= 0x10FFFD for c in ch)


def rate(ok, pages, B=2000):
    m, lo, hi = T.boot_ci_pages(ok, pages, B=B)
    return dict(n=int(len(ok)), p=round(100 * m, 2), ci95=[round(100 * lo, 2), round(100 * hi, 2)])


def main():
    C = T.lex()
    veq = C.var_eq_plus
    res = {}
    for b in ("B18", "B34", "L16", "TK"):
        D = T.load_base(b)
        meta = np.load(T.OUT / "emb" / f"{b}_crop_meta.npy")
        cells = pd.read_pickle(T.OUT / "cand" / f"{b}_cells.pkl")
        Rm = {s: C.R_of(s) for s in set(D.syllable)}
        # dựng lại luật ô neo và so với cột anchor
        anc = np.array((D.rule.str.startswith("s1_inter_s2_direct") & (D.label != "") & (D.label == D.ocr_char)).values, bool)
        anc &= np.array([l in Rm[s] for l, s in zip(D.label, D.syllable)])
        anc &= meta[:, 3] > 0
        same = bool((anc == cells.anchor.values).all())
        lab = D.label.values; gt = D.gt_char.values; pg = D.page.values
        has = gt != ""
        strict = np.array([l == g for l, g in zip(lab, gt)])
        dithe = np.array([bool(g) and (l == g or veq(l, g)) for l, g in zip(lab, gt)])
        pua = np.array([is_pua(l) or is_pua(g) for l, g in zip(lab, gt)])
        sets = {"S1_neo_co_chu_nguoi": anc & has,
                "S2_S1_pos_known": anc & has & D.pos_known.values,
                "S3_S1_pos_ok": anc & has & D.pos_ok.values}
        r = dict(so_o=int(len(D)), so_neo=int(anc.sum()), ty_le_neo_tren_o=round(100 * float(anc.mean()), 2),
                 dung_lai_luat_trung_cot_anchor=same, cac_tap={},
                 so_o_crop_ok_bang_0=int((meta[:, 3] <= 0).sum()), so_neo_co_ink_bang_0=int((meta[anc, 2] <= 0).sum()))
        for name, m in sets.items():
            r["cac_tap"][name] = dict(chat=rate(strict[m].astype(float), pg[m]), di_the=rate(dithe[m].astype(float), pg[m]),
                                      di_the_bo_PUA=rate(dithe[m & ~pua].astype(float), pg[m & ~pua]),
                                      so_o_PUA=int((m & pua).sum()))
        # theo tầng pipeline của ô neo (trên S1 và S3)
        tier = {}
        for name in ("S1_neo_co_chu_nguoi", "S3_S1_pos_ok"):
            m0 = sets[name]
            for t in sorted(set(D.tier.values[m0])):
                m = m0 & (D.tier.values == t)
                if m.sum() >= 30:
                    tier[f"{name}|{t}"] = rate(dithe[m].astype(float), pg[m], B=1000)
        r["theo_tang_pipeline_di_the"] = tier
        r["tong_neo_theo_tang"] = {t: int(((D.tier.values == t) & anc).sum()) for t in sorted(set(D.tier.values))}
        # ngưỡng 99 %: P(đúng>=99%) — ước lượng bằng tỉ lệ lần bootstrap có p >= 0,99 (S3, dị thể)
        m = sets["S3_S1_pos_ok"]
        up, inv = np.unique(pg[m], return_inverse=True)
        k = np.bincount(inv, weights=dithe[m].astype(float), minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
        rng = np.random.default_rng(20261003)
        idx = rng.integers(0, len(up), size=(4000, len(up)))
        rr = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
        r["P_bootstrap_p_ge_99pct_S3_dithe"] = round(float((rr >= 0.99).mean()), 4)
        res[b] = r
        s1, s3 = r["cac_tap"]["S1_neo_co_chu_nguoi"], r["cac_tap"]["S3_S1_pos_ok"]
        print(f"{b:4s} neo {r['so_neo']:6d}/{r['so_o']:6d} ({r['ty_le_neo_tren_o']:5.1f} %) luật==cột anchor: {same}")
        print(f"     S1 n={s1['di_the']['n']:6d} chặt {s1['chat']['p']:6.2f} dị thể {s1['di_the']['p']:6.2f} {s1['di_the']['ci95']} bỏ PUA {s1['di_the_bo_PUA']['p']:6.2f}")
        s2 = r["cac_tap"]["S2_S1_pos_known"]
        print(f"     S2 n={s2['di_the']['n']:6d} chặt {s2['chat']['p']:6.2f} dị thể {s2['di_the']['p']:6.2f} {s2['di_the']['ci95']}")
        print(f"     S3 n={s3['di_the']['n']:6d} chặt {s3['chat']['p']:6.2f} dị thể {s3['di_the']['p']:6.2f} {s3['di_the']['ci95']} bỏ PUA {s3['di_the_bo_PUA']['p']:6.2f} {s3['di_the_bo_PUA']['ci95']}  P(boot>=99%)={r['P_bootstrap_p_ge_99pct_S3_dithe']}")
        print("     theo tầng (dị thể):", {k: (v['n'], v['p']) for k, v in tier.items()})
        print("     tổng neo theo tầng:", r["tong_neo_theo_tang"])
    (OUT / "o_neo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
