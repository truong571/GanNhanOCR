"""t12_print_variant.py — L83/KVK: chữ kim của ô KHÔNG GOLD có trùng DỊ BẢN số hoá người (cùng câu, cùng CHỈ SỐ trong câu) không?
0 API. Khác TN8 (chỉ ô có âm QN trùng âm dị bản, same_pos = 1): ở đây lấy chữ dị bản theo CHỈ SỐ trong câu đã khớp
(auto_precision.match_verses, ≥ 75 % âm trùng, duy nhất) — nên cả ô mà âm QN của ta HỎNG (OCR QN sai, "Äi", "buỗi") vẫn
có chữ đối chiếu. Hai nguồn độc lập (kim đọc ảnh sách này ↔ người số hoá bản khác) trùng nhau ⇒ bằng chứng mạnh cho chữ kim.
Nền: trên GOLD, kim == dị bản 81,6 % (L83) / 90,6 % (KVK) (khác = dị bản hợp lệ). Đọc: tỉ lệ trùng của nhóm / tỉ lệ trùng
GOLD ≈ cận dưới độ đúng tương đối (giả định tỉ lệ khác-bản-hợp-lệ như GOLD). ĐỌC transcriptions từ ẢNH CHỤP.
Ra measure_out/_tn9/print_variant.json + print/<bộ>_var.pkl
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402

FULL = {"L83": "LucVanTien1883", "KVK": "KimVanKieu1884"}


def main():
    import auto_precision as AP
    C = T.lex()
    out = {}
    for b, full in FULL.items():
        cfg = dict(AP.CROSS_BOOKS[full])
        cfg["refs"] = list(cfg["refs"])
        cfg["trans"] = T.SNAP / "prepared" / full / "transcriptions"
        verses, vstat = AP.match_verses(full, cfg, 0.75)
        idx_ref = defaultdict(list)      # (page, col, syl) -> [(ref char, same_pos, ref name, qn_source)]
        for v in verses:
            for i in range(v["n"]):
                if i < len(v["ref_nom"]):
                    idx_ref[(v["page"], int(v["column"]), v["syl_offset"] + i)].append(
                        (v["ref_nom"][i], v["same_pos"][i], v["ref"], v["qn_source"]))
        D = pd.read_pickle(T.OUT / "print" / f"{b}_kv.pkl") if (T.OUT / "print" / f"{b}_kv.pkl").exists() else \
            pd.read_pickle(T.OUT / "print" / f"{b}.pkl")
        refs = [idx_ref.get((p, c, s), []) for p, c, s in zip(D.page, D.column, D.syl_i)]
        D["vref"] = ["|".join(r[0] for r in rr) for rr in refs]
        D["vsame"] = ["|".join(r[1] for r in rr) for rr in refs]
        D["vsrc"] = [rr[0][3] if rr else "" for rr in refs]
        D["kim_eq_vref"] = [bool(rr) and bool(k) and any(C.var_eq_plus(k, r[0]) for r in rr) for k, rr in zip(D.ocr_char, refs)]
        D["vref_inR"] = [bool(rr) and any(r[0] in C.R_of(s) for r in rr) for s, rr in zip(D.syllable, refs)]
        D.to_pickle(T.OUT / "print" / f"{b}_var.pkl")
        r = dict(verse_stat=vstat)
        for g, s in D.groupby("grp"):
            h = s[s.vref != ""]
            if len(h) == 0:
                r[g] = dict(n=int(len(s)), n_ref=0); continue
            same = h.vsame.str.contains("1")
            r[g] = dict(n=int(len(s)), n_ref=int(len(h)), kim_eq_vref=round(float(h.kim_eq_vref.mean()), 4),
                        kim_eq_vref_samepos=round(float(h.kim_eq_vref[same].mean()), 4) if same.any() else None,
                        kim_eq_vref_diffpos=round(float(h.kim_eq_vref[~same].mean()), 4) if (~same).any() else None,
                        n_diffpos=int((~same).sum()), vref_inR=round(float(h.vref_inR.mean()), 4),
                        by_src={k: [round(float(v.kim_eq_vref.mean()), 3), int(len(v))] for k, v in h.groupby("vsrc")})
        out[b] = r
        print(f"== {b}: {vstat}", flush=True)
        for g, v in sorted(r.items(), key=lambda kv: -kv[1]["n"] if isinstance(kv[1], dict) and "n" in kv[1] else 0):
            if isinstance(v, dict) and "n" in v:
                print(f"   {g:12s} {v}", flush=True)
    T.jdump(out, T.OUT / "print_variant.json")


if __name__ == "__main__":
    main()
