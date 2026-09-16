"""KIỂM CHỨNG KẾ HOẠCH v3 (docs/KE_HOACH_NANG_CAP_GAN_NHAN_FINAL.md) — 2026-09-15
Tái sinh ba con số của docs/DANH_GIA_KE_HOACH_NANG_CAP_2026-09-15.md mà thuc_nghiem.py
không in sẵn: (1) luật tier §5.1 của KẾ HOẠCH so với tier_v3 đã mô phỏng; (2) posterior
với ma trận hiệu chuẩn ở T=0,35 (kế hoạch) so với T=1,0 (đã mô phỏng); (3) ma trận
0/2,5/5,1/4,5/8,6 của kế hoạch so với CALIB 0/2,5/6,7/5,1/8,6.

    .venv/bin/python lab/gan_nhan_2026-09-13/thuc_nghiem_ke_hoach.py tier      # ~40 s
    .venv/bin/python lab/gan_nhan_2026-09-13/thuc_nghiem_ke_hoach.py T         # ~10 s
    .venv/bin/python lab/gan_nhan_2026-09-13/thuc_nghiem_ke_hoach.py matrix    # ~10 s
    .venv/bin/python lab/gan_nhan_2026-09-13/thuc_nghiem_ke_hoach.py all

Chỉ đọc repo + cols.pkl (cache của thuc_nghiem.py). Tất định.
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thuc_nghiem as T  # noqa: E402

PLAN = dict(COST_CONFIRM=0.0, COST_SIMILAR=2.5, COST_DICTMISS=5.1, COST_NODICT=4.5, COST_DEL=8.6, COST_INS=8.6)


def tier_plan(f, p, P_HI=0.8, P_MID=0.5):
    """Luật §5.1 của kế hoạch, chép nguyên văn. Vùng 0,5 ≤ p < 0,8 không được kế hoạch định
    nghĩa -> coi là REVIEW (cách đọc thận trọng nhất)."""
    if p < P_MID:
        return "REVIEW"
    if p < P_HI:
        return "REVIEW"                       # kế hoạch bỏ ngỏ
    if f["direct"]:
        return "CHAR_A"
    if f["sim_unique"] and (f["bigram"] or f["corpus2"]):
        return "CHAR_B"
    return "SYL"                              # ¬direct, KHÔNG cần ngữ cảnh


def cmd_tier():
    cols = T.load_cols()
    T.set_matrix(T.PROD)
    C = T.Corpus(cols)
    orig = T.tier_v3
    for name, fn in (("tier_v3 (đã mô phỏng trong RA_SOAT_GAN_NHAN §5.3)", orig),
                     ("luật §5.1 của KẾ HOẠCH (0,5≤p<0,8 -> REVIEW)", tier_plan)):
        T.tier_v3 = fn
        T.h(f"{name} — dữ liệu thật, CALIB + đệ quy neo cặp ≥2 trang khác, T=1,0")
        T._v3_crosstab(cols, C, recursive=True)
    T.tier_v3 = orig
    # SYL của kế hoạch tách theo có/không bằng chứng ngữ cảnh + benchmark ghép sai
    T.h("BENCHMARK 600 cột khớp số, PROD T=0,35 — SYL kế hoạch tách theo ngữ cảnh")
    T.set_matrix(T.PROD)
    all_chars = [c for col in cols for c in col["chars"] if c]
    eq = [c for c in cols if len(c["chars"]) == len(c["syl"]) and len(c["chars"]) >= 10 and c["tiers"] is not None]
    rng = random.Random(2026)
    sample = rng.sample(eq, 600)
    for kind in ("none", "drop_char", "drop_syl"):
        stat = defaultdict(lambda: [0, 0])
        for col in sample:
            pg = (col["book"], col["page"])
            c2, s2, truth = T.perturb(col["chars"], col["syl"], kind, rng, all_chars)
            pairs = T.viterbi_pairs(c2, s2)
            post = T.posterior_matches(c2, s2)
            for (i, j) in pairs:
                f = C.feats(c2, s2, i, j, pg)
                p = post.get((i, j), 0.0)
                ok = truth.get(i) == j
                tp, tv = tier_plan(f, p), orig(f, p)
                ctx = f["bigram"] or f["corpus4"] or f["tone"]
                for name in ("plan:" + tp + ("" if tp != "SYL" else ("+ctx" if ctx else "-ctx")), "v3:" + tv):
                    stat[name][0] += 1; stat[name][1] += not ok
        print(f"\n{kind}:")
        for name in sorted(stat):
            n, w = stat[name]
            print(f"  {name:14s} n={n:6d}  ghép sai {w:4d} ({w/max(1,n):.2%})")


def cmd_T():
    cols = T.load_cols()
    eq = [c for c in cols if len(c["chars"]) == len(c["syl"]) and len(c["chars"]) >= 10]
    rng = random.Random(7)
    sample = rng.sample(eq, 300)
    T.h("PHÂN BỐ p CỦA CẶP VITERBI (300 cột khớp số thật) — ngưỡng 0,5 / 0,8 có nghĩa với T nào")
    for mname, M in (("PROD", T.PROD), ("CALIB", T.CALIB), ("PLAN 5,1/4,5", PLAN)):
        T.set_matrix(M)
        for temp in (0.35, 1.0, 3.0):
            ps = []
            for col in sample:
                post = T.posterior_matches(col["chars"], col["syl"], T=temp)
                ps += [post.get(ij, 0.0) for ij in T.viterbi_pairs(col["chars"], col["syl"])]
            ps = np.array(ps)
            print(f"  {mname:12s} T={temp:<4}  p50={np.median(ps):.3f}  p<0,5 {np.mean(ps<0.5):5.1%}  "
                  f"0,5–0,8 {np.mean((ps>=0.5)&(ps<0.8)):5.1%}  p≥0,8 {np.mean(ps>=0.8):5.1%}  p≥0,99 {np.mean(ps>=0.99):5.1%}")
    T.set_matrix(T.PROD)


def cmd_matrix():
    cols = T.load_cols()
    T.h("MA TRẬN CHI PHÍ — khe giả / lệch chéo trên cột khớp số ≥10 chữ (như thuc_nghiem.py calib)")
    eq = [c for c in cols if len(c["chars"]) == len(c["syl"]) and len(c["chars"]) >= 10]
    for name, M in (("PROD 0/0,3/0,9/1,0 khe 0,7", T.PROD), ("CALIB 0/2,5/6,7/5,1 khe 8,6", T.CALIB),
                    ("PLAN  0/2,5/5,1/4,5 khe 8,6", PLAN)):
        T.set_matrix(M)
        gap_cols = off = pairs = 0
        for col in eq:
            ops = T.realign_column(col["chars"], col["syl"], T.qn, T.sim)
            if any(o["op"] != "match" for o in ops):
                gap_cols += 1
            for o in ops:
                if o["op"] == "match":
                    pairs += 1; off += o["nom_idx"] != o["syl_idx"]
        print(f"  {name:28s} cột có khe giả {gap_cols}/{len(eq)} · cặp lệch chéo {off}/{pairs} ({off/pairs:.2%})")
    T.set_matrix(T.PROD)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"tier": cmd_tier, "T": cmd_T, "matrix": cmd_matrix}
    for k in (steps if cmd == "all" else [cmd]):
        steps[k]()
