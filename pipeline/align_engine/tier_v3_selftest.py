"""Selftest tier_v3 (A-7, flow N5a–N5d): `CorpusStats.feats` PHẢI giống 100%
`thuc_nghiem.Corpus.feats` (lab/gan_nhan_2026-09-13) trên cols.pkl — vì mọi con số
mô phỏng của kế hoạch đo bằng lab; mã sản xuất lệch một cờ là số đó vô nghĩa.

    .venv/bin/python -m pipeline.align_engine.tier_v3_selftest [--n-cols 200]

Chỉ đọc; không ghi gì. Cần lab/gan_nhan_2026-09-13/cols.pkl (thuc_nghiem rebuild).
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab" / "gan_nhan_2026-09-13"))

_passed = _failed = 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name} {extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cols", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import thuc_nghiem as tn                                        # noqa: E402
    from pipeline.align_engine.tier_v3 import (                     # noqa: E402
        CorpusStats, build_bigram_syl_pages, context_evidence, rule_of,
        syl_bigram_count, tier_v3)

    if not tn.CACHE.exists():
        # cols.pkl là cache lab (gitignored, dựng bằng `thuc_nghiem.py rebuild`): thiếu thì
        # BỎ QUA có ghi rõ, vẫn in RESULT để scripts/run_all_selftests.sh phân biệt
        # "thiếu dữ liệu" với "gãy" (A-15) — 7 phép so với lab tự bỏ qua.
        print(f"[tier_v3 selftest] skip: không có {tn.CACHE} (chạy thuc_nghiem.py rebuild)")
        print("RESULT: 0 passed, 0 failed")
        return 0
    cols = pickle.load(open(tn.CACHE, "rb"))
    print(f"[tier_v3 selftest] cols.pkl: {len(cols)} cột")
    C_lab = tn.Corpus(cols)
    C = CorpusStats(cols, tn.qn, tn.sim)
    check("pair_pages giống lab (số khoá)", len(C.pair_pages) == len(C_lab.pair_pages),
          f"{len(C.pair_pages)} vs {len(C_lab.pair_pages)}")
    check("bigram_pages giống lab (số khoá)", len(C.bigram_pages) == len(C_lab.bigram_pages),
          f"{len(C.bigram_pages)} vs {len(C_lab.bigram_pages)}")
    check("pair_pages giống lab (nội dung)", dict(C.pair_pages) == dict(C_lab.pair_pages))
    check("bigram_pages giống lab (nội dung)", dict(C.bigram_pages) == dict(C_lab.bigram_pages))

    rng = random.Random(args.seed)
    sample = rng.sample(cols, min(args.n_cols, len(cols)))
    KEYS = ("direct", "sim_unique", "tone", "corpus2", "corpus4", "bigram")
    n_pairs = n_diff = 0
    first_diff = None
    tiers = {}
    for col in sample:
        pg = (col["book"], col["page"])
        chars, syls = col["chars"], col["syl"]
        for o in col["ops"]:
            if o["op"] != "match":
                continue
            i, j = o["nom_idx"], o["syl_idx"]
            f_lab = C_lab.feats(chars, syls, i, j, pg)
            f = C.feats(chars, syls, i, j, pg)
            n_pairs += 1
            if any(bool(f_lab[k]) != bool(f[k]) for k in KEYS):
                n_diff += 1
                first_diff = first_diff or (pg, i, j, {k: (f_lab[k], f[k]) for k in KEYS})
            # tier_v3 nguyên văn: cùng feats + cùng p -> cùng tier (dùng p = 0,9 và 0,6)
            for p in (0.9, 0.6, 0.2):
                if tn.tier_v3(f_lab, p) != tier_v3(f, p):
                    n_diff += 1
            t = tier_v3(f, 0.9)
            tiers[t] = tiers.get(t, 0) + 1
            # rule_of phải trả được với mọi tổ hợp cờ (không StopIteration)
            rule_of(t, f, 0.9)
            context_evidence(f)
    check(f"feats giống thuc_nghiem 100% trên {len(sample)} cột ({n_pairs} cặp)",
          n_diff == 0, f"lệch {n_diff}; đầu tiên: {first_diff}")
    print(f"  phân bố tier_v3 (p=0,9) trên mẫu: {tiers}")

    # 2-gram ÂM–ÂM (N5e): LOO + hai phía cộng
    bsp = build_bigram_syl_pages(cols)
    col = sample[0]
    pg = (col["book"], col["page"])
    syls = col["syl"]
    if len(syls) >= 3:
        n_self = syl_bigram_count(bsp, syls, 1, syls[1], pg)
        n_all = (len(bsp[(syls[0].lower(), syls[1].lower())])
                 + len(bsp[(syls[1].lower(), syls[2].lower())]))
        check("syl_bigram_count LOO: loại đúng trang đang xét (hai phía cộng)",
              n_self == n_all - 2, f"{n_self} vs {n_all} - 2")
    check("rule_of: ánh xạ N5d",
          rule_of("CHAR_A", {"direct": True}, 0.9) == "s1_inter_s2_direct"
          and rule_of("CHAR_B", {"direct": True}, 0.6) == "s1_inter_s2_direct_lowp"
          and rule_of("CHAR_B", {"direct": False}, 0.9) == "s1_inter_s2_similar"
          and rule_of("SYL", {"bigram": False, "corpus4": True, "tone": True}, 0.6) == "syl_ctx:corpus4"
          and rule_of("REVIEW", {"bigram": False, "corpus4": False, "tone": False}, 0.9) == "no_context"
          and rule_of("REVIEW", {"bigram": True, "corpus4": False, "tone": False}, 0.2) == "low_posterior")
    print(f"RESULT: {_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
