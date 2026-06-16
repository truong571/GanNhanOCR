"""APPLIED TEST #1 — is the structural (radical/IDS) route viable for THIS corpus?

Cross-script research said the durable transfer lever is the shared CJK STRUCTURAL
code (radicals / Ideographic Description Sequences), via CCR-CLIP / HierCode — a
prototype that can be built for ANY class, including the zero-real-crop tail where
S3 is weakest, purely from the character's structure. But every cited result is on
CHINESE benchmarks; whether it helps Nôm is inferred, not shown. The gating
question, recommended as "do this first", is empirical and needs NO GPU:

    Of THIS thesis's 1,591 Nôm classes — and especially the rare tail — how many
    actually have an IDS decomposition in the open CHISE/cjkvi-ids database?

This script answers it directly on the real class list. IDS files are the ones in
evaluation/ver_new/ids_data/ (cjkvi-ids: ids.txt + ids-ext-cdef.txt + hanyo-ids.txt).

Coverage levels reported:
  present       the char appears in the IDS table at all
  decomposable  its IDS contains an Ideographic Description Char (U+2FF0..2FFF),
                i.e. it splits into >=2 radical components (the useful case)
  atomic        present but its IDS is itself (a radical/leaf — still a usable code)
Broken down by Unicode block and by rare tail (classes with < RARE_N GOLD crops,
where S3 is weakest and the structural prototype would matter most).

Run:
  .venv/bin/python evaluation/ver_new/ids_coverage.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDS_DIR = HERE / "ids_data"
LABELS = HERE / "dataset_out" / "labels.csv"
RARE_N = 5                       # "rare tail" = char classes with < this many GOLD crops

IDC = set(range(0x2FF0, 0x3000))  # Ideographic Description Characters


def block(c: str) -> str:
    o = ord(c)
    if 0x4E00 <= o <= 0x9FFF: return "CJK Unified"
    if 0x3400 <= o <= 0x4DBF: return "Ext-A"
    if 0x20000 <= o <= 0x2A6DF: return "Ext-B"
    if 0x2A700 <= o <= 0x2EBEF: return "Ext-C..F"
    if 0xF900 <= o <= 0xFAFF: return "Compat"
    return "other"


def load_ids() -> dict[str, str]:
    """char -> first IDS expression, merged over all files (ids.txt has priority)."""
    ids: dict[str, str] = {}
    files = ["ids.txt", "ids-ext-cdef.txt", "hanyo-ids.txt"]
    for fn in files:
        p = IDS_DIR / fn
        if not p.exists():
            continue
        for ln in open(p, encoding="utf-8"):
            if ln.startswith("#") or "\t" not in ln:
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ch, expr = parts[1], parts[2]
            if ch and ch not in ids:        # first file wins
                ids[ch] = expr
    return ids


def radicals(ch: str, ids: dict[str, str], depth: int = 0, seen=None) -> set[str]:
    """Recursively expand to leaf (atomic) components using the IDS table."""
    if seen is None:
        seen = set()
    if ch in seen or depth > 12:
        return set()
    seen.add(ch)
    expr = ids.get(ch)
    if not expr or expr == ch:
        return {ch}                          # leaf / atomic radical
    leaves = set()
    for u in expr:
        o = ord(u)
        if o in IDC or o < 0x80:
            continue                         # structure operator or ASCII source-tag ([GTKV]) — skip
        if u == ch:
            leaves.add(u)
        else:
            leaves |= radicals(u, ids, depth + 1, seen)
    return leaves or {ch}


def main():
    # class list + per-class GOLD crop counts
    counts = Counter()
    for r in csv.DictReader(open(LABELS, encoding="utf-8")):
        if r["label_level"] == "char" and r["label"] and r["tier"] == "GOLD":
            counts[r["label"]] += 1
    # include SILVER-only classes too (full class set)
    classes = set(counts)
    for r in csv.DictReader(open(LABELS, encoding="utf-8")):
        if r["label_level"] == "char" and r["label"]:
            classes.add(r["label"])
    classes = [c for c in classes if len(c) == 1]
    print(f"classes: {len(classes)}  | IDS files: {[p.name for p in IDS_DIR.glob('*.txt')]}", flush=True)

    ids = load_ids()
    print(f"IDS table entries loaded: {len(ids)}", flush=True)

    by_block = defaultdict(lambda: {"n": 0, "present": 0, "decomp": 0, "atomic": 0})
    rare = {"n": 0, "present": 0, "decomp": 0}
    common = {"n": 0, "present": 0, "decomp": 0}
    rad_hist = Counter()
    examples_missing = []
    examples_decomp = []
    for c in classes:
        b = block(c)
        bb = by_block[b]; bb["n"] += 1
        expr = ids.get(c)
        present = expr is not None
        decomp = present and any(ord(u) in IDC for u in expr)
        atomic = present and not decomp
        if present: bb["present"] += 1
        if decomp: bb["decomp"] += 1
        if atomic: bb["atomic"] += 1
        tail = counts.get(c, 0) < RARE_N
        g = rare if tail else common
        g["n"] += 1
        if present: g["present"] += 1
        if decomp: g["decomp"] += 1
        if decomp and len(examples_decomp) < 6:
            rads = radicals(c, ids)
            examples_decomp.append(f"{c} (U+{ord(c):X}) = {expr}  -> radicals {{{''.join(sorted(rads))}}}")
        if decomp:
            rad_hist[len(radicals(c, ids))] += 1
        if not present and len(examples_missing) < 12:
            examples_missing.append(f"{c} (U+{ord(c):X}, {b})")

    n = len(classes)
    present = sum(bb["present"] for bb in by_block.values())
    decomp = sum(bb["decomp"] for bb in by_block.values())
    atomic = sum(bb["atomic"] for bb in by_block.values())

    print("\n" + "=" * 64)
    print(" IDS / RADICAL COVERAGE for the 1,591-class Nôm vocabulary")
    print("=" * 64)
    print(f"  present (has any IDS)     : {present}/{n}  ({present/n:.1%})")
    print(f"  decomposable (>=2 radicals): {decomp}/{n}  ({decomp/n:.1%})")
    print(f"  atomic (is itself a radical): {atomic}/{n}  ({atomic/n:.1%})")
    print("\n  by Unicode block:")
    print(f"   {'block':14s} {'n':>5s} {'present':>9s} {'decomp':>8s}")
    for b, bb in sorted(by_block.items(), key=lambda kv: -kv[1]["n"]):
        print(f"   {b:14s} {bb['n']:5d} {bb['present']/max(bb['n'],1):>8.0%} {bb['decomp']/max(bb['n'],1):>7.0%}")
    print(f"\n  RARE tail (<{RARE_N} GOLD crops): n={rare['n']}  present {rare['present']/max(rare['n'],1):.0%}  "
          f"decomposable {rare['decomp']/max(rare['n'],1):.0%}")
    print(f"  COMMON (>= {RARE_N})           : n={common['n']}  present {common['present']/max(common['n'],1):.0%}  "
          f"decomposable {common['decomp']/max(common['n'],1):.0%}")
    print(f"\n  radicals-per-char (decomposed): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(rad_hist.items())[:8]))
    print("\n  examples (decomposed):")
    for e in examples_decomp:
        print("   " + e)
    if examples_missing:
        print("\n  examples MISSING an IDS:")
        print("   " + " · ".join(examples_missing))

    verdict = ("VIABLE — most classes (incl. the rare tail) have a radical structure, so a "
               "HierCode/CCR-CLIP structural prototype can be built for them."
               if decomp / n > 0.85 and rare["decomp"] / max(rare["n"], 1) > 0.8 else
               "PARTIAL — coverage gaps (see missing list); structural route helps where present, "
               "keep the FD glyph as fallback for the uncovered tail.")
    print(f"\n  VERDICT: {verdict}")

    out = {"classes": n, "present": present, "decomposable": decomp, "atomic": atomic,
           "present_pct": round(present / n, 4), "decomposable_pct": round(decomp / n, 4),
           "by_block": {b: dict(bb) for b, bb in by_block.items()},
           "rare_tail": rare, "common": common, "verdict": verdict}
    p = HERE / "results" / "ids_coverage.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
