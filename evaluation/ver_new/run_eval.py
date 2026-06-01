"""A/B evaluation: current index-pairing  vs  banded anchored DP (ver_new).

Runs on the REAL data already in prepared/ (no model, no GPU, S3-independent):
  - Nôm side  : detected/page_*_ocr_cache.json  -> per-column [{char, bbox}]
  - QN side   : transcriptions/page_*.json       -> per-column syllables
  - Dictionary: dict/QuocNgu_SinoNom_TongHop3.csv + SinoNom_Similar_Dic_v2.csv

For every column it builds the Nôm↔QN pairing TWO ways and scores each by the
dictionary-confirmation rate (a high-precision proxy for alignment correctness:
measured ~99% precise where it fires) plus the GOLD/SILVER/REVIEW tiering:

  OLD = current step2 logic (index pairing + count force-equalize: drop leading
        chars when too many, pair prefix when too few).
  NEW = anchor_align.realign_column (banded dictionary-anchored DP).

Outputs a console report + results/summary.json + results/pairs_new.csv +
results/fixed_examples.txt.

Run:
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/run_eval.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

# repo root = two levels up (evaluation/ver_new/ -> repo)
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from core.text.dictionary import (  # noqa: E402
    load_qn_to_nom, load_similarity_dict,
)
from evaluation.ver_new.anchor_align import (  # noqa: E402
    realign_column, matched_pairs, is_confirmed,
)
from evaluation.ver_new.consensus import decide_label  # noqa: E402

BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]
OUT = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_page_columns(book: str):
    """Yield (page_name, nom_cols, qn_cols) where
    nom_cols = list[list[{'char':..}]]  (SinoNom OCR boxes per column)
    qn_cols  = list[list[str]]          (QN syllables per column)."""
    base = REPO / "prepared" / book
    nom_by_page: dict[str, list] = {}
    for f in glob.glob(str(base / "detected" / "page_*_ocr_cache.json")):
        pn = Path(f).stem.replace("_ocr_cache", "")
        d = json.load(open(f, encoding="utf-8"))
        nom_by_page[pn] = [[{"char": c.get("char")} for c in col]
                           for col in d.get("columns", [])]
    qn_by_page: dict[str, list] = {}
    for f in glob.glob(str(base / "transcriptions" / "page_*.json")):
        if f.endswith("_qn_ocr_cache.json"):
            continue
        pn = Path(f).stem
        d = json.load(open(f, encoding="utf-8"))
        qn_by_page[pn] = [c.get("syllables", []) for c in d.get("columns", [])]
    for pn in sorted(set(nom_by_page) & set(qn_by_page)):
        yield pn, nom_by_page[pn], qn_by_page[pn]


# ---------------------------------------------------------------------------
# OLD pairing — faithful to pipeline/step2_align.py count force-equalize
# ---------------------------------------------------------------------------
def old_pairing(nom_col: list, syllables: list[str]) -> list[dict]:
    """Reproduce current behaviour: when too many Nôm boxes drop the LEADING
    extras, when too few pair the syllable prefix, then zip 1-1 by position."""
    actual, expected = len(nom_col), len(syllables)
    if actual > expected:
        chars_used = nom_col[actual - expected:]
    else:
        chars_used = nom_col
    pairs = []
    for k in range(min(len(chars_used), len(syllables))):
        pairs.append({"ocr_char": chars_used[k]["char"], "syllable": syllables[k]})
    return pairs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qn_to_nom = load_qn_to_nom(str(REPO / "dict" / "QuocNgu_SinoNom_TongHop3.csv"))
    similar = load_similarity_dict(str(REPO / "dict" / "SinoNom_Similar_Dic_v2.csv"))

    # accumulators
    agg = {
        "old": {"pairs": 0, "confirmed": 0},
        "new": {"pairs": 0, "confirmed": 0},
    }
    # split by whether the column's Nôm/QN counts matched
    split = {
        "matched": {"cols": 0, "old_pairs": 0, "old_conf": 0, "new_pairs": 0, "new_conf": 0},
        "diverged": {"cols": 0, "old_pairs": 0, "old_conf": 0, "new_pairs": 0, "new_conf": 0},
    }
    tiers_new = Counter()
    rules_new = Counter()
    # head-to-head: per column, does NEW recover more confirmed pairs than OLD?
    col_better = col_worse = col_tie = 0
    fixed_examples = []          # columns where NEW strictly beats OLD
    pairs_rows = []              # per emitted NEW pair (for CSV)
    n_pages = 0
    eq_page_cols = 0             # cols on pages where #nom_cols == #qn_cols (the 3393 cohort)

    for book in BOOKS:
        for pn, nom_cols, qn_cols in load_page_columns(book):
            n_pages += 1
            page_eq = (len(nom_cols) == len(qn_cols))
            for ci in range(min(len(nom_cols), len(qn_cols))):
                nom_col, syls = nom_cols[ci], qn_cols[ci]
                if not syls:
                    continue
                matched = (len(nom_col) == len(syls))
                if page_eq:
                    eq_page_cols += 1

                # OLD
                op = old_pairing(nom_col, syls)
                old_conf = sum(is_confirmed(p["ocr_char"], p["syllable"],
                                            qn_to_nom, similar) for p in op)
                # NEW
                ops = realign_column(nom_col, syls, qn_to_nom, similar)
                mp = matched_pairs(ops)
                new_conf = sum(1 for p in mp if p["confirmed"])

                agg["old"]["pairs"] += len(op); agg["old"]["confirmed"] += old_conf
                agg["new"]["pairs"] += len(mp); agg["new"]["confirmed"] += new_conf
                s = split["matched" if matched else "diverged"]
                s["cols"] += 1
                s["old_pairs"] += len(op); s["old_conf"] += old_conf
                s["new_pairs"] += len(mp); s["new_conf"] += new_conf

                # tiers for NEW emitted pairs (S3 = None -> S3-independent floor).
                # `anchored` = this confirmed match has a confirmed neighbour in
                # the match-run, so its local register is certain even if the
                # whole column count diverged.
                conf_flags = [bool(p["confirmed"]) for p in mp]
                for k, p in enumerate(mp):
                    nbr = (k > 0 and conf_flags[k - 1]) or \
                          (k + 1 < len(mp) and conf_flags[k + 1])
                    anchored = bool(p["confirmed"]) and bool(nbr)
                    dec = decide_label(p["ocr_char"], p["syllable"], matched,
                                       qn_to_nom, similar, s3=None, anchored=anchored)
                    tiers_new[dec.tier] += 1
                    rules_new[dec.rule_id] += 1
                    pairs_rows.append({
                        "book": book[12:], "page": pn, "col": ci,
                        "ocr_char": p["ocr_char"], "syllable": p["syllable"],
                        "confirmed": int(p["confirmed"]), "tier": dec.tier,
                        "rule": dec.rule_id,
                    })

                if new_conf > old_conf:
                    col_better += 1
                    if len(fixed_examples) < 40:
                        fixed_examples.append({
                            "book": book[12:], "page": pn, "col": ci,
                            "old_conf": old_conf, "new_conf": new_conf,
                            "nom": "".join(c["char"] or "·" for c in nom_col),
                            "qn": " ".join(syls),
                        })
                elif new_conf < old_conf:
                    col_worse += 1
                else:
                    col_tie += 1

    # ----- write artefacts -------------------------------------------------
    summary = {
        "pages": n_pages,
        "columns_on_equal_col_pages": eq_page_cols,
        "overall": {
            "old": dict(agg["old"],
                        confirm_rate=round(agg["old"]["confirmed"] / max(agg["old"]["pairs"], 1), 4)),
            "new": dict(agg["new"],
                        confirm_rate=round(agg["new"]["confirmed"] / max(agg["new"]["pairs"], 1), 4)),
        },
        "by_column_type": split,
        "head_to_head_columns": {"new_better": col_better, "new_worse": col_worse, "tie": col_tie},
        "new_tiers": dict(tiers_new),
        "new_rules": dict(rules_new),
    }
    json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(OUT / "pairs_new.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["book", "page", "col", "ocr_char",
                                          "syllable", "confirmed", "tier", "rule"])
        w.writeheader(); w.writerows(pairs_rows)
    with open(OUT / "fixed_examples.txt", "w", encoding="utf-8") as f:
        for e in fixed_examples:
            f.write(f"[{e['book']} {e['page']} col{e['col']}] "
                    f"confirmed {e['old_conf']} -> {e['new_conf']}\n"
                    f"   Nôm: {e['nom']}\n   QN : {e['qn']}\n\n")

    # ----- console report --------------------------------------------------
    def rate(d): return d["confirmed"] / max(d["pairs"], 1) * 100
    print("=" * 70)
    print(" A/B: index-pairing (OLD)  vs  banded anchored DP (NEW)")
    print(f" {n_pages} pages, 3 books — dictionary-only (S3 not used)")
    print("=" * 70)
    print(f"\nOVERALL emitted pairs / dict-confirmed:")
    print(f"  OLD : {agg['old']['pairs']:6d} pairs | {agg['old']['confirmed']:6d} confirmed = {rate(agg['old']):.1f}%")
    print(f"  NEW : {agg['new']['pairs']:6d} pairs | {agg['new']['confirmed']:6d} confirmed = {rate(agg['new']):.1f}%")
    dconf = agg['new']['confirmed'] - agg['old']['confirmed']
    print(f"  Δ   : {agg['new']['pairs']-agg['old']['pairs']:+d} pairs, {dconf:+d} confirmed, "
          f"{rate(agg['new'])-rate(agg['old']):+.1f} pp confirm-rate")

    for t in ("matched", "diverged"):
        s = split[t]
        oldr = s["old_conf"] / max(s["old_pairs"], 1) * 100
        newr = s["new_conf"] / max(s["new_pairs"], 1) * 100
        print(f"\n{t.upper()}-count columns ({s['cols']} cols):")
        print(f"  OLD confirm {s['old_conf']}/{s['old_pairs']} = {oldr:.1f}%   "
              f"NEW confirm {s['new_conf']}/{s['new_pairs']} = {newr:.1f}%   ({newr-oldr:+.1f} pp)")

    print(f"\nPer-column head-to-head (confirmed-pair count):")
    print(f"  NEW better: {col_better}   tie: {col_tie}   NEW worse: {col_worse}")
    print(f"\nNEW tier breakdown (emitted pairs, S3-independent floor):")
    for t in ("GOLD", "SILVER", "REVIEW"):
        print(f"  {t:7s}: {tiers_new.get(t,0)}")
    print(f"\nArtefacts written to {OUT}/  (summary.json, pairs_new.csv, fixed_examples.txt)")
    print("Note: SILVER=0 expected without S3 (visual stack not provisioned). "
          "GOLD here is the S3-independent, dictionary-confirmed floor.")


if __name__ == "__main__":
    main()
