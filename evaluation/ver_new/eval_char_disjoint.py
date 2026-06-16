"""Char-disjoint test: can the SIMILAR-FONT glyph recognise a char that has NO
real crops — i.e. the UNSEEN-character regime where the similar images must carry
the whole signal?

Protocol (simulation, no retrain needed):
  - Hold out K char classes (seeded).
  - REMOVE their crop prototypes, so each held-out true char is backed ONLY by its
    similar-font glyph (gannhanocr-fd) — exactly like an unseen char.
  - On the held-out classes' TEST crops, measure retrieval@1 over the real
    candidate set R = {ocr_char} ∪ dict-readings:
        glyph_only : rank ALL candidates by similar-font-glyph cosine (fair, one
                     tier) -> what the similar images achieve for an unseen char.
        combined   : the live calibrated system (crop-backed DISTRACTORS can
                     out-score a glyph-only true char) -> exposes the
                     domain-confidence bias when the true char is crop-less.
  - A CONTROL group (non-held-out classes, crops kept) is measured the same way.

CAVEAT: the encoder BACKBONE trained on these chars, so the numbers are an UPPER
BOUND for true zero-shot. For the exact figure, exclude the holdout classes from
training and retrain the backbone on Kaggle, then re-run (the crops are already
excluded here, so the harness is identical — only the checkpoint changes).

Run:
  .venv/bin/python evaluation/ver_new/eval_char_disjoint.py --holdout 150 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402


def _candidates(ocr_char, syllable, qn):
    cands = []
    for c in ([ocr_char] if ocr_char else []) + qn.get((syllable or "").lower(), []):
        if _is_cjk(c) and c not in cands:
            cands.append(c)
    return cands


def _retrieval(rows_by_class, classes, vs3, qn, D, score_kind):
    """retrieval@1 over the real candidate set for the given classes."""
    hit = n = 0
    for ch in classes:
        for r in rows_by_class[ch]:
            ce = vs3.enc.embed_path(str(D / r["image"]))
            if ce is None:
                continue
            cands = _candidates(r["ocr_char"], r["syllable"], qn)
            if ch not in cands:
                cands.append(ch)
            if len(cands) < 2:
                continue
            n += 1
            tc = {c: vs3.tier_cosines(ce, c) for c in cands}
            if score_kind == "glyph":
                score = {c: tc[c].get("fd", -1.0) for c in cands}
            else:  # combined: per-tier calibrated P, best tier
                score = {}
                for c in cands:
                    ps = [vs3._p_tier(t, v) for t, v in tc[c].items()]
                    ps = [p for p in ps if p is not None]
                    score[c] = max(ps) if ps else 0.0
            if max(score, key=score.get) == ch:
                hit += 1
    return hit, n


def _decision_stats(rows_by_class, classes, vs3, qn, D, guard):
    """Use the production decide() to count asserted-correct / ASSERTED-WRONG
    (false SILVER) / abstained. The crop-bias fix should move asserted-wrong ->
    abstained for crop-less true chars."""
    asserted = a_correct = a_wrong = abstained = n = 0
    for ch in classes:
        for r in rows_by_class[ch]:
            ce = vs3.enc.embed_path(str(D / r["image"]))
            if ce is None:
                continue
            cands = _candidates(r["ocr_char"], r["syllable"], qn)
            if ch not in cands:
                cands.append(ch)
            if len(cands) < 2:
                continue
            n += 1
            dec = vs3.decide(ce, cands, guard=guard)
            if dec["reject"]:
                abstained += 1
            else:
                asserted += 1
                if dec["top_char"] == ch:
                    a_correct += 1
                else:
                    a_wrong += 1
    return dict(n=n, asserted=asserted, a_correct=a_correct, a_wrong=a_wrong, abstained=abstained)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "dataset_out"))
    ap.add_argument("--holdout", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   simfont_dir=str(REPO / paths["fd_cache_similar"]) if paths.get("fd_cache_similar") else "")

    D = Path(args.dataset)
    by_class = defaultdict(list)
    for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8")):
        if r["tier"] == "GOLD" and r["split"] == "test" and r["image"] and r["syllable"] and r["label"]:
            by_class[r["label"]].append(r)

    # eligible held-out classes: >=2 test crops AND a crop prototype to remove
    eligible = [c for c, rs in by_class.items() if len(rs) >= 2 and c in vs3.proto]
    random.seed(args.seed)
    random.shuffle(eligible)
    holdout = set(eligible[:args.holdout])
    control = eligible[args.holdout:args.holdout + len(holdout)]   # same-size control
    print(f"eligible classes (>=2 test crops, has proto): {len(eligible)}")
    print(f"held-out: {len(holdout)} | control: {len(control)}\n", flush=True)

    # remove held-out crop prototypes -> their true char is glyph-only (unseen-like)
    removed = [vs3.proto.pop(c, None) for c in holdout]
    print(f"removed {sum(x is not None for x in removed)} crop prototypes (held-out chars now glyph-only)\n", flush=True)

    print("=== retrieval@1 on HELD-OUT chars (no real crops -> similar-font glyph only) ===")
    for kind in ("glyph", "combined"):
        h, n = _retrieval(by_class, holdout, vs3, qn, D, kind)
        print(f"  {kind:9s}: {h/max(n,1):.1%}  ({h}/{n})", flush=True)

    print("\n=== CONTROL: same metric on classes that KEEP their crops ===")
    for kind in ("glyph", "combined"):
        h, n = _retrieval(by_class, control, vs3, qn, D, kind)
        print(f"  {kind:9s}: {h/max(n,1):.1%}  ({h}/{n})", flush=True)

    print("\n=== open-set crop-bias GUARD effect — production decide() on HELD-OUT ===")
    for guard in (False, True):
        s = _decision_stats(by_class, holdout, vs3, qn, D, guard)
        tag = "guard ON " if guard else "guard OFF"
        print(f"  {tag}: asserted {s['asserted']} (correct {s['a_correct']}, "
              f"WRONG {s['a_wrong']}) | abstained {s['abstained']}  -> "
              f"false-SILVER rate {s['a_wrong']/max(s['asserted'],1):.1%}", flush=True)
    print("  (guard ON should move WRONG asserts -> abstained: fewer false SILVER on crop-less chars)")

    print("\n  Headline: HELD-OUT glyph_only = the similar-font images' retrieval for a")
    print("  char with NO crops (the unseen regime). HELD-OUT combined < glyph_only shows")
    print("  the crop-bias: when the true char is crop-less, crop-backed distractors win")
    print("  unless candidates are compared on the shared glyph tier.")
    print("  CAVEAT: backbone trained on these chars -> UPPER BOUND vs true zero-shot.")


if __name__ == "__main__":
    main()
