"""APPLIED TEST #2 — can the ArcFace HEAD (an unused signal) rescue REVIEW?

Cross-script lesson (kuzushiji pseudo-label loop + Korean detect-then-cluster):
the strongest cheap lever on the 89%-S3/coverage REVIEW pile is a stronger VISUAL
classifier. We already have one for free: the encoder's ArcFace HEAD is a 1,591-way
classifier, and `visual_signal.decide()` currently ignores it (it scores cosine to
the reference bank only). The head is an INDEPENDENT signal — using it as a 4th
consensus vote, and rescuing REVIEW where the head AGREES with the reference bank,
is a no-retrain way to test the loop on THIS data.

Two parts, both on the real corpus:

  A) VALIDATE on GOLD val (ground truth): retrieval@1 over the real candidate set
     R = {ocr_char} u dict-readings for (a) the reference bank [current S3],
     (b) the ArcFace head restricted to R, (c) their AGREEMENT subset. The agree
     subset precision is the number that matters — it sets the expected precision
     of any REVIEW rescue.

  B) ESTIMATE the rescue on REVIEW `below_visual_threshold` rows (the 13.8k that
     are NOT a segmentation problem). Their crops are not materialised, so we cut
     them from the page via the stored bbox, embed, and count where head+bank agree
     on a dict reading -> the rescuable set, extrapolated to the full pile, at the
     GOLD-measured agree precision.

Run:
  .venv/bin/python evaluation/ver_new/head_rescue.py            # --review-n 2500
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
from evaluation.ver_new.bbox_fix import tighten_box                # noqa: E402

HERE = Path(__file__).resolve().parent


def cands_of(ocr_char, syllable, qn, true=None):
    cs = []
    for c in ([ocr_char] if ocr_char else []) + qn.get((syllable or "").lower(), []):
        if _is_cjk(c) and c not in cs:
            cs.append(c)
    if true and _is_cjk(true) and true not in cs:
        cs.append(true)
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--review-n", type=int, default=2500, help="REVIEW rows to sample for the estimate")
    args = ap.parse_args()

    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   simfont_dir=str(REPO / paths.get("fd_cache_similar", "")) if paths.get("fd_cache_similar") else "")
    enc = vs3.enc
    if not enc.has_head:
        sys.exit("checkpoint has no ArcFace head -> this test needs the head (it is present in nom-embed/best.pt).")
    lab2idx = {lab: i for i, lab in enc.classes.items()}
    name_map = {b["name"][12:]: b["name"] for b in cfg["books"]}
    data_root = REPO / cfg["paths"]["data_dir"]
    D = Path(args.dataset)

    def head_top(emb, R):
        """argmax of head logits over R ∩ vocab -> (char, top_logit, margin) or None."""
        lg = enc.logits(emb)
        sc = [(c, float(lg[lab2idx[c]])) for c in R if c in lab2idx]
        if not sc:
            return None
        sc.sort(key=lambda t: -t[1])
        margin = sc[0][1] - (sc[1][1] if len(sc) > 1 else -1.0)
        return sc[0][0], sc[0][1], margin

    # ---------- A) validate on GOLD val ----------
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == "GOLD" and r["split"] == "val" and r["image"] and r["syllable"] and r["label"]]
    nA = bank_hit = head_hit = agree = agree_hit = 0
    for r in rows:
        emb = enc.embed_path(str(D / r["image"]))
        if emb is None:
            continue
        true = r["label"]
        R = cands_of(r["ocr_char"], r["syllable"], qn, true)
        if len(R) < 2:
            continue
        nA += 1
        bank = vs3.decide(emb, R)["top_char"]
        ht = head_top(emb, R)
        head = ht[0] if ht else None
        bank_hit += (bank == true)
        head_hit += (head == true)
        if head is not None and head == bank:
            agree += 1
            agree_hit += (bank == true)
    print("=" * 64)
    print(" A) GOLD val — head as a new signal vs the reference bank")
    print("=" * 64)
    print(f"  decisions                : {nA}")
    print(f"  reference-bank retrieval@1: {bank_hit/max(nA,1):.1%}   (current S3)")
    print(f"  ArcFace-head retrieval@1  : {head_hit/max(nA,1):.1%}   (NEW, restricted to R)")
    print(f"  AGREE (head==bank)        : {agree/max(nA,1):.1%} coverage, "
          f"precision {agree_hit/max(agree,1):.1%}  <- the rescue-precision proxy")
    agree_prec = agree_hit / max(agree, 1)

    # ---------- B) rescue estimate on REVIEW below_visual_threshold ----------
    rev = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
           if r["tier"] == "REVIEW" and r["rule"] == "below_visual_threshold"
           and r["bbox"] and r["bbox"] != "null" and r["syllable"]]
    total_rev = len(rev)
    sample = rev[: args.review_n]
    pages = {}
    def page_img(book, page):
        key = (book, page)
        if key not in pages:
            full = name_map.get(book, book)
            p = data_root / full / "pages" / f"{page}.png"
            pages[key] = cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None
        return pages[key]

    nB = rescuable = head_in_R = 0
    for r in sample:
        R = qn.get((r["syllable"] or "").lower(), [])
        R = [c for c in R if _is_cjk(c)]
        if r["ocr_char"] and _is_cjk(r["ocr_char"]) and r["ocr_char"] not in R:
            R = [r["ocr_char"]] + R
        if len(R) < 1:
            continue
        img = page_img(r["book"], r["page"])
        if img is None:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in json.loads(r["bbox"])]
        except Exception:
            continue
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        tb = tighten_box(gray)
        if tb is not None:
            a, c, b, d = tb
            if b - a >= 8 and d - c >= 8:
                gray = gray[c:d, a:b]
        emb = enc.embed_gray(gray)
        nB += 1
        bank = vs3.decide(emb, R)["top_char"] if len(R) >= 2 else (R[0] if R else None)
        ht = head_top(emb, R)
        if ht is None:
            continue
        head, _, _ = ht
        # rescuable: head picks a DICT reading and the bank agrees (two visual votes)
        dict_readings = set(qn.get((r["syllable"] or "").lower(), []))
        if head in dict_readings and head == bank:
            rescuable += 1
        if head in dict_readings:
            head_in_R += 1

    rate = rescuable / max(nB, 1)
    est_total = int(rate * total_rev)
    print("\n" + "=" * 64)
    print(" B) REVIEW `below_visual_threshold` — rescue estimate (no retrain)")
    print("=" * 64)
    print(f"  pile size (below_visual_threshold): {total_rev}")
    print(f"  sampled & embeddable               : {nB}")
    print(f"  head picks a dict reading          : {head_in_R/max(nB,1):.1%}")
    print(f"  RESCUABLE (head==bank on a dict reading): {rescuable}/{nB} = {rate:.1%}")
    print(f"  -> extrapolated rescuable over the pile : ~{est_total} labels")
    print(f"  -> expected precision (GOLD agree proxy): ~{agree_prec:.1%}")
    print(f"\n  Net: a head+bank consensus vote could move ~{est_total} REVIEW rows to a")
    print(f"  new visual-consensus tier at ~{agree_prec:.0%} precision, WITHOUT retraining.")
    print("  (Retraining the encoder on GOLD+SILVER — the kuzushiji loop — would raise this.)")

    out = {"gold_val": {"n": nA, "bank_r1": round(bank_hit/max(nA,1), 4),
                        "head_r1": round(head_hit/max(nA,1), 4),
                        "agree_coverage": round(agree/max(nA,1), 4),
                        "agree_precision": round(agree_prec, 4)},
           "review_rescue": {"pile": total_rev, "sampled": nB, "rescuable_rate": round(rate, 4),
                            "estimated_rescued": est_total, "expected_precision": round(agree_prec, 4)}}
    p = HERE / "results" / "head_rescue.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  -> {p}")


if __name__ == "__main__":
    main()
