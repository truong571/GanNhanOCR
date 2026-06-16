"""Roadmap #4 — leave-one-book-out (LOBO) S3 evaluation: the honest, leakage-
controlled headline number.

The `retrieval@1 ~= 89%` figure is an UPPER BOUND: the crop prototypes and the
query crops come from the SAME 3 books (yen2/yen4/yen11) — same scanner, ink and
woodblock carving. A reviewer will (correctly) say it measures within-book
recall, not generalisation to a new book. With only 3 books the clean test is
leave-one-book-out: build the candidate prototype bank from the OTHER two books
(+ the FontDiffusion glyph, which never leaks), query on the held-out book, rotate.

For every held-out book B and every GOLD test crop in B we rank the real candidate
set R = {ocr_char} u dict-readings by the best cosine to the candidate's reference
bank, and report retrieval@1 + AURC under two reference regimes:

  same-book : prototypes may come from B itself (the leaky setting = current 89%)
  cross-book: prototypes only from the OTHER books + FD glyph (the deployment number)

The same-book MINUS cross-book gap IS the leakage. Reporting both — and expecting
cross-book to drop — is the contribution, per "A Metric Learning Reality Check"
(Musgrave et al., ECCV 2020) on leakage-aware evaluation.

NOTE: this varies only the REFERENCE bank, not the encoder — the backbone still
trained on all three books, so cross-book here isolates *prototype* leakage, not
*representation* leakage. A fully book-held-out encoder (retrain on Kaggle
excluding B) is the stricter test; state cross-book here as controlling reference
leakage with the backbone caveat.

Run:
  .venv/bin/python evaluation/ver_new/eval_book_disjoint.py            # all GOLD test crops
  .venv/bin/python evaluation/ver_new/eval_book_disjoint.py --n 500    # 500 queries/book (quick)
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
from evaluation.ver_new.s3_risk_coverage import risk_coverage      # noqa: E402

HERE = Path(__file__).resolve().parent


def _candidates(ocr_char, syllable, qn, true):
    cands = []
    for c in ([ocr_char] if ocr_char else []) + qn.get((syllable or "").lower(), []):
        if _is_cjk(c) and c not in cands:
            cands.append(c)
    if true and _is_cjk(true) and true not in cands:
        cands.append(true)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--proto-k", type=int, default=6, help="max prototypes per (book,class)")
    ap.add_argument("--n", type=int, default=0, help="cap queries per held-out book (0 = all)")
    args = ap.parse_args()

    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    # skip the global crop-proto build (pass a non-existent index) — we build our own
    # per-book banks below; we only need the encoder + FD glyph index from VisualS3.
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   proto_index=str(REPO / "__no_global_proto__"))
    enc, fd_index = vs3.enc, vs3.fd_index
    D = Path(args.dataset)

    # ---- load GOLD rows; build per-(book,class) prototype paths from TRAIN crops ----
    train_paths = defaultdict(list)        # (book,label) -> [crop paths]
    test_rows = defaultdict(list)          # book -> [row]
    for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8")):
        if r["tier"] != "GOLD" or not r["image"] or not r["label"] or not r["syllable"]:
            continue
        if r["split"] == "train":
            train_paths[(r["book"], r["label"])].append(str(D / r["image"]))
        elif r["split"] == "test":
            test_rows[r["book"]].append(r)
    books = sorted(test_rows)
    print(f"books: {books} | test crops/book: " +
          ", ".join(f"{b}:{len(test_rows[b])}" for b in books), flush=True)

    proto_emb_cache: dict = {}
    def proto_embs(book, ch):
        key = (book, ch)
        if key in proto_emb_cache:
            return proto_emb_cache[key]
        embs = [e for p in train_paths.get(key, [])[:args.proto_k]
                if (e := enc.embed_path(p)) is not None]
        proto_emb_cache[key] = embs
        return embs

    fd_cache: dict = {}
    def fd_emb(ch):
        if ch in fd_cache:
            return fd_cache[ch]
        p = fd_index.get(ch)
        e = enc.embed_path(p) if p else None
        fd_cache[ch] = e
        return e

    def score(ce, ch, allowed_books):
        best = -2.0
        for b in allowed_books:
            for e in proto_embs(b, ch):
                best = max(best, enc.cosine_raw(ce, e))
        fe = fd_emb(ch)
        if fe is not None:
            best = max(best, enc.cosine_raw(ce, fe))
        return best

    rows_out = []
    macro = {"same": [], "cross": []}      # collect per-book retrieval rates for macro avg
    print("\n=== leave-one-book-out retrieval@1 (reference-bank leakage isolated) ===")
    print(f"  {'held-out book':14s} {'n':>5s}  {'same-book':>10s} {'cross-book':>10s}  {'gap':>6s}  "
          f"{'AURC same':>9s} {'AURC cross':>10s}")
    for B in books:
        queries = test_rows[B][: args.n] if args.n else test_rows[B]
        cross_books = [b for b in books if b != B]
        recs = {"same": [], "cross": []}
        for r in queries:
            ce = enc.embed_path(str(D / r["image"]))
            if ce is None:
                continue
            true = r["label"]
            cands = _candidates(r["ocr_char"], r["syllable"], qn, true)
            if len(cands) < 2:
                continue
            for regime, allowed in (("same", books), ("cross", cross_books)):
                sc = {c: score(ce, c, allowed) for c in cands}
                top = max(sc, key=sc.get)
                recs[regime].append((float(sc[top]), 0.0, int(top == true)))
        out = {"book": B}
        for regime in ("same", "cross"):
            rr = recs[regime]
            acc = float(np.mean([c for _, _, c in rr])) if rr else 0.0
            _, _, _, aurc, _ = risk_coverage(rr) if len(rr) >= 10 else (0, 0, 0, float("nan"), 0)
            out[regime] = {"retrieval_at_1": round(acc, 4), "AURC": round(float(aurc), 4), "n": len(rr)}
            macro[regime].append(acc)
        gap = out["same"]["retrieval_at_1"] - out["cross"]["retrieval_at_1"]
        out["leakage_gap"] = round(gap, 4)
        rows_out.append(out)
        print(f"  {B:14s} {out['cross']['n']:5d}  {out['same']['retrieval_at_1']:>10.1%} "
              f"{out['cross']['retrieval_at_1']:>10.1%}  {gap:>6.1%}  "
              f"{out['same']['AURC']:>9.4f} {out['cross']['AURC']:>10.4f}", flush=True)

    msame, mcross = float(np.mean(macro["same"])), float(np.mean(macro["cross"]))
    print(f"\n  MACRO avg  same-book {msame:.1%}  cross-book {mcross:.1%}  "
          f"leakage gap {msame - mcross:.1%}")
    print("  cross-book = the honest deployment number (prototypes never from the query's book).")
    print("  CAVEAT: backbone trained on all 3 books -> this isolates REFERENCE leakage, not")
    print("  representation leakage; a book-held-out retrained encoder is the stricter test.")

    import json
    res = {"books": books, "proto_k": args.proto_k, "per_book": rows_out,
           "macro": {"same_book_retrieval_at_1": round(msame, 4),
                     "cross_book_retrieval_at_1": round(mcross, 4),
                     "leakage_gap": round(msame - mcross, 4)},
           "caveat": "reference-bank leakage only; backbone trained on all books."}
    out_p = HERE / "results" / "eval_book_disjoint.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {out_p}")


if __name__ == "__main__":
    main()
