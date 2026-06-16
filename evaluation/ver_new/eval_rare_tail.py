"""TEST for the RARE-TAIL problem — quantify (a) the S3 gap on rare vs common
classes, and (b) the POTENTIAL of the structural IDS/radical route (CCR-CLIP /
HierCode) to fix it, measured on the real corpus without training anything.

  (a) S3 retrieval@1 and the head's 1,591-way top-1 accuracy on GOLD test, split by
      class frequency: rare (< RARE_N GOLD crops) vs common. The rare-minus-common
      gap is exactly what the structural references would target.
  (b) STRUCTURAL DISAMBIGUATION POTENTIAL — for the rare-tail cases S3 gets WRONG
      (its top candidate != true), do the true and the wrongly-picked char have
      DIFFERENT IDS radical sets? If yes, a structural signal (image->radicals)
      could break the tie. The fraction with distinct radicals is an upper bound on
      how much CCR-CLIP/HierCode could recover here — a real, data-grounded "is it
      worth it?" number, complementing ids_coverage.py (which showed 96% of the rare
      tail HAS a decomposition).

Run:
  .venv/bin/python evaluation/ver_new/eval_rare_tail.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
from evaluation.ver_new.ids_coverage import load_ids, radicals     # noqa: E402

HERE = Path(__file__).resolve().parent
RARE_N = 5


def cands_of(ocr_char, syllable, qn, true):
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
    args = ap.parse_args()

    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]))
    enc = vs3.enc
    ids = load_ids()
    D = Path(args.dataset)
    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))

    freq = Counter(r["label"] for r in rows if r["tier"] == "GOLD" and r["label"])
    test = [r for r in rows if r["tier"] == "GOLD" and r["split"] == "test"
            and r["image"] and r["syllable"] and r["label"]]

    res = {}
    rare_errs_distinct = rare_errs = 0
    for grp in ("rare", "common"):
        sel = [r for r in test if (freq.get(r["label"], 0) < RARE_N) == (grp == "rare")]
        bank_hit = head_hit = n = 0
        for r in sel:
            emb = enc.embed_path(str(D / r["image"]))
            if emb is None:
                continue
            true = r["label"]
            R = cands_of(r["ocr_char"], r["syllable"], qn, true)
            if len(R) < 2:
                continue
            n += 1
            bank = vs3.decide(emb, R)["top_char"]
            bank_hit += (bank == true)
            tk = enc.predict_topk(emb, 1)
            head = tk[0][0] if tk else None
            head_hit += (head == true)
            if grp == "rare" and bank != true:
                rare_errs += 1
                rt = radicals(true, ids) if true in ids else {true}
                rp = radicals(bank, ids) if bank in ids else {bank}
                if rt != rp:                      # different radical signature -> structurally separable
                    rare_errs_distinct += 1
        res[grp] = {"n": n, "bank_retrieval@1": round(bank_hit / max(n, 1), 4),
                    "head_top1_1591way": round(head_hit / max(n, 1), 4)}

    print("=" * 64)
    print(" RARE-TAIL test — S3 gap + structural-route potential")
    print("=" * 64)
    for grp in ("common", "rare"):
        d = res[grp]
        print(f"  {grp:7s} (n={d['n']:4d}): bank retrieval@1 {d['bank_retrieval@1']:.1%} | "
              f"head 1591-way top1 {d['head_top1_1591way']:.1%}")
    gap = res["common"]["bank_retrieval@1"] - res["rare"]["bank_retrieval@1"]
    print(f"  -> rare-vs-common S3 gap: {gap*100:.1f} pts (the headroom for structural refs)")
    pot = rare_errs_distinct / max(rare_errs, 1)
    print(f"\n  STRUCTURAL POTENTIAL: of {rare_errs} rare-tail S3 errors, {rare_errs_distinct} "
          f"({pot:.1%}) have a DIFFERENT IDS radical set (true vs picked)")
    print(f"  -> upper bound on what an image->radical signal (CCR-CLIP/HierCode) could disambiguate here.")
    print(f"  (Combined with ids_coverage.py: 96% of the rare tail HAS a decomposition.)")

    out = {"rare_n_threshold": RARE_N, "groups": res, "rare_vs_common_gap": round(gap, 4),
           "rare_errors": rare_errs, "rare_errors_radically_distinct": rare_errs_distinct,
           "structural_potential": round(pot, 4)}
    p = HERE / "results" / "eval_rare_tail.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
