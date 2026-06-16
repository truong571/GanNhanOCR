"""NHÓM 1 (apply-now, no-retrain) implemented + REALLY evaluated on the corpus.

Implements the two cleanly-evaluable Group-1 levers and measures them on the real
data with a held-out ground-truthed split:

  #1  HEAD∩BANK CONSENSUS  — the ArcFace head (1,591-way, already in the ckpt but
      unused by decide()) is a 2nd, independent visual classifier. Accept a label
      only when head and reference-bank agree on a DICTIONARY reading of the syllable
      and the head margin clears a calibrated threshold. (kuzushiji pseudo-label /
      Korean detect-then-cluster lesson.)
  #2  VARIANT-AWARE candidates — expand each dict reading with its Unihan semantic/
      Z/simplified/traditional variants, score them, and map the winner back to the
      canonical reading. Targets the `below_visual_threshold` false-rejections where
      the true glyph is a regional VARIANT of a dict reading (Hanja-OCR lesson).

EVALUATION (honest, ground-truthed):
  CALIBRATE on GOLD val  -> pick the head-margin threshold τ at a target precision.
  TEST on GOLD test      -> REAL precision / coverage / yield (true label known),
                            baseline (#1) vs variant-aware (#1+#2).
  APPLY to REVIEW         -> real yield over the full `below_visual_threshold` pile
                            (crops cut from the page), expected precision = GOLD-test.
  Caveat: GOLD precision is an upper bound for REVIEW (on REVIEW the true char may be
  outside R); the final number still needs the human audit (verify.csv). Reported.

Outputs: results/group1_rescue.json + results/rescued_labels.csv (the new tier).

Run:
  .venv/bin/python evaluation/ver_new/group1_rescue.py --target 0.95 --review-n 0   # 0 = all REVIEW
"""
from __future__ import annotations

import argparse
import csv
import json
import re
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
UNIHAN = HERE / "ids_data" / "Unihan_Variants.txt"
VAR_FIELDS = {"kSemanticVariant", "kZVariant", "kSimplifiedVariant", "kTraditionalVariant"}
_UPAT = re.compile(r"U\+([0-9A-Fa-f]+)")


def load_variants() -> dict[str, set[str]]:
    """char -> set of logical-variant chars (Unihan semantic/Z/simp/trad)."""
    var: dict[str, set[str]] = defaultdict(set)
    if not UNIHAN.exists():
        return var
    for ln in open(UNIHAN, encoding="utf-8"):
        if ln.startswith("#") or "\t" not in ln:
            continue
        parts = ln.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[1] not in VAR_FIELDS:
            continue
        try:
            src = chr(int(_UPAT.match(parts[0]).group(1), 16))
        except Exception:
            continue
        for m in _UPAT.findall(parts[2]):
            try:
                tgt = chr(int(m, 16))
            except ValueError:
                continue
            if tgt != src:
                var[src].add(tgt); var[tgt].add(src)
    return var


class Rescuer:
    def __init__(self, vs3, qn, variants):
        self.vs3 = vs3; self.enc = vs3.enc; self.qn = qn; self.var = variants
        self.lab2idx = {lab: i for i, lab in self.enc.classes.items()}

    def _head_rank(self, emb, cands):
        lg = self.enc.logits(emb)
        sc = [(c, float(lg[self.lab2idx[c]])) for c in cands if c in self.lab2idx]
        if not sc:
            return None, 0.0
        sc.sort(key=lambda t: -t[1])
        margin = sc[0][1] - (sc[1][1] if len(sc) > 1 else -1.0)
        return sc[0][0], margin

    def decide(self, emb, ocr_char, syllable, variant_mode):
        """Return (label|None, margin) where label is a CANONICAL dict reading that
        head and bank agree on (None if no consensus). variant_mode expands candidates."""
        R0 = [c for c in self.qn.get((syllable or "").lower(), []) if _is_cjk(c)]
        if not R0:
            return None, 0.0
        canon = {r: r for r in R0}                       # candidate char -> canonical reading
        cands = list(R0)
        if ocr_char and _is_cjk(ocr_char) and ocr_char not in canon:
            cands.append(ocr_char); canon[ocr_char] = None      # ocr_char is NOT a valid label
        if variant_mode:
            for r in R0:
                for v in self.var.get(r, ()):
                    if _is_cjk(v) and v not in canon:
                        cands.append(v); canon[v] = r          # variant maps back to reading r
        if len(cands) < 2:
            # single dict reading: still require head+bank to back it
            cands = R0
        bank = self.vs3.decide(emb, cands)["top_char"] if len(cands) >= 2 else (R0[0] if R0 else None)
        head, margin = self._head_rank(emb, cands)
        if head is None or bank is None:
            return None, 0.0
        cb, ch = canon.get(bank), canon.get(head)
        # accept: both map to the SAME valid dict reading
        if ch is not None and ch == cb:
            return ch, margin
        return None, margin


def cut_embed(enc, img, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    tb = tighten_box(gray)
    if tb is not None:
        a, c, b, d = tb
        if b - a >= 8 and d - c >= 8:
            gray = gray[c:d, a:b]
    return enc.embed_gray(gray)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--target", type=float, default=0.95, help="target precision for the rescue tier")
    ap.add_argument("--review-n", type=int, default=0, help="REVIEW rows to apply on (0 = all)")
    args = ap.parse_args()

    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   simfont_dir=str(REPO / paths.get("fd_cache_similar", "")) if paths.get("fd_cache_similar") else "")
    if not vs3.enc.has_head:
        sys.exit("ckpt has no ArcFace head.")
    variants = load_variants()
    print(f"  Unihan variant links: {len(variants)} chars", flush=True)
    R = Rescuer(vs3, qn, variants)
    D = Path(args.dataset)
    name_map = {b["name"][12:]: b["name"] for b in cfg["books"]}
    data_root = REPO / cfg["paths"]["data_dir"]

    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))

    def gold(split):
        return [r for r in rows if r["tier"] == "GOLD" and r["split"] == split
                and r["image"] and r["syllable"] and r["label"]]

    # ---- collect signals on a split (ground-truthed) ----
    def collect(split, variant_mode):
        out = []  # (accepted_label or None, margin, true)
        for r in gold(split):
            emb = vs3.enc.embed_path(str(D / r["image"]))
            if emb is None:
                continue
            lab, margin = R.decide(emb, r["ocr_char"], r["syllable"], variant_mode)
            out.append((lab, margin, r["label"]))
        return out

    print("collecting GOLD val signals (calibration) ...", flush=True)
    val_base = collect("val", False)
    val_var = collect("val", True)

    def sweep(sig):
        """pick min margin τ that reaches target precision; maximise coverage."""
        cand = [s for s in sig if s[0] is not None]
        best = None
        for tau in np.linspace(0, 8, 81):
            acc = [(l, t) for (l, m, t) in cand if m >= tau]
            if len(acc) < 30:
                continue
            prec = sum(l == t for l, t in acc) / len(acc)
            cov = len(acc) / len(sig)
            if prec >= args.target and (best is None or cov > best["cov"]):
                best = {"tau": round(float(tau), 3), "prec": round(prec, 4), "cov": round(cov, 4), "n": len(acc)}
        if best is None:
            best = {"tau": 8.0, "prec": 0.0, "cov": 0.0, "n": 0, "note": "target unreachable"}
        return best

    cal_base, cal_var = sweep(val_base), sweep(val_var)
    print(f"  calibrated τ  base={cal_base}  variant={cal_var}", flush=True)

    # ---- REAL test on held-out GOLD test ----
    def evaluate(split, variant_mode, tau):
        sig = collect(split, variant_mode)
        acc = [(l, t) for (l, m, t) in sig if l is not None and m >= tau]
        prec = sum(l == t for l, t in acc) / len(acc) if acc else 0.0
        return {"precision": round(prec, 4), "coverage": round(len(acc) / max(len(sig), 1), 4),
                "accepted": len(acc), "n": len(sig)}

    print("evaluating on held-out GOLD test ...", flush=True)
    test_base = evaluate("test", False, cal_base["tau"])
    test_var = evaluate("test", True, cal_var["tau"])
    print(f"  GOLD test  #1(head∩bank)       : {test_base}")
    print(f"  GOLD test  #1+#2(variant-aware): {test_var}")

    # choose the better config (higher coverage at >= target precision on test)
    use_var = (test_var["precision"] >= args.target and test_var["coverage"] > test_base["coverage"])
    tau = cal_var["tau"] if use_var else cal_base["tau"]
    chosen = "variant-aware (#1+#2)" if use_var else "head∩bank (#1)"
    test_chosen = test_var if use_var else test_base
    print(f"  -> chosen config: {chosen} (τ={tau})", flush=True)

    # ---- APPLY to REVIEW below_visual_threshold (real yield) ----
    rev = [r for r in rows if r["tier"] == "REVIEW" and r["rule"] == "below_visual_threshold"
           and r["bbox"] and r["bbox"] != "null" and r["syllable"]]
    if args.review_n:
        rev = rev[: args.review_n]
    print(f"applying to REVIEW below_visual_threshold ({len(rev)} rows) ...", flush=True)
    page_cache = {}
    def page_img(book, page):
        k = (book, page)
        if k not in page_cache:
            p = data_root / name_map.get(book, book) / "pages" / f"{page}.png"
            page_cache[k] = cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None
        return page_cache[k]

    rescued = []
    seen_emb = 0
    for i, r in enumerate(rev):
        img = page_img(r["book"], r["page"])
        if img is None:
            continue
        try:
            emb = cut_embed(vs3.enc, img, json.loads(r["bbox"]))
        except Exception:
            emb = None
        if emb is None:
            continue
        seen_emb += 1
        lab, margin = R.decide(emb, r["ocr_char"], r["syllable"], use_var)
        if lab is not None and margin >= tau:
            rescued.append({**{k: r[k] for k in ("book", "page", "column", "ocr_char", "syllable", "bbox")},
                            "rescued_label": lab, "unicode": f"U+{ord(lab):04X}", "head_margin": round(margin, 3)})
        if (i + 1) % 3000 == 0:
            print(f"  ... {i+1}/{len(rev)}  rescued so far {len(rescued)}", flush=True)

    yield_rate = len(rescued) / max(seen_emb, 1)
    est_full = int(yield_rate * len(rev))      # == len(rescued) if review_n==0

    # current usable totals for context
    tiers = defaultdict(int)
    for r in rows:
        tiers[r["tier"]] += 1
    usable = tiers["GOLD"] + tiers["SILVER"]

    print("\n" + "=" * 66)
    print(" REAL RESULT — Group-1 rescue on the actual corpus")
    print("=" * 66)
    print(f"  config used                 : {chosen}  (τ={tau})")
    print(f"  GOLD-test precision/coverage: {test_chosen['precision']:.1%} / {test_chosen['coverage']:.1%} "
          f"({test_chosen['accepted']}/{test_chosen['n']})  <- ground-truthed")
    print(f"  REVIEW pile (below_visual)  : {len(rev)}  (embeddable {seen_emb})")
    print(f"  RESCUED (new labels)        : {len(rescued)}  ({yield_rate:.1%} of pile)")
    print(f"  expected precision on these : ~{test_chosen['precision']:.0%} (GOLD proxy; audit for final)")
    print(f"  usable char-labels: {usable}  ->  {usable + len(rescued)}  (+{len(rescued)}, "
          f"+{len(rescued)/max(usable,1)*100:.1f}%)  | SILVER {tiers['SILVER']} -> {tiers['SILVER']+len(rescued)}")

    # write rescued labels + json
    out_csv = HERE / "results" / "rescued_labels.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["book", "page", "column", "ocr_char", "syllable",
                                          "rescued_label", "unicode", "head_margin", "bbox"])
        w.writeheader(); w.writerows(rescued)
    res = {"target_precision": args.target, "variant_links": len(variants),
           "calibration": {"base": cal_base, "variant": cal_var},
           "gold_test": {"head_bank_#1": test_base, "variant_aware_#1+#2": test_var},
           "chosen_config": chosen, "tau": tau,
           "review_pile": len(rev), "embeddable": seen_emb,
           "rescued": len(rescued), "rescue_rate": round(yield_rate, 4),
           "expected_precision": test_chosen["precision"],
           "usable_before": usable, "usable_after": usable + len(rescued)}
    json.dump(res, open(HERE / "results" / "group1_rescue.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  -> results/group1_rescue.json , results/rescued_labels.csv ({len(rescued)} rows)")
    print("  NOTE: GOLD precision is an upper bound for REVIEW (true char may be outside R);")
    print("  audit a sample of rescued_labels.csv (export_eval_sample/measure_precision) for the final number.")


if __name__ == "__main__":
    main()
