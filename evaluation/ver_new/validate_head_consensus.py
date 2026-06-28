"""Validate the new SILVER sub-tier `s3_head_bank_consensus` (#4) BEFORE adopting it.

Two parts:
  A) GROUND-TRUTHED upper bound — on held-out GOLD test, simulate the exact
     production rule (bank-reject AND head_agree AND top∈dict) and measure how often
     the agreed char IS the true char. This is the OPTIMISTIC (easy-regime) precision.
  B) HUMAN-AUDIT PACKET — sample the ACTUAL s3_head_bank_consensus rescues (from a
     #4 build's labels.csv), cut their crops from the page, and render review.html +
     verify.csv so a human can judge precision on the REAL (hard) regime. Then:
       fill verify.csv (human_correct 1/0) -> measure_precision.py
       -> conformal_reject.py --audit  for the GUARANTEED threshold.

Run:
  # A) upper bound on GOLD test:
  .venv/bin/python evaluation/ver_new/validate_head_consensus.py --measure
  # B) build the audit packet from a #4 build:
  .venv/bin/python evaluation/ver_new/validate_head_consensus.py --packet /tmp/s4full --n 200
"""
from __future__ import annotations

import argparse
import csv
import html
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
FD_DIR = REPO / "gannhanocr-fd"


def fd_path(ch):
    if not ch or len(ch) != 1:
        return None
    hx = f"{ord(ch):X}"
    p = FD_DIR / hx[:2] / f"U+{hx}.png"
    if p.exists():
        return p
    hits = list(FD_DIR.rglob(f"U+{hx}.png"))
    return hits[0] if hits else None


def measure(args):
    """Part A — GOLD-test precision of the production head-consensus rule."""
    from pipeline.step0_setup import load_config
    from core.text.dictionary import load_qn_to_nom
    from evaluation.ver_new.visual_signal import VisualS3, _is_cjk
    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   simfont_dir=str(REPO / paths.get("fd_cache_similar", "")) if paths.get("fd_cache_similar") else "")
    if not vs3.enc.has_head:
        sys.exit("no ArcFace head — #4 unavailable.")
    D = Path(args.dataset)
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == "GOLD" and r["split"] == "test" and r["image"] and r["syllable"] and r["label"]]
    fired = correct = n = 0
    for r in rows:
        emb = vs3.enc.embed_path(str(D / r["image"]))
        if emb is None:
            continue
        true = r["label"]; syl = (r["syllable"] or "").lower()
        R = qn.get(syl, [])
        cands = []
        for c in ([r["ocr_char"]] if r["ocr_char"] else []) + R:
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if true not in cands:
            cands.append(true)
        if len(cands) < 2:
            continue
        n += 1
        dec = vs3.decide(emb, cands)
        # production head-consensus rule: bank rejected, head agrees, top is a dict reading
        if dec["reject"] and dec.get("head_agree") and dec["top_char"] in set(R):
            fired += 1
            correct += (dec["top_char"] == true)
    prec = correct / max(fired, 1)
    print("=" * 64)
    print(" A) GOLD-test precision of `s3_head_bank_consensus` (UPPER BOUND)")
    print("=" * 64)
    print(f"  GOLD test decisions      : {n}")
    print(f"  rule fired (rescued)     : {fired}  ({fired/max(n,1):.1%} of GOLD test)")
    print(f"  precision (agreed==true) : {prec:.3f}")
    print(f"\n  -> This is the EASY-regime upper bound. The real rescues are harder")
    print(f"     (below_visual_threshold) -> human-audit packet (--packet) for the true number.")
    import json
    json.dump({"gold_test_n": n, "fired": fired, "precision": round(prec, 4)},
              open(HERE / "results" / "validate_head_consensus.json", "w", encoding="utf-8"), indent=2)


def packet(args):
    """Part B — human-audit packet for the ACTUAL head-consensus rescues."""
    from pipeline.step0_setup import load_config
    cfg = load_config(args.config)
    name_map = {b["name"][12:]: b["name"] for b in cfg["books"]}
    data_root = REPO / cfg["paths"]["data_dir"]
    from evaluation.ver_new.bbox_fix import tighten_box

    src = Path(args.packet)
    rows = [r for r in csv.DictReader(open(src / "labels.csv", encoding="utf-8"))
            if r["tier"] == "SILVER" and r["rule"] == "s3_head_bank_consensus" and r["bbox"] and r["bbox"] != "null"]
    print(f"s3_head_bank_consensus rows in {src}: {len(rows)}")
    # stratify by class frequency band so the rare tail (error-prone) is covered
    freq = Counter(r["label"] for r in rows if r["label"])
    rare = [r for r in rows if freq.get(r["label"], 0) < 5]
    common = [r for r in rows if freq.get(r["label"], 0) >= 5]
    random.seed(args.seed); random.shuffle(rare); random.shuffle(common)
    half = args.n // 2
    sample = rare[:half] + common[:args.n - len(rare[:half])]
    random.shuffle(sample)

    out = HERE / "eval_sample_head"
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    pages = {}
    def page_img(book, page):
        k = (book, page)
        if k not in pages:
            p = data_root / name_map.get(book, book) / "pages" / f"{page}.png"
            pages[k] = cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None
        return pages[k]

    import json as _json
    vrows, cards = [], []
    for i, r in enumerate(sample):
        sid = f"H{i:04d}"
        img = page_img(r["book"], r["page"])
        crop_dst = ""
        if img is not None:
            try:
                x1, y1, x2, y2 = [int(v) for v in _json.loads(r["bbox"])]
                pw, ph = int((x2-x1)*0.12), int((y2-y1)*0.12)
                H, W = img.shape[:2]
                crop = img[max(0,y1-ph):min(H,y2+ph), max(0,x1-pw):min(W,x2+pw)]
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
                tb = tighten_box(g)
                if tb is not None:
                    a,c,b,d = tb
                    crop = crop[c:d, a:b]
                if crop.size:
                    cv2.imwrite(str(out/"imgs"/f"{sid}_crop.png"), crop); crop_dst = f"imgs/{sid}_crop.png"
            except Exception:
                pass
        ref = fd_path(r["label"]); ref_dst = ""
        if ref:
            shutil.copy(ref, out/"imgs"/f"{sid}_ref.png"); ref_dst = f"imgs/{sid}_ref.png"
        vrows.append({"sample_id": sid, "book": r["book"], "page": r["page"], "column": r["column"],
                      "ocr_char": r["ocr_char"], "syllable": r["syllable"], "label": r["label"],
                      "unicode": r["unicode"], "s3_cosine": r["s3_cosine"], "image": crop_dst,
                      "human_correct": "", "human_label": "", "notes": ""})
        cards.append(f"<tr><td>{sid}</td>"
                     f"<td>{'<img src=\"'+crop_dst+'\" height=90>' if crop_dst else '—'}</td>"
                     f"<td><b style='font-size:30px'>{html.escape(r['label'])}</b> {r['unicode']}<br>"
                     f"<small>âm {html.escape(r['syllable'])} · ocr {html.escape(r['ocr_char'])}</small></td>"
                     f"<td>{'<img src=\"'+ref_dst+'\" height=90>' if ref_dst else '—'}</td>"
                     f"<td style='font-size:22px;color:#aaa'>☐</td></tr>")

    fields = list(vrows[0].keys()) if vrows else []
    with open(out/"verify.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(vrows)
    open(out/"review.html", "w", encoding="utf-8").write(
        "<!doctype html><meta charset=utf-8><style>body{font-family:sans-serif;margin:20px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px;vertical-align:middle}"
        "th{background:#f5f5f5}</style>"
        f"<h2>Soát tier head∩bank ({len(vrows)} mẫu) — crop CÓ đúng chữ đề xuất không?</h2>"
        "<p>Điền <code>human_correct</code>=1 (đúng)/0 (sai) vào verify.csv; sai thì ghi chữ đúng vào human_label.</p>"
        "<table><tr><th>id</th><th>crop</th><th>đề xuất (head∩bank)</th><th>tham chiếu</th><th>✓?</th></tr>"
        + "".join(cards) + "</table>")
    print(f"  packet -> {out}  ({len(vrows)} mẫu: {len(rare[:half])} hiếm + {len(sample)-len(rare[:half])} phổ biến)")
    print(f"  mở {out}/review.html, điền {out}/verify.csv, rồi:")
    print(f"    measure_precision.py  (precision thật + Wilson CI)")
    print(f"    conformal_reject.py --audit {out}/verify.csv   (ngưỡng có bảo đảm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--measure", action="store_true", help="Part A: GOLD-test precision upper bound")
    ap.add_argument("--packet", default="", help="Part B: build audit packet from this #4 build dir")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    if args.measure:
        measure(args)
    if args.packet:
        packet(args)
    if not args.measure and not args.packet:
        print("pass --measure (Part A) and/or --packet <dir> (Part B)")


if __name__ == "__main__":
    main()
