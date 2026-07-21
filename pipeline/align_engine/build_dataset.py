"""Build the FINAL labeled dataset end-to-end (NEW pipeline, no A/B).

Two passes:
  PASS 1 — align every page (production-faithful: detect_nom_columns_v3 + parse_v5
           + bbox_fix offset + banded anchored DP + re-segment) and tier each pair
           via consensus.decide_label.
  PROMOTE — cross-page-consistent REVIEW "unconfirmed" pairs -> SYLLABLE tier
           (semantic/nghĩa borrowings the phonetic dict can't contain). [#6]
  SPLIT  — leakage-safe train/val/test by group=(book,page,column). [#4]
  PASS 2 — materialize crops (GOLD + SILVER + SYLLABLE; +REVIEW with --crop-review)
           with per-crop quality columns (ink%, size, md5, seg_flag). [#3,#5]

Tiers (label_level separates char- vs syllable-supervision):
  GOLD     label_level=char     dict-confirmed char (direct, or unique similar-bridge)
  SILVER   label_level=char     visual S3 (OFF until a Nôm-trained model replaces DINOv2)
  SYLLABLE label_level=syllable char unconfirmed but syllable reliable & cross-page-consistent
  REVIEW   label_level=''       not usable as a label (kept in manifest with bbox)

Run:
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/build_dataset.py
  # options: --no-tighten --pad 0.12 --limit N --out <dir> --crop-review --use-s3
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom, load_similarity_dict  # noqa: E402
from core.text.text_utils import is_plausible_qn_syllable  # noqa: E402
from pipeline.align_engine.align_production import align_page          # noqa: E402
from pipeline.align_engine.consensus import decide_label              # noqa: E402
from pipeline.align_engine.bbox_fix import tighten_box                # noqa: E402


def _book_code(name: str) -> str:
    """Short, meaningful book code for the labels.csv `book` field.

    'SachThanhTruyen2' -> 'stt2', 'SachThanhTruyen11' -> 'stt11' (STT = Sách
    Thánh Truyện). Replaces the old `book[12:]` substring hack, which sliced
    'SachThanhTruyen2' into the accidental, meaningless 'yen2'. The audit reverse
    map (audit_grid.book_to_scan_dir) keys off the trailing digits, so it keeps
    working for either code. Falls back to the lowercased name for other books.
    """
    m = re.match(r"SachThanhTruyen(\d+)$", name)
    return f"stt{m.group(1)}" if m else name.lower()

# SYLLABLE-tier gate (cross-page consistency of an unconfirmed char's reading).
SYL_MIN_OCC = 5      # the (char,syllable) must occur >= this many times corpus-wide
SYL_MIN_PAGES = 3    # on >= this many distinct pages
SYL_MIN_PURITY = 0.6  # and be the dominant syllable for that char by this share


def syllable_gate(records, unconf, min_occ=SYL_MIN_OCC, min_pages=SYL_MIN_PAGES,
                  min_purity=SYL_MIN_PURITY):
    """(ocr_char, LOWERCASED syllable) pairs passing the cross-page consistency gate.

    Case-insensitive on the syllable so 'Nhị' and 'nhị' merge into one class — the
    cased keying used to split them, diluting the occurrence/purity thresholds and
    dropping ~1,131 labels. Pure + deterministic; unit-tested in phase1_engine_selftest.
    """
    cnt = defaultdict(Counter)
    pages_of = defaultdict(lambda: defaultdict(set))
    for r in records:
        if r["tier"] == "REVIEW" and r["rule"] in unconf and r["ocr_char"]:
            syl = str(r["syllable"]).lower()
            if not is_plausible_qn_syllable(syl):
                continue      # garbage ('0'/'2017') must never become a SYLLABLE target
            cnt[r["ocr_char"]][syl] += 1
            pages_of[r["ocr_char"]][syl].add((r["book"], r["page"]))
    syl_ok = set()
    for ch, c in cnt.items():
        syl, n = c.most_common(1)[0]
        if (n >= min_occ and len(pages_of[ch][syl]) >= min_pages
                and n / sum(c.values()) >= min_purity):
            syl_ok.add((ch, syl))
    return syl_ok


def maybe_s3(p, page_png, qn_to_nom, vs3):
    if vs3 is None or not p.get("ocr_char"):
        return None
    if not (p.get("matched") or p.get("anchored")):
        return None
    cands = qn_to_nom.get((p["syllable"] or "").lower(), [])
    if p["ocr_char"] in cands:
        return None
    return vs3.compute(page_png, p.get("bbox"), p["ocr_char"], cands)


def _seg_flag(crop_gray) -> str:
    """Cheap advisory flag: 'tall' crops may be a merged 2-glyph or a tall char."""
    h, w = crop_gray.shape[:2]
    return "tall" if h > 1.8 * max(w, 1) else "ok"


def save_crop(img, bbox, pad, path: Path, tighten: bool = True) -> dict | None:
    """Cut + (tighten) + save a crop; return per-crop quality stats or None."""
    if img is None or not bbox:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if tighten:
        tb = tighten_box(gray)
        if tb is not None:
            a, c, b, d = tb
            crop = crop[c:d, a:b]
            gray = gray[c:d, a:b]
    if crop.size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    ch, cw = crop.shape[:2]
    return {
        "ink": round(float((gray < 128).mean()), 3),
        "w": cw, "h": ch,
        "md5": hashlib.md5(path.read_bytes()).hexdigest()[:12],
        "seg": _seg_flag(gray),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--out", default=str(REPO / "dataset_out"))
    ap.add_argument("--use-s3", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="fail loud if S3 can't load (else SILVER silently -> REVIEW)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--no-tighten", action="store_true")
    ap.add_argument("--crop-review", action="store_true",
                    help="also materialize REVIEW crops (kept in labels.csv either way)")
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--reseg", default="midpoint",
                    choices=["midpoint", "valley_n", "valley_guarded", "detector"],
                    help="column re-segmentation for crop boxes (default midpoint; valley_* are "
                         "opt-in experiments — see seg_valley_n_ab.py / seg_smart_ab.py). "
                         "valley_guarded needs the encoder (auto-loaded). detector uses a trained "
                         "char_detector/detector.pt (Kaggle; falls back to midpoint if absent).")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    paths = config["paths"]
    qn_to_nom = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    qn_dict_set = set(qn_to_nom.keys())
    similar = load_similarity_dict(str(REPO / paths["similar_dict"]))
    data_root = REPO / paths["data_dir"]

    vs3 = None
    if args.use_s3:
        from pipeline.align_engine.visual_signal import VisualS3
        try:
            print("Loading S3 (trained Nôm embedder + FD) ...", flush=True)
            # fd_cache_similar (optional): a glyph cache in a font SIMILAR to the
            # crops. When present it becomes the "simfont" reference tier (smaller
            # domain gap than FD). Absent -> simfont tier off (current state).
            simfont = str(REPO / paths["fd_cache_similar"]) if paths.get("fd_cache_similar") else ""
            vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                           simfont_dir=simfont)
        except Exception as e:
            if getattr(args, "strict", False):
                raise RuntimeError(
                    f"[S3 STRICT] S3 failed to load ({type(e).__name__}: {e}). "
                    "SILVER would silently collapse to REVIEW. Fix the nom-embed/best.pt "
                    "+ gannhanocr-fd checkpoints, or drop --strict to build GOLD-only."
                ) from e
            print(f"  [S3 OFF] {type(e).__name__}: {e}\n  -> SILVER bỏ qua; GOLD/SYLLABLE "
                  "vẫn chạy. (Cần checkpoint nom-embed/best.pt — train ở nom_classifier/.) "
                  "Dùng --strict để fail loud thay vì degrade âm thầm.",
                  flush=True)
            vs3 = None

    # encoder for reseg_mode=valley_guarded (the MLS guard) — reuse S3's if loaded,
    # else load just the encoder; absent -> _pick_reseg falls back to midpoint.
    reseg_encoder = vs3.enc if vs3 is not None else None
    if args.reseg == "valley_guarded" and reseg_encoder is None:
        try:
            from pipeline.align_engine.nom_classifier.infer import NomEncoder
            from pipeline.align_engine.visual_signal import _find_ckpt
            reseg_encoder = NomEncoder(_find_ckpt(REPO))
            print("  [reseg] valley_guarded: encoder loaded for the MLS guard.", flush=True)
        except Exception as e:
            print(f"  [reseg] valley_guarded needs the encoder ({e}) -> midpoint fallback.", flush=True)
    if args.reseg != "midpoint":
        print(f"  [reseg] mode = {args.reseg}", flush=True)

    # ---------- PASS 1: align all pages, collect records (no crop yet) ----------
    records = []
    pages_done = 0
    for b in config["books"]:
        book = b["name"]
        data_dir = data_root / book
        trans = sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
        trans = [t for t in trans if not t.endswith("_qn_ocr_cache.json")]
        if args.limit:
            trans = trans[:args.limit]
        print(f"[align] {book}: {len(trans)} pages ...", flush=True)
        for pi, tf in enumerate(trans):
            page = Path(tf).stem
            try:
                rec = align_page(page, data_dir, qn_dict_set, qn_to_nom, similar, "new",
                                 reseg_mode=args.reseg, encoder=reseg_encoder)
            except Exception as e:
                print(f"   [warn] {book}/{page}: {type(e).__name__}: {e}", flush=True)
                continue
            if rec is None:
                continue
            pages_done += 1
            page_png = str(data_dir / "pages" / f"{page}.png")
            for idx, p in enumerate(rec["pairs"]):
                s3 = maybe_s3(p, page_png, qn_to_nom, vs3) if vs3 else None
                dec = decide_label(p.get("ocr_char"), p["syllable"], p.get("matched", False),
                                   qn_to_nom, similar, s3=s3, anchored=p.get("anchored", False))
                records.append({
                    "book": _book_code(book), "page": page, "column": p["column"], "idx": idx,
                    "page_png": page_png, "ocr_char": p.get("ocr_char") or "",
                    # Canonical lowercase syllable for ALL tiers (not just SYLLABLE
                    # below): the QN→Nôm dict and decide_label are already case-folded,
                    # so keeping raw OCR case here only fragmented the reading vocabulary
                    # (Nhị/nhị/NHỊ → 3 classes) and inflated the distinct-syllable count.
                    # NFC + modern tone-mark placement are already applied upstream (parse_v5).
                    "syllable": str(p["syllable"]).lower(), "bbox": p.get("bbox"),
                    "tier": dec.tier, "rule": dec.rule_id, "label": dec.label or "",
                    "s3_cosine": round(s3.cosine, 3) if s3 else "",
                })

    # ---------- PROMOTE: cross-page-consistent unconfirmed -> SYLLABLE [#6] ----------
    # The unconfirmed pool = REVIEW rows with an ocr_char that S1∩S2 didn't confirm.
    # Without S3 the rule is 'unconfirmed_no_s3'; with S3 ON the S3-failed ones are
    # 'below_visual_threshold'. Both are eligible for the syllable tier (SILVER
    # already took the S3-confirmed ones), so SYLLABLE coexists with SILVER.
    UNCONF = {"unconfirmed_no_s3", "below_visual_threshold"}
    # Case-insensitive gate (fixes the case-split; +~1,131 labels). The promoted row
    # also stores the canonical lowercase syllable so its target class is not fragmented
    # into cased variants downstream.
    syl_ok = syllable_gate(records, UNCONF)
    n_promoted = 0
    for r in records:
        syl_l = str(r["syllable"]).lower()
        if (r["tier"] == "REVIEW" and r["rule"] in UNCONF
                and (r["ocr_char"], syl_l) in syl_ok):
            r["tier"], r["rule"] = "SYLLABLE", "nghia_consensus"
            r["syllable"] = syl_l
            n_promoted += 1

    # label_level + unicode
    for r in records:
        if r["tier"] in ("GOLD", "SILVER"):
            r["label_level"] = "char"
            lab = r["label"]
            r["unicode"] = f"U+{ord(lab):04X}" if len(lab) == 1 else ""
        elif r["tier"] == "SYLLABLE":
            r["label_level"] = "syllable"   # char unconfirmed; syllable is the target
            r["label"], r["unicode"] = "", ""
        else:
            r["label_level"], r["unicode"] = "", ""

    # ---------- SPLIT: leakage-safe by group=(book,page,column) [#4] ----------
    def split_of(group: str) -> str:
        h = int(hashlib.md5(group.encode()).hexdigest(), 16) % 100
        return "train" if h < 80 else ("val" if h < 90 else "test")
    for r in records:
        r["split_group"] = f"{r['book']}|{r['page']}|c{r['column']}"
    # singleton char classes -> force their WHOLE GROUP into train, so val/test
    # per-class metrics are well-defined AND a column never spans two splits.
    ccnt = Counter(r["label"] for r in records if r["label_level"] == "char" and r["label"])
    singleton_groups = {r["split_group"] for r in records
                        if r["label_level"] == "char" and r["label"] and ccnt[r["label"]] == 1}
    for r in records:
        g = r["split_group"]
        r["split"] = "train" if g in singleton_groups else split_of(g)

    # ---------- PASS 2: materialize crops + quality columns [#3,#5] ----------
    crop_tiers = {"GOLD", "SILVER", "SYLLABLE"} | ({"REVIEW"} if args.crop_review else set())
    by_page = defaultdict(list)
    for r in records:
        by_page[r["page_png"]].append(r)
    labels = []
    for png, recs in by_page.items():
        need = (not args.no_crops) and any(r["tier"] in crop_tiers for r in recs)
        img = cv2.imread(png, cv2.IMREAD_COLOR) if need else None
        for r in recs:
            img_rel = q = None
            if img is not None and r["tier"] in crop_tiers:
                fn = f"{r['book']}_{r['page']}_c{r['column']:02d}_{r['idx']:03d}.png"
                q = save_crop(img, r.get("bbox"), args.pad, out / r["tier"].lower() / fn,
                              tighten=not args.no_tighten)
                if q:
                    img_rel = f"{r['tier'].lower()}/{fn}"
            labels.append({
                "image": img_rel or "", "book": r["book"], "page": r["page"],
                "column": r["column"], "ocr_char": r["ocr_char"], "syllable": r["syllable"],
                "label": r["label"], "unicode": r["unicode"], "label_level": r["label_level"],
                "tier": r["tier"], "rule": r["rule"],
                "ink_pct": q["ink"] if q else "", "crop_w": q["w"] if q else "",
                "crop_h": q["h"] if q else "", "image_md5": q["md5"] if q else "",
                "seg_flag": q["seg"] if q else "",
                "s3_cosine": r.get("s3_cosine", ""),
                "split": r["split"], "split_group": r["split_group"],
                "bbox": json.dumps(r.get("bbox")),
            })

    # ---------- write manifest + summary ----------
    fields = ["image", "book", "page", "column", "ocr_char", "syllable", "label",
              "unicode", "label_level", "tier", "rule", "s3_cosine", "ink_pct",
              "crop_w", "crop_h", "image_md5", "seg_flag", "split", "split_group", "bbox"]
    with open(out / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(labels)

    tiers = Counter(r["tier"] for r in records)
    splits = Counter((r["tier"], r["split"]) for r in records if r["label_level"])
    char_classes = len(set(r["label"] for r in records if r["label_level"] == "char" and r["label"]))
    summary = {
        "pages": pages_done, "total_pairs": len(records),
        "tiers": dict(tiers), "syllable_promoted": n_promoted,
        "char_classes": char_classes,
        "usable_char": tiers["GOLD"] + tiers["SILVER"],
        "usable_total": tiers["GOLD"] + tiers["SILVER"] + tiers["SYLLABLE"],
        "split_counts": {f"{t}/{s}": n for (t, s), n in sorted(splits.items())},
    }
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f" DATASET -> {out}")
    print("=" * 64)
    print(f" pages {pages_done} | pairs {len(records)} | char classes {char_classes}")
    for t in ("GOLD", "SILVER", "SYLLABLE", "REVIEW"):
        print(f"   {t:9s}: {tiers.get(t, 0)}")
    print(f" SYLLABLE promoted from REVIEW: {n_promoted}")
    print(f" USABLE char-level (GOLD+SILVER): {summary['usable_char']}  | "
          f"+syllable: {summary['usable_total']}")
    print(f" manifest: {out}/labels.csv  ({len(fields)} cột)")


if __name__ == "__main__":
    main()
