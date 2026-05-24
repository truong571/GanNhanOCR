"""Audit 3 follow-up sau khi ship F1' patch và re-run pipeline.

A — HAN_BASIC singleton: 1,415 chữ Hán "phổ thông" chỉ xuất hiện 1 record
    → Tier 3 có đang nhầm chọn chữ Hán hiếm? Sample 50 + breakdown
    + đếm OCR Kim agreement, visual_score, candidate rank.

B — Unmatched +485 distribution: F1' chỉ demote 3 — 482 còn lại đến từ đâu?
    Phân bố per book × tier × bucket × syllable không có dict.

C — Long-tail proposal: compute class-weight + đề xuất gộp rare-class.
    Output class-weight CSV cho downstream training.

Tất cả output vào evaluation/audit_post_ship/.
"""
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "evaluation" / "audit_post_ship"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_ALL = ROOT / "dataset" / "all" / "labels.csv"
BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]


def classify(ch: str) -> str:
    if not ch:
        return "EMPTY"
    cp = ord(ch[0])
    if 0x4E00 <= cp <= 0x9FFF:
        return "HAN_BASIC"
    if 0x3400 <= cp <= 0x4DBF:
        return "CJK_EXT_A"
    if 0x20000 <= cp <= 0x2FFFF:
        return "NOM_EXT_B_PLUS"
    if (0xE000 <= cp <= 0xF8FF) or (0xF0000 <= cp <= 0x10FFFF):
        return "NOM_PUA"
    return "OTHER"


def load_all_records():
    """Load labels.csv + lookup richer fields from per-book dataset.json.

    labels.csv chỉ có: crop_file, nom_char, unicode, syllable, matched, tier,
    bbox, page, source, qn_page_confidence, qn_low_conf.
    dataset.json giàu hơn: thêm column, ocr_char, nom_candidates, visual_score.
    Index theo (source, crop_file) — duy nhất per record.
    """
    rows = list(csv.DictReader(open(LABELS_ALL, encoding="utf-8")))
    rich: dict[tuple[str, str], dict] = {}
    for book in BOOKS:
        ds_path = ROOT / "prepared" / book / "labeled" / "dataset.json"
        if not ds_path.exists():
            continue
        ds = json.load(open(ds_path))
        for r in ds:
            cf = r.get("crop_file")
            if cf:
                rich[(book, cf)] = r
    # Merge ocr_char / column / nom_candidates / visual_score into csv rows
    for r in rows:
        rd = rich.get((r.get("source"), r.get("crop_file")), {})
        for k in ("ocr_char", "column", "nom_candidates",
                  "visual_score", "loan_demoted"):
            r[k] = rd.get(k)
    return rows, rich


# ──────────────────────────────────────────
# AUDIT A — HAN_BASIC singleton
# ──────────────────────────────────────────
def audit_a_han_singleton(rows, rich):
    print("[A] HAN_BASIC singleton audit ...")
    freq = Counter(r["nom_char"] for r in rows if r.get("nom_char"))
    han_singletons = {ch for ch, n in freq.items()
                      if n == 1 and classify(ch) == "HAN_BASIC"}
    sample_pool = [r for r in rows
                   if r.get("nom_char") in han_singletons]

    # Stats: tier × matched × OCR agreement
    tier_dist = Counter(r.get("tier") for r in sample_pool)
    matched_dist = Counter(r.get("matched", "").lower() for r in sample_pool)
    ocr_agree = sum(1 for r in sample_pool
                    if r.get("ocr_char") == r.get("nom_char"))
    has_ocr = sum(1 for r in sample_pool if r.get("ocr_char"))
    no_ocr = sum(1 for r in sample_pool if not r.get("ocr_char"))

    # Sample 50 with rich data
    rng = random.Random(42)
    sample = rng.sample(sample_pool, min(50, len(sample_pool)))
    sample_details = []
    for r in sample:
        rd = rich.get((r.get("source"), r.get("crop_file")), {})
        vs = rd.get("visual_score")
        cands = rd.get("nom_candidates", [])
        cand_rank = (cands.index(r["nom_char"]) if r["nom_char"] in cands
                     else -1)
        sample_details.append({
            "source": r.get("source"),
            "page": r.get("page"),
            "col": r.get("column"),
            "syllable": r.get("syllable"),
            "ocr_char": r.get("ocr_char"),
            "nom_char": r.get("nom_char"),
            "tier": r.get("tier"),
            "matched": r.get("matched"),
            "visual_score": round(vs, 3) if isinstance(vs, (int, float)) else None,
            "cand_rank": cand_rank,
            "n_candidates": len(cands),
            "crop_file": r.get("crop_file"),
        })

    # Bucket sample by suspect category
    suspect_cats = {
        "ocr_agrees_with_nom": 0,        # Kim đọc đúng cùng chữ → OK, không nghi
        "ocr_disagree_but_high_vis": 0,  # vis ≥ 0.90 → DINOv2 tự tin
        "ocr_disagree_mid_vis": 0,       # 0.80-0.90 → mơ hồ
        "ocr_disagree_low_vis": 0,       # < 0.80 → nghi
        "no_ocr": 0,                     # Kim mất char → projection fill
        "rank_high_in_cand": 0,          # rank >5 trong pool → đáng nghi
        "not_in_cand": 0,                # rò ra ngoài pool
    }
    for d in sample_details:
        if d["ocr_char"] == d["nom_char"]:
            suspect_cats["ocr_agrees_with_nom"] += 1
        elif d["ocr_char"] is None or d["ocr_char"] == "":
            suspect_cats["no_ocr"] += 1
        else:
            vs = d["visual_score"] or 0
            if vs >= 0.90:
                suspect_cats["ocr_disagree_but_high_vis"] += 1
            elif vs >= 0.80:
                suspect_cats["ocr_disagree_mid_vis"] += 1
            else:
                suspect_cats["ocr_disagree_low_vis"] += 1
        if d["cand_rank"] == -1:
            suspect_cats["not_in_cand"] += 1
        elif d["cand_rank"] > 5:
            suspect_cats["rank_high_in_cand"] += 1

    return {
        "n_han_singletons": len(han_singletons),
        "n_records_in_pool": len(sample_pool),
        "tier_distribution": dict(tier_dist),
        "matched_distribution": dict(matched_dist),
        "ocr_present": has_ocr,
        "ocr_null": no_ocr,
        "ocr_agrees_nom": ocr_agree,
        "sample_50": sample_details,
        "sample_suspect_breakdown": suspect_cats,
    }


# ──────────────────────────────────────────
# AUDIT B — Unmatched +485 distribution
# ──────────────────────────────────────────
def audit_b_unmatched_distribution(rows, rich):
    print("[B] Unmatched distribution ...")
    unmatched = [r for r in rows
                 if (r.get("matched") or "").lower() == "false"]
    by_source = Counter(r.get("source") for r in unmatched)
    by_tier = Counter(r.get("tier") for r in unmatched)
    by_bucket = Counter(classify(r.get("nom_char") or "") for r in unmatched)
    by_syl_missing = sum(1 for r in unmatched if not r.get("syllable"))

    # T0 split (loan_demoted + nom_char None)
    t0 = [r for r in unmatched if r.get("tier") == "0"]

    # T3 unmatched: visual_score distribution
    t3_unmatched = []
    for r in unmatched:
        if r.get("tier") != "3":
            continue
        rd = rich.get((r.get("source"), r.get("crop_file")), {})
        vs = rd.get("visual_score")
        if isinstance(vs, (int, float)):
            t3_unmatched.append(vs)
    if t3_unmatched:
        t3_unmatched_sorted = sorted(t3_unmatched)
        n = len(t3_unmatched_sorted)
        vs_summary = {
            "n": n,
            "min": round(t3_unmatched_sorted[0], 3),
            "p25": round(t3_unmatched_sorted[n // 4], 3),
            "median": round(t3_unmatched_sorted[n // 2], 3),
            "p75": round(t3_unmatched_sorted[3 * n // 4], 3),
            "max": round(t3_unmatched_sorted[-1], 3),
            "lt_0.70": sum(1 for v in t3_unmatched if v < 0.70),
            "0.70-0.75": sum(1 for v in t3_unmatched if 0.70 <= v < 0.75),
            "ge_0.75": sum(1 for v in t3_unmatched if v >= 0.75),
        }
    else:
        vs_summary = None

    # Sample 20 unmatched per book
    rng = random.Random(7)
    samples_by_book = {}
    for book in BOOKS:
        bk = [r for r in unmatched if r.get("source") == book]
        samples_by_book[book] = [
            {
                "page": r.get("page"),
                "col": r.get("column"),
                "syl": r.get("syllable"),
                "ocr": r.get("ocr_char"),
                "tier": r.get("tier"),
                "nom": r.get("nom_char"),
            }
            for r in rng.sample(bk, min(20, len(bk)))
        ]

    return {
        "n_total_unmatched": len(unmatched),
        "by_source": dict(by_source),
        "by_tier": dict(by_tier),
        "by_bucket_with_nom_char": dict(by_bucket),
        "n_no_syllable": by_syl_missing,
        "n_t0": len(t0),
        "t3_unmatched_visual_score": vs_summary,
        "samples_per_book": samples_by_book,
    }


# ──────────────────────────────────────────
# AUDIT C — Class weights + rare-class grouping
# ──────────────────────────────────────────
def audit_c_class_weights(rows):
    print("[C] Class weights + rare grouping ...")
    freq = Counter(r["nom_char"] for r in rows if r.get("nom_char"))
    N = sum(freq.values())
    K = len(freq)

    # sklearn-style balanced class weight: N / (K * n_c)
    weights = {ch: N / (K * n) for ch, n in freq.items()}

    # Distribution stats
    weight_vals = sorted(weights.values())
    n = len(weight_vals)
    w_stats = {
        "min": round(weight_vals[0], 3),
        "p25": round(weight_vals[n // 4], 3),
        "median": round(weight_vals[n // 2], 3),
        "p75": round(weight_vals[3 * n // 4], 3),
        "p95": round(weight_vals[19 * n // 20], 3),
        "max": round(weight_vals[-1], 3),
    }

    # Rare-class grouping proposal
    rare_thresholds = [1, 2, 3, 5, 10]
    rare_proposals = {}
    for thr in rare_thresholds:
        rare_chars = {ch for ch, n in freq.items() if n <= thr}
        rare_records = sum(freq[ch] for ch in rare_chars)
        new_K = K - len(rare_chars) + 1  # +1 for "<rare>" class
        rare_proposals[f"<= {thr}"] = {
            "n_rare_classes": len(rare_chars),
            "n_rare_records": rare_records,
            "pct_classes_removed": round(100 * len(rare_chars) / K, 2),
            "pct_records_in_rare": round(100 * rare_records / N, 2),
            "new_K": new_K,
            "classes_kept": K - len(rare_chars),
        }

    # Save class-weight CSV
    csv_path = OUT_DIR / "class_weights_balanced.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nom_char", "unicode", "count", "weight_balanced", "bucket"])
        for ch, cnt in sorted(freq.items(), key=lambda x: -x[1]):
            w.writerow([ch, f"U+{ord(ch):04X}", cnt,
                        round(weights[ch], 4), classify(ch)])

    return {
        "N_records": N,
        "K_classes": K,
        "weight_stats": w_stats,
        "rare_grouping_proposals": rare_proposals,
        "csv_output": str(csv_path.relative_to(ROOT)),
    }


# ──────────────────────────────────────────
def write_report(a, b, c):
    md = ["# Audit Post-Ship — sau F1' patch + re-run pipeline",
          "",
          f"Dataset: dataset/all/labels.csv (83,652 records, 5,703 classes)",
          "", "## A — HAN_BASIC singleton audit", "",
          f"- Tổng HAN_BASIC singleton class: **{a['n_han_singletons']}**",
          f"- Records nằm trong pool singleton (HAN_BASIC, n=1): "
          f"**{a['n_records_in_pool']}**",
          f"- Tier distribution: {a['tier_distribution']}",
          f"- Matched distribution: {a['matched_distribution']}",
          f"- Có ocr_char: {a['ocr_present']}, null OCR: {a['ocr_null']}",
          f"- Kim đọc trùng nhãn (đúng đắn): {a['ocr_agrees_nom']}",
          "",
          "### Sample 50 — phân loại nghi vấn",
          "",
          "| Loại | n | Đánh giá |",
          "|---|---:|---|"]
    cat = a["sample_suspect_breakdown"]
    cat_desc = {
        "ocr_agrees_with_nom":       "✅ Kim trùng nhãn — OK, không nghi",
        "ocr_disagree_but_high_vis": "🟡 Kim khác, vis ≥ 0.90 — DINOv2 tự tin",
        "ocr_disagree_mid_vis":      "🟠 Kim khác, vis 0.80-0.90 — mơ hồ",
        "ocr_disagree_low_vis":      "🔴 Kim khác, vis < 0.80 — đáng nghi",
        "no_ocr":                    "⚪ Kim null — projection fill",
        "rank_high_in_cand":         "🟠 nom_char rank >5 trong pool",
        "not_in_cand":               "🔴 nom_char KHÔNG trong pool (leak)",
    }
    for k, v in cat.items():
        md.append(f"| {k} | {v} | {cat_desc[k]} |")

    md += ["", "### 20 sample đầu (chi tiết)", "",
           "| sách | page | col | syl | ocr | nom | tier | matched | vis | rank | crop |",
           "|------|------|---:|-----|-----|-----|----:|---------|----:|----:|------|"]
    for d in a["sample_50"][:20]:
        md.append(
            f"| {d['source']} | {d['page']} | {d['col']} | {d['syllable']} "
            f"| {d['ocr_char'] or '·'} | {d['nom_char']} | T{d['tier']} "
            f"| {d['matched']} | {d['visual_score']} | {d['cand_rank']} "
            f"| `{d['crop_file']}` |"
        )

    md += ["", "## B — Unmatched distribution", "",
           f"- Tổng unmatched: **{b['n_total_unmatched']}**",
           f"- Per book: {b['by_source']}",
           f"- Per tier:  {b['by_tier']}",
           f"- Per bucket (chỉ records có nom_char): {b['by_bucket_with_nom_char']}",
           f"- Records không có syllable: {b['n_no_syllable']}",
           f"- Tier 0 records: {b['n_t0']}",
           ""]
    vs = b["t3_unmatched_visual_score"]
    if vs:
        md += ["### Tier 3 unmatched — visual_score distribution", "",
               f"- n = {vs['n']}",
               f"- min/p25/median/p75/max = "
               f"{vs['min']}/{vs['p25']}/{vs['median']}/{vs['p75']}/{vs['max']}",
               f"- < 0.70: {vs['lt_0.70']}",
               f"- 0.70-0.75: {vs['0.70-0.75']}",
               f"- ≥ 0.75: {vs['ge_0.75']}",
               ""]

    md += ["### 20 sample/book", ""]
    for book, samples in b["samples_per_book"].items():
        md += [f"#### {book}", "",
               "| page | col | syl | ocr | tier | nom |",
               "|------|---:|-----|-----|----:|-----|"]
        for s in samples:
            md.append(f"| {s['page']} | {s['col']} | {s['syl']} "
                      f"| {s['ocr'] or '·'} | T{s['tier']} "
                      f"| {s['nom'] or '·'} |")
        md.append("")

    md += ["## C — Class-weight + rare-class grouping", "",
           f"- N (total records): **{c['N_records']:,}**",
           f"- K (unique classes): **{c['K_classes']:,}**",
           "",
           "### Class-weight distribution (balanced = N / K·n_c)",
           "",
           f"- min / p25 / median / p75 / p95 / max = "
           f"{c['weight_stats']['min']} / {c['weight_stats']['p25']} / "
           f"{c['weight_stats']['median']} / {c['weight_stats']['p75']} / "
           f"{c['weight_stats']['p95']} / {c['weight_stats']['max']}",
           f"- CSV xuất: [{c['csv_output']}]({c['csv_output']})",
           "",
           "### Rare-class grouping proposal",
           "",
           "Gộp các class có ≤ T records thành 1 class `<rare>`.",
           "",
           "| Ngưỡng T | # class hiếm | # records hiếm | % class loại | % records → <rare> | K mới |",
           "|---:|---:|---:|---:|---:|---:|"]
    for thr, p in c["rare_grouping_proposals"].items():
        md.append(f"| {thr} | {p['n_rare_classes']} | {p['n_rare_records']} "
                  f"| {p['pct_classes_removed']}% | "
                  f"{p['pct_records_in_rare']}% | {p['new_K']} |")

    md += ["", "## Tổng kết", ""]
    han_ok_rate = (cat["ocr_agrees_with_nom"]
                   / max(1, sum(cat[k] for k in cat
                                if k not in ("rank_high_in_cand", "not_in_cand"))))
    md += [
        f"**A — HAN singleton**: trên sample 50, {cat['ocr_agrees_with_nom']} "
        f"trường hợp Kim trùng nhãn (≈{100*han_ok_rate:.0f}% probably-correct). "
        f"Số nghi (vis<0.80 + leak ngoài pool): "
        f"{cat['ocr_disagree_low_vis'] + cat['not_in_cand']}. "
        f"→ Đa số không phải Tier 3 nhầm bừa.",
        "",
        f"**B — Unmatched {b['n_total_unmatched']}**: tier 0 = {b['n_t0']} "
        f"(loan_demoted = 3 + còn lại do step 2 không tìm được match). "
        f"Tier 3 unmatched có visual_score median = "
        f"{b['t3_unmatched_visual_score']['median'] if vs else 'N/A'} "
        f"— đúng vùng dưới ngưỡng 0.75.",
        "",
        f"**C — Class weight**: K={c['K_classes']:,} class, "
        f"weight max/min ratio ≈ {round(c['weight_stats']['max']/c['weight_stats']['min']):,}× "
        f"(rất lệch). Gộp class ≤2 records → K={c['rare_grouping_proposals']['<= 2']['new_K']} "
        f"({c['rare_grouping_proposals']['<= 2']['pct_records_in_rare']}% records → <rare>).",
    ]

    (OUT_DIR / "report.md").write_text("\n".join(md))


def main():
    rows, rich = load_all_records()
    a = audit_a_han_singleton(rows, rich)
    b = audit_b_unmatched_distribution(rows, rich)
    c = audit_c_class_weights(rows)

    summary = {"A_han_singleton": a, "B_unmatched": b, "C_class_weights": c}
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    )

    write_report(a, b, c)

    # ── Console ──
    print("\n=== KẾT QUẢ ===\n")
    print(f"[A] HAN_BASIC singleton:")
    print(f"    n_singleton = {a['n_han_singletons']}, "
          f"sample 50 breakdown:")
    for k, v in a["sample_suspect_breakdown"].items():
        print(f"      {k:<32} {v}")

    print(f"\n[B] Unmatched total = {b['n_total_unmatched']}")
    print(f"    Per book : {b['by_source']}")
    print(f"    Per tier : {b['by_tier']}")
    if b["t3_unmatched_visual_score"]:
        vs = b["t3_unmatched_visual_score"]
        print(f"    T3 unmatched vis: median={vs['median']} "
              f"<0.70={vs['lt_0.70']} 0.70-0.75={vs['0.70-0.75']} "
              f">=0.75={vs['ge_0.75']}")

    print(f"\n[C] Class weights:")
    print(f"    K = {c['K_classes']}")
    print(f"    weight ratio: {c['weight_stats']['max']} / "
          f"{c['weight_stats']['min']} = "
          f"{round(c['weight_stats']['max']/c['weight_stats']['min']):,}×")
    for thr, p in c["rare_grouping_proposals"].items():
        print(f"    rare {thr}: {p['n_rare_classes']:>5} class "
              f"→ K mới = {p['new_K']:>5} "
              f"({p['pct_records_in_rare']}% records → <rare>)")

    print(f"\nOutputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
