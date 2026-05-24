"""Ensemble test: full-page Kim ∪ per-col Kim → enriched ocr_char hint.

Strategy
────────
For each char position (page, col, idx):
  • full_ocr = char from full-page Kim cache (current production)
  • per_ocr  = char from per-col Kim crop (evaluation/percol_kim_test/)

Combine:
  • agree         → high-confidence ocr_char (both bản đồng ý)
  • disagree:
      pick whichever maps to QN syllable via qn_to_nom / nom_to_qn dict
      (i.e. the one that is a linguistically valid Nôm for this syllable)
  • neither in dict for this syllable → keep full_ocr (status quo), mark ambiguous

Then re-evaluate Tier 1+2 (dict-based) with the ensemble ocr_char and count:
  • Tier-1 promotions   : pairs that were Tier 2/3 but ensemble OCR turns
                          them into a clean dict match
  • Tier-1 contradictions : pairs that were Tier 1 with full but ensemble
                          OCR shows the previous match was a coincidence
  • OCR agreement rate   : how often the two bản agree
"""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(ROOT))

from core.text.dictionary import (  # noqa: E402
    load_qn_to_nom, build_nom_to_qn,
)
from core.ranking.ranker import tier1_dictionary_lookup  # noqa: E402

BOOK_DIR = ROOT / "prepared" / "SachThanhTruyen2"
PERCOL_DIR = ROOT / "evaluation" / "percol_kim_test" / "col_ocr"
OUT_DIR = ROOT / "evaluation" / "ensemble_kim_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAGES = ["page_0012", "page_0014"]

QN_DICT = ROOT / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"


def percol_chars_sorted(page: str, col: int) -> list[str]:
    """Read per-col Kim API JSON, return list of chars top-to-bottom."""
    f = PERCOL_DIR / page / f"col{col:02d}.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text())
    boxes = data.get("boxes") or []
    items = []
    for b in boxes:
        txt = (b.get("transcription") or "").strip()
        if not txt:
            continue
        pts = b.get("points") or []
        if not pts:
            continue
        y_top = pts[0][1]
        y_bot = pts[2][1] if len(pts) > 2 else pts[0][1]
        valid = [c for c in txt if not c.isspace()]
        if not valid:
            continue
        h = max(1.0, (y_bot - y_top)) / len(valid)
        for i, ch in enumerate(valid):
            items.append((y_top + h * (i + 0.5), ch))
    items.sort()
    return [c for _, c in items]


def align_percol_to_expected(percol: list[str], expected_n: int) -> list[str | None]:
    """Truncate or pad per-col char list to match expected count.

    If per-col has more chars (marker noise at top), drop from head.
    If fewer, pad tail with None.
    """
    if len(percol) >= expected_n:
        # Drop from head — matches the production marker-strip rule
        return percol[len(percol) - expected_n:]
    return percol + [None] * (expected_n - len(percol))


def build_ensemble(labeled_recs, qn_to_nom, nom_to_qn):
    """Walk records, attach (full_ocr, per_ocr, ensemble_ocr, source, tier_new)."""
    out = []
    by_pc: dict[tuple[str, int], list[dict]] = {}
    for r in labeled_recs:
        if r.get("type") != "match":
            continue
        by_pc.setdefault((r["page"], r["column"]), []).append(r)

    for (page, col), recs in by_pc.items():
        # Sort by bbox y_top to mirror reading order
        recs.sort(key=lambda r: (r.get("bbox") or [0, 0, 0, 0])[1])
        per_chars_raw = percol_chars_sorted(page, col)
        per_chars = align_percol_to_expected(per_chars_raw, len(recs))

        for i, r in enumerate(recs):
            full_ocr = r.get("ocr_char")
            per_ocr = per_chars[i] if i < len(per_chars) else None
            syl = r.get("syllable") or ""

            # Try Tier 1 with each
            char_full, matched_full, _ = tier1_dictionary_lookup(
                full_ocr, syl, qn_to_nom, nom_to_qn,
            )
            char_per, matched_per, _ = tier1_dictionary_lookup(
                per_ocr, syl, qn_to_nom, nom_to_qn,
            )

            # Ensemble decision
            if full_ocr and per_ocr and full_ocr == per_ocr:
                source = "agree"
                ens_ocr = full_ocr
            elif matched_full and not matched_per:
                source = "prefer_full_dict"
                ens_ocr = full_ocr
            elif matched_per and not matched_full:
                source = "prefer_per_dict"
                ens_ocr = per_ocr
            elif matched_full and matched_per:
                # Both produce a dict match → prefer agreement on resulting nom_char
                if char_full == char_per:
                    source = "both_match_same_nom"
                    ens_ocr = full_ocr
                else:
                    source = "both_match_diff_nom"  # ambiguous, pick full
                    ens_ocr = full_ocr
            else:
                # Neither matches dict
                if full_ocr and not per_ocr:
                    source, ens_ocr = "only_full", full_ocr
                elif per_ocr and not full_ocr:
                    source, ens_ocr = "only_per", per_ocr
                elif not full_ocr and not per_ocr:
                    source, ens_ocr = "none", None
                else:
                    source, ens_ocr = "disagree_no_dict", full_ocr

            char_ens, matched_ens, _ = tier1_dictionary_lookup(
                ens_ocr, syl, qn_to_nom, nom_to_qn,
            )

            out.append({
                "page": page,
                "column": col,
                "char_idx": i,
                "syllable": syl,
                "full_ocr": full_ocr,
                "per_ocr": per_ocr,
                "ens_ocr": ens_ocr,
                "source": source,
                "prev_nom": r.get("nom_char"),
                "prev_tier": r.get("tier"),
                "prev_matched": r.get("matched"),
                "t1_full": {"nom": char_full, "matched": matched_full},
                "t1_per": {"nom": char_per, "matched": matched_per},
                "t1_ens": {"nom": char_ens, "matched": matched_ens},
            })
    return out


def main():
    print("Loading dicts...")
    qn_to_nom = load_qn_to_nom(str(QN_DICT))
    nom_to_qn = build_nom_to_qn(qn_to_nom)

    ds = json.load(open(BOOK_DIR / "labeled" / "dataset.json"))
    debug_recs = [r for r in ds if r["page"] in PAGES]

    rows = build_ensemble(debug_recs, qn_to_nom, nom_to_qn)

    # ── Metrics ──
    n = len(rows)
    sources = Counter(r["source"] for r in rows)
    agree_total = sum(1 for r in rows
                      if r["full_ocr"] and r["per_ocr"]
                      and r["full_ocr"] == r["per_ocr"])
    paired_total = sum(1 for r in rows if r["full_ocr"] and r["per_ocr"])

    t1_full_matched = sum(1 for r in rows if r["t1_full"]["matched"])
    t1_per_matched = sum(1 for r in rows if r["t1_per"]["matched"])
    t1_ens_matched = sum(1 for r in rows if r["t1_ens"]["matched"])

    # Promotions: prev tier 2/3, ensemble t1 now matches
    promotions = [r for r in rows
                  if r["prev_tier"] in (2, 3) and r["t1_ens"]["matched"]
                  and not r["t1_full"]["matched"]]

    # Per-col-only saves: cases where full alone didn't find dict but per did
    per_only_saves = [r for r in rows
                      if r["t1_per"]["matched"] and not r["t1_full"]["matched"]]
    full_only_saves = [r for r in rows
                       if r["t1_full"]["matched"] and not r["t1_per"]["matched"]]

    # nom result contradictions (both dict-match but yield different nom)
    contradictions = [r for r in rows
                      if r["t1_full"]["matched"] and r["t1_per"]["matched"]
                      and r["t1_full"]["nom"] != r["t1_per"]["nom"]]

    summary = {
        "total_pairs": n,
        "ocr_agreement": {
            "agree": agree_total,
            "paired": paired_total,
            "rate_pct": round(100 * agree_total / max(1, paired_total), 2),
        },
        "source_counts": dict(sources),
        "tier1_dict_matches": {
            "full_only": t1_full_matched,
            "per_only":  t1_per_matched,
            "ensemble":  t1_ens_matched,
        },
        "n_promotions_prev_T2T3_to_T1": len(promotions),
        "n_per_only_saves":  len(per_only_saves),
        "n_full_only_saves": len(full_only_saves),
        "n_t1_nom_contradictions": len(contradictions),
    }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    (OUT_DIR / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )

    # ── Markdown report ──
    md = ["# Ensemble Kim OCR (full-page + per-col) — kết quả",
          "",
          f"Tổng cặp xét: **{n}**",
          "",
          "## 1. Mức đồng ý giữa hai bản đọc",
          "",
          f"- Cặp có cả 2 OCR: **{paired_total}**",
          f"- Đồng ý (cùng ký tự): **{agree_total}**  "
          f"({100*agree_total/max(1,paired_total):.1f}%)",
          f"- Bất đồng: **{paired_total-agree_total}**  "
          f"({100*(paired_total-agree_total)/max(1,paired_total):.1f}%)",
          "",
          "## 2. Tier-1 dict match (sau khi đổi gợi ý OCR)",
          "",
          "| Nguồn ocr_char | Số cặp Tier-1 dict match | Δ so với full |",
          "|---|---:|---:|",
          f"| Full-page (hiện tại) | {t1_full_matched} | — |",
          f"| Per-col only         | {t1_per_matched} | {t1_per_matched-t1_full_matched:+} |",
          f"| **Ensemble**         | **{t1_ens_matched}** | **{t1_ens_matched-t1_full_matched:+}** |",
          "",
          "## 3. Promotion / save count",
          "",
          f"- Promote Tier 2/3 → Tier 1 (nhờ ensemble): **{len(promotions)}**",
          f"- Per-col cứu được (full không tra ra dict): **{len(per_only_saves)}**",
          f"- Full cứu được (per-col không tra ra dict): **{len(full_only_saves)}**",
          f"- Cả 2 cùng tra ra dict nhưng nom khác nhau: **{len(contradictions)}**",
          "",
          "## 4. Nguồn ensemble (source breakdown)",
          "",
          "| source | n |",
          "|---|---:|"]
    for k, v in sources.most_common():
        md.append(f"| {k} | {v} |")
    md += ["",
           "## 5. Promotion list (Tier 2/3 → Tier 1 nhờ ensemble)",
           ""]
    if promotions:
        md += ["| page | col | idx | syl | full_ocr | per_ocr | ens_ocr | source | nom (mới) |",
               "|------|----:|----:|-----|---------|---------|---------|--------|-----------|"]
        for r in promotions[:60]:
            md.append(
                f"| {r['page']} | {r['column']} | {r['char_idx']} | {r['syllable']} "
                f"| {r['full_ocr'] or '·'} | {r['per_ocr'] or '·'} "
                f"| {r['ens_ocr'] or '·'} | {r['source']} "
                f"| {r['t1_ens']['nom']} |"
            )
    else:
        md.append("(Không có)")

    md += ["",
           "## 6. Bất đồng cả 2 cùng dict-match nhưng ra nom khác (cần audit thủ công)",
           ""]
    if contradictions:
        md += ["| page | col | idx | syl | full→ | per→ | prev_nom | prev_tier |",
               "|------|----:|----:|-----|------|------|----------|----------:|"]
        for r in contradictions[:30]:
            md.append(
                f"| {r['page']} | {r['column']} | {r['char_idx']} | {r['syllable']} "
                f"| {r['full_ocr']}→{r['t1_full']['nom']} "
                f"| {r['per_ocr']}→{r['t1_per']['nom']} "
                f"| {r['prev_nom']} | T{r['prev_tier']} |"
            )
    else:
        md.append("(Không có)")

    (OUT_DIR / "report.md").write_text("\n".join(md))

    # ── Self-test ──
    assert t1_ens_matched >= max(t1_full_matched, t1_per_matched) - 5, \
        "Ensemble nên ≥ max(full, per) (cho phép sai số dict tie-break)"

    print("\n=== KẾT QUẢ ===")
    print(f"OCR agreement: {agree_total}/{paired_total} ({100*agree_total/max(1,paired_total):.1f}%)")
    print(f"Tier-1 dict matches:")
    print(f"  full-only :  {t1_full_matched}")
    print(f"  per-only  :  {t1_per_matched}")
    print(f"  ensemble  :  {t1_ens_matched}  (Δ vs full = {t1_ens_matched-t1_full_matched:+})")
    print(f"Promotions T2/3→T1: {len(promotions)}")
    print(f"Per-col saves:      {len(per_only_saves)}")
    print(f"Full saves:         {len(full_only_saves)}")
    print(f"Contradictions:     {len(contradictions)}")
    print(f"\nOutput: {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()
