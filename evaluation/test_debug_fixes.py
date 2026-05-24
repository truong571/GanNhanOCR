"""Post-hoc fixes for the 4 issues found in debug_st2_first2.

Operates ON the labeled dataset (no core/ changes). Re-evaluates 2 pages of
SachThanhTruyen2 with 4 corrections and a stricter Tier-3 threshold.

Fixes
─────
F1 — Loan-word guard
    Catholic transliterations ("Bà Ma ri a" = Maria, "An ti ô ki" = Antiochia,
    "Rô ma" = Roma, "Mi sa" = Misa, "sa se do tê" = sacerdote, "Ina xio" =
    Ignatius, "Giê su" = Jesus, …) are sequences of 1-2 letter syllables that
    have NO canonical Han-Nom mapping. Step 3 currently forces them into the
    QN->Nom dict and picks rubbish. We detect such phrase spans inside each
    column and demote every syllable in the span to tier=0, matched=False.

F2 — Tier-3 confidence threshold raised 0.75 → 0.90
    Records with tier==3 and visual_score < 0.90 are demoted to matched=False.
    This is the user-requested cutoff.

F3 — Strict within-candidates
    When tier==3 and nom_char ∉ nom_candidates, demote: it means the ranker
    leaked outside the QN->Nom dict (rule violation per ranker.py docstring).

F4 — Skip-OCR Tier-1 promotion
    When ocr_char is null AND nom_candidates has exactly 1 entry that matches
    nom_char, promote tier 3 → tier 1 (dict already determines the answer
    uniquely; no visual ranking needed). Only applies when the current label
    is already matched (we don't invent new matches).

Outputs
───────
evaluation/debug_st2_first2/
  ├── fixed_pairs.json     full re-labeled records
  ├── fixes_report.md      side-by-side stats + per-fix change log
  └── fixes_diff.json      list of all records whose label changed
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
BOOK_DIR = ROOT / "prepared" / "SachThanhTruyen2"
OUT_DIR = ROOT / "evaluation" / "debug_st2_first2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAGES = ["page_0012", "page_0014"]

TIER3_MIN_SCORE = 0.90  # F2: raised from 0.75

# F1: known Catholic transliteration phrases observed in this corpus.
# Detection is substring-on-syllable-sequence (case-insensitive, accent-strip).
LOAN_PHRASES = [
    "ba ma ri a",      # Maria
    "ma ri a",
    "gie su",          # Jesus
    "ina xio",         # Ignatius
    "i na xio",
    "ina",
    "an ti o ki a",    # Antiochia
    "an ti o ki",
    "ro ma",           # Roma
    "mi sa",           # Misa
    "sa se do te",     # sacerdote
    "pha ri seu",      # Pharisaeus
    "ki to",           # Kitô (Christ)
    "phe ro",          # Pherô (Peter)
    "phao lo",         # Phaolô (Paul)
]


def strip_accents(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).lower()


def find_loan_spans(syllables: list[str]) -> set[int]:
    """Return indices (within `syllables`) covered by any loan phrase."""
    if not syllables:
        return set()
    flat = [strip_accents(s or "") for s in syllables]
    joined = " ".join(flat)
    # Index → char offset in joined string
    char_to_idx = []
    pos = 0
    for i, tok in enumerate(flat):
        for _ in tok:
            char_to_idx.append(i)
        if i < len(flat) - 1:
            char_to_idx.append(i)  # space belongs to neither, mark either
        pos += len(tok) + 1

    covered: set[int] = set()
    for phrase in LOAN_PHRASES:
        for m in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", joined):
            s, e = m.start(), m.end() - 1
            for k in range(s, e + 1):
                if k < len(char_to_idx):
                    covered.add(char_to_idx[k])
    return covered


def apply_fixes(records: list[dict]) -> tuple[list[dict], dict]:
    """Return (fixed_records, change_log)."""
    by_col: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_col[r.get("column")].append(r)

    fixed: list[dict] = []
    change_log = {"F1_loan": [], "F2_low_conf": [],
                  "F3_out_of_cand": [], "F4_promoted": []}

    for col in sorted(by_col):
        col_recs = by_col[col]
        syls = [r.get("syllable") or "" for r in col_recs]
        loan_idx = find_loan_spans(syls)

        for i, r in enumerate(col_recs):
            new = dict(r)
            tier = r.get("tier", 0)
            vis = r.get("visual_score")
            cands = r.get("nom_candidates") or []
            ocr = r.get("ocr_char")
            nom = r.get("nom_char")
            matched = r.get("matched", False)

            note = []

            # F1 — loan-word guard
            if i in loan_idx and r.get("type") == "match":
                new.update(matched=False, tier=0, nom_char=None,
                           unicode=None, fix_reason="F1_loan")
                change_log["F1_loan"].append({
                    "page": r["page"], "col": col, "syllable": r["syllable"],
                    "was": {"nom": nom, "tier": tier, "matched": matched},
                })
                fixed.append(new)
                continue

            # F4 — promote skip-OCR T1 (run before F2/F3 so promoted rows pass)
            if (tier == 3 and matched and ocr is None
                    and len(cands) == 1 and nom == cands[0]):
                new.update(tier=1, visual_score=None, fix_reason="F4_promoted")
                change_log["F4_promoted"].append({
                    "page": r["page"], "col": col, "syllable": r["syllable"],
                    "nom": nom, "was_vis": vis,
                })
                fixed.append(new)
                continue

            # F3 — strict within-candidates (only meaningful for T2/T3)
            if (tier in (2, 3) and matched and nom and cands
                    and nom not in cands):
                new.update(matched=False, fix_reason="F3_out_of_cand")
                change_log["F3_out_of_cand"].append({
                    "page": r["page"], "col": col, "syllable": r["syllable"],
                    "nom": nom, "tier": tier, "vis": vis,
                    "cands_head": cands[:5],
                })
                fixed.append(new)
                continue

            # F2 — Tier 3 threshold 0.90
            if (tier == 3 and matched and isinstance(vis, (int, float))
                    and vis < TIER3_MIN_SCORE):
                new.update(matched=False, fix_reason="F2_low_conf")
                change_log["F2_low_conf"].append({
                    "page": r["page"], "col": col, "syllable": r["syllable"],
                    "nom": nom, "vis": vis,
                })
                fixed.append(new)
                continue

            fixed.append(new)

    return fixed, change_log


def stats(records: list[dict]) -> dict:
    tiers = Counter(r.get("tier", 0) for r in records)
    matched = sum(1 for r in records if r.get("matched"))
    return {
        "total": len(records),
        "matched": matched,
        "unmatched": sum(1 for r in records
                         if r.get("type") == "match" and not r.get("matched")),
        "rate_pct": round(100 * matched / max(1, len(records)), 2),
        "tier1": tiers.get(1, 0),
        "tier2": tiers.get(2, 0),
        "tier3": tiers.get(3, 0),
        "tier0": tiers.get(0, 0),
    }


def write_report(before: dict, after: dict, change_log: dict):
    md = [
        "# Debug fixes — SachThanhTruyen2 page_0012 + page_0014",
        "",
        f"Tier-3 threshold: **0.75 → {TIER3_MIN_SCORE}**",
        "",
        "## So sánh trước / sau",
        "",
        "| Trang | metric | trước | sau | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for p in PAGES:
        b, a = before[p], after[p]
        for k in ("matched", "unmatched", "rate_pct",
                  "tier1", "tier2", "tier3", "tier0"):
            diff = a[k] - b[k]
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "·")
            sign = f"{diff:+}" if isinstance(diff, int) else f"{diff:+.2f}"
            md.append(f"| {p} | {k} | {b[k]} | {a[k]} | {sign} {arrow} |")
    md += ["", "## Số lượt sửa theo từng fix", "",
           "| Fix | Mô tả | Số record |",
           "|---|---|---:|"]
    desc = {
        "F1_loan": "Skip phiên âm Latin (Maria, Antiochia, Roma, Misa, ...)",
        "F2_low_conf": f"Tier 3 vis < {TIER3_MIN_SCORE} → unmatched",
        "F3_out_of_cand": "nom_char ∉ nom_candidates → unmatched",
        "F4_promoted": "ocr_char=null + 1 candidate → Tier 1",
    }
    for k in ("F1_loan", "F2_low_conf", "F3_out_of_cand", "F4_promoted"):
        md.append(f"| **{k}** | {desc[k]} | {len(change_log[k])} |")

    md += ["", "## Chi tiết từng fix", ""]
    for k in ("F1_loan", "F2_low_conf", "F3_out_of_cand", "F4_promoted"):
        items = change_log[k]
        if not items:
            continue
        md += [f"### {k} ({len(items)})", "",
               "| page | col | syllable | nom (was) | tier (was) | vis | note |",
               "|------|----:|----------|-----------|-----------:|----:|------|"]
        for it in items[:50]:
            page = it.get("page", "")
            col = it.get("col", "")
            syl = it.get("syllable", "")
            if k == "F1_loan":
                w = it.get("was", {})
                md.append(f"| {page} | {col} | {syl} | {w.get('nom')} "
                          f"| T{w.get('tier')} | — | demoted |")
            elif k == "F2_low_conf":
                md.append(f"| {page} | {col} | {syl} | {it.get('nom')} "
                          f"| T3 | {it.get('vis'):.3f} | < {TIER3_MIN_SCORE} |")
            elif k == "F3_out_of_cand":
                v = it.get("vis")
                vs = f"{v:.3f}" if isinstance(v, (int, float)) else "—"
                md.append(f"| {page} | {col} | {syl} | {it.get('nom')} "
                          f"| T{it.get('tier')} | {vs} | "
                          f"cands={it.get('cands_head')} |")
            elif k == "F4_promoted":
                wv = it.get("was_vis")
                wvs = f"{wv:.3f}" if isinstance(wv, (int, float)) else "—"
                md.append(f"| {page} | {col} | {syl} | {it.get('nom')} "
                          f"| T3→T1 | {wvs} | unique cand |")
        if len(items) > 50:
            md.append(f"| ... | | | (+{len(items)-50} more) | | | |")
        md.append("")

    (OUT_DIR / "fixes_report.md").write_text("\n".join(md))


def main():
    ds = json.load(open(BOOK_DIR / "labeled" / "dataset.json"))
    debug_recs = [r for r in ds if r["page"] in PAGES]

    before_per_page = {p: stats([r for r in debug_recs if r["page"] == p])
                       for p in PAGES}
    fixed, change_log = apply_fixes(debug_recs)
    after_per_page = {p: stats([r for r in fixed if r["page"] == p])
                      for p in PAGES}

    write_report(before_per_page, after_per_page, change_log)
    (OUT_DIR / "fixed_pairs.json").write_text(
        json.dumps({"records": fixed}, ensure_ascii=False, indent=2)
    )
    (OUT_DIR / "fixes_diff.json").write_text(
        json.dumps(change_log, ensure_ascii=False, indent=2)
    )

    # ── self-test assertions ──
    fixed_by_id = {(r["page"], r.get("column"), r.get("syllable"),
                    r.get("ocr_char")): r for r in fixed}

    # T1 — no Tier-3 record with vis < 0.90 still matched
    bad_t3 = [r for r in fixed if r.get("tier") == 3 and r.get("matched")
              and isinstance(r.get("visual_score"), (int, float))
              and r["visual_score"] < TIER3_MIN_SCORE]
    assert not bad_t3, f"F2 leak: {len(bad_t3)} T3 < {TIER3_MIN_SCORE} still matched"

    # T2 — no T2/T3 matched record with nom_char not in candidates
    leaks = [r for r in fixed if r.get("tier") in (2, 3) and r.get("matched")
             and r.get("nom_char") and r.get("nom_candidates")
             and r["nom_char"] not in r["nom_candidates"]]
    assert not leaks, f"F3 leak: {len(leaks)} out-of-candidate still matched"

    # T3 — every loan phrase syllable must be demoted (sample check on "Bà Ma ri a")
    p12 = [r for r in fixed if r["page"] == "page_0012" and r.get("column") == 4]
    syls_p12c4 = [r.get("syllable") for r in p12]
    if "Bà" in syls_p12c4 and "Ma" in syls_p12c4 and "ri" in syls_p12c4:
        loans = [r for r in p12 if r.get("syllable") in ("Bà", "Ma", "ri", "a")
                 and r.get("matched")]
        assert not loans, "F1 leak: Maria tokens still matched"

    # T4 — promoted records must have tier=1 and matched=True
    promos = [r for r in fixed if r.get("fix_reason") == "F4_promoted"]
    for r in promos:
        assert r["tier"] == 1 and r["matched"], "F4 inconsistency"

    print(f"✅ All assertions passed")
    print(f"Output dir: {OUT_DIR}")
    print(f"  - fixes_report.md    summary + per-fix changes")
    print(f"  - fixed_pairs.json   {len(fixed)} re-labeled records")
    print(f"  - fixes_diff.json    raw change log")
    print()
    print("Stats (before → after):")
    for p in PAGES:
        b, a = before_per_page[p], after_per_page[p]
        print(f"  {p}: matched {b['matched']}→{a['matched']} "
              f"({b['rate_pct']}% → {a['rate_pct']}%); "
              f"T1 {b['tier1']}→{a['tier1']}, "
              f"T3 {b['tier3']}→{a['tier3']}, "
              f"T0 {b['tier0']}→{a['tier0']}")
    print("Changes by fix:")
    for k, items in change_log.items():
        print(f"  {k}: {len(items)}")


if __name__ == "__main__":
    main()
