"""Test toàn bộ đề xuất Q1 + Fix #1..#5 trên 2 trang debug ST2.

Đề xuất / fix
─────────────
Q1   — Diacritic restore: nếu syllable không có trong QN dict, thử khôi phục
       trong khoảng cách dấu/dấu-mũ ≤ 1, đối chiếu set 19k QN syllables.
Q3   — Skip superscript footnote: lọc bỏ syllable kiểu '⁹', '¹⁰', số đơn lẻ
       không có phụ âm. (Sample này không có — chỉ check, không demote.)
F1'  — Loanword guard NHƯỢC (sửa lại F1 cũ): chỉ demote khi syllable KHÔNG
       có candidate trong qn_to_nom. Khi có (ví dụ "Bà"→婆) → giữ nhãn.
F2'  — Tier-3 threshold theo bucket:
         HAN_BASIC ≥ 0.85   (dict cover tốt, T3 chỉ là phụ trợ)
         NOM_EXT_B+ / PUA ≥ 0.90   (DINOv2 là nguồn chính, cần strict)
         Ext A           ≥ 0.87   (trung gian)
F3   — Hán-shared shortcut: nếu ocr_char ∈ HAN_BASIC VÀ ocr_char ∈ candidates
       → promote thành Tier 1 (bảo vệ Kim OCR đúng khỏi DINOv2 đè).
F4   — Audit cjk_block_score filter: report pool composition.
F5   — Audit candidate pool size: report distribution of len(nom_candidates).

So sánh
───────
  Baseline (production hiện tại)
  vs F1 strict (loại bỏ tất cả phiên âm, kết quả từ test trước)
  vs F1'+F2'+F3 combined (đề xuất mới)
  vs +Q1 (thêm diacritic restore)
"""
import csv
import json
import sys
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(ROOT))

from core.text.dictionary import (  # noqa: E402
    load_qn_to_nom, build_nom_to_qn, cjk_block_score,
)
from core.ranking.ranker import tier1_dictionary_lookup  # noqa: E402

BOOK_DIR = ROOT / "prepared" / "SachThanhTruyen2"
QN_DICT_CSV = ROOT / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"
OUT_DIR = ROOT / "evaluation" / "test_all_fixes"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAGES = ["page_0012", "page_0014"]


# ────────────────────────── Helpers ──────────────────────────
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def classify_codepoint(ch: str) -> str:
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


LOAN_PHRASES = [
    "ba ma ri a", "ma ri a", "gie su", "ina xio", "i na xio", "ina",
    "an ti o ki a", "an ti o ki", "ro ma", "mi sa", "sa se do te",
    "pha ri seu", "ki to", "phe ro", "phao lo",
]


def find_loan_spans(syllables: list[str]) -> set[int]:
    flat = [strip_accents(s or "") for s in syllables]
    joined = " ".join(flat)
    char_to_idx, pos = [], 0
    for i, tok in enumerate(flat):
        for _ in tok:
            char_to_idx.append(i)
        if i < len(flat) - 1:
            char_to_idx.append(i)
        pos += len(tok) + 1
    covered: set[int] = set()
    for phrase in LOAN_PHRASES:
        for m in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", joined):
            for k in range(m.start(), m.end()):
                if k < len(char_to_idx):
                    covered.add(char_to_idx[k])
    return covered


# Threshold theo bucket (F2')
T3_THR = {
    "HAN_BASIC": 0.85,
    "CJK_EXT_A": 0.87,
    "NOM_EXT_B_PLUS": 0.90,
    "NOM_PUA": 0.90,
    "OTHER": 0.90,
    "EMPTY": 0.90,
}


def build_qn_syl_set() -> set[str]:
    """Set of valid Vietnamese syllables from the QN-SinoNom dict."""
    out = set()
    with open(QN_DICT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qn = (row.get("QuocNgu") or "").strip()
            if qn:
                out.add(qn)
                out.add(qn.lower())
    return out


def diacritic_candidates(s: str, qn_set: set[str]) -> str | None:
    """Q1: try variants ≤ 1 diacritic away from s."""
    if not s or s in qn_set or s.lower() in qn_set:
        return None
    base = strip_accents(s)
    cands = [q for q in qn_set if strip_accents(q) == base]
    if not cands:
        return None
    # Pick the candidate closest in case (prefer lowercase if input lowercase)
    cands.sort(key=lambda q: (q.istitle() != s.istitle(), len(q)))
    return cands[0]


def superscript_token(s: str) -> bool:
    """Q3: superscript footnote markers, digit-only artifacts."""
    if not s:
        return False
    return bool(re.fullmatch(r"[¹²³⁰-⁹]+", s)) \
        or bool(re.fullmatch(r"\d+", s))


# ────────────────────────── Variants ──────────────────────────
def baseline(recs):
    return [dict(r) for r in recs]


def apply_f1_strict(recs, fix_reason="F1_strict_demote"):
    """Old F1 (per previous test): demote every syllable in loan span."""
    by_col = defaultdict(list)
    for r in recs:
        by_col[r["column"]].append(r)
    out = []
    for col in sorted(by_col):
        col_recs = by_col[col]
        syls = [r.get("syllable") or "" for r in col_recs]
        loan = find_loan_spans(syls)
        for i, r in enumerate(col_recs):
            nr = dict(r)
            if i in loan and r.get("type") == "match":
                nr.update(matched=False, tier=0, nom_char=None,
                          unicode=None, fix_reason=fix_reason)
            out.append(nr)
    return out


def apply_f1_relaxed(recs, qn_to_nom):
    """F1': only demote loan syllable when dict has NO candidate for it."""
    by_col = defaultdict(list)
    for r in recs:
        by_col[r["column"]].append(r)
    out, log = [], []
    for col in sorted(by_col):
        col_recs = by_col[col]
        syls = [r.get("syllable") or "" for r in col_recs]
        loan = find_loan_spans(syls)
        for i, r in enumerate(col_recs):
            nr = dict(r)
            if i in loan and r.get("type") == "match":
                syl = (r.get("syllable") or "").lower()
                cands = qn_to_nom.get(syl, []) or qn_to_nom.get(
                    r.get("syllable") or "", [])
                if not cands:
                    nr.update(matched=False, tier=0, nom_char=None,
                              unicode=None, fix_reason="F1_relaxed_demote")
                    log.append({"page": r["page"], "col": col,
                                "syl": r["syllable"], "reason": "no_dict"})
                else:
                    nr["fix_reason"] = "F1_relaxed_kept"
                    log.append({"page": r["page"], "col": col,
                                "syl": r["syllable"], "reason": "kept_has_dict",
                                "kept_nom": r.get("nom_char"),
                                "kept_tier": r.get("tier")})
            out.append(nr)
    return out, log


def apply_f2_bucket(recs):
    """Bucket-aware T3 threshold."""
    out, log = [], []
    for r in recs:
        nr = dict(r)
        if r.get("tier") == 3 and r.get("matched"):
            v = r.get("visual_score")
            ch = r.get("nom_char") or ""
            b = classify_codepoint(ch)
            thr = T3_THR.get(b, 0.90)
            if isinstance(v, (int, float)) and v < thr:
                nr.update(matched=False, fix_reason=f"F2_bucket_{b}_lt_{thr}")
                log.append({"page": r["page"], "col": r.get("column"),
                            "syl": r.get("syllable"), "nom": ch,
                            "bucket": b, "vis": v, "thr": thr})
        out.append(nr)
    return out, log


def apply_f3_han_shortcut(recs):
    """Han-shared shortcut: ocr_char ∈ HAN_BASIC ∩ candidates → Tier 1."""
    out, log = [], []
    for r in recs:
        nr = dict(r)
        if (r.get("tier") in (2, 3)
                and r.get("ocr_char")
                and classify_codepoint(r["ocr_char"]) == "HAN_BASIC"
                and r.get("nom_candidates")
                and r["ocr_char"] in r["nom_candidates"]):
            old_nom = r.get("nom_char")
            nr.update(nom_char=r["ocr_char"],
                      unicode=f"U+{ord(r['ocr_char']):04X}",
                      tier=1, matched=True, fix_reason="F3_han_shortcut")
            log.append({"page": r["page"], "col": r.get("column"),
                        "syl": r.get("syllable"),
                        "ocr_now_label": r["ocr_char"],
                        "was_nom": old_nom, "was_tier": r.get("tier")})
        out.append(nr)
    return out, log


def apply_q1_diacritic(recs, qn_set):
    """Q1: restore diacritics on syllable; doesn't change label here,
    but counts how many syllables would have been fixed (Tier 1 retry possible)."""
    out, log = [], []
    for r in recs:
        nr = dict(r)
        syl = r.get("syllable") or ""
        if syl and not superscript_token(syl):
            fixed = diacritic_candidates(syl, qn_set)
            if fixed:
                nr["syllable_fixed"] = fixed
                nr["syllable_original"] = syl
                log.append({"page": r["page"], "col": r.get("column"),
                            "from": syl, "to": fixed})
        out.append(nr)
    return out, log


def apply_q3_skip_footnote(recs):
    out, log = [], []
    for r in recs:
        nr = dict(r)
        syl = r.get("syllable") or ""
        if superscript_token(syl):
            nr.update(matched=False, tier=0, nom_char=None,
                      unicode=None, fix_reason="Q3_footnote")
            log.append({"page": r["page"], "col": r.get("column"), "syl": syl})
        out.append(nr)
    return out, log


# ────────────────────────── Stats ──────────────────────────
def stats(recs):
    tiers = Counter(r.get("tier", 0) for r in recs)
    m = sum(1 for r in recs if r.get("matched"))
    buckets = Counter(classify_codepoint(r.get("nom_char") or "")
                      for r in recs if r.get("nom_char"))
    return {
        "n": len(recs),
        "matched": m,
        "rate_pct": round(100 * m / max(1, len(recs)), 2),
        "T1": tiers.get(1, 0), "T2": tiers.get(2, 0),
        "T3": tiers.get(3, 0), "T0": tiers.get(0, 0),
        "buckets": dict(buckets),
    }


# ────────────────────────── Main ──────────────────────────
def main():
    print("Loading dict + records ...")
    qn_to_nom = load_qn_to_nom(str(QN_DICT_CSV))
    qn_set = build_qn_syl_set()
    ds = json.load(open(BOOK_DIR / "labeled" / "dataset.json"))
    recs0 = [r for r in ds if r["page"] in PAGES]

    # Pool size audit (F5)
    pool_sizes = [len(r.get("nom_candidates") or [])
                  for r in recs0 if r.get("nom_candidates")]
    pool_audit = {
        "n_with_candidates": len(pool_sizes),
        "max": max(pool_sizes or [0]),
        "mean": round(sum(pool_sizes) / max(1, len(pool_sizes)), 2),
        "n_at_cap_20": sum(1 for s in pool_sizes if s >= 20),
        "n_lt_5": sum(1 for s in pool_sizes if s < 5),
    }

    # cjk_block_score filter audit (F4)
    pua_in_pools = 0
    for r in recs0:
        for c in (r.get("nom_candidates") or []):
            if cjk_block_score(c) <= 0.1:
                pua_in_pools += 1
    f4_audit = {"pua_in_pools_total": pua_in_pools}

    # ── Build variants ──
    variants: dict[str, list[dict]] = {"baseline": baseline(recs0)}

    # F1 strict (cũ — reproduce kết quả test trước)
    variants["F1_strict"] = apply_f1_strict(baseline(recs0))

    # F1' relaxed
    v_f1r, log_f1r = apply_f1_relaxed(baseline(recs0), qn_to_nom)
    variants["F1_relaxed"] = v_f1r

    # F2' bucket (chỉ F2)
    v_f2, log_f2 = apply_f2_bucket(baseline(recs0))
    variants["F2_bucket_only"] = v_f2

    # F3 han shortcut (chỉ F3)
    v_f3, log_f3 = apply_f3_han_shortcut(baseline(recs0))
    variants["F3_only"] = v_f3

    # Combined: F1' + F3 (chạy F3 trước để Han shortcut promote, sau đó F1')
    combo = apply_f3_han_shortcut(baseline(recs0))[0]
    combo = apply_f1_relaxed(combo, qn_to_nom)[0]
    combo = apply_f2_bucket(combo)[0]
    variants["F1r+F2+F3"] = combo

    # +Q1 diacritic restore on top
    v_q1, log_q1 = apply_q1_diacritic(baseline(recs0), qn_set)
    combo_q1 = apply_f3_han_shortcut(v_q1)[0]
    combo_q1 = apply_f1_relaxed(combo_q1, qn_to_nom)[0]
    combo_q1 = apply_f2_bucket(combo_q1)[0]
    variants["+Q1 (full)"] = combo_q1

    # ── Compare ──
    rows = {k: stats(v) for k, v in variants.items()}

    # ── Self-tests ──
    # F1' kept at least one record that F1 strict demoted
    n_demoted_strict = sum(1 for r in variants["F1_strict"]
                           if r.get("fix_reason") == "F1_strict_demote")
    n_demoted_relax = sum(1 for r in variants["F1_relaxed"]
                          if r.get("fix_reason") == "F1_relaxed_demote")
    assert n_demoted_relax < n_demoted_strict, \
        f"F1' relaxed phải demote ÍT HƠN F1 strict ({n_demoted_relax} < {n_demoted_strict})"

    # F2 bucket: HAN_BASIC matched with vis < 0.85 must be demoted
    for r in variants["F2_bucket_only"]:
        if (r.get("tier") == 3 and r.get("matched")
                and r.get("nom_char")
                and classify_codepoint(r["nom_char"]) == "HAN_BASIC"):
            v = r.get("visual_score")
            if isinstance(v, (int, float)):
                assert v >= 0.85, f"F2 leak HAN_BASIC vis<{0.85}"

    # F3: any promoted record must have tier=1
    for r in variants["F3_only"]:
        if r.get("fix_reason") == "F3_han_shortcut":
            assert r["tier"] == 1 and r["matched"]

    # ── Save ──
    summary = {
        "variants": rows,
        "f4_audit": f4_audit,
        "f5_pool_audit": pool_audit,
        "change_counts": {
            "F1_strict_demoted": n_demoted_strict,
            "F1_relaxed_demoted": n_demoted_relax,
            "F1_relaxed_kept_due_to_dict":
                sum(1 for r in variants["F1_relaxed"]
                    if r.get("fix_reason") == "F1_relaxed_kept"),
            "F2_bucket_demoted": len(log_f2),
            "F3_promoted": len(log_f3),
            "Q1_diacritic_fixed": len(log_q1),
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    (OUT_DIR / "logs.json").write_text(json.dumps({
        "F1_relaxed": log_f1r, "F2_bucket": log_f2,
        "F3_han_shortcut": log_f3, "Q1_diacritic": log_q1,
    }, ensure_ascii=False, indent=2))

    # ── Report ──
    md = ["# Test toàn bộ đề xuất — ST2 page_0012 + page_0014", "",
          f"**Tổng records**: {len(recs0)}", "",
          "## So sánh các variant",
          "",
          "| Variant | matched | rate | T1 | T2 | T3 | T0 | Han | Ext A | NomB+ | PUA |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in rows.items():
        b = s["buckets"]
        md.append(
            f"| {name} | {s['matched']} | {s['rate_pct']}% | "
            f"{s['T1']} | {s['T2']} | {s['T3']} | {s['T0']} | "
            f"{b.get('HAN_BASIC',0)} | {b.get('CJK_EXT_A',0)} | "
            f"{b.get('NOM_EXT_B_PLUS',0)} | {b.get('NOM_PUA',0)} |"
        )

    md += ["", "## Đếm số lần sửa", "",
           "| Fix | Số record bị tác động |",
           "|---|---:|"]
    for k, v in summary["change_counts"].items():
        md.append(f"| {k} | {v} |")

    md += ["", "## F4 — audit cjk_block_score filter (>0.1) trong nom_candidates", ""]
    md.append(f"- Tổng entry PUA trong các pool nom_candidates: "
              f"**{f4_audit['pua_in_pools_total']}** (nếu >0 → filter `>0.1` "
              f"loại bỏ chúng trước Tier 3 ranking)")

    md += ["", "## F5 — audit pool size nom_candidates", "",
           f"- Records có pool: {pool_audit['n_with_candidates']}",
           f"- Pool size max: {pool_audit['max']}",
           f"- Pool size mean: {pool_audit['mean']}",
           f"- Records đạt cap 20: {pool_audit['n_at_cap_20']} "
           f"(nếu cao → cần tăng cap)",
           f"- Records pool < 5: {pool_audit['n_lt_5']}",
           ""]

    md += ["## F3 promotions (Hán-shared shortcut)", ""]
    if log_f3:
        md += ["| page | col | syl | OCR Han = nhãn mới | was nom | was tier |",
               "|------|----:|-----|----|---------|----------:|"]
        for it in log_f3[:40]:
            md.append(
                f"| {it['page']} | {it['col']} | {it['syl']} | "
                f"{it['ocr_now_label']} | {it['was_nom']} | T{it['was_tier']} |"
            )
    else:
        md.append("(không có)")

    md += ["", "## F1' decisions", ""]
    if log_f1r:
        kept = [x for x in log_f1r if x["reason"] == "kept_has_dict"]
        dropped = [x for x in log_f1r if x["reason"] == "no_dict"]
        md += [f"- **Giữ lại** (dict có entry): {len(kept)}",
               f"- **Demote** (dict không có): {len(dropped)}", ""]
        if kept:
            md += ["### Kept", "",
                   "| page | col | syl | nom |", "|---|---:|---|---|"]
            for x in kept[:20]:
                md.append(f"| {x['page']} | {x['col']} | {x['syl']} "
                          f"| {x.get('kept_nom')} |")

    md += ["", "## Q1 diacritic restore", ""]
    if log_q1:
        md += [f"- Số syllable được khôi phục dấu: **{len(log_q1)}**", "",
               "| page | col | từ | sang |", "|---|---:|---|---|"]
        for x in log_q1[:30]:
            md.append(f"| {x['page']} | {x['col']} | `{x['from']}` "
                      f"| `{x['to']}` |")
    else:
        md.append("(không có syllable nào sai dấu trong 2 trang sample)")

    (OUT_DIR / "report.md").write_text("\n".join(md))

    print("✅ All self-tests passed")
    print(f"\n=== So sánh variant ===")
    print(f"{'variant':<18} {'matched':>7} {'rate':>7} {'T1':>4} {'T3':>4} {'T0':>4} "
          f"{'HAN':>4} {'NomB+':>5} {'PUA':>4}")
    for name, s in rows.items():
        b = s["buckets"]
        print(f"{name:<18} {s['matched']:>7} {s['rate_pct']:>7.2f}% "
              f"{s['T1']:>4} {s['T3']:>4} {s['T0']:>4} "
              f"{b.get('HAN_BASIC',0):>4} {b.get('NOM_EXT_B_PLUS',0):>5} "
              f"{b.get('NOM_PUA',0):>4}")
    print(f"\nChange counts:")
    for k, v in summary["change_counts"].items():
        print(f"  {k}: {v}")
    print(f"\nOutputs: {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()
