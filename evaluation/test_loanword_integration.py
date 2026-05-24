"""Integration smoke test for F1' loanword guard in pipeline/step3_label.py.

Verifies the patched step3 code:
  1. Demotes a loan syllable that has NO entry in qn_to_nom (e.g. "Ina").
  2. KEEPS loan syllables that DO have an entry (e.g. "Bà" → 婆, "Rô" → 嚕).
  3. Does not affect non-loan syllables.

Uses the existing dict + the loanword module directly (no network, no API,
no GPU). Does not re-run step3 — instead replicates its loanword-decision
logic so we get a deterministic pass/fail without retriggering labeling
on a full book.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(ROOT))

from core.text.dictionary import load_qn_to_nom  # noqa: E402
from core.text.loanword import (  # noqa: E402
    find_loan_spans, should_demote_loan_syllable, LOAN_PHRASES,
)

QN_DICT = ROOT / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"


def main():
    qn_to_nom = load_qn_to_nom(str(QN_DICT))

    # Sanity on the module
    assert "ba ma ri a" in LOAN_PHRASES
    assert "ro ma" in LOAN_PHRASES

    # ── T1: span detection
    syls = ["Đức", "Bà", "Ma", "ri", "a", "cùng", "Giê", "su"]
    span = find_loan_spans(syls)
    assert span == {1, 2, 3, 4, 6, 7}, f"unexpected span: {span}"

    # ── T2: dict-aware demote — Bà/Ma/ri/a/Giê/su all HAVE dict entries
    for s in ("Bà", "Ma", "ri", "a", "Giê", "su"):
        assert not should_demote_loan_syllable(s, qn_to_nom), \
            f"{s!r} should be KEPT (dict has entry)"

    # ── T3: real loan syllable with no dict entry → demote
    # 'Ina' (mid-name of Ignatius) — verified absent in the dict during
    # the debug-fix test; if a future dict adds it, this assertion will
    # flip and we can simply update the fixture.
    assert "ina" not in qn_to_nom and "Ina" not in qn_to_nom, \
        "Test fixture stale: 'Ina' is now in dict; pick another loan-only syllable"
    assert should_demote_loan_syllable("Ina", qn_to_nom), \
        "Ina should be demoted (no dict entry)"

    # ── T4: ordinary syllable outside any span — no demotion
    plain = ["nguyệt", "nhật", "ngày", "tháng"]
    assert find_loan_spans(plain) == set()

    # ── T5: smoke against the patched step3 module — module imports & runs
    import importlib
    mod = importlib.import_module("pipeline.step3_label")
    assert hasattr(mod, "find_loan_spans"), "step3 didn't import loanword helper"
    assert hasattr(mod, "should_demote_loan_syllable"), \
        "step3 didn't import demote helper"

    # ── T6: replay against the 2 debug pages and verify decisions match
    ds_path = ROOT / "prepared" / "SachThanhTruyen2" / "labeled" / "dataset.json"
    if ds_path.exists():
        ds = json.load(open(ds_path))
        recs = [r for r in ds if r["page"] in ("page_0012", "page_0014")]
        # Build per-(page, col) syllable lists in order
        from collections import defaultdict
        by_pc = defaultdict(list)
        for r in recs:
            if r.get("type") == "match":
                by_pc[(r["page"], r["column"])].append(r)
        n_would_demote, n_would_keep = 0, 0
        for (page, col), pairs in by_pc.items():
            syls = [p.get("syllable") or "" for p in pairs]
            span = find_loan_spans(syls)
            for i, p in enumerate(pairs):
                if i not in span:
                    continue
                if should_demote_loan_syllable(p.get("syllable"), qn_to_nom):
                    n_would_demote += 1
                else:
                    n_would_keep += 1
        # Earlier F1' debug test reported: 1 demote, 32 kept
        assert n_would_demote == 1, \
            f"Expected 1 demote on 2 debug pages, got {n_would_demote}"
        assert n_would_keep == 32, \
            f"Expected 32 kept on 2 debug pages, got {n_would_keep}"
        print(f"  2 debug pages: would demote {n_would_demote}, "
              f"keep {n_would_keep}")

    print("✅ All loanword-guard tests pass")


if __name__ == "__main__":
    main()
