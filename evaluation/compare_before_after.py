"""So sánh dataset BEFORE (snapshot cũ) vs AFTER (run mới).

BEFORE: chạy với code cũ (PDF text trên Cac, VietOCR greedy trên Sach, không
        beamsearch, không confidence, no expanded saints/toponyms/OCR fixes).
        Snapshot ở evaluation/reports/coverage_BEFORE.md + tier1_diagnose_BEFORE.md.

AFTER:  chạy với code mới (VietOCR 2-pass beam+conf, expanded post-processing,
        chỉ 3 Sach books). Dataset hiện tại ở dataset/all/.

Output: evaluation/reports/compare.md
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "dataset" / "all" / "labels.csv"
DICT_QN = REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"
OUT = REPO / "evaluation" / "reports" / "compare.md"


def load_qn_set() -> set[str]:
    df = pd.read_csv(DICT_QN)
    col = next((c for c in df.columns if "quoc" in c.lower() or c.lower() == "qn"),
               df.columns[0])
    return {str(s).strip().lower() for s in df[col].dropna() if str(s).strip()}


# Hard-coded BEFORE metrics, captured from the old reports + manifests before
# the VietOCR / post-processing / beam-search upgrades. Only the 3 Sach books
# are listed here so AFTER (which is Sach-only) compares like-for-like.
BEFORE_SACH = {
    "SachThanhTruyen2":  {"rows": 17661, "unique_chars": 3013, "matched": 17106,
                          "tier1": 309, "tier2": 2430, "tier3": 14922},
    "SachThanhTruyen4":  {"rows": 17889, "unique_chars": 3193, "matched": 16981,
                          "tier1": 311, "tier2": 2386, "tier3": 15192},
    "SachThanhTruyen11": {"rows": 14725, "unique_chars": 2712, "matched": 14139,
                          "tier1": 114, "tier2": 1925, "tier3": 12686},
}


def main() -> None:
    if not LABELS.exists():
        print(f"!! {LABELS} not found — pipeline hasn't finished yet")
        return

    df = pd.read_csv(LABELS)
    qn = load_qn_set()

    # AFTER per-book
    after = {}
    for src, g in df.groupby("source"):
        after[src] = {
            "rows": int(len(g)),
            "unique_chars": int(g["nom_char"].nunique()),
            "matched": int(g["matched"].sum()),
            "tier1": int((g["tier"] == 1).sum()),
            "tier2": int((g["tier"] == 2).sum()),
            "tier3": int((g["tier"] == 3).sum()),
        }
        if "qn_page_confidence" in g.columns:
            confs = pd.to_numeric(g["qn_page_confidence"], errors="coerce").dropna()
            after[src]["qn_conf_mean"] = float(confs.mean()) if len(confs) else None
            after[src]["qn_low_conf_chars"] = int(
                pd.to_numeric(g.get("qn_low_conf", "").map(lambda v: 1 if str(v).lower() in ("true", "1") else 0),
                              errors="coerce").fillna(0).sum()
            )

    def pct(n, d): return f"{n/d*100:.1f}%" if d else "n/a"

    lines = [
        "# Compare: BEFORE vs AFTER (only Sach books)",
        "",
        "BEFORE  = pre-upgrade dataset (snapshot at start of this session)",
        "AFTER   = pipeline run with VietOCR 2-pass + beam + confidence + expanded post-processing",
        "",
        "## Per-book",
        "",
        "| Book | Metric | BEFORE | AFTER | Δ |",
        "|------|--------|-------:|------:|--:|",
    ]

    metrics = [
        ("rows",         "total rows"),
        ("unique_chars", "unique Nom chars"),
        ("matched",      "matched=True"),
        ("tier1",        "Tier 1 (dict)"),
        ("tier2",        "Tier 2 (similar)"),
        ("tier3",        "Tier 3 (visual)"),
    ]
    for book in sorted(set(BEFORE_SACH) | set(after)):
        b = BEFORE_SACH.get(book, {})
        a = after.get(book, {})
        for key, label in metrics:
            bv = b.get(key, 0)
            av = a.get(key, 0)
            delta = av - bv
            sign = "+" if delta > 0 else ""
            lines.append(f"| {book} | {label} | {bv:,} | {av:,} | {sign}{delta:,} |")
        lines.append(f"| {book} | match rate | {pct(b.get('matched',0), b.get('rows',1))} | "
                     f"{pct(a.get('matched',0), a.get('rows',1))} | — |")
        lines.append(f"| {book} | tier1 % | {pct(b.get('tier1',0), b.get('rows',1))} | "
                     f"{pct(a.get('tier1',0), a.get('rows',1))} | — |")
        if "qn_conf_mean" in a and a["qn_conf_mean"] is not None:
            lines.append(f"| {book} | avg QN conf (AFTER only) | — | "
                         f"{a['qn_conf_mean']:.3f} | — |")

    # Totals
    tot_before = {k: sum(b.get(k, 0) for b in BEFORE_SACH.values()) for k, _ in metrics}
    tot_after  = {k: sum(a.get(k, 0) for a in after.values()) for k, _ in metrics}
    lines += ["", "## Totals (3 Sach books)", "",
              "| Metric | BEFORE | AFTER | Δ |",
              "|--------|-------:|------:|--:|"]
    for key, label in metrics:
        bv, av = tot_before[key], tot_after[key]
        delta = av - bv
        sign = "+" if delta > 0 else ""
        lines.append(f"| {label} | {bv:,} | {av:,} | {sign}{delta:,} |")
    lines.append(f"| match rate | {pct(tot_before['matched'], tot_before['rows'])} | "
                 f"{pct(tot_after['matched'], tot_after['rows'])} | — |")
    lines.append(f"| **tier1 %** | **{pct(tot_before['tier1'], tot_before['rows'])}** | "
                 f"**{pct(tot_after['tier1'], tot_after['rows'])}** | — |")

    # Quality assessment
    lines += [
        "",
        "## Quality interpretation",
        "",
        "- **tier1 %**: % nhãn được xác nhận bằng từ điển song hướng QN↔Nôm.",
        "  Đây là metric quan trọng nhất — tier-1 đáng tin hơn tier-3 (visual).",
        "  Pre-upgrade: ~1-2% (vì syllable rác như 'due', 'nu6c', 'm<;>i' không khớp dict).",
        "  Post-upgrade kỳ vọng: ~40-60% (VietOCR + saint/toponym dict).",
        "",
        "- **matched rate**: tổng cả 3 tier. Trước = 96% nhưng phần lớn tier-3 không tin cậy bằng tier-1.",
        "  Sau khi nâng cấp, ngay cả khi matched rate không tăng,",
        "  composition chuyển từ tier-3 -> tier-1 cũng là cải thiện chất lượng lớn.",
        "",
    ]
    if any("qn_conf_mean" in v for v in after.values()):
        lines += [
            "- **avg QN conf** (chỉ có AFTER): VietOCR per-line confidence.",
            "  Dùng để filter trang chất lượng thấp khi train OCR model downstream.",
            "",
        ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    print("Summary:")
    print(f"  BEFORE total: {tot_before['rows']:,} rows, tier1 = {pct(tot_before['tier1'], tot_before['rows'])}")
    print(f"  AFTER  total: {tot_after['rows']:,} rows, tier1 = {pct(tot_after['tier1'], tot_after['rows'])}")


if __name__ == "__main__":
    main()
