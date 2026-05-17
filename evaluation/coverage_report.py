"""Tong ket dataset cuoi: matched-rate, tier distribution, long-tail per book.

    python evaluation/coverage_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "dataset" / "all" / "labels.csv"
OUT = REPO / "evaluation" / "reports" / "coverage.md"


def main() -> None:
    df = pd.read_csv(LABELS)
    total = len(df)
    matched = int(df["matched"].sum())
    unmatched = total - matched

    by_src = df.groupby("source").agg(
        rows=("nom_char", "size"),
        unique_chars=("nom_char", "nunique"),
        matched=("matched", "sum"),
    )
    by_src["match_rate"] = (by_src["matched"] / by_src["rows"] * 100).round(2)

    tier = df.groupby(["source", "tier"]).size().unstack(fill_value=0)

    # long-tail
    class_count = df.groupby("nom_char").size()
    tail = {
        "classes_total": int(class_count.size),
        "classes_1_sample": int((class_count == 1).sum()),
        "classes_lt_5": int((class_count < 5).sum()),
        "classes_ge_10": int((class_count >= 10).sum()),
        "classes_ge_50": int((class_count >= 50).sum()),
    }

    lines = [
        "# Coverage report",
        "",
        f"- Total samples: **{total:,}**",
        f"- Matched: **{matched:,}** ({matched/total*100:.2f}%)",
        f"- Unmatched: **{unmatched:,}** ({unmatched/total*100:.2f}%)",
        f"- Unique nom_char: **{class_count.size:,}**",
        "",
        "## Per-book breakdown",
        "",
        by_src.to_markdown(),
        "",
        "## Tier distribution per book",
        "",
        "Tier 1 = dict, Tier 2 = similar, Tier 3 = DINOv2+FD, 0 = none.",
        "",
        tier.to_markdown(),
        "",
        "## Long tail (class -> sample count)",
        "",
        f"- Total classes: {tail['classes_total']:,}",
        f"- Classes with 1 sample only: {tail['classes_1_sample']:,} "
        f"({tail['classes_1_sample']/tail['classes_total']*100:.1f}%)",
        f"- Classes with <5 samples: {tail['classes_lt_5']:,}",
        f"- Classes with >=10 samples (train-ready): {tail['classes_ge_10']:,}",
        f"- Classes with >=50 samples: {tail['classes_ge_50']:,}",
        "",
        "## Health flags",
        "",
    ]
    for src, row in by_src.iterrows():
        t1 = int(tier.loc[src, 1]) if 1 in tier.columns else 0
        t3 = int(tier.loc[src, 3]) if 3 in tier.columns else 0
        if row["rows"] >= 1000 and t1 / row["rows"] < 0.05:
            lines.append(
                f"- **{src}**: tier-1 only {t1/row['rows']*100:.1f}% "
                f"-> dictionary lookup is failing, tier-3 ({t3:,}) is masking it."
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
