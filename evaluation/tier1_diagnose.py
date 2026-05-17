"""Vi sao tier 1 (tu dien) chi gan duoc ~1-2% tren 3 sach Sach?

So sanh: Cac sach (tier 1 = 55-60%) vs Sach sach (tier 1 = 1-2%).
Doan: alignment lech, syllable bi noise, hoac syllable la ten rieng / tu lai.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "dataset" / "all" / "labels.csv"
DICT_QN = REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"
OUT = REPO / "evaluation" / "reports" / "tier1_diagnose.md"


def load_qn_set() -> set[str]:
    """Lay tap am tiet QN co trong tu dien (lowercase, stripped)."""
    df = pd.read_csv(DICT_QN)
    col = None
    for c in df.columns:
        if c.lower().startswith("quoc") or c.lower() == "qn":
            col = c
            break
    if col is None:
        col = df.columns[0]
    return {str(s).strip().lower() for s in df[col].dropna() if str(s).strip()}


def main() -> None:
    df = pd.read_csv(LABELS)
    qn_set = load_qn_set()

    df["syl_norm"] = df["syllable"].astype(str).str.strip().str.lower()
    df["in_dict"] = df["syl_norm"].isin(qn_set)

    rows = []
    for src, g in df.groupby("source"):
        n = len(g)
        tier1 = int((g["tier"] == 1).sum())
        in_dict = int(g["in_dict"].sum())
        empty = int(((g["syl_norm"] == "") | g["syl_norm"].isna()).sum())
        rows.append({
            "source": src,
            "rows": n,
            "tier1_pct": round(tier1 / n * 100, 2),
            "syl_in_dict_pct": round(in_dict / n * 100, 2),
            "syl_empty_pct": round(empty / n * 100, 2),
        })
    summary = pd.DataFrame(rows).set_index("source")

    # Top syllables NOT in dict, per source
    not_in = df[~df["in_dict"]]
    bad_syl_per_src = {}
    for src, g in not_in.groupby("source"):
        top = g["syl_norm"].value_counts().head(15)
        bad_syl_per_src[src] = top

    lines = [
        "# Tier-1 diagnostic",
        "",
        "Cot quan trong:",
        "- `tier1_pct`     = % mau duoc gan bang Tier-1 (tu dien).",
        "- `syl_in_dict_pct` = % syllable XUAT HIEN trong tu dien QN.",
        "- `syl_empty_pct` = % syllable rong / NaN.",
        "",
        "Neu `syl_in_dict_pct` cao ma `tier1_pct` thap -> alignment hong (syllable",
        "khop voi sai ky tu). Neu ca hai deu thap -> syllable bi noise tu Tesseract.",
        "",
        summary.to_markdown(),
        "",
        "## Top 15 syllable KHONG co trong tu dien (per source)",
        "",
    ]
    for src, top in bad_syl_per_src.items():
        lines += [
            f"### {src}",
            "",
            top.rename("count").to_frame().to_markdown(),
            "",
        ]

    lines += [
        "## Doc ket qua",
        "",
        "- Neu top-list co nhieu chuoi 1-2 ky tu (`fa`, `ay`, `ia`, `phai`, `ro`,...)",
        "  -> QN OCR bi Tesseract chia nho sai. Can: re-run buoc QN OCR voi config tot hon,",
        "  hoac dung text PDF embedded thay vi OCR.",
        "- Neu top-list co nhieu ten rieng / dia danh -> mo rong tu dien (add aliases).",
        "- Neu sach Sach co nhieu syllable in_dict NHUNG tier1 thap -> alignment lech,",
        "  xem lai chi phi xoa/them trong Buoc 2.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
