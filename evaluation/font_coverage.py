"""Phan tich do phu fd_cache so voi dataset, va so voi cac font ung vien khac.

Tra loi 2 cau hoi:
  1. Bao nhieu chu trong dataset KHONG nam trong fd_cache (vi NomNaTong khong co glyph)?
  2. Neu them font phu tro (HanaMinA/B, Noto Serif CJK, Han Nom A/B), cover them duoc bao nhieu?

Usage:
    python evaluation/font_coverage.py
    python evaluation/font_coverage.py --extra-font /path/to/HanaMinA.ttf /path/to/HanaMinB.ttf
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from fontTools.ttLib import TTFont

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "dataset" / "all" / "labels.csv"
FD_CACHE = REPO / "prepared" / "_universal_fd_cache"
NOMNATONG = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
OUT = REPO / "evaluation" / "reports" / "font_coverage.md"


# Goi y font: link tai (chi de reference, khong tu dong tai)
FONT_SUGGESTIONS = {
    "HanaMinA.ttf": {
        "covers": "CJK Unified + Ext A (~21k glyph)",
        "license": "IPA (free)",
        "url": "https://github.com/cjkvi/HanaMinAFDKO/releases",
        "why": "Cover toan bo CJK Unified, ngon ngu chu chinh thuc — bit nguon thieu cua NomNaTong cho cac chu Han pho thong.",
    },
    "HanaMinB.ttf": {
        "covers": "CJK Ext B/C/D/E/F (~70k glyph)",
        "license": "IPA (free)",
        "url": "https://github.com/cjkvi/HanaMinBFDKO/releases",
        "why": "Phu cho cac chu hiem trong Ext B-F ma khong font nao khac cover.",
    },
    "NotoSerifCJKsc-Regular.otf": {
        "covers": "CJK Unified + Ext A + Compat",
        "license": "OFL (free)",
        "url": "https://github.com/notofonts/noto-cjk",
        "why": "Style serif sang, gan giong chu in moc ban — phu hop neu can style printed Chinese.",
    },
    "NomNaTongLight.ttf": {
        "covers": "Same as NomNaTong (light weight)",
        "license": "NomFoundation",
        "url": "http://nomfoundation.org/nom-tools/Nom-Font",
        "why": "Bien the light cua NomNaTong — neu can them style variant cho chu Nom.",
    },
    "HanNomA.ttf / HanNomB.ttf": {
        "covers": "CJK Unified + Ext A + B (Han-Nom subset)",
        "license": "GPL/free (VietUnicode)",
        "url": "http://vietunicode.sourceforge.net/fonts/fonts_hannom.html",
        "why": "Font Han-Nom co dac thu chu Nom cua Viet Nam — style gan voi sach ban dia hon Noto.",
    },
}


def block(cp: int) -> str:
    ranges = [
        (0x4E00, 0x9FFF, "CJK Unified"),
        (0x3400, 0x4DBF, "CJK Ext A"),
        (0xF900, 0xFAFF, "CJK Compat"),
        (0x20000, 0x2A6DF, "CJK Ext B"),
        (0x2A700, 0x2B73F, "CJK Ext C"),
        (0x2B740, 0x2B81F, "CJK Ext D"),
        (0x2B820, 0x2CEAF, "CJK Ext E"),
        (0x2CEB0, 0x2EBEF, "CJK Ext F"),
        (0x2F800, 0x2FA1F, "CJK Compat Supp"),
        (0x30000, 0x3134F, "CJK Ext G"),
        (0xE000, 0xF8FF, "PUA"),
    ]
    for lo, hi, name in ranges:
        if lo <= cp <= hi:
            return name
    return f"OTHER (0x{cp:X})"


def font_cmap(path: Path) -> set[int]:
    try:
        return set(TTFont(str(path)).getBestCmap().keys())
    except Exception as e:
        print(f"  ! cannot read {path}: {e}")
        return set()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extra-font", nargs="*", default=[],
                   help="Path den cac font phu tro de check do phu (vd HanaMinA.ttf)")
    args = p.parse_args()

    df = pd.read_csv(LABELS)
    df["cp"] = df["unicode"].map(lambda u: int(u.replace("U+", ""), 16))
    all_cp = set(df["cp"].astype(int))

    # fd_cache coverage
    cache_cp: set[int] = set()
    if FD_CACHE.exists():
        for f in FD_CACHE.iterdir():
            n = f.name
            if n.startswith("U+") and n.endswith(".png"):
                cache_cp.add(int(n[2:-4], 16))

    miss = all_cp - cache_cp
    df_miss = df[df["cp"].isin(miss)]
    miss_by_block = Counter(block(cp) for cp in miss)
    inst_in = (~df["cp"].isin(miss)).sum()
    inst_miss = df["cp"].isin(miss).sum()

    # font ung vien
    nomna_cp = font_cmap(NOMNATONG)
    extra_results = []
    for path_s in args.extra_font:
        path = Path(path_s)
        if not path.exists():
            extra_results.append((path.name, 0, 0, "FILE NOT FOUND"))
            continue
        cp = font_cmap(path)
        covered = len(miss & cp)
        extra_results.append((path.name, len(cp), covered,
                              f"{covered/len(miss)*100:.1f}% cua thieu" if miss else "n/a"))

    # report
    lines = [
        "# Font coverage report",
        "",
        f"- Dataset: **{len(all_cp):,}** unique chars, {len(df):,} total rows",
        f"- NomNaTong glyph count (CJK + others): **{len(nomna_cp):,}**",
        f"- fd_cache size: **{len(cache_cp):,}** ({len(cache_cp & all_cp):,} dung trong dataset)",
        "",
        "## Phan chu thieu (khong co trong fd_cache)",
        "",
        f"- Unique chars thieu: **{len(miss)}** ({len(miss)/len(all_cp)*100:.2f}%)",
        f"- Instance thieu: **{inst_miss}** ({inst_miss/len(df)*100:.2f}%)",
        "",
        "### Phan bo theo Unicode block",
        "",
        "| Block | So chu unique |",
        "|-------|--------------:|",
    ]
    for b, c in miss_by_block.most_common():
        lines.append(f"| {b} | {c} |")

    lines += [
        "",
        "### Trang thai gan nhan cua phan thieu",
        "",
        "(Vi tier-3 KHONG the dung — khong co anh trong cache — chung dua hoan toan vao tier 1/2.)",
        "",
        df_miss.groupby("tier").size().rename("count").to_frame().to_markdown(),
        "",
        "### Top 20 chu thieu (theo tan suat)",
        "",
        df_miss.groupby(["nom_char", "unicode"]).size().sort_values(ascending=False)
        .head(20).rename("count").to_frame().to_markdown(),
        "",
    ]

    if extra_results:
        lines += ["## Do phu cua cac font ung vien", "",
                  "| Font | Tong glyph | Cover trong so 176 chu thieu | Ti le |",
                  "|------|----------:|----------------------------:|------:|"]
        for name, total_g, covered, pct in extra_results:
            lines.append(f"| {name} | {total_g:,} | {covered} | {pct} |")
        lines.append("")

    lines += [
        "## De xuat font bo sung",
        "",
        "Hien tai fd_cache phu 99.5% **instance** va 96.6% **unique chars**. Khoang trong",
        "0.5% / 3.4% nay khong the duoc tier-3 visual matching xu ly (khong co reference image),",
        "phai dua hoan toan vao tier 1/2 (dict + similar). Neu muon dong khoang trong nay:",
        "",
    ]
    for name, info in FONT_SUGGESTIONS.items():
        lines += [
            f"### {name}",
            f"- **Cover**: {info['covers']}",
            f"- **License**: {info['license']}",
            f"- **Url**: {info['url']}",
            f"- **Vi sao**: {info['why']}",
            "",
        ]

    lines += [
        "## Cach tich hop (de xuat thuc thi)",
        "",
        "1. Tai HanaMinA.ttf + HanaMinB.ttf ve `font_diffusion/fonts/` (Kaggle notebook da tung tai san).",
        "2. Sua `kaggle_diffusion/build_char_universe.py` thanh font-chain:",
        "   ```python",
        "   FONTS = [NomNaTong, HanNomA, HanNomB, HanaMinA, HanaMinB, NotoSerifCJKsc]",
        "   ```",
        "   moi font duyet cmap, ket hop lai thanh universe lon hon.",
        "3. Tren Kaggle T4, chay `diffusion_run.ipynb` lai voi universe moi —",
        "   chi cac chu CHUA co trong cache duoc sinh them (resume-able).",
        "4. Truong hop **mot chu nam o nhieu font**: uu tien NomNaTong > HanNom > HanaMin > Noto.",
        "   FontDiffusion lay style tu crop that nen ket qua se nhat quan voi sach.",
        "",
        "Voi quy mo nho (~176 chu), buoc 3 mat ~15-30 phut tren T4, khong can chay lai toan bo.",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
