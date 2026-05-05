"""Aggregate every chu-Nom char that needs FontDiffusion generation.

Scans all books in `config/pipeline.yaml`, runs tier 1 + tier 2 dictionary
checks, and collects the union of unfilled tier-3 candidates per book.

Output (under `colab_diffusion/exports/`):
    chars_<book>.txt          one Unicode char per line, per book
    chars_all.txt             union across books (deduplicated)
    style_<book>.png          a representative crop, used as style image
    summary.json              counts + paths
    MANIFEST.json             everything the Colab notebook needs to read

Run on the Mac side (NOT on Colab):
    python -m colab_diffusion.aggregate_chars

Then upload `colab_diffusion/exports/` to Colab and run
`generate_fd_cache.ipynb`. The notebook produces one fd_cache_<book>.zip
per book — download those back to the Mac and unzip into
`prepared/<book>/fd_cache/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from core.text.dictionary import (
    build_nom_to_qn, cjk_block_score, load_qn_to_nom, load_similarity_dict,
)
from core.ranking.ranker import (
    tier1_dictionary_lookup, tier2_similar_expansion,
)
from pipeline.step0_setup import load_config


def collect_book(
    book_name: str,
    data_dir: Path,
    qn_to_nom: dict,
    nom_to_qn: dict,
    similar_dict: dict,
) -> tuple[set[str], str | None]:
    """Return (chars_needing_tier3, style_image_path) for one book."""
    aligned_dir = data_dir / "aligned"
    aligned_files = sorted(aligned_dir.glob("page_*_aligned.json"))
    if not aligned_files:
        return set(), None

    chars: set[str] = set()
    style_image: str | None = None

    for af in aligned_files:
        with open(af, "r", encoding="utf-8") as f:
            alignment = json.load(f)
        for pair in alignment:
            if pair["type"] != "match":
                continue
            char_info = pair.get("char", {})
            syllable = pair.get("syllable", "")
            ocr_char = char_info.get("ocr_char") if char_info else None

            # Tier 1
            char, matched, s2 = tier1_dictionary_lookup(
                ocr_char, syllable, qn_to_nom, nom_to_qn,
            )
            if matched and char:
                continue

            # Tier 2
            if ocr_char:
                sim_char, _, _ = tier2_similar_expansion(
                    ocr_char, s2, similar_dict,
                )
                if sim_char:
                    continue

            # Needs tier 3 — strict: only s2 (no similar_dict union)
            filtered = [c for c in s2 if cjk_block_score(c) > 0.1]
            if not filtered:
                filtered = list(s2)
            chars.update(filtered[:20])

            # Style image: first existing crop in book
            if style_image is None and char_info:
                cf = char_info.get("crop_file", "")
                if cf:
                    p = data_dir / "detected" / cf
                    if p.exists():
                        style_image = str(p)

    return chars, style_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--out", default="colab_diffusion/exports")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_root = Path(paths["data_dir"])

    qn_to_nom = load_qn_to_nom(paths["qn_to_nom_dict"])
    nom_to_qn = build_nom_to_qn(qn_to_nom)
    similar_dict = load_similarity_dict(paths["similar_dict"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_book: dict[str, dict] = {}
    union: set[str] = set()

    print(f"Scanning {len(config['books'])} books from {args.config}")
    for book in config["books"]:
        name = book["name"]
        data_dir = data_root / name
        if not data_dir.exists():
            print(f"  [skip] {name}: prepared/{name}/ does not exist (run step 1+2 first)")
            continue

        chars, style_image = collect_book(
            name, data_dir, qn_to_nom, nom_to_qn, similar_dict,
        )

        # Write per-book char list
        chars_path = out_dir / f"chars_{name}.txt"
        with open(chars_path, "w", encoding="utf-8") as f:
            for c in sorted(chars):
                f.write(c + "\n")

        # Copy style image (or warn)
        style_dst = None
        if style_image:
            style_dst = out_dir / f"style_{name}.png"
            shutil.copy2(style_image, style_dst)
        else:
            print(f"  [warn] {name}: no style image found", file=sys.stderr)

        per_book[name] = {
            "chars_count": len(chars),
            "chars_file": chars_path.name,
            "style_image": style_dst.name if style_dst else None,
            "expected_output_dir": f"prepared/{name}/fd_cache/",
        }
        union.update(chars)
        print(f"  {name}: {len(chars):4d} chars, style={'OK' if style_image else 'MISSING'}")

    # Write union
    all_path = out_dir / "chars_all.txt"
    with open(all_path, "w", encoding="utf-8") as f:
        for c in sorted(union):
            f.write(c + "\n")

    # Manifest
    manifest = {
        "books": per_book,
        "union_chars": len(union),
        "union_file": all_path.name,
        "fontdiffusion_ckpt_subpath": paths["fontdiffusion_ckpt"],
        "fontdiffusion_phase1_ckpt_subpath": paths["fontdiffusion_phase1_ckpt"],
        "font_path_subpath": paths["font_path"],
    }
    with open(out_dir / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print()
    print(f"=== Aggregation summary ===")
    print(f"  Books processed:     {len(per_book)}")
    print(f"  Union unique chars:  {len(union)}")
    print(f"  Output directory:    {out_dir}/")
    print()
    print(f"Next steps:")
    print(f"  1. Upload {out_dir}/ + font_diffusion/ + dict/ to Colab.")
    print(f"  2. Run colab_diffusion/generate_fd_cache.ipynb on Colab.")
    print(f"  3. Download fd_cache_<book>.zip and unzip into prepared/<book>/fd_cache/.")


if __name__ == "__main__":
    main()
