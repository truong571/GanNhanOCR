"""Export list of tier 3 candidate characters for FontDiffusion generation.

Run locally:
  python tools/export_tier3_chars.py config/pipeline.yaml CacThanhTruyen2

Output:
  prepared/CacThanhTruyen2/tier3_chars.txt — one char per line
  prepared/CacThanhTruyen2/tier3_style.png — style reference image

Upload these 2 files + font_diffusion code to Colab/Kaggle,
run generate_fd_cache.ipynb, download fd_cache.zip, extract locally.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.text.dictionary import load_qn_to_nom, build_nom_to_qn, load_similarity_dict, cjk_block_score
from core.ranking.ranker import tier1_dictionary_lookup, tier2_similar_expansion
from pipeline.step0_setup import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("book", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_dir = Path(paths["data_dir"]) / args.book

    qn_to_nom = load_qn_to_nom(paths["qn_to_nom_dict"])
    nom_to_qn = build_nom_to_qn(qn_to_nom)
    similar_dict = load_similarity_dict(paths["similar_dict"])

    aligned_dir = data_dir / "aligned"
    aligned_files = sorted(aligned_dir.glob("page_*_aligned.json"))

    candidates = set()
    for aligned_path in aligned_files:
        with open(aligned_path) as f:
            alignment = json.load(f)
        for pair in alignment:
            if pair["type"] != "match":
                continue
            char_info = pair.get("char", {})
            syllable = pair.get("syllable", "")
            ocr_char = char_info.get("ocr_char") if char_info else None

            char, matched, s2 = tier1_dictionary_lookup(ocr_char, syllable, qn_to_nom, nom_to_qn)
            if matched and char:
                continue
            if ocr_char:
                sim_char, _, _ = tier2_similar_expansion(ocr_char, s2, similar_dict)
                if sim_char:
                    continue

            similar_chars = similar_dict.get(ocr_char, []) if ocr_char else []
            all_cands = list(dict.fromkeys(s2 + similar_chars))
            filtered = [c for c in all_cands if cjk_block_score(c) > 0.1] or all_cands
            candidates.update(filtered[:20])

    # Save chars list
    chars_path = data_dir / "tier3_chars.txt"
    with open(chars_path, "w", encoding="utf-8") as f:
        for c in sorted(candidates):
            f.write(f"{c}\n")

    # Export 1 style image per page (diverse writing styles)
    styles_dir = data_dir / "tier3_styles"
    styles_dir.mkdir(parents=True, exist_ok=True)
    style_count = 0

    for af in aligned_files:
        page_name = af.stem.replace("_aligned", "")
        with open(af) as f:
            alignment = json.load(f)
        for pair in alignment:
            if pair["type"] == "match" and pair.get("char"):
                cf = pair["char"].get("crop_file", "")
                if cf:
                    src = data_dir / "detected" / cf
                    if src.exists():
                        dst = styles_dir / f"{page_name}.png"
                        shutil.copy2(str(src), str(dst))
                        style_count += 1
                        break

    print(f"Exported {len(candidates)} chars -> {chars_path}")
    print(f"Exported {style_count} style images -> {styles_dir}/")
    print(f"\nUpload to Colab/Kaggle:")
    print(f"  1. {chars_path}")
    print(f"  2. {styles_dir}/ (folder with {style_count} style images)")
    print(f"  3. font_diffusion/ folder")
    print(f"  4. Run generate_fd_cache.ipynb")
    print(f"  5. Download fd_cache.zip -> extract to {data_dir}/fd_cache/")


if __name__ == "__main__":
    main()
