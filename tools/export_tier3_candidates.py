"""Export tier 3 candidate characters for FontDiffusion generation on Colab.

Usage:
    python tools/export_tier3_candidates.py config/pipeline.yaml CacThanhTruyen2

Output: prepared/{book}/tier3_candidates.txt (one Unicode char per line)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.step0_setup import load_config
from core.text.dictionary import load_qn_to_nom, build_nom_to_qn, load_similarity_dict, cjk_block_score
from core.ranking.ranker import tier1_dictionary_lookup, tier2_similar_expansion


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
    candidates = set()

    for af in sorted(aligned_dir.glob("page_*_aligned.json")):
        with open(af) as f:
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
            filtered = [c for c in all_cands if cjk_block_score(c) > 0.1]
            if not filtered:
                filtered = all_cands
            candidates.update(filtered[:20])

    out_path = data_dir / "tier3_candidates.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for char in sorted(candidates):
            f.write(f"{char}\tU+{ord(char):04X}\n")

    print(f"Exported {len(candidates)} candidates to {out_path}")


if __name__ == "__main__":
    main()
