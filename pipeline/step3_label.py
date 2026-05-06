"""Step 3: 3-tier label assignment — dictionary -> similar -> FontDiffusion+DINOv2.

2-pass approach for FontDiffusion optimization:
  Pass 1: Scan all pairs, run tier 1+2, collect tier 3 candidates
  Pass 2: Batch-generate FontDiffusion images for all candidates at once
  Pass 3: Label tier 3 using pre-generated images

Labels are either:
  matched=True  -> correct (BLACK in visualization)
  matched=False -> incorrect/unconfirmed (RED in visualization)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from core.text.dictionary import (
    load_qn_to_nom, build_nom_to_qn, load_similarity_dict,
)
from core.ranking.ranker import (
    assign_label, get_dinov2_ranker,
    tier1_dictionary_lookup, tier2_similar_expansion,
    tier3_visual_comparison,
)
from core.text.dictionary import cjk_block_score

from pipeline.step0_setup import load_config


def _collect_tier3_candidates(
    aligned_files: list[Path],
    data_dir: Path,
    qn_to_nom: dict,
    nom_to_qn: dict,
    similar_dict: dict,
) -> set[str]:
    """Pass 1: Scan all pairs, run tier 1+2, collect unique chars needing tier 3."""
    candidates = set()

    for aligned_path in aligned_files:
        with open(aligned_path, "r", encoding="utf-8") as f:
            alignment = json.load(f)

        for pair in alignment:
            if pair["type"] != "match":
                continue

            char_info = pair.get("char", {})
            syllable = pair.get("syllable", "")
            ocr_char = char_info.get("ocr_char") if char_info else None

            # Try tier 1
            char, matched, s2 = tier1_dictionary_lookup(
                ocr_char, syllable, qn_to_nom, nom_to_qn,
            )
            if matched and char:
                continue

            # Try tier 2
            if ocr_char:
                sim_char, sim_matched, _ = tier2_similar_expansion(
                    ocr_char, s2, similar_dict,
                )
                if sim_char:
                    continue

            # Needs tier 3 — collect QN->Nom dict candidates only.
            # Tier 3 ranks strictly within s2; similar_dict lookalikes are no
            # longer used at this stage (would risk picking an out-of-dict char).
            filtered = [c for c in s2 if cjk_block_score(c) > 0.1]
            if not filtered:
                filtered = list(s2)
            candidates.update(filtered[:20])

    return candidates


def label_book(config: dict, book_name: str, verbose: bool = True):
    """Run 3-tier labeling for all pages of a book."""
    paths = config["paths"]
    step3_cfg = config.get("step3", {})
    data_dir = Path(paths["data_dir"]) / book_name

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step 3: Label — {book_name}")
        print(f"{'='*60}")
        print("  Loading dictionaries...")

    qn_to_nom = load_qn_to_nom(paths["qn_to_nom_dict"])
    nom_to_qn = build_nom_to_qn(qn_to_nom)
    similar_dict = load_similarity_dict(paths["similar_dict"])

    if verbose:
        print(f"    QN->Nom: {len(qn_to_nom)} entries")
        print(f"    Similar: {len(similar_dict)} entries")

    # Optional: DINOv2 ranker
    dinov2 = None
    if step3_cfg.get("use_dinov2", False):
        font_path = paths.get("font_path")
        dinov2_model = step3_cfg.get("dinov2_model", "dinov2_vitb14_reg")
        dinov2 = get_dinov2_ranker(font_path, model_name=dinov2_model)
        if dinov2 and verbose:
            print("    DINOv2 ranker loaded.")

    font_path = paths.get("font_path")
    if font_path and not Path(font_path).exists():
        font_path = None

    aligned_dir = data_dir / "aligned"
    labeled_dir = data_dir / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    aligned_files = sorted(aligned_dir.glob("page_*_aligned.json"))
    if not aligned_files:
        print(f"[ERROR] No aligned files in {aligned_dir}", file=sys.stderr)
        return

    # ── FontDiffusion cache ──
    # fd_cache = {char: generated_image_path}
    fd_cache: dict[str, str] = {}
    fontdiffusion_ckpt = None
    require_fd = step3_cfg.get("require_fontdiffusion", True)

    if step3_cfg.get("use_fontdiffusion", False):
        ckpt = paths.get("fontdiffusion_ckpt")
        phase1_ckpt = paths.get("fontdiffusion_phase1_ckpt")

        # Load pre-generated caches from disk.
        # 1) Universal cache — produced once on Kaggle for the full NomNaTong
        #    universe (~21k chars), shared by every book.
        # 2) Per-book cache — local override, e.g. when fd_cache_universal is
        #    missing a char or a per-book style is preferred.
        universal_cache = Path(
            paths.get("fd_cache_universal", "prepared/_universal_fd_cache")
        )
        if universal_cache.exists():
            for png in universal_cache.rglob("U+*.png"):
                hex_str = png.stem.replace("U+", "")
                try:
                    char = chr(int(hex_str, 16))
                    fd_cache[char] = str(png)
                except ValueError:
                    pass
            if verbose and fd_cache:
                print(f"    FontDiffusion universal cache: {len(fd_cache)} images")

        cache_base = data_dir / "fd_cache"
        if cache_base.exists():
            local_added = 0
            for png in cache_base.rglob("U+*.png"):
                hex_str = png.stem.replace("U+", "")
                try:
                    char = chr(int(hex_str, 16))
                    fd_cache[char] = str(png)  # per-book overrides universal
                    local_added += 1
                except ValueError:
                    pass
            if verbose and local_added:
                print(f"    FontDiffusion per-book cache: +{local_added} overrides")

        # Determine which chars still need generation
        if verbose:
            print(f"\n  Scanning tier 3 candidates...")
        tier3_chars = _collect_tier3_candidates(
            aligned_files, data_dir, qn_to_nom, nom_to_qn, similar_dict,
        )
        missing_chars = sorted(c for c in tier3_chars if c not in fd_cache)
        if verbose:
            print(f"    Found {len(tier3_chars)} unique chars needing tier 3")
            print(f"    FontDiffusion cache missing: {len(missing_chars)} chars")

        ckpt_ok = ckpt and Path(ckpt).exists()
        if ckpt_ok:
            fontdiffusion_ckpt = ckpt

            if missing_chars:
                # Find a representative crop from this book to use as style image
                style_image = None
                for af in aligned_files:
                    with open(af, encoding="utf-8") as f:
                        alignment = json.load(f)
                    for pair in alignment:
                        if pair["type"] == "match" and pair.get("char"):
                            cf = pair["char"].get("crop_file", "")
                            if cf:
                                p = data_dir / "detected" / cf
                                if p.exists():
                                    style_image = str(p)
                                    break
                    if style_image:
                        break

                if not style_image:
                    msg = (
                        f"[step3] No style image found for {book_name}; cannot "
                        f"generate FontDiffusion cache for {len(missing_chars)} chars."
                    )
                    print(msg, file=sys.stderr)
                else:
                    try:
                        from core.ranking.fontdiffusion_gen import FontDiffusionGenerator

                        generator = FontDiffusionGenerator(
                            ckpt_dir=ckpt,
                            phase1_ckpt_dir=phase1_ckpt,
                            font_path=font_path or "font_diffusion/fonts/NomNaTong-Regular.ttf",
                            cache_dir=str(cache_base),
                        )
                        generated = generator.generate(
                            missing_chars, style_image, style_name=book_name,
                        )
                        fd_cache.update(generated)
                        if verbose:
                            print(f"    FontDiffusion: {len(generated)} images generated")
                    except Exception as e:
                        print(f"[step3] FontDiffusion error for {book_name}: "
                              f"{type(e).__name__}: {e}", file=sys.stderr)
            elif verbose:
                print("    FontDiffusion cache already covers tier 3 candidates")
        elif verbose and missing_chars:
            print(f"    [warn] FontDiffusion checkpoint not found at '{ckpt}'; "
                  f"{len(missing_chars)} chars cannot be generated locally.",
                  file=sys.stderr)

        # Hard-fail ONLY when cache is empty AND require_fontdiffusion=true.
        # When cache covers most chars, missing ones simply fall through to
        # final fallback (tier=1, matched=False) — honest, not silently wrong.
        if require_fd and not fd_cache:
            raise RuntimeError(
                f"[step3] No FontDiffusion cache available for {book_name} and "
                f"generation could not produce any images. Either:\n"
                f"  1. Generate cache via colab_diffusion/ on Colab, OR\n"
                f"  2. Set require_fontdiffusion=false in config (uses font-rendered "
                f"     fallback, lower accuracy).\n"
                f"  Aborting to avoid producing low-quality labels."
            )

    # ── Pass 2: Label all pairs ──
    if verbose:
        print(f"\n  Labeling...")

    all_labels = []
    tier_counts = {1: 0, 2: 0, 3: 0, 0: 0}
    matched_count = 0
    unmatched_count = 0
    gap_count = 0

    for aligned_path in aligned_files:
        page_name = aligned_path.stem.replace("_aligned", "")

        with open(aligned_path, "r", encoding="utf-8") as f:
            alignment = json.load(f)

        page_labels = []
        for pair in alignment:
            if pair["type"] in ("deletion", "insertion"):
                label = {
                    "page": page_name,
                    "column": pair.get("column"),
                    "type": pair["type"],
                    "syllable": pair.get("syllable"),
                    "nom_char": None,
                    "matched": False,
                    "tier": 0,
                }
                gap_count += 1
            else:
                # type == "match"
                char_info = pair.get("char", {})
                syllable = pair.get("syllable", "")
                ocr_char = char_info.get("ocr_char") if char_info else None

                # Resolve crop path for visual ranking.
                # Prefer the RAW crop (cropped directly from the original page
                # render). The cleaned crop has gone through stroke
                # normalisation / morph open / CC noise removal, which can
                # damage thin strokes and shift DINOv2 embeddings — biasing
                # the cosine similarity against fd_cache references that were
                # generated from the same raw style.
                ranking_crop_path = None
                if char_info:
                    crop_file = char_info.get("crop_file", "")
                    if crop_file:
                        p = data_dir / "detected" / crop_file
                        if p.exists():
                            ranking_crop_path = str(p)
                    if not ranking_crop_path:
                        cleaned_file = char_info.get("cleaned_file", "")
                        if cleaned_file:
                            p = data_dir / "detected" / cleaned_file
                            if p.exists():
                                ranking_crop_path = str(p)

                # 3-tier assignment
                result = assign_label(
                    ocr_char=ocr_char,
                    qn_syllable=syllable,
                    crop_path=ranking_crop_path,
                    qn_to_nom=qn_to_nom,
                    nom_to_qn=nom_to_qn,
                    similar_dict=similar_dict,
                    font_path=font_path,
                    dinov2_ranker=dinov2,
                    fontdiffusion_ckpt=fontdiffusion_ckpt,
                    fd_cache=fd_cache,
                    dinov2_threshold=step3_cfg.get("dinov2_threshold", 0.75),
                    classical_threshold=step3_cfg.get("classical_threshold", 0.55),
                    require_fontdiffusion=step3_cfg.get("require_fontdiffusion", True),
                )

                is_matched = bool(result["matched"])

                label = {
                    "page": page_name,
                    "column": pair.get("column"),
                    "type": "match",
                    "syllable": syllable,
                    "nom_char": result["nom_char"],
                    "unicode": f"U+{ord(result['nom_char']):04X}" if result["nom_char"] else None,
                    "matched": is_matched,
                    "tier": result["tier"],
                    "nom_candidates": result.get("nom_candidates", []),
                    "ocr_char": ocr_char,
                    "bbox": char_info.get("bbox") if char_info else None,
                    "crop_file": char_info.get("crop_file") if char_info else None,
                }
                # Diagnostic scores (only present when tier 3 ran or tier 1 sanity demoted)
                if result.get("visual_score") is not None:
                    label["visual_score"] = result["visual_score"]
                if result.get("sanity_score") is not None:
                    label["sanity_score"] = result["sanity_score"]
                tier_counts[result["tier"]] += 1
                if is_matched:
                    matched_count += 1
                else:
                    unmatched_count += 1

            page_labels.append(label)
            all_labels.append(label)

        if verbose:
            page_matched = sum(1 for l in page_labels if l.get("matched"))
            page_total = sum(1 for l in page_labels if l["type"] == "match")
            print(f"  {page_name}: {page_total} labeled, "
                  f"{page_matched} matched (black), "
                  f"{page_total - page_matched} unmatched (red)")

    # Save dataset.json
    with open(labeled_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=2)

    # Save labels.csv
    fieldnames = [
        "page", "column", "syllable", "nom_char", "unicode",
        "matched", "tier", "ocr_char", "bbox", "crop_file",
    ]
    with open(labeled_dir / "labels.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for label in all_labels:
            row = {k: label.get(k) for k in fieldnames}
            if row.get("bbox"):
                row["bbox"] = str(row["bbox"])
            writer.writerow(row)

    # Save summary
    total = len(all_labels)
    summary = {
        "book": book_name,
        "total_labels": total,
        "matched": matched_count,
        "unmatched": unmatched_count,
        "gaps": gap_count,
        "tiers": {str(k): v for k, v in tier_counts.items()},
    }
    with open(labeled_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n  Summary:")
        print(f"    Total labels:  {total}")
        print(f"    Matched (black):  {matched_count}")
        print(f"    Unmatched (red):  {unmatched_count}")
        print(f"    Gaps (skipped):   {gap_count}")
        if matched_count + unmatched_count > 0:
            rate = matched_count / (matched_count + unmatched_count) * 100
            print(f"    Match rate:       {rate:.1f}%")
        print(f"    Tier 1 (dict):    {tier_counts[1]}")
        print(f"    Tier 2 (similar): {tier_counts[2]}")
        print(f"    Tier 3 (visual):  {tier_counts[3]}")
        print(f"    Tier 0 (none):    {tier_counts[0]}")
        print(f"  Output: {labeled_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Step 3: 3-Tier Label Assignment")
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    parser.add_argument("book", type=str, help="Book name")
    args = parser.parse_args()

    config = load_config(args.config)
    label_book(config, args.book)


if __name__ == "__main__":
    main()
