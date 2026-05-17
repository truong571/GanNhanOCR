"""Sample 200 random char crops + their metadata for OCR engine comparison.

For each sample we collect:
  - crop_path     : path to the cropped char image
  - kimhannom_ch  : what Kimhannom OCR said this char is
  - syllable      : the QN syllable aligned to this position (post-LLM-fix)
  - dict_candidates: set of Nom chars dict says could match this syllable

This sample becomes the ground-truth-ish benchmark: any engine whose output
hits `dict_candidates` wins a Tier-1 point. Higher = engine recognised the
char in a way consistent with the (correct) Vietnamese reading.

Output: evaluation/test_multi_ocr/out/sample_crops.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
SEED = 42


def load_qn_to_nom() -> dict[str, set[str]]:
    df = pd.read_csv(REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv")
    qn_col = next(c for c in df.columns if "quoc" in c.lower())
    nom_col = next(c for c in df.columns if "nom" in c.lower())
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        qn = str(r[qn_col]).strip().lower()
        nom = str(r[nom_col]).strip()
        if not qn or nom == "nan":
            continue
        out.setdefault(qn, set()).update(ch for ch in nom if not ch.isspace())
    return out


def collect_candidates() -> list[dict]:
    """Walk aligned/ + detected/ to pair RAW Kimhannom OCR output with syllable.

    NB: dataset/all/labels.csv has the POST-VOTE char (tier 1/2/3), which is
    not a fair test target — we'd be measuring against an already-voted label.
    Instead read the raw Kimhannom char from prepared/<book>/aligned/*.json.
    """
    qn_to_nom = load_qn_to_nom()
    samples = []

    for book_dir in sorted((REPO / "prepared").iterdir()):
        if not book_dir.is_dir() or book_dir.name.startswith("_"):
            continue
        aligned_dir = book_dir / "aligned"
        if not aligned_dir.exists():
            continue
        for af in sorted(aligned_dir.glob("page_*_aligned.json")):
            data = json.loads(af.read_text())
            for pair in data:
                if pair.get("type") != "match":
                    continue
                ch = pair.get("char") or {}
                ocr_ch = ch.get("ocr_char", "")
                syllable = str(pair.get("syllable", "")).strip().lower()
                crop_file = ch.get("crop_file", "")
                if not (ocr_ch and syllable and crop_file):
                    continue
                candidates = qn_to_nom.get(syllable, set())
                if not candidates:
                    continue
                crop_path = book_dir / "detected" / crop_file
                if not crop_path.exists():
                    continue
                samples.append({
                    "crop_path": str(crop_path),
                    "source": book_dir.name,
                    "page": af.stem.replace("_aligned", ""),
                    "syllable": syllable,
                    "kimhannom_ch": ocr_ch,
                    "dict_candidates": sorted(candidates),
                    "n_candidates": len(candidates),
                })
    return samples


def main() -> None:
    samples = collect_candidates()
    print(f"Candidates with both crop + dict candidates: {len(samples):,}")

    random.seed(SEED)
    chosen = random.sample(samples, min(SAMPLE_SIZE, len(samples)))
    print(f"Sampled {len(chosen)} crops")

    # Stats on sample
    in_dict = sum(1 for s in chosen if s["kimhannom_ch"] in s["dict_candidates"])
    print(f"  Kimhannom in dict candidates: {in_dict}/{len(chosen)} = "
          f"{in_dict/len(chosen)*100:.1f}% (baseline Tier-1 score)")

    out_path = OUT / "sample_crops.json"
    out_path.write_text(json.dumps(chosen, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("Next: run an engine on these crops and score it the same way.")


if __name__ == "__main__":
    main()
