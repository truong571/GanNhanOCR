"""Option C — 1-vs-1 engine comparison on 200 random crops.

Runs each registered engine over the SAME sample of crops, scores tier-1 hits
(= engine's output is in the dict candidates for the aligned syllable).

The engine with the highest tier-1 rate is the best single replacement for
Kimhannom. If runner-up is close, consider 2-engine vote (test_vote.py).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.test_multi_ocr.engines import ENGINES

OUT = Path(__file__).resolve().parent / "out"
SAMPLE_PATH = OUT / "sample_crops.json"


def score_engine(engine, samples: list[dict]) -> dict:
    t0 = time.time()
    hits = 0
    blank = 0
    details = []
    for s in samples:
        kimhannom_ch = s["kimhannom_ch"]
        try:
            ch, conf = engine.recognize(s["crop_path"], kimhannom_ch=kimhannom_ch)
        except TypeError:
            ch, conf = engine.recognize(s["crop_path"])
        if not ch:
            blank += 1
        is_hit = ch in s["dict_candidates"]
        if is_hit:
            hits += 1
        details.append({
            "crop": Path(s["crop_path"]).name,
            "syllable": s["syllable"],
            "predicted": ch,
            "kimhannom_ch": kimhannom_ch,
            "is_hit": is_hit,
            "conf": round(conf, 3),
        })
    elapsed = time.time() - t0
    return {
        "engine": engine.name,
        "samples": len(samples),
        "hits": hits,
        "rate": round(hits / max(len(samples), 1) * 100, 2),
        "blank": blank,
        "elapsed_s": round(elapsed, 1),
        "details": details,
    }


def main() -> None:
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"Missing {SAMPLE_PATH}. Run sample_crops.py first.")
    samples = json.loads(SAMPLE_PATH.read_text())
    print(f"Loaded {len(samples)} sample crops")
    print()

    # Allow caller to pick a subset via env var
    import os
    only = os.environ.get("ONLY_ENGINES")
    engines_to_run = (
        {k: v for k, v in ENGINES.items() if k in set(only.split(","))}
        if only else ENGINES
    )
    print(f"Running engines: {list(engines_to_run.keys())}\n")

    results = []
    for name, eng in engines_to_run.items():
        print(f"Running {name}...", flush=True)
        r = score_engine(eng, samples)
        print(f"  hits: {r['hits']}/{r['samples']} = {r['rate']}%   "
              f"blank: {r['blank']}   time: {r['elapsed_s']}s")
        results.append(r)
    print()

    # Markdown report
    md = [
        "# Option C — 1-vs-1 engine comparison",
        "",
        f"Sample: {len(samples)} random crops from prepared/<book>/detected/.",
        "Score = tier-1 hits (engine output ∈ dict_candidates for the aligned",
        "VietOCR syllable). Higher = engine recognises the char in a way",
        "consistent with the (correct) Vietnamese reading.",
        "",
        "## Ranking",
        "",
        "| Engine | hits/N | Tier-1 % | Blank | Time |",
        "|--------|-------:|---------:|------:|-----:|",
    ]
    for r in sorted(results, key=lambda x: -x["rate"]):
        md.append(f"| {r['engine']} | {r['hits']}/{r['samples']} | "
                  f"**{r['rate']}%** | {r['blank']} | {r['elapsed_s']}s |")
    md += ["", "## Notes",
           "- Kimhannom baseline = score of `kimhannom-cached` engine.",
           "- An engine winning by ≥5pp is a real improvement worth swapping in.",
           "- Engines scoring close (±2pp) — try 2-engine vote (test_vote.py).",
           ""]

    (OUT / "compare_1v1.md").write_text("\n".join(md), encoding="utf-8")
    full = {"results": results, "samples": len(samples)}
    (OUT / "compare_1v1.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'compare_1v1.md'}")
    print(f"      {OUT / 'compare_1v1.json'}")


if __name__ == "__main__":
    main()
