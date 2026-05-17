# Option C — 1-vs-1 engine comparison

Sample: 200 random crops from prepared/<book>/detected/.
Score = tier-1 hits (engine output ∈ dict_candidates for the aligned
VietOCR syllable). Higher = engine recognises the char in a way
consistent with the (correct) Vietnamese reading.

## Ranking

| Engine | hits/N | Tier-1 % | Blank | Time |
|--------|-------:|---------:|------:|-----:|
| kimhannom-cached | 14/200 | **7.0%** | 0 | 0.0s |
| paddleocr-ch | 0/200 | **0.0%** | 97 | 58.3s |

## Notes
- Kimhannom baseline = score of `kimhannom-cached` engine.
- An engine winning by ≥5pp is a real improvement worth swapping in.
- Engines scoring close (±2pp) — try 2-engine vote (test_vote.py).
