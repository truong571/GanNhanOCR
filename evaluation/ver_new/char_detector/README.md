# `char_detector/` — count-constrained character detector (roadmap #5 / pain point A)

Replace the projection-valley / midpoint re-segmenter (`align_production._reseg_column`,
`core/image/char_segmenter.py`) for **diverged columns** with a learned character
**detector** whose proposals are reconciled with the **known character count `N`**
(= the QN syllable count of that column, from the alignment).

## Honest scope (read this first — measured, not assumed)
`review_breakdown.py --with-gaps` shows the **segmentation-addressable ceiling is
~3,585** pairs (1,749 `diverged_column` rows + 1,836 invisible alignment gaps),
versus **13,889** REVIEW rows that are an S3/coverage problem a detector cannot
touch. **27.5% of columns (1,099/3,998) diverge.** So the win from this detector is
**mostly crop QUALITY on diverged columns**, not a large REVIEW recovery. Build it
to (a) clean the crops in those 1,099 columns and (b) recover the ≤3,585 ceiling —
not as the main lever (that is S3 / pain point B). State this trade-off in the thesis.

## Pieces
| File | Status | Role |
|---|---|---|
| `count_constrained.py` | ✅ done + unit-tested | pure geometry: detector boxes + `N` → EXACTLY `N` boxes (merge closest / split tallest). The novel bit. |
| `bootstrap_boxes.py` | ✅ done + runs | `detect_manifest.json`: 66,630 boxes / 445 pages, from confirmed GOLD/SILVER/SYLLABLE bboxes in **original page coords** (no new annotation). |
| `train_centernet.py` | ⬜ Kaggle (stub below) | train a CenterNet/HRCenterNet on the manifest. |
| inference hook | ⬜ 1 edit | in `align_production._pair_new`, on a diverged column call the detector + `constrain_to_count`. |

## Step 1 — data (done locally)
```bash
.venv/bin/python evaluation/ver_new/char_detector/bootstrap_boxes.py
#   -> detect_manifest.json  (66,630 boxes; --complete-only for the clean 2,348 subset)
```
Boxes are page-space (the frame-offset is already corrected in `align_production._detect`,
so they are NOT the ~1.7-column-shifted OCR boxes). Caveat: columns with an alignment
gap may miss a box → a focal-loss detector (CenterNet) tolerates a few missing positives;
use `--complete-only` for a guaranteed-complete (but small) subset to sanity-check.

## Step 2 — train on Kaggle (P100/T4)
Method: **HRCenterNet** (Tang et al., *IEEE Big Data 2020*, arXiv 2012.05739,
code github.com/Tverous/HRCenterNet) — anchorless centre-point heatmap + size
regression for dense historical CJK (IoU ~0.81 on woodblock MTHv2). Recommended:
1. **Pretrain** on **TKH/MTHv2** (github.com/HCIILAB/TKH_MTH_Datasets_Release —
   char-level boxes; these are Han *print*, a domain shift → fine-tune is mandatory).
2. **Fine-tune** on `detect_manifest.json` (our woodblock Nôm boxes).
3. Single-class detection ("character"); the per-box Unicode labels are kept in the
   manifest for analysis / recognition reuse but the detector needs only boxes.
4. AMP, batch 8, input ~512–768 px (downscale pages), ~30–50 epochs — fits one
   Kaggle session. Pack pages + manifest as a Kaggle Dataset (mirror
   `nom_classifier/pack_for_kaggle.py`). Export `detector.pt`.

`train_centernet.py` is intentionally not committed as a runnable local script (it
needs the GPU + the external pretrain set); adapt the HRCenterNet repo's trainer to
read `detect_manifest.json` (image + boxes). Validate with per-char IoU/F1 vs the
current midpoint segmenter on a held-out page set.

## Step 3 — inference hook (the count constraint is the point)
In `evaluation/ver_new/align_production.py::_pair_new`, a column is "diverged" when
`len(cluster["chars"]) != len(syllables)`. There, instead of `_reseg_column`'s
midpoints, run the detector on the column window and reconcile to `N = len(syllables)`:

```python
from evaluation.ver_new.char_detector.count_constrained import constrain_to_count
# det_boxes = detector(page_img, column_window)  -> [(x1,y1,x2,y2,score), ...]
boxes = constrain_to_count(det_boxes, len(syllables))   # EXACTLY N, top->bottom
```
Then pair box[i] ↔ syllable[i] exactly as today. Gate on alignment confidence
(only fire on anchored/diverged columns) so a wrong `N` never drives a wrong split.
Optionally pass `valley_split=lambda b: <deepest projection valley in b>` to make the
under-segment split ink-aware (reuse `core/image/char_segmenter` valley code).

## Evaluation for the thesis
- **Segmentation table:** per-char IoU/F1 + diverged-column recovery rate, midpoint
  baseline vs detector+count-constraint, on held-out pages; TKH/MTHv2 as external ref.
- **Quality proxy:** S3 self-consistency (does the recovered crop's nearest trained
  class match its label?) before/after — reuse `NomEncoder.predict_topk`.
- Before/after crop panels on the worst diverged columns (qualitative figure).
