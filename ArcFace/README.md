# ArcFace — retrain the Nôm S3 encoder (Kaggle, self-contained)

Rebuilds the S3 visual signal to fix everything the audit measured, and bundles
all the data so the folder zips straight into a Kaggle dataset. Output is a
`best.pt` that is a **drop-in** for `pipeline/align_engine/nom_classifier/infer.py`.

## Why (what was wrong with the old S3)
| # | Problem measured | Fixed by |
|---|---|---|
| **P0** | scoring bug: `max over tiers of P` let a noisy crop-proto (P≤0.98) beat a strong glyph (P≤0.49) → **top-1 60.2% vs head-logit 78.1%, −18 pts** | `inference_s3_v2.py` (score by head-logit) + a strong exported head |
| **P1** | great ranker (retr@1 0.89) but near-random **error gate (AUC 0.57)** | `inference_s3_v2.py` Energy/MLS open-set + **conformal** FAR guarantee; `evaluate.py` proxy error-AUC |
| **P2** | **circular** training on own auto-labels + **86% page leak** in the split | `dataset.assign_splits` (page-disjoint / LOBO), tier-weighted loss (GOLD>SILVER), sub-center ArcFace, SAM/SWA, hard-neg batches |
| **P3** | long tail (522 singleton classes), empty simfont tier | class-balanced sampler, FD-glyph domain anchor, sub-centers |

## Files
```
prepare_data.py      bundle dataset_out crops + FD glyphs + similarity → data/
model.py             NomEmbedder (ckpt-compatible) + SubCenterArcMargin
dataset.py           page-disjoint/LOBO split, class-balanced + ConfusionBatchSampler
sam.py               SAM flat-minima optimizer (FMFP)
train.py             training loop (sub-center ArcFace, tier weights, SAM, SWA)
export_checkpoint.py collapse K sub-centers → (n_cls,embed) best.pt  (infer.py format)
evaluate.py          retr@1/@5 + proxy error-AUC (MLS/Energy) on page-disjoint test
kaggle_run.py        one-cell: train → export → evaluate
inference_s3_v2.py   the P0/P1 SCORING fix to paste into visual_signal.decide()
```

## Resume-safe (survives Kaggle session reset) — hub.py
Kaggle GPU dies at 9h / on interrupt. With `--hf-repo` set, train.py pushes a
**full-state `last.pt`** (model + head + optimizer + scheduler + epoch) to a private
HF repo every epoch, and on restart **pulls it and continues from the saved epoch**
— never from epoch 1. Without `--hf-repo`/token it just checkpoints locally
(`checkpoints/last.pt`), which "Save Version" persists. Resume is **ON by default**
(`--no-resume` to force fresh). Token: env `HF_TOKEN`, `--hf-token`, or a Kaggle
Secret named `HF_TOKEN`. Final `best.pt` is pushed too, so deploy pulls from HF.

## Run
**1. Bundle data (local, once):**
```bash
python ArcFace/prepare_data.py           # GOLD+SILVER crops + 1 FD glyph/class + similar_map
#   --tiers GOLD   (max de-circular)      --link  (fast if same disk)
```
Fills `ArcFace/data/{crops,glyphs,manifest.csv,similar_map.json}` (~0.5 GB).

**2. Upload:** zip `ArcFace/data/` → Kaggle *New Dataset*. Attach it + the `ArcFace/*.py`
code to a GPU notebook.

**3. Train (one cell), reset-safe:**
```bash
!python ArcFace/kaggle_run.py --hf-repo <user>/nom-embed-arcface   # SAM + page-disjoint, K=3, 30ep
# after a reset, run the SAME cell -> it resumes from HF and finishes the remaining epochs
!python ArcFace/kaggle_run.py --hf-repo <user>/nom-embed-arcface --split lobo --holdout stt4
```
Prints retrieval@1/@5 + proxy error-AUC; writes `checkpoints/best.pt` (+ pushes to HF).

**4. Deploy back:** `best.pt` → `nom-embed/best.pt`; route `VisualS3.decide` through
`inference_s3_v2.score_v2` (P0/P1); calibrate `conformal_tau` with `calibrate_conformal`
once GĐ0 human verdicts exist; then `bash run_pipeline.sh --from build --until publish`.

## Notes
- **No human labels yet** → auto-labels are still the supervision; page-disjoint split +
  tier down-weighting + sub-centers **reduce** (not eliminate) circularity. Honest
  error-AUC still needs GĐ0 verdicts — `evaluate.py` prints a proxy (upper bound).
- Determinism: seed 42; same split seed in train + evaluate → test pages never trained on.
- Exported head = **dominant (clean) sub-center per class**, so `infer.py` is unchanged.
- Kaggle: enable **Internet** (resnet18 pretrained + HF), add GPU T4x2, add the `HF_TOKEN` Secret.
