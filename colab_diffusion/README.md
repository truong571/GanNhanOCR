# colab_diffusion/

Batch-generate FontDiffusion `fd_cache/` for **all books** in one Colab session.

## Why this exists

Tier 3 of the labeling pipeline ([core/ranking/ranker.py](../core/ranking/ranker.py))
requires FontDiffusion-generated handwritten-style images to compare against
manuscript crops via DINOv2. Without these images, tier 3 has no usable
visual reference (font-rendered fallback was removed because the domain
gap between "clean printed" and "manuscript handwritten" produces
misleading similarity scores).

Generating images locally on a Mac is slow (~3-8s per char on MPS).
Generating them on Colab GPU is ~10× faster and free.

## Workflow

### On Mac (this repo)

1. Run steps 1+2 of the pipeline for every book that needs labeling
   (so `prepared/<book>/aligned/page_*_aligned.json` exists):

   ```sh
   ./run_pipeline.sh --step 1
   ./run_pipeline.sh --step 2
   ```

2. Aggregate the chu-Nom chars that need FontDiffusion images:

   ```sh
   PATH="$PWD/.venv/bin:$PATH" python -m colab_diffusion.aggregate_chars
   ```

   Output written under `colab_diffusion/exports/`:
   - `chars_<book>.txt` — chars to generate, one per line
   - `style_<book>.png` — representative crop, used as style reference
   - `chars_all.txt` — union (for inspection only)
   - `MANIFEST.json` — orchestration metadata

3. Push everything to GitHub (or zip the project) so Colab can pull it.

### On Colab

1. Set runtime to **GPU** (Runtime → Change runtime type → T4 / A100).
2. Open `colab_diffusion/generate_fd_cache.ipynb` directly via:
   ```
   https://colab.research.google.com/github/truong571/GanNhanOCR/blob/main/colab_diffusion/generate_fd_cache.ipynb
   ```
3. Run all cells top to bottom. The notebook is self-contained:
   - Cell 2: clones `https://github.com/truong571/GanNhanOCR.git`
   - Cell 3: verifies FontDiffusion checkpoint files (incl. `unet.safetensors`)
   - Cell 3a (optional): downloads missing `unet.safetensors` from HuggingFace Hub
   - Cell 4: refreshes `exports/` (idempotent)
   - Cell 5: loads FontDiffusion (~1 min)
   - Cell 6: generates ~10,500 images across 6 books (~30-50 min on T4)
   - Cell 7: downloads each `fd_cache_<book>.zip` to your local browser
4. **If `unet.safetensors` is missing on git** (size limits), either:
   - Upload it manually to `/content/GanNhanOCR/font_diffusion/ckpt/DRO-20260227-19P2/checkpoint_step_6000/` via the Colab file panel, or
   - Use cell 3a after pushing it to your HuggingFace Hub repo.

### Back on Mac

Unzip into the per-book `fd_cache/` directories:

```sh
for book in CacThanhTruyen2 CacThanhTruyen4 CacThanhTruyen11 \
            SachThanhTruyen2 SachThanhTruyen4 SachThanhTruyen11; do
  mkdir -p "prepared/$book/fd_cache"
  unzip -o "fd_cache_${book}.zip" -d "prepared/$book/fd_cache/"
done
```

Then run step 3 + 4:

```sh
./run_pipeline.sh --step 3
./run_pipeline.sh --step 4
```

## Layout after aggregation

```
colab_diffusion/
├── README.md                       (this file)
├── aggregate_chars.py              (run on Mac)
├── generate_fd_cache.ipynb         (run on Colab)
└── exports/                        (created by aggregate_chars.py)
    ├── MANIFEST.json
    ├── chars_all.txt
    ├── chars_CacThanhTruyen2.txt
    ├── chars_CacThanhTruyen4.txt
    ├── chars_CacThanhTruyen11.txt
    ├── chars_SachThanhTruyen2.txt
    ├── chars_SachThanhTruyen4.txt
    ├── chars_SachThanhTruyen11.txt
    ├── style_CacThanhTruyen2.png
    └── style_<other_books>.png
```

## Notes

- `aggregate_chars.py` uses the same tier-1+tier-2 logic as
  [pipeline/step3_label.py](../pipeline/step3_label.py) so the chars
  collected match exactly what step 3 will need.
- The list is **strict tier-3 candidates only** (s2 from QN→Nom dict);
  similar_dict expansion is not used — see Issue H fix in the codebase
  history for the rationale.
- Style image is the first existing crop found per book; one image per
  book is enough for FontDiffusion to capture the calligraphic style.
- `fd_cache/U+XXXX.png` filename convention is what step 3 expects.
