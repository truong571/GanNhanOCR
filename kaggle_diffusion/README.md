# kaggle_diffusion/

One-shot generator for the **universal chu-Nom fd_cache** — every CJK glyph in
NomNaTong (~21,837 chars) gets a FontDiffusion handwritten-style image. After
this runs once on Kaggle, no subsequent book ever needs Colab/Kaggle
re-generation; the pipeline pulls images from this cache directly.

## Layout

```
kaggle_diffusion/
├── README.md                         (this file)
├── build_char_universe.py            run on Mac → produces exports/
├── generate_universal_fd_cache.ipynb run on Kaggle (P100 GPU, ~6-8 h)
├── exports/
│   ├── char_universe.txt             21,837 chars, one per line
│   └── char_universe.json            block distribution metadata
└── style_references/
    ├── CacThanhTruyen2.png           ⭐ used as universal style image
    ├── CacThanhTruyen4.png           (alternatives if you want to A/B)
    ├── CacThanhTruyen11.png
    ├── SachThanhTruyen2.png
    ├── SachThanhTruyen4.png
    └── SachThanhTruyen11.png
```

## Workflow

### Pre-flight (Mac, one-time)

1. Re-extract the char universe in case font changes:

   ```sh
   PATH="$PWD/.venv/bin:$PATH" python kaggle_diffusion/build_char_universe.py
   ```

2. Push everything to git:

   ```sh
   git add kaggle_diffusion/ .gitmodules
   git commit -m "kaggle: universal fd_cache generator"
   git push
   ```

3. Create a HuggingFace Hub **dataset** repo (any name). This is where the
   generated PNGs land. It works as both checkpoint storage during generation
   and the permanent cache after.

4. Add `HF_TOKEN` to Kaggle Secrets:
   - Open the notebook on Kaggle
   - Add-ons → Secrets → Add a new secret
   - Name: `HF_TOKEN`, value: your HF Hub token (write access)

### On Kaggle (one-time, ~6-8 h)

1. Open https://www.kaggle.com/code → New Notebook
2. File → Import notebook → upload `generate_universal_fd_cache.ipynb`
3. Settings → Accelerator → **GPU P100** (much faster than T4)
4. Settings → Internet → On
5. Edit cell 2's `HF_REPO` to your repo (e.g. `truongmdn/gannhanocr-universal-fd-cache`)
6. Run all cells

What happens:
- Cell 1: install deps, verify GPU
- Cell 2: log into HF Hub, ensure repo exists
- Cell 3: clone GanNhanOCR + font_diffusion submodule
- Cell 4: download FontDiffusion checkpoints (~300 MB)
- Cell 5: pull any PNGs already on the HF repo (resume state)
- Cell 6: build the to-do list (universe minus already-done)
- Cell 7: load FontDiffusion (~1 min)
- Cell 8: **the long part** — generate in chunks of 500, push each chunk to HF Hub
- Cell 9: verify final coverage

If Kaggle resets the session before completion, just re-open the notebook
and run all cells — cells 5+6 detect the partial state and resume from where
it left off.

### After generation (Mac)

```sh
cd /path/to/GanNhanOCR
huggingface-cli download truongmdn/gannhanocr-universal-fd-cache \
    --repo-type=dataset \
    --local-dir prepared/_universal_fd_cache/

./run_pipeline.sh --step 3
./run_pipeline.sh --step 4
```

Step 3 will use `prepared/_universal_fd_cache/U+XXXX.png` for tier-3 visual
ranking on any book.

### Adding a new book later

```sh
# 1. Drop PDF into Data/
cp /path/to/NewBook.pdf Data/

# 2. Add to config/pipeline.yaml under `books:`
#      - name: NewBook
#        pdf: Data/NewBook.pdf
#        reocr: false

# 3. Run the pipeline end-to-end. No Kaggle/Colab needed.
./run_pipeline.sh --book NewBook
```

The universal cache covers any chu-Nom char in any Nom manuscript.

## Why these numbers

- **21,837** = NomNaTong glyph count ∩ CJK Unified/Compat/Ext A-F. Excludes
  PUA, ASCII, symbols.
- **~1.5 GB final size** = 21,837 PNGs at 256×256 grayscale, ~70 KB each.
- **~6-8 h on P100** = 21,837 chars at ~1 s/char average. The first batch is
  slower (model warm-up), later batches stabilise around 0.7-1 s.

## Choosing a different style

The universal cache is generated using one style image. The default is
`style_references/CacThanhTruyen2.png` (chosen for clean scan quality).

To use a different style, edit `STYLE_BOOK` in cell 8 to one of:
`CacThanhTruyen4`, `CacThanhTruyen11`, `SachThanhTruyen2`,
`SachThanhTruyen4`, `SachThanhTruyen11`.

To produce **multiple style variants** (e.g., a separate cache per book
style), run the notebook multiple times with different `STYLE_BOOK` values
and `HF_REPO` names. The pipeline supports per-book cache fallback to the
universal cache via `prepared/<book>/fd_cache/` overrides.
