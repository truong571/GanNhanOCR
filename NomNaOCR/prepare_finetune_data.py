"""Prepare NomNaOCR fine-tune data from the project's consensus labels.

Builds COLUMN-CHUNK images + sequence labels (the format NomNaOCR trains on) out of
the GOLD/SILVER positions in dataset_out/labels.csv, cropped from the page scans.

CONTIGUITY (critical): the CRNN reads a whole strip, so a crop must contain EXACTLY
the labelled chars — no unlabelled REVIEW glyph in between. We therefore anchor to
the kim OCR cache (the COMPLETE ordered column) and map each labelled position to
its true column index by matching `ocr_char` (== the cache char). A "run" = a maximal
set of CONSECUTIVE cache indices that are all labelled; runs are then cut into
≤--maxlen chunks (the CRNN's 27 CTC timesteps reliably decode ~≤10 chars).

Output (matches CRNNxCTC_finetune.ipynb):
  finetune_data/Datasets/Patches/<book>/<name>.jpg
  finetune_data/Datasets/Patches/{All,Validate,Test}.txt   # "<book>/<name>.jpg\t<text>"

Run (main venv — only needs PIL):
  .venv/bin/python NomNaOCR/prepare_finetune_data.py --tiers GOLD,SILVER --maxlen 10
"""
from __future__ import annotations
import argparse, ast, csv, json
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
LABELS = REPO / "dataset_out" / "labels.csv"


def book_to_dir(book: str) -> str:
    return f"SachThanhTruyen{''.join(ch for ch in book if ch.isdigit())}"


def greedy_subseq(needle_chars, hay_chars):
    """Map each needle position to a cache index (order-preserving). Returns list of
    (cache_idx or None) aligned to needle. Skips a needle char not found ahead."""
    out, j = [], 0
    for oc in needle_chars:
        k = j
        while k < len(hay_chars) and hay_chars[k] != oc:
            k += 1
        if k < len(hay_chars):
            out.append(k); j = k + 1
        else:
            out.append(None)          # not found ahead (rare); drops this position
    return out


def best_cache_col(ocr_chars, cache_cols):
    """Pick the cache column that contains the most of ocr_chars as a subsequence."""
    best_i, best_hits = -1, -1
    for i, col in enumerate(cache_cols):
        if not col:
            continue
        hits = sum(1 for k in greedy_subseq(ocr_chars, [c["char"] for c in col]) if k is not None)
        if hits > best_hits:
            best_hits, best_i = hits, i
    return best_i, best_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="GOLD,SILVER")
    ap.add_argument("--maxlen", type=int, default=10)
    ap.add_argument("--minlen", type=int, default=1)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "finetune_data"))
    ap.add_argument("--jpeg_q", type=int, default=92)
    args = ap.parse_args()
    tiers = set(t.strip() for t in args.tiers.split(","))
    out_root = Path(args.out) / "Datasets" / "Patches"
    out_root.mkdir(parents=True, exist_ok=True)

    # labelled positions grouped by (book,page,column): (y_top, ocr_char, label, split)
    groups = defaultdict(list)
    with open(LABELS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["tier"] not in tiers or not r["label"] or not r.get("ocr_char"):
                continue
            try:
                bb = ast.literal_eval(r["bbox"])
            except Exception:
                continue
            groups[(r["book"], r["page"], r["column"])].append(
                (bb[1], r["ocr_char"], r["label"], r.get("split", "train")))

    cache_cache: dict[str, list] = {}
    def cache_cols(book, page):
        k = f"{book}/{page}"
        if k not in cache_cache:
            p = REPO / "prepared" / book_to_dir(book) / "detected" / f"{page}_ocr_cache.json"
            cache_cache[k] = json.load(open(p, encoding="utf-8")).get("columns", []) if p.exists() else []
        return cache_cache[k]
    page_cache: dict[str, Image.Image] = {}
    def get_page(book, page):
        k = f"{book}/{page}"
        if k not in page_cache:
            p = REPO / "prepared" / book_to_dir(book) / "pages_denoised" / f"{page}.png"
            page_cache[k] = Image.open(p).convert("RGB") if p.exists() else None
        return page_cache[k]

    manifests = defaultdict(list)
    n_chunks = n_chars = n_drop = 0
    lengths, vocab = [], set()

    for (book, page, column), items in groups.items():
        items.sort(key=lambda it: it[0])                 # by y_top
        ccols = cache_cols(book, page)
        page_img = get_page(book, page)
        if not ccols or page_img is None:
            n_drop += len(items); continue
        ocr_seq = [it[1] for it in items]
        # choose the cache column: trust int(column)-1, else search by content
        ci = int(column) - 1
        if not (0 <= ci < len(ccols)) or best_cache_col(ocr_seq, [ccols[ci]])[1] < 0.6 * len(ocr_seq):
            ci, _ = best_cache_col(ocr_seq, ccols)
        if ci < 0:
            n_drop += len(items); continue
        col = ccols[ci]
        idxs = greedy_subseq(ocr_seq, [c["char"] for c in col])   # cache index per labelled item

        # runs of CONSECUTIVE cache indices (all labelled) -> carry (label, split, bbox)
        assigned = [(idxs[k], items[k][2], items[k][3]) for k in range(len(items)) if idxs[k] is not None]
        n_drop += sum(1 for k in idxs if k is None)
        runs, cur = [], []
        for k, lab, sp in assigned:
            if cur and k == cur[-1][0] + 1:
                cur.append((k, lab, sp))
            else:
                if cur:
                    runs.append(cur)
                cur = [(k, lab, sp)]
        if cur:
            runs.append(cur)

        cidx = 0
        for run in runs:
            for i in range(0, len(run), args.maxlen):
                ch = run[i:i + args.maxlen]
                if len(ch) < args.minlen:
                    continue
                bxs = [col[k]["bbox"] for k, _, _ in ch]
                region = (max(0, min(b[0] for b in bxs) - args.pad),
                          max(0, min(b[1] for b in bxs) - args.pad),
                          max(b[2] for b in bxs) + args.pad,
                          max(b[3] for b in bxs) + args.pad)
                crop = page_img.crop(tuple(map(int, region)))
                if crop.width < 4 or crop.height < 8:
                    continue
                text = "".join(lab for _, lab, _ in ch)
                sp = ch[0][2]
                split = "val" if sp == "val" else ("test" if sp == "test" else "train")
                name = f"{book}/{book}_{page}_c{ci}_{cidx:03d}.jpg"; cidx += 1
                fp = out_root / name; fp.parent.mkdir(parents=True, exist_ok=True)
                crop.save(fp, quality=args.jpeg_q)
                manifests[split].append(f"{name}\t{text}")
                n_chunks += 1; n_chars += len(text); lengths.append(len(text)); vocab |= set(text)

    (out_root / "All.txt").write_text("\n".join(manifests["train"]), encoding="utf-8")
    (out_root / "Validate.txt").write_text("\n".join(manifests["val"]), encoding="utf-8")
    if manifests["test"]:
        (out_root / "Test.txt").write_text("\n".join(manifests["test"]), encoding="utf-8")

    import numpy as np
    L = np.array(lengths) if lengths else np.array([0])
    print(f"tiers={sorted(tiers)} maxlen={args.maxlen}")
    print(f"chunks: {n_chunks} (train {len(manifests['train'])}, val {len(manifests['val'])}, "
          f"test {len(manifests['test'])}) | dropped positions: {n_drop}")
    print(f"chars: {n_chars} | chunk len: median {int(np.median(L))} mean {L.mean():.1f} max {L.max()}")
    print(f"unique chars: {len(vocab)}  ->  {out_root}")


if __name__ == "__main__":
    main()
