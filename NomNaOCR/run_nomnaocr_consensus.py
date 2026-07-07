"""Test NomNaOCR CRNN as recognizer #3 vs Kinhhannom on N pages (kim = coordinate base).

For each page: read kim's OCR cache (9 columns + per-char bbox) → crop each COLUMN
region from the page → NomNaOCR reads it top→bottom → align (Levenshtein) to that
column's kim chars → report per-column & overall agreement. Column = the 9-col grid
(kim authority); NomNaOCR maps 1-1 to a column (vertical model, no rotation).

Self-contained for the tf_env (only tensorflow + PIL + numpy + nomnaocr_rec).

Run (tf_env):
  tf_env/bin/python run_nomnaocr_consensus.py \
      --book SachThanhTruyen4 --n 10 \
      --pages_dir /path/to/prepared --weights weights/NomNaOCR_CRNNxCTC.h5 --vocab vocab.txt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from PIL import Image

from nomnaocr_rec import NomNaRecognizer

HERE = Path(__file__).resolve().parent
DEFAULT_PREPARED = HERE.parent / "prepared"        # repo-root/prepared


def align(a, b):
    """Levenshtein backtrace -> counts (match, sub, del_a, ins_b)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    i, j, mt, sb, dl, ins = n, m, 0, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            mt += a[i - 1] == b[j - 1]; sb += a[i - 1] != b[j - 1]; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dl += 1; i -= 1
        else:
            ins += 1; j -= 1
    return mt, sb, dl, ins


def col_region(col, pad=4):
    xs1 = [c["bbox"][0] for c in col]; ys1 = [c["bbox"][1] for c in col]
    xs2 = [c["bbox"][2] for c in col]; ys2 = [c["bbox"][3] for c in col]
    return (min(xs1) - pad, min(ys1) - pad, max(xs2) + pad, max(ys2) + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen4")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pages_dir", default=str(DEFAULT_PREPARED))
    ap.add_argument("--weights", default=str(HERE / "weights" / "NomNaOCR_CRNNxCTC.h5"))
    ap.add_argument("--vocab", default=str(HERE / "vocab.txt"))
    ap.add_argument("--chunk", type=int, default=0,
                    help="split each column into groups of N chars before recognizing (0=whole column). "
                         "The CRNN degrades past ~6-8 chars; try --chunk 6.")
    args = ap.parse_args()

    prepared = Path(args.pages_dir)
    pg_dir = prepared / args.book / "pages_denoised"
    det_dir = prepared / args.book / "detected"
    stems = [f.stem for f in sorted(pg_dir.glob("page_*.png"))
             if (det_dir / f"{f.stem}_ocr_cache.json").exists()][: args.n]

    print(f"Loading NomNaOCR CRNN (weights={Path(args.weights).name}, vocab={Path(args.vocab).name}) ...")
    rec = NomNaRecognizer(args.weights, args.vocab)
    print(f"  vocab_size={rec.vocab_size} (expect 7481)\n")

    print(f"{'page':14} {'kimC':>5} {'nnaC':>5} {'kCol':>5} {'match%':>7} {'sub':>4} {'del':>4} {'ins':>4}")
    print("-" * 60)
    T = dict(kim=0, nna=0, mt=0, sb=0, dl=0, ins=0)
    out_dir = HERE / "out"; out_dir.mkdir(exist_ok=True)
    for stem in stems:
        cache = json.loads((det_dir / f"{stem}_ocr_cache.json").read_text(encoding="utf-8"))
        if cache.get("coords_space") not in (None, "fullpage") and cache.get("framed"):
            print(f"{stem:14} SKIP (cache not fullpage coords)"); continue
        cols = [c for c in cache.get("columns", []) if c]
        page = Image.open(pg_dir / f"{stem}.png").convert("RGB")
        # optionally split each column into short chunks (CRNN degrades past ~6-8 chars)
        flat_imgs, spans = [], []
        for col in cols:
            groups = ([col[i:i + args.chunk] for i in range(0, len(col), args.chunk)]
                      if args.chunk and len(col) > args.chunk else [col])
            spans.append(len(groups))
            flat_imgs += [page.crop(tuple(map(int, col_region(g)))) for g in groups]
        flat = rec.recognize(flat_imgs)
        nna_cols, k = [], 0
        for n in spans:
            nna_cols.append("".join(flat[k:k + n])); k += n

        pmt = psb = pdl = pins = pk = pn = 0
        dump = [f"# {stem}  |  kim_cols={len(cols)}"]
        for ci, (col, nna) in enumerate(zip(cols, nna_cols)):
            kim = [c["char"] for c in col]
            mt, sb, dl, ins = align(kim, list(nna))
            pmt += mt; psb += sb; pdl += dl; pins += ins; pk += len(kim); pn += len(nna)
            dump.append(f"  cột{ci+1}: kim({len(kim)})={''.join(kim)}")
            dump.append(f"         nna({len(nna)})={nna}   match={mt} sub={sb} del={dl} ins={ins}")
        denom = pmt + psb or 1
        print(f"{stem:14} {pk:5} {pn:5} {len(cols):5} {100*pmt/denom:6.1f}% {psb:4} {pdl:4} {pins:4}")
        (out_dir / f"nna_{stem}.txt").write_text("\n".join(dump), encoding="utf-8")
        T["kim"] += pk; T["nna"] += pn; T["mt"] += pmt; T["sb"] += psb; T["dl"] += pdl; T["ins"] += pins

    d = T["mt"] + T["sb"] or 1
    print("-" * 60)
    print(f"TOTAL kim={T['kim']} nna={T['nna']} | match={T['mt']} ({100*T['mt']/d:.1f}%) "
          f"sub={T['sb']} del={T['dl']} ins={T['ins']}")
    print(f"per-column side-by-side -> {out_dir}/nna_<page>.txt")


if __name__ == "__main__":
    main()
