"""Dump NomNaOCR (fine-tuned) per-column readings for the tri-model consensus test.

Runs in the TF env (paddle/TF have no Py3.14 wheel). For each page: crop each kim
column into ≤--chunk char segments, recognize, concatenate per column.
Output: nomnaocr_cache/<book>_<page>.json = {"columns": [<nna string per kim column>]}

Run:
  <tf_env>/bin/python evaluation/tri_consensus/dump_nomnaocr.py --book SachThanhTruyen4 --n 10 --chunk 8
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NOMNA = REPO / "NomNaOCR"
sys.path.insert(0, str(NOMNA))
from nomnaocr_rec import NomNaRecognizer                    # noqa: E402


def col_region(col, pad=4):
    xs1 = [c["bbox"][0] for c in col]; ys1 = [c["bbox"][1] for c in col]
    xs2 = [c["bbox"][2] for c in col]; ys2 = [c["bbox"][3] for c in col]
    return (min(xs1) - pad, min(ys1) - pad, max(xs2) + pad, max(ys2) + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen4")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--weights", default=str(NOMNA / "weights" / "finetuned_CRNNxCTC.h5"))
    ap.add_argument("--vocab", default=str(NOMNA / "weights" / "finetuned_vocab.txt"))
    args = ap.parse_args()

    pg = REPO / "prepared" / args.book / "pages_denoised"
    det = REPO / "prepared" / args.book / "detected"
    out = HERE / "nomnaocr_cache"; out.mkdir(exist_ok=True)
    rec = NomNaRecognizer(args.weights, args.vocab)
    stems = [f.stem for f in sorted(pg.glob("page_*.png"))
             if (det / f"{f.stem}_ocr_cache.json").exists()][: args.n]

    for stem in stems:
        cache = json.loads((det / f"{stem}_ocr_cache.json").read_text(encoding="utf-8"))
        cols = [c for c in cache.get("columns", []) if c]
        page = Image.open(pg / f"{stem}.png").convert("RGB")
        flat, spans = [], []
        for col in cols:
            groups = ([col[i:i + args.chunk] for i in range(0, len(col), args.chunk)]
                      if len(col) > args.chunk else [col])
            spans.append(len(groups))
            flat += [page.crop(tuple(map(int, col_region(g)))) for g in groups]
        res = rec.recognize(flat)
        colstr, k = [], 0
        for n in spans:
            colstr.append("".join(res[k:k + n])); k += n
        (out / f"{args.book}_{stem}.json").write_text(
            json.dumps({"columns": colstr}, ensure_ascii=False), encoding="utf-8")
        print(f"{stem}: {len(cols)} cols, {sum(len(s) for s in colstr)} nna chars")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
