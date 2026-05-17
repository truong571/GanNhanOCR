"""POC: re-OCR Vietnamese text from Sach* PDFs using VietOCR.

The 3 SachThanhTruyen PDFs have a fake-ASCII text layer (Due, nu6c, m<;>i...),
so PyMuPDF can't recover Unicode. We instead RENDER the page to an image and
re-OCR with VietOCR.

Pipeline (no PaddleOCR — keeps deps slim):
  1. Render PDF page -> grayscale image
  2. Horizontal projection -> detect text line bboxes
  3. Crop each line, pass to VietOCR
  4. Print OCR'd lines side-by-side with the PyMuPDF garbage text

Outputs in evaluation/test_qn_ocr/out/:
  page_<book>_<page>_render.png     rendered page
  page_<book>_<page>_lines.png      visualization of detected line bboxes
  page_<book>_<page>_ocr.txt        VietOCR transcription
  comparison.md                     side-by-side comparison + dict-hit metric

Usage:
    PYTHONPATH=. .venv/bin/python evaluation/test_qn_ocr/run_poc.py
    PYTHONPATH=. .venv/bin/python evaluation/test_qn_ocr/run_poc.py --book SachThanhTruyen4 --page 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def render_page(pdf_path: Path, page_idx: int, dpi: int = 300) -> np.ndarray:
    doc = fitz.open(str(pdf_path))
    pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def pdf_text(pdf_path: Path, page_idx: int) -> str:
    doc = fitz.open(str(pdf_path))
    t = doc[page_idx].get_text("text")
    doc.close()
    return t


def detect_text_lines(img_rgb: np.ndarray, min_height: int = 18,
                      gap: int = 8) -> list[tuple[int, int, int, int]]:
    """Horizontal projection -> rows of text. Returns list of (x1, y1, x2, y2)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Binarize: ink = 1
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    proj = bw.sum(axis=1)  # ink per row
    h, w = bw.shape
    # threshold: rows with >2% pixels = ink
    th = max(int(w * 0.01), 5)
    is_ink = proj > th

    lines = []
    in_run = False
    start = 0
    for y in range(h):
        if is_ink[y] and not in_run:
            in_run = True
            start = y
        elif not is_ink[y] and in_run:
            in_run = False
            if y - start >= min_height:
                lines.append([start, y])
    if in_run and h - start >= min_height:
        lines.append([start, h])

    # merge close runs (gap pixels apart)
    merged = []
    for s, e in lines:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # trim horizontally per line
    out = []
    for s, e in merged:
        crop = bw[s:e]
        col = crop.sum(axis=0)
        nz = np.where(col > 1)[0]
        if not len(nz):
            continue
        x1, x2 = int(nz[0]), int(nz[-1]) + 1
        # add small padding
        pad = 5
        out.append((max(0, x1 - pad), max(0, s - pad),
                    min(w, x2 + pad), min(h, e + pad)))
    return out


def visualize_lines(img: np.ndarray, boxes: list, path: Path) -> None:
    vis = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(vis, str(i + 1), (x1 + 5, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def init_predictor():
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    cfg = Cfg.load_config_from_name("vgg_transformer")
    cfg["cnn"]["pretrained"] = False
    cfg["device"] = "cpu"
    cfg["predictor"]["beamsearch"] = False
    return Predictor(cfg)


def load_qn_dict() -> set[str]:
    df = pd.read_csv(REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv")
    col = df.columns[0]
    for c in df.columns:
        if "quoc" in c.lower() or c.lower() == "qn":
            col = c
            break
    return {str(s).strip().lower() for s in df[col].dropna() if str(s).strip()}


def syllable_hit_rate(text: str, dict_set: set[str]) -> tuple[int, int]:
    """Tokenize text into syllables (lowercase, alpha-only), return (hits, total)."""
    import re
    toks = re.findall(r"[A-Za-zÀ-ỹà-ỹĐđ]+", text)
    if not toks:
        return 0, 0
    hits = sum(1 for t in toks if t.lower() in dict_set)
    return hits, len(toks)


def process(pdf_path: Path, page_idx: int, predictor) -> dict:
    book = pdf_path.stem
    tag = f"{book}_p{page_idx + 1}"

    img = render_page(pdf_path, page_idx, dpi=300)
    render_path = OUT / f"{tag}_render.png"
    cv2.imwrite(str(render_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    boxes = detect_text_lines(img)
    visualize_lines(img, boxes, OUT / f"{tag}_lines.png")

    pil = Image.fromarray(img)
    t0 = time.time()
    lines_text = []
    for (x1, y1, x2, y2) in boxes:
        crop = pil.crop((x1, y1, x2, y2))
        if crop.size[0] < 10 or crop.size[1] < 10:
            continue
        try:
            txt = predictor.predict(crop)
            lines_text.append(txt.strip())
        except Exception as e:
            lines_text.append(f"<ERR: {e}>")
    elapsed = time.time() - t0

    ocr_txt = "\n".join(lines_text)
    (OUT / f"{tag}_ocr.txt").write_text(ocr_txt, encoding="utf-8")

    return {
        "book": book,
        "page": page_idx + 1,
        "num_lines": len(boxes),
        "elapsed_s": round(elapsed, 1),
        "pdf_text": pdf_text(pdf_path, page_idx),
        "ocr_text": ocr_txt,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="*", default=[
        "SachThanhTruyen2:1",
        "SachThanhTruyen4:1",
        "SachThanhTruyen11:1",
    ], help="book:page_idx (0-based) pairs")
    args = p.parse_args()

    print("Loading VietOCR predictor (vgg_transformer)...")
    predictor = init_predictor()
    print("Loading QN dictionary...")
    qn_dict = load_qn_dict()
    print(f"Dictionary: {len(qn_dict):,} unique syllables")
    print()

    rows = []
    for case in args.cases:
        book, idx_s = case.split(":")
        idx = int(idx_s)
        pdf_path = REPO / "Data" / f"{book}.pdf"
        if not pdf_path.exists():
            print(f"!! Missing {pdf_path}")
            continue
        print(f"=== {book} page {idx + 1} ===")
        r = process(pdf_path, idx, predictor)
        pdf_hits, pdf_total = syllable_hit_rate(r["pdf_text"], qn_dict)
        ocr_hits, ocr_total = syllable_hit_rate(r["ocr_text"], qn_dict)
        r.update({
            "pdf_dict_hit": f"{pdf_hits}/{pdf_total} = "
                            f"{(pdf_hits/pdf_total*100) if pdf_total else 0:.1f}%",
            "ocr_dict_hit": f"{ocr_hits}/{ocr_total} = "
                            f"{(ocr_hits/ocr_total*100) if ocr_total else 0:.1f}%",
        })
        print(f"  lines={r['num_lines']}  elapsed={r['elapsed_s']}s")
        print(f"  PDF dict hit: {r['pdf_dict_hit']}")
        print(f"  OCR dict hit: {r['ocr_dict_hit']}")
        print()
        rows.append(r)

    # comparison markdown
    md = ["# QN re-OCR POC comparison", "",
          "Compare PyMuPDF text (broken on Sach*) vs VietOCR re-OCR.",
          "",
          "## Summary", "",
          "| Book | Page | Lines | Time (s) | PDF dict hit | OCR dict hit |",
          "|------|-----:|------:|---------:|-------------:|-------------:|"]
    for r in rows:
        md.append(f"| {r['book']} | {r['page']} | {r['num_lines']} | "
                  f"{r['elapsed_s']} | {r['pdf_dict_hit']} | {r['ocr_dict_hit']} |")

    md += ["", "## Side-by-side per page", ""]
    for r in rows:
        md += [f"### {r['book']} page {r['page']}", "",
               "**PyMuPDF text (current pipeline):**", "", "```",
               r["pdf_text"][:600], "```", "",
               "**VietOCR re-OCR (proposed):**", "", "```",
               r["ocr_text"][:600], "```", ""]
    (OUT / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT / 'comparison.md'}")
    print(f"All artifacts in {OUT}")


if __name__ == "__main__":
    main()
