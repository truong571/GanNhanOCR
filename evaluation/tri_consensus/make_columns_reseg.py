"""Report cột dùng DETECTOR (CenterNet) neo theo SỐ ÂM QN → ĐỦ CHỮ mỗi cột + âm 1-1.

Khác make_column_example (dùng box kim thô, thiếu chữ): ở đây vị trí = box detector ép về
đúng số âm QN (bắt cả chữ mờ/dính kim sót). Mỗi vị trí gắn ÂM Quốc-Ngữ 1-1 (neo). kim/qwen/
nna được căn về các vị trí đó rồi vote.

Run:
  .venv/bin/python evaluation/tri_consensus/make_columns_reseg.py \
      --book SachThanhTruyen2 --page page_0012 --qwen_dir qwen_cache_235b
Output: evaluation/tri_columns_reseg_<book>_<page>.pdf
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from run_tri_consensus import (is_cjk, align_map, load_qn_dict, load_similar, make_decider)
from make_column_example import render, p_intro, TIER_COL   # tái dùng vẽ

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def kim_to_slots(kcol, dboxes):
    """Gán chữ kim vào các box detector theo y (monotonic). Slot kim sót -> '·'."""
    n = len(dboxes)
    dc = [(b[1] + b[3]) / 2 for b in dboxes]
    ks = sorted(kcol, key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
    slot = ["·"] * n
    di = 0
    for c in ks:
        ky = (c["bbox"][1] + c["bbox"][3]) / 2
        while di < n - 1 and abs(dc[di + 1] - ky) < abs(dc[di] - ky):
            di += 1
        slot[di] = c["char"]
        di = min(di + 1, n - 1)
    return slot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen2")
    ap.add_argument("--page", default="page_0012")
    ap.add_argument("--qwen_dir", default="qwen_cache_235b")
    args = ap.parse_args()
    from core.ocr.ocr_api import load_columns_fullpage
    from pipeline.align_engine.char_detector.detector_infer import DetectorInfer

    book, page = args.book, args.page
    cdir = REPO / "prepared" / book / "detected"
    pdir = REPO / "prepared" / book / "pages_denoised"
    tdir = REPO / "prepared" / book / "transcriptions"
    kcols = load_columns_fullpage(str(cdir / f"{page}_ocr_cache.json"), str(pdir / f"{page}.png"))
    qn = json.load(open(tdir / f"{page}.json", encoding="utf-8"))
    qsyl = {c["column"]: c["syllables"] for c in qn["columns"]}
    full_bgr = cv2.imread(str(pdir / f"{page}.png"))
    full_gray = Image.open(pdir / f"{page}.png").convert("L")

    det = DetectorInfer("train_crop/detector_r34.best.pt")
    pboxes = det.boxes_for_page(full_bgr)
    print(f"detector page boxes: {len(pboxes)}")

    qnd, sim = load_qn_dict(), load_similar()
    decide = make_decider(qnd, sim, True, True)

    qtext = json.load(open(HERE / args.qwen_dir / f"{book}_{page}.json", encoding="utf-8")).get("text", "")
    qlines = [[c for c in ln if is_cjk(c)] for ln in qtext.splitlines()]
    qlines = [l for l in qlines if l]
    nf = HERE / "nomnaocr_cache" / f"{book}_{page}.json"
    ncols = json.load(open(nf, encoding="utf-8")).get("columns", []) if nf.exists() else []

    outp = REPO / "evaluation" / f"tri_columns_reseg_{book}_{page}.pdf"
    tot = [0, 0, 0]; totn = 0
    with PdfPages(outp) as pdf:
        p_intro(pdf, book, page)
        for ci, kcol in enumerate(kcols):
            syls = qsyl.get(ci + 1, [])
            n = len(syls) if syls else len(kcol)
            xs1 = [c["bbox"][0] for c in kcol]; xs2 = [c["bbox"][2] for c in kcol]
            dboxes = det.column_boxes(pboxes, (min(xs1), max(xs2)), n)
            dboxes = sorted(dboxes, key=lambda b: (b[1] + b[3]) / 2)
            kim_slot = kim_to_slots(kcol, dboxes)
            qcol = qlines[ci] if ci < len(qlines) else []
            ncol = list(ncols[ci]) if ci < len(ncols) else []
            qmap = align_map(kim_slot, qcol)
            nmap = align_map(kim_slot, ncol)

            rows, crops, err = [], [], [0, 0, 0]
            for i, b in enumerate(dboxes):
                kim = kim_slot[i]
                qw, nn = qmap[i], nmap[i]
                syl = syls[i] if i < len(syls) else ""
                base = kim if kim != "·" else (qw or nn or "")   # kim sót -> lấy phiếu khác làm gốc
                lbl, tier = decide(base, qw, nn, syl)
                if not lbl:
                    lbl = qw or nn or kim
                err[0] += (kim == "·" or kim != lbl)
                err[1] += bool(qw) and qw != lbl
                err[2] += bool(nn) and nn != lbl
                x1, y1, x2, y2 = (int(v) for v in b[:4])
                g = full_gray.crop((x1 - 3, y1 - 3, x2 + 3, y2 + 3)); g.thumbnail((48, 48))
                crops.append(np.asarray(g))
                rows.append(dict(kim=kim, qw=qw, nn=nn, syl=syl, lbl=lbl, tier=tier, bbox=[x1, y1, x2, y2]))
            # strip cả cột (union dboxes)
            X1 = min(int(b[0]) for b in dboxes) - 4; Y1 = min(int(b[1]) for b in dboxes) - 4
            X2 = max(int(b[2]) for b in dboxes) + 4; Y2 = max(int(b[3]) for b in dboxes) + 4
            strip = np.asarray(full_gray.crop((X1, Y1, X2, Y2)))
            render(pdf, book, page, ci, rows, err, strip, crops)
            for j in range(3):
                tot[j] += err[j]
            totn += n
            print(f"  cột {ci+1}: {n} vị trí (neo QN) | sai kim={err[0]} qwen={err[1]} nna={err[2]}")
    print(f"TỔNG {totn} vị trí (ĐỦ theo QN) | sai kim={tot[0]} ({100*tot[0]/totn:.0f}%) "
          f"qwen={tot[1]} ({100*tot[1]/totn:.0f}%) nna={tot[2]} ({100*tot[2]/totn:.0f}%)")
    print(f"PDF -> {outp}")


if __name__ == "__main__":
    main()
