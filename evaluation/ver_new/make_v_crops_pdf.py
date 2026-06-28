"""PDF xem crop mà V (valley+glyph-DP) cắt ra, so với midpoint (A) trên CÙNG cột diverged.
Mỗi ví dụ: hàng A (đỏ) = crop midpoint (hay dính chữ) | hàng V (xanh) = crop V (sạch hơn).

Run:
  .venv/bin/python evaluation/ver_new/make_v_crops_pdf.py --limit 20 --n 12
Output: results/v_crops.pdf
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.align_production import _detect, _reseg_column  # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
from evaluation.ver_new.eval_forced_align import valley_glyph_dp, fd_path  # noqa: E402
from evaluation.ver_new.seg_valley_n_ab import two_blob            # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "v_crops.pdf"


def afont(sz):
    for c in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"]:
        if Path(c).exists():
            return ImageFont.truetype(c, sz)
    return ImageFont.load_default()


def thumb(page_bgr, box, H=104):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = page_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, False
    crop = page_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    tb = two_blob(gray)
    im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    s = H / max(im.height, 1)
    im = im.resize((max(6, int(im.width * s)), H))
    return im, tb


def row_strip(page_bgr, boxes, color, H=104, gap=8):
    thumbs = [thumb(page_bgr, b, H) for b in boxes]
    thumbs = [(t, tb) for (t, tb) in thumbs if t is not None]
    if not thumbs:
        return None
    W = sum(t.width for t, _ in thumbs) + gap * (len(thumbs) + 1)
    strip = Image.new("RGB", (W, H + 16), "white")
    d = ImageDraw.Draw(strip)
    x = gap
    for t, tb in thumbs:
        strip.paste(t, (x, 4))
        # viền: dính-chữ -> đỏ đậm, sạch -> màu nhóm
        bc = (210, 0, 0) if tb else color
        d.rectangle([x, 4, x + t.width, 4 + H], outline=bc, width=3 if tb else 2)
        x += t.width + gap
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config(str(REPO / "config" / "pipeline.yaml")); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"])); qn_set = set(qn.keys())
    data_root = REPO / paths["data_dir"]
    enc = VisualS3(REPO, fd_dir="").enc
    gcache = {}
    def gemb(ch):
        if ch not in gcache:
            p = fd_path(ch); gcache[ch] = enc.embed_path(str(p)) if p else None
        return gcache[ch]

    examples = []
    for b in cfg["books"]:
        if len(examples) >= args.n:
            break
        data_dir = data_root / b["name"]
        trans = [t for t in sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
                 if not t.endswith("_qn_ocr_cache.json")][: args.limit]
        for tf in trans:
            if len(examples) >= args.n:
                break
            page = Path(tf).stem
            try:
                det = _detect(page, data_dir, qn_set)
            except Exception:
                continue
            if not det:
                continue
            cols, qn_lines, iter_pairs, binary, _ = det
            page_bgr = cv2.imread(str(data_dir / "pages" / f"{page}.png"), cv2.IMREAD_COLOR)
            if page_bgr is None:
                continue
            page_gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
            for nom_idx, line_id in iter_pairs:
                if len(examples) >= args.n:
                    break
                cluster = cols[nom_idx]; syl = qn_lines[line_id]
                if not syl or not cluster.get("chars"):
                    continue
                N = len(syl)
                if N < 2 or len(cluster["chars"]) == N:
                    continue
                chars = cluster["chars"]
                if cluster.get("x_range"):
                    cx1, cx2 = int(cluster["x_range"][0]), int(cluster["x_range"][1])
                else:
                    cx1 = min(int(c["bbox"][0]) for c in chars); cx2 = max(int(c["bbox"][2]) for c in chars)
                cy1 = min(int(c["bbox"][1]) for c in chars); cy2 = max(int(c["bbox"][3]) for c in chars)
                if cx2 - cx1 < 10 or cy2 - cy1 < 20:
                    continue
                cand, have = [], 0
                for i in range(N):
                    s = str(syl[i]).strip().lower(); embs = []
                    for c in qn.get(s, []):
                        if _is_cjk(c):
                            e = gemb(c)
                            if e is not None:
                                embs.append(e)
                    cand.append(np.stack(embs) if embs else None); have += bool(embs)
                if have < max(2, N // 2):
                    continue
                A = _reseg_column(cluster) or []
                # chỉ lấy ví dụ mà A có dính-chữ (để khoe V sửa)
                a_merge = any(thumb(page_bgr, bx)[1] for bx in A)
                if not a_merge:
                    continue
                vboxes, _ = valley_glyph_dp(page_gray[cy1:cy2, cx1:cx2], N, cand, enc)
                V = [(cx1, cy1 + y, cx2, cy1 + y + h) for (y, h) in vboxes]
                syls = " ".join(str(syl[i]) for i in range(N))
                examples.append((page, N, syls, A, V, page_bgr))
        print(f"  [{b['name']}] gom {len(examples)} ví dụ", flush=True)

    # render PDF: 3 ví dụ / trang
    per = 3; pages = []
    ft = afont(20); fs = afont(15)
    for pi in range(0, len(examples), per):
        chunk = examples[pi:pi + per]
        blocks = []
        for (page, N, syls, A, V, pbgr) in chunk:
            ra = row_strip(pbgr, A, (140, 140, 140))
            rv = row_strip(pbgr, V, (0, 150, 0))
            if ra is None or rv is None:
                continue
            w = max(ra.width, rv.width) + 70
            hblk = ra.height + rv.height + 64
            blk = Image.new("RGB", (w, hblk), "white"); d = ImageDraw.Draw(blk)
            d.text((6, 6), f"{page} · {N} chữ · âm: {syls}", fill=(120, 0, 0), font=fs)
            d.text((6, 30), "A midpoint", fill=(120, 120, 120), font=fs); blk.paste(ra, (96, 26))
            d.text((6, 30 + ra.height + 6), "V (valley+glyph-DP)", fill=(0, 130, 0), font=fs)
            blk.paste(rv, (96, 26 + ra.height + 6))
            blocks.append(blk)
        if not blocks:
            continue
        W = max(b.width for b in blocks) + 40
        Hh = sum(b.height + 24 for b in blocks) + 60
        pg = Image.new("RGB", (W, Hh), "white"); d = ImageDraw.Draw(pg)
        d.text((20, 16), "Crop do V cắt vs Midpoint — viền ĐỎ = dính 2 chữ (two_blob)", fill=(120, 0, 0), font=ft)
        y = 52
        for blk in blocks:
            pg.paste(blk, (20, y)); y += blk.height + 24
        pages.append(pg)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not pages:
        print("Không gom được ví dụ (thử tăng --limit)."); return
    pp = [p.convert("P", palette=Image.ADAPTIVE, colors=128) for p in pages]
    pp[0].save(str(OUT), save_all=True, append_images=pp[1:], resolution=150)
    print(f"PDF -> {OUT}  ({len(pages)} trang, {len(examples)} ví dụ)")


if __name__ == "__main__":
    main()
