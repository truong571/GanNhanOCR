"""Debug S3 — XEM model train so ảnh như thế nào.

S3 (visual_signal.VisualS3) so ảnh đúng 3 bước:
  1. Tiền xử lý crop: grayscale -> pad về vuông (nền trắng) -> resize enc.size
     -> normalize (0.5/0.5) -> tensor 1x3xSxS.
  2. embed: NomEmbedder (ResNet + ArcFace) -> vector D chiều, rồi L2-normalize.
  3. so khớp: với MỖI ký tự ứng viên, lấy glyph FontDiffusion của nó, embed y
     hệt, tính COSINE(crop, glyph). Xếp hạng giảm dần -> top1 = nhãn S3.
     (cosine ở đây đã map về [0,1]: (cos+1)/2, clamp 0 — xem NomEncoder.cosine.)

Script này tái hiện đúng pipeline đó trên crop THẬT trong dataset_out, mở rộng
tập ứng viên bằng SinoNom_Similar_Dic_v2 (các chữ NHÌN GIỐNG = hard negatives)
để thấy model có tách đúng nhãn thật khỏi đám lookalike hay không, rồi in bảng
cosine + xuất ảnh montage [crop | glyph từng ứng viên kèm cosine].

Chạy:
  .venv/bin/python evaluation/ver_new/debug_s3.py                 # 1 SILVER + 1 GOLD demo
  .venv/bin/python evaluation/ver_new/debug_s3.py --tier SILVER --n 5
  .venv/bin/python evaluation/ver_new/debug_s3.py --tier REVIEW --n 3
  .venv/bin/python evaluation/ver_new/debug_s3.py --image dataset_out/gold/xxx.png --true 經
  .venv/bin/python evaluation/ver_new/debug_s3.py --no-img        # chỉ in bảng, không vẽ
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from evaluation.ver_new.nom_classifier.infer import NomEncoder  # noqa: E402
from evaluation.ver_new.visual_signal import _find_ckpt        # noqa: E402

DATASET = HERE / "dataset_out"
FD_DIR = REPO / "gannhanocr-fd"
SIMILAR = REPO / "Dict" / "SinoNom_Similar_Dic_v2.csv"
OUT = HERE / "debug_out"


def is_cjk(ch: str) -> bool:
    if not ch or len(ch) != 1:
        return False
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2EBEF
            or 0xF900 <= o <= 0xFAFF)


def build_fd_index(fd_dir: Path) -> dict[str, str]:
    idx: dict[str, str] = {}
    for png in fd_dir.rglob("U+*.png"):
        try:
            idx[chr(int(png.stem.replace("U+", ""), 16))] = str(png)
        except ValueError:
            pass
    return idx


def load_similar(path: Path) -> dict[str, list[str]]:
    sim: dict[str, list[str]] = {}
    if not path.exists():
        return sim
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ch = (row.get("Input Character") or "").strip()
            try:
                lst = ast.literal_eval(row.get("Top 20 Similar Characters") or "[]")
            except (ValueError, SyntaxError):
                lst = []
            if ch:
                sim[ch] = [c for c in lst if is_cjk(c)]
    return sim


def nom_font(size: int):
    for fp in [REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf",
               REPO / "font_diffusion" / "fonts" / "HanaMinA.ttf"]:
        if fp.exists():
            try:
                return ImageFont.truetype(str(fp), size)
            except OSError:
                pass
    return None


def explain_preprocess(gray: np.ndarray, size: int) -> str:
    h, w = gray.shape
    s = max(h, w)
    return (f"gray {w}x{h} -> pad vuông {s}x{s} (nền 255) -> resize {size}x{size} "
            f"-> /255 -> norm(mean .5/std .5) -> tensor 1x3x{size}x{size}")


def candidates_for(true_ch: str, ocr_ch: str, sim: dict, fd: dict, k: int) -> list[str]:
    """Tập ứng viên = {ocr, true} + lookalikes của cả hai (chỉ giữ chữ có glyph FD)."""
    seq: list[str] = []
    for c in [ocr_ch, true_ch]:
        if is_cjk(c) and c not in seq:
            seq.append(c)
    for base in [true_ch, ocr_ch]:
        for c in sim.get(base, [])[:k]:
            if c not in seq:
                seq.append(c)
    return [c for c in seq if c in fd]


def book_code_map(config_path: Path) -> dict[str, str]:
    """code (book[12:], vd 'yen2') -> tên sách đầy đủ ('SachThanhTruyen2')."""
    import yaml
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    return {b["name"][12:]: b["name"] for b in cfg["books"]}


def page_crop_gray(code2book: dict, book_code: str, page: str, bbox) -> np.ndarray | None:
    """Crop ĐÚNG NHƯ production: ảnh trang gốc + bbox thô (không pad)."""
    book = code2book.get(book_code)
    if not book or bbox is None:
        return None
    pp = REPO / "prepared" / book / "pages" / f"{page}.png"
    if not pp.exists():
        return None
    x1, y1, x2, y2 = (int(v) for v in (bbox if isinstance(bbox, list) else ast.literal_eval(bbox)))
    return np.asarray(Image.open(pp).convert("L").crop((x1, y1, x2, y2)))


def rank(enc: NomEncoder, crop_gray: np.ndarray, cands: list[str], fd: dict):
    crop_emb = enc.embed_gray(crop_gray)
    scored = []
    for c in cands:
        e = enc.embed_path(fd[c])
        cos = enc.cosine(crop_emb, e) if e is not None else 0.0
        scored.append((c, cos))
    scored.sort(key=lambda x: x[1], reverse=True)
    return crop_emb, scored


def print_sample(idx: int, tier: str, crop_path: Path, ocr_ch: str, true_ch: str,
                 gray: np.ndarray, crop_emb, scored, size: int,
                 src: str, recorded: str = ""):
    h = "=" * 72
    print(f"\n{h}\n[{idx}] tier={tier}  crop={crop_path.name}  (nguồn embed: {src})")
    print(f"     OCR(S1) = '{ocr_ch}' (U+{ord(ocr_ch):04X})" +
          (f"   |  nhãn thật = '{true_ch}' (U+{ord(true_ch):04X})" if true_ch else ""))
    print(f"  1) tiền xử lý: {explain_preprocess(gray, size)}")
    print(f"  2) embed crop: dim={crop_emb.shape[0]}  L2norm={np.linalg.norm(crop_emb):.2f}  "
          f"head=[{', '.join(f'{v:+.2f}' for v in crop_emb[:5])} ...]")
    print(f"  3) cosine( crop , glyph-FD ứng viên )  — đã map về [0,1]:")
    print(f"     {'rank':>4}  {'char':^4}  {'unicode':^8}  {'cosine':>7}   ghi chú")
    for i, (c, cos) in enumerate(scored, 1):
        note = []
        if c == true_ch:
            note.append("NHÃN THẬT")
        if c == ocr_ch:
            note.append("OCR")
        if i == 1:
            note.append("<- S3 chọn")
        bar = "█" * int(round(cos * 20))
        print(f"     {i:>4}  {c:^4}  U+{ord(c):04X}  {cos:>7.3f}  {bar:<20} {' '.join(note)}")
    if true_ch and scored:
        top_c = scored[0][0]
        ok = "✅ ĐÚNG (top1 = nhãn thật)" if top_c == true_ch else \
             ("⚠️  top1 = OCR (không sửa)" if top_c == ocr_ch else "❌ top1 khác cả hai")
        margin = scored[0][1] - (scored[1][1] if len(scored) > 1 else 0.0)
        print(f"     => {ok}   | margin top1-top2 = {margin:+.3f}")
        if recorded:
            cos_true = next((c for ch, c in scored if ch == true_ch), None)
            extra = (f" | cosine('{true_ch}' này)={cos_true:.3f} -> "
                     + ("KHỚP" if cos_true is not None and abs(cos_true - float(recorded)) < 0.02
                        else "LỆCH (do tập ứng viên/nguồn crop khác)")) if cos_true is not None else ""
            print(f"     s3_cosine ghi trong labels.csv = {recorded}{extra}")


def montage(crop_img: Image.Image, ocr_ch: str, true_ch: str, scored, fd: dict,
            out_path: Path, tile: int = 150):
    font = nom_font(40)
    small = ImageFont.load_default()
    cap = 60
    tiles = []

    def make_tile(img_l: Image.Image, head: str, lines: list[str], border):
        t = Image.new("RGB", (tile, tile + cap), (255, 255, 255))
        g = img_l.convert("L").resize((tile - 8, tile - 8))
        t.paste(g.convert("RGB"), (4, 4))
        d = ImageDraw.Draw(t)
        d.rectangle([1, 1, tile - 2, tile - 2], outline=border, width=3)
        d.text((6, tile + 2), head, fill=(0, 0, 0), font=small)
        for j, ln in enumerate(lines):
            d.text((6, tile + 16 + j * 13), ln, fill=(0, 0, 0), font=small)
        return t

    tiles.append(make_tile(crop_img, "CROP (mộc bản)",
                           [f"OCR={ocr_ch} U+{ord(ocr_ch):04X}"], (30, 60, 200)))

    for i, (c, cos) in enumerate(scored, 1):
        glyph = Image.open(fd[c])
        if c == true_ch:
            border = (20, 160, 40)        # xanh lá = nhãn thật
        elif c == ocr_ch:
            border = (210, 120, 0)        # cam = OCR
        else:
            border = (170, 170, 170)
        marks = []
        if c == true_ch:
            marks.append("THẬT")
        if c == ocr_ch:
            marks.append("OCR")
        if i == 1:
            marks.append("S3")
        tiles.append(make_tile(glyph, f"#{i} U+{ord(c):04X}",
                               [f"cos={cos:.3f}", " ".join(marks)], border))

    W = sum(t.width for t in tiles) + 8 * (len(tiles) + 1)
    H = max(t.height for t in tiles) + 16
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    x = 8
    for t in tiles:
        canvas.paste(t, (x, 8))
        x += t.width + 8
    # ghi chú char thật nếu có font Nôm (chữ to góc trên)
    if font and true_ch:
        ImageDraw.Draw(canvas)  # font ổn thì caption đã đủ; glyph FD chính là hình chữ
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def pick_rows(tier: str | None, n: int) -> list[dict]:
    rows = list(csv.DictReader(open(DATASET / "labels.csv", encoding="utf-8")))
    pool = [r for r in rows if r["image"] and (tier is None or r["tier"] == tier)]
    # ưu tiên ca có ocr != label (thấy rõ S3 sửa) khi tier=SILVER
    if tier == "SILVER":
        pool.sort(key=lambda r: r["ocr_char"] == r["label"])
    return pool[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["GOLD", "SILVER", "SYLLABLE", "REVIEW"])
    ap.add_argument("--n", type=int, default=0, help="số mẫu (mặc định: 1 SILVER + 1 GOLD)")
    ap.add_argument("--image", help="debug 1 crop cụ thể (kèm --true)")
    ap.add_argument("--true", dest="true_ch", default="", help="nhãn thật của --image")
    ap.add_argument("--ocr", default="", help="ocr_char của --image (mặc định = --true)")
    ap.add_argument("--k", type=int, default=6, help="số lookalike thêm vào mỗi gốc")
    ap.add_argument("--from-page", action="store_true",
                    help="crop ĐÚNG NHƯ production (ảnh trang gốc + bbox thô) để đối chiếu s3_cosine")
    ap.add_argument("--no-img", action="store_true", help="không xuất montage")
    args = ap.parse_args()

    print("Đang nạp encoder + FD index ...", flush=True)
    enc = NomEncoder(_find_ckpt(REPO))
    fd = build_fd_index(FD_DIR)
    sim = load_similar(SIMILAR)
    code2book = book_code_map(REPO / "config" / "pipeline.yaml")
    print(f"  encoder on {enc.device} | img={enc.size} | FD glyphs={len(fd)} | similar={len(sim)}")

    samples: list[dict] = []
    if args.image:
        p = Path(args.image)
        if not p.is_absolute():
            p = (DATASET / p) if (DATASET / p).exists() else (REPO / p)
        samples.append({"tier": "custom", "crop": p, "ocr": args.ocr or args.true_ch,
                        "true": args.true_ch, "book": "", "page": "", "bbox": None, "rec": ""})
    else:
        rows = (pick_rows(args.tier, args.n or 5) if args.tier
                else pick_rows("SILVER", 1) + pick_rows("GOLD", 1))
        for r in rows:
            samples.append({"tier": r["tier"], "crop": DATASET / r["image"], "ocr": r["ocr_char"],
                            "true": r["label"], "book": r["book"], "page": r["page"],
                            "bbox": r["bbox"], "rec": r["s3_cosine"]})

    for i, s in enumerate(samples, 1):
        crop_path, ocr_ch, true_ch = s["crop"], s["ocr"], s["true"]
        if not crop_path.exists():
            print(f"[{i}] BỎ QUA — không thấy crop {crop_path}")
            continue
        cands = candidates_for(true_ch, ocr_ch, sim, fd, args.k)
        if not cands:
            print(f"[{i}] BỎ QUA — không dựng được ứng viên có glyph FD "
                  f"(ocr={ocr_ch!r}, true={true_ch!r})")
            continue
        # nguồn crop để embed: trang gốc+bbox (giống production) hoặc crop png đã lưu
        gray, src = None, "crop PNG đã lưu (pad 0.18)"
        if args.from_page and s["bbox"]:
            gray = page_crop_gray(code2book, s["book"], s["page"], s["bbox"])
            if gray is not None:
                src = "trang gốc + bbox thô (= production)"
        if gray is None:
            gray = np.asarray(Image.open(crop_path).convert("L"))
        crop_emb, scored = rank(enc, gray, cands, fd)
        print_sample(i, s["tier"], crop_path, ocr_ch, true_ch, gray, crop_emb, scored,
                     enc.size, src, s["rec"])
        if not args.no_img:
            out = OUT / f"s3_{s['tier'].lower()}_{crop_path.stem}.png"
            montage(Image.fromarray(gray), ocr_ch, true_ch, scored, fd, out)
            print(f"  -> montage: {out.relative_to(REPO)}")

    print(f"\nXong {len(samples)} mẫu."
          + ("" if args.no_img else f"  Ảnh trong {OUT.relative_to(REPO)}/"))


if __name__ == "__main__":
    main()
