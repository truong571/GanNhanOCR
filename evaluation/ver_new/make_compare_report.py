"""PDF ngắn gọn: các CÁCH SO-KHỚP-ẢNH thay cho DINOv2 — kết quả + vướng mắc từng
cách + ví dụ trực quan chứng minh (crop ↔ glyph đúng/sai + điểm số).

Số retrieval@1 đo bởi compare_methods.py (nhúng hằng số). Ví dụ thất bại được tìm
LIVE: một crop mà encoder train chọn ĐÚNG còn cách không-train chọn SAI, kèm ảnh
+ điểm số thật. Chỉ dùng PIL (lưu PDF ở mode 'P' vì build Pillow này thiếu JPEG).

Chạy:
  .venv/bin/python evaluation/ver_new/make_compare_report.py
  -> evaluation/ver_new/BAOCAO_SoSanh_S3.pdf
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
import evaluation.ver_new.compare_methods as CM                    # noqa: E402

VI = "/System/Library/Fonts/Supplemental/Arial.ttf"
VIB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
NOM = str(REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf")
HANB = str(REPO / "font_diffusion" / "fonts" / "HanaMinB.ttf")   # Ext-B
HANC = str(REPO / "font_diffusion" / "fonts" / "HanaMinC.otf")   # Ext-C+
_ff: dict = {}


def nom_font(ch, size):
    """Font cho 1 chữ Nôm theo codepoint (NomNaTong cho BMP, HanaMin B/C cho Ext-B/C+)."""
    cp = ord(ch) if (ch and len(ch) == 1) else 0
    path = NOM if cp < 0x20000 else (HANB if cp < 0x2A700 else HANC)
    k = (path, size)
    if k not in _ff:
        try:
            _ff[k] = ImageFont.truetype(path, size)
        except Exception:
            _ff[k] = ImageFont.truetype(NOM, size)
    return _ff[k]
W, H, M = 1240, 1754, 90
INK, MUTE, ACC, OKC, BAD = (25, 25, 30), (95, 95, 100), (30, 90, 165), (30, 150, 70), (200, 50, 50)
OUT = HERE / "BAOCAO_SoSanh_S3.pdf"

# retrieval@1 đo bởi compare_methods.py (n=600 VAL; ngẫu nhiên ~5%)
RESULTS = [("trained (ResNet+ArcFace)", "có", 81.7), ("template_ncc", "không", 16.8),
           ("pixel_cos", "không", 16.0), ("dice", "không", 15.3),
           ("stroke_grid", "không", 14.0), ("proj_profile", "không", 13.8),
           ("hog", "không", 8.8), ("DINOv2 (zero-shot)", "không", 0.2)]
VUONG = [
    ("DINOv2 / CLIP (zero-shot)", "Encoder ảnh TỰ NHIÊN — embedding không phân biệt glyph "
     "Nôm (cosine ~bằng nhau giữa các chữ khác nhau). retrieval ~0% (≈ ngẫu nhiên)."),
    ("pixel_cos", "So từng điểm ảnh — nhạy với dịch/độ-đậm/nhiễu, không bất biến phong cách viết."),
    ("template_ncc", "Cần CĂN KHÍT; nét viết tay biến dạng phi tuyến (cong/đứt/ink bleed) làm "
     "tương quan chéo sụp. Chỉ tốt khi gần như bản sao."),
    ("dice (chồng nhị phân)", "Phụ thuộc nhị-phân-hoá + khít pixel; ngưỡng + thấm mực khiến "
     "overlap thấp NGAY CẢ khi cùng chữ."),
    ("stroke_grid (lưới mật-độ-nét)", "Lưới ink thô → mất chi tiết nét; chữ khác có phân bố ink "
     "tương tự không phân biệt được."),
    ("proj_profile (hình chiếu)", "Hình chiếu 1D quá thô — nhiều chữ chung profile, mất cấu trúc 2D."),
    ("hog (hướng gradient)", "Gradient của nét gãy/nhiễu hỗn loạn + cell thô → kém nhất (8.8%)."),
    ("trained (ResNet+ArcFace)", "HỌC bất biến phong cách → 81.7%. Vướng RIÊNG: cần GPU + dữ "
     "liệu train; precision là proxy; có bias miền crop↔glyph (đã thêm open-set guard)."),
]


def F(p, s):
    return ImageFont.truetype(p, s)


def tw(d, t, f):
    return d.textlength(t, font=f)


def wrap(d, t, f, mw):
    out, ln = [], ""
    for w in t.split():
        s = (ln + " " + w).strip()
        if tw(d, s, f) <= mw:
            ln = s
        else:
            out.append(ln); ln = w
    if ln:
        out.append(ln)
    return out


class Page:
    def __init__(self):
        self.img = Image.new("RGB", (W, H), (255, 255, 255)); self.d = ImageDraw.Draw(self.img); self.y = M

    def line(self, t, f, c=INK, g=6, x=M):
        self.d.text((x, self.y), t, fill=c, font=f); self.y += int(f.size * 1.4) + g

    def para(self, t, f, c=INK, g=8, x=M, mw=W - 2 * M):
        for l in wrap(self.d, t, f, mw):
            self.d.text((x, self.y), l, fill=c, font=f); self.y += int(f.size * 1.42)
        self.y += g

    def rule(self, g=14):
        self.y += 4; self.d.line([(M, self.y), (W - M, self.y)], fill=(225, 225, 228), width=2); self.y += g


def footer(p, n):
    p.d.text((M, H - 52), "So sánh các cách so-khớp-ảnh S3 (thay DINOv2)", fill=MUTE, font=F(VI, 19))
    p.d.text((W - M - 70, H - 52), f"Trang {n}", fill=MUTE, font=F(VI, 19))


def cap(p, x, y, prefix, ch, color, ps=18, cs=26):
    """Caption: tiền tố tiếng Việt (Arial) + chữ Nôm (font phủ rộng) — tránh ô vuông."""
    p.d.text((x, y + max(0, cs - ps)), prefix, fill=color, font=F(VIB, ps))
    xo = x + int(tw(p.d, prefix + " ", F(VIB, ps)))
    if ch:
        p.d.text((xo, y), ch, fill=color, font=nom_font(ch, cs))


def find_examples(vs3, qn, D, k=2, scan=160):
    """Tìm crop mà 'trained' chọn ĐÚNG còn 'dice' (không-train) chọn SAI."""
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == "GOLD" and r["split"] == "val" and r["image"] and r["syllable"]]
    random.seed(1); random.shuffle(rows)
    found = []
    gcache = {}

    def gly(ch):
        if ch in gcache:
            return gcache[ch]
        p = vs3.fd_index.get(ch)
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p else None
        if g is None:
            gcache[ch] = None; return None
        gf, bw = CM.prep(g)
        gcache[ch] = dict(path=p, gf=gf, bw=bw, emb=vs3.enc.embed_path(str(p)))
        return gcache[ch]

    for r in rows[:scan]:
        cg = cv2.imread(str(D / r["image"]), cv2.IMREAD_GRAYSCALE)
        if cg is None:
            continue
        cgf, cbw = CM.prep(cg)
        cemb = vs3.enc.embed_gray((cgf * 255).astype(np.uint8))
        true = r["label"]
        cands = []
        for c in ([r["ocr_char"]] if r["ocr_char"] else []) + qn.get((r["syllable"] or "").lower(), []):
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if true not in cands:
            cands.append(true)
        cands = [c for c in cands if gly(c) is not None]
        if len(cands) < 3 or true not in cands:
            continue
        tr = {c: CM.cos(cemb, gcache[c]["emb"]) for c in cands}
        di = {c: CM.s_dice(cbw, gcache[c]["bw"]) for c in cands}
        tw_, dw = max(tr, key=tr.get), max(di, key=di.get)
        if tw_ == true and dw != true:
            found.append(dict(r=r, true=true, crop=cg, dice_pick=dw,
                              tr_true=tr[true], tr_dice=tr[dw], di_true=di[true], di_dice=di[dw],
                              g_true=gcache[true]["path"], g_dice=gcache[dw]["path"]))
            if len(found) >= k:
                break
    return found


def paste(p, path_or_img, box, size=140):
    try:
        im = (Image.open(path_or_img) if not isinstance(path_or_img, np.ndarray)
              else Image.fromarray(path_or_img)).convert("RGB").resize((size, size))
        p.img.paste(im, box)
    except Exception:
        pass


def main():
    cfg = load_config(str(REPO / "config" / "pipeline.yaml")); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]))
    D = HERE / "dataset_out"
    pages = []

    # ---- PAGE 1: kết quả ----
    p = Page()
    p.line("BÁO CÁO — Các cách so-khớp-ảnh thay cho DINOv2", F(VIB, 27), c=ACC, g=8)
    p.para("So khớp crop chữ Nôm viết tay với glyph tham chiếu (tương đồng) để xếp hạng ký tự "
           "ứng viên (tầng S3). Đo retrieval@1 trên 600 quyết định VAL, tập ứng viên thật "
           "(~19 chữ → ngẫu nhiên ~5%).", F(VI, 21), g=14)
    p.rule()
    p.line("retrieval@1 theo phương pháp", F(VIB, 24), c=ACC, g=10)
    x0 = M + 430; bw = W - M - x0 - 110
    for i, (name, tr, v) in enumerate(RESULTS):
        yc = p.y + i * 58
        p.d.text((M, yc + 12), name, fill=INK, font=F(VI, 21))
        col = ACC if tr == "có" else (BAD if v < 6 else MUTE)
        p.d.rectangle([x0, yc, x0 + bw, yc + 40], fill=(238, 238, 242))
        p.d.rectangle([x0, yc, x0 + int(bw * v / 100), yc + 40], fill=col)
        p.d.text((x0 + int(bw * v / 100) + 10, yc + 9), f"{v:.1f}%", fill=INK, font=F(VIB, 21))
    p.y += len(RESULTS) * 58 + 8
    # chance line marker
    p.d.line([(x0 + int(bw * 0.05), p.y - len(RESULTS) * 58 - 8), (x0 + int(bw * 0.05), p.y)],
             fill=(150, 150, 150), width=1)
    p.para("Vạch xám mờ ≈ mức ngẫu nhiên (5%). Mọi cách KHÔNG-train (8–17%) chỉ nhỉnh hơn ngẫu "
           "nhiên; DINOv2 ≈ 0%. Chỉ encoder ĐÃ TRAIN đạt 81.7%.", F(VI, 20), c=MUTE, g=10)
    p.rule()
    p.line("Kết luận", F(VIB, 22), c=ACC)
    p.para("Không có cách so-khớp KHÔNG-train nào dùng được cho chữ Nôm viết tay — DINOv2 không "
           "phải ngoại lệ. Biến thiên nét viết tay (cong/đứt/nhiễu/thấm mực) phá mọi đặc trưng "
           "thủ công/pixel. Encoder TRAIN (học bất biến phong cách) gần như BẮT BUỘC.", F(VI, 21))
    footer(p, 1); pages.append(p.img)

    # ---- PAGE 2: vướng từng cách + montage crop↔glyph ----
    p = Page()
    p.line("Vướng mắc ở từng cách", F(VIB, 25), c=ACC, g=12)
    for name, why in VUONG:
        p.line("• " + name, F(VIB, 21), g=2)
        p.para(why, F(VI, 20), x=M + 24, mw=W - 2 * M - 24, g=8)
    p.rule()
    p.line("Vì sao khó: crop mộc bản (trái) vs glyph tương đồng (phải)", F(VIB, 21), c=ACC, g=10)
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == "GOLD" and r["split"] == "val" and r["image"] and r["label"]]
    random.seed(3); random.shuffle(rows)
    shown = 0; x = M; y0 = p.y
    for r in rows:
        gp = vs3.fd_index.get(r["label"])
        cp = D / r["image"]
        if not gp or not cp.exists():
            continue
        cx = M + (shown % 4) * 270
        paste(p, str(cp), (cx, y0), 120)
        paste(p, str(gp), (cx + 124, y0), 120)
        p.d.text((cx, y0 + 124), r["label"], fill=MUTE, font=nom_font(r["label"], 22))
        shown += 1
        if shown >= 4:
            break
    p.y = y0 + 165
    p.para("Cùng một chữ nhưng crop nhiễu/đứt nét/lệch nhẹ so với glyph → đối sánh pixel/cấu trúc "
           "thất bại; encoder train mới 'nhìn xuyên' phong cách.", F(VI, 20), c=MUTE)
    footer(p, 2); pages.append(p.img)

    # ---- PAGE 3: ví dụ chứng minh (trained đúng, dice sai) ----
    p = Page()
    p.line("Ví dụ chứng minh — vì sao cách không-train chọn SAI", F(VIB, 24), c=ACC, g=12)
    ex = find_examples(vs3, qn, D, k=2)
    if not ex:
        p.para("(Không tìm thấy ví dụ trong mẫu quét — chạy lại với scan lớn hơn.)", F(VI, 21))
    for e in ex:
        y0 = p.y
        paste(p, e["crop"], (M, y0), 150)
        p.d.text((M, y0 + 154), "crop viết tay", fill=MUTE, font=F(VI, 18))
        # correct glyph (trained's pick)
        paste(p, str(e["g_true"]), (M + 320, y0), 150)
        cap(p, M + 320, y0 + 154, "ĐÚNG (trained):", e["true"], OKC)
        # dice's wrong pick
        paste(p, str(e["g_dice"]), (M + 640, y0), 150)
        cap(p, M + 640, y0 + 154, "dice → SAI:", e["dice_pick"], BAD)
        p.y = y0 + 190
        p.para(f"trained: cos(crop, ĐÚNG)={e['tr_true']:.3f}  >  cos(crop, sai)={e['tr_dice']:.3f}  "
               f"=> chọn ĐÚNG", F(VI, 20), c=OKC, g=2)
        p.para(f"dice:    overlap(ĐÚNG)={e['di_true']:.3f}  <  overlap(sai)={e['di_dice']:.3f}  "
               f"=> chọn SAI  (nhị phân/khít pixel đánh lừa)", F(VI, 20), c=BAD, g=14)
        p.rule()
    p.para("Encoder train xếp đúng chữ #1 nhờ embedding bất biến phong cách; cách không-train bị "
           "nhiễu/độ-khít pixel đánh lừa nên xếp nhầm chữ nhìn-giống lên #1.", F(VI, 20), c=MUTE)
    footer(p, 3); pages.append(p.img)

    pal = [pg.convert("P", palette=Image.ADAPTIVE, colors=256) for pg in pages]
    pal[0].save(OUT, save_all=True, append_images=pal[1:], resolution=150.0)
    print(f"PDF -> {OUT}  ({len(pages)} trang, {OUT.stat().st_size // 1024} KB) · {len(ex)} ví dụ")


if __name__ == "__main__":
    main()
