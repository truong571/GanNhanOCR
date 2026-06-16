"""Xuất PDF báo cáo nghiên cứu: cải tiến tầng so-khớp-ảnh S3 (Bước 0/1/2 +
crop-bias guard) cho pipeline gán nhãn chữ Nôm.

Số liệu sống đọc từ dataset_out/summary.json + s3_calibration.json; các kết quả
đo (ablation, char-disjoint, tiến triển tier) nhúng dạng hằng số (đo trên lần
chạy này). Chỉ dùng PIL (không cần matplotlib) — vẽ bảng + bar chart + montage
crop↔glyph để soi trực quan.

Chạy:
  .venv/bin/python evaluation/ver_new/make_s3_report.py
  -> evaluation/ver_new/BAOCAO_S3_NghienCuu.pdf
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATASET = HERE / "dataset_out"
SAMPLE = HERE / "eval_sample"
OUT = HERE / "BAOCAO_S3_NghienCuu.pdf"

# fonts (macOS Arial covers Vietnamese; NomNaTong covers chữ Nôm)
VI = "/System/Library/Fonts/Supplemental/Arial.ttf"
VIB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
NOM = str(REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf")

W, H = 1240, 1754                 # A4 @ ~150 dpi
M = 90                            # margin
INK, MUTE, ACC = (25, 25, 30), (95, 95, 100), (30, 90, 165)
GOLD, SILVER, SYL, REV = (200, 160, 30), (130, 130, 140), (45, 110, 190), (210, 120, 40)


def F(path, s):
    return ImageFont.truetype(path, s)


def tw(draw, txt, font):
    return draw.textlength(txt, font=font)


def wrap(draw, txt, font, maxw):
    out, line = [], ""
    for word in txt.split():
        t = (line + " " + word).strip()
        if tw(draw, t, font) <= maxw:
            line = t
        else:
            if line:
                out.append(line)
            line = word
    if line:
        out.append(line)
    return out


class Page:
    def __init__(self):
        self.img = Image.new("RGB", (W, H), (255, 255, 255))
        self.d = ImageDraw.Draw(self.img)
        self.y = M

    def para(self, txt, font, color=INK, lh=1.42, gap=8, x=M, maxw=W - 2 * M):
        for ln in wrap(self.d, txt, font, maxw):
            self.d.text((x, self.y), ln, fill=color, font=font)
            self.y += int(font.size * lh)
        self.y += gap

    def line(self, txt, font, color=INK, gap=6, x=M):
        self.d.text((x, self.y), txt, fill=color, font=font)
        self.y += int(font.size * 1.4) + gap

    def rule(self, gap=14, color=(225, 225, 228)):
        self.y += 4
        self.d.line([(M, self.y), (W - M, self.y)], fill=color, width=2)
        self.y += gap


def footer(pg, n):
    f = F(VI, 19)
    pg.d.text((M, H - 52), "Cải tiến tầng so-khớp-ảnh S3 · pipeline gán nhãn chữ Nôm",
              fill=MUTE, font=f)
    pg.d.text((W - M - 70, H - 52), f"Trang {n}", fill=MUTE, font=f)


def bar_chart(pg, title, items, vmax, unit="%", colors=None, h=46, gap=20, val_fmt="{:.1f}"):
    """items = [(label, value)]. Horizontal bars."""
    f_t = F(VIB, 26); f_l = F(VI, 22); f_v = F(VIB, 22)
    pg.line(title, f_t, color=ACC, gap=10)
    x0 = M + 360
    barw = (W - M) - x0 - 110
    for i, (lab, v) in enumerate(items):
        yc = pg.y + i * (h + gap)
        pg.d.text((M, yc + h // 2 - 13), lab, fill=INK, font=f_l)
        pg.d.rectangle([x0, yc, x0 + barw, yc + h], fill=(238, 238, 242))
        ww = int(barw * min(v / vmax, 1.0))
        col = (colors[i] if colors else ACC)
        pg.d.rectangle([x0, yc, x0 + ww, yc + h], fill=col)
        pg.d.text((x0 + ww + 12, yc + h // 2 - 13), val_fmt.format(v) + unit, fill=INK, font=f_v)
    pg.y += len(items) * (h + gap) + 12


def table(pg, headers, rows, colw, hl_last=False):
    f_h = F(VIB, 21); f_c = F(VI, 21); f_cb = F(VIB, 21)
    x = M
    for hh, w in zip(headers, colw):
        pg.d.text((x + 8, pg.y + 6), hh, fill=ACC, font=f_h)
        x += w
    pg.y += 36
    pg.d.line([(M, pg.y), (M + sum(colw), pg.y)], fill=(210, 210, 215), width=2)
    pg.y += 6
    for ri, row in enumerate(rows):
        x = M
        fnt = f_cb if (hl_last and ri == len(rows) - 1) else f_c
        bg = (245, 248, 252) if (hl_last and ri == len(rows) - 1) else None
        if bg:
            pg.d.rectangle([M, pg.y - 2, M + sum(colw), pg.y + 32], fill=bg)
        for cell, w in zip(row, colw):
            pg.d.text((x + 8, pg.y + 3), str(cell), fill=INK, font=fnt)
            x += w
        pg.y += 36
    pg.y += 10


# ----------------------------- load live numbers -----------------------------
summ = json.load(open(DATASET / "summary.json", encoding="utf-8"))
cal = json.load(open(HERE / "s3_calibration.json", encoding="utf-8"))
T = summ["tiers"]

STAGES = [  # name, GOLD, SILVER, SYLLABLE, REVIEW, usable_total
    ("Gốc (baseline)", 51195, 6747, 5486, 18840, 63428),
    ("B0: mở khóa", 51195, 8470, 6881, 15722, 66546),
    ("B0+1+2: calibration", 51195, 9726, 5943, 15404, 66864),
    ("+ crop-bias guard (cuối)", T["GOLD"], T["SILVER"], T["SYLLABLE"], T["REVIEW"], summ["usable_total"]),
]
ABLATION = [("glyph-only (ảnh tương đồng)", 82.5), ("crop-only (crop thật)", 90.2),
            ("combined (hệ thống)", 89.4)]
CD = dict(held_glyph=83.8, held_comb=70.8, ctrl_glyph=88.5, ctrl_comb=95.5, false_off=160, false_on=46)

pages = []

# ============================ PAGE 1: tóm tắt + bối cảnh ============================
p = Page()
p.line("BÁO CÁO NGHIÊN CỨU", F(VIB, 30), color=ACC, gap=2)
p.para("Cải tiến tầng so-khớp-ảnh (S3) bằng tham chiếu đa-nguồn có hiệu chuẩn và "
       "cơ chế từ chối mở (open-set) trong pipeline tự động gán nhãn chữ Nôm",
       F(VIB, 27), gap=10)
p.para("Hệ thống GanNhanOCR — đối chiếu bản Hán-Nôm viết tay với bản dịch Quốc Ngữ "
       "song song để gán nhãn Unicode cho từng ký tự Nôm.", F(VI, 21), color=MUTE, gap=16)
p.rule()

p.line("Tóm tắt", F(VIB, 25), color=ACC)
p.para("Tầng S3 xếp hạng các ký tự ứng viên cho mỗi crop chữ Nôm viết tay bằng cách so "
       "khớp embedding của crop với ảnh tham chiếu, làm tín hiệu thứ ba phá thế khi OCR "
       "và từ điển bất đồng (sinh nhãn tầng SILVER). Báo cáo trình bày bốn cải tiến không "
       "cần huấn luyện lại: (1) mở khóa S3 ở cột lệch số; (2) ngân hàng tham chiếu đa-nguồn "
       "ưu tiên crop thật; (3) hiệu chuẩn xác suất theo từng tầng + ngưỡng theo độ chính xác "
       "mục tiêu kèm từ chối open-set; (4) guard chống thiên lệch miền (crop-bias). Kết quả: "
       f"nhãn dùng được tăng từ 63.428 lên {summ['usable_total']:,}, trong đó tầng SILVER "
       "100% được hậu thuẫn bằng crop thật và hiệu chuẩn tới ~95% (proxy); tầng GOLD bất "
       "biến tuyệt đối. Ảnh tham chiếu tương đồng nhận diện đúng 83,8% ký tự chưa từng "
       "huấn luyện; guard cắt 71% nhãn SILVER sai ở vùng open-set.", F(VI, 21), gap=14)
p.rule()

p.line("1. Bối cảnh & vấn đề", F(VIB, 25), color=ACC)
p.para("• S3 = so khớp crop viết tay với glyph tham chiếu (sinh bởi FontDiffusion theo "
       "phong cách mộc bản) qua một encoder Nôm tự train (ResNet + ArcFace, embedding "
       "256 chiều). DINOv2 zero-shot trước đó không phân biệt được chữ Nôm (retrieval 0%).",
       F(VI, 21))
p.para("• Bốn điểm yếu của S3 cũ: mỗi ứng viên chỉ có MỘT glyph tổng hợp làm tham chiếu "
       "(mong manh, còn khoảng cách miền synthetic↔viết tay); ngưỡng quyết định là hằng số "
       "đặt tay (τ=0,62 / δ=0,06) chưa hiệu chuẩn; S3 bị KHÓA ở các cột lệch số (không bao "
       "giờ chạy); và bị ÉP chọn một ứng viên kể cả khi ký tự đúng không nằm trong danh sách.",
       F(VI, 21))
footer(p, 1); pages.append(p.img)

# ============================ PAGE 2: phương pháp ============================
p = Page()
p.line("2. Phương pháp — bốn cải tiến", F(VIB, 26), color=ACC, gap=12)

def method(title, body):
    p.line(title, F(VIB, 23))
    p.para(body, F(VI, 21), gap=12)

method("Bước 0 — Mở khóa + đồng bộ tiền xử lý",
       "Sửa cờ neo cục bộ (anchored = có hàng xóm đã xác nhận, bỏ ràng buộc tự-xác-nhận) "
       "để S3 được phép chạy trên các cột lệch số — vốn bị chặn hoàn toàn. Cắt sát ink "
       "(tighten_box) crop truy vấn trước khi embed để khớp khung ảnh lúc huấn luyện. Lộ "
       "cosine thô phục vụ hiệu chuẩn. GOLD không đổi (chữ đã xác nhận trả về trước S3).")
method("Bước 1 — Ngân hàng tham chiếu đa-nguồn",
       "Thay 1 glyph tổng hợp bằng ngân hàng tham chiếu theo thứ tự ưu tiên: prototype "
       "từ CROP THẬT của lớp chữ (cùng miền, khoảng cách = 0; 1.578/1.591 lớp có sẵn) → "
       "glyph font-tương-đồng → glyph FontDiffusion (phủ đuôi dài/chữ hiếm). Không train lại.")
method("Bước 2 — Hiệu chuẩn theo tầng + từ chối open-set",
       "Hiệu chuẩn đẳng hướng (isotonic / PAVA) ánh xạ cosine→P(khớp) RIÊNG cho từng tầng, "
       "để điểm của tham chiếu crop và glyph SO SÁNH ĐƯỢC với nhau. Chọn ngưỡng τ tại độ "
       "chính xác mục tiêu trên tập VAL (thay hằng số đặt tay). Cho phép TỪ CHỐI (REVIEW) "
       "khi không ứng viên nào đạt ngưỡng — thay vì ép chọn nhãn sai.")
method("Crop-bias guard — chống thiên lệch miền",
       "Vì cosine crop↔crop cao hơn crop↔glyph, ứng viên có crop dễ thắng oan ký tự đúng "
       "chỉ-có-glyph (chữ hiếm/chưa thấy). Guard: nếu một ứng viên khác thắng winner ở "
       "TẦNG GLYPH quá một biên (0,10 cosine) — tức winner thắng chỉ nhờ crop — thì TỪ CHỐI "
       "(abstain) thay vì khẳng định, bảo vệ vùng open-set.")

p.rule()
p.line("Bất biến cốt lõi & an toàn", F(VIB, 23))
p.para("Mọi thay đổi giữ bất biến 1-chữ-Nôm = 1-âm-tiết, và KHÔNG chạm tầng GOLD (đối "
       "chiếu lớp-từng-dòng: 0 mất, 0 đổi nhãn qua cả bốn cải tiến). Toàn bộ mã đặt trong "
       "evaluation/; tham chiếu = đọc dữ liệu/checkpoint có sẵn, không huấn luyện lại.",
       F(VI, 21))
footer(p, 2); pages.append(p.img)

# ============================ PAGE 3: kết quả tiến triển ============================
p = Page()
p.line("3. Kết quả — tiến triển dataset", F(VIB, 26), color=ACC, gap=12)
table(p, ["Giai đoạn", "GOLD", "SILVER", "SYLLABLE", "REVIEW", "Dùng được"],
      [[s[0], f"{s[1]:,}", f"{s[2]:,}", f"{s[3]:,}", f"{s[4]:,}", f"{s[5]:,}"] for s in STAGES],
      [400, 130, 130, 150, 130, 160], hl_last=True)
p.para("GOLD giữ nguyên 51.195 (an toàn). SILVER (nhãn ký tự do thị giác) tăng nhờ mở khóa "
       "cột lệch + prototype crop thật; guard hạ 479 SILVER không-an-toàn-open-set, trong "
       "đó 245 được cứu lại ở mức âm tiết (SYLLABLE).", F(VI, 21), gap=18)

bar_chart(p, "Nhãn dùng được (GOLD+SILVER+SYLLABLE) qua các bước",
          [(s[0], s[5]) for s in STAGES], vmax=70000, unit="",
          colors=[MUTE, SYL, SYL, ACC], val_fmt="{:,.0f}")
p.rule()
p.line("Phân bố tầng cuối", F(VIB, 23))
total = sum(T.values())
xbar = M; ybar = p.y + 6; bw = W - 2 * M; bh = 56
for tier, col in (("GOLD", GOLD), ("SILVER", SILVER), ("SYLLABLE", SYL), ("REVIEW", REV)):
    ww = int(bw * T[tier] / total)
    p.d.rectangle([xbar, ybar, xbar + ww, ybar + bh], fill=col)
    lbl = f"{tier} {T[tier]:,}"
    lf = F(VIB, 18)
    if ww > tw(p.d, lbl, lf) + 14:
        p.d.text((xbar + 8, ybar + 18), lbl, fill=(255, 255, 255), font=lf)
    elif ww > 40:                       # narrow segment -> tier name only
        p.d.text((xbar + 6, ybar + 18), tier[:3], fill=(255, 255, 255), font=F(VIB, 16))
    xbar += ww
p.y = ybar + bh + 18
# legend (so every tier is named regardless of segment width)
lx = M
for tier, col in (("GOLD", GOLD), ("SILVER", SILVER), ("SYLLABLE", SYL), ("REVIEW", REV)):
    p.d.rectangle([lx, p.y + 2, lx + 20, p.y + 20], fill=col)
    t = f"{tier} {T[tier]:,}"
    p.d.text((lx + 26, p.y), t, fill=INK, font=F(VI, 18))
    lx += 32 + int(tw(p.d, t, F(VI, 18))) + 28
p.y += 34
p.para(f"Tổng {total:,} cặp · {summ['char_classes']:,} lớp ký tự · dùng được "
       f"{summ['usable_total']:,} ({100*summ['usable_total']/total:.0f}%).", F(VI, 21), color=MUTE)
footer(p, 3); pages.append(p.img)

# ============================ PAGE 4: đánh giá tầng so khớp ============================
p = Page()
p.line("4. Đánh giá tầng so-khớp-ảnh", F(VIB, 26), color=ACC, gap=12)

p.line("4.1 Hiệu chuẩn theo tầng (đo trên VAL)", F(VIB, 23))
ct = cal["tiers"]
table(p, ["Tầng tham chiếu", "P(khớp) tối đa", "Diễn giải"],
      [["crop (crop thật)", f"{max(ct['crop']['p']):.2f}", "cùng miền — phân biệt mạnh"],
       ["fd (ảnh tương đồng)", f"{max(ct['fd']['p']):.2f}", "1 glyph — trần thấp, làm dự phòng"]],
      [330, 230, 480])
p.para(f"Điểm vận hành: τ_p={cal['tau_p']} → precision {cal['measured_precision']:.1%}, "
       f"coverage {cal['coverage']:.1%}; retrieval@1 = {cal['retrieval_at_1']:.1%} trên tập "
       "ứng viên thật. Vì glyph trần ~0,49 < τ=0,50, SILVER trên thực tế YÊU CẦU crop thật "
       "hậu thuẫn (bằng chứng thật mới override OCR).", F(VI, 21), gap=16)

p.line("4.2 Ablation: từng nguồn tham chiếu đóng góp gì (retrieval@1, VAL)", F(VIB, 23))
bar_chart(p, "", ABLATION, vmax=100, colors=[REV, SYL, ACC])
p.para("Ảnh tương đồng MỘT MÌNH đạt 82,5% (DINOv2 xưa ~0%); crop thật 90,2%; kết hợp 89,4% "
       "≈ trần crop. Vì 99,2% lớp chữ đã có crop, ảnh tương đồng đóng vai dự phòng + miền "
       "hiệu chuẩn.", F(VI, 20), gap=14)

p.line("4.3 Char-disjoint: ký tự CHƯA huấn luyện & crop-bias guard", F(VIB, 23))
table(p, ["Chỉ số", "glyph-only", "combined"],
      [["Held-out (chữ unseen, không crop)", f"{CD['held_glyph']:.1f}%", f"{CD['held_comb']:.1f}%"],
       ["Control (giữ crop)", f"{CD['ctrl_glyph']:.1f}%", f"{CD['ctrl_comb']:.1f}%"]],
      [560, 220, 220])
p.para(f"Ảnh tương đồng nhận diện chữ CHƯA THẤY ở 83,8%. Guard cắt nhãn SILVER SAI ở vùng "
       f"open-set từ {CD['false_off']} xuống {CD['false_on']} (−71%), đẩy chúng sang REVIEW. "
       "(Caveat: backbone đã thấy các chữ này → cận trên của zero-shot thật.)", F(VI, 20))
footer(p, 4); pages.append(p.img)

# ============================ PAGE 5: minh hoạ + hạn chế + hướng tiếp ============================
p = Page()
p.line("5. Minh hoạ trực quan (crop viết tay ↔ glyph tương đồng)", F(VIB, 24), color=ACC, gap=12)
# montage from eval_sample
imgs = sorted((SAMPLE / "imgs").glob("S*_crop.png"))[:6] if (SAMPLE / "imgs").exists() else []
if imgs:
    cell = 150; pad = 26; x0 = M; y0 = p.y
    vrows = list(csv.DictReader(open(SAMPLE / "verify.csv", encoding="utf-8"))) if (SAMPLE / "verify.csv").exists() else []
    vmap = {r["sample_id"]: r for r in vrows}
    for i, cp in enumerate(imgs):
        sid = cp.name.split("_")[0]
        col = i % 3; rw = i // 3
        cx = x0 + col * (cell * 2 + pad + 40); cy = y0 + rw * (cell + 70)
        try:
            ci = Image.open(cp).convert("RGB").resize((cell, cell))
            p.img.paste(ci, (cx, cy))
        except Exception:
            pass
        rp = SAMPLE / "imgs" / f"{sid}_ref.png"
        if rp.exists():
            try:
                ri = Image.open(rp).convert("RGB").resize((cell, cell))
                p.img.paste(ri, (cx + cell + 8, cy))
            except Exception:
                pass
        r = vmap.get(sid, {})
        tier_txt = f"{r.get('tier','')} · "
        p.d.text((cx, cy + cell + 6), tier_txt, fill=MUTE, font=F(VI, 18))
        lab = r.get("label", "") or r.get("syllable", "")
        xoff = cx + int(tw(p.d, tier_txt, F(VI, 18)))
        # Nôm char needs NomNaTong (Arial -> tofu); syllable (Latin) stays Arial
        is_nom = bool(lab) and len(lab) == 1 and ord(lab) > 0x2E80
        p.d.text((xoff, cy + cell + (2 if is_nom else 6)), lab, fill=INK,
                 font=F(NOM, 24) if is_nom else F(VI, 18))
    p.y = y0 + 2 * (cell + 70) + 10
p.para("Trái = crop mộc bản thật · Phải = glyph tham chiếu tương đồng. Trích từ gói soi tay "
       "430 mẫu (eval_sample/) dùng để đo precision thật.", F(VI, 19), color=MUTE, gap=14)
p.rule()

p.line("6. Hạn chế & hướng tiếp", F(VIB, 24), color=ACC)
p.para("• Precision ~95% là PROXY (đo trên crop GOLD = vùng dễ); con số chốt cần người gán "
       "tay ~430 mẫu SILVER-eligible (công cụ export_eval_sample.py + measure_precision.py "
       "đã sẵn, có Wilson CI).", F(VI, 21))
p.para("• Char-disjoint 83,8% là cận trên (backbone đã thấy chữ); số zero-shot thật cần "
       "huấn luyện lại loại-lớp trên GPU — harness đã sẵn, chỉ thay checkpoint.", F(VI, 21))
p.para("• Tham chiếu crop tối ưu trong-corpus (near-duplicate cùng sách); khi khẳng định "
       "khái quát sang sách mới cần báo cáo riêng same/cross-book.", F(VI, 21))
p.para("• Hướng tiếp: chốt precision Bước 3; char-disjoint thật; chỉnh glyph_guard_margin "
       "theo cân bằng an-toàn/recall.", F(VI, 21), gap=14)
p.rule()
p.line("Phụ lục — tái lập (mã trong evaluation/ver_new/)", F(VIB, 21))
for cmd in ["calibrate_s3.py          # hiệu chuẩn τ + per-tier isotonic",
            "build_dataset.py --use-s3 # sinh dataset (4 cải tiến + guard)",
            "ablate_s3_refs.py         # ablation glyph/crop/combined",
            "eval_char_disjoint.py     # char-disjoint + guard off/on",
            "export_eval_sample.py     # gói 430 mẫu soi tay",
            "measure_precision.py      # precision + Wilson CI"]:
    p.line("  " + cmd, F(VI, 19), color=INK, gap=2)
footer(p, 5); pages.append(p.img)

# This Pillow build lacks the JPEG SAVE plugin (PDF defaults to DCTDecode for
# RGB/L). Palette mode "P" uses FlateDecode -> saves without JPEG, fine for a
# text+chart report with grayscale crops.
pal = [pg.convert("P", palette=Image.ADAPTIVE, colors=256) for pg in pages]
pal[0].save(OUT, save_all=True, append_images=pal[1:], resolution=150.0)
print(f"PDF -> {OUT}  ({len(pages)} trang, {OUT.stat().st_size // 1024} KB)")
