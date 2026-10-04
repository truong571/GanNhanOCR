"""BƯỚC 3b — chuỗi phông dự phòng với MÔ HÌNH THẬT trên MPS, qua đúng lớp RealGen của gói (import w_runner_khung_chay):
  - nạp RealGen như worker (cổng use_fst, setup_real copy code+ckpt vào cascade_tmp, kiểm băm ckpt);
  - sinh 3 chữ THẬT của kế hoạch cho MỖI phông (NomNaTong, HanaMinB, HanNomA, HanaMinA, PlangothicP2, NomNaTongLight2; theo font_idx;
    phông có ít hơn 3 chữ trong kế hoạch thì dùng hết) bằng ảnh phong cách một cuốn, trong MỘT lô nhiều phông (đúng đường worker_loop);
  - với mỗi phông dự phòng, 3 chữ "cặp" (có glyph ở cả NomNaTong và phông đó): sinh bằng nội dung phông dự phòng và bằng nội dung NomNaTong
    (CÙNG hạt giống, lô 1 ảnh) -> độ giống (IoU mực) so với nền "cùng chữ, NomNaTong, hai hạt giống khác nhau";
  - mực %, bbox, số thành phần liên thông, rỗng/trắng tinh; skip render_none; FontManager tên có dấu cách (HAN NOM A).
Chạy từ cwd = review/chay_thu_dau_cuoi/cascade_cwd. Ra: 04_font_cascade.json + 04_cascade_contact.png (ảnh tổng hợp để người xem; KHÔNG dùng làm số đo)
    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/.../r_chay_thu_dau_cuoi_04_font_cascade.py [--device mps] [--paired 3]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import argparse
import io
import json
import os
import time
from pathlib import Path

import r_chay_thu_dau_cuoi_lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps")
ap.add_argument("--paired", type=int, default=3)
ap.add_argument("--book", default="B34")
ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()

CW = L.R / "cascade_cwd"
CW.mkdir(parents=True, exist_ok=True)
os.chdir(CW)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

K = L.load_runner()
rep = L.Rep("04_font_cascade")
PACK = L.PACK
plan = K.load_plan(PACK)
TMP = L.R / "cascade_tmp"
t0 = time.time()
man = K.setup_real(PACK, TMP)
rep.data("setup_real_giay", round(time.time() - t0, 1))
fmap = json.loads((PACK / "fonts.json").read_text(encoding="utf-8"))
fpaths = {n: str(TMP / "font_diffusion" / "fonts" / f) for n, f in fmap.items()}
t0 = time.time()
gen = K.RealGen(TMP, TMP / "font_diffusion" / "ckpt" / "PROD", fpaths, a.device, a.batch, cache_dir=TMP / "_unused_cache")
rep.data("nap_RealGen_giay", round(time.time() - t0, 1))
rep.inv("C0.cong_use_fst_qua_va_khong_canh_bao_thieu_ckpt", gen.guard["ok"] and gen.guard["model_class"] == "FontDiffuserModelDPM" and not gen.guard["fst_attributes"] and not gen.guard["warnings"], gen.guard)

# phông -> chữ của kế hoạch (theo font_idx), phân biệt
names = plan["font_names"]
by_font = {n: [] for n in names}
seen = set()
for b, v in plan["books"].items():
    for c, i in zip(v["chars"], v["font_idx"]):
        if c not in by_font[names[i]]:
            by_font[names[i]].append(c)
cm = {}
from fontTools.ttLib import TTFont  # noqa: E402

for n, p in fpaths.items():
    f = TTFont(p, lazy=True); s = set()
    for st in f["cmap"].tables:
        s |= set(st.cmap)
    cm[n] = s
style = PACK / "data" / a.book / "style_0.png"
task = dict(book=a.book, model="base", style=0)


def decode(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)).convert("L"))


def stats(b: bytes) -> dict:
    im = Image.open(io.BytesIO(b))
    g = np.array(im.convert("L"))
    ink = g < 128
    import cv2
    n_cc, _, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(ink)
    bb = (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) if len(xs) else (0, 0)
    return dict(size=im.size, mode=im.mode, ink_pct=round(float(ink.mean() * 100), 1), bbox=bb, cc=int(sum(st[i, cv2.CC_STAT_AREA] >= 6 for i in range(1, n_cc))), trang_tinh=bool(g.min() > 240))


def iou(x: np.ndarray, y: np.ndarray) -> float:
    A, B = x < 128, y < 128
    u = (A | B).sum()
    return float((A & B).sum() / u) if u else 1.0


# ---- (1) 3 chữ kế hoạch cho mỗi phông, MỘT lô nhiều phông
chars, fonts, tag = [], [], []
for n in names:
    for c in by_font[n][:3]:
        chars.append(c); fonts.append(n); tag.append(n)
rep.data("chu_ke_hoach_theo_phong", {n: [f"U+{ord(c):04X}" for c in by_font[n][:3]] for n in names})
rep.data("so_chu_ke_hoach_moi_phong_trong_ca_goi", {n: len(by_font[n]) for n in names})
t0 = time.time()
imgs, sk = gen.generate(task, chars, fonts, style, K.seed_of("review/cascade/1"))
dt1 = time.time() - t0
rep.data("lo_nhieu_phong", dict(n=len(chars), giay=round(dt1, 1), giay_moi_anh=round(dt1 / max(1, len(chars)), 2), bo_qua=sk))
per = []
ok_all = True
for c, n in zip(chars, fonts):
    if c not in imgs:
        per.append(dict(chu=f"U+{ord(c):04X}", phong=n, loi="khong_sinh_duoc")); ok_all = False; continue
    s = stats(imgs[c]); s.update(chu=f"U+{ord(c):04X}", phong=n)
    per.append(s)
    ok_all &= s["size"] == (96, 96) and s["mode"] == "RGB" and not s["trang_tinh"] and 3 <= s["ink_pct"] <= 45 and s["cc"] >= 1
rep.data("anh_ke_hoach", per)
rep.inv("C1.moi_phong_sinh_dung_96x96_RGB_khong_trang_muc_hop_ly", ok_all and not sk and len(imgs) == len(chars), dict(n=len(imgs), bo_qua=sk, tom_tat={p["phong"]: p["ink_pct"] for p in per if "ink_pct" in p}))
rep.inv("C2.phong_ten_co_dau_cach_(HAN NOM A)_dung_duoc", any(p["phong"] == "HanNomA" and "ink_pct" in p for p in per), [p for p in per if p["phong"] == "HanNomA"][:3])
# nền "bình thường" theo NomNaTong (30 chữ NomNaTong của kế hoạch, lô 1)
ref_chars = by_font["NomNaTong"][100:130]
t0 = time.time()
ref_imgs, ref_sk = gen.generate(task, ref_chars, ["NomNaTong"] * len(ref_chars), style, K.seed_of("review/cascade/ref"))
dt_ref = time.time() - t0
ref_ink = np.array([stats(ref_imgs[c])["ink_pct"] for c in ref_chars if c in ref_imgs])
lo, hi = np.percentile(ref_ink, [5, 95])
rep.data("nen_NomNaTong_30_chu", dict(ink_p5=round(float(lo), 1), ink_tv=round(float(np.median(ref_ink)), 1), ink_p95=round(float(hi), 1), giay=round(dt_ref, 1), giay_moi_anh=round(dt_ref / len(ref_chars), 2), bo_qua=ref_sk))
fb_in_band = {p["phong"]: bool(lo * 0.5 <= p["ink_pct"] <= hi * 1.6) for p in per if "ink_pct" in p}
rep.inv("C3.muc_phong_du_phong_nam_trong_bien_do_0,5xp5..1,6xp95_cua_NomNaTong", all(fb_in_band.values()), dict(band=[round(float(lo * 0.5), 1), round(float(hi * 1.6), 1)], theo_phong=fb_in_band))

# ---- (2) chữ "cặp": nội dung phông dự phòng vs NomNaTong, cùng hạt giống, lô 1
pool = [c for c in by_font["NomNaTong"][200:3000]]
pair_res, contact = {}, []
S0, S1 = K.seed_of("review/pair/S0"), K.seed_of("review/pair/S1")
for n in [x for x in names if x != "NomNaTong"]:
    cand = [c for c in pool if ord(c) in cm[n]][: a.paired]
    rows = []
    for c in cand:
        g_f, _ = gen.generate(task, [c], [n], style, S0)
        g_n0, _ = gen.generate(task, [c], ["NomNaTong"], style, S0)
        g_n1, _ = gen.generate(task, [c], ["NomNaTong"], style, S1)
        if not (c in g_f and c in g_n0 and c in g_n1):
            rows.append(dict(chu=f"U+{ord(c):04X}", loi="thieu_anh")); continue
        A, B0, B1 = decode(g_f[c]), decode(g_n0[c]), decode(g_n1[c])
        rows.append(dict(chu=f"U+{ord(c):04X}", iou_phong_vs_NomNaTong_cung_seed=round(iou(A, B0), 3), iou_NomNaTong_hai_seed=round(iou(B0, B1), 3),
                         ink_phong=round(float((A < 128).mean() * 100), 1), ink_NomNaTong=round(float((B0 < 128).mean() * 100), 1)))
        contact.append((n, c, g_f[c], g_n0[c]))
    pair_res[n] = rows
rep.data("cap_phong_vs_NomNaTong", pair_res)
allr = [r for rows in pair_res.values() for r in rows if "iou_phong_vs_NomNaTong_cung_seed" in r]
if allr:
    m_font = float(np.mean([r["iou_phong_vs_NomNaTong_cung_seed"] for r in allr]))
    m_seed = float(np.mean([r["iou_NomNaTong_hai_seed"] for r in allr]))
    rep.data("iou_tb", dict(phong_vs_NomNaTong_cung_seed=round(m_font, 3), NomNaTong_hai_seed_khac_nhau=round(m_seed, 3), so_cap=len(allr)))
    # KHÔNG đặt ngưỡng đạt/trượt: IoU mực giữa hai ảnh sinh chỉ là thước thô (không phải phép nhận diện chữ). Ghi số liệu để người đọc tự đánh giá; xem thêm 04_cascade_contact.png.
    rep.data("C4_thong_tin_IoU", dict(iou_phong_vs_NomNaTong_cung_seed=round(m_font, 3), iou_NomNaTong_hai_seed_khac_nhau=round(m_seed, 3), ti_le=round(m_font / m_seed, 2), n=len(allr),
                                     ghi_chu="ti_le = độ giống (IoU mực) khi ĐỔI PHÔNG nội dung so với khi chỉ đổi hạt giống; không có phép nhận diện chữ nên chưa khẳng định nhận dạng đúng chữ"))
    rep.inv("C4.moi_cap_phong_du_phong_vs_NomNaTong_sinh_duoc_va_IoU_>0_(thong_tin,_khong_phai_nguong_chat_luong)", all(r["iou_phong_vs_NomNaTong_cung_seed"] > 0 for r in allr) and len(allr) == 15, dict(n=len(allr), iou_tb=round(m_font, 3), iou_hai_seed=round(m_seed, 3)))
# ---- (3) tạo ảnh tổng hợp cho người xem
tile = 96
rows_img = []
for n, c, gf, gn in contact:
    ct_f = Image.fromarray(np.full((tile, tile), 255, np.uint8))
    rows_img.append((n, c, gf, gn))
W = tile * 4 + 8 * 3
sheet = Image.new("RGB", (W, (tile + 6) * max(1, len(rows_img))), (255, 255, 255))
for i, (n, c, gf, gn) in enumerate(rows_img):
    from src.tools.utils import ttf2im  # noqa: E402
    cf = ttf2im(font=gen._font(n), char=c).resize((tile, tile))
    cn = ttf2im(font=gen._font("NomNaTong"), char=c).resize((tile, tile))
    for j, im in enumerate((cf, Image.open(io.BytesIO(gf)).convert("RGB"), cn, Image.open(io.BytesIO(gn)).convert("RGB"))):
        sheet.paste(im, (j * (tile + 8), i * (tile + 6)))
sheet.save(L.R / "04_cascade_contact.png")
rep.data("anh_tong_hop", "04_cascade_contact.png: mỗi hàng = [nội dung phông dự phòng | sinh từ nó | nội dung NomNaTong | sinh từ NomNaTong]; thứ tự hàng theo phông trong cap_phong_vs_NomNaTong")
rep.data("thu_tu_hang_anh_tong_hop", [(n, f"U+{ord(c):04X}") for n, c, _, _ in rows_img])
rep.save()
