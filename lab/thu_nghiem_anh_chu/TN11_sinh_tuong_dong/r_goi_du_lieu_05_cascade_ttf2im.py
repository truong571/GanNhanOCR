"""r_goi_du_lieu_05_cascade_ttf2im.py — HƯỚNG 5: RỦI RO TÍNH NĂNG — chuỗi phông dự phòng (cascade) và hàm ttf2im thật của gói.

Dùng CHÍNH src/tools/utils.py của gói (nạp từ code/font_diffusion/src/tools/utils.py bằng importlib, không nạp gói `src` để tránh tác dụng phụ), chính phông trong gói, pygame.freetype cỡ 128 như
FontManager.get_font -> load_ttf. Trên MỌI cặp (phông, chữ) của kế hoạch (51.730 chữ-cuốn = 10.200 cặp khác nhau):
  5a  ttf2im trả None / ném ngoại lệ / ảnh rỗng (toàn 255) / trùng ảnh .notdef của chính phông (pygame);
  5b  số đo ảnh nội dung theo phông: mực %, hộp mực (rộng, cao), độ dày nét (2·diện tích/chu vi), kích thước bề mặt pygame (cắt nhỏ khi > 128);
  5c  SO SÁNH THEO CẶP trên chữ chung: phông dự phòng so với NomNaTong (tỉ số mực, hộp, độ dày nét, IoU hình dạng) — kích thước/nền/độ đậm khác thế nào;
  5d  ca đặc biệt: chữ không có glyph (ttf2im có trả None không?), chữ stt11 bị bỏ, chữ PUA mặt phẳng 15.
Môi trường: pygame-ce trên Mac (khác pygame 2.6.1/SDL 2.28 trên Kaggle) — kết quả về rỗng/None là dấu hiệu, KHÔNG thay thế phép thử trên Kaggle.

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_05_cascade_ttf2im.py
Ra: 05_cascade_ttf2im.json.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import hashlib
import importlib.util
import io
import json
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402

import cv2  # noqa: E402


def load_fd_utils():
    p = Lb.PACK / "code" / "font_diffusion" / "src" / "tools" / "utils.py"
    spec = importlib.util.spec_from_file_location("fd_utils_goi", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, p


def img_stats(im, surf_size):
    """im: PIL RGB 128x128 từ ttf2im. Trả số đo trên kênh 0."""
    a = np.asarray(im)[:, :, 0]
    ink = a < 128
    n = int(ink.sum())
    d = dict(blank=(n == 0 and bool((a == 255).all())), ink_frac=round(float(ink.mean()), 5), n_ink=n, surf_w=int(surf_size[0]), surf_h=int(surf_size[1]),
             scaled=bool(max(surf_size) > 128), bg=int(np.median(a)))
    if n:
        ys, xs = np.nonzero(ink)
        d.update(bw=int(xs.max() - xs.min() + 1), bh=int(ys.max() - ys.min() + 1), cx=float(xs.mean()), cy=float(ys.mean()))
        cs, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        per = float(sum(cv2.arcLength(c, True) for c in cs))
        d["sw"] = round(2.0 * n / per, 3) if per > 0 else None
    else:
        d.update(bw=0, bh=0, cx=None, cy=None, sw=None)
    return d


def q(x, p):
    x = np.asarray([v for v in x if v is not None], float)
    return round(float(np.percentile(x, p)), 3) if len(x) else None


def main():
    t0 = time.time()
    inv = Lb.Inv()
    res: dict = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"))
    U, upath = load_fd_utils()
    import pygame
    res["moi_truong"] = dict(pygame=pygame.version.ver, pygame_ce=bool(getattr(pygame, "IS_CE", False)), sdl=".".join(map(str, pygame.get_sdl_version())), freetype=".".join(map(str, pygame.freetype.get_version())) if hasattr(pygame, "freetype") else None,
                             python=sys.version.split()[0], ttf2im_tep=str(upath.relative_to(Lb.REPO)), ghi_chu="Kaggle lần trước: pygame 2.6.1, SDL 2.28.4, Python 3.13.15 (measure_out/_tn11/kaggle_out/log_B34.txt)")
    plan = json.loads((Lb.PACK / "plan.json").read_text(encoding="utf-8"))
    fonts_json = json.loads((Lb.PACK / "fonts.json").read_text(encoding="utf-8"))
    fdir = Lb.PACK / "code" / "font_diffusion" / "fonts"
    fnames = plan["font_names"]
    pairs = defaultdict(set)
    for b, v in plan["books"].items():
        for c, i in zip(v["chars"], v["font_idx"]):
            pairs[(fnames[i], c)].add(b)
    fonts = {n: U.load_ttf(str(fdir / fonts_json[n])) for n in fnames}
    cm = {n: Lb.cmap_union(fdir / fonts_json[n]) for n in fnames}
    # ---- chữ ký .notdef của pygame: render mã không có trong cmap
    sig = {}
    for n in fnames:
        sg = {}
        for cp in (0xFFFF, 0x2FFFE, 0x10FFFE, 0x378, 0xFFFE, 0x2B81F):
            if cp in cm[n]:
                continue
            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    im = U.ttf2im(font=fonts[n], char=chr(cp))
                    sg[cp] = None if im is None else ("blank" if (np.asarray(im)[:, :, 0] == 255).all() else hashlib.md5(np.asarray(im).tobytes()).hexdigest())
                except Exception as e:  # noqa: BLE001
                    sg[cp] = f"EXC:{type(e).__name__}"
        sig[n] = sg
    res["5d_ket_qua_ttf2im_voi_ma_khong_co_glyph"] = {n: {hex(k): v for k, v in s.items()} for n, s in sig.items()}
    # ---- 5a/5b mọi cặp
    rows = {}
    none_pairs, exc_pairs, blank_pairs, notdef_pairs = [], [], [], []
    for (n, c), bks in sorted(pairs.items(), key=lambda x: (x[0][0], ord(x[0][1]))):
        font = fonts[n]
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                im = U.ttf2im(font=font, char=c)
        except Exception as e:  # noqa: BLE001
            exc_pairs.append((n, c, f"{type(e).__name__}: {str(e)[:60]}")); continue
        if im is None:
            none_pairs.append((n, c)); continue
        try:
            sz = font.render(c)[0].get_size()
        except Exception:  # noqa: BLE001
            sz = (0, 0)
        st = img_stats(im, sz)
        h = hashlib.md5(np.asarray(im).tobytes()).hexdigest()
        if st["blank"]:
            blank_pairs.append((n, c))
        if h in {v for v in sig[n].values() if isinstance(v, str) and v not in ("blank",) and not v.startswith("EXC")}:
            notdef_pairs.append((n, c))
        st["hash"] = h
        rows[(n, c)] = st
    res["5a_tong"] = dict(cap=len(pairs), cuon_chu=sum(len(v) for v in pairs.values()), ttf2im_tra_None=len(none_pairs), nem_ngoai_le=len(exc_pairs), anh_rong=len(blank_pairs), trung_notdef=len(notdef_pairs),
                          chu_cuon_bi_anh_huong=int(sum(len(pairs[(n, c)]) for n, c in none_pairs + blank_pairs + notdef_pairs) + sum(len(pairs[(n, c)]) for n, c, _ in exc_pairs)))
    res["5a_vi_du"] = dict(none=[f"{n}:U+{ord(c):05X}" for n, c in none_pairs[:15]], ngoai_le=[f"{n}:U+{ord(c):05X}:{e}" for n, c, e in exc_pairs[:15]], rong=[f"{n}:U+{ord(c):05X}" for n, c in blank_pairs[:15]],
                           notdef=[f"{n}:U+{ord(c):05X}" for n, c in notdef_pairs[:15]])
    inv.check("5a_ttf2im_khong_tra_None_tren_10200_cap", not none_pairs, len(none_pairs))
    inv.check("5a_ttf2im_khong_nem_ngoai_le_(RealGen_chi_bat_None)", not exc_pairs, [f"{n}:U+{ord(c):05X}" for n, c, _ in exc_pairs[:5]])
    inv.check("5a_khong_anh_rong_(toan_255)", not blank_pairs, len(blank_pairs))
    inv.check("5a_khong_anh_trung_.notdef_cua_chinh_phong", not notdef_pairs, len(notdef_pairs))
    # ---- 5b theo phông
    byf = defaultdict(list)
    for (n, c), st in rows.items():
        byf[n].append((c, st))
    perfont = {}
    for n in fnames:
        L = [st for _, st in byf[n]]
        perfont[n] = dict(n_cap=len(L), ink_frac_trung_vi=q([s["ink_frac"] for s in L], 50), ink_frac_p5=q([s["ink_frac"] for s in L], 5), ink_frac_p95=q([s["ink_frac"] for s in L], 95),
                          bbox_w_trung_vi=q([s["bw"] for s in L], 50), bbox_h_trung_vi=q([s["bh"] for s in L], 50), bbox_w_p5=q([s["bw"] for s in L], 5), bbox_h_p5=q([s["bh"] for s in L], 5),
                          nen_trung_vi=q([s["bg"] for s in L], 50), do_day_net_trung_vi=q([s["sw"] for s in L], 50), do_day_net_p5=q([s["sw"] for s in L], 5), do_day_net_p95=q([s["sw"] for s in L], 95),
                          be_mat_pygame_rong_trung_vi=q([s["surf_w"] for s in L], 50), be_mat_pygame_cao_trung_vi=q([s["surf_h"] for s in L], 50), so_anh_bi_thu_nho_do_be_mat_lon_hon_128=int(sum(s["scaled"] for s in L)),
                          so_cap_hop_muc_cham_vien=None)
    res["5b_theo_phong"] = perfont
    # ---- 5c so sánh theo cặp trên chữ chung với NomNaTong (chữ có trong cmap cả hai; lấy từ chữ kế hoạch gán NomNaTong để chắc chắn là chữ thật của gói)
    rng = np.random.default_rng(20261004)
    base_chars = [c for c, _ in byf["NomNaTong"]]
    paired = {}
    for n in fnames:
        if n == "NomNaTong":
            continue
        cand = sorted((c for c in base_chars if ord(c) in cm[n] and not Lb.is_pua(c)), key=ord)
        if not cand:
            paired[n] = dict(n_chu_chung=0); continue
        pick = [cand[i] for i in sorted(rng.choice(len(cand), size=min(700, len(cand)), replace=False))]
        ra, rb, rw, rh, ious, dx, dy = [], [], [], [], [], [], []
        n_ok = 0
        for c in pick:
            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    i1, i2 = U.ttf2im(font=fonts["NomNaTong"], char=c), U.ttf2im(font=fonts[n], char=c)
                except Exception:  # noqa: BLE001
                    continue
            if i1 is None or i2 is None:
                continue
            a1, a2 = np.asarray(i1)[:, :, 0] < 128, np.asarray(i2)[:, :, 0] < 128
            if a1.sum() == 0 or a2.sum() == 0:
                continue
            n_ok += 1
            ra.append(a2.mean() / a1.mean())
            y1, x1 = np.nonzero(a1); y2, x2 = np.nonzero(a2)
            rw.append((x2.max() - x2.min() + 1) / (x1.max() - x1.min() + 1)); rh.append((y2.max() - y2.min() + 1) / (y1.max() - y1.min() + 1))
            ious.append(float((a1 & a2).sum() / (a1 | a2).sum()))
            dx.append(float(x2.mean() - x1.mean())); dy.append(float(y2.mean() - y1.mean()))
        pc = [rows[(n2, c2)] for (n2, c2) in rows if n2 == n]
        paired[n] = dict(n_chu_chung=n_ok, ti_so_muc_trung_vi=q(ra, 50), ti_so_muc_p5=q(ra, 5), ti_so_muc_p95=q(ra, 95), ti_so_hop_rong_trung_vi=q(rw, 50), ti_so_hop_cao_trung_vi=q(rh, 50),
                         IoU_hinh_dang_trung_vi=q(ious, 50), IoU_p10=q(ious, 10), IoU_p90=q(ious, 90), lech_tam_x_trung_vi_px=q(dx, 50), lech_tam_y_trung_vi_px=q(dy, 50),
                         ghi_chu="IoU thấp là bình thường giữa hai kiểu chữ khác nhau; so với đối chứng NomNaTong vs NomNaTongLight2 bên dưới")
    # đối chứng: NomNaTong vs chính nó khác cỡ nét (Light2) cho mốc "cùng họ"
    res["5c_so_voi_NomNaTong_tren_chu_chung"] = paired
    # đối chứng ngẫu nhiên: IoU giữa NomNaTong của chữ A và NomNaTong của chữ B (khác chữ) — mốc "khác hẳn"
    perm = rng.permutation(len(base_chars))[:700]
    ctrl = []
    for i, j in zip(perm[:350], perm[350:700]):
        a1 = np.asarray(U.ttf2im(font=fonts["NomNaTong"], char=base_chars[int(i)]))[:, :, 0] < 128
        a2 = np.asarray(U.ttf2im(font=fonts["NomNaTong"], char=base_chars[int(j)]))[:, :, 0] < 128
        ctrl.append(float((a1 & a2).sum() / max((a1 | a2).sum(), 1)))
    res["5c_doi_chung_IoU_hai_chu_khac_nhau_cung_phong"] = dict(trung_vi=q(ctrl, 50), p90=q(ctrl, 90))
    # ---- phân bố chữ thật của gói: chữ gán phông dự phòng so với chữ gán NomNaTong (khác chữ => chỉ để đo dịch chuyển miền)
    base_rows = [st for _, st in byf["NomNaTong"]]
    dom = {}
    for n in fnames:
        if n == "NomNaTong":
            continue
        L = [st for _, st in byf[n]]
        if not L:
            continue
        lo, hi = q([s["ink_frac"] for s in base_rows], 5), q([s["ink_frac"] for s in base_rows], 95)
        wlo, whi = q([s["sw"] for s in base_rows], 5), q([s["sw"] for s in base_rows], 95)
        dom[n] = dict(n_cap=len(L), ink_frac_trung_vi=q([s["ink_frac"] for s in L], 50), ink_ngoai_p5_p95_cua_NomNaTong_pct=round(100 * float(np.mean([(s["ink_frac"] < lo or s["ink_frac"] > hi) for s in L])), 1),
                      do_day_net_trung_vi=q([s["sw"] for s in L], 50), net_ngoai_p5_p95_cua_NomNaTong_pct=round(100 * float(np.mean([(s["sw"] is not None and (s["sw"] < wlo or s["sw"] > whi)) for s in L])), 1),
                      bbox_h_trung_vi=q([s["bh"] for s in L], 50), bbox_w_trung_vi=q([s["bw"] for s in L], 50))
    dom["_NomNaTong_mốc"] = dict(ink_frac_trung_vi=q([s["ink_frac"] for s in base_rows], 50), ink_frac_p5_p95=[q([s["ink_frac"] for s in base_rows], 5), q([s["ink_frac"] for s in base_rows], 95)],
                                do_day_net_p5_p95=[q([s["sw"] for s in base_rows], 5), q([s["sw"] for s in base_rows], 95)], do_day_net_trung_vi=q([s["sw"] for s in base_rows], 50),
                                bbox_h_trung_vi=q([s["bh"] for s in base_rows], 50), bbox_w_trung_vi=q([s["bw"] for s in base_rows], 50))
    res["5c_dich_chuyen_mien_chu_that_cua_goi"] = dom
    # tỉ lệ chữ-cuốn và ảnh dùng phông dự phòng
    tot_bc = sum(len(v) for v in pairs.values())
    fb = {n: int(sum(len(v) for (m, c), v in pairs.items() if m == n)) for n in fnames}
    res["5c_ti_le_chu_cuon_theo_phong"] = dict(fb, tong=tot_bc, du_phong_pct=round(100 * (tot_bc - fb["NomNaTong"]) / tot_bc, 2),
                                              HanNomA_pct=round(100 * fb["HanNomA"] / tot_bc, 2), HanaMinB_pct=round(100 * fb["HanaMinB"] / tot_bc, 2),
                                              theo_cuon={b: round(100 * sum(1 for i in plan["books"][b]["font_idx"] if fnames[i] != "NomNaTong") / len(plan["books"][b]["chars"]), 2) for b in Lb.ORDER})
    # ---- 5d ca đặc biệt
    sp = {}
    for b in Lb.ORDER:
        for c in plan["books"][b]["uncovered"]:
            for n in ("NomNaTong", "PlangothicP2"):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    try:
                        im = U.ttf2im(font=fonts[n], char=c)
                        r = None if im is None else dict(blank=bool((np.asarray(im)[:, :, 0] == 255).all()), ink=float((np.asarray(im)[:, :, 0] < 128).mean()))
                    except Exception as e:  # noqa: BLE001
                        r = f"EXC:{type(e).__name__}"
                sp[f"{b}:U+{ord(c):05X}@{n}(co_trong_cmap={ord(c) in cm[n]})"] = r
    res["5d_chu_bi_bo_(uncovered)"] = sp
    pua15 = [c for (n, c) in rows if Lb.is_pua(c)]
    res["5d_PUA_mat_phang_15"] = dict(so_cap=len(pua15), anh_rong=sum(1 for (n, c) in blank_pairs if Lb.is_pua(c)), none=sum(1 for (n, c) in none_pairs if Lb.is_pua(c)),
                                      ink_frac_trung_vi=q([rows[(n, c)]["ink_frac"] for (n, c) in rows if Lb.is_pua(c)], 50))
    # so với PIL: pygame rỗng nhưng PIL không (và ngược lại)? (script 03 đã vẽ PIL: 0 rỗng)
    res["giay"] = round(time.time() - t0, 1)
    # ---- bất biến phụ: ảnh hợp lệ
    inv.check("5b_moi_anh_nen_trang_(trung_vi_nen==255)_va_co_muc", all(st["bg"] == 255 and st["n_ink"] > 0 for st in rows.values()), sum(1 for st in rows.values() if st["bg"] != 255 or st["n_ink"] == 0))
    inv.check("5b_moi_anh_hop_muc_>=_10px_moi_chieu (không là nét li ti)", all(st["bw"] >= 10 and st["bh"] >= 10 for st in rows.values()), [f"{n}:U+{ord(c):05X}:{st['bw']}x{st['bh']}" for (n, c), st in rows.items() if st["bw"] < 10 or st["bh"] < 10][:10])
    res["bat_bien"] = inv.summary(); res["bat_bien_chi_tiet"] = inv.rows
    Lb.jdump(res, "05_cascade_ttf2im.json")
    print(f"[05] xong {res['giay']}s; bất biến {res['bat_bien']['dat']}/{res['bat_bien']['tong']} rớt={res['bat_bien']['rot']}")


if __name__ == "__main__":
    main()
