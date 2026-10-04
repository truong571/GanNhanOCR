"""r_goi_du_lieu_03_chu_va_phong.py — HƯỚNG 3: DANH SÁCH CHỮ và GÁN PHÔNG của gói Kaggle.

  3a  chars_<cuốn>.txt: mỗi dòng đúng 1 mã điểm, không trùng; = tiền tố xếp hạng của S1(k=5) lọc phông (so với plans.pkl của khâu quy mô và rebuild.pkl của r_01);
      thứ tự ưu tiên giảm dần: khoá xếp hạng không giảm trong từng đoạn (S0 theo (−số lần, −g, mã); phần thêm theo (hạng tốt nhất, −cầu, −g, mã)), S0 đứng trước.
  3b  plan.json: chars + uncovered khớp chars_<cuốn>.txt (thứ tự, tập); font_idx khớp font_of_<cuốn>.tsv; phông của gói nằm trong fonts.json và có tệp.
  3c  phông được gán CÓ glyph (cmap fontTools: hợp mọi bảng con và bảng FreeType chọn); đầu tiên-trong-chuỗi-có-thể-vẽ; PUA chỉ phông họ NomNaTong.
  3d  VẼ THỬ bằng PIL/FreeType với ĐÚNG tệp phông trong gói (code/font_diffusion/fonts/*) trên MỌI cặp (phông, chữ) của kế hoạch (51.730 chữ-cuốn): rỗng / trùng ô .notdef / trùng ảnh nhiều chữ.
  3e  tiền đề "bảng PUA như nhau trong họ NomNaTong": so hình glyph các mã PUA có ở cả NomNaTong-Regular và NomNaTongLight2.

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_03_chu_va_phong.py
Ra: 03_chu_va_phong.json.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections
import hashlib
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402

CHAIN = ["NomNaTong[fd]", "HanaMinB", "HanNomA", "NomNaTongLight2", "HanaMinA", "NomNaTong[fonts]", "PlangothicP1", "PlangothicP2"]
PACK_NAME = {"NomNaTong[fd]": "NomNaTong", "HanaMinB": "HanaMinB", "HanNomA": "HanNomA", "NomNaTongLight2": "NomNaTongLight2", "HanaMinA": "HanaMinA", "PlangothicP2": "PlangothicP2"}
REPO_ONLY = {"NomNaTong[fonts]": "fonts/NomNaTong-Regular.ttf", "PlangothicP1": "fonts/PlangothicP1-Regular.ttf"}
PUA_FAMILY = {"NomNaTong", "NomNaTongLight2"}          # tên trong gói


def draw(font, ch, size):
    W = size * 3
    im = Image.new("L", (W, W), 0)
    ImageDraw.Draw(im).text((size // 2, size // 2), ch, font=font, fill=255)
    return im


def main():
    t0 = time.time()
    inv = Lb.Inv()
    res: dict = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"))
    plan = json.loads((Lb.PACK / "plan.json").read_text(encoding="utf-8"))
    fonts_json = json.loads((Lb.PACK / "fonts.json").read_text(encoding="utf-8"))
    fdir = Lb.PACK / "code" / "font_diffusion" / "fonts"
    fnames = plan["font_names"]
    # ------------------------------------------------------------------------------ 3b-0 phông của gói
    pf = {}
    for n in fnames:
        p = fdir / fonts_json[n]
        pf[n] = dict(tep=str(p.relative_to(Lb.PACK)), ton_tai=p.exists(), bytes=p.stat().st_size if p.exists() else None)
    res["3b_phong_cua_goi"] = pf
    inv.check("3b_moi_phong_trong_plan_co_trong_fonts_json_va_co_tep", all(n in fonts_json and pf[n]["ton_tai"] for n in fnames), pf)
    inv.check("3b_fonts_json_khong_co_phong_thua", set(fonts_json) == set(fnames), sorted(set(fonts_json) ^ set(fnames)))
    # ------------------------------------------------------------------------------ 3a/3b danh sách chữ
    per = {}
    allpairs = defaultdict(set)         # (tên phông gói, chữ) -> các cuốn
    plans_pkl = pickle.load(open(Lb.FULL / "quy_mo_va_ngan_sach" / "work" / "plans.pkl", "rb"))
    PL = plans_pkl["plans"]
    rebuild = pickle.load(open(Lb.OUT / "rebuild.pkl", "rb"))
    avail_cas = pickle.load(open(Lb.FULL / "quy_mo_va_ngan_sach" / "work" / "phong.pkl", "rb"))["avail_cas"]
    for b in Lb.ORDER:
        txt_raw = (Lb.SPEC / f"chars_{b}.txt").read_text(encoding="utf-8")
        txt = [c for c in txt_raw.split("\n") if c]
        tsv = {}
        for ln in (Lb.SPEC / f"font_of_{b}.tsv").read_text(encoding="utf-8").split("\n"):
            if "\t" in ln:
                c, n = ln.split("\t")
                tsv[c] = n
        pb = plan["books"][b]
        chars, unc, fidx = pb["chars"], pb["uncovered"], pb["font_idx"]
        d = dict(n_dong=len(txt), moi_dong_1_ma_diem=all(len(c) == 1 for c in txt), co_dong_trong=("" in txt_raw.split("\n")[:-1]),
                 ky_tu_la=[f"U+{ord(c):04X}" for c in txt if c.isspace() or ord(c) < 0x2E80 and not Lb.is_pua(c)][:5],
                 khong_trung=len(set(txt)) == len(txt), n_distinct=len(set(txt)))
        d["plan_chars_cong_uncovered_bang_txt_tap"] = set(chars) | set(unc) == set(txt) and len(chars) + len(unc) == len(txt)
        d["plan_chars_bang_txt_tru_uncovered_dung_thu_tu"] = list(chars) == [c for c in txt if c not in set(unc)]
        d["font_idx_dai_bang_chars"] = len(fidx) == len(chars)
        d["tsv_phu_het_txt"] = set(tsv) == set(txt)
        names = [fnames[i] for i in fidx]
        d["font_idx_khop_tsv"] = all(tsv.get(c) == n for c, n in zip(chars, names))
        d["uncovered_tsv_rong"] = all(tsv.get(c) == "" for c in unc)
        d["so_chu_khong_phong"] = len(unc)
        d["uncovered"] = [f"U+{ord(c):05X}" for c in unc]
        d["theo_phong"] = dict(Counter(names))
        d["theo_khoi"] = dict(Counter(Lb.block_of(c) for c in chars))
        for c, n in zip(chars, names):
            allpairs[(n, c)].add(b)
        # thứ tự ưu tiên (khoá của khâu quy mô, từ plans.pkl)
        p = PL[b]["sach"]
        n_self = plans_pkl["n_self"][b]; g_oth = plans_pkl["g_oth"][b]
        g = Counter(g_oth)
        for (rk, c), n in n_self.items():
            g[c] += n
        S0 = p["S0"]; best = p["best"]; e = p["e"]; dem = p["demand"]
        order = txt
        seg0 = [c for c in order if c in S0]
        seg1 = [c for c in order if c not in S0]
        k0 = [(-e[c], -g.get(c, 0), ord(c)) for c in seg0]
        k1 = [(best[c], -dem[c], -g.get(c, 0), ord(c)) for c in seg1]
        d["S0_dung_truoc_phan_con_lai"] = order[:len(seg0)] == seg0
        d["khoa_xep_hang_S0_khong_giam"] = all(a <= c for a, c in zip(k0, k0[1:]))
        d["khoa_xep_hang_phan_them_khong_giam"] = all(a <= c for a, c in zip(k1, k1[1:]))
        d["phan_them_deu_hang_<=5"] = all(best[c] <= Lb.K for c in seg1)
        d["n_S0_trong_danh_sach"] = len(seg0); d["n_phan_them"] = len(seg1)
        exp = [c for c in p["ranked"] if c in avail_cas][:len({c for c in p["S1"][Lb.K] if c in avail_cas})]
        d["danh_sach_la_tien_to_xep_hang_goc_loc_phong"] = (order == exp)
        d["n_S1_5_loc_phong_theo_plans_pkl"] = len(exp)
        d["bang_rebuild_doc_lap_r01"] = (rebuild["mine"][b] == txt)
        per[b] = d
    res["3a_3b_theo_cuon"] = per
    for key, nm in (("moi_dong_1_ma_diem", "moi_dong_dung_1_ma_diem"), ("khong_trung", "khong_trung_trong_cuon"), ("plan_chars_cong_uncovered_bang_txt_tap", "plan_chars+uncovered_khop_chars_txt"),
                    ("plan_chars_bang_txt_tru_uncovered_dung_thu_tu", "plan_chars_dung_thu_tu_chars_txt"), ("font_idx_khop_tsv", "font_idx_khop_font_of_tsv"), ("tsv_phu_het_txt", "font_of_tsv_phu_het"),
                    ("S0_dung_truoc_phan_con_lai", "S0_truoc_phan_them"), ("khoa_xep_hang_S0_khong_giam", "khoa_uu_tien_S0_khong_giam"), ("khoa_xep_hang_phan_them_khong_giam", "khoa_uu_tien_phan_them_khong_giam"),
                    ("phan_them_deu_hang_<=5", "phan_them_deu_hang_<=5"), ("danh_sach_la_tien_to_xep_hang_goc_loc_phong", "danh_sach_la_tien_to_S1(5)_loc_phong")):
        inv.check(f"3a_{nm}_10_cuon", all(v[key] for v in per.values()), [b for b, v in per.items() if not v[key]])
    inv.check("3a_bang_rebuild_doc_lap_9_cuon_tru_KVK", all(v["bang_rebuild_doc_lap_r01"] for b, v in per.items() if b != "KVK"), [b for b, v in per.items() if not v["bang_rebuild_doc_lap_r01"]])
    tot = sum(len(plan["books"][b]["chars"]) for b in Lb.ORDER)
    distinct_all = {c for b in Lb.ORDER for c in plan["books"][b]["chars"] + plan["books"][b]["uncovered"]}
    res["3a_tong"] = dict(chu_cuon_phu_duoc=tot, chu_cuon_ke_ca_uncovered=tot + sum(len(plan["books"][b]["uncovered"]) for b in Lb.ORDER), chu_khac_nhau_toan_goi=len(distinct_all),
                          cap_phong_chu_khac_nhau=len(allpairs), theo_khoi=dict(Counter(Lb.block_of(c) for c in distinct_all)),
                          chu_o_ca_10_cuon=sum(1 for c in distinct_all if all(c in set(plan["books"][b]["chars"]) | set(plan["books"][b]["uncovered"]) for b in Lb.ORDER)))
    inv.check("3a_51730_chu_cuon_va_10,2_nghin_chu_khac_nhau (tài liệu hướng dẫn nêu 51,7 nghìn ảnh / 10,2 nghìn chữ)", tot == 51730 and 10200 <= len(distinct_all) <= 10250, (tot, len(distinct_all)))
    # ------------------------------------------------------------------------------ 3c cmap
    cm_u, cm_ft, ft_key, files = {}, {}, {}, {}
    for n in fnames:
        files[n] = fdir / fonts_json[n]
        cm_u[n] = Lb.cmap_union(files[n])
        cm_ft[n], ft_key[n] = Lb.cmap_freetype(files[n])
    res["3c_cmap"] = {n: dict(n_hop=len(cm_u[n]), n_ban_FreeType_chon=len(cm_ft[n]), bang_FreeType_chon=list(ft_key[n]), n_chi_o_bang_phu=len(cm_u[n] - cm_ft[n])) for n in fnames}
    not_in_u = [(n, c) for (n, c) in allpairs if ord(c) not in cm_u[n]]
    not_in_ft = [(n, c) for (n, c) in allpairs if ord(c) not in cm_ft[n]]
    res["3c_cap_khong_co_glyph_theo_cmap_hop"] = [f"{n}:U+{ord(c):05X}" for n, c in not_in_u[:20]]
    res["3c_cap_chi_co_o_bang_phu_khong_o_bang_FreeType_chon"] = dict(n=len(not_in_ft), vi_du=[f"{n}:U+{ord(c):05X}" for n, c in not_in_ft[:20]])
    inv.check("3c_moi_cap_phong_chu_co_glyph_theo_cmap_hop_cua_tep_trong_goi", not not_in_u, f"{len(not_in_u)} cặp thiếu")
    inv.check("3c_moi_cap_co_glyph_trong_bang_cmap_ma_FreeType_chon (xấp xỉ)", not not_in_ft, f"{len(not_in_ft)} cặp chỉ ở bảng phụ")
    # PUA
    pua_pairs = [(n, c) for (n, c) in allpairs if Lb.is_pua(c)]
    pua_bad = [(n, c) for n, c in pua_pairs if n not in PUA_FAMILY]
    res["3c_PUA"] = dict(so_cap_PUA=len(pua_pairs), theo_phong=dict(Counter(n for n, _ in pua_pairs)), cap_PUA_o_phong_ngoai_ho_NomNaTong=[f"{n}:U+{ord(c):05X}" for n, c in pua_bad[:20]],
                         theo_khoi=dict(Counter(Lb.block_of(c) for _, c in pua_pairs)),
                         khoang_PUA="U+E000–U+F8FF, U+F0000–U+FFFFD, U+100000–U+10FFFD")
    inv.check("3c_chu_PUA_chi_gan_phong_ho_NomNaTong (NomNaTong, NomNaTongLight2)", not pua_bad, f"{len(pua_bad)} cặp sai")
    # đầu tiên trong chuỗi có thể vẽ: kiểm vi phạm "phông sớm hơn có glyph và vẽ được nhưng không được chọn"
    # (tính sau 3d vì cần kết quả vẽ)
    # ------------------------------------------------------------------------------ 3d vẽ thử PIL
    pil_res = {}
    sig_by_font = {}
    ftobj = {}
    for size in (64, 128):
        for n in fnames:
            key = (n, size)
            ftobj[key] = ImageFont.truetype(str(files[n]), size)
            sigs = {}
            for cp in (0xFFFF, 0x2FFFE, 0x10FFFE, 0x378, 0xFFFE):
                if cp not in cm_u[n]:
                    im = draw(ftobj[key], chr(cp), size)
                    sigs[cp] = None if im.getbbox() is None else hashlib.md5(im.tobytes()).hexdigest()
            sig_by_font[key] = sigs
    blank_pairs = {64: set(), 128: set()}
    notdef_pairs = {64: set(), 128: set()}
    hashes = {64: defaultdict(list), 128: defaultdict(list)}
    for size in (64, 128):
        for (n, c) in sorted(allpairs, key=lambda x: (x[0], ord(x[1]))):
            im = draw(ftobj[(n, size)], c, size)
            if im.getbbox() is None:
                blank_pairs[size].add((n, c)); continue
            h = hashlib.md5(im.tobytes()).hexdigest()
            hashes[size][(n, h)].append(c)
            if h in {v for v in sig_by_font[(n, size)].values() if v}:
                notdef_pairs[size].add((n, c))
    out3d = {}
    for n in fnames:
        npair = sum(1 for (m, c) in allpairs if m == n)
        out3d[n] = dict(n_cap=npair, n_cuon_chu=int(sum(len(v) for (m, c), v in allpairs.items() if m == n)),
                        rong_64=int(sum(1 for (m, c) in blank_pairs[64] if m == n)), rong_128=int(sum(1 for (m, c) in blank_pairs[128] if m == n)),
                        trung_notdef_64=int(sum(1 for (m, c) in notdef_pairs[64] if m == n)), trung_notdef_128=int(sum(1 for (m, c) in notdef_pairs[128] if m == n)),
                        notdef_ro_hay_khong={str(hex(cp)): ("rong" if v is None else "co_hinh") for cp, v in sig_by_font[(n, 128)].items()})
        dup = [(h, cs) for (m, h), cs in hashes[128].items() if m == n and len(cs) >= 2]
        out3d[n]["nhom_chu_cung_hinh_(>=2)_128"] = len(dup)
        out3d[n]["so_chu_trong_nhom_cung_hinh_(>=2)_128"] = int(sum(len(cs) for _, cs in dup))
        out3d[n]["nhom_cung_hinh_lon_nhat_128"] = max((len(cs) for _, cs in dup), default=0)
        out3d[n]["vi_du_nhom_(>=3)"] = [[f"U+{ord(c):05X}" for c in cs[:8]] for _, cs in sorted(dup, key=lambda x: -len(x[1]))[:3] if len(cs) >= 3]
    res["3d_ve_thu_PIL_theo_phong"] = out3d
    res["3d_tong"] = dict(cap_ve=len(allpairs), rong_64=len(blank_pairs[64]), rong_128=len(blank_pairs[128]), trung_notdef_64=len(notdef_pairs[64]), trung_notdef_128=len(notdef_pairs[128]))
    res["3d_cap_loi_vi_du"] = dict(rong_128=[f"{n}:U+{ord(c):05X}" for n, c in sorted(blank_pairs[128], key=lambda x: ord(x[1]))[:20]],
                                    notdef_128=[f"{n}:U+{ord(c):05X}" for n, c in sorted(notdef_pairs[128], key=lambda x: ord(x[1]))[:20]])
    # số chữ-cuốn bị ảnh hưởng
    aff = {sz: collections.Counter() for sz in (64, 128)}
    for sz in (64, 128):
        for (n, c) in blank_pairs[sz] | notdef_pairs[sz]:
            for b in allpairs[(n, c)]:
                aff[sz][b] += 1
    res["3d_chu_cuon_bi_anh_huong"] = {str(sz): dict(aff[sz]) for sz in aff}
    inv.check("3d_0_cap_ve_rong_hoac_trung_notdef_PIL_64px", not (blank_pairs[64] | notdef_pairs[64]), len(blank_pairs[64] | notdef_pairs[64]))
    inv.check("3d_0_cap_ve_rong_hoac_trung_notdef_PIL_128px (đúng cỡ ttf2im)", not (blank_pairs[128] | notdef_pairs[128]), len(blank_pairs[128] | notdef_pairs[128]))
    # đầu tiên trong chuỗi có thể vẽ
    name_of_chain = {k: PACK_NAME.get(k) for k in CHAIN}
    viol = []
    cache_ok = {}

    def ok_in(chain_name, c) -> bool:
        key = (chain_name, c)
        if key in cache_ok:
            return cache_ok[key]
        if chain_name in REPO_ONLY:
            p = Lb.REPO / REPO_ONLY[chain_name]
            cm = _repo_cmap(chain_name, p)
        else:
            nm = name_of_chain[chain_name]
            p = files[nm]; cm = cm_u[nm]
        v = ord(c) in cm
        if v and Lb.is_pua(c) and chain_name not in ("NomNaTong[fd]", "NomNaTong[fonts]", "NomNaTongLight2"):
            v = False
        if v:
            ft = _repo_font(chain_name, p)
            im = draw(ft, c, 64)
            v = im.getbbox() is not None
        cache_ok[key] = v
        return v
    rp_cm, rp_ft = {}, {}

    def _repo_cmap(k, p):
        if k not in rp_cm:
            rp_cm[k] = Lb.cmap_union(p)
        return rp_cm[k]

    def _repo_font(k, p):
        if k not in rp_ft:
            rp_ft[k] = ImageFont.truetype(str(p), 64)
        return rp_ft[k]
    # dùng phông trong gói cho các phông có trong gói
    for n in fnames:
        rp_ft[next(k for k, v in name_of_chain.items() if v == n)] = ftobj[(n, 64)]
        rp_cm[next(k for k, v in name_of_chain.items() if v == n)] = cm_u[n]
    for (n, c), bks in allpairs.items():
        pos = next(i for i, k in enumerate(CHAIN) if name_of_chain[k] == n)
        for k in CHAIN[:pos]:
            if ok_in(k, c):
                viol.append(dict(chu=f"U+{ord(c):05X}", duoc_gan=n, phong_som_hon_ve_duoc=k))
                break
    res["3c_vi_pham_dau_tien_trong_chuoi"] = dict(n=len(viol), vi_du=viol[:10])
    inv.check("3c_gan_phong_la_'phong_dau_tien_trong_chuoi_co_glyph_va_ve_duoc'", not viol, f"{len(viol)} vi phạm")
    # chữ uncovered: phông nào (kể cả ngoài gói) có
    unc_info = {}
    for b in Lb.ORDER:
        for c in plan["books"][b]["uncovered"]:
            unc_info[f"{b}:U+{ord(c):05X}"] = dict(trong_goi=[n for n in fnames if ord(c) in cm_u[n]],
                                                    ngoai_goi=[k for k, rel in REPO_ONLY.items() if ord(c) in Lb.cmap_union(Lb.REPO / rel)],
                                                    khoi=Lb.block_of(c), la_PUA=Lb.is_pua(c))
    res["3c_chu_khong_phong"] = unc_info
    # hai tệp NomNaTong-Regular.ttf cùng tên: khác nhau?
    a, b2 = Lb.REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf", Lb.REPO / "fonts/NomNaTong-Regular.ttf"
    res["3c_hai_tep_NomNaTong_Regular_cung_ten"] = dict(sha_font_diffusion=Lb.sha256_file(a)[:16], sha_fonts=Lb.sha256_file(b2)[:16], bytes=[a.stat().st_size, b2.stat().st_size],
                                                       cmap_chi_o_fonts=len(Lb.cmap_union(b2) - Lb.cmap_union(a)), cmap_chi_o_font_diffusion=len(Lb.cmap_union(a) - Lb.cmap_union(b2)))
    # ------------------------------------------------------------------------------ 3f chữ bị lọc vì không phông TRƯỚC khi dựng kế hoạch (không nằm trong `uncovered`)
    S15 = set().union(*rebuild["S1k"].values())
    avail_r = {c for c, n in rebuild["assign"].items() if n}
    dropped = sorted(S15 - avail_r, key=ord)
    res["3f_chu_loc_truoc_ke_hoach"] = dict(n_S1_5_hop=len(S15), n_co_phong=len(S15 & avail_r), n_bi_loc=len(dropped), ma=[f"U+{ord(c):05X}" for c in dropped], theo_khoi=dict(Counter(Lb.block_of(c) for c in dropped)),
                                            so_chu_thuoc_S0={b: int(sum(1 for c in dropped if c in rebuild["S0"][b])) for b in Lb.ORDER},
                                            ly_do="PUA chỉ có ở phông ngoài họ NomNaTong (quy tắc PUA) hoặc không phông nào có glyph (phong.json: chu_khong_phong)")
    # ------------------------------------------------------------------------------ 3e tiền đề PUA
    nn, lt = "NomNaTong", "NomNaTongLight2"
    shared = sorted(c for c in (cm_u[nn] & cm_u[lt]) if Lb.is_pua(chr(c)))
    f_nn, f_lt = ftobj[(nn, 64)], ftobj[(lt, 64)]

    def vec(font, ch):
        im = np.asarray(draw(font, ch, 64), dtype=np.float32) / 255.0
        ys, xs = np.nonzero(im > 0.5)
        if len(xs) == 0:
            return None
        c = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        import cv2
        d = cv2.resize(c, (24, 24), interpolation=cv2.INTER_AREA).ravel()
        d = d - d.mean()
        nr = float(np.linalg.norm(d))
        return d / nr if nr > 0 else None
    rng = np.random.default_rng(20261004)
    samp = [shared[i] for i in sorted(rng.choice(len(shared), size=min(1500, len(shared)), replace=False))] if shared else []
    A = [vec(f_nn, chr(cp)) for cp in samp]
    B = [vec(f_lt, chr(cp)) for cp in samp]
    ok = [(a is not None and b_ is not None) for a, b_ in zip(A, B)]
    same = np.array([float(a @ b_) for a, b_, o in zip(A, B, ok) if o])
    perm = rng.permutation(len(samp))
    diff = np.array([float(A[i] @ B[j]) for i, j in zip(range(len(samp)), perm) if ok[i] and ok[j] and i != j])
    # tương quan chữ không PUA (đối chứng "cùng bảng chắc chắn"): chữ CJK chung hai phông
    shared_cjk = sorted(c for c in (cm_u[nn] & cm_u[lt]) if 0x4E00 <= c <= 0x9FFF)
    s2 = [shared_cjk[i] for i in sorted(rng.choice(len(shared_cjk), size=min(1500, len(shared_cjk)), replace=False))]
    C = [(vec(f_nn, chr(c)), vec(f_lt, chr(c))) for c in s2]
    same_cjk = np.array([float(a @ b_) for a, b_ in C if a is not None and b_ is not None])
    res["3e_PUA_NomNaTong_vs_Light2"] = dict(n_PUA_chung=len(shared), n_mau=len(samp), tuong_quan_cung_ma_PUA=dict(trung_vi=round(float(np.median(same)), 3), p5=round(float(np.percentile(same, 5)), 3), p1=round(float(np.percentile(same, 1)), 3), ti_le_duoi_0_5=round(float((same < 0.5).mean()), 4)),
                                             tuong_quan_ma_PUA_khac_nhau_hoan_vi=dict(trung_vi=round(float(np.median(diff)), 3), p95=round(float(np.percentile(diff, 95)), 3)),
                                             doi_chung_CJK_chung=dict(n=len(s2), trung_vi=round(float(np.median(same_cjk)), 3), p5=round(float(np.percentile(same_cjk, 5)), 3), p1=round(float(np.percentile(same_cjk, 1)), 3)),
                                             chu_PUA_duoc_gan_Light2=[f"{b}:U+{ord(c):05X}" for (n, c), bs in allpairs.items() if n == lt and Lb.is_pua(c) for b in sorted(bs)][:12],
                                             cac_chu_gan_Light2_co_o_NomNaTong=[f"U+{ord(c):05X}" for (n, c) in allpairs if n == lt and ord(c) in cm_u[nn]])
    inv.check("3e_tien_de_PUA_cung_bang_NomNaTong_Light2: trung vi tuong quan cung ma PUA >= 0.9 va p1 >= 0.5", res["3e_PUA_NomNaTong_vs_Light2"]["tuong_quan_cung_ma_PUA"]["trung_vi"] >= 0.9 and res["3e_PUA_NomNaTong_vs_Light2"]["tuong_quan_cung_ma_PUA"]["p1"] >= 0.5,
              res["3e_PUA_NomNaTong_vs_Light2"]["tuong_quan_cung_ma_PUA"])
    res["bat_bien"] = inv.summary(); res["bat_bien_chi_tiet"] = inv.rows; res["giay"] = round(time.time() - t0, 1)
    Lb.jdump(res, "03_chu_va_phong.json")
    print(f"[03] xong {res['giay']}s; bất biến {res['bat_bien']['dat']}/{res['bat_bien']['tong']} rớt={res['bat_bien']['rot']}")


if __name__ == "__main__":
    main()
