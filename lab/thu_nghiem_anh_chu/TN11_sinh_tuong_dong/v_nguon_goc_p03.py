"""TN11 v_nguon_goc_p03 (03/10) — KIỂM ĐỊNH ĐỘC LẬP p03_truc_tiep_sinh_so_sanh.py (phiên khác). 0 mô hình, 0 API, chỉ ĐỌC.

Số do script sinh:
  1. 'fontdiffuser_global_stt' == 'font_nomnatong' vì sao: ảnh FD chung của p03 đọc từ gannhanocr-fd/<XX>/U+XXXX.png — tệp có tồn tại nhưng là CON TRỎ GIT-LFS
     (~129 byte, ASCII) nên cv2.imread trả None -> p03 dòng 171-175 rơi về ảnh phông (fd_im = f_im). Đếm trên 24 chữ ứng viên + toàn kho.
  2. 'nét sinh ra' (K05): cv2.dilate(k3, iterations=2) áp lên ảnh FD sinh — đo tỉ lệ điểm tối trước/sau (nét dày hay mảnh?); chỉ áp cho nhánh 'gen', không áp cho phông / FD chung.
  3. Tham số sinh của p03: guidance_scale mặc định của bộ sinh (p03 không đặt) so với 2.0 mà báo cáo nêu; số bước, thuật toán; thời gian suy từ mtime tệp.
  4. Thống kê p03 (n=10): Wilson 95 %, McNemar chính xác, kiểm dấu trên cos/margin theo ô; đối chiếu p02 (n=189).
  5. p04 (rare, n_sample=15) so với p05 (n_sample=20): cùng bộ, cùng mã họ HQC, khác thứ hạng HQC1/2/3.
Ra: measure_out/_tn11/verify/nguon_goc/p03_kiem_tra.json
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TN11 = REPO / "lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong"
M11 = REPO / "measure_out/_tn11"
OUT_DIR = M11 / "verify" / "nguon_goc"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0, ctr - hw) * 100, 1), round(min(1, ctr + hw) * 100, 1))


def exact_p(b, c):
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5).pvalue)


def grep_lines(path: Path, pat: str):
    rx = re.compile(pat)
    return [(i + 1, l.rstrip()) for i, l in enumerate(path.read_text(encoding="utf-8").splitlines()) if rx.search(l)]


def main():
    res = {}
    j = json.loads((M11 / "p03_direct/ket_qua_p03.json").read_text(encoding="utf-8"))
    det = j["details"]

    # ---------------------------------------------------------------- 1. con trỏ LFS
    chars = sorted({c for r in det for c in r["candidates"]})
    fd_root = REPO / "gannhanocr-fd"
    arc_root = REPO / "ArcFace/data/glyphs"
    arc_index = {}
    for root, _, fs in os.walk(arc_root):
        for fn in fs:
            if fn.startswith("U+") and fn.endswith(".png"):
                arc_index.setdefault(fn, Path(root) / fn)
    rows = []
    for c in chars:
        # đúng công thức đường dẫn của p03 dòng 168
        p = fd_root / f"{ord(c):02X}"[:2] / f"U+{ord(c):04X}.png"
        ex = p.exists()
        size = p.stat().st_size if ex else None
        head = p.read_bytes()[:40].decode("ascii", "replace") if ex else None
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if ex else None
        a = arc_index.get(f"U+{ord(c):04X}.png")
        rows.append({"char": c, "cp": f"U+{ord(c):04X}", "path_p03": str(p.relative_to(REPO)), "exists": ex, "size_bytes": size, "is_lfs_pointer": bool(head and head.startswith("version https://git-lfs")),
                     "cv2_imread_is_None": im is None, "p03_rơi_về_phông": (not ex) or im is None,
                     "ArcFace_glyphs_real_png": bool(a is not None and a.stat().st_size >= 1024)})
    n_lfs = sum(r["is_lfs_pointer"] for r in rows)
    n_none = sum(r["cv2_imread_is_None"] for r in rows)
    # toàn kho
    tot = small = 0
    for root, _, fs in os.walk(fd_root):
        for fn in fs:
            if fn.startswith("U+") and fn.endswith(".png"):
                tot += 1
                if (Path(root) / fn).stat().st_size < 1024:
                    small += 1
    same = sum(1 for r in det if r["cos_font"] == r["cos_fd"] and r["margin_font"] == r["margin_fd"] and r["top1_font"] == r["top1_fd"] and r["acc_font"] == r["acc_fd"])
    res["1_fd_chung_la_con_tro_lfs"] = {"n_chu_ung_vien_duy_nhat": len(rows), "n_tep_ton_tai": sum(r["exists"] for r in rows), "n_con_tro_LFS": int(n_lfs), "n_cv2_imread_None": int(n_none),
                                        "n_p03_roi_ve_phong": int(sum(r["p03_rơi_về_phông"] for r in rows)), "n_chu_co_anh_that_trong_ArcFace_glyphs": int(sum(r["ArcFace_glyphs_real_png"] for r in rows)),
                                        "kho_gannhanocr_fd_tong_png": tot, "kho_gannhanocr_fd_con_tro_LFS_duoi_1KB": small, "kho_gannhanocr_fd_anh_that": tot - small,
                                        "p03_json_so_hang_font_trung_het_fd": f"{same}/{len(det)}",
                                        "p03_dong_ma": {"fd_path": "p03:168", "exists/imread/None-fallback": "p03:169-175 (if not fd_path.exists(): fd_im=f_im ... if fd_im is None: fd_im=f_im)"},
                                        "chi_tiet": rows}
    print(f"[1] {len(rows)} chữ ứng viên duy nhất: tệp tồn tại {res['1_fd_chung_la_con_tro_lfs']['n_tep_ton_tai']}, con trỏ LFS {n_lfs}, cv2.imread=None {n_none}, rơi về phông {res['1_fd_chung_la_con_tro_lfs']['n_p03_roi_ve_phong']}/{len(rows)}; "
          f"kho {tot} PNG, con trỏ {small}, ảnh thật {tot - small}; JSON: hàng font≡fd {same}/{len(det)}", flush=True)

    # ---------------------------------------------------------------- 1b. chữ thiếu ảnh sinh -> nhánh 'gen' cũng rơi về phông (p03:177-183)
    from fontTools.ttLib import TTFont
    cmap = set(TTFont(str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")).getBestCmap())
    gdir = M11 / "p03_direct/gen_book_style/b34_style"
    miss = [c for c in chars if not (gdir / f"U+{ord(c):04X}.png").exists()]
    not_in_font = [c for c in chars if ord(c) not in cmap]
    cells_gt_missing = [r["cell_id"] for r in det if r["ground_truth"] in miss]
    cells_any_missing = [r["cell_id"] for r in det if any(c in miss for c in r["candidates"])]
    res["1b_gen_thieu_anh"] = {"n_chu_ung_vien": len(chars), "n_chu_co_anh_gen": len(chars) - len(miss), "chu_thieu_anh_gen": [f"U+{ord(c):04X}" for c in miss],
                               "chu_khong_co_trong_phong_NomNaTong": [f"U+{ord(c):04X}" for c in not_in_font], "trung_nhau": sorted(miss) == sorted(not_in_font),
                               "o_co_chu_dung_thieu_anh_gen": cells_gt_missing, "o_co_it_nhat_1_ung_vien_thieu_anh_gen": cells_any_missing, "so_o": len(det)}
    print(f"[1b] chữ thiếu ảnh gen {len(miss)}/{len(chars)} ({res['1b_gen_thieu_anh']['chu_thieu_anh_gen']}), trùng chữ ngoài phông: {res['1b_gen_thieu_anh']['trung_nhau']}; "
          f"ô có ứng viên thiếu {len(cells_any_missing)}/{len(det)}, ô có chữ ĐÚNG thiếu {len(cells_gt_missing)}", flush=True)

    # ---------------------------------------------------------------- 2. giãn nở (dilate)
    gen_dir = M11 / "p03_direct/gen_book_style/b34_style"
    k3 = np.ones((3, 3), np.uint8)
    dr = []
    for f in sorted(gen_dir.glob("U+*.png")):
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        d = cv2.dilate(im, k3, iterations=2)
        dr.append({"f": f.name, "shape": list(im.shape), "mean_before": round(float(im.mean()), 1), "mean_after": round(float(d.mean()), 1),
                   "dark_frac_before": round(float((im < 128).mean()), 4), "dark_frac_after": round(float((d < 128).mean()), 4)})
    # mật độ mực TRONG hộp bao nét (so được với crop thật: crop_meta cột 2 = tỉ lệ gray<128 trên crop đã siết hộp)
    def tight_dark(im):
        ys, xs = np.nonzero(im < 128)
        if len(ys) == 0:
            return 0.0
        sub = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return float((sub < 128).mean())
    for r, f in zip(dr, sorted(gen_dir.glob("U+*.png"))):
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        r["tight_dark_before"] = round(tight_dark(im), 4); r["tight_dark_after"] = round(tight_dark(cv2.dilate(im, k3, iterations=2)), 4)
    meta = np.load(REPO / "measure_out/_tn8/emb/B34_crop_meta.npy")
    okm = meta[:, 3] > 0
    ink_b34 = meta[okm, 2]
    db = np.array([r["dark_frac_before"] for r in dr]); da = np.array([r["dark_frac_after"] for r in dr])
    res["2_dilate"] = {"n_anh": len(dr), "nen_sang_net_toi (mean>128)": int(sum(r["mean_before"] > 128 for r in dr)),
                       "dark_frac_truoc_tb": round(float(db.mean()), 4), "dark_frac_sau_tb": round(float(da.mean()), 4),
                       "ti_le_sau_truoc_trung_vi": round(float(np.median(da / np.maximum(db, 1e-9))), 3), "n_anh_net_mong_di": int(sum(a < b for a, b in zip(da, db))),
                       "tight_bbox_dark_frac_truoc_tb": round(float(np.mean([r["tight_dark_before"] for r in dr])), 4), "tight_bbox_dark_frac_sau_tb": round(float(np.mean([r["tight_dark_after"] for r in dr])), 4),
                       "B34_crop_that_ink_frac_trung_vi": round(float(np.median(ink_b34)), 4), "B34_crop_that_ink_frac_p25_p75": [round(float(np.percentile(ink_b34, 25)), 4), round(float(np.percentile(ink_b34, 75)), 4)],
                       "dong_ma_p03": "p03:160 k3=np.ones((3,3)); p03:181 g_im = cv2.dilate(g_im, k3, iterations=2) — chỉ trong nhánh gen_path.exists(); font (p03:164-165) và FD chung (p03:167-176) KHÔNG qua dilate",
                       "chi_tiet": dr}
    print(f"[2] dilate(3x3,2) trên {len(dr)} ảnh FD sinh: nền sáng {res['2_dilate']['nen_sang_net_toi (mean>128)']}/{len(dr)}; tỉ lệ điểm tối {db.mean():.4f} -> {da.mean():.4f} "
          f"(trung vị sau/trước {res['2_dilate']['ti_le_sau_truoc_trung_vi']}); {res['2_dilate']['n_anh_net_mong_di']}/{len(dr)} ảnh mỏng đi | trong hộp bao nét: {res['2_dilate']['tight_bbox_dark_frac_truoc_tb']} -> {res['2_dilate']['tight_bbox_dark_frac_sau_tb']} "
          f"so với crop B34 thật (ink_frac trung vị {res['2_dilate']['B34_crop_that_ink_frac_trung_vi']}, IQR {res['2_dilate']['B34_crop_that_ink_frac_p25_p75']})", flush=True)

    # ---------------------------------------------------------------- 3. tham số sinh
    cfgp = REPO / "font_diffusion/src/configs/fontdiffuser.py"
    gl = grep_lines(cfgp, r'"--guidance_scale"|"--num_inference_steps"|"--algorithm_type"|"--order"|"--method"|"--skip_type"')
    # đọc default ngay sau tham số
    txt = cfgp.read_text(encoding="utf-8").splitlines()
    defaults = {}
    for ln, _ in gl:
        blk = " ".join(l.strip() for l in txt[ln - 1: ln + 6])
        name = re.search(r'"--(\w+)"', txt[ln - 1]).group(1)
        m = re.search(r"default=([^,\)\s]+)", blk)
        defaults[name] = {"line": ln, "default": m.group(1) if m else None}
    p02src = TN11 / "p02_fd_phong_cach_sach.py"
    p03src = TN11 / "p03_truc_tiep_sinh_so_sanh.py"
    gen_py = REPO / "core/ranking/fontdiffusion_gen.py"
    pipe_py = REPO / "font_diffusion/src/dpm_solver/pipeline_dpm_solver.py"
    samp_py = REPO / "font_diffusion/inference/sample_optimized.py"
    # thời gian suy từ mtime
    ts = sorted(f.stat().st_mtime for f in gen_dir.glob("U+*.png"))
    uniq = sorted(set(int(t) for t in ts))
    gaps = [b - a for a, b in zip(uniq, uniq[1:])]
    res["3_tham_so_sinh"] = {"parser_defaults": defaults,
                             "p03_co_dat_guidance_scale": bool(grep_lines(p03src, r"guidance_scale")),
                             "p03_goi_FontDiffusionGenerator": [ (ln, l.strip()) for ln, l in grep_lines(p03src, r"FontDiffusionGenerator\(|\.generate\(|batch_size=") ],
                             "p02_dat_guidance_scale": [(ln, l.strip()) for ln, l in grep_lines(p02src, r"guidance_scale")],
                             "gen_py_build_args_ghi_de_guidance": bool(grep_lines(gen_py, r"args\.guidance_scale\s*=")),
                             "pipeline_guidance_scale_dong": [(ln, l.strip()) for ln, l in grep_lines(pipe_py, r"guidance_scale")],
                             "sample_optimized_dong": [(ln, l.strip()) for ln, l in grep_lines(samp_py, r'guidance_scale')],
                             "so_anh_sinh": len(ts), "so_lo_mtime_khac_nhau": len(uniq), "khoang_cach_giua_cac_lo_giay": gaps,
                             "giay_moi_anh_theo_khoang_cach_lo (batch=4)": round(float(np.mean(gaps)) / 4, 2) if gaps else None,
                             "giay_moi_anh_min_max": [round(min(gaps) / 4, 2), round(max(gaps) / 4, 2)] if gaps else None}
    print(f"[3] p03 đặt guidance_scale? {res['3_tham_so_sinh']['p03_co_dat_guidance_scale']} | parser default: " +
          ", ".join(f"{k}={v['default']}(dòng {v['line']})" for k, v in defaults.items()) +
          f" | p02 đặt: {res['3_tham_so_sinh']['p02_dat_guidance_scale']} | {len(ts)} ảnh / {len(uniq)} lô, khoảng cách lô {gaps} s -> {res['3_tham_so_sinh']['giay_moi_anh_theo_khoang_cach_lo (batch=4)']} s/ảnh", flush=True)

    # ---------------------------------------------------------------- 4. thống kê p03 (n=10)
    n = len(det)
    A = {k: np.array([r[f"acc_{k}"] for r in det]) for k in ("font", "fd", "gen")}
    st = {}
    for k in A:
        st[k] = {"k": int(A[k].sum()), "n": n, "top1_pct": round(float(A[k].mean() * 100), 1), "wilson95": wilson(int(A[k].sum()), n)}
    b = int(((A["gen"] == 1) & (A["font"] == 0)).sum()); c = int(((A["gen"] == 0) & (A["font"] == 1)).sum())
    cosd = np.array([r["cos_gen"] - r["cos_font"] for r in det]); mard = np.array([r["margin_gen"] - r["margin_font"] for r in det])
    res["4_thong_ke_p03"] = {"top1": st, "mcnemar_gen_vs_font": {"o_gen_dung_font_sai": b, "o_gen_sai_font_dung": c, "p_chinh_xac_hai_phia": round(exact_p(b, c), 4)},
                             "cos_dung_gen_tru_font": {"n_duong": int((cosd > 0).sum()), "n_am": int((cosd < 0).sum()), "tb": round(float(cosd.mean()), 4), "p_dau_hai_phia": round(exact_p(int((cosd > 0).sum()), int((cosd < 0).sum())), 4)},
                             "margin_gen_tru_font": {"n_duong": int((mard > 0).sum()), "n_am": int((mard < 0).sum()), "tb": round(float(mard.mean()), 4), "p_dau_hai_phia": round(exact_p(int((mard > 0).sum()), int((mard < 0).sum())), 4)},
                             "so_o_chon_12_nhung_con_lai": n, "ghi_chu": "p03:77 chọn 12 ô; chỉ 10 ô có chữ đúng trong top-3 theo f_vW (p03:86-89)"}
    p02 = json.loads((M11 / "p02_B34/ket_qua.json").read_text(encoding="utf-8"))
    res["4_thong_ke_p03"]["p02_B34_n189_top8"] = {"f_font": p02["top1_trong_top_k"]["f_font"], "f_fd": p02["top1_trong_top_k"]["f_fd"], "f_gen_t0": p02["top1_trong_top_k"]["f_gen_t0"], "f_gen_t2": p02["top1_trong_top_k"]["f_gen_t2"],
                                                  "chenh_gen_t2_tru_fd_CI95": p02["chenh_lech_CI95"]["f_gen_t2 − f_fd"], "CFG_p02": 2.0}
    print(f"[4] p03 n={n}: font {st['font']['k']}/{n} {st['font']['wilson95']} | 'FD chung' {st['fd']['k']}/{n} | gen {st['gen']['k']}/{n} {st['gen']['wilson95']} | McNemar gen-vs-font b={b} c={c} p={res['4_thong_ke_p03']['mcnemar_gen_vs_font']['p_chinh_xac_hai_phia']} "
          f"| margin gen−font: {int((mard > 0).sum())} dương / {int((mard < 0).sum())} âm (tb {mard.mean():+.4f}) p_dấu={res['4_thong_ke_p03']['margin_gen_tru_font']['p_dau_hai_phia']}", flush=True)

    # ---------------------------------------------------------------- 5. p04 vs p05
    p04 = json.loads((M11 / "p04_he_quy_chieu/ket_qua_cac_bo.json").read_text(encoding="utf-8"))
    p05 = {r["book"]: r for r in json.loads((M11 / "p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
    tab = {}
    for b in ("B34", "B18", "L16", "TK"):
        a, e = p04[b], p05[b]
        t4 = [a["hqc1_white_canvas"]["top1"], a["hqc2_book_paper_canvas"]["top1"], a["hqc3_canonical_normalized"]["top1"]]
        t5 = [e["hqc1_white_canvas"]["top1"], e["hqc2_book_paper_canvas"]["top1"], e["hqc3_canonical_normalized"]["top1"]]
        rk = lambda t: "".join(map(str, np.argsort(-np.array(t), kind="stable") + 1))
        tab[b] = {"p04": {"n": a["n_cells"], "HQC1_2_3": t4, "thu_tu_giam": rk(t4)}, "p05": {"n": e["n_cells"], "HQC1_2_3": t5, "thu_tu_giam": rk(t5)},
                  "chenh_p05_tru_p04": [round(y - x, 1) for x, y in zip(t4, t5)]}
    res["5_p04_vs_p05"] = tab
    for b, v in tab.items():
        print(f"[5] {b}: p04 n={v['p04']['n']} {v['p04']['HQC1_2_3']} | p05 n={v['p05']['n']} {v['p05']['HQC1_2_3']} | chênh {v['chenh_p05_tru_p04']}", flush=True)

    (OUT_DIR / "p03_kiem_tra.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Đã ghi {OUT_DIR / 'p03_kiem_tra.json'}")


if __name__ == "__main__":
    main()
