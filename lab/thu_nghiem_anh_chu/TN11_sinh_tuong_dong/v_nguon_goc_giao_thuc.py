"""TN11 v_nguon_goc_giao_thuc (03/10) — KIỂM ĐỊNH ĐỘC LẬP giao thức của p04/p05 (phiên khác). 0 mô hình, 0 API, chỉ ĐỌC.

Trả lời bằng SỐ (script sinh) cho các câu hỏi:
  A. Mỗi bộ: số ô đích, "sự thật" lấy từ đâu (nhãn người hay cột label của pipeline), tỉ lệ chữ đúng nằm trong top-3 theo f_vW,
     Top-1 của chính f_vW (bộ chọn đã học) trên cùng tập, số ứng viên/ô, tỉ lệ ô đa-sự-thật, với bộ nhãn pipeline: label==ocr_char.
  B. `hash(book) % 100000` của p05: trị số theo từng tiến trình + độ chồng lấn tập ô chọn giữa các lần chạy (Jaccard); cố định PYTHONHASHSEED thì tất định.
  C. Hồ sơ mẫu giấy: trang nào, mấy trang, Otsu suy biến trên ảnh nhị phân -> ink_median = HẰNG SỐ 60.0 mặc định (stt2/4/11)?, patch giấy thật hay giả lập,
     độ ổn định qua nhiều trang.
  D. Ô neo tự động (quy tắc p02:46-50): độ chính xác so với nhãn người trên 4 bộ có sự thật; `meta[:,3]` là cờ ok hay mực?
Ra: measure_out/_tn11/verify/nguon_goc/giao_thuc.json

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_nguon_goc_giao_thuc.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT_DIR = REPO / "measure_out" / "_tn11" / "verify" / "nguon_goc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = T.OUT / "emb"
CAND_DIR = T.OUT / "cand"
SELF = Path(__file__).resolve()
P05_JSON = REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json"


# ------------------------------------------------------------------------------------------------------------------
# p05 dòng 176-201 (chọn ô) — sao chép nguyên logic, seed truyền vào
# ------------------------------------------------------------------------------------------------------------------
def load_target(book):
    D = T.load_base(book)
    F = pd.read_pickle(CAND_DIR / f"{book}.pkl")
    if "y" in F.columns and F.y.sum() > 0:
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        target_cells = list(tru.index)
        is_human = True
    else:
        lbl = D.label.values
        tgt = lbl[F.i.values]
        valid_mask = (F.c.values == tgt) & (tgt != "")
        target_cells = sorted(set(F.i.values[valid_mask]))
        F["y"] = np.where(valid_mask, 1, 0)
        is_human = False
    return D, F, target_cells, is_human


def p05_select(F, target_cells, seed, n_sample=20, k_cand=3):
    rng = np.random.default_rng(seed)
    selected_cells = np.sort(rng.choice(target_cells, size=min(n_sample, len(target_cells)), replace=False))
    S = F[F.i.isin(selected_cells)].copy()
    S["rk"] = S.f_vW.fillna(-9) if "f_vW" in S.columns else S.f_font.fillna(-9)
    S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(k_cand)
    has_truth = S.groupby("i").y.max()
    valid_cells = sorted(has_truth[has_truth == 1].index)
    return [int(x) for x in selected_cells], [int(x) for x in valid_cells]


def emit_selection(book):
    """Chạy như p05: seed = hash(book) % 100000 (hash ngẫu nhiên hoá theo tiến trình trừ khi PYTHONHASHSEED cố định)."""
    D, F, tc, _ = load_target(book)
    seed = hash(book) % 100000
    sel, val = p05_select(F, tc, seed)
    print(json.dumps({"book": book, "seed": seed, "selected": sel, "valid": val, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}))


def jaccard(a, b):
    a, b = set(a), set(b)
    return round(len(a & b) / max(1, len(a | b)), 3)


def part_b_hash():
    out = {"hash_gia_tri_theo_tien_trinh": {}, "chon_o_B34_theo_tien_trinh": [], "co_dinh_PYTHONHASHSEED": []}
    codes = []
    for _ in range(5):
        r = subprocess.run([sys.executable, "-c", "print(hash('B18')%100000, hash('B34')%100000, hash('L16')%100000, hash('TK')%100000)"],
                           capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"})
        codes.append([int(x) for x in r.stdout.split()])
    for j, b in enumerate(("B18", "B34", "L16", "TK")):
        out["hash_gia_tri_theo_tien_trinh"][b] = [c[j] for c in codes]
    runs = []
    for _ in range(3):
        r = subprocess.run([sys.executable, str(SELF), "--emit-selection", "B34"], capture_output=True, text=True,
                           env={k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"})
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    out["chon_o_B34_theo_tien_trinh"] = [{"seed": r["seed"], "n_valid": len(r["valid"]), "valid": r["valid"]} for r in runs]
    out["jaccard_cap_valid_B34"] = {f"{i}-{j}": jaccard(runs[i]["valid"], runs[j]["valid"]) for i in range(3) for j in range(i + 1, 3)}
    out["jaccard_cap_selected20_B34"] = {f"{i}-{j}": jaccard(runs[i]["selected"], runs[j]["selected"]) for i in range(3) for j in range(i + 1, 3)}
    fixed = []
    for _ in range(2):
        r = subprocess.run([sys.executable, str(SELF), "--emit-selection", "B34"], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "0"})
        fixed.append(json.loads(r.stdout.strip().splitlines()[-1]))
    out["co_dinh_PYTHONHASHSEED"] = {"seed": [f["seed"] for f in fixed], "valid_giong_nhau": fixed[0]["valid"] == fixed[1]["valid"]}
    return out


# ------------------------------------------------------------------------------------------------------------------
# A. giao thức theo bộ
# ------------------------------------------------------------------------------------------------------------------
def part_a_protocol():
    p05 = {r["book"]: r for r in json.loads(P05_JSON.read_text(encoding="utf-8"))}
    p01 = {}
    for b in ("B18", "B34", "L16", "TK"):
        p01[b] = json.loads((REPO / f"measure_out/_tn11/p01_{b}.json").read_text(encoding="utf-8"))
    res = {}
    for b in T.ORDER:
        t0 = time.time()
        D, F, tc, is_human = load_target(b)
        Ft = F[F.i.isin(tc)][["i", "c", "y", "f_vW", "f_font", "f_fd"]].copy()
        Ft = Ft.sort_values(["i", "f_vW"], ascending=[True, False])
        Ft["r"] = Ft.groupby("i").cumcount()
        g_all = Ft.groupby("i")
        n_cand_target = float(g_all.size().mean())
        top3 = Ft[Ft.r < 3]
        p_in3 = float(top3.groupby("i").y.max().mean())
        vw1 = float(Ft[Ft.r == 0].set_index("i").y.mean())
        multi_all = float((g_all.y.sum() >= 2).mean())
        multi_top3 = float((top3.groupby("i").y.sum() >= 2).mean())
        # Top-1 của f_vW chỉ trên các ô có chữ đúng trong top-3 (đúng điều kiện lọc của p05)
        in3 = top3.groupby("i").y.max()
        vw1_given_in3 = float(Ft[(Ft.r == 0) & Ft.i.isin(in3[in3 == 1].index)].set_index("i").y.mean())
        row = {"book": b, "n_D": int(len(D)), "n_cells_F": int(F.i.nunique()), "truth_source": "nhan_nguoi (F.y từ gt_char)" if is_human else "D.label của pipeline (khớp F.c == D.label, label != '')",
               "n_target_cells": int(len(tc)), "mean_cand_per_cell_all": round(float(len(F) / F.i.nunique()), 2), "mean_cand_per_target_cell": round(n_cand_target, 2),
               "P_truth_in_top3_by_f_vW": round(p_in3 * 100, 1), "f_vW_top1_over_all_cands_pct": round(vw1 * 100, 1),
               "f_vW_top1_given_truth_in_top3_pct": round(vw1_given_in3 * 100, 1),
               "multi_truth_cells_all_cands_pct": round(multi_all * 100, 1), "multi_truth_cells_top3_pct": round(multi_top3 * 100, 1),
               "chance_top1_3cand_pct": 33.3, "chance_top1_all_cands_pct": round(100 / n_cand_target, 1),
               "p05_n_cells_recorded": p05[b]["n_cells"], "p05_expected_n_if_20_selected": round(20 * p_in3, 1)}
        if not is_human:
            Dt = D.iloc[np.array(tc)]
            row["label_eq_ocr_char_pct_among_target"] = round(float((Dt.label == Dt.ocr_char).mean() * 100), 1)
            row["tier_among_target"] = {k: int(v) for k, v in Dt.tier.value_counts().head(5).items()}
            row["label_nonempty_pct_of_D"] = round(float((D.label != "").mean() * 100), 1)
        else:
            row["p01_f_font_top1_pct_26cand"] = p01[b]["top1"]["f_font"]["tat_ca"]
            row["p01_f_fd_top1_pct_26cand"] = p01[b]["top1"]["f_fd"]["tat_ca"]
            row["f_fd_NaN_pct_of_all_candidate_rows"] = round(float(F.f_fd.isna().mean() * 100), 1)
            row["f_fd_NaN_pct_of_truth_rows"] = round(float(F[F.y == 1].f_fd.isna().mean() * 100), 1)
        res[b] = row
        print(f"[A] {b:6s} target={row['n_target_cells']:6d} cand/ô={row['mean_cand_per_target_cell']:5.1f} P(đúng∈top3)={row['P_truth_in_top3_by_f_vW']:5.1f}% "
              f"f_vW@1(tất cả)={row['f_vW_top1_over_all_cands_pct']:5.1f}% f_vW@1|top3={row['f_vW_top1_given_truth_in_top3_pct']:5.1f}% đa-sự-thật(top3)={row['multi_truth_cells_top3_pct']}% "
              f"p05 n={row['p05_n_cells_recorded']} (kỳ vọng {row['p05_expected_n_if_20_selected']}) [{time.time() - t0:.0f}s]", flush=True)
    return res


# ------------------------------------------------------------------------------------------------------------------
# C. hồ sơ mẫu giấy (copy p05 dòng 73-124 có ghi thêm chẩn đoán)
# ------------------------------------------------------------------------------------------------------------------
def paper_model_diag(png: Path):
    page_img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    T_val, _ = cv2.threshold(page_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper_mask = page_img >= T_val
    paper_pixels, ink_pixels = page_img[paper_mask], page_img[~paper_mask]
    ink_default = len(ink_pixels) == 0
    bg = float(np.median(paper_pixels)) if len(paper_pixels) else 240.0
    ink = float(np.median(ink_pixels)) if len(ink_pixels) else 60.0
    h, w = page_img.shape
    min_ink, found = 999999, False
    for y in range(20, h - 148, 32):
        for x in range(20, w - 148, 32):
            n_ink = int(np.sum(page_img[y:y + 128, x:x + 128] < T_val))
            if n_ink < min_ink:
                min_ink, found = n_ink, True
            if min_ink == 0:
                break
        if min_ink == 0:
            break
    fallback = (not found) or min_ink > 200
    return {"otsu_T": float(T_val), "n_unique_gray": int(len(np.unique(page_img))), "frac_pixels_below_T": round(float((~paper_mask).mean()), 4),
            "bg": round(bg, 1), "ink": round(ink, 1), "contrast": round(bg - ink, 1), "ink_median_is_default_60": bool(ink_default),
            "patch_min_ink_px": (None if not found else min_ink), "patch_is_synthetic_gaussian": bool(fallback), "shape": [int(h), int(w)]}


def part_c_paper():
    p05 = {r["book"]: r for r in json.loads(P05_JSON.read_text(encoding="utf-8"))}
    res = {}
    for b in T.ORDER:
        cfg = T.BOOKS[b]
        pages = sorted((T.REPO / cfg["prep"] / "pages").glob("*.png"))
        idx = min(10, len(pages) - 1)
        samp = paper_model_diag(pages[idx])
        # nhiều trang: 12 trang cách đều
        sel = sorted(set(np.linspace(0, len(pages) - 1, 12).round().astype(int).tolist()))
        multi = [dict(page=pages[i].name, **paper_model_diag(pages[i])) for i in sel]
        pj = p05[b]["paper_stats"]
        res[b] = {"n_pages_in_dir": len(pages), "n_pages_used_by_p05": 1, "p05_page_index": idx, "p05_page": pages[idx].name, "sample_page_diag": samp,
                  "khop_JSON_p05": bool(abs(samp["bg"] - pj["bg"]) < 0.06 and abs(samp["ink"] - pj["ink"]) < 0.06 and abs(samp["contrast"] - pj["contrast"]) < 0.06),
                  "p05_json_paper_stats": pj,
                  "multi_page_12": {"bg": [min(m["bg"] for m in multi), max(m["bg"] for m in multi)], "ink": [min(m["ink"] for m in multi), max(m["ink"] for m in multi)],
                                    "contrast": [min(m["contrast"] for m in multi), max(m["contrast"] for m in multi)],
                                    "n_pages_ink_default": int(sum(m["ink_median_is_default_60"] for m in multi)),
                                    "n_pages_patch_synthetic": int(sum(m["patch_is_synthetic_gaussian"] for m in multi)),
                                    "n_pages_binary_2_levels": int(sum(m["n_unique_gray"] <= 2 for m in multi))}}
        print(f"[C] {b:6s} {len(pages):4d} trang; p05 dùng 1 trang {pages[idx].name}: bg/ink/contrast={samp['bg']}/{samp['ink']}/{samp['contrast']} | Otsu_T={samp['otsu_T']} "
              f"n_gray={samp['n_unique_gray']} ink_mặc_định_60={samp['ink_median_is_default_60']} patch_giả={samp['patch_is_synthetic_gaussian']} "
              f"| 12 trang: bg {res[b]['multi_page_12']['bg']} ink {res[b]['multi_page_12']['ink']} contrast {res[b]['multi_page_12']['contrast']} "
              f"ink_mặc_định {res[b]['multi_page_12']['n_pages_ink_default']}/12 patch_giả {res[b]['multi_page_12']['n_pages_patch_synthetic']}/12", flush=True)
    return res


# ------------------------------------------------------------------------------------------------------------------
# D. ô neo tự động (p02:46-50) — độ chính xác so nhãn người
# ------------------------------------------------------------------------------------------------------------------
def part_d_anchors():
    C = T.lex()
    res = {}
    for b in ("B34", "B18", "L16", "TK"):
        D = T.load_base(b)
        meta = np.load(EMB_DIR / f"{b}_crop_meta.npy")
        Rm = {s: C.R_of(s) for s in set(D.syllable)}
        base = np.array((D.rule.str.startswith("s1_inter_s2_direct") & (D.label != "") & (D.label == D.ocr_char)).values, bool)
        inR = np.array([l in Rm[s] for l, s in zip(D.label, D.syllable)])
        okflag = meta[:, 3] > 0                       # cột 3 của crop_meta = cờ ok (t02_embed.py:164), KHÔNG phải lượng mực
        inkfrac = meta[:, 2]                          # cột 2 = tỉ lệ điểm mực (gray<128)
        anc = base & inR & okflag
        anc_ink = base & inR & (inkfrac > 0)
        F = pd.read_pickle(CAND_DIR / f"{b}.pkl")[["i", "c", "y"]]
        tru = F[F.y == 1]
        truth_set = set(zip(tru.i.values.tolist(), tru.c.values.tolist()))
        cells_with_truth = set(tru.i.unique().tolist())
        ai = np.nonzero(anc)[0]
        has_gt = np.array([i in cells_with_truth for i in ai])
        ok_y = np.array([(int(i), D.label.values[i]) in truth_set for i in ai])
        strict = np.array([D.gt_char.values[i] != "" and D.gt_char.values[i] == D.label.values[i] for i in ai])
        has_gtchar = np.array([D.gt_char.values[i] != "" for i in ai])
        # theo tầng / quy tắc: quy tắc p02 dùng startswith('s1_inter_s2_direct') nên gồm cả 's1_inter_s2_direct|gate:...' (ô REVIEW bị cổng giữ)
        tier = D.tier.values
        by_tier = {}
        for tname in ("GOLD", "REVIEW", "SYLLABLE", "QUARANTINE", "GOLD_text_only"):
            m = np.array([tier[i] == tname for i in ai])
            mm = m & has_gt
            if mm.sum() > 0:
                by_tier[tname] = {"n_anchor": int(m.sum()), "n_co_su_that_nguoi": int(mm.sum()), "chinh_xac_pct": round(float(ok_y[mm].mean() * 100), 2)}
        exact_rule = np.array([D.rule.values[i] == "s1_inter_s2_direct" for i in ai])
        gold_exact = np.array([tier[i] == "GOLD" for i in ai]) & exact_rule & has_gt
        res[b] = {"by_tier": by_tier, "GOLD_va_quy_tac_chinh_xac_s1_inter_s2_direct": {"n": int(gold_exact.sum()), "chinh_xac_pct": (round(float(ok_y[gold_exact].mean() * 100), 2) if gold_exact.any() else None)},
                  "n_D": int(len(D)), "n_anchor": int(anc.sum()), "n_anchor_if_ink_gt_0": int(anc_ink.sum()), "meta_col3_is_ok_flag_all_ones_on_cells": bool(np.all(np.isin(meta[:, 3], [0, 1]))),
                  "anchor_cells_with_human_truth_in_cands": int(has_gt.sum()),
                  "precision_vs_human_F_y": (round(float(ok_y[has_gt].mean() * 100), 2) if has_gt.any() else None),
                  "anchor_cells_with_gt_char": int(has_gtchar.sum()),
                  "precision_strict_label_eq_gt_char": (round(float(strict[has_gtchar].mean() * 100), 2) if has_gtchar.any() else None),
                  "n_wrong_vs_F_y": int((~ok_y[has_gt]).sum()) if has_gt.any() else None}
        print(f"[D] {b:4s} neo={res[b]['n_anchor']:6d} (nếu 'mực>0' ở cột 2: {res[b]['n_anchor_if_ink_gt_0']}) | có sự thật người {res[b]['anchor_cells_with_human_truth_in_cands']} "
              f"| chính xác theo F.y {res[b]['precision_vs_human_F_y']}% | nghiêm ngặt label==gt_char {res[b]['precision_strict_label_eq_gt_char']}% (n={res[b]['anchor_cells_with_gt_char']}) "
              f"| theo tầng {res[b]['by_tier']} | GOLD & quy tắc chính xác {res[b]['GOLD_va_quy_tac_chinh_xac_s1_inter_s2_direct']}", flush=True)
    return res


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--emit-selection":
        emit_selection(sys.argv[2]); return
    out = {}
    out["B_hash"] = part_b_hash()
    print("[B]", json.dumps({k: v for k, v in out["B_hash"].items() if k not in ("chon_o_B34_theo_tien_trinh",)}, ensure_ascii=False))
    out["C_ho_so_giay"] = part_c_paper()
    out["D_o_neo"] = part_d_anchors()
    out["A_giao_thuc"] = part_a_protocol()
    (OUT_DIR / "giao_thuc.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Đã ghi {OUT_DIR / 'giao_thuc.json'}")


if __name__ == "__main__":
    main()
