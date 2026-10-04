"""w_hien_trang_du_lieu_03_tn8_cu.py — ĐỘ MỚI: bảng TN8 (measure_out/_tn8/base|cand|emb, dựng 30/09) so với nhãn HIỆN TẠI (đợt dựng lại 01/10).

Với mỗi cuốn: số dòng / khoá ô (page, cột, nom_idx, syl_idx) / thứ tự hàng (nhúng TN8 theo thứ tự hàng base) / nhãn, tầng, luật, hộp khác nhau;
ô neo TN8 (cand/<b>_cells.pkl: anchor) so với ô neo hiện tại; danh sách chữ ứng viên TN8 (cand/<b>.pkl: c) so với hiện tại (script 01).
Ra: measure_out/_tn11/full/hien_trang_du_lieu/tn8_cu.json. Chỉ ĐỌC measure_out/_tn8/ (không ghi). 0 API/GPU.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_03_tn8_cu.py [--books B34 ...]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import w_hien_trang_du_lieu_lib as L  # noqa: E402


def key(df):
    return (df.page + "|" + df.column.astype(int).astype(str) + "|" + df.nom_idx.astype(str) + "|" + df.syl_idx.astype(str)).to_numpy()


def book(ma):
    t0 = time.time()
    C = L.CFG[ma]
    b = pd.read_pickle(L.TN8 / "base" / f"{ma}.pkl").reset_index(drop=True)
    c = L.load_all(ma)
    inv = L.Inv()
    R = dict(ma=ma)
    R["mtime_base_tn8"] = L.mt(L.TN8 / "base" / f"{ma}.pkl")
    R["mtime_cand_tn8"] = L.mt(L.TN8 / "cand" / f"{ma}.pkl")
    R["mtime_emb_tn8"] = L.mt(L.TN8 / "emb" / f"{ma}_enc.npy")
    R["mtime_labels_hien_tai"] = L.mt(L.REPO / C["all_file"])
    R["n_base"] = int(len(b)); R["n_hien_tai"] = int(len(c))
    kb, kc = key(b), key(c)
    inv.check("khoa_o_duy_nhat_base", len(set(kb)) == len(kb), f"{len(kb) - len(set(kb))} trùng")
    inv.check("khoa_o_duy_nhat_hien_tai", len(set(kc)) == len(kc), f"{len(kc) - len(set(kc))} trùng")
    sb, sc = set(kb), set(kc)
    R["khoa_chung"] = len(sb & sc); R["chi_o_base"] = len(sb - sc); R["chi_o_hien_tai"] = len(sc - sb)
    R["cung_thu_tu_hang"] = bool(len(kb) == len(kc) and (kb == kc).all())
    # số hàng nhúng/crop_meta của TN8
    enc = np.load(L.TN8 / "emb" / f"{ma}_enc.npy", mmap_mode="r")
    meta = np.load(L.TN8 / "emb" / f"{ma}_crop_meta.npy")
    R["n_nhung_tn8"] = int(enc.shape[0]); R["n_crop_meta_tn8"] = int(meta.shape[0])
    inv.check("nhung_TN8_khop_so_hang_base", enc.shape[0] == len(b) == meta.shape[0], f"{enc.shape[0]} / {len(b)} / {meta.shape[0]}")
    # so sánh theo khoá chung
    cc = c.copy(); cc["_k"] = kc
    bb = b.copy(); bb["_k"] = kb
    M = bb[["_k", "ocr_char", "label", "tier", "rule", "bbox", "syllable", "image"]].merge(
        cc[["_k", "ocr_char", "label", "tier", "rule", "bbox", "syllable", "image"]], on="_k", suffixes=("_b", "_c"))
    n = max(len(M), 1)
    R["giong_nhau_tren_khoa_chung"] = {k: round(float((M[k + "_b"] == M[k + "_c"]).mean()), 4) for k in ("ocr_char", "label", "tier", "rule", "bbox", "syllable", "image")}
    R["so_khac_nhau_tren_khoa_chung"] = {k: int((M[k + "_b"] != M[k + "_c"]).sum()) for k in ("ocr_char", "label", "tier", "rule", "bbox", "syllable", "image")}
    # IoU hộp base vs hiện tại (khoá chung)
    def _b(v):
        try:
            x = json.loads(v)
            return [float(t) for t in x[:4]]
        except Exception:  # noqa: BLE001
            return [np.nan] * 4
    A = np.array([_b(v) for v in M.bbox_b]); Bq = np.array([_b(v) for v in M.bbox_c])
    ix = np.maximum(0, np.minimum(A[:, 2], Bq[:, 2]) - np.maximum(A[:, 0], Bq[:, 0])); iy = np.maximum(0, np.minimum(A[:, 3], Bq[:, 3]) - np.maximum(A[:, 1], Bq[:, 1]))
    inter = ix * iy
    iou = inter / np.maximum((A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]) + (Bq[:, 2] - Bq[:, 0]) * (Bq[:, 3] - Bq[:, 1]) - inter, 1e-9)
    iou = iou[~np.isnan(iou)]
    R["iou_hop"] = dict(n=int(len(iou)), giong_het=round(float((iou > 0.9999).mean()), 4), median=round(float(np.median(iou)), 4), p10=round(float(np.percentile(iou, 10)), 4),
                        ti_le_iou_lt_0_5=round(float((iou < 0.5).mean()), 4), ti_le_iou_lt_0_8=round(float((iou < 0.8).mean()), 4))
    # chuyển tầng
    tr = collections.Counter(zip(M.tier_b, M.tier_c))
    R["chuyen_tang_top"] = {f"{a}->{b_}": v for (a, b_), v in tr.most_common(12) if a != b_}
    R["tang_base"] = b.tier.value_counts().to_dict(); R["tang_hien_tai"] = c.tier.value_counts().to_dict()
    # ô neo
    cells = pd.read_pickle(L.TN8 / "cand" / f"{ma}_cells.pkl")
    a8 = cells["anchor"].to_numpy(bool)
    inv.check("cells_TN8_cung_so_hang_base", len(cells) == len(b), f"{len(cells)} vs {len(b)}")
    A8 = pd.DataFrame(dict(_k=kb[a8], label=b.label.to_numpy()[a8]))
    Ac = c[L.anchor_mask(c)]
    Ac = pd.DataFrame(dict(_k=key(Ac), label=Ac.label.to_numpy()))
    m = A8.merge(Ac, on="_k", how="outer", suffixes=("_8", "_c"), indicator=True)
    both = m[m._merge == "both"]
    R["neo"] = dict(
        n_tn8=int(len(A8)), n_hien_tai=int(len(Ac)), chung_cung_nhan=int((both.label_8 == both.label_c).sum()), chung_khac_nhan=int((both.label_8 != both.label_c).sum()),
        chi_tn8=int((m._merge == "left_only").sum()), chi_hien_tai=int((m._merge == "right_only").sum()))
    c8 = collections.Counter(A8.label); cc_ = collections.Counter(Ac.label)
    R["neo"].update(chu_tn8=len(c8), chu_hien_tai=len(cc_), chu_ge5_tn8=int(sum(v >= 5 for v in c8.values())), chu_ge5_hien_tai=int(sum(v >= 5 for v in cc_.values())),
                    chu_ge3_tn8=int(sum(v >= 3 for v in c8.values())), chu_ge3_hien_tai=int(sum(v >= 3 for v in cc_.values())),
                    chu_chi_tn8=len(set(c8) - set(cc_)), chu_chi_hien_tai=len(set(cc_) - set(c8)))
    R["neo"]["ti_le_neo_tn8_con_nguyen"] = round(R["neo"]["chung_cung_nhan"] / max(R["neo"]["n_tn8"], 1), 4)
    # chữ ứng viên
    F = pd.read_pickle(L.TN8 / "cand" / f"{ma}.pkl", )
    C8 = set(F["c"].unique()); nF = int(len(F))
    del F
    cur_txt = L.OUT / "chu_ung_vien" / f"{ma}.txt"
    if cur_txt.exists():
        Cc = set(cur_txt.read_text(encoding="utf-8").split("\n")) - {""}
        R["chu_ung_vien"] = dict(n_cap_o_x_chu_tn8=nF, n_chu_tn8=len(C8), n_chu_hien_tai=len(Cc), chung=len(C8 & Cc), chi_tn8=len(C8 - Cc), chi_hien_tai=len(Cc - C8),
                                 ti_le_chu_hien_tai_da_co_o_tn8=round(len(C8 & Cc) / max(len(Cc), 1), 4))
        inv.check("chu_ung_vien_tn8_la_tap_con_cua_hien_tai_hoac_gan", len(C8 - Cc) <= 0.02 * len(C8), f"chỉ ở TN8 {len(C8 - Cc)} / {len(C8)}")
    # kết luận cho cuốn
    g = R["giong_nhau_tren_khoa_chung"]
    cu = (not R["cung_thu_tu_hang"]) or g["label"] < 0.9999 or g["tier"] < 0.9999 or g["bbox"] < 0.9999 or R["chi_o_base"] or R["chi_o_hien_tai"]
    R["bang_tn8_cu"] = bool(cu)
    why = []
    if R["chi_o_base"] or R["chi_o_hien_tai"]:
        why.append(f"khoá ô lệch (chỉ base {R['chi_o_base']}, chỉ hiện tại {R['chi_o_hien_tai']})")
    if not R["cung_thu_tu_hang"]:
        why.append("thứ tự hàng khác (nhúng TN8 theo thứ tự hàng base không còn khớp)")
    if g["tier"] < 0.9999:
        why.append(f"{R['so_khac_nhau_tren_khoa_chung']['tier']} ô đổi tầng")
    if g["label"] < 0.9999:
        why.append(f"{R['so_khac_nhau_tren_khoa_chung']['label']} ô đổi nhãn")
    if g["bbox"] < 0.9999:
        why.append(f"{R['so_khac_nhau_tren_khoa_chung']['bbox']} ô đổi hộp (IoU median {R['iou_hop']['median']}; IoU<0,5 ở {R['iou_hop']['ti_le_iou_lt_0_5']:.1%}) — crop nhúng cũ")
    if g["rule"] < 0.9999:
        why.append(f"{R['so_khac_nhau_tren_khoa_chung']['rule']} ô đổi luật")
    R["ly_do_cu"] = why
    R["bat_bien"] = inv.summary(); R["bat_bien_chi_tiet"] = inv.rows
    R["giay"] = round(time.time() - t0, 1)
    print(f"[03] {ma}: base {len(b)} hiện tại {len(c)} khoá chung {R['khoa_chung']} cùng thứ tự {R['cung_thu_tu_hang']} | tầng khác {R['so_khac_nhau_tren_khoa_chung']['tier']} "
          f"nhãn khác {R['so_khac_nhau_tren_khoa_chung']['label']} hộp khác {R['so_khac_nhau_tren_khoa_chung']['bbox']} | neo tn8 {R['neo']['n_tn8']} hiện tại {R['neo']['n_hien_tai']} "
          f"nguyên {R['neo']['chung_cung_nhan']} | CŨ={R['bang_tn8_cu']} [{R['giay']}s]", flush=True)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", default=L.ORDER)
    a = ap.parse_args()
    out = {}
    for ma in a.books:
        out[ma] = book(ma)
    p = L.OUT / ("tn8_cu.json" if a.books == L.ORDER else "tn8_cu_" + "_".join(a.books) + ".json")
    L.jdump(out, p)
    print("->", L.rel(p))


if __name__ == "__main__":
    main()
