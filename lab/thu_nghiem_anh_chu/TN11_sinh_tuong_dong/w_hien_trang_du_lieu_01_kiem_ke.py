"""w_hien_trang_du_lieu_01_kiem_ke.py — KIỂM KÊ dữ liệu MỚI NHẤT của 10 cuốn cho gói Kaggle "sinh ảnh theo phong cách từng cuốn".

Với mỗi cuốn: tệp nhãn + crop + trang nguồn; số dòng theo tầng; số ô có tệp crop THẬT (kiểm tồn tại); kích thước/chế độ ảnh crop;
ô neo tự động (rule s1_inter_s2_direct* ∧ label == ocr_char ∧ label ∈ R(âm)) CÓ tệp crop: số chữ khác nhau, chữ ≥ 3 / ≥ 5 ô neo, tỉ lệ ô thuộc
chữ hiếm (< 2 ô neo); gold_exact (ok/text_only/uncertified/review) và crop chuẩn; danh sách chữ ứng viên (R(âm) ∪ kim [∪ lt1/lt2 của STT]).
Ra: measure_out/_tn11/full/hien_trang_du_lieu/{kiem_ke.json, neo/<ma>.csv, chu_ung_vien/<ma>.{txt,tsv}}. 0 API/GPU.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_01_kiem_ke.py [--books B34 ...]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import w_hien_trang_du_lieu_lib as L  # noqa: E402

WORKERS = 8


def pmap(fn, items):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, items, chunksize=256))


def img_info(path):
    """(w, h, mode) đọc phần đầu tệp PNG; None nếu không có/hỏng."""
    try:
        with Image.open(path) as im:
            return (im.size[0], im.size[1], im.mode)
    except Exception:  # noqa: BLE001
        return None


def exists(path):
    return os.path.isfile(path)


def kim_chars_stt(ma):
    """STT: mọi chữ trong cache kim hiện hành (kim_l1skel_l2: char/lt1/lt2 của từng hộp) — thay cho lt2 theo ô (không lưu cho ô không GOLD)."""
    d = L.REPO / L.CFG[ma]["prep"] / "kim_l1skel_l2"
    S = collections.Counter()
    n = 0
    for f in sorted(d.glob("page_*_ocr_cache.json")):
        try:
            z = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        n += 1
        for col in z.get("columns", []):
            for b in col:
                for k in ("char", "lt1", "lt2"):
                    v = b.get(k, "")
                    if v and len(v) == 1:
                        S[v] += 1
    return S, n


def book(ma, a_sizes=True):
    t0 = time.time()
    C = L.CFG[ma]
    root_all = L.REPO / C["all_root"]
    root_pub = L.REPO / "dataset" / C["ds"]
    D = L.load_all(ma)
    P = L.load_pub(ma)
    G = L.load_ge(ma)
    inv = L.Inv()
    R = dict(ma=ma, ten=C["name"], kind=C["kind"], crop_source=C["crop_source"])
    # ---- nguồn + tuổi
    prep = L.REPO / C["prep"]
    pages_dir = prep / "pages"
    R["nguon"] = dict(
        labels_pub=f"dataset/{C['ds']}/labels.csv" + (f" (lọc book=={C['book_val']})" if ma.startswith("stt") else ""),
        labels_trace=f"dataset/{C['ds']}/labels_trace.csv", gold_exact=f"dataset/{C['ds']}/gold_exact.csv" + (f" (lọc set8=={ma})" if ma.startswith("stt") else ""),
        labels_all=C["all_file"] + (f" (lọc book=={C['book_val']})" if ma.startswith("stt") else ""),
        crops_pub=f"dataset/{C['ds']}/gold/ (+ gold/chon_chu/ nếu có) , dataset/{C['ds']}/syllable/",
        crops_all=C["all_root"] + "/{gold,syllable}/", crops_copy_ALL=f"dataset/_ALL/crops/{C['ds']}/",
        crops_chuan=f"dataset/_ALL/crops_chuan/{C['ds']}/<book>/<page>/c<cột>_n<nom_idx>_s<syl_idx>.png (+ crops_chuan_128/)",
        pages=rel_p(pages_dir), config=C["yaml"], prepared=C["prep"])
    R["mtime"] = dict(labels_pub=L.mt(root_pub / "labels.csv"), labels_all=L.mt(L.REPO / C["all_file"]), gold_exact=L.mt(root_pub / "gold_exact.csv"),
                      labels_trace=L.mt(root_pub / "labels_trace.csv"))
    npg = len(list(pages_dir.glob("*.png"))) if pages_dir.exists() else 0
    R["so_trang_png"] = npg
    R["so_trang_trong_nhan"] = int(D.page.nunique())
    # ---- tầng
    R["so_dong_all"] = int(len(D)); R["so_dong_pub"] = int(len(P))
    R["tang_all"] = D.tier.value_counts().to_dict(); R["tang_pub"] = P.tier.value_counts().to_dict()
    inv.check("image_duy_nhat_trong_pub", P.image.is_unique, f"{P.image.duplicated().sum()} trùng")
    inv.check("image_khong_rong_o_pub", (P.image != "").all(), f"{(P.image == '').sum()} rỗng")
    inv.check("pub_la_tap_con_cua_all_theo_image", set(P.image) <= set(D.image[D.image != ""]), f"{len(set(P.image) - set(D.image))} ngoài")
    # ---- tồn tại tệp
    has_img_all = (D.image != "").to_numpy()
    paths_all = [str(root_all / im) if h else "" for im, h in zip(D.image, has_img_all)]
    ex_all = np.array(pmap(exists, [p if p else "/nonexistent" for p in paths_all]), bool) & has_img_all
    pub_set = set(P.image)
    in_pub = D.image.isin(pub_set).to_numpy()
    paths_pub_of_D = [str(root_pub / im) if h else "" for im, h in zip(D.image, in_pub)]
    ex_pub = np.array(pmap(exists, [p if p else "/nonexistent" for p in paths_pub_of_D]), bool) & in_pub
    D["_in_pub"] = in_pub; D["_ex_all"] = ex_all; D["_ex_pub"] = ex_pub
    tier_tab = {}
    for t, g in D.groupby("tier"):
        tier_tab[t] = dict(dong=int(len(g)), co_truong_image=int((g.image != "").sum()), ton_tai_o_build=int(g._ex_all.sum()),
                           trong_pub=int(g._in_pub.sum()), ton_tai_o_pub=int(g._ex_pub.sum()))
    R["crop_theo_tang"] = tier_tab
    R["so_o_co_anh_pub"] = int(D._ex_pub.sum()); R["so_o_co_anh_all"] = int(D._ex_all.sum())
    R["crop_thieu_pub"] = int((D._in_pub & ~D._ex_pub).sum()); R["crop_thieu_all"] = int((has_img_all & ~ex_all).sum())
    miss_t = D[D._in_pub & ~D._ex_pub].tier.value_counts().to_dict()
    R["crop_thieu_pub_theo_tang"] = miss_t
    inv.check("dong_pub_thieu_crop_chi_la_GOLD_text_only", set(miss_t) <= {"GOLD_text_only"}, f"thiếu theo tầng {miss_t}")
    inv.check("moi_GOLD_va_SYLLABLE_pub_co_tep_crop", int(((D.tier.isin(["GOLD", "SYLLABLE"])) & D._in_pub & ~D._ex_pub).sum()) == 0, "")
    inv.check("GOLD_text_only_co_crop_o_thu_muc_build", int(((D.tier == "GOLD_text_only") & ~D._ex_all).sum()) == 0,
              f"{int(((D.tier == 'GOLD_text_only') & ~D._ex_all).sum())} thiếu ở build")
    inv.check("image_duy_nhat_trong_all", D.image[D.image != ""].is_unique, f"{D.image[D.image != ''].duplicated().sum()} trùng")
    # ---- kích thước + chế độ (đọc đầu tệp): mọi tệp tồn tại (ưu tiên bản pub)
    D["_path"] = ["" for _ in range(len(D))]
    sel_pub = D._ex_pub.to_numpy(); sel_all_only = (D._ex_all & ~D._ex_pub).to_numpy()
    D.loc[sel_pub, "_path"] = np.array(paths_pub_of_D, dtype=object)[sel_pub]
    D.loc[sel_all_only, "_path"] = np.array(paths_all, dtype=object)[sel_all_only]
    D["_w"] = np.nan; D["_h"] = np.nan; D["_mode"] = ""
    if a_sizes:
        todo = D.index[D._path != ""].tolist()
        info = pmap(img_info, [D._path[i] for i in todo])
        bad = 0
        for i, v in zip(todo, info):
            if v is None:
                bad += 1
                continue
            D.at[i, "_w"], D.at[i, "_h"], D.at[i, "_mode"] = v
        R["crop_khong_doc_duoc"] = bad
        inv.check("khong_tep_crop_hong", bad == 0, f"{bad} tệp không mở được")
        ks = {}
        for name, m in (("GOLD", D.tier == "GOLD"), ("GOLD_text_only", D.tier == "GOLD_text_only"), ("SYLLABLE", D.tier == "SYLLABLE"),
                        ("REVIEW_co_anh", (D.tier == "REVIEW") & (D._path != "")), ("tat_ca_co_anh", D._path != "")):
            g = D[m & (D._path != "")]
            if len(g):
                asp = (g._w / g._h).to_numpy(float)
                ks[name] = dict(w=L.stats(g._w), h=L.stats(g._h), ti_le_w_tren_h_median=round(float(np.median(asp)), 3),
                                canh_ngan_median=float(np.median(np.minimum(g._w, g._h))), mode=g._mode.value_counts().to_dict())
        R["kich_thuoc_crop"] = ks
    # ---- ô neo
    A = L.anchor_mask(D)
    D["_neo"] = A
    D["_neo_crop"] = A & (D._path != "").to_numpy()
    n_neo_all = int(A.sum())
    Rn = {}
    Rn["dinh_nghia"] = "rule bắt đầu s1_inter_s2_direct ∧ label ≠ '' ∧ label == ocr_char ∧ label ∈ R(âm); đếm trên bảng MỌI tầng"
    Rn["tong_theo_dinh_nghia"] = n_neo_all
    Rn["theo_tang"] = D[A].tier.value_counts().to_dict()
    Rn["co_tep_crop"] = int(D._neo_crop.sum())
    Rn["co_tep_crop_theo_tang"] = D[D._neo_crop].tier.value_counts().to_dict()
    Rn["khong_tep_crop"] = int((A & ~D._neo_crop.to_numpy()).sum())
    na = D[D._neo_crop]
    cnt = collections.Counter(na.label)
    Rn["so_chu_khac_nhau"] = len(cnt)
    Rn["chu_ge3"] = int(sum(v >= 3 for v in cnt.values())); Rn["chu_ge5"] = int(sum(v >= 5 for v in cnt.values()))
    Rn["chu_ge1"] = len(cnt); Rn["chu_ge2"] = int(sum(v >= 2 for v in cnt.values()))
    Rn["o_neo_thuoc_chu_ge3"] = int(sum(v for v in cnt.values() if v >= 3)); Rn["o_neo_thuoc_chu_ge5"] = int(sum(v for v in cnt.values() if v >= 5))
    # hậu tố luật (mức độ qua tay chon_chu/gate)
    toks = collections.Counter(t.split(":")[0] + (":" + t.split(":")[1] if t.split(":")[0] in ("chon_chu", "gate") and ":" in t else "")
                               for r in na.rule for t in r.split("|")[1:])
    Rn["hau_to_luat"] = dict(toks.most_common(10))
    Rn["qua_chon_chu"] = int(na.rule.str.contains("chon_chu:").sum())
    Rn["qua_gate_qn_count_unfixed"] = int(na.rule.str.contains("gate:qn_count_unfixed").sum())
    Rn["qua_gate_crop_bad"] = int(na.rule.str.contains("gate:crop_bad").sum())
    Rn["thuan_khong_hau_to"] = int((~na.rule.str.contains(r"\|")).sum())
    # chữ hiếm: tỉ lệ ô thuộc chữ có < 2 ô neo
    gold = D[(D.tier == "GOLD") & (D.label != "")]
    Rn["ti_le_o_GOLD_thuoc_chu_hiem_lt2"] = round(float(np.mean([cnt.get(c, 0) < 2 for c in gold.label])), 4) if len(gold) else None
    Rn["ti_le_o_GOLD_thuoc_chu_lt5"] = round(float(np.mean([cnt.get(c, 0) < 5 for c in gold.label])), 4) if len(gold) else None
    kim = D[(D.ocr_char != "") & (D.ocr_char.str.len() == 1)]
    Rn["ti_le_o_moi_tang_theo_chu_kim_hiem_lt2"] = round(float(np.mean([cnt.get(c, 0) < 2 for c in kim.ocr_char])), 4)
    Rn["o_GOLD_xet"] = int(len(gold)); Rn["o_moi_tang_xet_kim"] = int(len(kim))
    # ---- gold_exact + crop chuẩn
    GE = {}
    GE["so_dong"] = int(len(G))
    GE["trang_thai"] = G.gold_exact.value_counts().to_dict()
    GE["evidence_level"] = G.evidence_level.value_counts().to_dict()
    GE["crop_status"] = G.crop_status.value_counts().to_dict()
    GE["policy_version"] = G.policy_version.value_counts().to_dict()
    Gi = G.set_index("image")
    inv.check("gold_exact_phu_dung_GOLD_cua_pub", set(G.image) == set(P.image[P.tier == "GOLD"]),
              f"ge {len(G)} vs GOLD pub {(P.tier == 'GOLD').sum()}; khác {len(set(G.image) ^ set(P.image[P.tier == 'GOLD']))}")
    D["_ge"] = D.image.map(Gi.gold_exact).fillna("(khong_co_dong_gold_exact)")
    cc = D[D._neo_crop]
    GE["o_neo_theo_trang_thai"] = cc._ge.value_counts().to_dict()
    ok = cc[cc._ge == "ok"]
    cc_path = D.image.map(Gi.crop_chuan).fillna("")
    D["_chuan_rel"] = cc_path
    chuan_exist = []
    for im, c in zip(D.image, cc_path):
        chuan_exist.append(bool(c) and os.path.isfile(str((root_pub / c))))
    D["_chuan_ex"] = np.array(chuan_exist, bool)
    oka = D[D._neo_crop & (D._ge == "ok")]
    GE["o_neo_ok_co_crop_chuan_ton_tai"] = int(oka._chuan_ex.sum())
    GE["o_ok_tong"] = int((G.gold_exact == "ok").sum())
    GE["o_ok_co_crop_chuan_ton_tai"] = int(D[D._ge == "ok"]._chuan_ex.sum())
    cnt_ok = collections.Counter(oka.label)
    GE["neo_ok_so_chu"] = len(cnt_ok); GE["neo_ok_chu_ge3"] = int(sum(v >= 3 for v in cnt_ok.values())); GE["neo_ok_chu_ge5"] = int(sum(v >= 5 for v in cnt_ok.values()))
    GE["neo_ok_n"] = int(len(oka))
    inv.check("moi_o_ok_co_crop_chuan", GE["o_ok_tong"] == GE["o_ok_co_crop_chuan_ton_tai"], f"{GE['o_ok_tong']} ok vs {GE['o_ok_co_crop_chuan_ton_tai']} tệp")
    R["gold_exact"] = GE
    # ---- chữ ứng viên (R(âm) ∪ kim [∪ lt1/lt2 STT])
    cand = collections.Counter(); candR = collections.Counter(); candK = collections.Counter()
    Rm = {s: L.R_of(s) for s in set(D.syllable)}
    for s, k in zip(D.syllable, D.ocr_char):
        for c in Rm[s]:
            candR[c] += 1
        if k and len(k) == 1:
            candK[k] += 1
    stt_extra = None
    if ma.startswith("stt"):
        stt_extra, npg_kim = kim_chars_stt(ma)
        R["kim_l1skel_l2_trang"] = npg_kim
    allc = set(candR) | set(candK) | (set(stt_extra) if stt_extra else set())
    ng = collections.Counter(D[D._neo_crop & (D.tier == "GOLD")].label)
    tsv = pd.DataFrame(dict(chu=sorted(allc)))
    tsv["u"] = ["U+%04X" % ord(c) for c in tsv.chu]
    tsv["so_o_trong_R"] = tsv.chu.map(candR).fillna(0).astype(int)
    tsv["so_o_kim_doc"] = tsv.chu.map(candK).fillna(0).astype(int)
    tsv["so_o_kim_lt1_lt2_cache"] = tsv.chu.map(stt_extra).fillna(0).astype(int) if stt_extra is not None else 0
    tsv["so_o_neo"] = tsv.chu.map(cnt).fillna(0).astype(int)
    tsv["so_o_neo_GOLD"] = tsv.chu.map(ng).fillna(0).astype(int)
    CU = dict(so_chu=len(allc), chi_trong_R=len(set(candR) - set(candK) - set(stt_extra or {})), trong_R=len(candR), kim_khong_trong_R=len(set(candK) - set(candR)),
              so_chu_co_o_neo_ge1=int((tsv.so_o_neo >= 1).sum()), so_chu_neo_ge3=int((tsv.so_o_neo >= 3).sum()), so_chu_neo_ge5=int((tsv.so_o_neo >= 5).sum()),
              so_chu_neo_lt2=int((tsv.so_o_neo < 2).sum()), ti_le_chu_ung_vien_neo_lt2=round(float((tsv.so_o_neo < 2).mean()), 4))
    if stt_extra is not None:
        CU["kim_lt1_lt2_cache_ngoai_R_va_kim"] = len(set(stt_extra) - set(candR) - set(candK))
    R["chu_ung_vien"] = CU
    # ---- ghi
    (L.OUT / "chu_ung_vien").mkdir(parents=True, exist_ok=True)
    (L.OUT / "chu_ung_vien" / f"{ma}.txt").write_text("\n".join(sorted(allc)), encoding="utf-8")
    tsv.to_csv(L.OUT / "chu_ung_vien" / f"{ma}.tsv", sep="\t", index=False)
    (L.OUT / "neo").mkdir(parents=True, exist_ok=True)
    keep = ["image", "tier", "rule", "label", "ocr_char", "syllable", "page", "column", "nom_idx", "syl_idx", "bbox", "image_md5",
            "crop_quality_flag", "stray_ink", "border_ink", "ink_pct", "gate_reason", "chon_chu"]
    X = D[D._neo_crop].copy()
    for c in keep:
        if c not in X.columns:
            X[c] = ""
    X = X[keep + ["_in_pub", "_path", "_w", "_h", "_mode", "_ge", "_chuan_rel", "_chuan_ex"]].rename(
        columns=dict(_in_pub="trong_pub", _path="duong_crop", _w="w", _h="h", _mode="mode", _ge="gold_exact", _chuan_rel="crop_chuan_rel", _chuan_ex="crop_chuan_ton_tai"))
    X["duong_crop"] = [L.rel(p) for p in X.duong_crop]
    X.insert(0, "ma", ma)
    X.to_csv(L.OUT / "neo" / f"{ma}.csv", index=False)
    R["neo"] = Rn
    R["bat_bien"] = inv.summary(); R["bat_bien_chi_tiet"] = inv.rows
    R["giay"] = round(time.time() - t0, 1)
    print(f"[01] {ma}: all {len(D)} pub {len(P)} tầng {R['tang_all']} | crop pub thiếu {R['crop_thieu_pub']} | neo {n_neo_all} có crop {Rn['co_tep_crop']} "
          f"chữ {Rn['so_chu_khac_nhau']} ≥3 {Rn['chu_ge3']} ≥5 {Rn['chu_ge5']} | ok∩neo {GE['neo_ok_n']} | chữ ứng viên {CU['so_chu']} [{R['giay']}s]", flush=True)
    return R


def rel_p(p):
    return L.rel(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", default=L.ORDER)
    ap.add_argument("--no-sizes", action="store_true")
    a = ap.parse_args()
    out = {}
    for ma in a.books:
        out[ma] = book(ma, not a.no_sizes)
    p = L.OUT / ("kiem_ke.json" if a.books == L.ORDER else "kiem_ke_" + "_".join(a.books) + ".json")
    L.jdump(out, p)
    print("->", L.rel(p))


if __name__ == "__main__":
    main()
