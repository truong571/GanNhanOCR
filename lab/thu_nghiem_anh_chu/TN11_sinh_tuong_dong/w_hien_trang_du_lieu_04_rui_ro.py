"""w_hien_trang_du_lieu_04_rui_ro.py — RỦI RO DỮ LIỆU cho gói Kaggle sinh ảnh theo sách: tệp crop thiếu/hỏng, md5 sai lệch, crop trùng, STT tính lẫn quyển + thư mục STT cũ,
rò nhãn người (cột/đường), bộ "evaluation_only".

1. Tệp: md5 thật của tệp crop (bản dataset/<Bộ>/ và bản thư mục dựng) so với cột image_md5 (12 hex đầu); bản giao == bản dựng.
2. Trùng: nhóm md5 trùng trong sách / chéo sách / trong ô neo (và nhóm trùng mang NHÃN KHÁC NHAU).
3. STT: trang trùng tên giữa ba quyển, khoá (page, cột, nom_idx, syl_idx) trùng giữa quyển (nếu bỏ cột book thì lẫn), thư mục dataset/SachThanhTruyen{2,4,11}/ (23/09) cũ so với hiện tại.
4. Rò nhãn người: cột của bảng TN8 (base/cand) + cột gold_exact.csv phụ thuộc nhãn/văn bản người + đường tệp có nhãn người (kiểm tồn tại) + luật chứa 'nguoi'/'qd01' trong nhãn hiện tại.
5. Bộ evaluation_only (TAP_DANH_GIA.md): số ô ô neo thuộc 4 bộ B18/B34/L16/TK.
Ra: measure_out/_tn11/full/hien_trang_du_lieu/rui_ro.json. 0 API/GPU.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_04_rui_ro.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import w_hien_trang_du_lieu_lib as L  # noqa: E402


def pmap(fn, items, w=8):
    with ThreadPoolExecutor(w) as ex:
        return list(ex.map(fn, items, chunksize=128))


def md5(p):
    try:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return ""


def tep_md5(ma, R, inv):
    C = L.CFG[ma]
    P = L.load_pub(ma)
    root_pub = L.REPO / "dataset" / C["ds"]
    root_all = L.REPO / C["all_root"]
    Pe = P[P.tier != "GOLD_text_only"]
    m_pub = pmap(md5, [str(root_pub / im) for im in Pe.image])
    m_bld = pmap(md5, [str(root_all / im) for im in Pe.image])
    pm = np.array([a[:12] for a in m_pub])
    ref = Pe.image_md5.to_numpy()
    R["md5"] = dict(n_tep=int(len(Pe)), pub_khac_cot_image_md5=int((pm != ref).sum()), pub_khac_ban_dung=int(sum(a != b for a, b in zip(m_pub, m_bld))),
                    khong_doc_duoc_pub=int(sum(a == "" for a in m_pub)), khong_doc_duoc_ban_dung=int(sum(a == "" for a in m_bld)))
    inv.check(f"{ma}_md5_tep_pub_khop_cot_image_md5", R["md5"]["pub_khac_cot_image_md5"] == 0, f"{R['md5']['pub_khac_cot_image_md5']} lệch/{len(Pe)}")
    inv.check(f"{ma}_ban_giao_bang_ban_dung", R["md5"]["pub_khac_ban_dung"] == 0, f"{R['md5']['pub_khac_ban_dung']} khác")
    Pe = Pe.assign(_md5=m_pub)
    return Pe


def main():
    t0 = time.time()
    inv = L.Inv()
    out = dict(sach={})
    allp = []
    for ma in L.ORDER:
        R = {}
        Pe = tep_md5(ma, R, inv)
        neo = pd.read_csv(L.OUT / "neo" / f"{ma}.csv", dtype=str, keep_default_na=False)
        neo_gold = neo[neo.tier == "GOLD"]
        # trùng md5 trong sách
        g = Pe.groupby("_md5").agg(n=("image", "size"), nl=("label", lambda s: len(set(x for x in s if x))))
        d = g[g.n > 1]
        R["trung_md5_trong_sach"] = dict(nhom=int(len(d)), dong_trong_nhom=int(d.n.sum()), nhom_mang_nhan_khac_nhau=int((d.nl > 1).sum()),
                                          dong_GOLD_trong_nhom=int(Pe[Pe._md5.isin(d.index) & (Pe.tier == "GOLD")].shape[0]))
        dn = neo_gold.groupby("image_md5").agg(n=("image", "size"), nl=("label", "nunique"))
        dn = dn[dn.n > 1]
        R["trung_md5_trong_neo_GOLD"] = dict(nhom=int(len(dn)), dong=int(dn.n.sum()), nhom_nhan_khac_nhau=int((dn.nl > 1).sum()))
        inv.check(f"{ma}_neo_GOLD_khong_trung_md5_khac_nhan", (dn.nl > 1).sum() == 0, f"{int((dn.nl > 1).sum())} nhóm")
        allp.append(Pe[["image_md5", "tier", "label"]].assign(ma=ma))
        out["sach"][ma] = R
        print(f"[04] {ma}: md5 lệch cột {R['md5']['pub_khac_cot_image_md5']} giao≠dựng {R['md5']['pub_khac_ban_dung']} | trùng md5 trong sách: nhóm {R['trung_md5_trong_sach']['nhom']} "
              f"dòng {R['trung_md5_trong_sach']['dong_trong_nhom']} nhãn khác {R['trung_md5_trong_sach']['nhom_mang_nhan_khac_nhau']} | trong neo GOLD {R['trung_md5_trong_neo_GOLD']}", flush=True)
    A = pd.concat(allp)
    # chéo sách (md5 12 hex)
    g = A.groupby("image_md5").ma.nunique()
    out["trung_md5_cheo_sach"] = dict(nhom=int((g > 1).sum()), chi_tiet=A[A.image_md5.isin(g[g > 1].index)].groupby("ma").size().to_dict())
    stt = A[A.ma.str.startswith("stt")]
    gs = stt.groupby("image_md5").ma.nunique()
    out["trung_md5_cheo_quyen_STT"] = int((gs > 1).sum())
    inv.check("khong_trung_md5_cheo_10_cuon", out["trung_md5_cheo_sach"]["nhom"] == 0, f"{out['trung_md5_cheo_sach']['nhom']} nhóm")

    # ---- tệp mồ côi: tệp .png dưới dataset/<Bộ>/{gold,syllable}/ không có trong labels.csv (và ngược lại)
    mo_coi = {}
    for ds in ["SachThanhTruyen", "Chrestomathie1872", "LucVanTien1883", "KimVanKieu1884", "LucVanTien1916", "TruyenKieu1872", "SachKinhThayCaBinh", "SachDungLyHoThan"]:
        root = L.REPO / "dataset" / ds
        disk = set()
        for sub in ("gold", "syllable"):
            for f in (root / sub).rglob("*.png"):
                disk.add(str(f.relative_to(root)))
        Pd = L.rd(root / "labels.csv")
        lab = set(Pd.image[Pd.tier != "GOLD_text_only"])
        n_chon_chu = sum(1 for x in disk if x.startswith("gold/chon_chu/"))
        mo_coi[ds] = dict(tep_tren_dia=len(disk), tep_trong_gold_chon_chu=n_chon_chu, dong_co_tep=len(lab), tep_khong_co_trong_labels=len(disk - lab), dong_khong_co_tep=len(lab - disk))
        inv.check(f"{ds}_khong_tep_mo_coi_va_khong_thieu", mo_coi[ds]["tep_khong_co_trong_labels"] == 0 and mo_coi[ds]["dong_khong_co_tep"] == 0, str(mo_coi[ds]))
    out["tep_mo_coi_dataset"] = mo_coi
    # ---- STT: lẫn quyển
    D = L.rd(L.REPO / "dataset_out/labels_final.csv")
    qs = ["stt2", "stt4", "stt11"]
    pg = {q: set(D[D.book == q].page) for q in qs}
    S = dict(trang_trung_ten={f"{a}&{b}": len(pg[a] & pg[b]) for i, a in enumerate(qs) for b in qs[i + 1:]}, so_trang=dict((q, len(pg[q])) for q in qs),
             trang_co_o_ca_3_quyen=len(pg["stt2"] & pg["stt4"] & pg["stt11"]))
    D["_k"] = D.page + "|" + D.column + "|" + D.nom_idx + "|" + D.syl_idx
    kq = D.groupby("_k").book.nunique()
    S["khoa_o_trung_giua_quyen"] = dict(so_khoa=int(len(kq)), khoa_o_ge2_quyen=int((kq >= 2).sum()), khoa_o_ca_3_quyen=int((kq == 3).sum()))
    Aall = pd.DataFrame()
    neos = {q: pd.read_csv(L.OUT / "neo" / f"{q}.csv", dtype=str, keep_default_na=False) for q in qs}
    for q in qs:
        n = neos[q]
        n = n[n.tier == "GOLD"].copy(); n["_k"] = n.page + "|" + n.column + "|" + n.nom_idx + "|" + n.syl_idx; n["q"] = q
        Aall = pd.concat([Aall, n[["_k", "q", "label"]]])
    ak = Aall.groupby("_k").q.nunique()
    two = Aall[Aall._k.isin(ak[ak >= 2].index)]
    lab_conf = two.groupby("_k").label.nunique()
    S["neo_GOLD_cung_khoa_o_giua_quyen"] = dict(khoa_co_neo_o_ge2_quyen=int((ak >= 2).sum()), trong_do_nhan_khac_nhau=int((lab_conf > 1).sum()),
                                                 o_neo_bi_anh_huong_neu_bo_cot_book=int(len(two)))
    S["quy_tac"] = ("PHẢI lọc cột book (stt2/stt4/stt11) khi đọc dataset/SachThanhTruyen/labels.csv, gold_exact.csv (set8) hoặc dataset_out/labels_final.csv; khoá ô phải gồm book; "
                    "đường ảnh (gold/stt11_page_XXXX_cNN_NNN.png) có tiền tố quyển nên đường ảnh KHÔNG trùng giữa quyển")
    inv.check("duong_anh_STT_khong_trung_giua_quyen", D.image[D.image != ""].is_unique, "")
    # thư mục STT cũ 23/09
    cu = {}
    for q, nm in zip(qs, ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]):
        o = L.rd(L.REPO / "dataset" / nm / "labels.csv")
        n = L.load_pub(q)
        m = o.merge(n, on="image", suffixes=("_cu", "_moi"))
        cu[q] = dict(thu_muc=f"dataset/{nm}/", mtime_labels=L.mt(L.REPO / "dataset" / nm / "labels.csv"), so_dong_cu=int(len(o)), so_dong_hien_tai=int(len(n)),
                     chung_theo_duong_anh=int(len(m)), cung_md5_anh=int((m.image_md5_cu == m.image_md5_moi).sum()), cung_nhan=int((m.label_cu == m.label_moi).sum()),
                     cung_hop=int((m.bbox_cu == m.bbox_moi).sum()), tang_cu=o.tier.value_counts().to_dict(), tang_hien_tai=n.tier.value_counts().to_dict(),
                     co_gold_exact=os.path.exists(L.REPO / "dataset" / nm / "gold_exact.csv"), co_labels_trace=os.path.exists(L.REPO / "dataset" / nm / "labels_trace.csv"))
    S["thu_muc_STT_cu_23_09"] = cu
    out["STT"] = S

    # ---- rò nhãn người
    leak = dict()
    base_cols = {}
    for ma in ["B18", "B34", "L16", "TK", "L83", "KVK", "Chr", "stt2"]:
        b = pd.read_pickle(L.TN8 / "base" / f"{ma}.pkl")
        base_cols[ma] = {c: (None if c not in b.columns else int(b[c].sum()) if b[c].dtype == bool else int((b[c].astype(str) != "").sum())) for c in L.HUMAN_COLS_TN8_BASE}
    leak["tn8_base_cot_nguoi_co_mat_va_so_dong_khac_rong_hoac_True"] = base_cols
    F = pd.read_pickle(L.TN8 / "cand" / "B34.pkl")
    leak["tn8_cand_cot_nguoi_co_mat_B34"] = {c: (c in F.columns) for c in L.HUMAN_COLS_TN8_CAND}
    del F
    leak["tn8_base_cot_nguoi"] = L.HUMAN_COLS_TN8_BASE
    leak["tn8_cand_cot_nguoi_hoac_suy_ra"] = L.HUMAN_COLS_TN8_CAND
    leak["tn8_emb_tep_nguoi"] = sorted(p.name for p in (L.TN8 / "emb").glob("hum_*"))
    ge = L.rd(L.REPO / "dataset/LucVanTien1916/gold_exact.csv").columns.tolist()
    geo = ["f_blank", "f_cut", "f_two", "f_ink", "bleed_new", "trunc_new", "tall_new", "ink_ratio", "dup_bbox", "ov_heavy", "int_foreign", "one_char_ok", "crop_status", "crop_flags",
           "ink_cx", "ink_cy", "core_loss", "core_loss_flag", "cell_uid", "book_set", "book", "set8", "image", "image_file_md5", "crop_chuan", "crop_chuan_md5", "crop_chuan_128",
           "crop_chuan_128_md5", "label"]
    leak["gold_exact_cot_hinh_hoc_duoc_phep"] = [c for c in ge if c in geo]
    leak["gold_exact_cot_KHONG_dua_vao_goi"] = [c for c in ge if c not in geo]
    leak["gold_exact_ly_do"] = ("gold_exact/reason/evidence_level + p_wood_*, viss_*, lobo_*, ta, ta_sub, mocr*, m_hom, simg, H1_new, lai, AINT… đều là đầu ra của bộ kiểm học từ crop NGƯỜI "
                                "(models/gold_exact/MANIFEST.json: 'nguyên mẫu người P/nh (Borg keep fold0-3 + TruyenKieu1872/LucVanTien1916 phần A)') và/hoặc văn bản người dị bản (ta_refs/mocr_refs). "
                                "ok = lai ∧ ¬luật A ∧ ¬M-OCR (docs/GOLD_CHINH_XAC_2026-09-27.md §3) ⇒ chọn ô theo ok = chọn gián tiếp theo nhãn/văn bản người (của sách KHÁC hoặc dị bản)")
    paths = ["prepared/SachKinhThayCaBinh/nom_transcriptions", "prepared/SachDungLyHoThan/nom_transcriptions", "data/SachKinhThayCaBinh/SachKinhThayCaBinh.xlsx",
             "data/SachKinhThayCaBinh/Sách kinh Thầy cả Bỉnh.xlsx", "data/SachDungLyHoThan", "data/LucVanTien1916/manifest.tsv", "data/LucVanTien1916/train.json",
             "data/LucVanTien1916/val.json", "data/LucVanTien1916/nomfoundation_lvt_phienam.json", "data/TruyenKieu1872/manifest.tsv", "data/TruyenKieu1872/train.json",
             "data/TruyenKieu1872/val.json", "data/TruyenKieu1872/nomfoundation_1872_phienam.json", "data/TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871_LieuVanDuong_phienam.json",
             "dataset/_BORG_NHAN_NGUOI", "models/gold_exact/ta_refs.pt", "models/gold_exact/mocr_refs.pt", "models/gold_exact/hand_tables_T_Kinh.pt", "models/gold_exact/vft_tables_T.pt",
             "measure_out/_tn8/base", "measure_out/_tn8/cand", "measure_out/_tn8/emb"]
    leak["duong_co_nhan_nguoi_hoac_suy_ra_KHONG_dua_vao_goi"] = {p: ("TON_TAI" if (L.REPO / p).exists() else "KHONG_THAY") for p in paths}
    # luật chứa dấu vết người trong nhãn hiện tại
    pat = re.compile(r"nguoi|human|qd01|quyet_dinh|hum_")
    toks = collections.Counter()
    for ma in L.ORDER:
        D1 = L.load_all(ma)
        for r in D1.rule.unique():
            for t in r.split("|"):
                if pat.search(t):
                    toks[(ma, t)] += int((D1.rule == r).sum())
    leak["luat_chua_nguoi_trong_nhan_hien_tai"] = {f"{k[0]}:{k[1]}": v for k, v in toks.items()}
    leak["luu_y_luat_lop_nham_nguoi"] = "'lop_nham:nguoi_2029A_vs_346B' là lớp nhầm của ÂM 'người' (㝵 U+2029A vs 㑲 U+346B), không phải phán quyết người"
    # dòng của gói ứng viên: cột an toàn của neo/<ma>.csv
    nh = pd.read_csv(L.OUT / "neo" / "B34.csv", nrows=1).columns.tolist()
    leak["neo_csv_cot"] = nh
    leak["neo_csv_chua_cot_nguoi"] = [c for c in nh if c in set(L.HUMAN_COLS_TN8_BASE) | set(L.HUMAN_COLS_TN8_CAND)]
    inv.check("neo_csv_khong_chua_cot_nhan_nguoi", leak["neo_csv_chua_cot_nguoi"] == [], str(leak["neo_csv_chua_cot_nguoi"]))
    out["ro_nhan_nguoi"] = leak
    # ---- evaluation_only
    ev = {}
    for ma in ["B18", "B34", "L16", "TK"]:
        C = L.CFG[ma]
        j = json.load(open(L.REPO / "dataset" / C["ds"] / "evaluation_only.json", encoding="utf-8"))
        ev[ma] = dict(evaluation_only=j.get("evaluation_only"), kind=j.get("kind"), ground_truth_file=j.get("ground_truth_file"), marked_on=j.get("marked_on"),
                      tap_danh_gia_md=f"dataset/{C['ds']}/TAP_DANH_GIA.md")
    ev["noi_dung_cam"] = ("TAP_DANH_GIA.md: 'KHÔNG đưa bất kỳ dòng nào của labels.csv này vào tập train/val của mô hình nào (nhận diện chữ, S3/ArcFace, detector, bộ kiểm chữ viết tay)'. "
                          "Tinh chỉnh FontDiffuser trên crop ô neo của 4 bộ này (TN11 p04/Kaggle đã làm với B18/B34) chạm điều cấm này theo nghĩa đen; chỉ dùng ảnh phong cách khi SUY LUẬN thì nhẹ hơn nhưng vẫn lấy dòng/crop của bộ. Cần quyết định của người dùng.")
    ev["so_o_neo_GOLD_4_bo"] = {ma: int((pd.read_csv(L.OUT / "neo" / f"{ma}.csv", dtype=str, keep_default_na=False).tier == "GOLD").sum()) for ma in ["B18", "B34", "L16", "TK"]}
    out["evaluation_only"] = ev
    out["bat_bien"] = inv.summary(); out["bat_bien_chi_tiet"] = inv.rows
    out["giay"] = round(time.time() - t0, 1)
    L.jdump(out, L.OUT / "rui_ro.json")
    print(f"[04] xong: trùng md5 chéo sách {out['trung_md5_cheo_sach']['nhom']} nhóm, chéo quyển STT {out['trung_md5_cheo_quyen_STT']} | bất biến {out['bat_bien']['dat']}/{out['bat_bien']['tong']} [{out['giay']}s]")
    print("->", L.rel(L.OUT / "rui_ro.json"))


if __name__ == "__main__":
    main()
