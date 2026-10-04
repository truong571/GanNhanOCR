"""r_goi_du_lieu_02_anh_phong_cach.py — HƯỚNG 2: ẢNH PHONG CÁCH của gói Kaggle (data/<cuốn>/style_0.png, style_1.png) đến từ ô nào và có sạch/đúng cuốn không.

Với 10 cuốn x 2 ảnh:
  1. nguồn: hạng 1/2 của hien_trang_du_lieu/style_ung_vien/<cuốn>.csv -> đường crop_chuan_cache_128 -> dựng lại style_image() (bản sao p11) và so BYTE với ảnh trong gói;
  2. ĐÚNG cuốn/quyển: đường cache (book_set/book/page/cột_nom_syl) khớp cuốn kỳ vọng; cell_uid có trong dataset/<Bộ>/gold_exact.csv; cột book của labels.csv; ảnh gốc cùng ô;
  3. ô neo tự động sạch: rule, label==ocr_char, tầng GOLD, label ∈ R(âm), cờ hình học gold_exact; tình trạng TRƯỚC/SAU chon_chu (bảng trước chon_chu cho 7 sách) và gate_reason;
  4. số đo ảnh: kiểu/kích thước, mực %, thành phần liên thông, bbox, mép, tỉ lệ xám trung gian, lệch tâm; không rỗng/không đen kín;
  5. khác cuốn không trùng ảnh: sha256 20 ảnh, cell_uid, NCC từng cặp;
  6. cơ cấu hồ ô neo (sạch thuần / qua chon_chu) và ĐỀ XUẤT thay ảnh cho cuốn mà ảnh hiện tại phụ thuộc chon_chu (ghi vào de_xuat_style/, KHÔNG sửa gói).

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_02_anh_phong_cach.py
Ra: 02_anh_phong_cach.json, de_xuat_style/*.png|json.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402

GEO = ["f_blank", "f_cut", "f_two", "bleed_new", "trunc_new", "tall_new", "dup_bbox", "ov_heavy", "int_foreign"]
KEY = ["book", "page", "column", "nom_idx", "syl_idx"]
BOOKSET = {b: Lb.DS[b] for b in Lb.ORDER}


def style_image(g: np.ndarray) -> np.ndarray:
    """== p11_dung_goi_toan_bo.style_image (bản sao để tái tạo; so byte với ảnh trong gói)."""
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw < 128).mean() > 0.5:
        bw = 255 - bw
    h, w = bw.shape
    s = int(max(h, w) * 1.2)
    out = np.full((s, s), 255, np.uint8)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = bw
    return cv2.resize(out, (128, 128), interpolation=cv2.INTER_AREA)


def stats(arr: np.ndarray) -> dict:
    ink = arr < 128
    n_cc, _, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    areas = sorted((int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n_cc)), reverse=True)
    ys, xs = np.nonzero(ink)
    bb = dict(x0=int(xs.min()), y0=int(ys.min()), w=int(xs.max() - xs.min() + 1), h=int(ys.max() - ys.min() + 1)) if len(xs) else dict(x0=-1, y0=-1, w=0, h=0)
    fr = np.zeros_like(ink); fr[:3, :] = True; fr[-3:, :] = True; fr[:, :3] = True; fr[:, -3:] = True
    cs, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = float(sum(cv2.arcLength(c, True) for c in cs))
    sw = round(2.0 * float(ink.sum()) / per, 2) if per > 0 else None          # độ dày nét ≈ 2·diện tích/chu vi (cùng cách đo ở r_05 cho glyph phông: NomNaTong ≈ 5,1 px ở 128 px)
    return dict(do_day_net_px=sw, muc_pct=round(float(ink.mean()) * 100, 2), thanh_phan_tong=int(n_cc - 1), thanh_phan_ge12px=int(sum(a >= 12 for a in areas)),
                thanh_phan_lon_nhat_pct_muc=round(100 * areas[0] / max(ink.sum(), 1), 1) if areas else 0.0,
                bbox=bb, bbox_w_pct=round(bb["w"] / 1.28, 1), bbox_h_pct=round(bb["h"] / 1.28, 1),
                muc_cham_vien_3px=int((ink & fr).sum()), xam_trung_gian_pct=round(float(((arr > 0) & (arr < 255)).mean()) * 100, 2),
                lech_tam_x=round(float(xs.mean() - 63.5), 1) if len(xs) else None, lech_tam_y=round(float(ys.mean() - 63.5), 1) if len(ys) else None,
                gia_tri_khac_nhau=int(len(np.unique(arr))))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def load_tables(b: str):
    want = set(KEY + ["syllable", "ocr_char", "label", "tier", "rule", "chon_chu", "chon_chu_truoc", "gate_reason", "ink_pct", "stray_ink", "image"])
    post = Lb.rd(Lb.REPO / Lb.POST[b], usecols=lambda c: c in want)
    if b in Lb.STT:
        post = post[post.book == Lb.BOOK_VAL[b]]
    post["_k"] = post[KEY].agg("|".join, axis=1)
    pre = None
    if b not in Lb.STT:
        pre = Lb.rd(Lb.REPO / Lb.PREF[b], usecols=lambda c: c in set(KEY + ["syllable", "ocr_char", "label", "tier", "rule", "gate_reason"]))
        pre["_k"] = pre[KEY].agg("|".join, axis=1)
    return post.set_index("_k"), (pre.set_index("_k") if pre is not None else None)


def main():
    t0 = time.time()
    inv = Lb.Inv()
    from pipeline.gold_exact import common as C
    res: dict = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"), sach={}, tong_hop={})
    spec = json.loads((Lb.SPEC / "spec.json").read_text(encoding="utf-8"))
    plan = json.loads((Lb.PACK / "plan.json").read_text(encoding="utf-8"))
    imgs = {}
    for b in Lb.ORDER:
        S = Lb.rd(Lb.HTD / "style_ung_vien" / f"{b}.csv").sort_values("hang", key=lambda s: s.astype(int))
        ds = Lb.DS[b]
        G = Lb.rd(Lb.REPO / "dataset" / ds / "gold_exact.csv")
        if b in Lb.STT:
            G = G[G.set8 == b]
        Gi = G.set_index("cell_uid")
        Pub = Lb.rd(Lb.REPO / "dataset" / ds / "labels.csv")
        if b in Lb.STT:
            Pub = Pub[Pub.book == Lb.BOOK_VAL[b]]
        Pi = Pub.set_index("image")
        post, pre = load_tables(b)
        rb = dict(anh=[])
        for j in range(2):
            r = S.iloc[j]
            p_pack = Lb.PACK / "data" / b / f"style_{j}.png"
            e = dict(j=j, hang=int(r.hang), chu=r.label, goi=str(p_pack.relative_to(Lb.PACK)))
            # ---- 1. dựng lại và so byte
            src = Lb.REPO / r.crop_chuan_cache_128
            g = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            e["nguon_cache_128"] = r.crop_chuan_cache_128
            e["nguon_doc_duoc"] = g is not None
            with Image.open(p_pack) as im:
                e["kieu_anh"] = dict(mode=im.mode, size=list(im.size), format=im.format, info_khoi_phu={k: str(v)[:40] for k, v in im.info.items()})
                arr = np.array(im)
            imgs[(b, j)] = arr
            e["dung_lai_style_image_trung_byte_voi_goi"] = bool(g is not None and np.array_equal(style_image(g), arr))
            e["sha256_png"] = Lb.sha256_file(p_pack)
            e["khop_tep_spec_styles"] = Lb.sha256_file(Lb.SPEC / "styles" / f"{b}_s{j}.png") == e["sha256_png"]
            e.update(stats(arr))
            e["kich_thuoc_mang"] = [int(x) for x in arr.shape]
            e["cache_kich_thuoc"] = [int(x) for x in g.shape] if g is not None else None
            # ---- 2. đúng cuốn/quyển
            parts = Path(r.crop_chuan_cache_128).parts
            exp_uid = f"{BOOKSET[b]}/{Lb.BOOK_VAL[b]}/{r.page}/c{r.column}/n{r.nom_idx}/s{r.syl_idx}"
            exp_path = ("prepared/_gold_exact/crop_chuan/sq128/" + "/".join(exp_uid.split("/")[:3]) + "/" + "_".join(exp_uid.split("/")[3:]) + ".png")
            e["duong_cache_dung_dinh_dang_va_cuon"] = (r.crop_chuan_cache_128 == exp_path)
            e["cell_uid_ky_vong"] = exp_uid
            e["cell_uid_co_trong_gold_exact"] = exp_uid in Gi.index
            if exp_uid in Gi.index:
                gr = Gi.loc[exp_uid]
                e["gold_exact_set8"] = gr.set8; e["gold_exact_book"] = gr.book; e["gold_exact_book_set"] = gr.book_set
                e["gold_exact_trang_thai"] = gr.gold_exact; e["gold_exact_crop_status"] = gr.crop_status
                e["co_hinh_hoc"] = {f: gr[f] for f in GEO + ["f_ink", "one_char_ok"]}
                e["cac_co_hinh_hoc_khong_bat"] = all(gr[f] in ("0", "") for f in GEO) and gr.one_char_ok == "1" and gr.crop_status == "ok"
                e["gold_exact_image"] = gr.image
                e["anh_goc_trung_duong_crop_csv"] = ("dataset/" + ds + "/" + gr.image) == r.duong_crop
                if gr.image in Pi.index:
                    pr = Pi.loc[gr.image]
                    e["labels_csv"] = dict(book=pr.book, page=pr.page, column=pr.column, tier=pr.tier, rule=pr.rule, label=pr.label, ocr_char=pr.ocr_char, syllable=pr.syllable)
                    e["labels_csv_book_dung_cuon"] = pr.book == Lb.BOOK_VAL[b]
                    e["labels_csv_page_khop"] = pr.page == r.page
                    e["labels_csv_label_khop_style_csv"] = pr.label == r.label
                    e["label_bang_ocr_char"] = pr.label == pr.ocr_char
                    e["tang_GOLD"] = pr.tier == "GOLD"
                    e["rule"] = pr.rule
                    e["rule_la_neo_tu_dong"] = pr.rule.startswith(Lb.ANCHOR_PREFIX)
                    e["rule_thuan_khong_hau_to"] = "|" not in pr.rule
                    e["label_thuoc_R_am"] = pr.label in C.R_of(pr.syllable)
                    # ảnh gốc của ô (crop dataset) — chỉ ghi kích thước; "cùng ô" được kiểm bằng dựng lại crop chuẩn bằng mã pipeline ở r_goi_du_lieu_02b
                    og = cv2.imread(str(Lb.REPO / r.duong_crop), cv2.IMREAD_GRAYSCALE)
                    e["anh_goc_doc_duoc"] = og is not None
                    if og is not None:
                        e["anh_goc_kich_thuoc"] = [int(og.shape[1]), int(og.shape[0])]
                        e["anh_goc_ti_le_phong_len_128"] = round(128 / max(og.shape), 2)
                else:
                    e["labels_csv"] = None
            # ---- 3. chon_chu / gate / trước chon_chu
            k = "|".join([Lb.BOOK_VAL[b], str(r.page), str(r.column), str(r.nom_idx), str(r.syl_idx)])
            if k in post.index:
                po = post.loc[k]
                e["sau_chon_chu"] = dict(tier=po.tier, rule=po.rule, label=po.label, chon_chu=po.get("chon_chu", ""), gate_reason=po.get("gate_reason", ""),
                                         chon_chu_truoc=po.get("chon_chu_truoc", ""), am=po.syllable)
            else:
                e["sau_chon_chu"] = None
            if pre is not None and k in pre.index:
                pp = pre.loc[k]
                e["truoc_chon_chu"] = dict(tier=pp.tier, rule=pp.rule, label=pp.label, gate_reason=pp.get("gate_reason", ""))
                e["nhan_truoc_chon_chu_la_GOLD_thuan"] = (pp.tier == "GOLD" and "|" not in pp.rule and pp.label == pp.ocr_char)
            elif b in Lb.STT:
                e["truoc_chon_chu"] = "STT: bảng nhãn duy nhất labels_final.csv (chon_chu TẮT cho STT: config/chon_chu.yaml enabled:false)"
            e["di_qua_chon_chu"] = bool(e.get("sau_chon_chu") and e["sau_chon_chu"]["chon_chu"] != "")
            e["di_qua_gate"] = bool(e.get("sau_chon_chu") and e["sau_chon_chu"]["gate_reason"] != "")
            rb["anh"].append(e)
        res["sach"][b] = rb
    # ---------------------------------------------------------------- tái lập khâu chọn ứng viên phong cách (05_tong_hop) CHỈ từ neo/<ma>.csv + gold_exact.csv (không nạp tệp đo nhãn người)
    HUMAN_COLS = {"gt_char", "slot", "pos_known", "pos_ok", "keep_level", "ref", "y", "y_ref", "f_hum", "n_hum", "f_humk", "f_vW", "f_vP", "nom_text", "hn_text", "SinoNom_Char"}
    repro = {}
    for b in Lb.ORDER:
        N = Lb.rd(Lb.HTD / "neo" / f"{b}.csv")
        G = Lb.rd(Lb.REPO / "dataset" / Lb.DS[b] / "gold_exact.csv")
        if b in Lb.STT:
            G = G[G.set8 == b]
        G = G.set_index("image")
        Ng = N[(N.tier == "GOLD") & (N.trong_pub == "True")].copy()
        Ng["w"] = Ng.w.astype(float); Ng["h"] = Ng.h.astype(float)
        for c in ("ink_pct", "stray_ink", "border_ink"):
            Ng[c] = pd.to_numeric(Ng[c], errors="coerce")
        geo = np.zeros(len(Ng), bool)
        for f in GEO:
            geo |= (Ng.image.map(G[f]).fillna("0") == "1").to_numpy()
        geo |= (Ng.image.map(G.one_char_ok).fillna("0") != "1").to_numpy()
        geo |= (Ng.image.map(G.crop_status).fillna("") != "ok").to_numpy()
        Ng["canh_ngan"] = np.minimum(Ng.w, Ng.h)
        Rk = Ng[~geo & (Ng.canh_ngan >= 24)].copy()
        cc = collections.Counter(Rk.label)
        hm, wm = Rk.h.median(), Rk.w.median()
        q1, q3 = Rk.ink_pct.quantile(0.25), Rk.ink_pct.quantile(0.75)
        sm = Rk.stray_ink.median()
        Sx = Rk[(Rk.h.between(0.8 * hm, 1.25 * hm)) & (Rk.w.between(0.8 * wm, 1.25 * wm)) & (Rk.ink_pct.between(q1, q3)) & ((Rk.stray_ink <= sm) | Rk.stray_ink.isna()) & (Rk.label.map(cc) >= 5)].copy()
        iqr = max(q3 - q1, 1e-6)
        Sx["diem"] = (Sx.h / hm - 1).abs() + (Sx.w / wm - 1).abs() + (Sx.ink_pct - Rk.ink_pct.median()).abs() / iqr
        Sx = Sx.sort_values(["diem", "image"]).drop_duplicates("label").head(12)
        cur = Lb.rd(Lb.HTD / "style_ung_vien" / f"{b}.csv").sort_values("hang", key=lambda x: x.astype(int))
        repro[b] = dict(n_pool_tai_lap=int(len(Rk)), top12_trung_style_ung_vien=(list(Sx.label) == list(cur.label) and [f"{p}|{c}|{n}|{y}" for p, c, n, y in zip(Sx.page, Sx.column, Sx.nom_idx, Sx.syl_idx)] == [f"{p}|{c}|{n}|{y}" for p, c, n, y in zip(cur.page, cur.column, cur.nom_idx, cur.syl_idx)]),
                        cot_neo=sorted(N.columns), cot_neo_la_nhan_nguoi=sorted(set(N.columns) & HUMAN_COLS), cot_style_csv_la_nhan_nguoi=sorted(set(cur.columns) & HUMAN_COLS),
                        cot_gold_exact_la_nhan_nguoi=sorted(set(G.columns) & HUMAN_COLS))
    res["tong_hop"]["tai_lap_chon_ung_vien_05"] = repro
    inv.check("tai_lap_05_(chi_tu_neo+gold_exact)_trung_12_ung_vien_dung_thu_tu_10_cuon", all(v["top12_trung_style_ung_vien"] for v in repro.values()), [b for b, v in repro.items() if not v["top12_trung_style_ung_vien"]])
    inv.check("cot_cua_neo/style_ung_vien/gold_exact_khong_co_cot_nhan_nguoi_(gt_char,ref,f_hum,f_vW,y,slot,...)", not any(v["cot_neo_la_nhan_nguoi"] or v["cot_style_csv_la_nhan_nguoi"] or v["cot_gold_exact_la_nhan_nguoi"] for v in repro.values()),
              {b: (v["cot_neo_la_nhan_nguoi"], v["cot_gold_exact_la_nhan_nguoi"]) for b, v in repro.items() if v["cot_neo_la_nhan_nguoi"] or v["cot_gold_exact_la_nhan_nguoi"]})
    # ---------------------------------------------------------------- tự điều kiện: ảnh sinh của CHÍNH chữ phong cách (style_0 là ảnh thật của chữ đó) -> bao nhiêu ô của sách mang chữ ấy (theo chữ kim, KHÔNG nhãn người)
    tu_dk = {}
    for b in Lb.ORDER:
        d_ = Lb.rd(Lb.REPO / Lb.PREF[b], usecols=lambda c: c in ("book", "ocr_char", "label", "tier"))
        if b in Lb.STT:
            d_ = d_[d_.book == Lb.BOOK_VAL[b]]
        ch0 = res["sach"][b]["anh"][0]["chu"]
        tu_dk[b] = dict(chu_style_0=ch0, cell_nguon=res["sach"][b]["anh"][0]["cell_uid_ky_vong"], so_o_kim_doc_chu_do=int((d_.ocr_char == ch0).sum()), tong_o=int(len(d_)), ti_le_o_pct=round(100 * float((d_.ocr_char == ch0).mean()), 3))
    res["tong_hop"]["tu_dieu_kien_chu_style_0"] = tu_dk
    # ---------------------------------------------------------------- bất biến theo ảnh
    A = [(b, e) for b in Lb.ORDER for e in res["sach"][b]["anh"]]
    inv.check("20_anh_dung_lai_style_image_trung_byte", all(e["dung_lai_style_image_trung_byte_voi_goi"] for _, e in A), [(b, e["j"]) for b, e in A if not e["dung_lai_style_image_trung_byte_voi_goi"]])
    inv.check("20_anh_mode_L_128x128", all(e["kieu_anh"]["mode"] == "L" and e["kieu_anh"]["size"] == [128, 128] for _, e in A), [(b, e["kieu_anh"]) for b, e in A if not (e["kieu_anh"]["mode"] == "L" and e["kieu_anh"]["size"] == [128, 128])])
    inv.check("20_anh_trung_tep_spec_styles", all(e["khop_tep_spec_styles"] for _, e in A))
    inv.check("20_anh_khong_rong_khong_den_kin (mực 3–40 %)", all(3 <= e["muc_pct"] <= 40 for _, e in A), [(b, e["j"], e["muc_pct"]) for b, e in A if not 3 <= e["muc_pct"] <= 40])
    inv.check("20_anh_duong_cache_dung_cuon_va_dinh_dang", all(e["duong_cache_dung_dinh_dang_va_cuon"] for _, e in A))
    inv.check("20_anh_cell_uid_co_trong_gold_exact_cua_cuon", all(e["cell_uid_co_trong_gold_exact"] for _, e in A))
    inv.check("20_anh_labels_csv_cot_book_dung_cuon", all(e.get("labels_csv_book_dung_cuon") for _, e in A), [(b, e["j"]) for b, e in A if not e.get("labels_csv_book_dung_cuon")])
    inv.check("20_anh_tang_GOLD_label==ocr_char_rule_s1_inter_s2_direct*_label∈R(am)", all(e.get("tang_GOLD") and e.get("label_bang_ocr_char") and e.get("rule_la_neo_tu_dong") and e.get("label_thuoc_R_am") for _, e in A))
    inv.check("20_anh_khong_co_hinh_hoc_gold_exact", all(e.get("cac_co_hinh_hoc_khong_bat") for _, e in A), [(b, e["j"]) for b, e in A if not e.get("cac_co_hinh_hoc_khong_bat")])
    # ---------------------------------------------------------------- khác cuốn không trùng
    shas = collections.Counter(e["sha256_png"] for _, e in A)
    uids = collections.Counter(e["cell_uid_ky_vong"] for _, e in A)
    inv.check("20_anh_sha256_khac_nhau_doi_mot", max(shas.values()) == 1, {k[:10]: v for k, v in shas.items() if v > 1})
    inv.check("20_anh_cell_uid_khac_nhau_doi_mot", max(uids.values()) == 1)
    keys = [(b, e["j"]) for b, e in A]
    mx = []
    for i in range(len(keys)):
        for k2 in range(i + 1, len(keys)):
            mx.append((ncc(imgs[keys[i]], imgs[keys[k2]]), keys[i], keys[k2]))
    mx.sort(reverse=True)
    res["tong_hop"]["NCC_lon_nhat_giua_cac_cap_anh_khac_nhau"] = [dict(ncc=round(v, 3), a=f"{a[0]}_s{a[1]}", b=f"{c[0]}_s{c[1]}") for v, a, c in mx[:6]]
    res["tong_hop"]["NCC_cung_chu_khac_cuon"] = []
    byc = collections.defaultdict(list)
    for b, e in A:
        byc[e["chu"]].append((b, e["j"]))
    for ch, lst in byc.items():
        if len(lst) > 1:
            res["tong_hop"]["NCC_cung_chu_khac_cuon"].append(dict(chu=ch, anh=[f"{b}_s{j}" for b, j in lst], ncc=[round(ncc(imgs[x], imgs[y]), 3) for ii, x in enumerate(lst) for y in lst[ii + 1:]]))
    inv.check("20_anh_NCC_cap_khac_nhau_<0.9 (không nhầm cùng một ảnh)", mx[0][0] < 0.9, mx[0])
    # ---------------------------------------------------------------- 6. hồ ô neo + đề xuất thay thế
    pool = {}
    outd = Lb.OUT / "de_xuat_style"
    for b in Lb.ORDER:
        N = Lb.rd(Lb.HTD / "neo_khuyen_nghi" / f"{b}.csv")
        N["w"] = N.w.astype(float); N["h"] = N.h.astype(float)
        for c in ("ink_pct", "stray_ink"):
            N[c] = pd.to_numeric(N[c], errors="coerce")
        plain = (N.rule == "s1_inter_s2_direct") & (N.chon_chu == "") & (N.gate_reason == "")
        cur = res["sach"][b]["anh"]
        pool[b] = dict(n_ho=int(len(N)), n_thuan_khong_chon_chu_khong_gate=int(plain.sum()), ti_le_thuan_pct=round(100 * float(plain.mean()), 1),
                       n_qua_chon_chu=int((N.chon_chu != "").sum()), n_qua_gate=int((N.gate_reason != "").sum()),
                       cac_chon_chu=dict(collections.Counter(N.chon_chu[N.chon_chu != ""]).most_common(6)),
                       hai_anh_hien_tai_thuan=[bool(e.get("di_qua_chon_chu") is False and e.get("di_qua_gate") is False) for e in cur],
                       hai_anh_hien_tai_hau_to_luat=[(e["labels_csv"]["rule"].split("|", 1)[1] if e.get("labels_csv") and "|" in e["labels_csv"]["rule"] else "") for e in cur],
                       hai_anh_hien_tai_qua_chon_chu=[e.get("di_qua_chon_chu") for e in cur],
                       hai_anh_hien_tai_rule=[e["labels_csv"]["rule"] if e.get("labels_csv") else None for e in cur])
    res["tong_hop"]["ho_o_neo_theo_cuon"] = pool
    # bất biến theo hồ: bỏ MỌI ô đã qua chon_chu/gate khỏi hồ (cờ hình học GIỮ NGUYÊN như 05) rồi chọn lại bằng đúng công thức 05 — hạng 1,2 có đổi không?
    inva = {}
    for b in Lb.ORDER:
        N = Lb.rd(Lb.HTD / "neo_khuyen_nghi" / f"{b}.csv")
        N["w"] = N.w.astype(float); N["h"] = N.h.astype(float)
        for c in ("ink_pct", "stray_ink"):
            N[c] = pd.to_numeric(N[c], errors="coerce")
        cur = res["sach"][b]["anh"]
        out_b = {}
        for ten, mask in (("hoan_toan_khong_chon_chu_khong_gate", (N.chon_chu == "") & (N.gate_reason == "")),):
            Rk = N[mask.to_numpy()].copy()
            cc = collections.Counter(Rk.label)
            hm, wm = Rk.h.median(), Rk.w.median()
            q1, q3 = Rk.ink_pct.quantile(0.25), Rk.ink_pct.quantile(0.75)
            sm = Rk.stray_ink.median()
            Sx = Rk[(Rk.h.between(0.8 * hm, 1.25 * hm)) & (Rk.w.between(0.8 * wm, 1.25 * wm)) & (Rk.ink_pct.between(q1, q3)) & ((Rk.stray_ink <= sm) | Rk.stray_ink.isna()) & (Rk.label.map(cc) >= 5)].copy()
            iqr = max(q3 - q1, 1e-6)
            Sx["diem"] = (Sx.h / hm - 1).abs() + (Sx.w / wm - 1).abs() + (Sx.ink_pct - Rk.ink_pct.median()).abs() / iqr
            Sx = Sx.sort_values(["diem", "image"]).drop_duplicates("label").head(2)
            keys = [f"{r.page}|{r.column}|{r.nom_idx}|{r.syl_idx}" for r in Sx.itertuples()]
            cur_keys = [e["cell_uid_ky_vong"].split("/")[2] + "|" + e["cell_uid_ky_vong"].split("/")[3][1:] + "|" + e["cell_uid_ky_vong"].split("/")[4][1:] + "|" + e["cell_uid_ky_vong"].split("/")[5][1:] for e in cur]
            out_b[ten] = dict(n_ho=int(len(Rk)), so_o_bi_loai=int((~mask).sum()), hang1_2_moi=list(Sx.label), hang1_2_hien_tai=[e["chu"] for e in cur], hai_hang_dau_giong_het=(keys == cur_keys))
        inva[b] = out_b
    res["tong_hop"]["bat_bien_khi_bo_o_qua_chon_chu"] = inva
    # đề xuất cho 4 bộ evaluation_only: lọc hồ về ô neo THUẦN (không qua chon_chu/gate, GOLD cả trước và sau), chọn lại bằng đúng công thức của 05_tong_hop
    prop = {}
    for b in Lb.ORDER:
        if b not in Lb.HUMAN:
            continue          # chỉ 4 bộ evaluation_only (B18 B34 L16 TK) cần ảnh phong cách không dựa vào quyết định của bộ chọn chữ học trên nhãn người; 6 bộ còn lại hạng 1,2 không đổi khi bỏ ô qua chon_chu (xem bat_bien_khi_bo_o_qua_chon_chu)
        N = Lb.rd(Lb.HTD / "neo" / f"{b}.csv")         # neo (mọi ô neo có crop) — để tự lọc cờ hình học KHÔNG gồm f_ink (phụ thuộc nhãn cuối)
        G = Lb.rd(Lb.REPO / "dataset" / Lb.DS[b] / "gold_exact.csv").set_index("image")
        pre_t = Lb.rd(Lb.REPO / Lb.PREF[b], usecols=lambda c: c in set(KEY + ["tier", "rule", "label", "ocr_char"]))
        pre_t["_k"] = pre_t[KEY].agg("|".join, axis=1)
        N["_k"] = N[["page", "column", "nom_idx", "syl_idx"]].agg("|".join, axis=1).map(lambda s: Lb.BOOK_VAL[b] + "|" + s)
        N = N.merge(pre_t[["_k", "tier", "rule", "label", "ocr_char"]].rename(columns=dict(tier="tier_truoc", rule="rule_truoc", label="label_truoc", ocr_char="ocr_truoc")), on="_k", how="left")
        ok = (N.tier == "GOLD") & (N.trong_pub == "True") & (N.tier_truoc == "GOLD") & (N.rule == "s1_inter_s2_direct") & (N.rule_truoc == "s1_inter_s2_direct") \
            & (N.chon_chu == "") & (N.gate_reason == "") & (N.label_truoc == N.ocr_truoc) & (N.label == N.label_truoc)
        flags_ok = np.ones(len(N), bool)
        for f in GEO:
            flags_ok &= (N.image.map(G[f]).fillna("0") != "1").to_numpy()
        flags_ok &= (N.image.map(G.crop_status).fillna("") == "ok").to_numpy()
        N["w"] = N.w.astype(float); N["h"] = N.h.astype(float)
        for c in ("ink_pct", "stray_ink"):
            N[c] = pd.to_numeric(N[c], errors="coerce")
        N["canh_ngan"] = np.minimum(N.w, N.h)
        Rk = N[ok.to_numpy() & flags_ok & (N.canh_ngan >= 24).to_numpy()].copy()
        # đường crop chuẩn 128 theo cell_uid (neo/<ma>.csv chưa có cột cache): dựng như 05_tong_hop
        Rk["cell_uid"] = Rk.image.map(G.cell_uid)
        Rk["c128"] = [("prepared/_gold_exact/crop_chuan/sq128/" + "/".join(u.split("/")[:3]) + "/" + "_".join(u.split("/")[3:]) + ".png") for u in Rk.cell_uid]
        Rk = Rk[[(Lb.REPO / p).is_file() for p in Rk.c128]]
        cc = collections.Counter(Rk.label)
        hm, wm = Rk.h.median(), Rk.w.median()
        q1, q3 = Rk.ink_pct.quantile(0.25), Rk.ink_pct.quantile(0.75)
        sm = Rk.stray_ink.median()
        Sx = Rk[(Rk.h.between(0.8 * hm, 1.25 * hm)) & (Rk.w.between(0.8 * wm, 1.25 * wm)) & (Rk.ink_pct.between(q1, q3)) & ((Rk.stray_ink <= sm) | Rk.stray_ink.isna()) & (Rk.label.map(cc) >= 5)].copy()
        iqr = max(q3 - q1, 1e-6)
        Sx["diem"] = (Sx.h / hm - 1).abs() + (Sx.w / wm - 1).abs() + (Sx.ink_pct - Rk.ink_pct.median()).abs() / iqr
        Sx = Sx.sort_values(["diem", "image"]).drop_duplicates("label").head(12)
        rows = []
        outd.mkdir(parents=True, exist_ok=True)
        for j, (_, r) in enumerate(Sx.head(2).iterrows()):
            g = cv2.imread(str(Lb.REPO / r.c128), cv2.IMREAD_GRAYSCALE)
            im = style_image(g)
            cv2.imwrite(str(outd / f"{b}_s{j}.png"), im)
            rows.append(dict(j=j, chu=r.label, cell_uid=r.cell_uid, nguon=r.c128, page=r.page, column=r.column, nom_idx=r.nom_idx, syl_idx=r.syl_idx, w=r.w, h=r.h, ink_pct=r.ink_pct, diem=round(float(r.diem), 5),
                             rule=r.rule, rule_truoc=r.rule_truoc, tier_truoc=r.tier_truoc, **stats(im), sha256=Lb.sha256_file(outd / f"{b}_s{j}.png")))
        prop[b] = dict(n_ho_thuan_sau_loc=int(len(Rk)), n_ung_vien_du_dieu_kien=int(len(Sx)), cach_chon="đúng công thức w_hien_trang_du_lieu_05_tong_hop (±25 % kích thước trung vị, mực [p25,p75], chữ ≥ 5 ô, điểm khoảng cách) nhưng trên hồ THUẦN: "
                       "tier GOLD cả trước và sau chon_chu, rule == 's1_inter_s2_direct' trước và sau, chon_chu rỗng, gate_reason rỗng, nhãn == chữ kim; cờ hình học = 9 cờ + crop_status (KHÔNG dùng f_ink/one_char_ok vì f_ink dùng trung vị theo nhãn cuối)",
                       de_xuat=rows)
    res["tong_hop"]["de_xuat_thay_anh"] = prop
    for b, pr in prop.items():
        inv.check(f"de_xuat_{b}_co_du_2_anh_thuan", len(pr["de_xuat"]) == 2 and all(r["rule"] == "s1_inter_s2_direct" and r["tier_truoc"] == "GOLD" for r in pr["de_xuat"]),
                  [(r["chu"], r["muc_pct"], r["thanh_phan_ge12px"]) for r in pr["de_xuat"]])
    res["bat_bien"] = inv.summary()
    res["bat_bien_chi_tiet"] = inv.rows
    res["giay"] = round(time.time() - t0, 1)
    Lb.jdump(res, "02_anh_phong_cach.json")
    print(f"[02] xong {res['giay']}s; bất biến {res['bat_bien']['dat']}/{res['bat_bien']['tong']} rớt={res['bat_bien']['rot']}")


if __name__ == "__main__":
    main()
