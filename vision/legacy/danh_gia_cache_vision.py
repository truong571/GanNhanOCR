#!/usr/bin/env python3
"""Đánh giá OFFLINE (0 request) các phản hồi Google Vision đã cache bởi stitched_vision_audit.py:
phân bổ ký hiệu Vision về trang gốc, ghép theo TỪNG Ô với nhãn pipeline (dataset/<Bộ>/labels.csv: bbox, ocr_char kim, label, tier),
và đo: độ phủ, tỉ lệ khớp ô, chữ Vision có thuộc R(âm) không, độ hiệu lực của phép kiểm "chữ có xuất hiện trên trang".

  .venv/bin/python lab/ocr_compare/danh_gia_cache_vision.py [--out measure_out/_vision_audit/danh_gia_cache.json]

Chỉ đọc: cache/google_vision/stitched/*.json, usage_ledger.json, prepared/*/pages_denoised (kích thước), dataset/*/labels.csv, Dict/QuocNgu_SinoNom.csv.
KHÔNG đọc khoá dịch vụ, KHÔNG gọi mạng.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = HERE / "cache" / "google_vision"
TARGET_H, SEP = 1800, 30   # giống stitch_pages() của stitched_vision_audit.py

# 10 mẫu cứng trong stitched_vision_audit.py (cặp liên tiếp = một ảnh ghép; thứ tự gọi: pair_1..pair_5)
ITEMS = [
    ("stt2", "SachThanhTruyen2", "page_0312", "trời", "(chờ duyệt)"), ("stt4", "SachThanhTruyen4", "page_0060", "mà", "麻"),
    ("stt11", "SachThanhTruyen11", "page_0016", "mà", "麻"), ("L83", "LucVanTien1883", "page_0048", "pha", "波"),
    ("KVK", "KimVanKieu1884", "page_0075", "nhà", "茹"), ("Chr", "Chrestomathie1872", "page_0021", "nói", "呐"),
    ("TK", "TruyenKieu1872", "page_0042", "nước", "渃"), ("L16", "LucVanTien1916", "page_0020", "sau", "𡢐"),
    ("B18", "SachKinhThayCaBinh", "page_0088", "chúa", "主"), ("B34", "SachDungLyHoThan", "page_0081", "chịu", "召"),
]


def cls(ch: str) -> str:
    if not ch:
        return "rỗng"
    c = ch[0]
    nm = unicodedata.name(c, "")
    cat = unicodedata.category(c)
    if "CJK" in nm:
        return "Hán"
    if cat == "Co":
        return "PUA"
    if nm.startswith("LATIN") or c.isalpha():
        return "Latin/khác chữ"
    if cat.startswith("P") or cat.startswith("S"):
        return "dấu câu"
    if cat.startswith("N"):
        return "số"
    return "khác"


def load_dict():
    R = defaultdict(set)
    with open(REPO / "dict" / "QuocNgu_SinoNom.csv", encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if len(row) >= 2:
                R[row[0].strip().lower()].add(row[1].strip())
    return R


def vision_symbols(resp):
    out = []
    for pg in resp.get("fullTextAnnotation", {}).get("pages", []):
        for b in pg.get("blocks", []):
            for pa in b.get("paragraphs", []):
                for w in pa.get("words", []):
                    for s in w.get("symbols", []):
                        v = s.get("boundingBox", {}).get("vertices", [])
                        if len(v) < 4:
                            continue
                        xs = [p.get("x", 0) for p in v]
                        ys = [p.get("y", 0) for p in v]
                        out.append(dict(ch=s.get("text", ""), cx=sum(xs) / 4, cy=sum(ys) / 4, w=max(xs) - min(xs), h=max(ys) - min(ys),
                                        x0v=xs[0], conf=s.get("confidence")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "ket_qua" / "danh_gia_cache.json"))
    a = ap.parse_args()
    ledger = json.load(open(CACHE / "usage_ledger.json", encoding="utf-8"))
    by_pair = {}
    for h in ledger["history"]:
        d = h.get("description", "")
        if d.startswith("stitched_pair_"):
            by_pair[int(d.split("_")[2].split(".")[0])] = h["hash"]
    R = load_dict()
    rng = random.Random(7)
    rows, per_page = [], []
    for pair in range(1, 6):
        f = CACHE / "stitched" / f"{by_pair[pair]}.json"
        resp = json.load(open(f, encoding="utf-8"))["responses"][0]
        syms = vision_symbols(resp)
        items = ITEMS[(pair - 1) * 2:(pair - 1) * 2 + 2]
        x = 0
        lay = []
        for key, bdir, page, syl, tgt in items:
            im = Image.open(REPO / "prepared" / bdir / "pages_denoised" / f"{page}.png")
            sc = TARGET_H / im.height
            w = int(im.width * sc)
            lay.append(dict(key=key, bdir=bdir, page=page, x0=x, x1=x + w, sc=sc, W=im.width, H=im.height, syl=syl, tgt=tgt))
            x += w + SEP
        for L in lay:
            ps = [s for s in syms if L["x0"] <= s["cx"] < L["x1"]]
            for s in ps:   # về toạ độ trang gốc
                s["px"], s["py"], s["pw"], s["ph"] = (s["cx"] - L["x0"]) / L["sc"], s["cy"] / L["sc"], s["w"] / L["sc"], s["h"] / L["sc"]
            lab = pd.read_csv(REPO / "dataset" / L["bdir"] / "labels.csv")
            lab = lab[lab["page"] == L["page"]].copy()
            bb = lab["bbox"].map(json.loads)
            lab["x0"], lab["y0"], lab["x1"], lab["y1"] = [bb.map(lambda b, i=i: b[i]) for i in range(4)]
            out_of = int(((lab["x1"] > L["W"] + 2) | (lab["y1"] > L["H"] + 2)).sum())
            page_text = "".join(s["ch"] for s in syms if L["x0"] <= s["x0v"] < L["x1"])    # đúng phép "p_text" của kịch bản gốc
            cells = []
            for r in lab.itertuples():
                bw, bh = r.x1 - r.x0, r.y1 - r.y0
                cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
                cand = [s for s in ps if r.x0 - 0.15 * bw <= s["px"] <= r.x1 + 0.15 * bw and r.y0 - 0.15 * bh <= s["py"] <= r.y1 + 0.15 * bh]
                m = min(cand, key=lambda s: (s["px"] - cx) ** 2 + (s["py"] - cy) ** 2) if cand else None
                lbl = r.label if isinstance(r.label, str) else ""
                kim = r.ocr_char if isinstance(r.ocr_char, str) else ""
                cells.append(dict(book=L["key"], page=L["page"], tier=r.tier, syl=str(r.syllable).lower(), label=lbl, kim=kim,
                                  v=(m["ch"] if m else None), vconf=(m["conf"] if m else None), found_page=bool(lbl) and (lbl in page_text)))
            used = {id(s) for s in ps}
            rows += cells
            med_h = float(np.median((lab["y1"] - lab["y0"]).values)) if len(lab) else 0.0
            per_page.append(dict(book=L["key"], page=L["page"], pair=pair, cells=len(cells), vision_symbols=len(ps), bbox_ngoai_anh=out_of,
                                 H=L["H"], scale=round(L["sc"], 3), cell_h_goc=round(med_h, 1), cell_h_sau_ghep=round(med_h * L["sc"], 1),
                                 classes=dict(Counter(cls(s["ch"]) for s in ps))))
    D = pd.DataFrame(rows)
    D["matched"] = D["v"].notna()
    D["agree_label"] = D["matched"] & (D["v"] == D["label"])
    D["agree_kim"] = D["matched"] & (D["v"] == D["kim"])
    D["v_in_R"] = D.apply(lambda r: bool(r["matched"]) and (r["v"] in R.get(r["syl"], set())), axis=1)
    D["v_han"] = D["v"].map(lambda c: cls(c) == "Hán" if isinstance(c, str) else False)

    def rate(m, base):
        return f"{100 * m.sum() / max(1, base.sum()):.1f} % ({m.sum()}/{base.sum()})"

    rep = dict(trang=per_page, tong_o=int(len(D)), ledger_da_dung=ledger["total_api_calls"])
    print(f"== {len(D)} ô pipeline trên {len(per_page)} trang; ledger đã dùng {ledger['total_api_calls']} request ==")
    for p in per_page:
        print(f"  {p['book']:6s} {p['page']}: ô pipeline {p['cells']:4d} | ký hiệu Vision {p['vision_symbols']:4d} | cao trang {p['H']} ×{p['scale']} | cao ô {p['cell_h_goc']}→{p['cell_h_sau_ghep']} px | {p['classes']}")
    allc = D["matched"]
    print(f"Độ phủ: Vision có ký hiệu ở {rate(allc, D['matched'] | ~D['matched'])} ô pipeline")
    tab = {}
    for tier in ["GOLD", "SYLLABLE", "REVIEW", "QUARANTINE", "ALL"]:
        s = D if tier == "ALL" else D[D["tier"] == tier]
        if not len(s):
            continue
        mt = s[s["matched"]]
        tab[tier] = dict(n=len(s), phu=float(s["matched"].mean()), khop_nhan_tren_o_khop=float(mt["agree_label"].mean()) if len(mt) else None,
                         khop_kim_tren_o_khop=float(mt["agree_kim"].mean()) if len(mt) else None,
                         vision_la_chu_Han=float(mt["v_han"].mean()) if len(mt) else None, vision_thuoc_Ram=float(mt["v_in_R"].mean()) if len(mt) else None)
        print(f"  [{tier:10s}] n={len(s):5d} | phủ {100 * tab[tier]['phu']:.1f} % | trên ô có ký hiệu: khớp nhãn {100 * (tab[tier]['khop_nhan_tren_o_khop'] or 0):.1f} %"
              f" · khớp kim {100 * (tab[tier]['khop_kim_tren_o_khop'] or 0):.1f} % · Vision ra chữ Hán {100 * (tab[tier]['vision_la_chu_Han'] or 0):.1f} % · thuộc R(âm) {100 * (tab[tier]['vision_thuoc_Ram'] or 0):.1f} %")
    rep["theo_tang"] = tab
    # theo cuốn (GOLD)
    G = D[(D["tier"] == "GOLD")]
    print("Theo cuốn (GOLD, trên ô có ký hiệu Vision): phủ | khớp nhãn")
    bk = {}
    for b, s in G.groupby("book"):
        mt = s[s["matched"]]
        bk[b] = dict(n=len(s), phu=float(s["matched"].mean()), khop=float(mt["agree_label"].mean()) if len(mt) else None)
        print(f"  {b:6s} n={len(s):4d} phủ {100 * bk[b]['phu']:.1f} % | khớp {100 * (bk[b]['khop'] or 0):.1f} %")
    rep["theo_cuon_GOLD"] = bk
    # bất đồng: ai có lý? (nhãn người không có → dùng R(âm) làm chỉ báo gián tiếp)
    dis = G[G["matched"] & ~G["agree_label"]]
    print(f"Bất đồng Vision≠nhãn (GOLD): {len(dis)} ô; trong đó Vision ra chữ thuộc R(âm) {100 * dis['v_in_R'].mean():.1f} % (nghi dị bản/lỗi pipeline), ngoài R(âm) {100 * (1 - dis['v_in_R'].mean()):.1f} % (nhiều khả năng Vision sai); Vision không phải chữ Hán {100 * (1 - dis['v_han'].mean()):.1f} %")
    rep["bat_dong_GOLD"] = dict(n=int(len(dis)), vision_trong_R=float(dis["v_in_R"].mean()), vision_khong_Han=float(1 - dis["v_han"].mean()))
    # phép kiểm gốc "chữ nhãn có trong văn bản trang" — tỉ lệ 'đã tìm thấy' theo từng nhóm ô GOLD
    ag = G[G["agree_label"]]
    un = G[~G["matched"]]
    print(f"Phép kiểm gốc 'chữ nhãn có trong văn bản trang' (ô GOLD): tìm thấy {100 * G['found_page'].mean():.1f} % tổng; "
          f"ô Vision khớp tại ô {100 * ag['found_page'].mean():.1f} % · ô Vision nói chữ khác {100 * dis['found_page'].mean():.1f} % · ô Vision KHÔNG có ký hiệu {100 * un['found_page'].mean():.1f} %")
    print(f"  ⇒ 'không tìm thấy' xảy ra ở {100 * (1 - G['found_page'].mean()):.1f} % ô GOLD, trong đó {100 * len(un[~un['found_page']]) / max(1, (~G['found_page']).sum()):.1f} % là ô Vision không đọc gì ở đó")
    rep["found_page"] = dict(GOLD=float(G["found_page"].mean()), khi_khop_o=float(ag["found_page"].mean()), khi_vision_noi_khac=float(dis["found_page"].mean()),
                             khi_vision_khong_co_ky_hieu=float(un["found_page"].mean()))
    # giá trị tiềm năng của Vision như tín hiệu độc lập (ngưỡng độ tin cậy ≥ 0,8)
    hc = G[G["matched"] & (G["vconf"].astype(float) >= 0.8)]
    c_ag, c_inR, c_outR = int(hc["agree_label"].sum()), int((~hc["agree_label"] & hc["v_in_R"]).sum()), int((~hc["agree_label"] & ~hc["v_in_R"]).sum())
    S = D[(D["tier"] == "SYLLABLE") & D["matched"] & (D["vconf"].astype(float) >= 0.8) & D["v_in_R"]]
    print(f"Vision conf≥0,8 trên ô GOLD: {len(hc)}/{len(G)} ô ({100 * len(hc) / len(G):.1f} %): xác nhận nhãn {c_ag} · khác nhãn nhưng thuộc R(âm) {c_inR} (ứng viên lỗi pipeline/dị bản) · khác nhãn ngoài R(âm) {c_outR}; "
          f"ô SYLLABLE có thể nâng (conf≥0,8 & thuộc R(âm)): {len(S)}/{int((D['tier'] == 'SYLLABLE').sum())}")
    rep["conf_cao"] = dict(n=int(len(hc)), xac_nhan=c_ag, khac_trong_R=c_inR, khac_ngoai_R=c_outR, syllable_nang_duoc=int(len(S)))
    # theo cuốn: Vision có thể xác nhận (conf≥0,8 & khớp) bao nhiêu % ô GOLD
    conf_book = {b: float(((s["matched"]) & (s["vconf"].astype(float) >= 0.8) & s["agree_label"]).mean()) for b, s in G.groupby("book")}
    print("Tỉ lệ ô GOLD được Vision XÁC NHẬN ở conf≥0,8:", {b: f"{100 * v:.1f} %" for b, v in conf_book.items()})
    rep["xac_nhan_conf_cao_theo_cuon"] = conf_book
    # chữ chẩn đoán: 10 mẫu cứng của kịch bản gốc
    own = []
    for key, bdir, page, syl, tgt in ITEMS:
        s = D[(D["book"] == key)]
        n_on_page = int((s["label"] == tgt).sum()) if tgt != "(chờ duyệt)" else 0
        own.append((key, tgt, n_on_page))
    print("10 mẫu của kịch bản gốc: số lần chữ mục tiêu xuất hiện trên chính trang đó theo nhãn pipeline =", [(k, t, n) for k, t, n in own])
    rep["mau_goc"] = own
    # độ tin cậy Vision và khớp
    mt = G[G["matched"] & G["vconf"].notna()].copy()
    if len(mt):
        mt["bin"] = pd.cut(mt["vconf"].astype(float), [0, 0.5, 0.8, 0.95, 1.01])
        cal = mt.groupby("bin", observed=True)["agree_label"].agg(["mean", "size"])
        print("Khớp nhãn theo độ tin cậy Vision:", {str(k): (round(float(v["mean"]), 3), int(v["size"])) for k, v in cal.iterrows()})
        rep["calib"] = {str(k): (float(v["mean"]), int(v["size"])) for k, v in cal.iterrows()}
    # chuẩn nhãn NGƯỜI (chỉ Borg B18/B34 có bảng nhãn người theo ô): Vision đúng bao nhiêu so với người
    hb = pd.read_csv(REPO / "dataset/_BORG_NHAN_NGUOI/labels.csv", usecols=["book", "page", "char", "bbox", "syllable"])
    hum = {}
    for key, bdir, page in (("B18", "SachKinhThayCaBinh", "page_0088"), ("B34", "SachDungLyHoThan", "page_0081")):
        s_ = hb[(hb["book"].str.lower() == bdir.lower()) & (hb["page"] == page) & hb["char"].notna() & hb["bbox"].notna()]
        pair = next(p_ for p_ in range(1, 6) if any(it[0] == key for it in ITEMS[(p_ - 1) * 2:(p_ - 1) * 2 + 2]))
        resp = json.load(open(CACHE / "stitched" / f"{by_pair[pair]}.json", encoding="utf-8"))["responses"][0]
        syms_all = vision_symbols(resp)
        items = ITEMS[(pair - 1) * 2:(pair - 1) * 2 + 2]
        x = 0
        L = None
        for k2, b2, p2, *_ in items:
            im = Image.open(REPO / "prepared" / b2 / "pages_denoised" / f"{p2}.png")
            sc = TARGET_H / im.height
            w = int(im.width * sc)
            if k2 == key:
                L = (x, x + w, sc)
            x += w + SEP
        ps = [dict(px=(q["cx"] - L[0]) / L[2], py=q["cy"] / L[2], ch=q["ch"], conf=q["conf"]) for q in syms_all if L[0] <= q["cx"] < L[1]]
        n = m_ = ok = 0
        okc = []
        for r in s_.itertuples():
            x0, y0, x1, y1 = json.loads(r.bbox)
            bw, bh = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cand = [q for q in ps if x0 - 0.15 * bw <= q["px"] <= x1 + 0.15 * bw and y0 - 0.15 * bh <= q["py"] <= y1 + 0.15 * bh]
            n += 1
            if cand:
                q = min(cand, key=lambda q: (q["px"] - cx) ** 2 + (q["py"] - cy) ** 2)
                m_ += 1
                ok += int(q["ch"] == r.char)
                okc.append((q["conf"], int(q["ch"] == r.char)))
        hi = [o for c, o in okc if c is not None and c >= 0.8]
        hum[key] = dict(o_nguoi=n, vision_co_ky_hieu=m_, vision_dung=ok, dung_tren_o_co_ky_hieu=(ok / m_ if m_ else None),
                        dung_khi_conf_ge_0_8=(sum(hi) / len(hi) if hi else None), n_conf_ge_0_8=len(hi))
        print(f"NHÃN NGƯỜI {key} {page}: {n} ô; Vision có ký hiệu {m_} ({100 * m_ / max(1, n):.1f} %); đúng {ok} = {100 * ok / max(1, m_):.1f} % trên ô có ký hiệu ({100 * ok / max(1, n):.1f} % trên mọi ô người); "
              f"khi conf≥0,8: {100 * (sum(hi) / len(hi) if hi else 0):.1f} % đúng (n={len(hi)})")
    rep["nhan_nguoi_borg"] = hum
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("->", a.out)


if __name__ == "__main__":
    main()
