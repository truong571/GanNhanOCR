"""Phân tích bổ sung cho kết hợp ba luồng (chạy SAU ket_hop.py, dùng chi_tiet/*_luong.csv):
 1. M1: các ô luật B1/D1 đề xuất chữ ∈R nhưng ≠K — là biến thể giản/phồn của K (OpenCC) hay chữ khác hẳn? (độ chính xác thay thế thật)
 2. M2: 100 ô cứu được — bao nhiêu là chữ 'gần hình' của K theo SinoNom_Similar (cầu tự dạng), bao nhiêu trùng nhãn pipeline
 3. M2: 171 ô engine mới XÁC NHẬN K (∉R) — K có âm nào trong từ điển gần với âm QN (bỏ dấu/thanh) không → nghi từ điển thiếu / VietOCR rụng dấu
 4. Phân bố theo |R| và theo K∈charset Paddle
 5. Chữ cứu được có phải chữ Hán thông dụng (URO) hay chữ Nôm riêng (Ext-B)?
Chạy: /tmp/venv_paddle/bin/python phan_tich_them.py   (cần opencc; pandas)  — ra phan_tich_them.json
"""
import sys, json, csv, ast, collections, unicodedata
from pathlib import Path
import pandas as pd
import opencc

REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
OUT = REPO / "lab/ocr_compare/ket_hop"
rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
t2s = opencc.OpenCC("t2s")
same_variant = lambda a, b: a != "" and b != "" and (a == b or t2s.convert(a) == t2s.convert(b))
inR = lambda c, r: c != "" and c in r


def load_similar():
    sim = {}
    with open(REPO / "Dict/SinoNom_Similar.csv", encoding="utf-8-sig") as f:
        rdr = csv.reader(f); next(rdr)
        for row in rdr:
            try: sim[row[0].strip()] = set(ast.literal_eval(row[1]))
            except Exception: pass
    return sim


def strip_tone(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d")


def main():
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import load_qn_to_nom, build_nom_to_qn
    q2n = load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv")); n2q = build_nom_to_qn(q2n)
    sim = load_similar()
    charset = set(open(REPO / "lab/ocr_compare/paddle/nom_vocab.txt", encoding="utf-8").read().split()) if (REPO / "lab/ocr_compare/paddle/nom_vocab.txt").exists() else set()
    res = {}
    mau = {s: rd(REPO / f"lab/ocr_compare/mau/{s}.csv").set_index("sample_id") for s in ("M1", "M2", "M3")}
    # ---------- 1. M1: đề xuất ∈R nhưng ≠K
    d1 = rd(OUT / "chi_tiet/M1_luong.csv"); d1["R"] = mau["M1"].loc[d1.sample_id, "R_candidates"].map(lambda x: set(t for t in x.split("|") if t)).values
    def b1(r):
        for c in (r.P, r.G):
            if inR(c, r.R): return c
        return ""
    d1["B1"] = d1.apply(b1, axis=1)
    fired = d1[d1.B1 != ""]; ne = fired[fired.B1 != fired.K]
    var = ne[[same_variant(a, b) for a, b in zip(ne.B1, ne.K)]]
    simk = ne[[b in sim.get(a, set()) or a in sim.get(b, set()) for a, b in zip(ne.B1, ne.K)]]
    simk_ids = set(simk.sample_id)
    res["M1_B1_de_xuat_neK"] = dict(
        fired=int(len(fired)), eqK=int((fired.B1 == fired.K).sum()), neK=int(len(ne)),
        neK_bien_the_gian_phon_OpenCC=int(len(var)), neK_gan_hinh_SinoNomSimilar=int(len(simk)),
        neK_khac_han=int(len(ne) - len(set(var.index) | set(simk.index))),
        precision_neu_coi_bien_the_la_dung=round(100 * (len(fired) - (len(ne) - len(var))) / len(fired), 1),
        vi_du_neK=[(r.sample_id, r.K, r.syllable, r.B1, "biến thể" if same_variant(r.B1, r.K) else ("gần hình" if r.sample_id in simk_ids else "khác")) for _, r in ne.head(41).iterrows()])
    # cùng cho A2 (P==G ∈R) trên M1
    a2 = d1[(d1.P != "") & (d1.P == d1.G) & pd.Series([inR(p, r) for p, r in zip(d1.P, d1.R)], index=d1.index)]
    ne2 = a2[a2.P != a2.K]
    res["M1_A2_de_xuat_neK"] = dict(fired=int(len(a2)), neK=int(len(ne2)), neK_bien_the=int(sum(same_variant(a, b) for a, b in zip(ne2.P, ne2.K))),
                                    vi_du=[(r.K, r.syllable, r.P) for _, r in ne2.iterrows()])
    # ---------- 2. M2: ô cứu được
    d2 = rd(OUT / "chi_tiet/M2_luong.csv"); d2["R"] = mau["M2"].loc[d2.sample_id, "R_candidates"].map(lambda x: set(t for t in x.split("|") if t)).values
    d2["B1"] = d2.apply(b1, axis=1); d2["src"] = ["P" if inR(p, r) else ("G" if inR(g, r) else "") for p, g, r in zip(d2.P, d2.G, d2.R)]
    resc = d2[d2.B1 != ""]
    res["M2_cuu"] = dict(
        n=int(len(resc)), nguon=dict(collections.Counter(resc.src)),
        ca_hai_dong_y=int(((resc.P == resc.G)).sum()),
        gan_hinh_K_theo_SinoNomSimilar=int(sum(b in sim.get(a, set()) or a in sim.get(b, set()) for a, b in zip(resc.B1, resc.K))),
        bien_the_gian_phon_cua_K=int(sum(same_variant(a, b) for a, b in zip(resc.B1, resc.K))),
        SILVER_trung_nhan_pipeline=f"{int(((resc.tier == 'SILVER') & (resc.B1 == resc.label)).sum())}/{int((resc.tier == 'SILVER').sum())}",
        SILVER_khac_nhan_pipeline=[(r.sample_id, r.K, r.syllable, r.B1, r.label, r.src) for _, r in resc[(resc.tier == "SILVER") & (resc.B1 != resc.label)].iterrows()],
        khoi_unicode_chu_cuu=dict(collections.Counter("URO" if ord(c) < 0x10000 else "ExtB+" for c in resc.B1)),
        khoi_unicode_K=dict(collections.Counter("URO" if ord(c) < 0x10000 else "ExtB+" for c in resc.K)),
        theo_R_size=dict(collections.Counter(pd.cut(resc.R_size.astype(int), [0, 4, 9, 19, 49, 999]).astype(str))),
        R_size_trung_vi_cuu=float(resc.R_size.astype(int).median()), R_size_trung_vi_khong_cuu=float(d2[d2.B1 == ""].R_size.astype(int).median()),
        vi_du=[(r.sample_id, r.tier, r.K, r.syllable, r.B1, r.label, r.src) for _, r in resc.head(30).iterrows()])
    # ---------- 3. M2: engine xác nhận K (∉R)
    conf = d2[(d2.P == d2.K) | (d2.G == d2.K)]
    def am_gan(k, syl):
        qs = n2q.get(k, [])
        s0 = strip_tone(syl)
        return dict(co_am_trong_tu_dien=bool(qs), cung_am_bo_dau=any(strip_tone(q) == s0 for q in qs), am=qs[:6])
    ag = [am_gan(k, s) for k, s in zip(conf.K, conf.syllable)]
    res["M2_engine_xac_nhan_K"] = dict(
        n=int(len(conf)), pct=round(100 * len(conf) / len(d2), 1),
        K_co_am_trong_tu_dien=int(sum(a["co_am_trong_tu_dien"] for a in ag)),
        K_cung_am_bo_dau_thanh=int(sum(a["cung_am_bo_dau"] for a in ag)),
        K_khong_co_trong_tu_dien=int(sum(not a["co_am_trong_tu_dien"] for a in ag)),
        theo_tier=dict(collections.Counter(conf.tier)),
        vi_du_cung_am_bo_dau=[(r.K, r.syllable, a["am"]) for (_, r), a in zip(conf.iterrows(), ag) if a["cung_am_bo_dau"]][:15],
        vi_du_khac_am=[(r.K, r.syllable, a["am"][:3]) for (_, r), a in zip(conf.iterrows(), ag) if a["co_am_trong_tu_dien"] and not a["cung_am_bo_dau"]][:15])
    # toàn M2: K có âm bỏ dấu trùng?
    ag_all = [am_gan(k, s) for k, s in zip(d2.K, d2.syllable)]
    res["M2_toan_bo_K_cung_am_bo_dau"] = int(sum(a["cung_am_bo_dau"] for a in ag_all))
    # ---------- 4. theo K ∈ charset Paddle (Paddle không thể đọc ra chữ ngoài charset)
    try:
        cs = json.load(open(REPO / "lab/ocr_compare/paddle/charset_coverage.json"))
    except Exception:
        cs = None
    res["ghi_chu_charset"] = "xem paddle/charset_coverage.json" if cs else "không có"
    # ---------- 5. M3 — engine đọc ra gì khi K=㝵
    d3 = rd(OUT / "chi_tiet/M3_luong.csv")
    res["M3"] = dict(P_top=collections.Counter(d3.P).most_common(6), G_top=collections.Counter(d3.G).most_common(6),
                     P_eq_G=int(((d3.P == d3.G) & (d3.P != "")).sum()), bat_ky_luong_ra_𠊚=int(sum((d3[c] == "𠊚").sum() for c in ("P", "G", "P6c", "P5c", "PVL", "Gf"))),
                     bat_ky_luong_ra_㝵=int(sum((d3[c] == "㝵").sum() for c in ("P", "G", "P6c", "P5c", "PVL", "Gf"))))
    json.dump(res, open(OUT / "phan_tich_them.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
