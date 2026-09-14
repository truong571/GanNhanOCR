"""KẾT HỢP ba luồng OCR (kinhhannom K, PaddleOCR P, GLM-OCR G) trên CÙNG bộ mẫu M1/M2/M3 (mau/, seed 20260911).

Luồng chính (chế độ CỘT — cấu hình tốt nhất của mỗi engine, đọc nguyên cột theo hộp cột kinhhannom):
  K  = ocr_char (kinhhannom, cache prepared/*/detected/*_ocr_cache.json)
  P  = PaddleOCR PP-OCRv6_medium_rec, cột xoay 90° CCW, gióng đơn điệu theo toạ độ CTC (pred_mono) — KHÔNG dùng chuỗi K
       P_y   = gióng theo y; P_all = mọi chữ CTC rơi vào ô
  G  = GLM-OCR (mlx) đọc nguyên cột, căn Levenshtein với chuỗi K để lấy chữ ở vị trí ô (col-ctx) — có thiên vị nhẹ về K
Luồng phụ (chế độ CROP đơn): P6c = v6 crop raw, P5c = v5 crop raw, PVL = PaddleOCR-VL-1.6 crop, Gf = GLM a-force-s1.

Trọng tài duy nhất: R(âm) = cột R_candidates trong mau/M*.csv (Dict/QuocNgu_SinoNom.csv).
Đối chứng: (i) chữ ngẫu nhiên theo tần suất corpus ∈ R; (ii) HOÁN VỊ: xáo P và G độc lập giữa các ô (giữ phân bố đầu ra
của engine, phá liên hệ với ảnh), 500 lần → số ô bắn / ∈R kỳ vọng khi KHÔNG có tín hiệu thị giác.

Chạy: /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/.venv/bin/python ket_hop.py
Ra:   ket_qua.json, bang_luat.csv, chi_tiet/{M1,M2,M3}_luong.csv, chi_tiet/M2_luat.csv
"""
import sys, json, math, random, collections
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
LAB = REPO / "lab/ocr_compare"
OUT = LAB / "ket_hop"
sys.path.insert(0, str(LAB / "paddle"))
from score import load_vocab, corpus_char_freq, cjk_chars  # noqa: E402

rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
first = lambda s: cjk_chars(s)[:1]
N_PERM = 500
SEED = 20260911


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}–{100*hi:.1f}]" if n else "0/0"


# ---------------------------------------------------------------- nạp luồng
def load_streams(s: str) -> pd.DataFrame:
    m = rd(LAB / f"mau/{s}.csv")
    m["R"] = m.R_candidates.map(lambda x: set(t for t in x.split("|") if t))
    d = m[["sample_id", "book", "page", "column", "tier", "rule", "ocr_char", "syllable", "label", "R", "R_size"]].copy()
    d = d.rename(columns={"ocr_char": "K"})
    p = rd(LAB / f"paddle/ket_qua/{s}_col-PP-OCRv6_medium_rec.csv")
    d = d.merge(p[["sample_id", "pred_mono", "pred", "pred_all"]].rename(columns={"pred_mono": "P", "pred": "P_y", "pred_all": "P_all"}), on="sample_id", how="left")
    g = rd(LAB / f"glm/out/{s}_col-ctx.csv")
    d = d.merge(g[["sample_id", "pred", "align_op"]].rename(columns={"pred": "G", "align_op": "G_op"}), on="sample_id", how="left")
    crop = {"P6c": f"paddle/ket_qua/{s}_PP-OCRv6_medium_rec_raw.csv", "P5c": f"paddle/ket_qua/{s}_PP-OCRv5_server_rec_raw.csv",
            "PVL": f"paddle/ket_qua/{s}_PaddleOCR-VL-1.6.csv", "Gf": f"glm/out/{s}_a-force-s1.csv"}
    for k, rel in crop.items():
        c = rd(LAB / rel)
        d = d.merge(c[["sample_id", "pred"]].rename(columns={"pred": k}), on="sample_id", how="left")
    d = d.fillna("")
    for k in ("P", "P_y", "G", "P6c", "P5c", "PVL", "Gf"):
        d[k] = d[k].map(first)
    d["P_all"] = d.P_all.map(cjk_chars)
    d["K_inR"] = [c in r for c, r in zip(d.K, d.R)]
    return d


inR = lambda c, r: c != "" and c in r
T2S = json.load(open(OUT / "t2s_map.json")) if (OUT / "t2s_map.json").exists() else {}  # phồn→giản (OpenCC), sinh bằng /tmp/venv_paddle
norm = lambda c: T2S.get(c, c)
same_var = lambda a, b: a != "" and b != "" and norm(a) == norm(b)   # cùng chữ sau chuẩn hoá giản/phồn


# ---------------------------------------------------------------- luật
def rules(d: pd.DataFrame, P=None, G=None) -> dict:
    """Trả về {tên luật: (fired: bool[], proposal: str[])} — proposal = chữ luật đề xuất ('' nếu không bắn).
    P, G có thể được truyền vào (hoán vị) để tính đối chứng."""
    K = d.K.values; P = d.P.values if P is None else P; G = d.G.values if G is None else G
    R = d.R.values; Pall = d.P_all.values
    P6, P5, PV, Gf = d.P6c.values, d.P5c.values, d.PVL.values, d.Gf.values
    n = len(d); out = {}

    def mk(f, prop):
        return np.array(f, bool), np.array(prop, object)

    # (a) ĐỒNG THUẬN
    out["A1 P==G (2 engine mới trùng nhau)"] = mk([p != "" and p == g for p, g in zip(P, G)], [p if p != "" and p == g else "" for p, g in zip(P, G)])
    out["A2 P==G ∧ ∈R"] = mk([p != "" and p == g and inR(p, r) for p, g, r in zip(P, G, R)], [p if p != "" and p == g and inR(p, r) else "" for p, g, r in zip(P, G, R)])
    out["A3 ≥2/3 trong {K,P,G} trùng (bất kỳ)"] = mk(
        [(p != "" and p == g) or (p != "" and p == k) or (g != "" and g == k) for k, p, g in zip(K, P, G)],
        [(p if (p != "" and (p == g or p == k)) else (g if g != "" and g == k else "")) for k, p, g in zip(K, P, G)])
    out["A4 K==P hoặc K==G (engine mới XÁC NHẬN K)"] = mk([(p != "" and p == k) or (g != "" and g == k) for k, p, g in zip(K, P, G)],
                                                          [k if (p != "" and p == k) or (g != "" and g == k) else "" for k, p, g in zip(K, P, G)])
    out["A5 K==P==G (3/3)"] = mk([p != "" and p == g == k for k, p, g in zip(K, P, G)], [p if p != "" and p == g == k else "" for k, p, g in zip(K, P, G)])
    # biến thể: so sánh sau chuẩn hoá giản/phồn (Paddle cột thiên phồn, GLM thiên giản → cùng chữ khác dạng)
    out["A1v P≈G (chuẩn hoá giản/phồn)"] = mk([same_var(p, g) for p, g in zip(P, G)], [p if same_var(p, g) else "" for p, g in zip(P, G)])
    out["A2v P≈G ∧ (P∈R ∨ G∈R) → lấy dạng ∈R"] = mk([same_var(p, g) and (inR(p, r) or inR(g, r)) for p, g, r in zip(P, G, R)],
                                                   [(p if inR(p, r) else g) if same_var(p, g) and (inR(p, r) or inR(g, r)) else "" for p, g, r in zip(P, G, R)])
    out["A4v K≈P hoặc K≈G (xác nhận K, chuẩn hoá giản/phồn)"] = mk([same_var(p, k) or same_var(g, k) for k, p, g in zip(K, P, G)],
                                                                [k if same_var(p, k) or same_var(g, k) else "" for k, p, g in zip(K, P, G)])
    # (b) TRƯỢT R → LẤY ENGINE KHÁC NẾU ∈R (chỉ áp cho ô K∉R; trên M1 giả lập 'không biết K')
    def fb(order):
        f, pr = [], []
        for i in range(n):
            c = ""
            for src in order:
                v = {"P": P[i], "G": G[i]}[src]
                if inR(v, R[i]):
                    c = v; break
            f.append(c != ""); pr.append(c)
        return mk(f, pr)
    out["B1 P∈R ? P : (G∈R ? G : ∅)"] = fb(("P", "G"))
    out["B2 G∈R ? G : (P∈R ? P : ∅)"] = fb(("G", "P"))
    out["B3 chỉ P∈R"] = mk([inR(p, r) for p, r in zip(P, R)], [p if inR(p, r) else "" for p, r in zip(P, R)])
    out["B4 chỉ G∈R"] = mk([inR(g, r) for g, r in zip(G, R)], [g if inR(g, r) else "" for g, r in zip(G, R)])
    out["B5 P∈R ∧ G∈R ∧ P==G (2 cầu ∈R)"] = out["A2 P==G ∧ ∈R"]
    out["B6 đúng 1 ứng viên ∈R trong {P,G} (không mâu thuẫn)"] = mk(
        [len({c for c in (p, g) if inR(c, r)}) == 1 for p, g, r in zip(P, G, R)],
        [list({c for c in (p, g) if inR(c, r)})[0] if len({c for c in (p, g) if inR(c, r)}) == 1 else "" for p, g, r in zip(P, G, R)])
    # (c) PHỦ QUYẾT — proposal = K (giữ) hoặc '' ; fired = phủ quyết K
    out["C1 phủ quyết nếu P≠K (P≠∅)"] = mk([p != "" and p != k for k, p in zip(K, P)], [""] * n)
    out["C2 phủ quyết nếu G≠K (G≠∅)"] = mk([g != "" and g != k for k, g in zip(K, G)], [""] * n)
    out["C3 phủ quyết nếu P≠K ∧ G≠K (cả hai ≠∅)"] = mk([p != "" and g != "" and p != k and g != k for k, p, g in zip(K, P, G)], [""] * n)
    out["C4 phủ quyết nếu P≠K ∨ G≠K (kể cả rỗng)"] = mk([p != k or g != k for k, p, g in zip(K, P, G)], [""] * n)
    out["C5 phủ quyết nếu P==G≠K"] = mk([p != "" and p == g and p != k for k, p, g in zip(K, P, G)], [""] * n)
    out["C6 phủ quyết nếu (P≠K ∧ P∈R) ∨ (G≠K ∧ G∈R)"] = mk([(p != k and inR(p, r)) or (g != k and inR(g, r)) for k, p, g, r in zip(K, P, G, R)], [""] * n)
    # (d) HỢP TẬP ĐỌC ∩ R, ĐẾM CẦU
    def union_rule(readers, name):
        f, pr, ncand, nbr = [], [], [], []
        for i in range(n):
            votes = collections.Counter()
            for rd_ in readers:
                for c in set(rd_[i]):
                    if inR(c, R[i]):
                        votes[c] += 1
            ncand.append(len(votes))
            if len(votes) == 1:
                c, b = votes.most_common(1)[0]; f.append(True); pr.append(c); nbr.append(b)
            elif len(votes) >= 2:
                (c1, b1), (c2, b2) = votes.most_common(2)
                if b1 > b2:
                    f.append(True); pr.append(c1); nbr.append(b1)
                else:
                    f.append(False); pr.append(""); nbr.append(0)
            else:
                f.append(False); pr.append(""); nbr.append(0)
        out[name] = mk(f, pr); out[name + " |ncand"] = (np.array(ncand), np.array(nbr))
    union_rule([P, Pall, G], "D1 ({P}∪P_all∪{G})∩R, ứng viên duy nhất/nhiều cầu nhất")
    union_rule([P, Pall, G, P6, P5, PV, Gf], "D2 mở rộng 7 luồng (thêm crop v6/v5/VL/GLM-force)∩R")
    # D3: ≥2 cầu (hai engine KHÁC NHAU cùng đọc ra chữ ∈R) trong 7 luồng — Paddle-family và GLM-family đếm riêng
    f, pr = [], []
    for i in range(n):
        fam = collections.defaultdict(set)
        for c in {P[i], *Pall[i], P6[i], P5[i], PV[i]}:
            if inR(c, R[i]): fam[c].add("paddle")
        for c in {G[i], Gf[i]}:
            if inR(c, R[i]): fam[c].add("glm")
        two = [c for c, s in fam.items() if len(s) >= 2]
        f.append(len(two) == 1); pr.append(two[0] if len(two) == 1 else "")
    out["D3 ∈R được CẢ Paddle lẫn GLM đọc ra (bất kỳ cấu hình)"] = mk(f, pr)
    return out


def summarize(d, rr, name, setname):
    fired, prop = rr
    R = d.R.values; K = d.K.values; n = len(d)
    nf = int(fired.sum())
    is_prop = np.array([p != "" for p in prop])
    in_r = np.array([inR(p, r) for p, r in zip(prop, R)])
    eqK = np.array([p != "" and p == k for p, k in zip(prop, K)])
    o = dict(luat=name, set=setname, n=n, fired=nf, fired_pct=round(100 * nf / n, 1), fired_ci=[round(100 * x, 1) for x in wilson(nf, n)])
    if is_prop.any():
        o["prop_n"] = int(is_prop.sum()); o["prop_inR"] = int(in_r.sum())
        o["prop_inR_pct_of_fired"] = round(100 * in_r.sum() / is_prop.sum(), 1)
        o["prop_inR_pct_of_n"] = round(100 * in_r.sum() / n, 2)
        o["prop_inR_ci_of_n"] = [round(100 * x, 2) for x in wilson(int(in_r.sum()), n)]
        o["prop_eqK"] = int(eqK.sum()); o["prop_eqK_pct_of_fired"] = round(100 * eqK.sum() / is_prop.sum(), 1)
        o["prop_inR_neK"] = int((in_r & ~eqK).sum())
    if setname == "M2":
        sil = d.tier.values == "SILVER"
        o["by_tier"] = {t: dict(n=int((d.tier.values == t).sum()), fired=int(fired[d.tier.values == t].sum()),
                                inR=int(in_r[d.tier.values == t].sum()),
                                inR_ci=[round(100 * x, 1) for x in wilson(int(in_r[d.tier.values == t].sum()), int((d.tier.values == t).sum()))]) for t in ("SILVER", "SYLLABLE")}
        o["by_book"] = {b: dict(n=int((d.book.values == b).sum()), fired=int(fired[d.book.values == b].sum()),
                                inR=int(in_r[d.book.values == b].sum())) for b in sorted(set(d.book))}
        lab = d.label.values
        m = sil & in_r
        o["SILVER_inR_eq_pipeline_label"] = f"{int((np.array([p == l for p, l in zip(prop, lab)]) & m).sum())}/{int(m.sum())}"
    if setname == "M3":
        cnt = collections.Counter(p for p in prop if p != "")
        o["dist_top"] = cnt.most_common(8)
        o["pct_𠊚"] = round(100 * cnt["𠊚"] / n, 1); o["pct_㝵"] = round(100 * cnt["㝵"] / n, 1)
    return o


def main():
    q2n, vocab = load_vocab(); freq = corpus_char_freq(); tot = sum(freq.values())
    rng = np.random.default_rng(SEED)
    res = {"tham_so": dict(n_perm=N_PERM, seed=SEED, P="PP-OCRv6 cột gióng đơn điệu (pred_mono)", G="GLM-OCR col-ctx"), "M1": {}, "M2": {}, "M3": {}}
    rows = []
    D = {}
    for s in ("M1", "M2", "M3"):
        d = load_streams(s); D[s] = d
        d.drop(columns=["R"]).assign(R_size=d.R_size).to_csv(OUT / f"chi_tiet/{s}_luong.csv", index=False)
        rr = rules(d)
        # đối chứng ngẫu nhiên theo tần suất: P(chữ ngẫu nhiên ∈ R(ô))
        p_freq = float(np.mean([sum(freq[c] for c in r) / tot for r in d.R]))
        p_unif = float(np.mean([len(r) / len(vocab) for r in d.R]))
        # hoán vị: xáo P, G (và các luồng crop) độc lập giữa các ô
        perm_stats = collections.defaultdict(list)
        for _ in range(N_PERM):
            P_ = d.P.values[rng.permutation(len(d))]; G_ = d.G.values[rng.permutation(len(d))]
            dd = d.copy()
            dd["P"] = P_; dd["G"] = G_
            for k in ("P_all", "P6c", "P5c", "PVL", "Gf"):
                dd[k] = d[k].values[rng.permutation(len(d))]
            rp = rules(dd)
            for name, v in rp.items():
                if name.endswith("|ncand"): continue
                f, pr = v
                perm_stats[name].append((int(f.sum()), int(sum(inR(p, r) for p, r in zip(pr, d.R.values)))))
        for name, v in rr.items():
            if name.endswith("|ncand"):
                ncand, nbr = v
                res[s][name] = dict(phan_bo_so_ung_vien=dict(collections.Counter(ncand.tolist())), phan_bo_so_cau=dict(collections.Counter(nbr[nbr > 0].tolist())))
                continue
            o = summarize(d, v, name, s)
            pf = np.array(perm_stats[name])
            o["perm_fired_mean"] = round(float(pf[:, 0].mean()), 1); o["perm_fired_p95"] = float(np.percentile(pf[:, 0], 95))
            o["perm_inR_mean"] = round(float(pf[:, 1].mean()), 3); o["perm_inR_p95"] = float(np.percentile(pf[:, 1], 95))
            # lift: nếu hoán vị chưa bao giờ đạt, dùng sàn 1/N_PERM → ghi dạng '≥'
            fl = max(float(pf[:, 0].mean()), 1.0 / N_PERM)
            o["lift_fired_vs_perm"] = (">" if pf[:, 0].mean() == 0 else "") + str(round(o["fired"] / fl, 1))
            if "prop_inR" in o:
                il = max(float(pf[:, 1].mean()), 1.0 / N_PERM)
                o["lift_inR_vs_perm"] = (">" if pf[:, 1].mean() == 0 else "") + str(round(o["prop_inR"] / il, 1))
                o["lift_inR_vs_freq"] = round((o["prop_inR"] / o["n"]) / p_freq, 1)
                o["p_perm_ge_obs"] = float((pf[:, 1] >= o["prop_inR"]).mean())
            o["p_random_freq"] = round(100 * p_freq, 3); o["p_random_uniform"] = round(100 * p_unif, 3)
            res[s][name] = o
            rows.append(o)
    # ------------------------------------------------ phủ quyết như bộ phân loại: M1 = K đúng (∈R), M2 = K sai (∉R)
    veto = {}
    for name in [k for k in res["M2"] if k.startswith("C")]:
        tpr = res["M2"][name]["fired"] / 600; fpr = res["M1"][name]["fired"] / 600
        veto[name] = dict(TPR_M2=round(100 * tpr, 1), FPR_M1=round(100 * fpr, 1), lift=round(tpr / fpr, 2) if fpr else None,
                          M3_fired_pct=res["M3"][name]["fired_pct"],
                          # áp lên corpus: GOLD 53.604 ô (K∈R) bị phủ quyết oan vs khối 27.770 ô bắt được
                          uoc_GOLD_bi_oan=int(round(fpr * 53604)), uoc_khoi_bat_duoc=int(round(tpr * 27770)),
                          precision_uoc=round(100 * tpr * 27770 / (tpr * 27770 + fpr * 53604), 1) if (tpr + fpr) else None)
    res["phu_quyet_phan_loai"] = veto
    # ------------------------------------------------ ngoại suy
    res["ngoai_suy"] = extrapolate(D["M2"], D["M1"], rules(D["M2"]), rules(D["M1"]))
    # ------------------------------------------------ M2 per-cell chi tiết luật
    rr2 = rules(D["M2"]); dd = D["M2"].drop(columns=["R"]).copy()
    for name, v in rr2.items():
        if name.endswith("|ncand"): continue
        f, pr = v; key = name.split(" ")[0]
        dd[key + "_fired"] = f; dd[key + "_prop"] = pr
    dd.to_csv(OUT / "chi_tiet/M2_luat.csv", index=False)
    json.dump(res, open(OUT / "ket_qua.json", "w"), ensure_ascii=False, indent=1, default=str)
    pd.DataFrame(rows).to_csv(OUT / "bang_luat.csv", index=False)
    print(json.dumps(res["ngoai_suy"], ensure_ascii=False, indent=1))
    print(json.dumps(veto, ensure_ascii=False, indent=1))


def extrapolate(d2, d1, rr2, rr1):
    """Suy rộng số ô 'có bằng chứng độc lập ∈R' từ M2 lên khối mục tiêu. Hậu phân tầng theo (tier, sách) về pool
    SILVER+SYLLABLE (R≠∅), bootstrap 2000 lần cho CI. REVIEW (11.442 ô trong khối 27.770) KHÔNG có trong M2 → kịch bản."""
    labels = rd(REPO / "dataset_out/labels.csv")
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import load_qn_to_nom
    from core.text.text_utils import normalize_tone_marks
    q2n = load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv"))
    labels["Rn"] = labels.syllable.map(lambda s: len(q2n.get(normalize_tone_marks(s.strip().lower()), [])))
    pool = labels[labels.tier.isin(["SILVER", "SYLLABLE"]) & (labels.Rn > 0)]
    W = pool.groupby(["tier", "book"]).size()
    rev = labels[(labels.tier == "REVIEW") & (labels.Rn > 0)]
    out = dict(pool_SILVER_SYLLABLE_R_nonempty=int(len(pool)), REVIEW_R_nonempty=int(len(rev)), khoi_muc_tieu=27770,
               ghi_chu=("Khối 27.770 = SILVER không cầu 5.959 + SYLLABLE 10.369 + REVIEW 11.442. M2 lấy từ SILVER+SYLLABLE (R≠∅: %d ô), "
                        "KHÔNG có ô REVIEW (không crop, S3 dưới ngưỡng) → REVIEW ngoại suy theo kịch bản 'như SYLLABLE' (giả định, chưa đo)." % len(pool)))
    rng = np.random.default_rng(SEED)
    strata = list(W.index)
    for name in ("B1 P∈R ? P : (G∈R ? G : ∅)", "B3 chỉ P∈R", "B4 chỉ G∈R", "A2 P==G ∧ ∈R", "A2v P≈G ∧ (P∈R ∨ G∈R) → lấy dạng ∈R", "B6 đúng 1 ứng viên ∈R trong {P,G} (không mâu thuẫn)",
                 "D2 mở rộng 7 luồng (thêm crop v6/v5/VL/GLM-force)∩R",
                 "D1 ({P}∪P_all∪{G})∩R, ứng viên duy nhất/nhiều cầu nhất", "D3 ∈R được CẢ Paddle lẫn GLM đọc ra (bất kỳ cấu hình)"):
        f2, pr2 = rr2[name]; inr2 = np.array([inR(p, r) for p, r in zip(pr2, d2.R.values)])
        # độ chính xác thay thế đo trên M1: trong các ô luật bắn, tỷ lệ đề xuất == K (K đúng theo R trên M1)
        f1, pr1 = rr1[name]; eq1 = np.array([p != "" and p == k for p, k in zip(pr1, d1.K.values)])
        prec = eq1[f1].mean() if f1.any() else float("nan"); prec_ci = wilson(int(eq1[f1].sum()), int(f1.sum()))
        eqv1 = np.array([same_var(p, k) for p, k in zip(pr1, d1.K.values)]); prec_v = eqv1[f1].mean() if f1.any() else float("nan")
        # hậu phân tầng
        def est(idx):
            tot = 0.0
            for (t, b) in strata:
                m = (d2.tier.values[idx] == t) & (d2.book.values[idx] == b)
                if m.sum() == 0: continue
                tot += W[(t, b)] * inr2[idx][m].mean()
            return tot
        all_idx = np.arange(len(d2))
        point = est(all_idx)
        boots = []
        for _ in range(2000):
            # bootstrap phân tầng
            idx = np.concatenate([rng.choice(np.where((d2.tier.values == t) & (d2.book.values == b))[0], size=int(((d2.tier.values == t) & (d2.book.values == b)).sum()), replace=True) for (t, b) in strata])
            boots.append(est(idx))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        syl_rate = inr2[d2.tier.values == "SYLLABLE"].mean(); syl_n = int((d2.tier.values == "SYLLABLE").sum()); syl_k = int(inr2[d2.tier.values == "SYLLABLE"].sum())
        sl, sh = wilson(syl_k, syl_n)
        out[name] = dict(
            M2_inR=f"{int(inr2.sum())}/{len(d2)}",
            ty_le_hau_phan_tang_pct=round(100 * point / len(pool), 2),
            so_o_SILVER_SYLLABLE=dict(diem=int(round(point)), ci95=[int(round(lo)), int(round(hi))]),
            so_o_REVIEW_kich_ban_nhu_SYLLABLE=dict(diem=int(round(syl_rate * len(rev))), ci95=[int(round(sl * len(rev))), int(round(sh * len(rev)))], gia_dinh="tỷ lệ SYLLABLE áp cho REVIEW; chưa đo"),
            tong_khoi_27770=dict(diem=int(round(point + syl_rate * len(rev))), ci95=[int(round(lo + sl * len(rev))), int(round(hi + sh * len(rev)))]),
            do_chinh_xac_thay_the_M1=dict(mo_ta="P(đề xuất == K | luật bắn) trên M1, K∈R; '∈R' chỉ là điều kiện cần",
                                          gia_tri_pct=round(100 * prec, 1) if prec == prec else None, ci95=[round(100 * x, 1) for x in prec_ci], n_fired_M1=int(f1.sum()),
                                          neu_coi_bien_the_gian_phon_la_dung_pct=round(100 * prec_v, 1) if prec_v == prec_v else None),
            so_o_dung_uoc_tinh_SILVER_SYLLABLE=dict(diem=int(round(point * prec)) if prec == prec else None,
                                                    ci95=[int(round(lo * prec_ci[0])), int(round(hi * prec_ci[1]))] if prec == prec else None),
        )
    return out


if __name__ == "__main__":
    main()
