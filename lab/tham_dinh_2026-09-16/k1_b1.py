"""K1 · THẨM ĐỊNH B-1 (KhoiB/train_oof_cnn.py + p_visual_oof.csv) — chỉ đọc, không sửa KhoiB/.

    .venv/bin/python lab/tham_dinh_2026-09-16/k1_b1.py            # ~1 phút

Đầu vào: KhoiB/KhoiB_kaggle_dataset.zip (labels_final.csv CŨ = git 3e062b1fa3, md5 17359c6e…),
KhoiB/p_visual_oof_results/*, lab/gan_nhan_2026-09-13/{cols.pkl,crops.npz}, dataset_out_v3/labels.csv.
Đầu ra: dataset_out_v3/khoi_b/k1_b1_tham_dinh.json (mọi số trong báo cáo K1)
        dataset_out_v3/khoi_b/p_visual_oof_rekey.csv (p_visual_oof + nom_idx_true để join đúng khoá v3)
"""
from __future__ import annotations
import hashlib, io, json, pickle, random, zipfile
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
KB = REPO / "KhoiB"; OUT = REPO / "dataset_out_v3/khoi_b"; OUT.mkdir(parents=True, exist_ok=True)
R = {}

# ---------------------------------------------------------------- nạp
with zipfile.ZipFile(KB / "KhoiB_kaggle_dataset.zip") as z:
    raw = z.read("labels_final.csv")
R["labels_cu_md5"] = hashlib.md5(raw).hexdigest()
df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
oof = pd.read_csv(KB / "p_visual_oof_results/p_visual_oof.csv", dtype=str, keep_default_na=False)
oof["p_visual"] = oof.p_visual.astype(float); oof["max_prob"] = oof.max_prob.astype(float)
ok = np.load(REPO / "lab/gan_nhan_2026-09-13/crops.npz")["ok"]
classes = json.load(open(KB / "p_visual_oof_results/classes.json")); cid = set(classes)
summ = json.load(open(KB / "p_visual_oof_results/summary_oof.json"))
assert len(df) == len(oof) == len(ok)
R["n"] = len(df); R["ok_true"] = int(ok.sum())
R["oof_cung_thu_tu_dong"] = int(((df.book == oof.book) & (df.page == oof.page) & (df.column == oof.column)).sum())
df["cum"] = df.groupby(["book", "page", "column"]).cumcount()
R["oof_nom_idx_la_cumcount"] = int((df.cum.values == oof.nom_idx.astype(int).values).sum())

# ---------------------------------------------------------------- 1. fold / lớp / rò
def page_fold(b, p): return int(hashlib.md5(f"{b}_{p}".encode()).hexdigest(), 16) % 5
df["fold"] = [page_fold(b, p) for b, p in zip(df.book, df.page)]
R["fold_khop_cong_thuc"] = int((df.fold.values == oof.fold.astype(int).values).sum())
pg = df.drop_duplicates(["book", "page"])
R["so_trang"] = len(pg); R["trang_thuoc_nhieu_fold"] = int((df.groupby(["book", "page"]).fold.nunique() > 1).sum())
R["trang_moi_fold"] = pg.fold.value_counts().sort_index().to_dict()
R["split_x_fold_trang"] = {s: pg[pg.split == s].fold.value_counts().sort_index().to_dict() for s in ["train", "val", "test"]}
m = df[df.image_md5 != ""]
R["md5_o_nhieu_fold"] = int((m.groupby("image_md5").fold.nunique() > 1).sum())
tr_mask = (df.split == "train") & df.tier.isin(["GOLD", "SYLLABLE"]) & ok
cl2 = sorted(s for s, n in Counter(df.loc[tr_mask, "syllable"]).items() if n >= 5)
R["lop_tai_lap"] = len(cl2); R["lop_khop_classes_json"] = cl2 == classes
inc = df.syllable.isin(cid)
R["o_am_ngoai_706"] = int((~inc).sum()); R["o_am_ngoai_706_theo_tier"] = df[~inc].tier.value_counts().to_dict()
R["p_visual_bang_0_du_am_trong_706"] = int((inc & (oof.p_visual == 0)).sum())
R["fold_train_val"] = [dict(fold=f, train=int(((df.fold != f) & df.tier.isin(["GOLD", "SYLLABLE"]) & ok & inc).sum()),
                            val=int(((df.fold == f) & df.tier.isin(["GOLD", "SYLLABLE"]) & ok & inc).sum())) for f in range(5)]
R["lab_train_split_train"] = int((tr_mask & inc).sum())

# ---------------------------------------------------------------- 2. nom_idx THẬT từ cols.pkl
cols = pickle.load(open(REPO / "lab/gan_nhan_2026-09-13/cols.pkl", "rb"))
nom_true = np.full(len(df), -1); syl_true = np.full(len(df), -1)
bad_cols = set(); n_join = 0
for c in cols:
    if c["rows"] is None: continue
    mp = [o for o in c["ops"] if o["op"] == "match"]; assert len(mp) == len(c["rows"])
    if any(df.ocr_char.iat[r] != o["ocr_char"] for r, o in zip(c["rows"], mp)):
        bad_cols.add((c["book"], c["page"], c["column"])); continue   # cột join nhầm (đếm bằng nhau tình cờ)
    n_join += 1
    for r, o in zip(c["rows"], mp):
        nom_true[r] = o["nom_idx"]; syl_true[r] = o["syl_idx"]
R["cols_pkl"] = dict(cot=len(cols), rows_none=sum(c["rows"] is None for c in cols), cot_ocr_char_lech=sorted(map(str, bad_cols)), cot_dung=n_join)
df["nom_true"] = nom_true; df["syl_true"] = syl_true
j = df[df.nom_true >= 0]; diff = j.cum != j.nom_true
R["o_co_nom_idx_that"] = len(j)
R["dong_cumcount_lech"] = dict(n=int(diff.sum()), pct=round(float(diff.mean()) * 100, 2),
                              theo_tier=j[diff].tier.value_counts().to_dict(), theo_split=j[diff].split.value_counts().to_dict(),
                              do_lech=(j.nom_true - j.cum)[diff].value_counts().sort_index().to_dict())
ck = j.book + "|" + j.page + "|" + j.column
cd = diff.groupby(ck).any(); R["cot_co_lech"] = dict(n=int(cd.sum()), tong=len(cd), pct=round(float(cd.mean()) * 100, 2))
rekey = oof.copy(); rekey.insert(4, "nom_idx_true", nom_true); rekey.insert(5, "syl_idx_true", syl_true)
rekey["old_row"] = np.arange(len(df)); rekey["old_tier"] = df.tier.values; rekey["old_rule"] = df.rule.values
rekey["old_syllable"] = df.syllable.values; rekey["old_bbox"] = df.bbox.values
rekey.to_csv(OUT / "p_visual_oof_rekey.csv", index=False)

# ---------------------------------------------------------------- 3. join sang v3
v3 = pd.read_csv(REPO / "dataset_out_v3/labels.csv", dtype=str, keep_default_na=False)
v3["k"] = v3.book + "|" + v3.page + "|" + v3.column + "|" + v3.nom_idx
R["v3"] = dict(n=len(v3), khoa_duy_nhat=bool(v3.k.is_unique), tier_v3=v3.tier_v3.value_counts().to_dict())
o = rekey.copy(); o["old_char"] = df.ocr_char.values
o["kw"] = o.book + "|" + o.page + "|" + o.column + "|" + o.nom_idx
o["kt"] = o.book + "|" + o.page + "|" + o.column + "|" + o.nom_idx_true.astype(str)
sw = v3.merge(o[["kw", "old_char", "old_syllable", "p_visual"]].rename(columns={"kw": "k"}), on="k", how="left")
hw = sw.p_visual.notna()
R["join_khoa_sai_cumcount"] = dict(o_v3_co_p=int(hw.sum()), ocr_char_khac=int((hw & (sw.ocr_char != sw.old_char)).sum()),
                                   syllable_khac=int((hw & (sw.syllable != sw.old_syllable)).sum()))
st = v3.merge(o[o.nom_idx_true >= 0][["kt", "old_char", "old_syllable", "old_bbox", "old_tier", "p_visual", "argmax_syl", "max_prob"]]
              .rename(columns={"kt": "k"}), on="k", how="left")
hit = st.p_visual.notna(); usable = st.tier_v3.isin(["CHAR_A", "CHAR_B", "SYL"])
def bb(s):
    try: return tuple(int(v) for v in json.loads(s))
    except Exception: return None
def iou(a, b):
    if a is None or b is None: return np.nan
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1); ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else np.nan
nb = st.bbox.map(bb); ob = st.old_bbox.map(lambda s: bb(s) if isinstance(s, str) else None)
chg = hit & (nb != ob); ious = np.array([iou(a, b) for a, b in zip(nb[chg], ob[chg])])
sd = hit & (st.syllable != st.old_syllable)
R["join_khoa_dung"] = dict(
    o_v3_co_p=int(hit.sum()), pct=round(float(hit.mean()) * 100, 2), ocr_char_khac=int((hit & (st.ocr_char != st.old_char)).sum()),
    theo_tier_v3={t: [int((hit & (st.tier_v3 == t)).sum()), int((st.tier_v3 == t).sum())] for t in sorted(st.tier_v3.unique())},
    usable=int(usable.sum()), usable_co_p=int((hit & usable).sum()),
    usable_khong_p=dict(n=int((usable & ~hit).sum()), box_source=st[usable & ~hit].box_source.value_counts().to_dict()),
    bbox_doi=dict(n=int(chg.sum()), pct_co_p=round(float(chg.sum() / hit.sum()) * 100, 2), usable=int((chg & usable).sum()),
                  theo_tier_v3=st[chg].tier_v3.value_counts().to_dict(), theo_box_source=st[chg].box_source.value_counts().to_dict(),
                  iou_p10_p50_p90=[round(float(np.nanpercentile(ious, q)), 3) for q in (10, 50, 90)],
                  iou_duoi_0_5=int((ious < 0.5).sum()), iou_duoi_0_8=int((ious < 0.8).sum())),
    syllable_v3_khac_cu=dict(n=int(sd.sum()), theo_tier_v3=st[sd].tier_v3.value_counts().to_dict(),
                             argmax_bang_am_v3=int((sd & (st.argmax_syl == st.syllable)).sum())),
    am_v3_ngoai_706_co_p=int((hit & ~st.syllable.isin(cid)).sum()))
rv = hit & (st.tier_v3 == "REVIEW"); g9 = rv & (st.argmax_syl == st.syllable) & (st.max_prob >= 0.9)
R["b3_so_bo_review_v3"] = dict(review_co_p=int(rv.sum()), qua_cong_0_9=int(g9.sum()), trong_do_bbox_doi=int((g9 & chg).sum()),
                               qua_cong_0_8=int((rv & (st.argmax_syl == st.syllable) & (st.max_prob >= 0.8)).sum()))

# ---------------------------------------------------------------- 4. chất lượng OOF
df["hit"] = oof.argmax_syl.values == df.syllable.values; df["p"] = oof.p_visual.values; df["mp"] = oof.max_prob.values
R["top1_theo_fold"] = [dict(fold=f, n=int(((df.fold == f) & df.tier.isin(["GOLD", "SYLLABLE"]) & inc).sum()),
                            top1=round(float(df.hit[(df.fold == f) & df.tier.isin(["GOLD", "SYLLABLE"]) & inc].mean()), 4),
                            summary_json=round(summ["folds"][f]["val_top1"], 4)) for f in range(5)]
buckets = [("GOLD-truc-tiep", (df.tier == "GOLD") & (df.rule == "s1_inter_s2_direct")),
           ("GOLD-cau-tu-dang", (df.tier == "GOLD") & df.rule.str.startswith("s1_inter_s2_similar")),
           ("GOLD-moi-rule", df.tier == "GOLD"), ("SYLLABLE", df.tier == "SYLLABLE"),
           ("SILVER_uncalibrated", df.tier == "SILVER_uncalibrated"), ("REVIEW", df.tier == "REVIEW")]
E1 = {"GOLD-truc-tiep": (4899, .778), "GOLD-cau-tu-dang": (512, .676), "SYLLABLE": (1131, .819), "REVIEW": (915, .456)}
R["top1_theo_tier_cu"] = {}
for name, mk in buckets:
    a = mk & inc; t = a & (df.split == "test"); p = df.p[a]
    R["top1_theo_tier_cu"][name] = dict(oof_n=int(a.sum()), oof_top1=round(float(df.hit[a].mean()), 3), test_n=int(t.sum()),
                                       test_top1=round(float(df.hit[t].mean()), 3), e1_lab=E1.get(name),
                                       p50=round(float(p.median()), 3), p10=round(float(p.quantile(.1)), 3),
                                       pct_p_duoi_0_05=round(float((p < 0.05).mean()) * 100, 1),
                                       pct_argmax_dung_va_P_tren_0_9=round(float(((df.mp[a] >= 0.9) & df.hit[a]).mean()) * 100, 1))
R["gold_theo_rule"] = df[(df.tier == "GOLD") & inc].groupby("rule").hit.agg(["size", "mean"]).round(3).reset_index().to_dict("records")

# ---------------------------------------------------------------- 5. E2b tương đương (chỉ CẬN vì thiếu vector)
rng = random.Random(5)
test_pages = set(map(tuple, df[df.split == "test"][["book", "page"]].drop_duplicates().itertuples(index=False)))
def e2b(only_test):
    n = 0; alo = []; ahi = []; det = Counter(); known = 0; tot = 0
    for c in cols:
        if c["rows"] is None or (c["book"], c["page"], c["column"]) in bad_cols: continue
        if not (len(c["chars"]) == len(c["syl"]) >= 10 and len(c["rows"]) == len(c["syl"]) and all(o["op"] == "match" for o in c["ops"])): continue
        if only_test and (c["book"], c["page"]) not in test_pages: continue
        rows = c["rows"]; N = len(rows); syl = [df.syllable.iat[r] for r in rows]
        if not all(s in cid for s in syl): continue
        n += 1; k2 = rng.randrange(3, N - 3)
        pr = np.array([df.p.iat[rows[i]] for i in range(k2)]); lo = []; hi = []
        for i in range(k2, N - 1):
            r = rows[i]; tot += 1; det["wrong"] += 1
            if oof.argmax_syl.iat[r] == syl[i + 1]:
                v = df.mp.iat[r]; lo.append(v); hi.append(v); known += 1; det["wrong_flag_lo"] += v < 0.05
            else:
                ub = max(0.0, min(df.mp.iat[r], 1.0 - df.p.iat[r] - df.mp.iat[r])); lo.append(ub); hi.append(0.0); det["wrong_flag_lo"] += ub < 0.05
        det["right"] += len(pr); det["right_flag"] += int((pr < 0.05).sum())
        alo.append(np.mean([1.0 if w < r else 0.5 if w == r else 0.0 for w in lo for r in pr]))
        ahi.append(np.mean([1.0 if w < r else 0.5 if w == r else 0.0 for w in hi for r in pr]))
    return dict(cot=n, auc_can_duoi=round(float(np.mean(alo)), 3), auc_can_tren=round(float(np.mean(ahi)), 3),
                o_sai_biet_P=known, o_sai=tot, oan_pct=round(det["right_flag"] / det["right"] * 100, 1), n_dung=det["right"],
                bat_can_duoi_pct=round(det["wrong_flag_lo"] / det["wrong"] * 100, 1))
R["e2b"] = dict(trang_test=e2b(True), moi_trang=e2b(False), lab_e2b=dict(auc=0.946, bat=93, oan=14.3, cot_thuc=89))
json.dump(R, open(OUT / "k1_b1_tham_dinh.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
