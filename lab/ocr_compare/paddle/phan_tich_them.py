"""Phân tích bổ sung cho cấu hình tốt nhất (cột v6, gióng theo y) và crop v6 raw:
Wilson CI, đối chứng hoán vị (xáo pred giữa các ô), so với nhãn pipeline trên SILVER, phân bố theo tier/sách."""
import sys, json, math, random, collections
import pandas as pd
sys.path.insert(0, "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle")
from score import load_vocab, corpus_char_freq, score_set, cjk_chars
K = "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle/ket_qua"
q2n, vocab = load_vocab(); freq = corpus_char_freq()


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


out = {}
for cfg in ("col-PP-OCRv6_medium_rec", "PP-OCRv6_medium_rec_raw", "PP-OCRv5_server_rec_raw"):
    o = {}
    for s in ("M1", "M2", "M3"):
        res = pd.read_csv(f"{K}/{s}_{cfg}.csv", dtype=str, keep_default_na=False)
        sc, d = score_set(s, res, vocab, freq)
        k = int(d.first_in_R.sum()); n = len(d)
        e = dict(in_R=f"{k}/{n} = {100*k/n:.1f}% CI95 {wilson(k, n)}")
        # đối chứng hoán vị: xáo chữ đọc ra giữa các ô (giữ phân bố đầu ra của engine), 300 lần
        rng = random.Random(0); firsts = list(d["first"]); Rs = list(d.R); hits = []
        for _ in range(300):
            rng.shuffle(firsts); hits.append(sum(f != "" and f in r for f, r in zip(firsts, Rs)) / n)
        e["perm_control_in_R"] = f"{100*sum(hits)/len(hits):.2f}%"
        e["lift_vs_perm"] = round((k / n) / max(1e-9, sum(hits) / len(hits)), 1)
        e["lift_vs_freq_random"] = round((k / n) / sc["p_random_in_R_freqweighted"], 1)
        if s == "M2":
            sil = d[d.tier == "SILVER"]; syl = d[d.tier == "SYLLABLE"]
            e["SILVER_in_R"] = f"{int(sil.first_in_R.sum())}/{len(sil)} = {100*sil.first_in_R.mean():.1f}%"
            e["SYLLABLE_in_R"] = f"{int(syl.first_in_R.sum())}/{len(syl)} = {100*syl.first_in_R.mean():.1f}%"
            rs = sil[sil.first_in_R]
            e["SILVER_rescued_eq_pipeline_label"] = f"{int((rs['first'] == rs.label).sum())}/{len(rs)}"
            e["by_book_in_R"] = {b: f"{int(g.first_in_R.sum())}/{len(g)}" for b, g in d.groupby("book")}
            e["by_rule_in_R"] = {b: f"{int(g.first_in_R.sum())}/{len(g)}" for b, g in d.groupby("rule")}
            e["rescued_examples"] = [(r.sample_id, r.ocr_char, r.syllable, r["first"], r.label) for _, r in d[d.first_in_R].head(12).iterrows()]
        if s == "M1":
            e["eq_kinhhannom"] = f"{int(d.eq_kinh.sum())}/{n} = {100*d.eq_kinh.mean():.1f}%"
            e["in_R_but_ne_kinh"] = int((d.first_in_R & ~d.eq_kinh).sum())
        if s == "M3":
            cnt = collections.Counter(d["first"]); e["top10"] = cnt.most_common(10)
            e["pct_得_variants"] = f"{cnt['得']+cnt['㝵']}/{n}"
        o[s] = e
    out[cfg] = o
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle/phan_tich_them.json", "w"), ensure_ascii=False, indent=1)
