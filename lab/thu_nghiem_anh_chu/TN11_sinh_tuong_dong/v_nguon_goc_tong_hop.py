"""TN11 v_nguon_goc_tong_hop (03/10) — gộp kết quả nhiễu lấy mẫu (v_nguon_goc_nhieu_mau.py) thành các số dùng trả lời. 0 mô hình, chỉ ĐỌC JSON.

Với mỗi bộ và mỗi tập hạt giống:
  - Top-1 HQC1/2/3 theo hạt giống: trung bình, độ lệch chuẩn (ddof=1), min–max, n sau lọc (có sẵn ở *_tom_tat.json; in lại gọn)
  - "chọn điều kiện tốt nhất" (best-of-3 như cột 'sau' của Bảng IV): mean(max(HQC1,HQC2,HQC3)) − mean(HQC1) trên các hạt giống
  - thứ hạng HQC1/2/3 theo hạt giống: số hạt giống mỗi HQC đứng đầu (kể cả hoà), số hạt giống có đúng thứ tự như bảng p05
  - SD qua hạt giống so với khoảng chênh max−min giữa 3 HQC mà p05 dùng, và tỉ lệ hạt giống có khoảng chênh ≥ khoảng chênh của p05
  - gộp ô duy nhất: Top-1, Wilson, chênh cặp + CI bootstrap + McNemar chính xác (có sẵn ở tóm tắt; in lại)
Ra: measure_out/_tn11/verify/nguon_goc/tong_hop_nhieu_mau_<tag>.json
    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_nguon_goc_tong_hop.py s0_9 [tag ...]
    ... v_nguon_goc_tong_hop.py --merge s0_9 s10_29 s0_29   (gộp các lượt khác hạt giống)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "measure_out/_tn11/verify/nguon_goc"


def order_str(v):
    return ">".join(f"HQC{i + 1}" for i in np.argsort(-np.array(v), kind="stable"))


def one(tag):
    runs_all = json.loads((OUT_DIR / f"nhieu_mau_{tag}.json").read_text(encoding="utf-8"))
    summ = json.loads((OUT_DIR / f"nhieu_mau_{tag}_tom_tat.json").read_text(encoding="utf-8"))
    out = {}
    for b, runs in runs_all.items():
        t = np.array([[r["top1"]["1"], r["top1"]["2"], r["top1"]["3"]] for r in runs], float)   # seeds x 3
        t2n = np.array([r["top1"]["2n"] for r in runs], float)
        ns = np.array([r["n_after_filter"] for r in runs])
        p05 = summ[b].get("p05_bang_goc")
        mx = t.max(1)
        wins = {f"HQC{i + 1}": int(np.sum(t[:, i] == mx)) for i in range(3)}
        strict_wins = {f"HQC{i + 1}": int(np.sum((t[:, i] == mx) & ((t == mx[:, None]).sum(1) == 1))) for i in range(3)}
        spread = mx - t.min(1)
        row = {"n_hat_giong": len(runs), "n_sau_loc": {"tb": round(float(ns.mean()), 1), "min": int(ns.min()), "max": int(ns.max())},
               "Top1_tb": [round(float(x), 1) for x in t.mean(0)], "Top1_sd": [round(float(x), 1) for x in t.std(0, ddof=1)], "Top1_min": [float(x) for x in t.min(0)], "Top1_max": [float(x) for x in t.max(0)],
               "HQC2n_tb_sd": [round(float(t2n.mean()), 1), round(float(t2n.std(ddof=1)), 1)],
               "best_of_3_tb": round(float(mx.mean()), 1), "best_of_3_tru_HQC1_tb": round(float((mx - t[:, 0]).mean()), 1),
               "so_hat_giong_HQC_dung_dau_ke_ca_hoa": wins, "so_hat_giong_HQC_dung_dau_duy_nhat": strict_wins,
               "khoang_chenh_max_min_giua_3_HQC_tb": round(float(spread.mean()), 1), "khoang_chenh_max_min_min_max": [float(spread.min()), float(spread.max())],
               "sd_trung_binh_qua_3_HQC": round(float(t.std(0, ddof=1).mean()), 1)}
        if p05:
            pv = [p05["HQC1"], p05["HQC2"], p05["HQC3"]]
            ps = max(pv) - min(pv)
            row["p05"] = {"n": p05["n_cells"], "HQC": pv, "thu_tu": order_str(pv), "khoang_chenh": round(ps, 1)}
            row["so_hat_giong_dung_thu_tu_p05"] = int(sum(order_str(list(x)) == order_str(pv) for x in t))
            row["so_hat_giong_khoang_chenh_ge_p05"] = int(np.sum(spread >= ps))
            row["thu_tu_trung_binh_qua_hat_giong"] = order_str(list(t.mean(0)))
        row["chenh_cap_qua_hat_giong"] = {k: summ[b][k] for k in summ[b] if k.startswith("diff_HQC")}
        row["gop_o_duy_nhat"] = summ[b]["gop_o_duy_nhat"]
        row["chan_doan"] = summ[b]["chan_doan"]
        row["sd_nhi_thuc_ky_vong"] = summ[b]["sd_nhi_thuc_ky_vong"]
        row["f_vW_hang1_tb"] = summ[b]["ref_vW_top1_mean"]; row["f_font_san_xuat_hang1_tb"] = summ[b]["ref_fontF_top1_mean"]
        out[b] = row
        print(f"[{tag}] {b:4s} n_sau_loc {row['n_sau_loc']['tb']} [{row['n_sau_loc']['min']}–{row['n_sau_loc']['max']}] | Top-1 TB HQC1/2/3 = {row['Top1_tb']} SD {row['Top1_sd']} min {row['Top1_min']} max {row['Top1_max']} "
              f"| 2n {row['HQC2n_tb_sd']} | best-of-3 {row['best_of_3_tb']} (so HQC1 +{row['best_of_3_tru_HQC1_tb']}) | đứng đầu(kể cả hoà) {wins} duy nhất {strict_wins}")
        if p05:
            print(f"        p05: n={p05['n_cells']} {pv} thứ tự {order_str(pv)} khoảng {ps:.1f} | hạt giống đúng thứ tự p05: {row['so_hat_giong_dung_thu_tu_p05']}/{len(runs)}; khoảng chênh ≥ p05: {row['so_hat_giong_khoang_chenh_ge_p05']}/{len(runs)}; "
                  f"khoảng chênh qua hạt giống {row['khoang_chenh_max_min_giua_3_HQC_tb']} [{row['khoang_chenh_max_min_min_max'][0]}–{row['khoang_chenh_max_min_min_max'][1]}]; SD TB {row['sd_trung_binh_qua_3_HQC']}; "
                  f"thứ tự trung bình {row['thu_tu_trung_binh_qua_hat_giong']}")
        g = row["gop_o_duy_nhat"]
        print(f"        gộp {g['n_o_duy_nhat']} ô duy nhất: " + " | ".join(f"{h} {g[h]['top1']} {g[h]['wilson95']}" for h in ("HQC1", "HQC2", "HQC3", "HQC2n")) + f" | f_vW@1 {g['f_vW_hang1_trong_top3']} f_font(sx)@1 {g['f_font_san_xuat_trong_top3']}")
        for k in ("HQC2_tru_HQC1", "HQC3_tru_HQC1", "HQC3_tru_HQC2", "HQC2n_tru_HQC2"):
            if k in g:
                v = g[k]
                print(f"          {k}: {v['mean_diff']:+.1f} CI95 {v['ci95_bootstrap_o']} (a đúng/b sai {v['o_a_dung_b_sai']} | a sai/b đúng {v['o_a_sai_b_dung']}) McNemar p={v['mcnemar_exact_p']}")
        c = row["chan_doan"]
        print(f"        chẩn đoán: ô đa-sự-thật {c['o_da_su_that_trong_top3']}/{c['tong_o_sau_loc']} hoà {c['o_hoa_diem_top1']} chữ đúng ngoài phông {c['o_chu_dung_khong_co_trong_phong']} "
              f"crop cắt lại ≠ sản xuất {c['o_crop_cat_lai_khac_crop_san_xuat']}/{c['tong_o_sau_loc']} cos(E,crop p05) {c['cos_E_vs_crop_cat_kieu_p05_tb']} cos(E,crop sản xuất) {c['cos_E_vs_crop_cat_kieu_san_xuat_tb']}")
    # gộp 4 bộ: mọi ô duy nhất của mọi bộ, so cặp (bootstrap theo ô; McNemar chính xác)
    from math import sqrt
    HS = ("1", "2", "3", "2n")
    rows = []
    for b, runs in runs_all.items():
        seen = set()
        for r in runs:
            for rec in r["records"]:
                if rec["cell_id"] not in seen:
                    seen.add(rec["cell_id"]); rows.append({**rec, "book": b})
    n = len(rows)
    acc = {h: np.array([r[f"acc{h}"] for r in rows]) for h in HS}
    def wil(k, n, z=1.959963984540054):
        p = k / n; den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den; hw = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return round((ctr - hw) * 100, 1), round((ctr + hw) * 100, 1)
    from scipy.stats import binomtest
    pooled = {"n_o_duy_nhat_4_bo": n, "top1": {f"HQC{h}": {"pct": round(float(acc[h].mean() * 100), 1), "wilson95": wil(int(acc[h].sum()), n)} for h in HS}}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(10000, n))
    for a, b in (("2", "1"), ("3", "1"), ("3", "2"), ("2n", "2")):
        d = acc[a] - acc[b]; bs = d[idx].mean(1) * 100
        bb = int(((acc[a] == 1) & (acc[b] == 0)).sum()); cc = int(((acc[a] == 0) & (acc[b] == 1)).sum())
        pooled[f"HQC{a}_tru_HQC{b}"] = {"mean_diff": round(float(d.mean() * 100), 1), "ci95_bootstrap": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                                        "a_dung_b_sai": bb, "a_sai_b_dung": cc, "mcnemar_p": round(float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0, 4)}
    pooled["f_vW_hang1_trong_top3_pct"] = round(float(np.mean([r["acc_vW_top1"] for r in rows]) * 100), 1)
    pooled["f_font_san_xuat_trong_top3_pct"] = round(float(np.mean([r["acc_fontF_top1"] for r in rows]) * 100), 1)
    out["_gop_4_bo"] = pooled
    print(f"[{tag}] GỘP 4 BỘ {n} ô duy nhất: " + " | ".join(f"{k} {v['pct']} {v['wilson95']}" for k, v in pooled["top1"].items()) + f" | f_vW@1 {pooled['f_vW_hang1_trong_top3_pct']} f_font(sx)@1 {pooled['f_font_san_xuat_trong_top3_pct']}")
    for k, v in pooled.items():
        if "_tru_" in k:
            print(f"      {k}: {v['mean_diff']:+.1f} CI95 {v['ci95_bootstrap']} (a đúng/b sai {v['a_dung_b_sai']} | a sai/b đúng {v['a_sai_b_dung']}) McNemar p={v['mcnemar_p']}")
    (OUT_DIR / f"tong_hop_nhieu_mau_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def merge(tags, outtag):
    """Gộp các lượt chạy (khác hạt giống) thành một tập, tính lại tóm tắt bằng đúng hàm summarize của v_nguon_goc_nhieu_mau.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import v_nguon_goc_nhieu_mau as V   # chỉ để dùng summarize(); không chạy main
    p05 = {r["book"]: r for r in json.loads((REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
    runs_all = {}
    for t in tags:
        d = json.loads((OUT_DIR / f"nhieu_mau_{t}.json").read_text(encoding="utf-8"))
        for b, runs in d.items():
            runs_all.setdefault(b, []).extend(runs)
    for b in runs_all:
        runs_all[b] = sorted(runs_all[b], key=lambda r: r["seed"])
        seeds = [r["seed"] for r in runs_all[b]]
        assert len(seeds) == len(set(seeds)), f"{b}: trùng hạt giống giữa các lượt"
    summaries = {b: V.summarize(b, runs, p05.get(b)) for b, runs in runs_all.items()}
    (OUT_DIR / f"nhieu_mau_{outtag}.json").write_text(json.dumps(runs_all, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (OUT_DIR / f"nhieu_mau_{outtag}_tom_tat.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Đã gộp {tags} -> {outtag}: " + ", ".join(f"{b}: {len(r)} hạt giống" for b, r in runs_all.items()))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--merge":
        merge(a[1:-1], a[-1])
        one(a[-1])
    else:
        for tag in a or ["s0_9"]:
            one(tag)
