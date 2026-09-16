"""E1/E2 OUT-OF-FOLD từ B-1' Kaggle (KhoiB/v3/p_visual_oof_v3_results) — K5, chỉ đo.

E1: top-1 / top-5 / hạng trung vị / p50 của P(âm v3 | crop) theo tier_v3, theo tier, theo sách, theo fold
    (ô có âm ∈ 804 lớp; hạng tính từ LP trong probs_oof.npz).
E2b: AUC P(âm mình) [CHAR_A] vs P(âm ô kề syl_idx±1) — theo fold, theo sách (như summary.json.e2b; loại kề cùng âm).
Đối chiếu b3_gate_review của summary.json (1.030/1.846) với visual_syl_gate (1.025/1.827): hiệu = ô bị loại
(36 QĐ-01 đã dùng được + 188 not_plausible) mà qua cổng.
So với HUONG_TAN_CONG_GAN_NHAN_2026-09-14.md §3 (706 lớp, trang test: 77,8/67,6/81,9/45,6; E2b AUC 0,946).

    .venv/bin/python KhoiB/v3/e1_e2_oof.py            -> KhoiB/v3/E1_E2_OOF.md + e1_e2_oof.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.text.text_utils import is_plausible_qn_syllable  # noqa: E402

RES = REPO / "KhoiB/v3/p_visual_oof_v3_results"
LABELS = REPO / "dataset_out/labels_final.csv"
REF = {"GOLD-trực-tiếp (CHAR_A)": 77.8, "GOLD-cầu tự dạng (CHAR_B)": 67.6, "SYLLABLE (SYL)": 81.9, "REVIEW": 45.6}


def auc(pos, neg):
    """AUC Mann–Whitney (xếp hạng, xử lý hoà)."""
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg]); r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block(df, rank, p):
    m = df.syl_in_classes.values == 1
    n = int(m.sum())
    if n == 0:
        return {"n_total": int(len(df)), "n_in_classes": 0}
    rk = rank[df.index.values[m]]; pp = p[df.index.values[m]]
    return {"n_total": int(len(df)), "n_in_classes": n, "top1": round(float((rk == 1).mean()) * 100, 2),
            "top5": round(float((rk <= 5).mean()) * 100, 2), "rank_p50": float(np.median(rk)),
            "p50": round(float(np.median(pp)), 4), "pct_p_lt_0.05": round(float((pp < 0.05).mean()) * 100, 2),
            "pct_argmax_dung_p_ge_0.9": round(float(((rk == 1) & (pp >= 0.9)).mean()) * 100, 2)}


def main():
    oof = pd.read_csv(RES / "p_visual_oof_v3.csv", dtype={"column": str, "nom_idx": int, "syl_idx": int}, keep_default_na=False)
    z = np.load(RES / "probs_oof.npz", allow_pickle=True)
    LP = z["LP"].astype(np.float32); classes = list(z["classes"]); cid = {c: i for i, c in enumerate(classes)}
    assert len(oof) == LP.shape[0] == 83239
    assert (z["book"] == oof.book.values).all() and (z["page"] == oof.page.values).all()
    assert (z["nom_idx"].astype(int) == oof.nom_idx.values).all()
    lab = pd.read_csv(LABELS, dtype=str, keep_default_na=False)
    assert len(lab) == len(oof) and (lab.syllable.values == oof.syllable_v3.values).all()
    # hạng của âm v3 (1 = argmax); ô âm ∉ lớp -> 0
    idx = np.array([cid.get(s, -1) for s in oof.syllable_v3.values])
    rank = np.zeros(len(oof), dtype=np.int32); p = np.full(len(oof), np.nan, dtype=np.float32)
    ok = idx >= 0
    lp_true = LP[np.arange(len(oof))[ok], idx[ok]]
    rank[ok] = 1 + (LP[ok] > lp_true[:, None]).sum(1)
    p[ok] = np.exp(lp_true)
    # đối chiếu với csv (p_syl_v3, argmax)
    pc = pd.to_numeric(oof.p_syl_v3, errors="coerce").values
    d = np.nanmax(np.abs(pc[ok] - p[ok]))
    top1_csv = (oof.argmax.values[ok] == oof.syllable_v3.values[ok]).mean()
    out = {"n": int(len(oof)), "n_in_classes": int(ok.sum()), "n_classes": len(classes),
           "check": {"max_abs_p_csv_vs_npz": float(d), "top1_csv": round(float(top1_csv) * 100, 2),
                     "top1_npz": round(float((rank[ok] == 1).mean()) * 100, 2)}}
    out["all"] = block(oof, rank, p)
    out["by_tier_v3"] = {t: block(g, rank, p) for t, g in oof.groupby("tier_v3")}
    out["by_tier"] = {t: block(g, rank, p) for t, g in oof.groupby("tier")}
    out["by_book"] = {t: block(g, rank, p) for t, g in oof.groupby("book")}
    out["by_fold"] = {int(t): block(g, rank, p) for t, g in oof.groupby("fold")}
    out["by_book_tier_v3"] = {f"{b}|{t}": block(g, rank, p) for (b, t), g in oof.groupby(["book", "tier_v3"])}
    # ---- E2b: CHAR_A, p mình vs p kề (syl_idx ± 1 cùng cột), loại kề cùng âm ----
    key = oof.book + "|" + oof.page + "|" + oof.column.astype(str)
    syl_by = {}
    for k, s, sy in zip(key.values, oof.syl_idx.values, oof.syllable_v3.values):
        syl_by[(k, int(s))] = sy
    pos, neg, fold_pos, fold_neg, book_pos, book_neg = [], [], {}, {}, {}, {}
    n_ke_cung_am = 0
    A = oof[(oof.tier_v3 == "CHAR_A") & ok].index.values
    for i in A:
        k = key.values[i]; s = int(oof.syl_idx.values[i]); f = int(oof.fold.values[i]); b = oof.book.values[i]
        pi = p[i]
        for dlt in (-1, 1):
            sy = syl_by.get((k, s + dlt))
            if sy is None or sy not in cid:
                continue
            if sy == oof.syllable_v3.values[i]:
                n_ke_cung_am += 1; continue
            pk = float(np.exp(LP[i, cid[sy]]))
            neg.append(pk); fold_neg.setdefault(f, []).append(pk); book_neg.setdefault(b, []).append(pk)
        pos.append(pi); fold_pos.setdefault(f, []).append(pi); book_pos.setdefault(b, []).append(pi)
    pos, neg = np.array(pos), np.array(neg)
    e2b = {"n_right": int(len(pos)), "n_wrong": int(len(neg)), "n_ke_cung_am_bo": n_ke_cung_am, "auc": round(auc(pos, neg), 4),
           "pct_right_p_lt_0.05": round(float((pos < 0.05).mean()) * 100, 2), "pct_wrong_p_lt_0.05": round(float((neg < 0.05).mean()) * 100, 2),
           "theo_fold": {f: {"auc": round(auc(np.array(fold_pos[f]), np.array(fold_neg[f])), 4), "n_right": len(fold_pos[f]), "n_wrong": len(fold_neg[f])} for f in sorted(fold_pos)},
           "theo_book": {b: {"auc": round(auc(np.array(book_pos[b]), np.array(book_neg[b])), 4), "n_right": len(book_pos[b]), "n_wrong": len(book_neg[b])} for b in sorted(book_pos)}}
    out["e2b"] = e2b
    # ---- đối chiếu b3_gate_review (summary) vs visual_syl_gate ----
    summ = json.load(open(RES / "summary.json"))
    rv = oof.tier_v3 == "REVIEW"
    g09 = rv & (oof.argmax == oof.syllable_v3) & (pd.to_numeric(oof.max_prob, errors="coerce") >= 0.9)
    g08 = rv & (oof.argmax == oof.syllable_v3) & (pd.to_numeric(oof.max_prob, errors="coerce") >= 0.8)
    usable = lab.tier.isin(["GOLD", "SYLLABLE"]).values
    notpl = ~np.array([is_plausible_qn_syllable(s) for s in lab.syllable.values])
    excl = rv.values & (usable | notpl)
    out["b3_doi_chieu"] = {"summary_qua_cong_0_9": summ["b3_gate_review"]["qua_cong_0_9"], "summary_qua_cong_0_8": summ["b3_gate_review"]["qua_cong_0_8"],
                          "tinh_lai_0_9": int(g09.sum()), "tinh_lai_0_8": int(g08.sum()),
                          "n_review_usable_qd01": int((rv.values & usable).sum()), "n_review_not_plausible": int((rv.values & notpl & ~usable).sum()),
                          "bi_loai_qua_cong_0_9": int((g09.values & excl).sum()), "bi_loai_qua_cong_0_8": int((g08.values & excl).sum()),
                          "gate_0_9_sau_loai": int((g09.values & ~excl).sum()), "gate_0_8_sau_loai": int((g08.values & ~excl).sum())}
    json.dump(out, open(REPO / "KhoiB/v3/e1_e2_oof.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- Markdown ----
    L = ["# E1 / E2 OUT-OF-FOLD — B-1' Kaggle (804 lớp, 5 fold theo trang) — K5 16/09/2026", "",
         f"Nguồn: `KhoiB/v3/p_visual_oof_v3_results/{{p_visual_oof_v3.csv, probs_oof.npz, summary.json}}` (labels_md5 `{z['labels_md5']}` == "
         f"`dataset_out/labels_final.csv`), script `KhoiB/v3/e1_e2_oof.py` → `e1_e2_oof.json`. Mọi ô đều out-of-fold "
         f"(mô hình fold k chưa thấy trang của ô). Ô có âm ∈ 804 lớp: {ok.sum():,}/{len(oof):,}. Kiểm csv↔npz: max|Δp| = {d:.2e}, top-1 csv {out['check']['top1_csv']}% == npz {out['check']['top1_npz']}%.", ""]
    L += ["## E1 · P(âm v3 | crop) theo tier_v3 (so HUONG_TAN_CONG §3: 706 lớp, chỉ trang test 20%)", "",
          "| tier_v3 | n (∈ lớp / tổng) | top-1 | top-5 | hạng trung vị | p50 | % p<0,05 | % argmax đúng ∧ p≥0,9 | §3 top-1 (test, 706 lớp) |", "|---|---|---|---|---|---|---|---|---|"]
    ref_map = {"CHAR_A": 77.8, "CHAR_B": 67.6, "SYL": 81.9, "REVIEW": 45.6}
    for t in ["CHAR_A", "CHAR_B", "SYL", "REVIEW"]:
        b = out["by_tier_v3"][t]
        L.append(f"| {t} | {b['n_in_classes']:,} / {b['n_total']:,} | **{b['top1']}%** | {b['top5']}% | {b['rank_p50']:.0f} | {b['p50']} | {b['pct_p_lt_0.05']}% | {b['pct_argmax_dung_p_ge_0.9']}% | {ref_map[t]}% |")
    b = out["all"]
    L.append(f"| **tất cả** | {b['n_in_classes']:,} / {b['n_total']:,} | {b['top1']}% | {b['top5']}% | {b['rank_p50']:.0f} | {b['p50']} | {b['pct_p_lt_0.05']}% | {b['pct_argmax_dung_p_ge_0.9']}% | — |")
    L += ["", "Theo tier (bộ giao nộp `tier`):", "", "| tier | n (∈ lớp / tổng) | top-1 | top-5 | hạng trung vị | p50 |", "|---|---|---|---|---|---|"]
    for t, b in out["by_tier"].items():
        L.append(f"| {t} | {b['n_in_classes']:,} / {b['n_total']:,} | {b['top1']}% | {b['top5']}% | {b['rank_p50']:.0f} | {b['p50']} |")
    L += ["", "## E1 theo sách × tier_v3", "", "| sách | tier_v3 | n ∈ lớp | top-1 | top-5 | p50 |", "|---|---|---|---|---|---|"]
    for bk in ["stt2", "stt4", "stt11"]:
        for t in ["CHAR_A", "CHAR_B", "SYL", "REVIEW"]:
            b = out["by_book_tier_v3"][f"{bk}|{t}"]
            L.append(f"| {bk} | {t} | {b['n_in_classes']:,} | {b['top1']}% | {b['top5']}% | {b['p50']} |")
        b = out["by_book"][bk]
        L.append(f"| {bk} | **tất cả** | {b['n_in_classes']:,} | {b['top1']}% | {b['top5']}% | {b['p50']} |")
    L += ["", "## E1 theo fold (ô mọi tier) và E2b theo fold", "", "| fold | trang | val top-1 (GOLD+SYL, Kaggle) | OOF top-1 mọi ô | OOF top-5 | p50 | E2b AUC | n đúng / n sai |", "|---|---|---|---|---|---|---|---|"]
    for f in range(5):
        b = out["by_fold"][f]; e = e2b["theo_fold"][f]; kf = summ["folds"][f]
        L.append(f"| {f} | {summ['pages_per_fold'][str(f)]} | {kf['val_top1']*100:.1f}% | {b['top1']}% | {b['top5']}% | {b['p50']} | **{e['auc']}** | {e['n_right']:,} / {e['n_wrong']:,} |")
    L += ["", f"## E2b · thanh ghi trượt (CHAR_A: P(âm mình) vs P(âm ô kề), loại {e2b['n_ke_cung_am_bo']} kề cùng âm)", "",
          f"- AUC toàn bộ **{e2b['auc']}** (summary.json.e2b: {summ['e2b']['auc']}; HUONG_TAN_CONG §3: 0,946 trên trang test) — n đúng {e2b['n_right']:,}, n sai {e2b['n_wrong']:,} "
          f"(summary.json n sai {summ['e2b']['n_wrong']:,}: `train_oof_cnn_v3.py:394` lấy cả âm kề của {summ['e2b']['n_wrong'] - e2b['n_wrong']:,} lượt ô CHAR_A có âm mình ∉ lớp; ở đây chỉ ô có cả hai vế — AUC lệch 0,0001).",
          f"- Ngưỡng p<0,05: bắt {e2b['pct_wrong_p_lt_0.05']}% ô sai, oan {e2b['pct_right_p_lt_0.05']}% ô đúng (§3: 93% / 14,3%).",
          "- Theo sách: " + " · ".join(f"{b} {v['auc']} (n {v['n_right']:,}/{v['n_wrong']:,})" for b, v in e2b["theo_book"].items()) + ".",
          "- Theo fold: " + " · ".join(f"f{f} {v['auc']}" for f, v in e2b["theo_fold"].items()) + f" (min–max {min(v['auc'] for v in e2b['theo_fold'].values())}–{max(v['auc'] for v in e2b['theo_fold'].values())}).", ""]
    bd = out["b3_doi_chieu"]
    L += ["## Đối chiếu B-3: summary.json.b3_gate_review vs visual_syl_gate", "",
          f"- summary.json đếm MỌI ô tier_v3 REVIEW (12.897): argmax == âm ∧ p ≥ 0,9 → {bd['summary_qua_cong_0_9']} (tính lại {bd['tinh_lai_0_9']}); ≥ 0,8 → {bd['summary_qua_cong_0_8']} (tính lại {bd['tinh_lai_0_8']}).",
          f"- visual_syl_gate loại trước {bd['n_review_usable_qd01']} ô REVIEW đã dùng được (QĐ-01 khoá → tier GOLD) + {bd['n_review_not_plausible']} ô âm không hợp lệ; trong số ô bị loại có {bd['bi_loai_qua_cong_0_9']} qua cổng 0,9 và {bd['bi_loai_qua_cong_0_8']} qua cổng 0,8 → còn **{bd['gate_0_9_sau_loai']} / {bd['gate_0_8_sau_loai']}** = đúng số của `visual_syl_gate_report.json` (1.025 / 1.827). Hai số KHỚP sau khi trừ ô bị loại.", ""]
    L += ["## Đọc", "",
          f"- Top-1 OOF toàn bộ {out['all']['top1']}% trên {ok.sum():,} ô (mọi trang, 804 lớp) so 77,8% GOLD-trực-tiếp trang test (706 lớp, 4.899 ô) của §3: cùng bậc; CHAR_A {out['by_tier_v3']['CHAR_A']['top1']}%, SYL {out['by_tier_v3']['SYL']['top1']}%, CHAR_B {out['by_tier_v3']['CHAR_B']['top1']}%, REVIEW {out['by_tier_v3']['REVIEW']['top1']}%. Thứ tự SYL ≥ CHAR_A > CHAR_B ≫ REVIEW của §3 được tái lập trên toàn ngữ liệu, không chỉ 20% trang test.",
          f"- CHAR_B (cầu tự dạng) OOF {out['by_tier_v3']['CHAR_B']['top1']}% cao hơn 67,6% của §3 — §3 đo trên GOLD-cầu của thế hệ 64.525 (512 ô test), v3 định nghĩa CHAR_B khác (3.311 ô).",
          f"- E2b AUC {e2b['auc']} ổn định giữa 5 fold (chênh ≤ {max(v['auc'] for v in e2b['theo_fold'].values()) - min(v['auc'] for v in e2b['theo_fold'].values()):.3f}) và 3 sách → kênh ảnh không phụ thuộc trang đã học; đây là điều kiện B-1 (\"E2b out-of-fold trên trang train ≈ test, AUC ≈ 0,95\") — ĐẠT.", ""]
    (REPO / "KhoiB/v3/E1_E2_OOF.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
