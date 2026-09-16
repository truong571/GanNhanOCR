"""B-3 · CỔNG THỊ GIÁC REVIEW → SYL (DANH_MUC_SUA_DOI_CUOI_2026-09-16 §1B B-3) — CHỈ ĐO, KHÔNG ĐỔI NHÃN.

Đọc bộ nhãn v3 (dataset_out_v3/labels_final.csv) + sidecar out-of-fold B-1' (p_visual_oof_v3.csv,
KhoiB/v3/train_oof_cnn_v3.py) → ô tier_v3 == REVIEW (và tier == REVIEW, chưa dùng được) mà
    argmax == syllable_v3  ∧  p_syl_v3 ≥ 0,9        (cổng chính, rule đề xuất `visual_syl_gate`)
    argmax == syllable_v3  ∧  p_syl_v3 ≥ 0,8        (cổng lỏng, báo riêng)
→ KhoiB/v3/visual_syl_candidates.csv (mọi ô REVIEW qua cổng 0,8; cột gate_09/gate_08) + báo cáo số theo
sách / tier_goc / rule / âm (KhoiB/v3/visual_syl_gate_report.json). Chỉ nhãn ÂM (SYL), mã chữ tự động = 0.

FAR ước trên CHAR_A (nhãn tin cậy nhất — GOLD trực tiếp): ô có âm KỀ (syl_idx ∓ 1 cùng cột, âm khác âm mình,
∈ lớp) mà
    (a) p_kề ≥ 0,9 ∧ p_mình < 0,5              (định nghĩa DANH_MUC / nhiệm vụ K4)
    (b) argmax == âm kề ∧ p_kề ≥ 0,9           (đúng điều kiện cổng áp cho nhãn SAI — FAR tương đương cổng)
tỉ lệ trên mẫu số = ô CHAR_A có ≥1 âm kề hợp lệ. Cả hai ở ngưỡng 0,9 và 0,8. Wilson 95%.

Thiếu p_visual_oof_v3.csv (B-1' chưa chạy Kaggle) → in lý do, ghi report {"available": false}, thoát 0.
Khoá join = (book, page, column, nom_idx) — cả hai tệp cùng thế hệ v3 (đối chiếu syllable_v3 == syllable,
labels_md5 trong probs_oof.npz nếu có).

    .venv/bin/python -m pipeline.tools.visual_syl_gate                       # mặc định đường v3
    .venv/bin/python -m pipeline.tools.visual_syl_gate --oof <csv> --out-dir <dir>
    .venv/bin/python -m pipeline.tools.visual_syl_gate --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.text.text_utils import is_plausible_qn_syllable  # noqa: E402

DEFAULT_LABELS = REPO / "dataset_out_v3/labels_final.csv"
DEFAULT_OOF = (REPO / "KhoiB/v3/p_visual_oof_v3_results/p_visual_oof_v3.csv",
               REPO / "KhoiB/v3/p_visual_oof_v3.csv")
DEFAULT_OUT = REPO / "KhoiB/v3"
KEY = ["book", "page", "column", "nom_idx"]
THR_MAIN, THR_LOOSE = 0.9, 0.8
RULE = "visual_syl_gate"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def find_oof(explicit=None) -> Path | None:
    for p in ([Path(explicit)] if explicit else DEFAULT_OOF):
        if p.exists():
            return p
    return None


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def run(labels_path: Path, oof_path: Path | None, out_dir: Path, thr_main=THR_MAIN, thr_loose=THR_LOOSE,
        quiet=False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {"labels": str(labels_path), "oof": str(oof_path) if oof_path else None,
                 "thr_main": thr_main, "thr_loose": thr_loose, "rule": RULE, "doi_nhan": False}
    if oof_path is None:
        rep.update(available=False, reason=("chưa có p_visual_oof_v3.csv (B-1' chưa chạy Kaggle — KhoiB/v3/README.md §3); "
                                            "đã tìm: " + " | ".join(str(p) for p in DEFAULT_OOF)))
        json.dump(rep, open(out_dir / "visual_syl_gate_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if not quiet:
            print(f"[B-3] KHÔNG SẴN SÀNG: {rep['reason']}\n  -> {out_dir / 'visual_syl_gate_report.json'} (available=false)")
        return rep
    rep["available"] = True
    df = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    oo = pd.read_csv(oof_path, dtype=str, keep_default_na=False)
    need = ["syllable_v3", "p_syl_v3", "argmax", "max_prob", "p_left", "p_right", "syl_in_classes", "ok"]
    miss = [c for c in KEY + need if c not in oo.columns]
    if miss:
        raise SystemExit(f"[B-3] {oof_path} thiếu cột {miss} — cần schema B-1' v3 (train_oof_cnn_v3.py), "
                         "không phải p_visual_oof.csv cũ (KhoiB/) — khoá nom_idx cũ là cumcount, sai 9,5%")
    for c in ("tier_v3", "nom_idx", "syl_idx", "tier", "rule", "syllable"):
        if c not in df.columns:
            raise SystemExit(f"[B-3] {labels_path} thiếu cột {c} — cần bộ v3")
    if "tier_goc" not in df.columns:
        df["tier_goc"] = ""
    # đối chiếu thế hệ
    npz = oof_path.parent / "probs_oof.npz"
    lm = _md5(labels_path)
    rep["labels_md5"] = lm
    if npz.exists():
        try:
            z = np.load(npz, allow_pickle=True)
            nm = str(z["labels_md5"]) if "labels_md5" in z else ""
            rep["oof_labels_md5"] = nm
            rep["labels_md5_match"] = (nm == lm) if nm else None
        except Exception as e:  # pragma: no cover
            rep["oof_labels_md5"] = f"lỗi đọc {npz.name}: {e}"
    key_df = df[KEY].agg("|".join, axis=1)
    key_oo = oo[KEY].agg("|".join, axis=1)
    if not key_oo.is_unique:
        raise SystemExit("[B-3] khoá (book,page,column,nom_idx) không duy nhất trong sidecar OOF")
    oo = oo.set_index(key_oo)
    m = df.assign(_k=key_df).join(oo[need + ["fold"] if "fold" in oo.columns else need], on="_k", how="left")
    joined = m["syllable_v3"].notna()
    rep["n_labels"] = int(len(df)); rep["n_oof"] = int(len(oo)); rep["n_joined"] = int(joined.sum())
    mism = int((m.loc[joined, "syllable_v3"] != m.loc[joined, "syllable"]).sum())
    rep["n_syllable_mismatch"] = mism
    if mism:
        print(f"[B-3 CẢNH BÁO] syllable_v3 (OOF) ≠ syllable (labels) ở {mism} ô — khác thế hệ? kiểm md5 trước khi tin số")
    m["p"] = m["p_syl_v3"].map(_f); m["pmax"] = m["max_prob"].map(_f)
    m["pl"] = m["p_left"].map(_f); m["pr"] = m["p_right"].map(_f)
    m["syl_in"] = m["syl_in_classes"].fillna("0").astype(str) == "1"

    # --- cổng REVIEW -> SYL (chỉ nhãn âm) ---
    rv = (m["tier_v3"] == "REVIEW")
    usable_already = rv & m["tier"].isin(["GOLD", "SYLLABLE"])          # 36 ô khoá QĐ-01 ở v3: đã dùng được, không qua cổng
    plaus = m["syllable"].map(lambda s: bool(is_plausible_qn_syllable(str(s).lower())))
    elig = rv & ~usable_already & (m["rule"] != "not_plausible") & plaus & joined & m["syl_in"]
    hit = elig & (m["argmax"] == m["syllable"])
    g09 = hit & (m["p"] >= thr_main)
    g08 = hit & (m["p"] >= thr_loose)
    rep["review"] = {
        "n_tier_v3_review": int(rv.sum()), "n_already_usable_qd01": int(usable_already.sum()),
        "n_not_plausible_or_rule": int((rv & ~usable_already & ((m["rule"] == "not_plausible") | ~plaus)).sum()),
        "n_review_not_joined": int((rv & ~usable_already & ~joined).sum()),
        "n_review_syl_outside_classes": int((rv & ~usable_already & joined & ~m["syl_in"]).sum()),
        "n_eligible": int(elig.sum()), "n_argmax_eq_syl": int(hit.sum()),
        f"n_gate_{thr_main}": int(g09.sum()), f"n_gate_{thr_loose}": int(g08.sum()),
        "p_quantiles_eligible": {q: (round(float(m.loc[elig, "p"].quantile(q)), 4) if elig.any() else None)
                                 for q in (0.5, 0.9, 0.95, 0.99)},
    }
    by = {}
    for col in ("book", "tier_goc", "rule"):
        by[col] = {}
        for v, g in m[elig].groupby(col):
            gi = g.index
            by[col][v or "(rỗng)"] = {"eligible": int(len(gi)), f"gate_{thr_main}": int(g09[gi].sum()),
                                      f"gate_{thr_loose}": int(g08[gi].sum())}
    by["syllable_top20_gate_main"] = dict(Counter(m.loc[g09, "syllable"]).most_common(20))
    rep["review_by"] = by

    # --- FAR ước trên CHAR_A ---
    ca = (m["tier_v3"] == "CHAR_A") & joined & m["syl_in"]
    pos = {(b, p, c, s): i for i, (b, p, c, s) in enumerate(zip(m.book, m.page, m.column, m.syl_idx.map(_f)))}
    syl_arr = m["syllable"].to_numpy(object)
    far = {}
    n_den = 0
    cnt = Counter()
    for i in np.where(ca.to_numpy())[0]:
        b, p, c, s = m.book.iat[i], m.page.iat[i], m.column.iat[i], _f(m.syl_idx.iat[i])
        own = syl_arr[i]; pm = m.p.iat[i]; am = m.argmax.iat[i]
        sides = []
        for d, pv in ((-1, m.pl.iat[i]), (1, m.pr.iat[i])):
            j = pos.get((b, p, c, s + d))
            if j is None or math.isnan(pv) or syl_arr[j] == own:
                continue
            sides.append((syl_arr[j], pv))
        if not sides:
            continue
        n_den += 1
        for thr, tag in ((thr_main, "09"), (thr_loose, "08")):
            if any(pv >= thr and pm < 0.5 for _, pv in sides):
                cnt[f"a_{tag}"] += 1
            if any(pv >= thr and am == sj for sj, pv in sides):
                cnt[f"b_{tag}"] += 1
    for k in ("a_09", "a_08", "b_09", "b_08"):
        lo, hi = wilson(cnt[k], n_den)
        far[k] = {"n": int(cnt[k]), "den": n_den, "rate": (cnt[k] / n_den if n_den else None),
                  "wilson95": [round(lo, 5), round(hi, 5)]}
    rep["far_char_a"] = {"dinh_nghia": {"a": "p_kề ≥ thr ∧ p_mình < 0,5", "b": "argmax == âm kề ∧ p_kề ≥ thr (điều kiện cổng áp cho nhãn sai)",
                                        "mau_so": "ô CHAR_A có ≥1 âm kề (syl_idx∓1 cùng cột) khác âm mình và ∈ lớp"},
                         "n_char_a_joined": int(ca.sum()), **far}
    # đối chứng: ô CHAR_A qua cổng với ĐÚNG âm mình (độ nhạy của cổng trên nhãn đúng)
    ca_hit = ca & (m["argmax"] == m["syllable"])
    rep["char_a_gate_recall"] = {f"gate_{thr_main}": int((ca_hit & (m.p >= thr_main)).sum()),
                                 f"gate_{thr_loose}": int((ca_hit & (m.p >= thr_loose)).sum()), "n": int(ca.sum())}

    # --- ghi ---
    cols = ["image", "book", "page", "column", "nom_idx", "syl_idx", "ocr_char", "syllable", "tier", "tier_v3", "rule",
            "tier_goc", "rule_goc", "bbox", "p_syl_v3", "argmax", "max_prob", "p_left", "p_right"]
    cols = [c for c in cols if c in m.columns]
    if "fold" in m.columns:
        cols.append("fold")
    out = m.loc[g08, cols].copy()
    out["gate_09"] = g09[g08].astype(int).to_numpy()
    out["gate_08"] = 1
    out["rule_de_xuat"] = RULE
    out = out.sort_values(["book", "page", "column"], key=lambda s_: s_ if s_.name != "column" else s_.map(_f))
    cand = out_dir / "visual_syl_candidates.csv"
    out.to_csv(cand, index=False)
    rep["candidates_csv"] = str(cand); rep["n_candidates_rows"] = int(len(out))
    json.dump(rep, open(out_dir / "visual_syl_gate_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if not quiet:
        r = rep["review"]; f = rep["far_char_a"]
        print(f"[B-3] join {rep['n_joined']:,}/{rep['n_labels']:,} ô (OOF {rep['n_oof']:,}; syllable lệch {mism}; "
              f"labels_md5 khớp: {rep.get('labels_md5_match')})")
        print(f"  REVIEW tier_v3 {r['n_tier_v3_review']:,} | đã dùng được (QĐ-01) {r['n_already_usable_qd01']} | "
              f"not_plausible {r['n_not_plausible_or_rule']} | không join {r['n_review_not_joined']} | âm ∉ lớp "
              f"{r['n_review_syl_outside_classes']} | đủ điều kiện {r['n_eligible']:,} | argmax == âm {r['n_argmax_eq_syl']:,}")
        print(f"  QUA CỔNG p ≥ {thr_main}: {r[f'n_gate_{thr_main}']:,} | p ≥ {thr_loose}: {r[f'n_gate_{thr_loose}']:,}  "
              f"(DANH_MUC kỳ vọng ≈894–961) — KHÔNG đổi nhãn; chờ Khối C 100–150 ô")
        for col in ("book", "tier_goc"):
            print(f"  theo {col}: " + " · ".join(f"{k}: {v['eligible']}/{v[f'gate_{thr_main}']}/{v[f'gate_{thr_loose}']}"
                                                for k, v in by[col].items()) + "  (đủ đk / ≥0,9 / ≥0,8)")
        print(f"  FAR CHAR_A (mẫu số {f['a_09']['den']:,}): (a) p_kề≥0,9∧p_mình<0,5 {f['a_09']['n']} = "
              f"{(f['a_09']['rate'] or 0):.4%} {f['a_09']['wilson95']} | (b) argmax==kề∧p_kề≥0,9 {f['b_09']['n']} = "
              f"{(f['b_09']['rate'] or 0):.4%} {f['b_09']['wilson95']} | ở 0,8: (a) {f['a_08']['n']} (b) {f['b_08']['n']}")
        print(f"  độ nhạy cổng trên CHAR_A (argmax == âm ∧ p ≥ 0,9): {rep['char_a_gate_recall'][f'gate_{thr_main}']:,}"
              f"/{rep['char_a_gate_recall']['n']:,}")
        print(f"  -> {cand} ({len(out):,} dòng) · {out_dir / 'visual_syl_gate_report.json'}")
    return rep


# --------------------------------------------------------------------------- selftest
def _selftest() -> int:
    import tempfile
    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name}")

    print("[visual_syl_gate selftest]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1 cột 6 ô: 0 CHAR_A 'an', 1 REVIEW 'ba' (qua 0,9), 2 REVIEW 'cha' (0,85: chỉ 0,8), 3 REVIEW 'mẹ' argmax khác,
        # 4 REVIEW not_plausible, 5 CHAR_A 'trời' (ảnh nói âm kề phải 'ba' (ô 6) 0,95, p mình 0,1 -> FAR a & b),
        # 6 REVIEW->GOLD khoá QĐ-01 'ba' (đã dùng được, không qua cổng)
        rows = [("an", "GOLD", "CHAR_A", "s1_inter_s2_direct"), ("ba", "REVIEW", "REVIEW", "no_context"),
                ("cha", "REVIEW", "REVIEW", "no_context"), ("mẹ", "REVIEW", "REVIEW", "no_context"),
                ("xx1", "REVIEW", "REVIEW", "not_plausible"), ("trời", "GOLD", "CHAR_A", "s1_inter_s2_direct"),
                ("ba", "GOLD", "REVIEW", "quyet_dinh_nguoi:qd01_cell_lock")]
        lab = pd.DataFrame([{"image": f"g{i}.png", "book": "stt2", "page": "page_0001", "column": "1", "nom_idx": str(i),
                             "syl_idx": str(i), "ocr_char": "人", "syllable": s, "tier": t, "tier_v3": tv, "rule": r,
                             "tier_goc": "", "rule_goc": "", "bbox": "[0, 0, 1, 1]"} for i, (s, t, tv, r) in enumerate(rows)])
        lab.to_csv(td / "labels_final.csv", index=False)
        oof = pd.DataFrame([
            {"book": "stt2", "page": "page_0001", "column": "1", "nom_idx": "0", "syl_idx": "0", "fold": "1", "syllable_v3": "an",
             "syl_in_classes": "1", "ok": "1", "p_syl_v3": "0.97", "argmax": "an", "max_prob": "0.97", "p_left": "", "p_right": "0.01"},
            {"book": "stt2", "page": "page_0001", "column": "1", "nom_idx": "1", "syl_idx": "1", "fold": "1", "syllable_v3": "ba",
             "syl_in_classes": "1", "ok": "1", "p_syl_v3": "0.95", "argmax": "ba", "max_prob": "0.95", "p_left": "0.01", "p_right": "0.01"},
            {"book": "stt2", "page": "page_0001", "column": "1", "nom_idx": "2", "syl_idx": "2", "fold": "1", "syllable_v3": "cha",
             "syl_in_classes": "1", "ok": "1", "p_syl_v3": "0.85", "argmax": "cha", "max_prob": "0.85", "p_left": "0.01", "p_right": "0.01"},
            {"book": "stt2", "page": "page_0001", "column": "1", "nom_idx": "3", "syl_idx": "3", "fold": "1", "syllable_v3": "mẹ",
             "syl_in_classes": "1", "ok": "1", "p_syl_v3": "0.05", "argmax": "cha", "max_prob": "0.93", "p_left": "0.93", "p_right": "0.01"},
            {"book": "stt2", "page": "page_0001", "column": "1", "nom_idx": "4", "syl_idx": "4", "fold": "1", "syllable_v3": "xx1",
             "syl_in_classes": "0", "ok": "1", "p_syl_v3": "", "argmax": "ba", "max_prob": "0.99", "p_left": "0.01", "p_right": "0.01"},
            {"book": "stt2", "page": "page_0001", "column": "1", "nom_idx": "5", "syl_idx": "5", "fold": "1", "syllable_v3": "trời",
             "syl_in_classes": "1", "ok": "1", "p_syl_v3": "0.1", "argmax": "ba", "max_prob": "0.95", "p_left": "", "p_right": "0.95"},
            {"book": "stt2", "page": "page_0001", "column": "1", "nom_idx": "6", "syl_idx": "6", "fold": "1", "syllable_v3": "ba",
             "syl_in_classes": "1", "ok": "1", "p_syl_v3": "0.99", "argmax": "ba", "max_prob": "0.99", "p_left": "0.01", "p_right": ""},
        ])
        oof.to_csv(td / "p_visual_oof_v3.csv", index=False)
        rep = run(td / "labels_final.csv", td / "p_visual_oof_v3.csv", td / "out", quiet=True)
        r = rep["review"]
        check("join 7/7", rep["n_joined"] == 7 and rep["n_syllable_mismatch"] == 0)
        check("REVIEW tier_v3 5, đã dùng được QĐ-01 1, not_plausible 1, đủ đk 3", r["n_tier_v3_review"] == 5
              and r["n_already_usable_qd01"] == 1 and r["n_not_plausible_or_rule"] == 1 and r["n_eligible"] == 3)
        check("argmax == âm 2; qua cổng 0,9 = 1; 0,8 = 2", r["n_argmax_eq_syl"] == 2 and r["n_gate_0.9"] == 1 and r["n_gate_0.8"] == 2)
        cand = pd.read_csv(td / "out/visual_syl_candidates.csv", dtype=str, keep_default_na=False)
        check("candidates.csv 2 dòng (ba gate_09=1, cha gate_09=0), rule_de_xuat", len(cand) == 2
              and dict(zip(cand.syllable, cand.gate_09)) == {"ba": "1", "cha": "0"} and (cand.rule_de_xuat == RULE).all())
        f = rep["far_char_a"]
        check("FAR mẫu số 2 (ô 0 kề 'ba' p_right 0,01; ô 5 kề 'mẹ' 0,95)", f["a_09"]["den"] == 2)
        check("FAR (a) 0,9: 1/2; (b) 0,9: 1/2 (ô 5: argmax 'ba' == kề phải, p_kề 0,95)", f["a_09"]["n"] == 1 and f["b_09"]["n"] == 1)
        check("Wilson 1/2 ≈ [0,095; 0,905]", abs(f["a_09"]["wilson95"][0] - 0.0945) < 0.01 and abs(f["a_09"]["wilson95"][1] - 0.9055) < 0.01)
        check("độ nhạy cổng CHAR_A 1/2 (ô 0 p 0,97)", rep["char_a_gate_recall"]["gate_0.9"] == 1 and rep["char_a_gate_recall"]["n"] == 2)
        check("theo sách: stt2 eligible 3", rep["review_by"]["book"]["stt2"]["eligible"] == 3)
        check("không đổi nhãn (doi_nhan False, labels không bị ghi)", rep["doi_nhan"] is False
              and pd.read_csv(td / "labels_final.csv", dtype=str, keep_default_na=False).equals(lab.astype(str)))
        # thiếu OOF -> available False, không crash
        rep0 = run(td / "labels_final.csv", None, td / "out0", quiet=True)
        check("thiếu OOF: available False + report ghi", rep0["available"] is False and (td / "out0/visual_syl_gate_report.json").exists())
        # schema cũ (KhoiB/p_visual_oof.csv) -> SystemExit rõ
        pd.DataFrame({"book": ["stt2"], "page": ["p"], "column": ["1"], "nom_idx": ["0"], "fold": ["1"], "p_visual": ["0.5"],
                      "argmax_syl": ["an"], "max_prob": ["0.5"]}).to_csv(td / "old.csv", index=False)
        try:
            run(td / "labels_final.csv", td / "old.csv", td / "out1", quiet=True); check("schema cũ -> SystemExit", False)
        except SystemExit as e:
            check("schema cũ -> SystemExit nêu cumcount", "cumcount" in str(e))
        check("wilson(0,0) nan; wilson(10,10) hi=1", math.isnan(wilson(0, 0)[0]) and wilson(10, 10)[1] == 1.0)
    print(f"RESULT: {passed} passed, {failed} failed")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="B-3: cổng thị giác REVIEW→SYL (chỉ đo)")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--oof", default=None, help="p_visual_oof_v3.csv (mặc định: KhoiB/v3/p_visual_oof_v3_results/)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--thr", type=float, default=THR_MAIN)
    ap.add_argument("--thr-loose", type=float, default=THR_LOOSE)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    run(Path(a.labels), find_oof(a.oof), Path(a.out_dir), a.thr, a.thr_loose)


if __name__ == "__main__":
    main()
