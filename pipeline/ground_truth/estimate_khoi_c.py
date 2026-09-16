"""Khối C (C-3) — từ verdict NGƯỜI của mẻ mù hai câu ra con số công bố được.

Đầu vào
-------
  <batch>/verdicts_<session_id>.jsonl   do HTML `phien_k.html` xuất (schema §"VERDICT")
  <batch>/_khoa/KHOA.jsonl              khoá ô -> nguồn (tầng, stratum, N_h, w, đáp án mồi, repeat_of…)
  <batch>/_khoa/manifest.jsonl          item_id -> stratum/weight + labels_sha256 + batch_sha256
  dataset_out/labels_final.csv          chỉ để đối chiếu sha256 (không đọc nội dung)

VERDICT (một dòng / ô, xuất từ `make_khoi_c_batch._HTML`, giao diện HAI PHA):
  {item_id, session_id, order, q1 ∈ {dung,sai_crop,sai_am,khong_ro,null}, q1_blind (Q1 lúc MÃ lộ), q1_am,
   q2 ∈ {dung,sai,khong_ro,null}, q2_code, has_code, t_first_shown, revealed_at, t_answer, dwell_q1_ms, dwell_ms,
   n_q1_change_after_reveal, visits, sitting_id, hist[], exported_at, source:"human"}
  Ước lượng dùng `q1_blind` (câu trả lời khi mọi ô còn trông y hệt nhau); `q1` cuối chỉ để báo số đổi.
  Verdict thế hệ cũ (không có q1_blind) -> dùng q1, và ghi rõ trong báo cáo.

Các phép đo (mỗi mục là một hàm thuần, kiểm bằng verdict GIẢ LẬP trong selftest)
-------------------------------------------------------------------------------
 (a) toàn vẹn   : sha256(KHOA.jsonl) == manifest.batch_sha256; sha256(labels_final) == manifest.labels_sha256;
                  mọi item_id trong verdict đều có trong khoá; has_code khớp has_q2; nguồn = người.
 (b) ô mồi      : mồi dương (QĐ-01) đạt ⇔ Q1=dung (Q2 = dung là theo PHÁN QUYẾT — mã 𠊚 nhận ra được nên chỉ
                  đếm "nhất quán với QĐ-01", không tính đạt/rớt); mồi âm (âm/mã ô kề) đạt ⇔ Q1∈{sai_am,sai_crop} ∧ Q2=sai.
                  BÁO ĐỘNG nếu < 90 %.
 (c) lặp ẩn     : κ Cohen Q1 (4 mức) và Q2 (3 mức) kèm KTC 95 % bootstrap (seed cố định), đồng thuận thô,
                  κ nhị phân (đúng/không), tách theo tầng gốc (lap_tang).
     dwell      : p10/p50/p90, ô < 1,5 s gắn cờ (ngưỡng DWELL_FLAG_MS), số ô đổi ý, số Q1 đổi SAU khi lộ mã.
 (d) precision  : theo tầng (Wilson 95 %) và theo tier_v3 (Horvitz–Thompson với stratum_N, FPC) cho
                  Q1-đúng, Q2-đúng, Q1∧Q2-đúng; GOLD = CHAR_A ∪ CHAR_B; USABLE = GOLD ∪ SYL (dân số ngoài QĐ-01);
                  QD01 = tầng mồi dương (SRS phân tầng trên 2.012 ô QĐ-01); USABLE_RE_DATASET = USABLE ∪ QD01
                  = đúng 70.326 ô re-dataset. `khong_ro` loại khỏi mẫu số, báo số.
 (e) T1 (B-3)   : tỉ lệ Q1 đúng trên ô qua cổng; cận trên Clopper–Pearson MỘT PHÍA 95 % của tỉ lệ lỗi
                  (0/300 -> 1,0 %; ≤ 3/300 -> ≤ 2,6 %); "đủ điều kiện thêm rule visual_syl_gate" ⇔ cận trên ≤ 3 %.
 (f) T2 chuỗi   : tỉ lệ sai_am theo ô và theo chuỗi (≥ 50 % ô sai_am -> đề xuất hạ REVIEW theo chuỗi).
     T3-B2      : âm cũ đúng / âm mới (argmax thị giác) đúng / cả hai sai.  T3-B5: hộp khoá QĐ-01 đúng crop?
     T6         : 'người' chưa khoá (283 ô, SRS 40): Q1 đúng / sai_am / sai_crop, HT về 283.
 (g) xuất       : docs/KET_QUA_KHOI_C_<ngày>.md + .json. KHÔNG ghi BANG_SO_LIEU (A-13/evidence làm sau khi người duyệt).

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.estimate_khoi_c                  # mẻ chính, 4 phiên
    .venv/bin/python -m pipeline.ground_truth.estimate_khoi_c --pilot          # phiên pilot 50 ô (C-1)
    .venv/bin/python -m pipeline.ground_truth.estimate_khoi_c --simulate DIR   # verdict giả lập để kiểm
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .estimate import AI_VERDICT_SOURCES, _verdict_source

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = REPO / "dataset_out" / "human_audit" / "khoi_c_2026-09-16"
DEFAULT_LABELS = REPO / "dataset_out" / "labels_final.csv"
DOCS = REPO / "docs"

Q1_VALUES = ("dung", "sai_crop", "sai_am", "khong_ro")
Q2_VALUES = ("dung", "sai", "khong_ro")
DWELL_FLAG_MS = 1500          # ô trả lời xong dưới 1,5 s -> cờ "quá nhanh"
DECOY_ALARM = 0.90            # mồi < 90 % -> báo động
T1_UPPER_MAX = 0.03           # B-3: cận trên tỉ lệ lỗi ≤ 3 % mới đủ điều kiện thêm rule
CHAIN_DEMOTE_FRAC = 0.50      # T2: ≥ 50 % ô trong chuỗi sai_am -> đề xuất hạ REVIEW theo chuỗi
GOLD_TIERS = ("CHAR_A", "CHAR_B")
USABLE_TIERS = ("CHAR_A", "CHAR_B", "SYL")
MAIN_TIERS = ("CHAR_A", "CHAR_B", "SYL", "REVIEW")
KAPPA_BOOT = 2000             # số lần bootstrap cho KTC κ
KAPPA_BOOT_SEED = 20260916
KAPPA_BANDS = ((0.8, "ổn định — precision công bố được"),
               (0.4, "công bố được nhưng PHẢI nêu κ kèm theo như một giới hạn"),
               (-1.0, "thiết kế chưa ổn — viết rubric chi tiết hơn rồi chấm lại"))


# ---------------------------------------------------------------------------
# nạp
def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return pd.DataFrame(rows)


def verdict_files(batch: Path, plan: dict, override: str | list[str] | None = None) -> list[Path]:
    """Tệp verdict của mẻ: theo `plan.sessions[].verdict_file` (chỉ tệp đã tồn tại), hoặc
    --verdicts (tệp / thư mục chứa verdicts*.jsonl / danh sách)."""
    if override:
        paths = [override] if isinstance(override, (str, Path)) else list(override)
        out: list[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                out += sorted(p.glob("verdicts*.jsonl"))
            else:
                out.append(p)
        return out
    return [batch / s["verdict_file"] for s in plan["sessions"] if (batch / s["verdict_file"]).exists()]


def load_verdicts(files: list[Path], include_ai: bool = False) -> tuple[pd.DataFrame, dict]:
    """Đọc verdict JSONL (schema HTML Khối C). Trùng item_id -> giữ bản `exported_at` mới nhất.
    Nguồn không phải người (source ≠ "human": ai_vision, simulated…) bị LOẠI trừ khi include_ai."""
    rows, skipped_ai, n_lines = [], 0, 0
    for fp in files:
        for i, ln in enumerate(Path(fp).read_text(encoding="utf-8").splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            n_lines += 1
            if "item_id" not in d:
                raise ValueError(f"{Path(fp).name}:{i + 1} thiếu item_id")
            src = _verdict_source(d)
            # CHẶT hơn estimate.py: mọi nguồn KHÔNG phải người (ai_vision, simulated…) đều bị loại
            # trừ khi include_ai — số ở đây là số công bố, verdict giả lập không được lọt vào.
            if (src in AI_VERDICT_SOURCES or src != "human") and not include_ai:
                skipped_ai += 1
                continue
            q1, q2 = d.get("q1"), d.get("q2")
            if q1 is not None and q1 not in Q1_VALUES:
                raise ValueError(f"{Path(fp).name}:{i + 1} q1 lạ {q1!r}")
            if q2 is not None and q2 not in Q2_VALUES:
                raise ValueError(f"{Path(fp).name}:{i + 1} q2 lạ {q2!r}")
            q1_blind = d.get("q1_blind")
            if q1_blind is not None and q1_blind not in Q1_VALUES:
                raise ValueError(f"{Path(fp).name}:{i + 1} q1_blind lạ {q1_blind!r}")
            hist = d.get("hist") or []
            rows.append({
                "item_id": str(d["item_id"]), "session_id": str(d.get("session_id") or ""),
                "order": d.get("order"), "q1_final": q1,
                # Q1 dùng để ước lượng = câu trả lời MÙ (trước khi lộ mã); thế hệ cũ không có -> q1
                "q1": q1_blind if q1_blind is not None else q1,
                "has_q1_blind": "q1_blind" in d,
                "q1_am": str(d.get("q1_am") or ""),
                "q2": q2, "q2_code": str(d.get("q2_code") or ""),
                "has_code": bool(d.get("has_code")), "dwell_ms": d.get("dwell_ms"),
                "dwell_q1_ms": d.get("dwell_q1_ms"),
                "n_q1_change_after_reveal": int(d.get("n_q1_change_after_reveal") or 0),
                "visits": d.get("visits"), "t_answer": d.get("t_answer"),
                "n_hist": int(sum(1 for h in hist if h.get("field") in ("q1", "q2"))),
                "exported_at": d.get("exported_at") or 0, "source": src, "file": Path(fp).name,
                "_line": i,
            })
    df = pd.DataFrame(rows)
    info = {"files": [str(f) for f in files], "n_lines": n_lines, "n_skipped_ai": skipped_ai}
    if df.empty:
        info["n_dup_item"] = 0
        return df, info
    for c in ("q1", "q1_final", "q2"):           # DataFrame(list[dict]) đổi None -> NaN: trả lại None
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    info["n_without_q1_blind"] = int((~df["has_q1_blind"]).sum())
    df = df.sort_values(["exported_at", "file", "_line"])
    n_dup = int(df["item_id"].duplicated().sum())
    df = df.drop_duplicates("item_id", keep="last").drop(columns="_line").reset_index(drop=True)
    info["n_dup_item"] = n_dup
    return df, info


# ---------------------------------------------------------------------------
# (a) toàn vẹn
def check_integrity(khoa_path: Path, manifest_path: Path, verdicts: pd.DataFrame,
                    labels_path: Path | None) -> tuple[dict, list[str]]:
    alarms: list[str] = []
    man = load_jsonl(manifest_path)
    khoa = load_jsonl(khoa_path)
    res: dict = {"khoa": str(khoa_path), "manifest": str(manifest_path), "n_khoa": int(len(khoa))}

    batch_sha = _sha256(khoa_path)
    man_batch = set(man["batch_sha256"].astype(str)) if "batch_sha256" in man.columns else set()
    res["batch_sha256"] = batch_sha
    res["batch_sha256_ok"] = man_batch == {batch_sha}
    if not res["batch_sha256_ok"]:
        alarms.append(f"KHOA.jsonl sha256 {batch_sha[:16]}… ≠ manifest {sorted(man_batch)}")

    man_labels = set(man["labels_sha256"].astype(str)) if "labels_sha256" in man.columns else set()
    res["labels_sha256_manifest"] = sorted(man_labels)
    if labels_path is not None and Path(labels_path).exists():
        cur = _sha256(Path(labels_path))
        res["labels_sha256_current"] = cur
        res["labels_sha256_ok"] = man_labels == {cur}
        if not res["labels_sha256_ok"]:
            alarms.append(f"labels_final.csv hiện hành sha256 {cur[:16]}… ≠ lúc dựng mẻ "
                          f"{[s[:16] + '…' for s in sorted(man_labels)]} — bộ nhãn đã đổi sau khi rút mẫu")
    else:
        res["labels_sha256_current"] = None
        res["labels_sha256_ok"] = None
        alarms.append("không có labels_final.csv để đối chiếu sha256")

    ids_khoa = set(khoa["item_id"].astype(str))
    ids_man = set(man["item_id"].astype(str))
    res["khoa_manifest_same_ids"] = ids_khoa == ids_man
    if not res["khoa_manifest_same_ids"]:
        alarms.append("KHOA và manifest không cùng tập item_id")

    if verdicts.empty:
        res.update({"n_verdict": 0, "n_unknown_id": 0, "n_graded": 0, "n_complete": 0})
        alarms.append("không có verdict nào")
        return res, alarms

    unknown = sorted(set(verdicts["item_id"]) - ids_khoa)
    res["n_verdict"] = int(len(verdicts))
    res["n_unknown_id"] = len(unknown)
    res["unknown_ids"] = unknown[:10]
    if unknown:
        alarms.append(f"{len(unknown)} item_id trong verdict KHÔNG có trong khoá (vd {unknown[:3]}) — "
                      "verdict của mẻ khác / phiên pilot lọt vào?")
    sess_known = set(man["session_id"].astype(str)) if "session_id" in man.columns else set()
    sess_seen = set(verdicts["session_id"])
    res["sessions_seen"] = sorted(sess_seen)
    res["sessions_missing"] = sorted(sess_known - sess_seen)
    if sess_seen - sess_known:
        alarms.append(f"session_id lạ: {sorted(sess_seen - sess_known)}")

    j = verdicts.merge(khoa[["item_id", "has_q2"]], on="item_id", how="inner")
    mism = int((j["has_code"] != j["has_q2"].astype(bool)).sum())
    res["n_has_code_mismatch"] = mism
    if mism:
        alarms.append(f"{mism} ô has_code (HTML) ≠ has_q2 (khoá) — HTML và khoá không cùng lần dựng")

    graded = verdicts["q1"].notna()
    complete = graded & (verdicts["q2"].notna() | ~verdicts["has_code"])
    res["n_graded"] = int(graded.sum())
    res["n_complete"] = int(complete.sum())
    res["n_ungraded_in_khoa"] = int(len(ids_khoa) - int(graded[verdicts["item_id"].isin(ids_khoa)].sum()))
    if res["n_ungraded_in_khoa"]:
        alarms.append(f"{res['n_ungraded_in_khoa']}/{len(ids_khoa)} ô chưa chấm — nếu bỏ sót liên quan độ khó "
                      "thì ước lượng chệch")
    src = verdicts["source"].value_counts().to_dict()
    res["sources"] = {str(k): int(v) for k, v in src.items()}
    return res, alarms


def join_khoa(verdicts: pd.DataFrame, khoa_path: Path) -> pd.DataFrame:
    """verdict ⋈ KHOA (inner), chỉ giữ ô đã trả lời Q1. Thêm cờ q1_ok/q2_ok/q1_scored/q2_scored."""
    khoa = load_jsonl(khoa_path)
    khoa["item_id"] = khoa["item_id"].astype(str)
    j = verdicts[verdicts["q1"].notna()].merge(khoa, on="item_id", how="inner", suffixes=("", "_khoa"))
    j["q1_scored"] = j["q1"] != "khong_ro"
    j["q1_ok"] = j["q1"] == "dung"
    j["q2_scored"] = j["q2"].isin(["dung", "sai"])
    j["q2_ok"] = j["q2"] == "dung"
    j["q12_scored"] = j["q1_scored"] & j["q2_scored"]
    j["q12_ok"] = j["q1_ok"] & j["q2_ok"]
    j["design_weight"] = pd.to_numeric(j["design_weight"], errors="coerce")
    j["stratum_N"] = pd.to_numeric(j["stratum_N"], errors="coerce")
    return j


# ---------------------------------------------------------------------------
# (b) ô mồi
def _norm(s) -> str:
    return unicodedata.normalize("NFC", str(s or "")).strip().lower()


def decoy_report(j: pd.DataFrame) -> dict:
    pos = j[j["tang"] == "T4_MOI_DUONG"]
    neg = j[j["tang"] == "T4_MOI_AM"]

    def _blk(g: pd.DataFrame, q1_pass, q2_pass) -> dict:
        n = int(len(g))
        if n == 0:
            return {"n": 0}
        p1 = q1_pass(g)
        p2 = q2_pass(g)
        both = p1 & p2
        k = int(both.sum())
        return {"n": n, "n_pass": k, "accuracy": k / n, "wilson_ci": stats.wilson_ci(k, n),
                "q1_pass": int(p1.sum()), "q2_pass": int(p2.sum()),
                "q1_counts": {v: int((g["q1"] == v).sum()) for v in Q1_VALUES},
                "q2_counts": {v: int((g["q2"] == v).sum()) for v in Q2_VALUES},
                "failed_ids": sorted(g.loc[~both, "item_id"].tolist())}

    # mồi dương: đạt ⇔ Q1 (mù) = dung. Q2 KHÔNG tính đạt/rớt: mã 𠊚 là phán quyết QĐ-01 mà người chấm
    # (chủ nhiệm) nhận ra ngay khi lộ mã -> chỉ đếm "nhất quán với phán quyết".
    out = {
        "pos": _blk(pos, lambda g: g["q1"] == "dung", lambda g: pd.Series(True, index=g.index)),
        "neg": _blk(neg, lambda g: g["q1"].isin(["sai_am", "sai_crop"]), lambda g: g["q2"] == "sai"),
        "alarm_threshold": DECOY_ALARM,
        "pos_rule": "Q1 == dung (Q2 chỉ đếm, không tính — mã QĐ-01 nhận ra được)",
        "neg_rule": "Q1 ∈ {sai_am, sai_crop} ∧ Q2 == sai",
    }
    if len(pos):
        out["pos"]["q2_dung_nhat_quan_qd01"] = int((pos["q2"] == "dung").sum())
        out["pos"]["q2_sai_trai_qd01"] = int((pos["q2"] == "sai").sum())
    # mồi âm: ghi "âm đúng là…" có khớp âm thật của ô không (thông tin phụ, không tính đạt/rớt)
    if len(neg):
        wrote = neg[neg["q1_am"] != ""]
        out["neg"]["n_wrote_am"] = int(len(wrote))
        out["neg"]["n_wrote_am_matches_true"] = int(
            sum(_norm(a) == _norm(t) for a, t in zip(wrote["q1_am"], wrote["syllable"])))
    n_all = out["pos"].get("n", 0) + out["neg"].get("n", 0)
    k_all = out["pos"].get("n_pass", 0) + out["neg"].get("n_pass", 0)
    out["all"] = {"n": n_all, "n_pass": k_all, "accuracy": (k_all / n_all) if n_all else None}
    out["alarm"] = bool(n_all and (
        (out["pos"].get("n") and out["pos"]["accuracy"] < DECOY_ALARM)
        or (out["neg"].get("n") and out["neg"]["accuracy"] < DECOY_ALARM)))
    return out


# ---------------------------------------------------------------------------
# (c) lặp ẩn + dwell
def _kappa_band(k: float | None) -> str | None:
    if k is None:
        return None
    for thr, txt in KAPPA_BANDS:
        if k >= thr:
            return txt
    return KAPPA_BANDS[-1][1]


def kappa_bootstrap_ci(pairs: list[tuple[str, str]], categories: tuple[str, ...], conf: float = 0.95,
                       n_boot: int = KAPPA_BOOT, seed: int = KAPPA_BOOT_SEED) -> tuple[float, float] | None:
    """KTC phần trăm bootstrap cho κ Cohen (lấy mẫu lại CẶP, seed cố định -> tất định).
    Trả None khi < 2 cặp hoặc mọi mẫu lại đều suy biến (pe = 1)."""
    if len(pairs) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(pairs, dtype=object)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        k = stats.cohens_kappa([tuple(x) for x in arr[idx]], categories)["kappa"]
        if k is not None and np.isfinite(k):
            ks.append(k)
    if len(ks) < n_boot // 2:
        return None
    a = (1 - conf) / 2
    return (float(np.percentile(ks, 100 * a)), float(np.percentile(ks, 100 * (1 - a))))


def repeat_report(j: pd.DataFrame, conf: float = 0.95) -> dict:
    cols = ["item_id", "repeat_of", "q1", "q2", "has_q2"] + (["lap_tang"] if "lap_tang" in j.columns else [])
    rep = j[j["tang"] == "T5_LAP"][cols]
    orig = j[["item_id", "q1", "q2"]].rename(columns={"item_id": "repeat_of", "q1": "q1_o", "q2": "q2_o"})
    p = rep.merge(orig, on="repeat_of", how="inner")
    out: dict = {"n_repeat_in_batch": int((j["tang"] == "T5_LAP").sum()), "n_pairs": int(len(p))}
    if p.empty:
        out["q1"] = out["q2"] = {"n": 0, "kappa": None}
        return out
    pairs1 = list(zip(p["q1_o"], p["q1"]))
    out["q1"] = stats.cohens_kappa(pairs1, Q1_VALUES)
    out["q1"]["ci_boot"] = kappa_bootstrap_ci(pairs1, Q1_VALUES, conf)
    pairs1b = [("dung" if a == "dung" else "khac", "dung" if b == "dung" else "khac") for a, b in pairs1]
    out["q1_binary"] = stats.cohens_kappa(pairs1b, ("dung", "khac"))
    out["q1_binary"]["ci_boot"] = kappa_bootstrap_ci(pairs1b, ("dung", "khac"), conf)
    p2 = p[p["q2"].notna() & p["q2_o"].notna()]
    pairs2 = list(zip(p2["q2_o"], p2["q2"]))
    out["q2"] = stats.cohens_kappa(pairs2, Q2_VALUES)
    out["q2"]["ci_boot"] = kappa_bootstrap_ci(pairs2, Q2_VALUES, conf)
    out["q1"]["band"] = _kappa_band(out["q1"]["kappa"])
    out["q2"]["band"] = _kappa_band(out["q2"]["kappa"])
    out["n_pairs_q2"] = int(len(p2))
    out["disagree_ids_q1"] = sorted(p.loc[p["q1"] != p["q1_o"], "item_id"].tolist())
    # đồng thuận thô Q1 kèm Wilson (ít nhạy tỉ lệ nền hơn κ)
    n_agree = int((p["q1"] == p["q1_o"]).sum())
    out["q1_agree_wilson"] = stats.wilson_ci(n_agree, len(p), conf)
    if "lap_tang" in p.columns:
        out["by_lap_tang"] = {}
        for t, g in p.groupby(p["lap_tang"].fillna("MAIN")):
            pr = list(zip(g["q1_o"], g["q1"]))
            kk = stats.cohens_kappa(pr, Q1_VALUES)
            out["by_lap_tang"][str(t)] = {"n": int(len(g)), "kappa": kk["kappa"],
                                          "observed_agreement": kk["observed_agreement"],
                                          "n_non_dung_orig": int((g["q1_o"] != "dung").sum())}
    return out


def dwell_report(v: pd.DataFrame, flag_ms: int = DWELL_FLAG_MS) -> dict:
    g = v[v["q1"].notna()].copy()
    d = pd.to_numeric(g["dwell_ms"], errors="coerce")
    ok = d.notna()
    out: dict = {"n_graded": int(len(g)), "n_with_dwell": int(ok.sum()), "flag_ms": flag_ms}
    if not ok.any():
        return out
    dd = d[ok] / 1000.0
    out.update({"p10_s": float(np.percentile(dd, 10)), "p50_s": float(np.percentile(dd, 50)),
                "p90_s": float(np.percentile(dd, 90)), "mean_s": float(dd.mean()),
                "total_min": float(dd.sum() / 60.0),
                "n_flag_fast": int((d[ok] < flag_ms).sum()),
                "frac_flag_fast": float((d[ok] < flag_ms).mean()),
                "flag_fast_ids": sorted(g.loc[ok & (d < flag_ms), "item_id"].tolist()),
                "n_revisited": int((pd.to_numeric(g["visits"], errors="coerce") > 1).sum())})
    need = 1 + g["has_code"].astype(int)
    out["n_changed_mind"] = int((g["n_hist"] > need).sum())
    if "n_q1_change_after_reveal" in g.columns:
        ch = pd.to_numeric(g["n_q1_change_after_reveal"], errors="coerce").fillna(0)
        out["n_q1_changed_after_reveal"] = int((ch > 0).sum())
        out["q1_changed_after_reveal_ids"] = sorted(g.loc[ch > 0, "item_id"].tolist())
        if "q1_final" in g.columns:
            out["n_q1_blind_ne_final"] = int((g["q1_final"].notna() & (g["q1_final"] != g["q1"])).sum())
    if "dwell_q1_ms" in g.columns:
        d1 = pd.to_numeric(g["dwell_q1_ms"], errors="coerce").dropna() / 1000.0
        if len(d1):
            out["q1_p50_s"] = float(np.median(d1))
            out["q1_p90_s"] = float(np.percentile(d1, 90))
    per = []
    for s, gs in g.groupby("session_id"):
        ds = pd.to_numeric(gs["dwell_ms"], errors="coerce").dropna() / 1000.0
        if len(ds):
            per.append({"session_id": s, "n": int(len(gs)), "p50_s": float(np.median(ds)),
                        "total_min": float(ds.sum() / 60.0), "n_flag_fast": int((ds * 1000 < flag_ms).sum())})
    out["per_session"] = per
    return out


# ---------------------------------------------------------------------------
# (d) precision theo tầng / tier
def _ht(g: pd.DataFrame, scored_col: str, ok_col: str, conf: float) -> dict:
    """Horvitz–Thompson trên các stratum có stratum_N; khong_ro loại khỏi n_h; kèm Wilson gộp."""
    s = g[g[scored_col]]
    n, k = int(len(s)), int(s[ok_col].sum())
    out: dict = {"n_audited": int(len(g)), "n_scored": n, "n_ok": k,
                 "n_khong_ro": int(len(g) - n)}
    if n == 0:
        return out
    out["precision"] = k / n
    out["wilson_ci"] = stats.wilson_ci(k, n, conf)
    out["cp_lower_one_sided"] = stats.cp_lower_bound(k, n, conf)
    triples, pop = [], 0
    for st, gg in s.groupby("stratum"):
        N = gg["stratum_N"].dropna()
        if N.empty:
            continue
        N_h = int(N.iloc[0])
        triples.append((max(N_h, len(gg)), int(len(gg)), int(gg[ok_col].sum())))
        pop += N_h
    if triples:
        pt, lo, hi = stats.stratified_mean_ci(triples, conf)
        out["weighted_precision"] = pt
        out["weighted_ci"] = (lo, hi)
        out["population_N"] = pop
        out["n_strata"] = len(triples)
    return out


def precision_report(j: pd.DataFrame, conf: float = 0.95) -> dict:
    m = j[j["tang"] == "MAIN"]
    out: dict = {"n_main": int(len(m)), "tiers": {}, "strata": [], "pooled": {}}
    for tier in MAIN_TIERS:
        g = m[m["tier_v3"] == tier]
        if g.empty:
            continue
        t = {"q1": _ht(g, "q1_scored", "q1_ok", conf),
             "q1_errors": {v: int((g["q1"] == v).sum()) for v in Q1_VALUES}}
        gq = g[g["has_q2"].astype(bool)]
        t["n_has_q2"] = int(len(gq))
        if len(gq):
            t["q2"] = _ht(gq, "q2_scored", "q2_ok", conf)
            t["q2_errors"] = {v: int((gq["q2"] == v).sum()) for v in Q2_VALUES}
            t["q1_and_q2"] = _ht(gq, "q12_scored", "q12_ok", conf)
            if len(gq) < len(g):
                t["q2_note"] = (f"chỉ {len(gq)}/{len(g)} ô tier này có mã (tier=GOLD nhưng tier_v3=SYL) — "
                                "Q2 không đại diện cho tier, đọc như số đếm; không suy HT")
                for k_ in ("q2", "q1_and_q2"):
                    for f_ in ("weighted_precision", "weighted_ci", "population_N", "n_strata"):
                        t[k_].pop(f_, None)
        out["tiers"][tier] = t
    for st, g in m.groupby("stratum"):
        s = g[g["q1_scored"]]
        n, k = int(len(s)), int(s["q1_ok"].sum())
        row = {"stratum": st, "tier_v3": st.split("|")[0], "N_h": int(g["stratum_N"].iloc[0]),
               "n_audited": int(len(g)), "n_scored": n, "q1_ok": k,
               "q1_precision": (k / n) if n else None, "q1_wilson": stats.wilson_ci(k, n, conf) if n else None,
               "sai_crop": int((g["q1"] == "sai_crop").sum()), "sai_am": int((g["q1"] == "sai_am").sum()),
               "khong_ro": int((g["q1"] == "khong_ro").sum())}
        gq = g[g["has_q2"].astype(bool) & g["q2_scored"]]
        if len(gq):
            row["q2_n"], row["q2_ok"] = int(len(gq)), int(gq["q2_ok"].sum())
        out["strata"].append(row)
    for name, tiers in (("GOLD", GOLD_TIERS), ("USABLE", USABLE_TIERS)):
        g = m[m["tier_v3"].isin(tiers)]
        if g.empty:
            continue
        blk = {"tiers": list(tiers), "q1": _ht(g, "q1_scored", "q1_ok", conf)}
        if name == "GOLD":
            gq = g[g["has_q2"].astype(bool)]
            blk["q2"] = _ht(gq, "q2_scored", "q2_ok", conf)
            blk["q1_and_q2"] = _ht(gq, "q12_scored", "q12_ok", conf)
        out["pooled"][name] = blk
    # tầng QĐ-01 = mồi dương (SRS phân tầng theo tier_v3 trên 2.012 ô QĐ-01, có stratum_N):
    # Q1 là số đo thật (người nhìn mù crop + 'người'); Q2 = nhất quán với phán quyết (không phải số đo).
    qd = j[(j["tang"] == "T4_MOI_DUONG") & j["stratum_N"].notna()]
    if len(qd):
        out["pooled"]["QD01"] = {"tiers": sorted(qd["tier_v3"].unique().tolist()),
                                 "q1": _ht(qd, "q1_scored", "q1_ok", conf),
                                 "q2_nhat_quan_qd01": _ht(qd, "q2_scored", "q2_ok", conf),
                                 "note": "Q2 trên ô QĐ-01: mã 𠊚 là phán quyết đã biết -> đọc là 'nhất quán', không phải precision mã"}
        # re-dataset 70.326 = USABLE (ngoài QĐ-01, 68.314) + QĐ-01 (2.012): HT trên hợp các tầng
        u = m[m["tier_v3"].isin(USABLE_TIERS)]
        if len(u):
            both = pd.concat([u, qd])
            out["pooled"]["USABLE_RE_DATASET"] = {
                "tiers": list(USABLE_TIERS) + ["QD01"], "q1": _ht(both, "q1_scored", "q1_ok", conf),
                "note": "dân số = re-dataset/labels.csv (CHAR_A+CHAR_B+SYL ngoài QĐ-01 ∪ 2.012 ô QĐ-01)"}
    return out


def t6_report(j: pd.DataFrame, conf: float = 0.95) -> dict:
    """T6 — 'người' chưa khoá (ngoài QĐ-01): người nhìn mù crop + ÂM 'người' (không có mã ở đa số ô)."""
    g = j[j["tang"] == "T6_NGUOI"]
    out: dict = {"n_audited": int(len(g))}
    if g.empty:
        return out
    s = g[g["q1_scored"]]
    n, k = int(len(s)), int(s["q1_ok"].sum())
    out.update({"n_scored": n, "n_ok": k, "q1_counts": {v: int((g["q1"] == v).sum()) for v in Q1_VALUES},
                "by_tier_v3": {t: {v: int((gg["q1"] == v).sum()) for v in Q1_VALUES}
                               for t, gg in g.groupby("tier_v3")}})
    if n:
        out["precision"] = k / n
        out["wilson_ci"] = stats.wilson_ci(k, n, conf)
        ht = _ht(g, "q1_scored", "q1_ok", conf)
        out["weighted_precision"] = ht.get("weighted_precision")
        out["weighted_ci"] = ht.get("weighted_ci")
        out["population_N"] = ht.get("population_N")
    wrote = g[(g["q1"] == "sai_am") & (g["q1_am"] != "")]
    out["sai_am_wrote"] = sorted(wrote["q1_am"].tolist())
    return out


# ---------------------------------------------------------------------------
# (e) T1 — cổng thị giác B-3
def t1_report(j: pd.DataFrame, conf: float = 0.95, upper_max: float = T1_UPPER_MAX) -> dict:
    g = j[j["tang"] == "T1_GATE"]
    out: dict = {"n_audited": int(len(g)), "upper_max": upper_max}
    if g.empty:
        out["eligible"] = None
        return out
    s = g[g["q1_scored"]]
    n, k = int(len(s)), int(s["q1_ok"].sum())
    err = n - k
    out.update({"n_scored": n, "n_ok": k, "n_err": err, "n_khong_ro": int(len(g) - n),
                "q1_counts": {v: int((g["q1"] == v).sum()) for v in Q1_VALUES}})
    if n == 0:
        out["eligible"] = None
        return out
    out["err_rate"] = err / n
    out["err_upper_cp95"] = stats.cp_upper_bound(err, n, conf)
    out["ok_wilson_ci"] = stats.wilson_ci(k, n, conf)
    # bảo thủ: khong_ro tính là lỗi
    n_all, err_all = int(len(g)), err + int(len(g) - n)
    out["err_upper_cp95_conservative"] = stats.cp_upper_bound(err_all, n_all, conf)
    out["eligible"] = bool(out["err_upper_cp95"] <= upper_max)
    out["eligible_conservative"] = bool(out["err_upper_cp95_conservative"] <= upper_max)
    ht = _ht(g, "q1_scored", "q1_ok", conf)
    out["weighted_precision"] = ht.get("weighted_precision")
    out["weighted_ci"] = ht.get("weighted_ci")
    out["per_book"] = [{"stratum": st, "n": int(gg["q1_scored"].sum()), "ok": int(gg["q1_ok"].sum()),
                        "khong_ro": int((~gg["q1_scored"]).sum())} for st, gg in g.groupby("stratum")]
    out["decision"] = (
        f"{'ĐỦ' if out['eligible'] else 'CHƯA ĐỦ'} điều kiện thêm rule visual_syl_gate: lỗi {err}/{n}, "
        f"cận trên CP một phía 95 % = {out['err_upper_cp95']:.2%} (ngưỡng ≤ {upper_max:.0%})"
        + ("" if out["eligible_conservative"] == out["eligible"]
           else f"; nếu tính {len(g) - n} ô KHÔNG RÕ là lỗi thì cận trên = "
                f"{out['err_upper_cp95_conservative']:.2%} — kết luận đổi, phải xem lại các ô đó"))
    return out


# ---------------------------------------------------------------------------
# (f) T2 chuỗi trượt · T3 B-2 / B-5
def t2_report(j: pd.DataFrame) -> dict:
    g = j[j["tang"] == "T2_CHAIN"].copy()
    out: dict = {"n_audited": int(len(g))}
    if g.empty:
        return out
    s = g[g["q1_scored"]]
    out.update({"n_scored": int(len(s)), "q1_counts": {v: int((g["q1"] == v).sum()) for v in Q1_VALUES},
                "sai_am_rate": float((s["q1"] == "sai_am").mean()) if len(s) else None,
                "sai_am_wilson": stats.wilson_ci(int((s["q1"] == "sai_am").sum()), len(s)) if len(s) else None})
    wrote = g[(g["q1"] == "sai_am") & (g["q1_am"] != "")]
    out["n_sai_am_wrote"] = int(len(wrote))
    out["n_sai_am_matches_argmax"] = int(sum(_norm(a) == _norm(b) for a, b in zip(wrote["q1_am"], wrote["argmax"])))
    chains = []
    for cid, gg in g.groupby("chain_id"):
        n_s = int(gg["q1_scored"].sum())
        n_sa = int((gg["q1"] == "sai_am").sum())
        frac = (n_sa / n_s) if n_s else None
        chains.append({"chain_id": cid, "n": int(len(gg)), "n_scored": n_s, "n_sai_am": n_sa,
                       "n_sai_crop": int((gg["q1"] == "sai_crop").sum()), "n_dung": int(gg["q1_ok"].sum()),
                       "frac_sai_am": frac, "demote": bool(frac is not None and frac >= CHAIN_DEMOTE_FRAC),
                       "n_qd01": int((gg["qd01_locked"].astype(str) == "1").sum())})
    out["chains"] = chains
    out["n_chains"] = len(chains)
    out["n_chains_demote"] = int(sum(c["demote"] for c in chains))
    out["n_cells_in_demote_chains"] = int(sum(c["n"] for c in chains if c["demote"]))
    out["demote_frac_threshold"] = CHAIN_DEMOTE_FRAC
    out["by_tier_v3"] = {t: {v: int((gg["q1"] == v).sum()) for v in Q1_VALUES}
                         for t, gg in g.groupby("tier_v3")}
    q = g[g["qd01_locked"].astype(str) == "1"]
    out["qd01_cells"] = {"n": int(len(q)), "q1_dung": int(q["q1_ok"].sum()),
                         "q2_dung": int(q["q2_ok"].sum()), "q2_sai": int((q["q2"] == "sai").sum())}
    return out


def t3_report(j: pd.DataFrame, conf: float = 0.95) -> dict:
    out: dict = {}
    b2 = j[j["tang"] == "T3_B2"]
    if len(b2):
        cls = []
        for r in b2.itertuples():
            same = _norm(r.syllable) == _norm(r.argmax)      # 2/19 ô B-2 chỉ đổi syl_idx, âm giữ nguyên
            if r.q1 == "dung":
                c = "am_cu_dung"
            elif r.q1 == "sai_am" and same:
                c = "ca_hai_sai"
            elif r.q1 == "sai_am" and r.q1_am and _norm(r.q1_am) == _norm(r.argmax):
                c = "am_moi_dung"
            elif r.q1 == "sai_am" and r.q1_am:
                c = "ca_hai_sai"
            elif r.q1 == "sai_am":
                c = "am_cu_sai_khong_ghi"
            else:
                c = r.q1                       # sai_crop / khong_ro
            cls.append({"item_id": r.item_id, "key": f"{r.book}|{r.page}|{r.column}|{r.nom_idx}",
                        "am_cu": r.syllable, "am_moi": r.argmax, "q1": r.q1, "q1_am": r.q1_am,
                        "q2": r.q2, "ket_luan": c})
        out["b2"] = {"n": int(len(b2)),
                     "counts": {c: int(sum(x["ket_luan"] == c for x in cls)) for c in
                                ("am_cu_dung", "am_moi_dung", "ca_hai_sai", "am_cu_sai_khong_ghi", "sai_crop", "khong_ro")},
                     "cells": cls}
    b5 = j[j["tang"] == "T3_B5"]
    if len(b5):
        s = b5[b5["q1_scored"]]
        n, k = int(len(s)), int(s["q1_ok"].sum())
        blk = {"n": int(len(b5)), "n_scored": n, "q1_counts": {v: int((b5["q1"] == v).sum()) for v in Q1_VALUES},
               "q2_counts": {v: int((b5["q2"] == v).sum()) for v in Q2_VALUES},
               "note": ("người chỉ thấy crop theo hộp ĐÃ KHOÁ (QĐ-01); hộp detector trước khoá "
                        "(bbox_truoc_lock) KHÔNG hiển thị. 'sai_crop' = hộp khoá sai; không suy ra hộp detector đúng.")}
        if n:
            blk["hop_khoa_dung"] = k / n
            blk["wilson_ci"] = stats.wilson_ci(k, n, conf)
            ht = _ht(b5, "q1_scored", "q1_ok", conf)
            blk["weighted_precision"] = ht.get("weighted_precision")
            blk["weighted_ci"] = ht.get("weighted_ci")
            blk["population_N"] = ht.get("population_N")
        out["b5"] = blk
    return out


def qd01_report(j: pd.DataFrame) -> dict:
    """Mọi ô QĐ-01 người nhìn lại mù (mồi dương + chuỗi + B-5): Q2 mã 𠊚 đúng?"""
    q = j[j["qd01_locked"].astype(str) == "1"]
    if q.empty:
        return {"n": 0}
    s2 = q[q["q2_scored"]]
    return {"n": int(len(q)), "by_tang": {t: int(v) for t, v in q["tang"].value_counts().items()},
            "q1_dung": int(q["q1_ok"].sum()), "q1_sai_crop": int((q["q1"] == "sai_crop").sum()),
            "q1_sai_am": int((q["q1"] == "sai_am").sum()),
            "q2_n": int(len(s2)), "q2_dung": int(s2["q2_ok"].sum()),
            "q2_wilson": stats.wilson_ci(int(s2["q2_ok"].sum()), len(s2)) if len(s2) else None}


# ---------------------------------------------------------------------------
# tổng hợp
def build(batch: Path, verdict_paths: list[Path], labels_path: Path | None, pilot: bool = False,
          conf: float = 0.95, include_ai: bool = False) -> dict:
    khoa = batch / "_khoa" / ("KHOA_pilot.jsonl" if pilot else "KHOA.jsonl")
    man = batch / "_khoa" / ("manifest_pilot.jsonl" if pilot else "manifest.jsonl")
    v, vinfo = load_verdicts(verdict_paths, include_ai=include_ai)
    integ, alarms = check_integrity(khoa, man, v, labels_path)
    rep: dict = {"batch": str(batch.relative_to(REPO)) if batch.is_relative_to(REPO) else str(batch),
                 "mode": "pilot" if pilot else "main", "conf": conf,
                 "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                 "verdict_input": vinfo, "integrity": integ}
    if v.empty:
        rep["alarms"] = alarms
        return rep
    j = join_khoa(v, khoa)
    rep["n_joined"] = int(len(j))
    rep["decoys"] = decoy_report(j)
    if rep["decoys"]["alarm"]:
        alarms.append(f"Ô MỒI dưới {DECOY_ALARM:.0%}: dương {rep['decoys']['pos'].get('accuracy')}, "
                      f"âm {rep['decoys']['neg'].get('accuracy')} — verdict không đủ tin cậy để công bố")
    rep["repeats"] = repeat_report(j, conf)
    k1 = rep["repeats"].get("q1", {}).get("kappa")
    if k1 is not None and k1 < 0.4:
        alarms.append(f"κ Q1 = {k1:.3f} < 0,4 — thiết kế chưa ổn (KE_HOACH_CHAM_TAY C.0.b)")
    rep["dwell"] = dwell_report(j)                      # chỉ ô thuộc khoá (id lạ đã bị báo ở toàn vẹn)
    if rep["dwell"].get("frac_flag_fast", 0) > 0.10:
        alarms.append(f"{rep['dwell']['n_flag_fast']} ô ({rep['dwell']['frac_flag_fast']:.0%}) trả lời < "
                      f"{DWELL_FLAG_MS / 1000:.1f} s")
    n_old = vinfo.get("n_without_q1_blind", 0)
    rep["q1_source"] = ("q1_blind (câu trả lời trước khi lộ mã)" if n_old == 0
                        else f"q1 cuối cho {n_old} verdict thế hệ cũ không có q1_blind")
    if n_old:
        alarms.append(f"{n_old} verdict không có q1_blind (HTML thế hệ trước 16/09 C3) — Q1 của các ô đó "
                      "được trả lời KHI ĐÃ THẤY mã/nhóm không mã: không mù giữa CHAR và SYL/REVIEW")
    rep["raw_counts"] = {"q1": {v_: int((j["q1"] == v_).sum()) for v_ in Q1_VALUES},
                         "q2": {v_: int((j["q2"] == v_).sum()) for v_ in Q2_VALUES},
                         "by_tang": {t: int(n) for t, n in j["tang"].value_counts().sort_index().items()}}
    if not pilot:
        rep["precision"] = precision_report(j, conf)
        rep["t1_gate"] = t1_report(j, conf)
        rep["t2_chains"] = t2_report(j)
        rep["t3"] = t3_report(j, conf)
        rep["t6_nguoi"] = t6_report(j, conf)
        rep["qd01"] = qd01_report(j)
    rep["alarms"] = alarms
    return rep


# ---------------------------------------------------------------------------
# giả lập (để kiểm ước lượng — KHÔNG BAO GIỜ là verdict thật; source="simulated")
SIM_ERR = {"CHAR_A": 0.04, "CHAR_B": 0.12, "SYL": 0.10, "REVIEW": 0.40, "T1_GATE": 0.0,
           "T2_CHAIN": 0.50, "T3_B2": 0.5, "T3_B5": 0.2, "T6_NGUOI": 0.25}
SIM_Q2_ERR = {"CHAR_A": 0.03, "CHAR_B": 0.10, "SYL": 0.0, "REVIEW": 0.0, "T3_B5": 0.0, "T2_CHAIN": 0.1, "T3_B2": 0.2,
              "T6_NGUOI": 0.0}


def simulate_verdicts(khoa: pd.DataFrame, seed: int = 1, noise: float = 0.05,
                      err: dict | None = None, q2_err: dict | None = None,
                      dwell_median_s: float = 6.0, p_change_after_reveal: float = 0.0
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Người chấm 'lý tưởng' + nhiễu.

    Sự thật được GIEO tất định theo tầng: ô thường sai với xác suất err[tang/tier_v3] (Q1 sai chia
    đều sai_crop / sai_am), Q2 sai với q2_err; ô mồi theo đáp án; ô lặp = y hệt bản gốc. Sau đó
    mỗi câu trả lời bị đổi sang giá trị KHÁC với xác suất `noise`. Verdict có `q1_blind` (= câu trả
    lời trước khi lộ mã) và, với xác suất `p_change_after_reveal` trên ô có mã, `q1` cuối bị đổi
    sang giá trị khác (mô phỏng người chấm sửa Q1 sau khi thấy glyph). Trả về (verdicts, truth).
    """
    err = dict(SIM_ERR if err is None else err)
    q2_err = dict(SIM_Q2_ERR if q2_err is None else q2_err)
    rng = np.random.default_rng(seed)
    kh = khoa.sort_values("audit_order").reset_index(drop=True)
    truth: dict[str, tuple[str, str | None]] = {}
    for r in kh.itertuples():
        has_q2 = bool(r.has_q2)
        if r.tang == "T5_LAP":
            continue                                   # điền sau, theo bản gốc
        if r.tang == "T4_MOI_DUONG":
            t1, t2 = "dung", "dung"
        elif r.tang == "T4_MOI_AM":
            t1, t2 = "sai_am", "sai"
        else:
            e = err.get(r.tang, err.get(getattr(r, "tier_v3", ""), 0.0)) if r.tang != "MAIN" \
                else err.get(getattr(r, "tier_v3", ""), 0.0)
            if rng.random() < e:
                t1 = "sai_am" if (r.tang in ("T2_CHAIN", "T3_B2") or rng.random() < 0.5) else "sai_crop"
            else:
                t1 = "dung"
            e2 = q2_err.get(r.tang, q2_err.get(getattr(r, "tier_v3", ""), 0.0)) if r.tang != "MAIN" \
                else q2_err.get(getattr(r, "tier_v3", ""), 0.0)
            t2 = ("sai" if rng.random() < e2 else "dung") if has_q2 else None
        truth[r.item_id] = (t1, t2 if has_q2 else None)
    for r in kh[kh["tang"] == "T5_LAP"].itertuples():
        truth[r.item_id] = truth[r.repeat_of]

    def _noisy(v, values):
        if v is None or rng.random() >= noise:
            return v
        return str(rng.choice([x for x in values if x != v]))

    rows, trows = [], []
    t0 = 1_758_000_000_000
    for i, r in enumerate(kh.itertuples()):
        t1, t2 = truth[r.item_id]
        a1, a2 = _noisy(t1, Q1_VALUES), _noisy(t2, Q2_VALUES)
        dwell = int(max(300, rng.lognormal(np.log(dwell_median_s * 1000), 0.6)))
        q1_am = ""
        if a1 == "sai_am":
            q1_am = str(getattr(r, "argmax", "") or "") if r.tang in ("T2_CHAIN", "T3_B2") else \
                (str(r.syllable) if r.tang == "T4_MOI_AM" else "")
        d1 = int(max(200, dwell * 0.7)) if a2 is not None else dwell
        hist = [{"ts": t0 + i * 7000 + d1, "field": "q1", "value": a1, "sitting": "sim", "after_reveal": False}]
        if a2 is not None:
            hist.append({"ts": t0 + i * 7000 + dwell + 10, "field": "q2", "value": a2, "sitting": "sim",
                         "after_reveal": False})
        a1_final, n_ch = a1, 0
        if a2 is not None and p_change_after_reveal > 0 and rng.random() < p_change_after_reveal:
            a1_final = str(rng.choice([x for x in Q1_VALUES if x != a1]))
            n_ch = 1
            hist.append({"ts": t0 + i * 7000 + dwell + 20, "field": "q1", "value": a1_final, "sitting": "sim",
                         "after_reveal": True})
        rows.append({"item_id": r.item_id, "session_id": r.session_id, "order": int(r.audit_order),
                     "q1": a1_final, "q1_blind": a1, "q1_am": q1_am, "q2": a2, "q2_code": "",
                     "has_code": bool(r.has_q2), "t_first_shown": t0 + i * 7000,
                     "revealed_at": t0 + i * 7000 + d1, "t_answer": t0 + i * 7000 + dwell,
                     "dwell_q1_ms": d1, "dwell_ms": dwell, "n_q1_change_after_reveal": n_ch,
                     "visits": 1, "sitting_id": "sim", "hist": hist, "exported_at": t0 + 10_000_000,
                     "source": "simulated"})
        trows.append({"item_id": r.item_id, "tang": r.tang, "truth_q1": t1, "truth_q2": t2,
                      "ans_q1": a1, "ans_q1_final": a1_final, "ans_q2": a2, "dwell_ms": dwell})
    return pd.DataFrame(rows), pd.DataFrame(trows)


def write_simulated(khoa_path: Path, out_dir: Path, seed: int = 1, noise: float = 0.05) -> list[Path]:
    kh = load_jsonl(khoa_path)
    v, truth = simulate_verdicts(kh, seed=seed, noise=noise)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sid, g in v.groupby("session_id"):
        p = out_dir / f"verdicts_{sid}.jsonl"
        recs = [{k: (None if (isinstance(x, float) and np.isnan(x)) else x) for k, x in r.items()}
                for r in g.to_dict("records")]
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        paths.append(p)
    truth.to_csv(out_dir / "_truth_simulated.csv", index=False)
    return paths


# ---------------------------------------------------------------------------
# markdown
def _pct(x, d: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{d}f} %"


def _ci(ci, d: int = 1) -> str:
    return "—" if not ci else f"[{100 * ci[0]:.{d}f}; {100 * ci[1]:.{d}f}]"


def _ht_row(name: str, b: dict) -> str:
    if not b or b.get("n_scored", 0) == 0:
        return f"| {name} | 0 | — | — | — | — | — |"
    return (f"| {name} | {b['n_scored']} | {b['n_ok']} | {_pct(b.get('precision'))} | "
            f"{_ci(b.get('wilson_ci'))} | {_pct(b.get('weighted_precision'))} | {_ci(b.get('weighted_ci'))} |")


def to_markdown(rep: dict) -> str:
    L: list[str] = []
    mode = rep["mode"]
    it = rep["integrity"]
    L.append(f"# Kết quả Khối C — {'PILOT 50 ô (C-1)' if mode == 'pilot' else 'mẻ chấm mù (C-2/C-3)'} · {rep['generated_at'][:10]}")
    L.append("")
    L.append(f"Mẻ `{rep['batch']}` · verdict: " + ", ".join(Path(f).name for f in rep["verdict_input"]["files"])
             + f" ({rep['verdict_input']['n_lines']} dòng, trùng id {rep['verdict_input']['n_dup_item']}, "
               f"bỏ nguồn máy {rep['verdict_input']['n_skipped_ai']}).")
    L.append("")
    L.append("## 0. Toàn vẹn")
    L.append("")
    L.append("| Kiểm | Kết quả |")
    L.append("|---|---|")
    L.append(f"| sha256(KHOA) == manifest.batch_sha256 | {'ĐÚNG' if it.get('batch_sha256_ok') else 'SAI'} (`{it.get('batch_sha256', '')[:16]}…`) |")
    ok_l = it.get("labels_sha256_ok")
    L.append(f"| sha256(labels_final.csv) == lúc dựng mẻ | {'ĐÚNG' if ok_l else ('SAI' if ok_l is False else 'không đối chiếu được')} "
             f"(`{(it.get('labels_sha256_current') or '')[:16]}…`) |")
    L.append(f"| item_id verdict ⊆ khoá | {it.get('n_unknown_id', 0)} id lạ / {it.get('n_verdict', 0)} verdict |")
    L.append(f"| has_code (HTML) == has_q2 (khoá) | {it.get('n_has_code_mismatch', 0)} lệch |")
    L.append(f"| ô đã chấm / đủ cả hai câu / tổng khoá | {it.get('n_graded', 0)} / {it.get('n_complete', 0)} / {it.get('n_khoa', 0)} |")
    L.append(f"| nguồn verdict | {it.get('sources', {})} |")
    L.append("")
    if rep.get("alarms"):
        L.append("**BÁO ĐỘNG:**")
        L.append("")
        for a in rep["alarms"]:
            L.append(f"- {a}")
        L.append("")
    else:
        L.append("Không có báo động.")
        L.append("")
    if "decoys" not in rep:
        return "\n".join(L) + "\n"

    L.append(f"Q1 dùng để ước lượng: **{rep.get('q1_source', 'q1')}**.")
    L.append("")
    d = rep["decoys"]
    L.append("## 1. Ô mồi (người chấm có nhìn thật không)")
    L.append("")
    L.append("| Mồi | n | đạt | độ chính xác | Wilson 95 % | Q1 đạt | Q2 đạt |")
    L.append("|---|---:|---:|---:|---|---:|---:|")
    for nm, key in (("dương (QĐ-01: đạt ⇔ Q1 = dung; Q2 chỉ đếm)", "pos"), ("âm (âm/mã ô kề: đáp án sai_am∨sai_crop / sai)", "neg")):
        b = d[key]
        if b.get("n"):
            q2p = (f"{b.get('q2_dung_nhat_quan_qd01', '—')} nhất quán / {b.get('q2_sai_trai_qd01', '—')} trái QĐ-01"
                   if key == "pos" else str(b["q2_pass"]))
            L.append(f"| {nm} | {b['n']} | {b['n_pass']} | {_pct(b['accuracy'])} | {_ci(b['wilson_ci'])} | {b['q1_pass']} | {q2p} |")
    L.append(f"| **tổng** | {d['all']['n']} | {d['all']['n_pass']} | {_pct(d['all']['accuracy'])} | | | |")
    L.append("")
    L.append(f"Ngưỡng báo động {DECOY_ALARM:.0%} → {'**BÁO ĐỘNG**' if d['alarm'] else 'đạt'}. "
             "Mồi dương: mã 𠊚 là phán quyết QĐ-01 mà người chấm nhận ra khi lộ mã → Q2 không phải phép đo.")
    L.append("")

    r = rep["repeats"]
    L.append("## 2. Lặp ẩn — κ nội tại")
    L.append("")
    L.append(f"{r['n_pairs']} cặp ghép được (mẻ có {r['n_repeat_in_batch']} ô lặp, phân tầng về tầng dễ khác 'dung').")
    L.append("")
    L.append("| Câu | n cặp | κ Cohen | KTC 95 % bootstrap | đồng thuận thô | kỳ vọng | ma trận (gốc→lặp, ô ≠ 0) | diễn giải |")
    L.append("|---|---:|---:|---|---:|---:|---|---|")
    for nm, key in (("Q1 (4 mức)", "q1"), ("Q1 nhị phân đúng/khác", "q1_binary"), ("Q2 (3 mức)", "q2")):
        b = r.get(key) or {}
        if b.get("n"):
            cb = b.get("ci_boot")
            L.append(f"| {nm} | {b['n']} | {b['kappa']:.3f} | {('[%.2f; %.2f]' % cb) if cb else '—'} | "
                     f"{_pct(b['observed_agreement'])} | {_pct(b['expected_agreement'])} | "
                     f"`{b['matrix']}` | {b.get('band') or ''} |")
    L.append("")
    if r.get("q1_agree_wilson"):
        L.append(f"Đồng thuận thô Q1 Wilson 95 %: {_ci(r['q1_agree_wilson'])}.")
    if r.get("by_lap_tang"):
        L.append("Theo tầng gốc: " + "; ".join(f"{t}: n={b['n']}, κ={b['kappa'] if b['kappa'] is None else round(b['kappa'], 2)}, "
                                              f"đồng thuận {_pct(b['observed_agreement'], 0)}, gốc≠dung {b['n_non_dung_orig']}"
                                              for t, b in r["by_lap_tang"].items()))
    L.append("")
    L.append("κ nhạy với tỷ lệ nền — trích cùng đồng thuận thô, KTC bootstrap và ma trận (KE_HOACH_CHAM_TAY C.0.b).")
    L.append("")

    w = rep["dwell"]
    L.append("## 3. Dwell (thời gian trả lời một ô)")
    L.append("")
    if w.get("n_with_dwell"):
        L.append(f"- n = {w['n_with_dwell']} ô có dwell · p10 **{w['p10_s']:.1f} s** · p50 **{w['p50_s']:.1f} s** · p90 **{w['p90_s']:.1f} s** · "
                 f"trung bình {w['mean_s']:.1f} s · tổng {w['total_min']:.0f} phút")
        L.append(f"- ô < {w['flag_ms'] / 1000:.1f} s gắn cờ: **{w['n_flag_fast']}** ({_pct(w['frac_flag_fast'])}) · xem lại ≥ 2 lần: {w['n_revisited']} · đổi ý: {w['n_changed_mind']}")
        if "n_q1_changed_after_reveal" in w:
            L.append(f"- Q1 đổi SAU khi lộ mã: **{w['n_q1_changed_after_reveal']}** ô (q1 cuối ≠ q1 mù: {w.get('n_q1_blind_ne_final', '—')}); "
                     f"dwell pha Q1 p50 {w.get('q1_p50_s', float('nan')):.1f} s · p90 {w.get('q1_p90_s', float('nan')):.1f} s")
        for s in w.get("per_session", []):
            L.append(f"  - {s['session_id']}: n={s['n']}, p50 {s['p50_s']:.1f} s, {s['total_min']:.0f} phút, cờ nhanh {s['n_flag_fast']}")
    else:
        L.append("- không có dwell (verdict chưa đủ hai câu?)")
    L.append("")
    rc = rep["raw_counts"]
    L.append(f"Đếm thô: Q1 {rc['q1']} · Q2 {rc['q2']} · theo tầng {rc['by_tang']}.")
    L.append("")
    if mode == "pilot":
        L.append("Pilot chỉ đo dwell/κ/mồi — KHÔNG trích số precision từ đây.")
        return "\n".join(L) + "\n"

    p = rep["precision"]
    L.append("## 4. Precision theo tier_v3 (mẻ chính 600 ô, Horvitz–Thompson với N_h, FPC)")
    L.append("")
    L.append("Wilson = trên mẫu gộp (không trọng số); HT = suy ra dân số tier. `khong_ro` loại khỏi mẫu số.")
    L.append("")
    L.append("| tier · câu | n chấm | đúng | precision mẫu | Wilson 95 % | HT dân số | CI 95 % (HT) |")
    L.append("|---|---:|---:|---:|---|---:|---|")
    for tier, t in p["tiers"].items():
        L.append(_ht_row(f"**{tier}** · Q1 (crop+âm)", t["q1"]))
        if "q2" in t:
            L.append(_ht_row(f"{tier} · Q2 (mã)" + (" ⚠" if "q2_note" in t else ""), t["q2"]))
            L.append(_ht_row(f"{tier} · Q1∧Q2", t["q1_and_q2"]))
    for nm, b in p["pooled"].items():
        L.append(_ht_row(f"**{nm}** = {'+'.join(b['tiers'])} · Q1", b["q1"]))
        if "q2" in b:
            L.append(_ht_row(f"{nm} · Q2", b["q2"]))
            L.append(_ht_row(f"{nm} · Q1∧Q2", b["q1_and_q2"]))
        if "q2_nhat_quan_qd01" in b:
            L.append(_ht_row(f"{nm} · Q2 nhất quán QĐ-01 (không phải precision)", b["q2_nhat_quan_qd01"]))
    L.append("")
    for nm, b in p["pooled"].items():
        if "note" in b:
            L.append(f"- {nm}: {b['note']}")
    L.append("")
    for tier, t in p["tiers"].items():
        note = f" — {t['q2_note']}" if "q2_note" in t else ""
        L.append(f"- {tier}: Q1 {t['q1_errors']}" + (f"; Q2 {t.get('q2_errors')}" if "q2_errors" in t else "") + note)
    L.append("")
    L.append("### 4b. Theo tầng (tier_v3 × lớp cột × box_source), Q1")
    L.append("")
    L.append("| tầng | N_h | n | đúng | precision | Wilson 95 % | sai_crop | sai_am | khong_ro | Q2 đúng/n |")
    L.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---|")
    for s in p["strata"]:
        q2s = f"{s['q2_ok']}/{s['q2_n']}" if "q2_n" in s else "—"
        L.append(f"| {s['stratum']} | {s['N_h']} | {s['n_scored']} | {s['q1_ok']} | {_pct(s['q1_precision'])} | "
                 f"{_ci(s['q1_wilson'])} | {s['sai_crop']} | {s['sai_am']} | {s['khong_ro']} | {q2s} |")
    L.append("")

    t1 = rep["t1_gate"]
    L.append("## 5. T1 — cổng thị giác B-3 (REVIEW→SYL, gate_09)")
    L.append("")
    if t1.get("n_scored"):
        L.append(f"- Q1 đúng **{t1['n_ok']}/{t1['n_scored']}** ({_pct(1 - t1['err_rate'])}; Wilson {_ci(t1['ok_wilson_ci'])}), "
                 f"KHÔNG RÕ {t1['n_khong_ro']}; Q1 {t1['q1_counts']}")
        L.append(f"- lỗi {t1['n_err']}/{t1['n_scored']} → cận trên Clopper–Pearson một phía 95 % = **{_pct(t1['err_upper_cp95'], 2)}** "
                 f"(bảo thủ, KHÔNG RÕ = lỗi: {_pct(t1['err_upper_cp95_conservative'], 2)})")
        L.append(f"- HT theo sách: {_pct(t1.get('weighted_precision'))} {_ci(t1.get('weighted_ci'))}; "
                 + "; ".join(f"{b['stratum']} {b['ok']}/{b['n']}" for b in t1["per_book"]))
        L.append(f"- **{t1['decision']}**")
    else:
        L.append("- chưa có ô T1 nào được chấm")
    L.append("")

    t2 = rep["t2_chains"]
    L.append("## 6. T2 — chuỗi thanh ghi trượt (k5 thr0.8 len3)")
    L.append("")
    if t2.get("n_scored"):
        L.append(f"- {t2['n_audited']} ô / {t2['n_chains']} chuỗi; Q1 {t2['q1_counts']}; tỉ lệ **sai_am {_pct(t2['sai_am_rate'])}** "
                 f"Wilson {_ci(t2['sai_am_wilson'])}; ghi âm đúng {t2['n_sai_am_wrote']} ô, trùng argmax thị giác {t2['n_sai_am_matches_argmax']}")
        L.append(f"- chuỗi có ≥ {t2['demote_frac_threshold']:.0%} ô sai_am → đề xuất hạ REVIEW theo chuỗi: "
                 f"**{t2['n_chains_demote']}/{t2['n_chains']}** chuỗi ({t2['n_cells_in_demote_chains']} ô)")
        L.append(f"- theo tier_v3: {t2['by_tier_v3']}; ô QĐ-01 trong chuỗi: {t2['qd01_cells']}")
        L.append("")
        L.append("| chuỗi | n | dung | sai_am | sai_crop | tỉ lệ sai_am | hạ? | QĐ-01 |")
        L.append("|---|---:|---:|---:|---:|---:|:---:|---:|")
        for c in t2["chains"]:
            L.append(f"| `{c['chain_id']}` | {c['n']} | {c['n_dung']} | {c['n_sai_am']} | {c['n_sai_crop']} | "
                     f"{_pct(c['frac_sai_am'], 0)} | {'CÓ' if c['demote'] else ''} | {c['n_qd01']} |")
    else:
        L.append("- chưa có ô T2 nào được chấm")
    L.append("")

    t3 = rep["t3"]
    L.append("## 7. T3 — B-2 đổi âm (19 ô) · B-5 hộp 3 nhánh lệch (30/117)")
    L.append("")
    if "b2" in t3:
        L.append(f"- B-2: {t3['b2']['counts']}")
        L.append("")
        L.append("| khoá | âm cũ (hiển thị) | âm mới (thị giác) | Q1 | ghi | Q2 | kết luận |")
        L.append("|---|---|---|---|---|---|---|")
        for c in t3["b2"]["cells"]:
            L.append(f"| `{c['key']}` | {c['am_cu']} | {c['am_moi']} | {c['q1']} | {c['q1_am']} | {c['q2'] or '—'} | {c['ket_luan']} |")
        L.append("")
    if "b5" in t3:
        b = t3["b5"]
        L.append(f"- B-5: hộp khoá QĐ-01 đúng crop **{_pct(b.get('hop_khoa_dung'))}** Wilson {_ci(b.get('wilson_ci'))} "
                 f"(HT ra 117 ô: {_pct(b.get('weighted_precision'))} {_ci(b.get('weighted_ci'))}); Q1 {b['q1_counts']}; Q2 {b['q2_counts']}. {b['note']}")
    L.append("")
    t6 = rep.get("t6_nguoi", {})
    if t6.get("n_scored"):
        L.append("## 7b. T6 — 'người' chưa khoá (ngoài QĐ-01)")
        L.append("")
        L.append(f"- {t6['n_audited']} ô / dân số {t6.get('population_N')}: Q1 đúng **{t6['n_ok']}/{t6['n_scored']}** "
                 f"({_pct(t6.get('precision'))}; Wilson {_ci(t6.get('wilson_ci'))}; HT {_pct(t6.get('weighted_precision'))} "
                 f"{_ci(t6.get('weighted_ci'))}); Q1 {t6['q1_counts']}; theo tier_v3 {t6['by_tier_v3']}")
        if t6.get("sai_am_wrote"):
            L.append(f"- âm người chấm ghi cho ô sai_am: {t6['sai_am_wrote']}")
        L.append("")
    q = rep["qd01"]
    if q.get("n"):
        L.append(f"## 8. Ô QĐ-01 nhìn lại mù ({q['n']} ô: {q['by_tang']})")
        L.append("")
        L.append(f"- Q1 dung {q['q1_dung']}, sai_crop {q['q1_sai_crop']}, sai_am {q['q1_sai_am']}; "
                 f"Q2 mã 𠊚 đúng **{q['q2_dung']}/{q['q2_n']}** Wilson {_ci(q['q2_wilson'])}")
        L.append("")
    L.append("---")
    L.append("Sinh bởi `pipeline/ground_truth/estimate_khoi_c.py`. Số ở đây CHƯA vào BANG_SO_LIEU — "
             "chỉ đưa vào qua `evidence()` (A-13) sau khi người duyệt.")
    return "\n".join(L) + "\n"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.estimate_khoi_c")
    ap.add_argument("--batch", default=str(DEFAULT_BATCH))
    ap.add_argument("--verdicts", nargs="*", help="tệp/thư mục verdict; mặc định theo plan.json trong --batch")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--pilot", action="store_true", help="phiên pilot 50 ô: báo cáo ngắn (mồi/κ/dwell)")
    ap.add_argument("--simulate", metavar="DIR", help="GHI verdict giả lập vào DIR rồi thoát (để kiểm)")
    ap.add_argument("--sim-seed", type=int, default=1)
    ap.add_argument("--sim-noise", type=float, default=0.05)
    ap.add_argument("--out-md", help="mặc định docs/KET_QUA_KHOI_C_<ngày>.md (pilot: <batch>/pilot_report.md)")
    ap.add_argument("--out-json")
    ap.add_argument("--include-ai-verdicts", action="store_true")
    ap.add_argument("--conf", type=float, default=0.95)
    args = ap.parse_args(argv)

    batch = Path(args.batch)
    if args.simulate:
        khoa = batch / "_khoa" / ("KHOA_pilot.jsonl" if args.pilot else "KHOA.jsonl")
        paths = write_simulated(khoa, Path(args.simulate), args.sim_seed, args.sim_noise)
        print(f"[sim] {len(paths)} tệp verdict GIẢ LẬP (source=simulated) -> {args.simulate}")
        return 0

    plan_name = "plan_pilot.json" if args.pilot else "plan.json"
    plan = json.loads((batch / plan_name).read_text(encoding="utf-8"))
    files = verdict_files(batch, plan, args.verdicts)
    if not files:
        raise SystemExit(f"không có tệp verdict nào (mong {[s['verdict_file'] for s in plan['sessions']]} trong {batch})")
    rep = build(batch, files, Path(args.labels), pilot=args.pilot, conf=args.conf,
                include_ai=args.include_ai_verdicts)
    if rep["verdict_input"]["n_skipped_ai"]:
        print(f"[cảnh báo] đã LOẠI {rep['verdict_input']['n_skipped_ai']} verdict nguồn máy")

    today = _dt.date.today().isoformat()
    if args.pilot:
        out_md = Path(args.out_md) if args.out_md else batch / "pilot_report.md"
        out_json = Path(args.out_json) if args.out_json else batch / "pilot_report.json"
    else:
        out_md = Path(args.out_md) if args.out_md else DOCS / f"KET_QUA_KHOI_C_{today}.md"
        out_json = Path(args.out_json) if args.out_json else DOCS / f"KET_QUA_KHOI_C_{today}.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(rep), ensure_ascii=False, indent=1), encoding="utf-8")
    out_md.write_text(to_markdown(rep), encoding="utf-8")

    it = rep["integrity"]
    print(f"[C3] toàn vẹn: batch_sha {it.get('batch_sha256_ok')} · labels_sha {it.get('labels_sha256_ok')} · "
          f"id lạ {it.get('n_unknown_id')} · chấm {it.get('n_graded')}/{it.get('n_khoa')}")
    if "decoys" in rep:
        d = rep["decoys"]
        print(f"[C3] mồi: dương {d['pos'].get('n_pass')}/{d['pos'].get('n')} · âm {d['neg'].get('n_pass')}/{d['neg'].get('n')}"
              f"{' — BÁO ĐỘNG' if d['alarm'] else ''}")
        r1 = rep["repeats"].get("q1", {})
        print(f"[C3] κ Q1 {r1.get('kappa')} (n={r1.get('n')}) · Q2 {rep['repeats'].get('q2', {}).get('kappa')}"
              f" · dwell p50 {rep['dwell'].get('p50_s')} s, cờ nhanh {rep['dwell'].get('n_flag_fast')}")
    if "precision" in rep:
        for tier, t in rep["precision"]["tiers"].items():
            q = t["q1"]
            print(f"[C3] {tier:<7} Q1 {q.get('n_ok')}/{q.get('n_scored')} HT {q.get('weighted_precision')}"
                  + (f" · Q2 {t['q2'].get('n_ok')}/{t['q2'].get('n_scored')}" if "q2" in t else ""))
        g = rep["precision"]["pooled"].get("GOLD", {})
        if g:
            print(f"[C3] GOLD   Q1 HT {g['q1'].get('weighted_precision')} {g['q1'].get('weighted_ci')} · "
                  f"Q1∧Q2 HT {g['q1_and_q2'].get('weighted_precision')}")
        print(f"[C3] T1: {rep['t1_gate'].get('decision')}")
    for a in rep.get("alarms", []):
        print(f"[BÁO ĐỘNG] {a}")
    print(f"[C3] -> {out_md}  {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
