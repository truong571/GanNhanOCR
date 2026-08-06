"""Self-test for Giai đoạn 1 remediation (no pytest).

Run:  .venv/bin/python -m pipeline.remediation.selftest
Uses a hand-built synthetic frame with known defects (so every count is checked exactly)
and then runs the whole remediation on the real labels.csv, asserting the invariants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import census as census_mod
from . import remediate as remediate_mod

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "dataset_out" / "labels.csv"

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


def _row(**kw):
    base = dict(image="", book="b", page="p", column=1, ocr_char="X", syllable="a",
                label="X", unicode="U+0058", label_level="char", tier="GOLD",
                rule="s1_inter_s2_direct", s3_cosine="", ink_pct=0.1, crop_w=100,
                crop_h=100, image_md5="", seg_flag="ok", split="train",
                split_group="b|p|c1", bbox="[0, 0, 10, 10]")
    base.update(kw)
    return base


def _synthetic() -> pd.DataFrame:
    rows = []
    # --- F1 cross-column, CONFLICTING labels (same md5, 2 columns, diff label) ---
    rows.append(_row(image="gold/a_c01_0.png", column=1, label="城", unicode="U+57CE",
                     image_md5="dupF1conflict", split="train", bbox="[10,10,20,20]"))
    rows.append(_row(image="gold/a_c02_0.png", column=2, label="㝵", unicode="U+3775",
                     image_md5="dupF1conflict", split="test", bbox="[10,10,20,20]"))
    # --- F1 cross-column, SAME label (pure duplicate, keep one) ---
    rows.append(_row(image="gold/b_c01_0.png", column=1, label="月", unicode="U+6708",
                     image_md5="dupF1same", split="train", bbox="[30,30,40,40]"))
    rows.append(_row(image="gold/b_c02_0.png", column=2, label="月", unicode="U+6708",
                     image_md5="dupF1same", split="val", bbox="[30,30,40,40]"))
    # --- AE-1 same-column dup-bbox, CONFLICTING ---
    rows.append(_row(image="gold/c_c01_0.png", column=1, label="未", unicode="U+672A",
                     image_md5="dupAE1", split="train", bbox="[50,50,60,60]"))
    rows.append(_row(image="gold/c_c01_1.png", column=1, label="末", unicode="U+672B",
                     image_md5="dupAE1", split="train", bbox="[50,50,60,60]"))
    # --- similar-bridge GOLD, LOW cosine (should demote) ---
    rows.append(_row(image="gold/d_c01_0.png", column=1, label="連", unicode="U+9023",
                     rule="s1_inter_s2_similar", s3_cosine=0.40, image_md5="simlow",
                     bbox="[70,70,80,80]"))
    # --- similar-bridge GOLD, HIGH cosine (should stay GOLD) ---
    rows.append(_row(image="gold/e_c01_0.png", column=1, label="建", unicode="U+5EFA",
                     rule="s1_inter_s2_similar", s3_cosine=0.90, image_md5="simhigh",
                     bbox="[90,90,100,100]"))
    # --- clean GOLD rows (untouched) ---
    for i in range(5):
        rows.append(_row(image=f"gold/clean_{i}.png", column=3, label="德",
                         unicode="U+5FB7", image_md5=f"clean{i}", split="train",
                         bbox=f"[{100+i},0,{110+i},10]"))
    # --- a REVIEW row (no image / md5) ---
    rows.append(_row(image="", tier="REVIEW", rule="below_visual_threshold",
                     label="", unicode="", image_md5="", s3_cosine=0.1))
    return pd.DataFrame(rows)


def test_census_synthetic() -> None:
    print("[census synthetic]")
    df = _synthetic()
    res = census_mod.run_census(df)
    check("dup_bbox rows == 2 (AE-1)", res.dup_bbox_rows == 2, str(res.dup_bbox_rows))
    check("cross_col rows == 4 (2 F1 groups)", res.cross_col_rows == 4, str(res.cross_col_rows))
    check("union rows == 6", res.union_rows == 6, str(res.union_rows))
    check("union groups == 3", res.union_groups == 3, str(res.union_groups))
    check("conflicting groups == 2", res.conflicting_groups == 2, str(res.conflicting_groups))
    check("provably-wrong == 2", res.provably_wrong_rows == 2, str(res.provably_wrong_rows))


def test_remediate_synthetic() -> None:
    print("[remediate synthetic]")
    df = _synthetic()
    out, rep = remediate_mod.remediate(df)
    # input not mutated (13 GOLD in the synthetic frame)
    check("input frame not mutated", (df["tier"] == "GOLD").sum() == 13,
          str((df["tier"] == "GOLD").sum()))
    # conflicting groups fully quarantined: dupF1conflict(2) + dupAE1(2) = 4
    # same-label group keeps 1, quarantines 1: dupF1same -> 1 quarantined
    check("quarantined rows == 5", rep.quarantined_rows == 5, str(rep.quarantined_rows))
    check("quarantined conflict == 4", rep.quarantined_conflict == 4, str(rep.quarantined_conflict))
    check("quarantined duplicate == 1", rep.quarantined_duplicate == 1, str(rep.quarantined_duplicate))
    check("kept representatives == 1", rep.kept_representatives == 1, str(rep.kept_representatives))
    q = out[out["tier"] == "QUARANTINE"]
    check("QUARANTINE tier applied", len(q) == 5, str(len(q)))
    check("quarantine rule tagged", q["rule"].str.contains("quarantine_dup").all())
    # demote: 1 low-cosine similar-bridge -> REVIEW; high-cosine stays GOLD
    check("demoted 1 similar-bridge", rep.demoted_similar_lowcos == 1, str(rep.demoted_similar_lowcos))
    demoted = out[out["rule"].str.contains("demoted_lowcos", na=False)]
    check("demoted row now REVIEW", (demoted["tier"] == "REVIEW").all() and len(demoted) == 1)
    check("high-cosine similar stays GOLD",
          (out[out["image"] == "gold/e_c01_0.png"]["tier"] == "GOLD").all())
    # split invariant: original leak present (F1 conflict train/test + same train/val),
    # closed as a side-effect of quarantine (leak is a subset of F1), residual asserted 0
    check("no md5 spans >1 split after", rep.md5_spanning_splits_after == 0)
    check("original split leak detected >0", rep.md5_spanning_splits_original >= 1,
          str(rep.md5_spanning_splits_original))
    # the surviving dupF1same representative must have a single split
    surv = out[(out["image_md5"] == "dupF1same") & (out["tier"] == "GOLD")]
    check("dupF1same reduced to 1 usable row", len(surv) == 1, str(len(surv)))
    # clean rows untouched
    check("5 clean GOLD survive", (out["image"].str.startswith("gold/clean_")).sum() == 5)
    check("usable dropped by 6 (5 quarantine + 1 demote)",
          rep.usable_before - rep.usable_after == 6,
          f"{rep.usable_before}->{rep.usable_after}")


def test_real() -> None:
    if not LABELS.exists():
        print(f"[warn] {LABELS} missing — skipping real-data test")
        return
    print("[real labels.csv]")
    df = pd.read_csv(LABELS, dtype={"image_md5": str})
    res = census_mod.run_census(df)
    # Census ĐO trên labels.csv HIỆN TẠI (sau engine-fix + dedup upstream). Số lịch sử
    # (thế hệ labels.csv cũ, không còn trên đĩa) ở docs/census_history.md — bằng chứng
    # engine-fix hoạt động: dup_bbox 701->0, cross_col 1686->8, union 2321->8, PW 1177->4.
    # dup_bbox: đo 0 (lịch sử 701) — dedup upstream đã xoá mọi trùng-bbox cùng cột.
    check("real dup_bbox == 0 (dedup closed; hist 701)", res.dup_bbox_rows == 0,
          str(res.dup_bbox_rows))
    # cross_col: BẤT BIẾN == 0. Lịch sử 1686 -> 8 (labels.csv 21/07) -> 0 (22/07).
    check("real cross_col == 0 (dedup closed; hist 1686->8->0)", res.cross_col_rows == 0,
          str(res.cross_col_rows))
    # union: BẤT BIẾN == 0 = union(dup_bbox=0, cross_col=0). Lịch sử 2321 -> 8 -> 0.
    check("real union == 0 (dedup closed; hist 2321->8->0)", res.union_rows == 0,
          str(res.union_rows))
    # provably-wrong: BẤT BIẾN == 0 (1 nhãn sai / nhóm xung đột; 0 nhóm còn lại).
    # Lịch sử ~1177 -> 4 -> 0.
    check("real provably-wrong == 0 (dedup closed; hist 1177->4->0)",
          res.provably_wrong_rows == 0, str(res.provably_wrong_rows))
    # CẤU TRÚC (không phụ thuộc thế hệ dữ liệu): union = |dup_bbox ∪ cross_col|.
    check("real union là hợp của 2 lớp con",
          max(res.dup_bbox_rows, res.cross_col_rows) <= res.union_rows
          <= res.dup_bbox_rows + res.cross_col_rows,
          f"union={res.union_rows} dup_bbox={res.dup_bbox_rows} cross={res.cross_col_rows}")
    out, rep = remediate_mod.remediate(df)
    check("real: outputs same #rows", len(out) == len(df))
    # BẤT BIẾN cốt lõi (giữ nguyên): không md5 nào span >1 split sau remediation.
    check("real: invariant md5 splits == 0", rep.md5_spanning_splits_after == 0)
    # Split-leak GỐC: đo 0 (lịch sử 288). Dedup upstream đã đóng lớp trùng md5 nên
    # labels.csv HIỆN TẠI vào remediation ĐÃ không còn md5 nào span >1 split — bất biến
    # giữ đúng từ đầu vào tới đầu ra (0 -> 0), không còn leak để vá tại bước này.
    check("real: original split leak == 0 (dedup closed; hist 288)",
          rep.md5_spanning_splits_original == 0, str(rep.md5_spanning_splits_original))
    # Demote similar-bridge: kiểm CẤU TRÚC, không phải số cứng — quarantine chạy TRƯỚC
    # nên hàng đã bị cách ly không còn tier GOLD và không bị demote nữa. Kỳ vọng =
    # |GOLD ∩ similar_bridge ∩ s3<τ| trừ đi phần đã bị quarantine cướp.
    # Lịch sử: 748 (khi quarantine lấy ~72 hàng) -> 925 (quarantine = 0).
    s3_in = pd.to_numeric(df["s3_cosine"], errors="coerce")
    cand_demote = (
        df["tier"].eq("GOLD")
        & df["rule"].eq(remediate_mod.SIMILAR_RULE)
        & s3_in.notna()
        & (s3_in < remediate_mod.TAU_SILVER)
    )
    q_mask = out["tier"].eq(remediate_mod.QUARANTINE_TIER)
    expect_demote = int((cand_demote & ~q_mask).sum())
    check("real: demoted similar-bridge == |GOLD∩bridge∩s3<τ| \\ quarantine",
          rep.demoted_similar_lowcos == expect_demote,
          f"{rep.demoted_similar_lowcos} vs expect {expect_demote}")
    # quarantined: BẤT BIẾN == 0. Lịch sử >1000 (~2321 hàng trùng) -> 8 -> 0. Không còn
    # hàng trùng md5/bbox nào để cách ly.
    check("real: quarantined == 0 (dedup closed; hist >1000->8->0)",
          rep.quarantined_rows == 0 and rep.quarantined_conflict == 0
          and rep.quarantined_duplicate == 0,
          f"rows={rep.quarantined_rows} conflict={rep.quarantined_conflict} "
          f"dup={rep.quarantined_duplicate}")
    # CẤU TRÚC: quarantine chỉ rút từ lớp trùng, và rows = conflict + duplicate thuần.
    check("real: quarantine ⊆ lớp trùng, rows = conflict + dup",
          rep.quarantined_rows <= res.union_rows
          and rep.quarantined_rows == rep.quarantined_conflict + rep.quarantined_duplicate,
          f"rows={rep.quarantined_rows} union={res.union_rows}")
    check("real: usable decreased", rep.usable_after < rep.usable_before,
          f"{rep.usable_before}->{rep.usable_after}")
    # idempotence: re-running remediation changes nothing further
    out2, rep2 = remediate_mod.remediate(out)
    check("real: idempotent (0 new quarantine)", rep2.quarantined_rows == 0,
          str(rep2.quarantined_rows))
    check("real: idempotent (0 new demote)", rep2.demoted_similar_lowcos == 0,
          str(rep2.demoted_similar_lowcos))
    print(f"       ({rep.summary()})")


def main() -> int:
    print("=" * 64)
    print("REMEDIATION SELFTEST")
    print("=" * 64)
    test_census_synthetic()
    test_remediate_synthetic()
    test_real()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
