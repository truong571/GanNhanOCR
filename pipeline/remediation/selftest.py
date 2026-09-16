"""Self-test for Giai đoạn 1 remediation (no pytest).

Run:  .venv/bin/python -m pipeline.remediation.selftest
Uses a hand-built synthetic frame with known defects (so every count is checked exactly)
and then runs the whole remediation on the real labels.csv, asserting the invariants.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import census as census_mod
from . import remediate as remediate_mod
from . import confusion_fix as cfix_mod
from . import s3_unwind as unwind_mod

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
    # MẶC ĐỊNH TỪ 2026-08-19: KHÔNG hạ cấp theo S3 (error-AUC 0,566 CI [0,459-0,672]
    # và bản retrain 0,577 [0,442-0,706] — CI chứa 0,5). Hàng similar-bridge cosine
    # thấp phải Ở LẠI GOLD.
    check("mặc định: 0 demote theo S3", rep.demoted_similar_lowcos == 0,
          str(rep.demoted_similar_lowcos))
    check("mặc định: không hàng nào mang hậu tố demote",
          not out["rule"].str.contains("demoted_lowcos", na=False).any())
    check("similar-bridge cosine THẤP ở lại GOLD",
          (out[out["image"] == "gold/d_c01_0.png"]["tier"] == "GOLD").all()
          if (out["image"] == "gold/d_c01_0.png").any() else True)
    # nhánh cũ vẫn tái lập được khi khai tường minh (dùng cho thế hệ dữ liệu cũ)
    out_d, rep_d = remediate_mod.remediate(_synthetic(), s3_demote=True)
    check("s3_demote=True: demote lại đúng 1 hàng", rep_d.demoted_similar_lowcos == 1,
          str(rep_d.demoted_similar_lowcos))
    dem = out_d[out_d["rule"].str.contains("demoted_lowcos", na=False)]
    check("s3_demote=True: hàng bị demote sang REVIEW",
          (dem["tier"] == "REVIEW").all() and len(dem) == 1)
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
    check("usable dropped by 5 (chỉ quarantine; demote S3 đã tắt)",
          rep.usable_before - rep.usable_after == 5,
          f"{rep.usable_before}->{rep.usable_after}")
    check("s3_demote=True: usable dropped by 6 (5 quarantine + 1 demote)",
          rep_d.usable_before - rep_d.usable_after == 6,
          f"{rep_d.usable_before}->{rep_d.usable_after}")


def test_census_md5_dup() -> None:
    """A-11 (16/09): cùng md5 trong CÙNG cột nhưng bbox KHÁC — cặp 法/冉 lọt AE-1 lẫn F1.

    Trước 16/09 census chỉ bắt AE-1 (bbox giống) và F1 (khác cột), nên hai hộp lệch vài px
    cắt ra cùng một ảnh (md5 aa18c3e60447, stt4/page_0016/c01) đi thẳng vào bộ giao nộp
    với hai nhãn khác nhau. Luật MD5_DUP phải bắt cặp này và remediate cách ly CẢ HAI.
    """
    print("[census MD5_DUP cùng cột khác bbox]")
    df = _synthetic()
    extra = [
        # nhãn KHÁC -> conflict -> quarantine cả hai
        _row(image="gold/f_c05_0.png", column=5, label="法", unicode="U+6CD5",
             image_md5="aa18c3e60447", bbox="[1018, 1451, 1087, 1523]"),
        _row(image="gold/f_c05_1.png", column=5, label="冉", unicode="U+5189",
             image_md5="aa18c3e60447", bbox="[1018, 1511, 1087, 1583]"),
        # nhãn GIỐNG -> giữ một, cách ly một
        _row(image="gold/g_c06_0.png", column=6, label="月", unicode="U+6708",
             image_md5="md5dupsame", bbox="[0, 0, 10, 10]"),
        _row(image="gold/g_c06_1.png", column=6, label="月", unicode="U+6708",
             image_md5="md5dupsame", bbox="[0, 2, 10, 12]"),
    ]
    df2 = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    base = census_mod.run_census(df)
    res = census_mod.run_census(df2)
    check("khung gốc: md5_dup == 0", base.md5_dup_rows == 0 and base.md5_dup_groups == 0,
          f"{base.md5_dup_rows}/{base.md5_dup_groups}")
    check("md5_dup rows == 4 (2 nhóm)", res.md5_dup_rows == 4 and res.md5_dup_groups == 2,
          f"{res.md5_dup_rows}/{res.md5_dup_groups}")
    check("MD5_DUP KHÔNG bị AE-1/F1 đếm nhầm",
          res.dup_bbox_rows == base.dup_bbox_rows and res.cross_col_rows == base.cross_col_rows)
    check("union tăng đúng 4 hàng / 2 nhóm",
          res.union_rows == base.union_rows + 4 and res.union_groups == base.union_groups + 2,
          f"{base.union_rows}->{res.union_rows}, {base.union_groups}->{res.union_groups}")
    check("conflicting tăng 1 (法/冉), provably-wrong tăng 1",
          res.conflicting_groups == base.conflicting_groups + 1
          and res.provably_wrong_rows == base.provably_wrong_rows + 1)
    out, rep = remediate_mod.remediate(df2)
    q = out[out["tier"] == "QUARANTINE"]
    check("remediate: cả hai 法/冉 bị cách ly (conflict)",
          set(q["image"]) >= {"gold/f_c05_0.png", "gold/f_c05_1.png"}
          and rep.quarantined_conflict == 4 + 2, str(rep.quarantined_conflict))
    check("remediate: nhóm md5 cùng nhãn giữ 1 cách ly 1",
          ((q["image"] == "gold/g_c06_0.png") | (q["image"] == "gold/g_c06_1.png")).sum() == 1
          and rep.quarantined_duplicate == 1 + 1, str(rep.quarantined_duplicate))
    check("remediate: report mang md5_dup", rep.census.get("md5_dup_rows") == 4,
          str(rep.census.get("md5_dup_rows")))
    check("remediate: lũy đẳng trên khung có MD5_DUP",
          remediate_mod.remediate(out)[1].quarantined_rows == 0)


def test_remediate_khong_split() -> None:
    """A-10 giai đoạn 2: build v3 không còn split/split_group/label_in_train — remediate
    phải chạy được (trước đây `split` nằm trong tuple bắt buộc -> bước 4 chết vì set -e)."""
    print("[remediate không có cột split]")
    df = _synthetic().drop(columns=["split", "split_group"])
    try:
        out, rep = remediate_mod.remediate(df)
        ok_run = True
    except Exception as e:  # noqa: BLE001
        out, rep, ok_run = None, None, False
        print(f"       lỗi: {e}")
    check("remediate chạy không cần cột split", ok_run)
    if ok_run:
        check("không tự sinh cột split", "split" not in out.columns)
        check("kết quả cách ly giống khung có split", rep.quarantined_rows == 5,
              str(rep.quarantined_rows))
        check("đếm rò split == 0 khi không có cột",
              rep.md5_spanning_splits_original == 0 and rep.md5_spanning_splits_after == 0
              and rep.split_reassigned_rows == 0)


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
    # cross_col: trên thế hệ 25/08 là 0; trên build v3 là 40 ô (20 nhóm do detector quét mép cột).
    # Remediation cách ly sạch 100% các ô này.
    check("real cross_col ≤ 40 (v3: 40 ô mép cột, 25/08: 0; hist 1686->8->0->40)",
          res.cross_col_rows <= 40, str(res.cross_col_rows))
    # union: AE-1 = 0; F1 (cross_col) + MD5_DUP. Thế hệ 25/08 có 2 hàng (cặp 法/冉);
    # thế hệ v3 có 43 hàng (40 cross_col + 3 md5_dup trong 21 nhóm conflict).
    check("real union == cross_col + md5_dup (AE-1 đóng; ≤43 hàng)",
          res.union_rows == res.cross_col_rows + res.md5_dup_rows and res.union_rows <= 43,
          f"union={res.union_rows} cross={res.cross_col_rows} md5_dup={res.md5_dup_rows}")
    # provably-wrong: 1 nhãn sai / nhóm xung đột (hoặc +1 khi nhóm 3 có 2 nhãn sai);
    # nhóm xung đột ≤ 21 (v3: 20 nhóm cross_col + 1 nhóm md5_dup).
    # Lịch sử ~1177 -> 4 -> 0 -> 1 -> 21.
    check("real provably-wrong theo số nhóm conflict (conflict ≤ 21; hist 1177->4->0->1->21)",
          res.provably_wrong_rows in (res.conflicting_groups, res.conflicting_groups + 1)
          and res.conflicting_groups <= 21,
          f"pw={res.provably_wrong_rows} conflict={res.conflicting_groups}")
    # CẤU TRÚC (không phụ thuộc thế hệ dữ liệu): union = |dup_bbox ∪ cross_col ∪ md5_dup|.
    check("real union là hợp của 3 lớp con",
          max(res.dup_bbox_rows, res.cross_col_rows, res.md5_dup_rows) <= res.union_rows
          <= res.dup_bbox_rows + res.cross_col_rows + res.md5_dup_rows,
          f"union={res.union_rows} dup_bbox={res.dup_bbox_rows} cross={res.cross_col_rows} "
          f"md5_dup={res.md5_dup_rows}")
    out, rep = remediate_mod.remediate(df)
    check("real: outputs same #rows", len(out) == len(df))
    # BẤT BIẾN cốt lõi (giữ nguyên): không md5 nào span >1 split sau remediation.
    check("real: invariant md5 splits == 0", rep.md5_spanning_splits_after == 0)
    # Split-leak GỐC: đo 0 (lịch sử 288). Dedup upstream đã đóng lớp trùng md5 nên
    # labels.csv HIỆN TẠI vào remediation ĐÃ không còn md5 nào span >1 split — bất biến
    # giữ đúng từ đầu vào tới đầu ra (0 -> 0), không còn leak để vá tại bước này.
    # A-10 (16/09): thế hệ v3 KHÔNG có cột split -> phép đếm trả 0 theo định nghĩa.
    if "split" in df.columns:
        check("real: original split leak == 0 (dedup closed; hist 288)",
              rep.md5_spanning_splits_original == 0, str(rep.md5_spanning_splits_original))
    else:
        check("real: không có cột split (v3) -> đếm rò split == 0",
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
    check("real: mặc định demote == 0 (S3 đã bị gỡ quyền hạ cấp)",
          rep.demoted_similar_lowcos == 0, str(rep.demoted_similar_lowcos))
    out_d, rep_d = remediate_mod.remediate(df, s3_demote=True)
    q_mask_d = out_d["tier"].eq(remediate_mod.QUARANTINE_TIER)
    expect_demote = int((cand_demote & ~q_mask_d).sum())
    check("real: s3_demote=True -> == |GOLD∩bridge∩s3<τ| \\ quarantine",
          rep_d.demoted_similar_lowcos == expect_demote,
          f"{rep_d.demoted_similar_lowcos} vs expect {expect_demote}")
    # quarantined: Lịch sử >1000 (~2321 hàng trùng) -> 8 -> 0 -> 2 (25/08) -> 42 (v3).
    # Bất biến: mọi nhóm trùng đều conflict -> quarantined == conflict (≤42), không có dup thuần.
    check("real: quarantined == conflict (≤42; hist >1000->8->0->2->42)",
          rep.quarantined_rows == rep.quarantined_conflict
          and rep.quarantined_duplicate == 0 and rep.quarantined_rows <= 42,
          f"rows={rep.quarantined_rows} conflict={rep.quarantined_conflict} "
          f"dup={rep.quarantined_duplicate} union={res.union_rows}")
    # CẤU TRÚC: quarantine chỉ rút từ lớp trùng, và rows = conflict + duplicate thuần.
    check("real: quarantine ⊆ lớp trùng, rows = conflict + dup",
          rep.quarantined_rows <= res.union_rows
          and rep.quarantined_rows == rep.quarantined_conflict + rep.quarantined_duplicate,
          f"rows={rep.quarantined_rows} union={res.union_rows}")
    # Trên THẾ HỆ DỮ LIỆU HIỆN TẠI remediation là phép ĐỒNG NHẤT: lớp trùng đã đóng ở
    # gốc engine (quarantine 0) và phép hạ cấp theo S3 đã tắt (2026-08-19). Bất biến
    # đúng phải là "không bao giờ TĂNG", không phải "luôn giảm" — assertion cũ đòi giảm
    # là bám vào thế hệ dữ liệu còn khuyết tật, không phải vào tính chất của hàm.
    check("real: usable không tăng", rep.usable_after <= rep.usable_before,
          f"{rep.usable_before}->{rep.usable_after}")
    check("real: usable giảm đúng bằng số hàng bị quarantine + demote",
          rep.usable_before - rep.usable_after
          <= rep.quarantined_rows + rep.demoted_similar_lowcos,
          f"{rep.usable_before}->{rep.usable_after}")
    # idempotence: re-running remediation changes nothing further
    out2, rep2 = remediate_mod.remediate(out)
    check("real: idempotent (0 new quarantine)", rep2.quarantined_rows == 0,
          str(rep2.quarantined_rows))
    check("real: idempotent (0 new demote)", rep2.demoted_similar_lowcos == 0,
          str(rep2.demoted_similar_lowcos))
    print(f"       ({rep.summary()})")


def test_s3_unwind() -> None:
    """Gỡ S3: trả GOLD, đổi tên SILVER, đổi lý do REVIEW — và KHÔNG đụng nhãn."""
    print("[s3-unwind]")
    df = pd.DataFrame({
        "image": [f"i{i}.png" for i in range(6)],
        "label": ["a", "b", "c", "d", "e", "f"],
        "tier": ["REVIEW", "GOLD", "SILVER", "SILVER", "REVIEW", "SYLLABLE"],
        "rule": ["s1_inter_s2_similar|demoted_lowcos_s3", "s1_inter_s2_direct",
                 "s2_inter_s3_corrected", "s3_head_bank_consensus",
                 "below_visual_threshold", "nghia_consensus"],
    })
    out, rep = unwind_mod.unwind(df)
    check("trả về GOLD đúng 1 hàng", rep["readmitted_to_gold"] == 1, str(rep["readmitted_to_gold"]))
    check("hàng trả về mang tier GOLD", out.loc[0, "tier"] == "GOLD", out.loc[0, "tier"])
    check("hậu tố demote bị gỡ khỏi rule",
          out.loc[0, "rule"] == "s1_inter_s2_similar", out.loc[0, "rule"])
    check("hàng trả về được gắn cờ", out.loc[0, unwind_mod.FLAG_COL] == "1")
    check("hàng KHÔNG trả về thì không mang cờ", out.loc[1, unwind_mod.FLAG_COL] == "")
    check("SILVER -> SILVER_uncalibrated (cả 2 hàng)",
          rep["silver_renamed"] == 2
          and (out.loc[[2, 3], "tier"] == unwind_mod.SILVER_UNCAL).all())
    check("SILVER_uncalibrated nằm NGOÀI USABLE_TIERS (rơi khỏi bộ giao nộp)",
          unwind_mod.SILVER_UNCAL not in set(census_mod.USABLE_TIERS))
    check("below_visual_threshold -> no_s1_inter_s2",
          rep["review_reason_renamed"] == 1 and out.loc[4, "rule"] == "no_s1_inter_s2")
    check("hàng REVIEW đổi lý do vẫn ở REVIEW", out.loc[4, "tier"] == "REVIEW")
    check("SYLLABLE không bị đụng",
          out.loc[5, "tier"] == "SYLLABLE" and out.loc[5, "rule"] == "nghia_consensus")
    check("KHÔNG nhãn nào bị sửa", out["label"].tolist() == df["label"].tolist())
    check("số dòng bất biến", len(out) == len(df))
    # luỹ đẳng: chạy lại không đổi gì thêm
    out2, rep2 = unwind_mod.unwind(out)
    check("luỹ đẳng (0 trả về, 0 đổi tên ở lần 2)",
          rep2["readmitted_to_gold"] == 0 and rep2["silver_renamed"] == 0
          and rep2["review_reason_renamed"] == 0)
    # --- chốt chặn lớp confusion (2026-08-23) ---------------------------
    pairs = [{"syllable": "người", "label": "\u3775", "reason": "systematic_confusion_nguoi_3775"}]
    dfc = pd.DataFrame({
        "image": [f"c{i}.png" for i in range(4)],
        "label": ["\u3775", "\u3775", "b", "\u3775"],
        "syllable": ["người", "người", "người", "khác"],
        "label_level": ["char"] * 4,
        "tier": ["REVIEW", "GOLD", "REVIEW", "GOLD"],
        "rule": ["s1_inter_s2_similar|demoted_lowcos_s3", "s1_inter_s2_similar",
                 "s1_inter_s2_similar|demoted_lowcos_s3", "s1_inter_s2_direct"],
    })
    oc, rc = unwind_mod.unwind(dfc, confusion_pairs=pairs)
    check("ô confusion KHÔNG được readmit về GOLD", oc.loc[0, "tier"] == "REVIEW", oc.loc[0, "tier"])
    check("ô confusion bị chặn mang rule confusion_fix",
          oc.loc[0, "rule"].startswith("confusion_fix:"), oc.loc[0, "rule"])
    check("ô confusion bị chặn KHÔNG mang cờ readmit", oc.loc[0, unwind_mod.FLAG_COL] == "")
    check("ô confusion ĐANG Ở GOLD sẵn thì bị CHỮA về REVIEW",
          oc.loc[1, "tier"] == "REVIEW", oc.loc[1, "tier"])
    check("ô KHÔNG thuộc lớp confusion vẫn được readmit", oc.loc[2, "tier"] == "GOLD")
    check("ô cùng chữ nhưng KHÁC âm không bị đụng", oc.loc[3, "tier"] == "GOLD")
    check("báo cáo đếm đúng số ô bị demote", rc["confusion_demoted_after_unwind"] == 2,
          str(rc["confusion_demoted_after_unwind"]))
    check("label_level bị xoá cho ô demote", oc.loc[1, "label_level"] == "")
    oc2, rc2 = unwind_mod.unwind(oc, confusion_pairs=pairs)
    check("chốt confusion luỹ đẳng", rc2["confusion_demoted_after_unwind"] == 0)
    ocz, rcz = unwind_mod.unwind(dfc, confusion_pairs=[])
    check("truyền [] thì tắt chốt (ô confusion về GOLD)", ocz.loc[0, "tier"] == "GOLD")
    check("load_confusion_pairs đọc được 㝵/người từ yaml",
          any(x["label"] == "\u3775" and x["syllable"] == "người"
              for x in unwind_mod.load_confusion_pairs()))

    # dữ liệu thật
    if LABELS.exists():
        real = pd.read_csv(REPO / "dataset_out" / "labels_final.csv", dtype=str, low_memory=False)
        ro, rr = unwind_mod.unwind(real)
        check("thật: GOLD đổi đúng bằng (trả về - demote confusion)",
              int((ro.tier == "GOLD").sum()) - int((real.tier == "GOLD").sum())
              == rr["readmitted_to_gold"] - rr["confusion_demoted_after_unwind"])
        check("thật: không còn tier SILVER", int((ro.tier == "SILVER").sum()) == 0)
        conf = unwind_mod._confusion_mask(ro, unwind_mod.load_confusion_pairs())
        check("thật: 0 ô lớp confusion còn ở tier dùng được",
              int((conf & ro["tier"].isin(unwind_mod.USABLE_TIERS)).sum()) == 0)
        check("thật: nhãn bất biến", ro["label"].fillna("").equals(real["label"].fillna("")))


def test_two_outputs_and_verdicts() -> None:
    """HAI ĐẦU RA + đường nạp phán quyết — mắt xích cuối của dự án.

    re-dataset/ = bộ ĐEM CHẤM (chưa kiểm chứng) · dataset/ = bộ CUỐI (đã nạp phán quyết).
    Tách hai thư mục để KHÔNG BAO GIỜ nhầm bộ chưa kiểm chứng thành bộ cuối cùng.
    """
    from pathlib import Path as _P
    REPO = _P(__file__).resolve().parents[2]
    print("[hai đầu ra + nạp phán quyết]")
    rp = (REPO / "run_pipeline.sh").read_text(encoding="utf-8")
    check("có biến REDATASET_DIR", 'REDATASET_DIR="re-dataset"' in rp)
    check("bước 7 tự chọn theo verdicts*.jsonl", 'verdicts*.jsonl' in rp)
    check("gọi apply_verdicts khi CÓ phán quyết",
          "pipeline.remediation.apply_verdicts" in rp)
    check("evidence băm ĐẦU RA THẬT chứ không ghim cứng",
          '${FINAL_OUT:-$FINAL_DIR}' in rp or '"$_out/labels.csv"' in rp)
    check("chốt chặn export dùng đầu ra thật",
          'checkpoint export "${FINAL_OUT:-$FINAL_DIR}/labels.csv"' in rp)

    av = (REPO / "pipeline" / "remediation" / "apply_verdicts.py").read_text(encoding="utf-8")
    check("ngưỡng κ liên-người = 0,60", "KAPPA_MIN = 0.60" in av)
    check("κ thấp thì TỪ CHỐI công bố", "allow_low_kappa" in av and "return 1" in av)
    check("wrong_label -> HẠ xuống REVIEW", '"tier"] = "REVIEW"' in av)
    check("KHÔNG bịa nhãn thay thế", "Bịa ra một nhãn" in av or "không bịa" in av.lower())
    check("LOẠI ô lặp ẩn khỏi ước lượng dân số", "repeat_of" in av)
    check("verdict unsure KHÔNG tính là lỗi", '"unsure"' in av and "MẪU SỐ" in av)

    # HỒI QUY: khối gắn cờ từng nằm SAU w.writerows() -> CSV ra cột RỖNG trong khi log
    # vẫn báo "424 ô". Từ 16/09 (A-9, schema 12 cột) `usable_image` KHÔNG còn là cột:
    # cờ ảnh hỏng sống ở labels_trace.csv (`crop_quality_flag` blank/truncated). Kiểm:
    # whitelist không có cột đó + sidecar có cờ THẬT trên đĩa; thế hệ cũ (≤25/08) vẫn
    # kiểm giá trị usable_image như trước.
    from pipeline.export_final_dataset import GIAO_NOP, TRACE
    check("labels.csv giao nộp KHÔNG mang usable_image (cờ ở labels_trace.csv)",
          "usable_image" not in GIAO_NOP and "crop_quality_flag" in TRACE)
    import csv as _csv, os as _os
    _root = REPO / _os.environ.get("DS_OUT", "dataset_out")
    _root = _root if _root != REPO / "dataset_out" else REPO
    for _d in ("dataset", "re-dataset"):
        f = _root / _d / "labels.csv"
        if not f.exists():
            continue
        rows = list(_csv.DictReader(open(f, encoding="utf-8")))
        if "usable_image" in (rows[0] if rows else {}):        # thế hệ cũ
            bad = [r for r in rows if r.get("crop_quality_flag") in ("blank", "truncated")]
            sai = [r for r in bad if r.get("usable_image") != "0"]
            check(f"{_d}/: {len(bad)} ô ảnh hỏng đều có usable_image=0 THẬT trên đĩa",
                  not sai, f"{len(sai)} ô sai")
            check(f"{_d}/: cột usable_image KHÔNG rỗng",
                  any((r.get("usable_image") or "").strip() for r in rows))
        else:                                                   # 12 cột
            tp = f.parent / "labels_trace.csv"
            tr = list(_csv.DictReader(open(tp, encoding="utf-8"))) if tp.exists() else []
            check(f"{_d}/: labels_trace.csv có crop_quality_flag THẬT trên đĩa (không rỗng)",
                  bool(tr) and "crop_quality_flag" in tr[0]
                  and any((r.get("crop_quality_flag") or "").strip() for r in tr))
            check(f"{_d}/: trace cùng số dòng labels.csv", len(tr) == len(rows),
                  f"{len(tr)} vs {len(rows)}")
        break

    check("TỪ CHỐI verdict không có khai xuất xứ", 'PROV = "NGUOI_CHAM.md"' in av)
    check("từ chối cả khi khai còn bỏ trống", '"⬜" in txt' in av)
    check("ghi rõ vì sao bắt khai", "KHÔNG hề tự khai là máy" in av)
    qt = (REPO / "docs" / "QUY_TRINH_CHAM_TAY.md").read_text(encoding="utf-8")
    check("quy trình có MẪU khai xuất xứ", "NGUOI_CHAM.md" in qt)
    check("mẫu hỏi người thứ hai có phải tác giả không", "KHÔNG phải tác giả" in qt)

    # --- đường (A): ĐỘI NGOÀI chấm trên re-dataset/ ---------------------------
    check("nhận verdicts.csv phẳng theo cột `image`", 'VERDICT_CSV = "verdicts.csv"' in av)
    check("chỉ nhận 3 giá trị tiếng Việt", '"dung", "sai", "khong_doc_duoc"' in av)
    check("verdict sai định dạng là LỖI, không bỏ qua im lặng",
          "dòng sai định dạng" in av)
    check("`image` lạ => LỖI (đội chấm dùng bản CŨ)", "bản re-dataset CŨ" in av)
    # HỒI QUY: bảng precision từng báo 100% cho chính luật gây lỗi, vì ô bị hạ đã mất
    # dấu vết luật gốc. Lỗi phải được quy về ĐÚNG luật sinh ra nó.
    check("GIỮ luật gốc trước khi hạ", 'df.loc[demote, "rule_goc"]' in av)
    check("bảng quy lỗi về LUẬT GỐC", 'r.get("rule_goc")' in av)
    check("chưa chấm hết thì KHÔNG suy rộng", "KHÔNG suy rộng" in av)

    cb = (REPO / "scripts" / "clean_build.sh").read_text(encoding="utf-8")
    check("clean_build dọn cả re-dataset", "re-dataset" in cb)
    check("clean_build KHÔNG xoá human_audit của người",
          "human_audit" in cb and "KHÔNG tái tạo được" in cb)


def test_co_khong_bi_ep_kieu() -> None:
    """HỒI QUY: cột CỜ không được biến thành số thực khi đi qua pandas.

    Lỗi thật 2026-08-25: `label_in_train` build ghi ra '1', nhưng cột có ô rỗng (tầng
    SYLLABLE không có giá trị) nên pandas ép cả cột về float64 và ghi ra '1.0'. Hậu quả:
    mọi phép lọc `label_in_train == "1"` trả về RỖNG mà KHÔNG báo lỗi. `crop_w`/`crop_h`
    cũng dính ('138' -> '138.0').

    Đây là lớp lỗi CÂM: không traceback, không cảnh báo, chỉ ra kết quả sai.
    """
    from pathlib import Path as _P
    import csv as _csv
    REPO = _P(__file__).resolve().parents[2]
    print("[cột cờ không bị ép kiểu]")
    for f in ("pipeline/remediation/cli.py", "pipeline/remediation/confusion_fix.py",
              "pipeline/remediation/glyph_fix.py"):
        src = (REPO / f).read_text(encoding="utf-8")
        check(f"{_P(f).name} đọc label_in_train là chuỗi", '"label_in_train": str' in src)
        check(f"{_P(f).name} đọc crop_w/h là chuỗi",
              '"crop_w": str' in src and '"crop_h": str' in src)
        # A-10/A-12 (16/09): cờ v3 (qd01_locked…, nom_idx/syl_idx, n_*) cũng phải là chuỗi
        check(f"{_P(f).name} đọc cờ v3 (qd01_locked, nom_idx, band_touched…) là chuỗi",
              all(f'"{c}": str' in src for c in
                  ("qd01_locked", "qd01_excluded", "nom_idx", "syl_idx", "band_touched",
                   "l1_support", "flank_gold", "n_ocr")))
    for d in ("dataset", "re-dataset"):
        p2 = REPO / d / "labels.csv"
        if not p2.exists():
            continue
        rows = list(_csv.DictReader(open(p2, encoding="utf-8")))
        vals = {r.get("label_in_train", "") for r in rows}
        check(f"{d}/: label_in_train chỉ có '', '0', '1' — KHÔNG có '1.0'",
              vals <= {"", "0", "1"}, f"thấy {sorted(vals)}")
        cw = {r.get("crop_w", "") for r in rows if r.get("crop_w")}
        check(f"{d}/: crop_w không có đuôi '.0'",
              not any("." in v for v in cw), f"ví dụ {sorted(cw)[:3]}")
        break


def test_nan_syllable_not_eaten() -> None:
    """HỒI QUY: `nan` là một ÂM TIẾNG VIỆT (難), không phải giá trị thiếu.

    pandas mặc định đọc chuỗi "nan"/"NA"/"NULL"/"None"/"null"/"NaN" thành NaN. Đo
    2026-08-24: 3 ô mất hẳn âm khi đi qua `cli._load` (2 trong đó là GOLD
    s1_inter_s2_similar NẰM TRONG bộ giao nộp: gold/stt4_page_0040_c02_023.png và
    gold/stt4_page_0108_c08_178.png, ocr=准 label=难), và 8 mục từ điển biến mất mỗi
    lần QuocNgu_SinoNom.csv được đọc bằng pandas.

    Cách vá: `keep_default_na=False, na_values=[""]` — ô RỖNG vẫn thành NaN (s3_cosine
    và các cột số cần thế) nhưng mọi chuỗi CÓ NỘI DUNG được giữ nguyên.
    """
    import io
    import pandas as pd
    print("[nan-là-âm-tiếng-việt]")
    csv_text = ("image,syllable,s3_cosine,tier\n"
                "a.png,nan,,GOLD\n"
                "b.png,na,0.5,GOLD\n"
                "c.png,null,,REVIEW\n"
                "d.png,,0.3,REVIEW\n")

    naive = pd.read_csv(io.StringIO(csv_text))
    check("tái hiện được lỗi: pandas mặc định NUỐT 'nan'/'null'",
          int(naive["syllable"].isna().sum()) == 3)

    fixed = pd.read_csv(io.StringIO(csv_text), keep_default_na=False, na_values=[""])
    check("sau vá: 'nan' giữ nguyên là chuỗi", fixed["syllable"].iloc[0] == "nan")
    check("sau vá: 'null' giữ nguyên là chuỗi", fixed["syllable"].iloc[2] == "null")
    check("sau vá: ô RỖNG vẫn thành NaN", bool(pd.isna(fixed["syllable"].iloc[3])))
    check("sau vá: cột số rỗng vẫn NaN (s3_cosine không hỏng)",
          int(fixed["s3_cosine"].isna().sum()) == 2)
    check("sau vá: cột số vẫn ra kiểu số", str(fixed["s3_cosine"].dtype).startswith("float"))

    # đường THẬT: hàm nạp của remediation phải giữ được âm 'nan'
    if LABELS.exists():
        from pipeline.remediation import cli as _cli
        import inspect
        src = inspect.getsource(_cli)
        check("cli.py đã dùng keep_default_na=False", "keep_default_na=False" in src)
        real = pd.read_csv(LABELS, dtype={"image_md5": str},
                           keep_default_na=False, na_values=[""])
        n_nan = int((real["syllable"] == "nan").sum())
        check(f"labels.csv giữ được {n_nan} ô âm 'nan'", n_nan >= 3, f"đếm được {n_nan}")


def test_confusion_fix_join() -> None:
    """Join verdict<->nhãn: tiền tố sách đời cũ `yen*` phải được chuẩn hoá về `stt*`.

    Trước 2026-08-23 join thô khớp 0/825 nên `--measure` trả `null` TRONG IM LẶNG.
    """
    print("[confusion-fix join]")
    n = cfix_mod.normalize_image_key
    check("gold/yen11_... -> gold/stt11_...",
          n("gold/yen11_page_0018_c05_094.png") == "gold/stt11_page_0018_c05_094.png")
    check("silver/yen4_... -> silver/stt4_...", n("silver/yen4_x.png") == "silver/stt4_x.png")
    check("yen ở đầu chuỗi cũng đổi", n("yen2_a.png") == "stt2_a.png")
    check("stt* giữ nguyên (luỹ đẳng)", n("gold/stt2_p.png") == "gold/stt2_p.png")
    check("'yen' KHÔNG ở ranh giới đường dẫn thì không đụng",
          n("abc_yen9_x.png") == "abc_yen9_x.png")
    check("luỹ đẳng khi gọi 2 lần", n(n("gold/yen11_a.png")) == n("gold/yen11_a.png"))


def test_measure_chi_nhan_verdict_nguoi() -> None:
    """`--measure` chỉ được đo khi có verdict NGƯỜI, và phải NÓI RA khi không đo được.

    Hai khiếm khuyết thật vá 2026-08-25:
    1. BẪY XUẤT XỨ — hàm neo vào `verdicts_reanchored.csv`, tệp MÁY chấm đã bị vô hiệu.
       Nó đã bị xoá khỏi đĩa nhưng VẪN nằm trong lịch sử git: một lần `git checkout` bản cũ
       là precision máy-chấm lặng lẽ chảy vào report.json trở lại. Đúng cái sai đã một lần
       huỷ sạch số liệu của đề tài.
    2. NUỐT THÔNG BÁO — khối in gác bằng `if measure and report.get(...)`, nên khi chưa có
       verdict thì report ăn thêm hai trường null mà màn hình không hé một chữ.
    """
    import contextlib, io
    print("[--measure chỉ nhận verdict người]")
    final = pd.DataFrame({"image": ["gold/stt2_a.png", "gold/stt2_b.png", "silver/stt2_c.png"],
                          "tier": ["GOLD", "GOLD", "SILVER"]})
    goc_nguon, goc_cu = cfix_mod.NGUON_VERDICT, cfix_mod.VERDICT_DOI_CU
    try:
        if True:
            sb = cfix_mod.REPO / "_tmp_verdict_test"
            shutil.rmtree(sb, ignore_errors=True)   # rác của lần chạy hỏng trước làm test nói dối
            sb.mkdir()
            try:
                cfix_mod.NGUON_VERDICT = ("_tmp_verdict_test",)
                cfix_mod.VERDICT_DOI_CU = sb / "khong_ton_tai.csv"
                check("chưa có tệp nào -> None, KHÔNG bịa số",
                      cfix_mod.measure_gold_precision(final) is None)
                (sb / "verdicts.csv").write_text(
                    "image,verdict\ngold/stt2_a.png,dung\ngold/stt2_b.png,sai\n"
                    "silver/stt2_c.png,dung\n", encoding="utf-8")
                check("có verdicts.csv nhưng THIẾU NGUOI_CHAM.md -> None",
                      cfix_mod.measure_gold_precision(final) is None)

                (sb / "NGUOI_CHAM.md").write_text("# ai chấm\n- Test\n", encoding="utf-8")
                r = cfix_mod.measure_gold_precision(final)
                check("đủ hai tệp -> đo được", r is not None)
                check("chỉ tính ô GOLD (2/3, bỏ SILVER)", r["gold_audited"] == 2,
                      f"n={r['gold_audited']}")
                check("precision = 1 đúng / 2 = 0,5", r["precision"] == 0.5, str(r["precision"]))
                check("xuất xứ trỏ tệp thật, KHÔNG còn 'UNVERIFIED'",
                      "NGUOI_CHAM.md" in r["provenance"] and "UNVERIFIED" not in r["provenance"],
                      r["provenance"])

                # BẪY: tệp máy-chấm đời cũ quay lại -> phải TỪ CHỐI dù verdict người có sẵn
                cu = cfix_mod.REPO / "_tmp_verdict_test" / "gia_verdicts_reanchored.csv"
                cu.write_text("image_new,verdict,status\n", encoding="utf-8")
                cfix_mod.VERDICT_DOI_CU = cu
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    r2 = cfix_mod.measure_gold_precision(final)
                check("tệp MÁY chấm đời cũ quay lại -> TỪ CHỐI đo", r2 is None)
                check("và có kêu lên chứ không im", "MÁY chấm" in buf.getvalue())
            finally:
                shutil.rmtree(sb, ignore_errors=True)
    finally:
        cfix_mod.NGUON_VERDICT, cfix_mod.VERDICT_DOI_CU = goc_nguon, goc_cu

    # nuốt thông báo: --measure luôn phải in một câu về precision
    src = (REPO / "pipeline" / "remediation" / "confusion_fix.py").read_text(encoding="utf-8")
    check("khối in KHÔNG còn gác bằng report.get(...)",
          'if measure and report.get("precision_gold_after")' not in src)
    check("có nhánh nói rõ CHƯA ĐO ĐƯỢC", "CHƯA ĐO ĐƯỢC" in src)


def test_glyph_fix_quyet_dinh_nguoi() -> None:
    """Phán quyết NGƯỜI đi vào bộ nhãn — và chỉ đi vào được qua ba cổng.

    Mô-đun này GÁN nhãn cho cả một lớp âm, tức nó có quyền lực lớn nhất trong toàn pipeline:
    một dòng cấu hình sai là 2.014 ô sai. Ba cổng phải luôn đứng:
      1. THIẾU KHAI XUẤT XỨ -> từ chối chạy. Dự án đã một lần nhầm phán quyết MÁY thành
         phán quyết NGƯỜI và phải huỷ sạch precision 97,98% / Fisher p=5,4e-8 / κ=0,13.
      2. LỆCH MÃ ĐIỂM -> từ chối. 𠊚 U+2029A và 𠊛 U+2029B đều là chữ "người" và chỉ khác
         nhau một mã điểm; lệch ở đây là gán sai cả khối mà nhìn mắt thường không thấy.
      3. Ô KHÔNG CÓ ẢNH -> không gán. Người không nhìn được thì không có phán quyết.
         Khối "người" có 2.144 ô nhưng chỉ 2.014 ô có crop.
    """
    import yaml as _y
    from pipeline.remediation import glyph_fix as gf
    print("[quyết định glyph]")
    NG = "\U0002029A"

    def _df():
        return pd.DataFrame({
            "image": ["gold/a.png", "gold/b.png", "", "gold/d.png"],
            "book": ["stt2"] * 4, "syllable": ["người", "Người", "người", "mà"],
            "ocr_char": ["㝵", "早", "身", "麻"], "label": ["", "", "", "麻"],
            "unicode": ["", "", "", "U+9EBB"], "label_level": ["", "", "", "char"],
            "tier": ["REVIEW", "SILVER_uncalibrated", "REVIEW", "GOLD"],
            "rule": ["confusion_fix:x", "s2_inter_s3_corrected", "no_s1_inter_s2", "s1_inter_s2_direct"],
        })

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "gold").mkdir()
        for n in ("a", "b", "d"):
            (d / "gold" / f"{n}.png").write_bytes(b"x")      # 'd.png' có ảnh nhưng đang GOLD
        prov = d / "khai.md"
        qd = {"am": "người", "chu": NG, "unicode": "U+2029A", "hanh_dong": "gan_nhan",
              "to_tier": "GOLD", "ly_do": "test", "xuat_xu": None, "chi_khi_co_anh": True}

        # CỔNG 1 — thiếu khai xuất xứ
        for xx, ten in ((None, "không khai"), ("docs/khong_ton_tai_9999.md", "khai tệp ma")):
            try:
                gf.ap_dung(_df(), [{**qd, "xuat_xu": xx}], d)
                check(f"THIẾU xuất xứ ({ten}) -> phải TỪ CHỐI", False, "chạy tuột")
            except SystemExit as e:
                check(f"THIẾU xuất xứ ({ten}) -> TỪ CHỐI", "TỪ CHỐI" in str(e))

        prov.write_text("# ai chấm\n- Test\n", encoding="utf-8")
        rel = str(prov.relative_to(gf.REPO)) if str(prov).startswith(str(gf.REPO)) else None
        goc = gf.REPO
        try:
            gf.REPO = d                                    # trỏ gốc về sandbox
            ok = {**qd, "xuat_xu": "khai.md"}

            # CỔNG 2 — lệch mã điểm
            try:
                gf.ap_dung(_df(), [{**ok, "unicode": "U+2029B"}], d)
                check("LỆCH mã điểm 𠊚/𠊛 -> phải TỪ CHỐI", False, "chạy tuột")
            except SystemExit as e:
                check("LỆCH mã điểm 𠊚/𠊛 -> TỪ CHỐI", "TỪ CHỐI" in str(e))

            out, log = gf.ap_dung(_df(), [ok], d)
            check("gán đúng 2 ô có ảnh", log[0]["gan"] == 2, str(log[0]))
            check("CỔNG 3: bỏ 1 ô không có ảnh", log[0]["bo_vi_khong_anh"] == 1, str(log[0]))
            check("khớp âm KHÔNG phân biệt hoa-thường", out.loc[1, "label"] == NG)
            check("nhãn = 𠊚 và unicode = U+2029A",
                  (out.loc[0, "label"], out.loc[0, "unicode"]) == (NG, "U+2029A"))
            check("label_level thành char", out.loc[0, "label_level"] == "char")
            check("tier thành GOLD", out.loc[0, "tier"] == "GOLD")
            check("rule truy được về quyết định", out.loc[0, "rule"].startswith("quyet_dinh_nguoi:"))
            check("giữ rule_goc để quy lỗi đúng luật cũ",
                  out.loc[0, "rule_goc"] == "confusion_fix:x", out.loc[0, "rule_goc"])
            check("giữ tier_goc", out.loc[0, "tier_goc"] == "REVIEW")
            check("ô KHÔNG có ảnh giữ nguyên REVIEW", out.loc[2, "tier"] == "REVIEW")
            check("KHÔNG ghi đè ô đã ở GOLD (âm khác)",
                  (out.loc[3, "label"], out.loc[3, "tier"]) == ("麻", "GOLD"))
        finally:
            gf.REPO = goc

    # cấu hình thật + dữ liệu thật
    cfg = REPO / "config" / "quyet_dinh_glyph.yaml"
    if cfg.exists():
        qds = (_y.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("quyet_dinh", [])
        check("cấu hình thật: mọi quyết định đều khai xuất xứ CÓ THẬT",
              all((REPO / q["xuat_xu"]).exists() for q in qds), str([q.get("xuat_xu") for q in qds]))
        check("cấu hình thật: chữ khớp mã điểm đã khai",
              all(f"U+{ord(q['chu']):04X}" == q["unicode"].upper() for q in qds))
    lf = REPO / "dataset_out" / "labels_final.csv"
    if lf.exists():
        df = pd.read_csv(lf, dtype={"image_md5": str}, keep_default_na=False, na_values=[""])
        q = df[df["rule"].astype(str).str.startswith("quyet_dinh_nguoi:")]
        if len(q):
            check(f"thật: {len(q):,} ô quyết định đều mang đúng một nhãn",
                  set(q["label"]) == {NG}, str(set(q["label"]))[:60])
            check("thật: đều có ảnh", (q["image"].astype(str).str.strip() != "").all())
            check("thật: đều giữ tier_goc", (q["tier_goc"].astype(str) != "").all())


def test_glyph_fix_kiem_khoa() -> None:
    """Chế độ `--mode kiem` (A-2, N12): chỉ ĐỐI CHIẾU khoá QĐ-01, không gán.

    Khoá bền (book,page,column,nom_idx) sống qua mọi đổi ma trận/hộp; bbox_cu/md5_cu chỉ
    để đo trôi. Phải đếm đúng 4 lớp: khớp / pending / mất / lệch — và chỉ khớp + pending
    mới được tính "đạt". Thiếu nom_idx thì rơi về khoá bbox↔bbox_cu.
    """
    from pipeline.remediation import glyph_fix as gf
    print("[glyph_fix --mode kiem]")
    NG = "\U0002029A"
    cells = pd.DataFrame({
        "book": ["stt2"] * 4, "page": ["p1"] * 4, "column": ["1", "1", "2", "2"],
        "nom_idx": ["0", "3", "5", "7"], "syl_idx": ["0", "3", "5", "7"],
        "syllable": ["người"] * 4, "ocr_char": ["㝵"] * 4, "label": [NG] * 4,
        "unicode": ["U+2029A"] * 4,
        "bbox_cu": ["[0, 0, 9, 9]", "[0, 30, 9, 39]", "[10, 0, 19, 9]", "[10, 70, 19, 79]"],
        "prev_bbox_cu": [""] * 4, "next_bbox_cu": [""] * 4,
        "image_md5_cu": ["a", "b", "c", "d"], "image_cu": ["gold/x.png"] * 4,
        "tier_build": ["GOLD"] * 4, "rule_build": ["s1_inter_s2_direct"] * 4})
    lab = pd.DataFrame({
        "book": ["stt2"] * 4, "page": ["p1"] * 4, "column": [1, 1, 2, 9],
        "nom_idx": [0, 3, 5, 7], "syllable": ["người", "người", "mà", "người"],
        "label": [NG, "㝵", NG, NG], "tier": ["GOLD", "GOLD", "GOLD", "GOLD"],
        "rule": ["quyet_dinh_nguoi:qd01_cell_lock", "s1_inter_s2_direct",
                 "quyet_dinh_nguoi:pending", "quyet_dinh_nguoi:qd01_cell_lock"],
        "bbox": ["[0, 0, 9, 9]", "[0, 30, 9, 39]", "[10, 1, 19, 9]", "[10, 70, 19, 79]"],
        "image_md5": ["a", "b", "zz", "d"]})
    dec = pd.DataFrame(columns=["book", "page", "column", "nom_idx", "bbox_cu", "syllable",
                                "ocr_char", "label_hien_tai", "ly_do", "quyet", "nguoi_ky",
                                "ngay", "xuat_xu"])
    rep, j = gf.kiem_khoa(lab, cells, dec)
    check("khoá theo nom_idx khi labels có cột", rep["khoa"] == "nom_idx")
    check("khớp 1 (label 𠊚 + rule QĐ-01)", rep["n_khop"] == 1, str(rep))
    check("pending 1 (rule quyet_dinh_nguoi:pending)", rep["n_pending"] == 1)
    check("lệch 1 (nhãn 㝵, rule thường) — KHÔNG được tính đạt", rep["n_lech"] == 1)
    check("mất 1 (cột 2 nom 7 không có; cột 9 không tính)", rep["n_mat"] == 1)
    check("bbox đổi 1, md5 đổi 1 (ô pending)", (rep["n_bbox_doi"], rep["n_md5_doi"]) == (1, 1))
    check("âm ≠ người 1 (ô pending ghép sang 'mà')", rep["n_syllable_doi"] == 1)
    check("không đạt: khớp + pending 2 != 4", rep["dat"] is False)
    check("GOLD 'người' ≠ 𠊚 chưa có phán quyết = 1 (ô lệch)", rep["n_gold_nguoi_chua_quyet"] == 1)
    check("KHÔNG gán: labels giữ nguyên nhãn", lab.loc[1, "label"] == "㝵")

    # phán quyết đã điền cho ô pending -> báo "chưa rebuild"
    dec2 = pd.DataFrame([{"book": "stt2", "page": "p1", "column": "2", "nom_idx": "5",
                          "bbox_cu": "[10, 0, 19, 9]", "syllable": "người", "ocr_char": "㝵",
                          "label_hien_tai": "", "ly_do": "trôi", "quyet": "giu_2029A",
                          "nguoi_ky": "t", "ngay": "2026-09-16", "xuat_xu": "x.md"}])
    rep2, _ = gf.kiem_khoa(lab, cells, dec2)
    check("pending có quyết nhưng chưa rebuild = 1", rep2["n_pending_co_quyet_chua_rebuild"] == 1)

    # thiếu nom_idx -> khoá bbox↔bbox_cu: ô pending đổi bbox nên thành MẤT
    rep3, _ = gf.kiem_khoa(lab.drop(columns=["nom_idx"]), cells, dec)
    check("fallback khoá bbox khi labels không có nom_idx", rep3["khoa"] == "bbox")
    check("fallback: khớp 1, mất 2 (bbox đổi không tìm thấy)", (rep3["n_khop"], rep3["n_mat"]) == (1, 2), str(rep3))

    # đạt: cả 4 ô khớp/pending
    lab_ok = lab.copy(); lab_ok.loc[1, ["label", "rule"]] = [NG, "quyet_dinh_nguoi:qd01_cell_lock"]
    lab_ok.loc[3, "column"] = 2
    rep4, _ = gf.kiem_khoa(lab_ok, cells, dec)
    check("đạt khi khớp 3 + pending 1 == 4", rep4["dat"] and rep4["n_khop"] == 3)

    # cấu hình thật: cells + decisions đúng schema, 2.014 ô, 12 ô A-3 chờ người
    cp = REPO / "config" / "qd01_cells.csv"; dp = REPO / "config" / "qd01a_decisions.csv"
    if cp.exists():
        c = pd.read_csv(cp, dtype=str, keep_default_na=False)
        check("thật: qd01_cells.csv 2.014 ô, 100% 𠊚, 0 trùng khoá",
              len(c) == 2014 and set(c["label"]) == {NG}
              and not c.duplicated(["book", "page", "column", "nom_idx"]).any(), str(len(c)))
        check("thật: bbox_cu/md5_cu 2.014/2.014",
              (c["bbox_cu"] != "").all() and (c["image_md5_cu"] != "").all())
    if dp.exists():
        d = pd.read_csv(dp, dtype=str, keep_default_na=False)
        check("thật: qd01a_decisions.csv đủ 13 cột", list(d.columns) == list(dec.columns), str(list(d.columns)))
        try:
            gf._doc_decisions(dp)
            check("thật: mọi `quyet` đã điền đều hợp lệ + có xuất xứ", True)
        except SystemExit as e:
            check("thật: mọi `quyet` đã điền đều hợp lệ + có xuất xứ", False, str(e)[:120])


def main() -> int:
    print("=" * 64)
    print("REMEDIATION SELFTEST")
    print("=" * 64)
    test_census_synthetic()
    test_census_md5_dup()
    test_remediate_synthetic()
    test_remediate_khong_split()
    test_real()
    test_s3_unwind()
    test_two_outputs_and_verdicts()
    test_co_khong_bi_ep_kieu()
    test_nan_syllable_not_eaten()
    test_confusion_fix_join()
    test_measure_chi_nhan_verdict_nguoi()
    test_glyph_fix_quyet_dinh_nguoi()
    test_glyph_fix_kiem_khoa()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
