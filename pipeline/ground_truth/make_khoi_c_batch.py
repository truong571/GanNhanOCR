"""Khối C (C-2) — mẻ chấm MÙ TUYỆT ĐỐI trên bộ giao nộp `labels_final.csv`, hai câu tách.

VÌ SAO PHẢI DỰNG LẠI (không dùng thẳng `audit_grid.build_audit`)
------------------------------------------------------------------
Rà C0 (16/09) trên `audit_grid.py` + `make_combined_batch.py` cho 6 chỗ không đáp ứng
Khối C, mỗi chỗ đều là thứ đã làm hỏng mẻ 04/08 (κ = 0,13):

  1. MỘT câu hỏi trộn hai chiều (crop / nhãn).  Khối C tách hẳn:
        Q1  "ô này cắt đúng MỘT chữ, và chữ đó đọc là ÂM hiển thị?"  (chiều CROP+ÂM)
        Q2  "MÃ Unicode hiển thị có đúng chữ trên crop?"              (chiều MÃ, chỉ khi có mã)
     và mỗi lựa chọn có ĐỊNH NGHĨA + NGƯỠNG (sai_crop = cắt cụt > 1/3 chữ / dính ≥ 2 chữ /
     không phải chữ) — bài học 04/08: 'wrong_image' không ngưỡng nên không tái lập.
  2. item_id = sha1(image)[:16] không muối -> suy ngược được từ labels_final. Ở đây id là
     UUID rút gọn rút từ RNG có seed; ánh xạ chỉ nằm trong `_khoa/KHOA.jsonl`.
  3. Không có dwell / session / lịch sử sửa (chỉ `ts` lúc bấm, ghi đè). Giao diện mới ghi
     `dwell_ms` từng ô (thời gian ô hiển thị cho tới khi trả lời xong), `session_id`,
     `sitting_id`, `order`, `hist[]` (mọi lần đổi ý), và KHOÁ sau khi Xuất.
  4. `canh_bao` (blank/truncated) hiện ra HTML — với Q1 đó là gợi ý đáp án -> giấu.
  5. `cands` top-12 từ điển làm CHAR_B vắng 24,5 % vs CHAR_A 11,1 % -> rò tầng -> bỏ hẳn.
  6. Hàng không có tệp crop bị BỎ (n_no_crop) -> REVIEW / B-3 không dựng được. Ở đây crop
     được CẮT LẠI từ `prepared/<Sách>/pages/<page>.png` bằng đúng công thức của
     `build_dataset.save_crop` (pad 0,12 + carve hàng xóm + tighten). Kiểm 16/09: 4/4 ô
     cắt lại byte-identical với tệp trong dataset_out — người chấm không phân biệt được.

THIẾT KẾ MẺ (seed 20260916, tất định)
--------------------------------------
Mẻ chính 600 ô = CHAR_A 300 · CHAR_B 100 · SYL 150 · REVIEW 50 (đối chứng), rút từ
labels_final loại QĐ-01 (đã có người nhìn) và QUARANTINE. Trong mỗi tier_v3 phân tầng phụ
(lớp cột n_ocr==n_qn / khác) × (box_source detector / legacy_locked_col / midpoint+split),
sàn 5 ô/tầng, phần còn lại tỉ lệ thuận (`make_combined_batch._allocate`), SRS trong từng
tầng, ghi `stratum_N` và `design_weight = N_h/n_h` cho ước lượng Horvitz–Thompson.

Tầng bổ sung (design_weight ghi khi là mẫu ngẫu nhiên của một dân số rõ; NaN khi chủ đích):
  T1  300 ô B-3 qua cổng thị giác (gate_09, 1.025 ô REVIEW), 100/sách, SRS trong sách.
      (C3 16/09: 150 ô chỉ đạt "≤ 3 %" khi 0 lỗi — P(đạt | lỗi thật 1 %) = 0,22; 300 ô cho phép
      ≤ 3 lỗi, P(đạt | 1 %) = 0,65, P(đạt | 0,5 %) = 0,93.)
  T2  ≈100 ô chuỗi "thanh ghi trượt" (k5 thr0.8 len3): rút NGUYÊN CHUỖI, ưu tiên (0) chuỗi
      chứa ô QĐ-01 (cả 8 ô), (1) run ≥ 4, (2) còn lại. Mẫu cụm -> không có design_weight.
  T3  19 ô B-2 đổi âm (census) + 30/117 ô B-5 hộp 3 nhánh lệch (SRS, weight 117/30).
  T4  ô MỒI ≈ 90: 45 dương = ô QĐ-01 người đã xem, SRS PHÂN TẦNG theo tier_v3 trên cả 2.012 ô
      QĐ-01 (có design_weight -> đồng thời là tầng "QĐ-01" của dân số re-dataset 70.326);
      đáp án Q1 dung; Q2 dung là theo PHÁN QUYẾT (mã 𠊚 nhận ra được -> chỉ đếm, không tính đạt/rớt).
      45 âm = ô CHAR_A sạch hiển thị ÂM/MÃ của ô kề syl_idx±1 (đáp án Q1 sai_am, Q2 sai).
  T5  LẶP ẨN 100 ô, PHÂN TẦNG về phía tầng dễ khác câu trả lời (MAIN CHAR_A 20 · CHAR_B 15 · SYL 15 ·
      REVIEW 15 · T1 15 · T2 20) để κ có KTC hẹp (≈ ±0,13 thay vì ±0,27 khi SRS 75 ô toàn 'dung');
      id khác, cách ≥ 100 vị trí (`interleave`).
  T6  40 ô âm "người" CHƯA KHOÁ (283 ô ngoài QĐ-01, SRS, weight 283/40): pha loãng dấu hiệu
      'người' ⇒ QĐ-01 ở Q1 (trước: 84/89 ô 'người' trong mẻ là QĐ-01) và đo thẳng khối QĐ-01a.
Tổng ≈ 1.280 lượt, xáo trộn toàn bộ theo seed, chia 5 phiên ≤ 270 lượt.

MÙ TUYỆT ĐỐI: HTML chỉ chứa id ngẫu nhiên, ảnh base64, ÂM, MÃ, glyph tham chiếu. Không
tier / rule / p / box_source / tầng / đường dẫn / tên tệp / cảnh báo. Ô mồi và ô lặp không
khác gì ô thường. `check_blind()` grep lại HTML (đã bỏ payload base64) trước khi ghi xong.

HAI PHA TRÊN MỘT Ô (C3 16/09 — vá lỗi lộ nhóm "không mã"): MÃ + glyph tham chiếu CHỈ hiện SAU KHI
Q1 đã trả lời. Trước đó mọi ô trông y hệt nhau (crop · ngữ cảnh · ÂM) — người chấm không biết ô có
mã hay không (= không biết CHAR vs SYL/REVIEW/T1), và Q1 được trả lời trong CÙNG điều kiện thông
tin ở mọi tier (không có glyph "gợi ý" cho ô CHAR). Verdict ghi `q1_blind` (câu trả lời Q1 lúc lộ
mã) tách với `q1` (cuối cùng) và đếm `n_q1_change_after_reveal`; ước lượng dùng `q1_blind`.

PILOT (C-1, `--pilot`): phiên riêng 50 lượt = 40 ô (CHAR_A 20 · CHAR_B 7 · SYL 10 · REVIEW 3, SRS trong
tier_v3 từ cùng dân số nhưng LOẠI mọi ô đã nằm trong mẻ chính) + 5 mồi (2 dương QĐ-01, 3 âm) + 5 lặp ẩn
(gap ≥ 15). Seed riêng (SEED+1), id riêng, khoá riêng `_khoa/KHOA_pilot.jsonl` + `manifest_pilot.jsonl`,
`plan_pilot.json`. Mục đích: đo dwell / κ / mồi TRƯỚC khi chấm thật; không trích precision từ pilot.

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.make_khoi_c_batch            # mẻ chính 4 phiên
    .venv/bin/python -m pipeline.ground_truth.make_khoi_c_batch --pilot    # phiên pilot 50 ô (cần mẻ chính đã dựng)
"""
from __future__ import annotations

import argparse
import hashlib
import html as _html
import json
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import audit_grid
from .make_combined_batch import _allocate, interleave

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO / "dataset_out" / "labels_final.csv"
DEFAULT_OUT = REPO / "dataset_out" / "human_audit" / "khoi_c_2026-09-16"
KHOIB = REPO / "KhoiB" / "v3"
PREPARED = REPO / "prepared"
FD_DIR = REPO / "gannhanocr-fd"
FONT = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
DATASET_DIR = REPO / "dataset_out"

SEED = 20260916
CROP_PAD = 0.12          # config/pipeline.yaml step2.crop_pad_frac — CÙNG công thức build_dataset
MAIN_N = {"CHAR_A": 300, "CHAR_B": 100, "SYL": 150, "REVIEW": 50}
MIN_CELL = 5
N_T1, N_T2, N_B5, N_MOI_DUONG, N_MOI_AM, N_NGUOI = 300, 100, 30, 45, 45, 40
# lặp ẩn phân tầng: (tang, tier_v3 hoặc None) -> n. Tổng 100.
LAP_ALLOC = {("MAIN", "CHAR_A"): 20, ("MAIN", "CHAR_B"): 15, ("MAIN", "SYL"): 15, ("MAIN", "REVIEW"): 15,
             ("T1_GATE", None): 15, ("T2_CHAIN", None): 20}
N_LAP = sum(LAP_ALLOC.values())
MIN_GAP = 100
N_SESSIONS = 5
MAX_PER_SESSION = 270
PILOT_SEED_OFFSET = 1
PILOT_N = {"CHAR_A": 20, "CHAR_B": 7, "SYL": 10, "REVIEW": 3}
PILOT_N_MOI_DUONG, PILOT_N_MOI_AM, PILOT_MIN_GAP = 2, 3, 15
PILOT_LAP_ALLOC = {("MAIN", "CHAR_A"): 2, ("MAIN", "CHAR_B"): 1, ("MAIN", "SYL"): 1, ("MAIN", "REVIEW"): 1}
PILOT_N_LAP = sum(PILOT_LAP_ALLOC.values())
PILOT_SESSION_ID = "KC20260916-PILOT"

# Chuỗi KHÔNG được xuất hiện trong HTML người chấm (sau khi bỏ payload base64).
LEAK_PATTERNS = ("gold/", "syllable/", "review/", "tier", "rule", "stt2", "stt4", "stt11",
                 "page_", "CHAR_A", "CHAR_B", "REVIEW", "box_source", "design_weight",
                 "stratum", "qd01", "moi_", "repeat", "gate", "chain", "decoy", "KHOA",
                 "nguoi_chua", "T4_", "T5_", "T6_")

Q1_CHOICES = [
    ("1", "dung", "ĐÚNG", "một chữ trọn vẹn, và chữ đó đọc đúng là ÂM hiển thị"),
    ("2", "sai_crop", "SAI CROP", "cắt cụt > 1/3 chữ · dính ≥ 2 chữ · hoặc không phải chữ"),
    ("3", "sai_am", "SAI ÂM", "crop là MỘT chữ hoàn chỉnh, nhưng KHÔNG đọc là âm này"),
    ("4", "khong_ro", "KHÔNG RÕ", "không đủ căn cứ, kể cả sau khi xem ảnh ngữ cảnh"),
]
Q2_CHOICES = [
    ("5", "dung", "MÃ ĐÚNG", "glyph tham chiếu đúng là chữ trên crop (dị thể chấp nhận)"),
    ("6", "sai", "MÃ SAI", "chữ trên crop là một chữ KHÁC — ghi mã đúng nếu biết"),
    ("7", "khong_ro", "KHÔNG RÕ", "không đủ căn cứ"),
]


# ---------------------------------------------------------------------------
# khung nhãn
def _key(r) -> str:
    return f"{r['book']}|{r['page']}|{r['column']}|{r['nom_idx']}"


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"image", "book", "page", "column", "nom_idx", "syl_idx", "syllable", "label",
            "unicode", "tier", "tier_v3", "rule", "box_source", "count_source", "n_ocr",
            "n_qn", "n_det", "qd01_locked", "bbox", "image_md5", "crop_quality_flag"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"labels thiếu cột {sorted(miss)} — không phải thế hệ v3")
    df["key"] = df.apply(_key, axis=1)
    if df["key"].duplicated().any():
        raise SystemExit("khoá (book,page,column,nom_idx) trùng — không nối được sidecar KhoiB")
    df["col_class"] = np.where(df["n_ocr"] == df["n_qn"], "eq", "ne")
    df["box_class"] = df["box_source"].map(
        lambda s: "midpoint_split" if s in ("midpoint", "split") else s)
    return df


def _srs_idx(index: np.ndarray, n: int, tag: str, seed: int) -> np.ndarray:
    """SRS tất định: seed con dẫn xuất sha1 từ (seed, tag) — cùng cách draw_tier."""
    s = int(hashlib.sha1(f"{seed}:{tag}".encode()).hexdigest()[:8], 16) % (2 ** 31)
    n = min(n, len(index))
    if n <= 0:
        return index[:0]
    return np.random.default_rng(s).choice(index, size=n, replace=False)


# ---------------------------------------------------------------------------
# mẻ chính
def draw_main(df: pd.DataFrame, seed: int, n_by_tier: dict[str, int] | None = None
              ) -> tuple[pd.DataFrame, list[dict]]:
    """600 ô phân tầng (tier_v3 × lớp cột × box_source), SRS trong tầng, có trọng số."""
    n_by_tier = dict(MAIN_N if n_by_tier is None else n_by_tier)
    base = df[(df["qd01_locked"] == "0") & (df["tier"] != "QUARANTINE") & (df["bbox"] != "")]
    parts, cells = [], []
    for tier, n in n_by_tier.items():
        if tier == "REVIEW":
            pool = base[base["tier_v3"] == "REVIEW"]                 # không có tệp crop -> cắt lại
        else:
            pool = base[(base["tier_v3"] == tier) & (base["image"] != "")]
        pool = pool.sort_values("key")
        sizes = {f"{a}|{b}": int(v) for (a, b), v in
                 pool.groupby(["col_class", "box_class"]).size().items()}
        alloc = _allocate(sizes, n, MIN_CELL)
        for cell, n_h in sorted(alloc.items()):
            if n_h <= 0:
                continue
            cc, bc = cell.split("|")
            sub = pool[(pool["col_class"] == cc) & (pool["box_class"] == bc)]
            N_h = len(sub)
            idx = _srs_idx(sub.index.to_numpy(), n_h, f"main:{tier}:{cell}", seed)
            d = df.loc[idx].copy()
            d["tang"] = "MAIN"
            d["stratum"] = f"{tier}|{cell}"
            d["stratum_N"] = N_h
            d["design_weight"] = N_h / float(len(idx))
            parts.append(d)
            cells.append({"tang": "MAIN", "stratum": f"{tier}|{cell}", "tier_v3": tier,
                          "col_class": cc, "box_class": bc, "N_h": N_h, "n_h": int(len(idx)),
                          "design_weight": round(N_h / len(idx), 4)})
    out = pd.concat(parts)
    if out["key"].duplicated().any():
        raise RuntimeError("mẻ chính rút trùng ô")
    return out, cells


# ---------------------------------------------------------------------------
# tầng bổ sung
def draw_t1_gate(df: pd.DataFrame, used: set[str], seed: int, n: int = N_T1
                 ) -> tuple[pd.DataFrame, list[dict]]:
    g = pd.read_csv(KHOIB / "visual_syl_candidates.csv", dtype=str, keep_default_na=False)
    g = g[g["gate_09"] == "1"].copy()
    g["key"] = g.apply(_key, axis=1)
    N_gate = len(g)
    g = g[~g["key"].isin(used)]
    joined = df[df["key"].isin(g["key"])].merge(
        g[["key", "p_syl_v3", "argmax", "max_prob"]], on="key", how="inner")
    bad = joined[joined["syllable"] != joined["argmax"]]
    if len(bad):
        raise RuntimeError(f"gate_09 nhưng argmax ≠ âm ghép ở {len(bad)} ô — sidecar lệch thế hệ")
    joined = joined.sort_values("key").reset_index(drop=True)
    sizes = {str(k): int(v) for k, v in joined.groupby("book").size().items()}
    # rải đều 3 sách: sàn = n/3, phần lẻ theo tỉ lệ
    alloc = _allocate(sizes, n, min_cell=n // len(sizes))
    parts, cells = [], []
    for book, n_h in sorted(alloc.items()):
        sub = joined[joined["book"] == book]
        idx = _srs_idx(sub.index.to_numpy(), n_h, f"t1:{book}", seed)
        d = sub.loc[idx].copy()
        d["tang"] = "T1_GATE"
        d["stratum"] = f"T1|gate09|{book}"
        d["stratum_N"] = len(sub)
        d["design_weight"] = len(sub) / float(len(idx))
        parts.append(d)
        cells.append({"tang": "T1_GATE", "stratum": f"T1|gate09|{book}", "N_h": int(len(sub)),
                      "n_h": int(len(idx)), "design_weight": round(len(sub) / len(idx), 4),
                      "N_gate_total": N_gate})
    return pd.concat(parts), cells


def chain_ids(k5: pd.DataFrame) -> pd.Series:
    """Chuỗi = ô liền nhau (nom_idx cách 1) cùng (book,page,column,shift). Tái lập 114 chuỗi."""
    k5 = k5.copy()
    k5["_n"] = k5["nom_idx"].astype(int)
    k5 = k5.sort_values(["book", "page", "column", "shift", "_n"])
    cid, out = 0, {}
    prev = None
    for i, r in k5.iterrows():
        g = (r["book"], r["page"], r["column"], r["shift"])
        if prev is None or g != prev[0] or r["_n"] != prev[1] + 1:
            cid += 1
        out[i] = f"{r['book']}|{r['page']}|c{r['column']}|s{r['shift']}|#{cid}"
        prev = (g, r["_n"])
    return pd.Series(out)


def draw_t2_chains(df: pd.DataFrame, used: set[str], seed: int, n_target: int = N_T2
                   ) -> tuple[pd.DataFrame, list[dict]]:
    k5 = pd.read_csv(KHOIB / "k5_register_shift_thr0.8_len3.csv", dtype=str, keep_default_na=False)
    k5["key"] = k5.apply(_key, axis=1)
    k5["chain_id"] = chain_ids(k5)
    chains = k5.groupby("chain_id").agg(run_len=("key", "size"),
                                        n_qd01=("qd01_locked", lambda s: int((s == "1").sum())),
                                        any_used=("key", lambda s: bool(s.isin(used).any())))
    chains = chains[~chains["any_used"]]
    # ưu tiên: (0) chuỗi có ô QĐ-01 — cả 8 ô QĐ-01 trong chuỗi phải được người nhìn lại;
    # (1) run ≥ 4; (2) còn lại. Trong nhóm xáo tất định theo seed.
    rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:t2".encode()).hexdigest()[:8], 16))
    chains["_r"] = rng.random(len(chains))
    chains["_pri"] = np.where(chains["n_qd01"] > 0, 0, np.where(chains["run_len"] >= 4, 1, 2))
    chains = chains.sort_values(["_pri", "_r"])
    picked, tot = [], 0
    for cid, r in chains.iterrows():
        if tot >= n_target:
            break
        picked.append(cid)
        tot += int(r["run_len"])
    sel = k5[k5["chain_id"].isin(picked)]
    d = df[df["key"].isin(sel["key"])].merge(
        sel[["key", "chain_id", "shift", "run_len", "argmax", "max_prob", "p_syl"]],
        on="key", how="inner")
    if len(d) != len(sel):
        raise RuntimeError(f"chuỗi trượt: {len(sel)} ô sidecar nhưng chỉ nối được {len(d)} ô")
    d["tang"] = "T2_CHAIN"
    d["stratum"] = "T2|chain"
    d["stratum_N"] = int(len(k5))
    d["design_weight"] = np.nan                                   # mẫu cụm, chủ đích
    cells = [{"tang": "T2_CHAIN", "stratum": "T2|chain", "n_chains": len(picked),
              "n_chains_total": int(k5["chain_id"].nunique()), "N_h": int(len(k5)),
              "n_h": int(len(d)), "n_qd01": int((d["qd01_locked"] == "1").sum()),
              "run_len_hist": {str(k): int(v) for k, v in
                               sel.groupby("chain_id").size().value_counts().sort_index().items()},
              "design_weight": None}]
    return d, cells


def draw_t3(df: pd.DataFrame, used: set[str], seed: int, n_b5: int = N_B5
            ) -> tuple[pd.DataFrame, list[dict]]:
    b2 = json.load(open(KHOIB / "b2_changed_cells.json", encoding="utf-8"))
    b2_rows = b2["resyl_usable_both_sample"]
    b2_keys = [r["k"] for r in b2_rows]
    b2_vis = {r["k"]: r["syl_vis"] for r in b2_rows}
    d2 = df[df["key"].isin(b2_keys) & ~df["key"].isin(used)].copy()
    if len(d2) != len(b2_keys):
        raise RuntimeError(f"B-2: {len(b2_keys)} khoá nhưng nối được {len(d2)}")
    d2["tang"] = "T3_B2"
    d2["stratum"] = "T3|b2"
    d2["stratum_N"] = len(b2_keys)
    d2["design_weight"] = 1.0                                     # census
    d2["argmax"] = d2["key"].map(b2_vis)

    b5 = pd.read_csv(KHOIB / "b5" / "qd01_3nhanh_lech_117.csv", dtype=str, keep_default_na=False)
    b5["key"] = b5.apply(_key, axis=1)
    N5 = len(b5)
    b5 = b5[~b5["key"].isin(used | set(d2["key"]))].sort_values("key")
    idx = _srs_idx(b5.index.to_numpy(), n_b5, "t3:b5", seed)
    sel = b5.loc[idx]
    d5 = df[df["key"].isin(sel["key"])].merge(
        sel[["key", "bbox_truoc_lock", "iou", "box_source_truoc_lock"]], on="key", how="inner")
    if len(d5) != len(sel):
        raise RuntimeError("B-5: không nối đủ ô vào labels_final")
    d5["tang"] = "T3_B5"
    d5["stratum"] = "T3|b5"
    d5["stratum_N"] = N5
    d5["design_weight"] = N5 / float(len(d5))
    cells = [{"tang": "T3_B2", "stratum": "T3|b2", "N_h": len(b2_keys), "n_h": int(len(d2)),
              "design_weight": 1.0},
             {"tang": "T3_B5", "stratum": "T3|b5", "N_h": N5, "n_h": int(len(d5)),
              "design_weight": round(N5 / len(d5), 4)}]
    return pd.concat([d2, d5]), cells


def draw_t4_decoys(df: pd.DataFrame, used: set[str], seed: int,
                   n_pos: int = N_MOI_DUONG, n_neg: int = N_MOI_AM
                   ) -> tuple[pd.DataFrame, list[dict]]:
    # dương: QĐ-01 người đã xem — SRS PHÂN TẦNG theo tier_v3 trên TOÀN BỘ ô QĐ-01 có crop
    # (2.012 = đúng phần QĐ-01 của re-dataset) -> có stratum_N / design_weight, dùng được như
    # tầng "QĐ-01" khi suy precision cho dân số re-dataset 70.326. `used` chỉ loại ô đã ở tầng khác
    # (chuỗi T2 / B-5) — N_h vẫn là dân số đầy đủ để trọng số khép kín.
    pos_all = df[(df["qd01_locked"] == "1") & (df["image"] != "")]
    pos = pos_all[~pos_all["key"].isin(used)].sort_values("key")
    sizes = {str(k): int(v) for k, v in pos_all.groupby("tier_v3").size().items()}
    alloc = _allocate(sizes, n_pos, min_cell=min(3, n_pos // 4))   # 45 -> sàn 3/tier; pilot 2 -> không sàn
    parts, cells_pos = [], []
    for tier, n_h in sorted(alloc.items()):
        if n_h <= 0:
            continue
        sub = pos[pos["tier_v3"] == tier]
        idx = _srs_idx(sub.index.to_numpy(), n_h, f"t4pos:{tier}", seed)
        d = sub.loc[idx].copy()
        d["stratum"] = f"T4|moi_duong|{tier}"
        d["stratum_N"] = sizes[tier]
        d["design_weight"] = sizes[tier] / float(len(idx))
        parts.append(d)
        cells_pos.append({"tang": "T4_MOI_DUONG", "stratum": f"T4|moi_duong|{tier}", "tier_v3": tier,
                          "N_h": sizes[tier], "n_h": int(len(idx)),
                          "design_weight": round(sizes[tier] / len(idx), 4),
                          "dap_an": {"q1": "dung", "q2": "dung (theo phán quyết QĐ-01, chỉ đếm)"}})
    dpos = pd.concat(parts)
    dpos["tang"] = "T4_MOI_DUONG"
    dpos["decoy_q1"] = "dung"
    dpos["decoy_q2"] = "dung"

    # âm: ô CHAR_A sạch, hiển thị ÂM/MÃ của ô kề syl_idx±1 (cùng cột) có mã, khác âm & khác mã
    cand = df[(df["tier_v3"] == "CHAR_A") & (df["qd01_locked"] == "0") & (df["image"] != "")
              & (df["crop_quality_flag"] == "ok") & (df["tier"] != "QUARANTINE")
              & ~df["key"].isin(used | set(dpos["key"]))]
    by_col = {k: g for k, g in df[df["label"] != ""].groupby(["book", "page", "column"])}
    rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:t4neg".encode()).hexdigest()[:8], 16))
    order = cand.sort_values("key").index.to_numpy().copy()
    rng.shuffle(order)
    rows, taken = [], 0
    for i in order:
        if taken >= n_neg:
            break
        r = df.loc[i]
        col = by_col.get((r["book"], r["page"], r["column"]))
        if col is None:
            continue
        si = int(r["syl_idx"])
        sides = [1, -1] if rng.random() < 0.5 else [-1, 1]
        nb = None
        for s in sides:
            m = col[(col["syl_idx"].astype(int) == si + s) & (col["tier_v3"].isin(["CHAR_A", "CHAR_B"]))
                    & (col["syllable"] != r["syllable"]) & (col["label"] != r["label"])
                    & (col["qd01_locked"] == "0")]
            if len(m) == 1:
                nb = m.iloc[0]
                break
        if nb is None:
            continue
        rr = r.copy()
        rr["shown_syllable"], rr["shown_label"], rr["shown_unicode"] = \
            nb["syllable"], nb["label"], nb["unicode"]
        rr["decoy_from_key"] = nb["key"]
        rows.append(rr)
        taken += 1
    if taken < n_neg:
        raise RuntimeError(f"mồi âm: chỉ ghép được {taken}/{n_neg}")
    dneg = pd.DataFrame(rows)
    dneg["tang"] = "T4_MOI_AM"
    dneg["stratum"] = "T4|moi_am"
    dneg["decoy_q1"] = "sai_am"
    dneg["decoy_q2"] = "sai"
    dneg["stratum_N"] = np.nan
    dneg["design_weight"] = np.nan
    out = pd.concat([dpos, dneg])
    cells = cells_pos + [{"tang": "T4_MOI_AM", "stratum": "T4|moi_am", "n_h": int(len(dneg)),
                          "dap_an": {"q1": "sai_am", "q2": "sai"}, "design_weight": None}]
    return out, cells


def draw_t6_nguoi(df: pd.DataFrame, used: set[str], seed: int, n: int = N_NGUOI
                  ) -> tuple[pd.DataFrame, list[dict]]:
    """T6: ô âm 'người' CHƯA KHOÁ (ngoài QĐ-01), SRS có trọng số.

    Hai việc: (1) pha loãng dấu hiệu 'người' ⇒ QĐ-01 ở pha Q1 (trong mẻ 16/09 có 84/89 ô 'người' là
    QĐ-01 — người chấm biết QĐ-01 sẽ đoán ra); (2) đo thẳng khối 'người' còn lại (QĐ-01a: 14 ô `bo`,
    202 SYL, 39 REVIEW, 42 CHAR_B/GOLD) — precision Q1 của khối này chưa từng được người nhìn mù."""
    pool_all = df[(df["syllable"] == "người") & (df["qd01_locked"] == "0") & (df["tier"] != "QUARANTINE")
                  & (df["bbox"] != "")]
    pool = pool_all[~pool_all["key"].isin(used)].sort_values("key")
    N = int(len(pool_all))
    idx = _srs_idx(pool.index.to_numpy(), n, "t6:nguoi", seed)
    d = pool.loc[idx].copy()
    d["tang"] = "T6_NGUOI"
    d["stratum"] = "T6|nguoi_chua_khoa"
    d["stratum_N"] = N
    d["design_weight"] = N / float(len(idx))
    cells = [{"tang": "T6_NGUOI", "stratum": "T6|nguoi_chua_khoa", "N_h": N, "n_h": int(len(d)),
              "design_weight": round(N / len(idx), 4),
              "tier_v3_mix": {str(k): int(v) for k, v in d["tier_v3"].value_counts().sort_index().items()},
              "N_tier_v3": {str(k): int(v) for k, v in pool_all["tier_v3"].value_counts().sort_index().items()}}]
    return d, cells


# ---------------------------------------------------------------------------
# id mù + lặp ẩn
def assign_ids(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """UUID4 rút gọn (12 hex) rút từ RNG có seed, gán theo thứ tự (tang, key) tất định.
    Không dẫn xuất từ image/md5 nên KHÔNG suy ngược được từ labels_final."""
    rows = rows.sort_values(["tang", "key"]).reset_index(drop=True)
    rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:ids".encode()).hexdigest()[:8], 16))
    ids = []
    seen = set()
    for _ in range(len(rows)):
        while True:
            u = uuid.UUID(bytes=rng.bytes(16), version=4).hex[:12]
            if u not in seen:
                seen.add(u)
                ids.append(u)
                break
    rows["item_id"] = ids
    return rows


def add_hidden_repeats(rows: pd.DataFrame, seed: int, alloc: dict | None = None) -> pd.DataFrame:
    """Lặp ẩn PHÂN TẦNG theo (tang, tier_v3): SRS trong từng ô của `alloc` (mặc định LAP_ALLOC).

    Vì sao không SRS trên mẻ chính: κ Cohen với tỉ lệ nền 'dung' ≈ 90 % có pe ≈ 0,82 nên
    SE(κ) ≈ sqrt(po(1-po)/(n(1-pe)^2)) ≈ 0,14 với n = 75 (KTC ±0,27). Kéo về các tầng dự
    kiến ≈ 20 % khác 'dung' (REVIEW, T1, T2, CHAR_B) -> pe ≈ 0,68, SE ≈ 0,07 với n = 100."""
    alloc = dict(LAP_ALLOC if alloc is None else alloc)
    rows = rows.reset_index(drop=True)            # nhãn index phải duy nhất (concat nhiều tầng có thể trùng)
    parts = []
    for (tang, tier), n in sorted(alloc.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        sub = rows[rows["tang"] == tang]
        if tier is not None:
            sub = sub[sub["tier_v3"] == tier]
        if sub.empty or n <= 0:
            continue
        idx = _srs_idx(sub.sort_values("key").index.to_numpy(), n, f"t5:{tang}:{tier}", seed)
        parts.append(rows.loc[idx])
    rep = pd.concat(parts).copy()
    rep["repeat_of"] = rep["item_id"]
    rep["lap_tang"] = rep["tang"]                  # tầng gốc, để ước lượng tách κ theo tầng
    rep["tang"] = "T5_LAP"
    rep["stratum"] = "__repeat__"                 # interleave() + estimate nhận dạng theo đây
    rep["stratum_N"] = np.nan
    rep["design_weight"] = np.nan
    rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:ids:rep".encode()).hexdigest()[:8], 16))
    taken = set(rows["item_id"])
    ids = []
    for _ in range(len(rep)):
        while True:
            u = uuid.UUID(bytes=rng.bytes(16), version=4).hex[:12]
            if u not in taken:
                taken.add(u)
                ids.append(u)
                break
    rep["item_id"] = ids
    return rep


# ---------------------------------------------------------------------------
# ảnh: cắt lại crop y hệt build_dataset.save_crop; ngữ cảnh cột ±2 ô
def _cut_like_build(img_bgr, gray_full, bbox, prev_bbox, next_bbox):
    import cv2
    from pipeline.align_engine.bbox_fix import carve_neighbor_ink, tighten_box
    H, W = img_bgr.shape[:2]
    ox1, oy1, ox2, oy2 = (int(v) for v in bbox)
    pw, ph = int((ox2 - ox1) * CROP_PAD), int((oy2 - oy1) * CROP_PAD)
    x1, y1 = max(0, ox1 - pw), max(0, oy1 - ph)
    x2, y2 = min(W, ox2 + pw), min(H, oy2 + ph)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = crop.copy()
    if prev_bbox is not None or next_bbox is not None:
        crop = carve_neighbor_ink(crop, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    tb = tighten_box(gray)
    if tb is not None:
        a, c, b, d = tb
        crop = crop[c:d, a:b]
    if crop.size == 0:
        return None
    return Image.fromarray(crop[:, :, ::-1])


def _column_context(scan: Image.Image, bbox: list[int], pad_y: float = 2.2, pad_x: float = 0.55,
                    out_h: int = 460) -> Image.Image | None:
    """Ngữ cảnh CỘT: ±2,2 ô theo chiều dọc, ±0,55 ô theo chiều ngang, khung đỏ quanh ô."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    W, H = scan.size
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = max(0, int(x1 - bw * pad_x)), max(0, int(y1 - bh * pad_y))
    cx2, cy2 = min(W, int(x2 + bw * pad_x)), min(H, int(y2 + bh * pad_y))
    ctx = scan.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    ImageDraw.Draw(ctx).rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
                                  outline=(220, 30, 30), width=3)
    if ctx.height > out_h:
        s = out_h / ctx.height
        ctx = ctx.resize((max(1, int(ctx.width * s)), out_h))
    return ctx


def _neighbors(df_page: pd.DataFrame) -> dict[str, tuple]:
    """prev/next bbox theo thứ tự y trong CÙNG CỘT, trên MỌI hàng của trang (như build_dataset)."""
    out = {}
    for _, col in df_page[df_page["bbox"] != ""].groupby("column"):
        col = col.assign(_bb=col["bbox"].map(json.loads))
        col = col.assign(_cy=col["_bb"].map(lambda b: (b[1] + b[3]) / 2.0)).sort_values("_cy")
        bbs = list(col["_bb"])
        keys = list(col["key"])
        for i, k in enumerate(keys):
            out[k] = (bbs[i - 1] if i > 0 else None, bbs[i + 1] if i < len(bbs) - 1 else None)
    return out


def render_items(rows: pd.DataFrame, df: pd.DataFrame, fonts: list[Path]) -> tuple[dict, dict]:
    """Dựng dict item_id -> dữ liệu HTML (ảnh base64, ÂM, MÃ). Duyệt theo trang để tiết kiệm RAM.
    Trả về (items, meta) với meta = crop_source / kích thước để ghi KHOA."""
    import cv2
    items, meta = {}, {}
    rows = rows.sort_values(["book", "page", "audit_order"])
    for (book, page), grp in rows.groupby(["book", "page"], sort=False):
        sp = PREPARED / audit_grid.book_to_scan_dir(book) / "pages" / f"{page}.png"
        img = cv2.imread(str(sp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"thiếu trang scan {sp}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scan = Image.fromarray(img[:, :, ::-1])
        nbrs = None
        for _, r in grp.iterrows():
            bbox = audit_grid._parse_bbox(r["bbox"])
            if r["image"]:
                crop = audit_grid._load_crop(DATASET_DIR, r["image"])
                src = "file"
                if crop is None:
                    raise FileNotFoundError(f"thiếu crop {r['image']}")
            else:
                if nbrs is None:
                    nbrs = _neighbors(df[(df["book"] == book) & (df["page"] == page)])
                pv, nx = nbrs.get(r["key"], (None, None))
                crop = _cut_like_build(img, gray, bbox, pv, nx)
                src = "scan_cut"
                if crop is None:
                    raise RuntimeError(f"không cắt được {r['key']}")
            ctx = _column_context(scan, bbox) if bbox else None
            uni = str(r.get("shown_unicode") or "")
            ref = audit_grid._reference_glyph(FD_DIR, uni, fonts) if uni else None
            items[r["item_id"]] = {
                "id": r["item_id"],
                "crop": audit_grid._data_uri(crop), "cw": crop.width, "ch": crop.height,
                "ctx": audit_grid._data_uri(ctx) if ctx is not None else "",
                "ref": audit_grid._data_uri(ref) if ref is not None else "",
                "syl": str(r.get("shown_syllable") or ""),
                "code": str(r.get("shown_label") or ""),
                "uni": uni,
            }
            meta[r["item_id"]] = {"crop_source": src, "crop_w": crop.width, "crop_h": crop.height,
                                  "has_ref_glyph": ref is not None, "has_ctx": ctx is not None}
        scan.close()
    return items, meta


# ---------------------------------------------------------------------------
# HTML một phiên — một ô một màn, hai câu, dwell, lịch sử, khoá sau Xuất
_HTML = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#f6f5f1; --card:#fff; --ink:#1f1f1f; --line:#d9d9d9; --accent:#35558a; --mute:#777; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Heiti SC",Arial,sans-serif; }
  header { position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line);
    padding:10px 16px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; z-index:10; }
  header b { font-size:15px; }
  .bar { flex:1; height:10px; background:#eee; border-radius:6px; overflow:hidden; min-width:140px; }
  .bar > i { display:block; height:100%; background:var(--accent); width:0; }
  button { font:inherit; padding:7px 12px; border:1px solid var(--line); background:#fff;
    border-radius:6px; cursor:pointer; }
  button:hover { background:#eef; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  main { max-width:980px; margin:0 auto; padding:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; }
  .imgs { display:flex; gap:22px; align-items:flex-start; flex-wrap:wrap; }
  .imgs figure { margin:0; text-align:center; }
  .imgs img { border:1px solid var(--line); background:#fff; image-rendering:auto; }
  figcaption { font-size:11px; color:var(--mute); text-transform:uppercase; letter-spacing:.06em; margin-top:5px; }
  .lab { font-size:64px; line-height:1; margin:4px 0 2px; font-family:"Han-nom Minh","HanaMinB","HanaMinA","NomNaTong",serif; }
  .syl { font-size:30px; font-weight:700; margin:6px 0; }
  .meta { color:#555; font-size:14px; }
  .nocode { color:var(--mute); font-size:15px; margin:8px 0; }
  .q { margin-top:18px; padding-top:14px; border-top:1px dashed var(--line); }
  .q h3 { margin:0 0 8px; font-size:15px; }
  .choices { display:flex; gap:10px; flex-wrap:wrap; }
  .choices button { font-size:14px; text-align:left; line-height:1.25; }
  .choices button small { display:block; color:var(--mute); font-weight:400; font-size:11.5px; }
  .k1 { border-color:#2e6e4c; } .k1.sel { background:#2e6e4c; color:#fff; }
  .k2 { border-color:#a03b32; } .k2.sel { background:#a03b32; color:#fff; }
  .k3 { border-color:#9a6b10; } .k3.sel { background:#9a6b10; color:#fff; }
  .k4 { border-color:#777; }    .k4.sel { background:#777; color:#fff; }
  .k5 { border-color:#2e6e4c; } .k5.sel { background:#2e6e4c; color:#fff; }
  .k6 { border-color:#a03b32; } .k6.sel { background:#a03b32; color:#fff; }
  .k7 { border-color:#777; }    .k7.sel { background:#777; color:#fff; }
  .sel small { color:#eee; }
  input.free { font:inherit; padding:5px 8px; border:1px solid var(--line); border-radius:6px; margin-top:8px; width:260px; }
  .nav { display:flex; gap:10px; align-items:center; margin-top:16px; flex-wrap:wrap; }
  .idx { color:#aaa; font-size:12px; float:right; }
  .hint { color:#888; font-size:12px; }
  .lock { background:#fff4e5; border:1px solid #f0c27a; padding:8px 12px; border-radius:8px; margin-bottom:12px; font-size:14px; }
  .done { color:#2e6e4c; font-weight:600; }
  textarea { width:100%; height:120px; font:12px/1.3 monospace; margin-top:8px; }
</style></head>
<body>
<header>
  <b>__TITLE__</b>
  <span class="hint">Q1: phím 1·2·3·4 &nbsp;·&nbsp; Q2: phím 5·6·7 &nbsp;·&nbsp; ←/→ chuyển ô</span>
  <div class="bar"><i id="prog"></i></div>
  <span id="count" style="font-variant-numeric:tabular-nums">0/0</span>
  <button id="next-undone" title="nhảy tới ô chưa chấm">Ô chưa chấm</button>
  <button id="export">Xuất JSONL</button>
</header>
<main>
  <div id="lockbox" class="lock" hidden></div>
  <div id="card" class="card"></div>
  <div class="nav">
    <button id="prev">← Trước</button>
    <button id="next">Sau →</button>
    <span class="hint" id="navhint"></span>
  </div>
  <div id="exportbox" hidden>
    <p class="hint">Nếu trình duyệt không tự tải tệp, sao chép nội dung dưới đây và lưu thành <code>__FNAME__</code>:</p>
    <textarea id="exporttxt" readonly></textarea>
  </div>
</main>
<script>
const ITEMS = __ITEMS__;
const SESSION = "__SESSION__";
const FNAME = "__FNAME__";
const KEY = "khoi_c::" + SESSION;
const Q1 = __Q1__;
const Q2 = __Q2__;
const Q1K = Object.fromEntries(Q1.map(c => [c[0], c[1]]));
const Q2K = Object.fromEntries(Q2.map(c => [c[0], c[1]]));
const SITTING = SESSION + ":" + Date.now().toString(36);

function loadStore() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { return {}; } }
let store = loadStore();
if (!store.__meta) store.__meta = { session_id: SESSION, created: Date.now(), sittings: [] };
store.__meta.sittings.push({ sitting_id: SITTING, t: Date.now() });
function persist() { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} }
persist();

function rec(id) {
  if (!store[id]) store[id] = { q1: null, q1_am: "", q2: null, q2_code: "", t_first_shown: null,
    t_answer: null, dwell_ms: null, dwell_acc: 0, visits: 0, hist: [],
    q1_blind: null, revealed_at: null, dwell_q1_ms: null, n_q1_change_after_reveal: 0 };
  return store[id];
}
function hasCode(it) { return !!it.code; }
function revealed(r) { return !!r.revealed_at; }
function complete(it, r) { return !!r.q1 && (!hasCode(it) || !!r.q2); }
function isLocked() { return !!store.__meta.locked; }

let cur = 0, shownAt = null;
function leaveCurrent() {
  if (shownAt === null) return;
  const r = rec(ITEMS[cur].id);
  r.dwell_acc = (r.dwell_acc || 0) + (Date.now() - shownAt);
  shownAt = null;
}
function show(i) {
  leaveCurrent();
  cur = Math.max(0, Math.min(ITEMS.length - 1, i));
  const it = ITEMS[cur], r = rec(it.id);
  shownAt = Date.now();
  r.visits = (r.visits || 0) + 1;
  if (!r.t_first_shown) r.t_first_shown = shownAt;
  persist(); render(); window.scrollTo({ top: 0 });
}
function answer(field, value) {
  if (isLocked()) return;
  const it = ITEMS[cur], r = rec(it.id);
  if (field === "q1" && !Q1K[value] && !Object.values(Q1K).includes(value)) return;
  // Q2 chỉ nhận sau khi MÃ đã lộ (tức Q1 đã trả lời) — pha 2
  if (field === "q2" && (!hasCode(it) || !revealed(r) || (!Q2K[value] && !Object.values(Q2K).includes(value)))) return;
  const v = (field === "q1") ? (Q1K[value] || value) : (field === "q2") ? (Q2K[value] || value) : value;
  if (r[field] === v) return;
  const wasComplete = complete(it, r);
  r[field] = v;
  const ts = Date.now();
  const after = (field === "q1") && revealed(r);
  r.hist.push({ ts: ts, field: field, value: v, sitting: SITTING, after_reveal: after });
  if (after) r.n_q1_change_after_reveal = (r.n_q1_change_after_reveal || 0) + 1;
  if (field === "q1" && !revealed(r)) {
    // pha 1 kết thúc: chốt câu trả lời MÙ, ghi dwell pha 1, rồi mới lộ MÃ + Q2
    r.q1_blind = v;
    r.revealed_at = ts;
    r.dwell_q1_ms = (r.dwell_acc || 0) + (shownAt ? ts - shownAt : 0);
  }
  if (field === "q1" && v !== "sai_am") r.q1_am = "";
  if (field === "q2" && v !== "sai") r.q2_code = "";
  if (!wasComplete && complete(it, r)) {
    r.t_answer = ts;
    r.dwell_ms = (r.dwell_acc || 0) + (shownAt ? ts - shownAt : 0);
    r.order = cur;
    r.sitting_id = SITTING;
  }
  persist(); render();
  if (!wasComplete && complete(it, r)) setTimeout(() => { if (cur < ITEMS.length - 1) show(cur + 1); }, 220);
}
function freeText(field, value) {
  if (isLocked()) return;
  const r = rec(ITEMS[cur].id);
  if (r[field] === value) return;
  r[field] = value;
  r.hist.push({ ts: Date.now(), field: field, value: value, sitting: SITTING });
  persist();
}
function nDone() { return ITEMS.filter(it => complete(it, rec(it.id))).length; }
function refresh() {
  const done = nDone(), total = ITEMS.length;
  document.getElementById("count").textContent = done + "/" + total;
  document.getElementById("prog").style.width = (total ? 100 * done / total : 0) + "%";
  const lb = document.getElementById("lockbox");
  lb.hidden = !isLocked();
  if (isLocked()) lb.innerHTML = "Phiên này đã <b>Xuất</b> lúc " + new Date(store.__meta.locked).toLocaleString() +
    " và đang KHOÁ. Muốn sửa thì bấm <button id='unlock'>Mở khoá (ghi lịch sử)</button> — mọi thay đổi sau đó đều được ghi lại.";
  const u = document.getElementById("unlock");
  if (u) u.onclick = () => { store.__meta.locked = null; store.__meta.unlocks = (store.__meta.unlocks || []); store.__meta.unlocks.push(Date.now()); persist(); render(); };
}
function btn(cls, k, code, lbl, def, sel, onclick) {
  return '<button class="' + cls + (sel ? ' sel' : '') + '" onclick="' + onclick + '">' +
    k + ' · ' + lbl + '<small>' + def + '</small></button>';
}
function render() {
  const it = ITEMS[cur], r = rec(it.id);
  const scale = Math.min(3, 400 / it.ch, 340 / it.cw);
  const w = Math.round(it.cw * scale);
  let h = '<span class="idx">#' + (cur + 1) + ' / ' + ITEMS.length + '</span>';
  const rv = revealed(r);
  h += '<div class="imgs">';
  h += '<figure><img src="' + it.crop + '" style="width:' + w + 'px"><figcaption>crop (phóng ' + scale.toFixed(1) + '×)</figcaption></figure>';
  if (it.ctx) h += '<figure><img src="' + it.ctx + '" style="max-height:460px"><figcaption>vị trí trên trang (khung đỏ)</figcaption></figure>';
  h += '<figure>';
  // PHA 1: mọi ô trông y hệt nhau — MÃ và glyph tham chiếu chỉ lộ SAU khi Q1 đã trả lời
  if (!rv) {
    h += '<div class="nocode">MÃ (nếu có) hiện sau khi<br>trả lời Q1</div>';
  } else if (it.code) {
    h += '<div class="lab">' + it.code + '</div>';
    if (it.ref) h += '<img src="' + it.ref + '" style="width:120px"><figcaption>glyph tham chiếu của MÃ</figcaption>';
    else h += '<figcaption>(không có font cho mã này)</figcaption>';
    h += '<div class="meta">' + it.uni + '</div>';
  } else {
    h += '<div class="nocode">ô này không có mã Unicode<br>(Q2 không áp dụng)</div>';
  }
  h += '</figure></div>';
  h += '<div class="syl">ÂM hiển thị: <span style="color:var(--accent)">' + (it.syl || '—') + '</span></div>';
  h += '<div class="q"><h3>Q1 · Ô này cắt đúng MỘT chữ, và chữ đó đọc là ÂM hiển thị?</h3><div class="choices">';
  Q1.forEach(c => { h += btn('k' + c[0], c[0], c[1], c[2], c[3], r.q1 === c[1], "answer('q1','" + c[1] + "')"); });
  h += '</div>';
  if (r.q1 === 'sai_am') h += '<input class="free" id="q1am" placeholder="âm đúng là… (nếu biết, không bắt buộc)" value="' + (r.q1_am || '').replace(/"/g, '&quot;') + '">';
  if (rv && r.q1 !== r.q1_blind) h += '<p class="hint">Q1 đã đổi sau khi lộ mã (câu trả lời mù: ' + r.q1_blind + ') — được ghi lại.</p>';
  h += '</div>';
  if (it.code && rv) {
    h += '<div class="q"><h3>Q2 · MÃ Unicode hiển thị (' + it.uni + ') có đúng chữ trên crop?</h3><div class="choices">';
    Q2.forEach(c => { h += btn('k' + c[0], c[0], c[1], c[2], c[3], r.q2 === c[1], "answer('q2','" + c[1] + "')"); });
    h += '</div>';
    if (r.q2 === 'sai') h += '<input class="free" id="q2code" placeholder="mã/chữ đúng là… (nếu biết, không bắt buộc)" value="' + (r.q2_code || '').replace(/"/g, '&quot;') + '">';
    h += '</div>';
  }
  const nAns = r.hist.filter(x => x.field === 'q1' || x.field === 'q2').length;
  if (complete(it, r)) h += '<p class="done">✓ đã chấm ô này' + (nAns > (hasCode(it) ? 2 : 1) ? ' (có sửa — lịch sử được ghi)' : '') + '</p>';
  document.getElementById("card").innerHTML = h;
  const a = document.getElementById("q1am"); if (a) a.onchange = e => freeText('q1_am', e.target.value.trim());
  const b = document.getElementById("q2code"); if (b) b.onchange = e => freeText('q2_code', e.target.value.trim());
  document.getElementById("prev").disabled = cur === 0;
  document.getElementById("next").disabled = cur === ITEMS.length - 1;
  document.getElementById("navhint").textContent = (cur === ITEMS.length - 1 && nDone() === ITEMS.length) ? "Hết phiên — bấm Xuất JSONL." : "";
  refresh();
}
function firstUndone() { const i = ITEMS.findIndex(it => !complete(it, rec(it.id))); return i < 0 ? ITEMS.length - 1 : i; }
document.getElementById("prev").onclick = () => show(cur - 1);
document.getElementById("next").onclick = () => show(cur + 1);
document.getElementById("next-undone").onclick = () => show(firstUndone());
document.addEventListener("keydown", e => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (Q1K[e.key]) answer('q1', e.key);
  else if (Q2K[e.key]) answer('q2', e.key);
  else if (e.key === "ArrowRight") show(cur + 1);
  else if (e.key === "ArrowLeft") show(cur - 1);
});
document.addEventListener("visibilitychange", () => {
  // tab ẩn: chốt thời gian đã xem, không cộng dwell khi người chấm rời màn hình
  if (document.hidden) leaveCurrent(); else if (shownAt === null) shownAt = Date.now();
});
document.getElementById("export").onclick = () => {
  leaveCurrent(); shownAt = Date.now();
  const done = nDone();
  if (done < ITEMS.length && !confirm("Còn " + (ITEMS.length - done) + " ô chưa chấm. Vẫn xuất?")) return;
  const now = Date.now();
  const lines = ITEMS.map((it, i) => { const r = rec(it.id); return JSON.stringify({
    item_id: it.id, session_id: SESSION, order: i, q1: r.q1, q1_blind: r.q1_blind, q1_am: r.q1_am || "",
    q2: hasCode(it) ? r.q2 : null, q2_code: r.q2_code || "", has_code: hasCode(it),
    t_first_shown: r.t_first_shown, revealed_at: r.revealed_at, t_answer: r.t_answer,
    dwell_q1_ms: r.dwell_q1_ms, dwell_ms: r.dwell_ms, n_q1_change_after_reveal: r.n_q1_change_after_reveal || 0,
    visits: r.visits, sitting_id: r.sitting_id || null, hist: r.hist, exported_at: now,
    source: "human" }); });
  const txt = lines.join("\n") + "\n";
  store.__meta.locked = now; store.__meta.exports = (store.__meta.exports || []); store.__meta.exports.push(now); persist();
  const blob = new Blob([txt], { type: "application/x-ndjson" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = FNAME; a.click();
  document.getElementById("exportbox").hidden = false; document.getElementById("exporttxt").value = txt;
  render();
};
window.addEventListener("beforeunload", () => { leaveCurrent(); persist(); });
show(firstUndone());
</script>
</body></html>
"""


def render_session_html(items: list[dict], session_id: str, title: str, fname: str) -> str:
    q1 = json.dumps([list(c) for c in Q1_CHOICES], ensure_ascii=False)
    q2 = json.dumps([list(c) for c in Q2_CHOICES], ensure_ascii=False)
    return (_HTML.replace("__TITLE__", _html.escape(title))
            .replace("__SESSION__", session_id)
            .replace("__FNAME__", fname)
            .replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
            .replace("__Q1__", q1).replace("__Q2__", q2))


def _nguoi_leak_stats(order: pd.DataFrame) -> dict:
    """Ở pha Q1 người chấm chỉ thấy ÂM: nếu 'người' ⇒ gần chắc QĐ-01 thì mồi dương/B-5 không còn mù.
    Ghi tỉ lệ để kiểm (T6 pha loãng)."""
    m = order[order["shown_syllable"] == "người"]
    qd = (m["qd01_locked"].astype(str) == "1").sum()
    return {"n_shown_nguoi": int(len(m)), "n_qd01": int(qd), "n_khong_khoa": int(len(m) - qd),
            "frac_qd01": (float(qd) / len(m)) if len(m) else None,
            "n_shown_2029A": int((m["shown_label"] == "𠊚").sum())}


def check_blind(html_text: str, patterns: tuple[str, ...] = LEAK_PATTERNS) -> dict[str, int]:
    """Đếm chuỗi rò rỉ trong HTML SAU KHI bỏ payload base64 (base64 ngẫu nhiên có thể
    chứa 'tier'/'rule' — ~2 lần/30 MB — nên phải loại trước khi grep)."""
    stripped = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "data:IMG", html_text)
    return {p: stripped.count(p) for p in patterns if stripped.count(p)}


# ---------------------------------------------------------------------------
KHOA_FIELDS = ("tang", "stratum", "stratum_N", "design_weight", "repeat_of", "lap_tang",
               "book", "page", "column", "nom_idx", "syl_idx", "tier", "tier_v3", "rule",
               "box_source", "box_class", "col_class", "count_source", "n_ocr", "n_qn", "n_det",
               "qd01_locked", "image", "image_md5", "bbox", "crop_quality_flag",
               "syllable", "label", "unicode", "shown_syllable", "shown_label", "shown_unicode",
               "decoy_q1", "decoy_q2", "decoy_from_key",
               "chain_id", "shift", "run_len", "argmax", "max_prob", "p_syl", "p_syl_v3",
               "bbox_truoc_lock", "iou", "box_source_truoc_lock")


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_khoi_c_batch")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-sessions", type=int, default=N_SESSIONS)
    ap.add_argument("--pilot", action="store_true", help="chỉ dựng phiên pilot 50 ô (cần mẻ chính đã có)")
    args = ap.parse_args(argv)

    labels_path = Path(args.labels)
    out = Path(args.out)
    khoa_dir = out / "_khoa"
    khoa_dir.mkdir(parents=True, exist_ok=True)
    labels_sha = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    print(f"[C] labels {labels_path.relative_to(REPO)} sha256={labels_sha[:16]}…")

    df = load_labels(labels_path)
    if args.pilot:
        build_pilot(df, out, labels_path, args.seed + PILOT_SEED_OFFSET)
        return 0
    used: set[str] = set()

    main_rows, cells = draw_main(df, args.seed)
    used |= set(main_rows["key"])
    t1, c1 = draw_t1_gate(df, used, args.seed); used |= set(t1["key"])
    t2, c2 = draw_t2_chains(df, used, args.seed); used |= set(t2["key"])
    t3, c3 = draw_t3(df, used, args.seed); used |= set(t3["key"])
    t4, c4 = draw_t4_decoys(df, used, args.seed); used |= set(t4["key"])
    t6, c6 = draw_t6_nguoi(df, used, args.seed); used |= set(t6["key"])
    cells += c1 + c2 + c3 + c4 + c6

    allrows = pd.concat([main_rows, t1, t2, t3, t4, t6], ignore_index=True)
    for c in ("shown_syllable", "shown_label", "shown_unicode"):
        src = {"shown_syllable": "syllable", "shown_label": "label", "shown_unicode": "unicode"}[c]
        if c not in allrows.columns:
            allrows[c] = allrows[src]
        else:
            allrows[c] = allrows[c].where(allrows[c].notna() & (allrows[c] != ""), allrows[src])
    for c in ("decoy_q1", "decoy_q2", "decoy_from_key", "repeat_of", "lap_tang", "chain_id", "shift", "run_len",
              "argmax", "max_prob", "p_syl", "p_syl_v3", "bbox_truoc_lock", "iou",
              "box_source_truoc_lock"):
        if c not in allrows.columns:
            allrows[c] = np.nan
    if allrows["key"].duplicated().any():
        raise RuntimeError("một ô xuất hiện ở hai tầng")
    allrows = assign_ids(allrows, args.seed)
    reps = add_hidden_repeats(allrows, args.seed)
    order = interleave(allrows, reps, args.seed, MIN_GAP)
    n_total = len(order)
    n_sessions = args.n_sessions
    per = -(-n_total // n_sessions)
    if per > MAX_PER_SESSION:
        raise SystemExit(f"{per} lượt/phiên > {MAX_PER_SESSION} — tăng --n-sessions")
    order["session"] = (order["audit_order"] // per).astype(int) + 1
    order["session_id"] = order["session"].map(lambda k: f"KC20260916-S{k}")

    # kiểm khoảng cách ô lặp
    pos = dict(zip(order["item_id"], order["audit_order"]))
    gaps = [abs(pos[r.item_id] - pos[r.repeat_of]) for r in order[order["stratum"] == "__repeat__"].itertuples()]
    print(f"[C] {n_total} lượt = {len(allrows)} ô + {len(reps)} lặp ẩn; gap lặp min={min(gaps)} "
          f"(yêu cầu ≥ {MIN_GAP}); {n_sessions} phiên × ≤ {per}")

    fonts = audit_grid.build_font_chain(FONT)
    items, meta = render_items(order, df, fonts)

    # --- KHOÁ (chỉ trong _khoa/) --------------------------------------------------
    khoa_lines, man_lines = [], []
    for r in order.sort_values("audit_order").itertuples(index=False):
        rd = r._asdict()
        k = {"item_id": rd["item_id"], "session": int(rd["session"]), "session_id": rd["session_id"],
             "audit_order": int(rd["audit_order"])}
        for f in KHOA_FIELDS:
            k[f] = _clean(rd.get(f))
        k.update(meta[rd["item_id"]])
        k["has_q2"] = bool(k["shown_label"])
        khoa_lines.append(json.dumps(k, ensure_ascii=False))
    (khoa_dir / "KHOA.jsonl").write_text("\n".join(khoa_lines) + "\n", encoding="utf-8")
    batch_sha = hashlib.sha256((khoa_dir / "KHOA.jsonl").read_bytes()).hexdigest()
    for ln in khoa_lines:
        k = json.loads(ln)
        man_lines.append(json.dumps({
            "item_id": k["item_id"], "session": k["session"], "session_id": k["session_id"],
            "audit_order": k["audit_order"], "tang": k["tang"], "stratum": k["stratum"],
            "stratum_N": k["stratum_N"], "design_weight": k["design_weight"],
            "repeat_of": k["repeat_of"], "has_q2": k["has_q2"],
            "labels_sha256": labels_sha, "labels_path": str(labels_path.relative_to(REPO)),
            "batch_sha256": batch_sha, "seed": args.seed}, ensure_ascii=False))
    (khoa_dir / "manifest.jsonl").write_text("\n".join(man_lines) + "\n", encoding="utf-8")

    # --- HTML từng phiên -----------------------------------------------------------
    html_paths, leaks = [], {}
    for s in range(1, n_sessions + 1):
        sub = order[order["session"] == s].sort_values("audit_order")
        sid = f"KC20260916-S{s}"
        fname = f"verdicts_{sid}.jsonl"
        its = [items[i] for i in sub["item_id"]]
        txt = render_session_html(its, sid, f"Khối C · phiên {s}/{n_sessions} · {len(its)} ô", fname)
        fp = out / f"phien_{s}.html"
        fp.write_text(txt, encoding="utf-8")
        html_paths.append(fp)
        lk = check_blind(txt)
        if lk:
            leaks[fp.name] = lk
    if leaks:
        raise SystemExit(f"RÒ RỈ trong HTML: {leaks}")

    # --- plan + README --------------------------------------------------------------
    tang_counts = {str(k): int(v) for k, v in order["tang"].value_counts().sort_index().items()}
    plan = {
        "batch": "khoi_c_2026-09-16", "seed": args.seed,
        "labels_path": str(labels_path.relative_to(REPO)), "labels_sha256": labels_sha,
        "batch_sha256_KHOA": batch_sha,
        "n_items_total": n_total, "n_cells": int(len(allrows)), "n_repeat": int(len(reps)),
        "min_gap_repeat": MIN_GAP, "repeat_gap_min_observed": int(min(gaps)),
        "sessions": [{"session": s, "session_id": f"KC20260916-S{s}", "html": f"phien_{s}.html",
                      "n": int((order["session"] == s).sum()),
                      "verdict_file": f"verdicts_KC20260916-S{s}.jsonl"} for s in range(1, n_sessions + 1)],
        "tang": tang_counts,
        "strata": cells,
        "crop_source": {k: int(v) for k, v in pd.Series([m["crop_source"] for m in meta.values()]).value_counts().items()},
        "missing_ref_glyph": int(sum(1 for m in meta.values() if not m["has_ref_glyph"])),
        "q2_items": int(sum(1 for m in khoa_lines if json.loads(m)["has_q2"])),
        "questions": {"q1": [c[1] for c in Q1_CHOICES], "q2": [c[1] for c in Q2_CHOICES]},
        "two_phase": {"q1_blind_before_code": True,
                      "note": "MÃ + glyph chỉ lộ sau khi Q1 đã trả lời; verdict ghi q1_blind / q1 / n_q1_change_after_reveal"},
        "lap_alloc": {f"{t}|{tier or '*'}": n for (t, tier), n in LAP_ALLOC.items()},
        "blind_check": {"patterns": list(LEAK_PATTERNS), "leaks": leaks,
                        "nguoi_shown": _nguoi_leak_stats(order)},
        "note_665_syl_co_ma": ("tier_v3=SYL nhưng tier=GOLD (665 ô toàn bộ) vẫn có `label` trong bộ "
                               "giao nộp -> Q2 vẫn hỏi cho các ô đó (đo đúng cái đem nộp)."),
    }
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(_readme(plan), encoding="utf-8")

    print("[C] tầng: " + ", ".join(f"{k}={v}" for k, v in tang_counts.items()))
    for c in cells:
        print(f"  {c['stratum']:<32} N={c.get('N_h', '-'):>6}  n={c['n_h']:>3}  w={c.get('design_weight')}")
    for p in html_paths:
        print(f"[C] {p.relative_to(REPO)}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"[C] khoá: {khoa_dir.relative_to(REPO)}/KHOA.jsonl (sha256 {batch_sha[:16]}…), manifest.jsonl")
    print(f"[C] rò rỉ HTML (sau bỏ base64): {leaks or 'không'}")
    return 0


# ---------------------------------------------------------------------------
# PILOT (C-1)
def draw_pilot_main(df: pd.DataFrame, used: set[str], seed: int, n_by_tier: dict[str, int] | None = None
                    ) -> tuple[pd.DataFrame, list[dict]]:
    """SRS trong từng tier_v3 (không tầng phụ — pilot không dùng để ước lượng), LOẠI ô đã ở mẻ chính."""
    n_by_tier = dict(PILOT_N if n_by_tier is None else n_by_tier)
    base = df[(df["qd01_locked"] == "0") & (df["tier"] != "QUARANTINE") & (df["bbox"] != "")
              & ~df["key"].isin(used)]
    parts, cells = [], []
    for tier, n in n_by_tier.items():
        pool = base[base["tier_v3"] == tier] if tier == "REVIEW" else \
            base[(base["tier_v3"] == tier) & (base["image"] != "")]
        pool = pool.sort_values("key")
        idx = _srs_idx(pool.index.to_numpy(), n, f"pilot:{tier}", seed)
        d = df.loc[idx].copy()
        d["tang"] = "MAIN"
        d["stratum"] = f"{tier}|pilot"
        d["stratum_N"] = len(pool)
        d["design_weight"] = len(pool) / float(len(idx))
        parts.append(d)
        cells.append({"tang": "MAIN", "stratum": f"{tier}|pilot", "tier_v3": tier, "N_h": int(len(pool)),
                      "n_h": int(len(idx)), "design_weight": round(len(pool) / len(idx), 4)})
    out = pd.concat(parts)
    if out["key"].duplicated().any() or out["key"].isin(used).any():
        raise RuntimeError("pilot rút trùng ô hoặc trùng mẻ chính")
    return out, cells


def build_pilot(df: pd.DataFrame, out: Path, labels_path: Path, seed: int) -> dict:
    """Phiên pilot 50 lượt, tách biệt hoàn toàn với mẻ chính (ô, id, khoá, seed)."""
    khoa_dir = out / "_khoa"
    main_khoa = khoa_dir / "KHOA.jsonl"
    if not main_khoa.exists():
        raise SystemExit("chưa có _khoa/KHOA.jsonl — dựng mẻ chính trước rồi mới --pilot")
    used = {f"{k['book']}|{k['page']}|{k['column']}|{k['nom_idx']}"
            for k in (json.loads(ln) for ln in main_khoa.read_text(encoding="utf-8").splitlines() if ln.strip())}
    labels_sha = hashlib.sha256(labels_path.read_bytes()).hexdigest()

    rows, cells = draw_pilot_main(df, used, seed)
    used2 = used | set(rows["key"])
    t4, c4 = draw_t4_decoys(df, used2, seed, n_pos=PILOT_N_MOI_DUONG, n_neg=PILOT_N_MOI_AM)
    cells += c4
    allrows = pd.concat([rows, t4], ignore_index=True)
    for c in ("shown_syllable", "shown_label", "shown_unicode"):
        src = {"shown_syllable": "syllable", "shown_label": "label", "shown_unicode": "unicode"}[c]
        if c not in allrows.columns:
            allrows[c] = allrows[src]
        else:
            allrows[c] = allrows[c].where(allrows[c].notna() & (allrows[c] != ""), allrows[src])
    for c in ("decoy_q1", "decoy_q2", "decoy_from_key", "repeat_of", "lap_tang", "chain_id", "shift", "run_len",
              "argmax", "max_prob", "p_syl", "p_syl_v3", "bbox_truoc_lock", "iou", "box_source_truoc_lock"):
        if c not in allrows.columns:
            allrows[c] = np.nan
    if allrows["key"].isin(used).any():
        raise RuntimeError("pilot dính ô của mẻ chính")
    allrows = assign_ids(allrows, seed)
    reps = add_hidden_repeats(allrows, seed, alloc=PILOT_LAP_ALLOC)
    order = interleave(allrows, reps, seed, PILOT_MIN_GAP)
    order["session"] = 0
    order["session_id"] = PILOT_SESSION_ID
    pos = dict(zip(order["item_id"], order["audit_order"]))
    gaps = [abs(pos[r.item_id] - pos[r.repeat_of]) for r in order[order["stratum"] == "__repeat__"].itertuples()]

    fonts = audit_grid.build_font_chain(FONT)
    items, meta = render_items(order, df, fonts)
    khoa_lines, man_lines = [], []
    for r in order.sort_values("audit_order").itertuples(index=False):
        rd = r._asdict()
        k = {"item_id": rd["item_id"], "session": 0, "session_id": PILOT_SESSION_ID,
             "audit_order": int(rd["audit_order"])}
        for f in KHOA_FIELDS:
            k[f] = _clean(rd.get(f))
        k.update(meta[rd["item_id"]])
        k["has_q2"] = bool(k["shown_label"])
        khoa_lines.append(json.dumps(k, ensure_ascii=False))
    kp = khoa_dir / "KHOA_pilot.jsonl"
    kp.write_text("\n".join(khoa_lines) + "\n", encoding="utf-8")
    batch_sha = hashlib.sha256(kp.read_bytes()).hexdigest()
    for ln in khoa_lines:
        k = json.loads(ln)
        man_lines.append(json.dumps({
            "item_id": k["item_id"], "session": 0, "session_id": PILOT_SESSION_ID,
            "audit_order": k["audit_order"], "tang": k["tang"], "stratum": k["stratum"],
            "stratum_N": k["stratum_N"], "design_weight": k["design_weight"],
            "repeat_of": k["repeat_of"], "has_q2": k["has_q2"],
            "labels_sha256": labels_sha, "labels_path": str(labels_path.relative_to(REPO)),
            "batch_sha256": batch_sha, "seed": seed, "pilot": True}, ensure_ascii=False))
    (khoa_dir / "manifest_pilot.jsonl").write_text("\n".join(man_lines) + "\n", encoding="utf-8")

    fname = f"verdicts_{PILOT_SESSION_ID}.jsonl"
    its = [items[i] for i in order.sort_values("audit_order")["item_id"]]
    txt = render_session_html(its, PILOT_SESSION_ID, f"Khối C · PILOT · {len(its)} ô", fname)
    leaks = check_blind(txt)
    if leaks:
        raise SystemExit(f"RÒ RỈ trong pilot HTML: {leaks}")
    fp = out / "pilot_50.html"
    fp.write_text(txt, encoding="utf-8")

    plan = {
        "batch": "khoi_c_2026-09-16", "pilot": True, "seed": seed,
        "labels_path": str(labels_path.relative_to(REPO)), "labels_sha256": labels_sha,
        "batch_sha256_KHOA": batch_sha, "main_batch_sha256_KHOA": hashlib.sha256(main_khoa.read_bytes()).hexdigest(),
        "n_items_total": int(len(order)), "n_cells": int(len(allrows)), "n_repeat": int(len(reps)),
        "min_gap_repeat": PILOT_MIN_GAP, "repeat_gap_min_observed": int(min(gaps)) if gaps else None,
        "disjoint_from_main": True,
        "sessions": [{"session": 0, "session_id": PILOT_SESSION_ID, "html": fp.name, "n": int(len(order)),
                      "verdict_file": fname}],
        "tang": {str(k): int(v) for k, v in order["tang"].value_counts().sort_index().items()},
        "strata": cells,
        "crop_source": {k: int(v) for k, v in pd.Series([m["crop_source"] for m in meta.values()]).value_counts().items()},
        "blind_check": {"patterns": list(LEAK_PATTERNS), "leaks": leaks},
        "purpose": "C-1: đo dwell / κ nội tại / mồi trước khi chấm thật — KHÔNG trích precision",
    }
    (out / "plan_pilot.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = out / "README.md"
    if readme.exists() and "pilot_50.html" not in readme.read_text(encoding="utf-8"):
        with readme.open("a", encoding="utf-8") as fh:
            fh.write(f"""
## Pilot (C-1) — chấm TRƯỚC 4 phiên

`pilot_50.html` — {len(order)} lượt ({len(allrows)} ô + {len(reps)} lặp), ~5 phút. Xuất `{fname}` vào
chính thư mục này rồi chạy `.venv/bin/python -m pipeline.ground_truth.estimate_khoi_c --pilot` để xem
dwell / κ / mồi. Ô pilot KHÔNG trùng ô nào của 4 phiên chính. Hướng dẫn đầy đủ:
`docs/HUONG_DAN_CHAM_KHOI_C_2026-09-16.md`.
""")
    print(f"[C-pilot] {len(order)} lượt = {len(allrows)} ô + {len(reps)} lặp; gap lặp min={min(gaps) if gaps else '-'}; "
          f"tầng {plan['tang']}; crop {plan['crop_source']}; rò rỉ {leaks or 'không'}")
    print(f"[C-pilot] {fp.relative_to(REPO)} ({fp.stat().st_size / 1e6:.1f} MB); khoá {kp.relative_to(REPO)} "
          f"(sha256 {batch_sha[:16]}…)")
    return plan


def _readme(plan: dict) -> str:
    q1 = "\n".join(f"| **{k}** | **{lbl}** | {d} |" for k, _, lbl, d in Q1_CHOICES)
    q2 = "\n".join(f"| **{k}** | **{lbl}** | {d} |" for k, _, lbl, d in Q2_CHOICES)
    ses = "\n".join(f"| {s['session']} | `{s['html']}` | {s['n']} | `{s['verdict_file']}` |"
                    for s in plan["sessions"])
    n = plan["n_items_total"]
    return f"""# Khối C — mẻ chấm mù (16/09/2026)

**{n} ô** chia **{len(plan['sessions'])} phiên** (mỗi phiên một tệp HTML, mở offline, tiến độ tự
lưu trong trình duyệt). Ước tính **5–8 s/ô** → mỗi phiên ≈ 25–35 phút, cả mẻ ≈ 2,5 giờ.
Không chấm hơn một phiên liền — nghỉ giữa các phiên.

## Hướng dẫn 10 dòng

1. Mở `phien_1.html` bằng trình duyệt (Chrome/Safari). Chấm xong phiên 1 mới sang phiên 2.
2. Mỗi màn là MỘT ô, **hai pha**: pha 1 chỉ có crop phóng to · vị trí trên trang (khung đỏ) · ÂM hiển thị;
   **MÃ + glyph tham chiếu chỉ hiện SAU khi bạn trả lời Q1** (pha 2). Ô không có mã thì sau Q1 tự sang ô sau.
3. **Q1** (bắt buộc, trả lời KHI CHƯA THẤY MÃ): *Ô này cắt đúng MỘT chữ, và chữ đó đọc là ÂM hiển thị?* — phím **1·2·3·4**.
4. **Q2** (chỉ khi có mã, hiện sau Q1): *MÃ Unicode hiển thị có đúng chữ trên crop?* — phím **5·6·7**.
   Sau khi thấy mã bạn vẫn được đổi Q1, nhưng câu trả lời mù được giữ riêng và lần đổi được ghi lại.
5. Trả lời xong cả hai câu, tự chuyển ô sau. `←`/`→` xem lại; nút **Ô chưa chấm** nhảy tới ô còn trống.
6. Crop khó nhìn → nhìn khung đỏ trên ảnh trang. Vẫn không đủ căn cứ → **KHÔNG RÕ**, đừng đoán.
7. Đổi ý được; mọi lần đổi đều được ghi lại — đó là dữ liệu, không phải lỗi.
8. Chấm **hết** theo thứ tự, không bỏ ô khó (bỏ chọn lọc làm khoảng tin cậy mất hiệu lực).
9. Xong phiên → bấm **Xuất JSONL** → lưu tệp vào **chính thư mục này** (tên tệp có sẵn). Sau khi xuất, phiên bị khoá.
10. Không mở thư mục `_khoa/`. Không so ô này với ô khác, không tìm ô lặp.

## Q1 — ba lựa chọn có định nghĩa

| Phím | Lựa chọn | Định nghĩa |
|:---:|---|---|
{q1}

Ranh giới cần nhớ: mất **dưới 1/3** chữ hoặc dính **một chút mực** của chữ bên cạnh mà vẫn đọc ra
trọn chữ → **không** phải SAI CROP. Dị thể (viết khác nét nhưng cùng chữ, cùng âm) → **ĐÚNG**.

## Q2 — chỉ khi có mã

| Phím | Lựa chọn | Định nghĩa |
|:---:|---|---|
{q2}

## Phiên

| Phiên | Tệp | Số ô | Tệp xuất |
|:---:|---|---:|---|
{ses}

Số liệu thiết kế (tầng, cỡ mẫu, trọng số, sha256 bộ nhãn): `plan.json`. Ánh xạ ô → nguồn chỉ nằm
trong `_khoa/` và chỉ dùng ở bước ước lượng (`pipeline.ground_truth.estimate_khoi_c`, C-3).
"""


if __name__ == "__main__":
    raise SystemExit(main())
