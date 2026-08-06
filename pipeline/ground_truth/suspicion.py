"""Per-crop suspicion ranking + risk stratification over the exported dataset.

Consumes dataset_out/labels.csv and produces, for every *usable* crop (tier in
GOLD/SILVER/SYLLABLE with a real image), a transparent risk score and a mutually
exclusive `stratum` label used to steer the human audit sample toward the strata a
crop-error is most likely to hide in.

The score is a heuristic AUDIT-PRIORITISATION prior, NOT a probability of error — the
human audit is what actually measures correctness. Every component is derived from a
signal already verified against the corpus in the 3-round evaluation:

  dup_defect        AE-1 (same bbox, same column) OR F1 (same md5, >1 column)  — provably
                    compromised duplicate crops (union census = 2,321 rows).
  similar_bridge    rule s1_inter_s2_similar — OCR char replaced by a look-alike.
  s3_low            recorded S3 cosine below the SILVER acceptance threshold.
  head_bank         SILVER rule s3_head_bank_consensus (measured median cosine ~0.32).
  quality_flag      aspect-ratio or ink-percentile outlier.
  s3_missing        rows where NO visual signal exists at all (baseline, mild).

Các thành phần dưới đây chỉ bật khi khung đã đi qua `s3_signals.attach()` (tức có
`dataset_out/fusion/s3_corpus.csv`); nếu không có, hàm chạy y hệt như trước:

  head_disagree     đầu ArcFace xếp một chữ KHÁC cao điểm hơn chính nhãn đang gán
                    (`head_isarg == 0`). Đây là bằng chứng mạnh nhất và KHÔNG cần
                    ngưỡng — không có số ma nào phải chọn.
  bank_low          `bank_cos` nằm trong nhóm thấp nhất theo PHÂN VỊ của chính khung.

CẢNH BÁO VỀ TRỌNG SỐ: hai trọng số `r_head_disagree` / `r_bank_low` là TIÊN NGHIỆM CHƯA
ĐƯỢC KIỂM ĐỊNH — sức phân biệt thật của tín hiệu đầu ArcFace trên nhãn NGƯỜI chưa từng
được đo (các số AUC đang lưu hành đều đo trên verdict do MÁY chấm). Chúng chỉ ảnh hưởng
tới việc mẫu audit được ưu tiên ra sao, KHÔNG bao giờ đổi `tier`/`label`. Bước B (sau khi
có verdict người) sẽ đo AUC thật rồi mới thay các số này.

LƯU Ý THANG ĐO: `bank_cos` KHÔNG cùng thang với cột `s3_cosine` của engine (đo trên 15.267
hàng có cả hai: tương quan 0,836 nhưng lệch tuyệt đối trung bình 0,13; trung vị 0,676 so
với 0,780). Vì vậy ngưỡng τ=0,62 của SILVER KHÔNG được tái sử dụng cho `bank_cos`; ở đây
chỉ dùng phân vị, giống cách `ink_pct` đã làm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["RiskConfig", "USABLE_TIERS", "STRATA", "add_suspicion"]

USABLE_TIERS = ("GOLD", "SILVER", "SYLLABLE")

# Stratum priority: the FIRST matching rule (top to bottom) wins, so strata are
# mutually exclusive and a row lands in its highest-risk applicable bucket.
STRATA = (
    "dup_defect",
    "head_disagree",          # chỉ xuất hiện khi khung có tín hiệu S3 corpus-wide
    "similar_bridge_lowcos",
    "similar_bridge",
    "silver_headbank",
    "s3_low",
    "quality_flag",
    "gold_direct",
    "other",
)


@dataclass(frozen=True)
class RiskConfig:
    tau_silver: float = 0.62          # SILVER acceptance threshold (consensus.py)
    s3_very_low: float = 0.50
    aspect_hi: float = 1.8            # crop_h / crop_w  (tall)
    aspect_lo: float = 0.55           # crop_h / crop_w  (wide)
    ink_lo_pct: float = 1.0           # percentile thresholds for ink_pct outliers
    ink_hi_pct: float = 99.0
    bank_cos_lo_pct: float = 10.0     # phân vị dưới của bank_cos -> cờ bank_low
    # noisy-OR risk contributions (documented priors, in [0,1])
    r_dup: float = 0.90
    r_similar_verylow: float = 0.70
    r_similar_low: float = 0.50
    r_similar: float = 0.30
    r_s3_verylow: float = 0.60
    r_s3_low: float = 0.35
    r_headbank: float = 0.40
    r_aspect: float = 0.20
    r_ink: float = 0.15
    r_s3_missing: float = 0.10
    # TIÊN NGHIỆM CHƯA KIỂM ĐỊNH — chỉ dùng để ưu tiên mẫu audit, không đổi nhãn.
    # Bước B đo AUC trên verdict NGƯỜI rồi mới chốt lại hai số này.
    r_head_disagree: float = 0.45
    r_bank_low: float = 0.25


def _f(series: pd.Series) -> pd.Series:
    """Parse a possibly-blank numeric column to float with NaN for blanks."""
    return pd.to_numeric(series, errors="coerce")


def add_suspicion(labels: pd.DataFrame, cfg: RiskConfig | None = None) -> pd.DataFrame:
    """Return usable rows augmented with risk-component flags, `suspicion`, `stratum`.

    Input must be the labels.csv frame. Output preserves the original index of the
    usable subset and adds columns; the input is not mutated.
    """
    cfg = cfg or RiskConfig()
    df = labels.copy()

    # --- restrict to usable crops with a real image ------------------------- #
    img = df["image"].fillna("").astype(str)
    usable = df["tier"].isin(USABLE_TIERS) & (img.str.len() > 0)
    df = df.loc[usable].copy()
    if df.empty:
        raise ValueError("no usable rows found (tier in GOLD/SILVER/SYLLABLE with image)")

    s3 = _f(df["s3_cosine"])
    df["s3_val"] = s3
    df["s3_present"] = s3.notna()

    # --- duplicate-crop defects (AE-1 same-col dup-bbox, F1 cross-col md5) --- #
    md5 = df["image_md5"].fillna("").astype(str)
    # AE-1: identical bbox appearing on >1 row within the same (book,page,column)
    grp_bbox = df.groupby(["book", "page", "column", "bbox"], dropna=False)["image"]
    df["dup_bbox"] = grp_bbox.transform("size") > 1
    # F1: same md5 spanning >1 distinct column within the same (book,page)
    cols_per_md5 = (
        df.assign(_md5=md5)
        .groupby(["book", "page", "_md5"], dropna=False)["column"]
        .transform("nunique")
    )
    df["cross_col"] = (md5.str.len() > 0) & (cols_per_md5 > 1)
    df["dup_defect"] = df["dup_bbox"] | df["cross_col"]

    # --- rule-based signals ------------------------------------------------- #
    rule = df["rule"].fillna("").astype(str)
    df["similar_bridge"] = rule.eq("s1_inter_s2_similar")
    df["head_bank"] = rule.eq("s3_head_bank_consensus")
    df["gold_direct"] = rule.eq("s1_inter_s2_direct")

    # --- S3 cosine risk ----------------------------------------------------- #
    df["s3_low"] = df["s3_present"] & (s3 < cfg.tau_silver)
    df["s3_verylow"] = df["s3_present"] & (s3 < cfg.s3_very_low)

    # --- tín hiệu S3 corpus-wide (chỉ có sau s3_signals.attach) -------------- #
    # Tách khỏi khối trên vì đây là ĐẠI LƯỢNG KHÁC, không cùng thang với s3_cosine.
    if "s3_head_isarg" in df.columns:
        isarg = _f(df["s3_head_isarg"])
        # Cờ hiện diện lấy từ s3_signals.attach nếu có: nó dựa trên việc hàng CÓ MẶT trong
        # corpus, không dựa trên head_* — vì nhãn ngoài từ vựng ArcFace vẫn có bank_cos/mls.
        if "s3_signals_present" in df.columns:
            df["s3_signals_present"] = df["s3_signals_present"].astype(bool)
        else:
            df["s3_signals_present"] = isarg.notna()
        # KHÔNG cần ngưỡng: đầu ArcFace xếp một chữ khác trên chính nhãn đang gán.
        df["head_disagree"] = isarg.notna() & (isarg == 0)
    else:
        df["s3_signals_present"] = False
        df["head_disagree"] = False

    if "s3_bank_cos" in df.columns:
        bank = _f(df["s3_bank_cos"])
        df["bank_cos_val"] = bank
        if bank.notna().any():
            bank_lo = float(np.nanpercentile(bank.to_numpy(dtype=float), cfg.bank_cos_lo_pct))
        else:
            bank_lo = -np.inf
        df["bank_low"] = bank.notna() & (bank < bank_lo)
    else:
        df["bank_cos_val"] = np.nan
        df["bank_low"] = False

    # s3_missing = KHÔNG có tín hiệu thị giác nào. Trước đây 94% GOLD rơi vào đây vì
    # engine bỏ qua S3 khi OCR khớp từ điển; nay s3_corpus đã chấm bù nên các hàng đó
    # KHÔNG còn mù nữa và không đáng bị tính rủi ro "chưa từng soi ảnh".
    df["s3_missing"] = ~df["s3_present"] & ~df["s3_signals_present"]

    # --- crop-quality flags ------------------------------------------------- #
    cw = _f(df["crop_w"]).replace(0, np.nan)
    ch = _f(df["crop_h"])
    aspect = ch / cw
    df["aspect"] = aspect
    df["aspect_out"] = (aspect > cfg.aspect_hi) | (aspect < cfg.aspect_lo)
    ink = _f(df["ink_pct"])
    if ink.notna().any():
        lo = np.nanpercentile(ink, cfg.ink_lo_pct)
        hi = np.nanpercentile(ink, cfg.ink_hi_pct)
    else:
        lo, hi = -np.inf, np.inf
    df["ink_out"] = ink.notna() & ((ink < lo) | (ink > hi))
    df["quality_flag"] = df["aspect_out"] | df["ink_out"]

    # --- noisy-OR suspicion score ------------------------------------------ #
    keep = np.ones(len(df), dtype=float)  # running product of (1 - r_i)

    def apply(mask: pd.Series | np.ndarray, r: float) -> None:
        nonlocal keep
        keep = keep * np.where(np.asarray(mask, dtype=bool), 1.0 - r, 1.0)

    apply(df["dup_defect"], cfg.r_dup)
    # similar-bridge risk depends on the S3 evidence available for that bridge
    sim = df["similar_bridge"].to_numpy()
    sv = df["s3_verylow"].to_numpy()
    sl = df["s3_low"].to_numpy()
    apply(sim & sv, cfg.r_similar_verylow)
    apply(sim & sl & ~sv, cfg.r_similar_low)
    apply(sim & ~sl, cfg.r_similar)
    # non-similar S3 risk
    nonsim = ~sim
    apply(nonsim & df["s3_verylow"].to_numpy(), cfg.r_s3_verylow)
    apply(nonsim & sl & ~df["s3_verylow"].to_numpy(), cfg.r_s3_low)
    apply(df["head_bank"], cfg.r_headbank)
    apply(df["aspect_out"], cfg.r_aspect)
    apply(df["ink_out"], cfg.r_ink)
    apply(df["s3_missing"] & ~df["dup_defect"], cfg.r_s3_missing)
    apply(df["head_disagree"], cfg.r_head_disagree)
    apply(df["bank_low"], cfg.r_bank_low)

    df["suspicion"] = 1.0 - keep

    # --- mutually-exclusive stratum (priority order) ------------------------ #
    stratum = np.full(len(df), "other", dtype=object)
    # assign from lowest to highest priority so higher priority overwrites
    stratum = np.where(df["gold_direct"].to_numpy(), "gold_direct", stratum)
    stratum = np.where(df["quality_flag"].to_numpy(), "quality_flag", stratum)
    stratum = np.where((nonsim & sl), "s3_low", stratum)
    stratum = np.where(df["head_bank"].to_numpy(), "silver_headbank", stratum)
    stratum = np.where(sim, "similar_bridge", stratum)
    stratum = np.where(sim & sl, "similar_bridge_lowcos", stratum)
    # head_disagree đứng trên mọi tầng dựa-trên-luật: đây là bằng chứng TRỰC TIẾP từ ảnh
    # rằng nhãn có thể sai, trong khi các tầng kia chỉ suy từ luật sinh nhãn.
    stratum = np.where(df["head_disagree"].to_numpy(), "head_disagree", stratum)
    stratum = np.where(df["dup_defect"].to_numpy(), "dup_defect", stratum)
    df["stratum"] = stratum

    # human-readable reason for the audit UI / logs
    df["risk_reason"] = _reasons(df)

    return df.sort_values("suspicion", ascending=False, kind="stable")


def _reasons(df: pd.DataFrame) -> pd.Series:
    parts = []
    for _, r in df.iterrows():
        tags = []
        if r["dup_bbox"]:
            tags.append("dup-bbox(AE-1)")
        if r["cross_col"]:
            tags.append("cross-col(F1)")
        if r["similar_bridge"]:
            tags.append("similar-bridge")
        if r["head_bank"]:
            tags.append("silver-headbank")
        if r["s3_present"] and r["s3_low"]:
            tags.append(f"s3={r['s3_val']:.2f}<τ")
        if r.get("head_disagree", False):
            tags.append("head-disagree")
        if r.get("bank_low", False):
            tags.append(f"bank={r['bank_cos_val']:.2f}(p10)")
        if r["aspect_out"]:
            tags.append(f"aspect={r['aspect']:.2f}")
        if r["ink_out"]:
            tags.append("ink-outlier")
        if not tags and r["s3_missing"]:
            tags.append("gold-direct(no-s3)")
        parts.append(",".join(tags) if tags else "clean")
    return pd.Series(parts, index=df.index)


def stratum_summary(ranked: pd.DataFrame) -> pd.DataFrame:
    """Population size + mean suspicion per stratum, ordered by STRATA priority."""
    g = (
        ranked.groupby("stratum")
        .agg(n=("image", "size"), mean_suspicion=("suspicion", "mean"))
        .reset_index()
    )
    order = {s: i for i, s in enumerate(STRATA)}
    g["_o"] = g["stratum"].map(order).fillna(len(STRATA))
    return g.sort_values("_o").drop(columns="_o").reset_index(drop=True)
