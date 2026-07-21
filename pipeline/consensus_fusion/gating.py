"""Asymmetric promote/demote gate — promotion is hard, demotion is easy.

Because pixel-correlated votes inflate false consensus, a GOLD promotion must clear
MULTIPLE channels while a demotion needs only ONE strong negative signal (report §07):

  promote_gold  iff  P >= tau_promote
                     AND s3 passes (must_pass channel)         <- the independent gate
                     AND qwen did not abstain
                     AND no crop-quality flag (dup/cross-col/aspect/ink)

  demote_review iff  qwen disagrees with a stable reading      (blind verifier says other)
                     OR s3 below floor while nna merely echoes kim
                     OR nna_lobo disagrees                       (informative dissent)
                     OR dict-implausible with no glyph support

[LƯU Ý — vai must_pass của S3 là thiết kế đo vòng-3 ĐÃ BỊ BÁC (S3 AUC<=0.6 khi bắt-lỗi
thật; bank_cos=0.566). S3 hiện dùng như RANKER/FILTER, KHÔNG phải cổng must-pass. Ngưỡng
s3_pass/s3_floor giữ nguyên giá trị theo code-freeze — xem FLOW §7 (FLOW_TONG_THE_CHOT)
và docs/BANG_SO_LIEU_CHINH_THUC.md.]

Everything else is 'keep'. A demoted GOLD crop is never lost — it drops to REVIEW for
human/adjudicator follow-up, so the recall cost of a strict S3 gate is just review work.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["GateConfig", "GateResult", "apply_gate"]


@dataclass
class GateConfig:
    tau_promote: float = 0.90      # calibrated P threshold for GOLD (set for >=99% prec)
    # [s3_pass/s3_floor: ngưỡng đo vòng-3 ĐÃ BỊ BÁC (S3 AUC<=0.6 khi bắt-lỗi thật);
    #  GIỮ NGUYÊN giá trị theo code-freeze; S3 dùng như ranker/filter, KHÔNG phải cổng
    #  must-pass — xem FLOW §7 / docs/BANG_SO_LIEU_CHINH_THUC.md]
    s3_pass: float = 0.29          # S3 head-logit operating point (85% recall / 3.5% FAR)
    s3_floor: float = 0.15         # below this S3 actively contradicts the label
    dict_floor: float = 0.05       # dict prior below this = implausible


@dataclass
class GateResult:
    decision: pd.Series            # 'promote_gold' | 'keep' | 'demote_review'
    reason: pd.Series
    counts: dict

    def summary(self) -> str:
        return " | ".join(f"{k}={v}" for k, v in self.counts.items())


def _col(df: pd.DataFrame, name: str, default):
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def apply_gate(P, flags: pd.DataFrame, scores: pd.DataFrame,
               cfg: GateConfig | None = None) -> GateResult:
    """Apply the asymmetric gate. Returns a decision + reason per crop.

    P      calibrated P(label correct), aligned to flags.index
    flags  booleans: qwen_abstain, quality_flag, qwen_disagree, nna_disagree, nna_echoes_kim
    scores numeric: s3, dict   (used for must-pass / floor checks; NaN treated as absent)
    """
    cfg = cfg or GateConfig()
    idx = flags.index
    P = pd.Series(np.asarray(P, float), index=idx)

    s3 = pd.to_numeric(_col(scores, "s3", np.nan), errors="coerce")
    dct = pd.to_numeric(_col(scores, "dict", np.nan), errors="coerce")
    qwen_abstain = _col(flags, "qwen_abstain", False).astype(bool)
    quality_flag = _col(flags, "quality_flag", False).astype(bool)
    qwen_disagree = _col(flags, "qwen_disagree", False).astype(bool)
    nna_disagree = _col(flags, "nna_disagree", False).astype(bool)
    nna_echoes_kim = _col(flags, "nna_echoes_kim", False).astype(bool)

    # S3 pass requires the score be PRESENT and above the operating point.
    s3_present = s3.notna()
    s3_ok = s3_present & (s3 >= cfg.s3_pass)
    s3_bad = s3_present & (s3 < cfg.s3_floor)
    dict_implausible = dct.notna() & (dct < cfg.dict_floor)

    decision = pd.Series("keep", index=idx, dtype=object)
    reason = pd.Series("", index=idx, dtype=object)

    # --- DEMOTE (any one strong negative) ---------------------------------- #
    demote = (
        qwen_disagree
        | (s3_bad & nna_echoes_kim)
        | nna_disagree
        | (dict_implausible & ~s3_ok)
    )
    demote_reason = np.select(
        [qwen_disagree, s3_bad & nna_echoes_kim, nna_disagree, dict_implausible & ~s3_ok],
        ["qwen_disagree", "s3_low+nna_echo", "nna_lobo_disagree", "dict_implausible"],
        default="",
    )
    decision[demote] = "demote_review"
    reason[demote] = demote_reason[demote.to_numpy()]

    # --- PROMOTE (must clear every gate) ----------------------------------- #
    promote = (
        ~demote
        & (P >= cfg.tau_promote)
        & s3_ok
        & ~qwen_abstain
        & ~quality_flag
    )
    decision[promote] = "promote_gold"
    reason[promote] = "multi_channel_pass"

    counts = decision.value_counts().to_dict()
    return GateResult(decision=decision, reason=reason,
                      counts={k: int(v) for k, v in counts.items()})
