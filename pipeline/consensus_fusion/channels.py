"""The five evidence channels of the go-live design (report §07).

Votes are weighted by DISTINCT EVIDENCE, not by count: each channel declares its role
and whether it is pixel-independent of the kim crop. The fuser (fusion.py) consumes the
numeric `score` columns; the gate (gating.py) consumes the roles + flags.

  kim        proposer   — provides the label + coords; never verifies itself
  qwen       verifier   — BLIND (never shown kim's label; sycophancy guard); may abstain
  nna_lobo   demote_only — leave-one-book-out NomNaOCR; its disagreement demotes, its
                           agreement adds nothing (student of kim's labels)
  s3         must_pass  — ArcFace head-logit; the ONLY input-independent channel; a GOLD
                           promotion REQUIRES it to pass
                           [LƯU Ý — vai must_pass đo vòng-3 ĐÃ BỊ BÁC (S3 AUC<=0.6 khi
                           bắt-lỗi thật; bank_cos=0.566); S3 hiện dùng như ranker/filter,
                           KHÔNG phải cổng must-pass — xem FLOW §7 /
                           docs/BANG_SO_LIEU_CHINH_THUC.md]
  dict       prior      — QN-dictionary plausibility; never promotes alone
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["Role", "ChannelSpec", "DEFAULT_CHANNELS", "ChannelTable"]


class Role:
    PROPOSER = "proposer"
    VERIFIER = "verifier"
    DEMOTE_ONLY = "demote_only"
    MUST_PASS = "must_pass"
    PRIOR = "prior"


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    role: str
    independent: bool          # pixel-independent of the kim crop?
    is_vote: bool = True       # produces a discrete char vote (for n_eff)?
    higher_is_better: bool = True


DEFAULT_CHANNELS = (
    ChannelSpec("kim", Role.PROPOSER, independent=False),
    ChannelSpec("qwen", Role.VERIFIER, independent=False),
    ChannelSpec("nna_lobo", Role.DEMOTE_ONLY, independent=False),
    ChannelSpec("s3", Role.MUST_PASS, independent=True, is_vote=False),
    ChannelSpec("dict", Role.PRIOR, independent=True, is_vote=False),
)


class ChannelTable:
    """Per-crop channel data aligned by crop_id.

    Holds three aligned frames:
      votes   discrete char prediction per voting channel (for n_eff)
      scores  numeric evidence per channel (for the fuser)   [NaN allowed]
      flags   per-crop booleans for gating (qwen_abstain, s3_pass, quality_flag, ...)
              [s3_pass ở đây là tên cờ theo thiết kế must-pass CŨ; must-pass đã bị bác,
               S3 dùng như ranker/filter — xem FLOW §7]
    """

    def __init__(self, specs=DEFAULT_CHANNELS):
        self.specs = list(specs)
        self.by_name = {s.name: s for s in self.specs}

    def build(self, votes: pd.DataFrame | None, scores: pd.DataFrame,
              flags: pd.DataFrame | None = None) -> "ChannelTable":
        if scores.index.duplicated().any():
            raise ValueError("scores index (crop_id) must be unique")
        self.scores = scores
        self.votes = votes if votes is not None else pd.DataFrame(index=scores.index)
        self.flags = flags if flags is not None else pd.DataFrame(index=scores.index)
        # align everything to the scores index
        self.votes = self.votes.reindex(scores.index)
        self.flags = self.flags.reindex(scores.index)
        return self

    def vote_columns(self) -> list[str]:
        return [s.name for s in self.specs if s.is_vote and s.name in self.votes.columns]

    def score_columns(self) -> list[str]:
        return [s.name for s in self.specs if s.name in self.scores.columns]

    def feature_matrix(self):
        """(X, names) for the fuser — numeric score columns in spec order."""
        cols = self.score_columns()
        return self.scores[cols].to_numpy(dtype=float), cols
