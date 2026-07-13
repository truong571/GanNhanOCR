"""Effective independence of a set of votes — the honest thesis metric.

Pixel-correlated OCR votes (kim, qwen, nomnaocr all read the same crop) are NOT
independent, so counting them as three votes overstates the evidence. Following
"Nine Judges, Two Effective Votes" (arXiv:2605.29800), the number of *effective*
independent votes is the Kish design effect

    n_eff = m / (1 + (m - 1) * phi_bar)

where m is the vote count and phi_bar the mean pairwise error correlation. With
truth available we correlate error indicators (vote != truth); without truth we fall
back to disagreement correlation as a proxy. Report n_eff in the thesis so a panel
cannot object that "three models" were treated as three independent sources.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["NeffResult", "error_matrix", "mean_pairwise_phi", "kish_neff", "vote_neff"]


@dataclass
class NeffResult:
    m: int                       # number of votes
    n_eff: float                 # effective independent votes (Kish)
    phi_bar: float               # mean pairwise error correlation
    pairwise: dict               # {(a,b): phi}
    basis: str                   # 'error' (vs truth) or 'disagreement' (proxy)

    def summary(self) -> str:
        return (f"{self.m} votes -> n_eff={self.n_eff:.2f} "
                f"(phi_bar={self.phi_bar:.3f}, basis={self.basis})")


def error_matrix(votes: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
    """0/1 error indicator per (row, channel): 1 where the vote differs from truth.

    Rows where a channel abstained (NaN/empty vote) are left NaN and excluded pairwise.
    """
    err = {}
    t = truth.astype(str)
    for col in votes.columns:
        v = votes[col]
        present = v.notna() & (v.astype(str).str.len() > 0)
        e = (v.astype(str) != t).astype(float)
        e[~present] = np.nan
        err[col] = e
    return pd.DataFrame(err, index=votes.index)


def disagreement_matrix(votes: pd.DataFrame) -> pd.DataFrame:
    """Proxy when truth is absent: correlate each channel's disagreement with the
    per-row majority vote (1 = this channel dissents from the row majority)."""
    v = votes.astype("object")
    out = {}
    maj = []
    for _, row in v.iterrows():
        vals = [x for x in row if isinstance(x, str) and x] or \
               [x for x in row if pd.notna(x)]
        if vals:
            maj.append(pd.Series(vals).value_counts().idxmax())
        else:
            maj.append(np.nan)
    maj = pd.Series(maj, index=v.index)
    for col in v.columns:
        s = v[col]
        present = s.notna() & (s.astype(str).str.len() > 0)
        d = (s.astype(str) != maj.astype(str)).astype(float)
        d[~present] = np.nan
        out[col] = d
    return pd.DataFrame(out, index=v.index)


def mean_pairwise_phi(ind: pd.DataFrame) -> tuple[float, dict]:
    """Mean pairwise Pearson correlation of the 0/1 indicator columns.

    Zero-variance columns (a channel always right or always wrong on the overlap) yield
    an undefined correlation and are skipped for that pair. Returns (mean, {pair: phi}).
    """
    cols = list(ind.columns)
    pairs: dict = {}
    vals = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = ind[cols[i]], ind[cols[j]]
            mask = a.notna() & b.notna()
            if mask.sum() < 2:
                continue
            aa, bb = a[mask].to_numpy(), b[mask].to_numpy()
            if aa.std() == 0 or bb.std() == 0:
                phi = 0.0
            else:
                phi = float(np.corrcoef(aa, bb)[0, 1])
            pairs[(cols[i], cols[j])] = phi
            vals.append(phi)
    return (float(np.mean(vals)) if vals else 0.0), pairs


def kish_neff(m: int, phi_bar: float) -> float:
    """Effective independent votes n_eff = m / (1 + (m-1)*phi_bar), clamped to [1, m]."""
    if m <= 1:
        return float(m)
    denom = 1.0 + (m - 1) * phi_bar
    if denom <= 0:
        return float(m)                 # negative correlation -> super-efficient; cap at m
    return float(min(m, max(1.0, m / denom)))


def vote_neff(votes: pd.DataFrame, truth: pd.Series | None = None) -> NeffResult:
    """Effective independence of the OCR vote columns.

    votes: DataFrame with one column of discrete char predictions per channel.
    truth: optional gold labels; when given, uses error correlation (preferred),
           otherwise disagreement-from-majority as a proxy.
    """
    m = votes.shape[1]
    if truth is not None:
        ind = error_matrix(votes, truth)
        basis = "error"
    else:
        ind = disagreement_matrix(votes)
        basis = "disagreement"
    phi_bar, pairs = mean_pairwise_phi(ind)
    return NeffResult(m=m, n_eff=kish_neff(m, phi_bar), phi_bar=phi_bar,
                      pairwise=pairs, basis=basis)
