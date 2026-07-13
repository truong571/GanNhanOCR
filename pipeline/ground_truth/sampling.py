"""Deterministic audit sampling: stratified (for estimation) and SRS (for acceptance).

The stratified sampler over-samples known-risk strata to spend the human budget where
errors hide, and records a Horvitz–Thompson `design_weight` = N_h / n_h per row so the
population precision can be reweighted back without bias (see estimate.py).

All draws are seeded and reproducible: same (frame, seed) -> identical sample.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_OVERSAMPLE", "stratified_sample", "simple_random_sample"]

# Multiply the proportional allocation of these risk strata (reweighted out later).
DEFAULT_OVERSAMPLE = {
    "dup_defect": 3.0,
    "similar_bridge_lowcos": 3.0,
    "similar_bridge": 2.0,
    "silver_headbank": 2.0,
    "s3_low": 2.0,
    "quality_flag": 1.5,
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _stable_key(row_id: str, seed: int) -> int:
    """A per-row stable hash so item ordering does not leak the stratum."""
    h = hashlib.sha1(f"{seed}:{row_id}".encode()).hexdigest()
    return int(h[:12], 16)


def stratified_sample(
    ranked: pd.DataFrame,
    n_total: int,
    seed: int = 42,
    oversample: dict[str, float] | None = None,
    min_per_stratum: int = 8,
    id_col: str = "image",
) -> pd.DataFrame:
    """Draw ~n_total rows, allocated proportionally x oversample, seeded per stratum.

    Returns a copy of the sampled rows with added columns:
      item_id       stable blinded id (sha1 of id_col)
      design_weight N_h / n_h  (population weight for reweighting)
      audit_order   deterministic shuffled display order (hides stratum)
    Allocation is capped at each stratum's population size; small strata get at least
    min(min_per_stratum, N_h).
    """
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if id_col not in ranked.columns:
        raise ValueError(f"id_col {id_col!r} not in frame")
    if "stratum" not in ranked.columns:
        raise ValueError("frame must carry a 'stratum' column (run add_suspicion first)")
    over = dict(DEFAULT_OVERSAMPLE if oversample is None else oversample)

    sizes = ranked.groupby("stratum").size()
    weights = {s: n * over.get(s, 1.0) for s, n in sizes.items()}
    wsum = sum(weights.values())
    if wsum <= 0:
        raise ValueError("empty population")

    # initial proportional-with-oversample allocation
    alloc: dict[str, int] = {}
    for s, N_h in sizes.items():
        target = int(round(n_total * weights[s] / wsum))
        target = max(target, min(min_per_stratum, int(N_h)))
        alloc[s] = min(target, int(N_h))

    # reconcile the total back toward n_total (strata not yet exhausted absorb the delta)
    _reconcile(alloc, sizes.to_dict(), n_total)

    parts = []
    for s, n_h in alloc.items():
        if n_h <= 0:
            continue
        pool = ranked[ranked["stratum"] == s]
        idx = _rng(seed + _stable_key(s, seed) % 100000).choice(
            pool.index.to_numpy(), size=n_h, replace=False
        )
        chunk = ranked.loc[idx].copy()
        chunk["design_weight"] = float(sizes[s]) / float(n_h)
        parts.append(chunk)

    sample = pd.concat(parts).copy()
    sample["item_id"] = sample[id_col].map(
        lambda v: hashlib.sha1(str(v).encode()).hexdigest()[:16]
    )
    if sample["item_id"].duplicated().any():
        raise RuntimeError("item_id collision — id_col is not unique")
    # deterministic display order, independent of stratum, to preserve blinding
    order_key = sample["item_id"].map(lambda k: _stable_key(k, seed))
    sample = sample.assign(_ord=order_key).sort_values("_ord").drop(columns="_ord")
    sample = sample.reset_index(drop=False).rename(columns={"index": "source_row"})
    sample["audit_order"] = np.arange(len(sample))
    return sample


def _reconcile(alloc: dict[str, int], sizes: dict[str, int], n_total: int) -> None:
    """Nudge the allocation total to n_total, respecting per-stratum caps."""
    for _ in range(10000):
        cur = sum(alloc.values())
        if cur == n_total:
            return
        if cur < n_total:
            # add one to the stratum with the most remaining headroom
            cand = [s for s in alloc if alloc[s] < sizes[s]]
            if not cand:
                return
            s = max(cand, key=lambda s: sizes[s] - alloc[s])
            alloc[s] += 1
        else:
            # remove one from the largest allocation above its floor
            cand = [s for s in alloc if alloc[s] > 0]
            if not cand:
                return
            s = max(cand, key=lambda s: alloc[s])
            alloc[s] -= 1


def simple_random_sample(
    ranked: pd.DataFrame, n: int, seed: int = 42, id_col: str = "image"
) -> pd.DataFrame:
    """Unstratified SRS — the correct design for an acceptance-sampling claim.

    Every row has equal inclusion probability, so design_weight is constant (N/n) and
    a Clopper–Pearson bound on the sample applies directly to the population.
    """
    if n <= 0 or n > len(ranked):
        raise ValueError(f"need 0 < n <= population ({len(ranked)}), got {n}")
    idx = _rng(seed).choice(ranked.index.to_numpy(), size=n, replace=False)
    sample = ranked.loc[idx].copy()
    sample["design_weight"] = float(len(ranked)) / float(n)
    sample["item_id"] = sample[id_col].map(
        lambda v: hashlib.sha1(str(v).encode()).hexdigest()[:16]
    )
    order_key = sample["item_id"].map(lambda k: _stable_key(k, seed))
    sample = sample.assign(_ord=order_key).sort_values("_ord").drop(columns="_ord")
    sample = sample.reset_index(drop=False).rename(columns={"index": "source_row"})
    sample["audit_order"] = np.arange(len(sample))
    return sample
