"""Rigorous frequentist statistics for the ground-truth audit (Giai đoạn 0).

Pure functions, no side effects, verified against scipy in selftest.py. These are the
numbers that go into the thesis, so every interval here is a named, defensible method:

  - Wilson score interval          (recommended near-boundary proportions)
  - Clopper–Pearson "exact"        (conservative fallback reviewers accept without argument)
  - Acceptance-sampling plan (n,c) (one-sided lower-bound claim "precision >= p0")
  - Required-n for a target CI half-width
  - Prediction-Powered Inference   (PPI, Angelopoulos et al. 2023) using a surrogate score

No sklearn / no cleanlab dependency — only numpy + scipy.stats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, binom, norm

__all__ = [
    "wilson_ci",
    "wilson_lower",
    "clopper_pearson_ci",
    "cp_lower_bound",
    "cp_upper_bound",
    "required_n_for_halfwidth",
    "AcceptancePlan",
    "acceptance_plan",
    "PPIResult",
    "ppi_mean_ci",
    "stratified_mean_ci",
]


# --------------------------------------------------------------------------- #
# Binomial proportion intervals
# --------------------------------------------------------------------------- #
def _z(conf: float, two_sided: bool = True) -> float:
    """Normal quantile for a confidence level."""
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0,1), got {conf}")
    alpha = 1.0 - conf
    return float(norm.ppf(1.0 - (alpha / 2.0 if two_sided else alpha)))


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion k/n.

    Returns (lo, hi), both clamped to [0, 1]. n must be > 0.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    z = _z(conf, two_sided=True)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_lower(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Wilson lower confidence bound for k/n."""
    if n <= 0:
        raise ValueError("n must be > 0")
    z = _z(conf, two_sided=False)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, centre - half)


def clopper_pearson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper–Pearson (exact, Beta-quantile) interval for k/n."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def cp_lower_bound(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper–Pearson lower bound on the success proportion k/n."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if k == 0:
        return 0.0
    alpha = 1.0 - conf
    return float(beta.ppf(alpha, k, n - k + 1))


def cp_upper_bound(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper–Pearson upper bound on the success proportion k/n."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if k == n:
        return 1.0
    alpha = 1.0 - conf
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


# --------------------------------------------------------------------------- #
# Sample-size planning
# --------------------------------------------------------------------------- #
def required_n_for_halfwidth(
    p: float, halfwidth: float, conf: float = 0.95, method: str = "wilson"
) -> int:
    """Smallest n whose two-sided CI half-width at proportion p is <= halfwidth.

    method: 'wald' (closed form), 'wilson', or 'cp' (searched, exact width at k=round(p*n)).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    if not 0.0 < halfwidth < 1.0:
        raise ValueError("halfwidth must be in (0,1)")
    z = _z(conf, two_sided=True)
    if method == "wald":
        return int(np.ceil(z * z * p * (1 - p) / (halfwidth * halfwidth)))

    if method not in ("wilson", "cp"):
        raise ValueError("method must be 'wald', 'wilson', or 'cp'")

    def half_at(n: int) -> float:
        k = int(round(p * n))
        k = min(max(k, 0), n)
        lo, hi = (wilson_ci(k, n, conf) if method == "wilson"
                  else clopper_pearson_ci(k, n, conf))
        return (hi - lo) / 2.0

    # Wald gives a good starting point; expand a window around it and binary-search.
    n0 = max(2, int(np.ceil(z * z * p * (1 - p) / (halfwidth * halfwidth))))
    lo_n, hi_n = 2, max(4, n0 * 3)
    while half_at(hi_n) > halfwidth:
        hi_n *= 2
        if hi_n > 10_000_000:
            raise RuntimeError("required_n did not converge")
    while lo_n < hi_n:
        mid = (lo_n + hi_n) // 2
        if half_at(mid) <= halfwidth:
            hi_n = mid
        else:
            lo_n = mid + 1
    return lo_n


@dataclass
class AcceptancePlan:
    """A one-sided acceptance-sampling plan.

    Draw `n` items; ACCEPT the claim "precision >= p0" iff observed defects <= `c`.
    Guarantee: accepting with <= c defects implies a Clopper–Pearson one-sided lower
    bound on precision >= p0 at confidence `conf` (this is `lcb_at_c`).
    """
    n: int
    c: int
    p0: float
    conf: float
    lcb_at_c: float          # CP lower bound on precision when exactly c defects seen
    p_assumed: float | None  # true precision assumed for the power calc
    power: float | None      # P(accept | precision == p_assumed)


def _max_c_for_n(n: int, p0: float, conf: float) -> tuple[int, float]:
    """Largest defect count c such that CP-LCB(successes = n-c) >= p0. (-1 if none.)"""
    best_c, best_lcb = -1, 0.0
    # defects from 0 upward; LCB decreases monotonically as defects rise, so stop early.
    for d in range(0, n + 1):
        lcb = cp_lower_bound(n - d, n, conf)
        if lcb >= p0:
            best_c, best_lcb = d, lcb
        else:
            break
    return best_c, best_lcb


def acceptance_plan(
    p0: float,
    conf: float = 0.95,
    p_assumed: float | None = None,
    power: float = 0.90,
    n_min: int = 10,
    n_max: int = 20000,
) -> AcceptancePlan:
    """Design an acceptance-sampling plan for the claim "precision >= p0".

    If p_assumed is given, returns the SMALLEST n whose plan reaches `power` =
    P(accept | true precision == p_assumed). Otherwise returns the smallest n that
    admits any acceptance number c >= 0 (i.e. a zero-defect plan is feasible).
    """
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0,1)")
    if p_assumed is not None and p_assumed <= p0:
        raise ValueError("p_assumed must exceed p0 for a meaningful power calc")

    for n in range(n_min, n_max + 1):
        c, lcb = _max_c_for_n(n, p0, conf)
        if c < 0:
            continue
        if p_assumed is None:
            return AcceptancePlan(n, c, p0, conf, lcb, None, None)
        achieved = float(binom.cdf(c, n, 1.0 - p_assumed))
        if achieved >= power:
            return AcceptancePlan(n, c, p0, conf, lcb, p_assumed, achieved)
    raise RuntimeError(f"no plan found up to n_max={n_max}")


# --------------------------------------------------------------------------- #
# Prediction-Powered Inference (PPI) for a mean
# --------------------------------------------------------------------------- #
@dataclass
class PPIResult:
    theta: float             # PPI point estimate of the mean (e.g. precision)
    lo: float
    hi: float
    classical_theta: float   # mean of labelled y only
    classical_lo: float
    classical_hi: float
    n_labeled: int
    n_unlabeled: int


def ppi_mean_ci(
    y_labeled: np.ndarray,
    f_labeled: np.ndarray,
    f_unlabeled: np.ndarray,
    conf: float = 0.95,
    clip01: bool = True,
) -> PPIResult:
    """Prediction-powered CI for E[Y] using surrogate predictions f.

    y_labeled   : observed outcomes on the audited items (0/1 correctness).
    f_labeled   : surrogate score on the SAME audited items.
    f_unlabeled : surrogate score on the unaudited population.

    Valid for ANY surrogate f (the rectifier removes f's bias); it merely tightens
    the interval when f correlates with y. Ref: Angelopoulos, Bates, Fannjiang,
    Jordan, Zrnic, "Prediction-Powered Inference", Science 2023.
    """
    y = np.asarray(y_labeled, dtype=float)
    fl = np.asarray(f_labeled, dtype=float)
    fu = np.asarray(f_unlabeled, dtype=float)
    n = y.size
    N = fu.size
    if n == 0:
        raise ValueError("need at least one labelled point")
    if fl.size != n:
        raise ValueError("y_labeled and f_labeled must have equal length")
    z = _z(conf, two_sided=True)

    rectifier = y - fl                         # bias correction on labelled set
    theta = (fu.mean() if N > 0 else fl.mean()) + rectifier.mean()

    var_rect = rectifier.var(ddof=1) / n if n > 1 else rectifier.var() / max(n, 1)
    var_f = (fu.var(ddof=1) / N) if N > 1 else 0.0
    se = float(np.sqrt(var_rect + var_f))
    lo, hi = theta - z * se, theta + z * se

    # Classical: labelled-only Wald on the mean of y.
    cse = float(np.sqrt(y.var(ddof=1) / n)) if n > 1 else 0.0
    clo, chi = y.mean() - z * cse, y.mean() + z * cse

    if clip01:
        theta = min(1.0, max(0.0, theta))
        lo, hi = max(0.0, lo), min(1.0, hi)
        clo, chi = max(0.0, clo), min(1.0, chi)
    return PPIResult(theta, lo, hi, float(y.mean()), clo, chi, n, N)


# --------------------------------------------------------------------------- #
# Stratified estimator (for design-weighted population precision)
# --------------------------------------------------------------------------- #
def stratified_mean_ci(
    strata: list[tuple[int, int, int]], conf: float = 0.95
) -> tuple[float, float, float]:
    """Stratified estimate of a population proportion with FPC.

    strata: list of (N_h population size, n_h sample size, k_h successes in sample).
    Returns (point, lo, hi). Uses the standard stratified-mean variance with the
    finite-population correction; normal CI.
    """
    N = sum(N_h for N_h, _, _ in strata)
    if N <= 0:
        raise ValueError("total population must be > 0")
    point = 0.0
    var = 0.0
    for N_h, n_h, k_h in strata:
        if n_h <= 0 or N_h <= 0:
            continue
        p_h = k_h / n_h
        w_h = N_h / N
        point += w_h * p_h
        if n_h >= 2:
            s2 = p_h * (1 - p_h) * n_h / (n_h - 1)   # sample variance of a 0/1 var
        else:
            s2 = 0.0
        fpc = max(0.0, 1.0 - n_h / N_h)
        var += (w_h ** 2) * fpc * s2 / n_h
    z = _z(conf, two_sided=True)
    half = z * float(np.sqrt(var))
    return (point, max(0.0, point - half), min(1.0, point + half))
