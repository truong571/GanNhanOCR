"""Turn human verdicts back into defensible precision numbers.

Joins verdicts.jsonl (from the audit tool) to the manifest, then reports:
  - sample precision with Wilson + Clopper–Pearson CIs
  - the acceptance decision for a "precision >= p0" claim (SRS designs)
  - the design-weighted / stratified population precision (stratified samples)
  - a PPI-tightened CI using S3 cosine as the surrogate
  - the wrong-label vs wrong-image breakdown (the two defect families)

`correct` for precision = verdict == "correct"; `unsure` rows are excluded from the
precision denominator by default (reported separately) so ambiguity never silently
counts as either right or wrong.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats

__all__ = ["load_verdicts", "join_manifest", "PrecisionReport", "estimate"]

VALID_VERDICTS = ("correct", "wrong_label", "wrong_image", "unsure")
_PPI_MIN_COVERAGE = 0.90   # surrogate must cover >=90% of both labelled + population


def load_verdicts(path: str | Path) -> pd.DataFrame:
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        if "item_id" not in d or "verdict" not in d:
            raise ValueError(f"verdict line missing keys: {ln}")
        if d["verdict"] not in VALID_VERDICTS:
            raise ValueError(f"unknown verdict {d['verdict']!r}")
        rows.append({"item_id": str(d["item_id"]), "verdict": d["verdict"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no verdicts loaded")
    if df["item_id"].duplicated().any():
        # keep the last verdict per item (auditor may revise)
        df = df.drop_duplicates("item_id", keep="last")
    return df


def load_manifest(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    df = pd.DataFrame(rows)
    if "item_id" not in df.columns:
        raise ValueError("manifest has no item_id")
    return df


def join_manifest(verdicts: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    j = verdicts.merge(manifest, on="item_id", how="left", validate="one_to_one")
    missing = j["stratum"].isna().sum() if "stratum" in j.columns else j.isna().all(axis=1).sum()
    if missing:
        raise ValueError(f"{missing} verdicts have no matching manifest row")
    j["correct"] = (j["verdict"] == "correct").astype(int)
    j["is_wrong_label"] = (j["verdict"] == "wrong_label").astype(int)
    j["is_wrong_image"] = (j["verdict"] == "wrong_image").astype(int)
    j["is_unsure"] = (j["verdict"] == "unsure").astype(int)
    return j


@dataclass
class PrecisionReport:
    n_audited: int
    n_scored: int              # excludes unsure
    n_correct: int
    n_wrong_label: int
    n_wrong_image: int
    n_unsure: int
    precision: float
    wilson_ci: tuple[float, float]
    cp_ci: tuple[float, float]
    cp_lower_one_sided: float
    weighted_precision: float | None
    weighted_ci: tuple[float, float] | None
    ppi_precision: float | None
    ppi_ci: tuple[float, float] | None
    ppi_note: str | None
    acceptance: dict | None
    per_stratum: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _weighted(joined: pd.DataFrame, conf: float) -> tuple[float, tuple[float, float]] | None:
    """Stratified population precision using population sizes implied by design_weight.

    design_weight = N_h / n_h, so N_h = design_weight * n_h (constant within a stratum).
    """
    if "stratum" not in joined.columns or "design_weight" not in joined.columns:
        return None
    scored = joined[joined["is_unsure"] == 0]
    if scored.empty:
        return None
    strata = []
    for s, g in scored.groupby("stratum"):
        n_h = len(g)
        w = g["design_weight"].dropna()
        if w.empty:
            continue
        N_h = int(round(float(w.iloc[0]) * n_h))
        k_h = int(g["correct"].sum())
        strata.append((max(N_h, n_h), n_h, k_h))
    if not strata:
        return None
    point, lo, hi = stats.stratified_mean_ci(strata, conf)
    return point, (lo, hi)


def estimate(
    joined: pd.DataFrame,
    conf: float = 0.95,
    p0: float | None = None,
    design: str = "stratified",
    surrogate_col: str = "s3_cosine",
    unlabeled_scores: np.ndarray | None = None,
) -> PrecisionReport:
    """Compute the full precision report from a joined verdict+manifest frame.

    design: 'srs' enables the acceptance decision (requires an SRS sample); any value
    enables the stratified/weighted estimate when design_weight is present.
    p0: target precision for the acceptance claim (e.g. 0.97).
    unlabeled_scores: surrogate scores over the un-audited usable population for PPI.
    """
    n_aud = len(joined)
    scored = joined[joined["is_unsure"] == 0]
    n_scored = len(scored)
    k = int(scored["correct"].sum())
    n_wl = int(joined["is_wrong_label"].sum())
    n_wi = int(joined["is_wrong_image"].sum())
    n_un = int(joined["is_unsure"].sum())

    if n_scored == 0:
        raise ValueError("no scorable verdicts (all unsure?)")
    precision = k / n_scored
    wci = stats.wilson_ci(k, n_scored, conf)
    cci = stats.clopper_pearson_ci(k, n_scored, conf)
    cp_low = stats.cp_lower_bound(k, n_scored, conf)

    weighted = _weighted(joined, conf)
    w_point = weighted[0] if weighted else None
    w_ci = weighted[1] if weighted else None

    # acceptance decision (defensible only for an SRS design)
    acceptance = None
    if p0 is not None:
        defects = n_scored - k
        lcb = cp_low
        acceptance = {
            "p0": p0,
            "defects": defects,
            "n": n_scored,
            "one_sided_lower_bound": lcb,
            "accept": bool(lcb >= p0),
            "design": design,
            "note": ("valid SRS acceptance claim" if design == "srs"
                     else "stratified sample — use weighted_precision, not this SRS bound"),
        }

    # PPI using the surrogate on labelled + unlabelled.
    # GUARD: PPI is only valid when the surrogate covers (nearly) the WHOLE population;
    # if it is missing on a biased subset (e.g. s3_cosine is blank on all GOLD-direct
    # rows) the unlabelled mean is taken over a skewed subset and the estimate is junk.
    # In that case we skip PPI honestly rather than emit a misleading number.
    ppi_p = ppi_ci = None
    ppi_note = None
    if surrogate_col in scored.columns and unlabeled_scores is not None:
        f_lab_all = pd.to_numeric(scored[surrogate_col], errors="coerce").to_numpy()
        y_all = scored["correct"].to_numpy(dtype=float)
        lab_mask = ~np.isnan(f_lab_all)
        fu_all = np.asarray(unlabeled_scores, dtype=float)
        fu = fu_all[~np.isnan(fu_all)]
        lab_cov = float(lab_mask.mean()) if lab_mask.size else 0.0
        unl_cov = (fu.size / fu_all.size) if fu_all.size else 0.0
        if lab_cov >= _PPI_MIN_COVERAGE and unl_cov >= _PPI_MIN_COVERAGE and fu.size >= 2:
            res = stats.ppi_mean_ci(y_all[lab_mask], f_lab_all[lab_mask], fu, conf)
            ppi_p, ppi_ci = res.theta, (res.lo, res.hi)
        else:
            ppi_note = (
                f"PPI skipped: surrogate {surrogate_col!r} coverage too low "
                f"(labelled {lab_cov:.0%}, population {unl_cov:.0%}; need "
                f">={_PPI_MIN_COVERAGE:.0%}). Provide a population-wide calibrated "
                f"surrogate (e.g. S3 head-logit over all crops) to enable PPI.")

    per_stratum = []
    if "stratum" in joined.columns:
        for s, g in joined.groupby("stratum"):
            gs = g[g["is_unsure"] == 0]
            if len(gs) == 0:
                continue
            kk = int(gs["correct"].sum())
            per_stratum.append({
                "stratum": s,
                "n": len(gs),
                "correct": kk,
                "precision": kk / len(gs),
                "wrong_label": int(g["is_wrong_label"].sum()),
                "wrong_image": int(g["is_wrong_image"].sum()),
                "unsure": int(g["is_unsure"].sum()),
            })

    return PrecisionReport(
        n_audited=n_aud, n_scored=n_scored, n_correct=k,
        n_wrong_label=n_wl, n_wrong_image=n_wi, n_unsure=n_un,
        precision=precision, wilson_ci=wci, cp_ci=cci, cp_lower_one_sided=cp_low,
        weighted_precision=w_point, weighted_ci=w_ci,
        ppi_precision=ppi_p, ppi_ci=ppi_ci, ppi_note=ppi_note,
        acceptance=acceptance, per_stratum=per_stratum,
    )
