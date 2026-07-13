"""Phase-1 remediation of the committed dataset — the PROVEN fixes, in order.

Applies, strictly in the mandatory sequence (P0-C), deterministic corrections to
labels.csv without re-running the pipeline:

  1. QUARANTINE the AE-1 ∪ F1 duplicate-crop defects (one crop, >1 label). Conflicting
     groups are fully quarantined (no copy is trustworthy); identical-label duplicate
     groups keep one representative and quarantine the rest.
  2. DEMOTE similar-bridge GOLD rows whose recorded S3 cosine is below TAU_SILVER —
     the visual evidence already contradicts the look-alike substitution.
  3. DEDUP-BY-MD5 SPLIT: force every surviving image_md5 into a single split, closing
     the train/test pixel leak (P0-D), then ASSERT the invariant holds.

Every changed row is retagged in place (tier -> QUARANTINE, or GOLD -> REVIEW) and kept
in the file for auditability; nothing is silently deleted. Returns a change report.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import pandas as pd

from .census import USABLE_TIERS, annotate, run_census

__all__ = ["TAU_SILVER", "RemediationReport", "remediate"]

TAU_SILVER = 0.62          # SILVER acceptance threshold (consensus.py)
QUARANTINE_TIER = "QUARANTINE"
SIMILAR_RULE = "s1_inter_s2_similar"


@dataclass
class RemediationReport:
    n_rows: int
    census: dict
    quarantined_rows: int
    quarantined_conflict: int
    quarantined_duplicate: int
    kept_representatives: int
    demoted_similar_lowcos: int
    split_reassigned_rows: int
    md5_spanning_splits_original: int   # in the committed data, before any step
    md5_spanning_splits_residual: int   # after quarantine, before the dedup-split step
    md5_spanning_splits_after: int      # final (asserted 0)
    tier_before: dict = field(default_factory=dict)
    tier_after: dict = field(default_factory=dict)
    usable_before: int = 0
    usable_after: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"quarantined {self.quarantined_rows} "
            f"(conflict {self.quarantined_conflict}, dup {self.quarantined_duplicate}, "
            f"kept {self.kept_representatives}); "
            f"demoted {self.demoted_similar_lowcos} similar-bridge; "
            f"split-leak (original {self.md5_spanning_splits_original} md5 -> "
            f"final {self.md5_spanning_splits_after}); "
            f"usable {self.usable_before}->{self.usable_after}")


def _group_key(row_md5: str, book, page, column, bbox) -> str:
    """Stable defect-group key: md5 when present, else the geometric identity."""
    return row_md5 if row_md5 else f"{book}|{page}|{column}|{bbox}"


def remediate(df: pd.DataFrame, tau_silver: float = TAU_SILVER) -> tuple[pd.DataFrame, RemediationReport]:
    """Return (remediated_frame, report). Input is not mutated."""
    out = df.copy()
    for col in ("tier", "rule", "split", "image_md5", "label", "bbox"):
        if col not in out.columns:
            raise ValueError(f"labels frame missing required column {col!r}")

    tier_before = out["tier"].value_counts().to_dict()
    usable_before = int(out["tier"].isin(USABLE_TIERS).sum())
    census = run_census(out)

    # original train/test split leak in the committed usable data (P0-D)
    orig = out[out["tier"].isin(USABLE_TIERS)]
    orig_md5 = orig["image_md5"].fillna("").astype(str)
    spans_original = _count_md5_spanning_splits(orig[orig_md5.str.len() > 0],
                                                orig_md5[orig_md5.str.len() > 0])

    # ---- Step 1: QUARANTINE duplicate-crop defects (conflict->all, dup->keep one) ----
    u = annotate(out)  # usable rows with dup flags + dup_group (md5 for defects)
    defect = u[u["dup_defect"]].copy()
    md5 = defect["image_md5"].fillna("").astype(str)
    defect["_gkey"] = [
        _group_key(md5.iloc[i], r.book, r.page, r.column, r.bbox)
        for i, (_, r) in enumerate(defect.iterrows())
    ]

    quarantine_idx: list = []
    kept_reps = 0
    q_conflict = q_dup = 0
    for _, g in defect.groupby("_gkey", sort=False):
        labels = g["label"].fillna("").astype(str)
        if labels.nunique() > 1:
            quarantine_idx.extend(g.index.tolist())      # conflict -> quarantine all
            q_conflict += len(g)
        else:
            # identical label -> keep the deterministic-first, quarantine the rest
            rep = g["image"].astype(str).idxmin()
            rest = [i for i in g.index if i != rep]
            quarantine_idx.extend(rest)
            kept_reps += 1
            q_dup += len(rest)

    out.loc[quarantine_idx, "rule"] = (
        out.loc[quarantine_idx, "rule"].fillna("").astype(str) + "|quarantine_dup")
    out.loc[quarantine_idx, "tier"] = QUARANTINE_TIER

    # ---- Step 2: DEMOTE similar-bridge GOLD with S3 cosine below tau ----
    s3 = pd.to_numeric(out["s3_cosine"], errors="coerce")
    demote_mask = (
        (out["tier"] == "GOLD")
        & (out["rule"] == SIMILAR_RULE)
        & s3.notna()
        & (s3 < tau_silver)
    )
    demote_idx = out.index[demote_mask].tolist()
    out.loc[demote_idx, "rule"] = SIMILAR_RULE + "|demoted_lowcos_s3"
    out.loc[demote_idx, "tier"] = "REVIEW"

    # ---- Step 3: DEDUP-BY-MD5 SPLIT (close the train/test pixel leak) ----
    usable_mask = out["tier"].isin(USABLE_TIERS)
    um = out[usable_mask].copy()
    um_md5 = um["image_md5"].fillna("").astype(str)
    has_md5 = um_md5.str.len() > 0
    spans_before = _count_md5_spanning_splits(um[has_md5], um_md5[has_md5])

    reassigned = 0
    if "split" in out.columns:
        # canonical split per md5 = split of its deterministic-first row (by image name)
        work = um[has_md5].assign(_md5=um_md5[has_md5])
        first = (work.sort_values("image").groupby("_md5")["split"].first())
        new_split = work["_md5"].map(first)
        changed = new_split.values != work["split"].values
        idx_changed = work.index[changed]
        out.loc[idx_changed, "split"] = new_split[changed].values
        reassigned = int(changed.sum())

    # recompute invariant
    um2 = out[out["tier"].isin(USABLE_TIERS)]
    um2_md5 = um2["image_md5"].fillna("").astype(str)
    spans_after = _count_md5_spanning_splits(um2[um2_md5.str.len() > 0],
                                             um2_md5[um2_md5.str.len() > 0])

    # ---- assertions: the invariants Phase 1 promises ----
    assert spans_after == 0, f"md5 still spanning >1 split after dedup: {spans_after}"
    _assert_no_conflicting_usable_dupes(out)

    report = RemediationReport(
        n_rows=len(out),
        census=census.__dict__,
        quarantined_rows=len(quarantine_idx),
        quarantined_conflict=q_conflict,
        quarantined_duplicate=q_dup,
        kept_representatives=kept_reps,
        demoted_similar_lowcos=len(demote_idx),
        split_reassigned_rows=reassigned,
        md5_spanning_splits_original=spans_original,
        md5_spanning_splits_residual=spans_before,
        md5_spanning_splits_after=spans_after,
        tier_before={str(k): int(v) for k, v in tier_before.items()},
        tier_after={str(k): int(v) for k, v in out["tier"].value_counts().items()},
        usable_before=usable_before,
        usable_after=int(out["tier"].isin(USABLE_TIERS).sum()),
    )
    return out, report


def _count_md5_spanning_splits(df: pd.DataFrame, md5: pd.Series) -> int:
    if df.empty:
        return 0
    n = df.assign(_md5=md5).groupby("_md5")["split"].nunique()
    return int((n > 1).sum())


def _assert_no_conflicting_usable_dupes(df: pd.DataFrame) -> None:
    """No usable row may remain in a conflicting-label duplicate-crop group."""
    u = df[df["tier"].isin(USABLE_TIERS)]
    if u.empty:
        return
    ann = annotate(df)  # recompute on the full remediated frame
    ann = ann[ann["tier"].isin(USABLE_TIERS) & ann["dup_defect"]]
    if ann.empty:
        return
    md5 = ann["image_md5"].fillna("").astype(str)
    for _, g in ann.assign(_m=md5).groupby("_m"):
        labels = g["label"].fillna("").astype(str)
        assert labels.nunique() <= 1, (
            f"conflicting-label duplicate survived remediation for md5 group "
            f"(labels={sorted(labels.unique())})")
