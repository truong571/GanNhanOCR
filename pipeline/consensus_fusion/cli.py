"""CLI for Giai đoạn 2 — vote fusion.

    # effective independence of the OCR votes
    .venv/bin/python -m pipeline.consensus_fusion neff --votes votes.csv --truth label

    # fit the calibrated fuser on audit labels, predict + gate all crops
    .venv/bin/python -m pipeline.consensus_fusion fuse \
        --features channels.csv --label-col y --score-cols s3,dict,qwen_agree,nna_agree

    # synthetic end-to-end demonstration (no data needed)
    .venv/bin/python -m pipeline.consensus_fusion demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import fusion, gating, independence

REPO = Path(__file__).resolve().parents[2]


def cmd_neff(args) -> None:
    df = pd.read_csv(args.votes, dtype=str)
    truth = df[args.truth] if args.truth and args.truth in df.columns else None
    vote_cols = [c for c in df.columns if c not in ({args.truth, args.id} if args.truth else {args.id})]
    votes = df[vote_cols]
    res = independence.vote_neff(votes, truth)
    print(f"[neff] {res.summary()}")
    for (a, b), phi in sorted(res.pairwise.items()):
        print(f"       phi({a},{b}) = {phi:+.3f}")


def cmd_fuse(args) -> None:
    df = pd.read_csv(args.features)
    score_cols = [c.strip() for c in args.score_cols.split(",") if c.strip()]
    for c in score_cols:
        if c not in df.columns:
            raise SystemExit(f"score column {c!r} not in {args.features}")
    y = pd.to_numeric(df[args.label_col], errors="coerce")
    labeled = y.notna()
    if labeled.sum() < 10:
        raise SystemExit(f"need >=10 labeled rows to fit; got {int(labeled.sum())}")
    X = df[score_cols].to_numpy(float)
    fuser = fusion.LogisticFuser(l2=args.l2).fit(X[labeled.to_numpy()],
                                                 y[labeled].to_numpy(), names=score_cols)
    raw = fuser.predict_proba(X, names=score_cols)
    cal = fusion.IsotonicCalibrator().fit(raw[labeled.to_numpy()], y[labeled].to_numpy())
    P = cal.transform(raw)
    auc = fusion.roc_auc(raw[labeled.to_numpy()], y[labeled].to_numpy())

    flags = df[[c for c in df.columns if c.startswith("flag_")]].rename(
        columns=lambda c: c[5:]) if any(c.startswith("flag_") for c in df.columns) else pd.DataFrame(index=df.index)
    scores = df[[c for c in ("s3", "dict") if c in df.columns]]
    gate = gating.apply_gate(P, flags.reindex(df.index).fillna(False), scores, gating.GateConfig(
        tau_promote=args.tau))
    out = df.copy()
    out["fused_P"] = P
    out["decision"] = gate.decision.values
    out["reason"] = gate.reason.values
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"[fuse] fit on {int(labeled.sum())} audit rows, train AUC={auc:.3f}")
    print(f"[fuse] gate: {gate.summary()}")
    print(f"[fuse] -> {outp}")


def cmd_demo(args) -> None:
    from .selftest import synthetic_demo
    synthetic_demo()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.consensus_fusion",
                                description="Giai đoạn 2 — vote fusion (SOTA)")
    sub = p.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("neff", help="effective independence (Kish n_eff) of votes")
    n.add_argument("--votes", required=True)
    n.add_argument("--truth", default=None, help="truth column (else disagreement proxy)")
    n.add_argument("--id", default="crop_id")
    n.set_defaults(func=cmd_neff)

    f = sub.add_parser("fuse", help="fit calibrated fuser on audit labels, predict + gate")
    f.add_argument("--features", required=True)
    f.add_argument("--label-col", default="y")
    f.add_argument("--score-cols", required=True)
    f.add_argument("--l2", type=float, default=1.0)
    f.add_argument("--tau", type=float, default=0.90)
    f.add_argument("--out", default=str(REPO / "dataset_out" / "fusion" / "fused.csv"))
    f.set_defaults(func=cmd_fuse)

    d = sub.add_parser("demo", help="synthetic end-to-end demonstration")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
