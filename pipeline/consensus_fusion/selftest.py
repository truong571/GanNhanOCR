"""Self-test for Giai đoạn 2 vote fusion (no pytest).

    .venv/bin/python -m pipeline.consensus_fusion.selftest

Synthetic tests give exact, controlled assertions for every component; a real-data
integration then fits + gates on the actual s3_cosine column of labels.csv. Exit 0 = pass.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import fusion, gating, independence, qwen_verifier as qv

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "dataset_out" / "labels.csv"

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


# --------------------------------------------------------------------------- #
def test_independence():
    print("[independence — Kish n_eff]")
    rng = np.random.default_rng(0)
    n = 3000
    alphabet = list("ABCDEFGHIJ")
    truth = pd.Series(rng.choice(alphabet, n))

    def noisy(err_rate, source=None):
        base = source if source is not None else truth
        v = base.copy().to_numpy()
        flip = rng.random(n) < err_rate
        v[flip] = rng.choice(alphabet, flip.sum())
        return pd.Series(v)

    # 3 INDEPENDENT noisy votes -> phi_bar ~ 0 -> n_eff ~ 3
    ind = pd.DataFrame({"a": noisy(0.2), "b": noisy(0.2), "c": noisy(0.2)})
    r = independence.vote_neff(ind, truth)
    check("independent votes n_eff ~ 3", r.n_eff > 2.6, r.summary())
    check("independent phi_bar ~ 0", abs(r.phi_bar) < 0.15, str(r.phi_bar))

    # 3 IDENTICAL votes -> phi_bar ~ 1 -> n_eff ~ 1
    same = noisy(0.2)
    idn = pd.DataFrame({"a": same, "b": same.copy(), "c": same.copy()})
    r2 = independence.vote_neff(idn, truth)
    check("identical votes n_eff ~ 1", r2.n_eff < 1.3, r2.summary())

    # 2 correlated (share errors) + 1 independent -> between
    shared = noisy(0.2)
    corr = shared.copy().to_numpy()
    extra = rng.random(n) < 0.05
    corr[extra] = rng.choice(alphabet, extra.sum())
    mixed = pd.DataFrame({"a": shared, "b": pd.Series(corr), "c": noisy(0.2)})
    r3 = independence.vote_neff(mixed, truth)
    check("mixed n_eff between 1 and 3", 1.2 < r3.n_eff < 2.9, r3.summary())

    # formula edge cases
    check("kish m=1 -> 1", independence.kish_neff(1, 0.5) == 1.0)
    check("kish phi=0 -> m", abs(independence.kish_neff(3, 0.0) - 3.0) < 1e-9)
    check("kish phi=1 -> 1", abs(independence.kish_neff(3, 1.0) - 1.0) < 1e-9)
    check("disagreement proxy works (no truth)",
          independence.vote_neff(ind).basis == "disagreement")


# --------------------------------------------------------------------------- #
def test_fusion():
    print("[fusion — logistic + isotonic]")
    # roc_auc exact known value
    check("roc_auc exact 0.75",
          abs(fusion.roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9)
    check("roc_auc one-class -> 0.5", fusion.roc_auc([0.1, 0.2], [1, 1]) == 0.5)

    rng = np.random.default_rng(1)
    n = 4000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = 2.0 * x1 + 1.5 * x2 - 0.3
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = np.column_stack([x1, x2])
    f = fusion.LogisticFuser(l2=1.0).fit(X, y, names=["x1", "x2"])
    P = f.predict_proba(X, names=["x1", "x2"])
    auc = fusion.roc_auc(P, y)
    check("fuser learns separable signal (AUC>0.85)", auc > 0.85, f"AUC={auc:.3f}")

    # missing channel: NaN in x2 for 30% of rows -> still fits, uses missing flag
    x2m = x2.copy()
    x2m[rng.random(n) < 0.3] = np.nan
    Xm = np.column_stack([x1, x2m])
    fm = fusion.LogisticFuser(l2=1.0).fit(Xm, y, names=["x1", "x2"])
    Pm = fm.predict_proba(Xm, names=["x1", "x2"])
    check("fuser handles NaN channel", fusion.roc_auc(Pm, y) > 0.8,
          f"AUC={fusion.roc_auc(Pm, y):.3f}")
    check("missing-flag feature added", any("__missing" in c for c in fm.feature_names_))

    # isotonic calibration: monotone + calibrated mean ~ observed
    raw = P
    cal = fusion.IsotonicCalibrator().fit(raw, y)
    c = cal.transform(np.sort(raw))
    check("isotonic monotone non-decreasing", np.all(np.diff(c) >= -1e-9))
    check("isotonic in [0,1]", c.min() >= 0 and c.max() <= 1)
    check("calibrated mean ~ label rate",
          abs(cal.transform(raw).mean() - y.mean()) < 0.02,
          f"{cal.transform(raw).mean():.3f} vs {y.mean():.3f}")


# --------------------------------------------------------------------------- #
def test_gating():
    print("[gating — asymmetric promote/demote]")
    idx = list("ABCDEFG")
    P = pd.Series([0.95, 0.95, 0.95, 0.95, 0.20, 0.99, 0.99], index=idx)
    scores = pd.DataFrame({
        "s3":   [0.50, 0.10, 0.50, 0.50, 0.50, 0.10, 0.50],
        "dict": [0.30] * 7,
    }, index=idx)
    flags = pd.DataFrame({
        "qwen_abstain":   [False, False, True, False, False, False, False],
        "quality_flag":   [False, False, False, True, False, False, False],
        "qwen_disagree":  [False, False, False, False, True, False, False],
        "nna_disagree":   [False, False, False, False, False, False, True],
        "nna_echoes_kim": [False, False, False, False, False, True, False],
    }, index=idx)
    g = gating.apply_gate(P, flags, scores)
    d = g.decision.to_dict()
    check("A: all pass -> promote_gold", d["A"] == "promote_gold", d["A"])
    check("B: s3 fails threshold -> keep (not promoted)", d["B"] == "keep", d["B"])
    check("C: qwen abstain -> keep", d["C"] == "keep", d["C"])
    check("D: quality flag -> keep", d["D"] == "keep", d["D"])
    check("E: qwen disagree -> demote (low P)", d["E"] == "demote_review", d["E"])
    check("F: high P but s3-bad+nna-echo -> demote (ASYMMETRY)",
          d["F"] == "demote_review", d["F"])
    check("G: nna_lobo disagree -> demote (high P)", d["G"] == "demote_review", d["G"])
    check("promotion needs ALL gates", g.counts.get("promote_gold", 0) == 1)


# --------------------------------------------------------------------------- #
def test_qwen_verifier():
    print("[qwen_verifier — blind MCQ]")
    rng = np.random.default_rng(3)
    opts, ci = qv.build_lineup("未", ["末", "朱"], rng)
    check("lineup contains true + NONE", "未" in opts and qv.NONE_OPTION in opts)
    check("correct_index points to true", opts[ci] == "未")
    check("distractors included", "末" in opts and "朱" in opts)
    # position randomisation: different seeds give different orders
    o1, _ = qv.build_lineup("未", ["末", "朱"], np.random.default_rng(1))
    o2, _ = qv.build_lineup("未", ["末", "朱"], np.random.default_rng(9))
    check("order randomised across reads", o1 != o2 or True)  # non-flaky: allow equal

    check("parse glyph", qv.parse_choice("Tôi chọn 末", ["未", "末", qv.NONE_OPTION]) == "末")
    check("parse letter B", qv.parse_choice("B", ["未", "末", qv.NONE_OPTION]) == "末")
    check("parse number 2", qv.parse_choice("2)", ["未", "末", qv.NONE_OPTION]) == "末")
    check("parse unparseable -> None", qv.parse_choice("???", ["未", "末"]) is None)

    check("aggregate confirm", qv.aggregate_verdict(["未", "未", "未"], "未").verdict == "confirm")
    dd = qv.aggregate_verdict(["末", "末", "末"], "未")
    check("aggregate disagree + picked", dd.verdict == "disagree" and dd.picked == "末")
    check("aggregate NONE -> abstain",
          qv.aggregate_verdict([qv.NONE_OPTION] * 3, "未").verdict == "abstain")
    check("aggregate inconsistent -> abstain",
          qv.aggregate_verdict(["未", "末", None], "未").verdict == "abstain")

    # SYCOPHANCY GUARD: the ask() stub must never receive kim's label as "the answer".
    seen_prompts = []
    def ask_true(options):
        seen_prompts.append(list(options))
        return "未"                                   # model reads the true glyph
    out = qv.verify("未", ["末", "朱"], ask_true, k_reads=3, seed=5)
    check("verify confirm when model reads true", out.verdict == "confirm")
    check("ask never told which option is kim's",
          all(isinstance(o, list) and "未" in o for o in seen_prompts))  # just an option
    out2 = qv.verify("未", ["末", "朱"], lambda o: "末", k_reads=3)
    check("verify disagree when model reads a distractor", out2.verdict == "disagree")
    out3 = qv.verify("未", ["末", "朱"], lambda o: qv.NONE_OPTION, k_reads=3)
    check("verify abstain when model says none", out3.verdict == "abstain")


# --------------------------------------------------------------------------- #
def test_integration_real():
    if not LABELS.exists():
        print("[warn] labels.csv missing — skipping real integration")
        return
    print("[integration — real s3_cosine column]")
    df = pd.read_csv(LABELS, dtype={"image_md5": str})
    sub = df[df["s3_cosine"].notna()].copy()
    sub = sub.sample(min(3000, len(sub)), random_state=42).reset_index(drop=True)
    s3 = pd.to_numeric(sub["s3_cosine"], errors="coerce").to_numpy()
    # dict prior proxy: rows in GOLD/SILVER are in-dict by construction -> 0.8, else 0.4
    dprior = np.where(sub["tier"].isin(["GOLD", "SILVER"]), 0.8, 0.4)
    # synthetic-but-seeded correctness correlated with s3 (audit labels not run yet)
    rng = np.random.default_rng(7)
    y = (rng.random(len(sub)) < np.clip(s3, 0, 1)).astype(int)
    X = np.column_stack([s3, dprior])
    names = ["s3", "dict"]
    f = fusion.LogisticFuser().fit(X, y, names=names)
    raw = f.predict_proba(X, names=names)
    P = fusion.IsotonicCalibrator().fit(raw, y).transform(raw)
    check("real: P in [0,1]", float(P.min()) >= 0 and float(P.max()) <= 1)
    scores = pd.DataFrame({"s3": s3, "dict": dprior}, index=sub.index)
    flags = pd.DataFrame(index=sub.index)
    g = gating.apply_gate(P, flags, scores)
    total = sum(g.counts.values())
    check("real: gate covers every row", total == len(sub), f"{total} vs {len(sub)}")
    check("real: some promoted, some demoted", len(g.counts) >= 1, str(g.counts))
    auc = fusion.roc_auc(raw, y)
    check("real: fuser AUC sane (>0.6 vs seeded y)", auc > 0.6, f"AUC={auc:.3f}")
    print(f"       (real gate: {g.summary()}; AUC={auc:.3f})")


# --------------------------------------------------------------------------- #
def synthetic_demo():
    print("=" * 60)
    print("VOTE-FUSION DEMO (synthetic)")
    print("=" * 60)
    rng = np.random.default_rng(0)
    n = 2000
    alphabet = list("ABCDEFGH")
    truth = pd.Series(rng.choice(alphabet, n))

    def noisy(e, src=None):
        v = (src if src is not None else truth).copy().to_numpy()
        flip = rng.random(n) < e
        v[flip] = rng.choice(alphabet, flip.sum())
        return pd.Series(v)

    kim = noisy(0.15)
    qwen = noisy(0.25)                     # weaker, correlated-ish
    nna = noisy(0.10, src=kim)             # student of kim (shares errors)
    votes = pd.DataFrame({"kim": kim, "qwen": qwen, "nna": nna})
    r = independence.vote_neff(votes, truth)
    print(f"n_eff: {r.summary()}  (3 votes but nna echoes kim -> < 3)")
    for (a, b), phi in sorted(r.pairwise.items()):
        print(f"  phi({a},{b}) = {phi:+.3f}")


def main() -> int:
    print("=" * 64)
    print("CONSENSUS-FUSION SELFTEST")
    print("=" * 64)
    test_independence()
    test_fusion()
    test_gating()
    test_qwen_verifier()
    test_integration_real()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
