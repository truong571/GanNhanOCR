"""Self-contained test suite for the ground-truth stage (no pytest dependency).

Run:  .venv/bin/python -m pipeline.ground_truth.selftest
It exercises the pure statistics against independent scipy computations and textbook
values, then runs the whole rank -> sample -> grid -> estimate flow on the real
labels.csv with synthetic verdicts, asserting every invariant. Exit code 0 = all pass.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, norm

from . import audit_grid, estimate as est_mod, sampling, stats, suspicion

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "dataset_out" / "labels.csv"

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
def test_stats() -> None:
    print("[stats]")
    # Wilson vs closed form (independent recompute)
    k, n, conf = 1116, 1150, 0.95
    z = norm.ppf(0.975)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = stats.wilson_ci(k, n, conf)
    check("wilson matches manual", approx(lo, centre - half) and approx(hi, centre + half),
          f"{lo},{hi} vs {centre-half},{centre+half}")
    check("wilson n=1150 k=1116 ~ [0.959,0.979]", 0.958 < lo < 0.961 and 0.978 < hi < 0.981,
          f"[{lo:.4f},{hi:.4f}]")

    # Clopper-Pearson vs scipy.beta directly
    lo2, hi2 = stats.clopper_pearson_ci(17, 100, 0.95)
    check("CP matches scipy.beta", approx(lo2, beta.ppf(0.025, 17, 84)) and
          approx(hi2, beta.ppf(0.975, 18, 83)))
    check("CP edge k=0 lower=0", stats.cp_lower_bound(0, 50) == 0.0)
    check("CP edge k=n upper=1", stats.cp_upper_bound(50, 50) == 1.0)
    check("CP one-sided lower = beta.ppf(alpha)", approx(
        stats.cp_lower_bound(833, 850, 0.95), beta.ppf(0.05, 833, 18)))

    # Acceptance plan reproduces the thesis figure n=850, c=17
    plan = stats.acceptance_plan(0.97, 0.95, p_assumed=0.985, power=0.90)
    check("acceptance n in [700,1000]", 700 <= plan.n <= 1000, f"n={plan.n}")
    check("acceptance LCB(c) >= p0", plan.lcb_at_c >= 0.97, f"lcb={plan.lcb_at_c}")
    # verify c is truly the max: c+1 defects must break the bound
    check("acceptance c is maximal",
          stats.cp_lower_bound(plan.n - (plan.c + 1), plan.n, 0.95) < 0.97)
    check("acceptance power matches binom.cdf",
          approx(plan.power, float(binom.cdf(plan.c, plan.n, 0.015)), 1e-9))
    print(f"       (plan: n={plan.n}, c={plan.c}, LCB={plan.lcb_at_c:.4f}, power={plan.power:.3f})")

    # Required n for half-width
    n01 = stats.required_n_for_halfwidth(0.97, 0.01, 0.95, "wilson")
    check("required_n +/-1% ~ 1140", 1100 <= n01 <= 1200, f"n={n01}")
    # monotone: tighter needs more
    n005 = stats.required_n_for_halfwidth(0.97, 0.005, 0.95, "wilson")
    check("tighter CI needs larger n", n005 > n01, f"{n005} vs {n01}")
    hw = (stats.wilson_ci(round(0.97 * n01), n01, 0.95)[1] -
          stats.wilson_ci(round(0.97 * n01), n01, 0.95)[0]) / 2
    check("required_n actually achieves half-width", hw <= 0.01 + 1e-9, f"hw={hw:.5f}")

    # PPI: perfect surrogate -> tiny interval; zero-signal surrogate ~ classical; both unbiased
    rng = np.random.default_rng(0)
    N = 5000
    y_all = (rng.random(N) < 0.9).astype(float)
    f_perfect = y_all.copy()
    idx = rng.choice(N, 300, replace=False)
    mask = np.ones(N, bool); mask[idx] = False
    r_perfect = stats.ppi_mean_ci(y_all[idx], f_perfect[idx], f_perfect[mask])
    ppi_w = r_perfect.hi - r_perfect.lo
    classical_w = r_perfect.classical_hi - r_perfect.classical_lo
    # perfect surrogate: PPI residual is only var(f)/N over the large unlabeled set,
    # so the interval must be dramatically tighter than the labelled-only classical one.
    check("PPI perfect surrogate >> tighter than classical", ppi_w < classical_w / 3,
          f"ppi={ppi_w:.4f} classical={classical_w:.4f}")
    f_noise = rng.random(N)
    r_noise = stats.ppi_mean_ci(y_all[idx], f_noise[idx], f_noise[mask])
    check("PPI unbiased (covers true 0.9)", r_noise.lo <= 0.9 <= r_noise.hi,
          f"[{r_noise.lo:.3f},{r_noise.hi:.3f}]")
    check("PPI classical also covers truth", r_noise.classical_lo <= 0.9 <= r_noise.classical_hi)

    # stratified estimator sanity: equal strata reduces to pooled proportion
    pt, slo, shi = stats.stratified_mean_ci([(1000, 100, 90), (1000, 100, 80)], 0.95)
    check("stratified point = mean of strata props", approx(pt, 0.85, 1e-9), f"pt={pt}")
    check("stratified CI ordered", slo < pt < shi)


# --------------------------------------------------------------------------- #
def test_suspicion(labels: pd.DataFrame) -> pd.DataFrame:
    print("[suspicion]")
    ranked = suspicion.add_suspicion(labels)
    check("only usable tiers", set(ranked["tier"]).issubset(set(suspicion.USABLE_TIERS)))
    check("all have image", (ranked["image"].astype(str).str.len() > 0).all())
    check("suspicion in [0,1]", ranked["suspicion"].between(0, 1).all())
    check("strata within STRATA set", set(ranked["stratum"]).issubset(set(suspicion.STRATA)))
    check("sorted descending", ranked["suspicion"].is_monotonic_decreasing)

    # cross-check against the census MEASURED on the CURRENT labels.csv.
    # Số lịch sử (thế hệ labels.csv TRƯỚC engine-fix + dedup upstream, không còn trên đĩa)
    # cao hơn hẳn; dedup đã đóng lớp trùng bbox/md5 nên các giá trị dưới đây tụt về mức
    # "đã sạch". Bảng before/after = bằng chứng engine-fix hoạt động: docs/census_history.md.
    dup_bbox = int(ranked["dup_bbox"].sum())
    cross = int(ranked["cross_col"].sum())
    sim = int(ranked["similar_bridge"].sum())
    # dup_bbox: đo 0 (lịch sử 701) — dedup upstream đã xoá mọi trùng-bbox cùng cột.
    check("dup_bbox == 0 (dedup closed; hist 701)", dup_bbox == 0, f"got {dup_bbox}")
    # cross_col: đo 8 (lịch sử 1686) = 4 nhóm xung đột nhãn còn sót cross-column.
    check("cross_col == 8 (dedup closed; hist 1686)", cross == 8, f"got {cross}")
    # similar_bridge: đo 3850 (lịch sử 3856) — 6 hàng bridge biến động theo lần tái sinh
    # labels.csv upstream (đây KHÔNG phải lớp dedup, chỉ là dao động nhỏ của census).
    check("similar_bridge == 3850 (hist 3856)", sim == 3850, f"got {sim}")
    # dup_defect union: đo 8 (lịch sử 2321) = đúng 8 hàng cross_col xung đột; dup_bbox=0
    # nên union == cross_col. Dedup upstream đã đóng lớp trùng (REVIEW không có image, loại).
    dup_union = int(ranked["dup_defect"].sum())
    check("dup_defect union == 8 (dedup closed; hist 2321)", dup_union == 8, f"got {dup_union}")

    # every dup_defect row must land in the top-priority stratum
    check("dup_defect -> stratum dup_defect",
          (ranked.loc[ranked["dup_defect"], "stratum"] == "dup_defect").all())
    # non-mutating
    check("input not mutated", "suspicion" not in labels.columns)
    print(suspicion.stratum_summary(ranked).to_string(index=False))
    return ranked


# --------------------------------------------------------------------------- #
def test_sampling(ranked: pd.DataFrame) -> pd.DataFrame:
    print("[sampling]")
    s1 = sampling.stratified_sample(ranked, 1150, seed=42)
    s2 = sampling.stratified_sample(ranked, 1150, seed=42)
    check("stratified deterministic", s1["item_id"].tolist() == s2["item_id"].tolist())
    check("stratified size ~ target", abs(len(s1) - 1150) <= 2, f"got {len(s1)}")
    check("unique item_id", not s1["item_id"].duplicated().any())
    check("design_weight positive", (s1["design_weight"] > 0).all())
    # HT check: sum of design weights ~ population size
    ht = s1["design_weight"].sum()
    check("sum(design_weight) ~ population", abs(ht - len(ranked)) / len(ranked) < 0.02,
          f"HT={ht:.0f} vs N={len(ranked)}")
    # risk strata are over-represented vs their population share
    pop_share = (ranked["stratum"] == "dup_defect").mean()
    samp_share = (s1["stratum"] == "dup_defect").mean()
    check("dup_defect oversampled", samp_share > pop_share, f"{samp_share:.3f} vs {pop_share:.3f}")
    seed_diff = sampling.stratified_sample(ranked, 1150, seed=7)
    check("different seed -> different sample",
          seed_diff["item_id"].tolist() != s1["item_id"].tolist())

    srs = sampling.simple_random_sample(ranked, 850, seed=42)
    check("srs size exact", len(srs) == 850)
    check("srs constant weight", srs["design_weight"].nunique() == 1)
    check("srs weight = N/n", approx(srs["design_weight"].iloc[0], len(ranked) / 850, 1e-6))
    return s1


# --------------------------------------------------------------------------- #
def test_grid(sample: pd.DataFrame) -> None:
    print("[grid]")
    small = sample.head(6).copy()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        res = audit_grid.build_audit(
            sample=small,
            dataset_dir=REPO / "dataset_out",
            prepared_dir=REPO / "prepared",
            fd_dir=REPO / "gannhanocr-fd",
            out_html=td / "audit.html",
            out_manifest=td / "manifest.jsonl",
            qn_dict=None,
            font_path=REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf",
            with_context=True,
        )
        html_text = (td / "audit.html").read_text(encoding="utf-8")
        manifest = [json.loads(l) for l in (td / "manifest.jsonl").read_text().splitlines() if l.strip()]
        check("grid produced items", res["items"] >= 1, str(res))
        check("manifest lines == items", len(manifest) == res["items"])
        check("html has verdict buttons", "wrong_label" in html_text and "wrong_image" in html_text)
        check("html embeds crops (data-uri)", "data:image/png;base64," in html_text)
        # BLINDING: tier / rule / stratum must not leak into the auditor-facing HTML
        leaked = [w for w in ("s1_inter_s2", "dup_defect", "GOLD", "SILVER", "stratum")
                  if w in html_text]
        check("html is blinded (no tier/rule/stratum)", not leaked, f"leaked={leaked}")
        # manifest DOES carry the hidden fields.
        # GUARD: khi thiếu ảnh crop (vd chạy trên clone sạch — crop bị gitignore),
        # grid trả items=0 -> manifest rỗng -> manifest[0] ném IndexError, làm CẢ suite
        # crash và KHÔNG in dòng RESULT. Hậu quả: 56 assertion còn lại biến mất khỏi
        # báo cáo và run_all_selftests.sh chuyển từ "FAIL có số" sang "không chạy được".
        # Biến kiểm định thành MÙ còn nguy hiểm hơn một assertion đỏ -> phải fail có số.
        if not manifest:
            check("manifest carries hidden fields", False,
                  "manifest RỖNG (grid ra 0 item — thiếu ảnh crop?), bỏ qua 2 assertion sau")
            check("manifest has design_weight", False, "manifest RỖNG")
        else:
            check("manifest carries hidden fields",
                  all(k in manifest[0] for k in ("tier", "rule", "stratum", "label")))
            check("manifest has design_weight", "design_weight" in manifest[0])


# --------------------------------------------------------------------------- #
def test_estimate(ranked: pd.DataFrame) -> None:
    print("[estimate]")
    # Build a controlled synthetic ground truth: 4% of the SRS sample are wrong.
    srs = sampling.simple_random_sample(ranked, 850, seed=123)
    rng = np.random.default_rng(1)
    verdict_rows, manifest_rows = [], []
    n_wrong = 0
    for _, r in srs.iterrows():
        # inject errors preferentially where suspicion is high (realistic)
        p_wrong = 0.02 + 0.10 * float(r["suspicion"])
        if rng.random() < p_wrong:
            v = "wrong_image" if rng.random() < 0.5 else "wrong_label"
            n_wrong += 1
        elif rng.random() < 0.03:
            v = "unsure"
        else:
            v = "correct"
        verdict_rows.append({"item_id": r["item_id"], "verdict": v, "ts": 1})
        # full-coverage synthetic surrogate correlated with correctness (for PPI path)
        base = 0.85 if v == "correct" else 0.45
        surr = float(min(1.0, max(0.0, base + rng.normal(0, 0.1))))
        m = {"item_id": r["item_id"], "stratum": r["stratum"],
             "design_weight": float(r["design_weight"]),
             "s3_cosine": r["s3_cosine"], "surrogate": surr,
             "tier": r["tier"], "rule": r["rule"], "label": r["label"]}
        manifest_rows.append(m)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "verdicts.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in verdict_rows), encoding="utf-8")
        (td / "manifest.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in manifest_rows), encoding="utf-8")
        v = est_mod.load_verdicts(td / "verdicts.jsonl")
        m = est_mod.load_manifest(td / "manifest.jsonl")
        joined = est_mod.join_manifest(v, m)
        check("join keeps all rows", len(joined) == len(srs))

        # (a) low-coverage surrogate (s3_cosine, blank on gold_direct) -> PPI SKIPPED
        unlabeled_s3 = pd.to_numeric(ranked["s3_cosine"], errors="coerce").to_numpy()
        rep = est_mod.estimate(joined, conf=0.95, p0=0.97, design="srs",
                               surrogate_col="s3_cosine", unlabeled_scores=unlabeled_s3)
        check("precision in (0,1)", 0 < rep.precision < 1, f"{rep.precision}")
        check("wilson brackets precision", rep.wilson_ci[0] <= rep.precision <= rep.wilson_ci[1])
        check("cp brackets precision", rep.cp_ci[0] <= rep.precision <= rep.cp_ci[1])
        check("scored + unsure == audited", rep.n_scored + rep.n_unsure == rep.n_audited)
        check("correct + wrong == scored",
              rep.n_correct + (rep.n_scored - rep.n_correct) == rep.n_scored)
        check("acceptance present for p0", rep.acceptance is not None)
        check("acceptance lower<=precision", rep.acceptance["one_sided_lower_bound"] <= rep.precision)
        check("low-coverage surrogate -> PPI skipped", rep.ppi_precision is None)
        check("PPI skip records a note", bool(rep.ppi_note))
        check("per_stratum non-empty", len(rep.per_stratum) > 0)

        # (b) full-coverage surrogate -> PPI computed and sane
        unl_full = np.clip(rng.normal(0.8, 0.15, size=len(ranked)), 0, 1)
        rep2 = est_mod.estimate(joined, conf=0.95, p0=0.97, design="srs",
                                surrogate_col="surrogate", unlabeled_scores=unl_full)
        check("full-coverage surrogate -> PPI computed", rep2.ppi_precision is not None)
        check("PPI brackets sane", rep2.ppi_ci[0] <= rep2.ppi_precision <= rep2.ppi_ci[1])
        # revised-verdict dedup
        dup = pd.concat([v, pd.DataFrame([{"item_id": v.iloc[0]["item_id"], "verdict": "unsure"}])])
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(json.dumps({"item_id": r.item_id, "verdict": r.verdict})
                              for r in dup.itertuples()))
            fp = f.name
        v2 = est_mod.load_verdicts(fp)
        check("revised verdict deduped", not v2["item_id"].duplicated().any())
        Path(fp).unlink()
        print(f"       (synthetic: precision={rep.precision:.4f}, injected_wrong={n_wrong}, "
              f"CP-LCB={rep.cp_lower_one_sided:.4f})")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 64)
    print("GROUND-TRUTH SELFTEST")
    print("=" * 64)
    test_stats()
    if not LABELS.exists():
        print(f"[warn] {LABELS} not found — skipping data-dependent tests")
    else:
        labels = pd.read_csv(LABELS, dtype={"image_md5": str})
        ranked = test_suspicion(labels)
        sample = test_sampling(ranked)
        test_grid(sample)
        test_estimate(ranked)
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
