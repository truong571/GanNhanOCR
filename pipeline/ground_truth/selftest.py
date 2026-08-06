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

from . import (
    audit_grid, estimate as est_mod, make_confusion_batch, make_gold_batch,
    make_retest_batch, s3_signals, sampling, stats, suspicion,
)

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
    # cross_col: BẤT BIẾN == 0. Lịch sử 1686 -> 8 (labels.csv 21/07) -> 0 (22/07).
    check("cross_col == 0 (dedup closed; hist 1686->8->0)", cross == 0, f"got {cross}")
    # similar_bridge: đo 3850 (lịch sử 3856) — 6 hàng bridge biến động theo lần tái sinh
    # labels.csv upstream (đây KHÔNG phải lớp dedup, chỉ là dao động nhỏ của census).
    check("similar_bridge == 3850 (hist 3856)", sim == 3850, f"got {sim}")
    # dup_defect union: BẤT BIẾN == 0 = union(dup_bbox=0, cross_col=0). Lịch sử 2321 -> 8 -> 0.
    # Dedup upstream đã đóng lớp trùng (REVIEW không có image, loại khỏi tập usable).
    dup_union = int(ranked["dup_defect"].sum())
    check("dup_defect union == 0 (dedup closed; hist 2321->8->0)", dup_union == 0,
          f"got {dup_union}")
    # CẤU TRÚC (độc lập thế hệ dữ liệu): union = |dup_bbox ∪ cross_col|.
    check("dup_defect union là hợp của 2 lớp con",
          max(dup_bbox, cross) <= dup_union <= dup_bbox + cross,
          f"union={dup_union} dup_bbox={dup_bbox} cross={cross}")

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
    # MỌI tầng rủi ro CÓ MẶT trong dân số phải được lấy vượt tỷ lệ. Không hard-code
    # 'dup_defect': lớp trùng đã đóng (N=0) nên phép so 0 > 0 là vô nghĩa, tự đỏ giả.
    bad = []
    for st, over in sampling.DEFAULT_OVERSAMPLE.items():
        n_pop = int((ranked["stratum"] == st).sum())
        if n_pop == 0 or over <= 1.0:
            continue                      # tầng rỗng: không có gì để lấy vượt tỷ lệ
        pop_share = n_pop / len(ranked)
        samp_share = float((s1["stratum"] == st).mean())
        if samp_share <= pop_share:
            bad.append(f"{st} {samp_share:.4f}<={pop_share:.4f}")
    check("mọi tầng rủi ro không rỗng đều được oversample", not bad, "; ".join(bad))
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
def _toy_labels(n: int = 12) -> pd.DataFrame:
    """Khung labels tối giản nhưng đủ cột cho add_suspicion."""
    return pd.DataFrame({
        "image": [f"gold/x_{i:03d}.png" for i in range(n)],
        "label": ["二"] * n,
        "unicode": ["U+4E8C"] * n,
        "syllable": ["nhị"] * n,
        "tier": ["GOLD"] * n,
        "rule": ["s1_inter_s2_direct"] * n,
        "book": ["stt2"] * n,
        "page": [f"{i:04d}" for i in range(n)],
        "column": ["c01"] * n,
        "bbox": ["[1,2,3,4]"] * n,
        "s3_cosine": [""] * n,
        "ink_pct": [0.3] * n,
        "crop_w": [50] * n,
        "crop_h": [50] * n,
        "image_md5": [f"{i:012x}" for i in range(n)],
        "split": ["train"] * n,
    })


def _toy_corpus(labels: pd.DataFrame, margins: list[float]) -> pd.DataFrame:
    n = len(labels)
    return pd.DataFrame({
        "image": labels["image"],
        "label": labels["label"],
        "tier": labels["tier"],
        "head_cos": [0.5] * n,
        "head_prob": [0.5] * n,
        "head_margin": margins,
        "head_isarg": [1.0 if m > 0 else 0.0 for m in margins],
        "bank_cos": [0.5 + 0.01 * i for i in range(n)],
        "mls": [1.0] * n,
    })


def test_s3_signals() -> None:
    print("[s3_signals]")
    lab = _toy_labels(12)
    margins = [float(i) - 5.0 for i in range(12)]
    n_disagree = sum(1 for m in margins if m <= 0)     # _toy_corpus: isarg=0 khi m <= 0
    corp = _toy_corpus(lab, margins)

    out, rep = s3_signals.attach(lab, corp)
    check("attach giữ nguyên số hàng", len(out) == len(lab), f"{len(out)} vs {len(lab)}")
    check("attach không sửa khung gốc", "s3_head_cos" not in lab.columns)
    check("đủ 6 cột tín hiệu", all(c in out.columns for c in s3_signals.SIGNAL_COLS))
    check("coverage = 100%", rep.coverage == 1.0, f"{rep.coverage}")
    check("head_disagree đúng số", int(out["s3_head_disagree"].sum()) == n_disagree,
          f"{int(out['s3_head_disagree'].sum())} vs {n_disagree}")

    # hàng KHÔNG có trong corpus -> không có tín hiệu, và KHÔNG bị coi là bất đồng
    part, rep2 = s3_signals.attach(lab, corp.iloc[:8])
    check("hàng ngoài corpus -> s3_signals_present False",
          int(part["s3_signals_present"].sum()) == 8,
          str(int(part["s3_signals_present"].sum())))
    check("NaN không bị tính là head_disagree",
          not part.loc[8:, "s3_head_disagree"].any())
    check("coverage bộ phận đúng", approx(rep2.coverage, 8 / 12, 1e-9), f"{rep2.coverage}")

    # nhãn ngoài từ vựng ArcFace: head_* là NaN nhưng bank_cos vẫn có -> VẪN là "đã chấm"
    oov = corp.copy()
    oov.loc[0, ["head_cos", "head_prob", "head_margin", "head_isarg"]] = np.nan
    out3, _ = s3_signals.attach(lab, oov)
    check("nhãn ngoài từ vựng vẫn tính là ĐÃ CHẤM",
          bool(out3.loc[0, "s3_signals_present"]) and not bool(out3.loc[0, "s3_head_present"]))

    # chống lệch thế hệ dữ liệu
    stale = corp.copy()
    stale.loc[:, "label"] = "三"
    try:
        s3_signals.attach(lab, stale)
        check("bắt được corpus lệch thế hệ", False, "không ném lỗi")
    except ValueError as e:
        check("bắt được corpus lệch thế hệ", "LỆCH THẾ HỆ" in str(e))
    out4, rep4 = s3_signals.attach(lab, stale, strict=False)
    check("strict=False vẫn chạy nhưng báo số lệch", rep4.label_mismatch == 12,
          str(rep4.label_mismatch))

    dup = pd.concat([corp, corp.iloc[:1]], ignore_index=True)
    try:
        s3_signals.attach(lab, dup)
        check("bắt được `image` trùng trong corpus", False, "không ném lỗi")
    except Exception as e:                                   # noqa: BLE001
        check("bắt được `image` trùng trong corpus", isinstance(e, (ValueError, Exception)))


def test_suspicion_backcompat(labels: pd.DataFrame) -> None:
    """Không có tín hiệu S3 -> add_suspicion phải cho kết quả Y HỆT trước đây."""
    print("[suspicion · tương thích ngược]")
    base = suspicion.add_suspicion(labels)
    check("không có tín hiệu -> không sinh tầng head_disagree",
          "head_disagree" not in set(base["stratum"]))
    check("không có tín hiệu -> cờ head_disagree toàn False",
          not base["head_disagree"].any())
    # s3_missing giữ nghĩa cũ: mọi hàng thiếu s3_cosine
    s3 = pd.to_numeric(base["s3_cosine"], errors="coerce")
    check("s3_missing == thiếu s3_cosine (nghĩa cũ)",
          int(base["s3_missing"].sum()) == int(s3.isna().sum()),
          f"{int(base['s3_missing'].sum())} vs {int(s3.isna().sum())}")

    lab = _toy_labels(12)
    margins = [float(i) - 5.0 for i in range(12)]
    n_disagree = sum(1 for m in margins if m <= 0)
    att, _ = s3_signals.attach(lab, _toy_corpus(lab, margins))
    withsig = suspicion.add_suspicion(att)
    check("có tín hiệu -> s3_missing về 0", int(withsig["s3_missing"].sum()) == 0,
          str(int(withsig["s3_missing"].sum())))
    check("có tín hiệu -> sinh tầng head_disagree",
          int((withsig["stratum"] == "head_disagree").sum()) == n_disagree,
          f"{int((withsig['stratum'] == 'head_disagree').sum())} vs {n_disagree}")
    check("head_disagree làm suspicion cao hơn",
          withsig.loc[withsig["head_disagree"], "suspicion"].min()
          > withsig.loc[~withsig["head_disagree"], "suspicion"].max())


def test_gold_batch() -> None:
    print("[make_gold_batch · mẻ hai tầng]")
    lab = _toy_labels(60)
    corp = _toy_corpus(lab, [float(i) - 30.0 for i in range(60)])
    att, _ = s3_signals.attach(lab, corp)
    gold = suspicion.add_suspicion(att)

    s = make_gold_batch.build_sample(gold, n_srs=20, n_active=10, seed=42)
    srs = s[s["audit_batch"] == make_gold_batch.BATCH_SRS]
    act = s[s["audit_batch"] == make_gold_batch.BATCH_ACTIVE]
    check("đúng kích thước 2 tầng", len(srs) == 20 and len(act) == 10,
          f"{len(srs)}/{len(act)}")
    check("hai tầng KHÔNG giao nhau", not set(srs["image"]) & set(act["image"]))
    check("item_id duy nhất", not s["item_id"].duplicated().any())
    check("design_weight SRS = N/n",
          approx(float(srs["design_weight"].iloc[0]), len(gold) / 20, 1e-9),
          str(srs["design_weight"].iloc[0]))
    check("design_weight tầng chủ đích phải RỖNG (chặn gộp nhầm)",
          bool(act["design_weight"].isna().all()))
    check("tầng chủ đích đúng là margin thấp nhất",
          float(act["s3_head_margin"].max()) <= float(srs["s3_head_margin"].min())
          or float(act["s3_head_margin"].mean()) < float(srs["s3_head_margin"].mean()),
          f"act max={act['s3_head_margin'].max()} srs min={srs['s3_head_margin'].min()}")
    check("audit_order liên tục 0..n-1",
          s["audit_order"].tolist() == list(range(len(s))))
    # LÀM MÙ: thứ tự hiển thị không được gom tầng thành khối
    seq = s.sort_values("audit_order")["audit_batch"].tolist()
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    check("thứ tự hiển thị đã trộn 2 tầng (không lộ tầng)", switches >= 5,
          f"chỉ {switches} lần đổi tầng")

    s2 = make_gold_batch.build_sample(gold, n_srs=20, n_active=10, seed=42)
    check("cùng seed -> cùng mẫu", s["item_id"].tolist() == s2["item_id"].tolist())
    s3_ = make_gold_batch.build_sample(gold, n_srs=20, n_active=10, seed=7)
    check("khác seed -> khác mẫu", s["item_id"].tolist() != s3_["item_id"].tolist())

    nosig = suspicion.add_suspicion(_toy_labels(60))
    try:
        make_gold_batch.build_sample(nosig, n_srs=20, n_active=10, seed=42)
        check("thiếu tín hiệu S3 -> báo lỗi rõ ràng", False, "không ném lỗi")
    except (ValueError, KeyError) as e:
        check("thiếu tín hiệu S3 -> báo lỗi rõ ràng", True, str(e)[:40])


def test_confusion_batch() -> None:
    print("[make_confusion_batch · mẻ lớp nhầm lẫn]")
    n = 90
    lab = _toy_labels(n)
    # 30 hàng đầu là lớp nghi vấn 奴/"nó", phần còn lại là chữ khác
    lab.loc[:29, "label"] = "奴"
    lab.loc[:29, "syllable"] = "nó"
    lab.loc[30:, "label"] = "三"
    lab.loc[30:, "syllable"] = "tam"
    gold = suspicion.add_suspicion(lab)

    excl = set(gold["image"].iloc[:5])          # 5 hàng lớp đã chấm ở mẻ trước
    s, info = make_confusion_batch.build_sample(
        gold, "奴", "nó", n_target=10, n_control=8, exclude=excl, seed=42)
    tgt = s[s["audit_batch"] == make_confusion_batch.BATCH_TARGET]
    ctl = s[s["audit_batch"] == make_confusion_batch.BATCH_CONTROL]

    check("đúng kích thước 2 nhóm", len(tgt) == 10 and len(ctl) == 8, f"{len(tgt)}/{len(ctl)}")
    check("mục tiêu toàn đúng cặp chữ–âm",
          set(tgt["label"]) == {"奴"} and set(tgt["syllable"]) == {"nó"})
    check("đối chứng KHÔNG lẫn hàng của lớp",
          not ((ctl["label"] == "奴") & (ctl["syllable"] == "nó")).any())
    check("loại đúng các ô đã chấm ở mẻ trước", not set(s["image"]) & excl)
    check("dân số lớp đếm CẢ hàng đã chấm", info["class_population"] == 30,
          str(info["class_population"]))
    check("dân số rút mẫu TRỪ hàng đã chấm", info["class_unaudited"] == 25,
          str(info["class_unaudited"]))
    check("design_weight mục tiêu = N_lớp_chưa_chấm / n",
          approx(float(tgt["design_weight"].iloc[0]), 25 / 10, 1e-9),
          str(tgt["design_weight"].iloc[0]))
    check("design_weight đối chứng = N_còn_lại / n",
          approx(float(ctl["design_weight"].iloc[0]), 60 / 8, 1e-9),
          str(ctl["design_weight"].iloc[0]))
    check("stratum = nhóm (để ước lượng phân tầng đúng dân số)",
          set(s["stratum"]) == {make_confusion_batch.BATCH_TARGET,
                                make_confusion_batch.BATCH_CONTROL})
    check("giữ lại thứ hạng rủi ro gốc ở risk_stratum", "risk_stratum" in s.columns)
    check("item_id duy nhất", not s["item_id"].duplicated().any())
    seq = s.sort_values("audit_order")["audit_batch"].tolist()
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    check("đã trộn 2 nhóm (chống hiệu ứng mỏ neo)", switches >= 4, f"chỉ {switches} lần đổi")

    s2, _ = make_confusion_batch.build_sample(
        gold, "奴", "nó", n_target=10, n_control=8, exclude=excl, seed=42)
    check("cùng seed -> cùng mẫu", s["item_id"].tolist() == s2["item_id"].tolist())
    try:
        make_confusion_batch.build_sample(gold, "龍", "long", 5, 5)
        check("lớp rỗng -> báo lỗi rõ", False, "không ném lỗi")
    except ValueError:
        check("lớp rỗng -> báo lỗi rõ", True)
    try:
        make_confusion_batch.build_sample(gold, "奴", "nó", n_target=999, n_control=5)
        check("xin nhiều hơn dân số -> báo lỗi rõ", False, "không ném lỗi")
    except ValueError:
        check("xin nhiều hơn dân số -> báo lỗi rõ", True)


def test_retest_batch() -> None:
    print("[make_retest_batch · kiểm tra lặp]")
    rng = np.random.default_rng(0)
    rows = []
    for b, n_c, n_w in [("batch_a", 40, 6), ("batch_b", 30, 10)]:
        for i in range(n_c):
            rows.append({"item_id": f"{b}c{i}", "image": f"gold/{b}_c{i}.png",
                         "orig_verdict": "correct", "orig_batch": b, "orig_group": "srs"})
        for i in range(n_w):
            v = ["wrong_label", "wrong_image", "unsure"][i % 3]
            rows.append({"item_id": f"{b}w{i}", "image": f"gold/{b}_w{i}.png",
                         "orig_verdict": v, "orig_batch": b, "orig_group": "srs"})
    graded = pd.DataFrame(rows)

    # chỉ có 16 ô không-correct, ngân sách muốn 20 -> phần thiếu dồn sang nhóm correct
    s = make_retest_batch.build_sample(graded, n_total=40, seed=42)
    check("trả đúng cỡ mẫu dù một nhóm thiếu hàng", len(s) == 40, str(len(s)))
    check("design_weight RỖNG toàn bộ (chặn dùng làm precision)",
          bool(s["design_weight"].isna().all()))
    n_nc = int(s["orig_verdict"].isin(make_retest_batch.NONCORRECT).sum())
    check("vét hết nhóm không-correct khi thiếu", n_nc == 16, str(n_nc))
    check("lấy vượt tỷ lệ nhóm không-correct so với dân số",
          n_nc / len(s) > 16 / len(graded), f"{n_nc/len(s):.2f} vs {16/len(graded):.2f}")
    check("có mặt CẢ HAI buổi chấm", s["orig_batch"].nunique() == 2,
          str(s["orig_batch"].unique()))
    check("không lặp ảnh trong mẻ", not s["image"].duplicated().any())
    check("giữ nguyên verdict cũ để đối chiếu", "orig_verdict" in s.columns)
    check("orig_verdict nằm trong danh sách trường ẩn của grid",
          "orig_verdict" in audit_grid._HIDDEN_FIELDS)
    seq = s.sort_values("audit_order")["orig_verdict"].tolist()
    switches = sum(1 for a, b in zip(seq, seq[1:]) if (a == "correct") != (b == "correct"))
    check("đã xáo trộn (không gom nhóm thành khối)", switches >= 8, f"chỉ {switches}")

    s2 = make_retest_batch.build_sample(graded, n_total=40, seed=42)
    check("cùng seed -> cùng mẫu", s["item_id"].tolist() == s2["item_id"].tolist())

    # nhóm lỗi ít hơn ngân sách -> vẫn chạy, chỉ lấy hết những gì có
    small = graded[graded["orig_verdict"].eq("correct") | graded["item_id"].str.contains("aw")]
    s3 = make_retest_batch.build_sample(small, n_total=40, seed=1)
    check("thiếu hàng lỗi vẫn rút được mẫu", len(s3) <= 40 and len(s3) > 0, str(len(s3)))


def test_crop_bleed() -> None:
    print("[crop_bleed · đo chất lượng crop bằng hình học]")
    import cv2
    from . import crop_bleed as CB

    # ảnh tổng hợp: thân chữ trong bbox + một mảnh TÁCH RỜI nằm ngoài bbox
    img = np.full((60, 40, 3), 255, np.uint8)
    img[20:40, 10:30] = 0            # thân, sẽ nằm trong bbox
    img[2:8, 12:28] = 0              # mảnh rời phía trên, ngoài bbox
    geom = CB.CropGeom(crop=img, x0=100, y0=200)
    own = [110, 220, 130, 240]       # bbox = vùng thân (toạ độ trang)

    det = CB.detached_fraction(geom, own)
    body, frag = 20 * 20, 6 * 16
    check("detached_frac = mảnh rời / tổng mực",
          approx(det, frag / (body + frag), 1e-6), f"{det:.4f}")

    # nét của CHÍNH chữ vươn ra ngoài bbox mà vẫn dính liền -> KHÔNG bị tính oan
    img2 = np.full((60, 40, 3), 255, np.uint8)
    img2[20:40, 10:30] = 0
    img2[14:20, 18:22] = 0           # đuôi nối liền thân, thò lên trên bbox
    det2 = CB.detached_fraction(CB.CropGeom(crop=img2, x0=100, y0=200), own)
    check("nét liền thò ra ngoài bbox KHÔNG bị tính là ngoại lai",
          approx(det2, 0.0, 1e-9), f"{det2:.4f}")

    # chữ cấu trúc ⿱: hai bộ phận rời nhau nhưng ĐỀU trong bbox -> không bị tính oan
    img3 = np.full((60, 40, 3), 255, np.uint8)
    img3[22:28, 12:28] = 0
    img3[32:38, 12:28] = 0
    det3 = CB.detached_fraction(CB.CropGeom(crop=img3, x0=100, y0=200), own)
    check("chữ ⿱ (2 bộ phận rời trong bbox) KHÔNG bị tính oan",
          approx(det3, 0.0, 1e-9), f"{det3:.4f}")

    # không còn mực nào trong bbox -> crop hỏng hoàn toàn
    img4 = np.full((60, 40, 3), 255, np.uint8)
    img4[2:8, 12:28] = 0
    det4 = CB.detached_fraction(CB.CropGeom(crop=img4, x0=100, y0=200), own)
    check("không còn mực trong bbox -> detached = 1.0", approx(det4, 1.0, 1e-9), f"{det4}")

    # bleed_fraction: mực nằm trong bbox của CHỮ KHÁC
    other = [100, 200, 140, 210]     # phủ hàng 0..10 của crop
    frac, n_f, n_tot = CB.bleed_fraction(geom, own, [other])
    check("bleed_frac = mực trong bbox chữ khác", n_f == 6 * 16 and n_tot == body + frag,
          f"{n_f}/{n_tot}")
    f0, _, _ = CB.bleed_fraction(geom, own, [])
    check("không có chữ khác -> bleed_frac = 0", approx(f0, 0.0, 1e-9))

    # BẤT BIẾN QUAN TRỌNG NHẤT: phép tái lập phải trùng md5 với crop đã lưu.
    # Nếu sai, mọi con số đo trên crop đều đo trên một phép cắt KHÁC.
    prep = REPO / "prepared"
    if LABELS.exists() and prep.exists():
        lab = pd.read_csv(LABELS, dtype={"image_md5": str})
        res = CB.measure_corpus(lab, prep, pad=0.12, limit_pages=3, progress=False)
        if len(res):
            check("tái lập crop khớp md5 100% (bằng chứng phép đo đúng khung)",
                  bool(res["md5_match"].all()),
                  f"{int(res['md5_match'].sum())}/{len(res)}")
            check("hàng không có ảnh bị loại khỏi phép đo",
                  bool(res["image"].map(lambda v: isinstance(v, str) and bool(v)).all()))


def test_label_only_mode() -> None:
    print("[audit_grid · chế độ chỉ-hỏi-nhãn]")
    mk = {"id": "a", "crop": "", "ref": "", "ctx": "", "label": "x", "syl": "s", "cands": ""}
    full = audit_grid._render_html([mk], "t", ["a"], "full")
    lab = audit_grid._render_html([mk], "t", ["a"], "label_only")
    check("chế độ full vẫn đủ 4 mức", full.count('"wrong_image"') == 1)
    check("chế độ label_only BỎ HẲN wrong_image", '"wrong_image"' not in lab)
    check("label_only vẫn giữ correct/wrong_label/unsure",
          all(f'"{v}"' in lab for v in ("correct", "wrong_label", "unsure")))
    check("số nút đúng bằng số lựa chọn",
          len(audit_grid.CHOICE_SETS["label_only"]) == 3
          and len(audit_grid.CHOICE_SETS["full"]) == 4)
    try:
        audit_grid._render_html([mk], "t", ["a"], "khong-ton-tai")
        check("mode sai -> báo lỗi rõ", False, "không ném lỗi")
    except ValueError:
        check("mode sai -> báo lỗi rõ", True)


def test_purposive_exclusion() -> None:
    """Mẫu chủ đích PHẢI bị loại khỏi precision — nếu không, số báo cáo sẽ thấp giả."""
    print("[estimate · loại mẫu chủ đích]")
    n_srs, n_act = 40, 20
    rows = []
    for i in range(n_srs):                       # tầng xác suất: 2 lỗi / 40
        rows.append({"item_id": f"s{i}", "verdict": "wrong_label" if i < 2 else "correct",
                     "design_weight": 100.0, "stratum": "gold_direct", "audit_batch": "srs"})
    for i in range(n_act):                       # tầng chủ đích: 10 lỗi / 20
        rows.append({"item_id": f"a{i}", "verdict": "wrong_label" if i < 10 else "correct",
                     "design_weight": np.nan, "stratum": "head_disagree",
                     "audit_batch": "active_lowmargin"})
    j = pd.DataFrame(rows)
    j["correct"] = (j["verdict"] == "correct").astype(int)
    j["is_wrong_label"] = (j["verdict"] == "wrong_label").astype(int)
    j["is_wrong_image"] = 0
    j["is_unsure"] = 0

    rep = est_mod.estimate(j, conf=0.95, p0=0.90, design="srs")
    check("chỉ chấm điểm trên tầng xác suất", rep.n_scored == n_srs, str(rep.n_scored))
    check("precision = 38/40, KHÔNG phải 48/60",
          approx(rep.precision, 38 / 40, 1e-9), f"{rep.precision:.4f}")
    check("báo rõ số ô bị loại", rep.n_purposive_excluded == n_act,
          str(rep.n_purposive_excluded))
    check("có thống kê mô tả cho tầng chủ đích",
          rep.purposive is not None and approx(rep.purposive["error_rate"], 0.5, 1e-9))
    pooled = 48 / 60
    check("nếu gộp nhầm thì precision sẽ thấp hơn hẳn", rep.precision > pooled,
          f"{rep.precision:.4f} vs gộp {pooled:.4f}")

    # tương thích ngược: mọi hàng đều có trọng số -> không loại gì
    j2 = j.copy()
    j2["design_weight"] = 100.0
    rep2 = est_mod.estimate(j2, conf=0.95, design="srs")
    check("toàn mẫu xác suất -> không loại hàng nào",
          rep2.n_purposive_excluded == 0 and rep2.n_scored == n_srs + n_act,
          f"{rep2.n_purposive_excluded}/{rep2.n_scored}")


def test_verdict_dir() -> None:
    """README hướng dẫn truyền cả THƯ MỤC verdicts — phải thật sự chạy được."""
    print("[estimate · nạp verdict từ thư mục]")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "verdicts_001.jsonl").write_text(
            '{"item_id":"a","verdict":"correct"}\n', encoding="utf-8")
        (d / "verdicts_002.jsonl").write_text(
            '{"item_id":"b","verdict":"wrong_label"}\n', encoding="utf-8")
        v = est_mod.load_verdicts(d)
        check("gộp nhiều file verdicts*.jsonl", len(v) == 2, str(len(v)))
        v1 = est_mod.load_verdicts(d / "verdicts_001.jsonl")
        check("vẫn nhận một file đơn lẻ", len(v1) == 1, str(len(v1)))
        try:
            est_mod.load_verdicts(Path(td) / "khong-ton-tai")
            check("thư mục rỗng -> báo lỗi rõ", False, "không ném lỗi")
        except (FileNotFoundError, OSError):
            check("thư mục rỗng -> báo lỗi rõ", True)


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 64)
    print("GROUND-TRUTH SELFTEST")
    print("=" * 64)
    test_stats()
    test_s3_signals()
    test_gold_batch()
    test_confusion_batch()
    test_retest_batch()
    test_crop_bleed()
    test_label_only_mode()
    test_purposive_exclusion()
    test_verdict_dir()
    if not LABELS.exists():
        print(f"[warn] {LABELS} not found — skipping data-dependent tests")
    else:
        labels = pd.read_csv(LABELS, dtype={"image_md5": str})
        test_suspicion_backcompat(labels)
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
