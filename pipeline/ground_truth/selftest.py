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
    audit_grid, estimate as est_mod, estimate_khoi_c as ekc, make_confusion_batch, make_gold_batch,
    make_khoi_c_batch as kc, make_retest_batch, report_combined, s3_signals, sampling,
    stats, suspicion,
)

REPO = Path(__file__).resolve().parents[2]
LABELS = (
    REPO / "dataset_out" / "labels_final.csv"
    if (REPO / "dataset_out" / "labels_final.csv").exists()
    else REPO / "dataset_out" / "labels.csv"
)

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
    # similar_bridge: PHỤ THUỘC DỮ LIỆU — đổi sau MỖI lần bộ nhãn đổi.
    # Lịch sử: 3856 -> 3850 -> 4098 (gộp SinoNom_Similar_Đạt_v0, 2026-08-19)
    #          -> 4100 (T1 2026-08-24: chuẩn hoá dấu phụ TRƯỚC align)
    #          -> 4102 (T2 2026-08-24: sửa dò cột min_len + fallback bóc marker)
    #          -> 4127 (2026-08-25: cứu 3 trang bị ghi đè vì trùng số trang in;
    #                   ngữ liệu 445 -> 448 trang, mọi con số theo đó nhích lên)
    # HAI PHÉP KIỂM, hai mục đích khác nhau:
    #   (a) BĂNG rộng — bắt SỤP THẬT (luật similar hỏng, từ điển tự dạng không nạp
    #       được). Phép này BỀN, không phải sửa khi dữ liệu nhích.
    #   (b) MỐC chính xác — canary bắt "đổi dữ liệu mà quên cập nhật tài liệu". Hỏng ở
    #       đây là VIỆC BẢO TRÌ, không phải lỗi mã: đối chiếu với bộ nhãn rồi cập nhật.
    check("similar_bridge trong băng lành mạnh 2500-4700 (bắt sụp thật)",
          2500 <= sim <= 4700, f"got {sim}")
    check("similar_bridge == 2964 (v3) / 4127 (legacy) — MỐC dữ liệu, cập nhật khi bộ nhãn đổi "
          "(hist 3856->3850->4098->T1 4100->T2 4102->448 trang->v3 2964)",
          sim in (4127, 2964), f"got {sim}")
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
            check("manifest KHÔNG có labels_sha256 khi không truyền labels_path",
                  "labels_sha256" not in manifest[0])
        # A-14: labels_path -> sha256 của tệp nhãn đã rút mẫu ghi vào TỪNG dòng manifest
        # (bộ đem đo = bộ đem nộp, so được với CHECKSUMS.txt bằng máy)
        lp = td / "labels_fake.csv"
        lp.write_text("image,tier\nx.png,GOLD\n", encoding="utf-8")
        import hashlib
        want = hashlib.sha256(lp.read_bytes()).hexdigest()
        res2 = audit_grid.build_audit(
            sample=small, dataset_dir=REPO / "dataset_out", prepared_dir=REPO / "prepared",
            fd_dir=REPO / "gannhanocr-fd", out_html=td / "audit2.html",
            out_manifest=td / "manifest2.jsonl", qn_dict=None,
            font_path=REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf",
            with_context=False, labels_path=lp)
        man2 = [json.loads(l) for l in (td / "manifest2.jsonl").read_text().splitlines() if l.strip()]
        check("labels_sha256 trả về đúng sha256 của labels_path", res2["labels_sha256"] == want)
        check("mọi dòng manifest mang labels_sha256 + labels_path",
              bool(man2) and all(m.get("labels_sha256") == want and m.get("labels_path") == str(lp)
                                 for m in man2))
        try:
            audit_grid.build_audit(
                sample=small, dataset_dir=REPO / "dataset_out", prepared_dir=REPO / "prepared",
                fd_dir=REPO / "gannhanocr-fd", out_html=td / "audit3.html",
                out_manifest=td / "manifest3.jsonl", with_context=False,
                labels_path=td / "khong_ton_tai.csv")
            check("labels_path không tồn tại -> FileNotFoundError", False)
        except FileNotFoundError:
            check("labels_path không tồn tại -> FileNotFoundError", True)


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
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def _combined_fixture(d: Path) -> None:
    """Mẻ gộp tổng hợp, mọi con số chọn sao cho tính tay ra được.

    Bốn tầng khác dân số và khác tỷ lệ lỗi (để lộ ra ngay nếu quên trọng số), một tier
    toàn "không đọc được", hai ô chưa chấm, và 12 ô lặp có ma trận (8,1,1,2) -> κ = 5/9.
    """
    man: list[dict] = []
    ver: list[dict] = []

    def add(iid: str, stratum: str, N_h: int, n_h: int, verdict: str | None) -> None:
        man.append({"item_id": iid, "stratum": stratum, "stratum_N": N_h,
                    "design_weight": round(N_h / n_h, 4), "image": f"{iid}.png"})
        if verdict:
            ver.append({"item_id": iid, "verdict": verdict})

    for i in range(20):        # GOLD|stt2 — N=1.000: 3 sai nhãn, 1 không đọc được, 16 đúng
        add(f"g{i}", "GOLD|stt2", 1000, 20,
            "wrong_label" if i < 3 else ("unsure" if i == 3 else "correct"))
    for i in range(20):        # GOLD|stt4 — N=3.000: đúng hết (tầng nặng, sạch)
        add(f"h{i}", "GOLD|stt4", 3000, 20, "correct")
    for i in range(10):        # SILVER|stt2 — N=500: 10 ô nhưng CHỈ 8 ô được chấm
        add(f"s{i}", "SILVER|stt2", 500, 10,
            None if i >= 8 else ("wrong_label" if i == 0 else "correct"))
    for i in range(5):         # SYLLABLE|stt2 — toàn "không đọc được"
        add(f"y{i}", "SYLLABLE|stt2", 200, 5, "unsure")

    repeats = ([(f"g{i}", "correct") for i in range(4, 12)]      # đúng -> đúng  ×8
               + [("g12", "wrong_label")]                        # đúng -> sai   ×1
               + [("g0", "correct")]                             # sai  -> đúng  ×1
               + [("g1", "wrong_label"), ("g2", "wrong_label")])  # sai  -> sai   ×2
    for i, (orig, v) in enumerate(repeats):
        man.append({"item_id": f"r{i}", "stratum": "__repeat__", "design_weight": None,
                    "repeat_of": orig, "image": f"{orig}.png"})
        ver.append({"item_id": f"r{i}", "verdict": v})
    # ô lặp trỏ về ô chưa hề được chấm -> không ghép cặp được, phải báo riêng
    man.append({"item_id": "r99", "stratum": "__repeat__", "design_weight": None,
                "repeat_of": "khong-co-that", "image": "x.png"})
    ver.append({"item_id": "r99", "verdict": "correct"})

    _write_jsonl(d / "manifest.jsonl", man)
    _write_jsonl(d / "verdicts.jsonl", ver)
    (d / "plan.json").write_text(json.dumps(
        {"tiers": [{"tier": "GOLD", "N": 4200, "N_unaudited": 4000}]}), encoding="utf-8")


def test_report_combined() -> None:
    """Module sinh BẢNG HEADLINE của luận văn — sai ở đây là sai thẳng vào kết quả."""
    print("[report_combined · precision theo tier + κ nội tại]")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _combined_fixture(d)
        rep = report_combined.build(d, conf=0.95)
        tiers = {t["tier"]: t for t in rep["tiers"]}

        check("tách đúng ba tier, ô lặp không thành 'tier' thứ tư",
              set(tiers) == {"GOLD", "SILVER", "SYLLABLE"}, str(sorted(tiers)))

        g = tiers["GOLD"]
        check("ô lặp KHÔNG lọt vào precision của tier",
              g["n_audited"] == 40 and g["n_scored"] == 39,
              f"n_audited={g['n_audited']} n_scored={g['n_scored']}")
        check("GOLD precision = 36/39", approx(g["precision"], 36 / 39, 1e-12),
              f"{g['precision']:.6f}")
        check("'không đọc được' bị loại khỏi mẫu số nhưng vẫn báo riêng",
              g["n_unsure"] == 1 and g["n_wrong_label"] == 3,
              f"unsure={g['n_unsure']} wrong={g['n_wrong_label']}")

        w_expect = (1000 * (16 / 19) + 3000 * 1.0) / 4000
        check("precision có trọng số lấy N_h từ stratum_N, không phải trung bình cộng",
              approx(g["weighted_precision"], w_expect, 1e-9)
              and abs(g["weighted_precision"] - g["precision"]) > 0.01,
              f"{g['weighted_precision']:.6f} vs kỳ vọng {w_expect:.6f}")
        check("population_N cộng từ các tầng của tier", g["population_N"] == 4000,
              str(g.get("population_N")))

        lcb = float(beta.ppf(0.05, 36, 39 - 36 + 1))
        check("cận dưới một phía khớp Clopper–Pearson tính độc lập",
              approx(g["cp_lower_one_sided"], lcb, 1e-9),
              f"{g['cp_lower_one_sided']:.6f} vs {lcb:.6f}")
        acc = g["acceptance"]
        check("ngưỡng GOLD mặc định 0,97 và mẻ này CHƯA đạt",
              acc["p0"] == 0.97 and acc["accept"] is False and acc["defects"] == 3,
              str(acc))
        g_low = {t["tier"]: t for t in
                 report_combined.build(d, p0={"GOLD": 0.5})["tiers"]}["GOLD"]
        check("hạ p0 xuống dưới cận dưới thì kết luận đảo thành ĐẠT",
              g_low["acceptance"]["accept"] is True)

        s = tiers["SILVER"]
        check("ô chưa chấm không âm thầm được tính là đúng",
              s["n_scored"] == 8 and approx(s["precision"], 7 / 8, 1e-12),
              f"n={s['n_scored']} p={s.get('precision')}")
        check("số ô CHƯA chấm được báo ra", rep["n_ungraded"] == 2, str(rep["n_ungraded"]))
        check("n_items_in_batch đếm cả ô lặp", rep["n_items_in_batch"] == 68,
              str(rep["n_items_in_batch"]))

        y = tiers["SYLLABLE"]
        check("tier toàn 'không đọc được' -> KHÔNG bịa ra precision",
              "precision" not in y and y["n_unsure"] == 5, str(y))

        ov = rep["overall_usable"]
        ov_expect = (1000 * (16 / 19) + 3000 * 1.0 + 500 * (7 / 8)) / 4500
        check("toàn tập gộp bằng Horvitz–Thompson trên MỌI tầng của MỌI tier",
              approx(ov["weighted_precision"], ov_expect, 1e-9)
              and ov["population_N"] == 4500,
              f"{ov['weighted_precision']:.6f} vs {ov_expect:.6f}, N={ov['population_N']}")

        rel = rep["reliability"]
        check("κ nội tại khớp giá trị tính tay 5/9", approx(rel["kappa"], 5 / 9, 1e-9),
              f"{rel['kappa']:.6f}")
        check("đồng thuận thô = 10/12 (báo cạnh κ, không để κ đứng một mình)",
              approx(rel["observed_agreement"], 10 / 12, 1e-12))
        check("ma trận đảo verdict đúng chiều lần 1 -> lần 2",
              rel["matrix"].get("correct->wrong_label") == 1
              and rel["matrix"].get("wrong_label->correct") == 1
              and rel["matrix"].get("correct->correct") == 8, str(rel["matrix"]))
        check("ô lặp không ghép được đếm riêng, không lặng lẽ biến mất",
              rel["n_repeat_items"] == 13 and rel["n_repeat_unmatched"] == 1,
              f"{rel['n_repeat_items']}/{rel['n_repeat_unmatched']}")
        check("κ báo kèm theo từng tier", approx(rel["per_tier"]["GOLD"]["kappa"], 5 / 9, 1e-9))

        check("khung rút mẫu đọc từ plan.json (để nói đúng phạm vi suy rộng)",
              bool(rep["sampling_frame"]) and rep["sampling_frame"][0]["N_frame"] == 4000)

        md = report_combined._md(rep)
        check("bảng Markdown dựng được, đủ ba tier + κ",
              all(t in md for t in ("GOLD", "SILVER", "SYLLABLE")) and "Cohen's κ" in md)
        check("tier không có precision vẫn render, không vỡ bảng", "| SYLLABLE | — |" in md)
        check("dòng toàn tập có mặt trong bảng", "Toàn tập có nhãn" in md)

        # verdict của MẺ KHÁC lọt vào -> phải chặn, vì nó phá cả mẫu số lẫn khung mẫu
        _write_jsonl(d / "verdicts_lac.jsonl", [{"item_id": "lac-de", "verdict": "correct"}])
        try:
            report_combined.build(d)
            check("verdict lạc mẻ khác -> chặn", False, "không ném lỗi")
        except ValueError:
            check("verdict lạc mẻ khác -> chặn", True)
        rep2 = report_combined.build(d, drop_unknown=True)
        check("--drop-unknown bỏ đúng verdict lạc, số còn lại không đổi",
              rep2["n_verdicts"] == rep["n_verdicts"]
              and approx(rep2["tiers"][0]["precision"], rep["tiers"][0]["precision"], 1e-12),
              f"{rep2['n_verdicts']} vs {rep['n_verdicts']}")


def test_interrater() -> None:
    """κ liên người là câu trả lời cho 'ai kiểm tra lại tác giả' — phải đúng."""
    print("[report_combined · κ liên người]")
    pairs = ([("correct", "correct")] * 8 + [("correct", "wrong_label")]
             + [("wrong_label", "correct")] + [("wrong_label", "wrong_label")] * 2)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        man = [{"item_id": f"i{i}", "stratum": "GOLD|stt2", "orig_verdict": v1}
               for i, (v1, _) in enumerate(pairs)]
        ver = [{"item_id": f"i{i}", "verdict": v2} for i, (_, v2) in enumerate(pairs)]
        man.append({"item_id": "i99", "stratum": "SILVER|stt2", "orig_verdict": "correct"})
        _write_jsonl(d / "manifest.jsonl", man)
        _write_jsonl(d / "verdicts.jsonl", ver)

        r = report_combined.interrater(d)
        check("κ liên người khớp giá trị tính tay 5/9", approx(r["kappa"], 5 / 9, 1e-9),
              f"{r['kappa']:.6f}")
        check("ô người thứ hai chưa chấm được báo riêng", r["n_ungraded"] == 1,
              str(r["n_ungraded"]))
        cond = r["conditional_agreement"]
        check("đồng thuận CÓ ĐIỀU KIỆN tách theo verdict của người 1",
              cond["correct"]["n"] == 9 and cond["correct"]["trùng khớp"] == 8
              and cond["wrong_label"]["n"] == 3 and cond["wrong_label"]["trùng khớp"] == 2,
              str(cond))
        ov = r["defect_overlap"]
        check("bảng chồng lấn lỗi đếm đúng theo hướng bất đồng",
              ov["cả hai gọi là lỗi"] == 2 and ov["chỉ người 1 gọi là lỗi"] == 1
              and ov["chỉ người 2 gọi là lỗi"] == 1
              and ov["cả hai gọi là đúng/không đọc được"] == 8, str(ov))
        lo, hi = r["agreement_ci"]
        check("có CI cho đồng thuận thô", 0.0 <= lo < 10 / 12 < hi <= 1.0, f"[{lo}, {hi}]")
        md = report_combined._md_interrater(r)
        check("trang κ liên người dựng được", "Cohen's κ" in md and "Landis & Koch" in md)

    # thư mục KHÔNG phải mẻ liên người -> chặn, đừng lặng lẽ trả κ vô nghĩa
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_jsonl(d / "manifest.jsonl", [{"item_id": "a", "stratum": "GOLD|stt2"}])
        _write_jsonl(d / "verdicts.jsonl", [{"item_id": "a", "verdict": "correct"}])
        try:
            report_combined.interrater(d)
            check("thiếu orig_verdict -> báo lỗi rõ", False, "không ném lỗi")
        except ValueError:
            check("thiếu orig_verdict -> báo lỗi rõ", True)


def test_khoi_c_batch() -> None:
    """Khối C (16/09): mẻ mù hai câu — các hàm thuần phải tất định và KHÔNG rò tầng."""
    print("[khoi_c · mẻ mù hai câu]")
    # 1. chuỗi trượt: liền nom_idx cùng (book,page,column,shift) mới là một chuỗi
    k5 = pd.DataFrame({
        "book": ["b"] * 8, "page": ["p"] * 8, "column": ["1"] * 5 + ["2"] * 3,
        "shift": ["1"] * 3 + ["-1"] * 2 + ["1"] * 3,
        "nom_idx": ["3", "4", "5", "8", "9", "1", "2", "7"],
    })
    cid = kc.chain_ids(k5)
    check("chuỗi tách theo shift và theo khoảng trống nom_idx", cid.nunique() == 4, str(cid.tolist()))
    check("hai ô liền nhau cùng shift vào cùng chuỗi", cid[0] == cid[1] == cid[2] and cid[3] == cid[4])

    # 2. check_blind: chuỗi cấm nằm TRONG base64 không tính, nằm NGOÀI thì bắt
    inner = 'x<img src="data:image/png;base64,QUJDtierRUVFgold/">'
    check("chuỗi cấm bên trong payload base64 được bỏ qua", kc.check_blind(inner) == {})
    check("chuỗi cấm ngoài payload bị bắt", kc.check_blind(inner + "<b>tier</b>") == {"tier": 1})

    # 3. id mù: tất định theo seed, 12 hex, KHÔNG dẫn xuất từ image
    rows = pd.DataFrame({"tang": ["MAIN"] * 3, "key": ["k1", "k2", "k3"],
                         "image": ["gold/a.png", "gold/b.png", "gold/c.png"]})
    a = kc.assign_ids(rows, 1)["item_id"].tolist()
    b = kc.assign_ids(rows, 1)["item_id"].tolist()
    c = kc.assign_ids(rows, 2)["item_id"].tolist()
    import hashlib as _h
    check("id tất định theo seed", a == b and a != c)
    check("id 12 hex, duy nhất", all(len(x) == 12 for x in a) and len(set(a)) == 3)
    check("id không phải sha1(image) như sampling cũ",
          all(x != _h.sha1(im.encode()).hexdigest()[:12] for x, im in zip(a, rows["image"])))

    # 4. mồi âm: hiển thị ÂM/MÃ của ô kề — phải KHÁC âm và mã thật; mồi dương = QĐ-01
    n = 12
    df = pd.DataFrame({
        "book": ["stt2"] * n, "page": ["page_0001"] * n, "column": ["1"] * n,
        "nom_idx": [str(i) for i in range(n)], "syl_idx": [str(i) for i in range(n)],
        "syllable": [f"am{i}" for i in range(n)], "label": [chr(0x4E00 + i) for i in range(n)],
        "unicode": [f"U+{0x4E00 + i:04X}" for i in range(n)],
        "tier": ["GOLD"] * n, "tier_v3": ["CHAR_A"] * n, "rule": ["r"] * n,
        "box_source": ["detector"] * n, "count_source": ["equal_qn"] * n,
        "n_ocr": ["12"] * n, "n_qn": ["12"] * n, "n_det": ["12"] * n,
        "qd01_locked": ["1", "1", "1"] + ["0"] * (n - 3),
        "bbox": ["[0, 0, 10, 10]"] * n, "image": [f"gold/{i}.png" for i in range(n)],
        "image_md5": ["x"] * n, "crop_quality_flag": ["ok"] * n,
    })
    df["key"] = df.apply(kc._key, axis=1)
    df["col_class"], df["box_class"] = "eq", "detector"
    out, cells = kc.draw_t4_decoys(df, set(), seed=7, n_pos=3, n_neg=4)
    neg = out[out["tang"] == "T4_MOI_AM"]
    pos = out[out["tang"] == "T4_MOI_DUONG"]
    check("mồi dương = đúng các ô QĐ-01, đáp án dung/dung",
          len(pos) == 3 and (pos["qd01_locked"] == "1").all()
          and (pos["decoy_q1"] == "dung").all() and (pos["decoy_q2"] == "dung").all())
    check("mồi dương là SRS phân tầng theo tier_v3 CÓ trọng số (stratum_N = dân số QĐ-01, w = N/n)",
          (pos["stratum"] == "T4|moi_duong|CHAR_A").all() and (pos["stratum_N"] == 3).all()
          and (pos["design_weight"] == 1.0).all()
          and any(c["stratum"] == "T4|moi_duong|CHAR_A" and c["N_h"] == 3 for c in cells))
    check("mồi âm KHÔNG có trọng số (chủ đích)", neg["design_weight"].isna().all())
    check("mồi âm hiển thị âm & mã KHÁC ô thật, đáp án sai_am/sai",
          len(neg) == 4 and (neg["shown_syllable"] != neg["syllable"]).all()
          and (neg["shown_label"] != neg["label"]).all()
          and (neg["decoy_q1"] == "sai_am").all() and (neg["decoy_q2"] == "sai").all())
    nb_ok = all(abs(int(r.decoy_from_key.split("|")[-1]) - int(r.nom_idx)) == 1 for r in neg.itertuples())
    check("ô kề đúng nghĩa syl_idx±1 cùng cột", nb_ok)
    check("mồi âm không lấy ô QĐ-01 làm ô thật", (neg["qd01_locked"] == "0").all())

    # 4b. T6 'người' chưa khoá: chỉ ô syllable=người, qd01=0, có bbox; N = dân số đầy đủ; w = N/n
    df6 = df.copy()
    df6.loc[df6.index[3:9], "syllable"] = "người"          # 6 ô 'người' không khoá (idx 3..8), 3 ô QĐ-01 ở 0..2
    t6, c6 = kc.draw_t6_nguoi(df6, {df6["key"].iloc[3]}, seed=7, n=4)
    check("T6: 4 ô 'người' chưa khoá, không lấy ô đã dùng, stratum_N = 6 (dân số đầy đủ), w = 1,5",
          len(t6) == 4 and (t6["syllable"] == "người").all() and (t6["qd01_locked"] == "0").all()
          and df6["key"].iloc[3] not in set(t6["key"]) and (t6["stratum_N"] == 6).all()
          and abs(float(t6["design_weight"].iloc[0]) - 1.5) < 1e-12 and c6[0]["N_h"] == 6)

    # 4c. lặp ẩn PHÂN TẦNG theo (tang, tier_v3): đúng số mỗi ô, id mới, repeat_of trỏ về gốc, lap_tang giữ tầng gốc
    rows_all = pd.concat([out.assign(item_id=[f"{i:012x}" for i in range(len(out))]),
                          t6.assign(item_id=[f"{i + 100:012x}" for i in range(len(t6))])])
    rep = kc.add_hidden_repeats(rows_all, seed=3, alloc={("T4_MOI_DUONG", "CHAR_A"): 2, ("T6_NGUOI", None): 3})
    check("lặp ẩn phân tầng: 2 + 3 = 5 ô, tang=T5_LAP, lap_tang = tầng gốc, id mới, repeat_of ∈ gốc",
          len(rep) == 5 and (rep["tang"] == "T5_LAP").all()
          and rep["lap_tang"].value_counts().to_dict() == {"T6_NGUOI": 3, "T4_MOI_DUONG": 2}
          and not set(rep["item_id"]) & set(rows_all["item_id"]) and set(rep["repeat_of"]) <= set(rows_all["item_id"]))
    rep2 = kc.add_hidden_repeats(rows_all, seed=3, alloc={("T4_MOI_DUONG", "CHAR_A"): 2, ("T6_NGUOI", None): 3})
    check("lặp ẩn tất định theo seed", rep["item_id"].tolist() == rep2["item_id"].tolist()
          and rep["repeat_of"].tolist() == rep2["repeat_of"].tolist())

    # 5. HTML phiên: không rò, đủ item, đủ hai câu, và hàng không mã thì bỏ Q2 ở phía JS
    items = [{"id": "abc123def456", "crop": "data:image/png;base64,QUJD", "cw": 10, "ch": 12,
              "ctx": "", "ref": "", "syl": "người", "code": "𠊚", "uni": "U+2029A"},
             {"id": "0123456789ab", "crop": "data:image/png;base64,QUJD", "cw": 10, "ch": 12,
              "ctx": "", "ref": "", "syl": "là", "code": "", "uni": ""}]
    html_txt = kc.render_session_html(items, "KC-TEST-S1", "Khối C · thử", "verdicts_KC-TEST-S1.jsonl")
    check("HTML phiên không chứa chuỗi rò tầng", kc.check_blind(html_txt) == {}, str(kc.check_blind(html_txt)))
    check("HTML mang đủ item và hai bộ câu hỏi",
          "abc123def456" in html_txt and "0123456789ab" in html_txt
          and '"sai_crop"' in html_txt and '"khong_ro"' in html_txt and "Q2 ·" in html_txt)
    check("lựa chọn Q1 có định nghĩa ngưỡng 1/3 chữ", "1/3" in html_txt)
    # 5b. HAI PHA: mã/glyph chỉ hiện sau Q1; Q2 bị chặn khi chưa lộ; verdict có q1_blind + đếm đổi sau lộ
    check("HTML hai pha: có thông báo pha 1, chặn Q2 khi chưa lộ mã, xuất q1_blind / n_q1_change_after_reveal",
          "hiện sau khi" in html_txt and "!revealed(r)" in html_txt and "q1_blind: r.q1_blind" in html_txt
          and "n_q1_change_after_reveal" in html_txt and "after_reveal" in html_txt and "dwell_q1_ms" in html_txt)
    check("HTML pha 1 KHÔNG in sẵn 'chưa có mã' cho ô không mã (chỉ lộ sau Q1 qua JS)",
          "chưa có mã Unicode" not in html_txt)
    # 5c. thống kê lộ 'người' -> QĐ-01
    ordr = pd.DataFrame({"shown_syllable": ["người", "người", "là"], "qd01_locked": ["1", "0", "0"],
                         "shown_label": ["𠊚", "", "羅"]})
    st = kc._nguoi_leak_stats(ordr)
    check("_nguoi_leak_stats: 2 ô 'người', 1 QĐ-01, tỉ lệ 0,5, 1 ô 𠊚",
          st["n_shown_nguoi"] == 2 and st["n_qd01"] == 1 and st["frac_qd01"] == 0.5 and st["n_shown_2029A"] == 1)


def _khoi_c_fixture(d: Path) -> tuple[Path, Path, Path]:
    """KHOA/manifest tổng hợp có đủ 8 tầng, viết đúng schema của make_khoi_c_batch (không cần dữ liệu)."""
    import hashlib as _h
    rows = []
    nid = [0]

    def add(tang, stratum, N, **kw):
        nid[0] += 1
        r = {"item_id": f"{nid[0]:012x}", "session": 1 + (nid[0] % 2), "session_id": f"KC-T-S{1 + (nid[0] % 2)}",
             "audit_order": 0, "tang": tang, "stratum": stratum, "stratum_N": N,
             "design_weight": None, "repeat_of": None, "book": "stt2", "page": f"page_{nid[0]:04d}",
             "column": "1", "nom_idx": str(nid[0]), "tier_v3": stratum.split("|")[0], "qd01_locked": "0",
             "syllable": f"am{nid[0]}", "label": "", "argmax": None, "chain_id": None, "has_q2": False}
        r.update(kw)
        rows.append(r)
        return r["item_id"]

    main_ids = []
    for st, N, n, q2 in (("CHAR_A|eq|detector", 1000, 10, True), ("CHAR_A|ne|detector", 500, 10, True),
                         ("CHAR_B|eq|detector", 100, 6, True), ("SYL|eq|detector", 300, 6, False),
                         ("REVIEW|eq|detector", 200, 4, False)):
        for i in range(n):
            main_ids.append(add("MAIN", st, N, design_weight=N / n, has_q2=q2 or (st.startswith("SYL") and i == 0),
                                label="字" if (q2 or (st.startswith("SYL") and i == 0)) else ""))
    for b, N in (("stt2", 400), ("stt4", 350)):
        for _ in range(75):
            add("T1_GATE", f"T1|gate09|{b}", N, tier_v3="REVIEW", design_weight=N / 75, book=b)
    for cid, n, qd in (("c#1", 3, 1), ("c#2", 4, 0)):
        for i in range(n):
            add("T2_CHAIN", "T2|chain", 404, tier_v3="SYL", chain_id=cid, argmax="thế",
                qd01_locked="1" if (qd and i == 0) else "0", has_q2=(i == 0), label="字" if i == 0 else "")
    add("T3_B2", "T3|b2", 3, tier_v3="CHAR_B", design_weight=1.0, syllable="trên", argmax="lên", has_q2=True, label="連")
    add("T3_B2", "T3|b2", 3, tier_v3="SYL", design_weight=1.0, syllable="vậy", argmax="làm")
    add("T3_B2", "T3|b2", 3, tier_v3="CHAR_B", design_weight=1.0, syllable="sự", argmax="sự", has_q2=True, label="事")
    for _ in range(4):
        add("T3_B5", "T3|b5", 117, tier_v3="CHAR_A", design_weight=117 / 4, qd01_locked="1", has_q2=True, label="𠊚")
    # mồi dương = tầng QĐ-01 có trọng số (2 tầng phụ: CHAR_A 2/40, SYL 1/10) -> dân số QĐ-01 = 50
    for _ in range(2):
        add("T4_MOI_DUONG", "T4|moi_duong|CHAR_A", 40, tier_v3="CHAR_A", qd01_locked="1", has_q2=True, label="𠊚",
            design_weight=20.0, decoy_q1="dung", decoy_q2="dung", syllable="người")
    add("T4_MOI_DUONG", "T4|moi_duong|SYL", 10, tier_v3="SYL", qd01_locked="1", has_q2=True, label="𠊚",
        design_weight=10.0, decoy_q1="dung", decoy_q2="dung", syllable="người")
    for _ in range(3):
        add("T4_MOI_AM", "T4|moi_am", None, tier_v3="CHAR_A", has_q2=True, label="施", decoy_q1="sai_am", decoy_q2="sai")
    for _ in range(4):
        add("T6_NGUOI", "T6|nguoi_chua_khoa", 20, tier_v3="SYL", design_weight=5.0, syllable="người")
    by_id = {r["item_id"]: r for r in rows}
    t1_ids = [r["item_id"] for r in rows if r["tang"] == "T1_GATE"][:2]
    t2_ids = [r["item_id"] for r in rows if r["tang"] == "T2_CHAIN"][:2]
    for oid in main_ids[:6] + t1_ids + t2_ids:
        o = by_id[oid]
        add("T5_LAP", "__repeat__", None, tier_v3=o["tier_v3"], repeat_of=oid, has_q2=o["has_q2"], label=o["label"],
            lap_tang=o["tang"])
    for i, r in enumerate(rows):
        r["audit_order"] = i
    labels = d / "labels_final.csv"
    labels.write_text("image,label\na.png,字\n", encoding="utf-8")
    lsha = _h.sha256(labels.read_bytes()).hexdigest()
    kd = d / "_khoa"
    kd.mkdir()
    khoa = kd / "KHOA.jsonl"
    khoa.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    bsha = _h.sha256(khoa.read_bytes()).hexdigest()
    man = kd / "manifest.jsonl"
    man.write_text("\n".join(json.dumps({
        "item_id": r["item_id"], "session": r["session"], "session_id": r["session_id"],
        "audit_order": r["audit_order"], "tang": r["tang"], "stratum": r["stratum"], "stratum_N": r["stratum_N"],
        "design_weight": r["design_weight"], "repeat_of": r["repeat_of"], "has_q2": r["has_q2"],
        "labels_sha256": lsha, "labels_path": "labels_final.csv", "batch_sha256": bsha, "seed": 1})
        for r in rows) + "\n", encoding="utf-8")
    (d / "plan.json").write_text(json.dumps({"sessions": [
        {"session": 1, "session_id": "KC-T-S1", "verdict_file": "verdicts_KC-T-S1.jsonl"},
        {"session": 2, "session_id": "KC-T-S2", "verdict_file": "verdicts_KC-T-S2.jsonl"}]}), encoding="utf-8")
    return khoa, man, labels


def test_estimate_khoi_c() -> None:
    """Khối C (C-3): ước lượng từ verdict — kiểm bằng verdict GIẢ LẬP có đáp án gieo sẵn."""
    print("[khoi_c · ước lượng (estimate_khoi_c)]")
    # 0. κ tổng quát trong stats == κ 3 hạng mục của report_combined trên cùng dữ liệu 5/9
    pairs = [("correct", "correct")] * 5 + [("correct", "wrong_label"), ("wrong_label", "correct"),
                                            ("unsure", "unsure"), ("wrong_label", "wrong_label")]
    check("stats.cohens_kappa khớp report_combined.cohens_kappa",
          approx(stats.cohens_kappa(pairs, report_combined.VERDICT_ORDER)["kappa"],
                 report_combined.cohens_kappa(pairs)["kappa"], 1e-12))
    check("κ = 1 khi đồng thuận tuyệt đối, 0 khi độc lập",
          stats.cohens_kappa([("a", "a"), ("b", "b")], ("a", "b"))["kappa"] == 1.0
          and approx(stats.cohens_kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")], ("a", "b"))["kappa"], 0.0))

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        khoa, man, labels = _khoi_c_fixture(d)
        kh = ekc.load_jsonl(khoa)
        err = {"CHAR_A": 0.2, "CHAR_B": 0.3, "SYL": 0.25, "REVIEW": 0.5, "T1_GATE": 0.0,
               "T2_CHAIN": 0.6, "T3_B2": 0.5, "T3_B5": 0.25}
        q2e = {"CHAR_A": 0.1, "CHAR_B": 0.2, "SYL": 0.0, "REVIEW": 0.0, "T3_B5": 0.0, "T2_CHAIN": 0.0, "T3_B2": 0.0}
        v0, truth = ekc.simulate_verdicts(kh, seed=3, noise=0.0, err=err, q2_err=q2e)
        sim_dir = d / "sim0"
        sim_dir.mkdir()
        for sid, g in v0.groupby("session_id"):
            recs = [{k: (None if (isinstance(x, float) and np.isnan(x)) else x) for k, x in r.items()}
                    for r in g.to_dict("records")]
            (sim_dir / f"verdicts_{sid}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        files = sorted(sim_dir.glob("verdicts*.jsonl"))

        # 1. nguồn không phải người bị loại mặc định
        v_h, info_h = ekc.load_verdicts(files)
        check("verdict source=simulated bị LOẠI mặc định (0 nạp, đếm bỏ)",
              v_h.empty and info_h["n_skipped_ai"] == len(kh))
        rep_h = ekc.build(d, files, labels)
        check("không verdict -> báo động rõ, không crash", "không có verdict nào" in " ".join(rep_h["alarms"]))

        # 2. nạp + toàn vẹn (noise 0, người lý tưởng)
        rep = ekc.build(d, files, labels, include_ai=True)
        it = rep["integrity"]
        check("toàn vẹn: batch_sha256 + labels_sha256 khớp, 0 id lạ, 0 lệch has_code, chấm đủ",
              it["batch_sha256_ok"] and it["labels_sha256_ok"] and it["n_unknown_id"] == 0
              and it["n_has_code_mismatch"] == 0 and it["n_graded"] == len(kh) and not rep["alarms"], str(rep["alarms"]))

        # 3. ô mồi 100 %, κ = 1, dwell khớp bảng sự thật
        dcy = rep["decoys"]
        check("mồi: dương 3/3, âm 3/3, không báo động",
              dcy["pos"]["n_pass"] == 3 and dcy["neg"]["n_pass"] == 3 and not dcy["alarm"])
        rp = rep["repeats"]
        check("lặp ẩn: 10 cặp (6 MAIN + 2 T1 + 2 T2), κ Q1 = 1, đồng thuận 100 %, KTC bootstrap = [1; 1], tách theo tầng gốc",
              rp["n_pairs"] == 10 and rp["q1"]["kappa"] == 1.0 and rp["q1"]["observed_agreement"] == 1.0
              and rp["q1"]["ci_boot"] == (1.0, 1.0)
              and set(rp["by_lap_tang"]) == {"MAIN", "T1_GATE", "T2_CHAIN"} and rp["by_lap_tang"]["MAIN"]["n"] == 6)
        check("mồi dương đạt theo Q1; Q2 chỉ đếm 'nhất quán QĐ-01'",
              "q2_dung_nhat_quan_qd01" in dcy["pos"] and dcy["pos"]["q2_dung_nhat_quan_qd01"] == 3
              and "Q1 == dung" in dcy["pos_rule"])
        check("Q1 dùng = q1_blind (verdict giả lập thế hệ hai pha)", rep["q1_source"].startswith("q1_blind")
              and rep["dwell"]["n_q1_changed_after_reveal"] == 0)
        dw = rep["dwell"]
        exp_fast = int((truth["dwell_ms"] < ekc.DWELL_FLAG_MS).sum())
        check("dwell: số ô < 1,5 s bằng đếm trực tiếp trên bảng sự thật, p50 tính được",
              dw["n_flag_fast"] == exp_fast and dw["n_with_dwell"] == len(kh) and dw["p50_s"] > 0)

        # 4. precision theo tier khớp bảng sự thật; HT khớp tính tay
        tr = truth.merge(kh[["item_id", "stratum", "stratum_N", "tier_v3", "has_q2"]], on="item_id")
        mm = tr[tr["tang"] == "MAIN"]
        okA = mm[mm["tier_v3"] == "CHAR_A"]
        kA, nA = int((okA["truth_q1"] == "dung").sum()), len(okA)
        pa = rep["precision"]["tiers"]["CHAR_A"]["q1"]
        check("CHAR_A Q1: n_ok/n_scored đúng bằng sự thật gieo", pa["n_ok"] == kA and pa["n_scored"] == nA, f"{pa}")
        trip = [(int(g["stratum_N"].iloc[0]), len(g), int((g["truth_q1"] == "dung").sum()))
                for _, g in okA.groupby("stratum")]
        pt, lo, hi = stats.stratified_mean_ci(trip)
        check("CHAR_A Q1 HT == stratified_mean_ci tính tay từ (N_h, n_h, k_h)",
              approx(pa["weighted_precision"], pt, 1e-12) and approx(pa["weighted_ci"][0], lo, 1e-12))
        q2A = rep["precision"]["tiers"]["CHAR_A"]["q2"]
        check("CHAR_A Q2: n_ok đúng sự thật", q2A["n_ok"] == int((okA["truth_q2"] == "dung").sum()))
        gold = mm[mm["tier_v3"].isin(("CHAR_A", "CHAR_B"))]
        tripG = [(int(g["stratum_N"].iloc[0]), len(g), int((g["truth_q1"] == "dung").sum()))
                 for _, g in gold.groupby("stratum")]
        pg = rep["precision"]["pooled"]["GOLD"]["q1"]
        check("GOLD = CHAR_A+CHAR_B: HT trên 3 tầng khớp tính tay",
              pg["n_strata"] == 3 and approx(pg["weighted_precision"], stats.stratified_mean_ci(tripG)[0], 1e-12))
        syl = rep["precision"]["tiers"]["SYL"]
        check("SYL: Q2 chỉ 1/6 ô có mã -> ghi chú, KHÔNG suy HT cho Q2",
              "q2_note" in syl and "weighted_precision" not in syl["q2"] and syl["q2"]["n_scored"] == 1)
        pq = rep["precision"]["pooled"]["QD01"]
        pu = rep["precision"]["pooled"]["USABLE_RE_DATASET"]
        tq = tr[tr["tang"] == "T4_MOI_DUONG"]
        tripQ = [(int(g["stratum_N"].iloc[0]), len(g), int((g["truth_q1"] == "dung").sum())) for _, g in tq.groupby("stratum")]
        tripU = [(int(g["stratum_N"].iloc[0]), len(g), int((g["truth_q1"] == "dung").sum()))
                 for _, g in pd.concat([mm[mm["tier_v3"].isin(("CHAR_A", "CHAR_B", "SYL"))], tq]).groupby("stratum")]
        check("QĐ-01 = tầng mồi dương có N_h (dân số 50, 2 tầng); USABLE_RE_DATASET = USABLE ∪ QĐ-01 (dân số 1950) khớp tính tay",
              pq["q1"]["population_N"] == 50 and pq["q1"]["n_strata"] == 2
              and pu["q1"]["population_N"] == 1000 + 500 + 100 + 300 + 50
              and approx(pu["q1"]["weighted_precision"], stats.stratified_mean_ci(tripU)[0], 1e-12)
              and approx(pq["q1"]["weighted_precision"], stats.stratified_mean_ci(tripQ)[0], 1e-12))
        t6 = rep["t6_nguoi"]
        t6t = tr[tr["tang"] == "T6_NGUOI"]
        check("T6 'người' chưa khoá: n = 4, HT về dân số 20, precision = sự thật gieo",
              t6["n_audited"] == 4 and t6["population_N"] == 20
              and approx(t6["precision"], (t6t["truth_q1"] == "dung").mean(), 1e-12))

        # 5. T1: 0/150 lỗi -> cận trên CP 1,98 % -> đủ điều kiện; 5 lỗi -> không
        t1 = rep["t1_gate"]
        check("T1: 0/150 lỗi -> cận trên CP một phía = 1,98 % ≤ 3 % -> ĐỦ điều kiện",
              t1["n_err"] == 0 and t1["n_scored"] == 150 and approx(t1["err_upper_cp95"], 0.019773, 1e-5)
              and t1["eligible"] is True)
        # 6. T2 / T3 khớp sự thật
        t2 = rep["t2_chains"]
        exp_dem = 0
        for cid, g in tr[tr["tang"] == "T2_CHAIN"].merge(kh[["item_id", "chain_id"]], on="item_id").groupby("chain_id"):
            exp_dem += int((g["truth_q1"] == "sai_am").mean() >= 0.5)
        check("T2: số chuỗi đề xuất hạ (≥ 50 % sai_am) khớp sự thật; ghi âm == argmax",
              t2["n_chains"] == 2 and t2["n_chains_demote"] == exp_dem
              and t2["n_sai_am_matches_argmax"] == t2["n_sai_am_wrote"] == t2["q1_counts"]["sai_am"])
        b5 = rep["t3"]["b5"]
        b5t = tr[tr["tang"] == "T3_B5"]
        check("T3-B5: hộp khoá đúng = k/n sự thật, HT về 117 ô",
              approx(b5["hop_khoa_dung"], (b5t["truth_q1"] == "dung").mean(), 1e-12) and b5["population_N"] == 117)

        # 7. can thiệp có chủ đích vào verdict: mồi rớt, lặp lệch, khong_ro, B-2, id lạ, trùng, labels đổi
        v1 = v0.copy()
        b2ids = kh[kh["tang"] == "T3_B2"]["item_id"].tolist()
        # verdict hai pha: ước lượng đọc q1_blind -> can thiệp phải đổi CẢ q1 lẫn q1_blind
        v1.loc[v1["item_id"] == b2ids[0], ["q1", "q1_blind", "q1_am"]] = ["sai_am", "sai_am", "lên"]   # âm mới đúng
        v1.loc[v1["item_id"] == b2ids[1], ["q1", "q1_blind", "q1_am"]] = ["dung", "dung", ""]          # âm cũ đúng
        v1.loc[v1["item_id"] == b2ids[2], ["q1", "q1_blind", "q1_am"]] = ["sai_am", "sai_am", "sự"]    # âm không đổi -> cả hai sai
        pos_id = kh[kh["tang"] == "T4_MOI_DUONG"]["item_id"].iloc[0]
        v1.loc[v1["item_id"] == pos_id, ["q1", "q1_blind"]] = "sai_crop"
        rep_id = kh[kh["tang"] == "T5_LAP"]["item_id"].iloc[0]
        orig_q1 = v1.loc[v1["item_id"] == rep_id, "q1_blind"].iloc[0]
        v1.loc[v1["item_id"] == rep_id, ["q1", "q1_blind"]] = "khong_ro" if orig_q1 != "khong_ro" else "dung"
        a_ids = kh[kh["stratum"] == "CHAR_A|eq|detector"]["item_id"].tolist()
        v1.loc[v1["item_id"] == a_ids[7], ["q1", "q1_blind"]] = "khong_ro"                          # ô KHÔNG bị lặp
        t1_ids = kh[kh["tang"] == "T1_GATE"]["item_id"].tolist()[:5]
        v1.loc[v1["item_id"].isin(t1_ids), ["q1", "q1_blind"]] = "sai_am"
        t1_reps = kh[kh["repeat_of"].isin(t1_ids)]["item_id"].tolist()                  # bản lặp của ô T1 đổi theo (người nhất quán)
        v1.loc[v1["item_id"].isin(t1_reps), ["q1", "q1_blind"]] = "sai_am"
        recs = [{k: (None if (isinstance(x, float) and np.isnan(x)) else x) for k, x in r.items()}
                for r in v1.to_dict("records")]
        recs.append(dict(recs[0], item_id="ffffffffffff"))                              # id lạ
        recs.append(dict(recs[1], q1="khong_ro", q1_blind="khong_ro", exported_at=recs[1]["exported_at"] - 5))  # bản CŨ hơn -> bị bỏ
        f1 = d / "verdicts_edit.jsonl"
        f1.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        rep1 = ekc.build(d, [f1], labels, include_ai=True)
        it1 = rep1["integrity"]
        check("id lạ bị bắt (1) và báo động", it1["n_unknown_id"] == 1 and any("KHÔNG có trong khoá" in a for a in rep1["alarms"]))
        check("trùng item_id: giữ bản exported_at MỚI nhất (bản cũ khong_ro bị bỏ)",
              rep1["verdict_input"]["n_dup_item"] == 1
              and rep1["raw_counts"]["q1"]["khong_ro"] == 1 + int(orig_q1 != "khong_ro"))
        check("mồi dương rớt 1/3 (Q1 mù = sai_crop) -> độ chính xác 66,7 %, BÁO ĐỘNG",
              approx(rep1["decoys"]["pos"]["accuracy"], 2 / 3, 1e-12) and rep1["decoys"]["alarm"]
              and rep1["decoys"]["pos"]["failed_ids"] == [pos_id])
        check("lặp lệch 1/10 -> κ Q1 < 1, id lệch được liệt kê",
              rep1["repeats"]["q1"]["kappa"] < 1.0 and rep1["repeats"]["disagree_ids_q1"] == [rep_id])
        pa1 = rep1["precision"]["tiers"]["CHAR_A"]["q1"]
        check("khong_ro loại khỏi mẫu số: n_scored giảm 1, n_khong_ro = 1",
              pa1["n_scored"] == nA - 1 and pa1["n_khong_ro"] == 1)
        t1b = rep1["t1_gate"]
        check("T1: 5/150 lỗi -> cận trên CP > 3 % -> CHƯA đủ điều kiện",
              t1b["n_err"] == 5 and t1b["err_upper_cp95"] > 0.03 and t1b["eligible"] is False)
        b2 = rep1["t3"]["b2"]["counts"]
        check("T3-B2: phân loại âm mới đúng / âm cũ đúng / cả hai sai (âm không đổi)",
              b2["am_moi_dung"] == 1 and b2["am_cu_dung"] == 1 and b2["ca_hai_sai"] == 1)
        md = ekc.to_markdown(rep1)
        check("markdown có đủ mục 0-8 và JSON hoá được",
              all(h in md for h in ("## 0. Toàn vẹn", "## 1. Ô mồi", "## 2. Lặp ẩn", "## 3. Dwell",
                                    "## 4. Precision", "## 5. T1", "## 6. T2", "## 7. T3"))
              and json.dumps(ekc._jsonable(rep1), ensure_ascii=False))

        # 7b. hai pha: q1 cuối đổi sau khi lộ mã -> ước lượng vẫn dùng q1_blind; verdict thế hệ cũ -> báo động
        v7, _ = ekc.simulate_verdicts(kh, seed=3, noise=0.0, err=err, q2_err=q2e, p_change_after_reveal=0.3)  # cùng sự thật seed 3
        n_ch = int((v7["n_q1_change_after_reveal"] > 0).sum())
        recs7 = [{k: (None if (isinstance(x, float) and np.isnan(x)) else x) for k, x in r.items()}
                 for r in v7.to_dict("records")]
        f7 = d / "verdicts_reveal.jsonl"
        f7.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs7) + "\n", encoding="utf-8")
        rep7 = ekc.build(d, [f7], labels, include_ai=True)
        pa7 = rep7["precision"]["tiers"]["CHAR_A"]["q1"]
        check("đổi Q1 sau khi lộ mã ở 30 % ô có mã: precision CHAR_A KHÔNG đổi (dùng q1_blind), số đổi được đếm",
              n_ch > 0 and pa7["n_ok"] == pa["n_ok"] and pa7["n_scored"] == pa["n_scored"]
              and rep7["dwell"]["n_q1_changed_after_reveal"] == n_ch
              and rep7["dwell"]["n_q1_blind_ne_final"] == n_ch, f"n_ch={n_ch} {rep7['dwell'].get('n_q1_changed_after_reveal')}")
        old = [{k: x for k, x in r.items() if k not in ("q1_blind", "dwell_q1_ms", "n_q1_change_after_reveal", "revealed_at")}
               for r in recs7]
        f8 = d / "verdicts_old.jsonl"
        f8.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in old) + "\n", encoding="utf-8")
        rep8 = ekc.build(d, [f8], labels, include_ai=True)
        check("verdict thế hệ cũ (không q1_blind): dùng q1 cuối + BÁO ĐỘNG 'không mù giữa CHAR và SYL'",
              rep8["verdict_input"]["n_without_q1_blind"] == len(kh)
              and any("q1_blind" in a for a in rep8["alarms"]) and rep8["q1_source"].startswith("q1 cuối"))

        # 8. bộ nhãn đổi sau khi rút mẫu -> labels_sha256 SAI + báo động
        labels.write_text("image,label\na.png,子\n", encoding="utf-8")
        rep2 = ekc.build(d, files, labels, include_ai=True)
        check("labels_final.csv đổi -> labels_sha256_ok False + báo động",
              rep2["integrity"]["labels_sha256_ok"] is False and any("đã đổi" in a for a in rep2["alarms"]))

        # 9. nhiễu 5 %: mồi ≈ 90 % (2 câu × 95 %), κ < 1 nhưng > 0,5, precision lệch sự thật ≤ 10 điểm
        v5, truth5 = ekc.simulate_verdicts(kh, seed=11, noise=0.05, err=err, q2_err=q2e)
        f5 = d / "verdicts_n5.jsonl"
        recs5 = [{k: (None if (isinstance(x, float) and np.isnan(x)) else x) for k, x in r.items()}
                 for r in v5.to_dict("records")]
        f5.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs5) + "\n", encoding="utf-8")
        rep5 = ekc.build(d, [f5], labels if False else None, include_ai=True)
        n_noisy = int((truth5["truth_q1"] != truth5["ans_q1"]).sum())
        check("nhiễu 5 %: bộ giả lập thật sự đổi câu trả lời, ước lượng vẫn chạy, κ Q1 trong (0,5; 1]",
              0 < n_noisy < 0.15 * len(kh) and 0.5 < rep5["repeats"]["q1"]["kappa"] <= 1.0, f"noisy={n_noisy}")
        pa5 = rep5["precision"]["tiers"]["CHAR_A"]["q1"]
        t5A = truth5.merge(kh[["item_id", "tier_v3"]], on="item_id")
        t5A = t5A[(t5A["tang"] == "MAIN") & (t5A["tier_v3"] == "CHAR_A") & (t5A["ans_q1"] != "khong_ro")]
        check("nhiễu 5 %: precision CHAR_A == đếm trên câu trả lời NHIỄU (không phải sự thật), tỉ lệ nhiễu 1–12 %",
              pa5["n_ok"] == int((t5A["ans_q1"] == "dung").sum()) and pa5["n_scored"] == len(t5A)
              and 0.01 <= n_noisy / len(kh) <= 0.12, f"{pa5} noisy={n_noisy}/{len(kh)}")
        check("không có labels -> labels_sha256_ok None + báo động 'không đối chiếu'",
              rep5["integrity"]["labels_sha256_ok"] is None and any("không có labels" in a for a in rep5["alarms"]))

        # 10. pilot: dùng KHOA_pilot/manifest_pilot, KHÔNG có precision
        import shutil
        shutil.copy(khoa, d / "_khoa" / "KHOA_pilot.jsonl")
        shutil.copy(man, d / "_khoa" / "manifest_pilot.jsonl")
        repp = ekc.build(d, files, None, pilot=True, include_ai=True)
        check("--pilot: báo cáo ngắn (mồi/κ/dwell), không có precision/T1", "precision" not in repp and "decoys" in repp
              and "Pilot chỉ đo" in ekc.to_markdown(repp))


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
    test_report_combined()
    test_interrater()
    test_khoi_c_batch()
    test_estimate_khoi_c()
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
