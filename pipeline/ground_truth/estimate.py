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

NGUỒN VERDICT: mặc định module này CHỈ nhận verdict do NGƯỜI chấm. Bản ghi có
`source` ∈ AI_VERDICT_SOURCES (vd 'ai_vision') bị LOẠI, trừ khi khai báo tường minh
include_ai=True (cli: --include-ai-verdicts). Verdict thiếu trường `source` được coi là
người chấm. Điều này ngăn nhãn do MÁY chấm âm thầm trở thành ground truth cho precision
+ CI + acceptance (khớp semantics ở pipeline.consensus_fusion.fuse_stage).
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

# Nguồn verdict do MÁY chấm. MẶC ĐỊNH bị loại khỏi phép tính precision/CI/acceptance vì
# nhãn máy KHÔNG phải ground truth (khớp AI_VERDICT_SOURCES ở consensus_fusion.fuse_stage).
AI_VERDICT_SOURCES = {"ai_vision"}


def _verdict_source(rec: dict) -> str:
    """Nguồn của một bản ghi verdict; vắng trường `source` => coi là người chấm."""
    return str(rec.get("source") or "human").strip().lower()


def _verdict_files(path: str | Path) -> list[Path]:
    """Một file, hoặc MỌI verdicts*.jsonl trong một thư mục.

    Công cụ chấm xuất ra từng phần (`verdicts_001.jsonl`, `verdicts_002.jsonl`, ...) nên
    truyền cả thư mục là cách dùng tự nhiên — và đúng như hướng dẫn trong README của các
    mẻ audit.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("verdicts*.jsonl"))
        if not files:
            raise FileNotFoundError(f"không thấy verdicts*.jsonl nào trong {p}")
        return files
    return [p]


def load_verdicts(path: str | Path, include_ai: bool = False) -> pd.DataFrame:
    """Đọc verdicts.jsonl (hoặc cả thư mục chứa verdicts*.jsonl) thành frame.

    MẶC ĐỊNH chỉ nhận verdict của NGƯỜI chấm: bản ghi có source ∈ AI_VERDICT_SOURCES
    (vd 'ai_vision') bị LOẠI, trừ khi include_ai=True. Verdict thiếu trường `source`
    được coi là người chấm. In số nạp/bỏ theo từng nguồn để minh bạch.
    """
    rows = []
    kept: dict[str, int] = {}
    skipped: dict[str, int] = {}
    files = _verdict_files(path)
    for fp in files:
        for ln in fp.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            if "item_id" not in d or "verdict" not in d:
                raise ValueError(f"{fp.name}: verdict line missing keys: {ln}")
            if d["verdict"] not in VALID_VERDICTS:
                raise ValueError(f"{fp.name}: unknown verdict {d['verdict']!r}")
            src = _verdict_source(d)
            if src in AI_VERDICT_SOURCES and not include_ai:
                skipped[src] = skipped.get(src, 0) + 1
                continue
            kept[src] = kept.get(src, 0) + 1
            rows.append({"item_id": str(d["item_id"]), "verdict": d["verdict"], "source": src})

    def _fmt(dd: dict[str, int]) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(dd.items())) or "0"
    print(f"[estimate] verdicts từ {len(files)} file: nạp {sum(kept.values())} "
          f"({_fmt(kept)}), bỏ qua {sum(skipped.values())} ({_fmt(skipped)})")
    if skipped and not include_ai:
        print(f"[cảnh báo] đã LOẠI {sum(skipped.values())} verdict source=ai_vision "
              f"(dùng --include-ai-verdicts nếu thực sự muốn dùng nhãn máy)")

    df = pd.DataFrame(rows)
    if df.empty:
        if skipped and not include_ai:
            raise ValueError(
                f"toàn bộ verdict là AI (source=ai_vision; {sum(skipped.values())} dòng bị loại); "
                f"dùng --include-ai-verdicts nếu THỰC SỰ muốn dùng nhãn máy làm ground truth")
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


def join_manifest(verdicts: pd.DataFrame, manifest: pd.DataFrame,
                  drop_unknown: bool = False) -> pd.DataFrame:
    """Ghép verdict với manifest. Verdict không có hàng manifest tương ứng là LỖI THẬT.

    Nguyên nhân thường gặp: bản xuất kéo theo verdict của MẺ KHÁC. Các mẻ dựng trước
    2026-08-03 dùng chung một khoá localStorage (`gt_audit_verdicts`) nên kho verdict bị
    trộn giữa các mẻ; khoá nay đã tách theo mẻ. Với các bản xuất cũ, dùng drop_unknown=True
    để bỏ phần lạc — hàm sẽ IN RÕ số bị bỏ chứ không im lặng.
    """
    known = set(manifest["item_id"].astype(str))
    unknown = verdicts.loc[~verdicts["item_id"].astype(str).isin(known)]
    if len(unknown):
        if not drop_unknown:
            raise ValueError(
                f"{len(unknown)} verdict không có hàng manifest tương ứng "
                f"(vd {unknown['item_id'].iloc[0]!r}). Gần như chắc chắn là verdict của MẺ "
                f"KHÁC lọt vào do localStorage dùng chung khoá trước 2026-08-03. "
                f"Dùng drop_unknown=True / --drop-unknown-verdicts để bỏ chúng.")
        print(f"[estimate] BỎ {len(unknown)} verdict lạc (không thuộc mẻ này): "
              f"{', '.join(unknown['item_id'].astype(str).head(5))}")
        verdicts = verdicts.loc[verdicts["item_id"].astype(str).isin(known)]

    j = verdicts.merge(manifest, on="item_id", how="left", validate="one_to_one")
    missing = j["stratum"].isna().sum() if "stratum" in j.columns else j.isna().all(axis=1).sum()
    if missing:
        raise ValueError(f"{missing} verdicts have no matching manifest row")
    n_ungraded = len(known) - len(j)
    if n_ungraded > 0:
        print(f"[estimate] CẢNH BÁO: {n_ungraded}/{len(known)} ô trong mẻ CHƯA được chấm — "
              f"chúng bị loại khỏi mẫu. Nếu việc bỏ sót có liên quan tới độ khó của ô thì "
              f"ước lượng sẽ chệch; kiểm lại trước khi trích dẫn.")
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
    # Số ô bị LOẠI vì thuộc mẫu chủ đích (design_weight rỗng) + thống kê mô tả của chúng.
    # Luôn báo ra để việc loại là minh bạch, không phải cắt bớt âm thầm.
    n_purposive_excluded: int = 0
    purposive: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _split_purposive(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tách (mẫu xác suất, mẫu chủ đích). Chủ đích = design_weight rỗng.

    Chỉ tách khi khung có CẢ HAI loại; nếu toàn bộ đều có trọng số (mọi mẻ cũ) thì
    không đổi gì — hàm là no-op và tương thích ngược tuyệt đối.
    """
    if "design_weight" not in joined.columns:
        return joined, joined.iloc[0:0]
    w = pd.to_numeric(joined["design_weight"], errors="coerce")
    is_purposive = w.isna()
    if is_purposive.all():
        print("[estimate] CẢNH BÁO: KHÔNG hàng nào có design_weight — mẻ này không phải "
              "mẫu xác suất (vd mẻ kiểm tra lặp, lấy vượt tỷ lệ nhóm lỗi). Mọi con số "
              "precision/CI dưới đây CHỈ mô tả chính mẫu, KHÔNG suy rộng ra dân số được.")
    if not is_purposive.any() or is_purposive.all():
        # toàn xác suất -> giữ nguyên; toàn chủ đích -> để nguyên cho người dùng tự chịu
        # trách nhiệm (và cảnh báo ở tầng CLI), tránh trả về khung rỗng gây lỗi khó hiểu.
        return joined, joined.iloc[0:0]
    return joined.loc[~is_purposive].copy(), joined.loc[is_purposive].copy()


def _purposive_summary(purposive: pd.DataFrame) -> dict | None:
    """Thống kê MÔ TẢ cho tầng chủ đích — cố ý KHÔNG có CI, vì CI ở đây vô nghĩa."""
    if purposive.empty:
        return None
    s = purposive[purposive["is_unsure"] == 0]
    n = len(s)
    k = int(s["correct"].sum()) if n else 0
    batches = (sorted(set(purposive["audit_batch"].dropna().astype(str)))
               if "audit_batch" in purposive.columns else [])
    return {
        "n_audited": len(purposive),
        "n_scored": n,
        "n_correct": k,
        "error_rate": (1.0 - k / n) if n else None,
        "audit_batch": batches,
        "note": ("Mẫu CHỦ ĐÍCH (design_weight rỗng) — đã LOẠI khỏi mọi ước lượng "
                 "precision. Chỉ dùng để đo AUC / hiệu chỉnh ngưỡng. Tỷ lệ lỗi ở đây "
                 "CAO hơn dân số theo đúng thiết kế và KHÔNG được trích dẫn như "
                 "precision của tier."),
    }


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
    # --- LOẠI phần mẫu CHỦ ĐÍCH khỏi mọi ước lượng precision ------------------ #
    # Mẻ hai tầng (make_gold_batch) trộn một tầng xác suất với một tầng chọn-chủ-đích
    # theo margin thấp. Tầng chủ đích có tỷ lệ lỗi cao hơn hẳn dân số theo đúng thiết kế;
    # gộp nó vào Clopper–Pearson sẽ cho precision THẤP GIẢ. Dấu hiệu nhận biết là
    # design_weight rỗng — mẫu không xác suất thì không có trọng số quy về dân số.
    joined, purposive = _split_purposive(joined)
    n_purposive = len(purposive)

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
        n_purposive_excluded=n_purposive, purposive=_purposive_summary(purposive),
    )
