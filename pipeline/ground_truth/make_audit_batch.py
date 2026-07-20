"""P1.2 — Dựng MẺ AUDIT cho một tier (mặc định SILVER) để ĐO precision chưa biết.

Vì sao: audit GOLD (825 verdict) 100% là GOLD → SILVER (10.856) + SYLLABLE (6.751) precision
CHƯA ĐO. 98% GOLD KHÔNG suy rộng được. Mẻ này lấy mẫu KHÔNG THIÊN LỆCH (stratified random
theo RULE, KHÔNG rank theo nghi ngờ) → chấm tay → estimate cho precision + CI + per-rule.

Thiết kế: stratify theo `rule` (biết rule nào của SILVER đáng tin), sàn min mỗi rule để rule
nhỏ vẫn ước lượng được; design_weight = N_h/n_h để quy về dân số (Horvitz–Thompson). Dùng lại
sampling.stratified_sample + audit_grid.build_audit (đã test) — chỉ override stratum=rule.

Chạy:
  .venv/bin/python -m pipeline.ground_truth.make_audit_batch --tier SILVER --n 400
  # rồi mở dataset_out/ground_truth/audit_SILVER/audit_*.html, chấm, lưu verdicts_*.jsonl vào đó
  # sau đó: estimate (xem cuối file)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import audit_grid, sampling, stats
from .cli import _load_config, _paths

REPO = Path(__file__).resolve().parents[2]

_README = """# Mẻ audit {tier} — hướng dẫn chấm

Mục tiêu: đo PRECISION tier {tier} (chưa từng đo — audit GOLD 100% là GOLD). Mẫu {n} crop,
lấy NGẪU NHIÊN phân tầng theo rule (không thiên lệch), design_weight quy về dân số {pool}.

## Chấm (giống audit GOLD đã làm)
1. Mở lần lượt `audit_001.html`, `audit_002.html`, ... trong trình duyệt.
2. Mỗi ô: crop + glyph tham chiếu (nếu có) + ngữ cảnh trang + âm QN + ứng viên từ điển.
   Bấm 1 trong: **correct** (nhãn đúng) / **wrong_label** (crop đúng chữ nhưng nhãn sai) /
   **wrong_image** (crop cắt lỗi/dính/nhầm ô) / **unsure** (không chắc). Tiến độ tự lưu.
3. Chấm hết mọi batch → **Download JSON** → lưu thành `verdicts_001.jsonl`, `verdicts_002.jsonl`...
   ĐẶT NGAY trong thư mục này.

## Đo sau khi chấm
```
.venv/bin/python -m pipeline.ground_truth estimate \\
    --verdicts {dir} --manifest {dir}/manifest.jsonl --design stratified --p0 0.90
```
→ precision {tier} + CI95 (Wilson/Clopper-Pearson) + per-rule + acceptance.

## Phân tầng (design_weight = N_h / n_h)
{strata_table}
"""


def _write_plan_readme(tier: str, sample: pd.DataFrame, pool_size: int,
                       out_dir: Path, seed: int, source: str, conf: float) -> dict:
    """Ghi plan.json (thiết kế mẫu, cho tái lập + acceptance) + README.md."""
    strata = []
    for s, g in sample.groupby("stratum"):
        strata.append({"stratum": str(s), "n_h": int(len(g)),
                       "design_weight": round(float(g["design_weight"].iloc[0]), 3),
                       "N_h": int(round(float(g["design_weight"].iloc[0]) * len(g)))})
    # acceptance SRS tham chiếu cho claim precision>=p0 (thiết kế thật là stratified weighted)
    try:
        ap = stats.acceptance_plan(0.90, conf, 0.93, 0.90)
        acc = {"p0": 0.90, "n_srs_ref": ap.n, "accept_if_defects_le": ap.c,
               "lcb_at_c": round(ap.lcb_at_c, 4)}
    except Exception:
        acc = None
    plan = {
        "tier": tier, "n": int(len(sample)), "pool": int(pool_size),
        "coverage": round(len(sample) / pool_size, 4) if pool_size else None,
        "design": "stratified_random_by_rule", "seed": seed, "source": source,
        "conf": conf, "strata": strata,
        "estimation": ("stratified design-weighted (Horvitz–Thompson) overall + "
                       "Wilson/Clopper–Pearson per-rule; pipeline.ground_truth estimate --design stratified"),
        "acceptance_srs_reference": acc,
        "note": ("Mẫu KHÔNG rank theo nghi ngờ (unbiased để ĐO precision). "
                 "design_weight = N_h/n_h quy per-rule về dân số tier."),
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    tbl = "\n".join(f"- {s['stratum']}: n={s['n_h']} / N={s['N_h']} (w={s['design_weight']})"
                    for s in strata)
    (out_dir / "README.md").write_text(_README.format(
        tier=tier, n=len(sample), pool=pool_size, dir=out_dir, strata_table=tbl))
    return plan


def run(tier: str, n: int, labels_csv: Path, out_dir: Path, seed: int,
        min_per_rule: int, config: Path, batch_size: int) -> dict:
    df = pd.read_csv(labels_csv, dtype={"image_md5": str})
    pool = df[df["tier"] == tier].copy()
    if pool.empty:
        raise SystemExit(f"[audit] tier {tier!r} rỗng trong {labels_csv}")
    # stratify theo RULE (không phải risk-stratum) để đo per-rule + không thiên lệch
    pool["stratum"] = pool["rule"].astype(str)
    n = min(n, len(pool))

    sample = sampling.stratified_sample(pool, n, seed=seed, min_per_stratum=min_per_rule)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_dir / "sample.csv", index=False)
    _write_plan_readme(tier, sample, len(pool), out_dir, seed, Path(labels_csv).name, 0.95)

    cfg = _load_config(config)
    paths = _paths(cfg)
    qn_dict = None
    try:
        from core.text.dictionary import load_qn_to_nom
        if paths["qn_dict"].exists():
            qn_dict = load_qn_to_nom(str(paths["qn_dict"]))
    except Exception:
        pass

    res = audit_grid.build_audit(
        sample=sample,
        dataset_dir=paths["dataset_dir"],
        prepared_dir=paths["prepared_dir"],
        fd_dir=paths["fd_dir"],
        out_html=out_dir / "audit.html",
        out_manifest=out_dir / "manifest.jsonl",
        qn_dict=qn_dict,
        font_path=paths["font"],
        with_context=True,
        batch_size=batch_size,
    )

    # JSON máy-đọc-được cho mỗi audit_XXX.html
    from . import batch_json
    batch_json.export_dir(out_dir, batch_size, qn_dict or {})

    alloc = sample.groupby("stratum").agg(
        n=("item_id", "size"),
        design_weight=("design_weight", "first")).to_dict("index")
    print("=" * 64)
    print(f" MẺ AUDIT {tier} — n={len(sample)} (stratified theo rule, seed={seed})")
    print("=" * 64)
    print(f" Dân số {tier} = {len(pool)}; phủ mẫu = {100*len(sample)/len(pool):.1f}%")
    print(" Phân bổ theo rule (n / design_weight = N_h/n_h):")
    for r, a in alloc.items():
        print(f"   {r:26s}: n={a['n']:3d}  w={a['design_weight']:.1f}")
    print(f"\n HTML để chấm : {out_dir}/audit_*.html   (grid: {res})")
    print(f" Manifest     : {out_dir}/manifest.jsonl")
    print(f"\n → Chấm xong lưu verdicts_*.jsonl vào {out_dir}/, rồi:")
    print(f"   .venv/bin/python -m pipeline.ground_truth estimate \\")
    print(f"       --verdicts {out_dir} --manifest {out_dir}/manifest.jsonl \\")
    print(f"       --design stratified --p0 0.90")
    return {"tier": tier, "n": len(sample), "pool": len(pool), "alloc": alloc}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_audit_batch")
    ap.add_argument("--tier", default="SILVER")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--labels", default=str(REPO / "dataset_out" / "labels_final.csv"))
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-rule", type=int, default=40)
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--batch-size", type=int, default=150)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else (REPO / "dataset_out" / "ground_truth" / f"audit_{args.tier}")
    run(args.tier, args.n, Path(args.labels), out, args.seed,
        args.min_per_rule, Path(args.config), args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
