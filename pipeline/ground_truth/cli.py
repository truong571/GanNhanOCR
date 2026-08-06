"""Command-line entry point for Giai đoạn 0 — Tạo ground truth.

Run from the repo root, e.g.:

    .venv/bin/python -m pipeline.ground_truth rank
    .venv/bin/python -m pipeline.ground_truth plan --p0 0.97 --p-assumed 0.985
    .venv/bin/python -m pipeline.ground_truth sample --n 1150 --design stratified
    .venv/bin/python -m pipeline.ground_truth grid
    .venv/bin/python -m pipeline.ground_truth estimate --verdicts out/verdicts.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit_grid, estimate as est_mod, s3_signals, sampling, stats, suspicion

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO / "dataset_out" / "labels.csv"
DEFAULT_OUT = REPO / "dataset_out" / "ground_truth"
DEFAULT_CONFIG = REPO / "config" / "pipeline.yaml"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _paths(cfg: dict) -> dict:
    p = cfg.get("paths", {}) if cfg else {}
    return {
        "dataset_dir": REPO / "dataset_out",
        "prepared_dir": REPO / p.get("data_dir", "prepared"),
        "fd_dir": REPO / p.get("fd_cache_universal", "gannhanocr-fd"),
        "qn_dict": REPO / p.get("qn_to_nom_dict", "Dict/QuocNgu_SinoNom_TongHop3.csv"),
        "font": REPO / p.get("font_path", "font_diffusion/fonts/NomNaTong-Regular.ttf"),
    }


def _load_labels(args) -> pd.DataFrame:
    """Đọc labels.csv và GẮN tín hiệu S3 corpus-wide nếu có.

    Không có bước gắn này thì ~94% GOLD (luật s1_inter_s2_direct) chưa từng được đối chiếu
    với ảnh, và xếp hạng nghi ngờ hoàn toàn mù trước lớp đó — xem s3_signals.py.
    """
    labels = pd.read_csv(args.labels, dtype={"image_md5": str})
    if getattr(args, "no_s3_signals", False):
        print("[labels] BỎ QUA tín hiệu S3 corpus-wide (--no-s3-signals)")
        return labels
    try:
        labels, rep = s3_signals.attach(labels)
        print(f"[labels] {rep}")
    except FileNotFoundError as e:
        print(f"[labels] CẢNH BÁO: {e}\n"
              f"[labels] -> chạy tiếp KHÔNG có tín hiệu S3; ~94% GOLD sẽ không được soi ảnh.")
    return labels


def _ranked(args) -> pd.DataFrame:
    """Load or compute the ranked frame (cached to labels_ranked.csv)."""
    cache = Path(args.out) / "labels_ranked.csv"
    if cache.exists() and not getattr(args, "force", False):
        return pd.read_csv(cache, dtype={"image_md5": str})
    ranked = suspicion.add_suspicion(_load_labels(args))
    cache.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(cache, index=False)
    return ranked


# --------------------------------------------------------------------------- #
def cmd_rank(args) -> None:
    ranked = suspicion.add_suspicion(_load_labels(args))
    out = Path(args.out) / "labels_ranked.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)
    summ = suspicion.stratum_summary(ranked)
    print(f"[rank] {len(ranked):,} usable crops ranked -> {out}")
    print(summ.to_string(index=False))
    print(f"[rank] provably-compromised dup_defect rows: "
          f"{int(ranked['dup_defect'].sum()):,}")
    blind = int(ranked["s3_missing"].sum())
    print(f"[rank] hàng CHƯA từng được đối chiếu ảnh (s3_missing): {blind:,} "
          f"({blind / len(ranked):.1%})")
    if "head_disagree" in ranked.columns:
        print(f"[rank] head_disagree (đầu ArcFace chọn chữ khác nhãn): "
              f"{int(ranked['head_disagree'].sum()):,}")


def cmd_plan(args) -> None:
    print(f"[plan] target precision claim p0 = {args.p0}, confidence {args.conf}")
    if args.p_assumed:
        plan = stats.acceptance_plan(args.p0, args.conf, args.p_assumed, args.power)
        print(f"  Acceptance sampling (SRS): n = {plan.n}, accept if defects <= {plan.c}")
        print(f"    -> if <= {plan.c} defects, one-sided {args.conf:.0%} lower bound "
              f"= {plan.lcb_at_c:.4f} >= {args.p0}")
        print(f"    -> power at true precision {args.p_assumed}: {plan.power:.3f}")
    for hw in (0.01, 0.005):
        n = stats.required_n_for_halfwidth(args.p0, hw, args.conf, "wilson")
        print(f"  Two-sided Wilson CI +/-{hw:.3f} at p={args.p0}: n = {n}")


def cmd_sample(args) -> None:
    ranked = _ranked(args)
    if args.design == "srs":
        sample = sampling.simple_random_sample(ranked, args.n, args.seed)
    else:
        sample = sampling.stratified_sample(ranked, args.n, args.seed)
    out = Path(args.out) / f"sample_{args.design}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"[sample] design={args.design} n={len(sample)} seed={args.seed} -> {out}")
    if "stratum" in sample.columns:
        print(sample.groupby("stratum").size().to_string())


def cmd_grid(args) -> None:
    cfg = _load_config(Path(args.config))
    paths = _paths(cfg)
    sample = pd.read_csv(args.sample, dtype={"image_md5": str})
    qn_dict = None
    try:
        from core.text.dictionary import load_qn_to_nom
        if paths["qn_dict"].exists():
            qn_dict = load_qn_to_nom(str(paths["qn_dict"]))
    except Exception as e:  # dictionary is a nice-to-have for the UI, not required
        print(f"[grid] warning: dictionary not loaded ({e})", file=sys.stderr)
    res = audit_grid.build_audit(
        sample=sample,
        dataset_dir=paths["dataset_dir"],
        prepared_dir=paths["prepared_dir"],
        fd_dir=paths["fd_dir"],
        out_html=Path(args.out) / "audit.html",
        out_manifest=Path(args.out) / "manifest.jsonl",
        qn_dict=qn_dict,
        font_path=paths["font"],
        with_context=not args.no_context,
        batch_size=args.batch_size,
    )
    print(f"[grid] {json.dumps(res, ensure_ascii=False, indent=2)}")


def cmd_estimate(args) -> None:
    include_ai = getattr(args, "include_ai", False)
    if include_ai:
        print("[cảnh báo] ĐANG dùng verdict do MÁY chấm (source=ai_vision) làm ground truth "
              "để tính precision/CI/acceptance — kết quả KHÔNG phải precision trên nhãn "
              "người chấm, chỉ dùng để thăm dò")
    verdicts = est_mod.load_verdicts(args.verdicts, include_ai=include_ai)
    manifest = est_mod.load_manifest(args.manifest)
    joined = est_mod.join_manifest(verdicts, manifest,
                                   drop_unknown=args.drop_unknown)

    unlabeled = None
    surrogate = args.surrogate
    if args.ranked and Path(args.ranked).exists():
        ranked = pd.read_csv(args.ranked, dtype={"image_md5": str})
        audited = set(manifest["item_id"])
        import hashlib
        ranked["item_id"] = ranked["image"].map(
            lambda v: hashlib.sha1(str(v).encode()).hexdigest()[:16])
        un = ranked[~ranked["item_id"].isin(audited)]
        # PPI chỉ hợp lệ khi biến thay thế phủ (gần) TOÀN dân số. `s3_cosine` của engine
        # chỉ phủ ~30-40% (nó bỏ qua mọi hàng GOLD-direct), nên PPI vẫn luôn bị bỏ. Các cột
        # từ s3_corpus phủ 100% phần chấm được -> chọn cột phủ rộng nhất và NÓI RÕ ra.
        cands = [c for c in ("s3_bank_cos", "s3_head_cos", "s3_cosine")
                 if c in un.columns and c in joined.columns]
        cov = {c: float(pd.to_numeric(un[c], errors="coerce").notna().mean()) for c in cands}
        if surrogate == "auto":
            surrogate = max(cov, key=cov.get) if cov else "s3_cosine"
        if cov:
            print("[estimate] độ phủ biến thay thế trên dân số chưa chấm: "
                  + ", ".join(f"{c}={v:.1%}" for c, v in cov.items())
                  + f"  -> dùng {surrogate!r}")
        unlabeled = pd.to_numeric(un.get(surrogate), errors="coerce").to_numpy()
    elif surrogate == "auto":
        surrogate = "s3_cosine"

    rep = est_mod.estimate(
        joined, conf=args.conf, p0=args.p0, design=args.design,
        surrogate_col=surrogate, unlabeled_scores=unlabeled,
    )
    if rep.n_purposive_excluded:
        p = rep.purposive or {}
        print(f"[estimate] ĐÃ LOẠI {rep.n_purposive_excluded} ô thuộc mẫu CHỦ ĐÍCH "
              f"({', '.join(p.get('audit_batch') or ['design_weight rỗng'])}) khỏi mọi "
              f"ước lượng precision.")
        if p.get("error_rate") is not None:
            print(f"           (tỷ lệ lỗi trong tầng chủ đích: {p['error_rate']:.1%} trên "
                  f"{p['n_scored']} ô — chỉ để hiệu chỉnh ngưỡng, KHÔNG phải precision)")
    out = Path(args.out) / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[estimate] audited={rep.n_audited} scored={rep.n_scored} "
          f"correct={rep.n_correct} unsure={rep.n_unsure}")
    print(f"  precision            = {rep.precision:.4f}")
    print(f"  Wilson 95% CI        = [{rep.wilson_ci[0]:.4f}, {rep.wilson_ci[1]:.4f}]")
    print(f"  Clopper-Pearson CI   = [{rep.cp_ci[0]:.4f}, {rep.cp_ci[1]:.4f}]")
    print(f"  one-sided lower      = {rep.cp_lower_one_sided:.4f}")
    if rep.weighted_precision is not None:
        print(f"  weighted precision   = {rep.weighted_precision:.4f} "
              f"[{rep.weighted_ci[0]:.4f}, {rep.weighted_ci[1]:.4f}]")
    if rep.ppi_precision is not None:
        print(f"  PPI precision        = {rep.ppi_precision:.4f} "
              f"[{rep.ppi_ci[0]:.4f}, {rep.ppi_ci[1]:.4f}]")
    elif rep.ppi_note:
        print(f"  PPI                  = skipped ({rep.ppi_note.split(':',1)[1].strip()})")
    if rep.acceptance is not None:
        a = rep.acceptance
        print(f"  acceptance (p0={a['p0']}): defects={a['defects']} "
              f"LCB={a['one_sided_lower_bound']:.4f} -> "
              f"{'ACCEPT' if a['accept'] else 'REJECT'} ({a['note']})")
    print(f"  wrong_label={rep.n_wrong_label} wrong_image={rep.n_wrong_image}")
    print(f"[estimate] full report -> {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.ground_truth",
                                description="Giai đoạn 0 — Tạo ground truth")
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--conf", type=float, default=0.95)
    p.add_argument("--no-s3-signals", action="store_true", dest="no_s3_signals",
                   help="KHÔNG gắn tín hiệu S3 corpus-wide (dataset_out/fusion/s3_corpus.csv); "
                        "chỉ để tái lập lại cách xếp hạng cũ, khi đó ~94%% GOLD không được soi ảnh")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rank", help="score every usable crop by error suspicion")
    r.set_defaults(func=cmd_rank)

    pl = sub.add_parser("plan", help="sample-size / acceptance planning (pure stats)")
    pl.add_argument("--p0", type=float, default=0.97)
    pl.add_argument("--p-assumed", type=float, default=0.985, dest="p_assumed")
    pl.add_argument("--power", type=float, default=0.90)
    pl.set_defaults(func=cmd_plan)

    s = sub.add_parser("sample", help="draw an audit sample")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--design", choices=["stratified", "srs"], default="stratified")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--force", action="store_true", help="recompute the ranking cache")
    s.set_defaults(func=cmd_sample)

    g = sub.add_parser("grid", help="render blinded audit HTML + manifest")
    g.add_argument("--sample", required=True)
    g.add_argument("--no-context", action="store_true", help="skip scan-context crops")
    g.add_argument("--batch-size", type=int, default=150,
                   help="split HTML into batches of this many items (0 = single file)")
    g.set_defaults(func=cmd_grid)

    e = sub.add_parser("estimate", help="verdicts -> precision + CI + acceptance + PPI")
    e.add_argument("--verdicts", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--ranked", default=str(DEFAULT_OUT / "labels_ranked.csv"),
                   help="ranked csv for PPI unlabeled surrogate scores")
    e.add_argument("--p0", type=float, default=None)
    e.add_argument("--design", choices=["stratified", "srs"], default="stratified")
    e.add_argument("--drop-unknown-verdicts", action="store_true", dest="drop_unknown",
                   help="bỏ verdict không thuộc mẻ (bản xuất cũ bị trộn localStorage)")
    e.add_argument("--surrogate", default="auto",
                   choices=["auto", "s3_bank_cos", "s3_head_cos", "s3_cosine"],
                   help="biến thay thế cho PPI; 'auto' chọn cột phủ dân số rộng nhất")
    e.add_argument("--include-ai-verdicts", action="store_true", dest="include_ai",
                   help="CHO PHÉP dùng verdict do MÁY chấm (source=ai_vision) làm ground "
                        "truth để tính precision; mặc định TẮT — chỉ verdict người chấm")
    e.set_defaults(func=cmd_estimate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
