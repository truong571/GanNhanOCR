"""Giai đoạn 2 glue — chạy vote-fusion như một LỚP PHỦ trong pipeline chính.

Path A (0 chi phí): chỉ dùng hai kênh ĐỘC LẬP-ĐẦU-VÀO đã có sẵn trong đầu ra của engine
đóng băng —
  s3    = cosine đầu ArcFace (visual_signal),  ~41% crop có điểm
  dict  = độ khả tín từ điển QN (label ∈ qn_to_nom[syllable])
— hiệu chỉnh trên 846 phán quyết human-audit (ground_truth/verdicts_*.jsonl), rồi áp
gate bất đối xứng (gating.py). Các kênh tương quan nặng (qwen, nna_lobo) VẮNG ở Path A;
fuser tự coi chúng là missing (mean-impute + cờ missing). Đây là bằng chứng khung SOTA
chạy end-to-end trên dữ liệu thật mà KHÔNG cần API/train; thêm kênh nặng sau (Path B —
xem README §Drivers).

KHÔNG phá gốc: không đụng crops / gold / silver / labels_remediated.csv. Chỉ ghi thêm:
    dataset_out/fusion/channels.csv      feature + y mỗi crop (đầu vào cho fuser)
    dataset_out/fusion/fused.csv         fused_P + quyết định/lý do gate mỗi crop
    dataset_out/fusion/labels_fused.csv  labels_remediated + cột fusion + tier_fused (đề xuất)
    dataset_out/fusion/summary.json      AUC, đếm gate, coverage, số tier đổi

Chạy:
    .venv/bin/python -m pipeline.consensus_fusion.fuse_stage --config config/pipeline.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.text.dictionary import load_qn_to_nom
from . import fusion, gating

REPO = Path(__file__).resolve().parents[2]

# verdict (human-audit) -> nhãn nhị phân y cho fuser
_VERDICT_Y = {"correct": 1.0, "wrong_label": 0.0, "wrong_image": 0.0, "unsure": np.nan}

# Nguồn verdict do MÁY chấm. MẶC ĐỊNH bị loại: nếu nạp verdict AI làm nhãn y để hiệu chỉnh
# fuser, ta biến nhãn máy thành "ground truth" (đúng blocker khoa học "SILVER = AI-audit").
AI_VERDICT_SOURCES = {"ai_vision"}


def verdict_source(rec: dict) -> str:
    """Nguồn của một bản ghi verdict; vắng trường `source` => coi là người chấm."""
    return str(rec.get("source") or "human").strip().lower()


def load_verdicts(gt_dir: Path, include_ai: bool = False,
                  tag: str = "fuse_stage") -> dict[str, str]:
    """Gộp mọi verdicts_*.jsonl theo item_id (batch sau ghi đè batch trước).

    Quét ĐỆ QUY vì file verdict thật nằm trong thư mục con (audit_SILVER/verdicts_ai.jsonl).
    MẶC ĐỊNH chỉ nhận verdict của NGƯỜI; verdict source ∈ AI_VERDICT_SOURCES bị bỏ qua trừ
    khi khai báo tường minh include_ai=True (--include-ai-verdicts).
    """
    verd: dict[str, str] = {}
    kept: dict[str, int] = {}
    skipped: dict[str, int] = {}
    files = sorted(glob.glob(str(gt_dir / "**" / "verdicts_*.jsonl"), recursive=True))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    src = verdict_source(r)
                    if src in AI_VERDICT_SOURCES and not include_ai:
                        skipped[src] = skipped.get(src, 0) + 1
                        continue
                    kept[src] = kept.get(src, 0) + 1
                    verd[str(r["item_id"])] = str(r["verdict"])
    report_verdict_sources(tag, files, kept, skipped, include_ai)
    return verd


def report_verdict_sources(tag: str, files: list[str], kept: dict[str, int],
                           skipped: dict[str, int], include_ai: bool) -> None:
    """In minh bạch: nạp bao nhiêu / bỏ bao nhiêu verdict, theo từng nguồn."""
    def _fmt(d: dict[str, int]) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(d.items())) or "0"
    print(f"[{tag}] verdicts: {len(files)} file, nạp {sum(kept.values())} ({_fmt(kept)}), "
          f"bỏ qua {sum(skipped.values())} ({_fmt(skipped)})")
    if skipped:
        n_ai = sum(skipped.values())
        print(f"[cảnh báo] bỏ qua {n_ai} verdict source=ai_vision "
              f"(dùng --include-ai-verdicts nếu thực sự muốn)")
    if include_ai and kept:
        ai_kept = sum(v for k, v in kept.items() if k in AI_VERDICT_SOURCES)
        if ai_kept:
            print(f"[cảnh báo] ĐANG dùng {ai_kept} verdict do MÁY chấm làm nhãn — "
                  f"kết quả KHÔNG phải ground truth người chấm, chỉ dùng để thăm dò")


def _load_item_to_image(gt_dir: Path) -> dict[str, str]:
    """manifest.jsonl (quét đệ quy): item_id -> image path (khớp cột `image` của labels)."""
    id2img: dict[str, str] = {}
    for man in sorted(glob.glob(str(gt_dir / "**" / "manifest.jsonl"), recursive=True)):
        with open(man, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    m = json.loads(line)
                    id2img[str(m["item_id"])] = str(m["image"])
    return id2img


def build_channels(rem: pd.DataFrame, verd: dict[str, str],
                   id2img: dict[str, str], qn_to_nom: dict[str, list[str]]) -> pd.DataFrame:
    """Một dòng / crop (dedup theo image). Cột: image, tier, s3, dict, flag_quality_flag, y."""
    # dedup theo image: ưu tiên giữ dòng có s3_cosine (kênh must_pass) để không mất điểm.
    df = rem.copy()
    df["_s3num"] = pd.to_numeric(df.get("s3_cosine"), errors="coerce")
    df = (df.sort_values("_s3num", ascending=False, na_position="last")
            .drop_duplicates("image", keep="first")
            .reset_index(drop=True))

    ch = pd.DataFrame({"image": df["image"].astype(str)})
    ch["tier"] = df.get("tier")
    ch["label"] = df.get("label")
    ch["syllable"] = df.get("syllable")
    ch["s3"] = df["_s3num"].to_numpy()

    # dict: 1.0 nếu nhãn Nôm nằm trong ứng viên từ điển của âm QN, else 0.0
    def _dict_plaus(row) -> float:
        cands = qn_to_nom.get(str(row["syllable"] or "").lower(), [])
        return 1.0 if str(row["label"]) in cands else 0.0
    ch["dict"] = df.apply(_dict_plaus, axis=1).to_numpy()

    # cờ chất lượng crop -> gate.quality_flag (chặn promote)
    seg = df.get("seg_flag")
    if seg is None:
        ch["flag_quality_flag"] = False
    else:
        s = seg.astype(str).str.strip().str.lower()
        ch["flag_quality_flag"] = ~s.isin(["", "nan", "none", "0", "false", "ok", "clean"])

    # y từ verdict (join qua image)
    img2y: dict[str, float] = {}
    for item_id, v in verd.items():
        img = id2img.get(item_id)
        if img is not None:
            img2y[img] = _VERDICT_Y.get(v, np.nan)
    ch["y"] = ch["image"].map(img2y)
    return ch


def run(config: str, tau: float, l2: float, include_ai: bool = False) -> dict:
    cfg = yaml.safe_load((REPO / config).read_text())
    paths = cfg["paths"]
    out_root = REPO / cfg.get("output", {}).get("dir", "dataset_out")
    rem_path = out_root / "labels_remediated.csv"
    gt_dir = out_root / "ground_truth"
    fus_dir = out_root / "fusion"
    fus_dir.mkdir(parents=True, exist_ok=True)

    if not rem_path.exists():
        print(f"[fuse_stage] SKIP: thiếu {rem_path} (chạy Giai đoạn 1 remediation trước)")
        return {"skipped": "no_remediated"}
    verd = load_verdicts(gt_dir, include_ai=include_ai)
    id2img = _load_item_to_image(gt_dir)
    if not verd or not id2img:
        print(f"[fuse_stage] SKIP: thiếu verdicts NGƯỜI chấm/manifest trong {gt_dir} "
              f"(chạy Giai đoạn 0; --include-ai-verdicts nếu muốn dùng verdict máy)")
        return {"skipped": "no_verdicts"}

    rem = pd.read_csv(rem_path, dtype={"image_md5": str})
    qn_to_nom = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    ch = build_channels(rem, verd, id2img, qn_to_nom)
    ch.to_csv(fus_dir / "channels.csv", index=False)

    labeled = ch["y"].notna()
    n_lab = int(labeled.sum())
    if n_lab < 10:
        print(f"[fuse_stage] SKIP fit: chỉ {n_lab} dòng có nhãn (cần >=10)")
        return {"skipped": "too_few_labels", "labeled": n_lab}

    names = ["s3", "dict"]
    X = ch[names].to_numpy(float)
    yv = ch["y"].to_numpy(float)
    lab_np = labeled.to_numpy()

    fuser = fusion.LogisticFuser(l2=l2).fit(X[lab_np], yv[lab_np], names=names)
    raw = fuser.predict_proba(X, names=names)
    cal = fusion.IsotonicCalibrator().fit(raw[lab_np], yv[lab_np])
    P = cal.transform(raw)
    auc = fusion.roc_auc(raw[lab_np], yv[lab_np])

    flags = pd.DataFrame({"quality_flag": ch["flag_quality_flag"].astype(bool)}, index=ch.index)
    scores = ch[["s3", "dict"]]
    gate = gating.apply_gate(P, flags, scores, gating.GateConfig(tau_promote=tau))

    # --- LỚP PHỦ: đề xuất tier_fused, KHÔNG ghi đè tier gốc -------------------- #
    out = ch.copy()
    out["fused_P"] = P
    out["fusion_decision"] = gate.decision.values
    out["fusion_reason"] = gate.reason.values
    tier_fused = out["tier"].astype(object).copy()
    promote = (out["fusion_decision"] == "promote_gold") & (out["tier"] != "GOLD")
    demote = out["fusion_decision"] == "demote_review"
    tier_fused[promote] = "GOLD"
    tier_fused[demote] = "REVIEW"
    out["tier_fused"] = tier_fused
    out.to_csv(fus_dir / "fused.csv", index=False)

    # labels_fused.csv = remediated (dedup theo image) + cột fusion (chỉ thêm, không sửa)
    rem_dedup = (rem.assign(_s3=pd.to_numeric(rem.get("s3_cosine"), errors="coerce"))
                    .sort_values("_s3", ascending=False, na_position="last")
                    .drop_duplicates("image", keep="first").drop(columns="_s3"))
    merged = rem_dedup.merge(
        out[["image", "fused_P", "fusion_decision", "fusion_reason", "tier_fused"]],
        on="image", how="left")
    merged.to_csv(fus_dir / "labels_fused.csv", index=False)

    s3_cov_lab = float(np.isfinite(X[lab_np, 0]).mean())
    summary = {
        "labeled_rows": n_lab,
        "positives": int((yv[lab_np] == 1).sum()),
        "negatives": int((yv[lab_np] == 0).sum()),
        "s3_coverage_all": float(np.isfinite(X[:, 0]).mean()),
        "s3_coverage_labeled": s3_cov_lab,
        "train_auc": round(float(auc), 4),
        "gate": gate.counts,
        "tier_promote_suggested": int(promote.sum()),
        "tier_demote_suggested": int(demote.sum()),
        "crops_scored": int(len(out)),
        "tau_promote": tau,
        "include_ai_verdicts": bool(include_ai),
        "verdict_label_source": "ai+human" if include_ai else "human_only",
        "channels_present": ["s3", "dict"],
        "channels_absent": ["qwen", "nna_lobo"],
        "note": "Path A overlay: non-destructive; heavy channels absent; n_eff needs >=2 votes (Path B).",
    }
    (fus_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"[fuse_stage] fit trên {n_lab} nhãn audit "
          f"({summary['positives']}+/{summary['negatives']}-), train AUC={auc:.3f}, "
          f"s3-cov(labeled)={s3_cov_lab:.0%}")
    print(f"[fuse_stage] gate: {gate.summary()}  "
          f"-> promote {int(promote.sum())} | demote {int(demote.sum())} (đề xuất, overlay)")
    print(f"[fuse_stage] -> {fus_dir}/  (channels.csv, fused.csv, labels_fused.csv, summary.json)")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.consensus_fusion.fuse_stage",
        description="Giai đoạn 2 (Path A) — vote fusion overlay trong pipeline chính")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--tau", type=float, default=0.90, help="ngưỡng P hiệu chỉnh để promote GOLD")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--include-ai-verdicts", action="store_true",
                   help="CHO PHÉP dùng verdict do MÁY chấm (source=ai_vision) làm nhãn y; "
                        "mặc định TẮT — chỉ verdict người chấm mới được coi là ground truth")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.config, args.tau, args.l2, include_ai=args.include_ai_verdicts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
