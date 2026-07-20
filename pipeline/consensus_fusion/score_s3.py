"""Driver kênh S3 must-pass — chấm điểm visual cho MỌI crop theo NHÃN ĐÃ GÁN.

Khác `maybe_s3` của engine (chỉ chạy ca phá-hòa, BỎ GOLD-direct nên chỉ phủ ~30-41%):
ở đây mỗi crop được chấm với CHÍNH nhãn của nó — kể cả GOLD-direct — để cổng must-pass
có thể BẮT nhãn GOLD sai (kim∩dict đồng thuận nhưng ảnh thực ra là chữ khác).

Hai chế độ:
  --validate : chấm 846 crop human-audit (dataset_out/ground_truth/verdicts_*.jsonl),
               so AUC nhiều tín hiệu S3 vs verdict correct/wrong → CHỌN tín hiệu must-pass
               bằng đo trên nhãn thật, không tin memory (kiểm/bác claim "head +18pt").
  --all      : chấm mọi crop char (dedup theo image) trong labels_remediated.csv →
               dataset_out/fusion/s3_corpus.csv (kênh s3 phủ ~100% cho fuse_stage).

Tín hiệu chấm mỗi crop (embed thẳng crop PNG qua NomEncoder đã train):
  head_cos    cosine-logit của NHÃN từ đầu ArcFace (1591 lớp)   — "crop có phải nhãn?"
  head_prob   softmax(logits)[nhãn]
  head_margin logit[nhãn] − max(logit lớp khác)  (>0 nếu nhãn là argmax)
  head_isarg  1.0 nếu argmax head == nhãn, else 0.0
  bank_cos    max cosine tới ngân hàng tham chiếu của nhãn (proto/simfont/fd) — tín hiệu "production"
  mls         max-logit-score (độ giống-glyph, độc lập nhãn; OOD)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import fusion

REPO = Path(__file__).resolve().parents[2]
_VERDICT_Y = {"correct": 1.0, "wrong_label": 0.0, "wrong_image": 0.0, "unsure": np.nan}
SIGNALS = ["head_cos", "head_prob", "head_margin", "head_isarg", "bank_cos", "mls"]


def _load_vs3():
    cfg = yaml.safe_load((REPO / "config" / "pipeline.yaml").read_text())
    paths = cfg["paths"]
    from pipeline.align_engine.visual_signal import VisualS3
    simfont = str(REPO / paths["fd_cache_similar"]) if paths.get("fd_cache_similar") else ""
    return VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]), simfont_dir=simfont)


def score_crop(vs3, crop_path: str, label: str) -> dict:
    """Các tín hiệu S3 cho một crop vs nhãn đã gán. NaN nếu embed/nhãn không dùng được."""
    out = {s: np.nan for s in SIGNALS}
    if not label or len(label) != 1:
        return out
    emb = vs3.enc.embed_path(crop_path)
    if emb is None:
        return out
    # đầu ArcFace (nếu nhãn ∈ vocab)
    lg = vs3.enc.logits(emb)
    if lg is not None and label in vs3.lab2idx:
        idx = vs3.lab2idx[label]
        li = float(lg[idx])
        out["head_cos"] = li
        z = lg - lg.max()
        ez = np.exp(z)
        out["head_prob"] = float(ez[idx] / ez.sum())
        others = np.delete(lg, idx)
        out["head_margin"] = li - float(others.max()) if others.size else li
        out["head_isarg"] = 1.0 if int(np.argmax(lg)) == idx else 0.0
        out["mls"] = float(lg.max())
    elif lg is not None:
        out["mls"] = float(lg.max())
    # ngân hàng tham chiếu (production bank cosine)
    tc = vs3.tier_cosines(emb, label)          # {tier: cosine_raw}
    if tc:
        out["bank_cos"] = float(max(tc.values()))
    return out


def _load_audited() -> list[dict]:
    gt = REPO / "dataset_out" / "ground_truth"
    man = {}
    with open(gt / "audit_gold" / "manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                m = json.loads(line)
                man[str(m["item_id"])] = m
    verd = {}
    for f in sorted(glob.glob(str(gt / "verdicts_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    verd[str(r["item_id"])] = str(r["verdict"])
    rows = []
    for item_id, v in verd.items():
        m = man.get(item_id)
        if not m:
            continue
        rows.append({"image": m["image"], "label": m.get("label", ""),
                     "y": _VERDICT_Y.get(v, np.nan), "verdict": v})
    return rows


def cmd_validate(args) -> None:
    vs3 = _load_vs3()
    rows = _load_audited()
    root = REPO / "dataset_out"
    print(f"[validate] chấm {len(rows)} crop audited theo NHÃN đã gán ...", flush=True)
    recs = []
    for i, r in enumerate(rows):
        sc = score_crop(vs3, str(root / r["image"]), r["label"])
        recs.append({**r, **sc})
        if (i + 1) % 200 == 0:
            print(f"   {i + 1}/{len(rows)}", flush=True)
    df = pd.DataFrame(recs)
    y = df["y"].to_numpy(float)
    lab = np.isfinite(y)
    print(f"\n[validate] {int(lab.sum())} có nhãn ({int((y[lab]==1).sum())}+/"
          f"{int((y[lab]==0).sum())}−). AUC mỗi tín hiệu S3 vs verdict (correct=1):\n")
    table = []
    for s in SIGNALS:
        v = df[s].to_numpy(float)
        ok = lab & np.isfinite(v)
        cov = float(np.isfinite(v).mean())
        if ok.sum() < 5 or len(set(y[ok])) < 2:
            table.append((s, cov, float("nan"), int(ok.sum())))
            continue
        auc = fusion.roc_auc(v[ok], y[ok])
        table.append((s, cov, auc, int(ok.sum())))
    table.sort(key=lambda t: (-(t[2] if t[2] == t[2] else -1)))
    print(f"   {'signal':12s} {'coverage':>9s} {'AUC':>7s} {'n':>6s}")
    for s, cov, auc, n in table:
        print(f"   {s:12s} {cov:9.1%} {auc:7.3f} {n:6d}")
    best = next((t for t in table if t[2] == t[2]), None)
    if best:
        print(f"\n[validate] => must-pass nên dùng '{best[0]}' (AUC {best[2]:.3f}, "
              f"coverage {best[1]:.0%}). So bank_cos (production) để kiểm claim head.")
    outp = REPO / "dataset_out" / "fusion" / "s3_validate.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    print(f"[validate] -> {outp}")


def cmd_all(args) -> None:
    vs3 = _load_vs3()
    root = REPO / "dataset_out"
    rem = pd.read_csv(root / "labels_remediated.csv", dtype={"image_md5": str})
    rem = rem[rem["image"].astype(bool) & rem["label"].astype(str).str.len().eq(1)]
    rem = rem.drop_duplicates("image", keep="first").reset_index(drop=True)
    if args.limit:
        rem = rem.iloc[:args.limit]
    print(f"[all] chấm {len(rem)} crop char (dedup) ...", flush=True)
    recs = []
    for i, (_, r) in enumerate(rem.iterrows()):
        sc = score_crop(vs3, str(root / r["image"]), str(r["label"]))
        recs.append({"image": r["image"], "label": r["label"], "tier": r["tier"], **sc})
        if (i + 1) % 500 == 0:
            print(f"   {i + 1}/{len(rem)}", flush=True)
    outp = root / "fusion" / "s3_corpus.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(outp, index=False)
    print(f"[all] -> {outp}  ({len(recs)} crop, phủ must-pass corpus-wide)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.consensus_fusion.score_s3")
    ap.add_argument("--validate", action="store_true", help="đo AUC tín hiệu S3 trên 846 verdict")
    ap.add_argument("--all", action="store_true", help="chấm mọi crop char -> s3_corpus.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if args.validate:
        cmd_validate(args)
    elif args.all:
        cmd_all(args)
    else:
        ap.error("chọn --validate hoặc --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
