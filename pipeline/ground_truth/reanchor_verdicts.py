"""P0.2 — Neo lại 846 verdict human-audit về dataset THẾ HỆ HIỆN TẠI.

Vì sao cần: verdict cũ neo theo item_id = sha1(image_path) của thế hệ cũ. Khi build lại
(nhất là yen4/yen11 re-OCR), tên file + bbox đổi → khớp image-path chỉ còn 33% (mồ côi 67%).
Nhưng verdict là phán quyết về (VỊ TRÍ ký tự trên trang, NHÃN được gán) — vị trí bền hơn
tên file. Nên ta neo theo (book, page, bbox-IoU) và KIỂM nhãn.

Quan trọng: re-OCR có thể đổi NHÃN ở cùng vị trí. Verdict cũ ("correct"/"wrong") nói về
NHÃN CŨ. Nếu nhãn mới KHÁC nhãn đã audit → verdict không còn áp được → đánh dấu re-audit,
KHÔNG mang mù verdict cũ sang.

Đầu vào:
  --old-manifest  dataset_out/ground_truth/audit_gold/manifest.jsonl  (item_id,book,page,bbox,label)
  --verdicts      gộp mọi verdicts_*.jsonl                             (item_id,verdict)
  --new-labels    dataset_out/labels.csv (thế hệ HIỆN TẠI)            (book,page,bbox,label,image,tier)
Đầu ra:
  dataset_out/ground_truth/verdicts_reanchored.csv
    item_id, verdict, book, page, iou, image_old(none), image_new, label_old, label_new,
    label_match, tier_new, status  (matched | label_changed | orphan)

Chạy:
  .venv/bin/python -m pipeline.ground_truth.reanchor_verdicts \
      --old-manifest dataset_out/ground_truth/audit_gold/manifest.jsonl \
      --verdicts dataset_out/ground_truth \
      --new-labels dataset_out/labels.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
IOU_MIN = 0.5          # ngưỡng khớp vị trí


def _parse_bbox(v) -> tuple[float, float, float, float] | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if not v or len(v) != 4:
        return None
    return tuple(float(x) for x in v)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _load_verdicts(path: Path) -> dict[str, str]:
    files = ([path] if path.is_file()
             else sorted(Path(p) for p in glob.glob(str(path / "verdicts_*.jsonl"))))
    out = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[str(r["item_id"])] = str(r["verdict"])
    return out


def run(old_manifest: Path, verdicts_path: Path, new_labels: Path) -> pd.DataFrame:
    man = {}
    with open(old_manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                m = json.loads(line)
                man[str(m["item_id"])] = m
    verd = _load_verdicts(verdicts_path)

    lab = pd.read_csv(new_labels, dtype=str)
    lab["_bbox"] = lab["bbox"].map(_parse_bbox)
    # index theo (book, page) để tìm ứng viên nhanh
    by_bp: dict[tuple, list] = {}
    for _, r in lab.iterrows():
        if r["_bbox"] is None:
            continue
        by_bp.setdefault((str(r["book"]), str(r["page"])), []).append(r)

    rows = []
    for item_id, v in verd.items():
        m = man.get(item_id)
        if not m:
            continue
        book, page = str(m.get("book")), str(m.get("page"))
        bb_old = _parse_bbox(m.get("bbox"))
        lab_old = str(m.get("label") or "")
        cands = by_bp.get((book, page), [])
        best, best_iou = None, 0.0
        if bb_old is not None:
            for r in cands:
                j = _iou(bb_old, r["_bbox"])
                if j > best_iou:
                    best, best_iou = r, j
        if best is None or best_iou < IOU_MIN:
            rows.append({"item_id": item_id, "verdict": v, "book": book, "page": page,
                         "iou": round(best_iou, 3), "image_new": "", "label_old": lab_old,
                         "label_new": "", "label_match": False, "tier_new": "", "status": "orphan"})
            continue
        lab_new = str(best["label"] or "")
        match = (lab_new == lab_old)
        rows.append({"item_id": item_id, "verdict": v, "book": book, "page": page,
                     "iou": round(best_iou, 3), "image_new": best["image"],
                     "label_old": lab_old, "label_new": lab_new, "label_match": match,
                     "tier_new": best["tier"],
                     "status": "matched" if match else "label_changed"})
    df = pd.DataFrame(rows)

    outp = REPO / "dataset_out" / "ground_truth" / "verdicts_reanchored.csv"
    df.to_csv(outp, index=False)

    n = len(df)
    st = df["status"].value_counts().to_dict()
    print("=" * 64)
    print(f" RE-ANCHOR {n} verdict → dataset hiện tại (IoU≥{IOU_MIN})")
    print("=" * 64)
    print(f"   matched        : {st.get('matched', 0):4d}  (vị trí + nhãn khớp → verdict DÙNG được)")
    print(f"   label_changed  : {st.get('label_changed', 0):4d}  (vị trí khớp, NHÃN đổi → phải re-audit)")
    print(f"   orphan         : {st.get('orphan', 0):4d}  (không thấy vị trí → crop biến mất/không build)")
    if "book" in df:
        print("\n   theo book (matched / tổng):")
        for b, g in df.groupby("book"):
            print(f"     {b}: {int((g['status']=='matched').sum())}/{len(g)}")
    usable = df[df["status"] == "matched"]
    if len(usable):
        vc = usable["verdict"].value_counts().to_dict()
        print(f"\n   verdict DÙNG được: {vc}")
    print(f"\n -> {outp}")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.reanchor_verdicts")
    ap.add_argument("--old-manifest",
                    default=str(REPO / "dataset_out/ground_truth/audit_gold/manifest.jsonl"))
    ap.add_argument("--verdicts", default=str(REPO / "dataset_out/ground_truth"))
    ap.add_argument("--new-labels", default=str(REPO / "dataset_out/labels.csv"))
    args = ap.parse_args(argv)
    run(Path(args.old_manifest), Path(args.verdicts), Path(args.new_labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
