"""Đo KHOẢNG CÁCH MIỀN (domain gap) crop-thật ↔ glyph-gannhanocr-fd.

Câu hỏi: ảnh sinh trong gannhanocr-fd (font sạch, nền trắng) có đủ GIỐNG ảnh crop
thật (ván khắc, nền giấy) để encoder so khớp được không?

Đo 3 thứ:
  1) Độ giống (cosine) trong encoder:
       - cùng chữ, crop↔crop      (in-domain, ngưỡng trên)
       - cùng chữ, crop↔FD-glyph  (cross-domain — cái ta quan tâm)
       - khác chữ, crop↔FD-glyph  (negative — phải thấp hơn hẳn)
     Nếu (cùng-chữ crop↔FD) >> (khác-chữ) → encoder ĐÃ bắc cầu được khoảng cách miền.
  2) retrieval@1 khi CHỈ có glyph FD làm tham chiếu (kịch bản chữ 0-crop)
     vs khi có crop thật làm tham chiếu → chênh lệch = giá phải trả của domain gap.
  3) Dump vài cặp (crop, FD-glyph) cùng chữ để xem tận mắt.

Run:
  .venv/bin/python evaluation/ver_new/eval_fd_domain_gap.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402

HERE = Path(__file__).resolve().parent
FD = REPO / "gannhanocr-fd"
D = HERE / "dataset_out"


def fd_path(ch):
    hx = f"{ord(ch):X}"
    for q in (FD / f"U+{hx}.png", FD / hx[:2] / f"U+{hx}.png"):
        if q.exists():
            return q
    h = list(FD.rglob(f"U+{hx}.png"))
    return h[0] if h else None


def cos(a, b):
    return float(np.dot(a, b))


def main():
    random.seed(0)
    enc = VisualS3(REPO, fd_dir="").enc
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == "GOLD" and r["image"] and r["label"] and _is_cjk(r["label"])]
    by_char = defaultdict(list)
    for r in rows:
        by_char[r["label"]].append(r["image"])
    # chars with >=2 real crops AND an fd glyph
    chars = [c for c, imgs in by_char.items() if len(imgs) >= 2 and fd_path(c)]
    random.shuffle(chars)
    sample = chars[:400]
    print(f"chars usable (>=2 crop + fd glyph): {len(chars)} | sampling {len(sample)}", flush=True)

    # cache crop embeds (1 query + 1 other per char) and fd embeds
    crop_emb, fd_emb = {}, {}
    for c in sample:
        e0 = enc.embed_path(str(D / by_char[c][0]))
        e1 = enc.embed_path(str(D / by_char[c][1]))
        ef = enc.embed_path(str(fd_path(c)))
        if e0 is None or e1 is None or ef is None:
            continue
        crop_emb[c] = (e0, e1); fd_emb[c] = ef

    keys = list(crop_emb.keys())
    in_dom, cross_same, cross_neg = [], [], []
    for c in keys:
        q, other = crop_emb[c]
        in_dom.append(cos(q, other))
        cross_same.append(cos(q, fd_emb[c]))
        d = random.choice([k for k in keys if k != c])
        cross_neg.append(cos(q, fd_emb[d]))

    def stat(xs):
        a = np.array(xs); return float(a.mean()), float(a.std())

    print("\n" + "=" * 64)
    print(" 1) Độ giống trong encoder (cosine, càng cao càng giống)")
    print("=" * 64)
    for name, xs in [("cùng chữ:  crop ↔ crop   (in-domain)", in_dom),
                     ("cùng chữ:  crop ↔ FD-glyph (cross-domain)", cross_same),
                     ("KHÁC chữ:  crop ↔ FD-glyph (negative)", cross_neg)]:
        m, s = stat(xs); print(f"  {name:42s} {m:+.3f} ± {s:.3f}")
    sep = (stat(cross_same)[0] - stat(cross_neg)[0])
    print(f"  -> tách biệt cùng-chữ vs khác-chữ (cross-domain): {sep:+.3f}  "
          f"({'TỐT, bắc cầu được' if sep > 0.15 else 'YẾU'})")

    # 2) retrieval@1: query crop, references = FD glyphs (all sampled chars) vs crop protos
    print("\n" + "=" * 64)
    print(" 2) retrieval@1: tìm đúng chữ cho 1 crop (trong", len(keys), "chữ)")
    print("=" * 64)
    fd_mat = np.stack([fd_emb[c] for c in keys])           # FD references
    proto_mat = np.stack([crop_emb[c][1] for c in keys])   # real-crop references (the "other" crop)
    hit_fd = hit_proto = 0
    for i, c in enumerate(keys):
        q = crop_emb[c][0]
        if keys[int((fd_mat @ q).argmax())] == c:
            hit_fd += 1
        if keys[int((proto_mat @ q).argmax())] == c:
            hit_proto += 1
    print(f"  CHỈ glyph FD làm mẫu   : {hit_fd/len(keys):.1%}   (kịch bản chữ 0-crop)")
    print(f"  crop THẬT làm mẫu      : {hit_proto/len(keys):.1%}   (chữ đã có crop)")
    print(f"  -> chênh lệch = giá của domain gap: {(hit_proto-hit_fd)/len(keys):+.1%}")

    # 3) dump examples for visual
    ex = []
    for c in keys[:6]:
        ex.append({"char": c, "unicode": f"U+{ord(c):X}",
                   "crop": str((D / by_char[c][0])), "fd": str(fd_path(c)),
                   "cos_crop_fd": round(cos(crop_emb[c][0], fd_emb[c]), 3)})
    print("\n  Ví dụ (đường dẫn để xem tận mắt):")
    for e in ex:
        print(f"   {e['char']} {e['unicode']} cos(crop,FD)={e['cos_crop_fd']:+.2f}")

    out = {"n_chars": len(keys),
           "cos_in_domain": round(stat(in_dom)[0], 4),
           "cos_cross_same": round(stat(cross_same)[0], 4),
           "cos_cross_neg": round(stat(cross_neg)[0], 4),
           "separation": round(sep, 4),
           "retr_fd_only": round(hit_fd / len(keys), 4),
           "retr_crop_proto": round(hit_proto / len(keys), 4),
           "examples": ex}
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / "eval_fd_domain_gap.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  -> results/eval_fd_domain_gap.json")


if __name__ == "__main__":
    main()
