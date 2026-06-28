"""Dựng manifest huấn luyện CenterNet TỪ labels.csv — chọn tier linh hoạt.

(Tái hiện đúng logic đã kiểm chứng của bootstrap_boxes.py, nhưng tự-chứa trong test/
và cho chọn tier qua tham số.)

Theo Data Registry: mỗi dòng labels.csv mang `bbox` (toạ độ TRANG đã sửa offset) +
`tier` (GOLD/SILVER/SYLLABLE/REVIEW) + book/page/column. Script gom theo trang ->
manifest [{image, book, page, boxes, labels, n_boxes}] mà trainer dùng trực tiếp.

  • Box LẤY từ các tier `--tiers` (mặc định GOLD,SILVER,SYLLABLE — đều là hộp ĐỊNH-VỊ-
    TỐT; REVIEW là cột diverged, vị trí sai -> luôn loại).
  • `--complete-only`: chỉ giữ box ở CỘT HOÀN CHỈNH (không còn dòng REVIEW). Sạch nhất
    nhưng RẤT ít box (tránh "completeness trap"); MẶC ĐỊNH TẮT để giữ nhiều box như
    detect_manifest.json gốc (~66k).

Đường dẫn ảnh TRANG lấy từ manifest "nguồn path" (mặc định detect_manifest.json đầy đủ).

Chạy:
  .venv/bin/python test/build_manifest.py --tiers GOLD,SILVER,SYLLABLE \
      --out test/manifest_gss.json
  .venv/bin/python test/build_manifest.py --tiers GOLD --complete-only \
      --out test/manifest_gold_clean.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEF_LABELS = REPO / "evaluation/ver_new/dataset_out/labels.csv"
DEF_PAGEMAP = REPO / "evaluation/ver_new/char_detector/detect_manifest.json"


def _page_path_map(pagemap_json):
    """(book,page) -> đường dẫn ảnh trang gốc, lấy từ manifest nguồn."""
    m = {}
    for it in json.load(open(pagemap_json, encoding="utf-8")):
        m[(it["book"], it["page"])] = it["image"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(DEF_LABELS))
    ap.add_argument("--pagemap", default=str(DEF_PAGEMAP),
                    help="manifest nguồn để ánh xạ (book,page)->ảnh trang")
    ap.add_argument("--tiers", default="GOLD,SILVER,SYLLABLE",
                    help="tier LẤY box (phân tách dấu phẩy)")
    ap.add_argument("--complete-only", action="store_true",
                    help="chỉ giữ box ở cột KHÔNG còn REVIEW (sạch nhất, ít box hơn nhiều)")
    ap.add_argument("--out", default=str(HERE / "manifest_custom.json"))
    a = ap.parse_args()

    keep = set(t.strip().upper() for t in a.tiers.split(",") if t.strip())
    pmap = _page_path_map(a.pagemap)
    rows = list(csv.DictReader(open(a.labels, encoding="utf-8")))

    # hoàn chỉnh = cột KHÔNG có dòng REVIEW (độc lập với `keep`)
    col_tiers = defaultdict(set)
    for r in rows:
        col_tiers[(r["book"], r["page"], r["column"])].add(r["tier"])

    def complete(r):
        return "REVIEW" not in col_tiers[(r["book"], r["page"], r["column"])]

    by_page = defaultdict(lambda: {"boxes": [], "labels": []})
    kept = dropped = tiny = 0
    for r in rows:
        if r["tier"] not in keep:
            continue
        if not r["bbox"] or r["bbox"] in ("null", "None", "[]"):
            continue
        if a.complete_only and not complete(r):
            dropped += 1
            continue
        try:
            bb = [float(v) for v in ast.literal_eval(r["bbox"])]
        except Exception:
            continue
        if bb[2] - bb[0] < 4 or bb[3] - bb[1] < 4:
            tiny += 1
            continue
        by_page[(r["book"], r["page"])]["boxes"].append(bb)
        by_page[(r["book"], r["page"])]["labels"].append(r.get("unicode", ""))
        kept += 1

    man, miss = [], 0
    for (book, page), d in by_page.items():
        img = pmap.get((book, page))
        if img is None:
            miss += 1
            continue
        man.append({"image": img, "book": book, "page": page,
                    "boxes": d["boxes"], "labels": d["labels"], "n_boxes": len(d["boxes"])})
    man.sort(key=lambda it: (it["book"], it["page"]))
    json.dump(man, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)

    nb = sorted(it["n_boxes"] for it in man)
    print(f"[build_manifest] tiers={sorted(keep)} | complete_only={a.complete_only}")
    print(f"  box giữ {kept} | bỏ (cột chưa hoàn chỉnh) {dropped} | bỏ (hộp quá nhỏ) {tiny} "
          f"| thiếu path {miss}")
    print(f"  -> {a.out} : {len(man)} trang"
          + (f", box/trang min {nb[0]} med {nb[len(nb)//2]} max {nb[-1]}" if nb else ""))
    print("  Kế tiếp: pack_for_kaggle.py --manifest <file này>, hoặc trainer --manifest.")


if __name__ == "__main__":
    main()
