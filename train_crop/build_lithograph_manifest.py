"""Dựng tập NHÃN YẾU tự động cho detector v2 (thạch bản + STT) → train_crop/data_lithograph/.

Không có người vẽ hộp. Nhãn hộp sinh tự động từ hai nguồn, cùng định dạng manifest của
train_crop (data_centernet.CenterNetDataset.from_manifest: [{image, boxes, ...}]):

  • Thạch bản (LVT1883 105 trang, KVK1884 163 trang, ảnh prepared/<book>/pages/*.png = ảnh
    pipeline thấy): Ô THAM CHIẾU = hộp TẦNG của kim (1 hộp / tầng-cột) cắt tại N−1 khe mực
    yếu nhất (pipeline.align_engine.char_detector.pitch_decode.ink_cut_cells, N = số âm QN của
    câu, transcriptions/<page>.json). Chỉ lấy tầng "verified" (số chữ kim == N). Tầng không
    verified và tầng có ô xấu (blank ink < 5 %, mực chạm mép > 20 % = cắt vào thân chữ,
    h/p ∉ [0,6; 1,5]) → `ignore_boxes`: trainer TÔ TRẮNG vùng đó trước khi học (không có
    glyph không nhãn dạy detector bỏ sót). Số câu in / rác lề không có nhãn → detector học
    KHÔNG bắt (đúng ý: đây là nguồn +1 của I5).
  • STT (445 trang, train_crop/detect_manifest.json = hộp kim/pipeline tier GOLD/SILVER/
    SYLLABLE — cách train_crop v1 đang dùng): giữ nguyên hộp; thêm `ignore_boxes` = bbox tier
    REVIEW từ dataset_out/labels.csv (v1 để glyph REVIEW không nhãn → trần F1 0,84).

Chia tập PAGE-DISJOINT THEO SÁCH: trong mỗi sách, trang thứ 10k+3 → val, 10k+7 → test, còn lại
train (mọi sách có mặt ở 3 tập); --lobo <book> giữ nguyên sách đó làm test (đo tổng quát hoá
liên sách). `domain` ∈ {litho, stt} để trainer lấy mẫu cân bằng 50/50.

Chạy:
  .venv/bin/python train_crop/build_lithograph_manifest.py            # toàn bộ → train_crop/data_lithograph/
  .venv/bin/python train_crop/build_lithograph_manifest.py --limit 6  # thử
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "measure"))

LITHO_BOOKS = {"LucVanTien1883": "lucvantien_manuscript", "KimVanKieu1884": "canvas"}

def _stt_key(book: str) -> str:
    """Mã sách STT về một dạng: labels.csv ghi 'stt2', detect_manifest.json ghi 'yen2' (tên cũ),
    config ghi 'SachThanhTruyen2' → 'stt2'."""
    import re
    m = re.match(r"^(?:yen|stt|SachThanhTruyen)(\d+)$", str(book))
    return f"stt{m.group(1)}" if m else str(book)

OUT_DIR = HERE / "data_lithograph"
INK_MIN, BORDER_MAX, HP_LO, HP_HI = 0.05, 0.20, 0.6, 1.5


def _binarize(gray):
    import cv2
    _, b = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (b > 0).astype(np.uint8)


def _cell_ok(binimg, c, p):
    x1, y1, x2, y2 = [int(v) for v in c[:4]]
    sub = binimg[max(0, y1):y2, max(0, x1):x2]
    if sub.size == 0 or sub.shape[0] < 4:
        return False, "empty"
    ink = float(sub.mean())
    border = float(max(sub[:2].mean(), sub[-2:].mean()))
    hp = (y2 - y1) / p
    if ink < INK_MIN:
        return False, "blank"
    if border > BORDER_MAX:
        return False, "cut_glyph"
    if not (HP_LO <= hp <= HP_HI):
        return False, "h_over_p"
    return True, "ok"


def litho_items(book, limit=0):
    """Mỗi trang thạch bản → item {image, book, page, domain, boxes, ignore_boxes, stats}."""
    import cv2
    from box_ref_eval import ref_cells_for_page
    pdir = REPO / "prepared" / book
    pages = sorted(p.stem for p in (pdir / "transcriptions").glob("page_*.json") if "_qn_ocr" not in p.stem)
    if limit:
        pages = pages[:limit]
    pitch_by_page = {}
    lp = REPO / "measure_out" / book / "layout" / "layout_pages.csv"
    if lp.exists():
        for r in csv.DictReader(open(lp, encoding="utf-8")):
            uid = int(r["uid"])
            pg = uid if book == "LucVanTien1883" else 167 - uid
            if r.get("col_pitch"):
                pitch_by_page[f"page_{pg:04d}"] = float(r["col_pitch"])
    items = []
    for page in pages:
        img = pdir / "pages" / f"{page}.png"
        cache_p = pdir / "detected" / f"{page}_ocr_cache.json"
        if not img.exists() or not cache_p.exists():
            continue
        gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        cache = json.load(open(cache_p, encoding="utf-8"))
        trans = json.load(open(pdir / "transcriptions" / f"{page}.json", encoding="utf-8"))
        cols = ref_cells_for_page(gray, cache, trans, pitch_by_page.get(page))
        binimg = _binarize(gray)
        boxes, labels, ignore, tiers, st = [], [], [], [], Counter()
        for col in cols:
            for t in col["tiers"]:
                st["tiers"] += 1
                if not t["verified"]:
                    st["tiers_unverified"] += 1; ignore.append([int(v) for v in t["box"]]); continue
                oks = [_cell_ok(binimg, c, t["pitch"]) for c in t["cells"]]
                for _, why in oks:
                    st[f"cell_{why}"] += 1
                if all(ok for ok, _ in oks):
                    st["tiers_kept"] += 1
                    tiers.append([int(v) for v in t["box"]] + [t["n_ref"], len(boxes)])   # [x1,y0,x2,y1,N,start_idx]
                    boxes += [[int(v) for v in c] for c in t["cells"]]
                    labels += [""] * len(t["cells"])
                else:
                    st["tiers_bad_cell"] += 1; ignore.append([int(v) for v in t["box"]])
        if not boxes:
            continue
        items.append({"image": str(img), "book": book, "page": page, "domain": "litho",
                      "boxes": boxes, "labels": labels, "n_boxes": len(boxes),
                      "ignore_boxes": ignore, "tiers": tiers, "stats": dict(st)})
    return items


def stt_items(limit=0):
    """STT từ detect_manifest.json (hộp GOLD/SILVER/SYLLABLE) + ignore = bbox REVIEW của labels.csv."""
    man = json.load(open(HERE / "detect_manifest.json", encoding="utf-8"))
    review = defaultdict(list)
    lab = REPO / "dataset_out" / "labels.csv"
    if lab.exists():
        for r in csv.DictReader(open(lab, encoding="utf-8")):
            if r["tier"] == "REVIEW" and r["bbox"] and r["bbox"] not in ("null", "None", "[]"):
                try:
                    bb = [int(float(v)) for v in ast.literal_eval(r["bbox"])]
                except Exception:
                    continue
                review[(_stt_key(r["book"]), r["page"])].append(bb)
    items = []
    for it in man:
        items.append({"image": it["image"], "book": it["book"], "page": it["page"], "domain": "stt",
                      "boxes": it["boxes"], "labels": it.get("labels", [""] * len(it["boxes"])),
                      "n_boxes": len(it["boxes"]),
                      "ignore_boxes": review.get((_stt_key(it["book"]), it["page"]), []), "tiers": [],
                      "stats": {"n_review_ignore": len(review.get((_stt_key(it["book"]), it["page"]), []))}})
    items.sort(key=lambda x: (x["book"], x["page"]))
    return items[:limit] if limit else items


def assign_split(items, lobo=""):
    by_book = defaultdict(list)
    for it in items:
        by_book[it["book"]].append(it)
    for book, its in by_book.items():
        its.sort(key=lambda x: x["page"])
        for k, it in enumerate(its):
            if lobo and book == lobo:
                it["split"] = "test"
            elif k % 10 == 3:
                it["split"] = "val"
            elif k % 10 == 7:
                it["split"] = "test"
            else:
                it["split"] = "train"
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=0, help="số trang mỗi sách (thử)")
    ap.add_argument("--no-stt", action="store_true")
    ap.add_argument("--lobo", default="", help="giữ nguyên 1 sách làm test (vd KimVanKieu1884)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = []
    for b in LITHO_BOOKS:
        its = litho_items(b, a.limit)
        print(f"[litho] {b}: {len(its)} trang, {sum(i['n_boxes'] for i in its)} ô nhãn yếu, "
              f"{sum(len(i['ignore_boxes']) for i in its)} vùng ignore", flush=True)
        items += its
    if not a.no_stt:
        its = stt_items(a.limit * 3 if a.limit else 0)
        print(f"[stt] {len(its)} trang, {sum(i['n_boxes'] for i in its)} hộp, "
              f"{sum(len(i['ignore_boxes']) for i in its)} hộp REVIEW ignore", flush=True)
        items += its
    assign_split(items, a.lobo)
    json.dump(items, open(out / "manifest_all.json", "w", encoding="utf-8"), ensure_ascii=False)
    stats = {"n_items": len(items), "by_domain_split": {}, "litho_cell_stats": Counter(), "lobo": a.lobo}
    for sp in ("train", "val", "test"):
        sub = [i for i in items if i["split"] == sp]
        json.dump(sub, open(out / f"manifest_{sp}.json", "w", encoding="utf-8"), ensure_ascii=False)
        for dom in ("litho", "stt"):
            d = [i for i in sub if i["domain"] == dom]
            stats["by_domain_split"][f"{dom}/{sp}"] = {"pages": len(d), "boxes": sum(i["n_boxes"] for i in d),
                                                       "ignore": sum(len(i["ignore_boxes"]) for i in d)}
    for i in items:
        if i["domain"] == "litho":
            stats["litho_cell_stats"].update(i["stats"])
    stats["litho_cell_stats"] = dict(stats["litho_cell_stats"])
    json.dump(stats, open(out / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats["by_domain_split"], ensure_ascii=False))
    print(json.dumps(stats["litho_cell_stats"], ensure_ascii=False))
    print(f"-> {out}/manifest_{{all,train,val,test}}.json + stats.json")


if __name__ == "__main__":
    main()
