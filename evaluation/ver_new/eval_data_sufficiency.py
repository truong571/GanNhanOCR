"""Dữ liệu hiện tại đã ĐỦ để train cho kết quả tốt nhất chưa?
Quyết định bởi PHÂN BỐ SỐ MẪU/LỚP (1591 lớp), nhất là đuôi dài. Đối chiếu với
retrieval đã đo (đuôi hiếm 46% vs phổ biến 91%).

Run: .venv/bin/python evaluation/ver_new/eval_data_sufficiency.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "dataset_out"


def main():
    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))
    gold = Counter(r["label"] for r in rows if r["tier"] == "GOLD" and r["label"])
    usable = Counter(r["label"] for r in rows if r["tier"] in ("GOLD", "SILVER") and r["label"])
    classes = set(r["label"] for r in rows if r["tier"] in ("GOLD", "SILVER", "SYLLABLE", "REVIEW") and r["label"])
    test = Counter(r["label"] for r in rows if r["tier"] == "GOLD" and r["split"] == "test" and r["label"])
    ncls = len(usable)
    tot = sum(gold.values())

    def band_report(cnt, name):
        bands = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 49), (50, 10 ** 9)]
        print(f"\n  Phân bố số crop/lớp ({name}, {len(cnt)} lớp, {sum(cnt.values())} crop):")
        print(f"    {'dải':>10} | {'#lớp':>6} | {'%lớp':>6} | {'#crop':>7} | {'%crop':>6}")
        tc = sum(cnt.values())
        for lo, hi in bands:
            cl = [c for c, n in cnt.items() if lo <= n <= hi]
            crops = sum(cnt[c] for c in cl)
            lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+")
            print(f"    {lab:>10} | {len(cl):>6} | {len(cl)/max(len(cnt),1)*100:>5.1f}% | {crops:>7} | {crops/max(tc,1)*100:>5.1f}%")

    print("=" * 60)
    print(" DỮ LIỆU HIỆN TẠI — đủ train tốt nhất chưa?")
    print("=" * 60)
    print(f"  Tổng lớp chữ (xuất hiện): {len(classes)}  | có nhãn GOLD+SILVER: {ncls}")
    print(f"  Tổng GOLD crop: {tot}  | trung bình {tot/max(ncls,1):.1f}/lớp  | "
          f"trung vị {sorted(gold.values())[len(gold)//2] if gold else 0}/lớp")
    # Zipf
    top = sorted(gold.values(), reverse=True)
    top10 = sum(top[: max(1, len(top)//10)])
    print(f"  10% lớp đông nhất giữ {top10/max(tot,1)*100:.0f}% số crop (lệch Zipf)")

    band_report(gold, "GOLD")

    starve5 = [c for c in classes if gold.get(c, 0) < 5]
    starve10 = [c for c in classes if gold.get(c, 0) < 10]
    zero = [c for c in classes if gold.get(c, 0) == 0]
    print(f"\n  ĐUÔI ĐÓI DỮ LIỆU:")
    print(f"    lớp có 0 GOLD crop : {len(zero)}  ({len(zero)/max(len(classes),1)*100:.0f}% lớp) — chỉ học được qua glyph sinh")
    print(f"    lớp <5 GOLD crop   : {len(starve5)}  ({len(starve5)/max(len(classes),1)*100:.0f}% lớp) — few-shot, khó")
    print(f"    lớp <10 GOLD crop  : {len(starve10)}  ({len(starve10)/max(len(classes),1)*100:.0f}% lớp)")

    # test adequacy
    test_cls = len([c for c in classes if test.get(c, 0) >= 1])
    print(f"\n  ĐÁNH GIÁ (test split): {sum(test.values())} crop test | {test_cls}/{len(classes)} lớp có ≥1 mẫu test")
    print(f"    -> {len(classes)-test_cls} lớp KHÔNG có mẫu test (không đo được/lớp) — chủ yếu là đuôi hiếm")

    print(f"\n  ĐỐI CHIẾU retrieval đã đo: đuôi hiếm(<5) 46% vs phổ biến 91% (chênh 45đ)")
    print("  => Phổ biến: ĐỦ dữ liệu (đã ~90%). Đuôi hiếm: THIẾU (data-starved) = trần đang bị kéo xuống bởi đuôi.")

    out = {"classes": len(classes), "usable_classes": ncls, "gold_crops": tot,
           "avg_per_class": round(tot/max(ncls,1), 1),
           "zero_crop_classes": len(zero), "lt5_classes": len(starve5), "lt10_classes": len(starve10),
           "test_classes_covered": test_cls, "test_crops": sum(test.values())}
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / "eval_data_sufficiency.json", "w"), indent=2)
    print(f"\n  -> results/eval_data_sufficiency.json")


if __name__ == "__main__":
    main()
