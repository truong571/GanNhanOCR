"""Xuất BỘ CHỮ TỪ ĐIỂN KHÔNG CÓ — mỗi dòng một cặp (chữ Nôm, âm) mà
`Dict/QuocNgu_SinoNom.csv` không chứa.

Đây chính là khoảng trống sinh ra tier SYLLABLE: kinhhannom đọc ra chữ, nhưng từ điển ÂM
không có cặp đó vì phần lớn là mượn NGHĨA (皇 đọc "thánh", 行 đọc "đức"). Đã đo: Unihan
17.0 kVietnamese xác nhận 0/316 cặp SYLLABLE, nên không nguồn công khai nào lấp được —
phải chấm tay từ chính corpus.

Mỗi dòng mang sẵn mọi thứ cần để phán đoán mà không phải mở lại pipeline:

  n_rows/n_pages/n_books  cặp này xuất hiện bao nhiêu, rải ra mấy trang mấy sách
                          (rải rộng = quy ước thật; dồn một trang = nghi lỗi cục bộ)
  char_in_dict            chữ CÓ trong từ điển nhưng đọc âm khác  -> mượn nghĩa
                          chữ KHÔNG có ở đâu cả                   -> nghi chữ hiếm / OCR sai
  dict_readings_of_char   các âm mà từ điển gán cho chính chữ này
  dict_chars_for_syllable các chữ mà từ điển gán cho âm này — cột QUAN TRỌNG NHẤT: nếu
                          một chữ trong đây gần giống chữ đang xét thì đây là OCR NHẦM
                          TỰ DẠNG chứ không phải mượn nghĩa (vd 而 vs 爫 cùng đọc "làm")
  similar_hit             chữ nào trong cột trên nối được qua SinoNom_Similar
  gold_count              chữ này đã từng được xác nhận làm nhãn GOLD bao nhiêu lần
  sample_images           tối đa 5 ảnh crop để mở xem ngay

    .venv/bin/python -m pipeline.tools.dict_gap [--tiers SYLLABLE,REVIEW] [--out ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.dict_gap")
    ap.add_argument("--labels", default=str(REPO / "dataset_out" / "labels_final.csv"))
    ap.add_argument("--out", default=str(REPO / "dataset_out" / "dict_gap.csv"))
    ap.add_argument("--tiers", default="SYLLABLE,REVIEW",
                    help="tier nào được tính (GOLD/SILVER đã được từ điển xác nhận)")
    ap.add_argument("--min-rows", type=int, default=1)
    args = ap.parse_args(argv)

    from core.text.dictionary import load_qn_to_nom, load_similarity_dict
    qn = load_qn_to_nom(str(REPO / "Dict" / "QuocNgu_SinoNom.csv"))
    sim = load_similarity_dict(str(REPO / "Dict" / "SinoNom_Similar.csv"))
    readings_of = {}
    for syl, chars in qn.items():
        for ch in chars:
            readings_of.setdefault(ch, set()).add(syl)

    d = pd.read_csv(args.labels, dtype=str, low_memory=False)
    gold_count = d[d["tier"] == "GOLD"]["label"].value_counts()
    tiers = {t.strip().upper() for t in args.tiers.split(",") if t.strip()}
    sub = d[d["tier"].isin(tiers)].copy()
    sub = sub[sub["ocr_char"].notna() & sub["syllable"].notna()]
    sub["syllable"] = sub["syllable"].str.lower()
    # chỉ giữ cặp từ điển KHÔNG có
    sub = sub[[ch not in qn.get(sy, []) for ch, sy in zip(sub["ocr_char"], sub["syllable"])]]

    rows = []
    for (ch, syl), g in sub.groupby(["ocr_char", "syllable"]):
        if len(g) < args.min_rows:
            continue
        cands = qn.get(syl, [])
        near = [c for c in cands if ch in sim.get(c, []) or c in sim.get(ch, [])]
        rows.append({
            "ocr_char": ch, "syllable": syl,
            "n_rows": len(g),
            "n_pages": g.groupby(["book", "page"]).ngroups,
            "n_books": g["book"].nunique(),
            "tiers": ",".join(sorted(g["tier"].unique())),
            "char_in_dict": bool(ch in readings_of),
            "dict_readings_of_char": " ".join(sorted(readings_of.get(ch, ()))[:12]),
            "n_dict_chars_for_syllable": len(cands),
            "dict_chars_for_syllable": " ".join(cands[:20]),
            "similar_hit": " ".join(near[:6]),
            "gold_count": int(gold_count.get(ch, 0)),
            "verdict": "",          # cột để người chấm điền: nghia | ocr_sai | khac
            "true_char": "",        # nếu ocr_sai: chữ đúng là gì
            "sample_images": " ".join(g["image"].dropna().astype(str).head(5).tolist()),
        })

    out = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[gap] {len(out):,} cặp (chữ, âm) từ điển không có = "
          f"{int(out['n_rows'].sum()):,} ô ảnh")
    print(f"[gap]   chữ hoàn toàn không có trong từ điển: "
          f"{int((~out['char_in_dict']).sum()):,} cặp / "
          f"{int(out.loc[~out['char_in_dict'], 'n_rows'].sum()):,} ô")
    print(f"[gap]   nối được qua từ điển tự dạng gần giống (NGHI OCR SAI): "
          f"{int((out['similar_hit'] != '').sum()):,} cặp / "
          f"{int(out.loc[out['similar_hit'] != '', 'n_rows'].sum()):,} ô")
    print(f"[gap] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
