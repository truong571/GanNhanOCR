"""Sinh bảng ỨNG VIÊN MỞ RỘNG TỪ ĐIỂN — có ĐỦ cột để người duyệt phân xử.

VÌ SAO CÓ TỆP NÀY
-----------------
11.007 ô bị bỏ với lý do `no_s1_inter_s2`: S1 đọc được chữ, S2 có âm, S3 có điểm — nhưng
chữ không nằm trong danh sách âm đọc của từ điển. Trong đó 2.536 ô có cặp (chữ, âm) xuất
hiện ở **>=2 sách**, tức không phải nhiễu ngẫu nhiên.

HAI GIẢ THUYẾT CẠNH TRANH, VÀ CHÚNG PHẢI ĐƯỢC PHÂN XỬ CHỨ KHÔNG ĐƯỢC ĐOÁN:
  (A) TỪ ĐIỂN THIẾU — đây là chữ Nôm thật, quy ước thật của văn bản, mà
      QuocNgu_SinoNom.csv chưa thu. Ví dụ ứng viên: 䍐/ra, 乇/đã.
  (B) OCR NHẦM TỰ DẠNG — S1 đọc nhầm sang một chữ nhìn giống. Cùng một mô hình OCR,
      cùng nét khắc, cùng thể loại thì **cùng một cái nhầm sẽ tái xuất ở cả ba cuốn**,
      nên riêng tiêu chí ">=2 sách" KHÔNG phân biệt được (A) với (B).

Bản đầu của bảng này (2026-08-25) đã BỎ MẤT đúng hai cột dùng để phân xử, khiến người
duyệt không có căn cứ. Đây là chỗ vá.

BA CỘT PHÂN XỬ
--------------
  similar_hit   chữ trong từ điển ĐỌC ĐÚNG âm đó VÀ nhìn giống chữ OCR.
                Có giá trị -> nghiêng mạnh về (B). Rỗng -> (B) khó giải thích.
  char_in_dict  chữ OCR có mặt trong từ điển ở BẤT KỲ âm nào khác không, và là âm gì.
                Nếu chữ đó vốn đọc âm khác hẳn -> nghiêng về (B) hoặc mượn âm hiếm.
  gold_count    chữ OCR đó đã được dùng làm nhãn GOLD bao nhiêu lần ở âm KHÁC.
                Cao -> chữ này pipeline vẫn đọc tốt ở chỗ khác, nên ở đây khó là OCR hỏng.

    python -m pipeline.tools.dict_candidates
    python -m pipeline.tools.dict_candidates --min-books 2 --min-count 3
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def build(labels: Path, min_books: int, min_count: int) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import dict_dir, load_qn_to_nom, load_similarity_dict
    from pipeline.tools.sem_score import SemScorer, load_meanings, TAU_CONFIRM

    qn = load_qn_to_nom(str(dict_dir() / "QuocNgu_SinoNom.csv"))
    sim = load_similarity_dict(str(dict_dir() / "SinoNom_Similar.csv"))
    sc = SemScorer(load_meanings(), qn)

    rows = list(csv.DictReader(open(labels, encoding="utf-8")))
    # chữ OCR nào đã từng làm nhãn GOLD, ở âm nào
    gold = collections.Counter()
    for r in rows:
        if r["tier"] == "GOLD" and (r.get("label") or "").strip():
            gold[r["label"]] += 1
    # từ điển ngược: chữ -> các âm nó đọc
    nom2qn = collections.defaultdict(list)
    for syl, chars in qn.items():
        for c in chars:
            nom2qn[c].append(syl)

    pool = collections.defaultdict(lambda: {"n": 0, "books": set(), "pages": set(), "img": []})
    for r in rows:
        if r["rule"] != "no_s1_inter_s2":
            continue
        oc, syl = r["ocr_char"], (r["syllable"] or "").lower()
        R = qn.get(syl)
        if not R or oc in R:
            continue
        if [s for s in dict.fromkeys(sim.get(oc, ())) if s in R]:
            continue                       # có cầu nối -> luật bắc cầu đã lo
        d = pool[(oc, syl)]
        d["n"] += 1
        d["books"].add(r["book"])
        d["pages"].add(r["page"])
        # Ô REVIEW KHÔNG được cắt crop (crop_tiers loại REVIEW), nên cột ảnh mẫu luôn
        # rỗng. Người duyệt cần TOẠ ĐỘ để mở đúng chỗ trên ảnh trang gốc.
        if len(d["img"]) < 3:
            d["img"].append(f"{r['book']}/{r['page']} c{r['column']} {r.get('bbox','')}")

    out = []
    for (oc, syl), v in pool.items():
        if len(v["books"]) < min_books or v["n"] < min_count:
            continue
        R = set(qn[syl])
        # (B): chữ trong từ điển đọc đúng âm này VÀ nhìn giống chữ OCR
        hit = [c for c in R if oc in sim.get(c, ()) or c in sim.get(oc, ())]
        other = nom2qn.get(oc, [])
        s = sc.score(oc, syl)
        out.append({
            "ocr_char": oc, "syllable": syl,
            "so_o": v["n"], "so_sach": len(v["books"]), "so_trang": len(v["pages"]),
            "similar_hit": "".join(hit[:4]),
            "char_in_dict": "".join(other[:5]),
            "gold_count": gold.get(oc, 0),
            "diem_unihan": f"{s:.4f}" if s is not None else "",
            "nghieng_ve": ("B_OCR_NHAM" if hit else
                           ("B_nghi" if (other and syl not in other and gold.get(oc, 0) >= 20)
                            else "A_TU_DIEN_THIEU")),
            # vị trí để MỞ ẢNH TRANG GỐC prepared/<sách>/pages/<trang>.png rồi soi bbox
            "vi_tri_1": (v["img"] + ["", ""])[0],
            "vi_tri_2": (v["img"] + ["", ""])[1],
            "NGUOI_DUYET": "", "GHI_CHU": "",
        })
    out.sort(key=lambda r: (-r["so_o"], r["ocr_char"]))
    stat = collections.Counter(r["nghieng_ve"] for r in out)
    return out, {"cap": len(out), "o": sum(r["so_o"] for r in out), "nghieng": dict(stat)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.dict_candidates")
    ap.add_argument("--labels", default=str(REPO / "dataset_out" / "labels_final.csv"))
    ap.add_argument("--out", default=str(REPO / "docs" / "UNG_VIEN_MO_RONG_TU_DIEN.csv"))
    ap.add_argument("--min-books", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args(argv)

    rows, st = build(Path(args.labels), args.min_books, args.min_count)
    if not rows:
        print("[ứng viên] không có cặp nào qua ngưỡng", file=sys.stderr)
        return 1
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ứng viên] {st['cap']:,} cặp phủ {st['o']:,} ô -> {args.out}")
    for k, v in sorted(st["nghieng"].items()):
        print(f"    {k:20} {v:5,} cặp")
    print("    A_TU_DIEN_THIEU = không giải thích được bằng nhầm tự dạng -> ứng viên THẬT")
    print("    B_OCR_NHAM      = có chữ nhìn giống trong từ điển đọc đúng âm -> nghi OCR sai")
    print("    B_nghi          = chữ này đọc âm khác trong từ điển VÀ đã làm nhãn GOLD >=20 lần")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
