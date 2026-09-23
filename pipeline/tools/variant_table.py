"""Sinh bảng ỨNG VIÊN CHUẨN HOÁ DỊ THỂ — máy đề xuất, NGƯỜI quyết.

VÌ SAO — VÀ VÌ SAO PHẢI CẨN THẬN
--------------------------------
Đo được: 500/1.321 âm Quốc ngữ trong bộ nhãn ứng với >1 mã chữ, và 5.377/50.156 ô
(10,7%) mang mã KHÔNG phải mã trội của âm đó.

⚠️ CON SỐ ĐÓ **KHÔNG PHẢI LỖI**. Trong chữ Nôm, một âm mượn NHIỀU chữ khác nhau là
BẢN CHẤT của hệ chữ, không phải mâu thuẫn. `thì` khắc bằng 時 hay 寺 là hai lựa chọn
tự dạng THẬT của người khắc, phải giữ nguyên.

Chỉ một tập con cần chuẩn hoá: **dị thể của CÙNG MỘT chữ** — 事/亊, 時/时, 徳/德, 爲/為.
Gộp nhầm hai chữ Nôm khác nhau lại là PHÁ chứng cứ tự dạng, tệ hơn là để nguyên.

VÌ SAO KHÔNG TỰ ĐỘNG HOÁ ĐƯỢC
-----------------------------
Đã thử NFKC (phân rã tương thích Unicode): bắt được **0/xxx cặp**. Các cặp giản/phồn
không trùng NFKC. Nên KHÔNG có phép kiểm máy nào đủ tin để tự gộp.

CÁCH LÀM Ở ĐÂY: máy lọc ra ứng viên bằng HAI điều kiện cùng lúc —
  (1) hai chữ cùng được dùng làm nhãn cho CÙNG MỘT âm, và
  (2) hai chữ nằm trong danh sách NHÌN GIỐNG của nhau (SinoNom_Similar)
rồi ghi tần suất để người biết chữ Nôm phán. Công cụ này **KHÔNG sửa bộ nhãn**.

    python -m pipeline.tools.variant_table
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def build(labels: Path) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import dict_dir, load_similarity_dict
    sim = load_similarity_dict(str(dict_dir() / "SinoNom_Similar.csv"))

    rows = [r for r in csv.DictReader(open(labels, encoding="utf-8"))
            if (r.get("label") or "").strip()]
    by = collections.defaultdict(collections.Counter)
    where = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        syl = (r["syllable"] or "").lower()
        by[syl][r["label"]] += 1
        where[syl][r["label"]].add(r["book"])

    out = []
    for syl, cnt in by.items():
        if len(cnt) < 2:
            continue
        ks = [k for k, _ in cnt.most_common()]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = ks[i], ks[j]
                mutual = (b in sim.get(a, ())) and (a in sim.get(b, ()))
                oneway = (b in sim.get(a, ())) or (a in sim.get(b, ()))
                if not oneway:
                    continue                    # không nhìn giống -> hai chữ KHÁC nhau, giữ nguyên
                out.append({
                    "syllable": syl,
                    "ma_troi": a, "so_o_troi": cnt[a], "sach_troi": len(where[syl][a]),
                    "ma_phu": b, "so_o_phu": cnt[b], "sach_phu": len(where[syl][b]),
                    "giong_hai_chieu": "CO" if mutual else "mot_chieu",
                    "ty_le": f"{cnt[b] / max(cnt[a], 1):.3f}",
                    # gợi ý: lệch tần suất RẤT lớn + giống hai chiều -> nhiều khả năng dị thể
                    "may_nghi": ("DI_THE" if mutual and cnt[b] / max(cnt[a], 1) < 0.25
                                 else "CAN_NGUOI_XEM"),
                    "NGUOI_DUYET": "", "GHI_CHU": "",
                })
    out.sort(key=lambda r: -r["so_o_phu"])
    st = collections.Counter(r["may_nghi"] for r in out)
    return out, {"cap": len(out), "o_phu": sum(r["so_o_phu"] for r in out), "nghi": dict(st)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.variant_table")
    ap.add_argument("--labels", default=str(REPO / "dataset" / "SachThanhTruyen" / "labels.csv"))
    ap.add_argument("--out", default=str(REPO / "docs" / "UNG_VIEN_CHUAN_HOA_DI_THE.csv"))
    args = ap.parse_args(argv)
    rows, st = build(Path(args.labels))
    if not rows:
        print("[dị thể] không có cặp nào vừa cùng âm vừa nhìn giống"); return 0
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[dị thể] {st['cap']:,} cặp ứng viên, phủ {st['o_phu']:,} ô mã phụ -> {args.out}")
    for k, v in sorted(st["nghi"].items()):
        print(f"    {k:16} {v:4,} cặp")
    print("    CÔNG CỤ NÀY KHÔNG SỬA BỘ NHÃN — chỉ đề xuất để người biết chữ Nôm phán.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
