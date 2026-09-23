"""Ghi 3 cột chất lượng crop vào bộ nhãn — để người chấm KHÔNG phải chấm chiều CROP bằng mắt.

VÌ SAO
------
Đặc tả T4.4 đặt hàng đúng việc này ("ghi `crop_quality_flag` + `stray_ink` + `border_ink`
vào labels.csv để mẻ chấm sau này KHÔNG phải chấm chiều CROP bằng mắt nữa") nhưng nó chưa
từng được làm. Nó quan trọng vì một lý do đã đo được:

  Mẻ chấm 2026-08-04 hỏng ở CHIỀU CROP chứ không ở chiều NHÃN — κ = 0,14 cho câu hỏi
  "ảnh có đúng chữ không", trong khi chiều nhãn 0/20 báo động giả. Người không nhất quán
  khi phải vừa đọc chữ vừa phán xét ảnh cắt.

Máy đo hình học thì nhất quán tuyệt đối. Nên tách bạch: MÁY chấm chiều crop, NGƯỜI chấm
chiều nhãn. Ba cột này là phần của máy.

VÌ SAO KHÔNG NHÉT VÀO save_crop
-------------------------------
Để nó chạy được trên bộ nhãn ĐÃ CÓ mà không phải cắt lại 57k ảnh. Cắt lại là đổi
`image_md5` và toàn bộ chuỗi băm; ba cột này không đáng giá đó. `run_pipeline.sh` gọi nó
ngay sau bước build nên kết quả luôn đồng bộ.

GIỚI HẠN PHẢI NÓI RÕ
--------------------
Cờ này đo HÌNH HỌC, không đo đúng/sai nhãn. `bleed` nghĩa là "có mực lạ trong khung",
KHÔNG nghĩa là "nhãn sai". Dùng nó để XẾP ƯU TIÊN và để CẢNH BÁO người chấm, tuyệt đối
không dùng làm phán quyết.

    python -m pipeline.tools.enrich_crop_quality --labels dataset_out/labels.csv
    python -m pipeline.tools.enrich_crop_quality --check    # exit 1 nếu thiếu cột
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COLS = ("crop_quality_flag", "stray_ink", "border_ink")


def enrich(labels: Path, src_root: Path, verbose: bool = True, bin_root: Path | None = None) -> dict:
    """`bin_root` (2026-09-23) = thư mục chứa BẢN ĐÃ XỬ LÝ của cùng những crop đó
    (`$out/crops_bin/<tier>/<cùng tên>.png`, do build_dataset sinh khi sách khai
    `books[].crop_source: original`). Ba cột này đo HÌNH HỌC MỰC bằng ngưỡng cố định
    (`gray < 128`), nên phải đo trên bản đã xử lý — nếu đo trên crop gốc (nền giấy
    còn nguyên) thì `stray_ink`/`border_ink`/`crop_quality_flag` đổi hàng loạt chỉ vì
    đổi nguồn điểm ảnh, kéo theo cổng cơ chế B4' đọc `crop_quality_flag`. Vắng tham
    số: tự lấy `src_root/crops_bin` nếu có; không có thì đo thẳng crop giao nộp
    (đúng hành vi cũ, 0 thay đổi cho STT và cho mọi bộ `crop_source: processed`).
    """
    import cv2
    from pipeline.align_engine import crop_quality as CQ

    if bin_root is None:
        cand = src_root / "crops_bin"
        bin_root = cand if cand.is_dir() else None
    n_bin = 0

    with open(labels, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys()) if rows else []
    for c in COLS:
        if c not in fields:
            fields.append(c)

    stat: dict[str, int] = {"ok": 0, "bleed": 0, "truncated": 0, "blank": 0,
                            "không có ảnh": 0, "đọc lỗi": 0}
    for r in rows:
        img = (r.get("image") or "").strip()
        if not img:
            for c in COLS:
                r[c] = ""
            stat["không có ảnh"] += 1
            continue
        path = src_root / img
        if bin_root is not None and (bin_root / img).exists():
            path = bin_root / img
            n_bin += 1
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            for c in COLS:
                r[c] = ""
            stat["đọc lỗi"] += 1
            continue
        m = CQ.measure(g)
        r["crop_quality_flag"] = m["crop_quality_flag"]
        r["stray_ink"] = m["stray_ink"]
        r["border_ink"] = m["border_ink"]
        stat[m["crop_quality_flag"]] = stat.get(m["crop_quality_flag"], 0) + 1

    tmp = labels.with_suffix(labels.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(labels)

    if verbose:
        n = sum(v for k, v in stat.items() if k in ("ok", "bleed", "truncated", "blank"))
        print(f"  [chất lượng crop] {n:,} ảnh đã đo"
              + (f" ({n_bin:,} đo trên crops_bin/ = bản đã xử lý; crop giao nộp là ảnh quét gốc)"
                 if n_bin else ""))
        for k in ("ok", "bleed", "truncated", "blank"):
            if stat.get(k):
                print(f"    {k:12} {stat[k]:7,}  {100 * stat[k] / max(n, 1):5.2f}%")
        if stat["đọc lỗi"]:
            print(f"    🔴 đọc lỗi  {stat['đọc lỗi']:,}")
    return stat


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.enrich_crop_quality")
    ap.add_argument("--labels", default=str(REPO / "dataset_out" / "labels.csv"))
    ap.add_argument("--src-root", default=str(REPO / "dataset_out"))
    ap.add_argument("--bin-root", default="",
                    help="thư mục bản ĐÃ XỬ LÝ của cùng bộ crop (mặc định: <src-root>/crops_bin nếu có). "
                         "Ba cột chất lượng luôn đo trên bản này khi có, để không đổi khi crop giao nộp "
                         "chuyển sang ảnh quét gốc (books[].crop_source: original).")
    ap.add_argument("--check", action="store_true",
                    help="chỉ kiểm đã có 3 cột chưa; exit 1 nếu thiếu")
    args = ap.parse_args(argv)

    p = Path(args.labels)
    if not p.exists():
        print(f"[chất lượng crop] không có {p}", file=sys.stderr)
        return 1
    with open(p, encoding="utf-8") as fh:
        head = next(csv.reader(fh), [])
    if args.check:
        miss = [c for c in COLS if c not in head]
        if miss:
            print(f"[chất lượng crop] THIẾU cột {miss} trong {p.name} — chạy:\n"
                  f"    python -m pipeline.tools.enrich_crop_quality --labels {p}",
                  file=sys.stderr)
            return 1
        print(f"[chất lượng crop] đủ 3 cột trong {p.name}")
        return 0

    enrich(p, Path(args.src_root), bin_root=(Path(args.bin_root) if args.bin_root else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
