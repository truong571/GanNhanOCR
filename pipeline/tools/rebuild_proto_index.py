"""Sinh lại `pipeline/align_engine/data/index.csv` — chỉ mục crop-proto của S3.

VÌ SAO
------
`index.csv` liệt kê các crop GOLD split=train dùng làm NGUYÊN MẪU THỊ GIÁC cho S3
(`visual_signal.py:187`). Nó được tạo MỘT LẦN rồi để đó, không có bộ sinh — nên sau vài
thế hệ dữ liệu nó trỏ vào một bộ nhãn đã bị thay thế.

Đo 2026-08-24: **51.195/51.195 dòng trỏ sách tên cũ `yen2/yen4/yen11`**, trong khi bộ nhãn
hiện hành toàn `stt2/stt4/stt11` — giao nhau theo đường dẫn = **0**. Các tệp `yen*` vẫn còn
trên đĩa nên preflight cũ báo XANH ("tệp tồn tại"), nhưng nguyên mẫu đến từ một thế hệ nhãn
đã bỏ. Preflight nay kiểm cả THẾ HỆ (KHỐI 1.7), và đây là bộ sinh để vá.

PHẠM VI ẢNH HƯỞNG — nhỏ và nằm NGOÀI bộ giao nộp
------------------------------------------------
S3 chỉ quyết các tier SILVER (`s2_inter_s3_corrected`, `s1_inter_s3_out_of_dict`,
`s3_head_bank_consensus`). GOLD = S1∩S2, `consensus.py:92-93` trả về TRƯỚC mọi lần đọc S3,
và `--s3-demote` đã tắt mặc định. Nên sinh lại chỉ mục **không đụng bộ giao nộp**; nó chỉ
đổi thành phần SILVER_uncalibrated, vốn đã nằm ngoài `USABLE_TIERS`.

Hiệu lực có ở LẦN BUILD KẾ TIẾP (nguyên mẫu được cache theo mtime của index.csv).

    python -m pipeline.tools.rebuild_proto_index
    python -m pipeline.tools.rebuild_proto_index --check   # exit 1 nếu lệch thế hệ
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "dataset_out" / "labels_final.csv"
INDEX = REPO / "pipeline" / "align_engine" / "data" / "index.csv"
FIELDS = ["path", "label", "unicode", "split", "source"]


def current_books(labels: Path = LABELS) -> set[str]:
    with open(labels, encoding="utf-8") as fh:
        return {r["book"] for r in csv.DictReader(fh) if r.get("book")}


def generation_of(index: Path = INDEX) -> set[str]:
    """Các tiền tố sách mà chỉ mục đang trỏ tới."""
    out: set[str] = set()
    if not index.exists():
        return out
    with open(index, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = re.search(r"/([a-z]+\d+)_", r.get("path", ""))
            if m:
                out.add(m.group(1))
    return out


def split_lech(labels: Path = LABELS, index: Path = INDEX) -> tuple[int, int]:
    """(số nguyên mẫu gắn train mà NAY thuộc val/test, tổng nguyên mẫu train).

    Phép kiểm thế hệ cũ chỉ so TIỀN TỐ SÁCH, nên nó mù với việc đổi ĐỊNH NGHĨA split.
    Đúng chuyện đã xảy ra 2026-08-25: chia tách chuyển từ mức CỘT sang mức TRANG, index.csv
    không ai sinh lại, và 20,6% nguyên mẫu gắn `split=train` hoá ra nằm trên trang val/test
    — trong khi `--check` vẫn báo "cùng thế hệ". Bảo đảm "nguyên mẫu rời khỏi val/test" của
    `build_rows` bị phá mà không một tín hiệu nào.
    """
    if not index.exists() or not labels.exists():
        return 0, 0
    now: dict[tuple[str, str], str] = {}
    with open(labels, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            now.setdefault((r.get("book", ""), r.get("page", "")), r.get("split", ""))
    # `_p####` là hậu tố tách trang trùng số (step1_extract) — phải nuốt cả nó, nếu không
    # `page_0010_p0028` bị cắt thành `page_0010` và quy sai về trang khác.
    pat = re.compile(r"/([a-z]+\d+)_(page_\d+(?:_p\d+)?)")
    lech = tong = 0
    with open(index, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != "train":
                continue
            tong += 1
            m = pat.search(r.get("path", ""))
            if m and now.get((m.group(1), m.group(2))) in ("val", "test"):
                lech += 1
    return lech, tong


def build_rows(labels: Path = LABELS, src_root: Path = REPO / "dataset_out") -> list[dict]:
    """Crop GOLD split=train có ảnh THẬT trên đĩa.

    Chỉ lấy `split == 'train'`: nguyên mẫu phải rời khỏi val/test, nếu không phép đánh giá
    giữ-lại ở Bước 3 sẽ tự chấm điểm cho mình (chú thích ở visual_signal.py:180-184).
    """
    rows = []
    with open(labels, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("tier") != "GOLD" or r.get("split") != "train":
                continue
            img, lab = r.get("image"), r.get("label")
            if not img or not lab or not (src_root / img).exists():
                continue
            rows.append({"path": f"dataset_out/{img}", "label": lab,
                         "unicode": r.get("unicode", ""), "split": "train",
                         "source": "crop"})
    rows.sort(key=lambda x: x["path"])          # tất định
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.rebuild_proto_index")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", default=str(INDEX))
    ap.add_argument("--check", action="store_true",
                    help="chỉ kiểm thế hệ, KHÔNG ghi; exit 1 nếu lệch")
    args = ap.parse_args(argv)

    want = current_books(Path(args.labels))
    have = generation_of(Path(args.out))
    if args.check:
        if have and not (have & want):
            print(f"[proto-index] LỆCH THẾ HỆ: chỉ mục trỏ {sorted(have)}, "
                  f"bộ nhãn dùng {sorted(want)} — giao nhau = 0.\n"
                  f"    chạy: python -m pipeline.tools.rebuild_proto_index", file=sys.stderr)
            return 1
        lech, tong = split_lech(Path(args.labels), Path(args.out))
        if lech:
            print(f"[proto-index] LỆCH CHIA TÁCH: {lech:,}/{tong:,} nguyên mẫu "
                  f"({100*lech/tong:.1f}%) gắn split=train nhưng trang của chúng NAY là "
                  f"val/test — nguyên mẫu thị giác đã rò sang phần held-out.\n"
                  f"    chạy: python -m pipeline.tools.rebuild_proto_index", file=sys.stderr)
            return 1
        print(f"[proto-index] cùng thế hệ ({sorted(have) or 'chưa có chỉ mục'}) · "
              f"chia tách khớp ({tong:,} nguyên mẫu train, 0 rò)")
        return 0

    rows = build_rows(Path(args.labels))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[proto-index] {len(rows):,} crop GOLD/train -> {out.relative_to(REPO)}")
    print(f"[proto-index] thế hệ: {sorted(have) or '(trống)'} -> {sorted(current_books(Path(args.labels)))}")
    print("[proto-index] hiệu lực ở LẦN BUILD KẾ TIẾP (nguyên mẫu cache theo mtime index.csv).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
