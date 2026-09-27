"""Đóng dấu một thư mục `dataset/<Book>/` là **TẬP ĐÁNH GIÁ** — không được trộn vào tập huấn luyện.

Vì sao cần: hai bộ IHR-NomDB (LucVanTien1916, TruyenKieu1872) có **nhãn người từng chữ**
(`data/<book>/manifest.tsv`). Chạy pipeline lên chúng là để ĐO độ đúng end-to-end, không phải để
thêm dữ liệu huấn luyện. Nếu bộ ra lọt vào train/val của bất kỳ mô hình nào thì mọi con số đánh giá
về sau đều vô nghĩa. `pipeline/tools/make_dataset_docs.py` không biết điều đó nên phải đóng dấu riêng.

Công cụ này chỉ GHI THÊM hai tệp vào thư mục export, không sửa `labels.csv` và không đọc `data/`:
  TAP_DANH_GIA.md          cảnh báo cho người đọc
  evaluation_only.json      cờ máy đọc được (`evaluation_only: true`) cho cổng CI về sau

Chạy:
  .venv/bin/python -m pipeline.tools.mark_eval_dataset --dataset dataset/LucVanTien1916 \
      --book LucVanTien1916 --gt data/LucVanTien1916/manifest.tsv
  .venv/bin/python -m pipeline.tools.mark_eval_dataset --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MARK_MD = "TAP_DANH_GIA.md"
MARK_JSON = "evaluation_only.json"

TEMPLATE = """# ⚠️ TẬP ĐÁNH GIÁ — KHÔNG TRỘN VÀO TẬP HUẤN LUYỆN

Bộ dữ liệu trong thư mục này sinh từ **{book}**, một bản in **đã có nhãn chữ Nôm do NGƯỜI gán**
(`{gt}`, IHR-NomDB). Pipeline chạy lên bản này **chỉ để ĐO độ đúng end-to-end** bằng cách so nhãn máy
với nhãn người; nó **không** phải nguồn dữ liệu huấn luyện.

## Ba điều cấm

1. **KHÔNG** đưa bất kỳ dòng nào của `labels.csv` này vào tập train/val của mô hình nào
   (nhận diện chữ, S3/ArcFace, detector). Nếu trộn, mọi phép đo "precision so nhãn người" về sau
   đều vô nghĩa vì mô hình đã thấy đáp án.
2. **KHÔNG** ghi đè hay sửa `{gt}` bằng nhãn máy. Adapter sinh ra bộ này (`pipeline/tools/ingest_ihr_book.py`)
   không đọc trường nhãn chữ Nôm nào của bộ gốc — phép đo chỉ giữ được tính độc lập khi điều đó còn đúng.
3. **KHÔNG** trích con số precision của bộ này làm "độ đúng của pipeline" nói chung: quốc ngữ đầu vào ở
   đây là **phiên âm của người** và ô cột là **ô người vẽ**, nên đó là **cận trên của phương pháp**,
   không phải độ đúng dự kiến trên sách có quốc ngữ do OCR đọc.

## Dùng đúng cách

- Báo cáo số: `measure_out/{book}/ihr_endtoend/summary.json`
  (`.venv/bin/python scripts/measure/ihr_endtoend_eval.py --book {book}`).
- Diễn giải + giới hạn: `docs/CHAY_3_BO_CON_LAI_2026-09-23.md` §4.

Đóng dấu ngày {today} bởi `pipeline/tools/mark_eval_dataset.py`.
"""


# 2026-09-27: bản chép tay Vatican Borgiano Tonchinese (--kind borg) — cùng ba điều cấm, khác nguồn/adapter/phép đo.
TEMPLATE_BORG = """# ⚠️ TẬP ĐÁNH GIÁ — KHÔNG TRỘN VÀO TẬP HUẤN LUYỆN

Bộ dữ liệu trong thư mục này sinh từ **{book}**, một bản **CHÉP TAY** chữ Nôm Công giáo (Vatican, Borgiano
Tonchinese) **đã có nhãn chữ Nôm do NGƯỜI phiên** (`{gt}`, cột `SinoNom_Char`). Pipeline chạy lên bản này
**chỉ để ĐO độ đúng trên chữ viết tay** bằng cách so nhãn máy với nhãn người; nó **không** phải nguồn dữ liệu huấn luyện.

## Ba điều cấm

1. **KHÔNG** đưa bất kỳ dòng nào của `labels.csv` này vào tập train/val của mô hình nào
   (nhận diện chữ, S3/ArcFace, detector, bộ kiểm chữ viết tay). Nếu trộn, mọi phép đo "độ đúng so nhãn người" về sau
   đều vô nghĩa vì mô hình đã thấy đáp án.
2. **KHÔNG** ghi đè hay sửa `{gt}` (hay `prepared/{book}/`) bằng nhãn máy. Adapter sinh ra bộ này
   (`pipeline/tools/ingest_borg_book.py`) không đọc cột chữ Nôm người — phép đo chỉ độc lập khi điều đó còn đúng.
3. **KHÔNG** trích con số của bộ này làm "độ đúng của pipeline" nói chung: quốc ngữ đầu vào ở đây là **phiên âm
   của người** (không phải OCR quốc ngữ), nên đó là **cận trên của phương pháp trên chữ viết tay**.

## Dùng đúng cách

- Báo cáo số: `measure_out/{book}/borg_endtoend/summary.json`
  (`.venv/bin/python scripts/measure/borg_endtoend_eval.py --book {book}`).
- Diễn giải + giới hạn: `docs/BORG_DANH_GIA_2026-09-27.md`.

Đóng dấu ngày {today} bởi `pipeline/tools/mark_eval_dataset.py`.
"""
KINDS = {
    "ihr": dict(template=TEMPLATE, report="docs/CHAY_3_BO_CON_LAI_2026-09-23.md", metrics="measure_out/{book}/ihr_endtoend/summary.json",
                reason="Bản in có nhãn chữ Nôm do người gán; bộ này chỉ dùng để đo độ đúng end-to-end, không được đưa vào tập huấn luyện."),
    "borg": dict(template=TEMPLATE_BORG, report="docs/BORG_DANH_GIA_2026-09-27.md",
                 metrics="measure_out/{book}/borg_endtoend/summary.json",
                 reason="Bản chép tay có nhãn chữ Nôm do người phiên (Excel SinoNom_Char); bộ này chỉ dùng để đo độ đúng "
                        "trên chữ viết tay, không được đưa vào tập huấn luyện."),
}


def mark(dataset: Path, book: str, gt: str, reason: str = "", kind: str = "ihr") -> dict:
    """Ghi TAP_DANH_GIA.md + evaluation_only.json vào thư mục export. Trả bản ghi JSON đã ghi.
    kind = ihr (mặc định, y hệt trước 27/09) | borg (bản chép tay Vatican)."""
    dataset = Path(dataset)
    if not dataset.is_dir():
        raise FileNotFoundError(f"không thấy thư mục export {dataset}")
    k = KINDS[kind]
    today = date.today().isoformat()
    (dataset / MARK_MD).write_text(k["template"].format(book=book, gt=gt, today=today), encoding="utf-8")
    rec = dict(evaluation_only=True, book=book, ground_truth_file=gt, marked_on=today,
               marked_by="pipeline/tools/mark_eval_dataset.py",
               reason=reason or k["reason"],
               report=k["report"],
               metrics=k["metrics"].format(book=book))
    if kind != "ihr":
        rec["kind"] = kind
    (dataset / MARK_JSON).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def is_marked(dataset: Path) -> bool:
    """True khi thư mục export đã được đóng dấu tập đánh giá (cổng CI đọc hàm này)."""
    p = Path(dataset) / MARK_JSON
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("evaluation_only"))
    except Exception:      # noqa: BLE001
        return False


def selftest() -> int:
    n = ok = 0

    def chk(name, cond):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'} {name}")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "dataset_x"
        chk("thư mục chưa có -> báo lỗi", _raises(lambda: mark(d, "B", "g")))
        d.mkdir()
        chk("chưa đóng dấu -> is_marked False", not is_marked(d))
        rec = mark(d, "LucVanTien1916", "data/LucVanTien1916/manifest.tsv")
        chk("ghi TAP_DANH_GIA.md", (d / MARK_MD).exists())
        chk("ghi evaluation_only.json", (d / MARK_JSON).exists())
        chk("cờ evaluation_only = True", rec["evaluation_only"] is True)
        chk("is_marked True sau khi đóng dấu", is_marked(d))
        txt = (d / MARK_MD).read_text(encoding="utf-8")
        chk("md nêu tên sách", "LucVanTien1916" in txt)
        chk("md nêu tệp nhãn người", "manifest.tsv" in txt)
        chk("md có cảnh báo không trộn train", "KHÔNG TRỘN VÀO TẬP HUẤN LUYỆN" in txt)
        chk("md có 3 điều cấm", txt.count("**KHÔNG**") >= 3)
        j = json.loads((d / MARK_JSON).read_text(encoding="utf-8"))
        chk("json ghi đường dẫn báo cáo số", j["metrics"].endswith("ihr_endtoend/summary.json"))
        chk("json ghi tệp GT", j["ground_truth_file"].endswith("manifest.tsv"))
        chk("đóng dấu 2 lần không lỗi", bool(mark(d, "B2", "g2")))
        chk("đóng dấu lại thì cập nhật tên sách",
            json.loads((d / MARK_JSON).read_text(encoding="utf-8"))["book"] == "B2")
        chk("json hỏng -> is_marked False", _corrupt_then_check(d))
        d2 = Path(td) / "dataset_borg"; d2.mkdir()
        rb = mark(d2, "SachDungLyHoThan", "data/SachDungLyHoThan/SachDungLyHoThan.xlsx", kind="borg")
        tb = (d2 / MARK_MD).read_text(encoding="utf-8")
        chk("borg: cờ + kind + đường đo borg_endtoend", rb["evaluation_only"] and rb["kind"] == "borg"
            and rb["metrics"].endswith("borg_endtoend/summary.json") and is_marked(d2))
        chk("borg: md nêu CHÉP TAY + 3 điều cấm", "CHÉP TAY" in tb and tb.count("**KHÔNG**") >= 3)
        chk("ihr mặc định không có khoá kind", "kind" not in mark(d, "LucVanTien1916", "g"))
    print(f"mark_eval_dataset selftest: {ok}/{n}")
    return 0 if ok == n else 1


def _raises(fn) -> bool:
    try:
        fn()
    except FileNotFoundError:
        return True
    except Exception:      # noqa: BLE001
        return False
    return False


def _corrupt_then_check(d: Path) -> bool:
    (d / MARK_JSON).write_text("{khong phai json", encoding="utf-8")
    return not is_marked(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="thư mục export, vd dataset/LucVanTien1916")
    ap.add_argument("--book", default="")
    ap.add_argument("--gt", default="", help="tệp nhãn người, vd data/<book>/manifest.tsv")
    ap.add_argument("--reason", default="")
    ap.add_argument("--kind", default="ihr", choices=sorted(KINDS), help="ihr (mặc định) | borg (bản chép tay Vatican)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.dataset or not a.book:
        ap.error("cần --dataset và --book (hoặc --selftest)")
    gt = a.gt or f"data/{a.book}/manifest.tsv"
    rec = mark(Path(a.dataset), a.book, gt, a.reason, kind=a.kind)
    print(f"[mark] {a.dataset}: đóng dấu TẬP ĐÁNH GIÁ ({MARK_MD}, {MARK_JSON}) — GT {rec['ground_truth_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
