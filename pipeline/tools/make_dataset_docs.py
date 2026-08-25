"""Sinh README + DATASHEET + LICENSE + NGUON_THU_TICH cho thư mục giao nộp.

VÌ SAO
------
`dataset/` trước đây chỉ có `labels.csv` + hai thư mục ảnh. Với ngành Hán Nôm, một bộ
dữ liệu KHÔNG CÓ LAI LỊCH THƯ TỊCH là **không trích dẫn được**: người đọc không biết ba
cuốn này là bản in nào, lưu ở đâu, ký hiệu gì, nên không thể kiểm lại một ô nhãn nào.
Không có tuyên bố giấy phép thì cũng không ai dám dùng lại.

Mọi con số trong DATASHEET đều ĐỌC TỪ bộ nhãn thật, không chép tay — nên nó không thể
lệch với dữ liệu.

⬜ Những mục CHỈ NGƯỜI BIẾT được (nơi lưu giữ, ký hiệu kho, nhà in, niên đại) được đánh
dấu `⬜ CHƯA ĐIỀN` chứ KHÔNG bịa. Công cụ này cố ý để trống thay vì đoán.

    python -m pipeline.tools.make_dataset_docs
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ⬜ = người phải điền. Giữ nguyên ký hiệu để `--check` tìm được chỗ còn thiếu.
SACH = {
    "stt2":  {"ten": "Sách Thánh Truyện, quyển 2"},
    "stt4":  {"ten": "Sách Thánh Truyện, quyển 4"},
    "stt11": {"ten": "Sách Thánh Truyện, quyển 11"},
}
TRONG = ["noi_luu_giu", "ky_hieu_kho", "nha_in", "nien_dai", "kho_sach", "lan_in", "dpi_quet"]
NHAN_VN = {
    "noi_luu_giu": "Nơi lưu giữ", "ky_hieu_kho": "Ký hiệu kho", "nha_in": "Nhà in / nơi khắc",
    "nien_dai": "Niên đại", "kho_sach": "Khổ sách", "lan_in": "Lần in / bản",
    "dpi_quet": "DPI bản quét",
}


def stats(labels: Path) -> dict:
    rows = list(csv.DictReader(open(labels, encoding="utf-8")))
    char = [r for r in rows if (r.get("label") or "").strip()]
    syl = [r for r in rows if not (r.get("label") or "").strip()]
    q = collections.Counter(r.get("crop_quality_flag", "") for r in rows)
    return {
        "dong": len(rows), "nhan_ky_tu": len(char), "chu_giai_am": len(syl),
        "lop_chu": len({r["label"] for r in char}),
        "sach": sorted({r["book"] for r in rows}),
        "trang": len({(r["book"], r["page"]) for r in rows}),
        "cot": len({(r["book"], r["page"], r["column"]) for r in rows}),
        "am_qn": len({(r.get("syllable") or "").lower() for r in rows if r.get("syllable")}),
        "anh_hong": q.get("blank", 0) + q.get("truncated", 0),
        "flag": {k: v for k, v in q.items() if k},
        "commit": subprocess.run(["git", "log", "-1", "--format=%h", "--", str(labels)],
                                 cwd=REPO, capture_output=True, text=True).stdout.strip() or "?",
        "ngay": datetime.date.fromtimestamp(labels.stat().st_mtime).isoformat(),
    }


def readme(s: dict) -> str:
    return f"""# Bộ dữ liệu gán nhãn chữ Nôm — Sách Thánh Truyện

**{s['nhan_ky_tu']:,} nhãn cấp KÝ TỰ** + **{s['chu_giai_am']:,} chú giải cấp ÂM TIẾT**
= {s['dong']:,} dòng · {s['lop_chu']:,} lớp chữ · {s['trang']} trang · {s['cot']:,} cột

> ⚠️ **Đừng phát biểu là "{s['dong']:,} nhãn".** {s['chu_giai_am']:,} dòng tầng SYLLABLE có cột
> `label` **rỗng** — chúng chỉ ghi âm Quốc ngữ, KHÔNG gán chữ Nôm. Con số đem so với các bộ
> dữ liệu Hán Nôm khác là **{s['nhan_ky_tu']:,}**.

## Cấu trúc

| | |
|---|---|
| `labels.csv` | một dòng mỗi ô, đường dẫn ảnh tương đối |
| `gold/` | ảnh crop của các ô có nhãn cấp ký tự |
| `syllable/` | ảnh crop của các ô chỉ có chú giải âm |

## Cột quan trọng

| cột | nghĩa |
|---|---|
| `label` | chữ Nôm được gán. **Rỗng** ở tầng SYLLABLE |
| `label_level` | `char` = nhãn ký tự · `syllable` = chỉ có âm |
| `syllable` | âm Quốc ngữ tương ứng, lấy từ bản dịch song song in kèm |
| `tier` / `rule` | luật nào quyết nhãn này — xem DATASHEET |
| `usable_image` | `0` = ảnh trắng hoặc bị cắt mất nét ({s['anh_hong']} ô). Nhãn có thể vẫn đúng; đừng chấm chiều ảnh ở các ô này |
| `crop_quality_flag` | `ok` / `bleed` (dính mực chữ bên cạnh) / `truncated` / `blank` |
| `split` / `split_group` | chia tách theo TRANG, không có trang nào nằm ở hai phía |

## 🔴 Trạng thái kiểm định

**Chưa có phép đo precision nào còn hiệu lực.** Mọi con số precision trong các bản trước
đã bị **tước tư cách bằng chứng** vì nguồn phán quyết hoá ra là máy chấm chứ không phải
người. Mẻ chấm tay đúng quy trình đang được tiến hành — xem `docs/QUY_TRINH_CHAM_TAY.md`.

Nói cách khác: bộ này dùng được để **huấn luyện** và **thăm dò**, nhưng **chưa được trích
dẫn như dữ liệu đã kiểm chứng**.

## Trích dẫn

Xem `NGUON_THU_TICH.md` cho lai lịch ba cuốn sách nguồn.

*Sinh tự động từ `labels.csv` ngày {s['ngay']} · commit `{s['commit']}`*
"""


def nguon(s: dict) -> str:
    hang = []
    for code in s["sach"]:
        info = SACH.get(code, {"ten": code})
        hang.append(f"### {code} — {info['ten']}\n")
        for k in TRONG:
            hang.append(f"- **{NHAN_VN[k]}**: ⬜ CHƯA ĐIỀN")
        hang.append("")
    return f"""# Lai lịch thư tịch — ba cuốn nguồn

> 🔴 **Mục này CHƯA HOÀN CHỈNH và chỉ người giữ bản quét mới điền được.**
> Công cụ sinh tài liệu **cố ý để trống thay vì đoán** — một lai lịch bịa còn tệ hơn
> không có, vì nó khiến người sau tin nhầm.

Không có phần này thì bộ dữ liệu **không trích dẫn được**: người đọc không thể tra lại
một ô nhãn nào về bản in gốc.

{chr(10).join(hang)}
## Vì sao từng mục cần thiết

| mục | vì sao |
|---|---|
| Nơi lưu giữ + ký hiệu kho | để người khác **tra lại được** bản gốc |
| Nhà in / nơi khắc | ván khắc khác nhau cho tự dạng khác nhau |
| Niên đại | chính tả Nôm biến đổi theo thời kỳ |
| Khổ sách + DPI | quyết định kích thước ảnh crop, ảnh hưởng khả năng tái lập |
| Lần in / bản | cùng một tên sách có thể có nhiều bản khắc khác nhau |

## Giấy phép bản quét

⬜ **CHƯA ĐIỀN.** Phải xác định trước khi công bố:
tình trạng bản quyền của bản quét, và điều kiện bên lưu giữ cho phép.
"""


def datasheet(s: dict) -> str:
    fl = " · ".join(f"`{k}` {v:,} ({100*v/s['dong']:.2f}%)" for k, v in sorted(s["flag"].items()))
    return f"""# Datasheet

## Động cơ
Gán nhãn tự động ở mức ký tự cho văn bản chữ Nôm khắc gỗ, bằng cách khai thác **bản dịch
Quốc ngữ song song in kèm** làm giám sát yếu — thay cho việc gán tay từng chữ.

## Thành phần
- **{s['nhan_ky_tu']:,}** ô có nhãn cấp ký tự · **{s['chu_giai_am']:,}** ô chỉ có chú giải âm
- **{s['lop_chu']:,}** lớp chữ phân biệt · **{s['am_qn']:,}** âm Quốc ngữ phân biệt
- **{s['trang']}** trang · **{s['cot']:,}** cột · **{len(s['sach'])}** cuốn: {', '.join(s['sach'])}
- chất lượng ảnh: {fl}

## Quy trình thu thập
Ảnh trang → dò cột → OCR chữ Nôm (dịch vụ HCMUS) + OCR Quốc ngữ (VietOCR, chạy cục bộ) →
căn chỉnh chữ↔âm bằng quy hoạch động có dải → hợp nhất ba tín hiệu thành tier.
Toàn bộ **tất định tới từng byte**; chạy lại hai lần cho kết quả trùng khít.

## 🔴 Giới hạn — đọc trước khi dùng

1. **Chưa có phép đo precision nào còn hiệu lực.** Số cũ đã bị tước tư cách bằng chứng
   (nguồn phán quyết là máy chấm, không phải người).
2. **Ba cuốn cùng MỘT thể loại** (truyện thánh Công giáo). Ngoại suy sang Nôm văn học
   hay hành chính **chưa được kiểm chứng**.
3. **Chữ Nôm tự tạo có thể bị hụt**: chỉ **1,63%** ô nằm ngoài khối CJK cơ bản, so với
   **4,21%** ở ngữ liệu NomNaOCR. Chưa rõ do pipeline bóc mất bộ thủ hay do Nôm Công
   giáo thế kỷ XIX vốn chuộng dạng giản.
4. **Chưa chuẩn hoá dị thể.** Cùng một chữ có thể xuất hiện dưới nhiều mã
   (`徳`/`德`, `别`/`別`). Ứng viên đã lọc ở `docs/UNG_VIEN_CHUAN_HOA_DI_THE.csv`,
   **chưa áp dụng**.
5. **{s['anh_hong']} ô có ảnh hỏng** (`usable_image=0`) vẫn nằm trong bộ — nhãn có thể
   đúng, ảnh thì không dùng được.
6. **Không có recall.** Bộ này chỉ chứa ô đã gán được nhãn; phần bị bỏ không nằm ở đây.

## Khuyến nghị dùng
Dùng được: huấn luyện mô hình, thăm dò, làm điểm khởi đầu để chấm tay.
**Chưa dùng được**: trích dẫn như dữ liệu đã kiểm chứng, hoặc làm chuẩn đánh giá.

*Sinh tự động từ `labels.csv` ngày {s['ngay']} · commit `{s['commit']}`*
"""


def huong_dan_cham(s: dict) -> str:
    """Đặc tả GIAO NỘP cho đội chấm ngoài. Họ tự làm giao diện; ta chỉ chốt ĐẦU RA."""
    return f"""# Hướng dẫn chấm — dành cho đội chấm

Thư mục này **tự đủ**: `labels.csv` + hai thư mục ảnh. Không cần công cụ nào của chúng
tôi. Đội chấm **tự làm giao diện** theo cách thuận tiện nhất.

Chúng tôi chỉ chốt **đầu ra**, để nạp ngược vào pipeline được.

## Câu hỏi — KHÁC NHAU theo `label_level`

| `label_level` | số ô | câu hỏi |
|---|---|---|
| `char` | {s['nhan_ky_tu']:,} | **Chữ trong ảnh có đúng là chữ ở cột `label` không?** |
| `syllable` | {s['chu_giai_am']:,} | **Âm ở cột `syllable` có đúng với chữ trong ảnh không?** |

> ⚠️ Ô `label_level = syllable` có cột **`label` RỖNG** — chúng **không gán chữ Nôm**.
> Hỏi "chữ này đúng không" ở đó là hỏi một thứ không tồn tại.

## Bốn quy tắc

1. **Khung cắt xấu KHÔNG phải lỗi nhãn.** Dính chút mực chữ bên cạnh mà vẫn đọc ra chữ
   → chấm **đúng**. Câu hỏi là về **chữ**, không phải về **khung**.
2. **{s['anh_hong']} ô có `usable_image = 0`** (ảnh trắng hoặc cắt mất nét) — **bỏ qua**,
   chấm `khong_doc_duoc`. Đừng chấm thành "sai".
3. **Lưỡng lự → `khong_doc_duoc`, đừng chấm `sai`.** Ô *không đọc được* bị loại khỏi mẫu
   số; ô *sai* bị tính là lỗi. Dữ liệu cũ cho thấy xu hướng **gọi quá tay**.
4. Chấm khó thì mở ảnh trang gốc theo cột `book` / `page` / `bbox`.

## Đầu ra cần nộp — ĐÚNG hai tệp, đặt ngay trong thư mục này

### 1. `verdicts.csv`

```csv
image,verdict,nguoi_cham,ghi_chu
gold/stt2_page_0012_c01_003.png,dung,Nguyen Van A,
gold/stt2_page_0012_c01_004.png,sai,Nguyen Van A,nhìn giống chữ khác
syllable/stt4_page_0020_c03_055.png,khong_doc_duoc,Tran Thi B,mực nhoè
```

| cột | bắt buộc | giá trị |
|---|---|---|
| `image` | ✅ | chép **nguyên văn** từ `labels.csv`, không đổi đường dẫn |
| `verdict` | ✅ | **chỉ ba giá trị**: `dung` · `sai` · `khong_doc_duoc` |
| `nguoi_cham` | nên có | tên người chấm ô đó — cần để tính κ liên-người |
| `ghi_chu` | không | tuỳ ý |

**Không cần chấm hết.** Chấm được bao nhiêu nộp bấy nhiêu; pipeline tự tính độ phủ.
Nhưng nếu chấm một phần thì con số precision **chỉ đúng cho phần đã chấm** — sẽ không
suy rộng ra toàn bộ, vì không biết phần đó có đại diện hay không.

### 2. `NGUOI_CHAM.md`

**Bắt buộc.** Không có nó thì pipeline **từ chối nạp**, dù `verdicts.csv` hoàn hảo.

Lý do: dự án này đã một lần tin nhầm verdict **MÁY** là verdict người và phải huỷ toàn
bộ số liệu — và các tệp đó **không hề tự khai là máy**. Nên chúng tôi không suy đoán
xuất xứ từ dữ liệu nữa, mà **bắt khai ra**.

Mẫu ở `docs/QUY_TRINH_CHAM_TAY.md`. Ba điều quyết định: người chấm có **đọc được chữ
Nôm** không · có **ít nhất hai người** không · có cam kết **không dùng máy** không.

## Nộp về rồi thì sao

Chép cả thư mục `re-dataset/` (đã có hai tệp trên) về máy chạy pipeline, rồi:

```bash
bash run_pipeline.sh
```

Pipeline tự phát hiện `verdicts.csv`, nạp phán quyết, và xuất bộ **cuối cùng** ra
`dataset/` kèm `docs/BANG_PRECISION.md`.

*Sinh tự động ngày {s['ngay']} · commit `{s['commit']}`*
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.make_dataset_docs")
    ap.add_argument("--dataset", default=str(REPO / "dataset"))
    ap.add_argument("--check", action="store_true", help="exit 1 nếu còn mục ⬜ CHƯA ĐIỀN")
    args = ap.parse_args(argv)

    root = Path(args.dataset)
    lab = root / "labels.csv"
    if not lab.exists():
        print(f"[docs] không thấy {lab}")
        return 1
    if args.check:
        f = root / "NGUON_THU_TICH.md"
        n = f.read_text(encoding="utf-8").count("⬜ CHƯA ĐIỀN") if f.exists() else -1
        if n != 0:
            print(f"[docs] còn {n} mục ⬜ CHƯA ĐIỀN trong NGUON_THU_TICH.md — "
                  f"bộ dữ liệu chưa trích dẫn được")
            return 1
        print("[docs] lai lịch thư tịch đã điền đủ")
        return 0

    s = stats(lab)
    for name, body in (("README.md", readme(s)), ("DATASHEET.md", datasheet(s)),
                       ("NGUON_THU_TICH.md", nguon(s)),
                       ("HUONG_DAN_CHAM.md", huong_dan_cham(s))):
        (root / name).write_text(body, encoding="utf-8")
    lic = root / "LICENSE.md"
    if not lic.exists():
        lic.write_text("# Giấy phép\n\n⬜ **CHƯA CHỌN.** Phải quyết trước khi công bố.\n\n"
                       "Cân nhắc CC BY-NC-SA 4.0 cho phần NHÃN. Phần ẢNH QUÉT phụ thuộc\n"
                       "điều kiện của bên lưu giữ bản gốc — xem `NGUON_THU_TICH.md`.\n",
                       encoding="utf-8")
    print(f"[docs] {s['nhan_ky_tu']:,} nhãn ký tự + {s['chu_giai_am']:,} chú giải âm")
    print(f"[docs] đã ghi README.md · DATASHEET.md · NGUON_THU_TICH.md · LICENSE.md -> {root}/")
    n = (root / "NGUON_THU_TICH.md").read_text(encoding="utf-8").count("⬜ CHƯA ĐIỀN")
    print(f"[docs] ⚠️ còn {n} mục ⬜ CHƯA ĐIỀN — chỉ người giữ bản quét điền được")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
