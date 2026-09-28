"""doc_text.py — khối văn bản mô tả `gold_exact.csv` cho README/DATASHEET của bộ giao nộp (chỉ chuỗi, không nạp numpy/torch).

Dùng chung bởi pipeline/tools/merge_datasets.py (dataset/_ALL/) và pipeline/tools/make_dataset_docs.py (dataset/<Bộ>/) để hai nơi
không lệch nhau. Mức chắc (evidence_level) phải trùng common.EVIDENCE — selftest của gói kiểm điều này.
KHÔNG chứa chuỗi tên tầng text-only của bộ giao nộp (selftest mechanism_gates dò chuỗi đó để biết README có khối tầng hay không).
"""
from __future__ import annotations

EVIDENCE_DOC = {
    "do_tren_nhan_nguoi": ("L16", "TK", "B18", "B34"),   # B18/B34 = Borg chép tay (27/09), chỉ khi có trong bộ gộp
    "uoc_luong": ("KVK", "L83"),
    "suy_doan": ("stt2", "stt4", "stt11", "Chr"),
}

TRANG_THAI = [
    ("ok", "ảnh + chữ cùng qua mọi cổng tự động; ảnh giao = **crop chuẩn** (`crop_chuan`, khung vuông một chữ) — "
           "cột `image` gốc vẫn giữ nguyên trong `labels.csv`"),
    ("text_only", "nhãn chữ giữ, nhưng ảnh không chắc là đúng MỘT chữ của đúng khe (crop trắng/cắt nét/hai chữ, "
                  "hộp một-hộp-hai-cột, số chữ OCR ≠ số âm QN, trượt so với hộp chữ kim)"),
    ("uncertified", "không có bằng chứng ảnh↔chữ đủ mạnh (bộ kiểm ảnh dưới ngưỡng TN1), dị bản người không chứng nhãn, "
                    "hoặc (STT, khi đủ cache lt2) lần đọc kim lt2 không xác nhận nhãn"),
    ("review", "luật A (ảnh của ô khác, nhãn rescue, cầu tự dạng, văn bản yếu) hoặc M-OCR t50 — cần xem lại"),
]


# Giới hạn phải đi kèm mọi mô tả gold_exact.csv (README bộ gộp + bộ nguồn, GOLD_EXACT.md) — số lấy từ TN4 §5.1 / TN3 §6.
GIOI_HAN = """>
> Biên độ theo chọn ngưỡng (TN4 §5.1, bootstrap trang tune B = 300, 5 %–95 %): số ô giữ L16 1.343–6.670, TK 9.200–12.137 —
> con số ok là của MỘT điểm vận hành đăng ký trước, không phải hằng số. **Lần đọc thứ hai STT bằng kim lt2 (H7)**: luật CHỈ HẠ
> (ok -> `uncertified` khi nhãn ≠ chữ lt2 hoặc không ghép được), CHỈ bật cho bộ STT có ĐỦ cache lt2 (trạng thái từng bộ:
> `GOLD_EXACT.md`); hiệu chuẩn trên thạch bản (ước lượng) — với chữ viết tay vẫn là SUY ĐOÁN; TN3 §6: kể cả xoá hết lỗi riêng
> của STT, cận bi quan vẫn ≤ 97,31 % → STT vẫn là suy đoán.
"""

BEGIN, END = "<!-- gold_exact:begin -->", "<!-- gold_exact:end -->"
DS_BEGIN, DS_END = "<!-- gold_exact_ds:begin -->", "<!-- gold_exact_ds:end -->"
LEGACY_HEAD = "## GOLD chính xác — `gold_exact.csv`"
LEGACY_DS = "**GOLD chính xác:**"


def block_md(where: str = "all", heading: str = "##") -> str:
    """where = 'all' (dataset/_ALL) | 'book' (dataset/<Bộ>, chỉ dùng khi tệp gold_exact.csv CÓ trong thư mục bộ)."""
    per_book = ("" if where != "all" else
                "\nBản lọc theo bộ `dataset/<Bộ>/gold_exact.csv` CHỈ có ở những bộ mà bước gold_exact đã ghi trong lượt này "
                "(danh sách thật: `GOLD_EXACT.md` §8); dựng lại một bộ (`--book <Bộ>`) xoá bản lọc của bộ đó cho tới lần "
                "gộp + gold_exact sau (`./run_pipeline.sh --merge`).\n")
    paths = ("`crop_chuan` = `crops_chuan/<book_set>/<book>/<page>/c<cột>_n<nom_idx>_s<syl_idx>.png` "
             "(+ `crop_chuan_128` 128×128), tương đối với `dataset/_ALL/`" if where == "all" else
             "`crop_chuan` = `../_ALL/crops_chuan/…` (ảnh crop chuẩn chỉ nằm ở bộ gộp, tệp này không chép crop); "
             "`image` = đường ảnh gốc tương đối với thư mục bộ này (như `labels.csv`)")
    st = "\n".join(f"| `{k}` | {v} |" for k, v in TRANG_THAI)
    ev = " · ".join(f"`{k}`: {', '.join(v)}" for k, v in EVIDENCE_DOC.items())
    return f"""{heading} GOLD chính xác — `gold_exact.csv` (bước sau khi gộp, 0 API)

`gold_exact.csv` = một dòng cho MỖI ô `tier == "GOLD"` (khoá `cell_uid`), sinh bởi `python -m pipeline.gold_exact --publish`
(`./run_pipeline.sh --gold-exact on`, mặc định). Bước này **không sửa nhãn và không đổi `labels.csv`**; nó chỉ gắn cho mỗi ô
GOLD đúng một trong 4 trạng thái (luật đầu tiên khớp thắng, theo thứ tự review → text_only → uncertified → ok):

| `gold_exact` | nghĩa |
|---|---|
{st}

Cột chính: `gold_exact`, `reason` (luật đã quyết), `evidence_level`, `policy_version`, `config_sha16`
(16 hex đầu sha256 `config/gold_exact.yaml`), `image` + `image_file_md5` (ảnh gốc), {paths},
`crop_chuan_md5` / `crop_chuan_128_md5` (chỉ ô ok), các điểm/cờ đã dùng (`p_wood_*`, `viss_*`, `lobo_*`, `ta`, `cnt`, `bc`, …)
và cột thông tin `core_loss` / `core_loss_flag` (KHÔNG còn là cổng — đo trước trên nhãn người cho kết quả âm).

`evidence_level` = mức chắc của độ chính xác ô ok: {ev}.

> ⚠️ **Độ chính xác chỉ ĐO được ở L16/TK** (nhãn người IHR-NomDB, xem `GOLD_EXACT.md` §2 của bộ gộp — hai bộ này là TẬP
> ĐÁNH GIÁ). **KVK/L83 chỉ là ước lượng** (qua dị bản người); **STT/Chr là suy đoán** (không có sự thật nào để đo).
> Số ô ok phụ thuộc ngưỡng đã chọn (TN1 τ = 0,995) — đừng trích "ok" như tập đã kiểm chứng ở các bộ không có nhãn người.
{GIOI_HAN}{per_book}"""


def datasheet_line() -> str:
    return ("`gold_exact.csv` (4 trạng thái ok/text_only/uncertified/review cho mỗi ô GOLD, bước gold_exact sau khi gộp): "
            "độ chính xác của ô ok chỉ ĐO được trên L16/TK (nhãn người); KVK/L83 là ước lượng; STT/Chr là suy đoán — "
            "xem `GOLD_EXACT.md` và `docs/GOLD_CHINH_XAC_2026-09-27.md`.")


def absent_md(heading: str = "##") -> str:
    """Khối README của bộ nguồn khi thư mục bộ KHÔNG có gold_exact.csv (không mô tả tệp không tồn tại)."""
    return f"""{heading} GOLD chính xác — `gold_exact.csv`

Thư mục này hiện **không có** `gold_exact.csv`. Tệp đó (4 trạng thái ok/text_only/uncertified/review cho mỗi ô GOLD) chỉ do
bước gold_exact sinh NGAY SAU bước gộp (`./run_pipeline.sh --merge` hoặc `--book all`) và bị xoá mỗi khi bộ này được dựng lại
(`--book <Bộ>`) hay bộ gộp được gộp lại — để không bao giờ lệch `labels.csv`. Mô tả: `../_ALL/README.md`, `../_ALL/GOLD_EXACT.md`.
"""


def book_section(present: bool, heading: str = "##") -> str:
    """Khối README của bộ nguồn, bọc marker để publish / bước gộp đổi qua lại khi tệp xuất hiện / bị xoá."""
    body = block_md("book", heading) if present else absent_md(heading)
    return f"{BEGIN}\n{body.rstrip()}\n{END}\n"


def datasheet_block(present: bool) -> str:
    line = datasheet_line() if present else ("thư mục này hiện không có `gold_exact.csv` (chỉ sinh sau bước gộp + gold_exact: "
                                             "`./run_pipeline.sh --merge`; xem `../_ALL/GOLD_EXACT.md`).")
    return f"{DS_BEGIN}\n{LEGACY_DS} {line}\n{DS_END}"


def _swap(text: str, begin: str, end: str, new: str, legacy_start: str | None, legacy_is_line: bool) -> tuple[str, bool]:
    if begin in text and end in text:
        a = text.index(begin); b = text.index(end, a) + len(end)
        if text[b:b + 1] == "\n" and new.endswith("\n"):
            b += 1
        return text[:a] + new + text[b:], True
    if legacy_start and legacy_start in text:              # bản cũ chưa có marker (README sinh trước 28/09)
        a = text.index(legacy_start)
        if legacy_is_line:
            b = text.find("\n", a); b = len(text) if b < 0 else b
            return text[:a] + new.rstrip("\n") + text[b:], True
        b = text.find("\n## ", a + len(legacy_start))
        b = len(text) if b < 0 else b + 1
        return text[:a] + new + text[b:], True
    return text, False


def refresh_book_docs(book_dir, present: bool) -> list[str]:
    """Đổi khối gold_exact trong README.md / DATASHEET.md của bộ nguồn theo việc tệp có mặt hay không. Trả tên tệp đã sửa.
    Không có marker lẫn khối cũ -> không đụng tệp (không chèn thêm)."""
    from pathlib import Path
    d = Path(book_dir)
    done = []
    for name, begin, end, new, legacy, is_line in (
            ("README.md", BEGIN, END, book_section(present), LEGACY_HEAD, False),
            ("DATASHEET.md", DS_BEGIN, DS_END, datasheet_block(present), LEGACY_DS, True)):
        f = d / name
        if not f.is_file():
            continue
        t = f.read_text(encoding="utf-8")
        t2, hit = _swap(t, begin, end, new, legacy, is_line)
        if hit and t2 != t:
            f.write_text(t2, encoding="utf-8"); done.append(name)
    return done
