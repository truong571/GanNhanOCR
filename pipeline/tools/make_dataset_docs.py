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
    # SCHEMA 12 CỘT (16/09): cờ chất lượng ảnh nằm ở sidecar labels_trace.csv, còn danh
    # sách cột của từng trang ở columns.csv. Thế hệ cũ (≤25/08, 30+ cột) vẫn đọc được:
    # cột nằm ngay trong labels.csv thì lấy ở đó.
    root = labels.parent
    trace_p, cols_p = root / "labels_trace.csv", root / "columns.csv"
    trace = list(csv.DictReader(open(trace_p, encoding="utf-8"))) if trace_p.exists() else []
    flag_src = trace if (trace and "crop_quality_flag" in trace[0]) else rows
    q = collections.Counter(r.get("crop_quality_flag", "") for r in flag_src)
    # TRANG KHÔNG ĐỦ 9 CỘT: đo từ columns.csv (dựng trên TOÀN BỘ hàng, kể cả REVIEW — đếm
    # trên hàng đã lọc thì trang lành có nguyên cột rơi vào REVIEW sẽ mang tiếng oan).
    # Không có columns.csv (thế hệ cũ) thì đọc cờ page_cot_lech cũ.
    if cols_p.exists():
        _pg = collections.defaultdict(set)
        for r in csv.DictReader(open(cols_p, encoding="utf-8")):
            _pg[(r.get("book"), r.get("page"))].add(r.get("column"))
        trang_lech = sorted(k for k, v in _pg.items() if len(v) != 9)
        _co_trang = {(r.get("book"), r.get("page")) for r in rows}
        trang_lech = [k for k in trang_lech if k in _co_trang]
    else:
        trang_lech = sorted({(r.get("book"), r.get("page")) for r in rows
                             if r.get("page_cot_lech") == "1"})
    _tl = set(trang_lech)
    _lech = sum(1 for r in rows if (r.get("book"), r.get("page")) in _tl)
    # ĐO tỷ lệ chữ ngoài khối CJK cơ bản. Trước 2026-08-25 con số này là chuỗi ghim cứng
    # "1,63%" nằm lọt giữa một f-string mà mọi số quanh nó đều động — đo lại được 2,25%.
    _ngoai = sum(1 for r in char if not ("\u4e00" <= r["label"] <= "\u9fff"))
    return {
        "dong": len(rows), "nhan_ky_tu": len(char), "chu_giai_am": len(syl),
        "lop_chu": len({r["label"] for r in char}),
        "sach": sorted({r["book"] for r in rows}),
        "trang": len({(r["book"], r["page"]) for r in rows}),
        "cot": len({(r["book"], r["page"], r["column"]) for r in rows}),
        "am_qn": len({(r.get("syllable") or "").lower() for r in rows if r.get("syllable")}),
        "anh_hong": q.get("blank", 0) + q.get("truncated", 0),
        "flag": {k: v for k, v in q.items() if k},
        "cot_lech": _lech, "trang_cot_lech": len(trang_lech),
        "ds_trang_lech": [f"{b}/{pg}" for b, pg in trang_lech],
        "ngoai_cjk": _ngoai,
        "ngoai_cjk_pct": (100 * _ngoai / len(char)) if char else 0.0,
        "cot_labels": list(rows[0].keys()) if rows else [],
        "cot_trace": list(trace[0].keys()) if trace else [],
        "co_columns_csv": cols_p.exists(),
        "commit": subprocess.run(["git", "log", "-1", "--format=%h", "--", str(labels)],
                                cwd=REPO, capture_output=True, text=True).stdout.strip() or "?",
        "ngay": datetime.date.fromtimestamp(labels.stat().st_mtime).isoformat(),
    }


def _pct(x: float) -> str:
    """Dấu thập phân kiểu Việt — để không đứng cạnh '4,21%' mà viết '2.25%'."""
    return f"{x:.2f}".replace(".", ",")


# Nghĩa từng cột giao nộp (12 cột, DANH_MUC_SUA_DOI_CUOI_2026-09-16 §2) và sidecar.
NGHIA_COT = {
    "image": "đường ảnh crop tương đối, **khoá chính** (duy nhất)",
    "book": "mã sách (`stt2`/`stt4`/`stt11`)",
    "page": "tên trang trên bản quét — **đơn vị chia tách** nếu người dùng cần",
    "column": "số cột trên trang (1–9, bố cục ván khắc luôn 9 cột)",
    "ocr_char": "chữ OCR Nôm (S1) — bằng chứng trực tiếp, không rỗng",
    "syllable": "âm Quốc ngữ (lower/NFC) lấy từ bản phiên âm in kèm — nhãn tầng SYLLABLE",
    "label": "chữ Nôm được gán. **Rỗng** ở tầng SYLLABLE",
    "unicode": "mã của `label` (`U+XXXX`), để Excel không hiện được Ext-B vẫn tra được",
    "tier": "`GOLD` = nhãn cấp ký tự · `SYLLABLE` = chỉ có âm (thay cột `label_level` cũ)",
    "rule": "luật quyết nhãn (`quyet_dinh_nguoi:*` = phán quyết NGƯỜI, QĐ-01) — xem DATASHEET",
    "bbox": "toạ độ `[x1, y1, x2, y2]` trên ảnh trang gốc — không tái lập được từ crop",
    "image_md5": "md5 (12 hex đầu) của tệp crop: kiểm toàn vẹn khi sao chép, bắt trùng crop",
}
NGHIA_TRACE = {
    "nom_idx": "thứ tự chữ trong cột OCR Nôm (khoá bền cùng `book,page,column`)",
    "syl_idx": "thứ tự âm trong cột Quốc ngữ",
    "syllable_ocr": "âm Quốc ngữ NGUYÊN VĂN VietOCR (lower) — \"ghi đè bản in\" đo trên cột này",
    "syllable_raw": "âm sau `normalize_column` (vá dấu), TRƯỚC mọi sửa L1",
    "tier_v3": "tầng theo luật v3 (`CHAR_A`/`CHAR_B`/`SYL`/…) trước khi ánh xạ sang `tier`",
    "tier_goc": "tier trước khi bị hạ (rỗng nếu chưa từng hạ)",
    "rule_goc": "luật trước khi bị hạ (rỗng nếu chưa từng hạ)",
    "p_register": "xác suất hậu nghiệm P(chữ i ↔ âm j) — forward–backward trên lưới DP có băng",
    "dict_support": "số chữ từ điển cho âm này (|R(s)|); `corpus` khi nhãn do `corpus_readings`",
    "context_evidence": "cờ ngữ cảnh ĐÚNG, nối bằng `|`: `bigram|corpus4|tone|corpus2|direct|sim_unique`",
    "l1_support": "L1: n(âm mới) − n(âm gốc) theo 2-gram âm–âm (LOO trang); > 0 mới đổi âm",
    "l1_tie": "1 = L1 hoà → giữ âm gốc",
    "flank_gold": "số ô kề (`syl_idx` ± 1, cùng cột) có `tier_v3 == CHAR_A` ∈ {0, 1, 2}",
    "box_source": "nguồn hộp `bbox`: `detector` / `split` / `midpoint` / `legacy_locked_col` / `qd01_locked`",
    "qd01_locked": "1 = ô khoá theo phán quyết NGƯỜI QĐ-01 (giữ `bbox` cũ)",
    "qd01_excluded": "1 = ô QĐ-01 bị loại (`bo`) hoặc còn chờ người",
    "label_canonical": "mã chuẩn dị thể theo bảng đã ký (mặc định = `label`; `label` không đổi)",
    "crop_quality_flag": "`ok` / `bleed` (dính mực chữ bên) / `truncated` / `blank` — "
                         "**`blank`/`truncated` = ảnh hỏng, đừng chấm chiều ảnh**",
    "stray_ink": "tỷ lệ mực lạc (không thuộc chữ chính)", "border_ink": "mực sát biên crop",
    "ink_pct": "tỷ lệ điểm mực trên crop",
    "crop_w": "rộng crop (px)", "crop_h": "cao crop (px)", "seg_flag": "cờ tách chữ của bước crop",
    "s3_cosine": "cosine với nguyên mẫu thị giác S3 (rỗng khi không tính / S3 tắt)",
}


def _khoi_cot(s: dict) -> str:
    """Bảng cột đọc từ CHÍNH header trên đĩa — cột lạ (thế hệ cũ) vẫn liệt kê, không nói dối."""
    t = ["## Cột của `labels.csv`", "", "| cột | nghĩa |", "|---|---|"]
    for c in s["cot_labels"]:
        t.append(f"| `{c}` | {NGHIA_COT.get(c, '(cột thế hệ cũ, xem lịch sử README)')} |")
    if s["cot_trace"]:
        t += ["", "## Sidecar `labels_trace.csv` — chẩn đoán, CÙNG số dòng và thứ tự, khoá `image`", "",
              "Không cần cho việc dùng nhãn; giữ để truy vết vì sao một ô được gán như vậy.", "",
              "| cột | nghĩa |", "|---|---|"]
        for c in s["cot_trace"]:
            if c == "image":
                continue
            t.append(f"| `{c}` | {NGHIA_TRACE.get(c, '')} |")
    if s["co_columns_csv"]:
        t += ["", "## `columns.csv` — một dòng mỗi cột trang, khoá `book,page,column`", "",
              "Dựng trên TOÀN BỘ ô (kể cả ô không giao nộp) nên đủ để suy trang thiếu cột;",
              "các cột `n_ocr` (số chữ OCR Nôm), `n_qn` (số âm Quốc ngữ), `n_det` (số hộp",
              "detector), `count_source` (`equal_qn`/`equal_ocr`/`conflict`/`legacy_locked_col`)",
              "chỉ có ở bộ dựng từ 16/09 trở đi."]
    return "\n".join(t)


def _khoi_khong_chia(s: dict) -> str:
    """Bộ giao nộp KHÔNG chia train/val/test (ràng buộc 16/09, A-10): ghi công thức cũ để ai cần tự chia."""
    return "\n".join([
        "## Không chia train/val/test", "",
        "Bộ này **không** mang cột `split`/`split_group`/`label_in_train` (bỏ từ 16/09). Ba cột ấy",
        "là hàm thuần của `book` + `page`, không mất thông tin khi bỏ; người dùng tự chia theo",
        "**trang** — hai cột cạnh nhau trên cùng một trang dùng chung nét bút, chung mực, chung",
        "lần quét, chia theo cột thì mô hình học được *diện mạo trang* rồi được chấm lại trên",
        "chính trang đó.", "",
        "Công thức đã dùng cho các bản tới 25/08 (tái lập được):", "",
        "```python",
        "h = int(hashlib.md5(f'{book}|{page}'.encode()).hexdigest(), 16) % 100",
        "split = 'train' if h < 80 else 'val' if h < 90 else 'test'",
        "```", "",
        f"Khi đánh giá **cấp ký tự** hãy tự tính \"lớp chữ có mặt trong train\" trên phần train",
        f"của chính phép chia mình dùng, và tính trong phạm vi `tier == \"GOLD\"` — lọc trên cả bộ",
        f"sẽ vứt luôn {s['chu_giai_am']:,} dòng SYLLABLE có `label` rỗng."])


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
| `labels.csv` | một dòng mỗi ô, **{len(s['cot_labels'])} cột cố định**, đường dẫn ảnh tương đối |
| `labels_trace.csv` | sidecar chẩn đoán, cùng số dòng/thứ tự, khoá `image` (không cần để dùng nhãn) |
| `columns.csv` | một dòng mỗi cột trang (`book,page,column`) |
| `gold/` | ảnh crop của các ô có nhãn cấp ký tự |
| `syllable/` | ảnh crop của các ô chỉ có chú giải âm |

{_khoi_cot(s)}

{_khoi_khong_chia(s)}

## Ảnh hỏng và trang thiếu cột

- **{s['anh_hong']} ô** có ảnh trắng hoặc bị cắt mất nét (`crop_quality_flag` = `blank`/`truncated`
  trong `labels_trace.csv`). Nhãn có thể vẫn đúng; đừng chấm chiều ảnh ở các ô này.
- **{s['cot_lech']} ô trên {s['trang_cot_lech']} trang không đủ 9 cột** ({', '.join(s['ds_trang_lech']) or 'không có'}):
  ghép cột Nôm↔Quốc ngữ trên trang đó có thể đã trượt — suy từ `columns.csv`.

## 🔴 Trạng thái kiểm định

**Chưa có phép đo precision nào còn hiệu lực.** Mọi con số precision trong các bản trước
đã bị **tước tư cách bằng chứng** vì nguồn phán quyết hoá ra là máy chấm chứ không phải
người.

Nghĩa là: bộ này dùng được để **huấn luyện** và **thăm dò**, nhưng **chưa được trích dẫn
như dữ liệu đã kiểm chứng**.

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
| Người chép + niên đại chép | ĐÂY LÀ BẢN CHÉP TAY BÚT LÔNG. Nét bút mỗi người một khác, nên tự dạng phụ thuộc tay người chép, không cố định như chữ in |
| Niên đại bản được chép lại | chính tả Nôm biến đổi theo thời kỳ |
| Chất liệu + khổ giấy + DPI | quyết định kích thước ảnh crop, ảnh hưởng khả năng tái lập |
| Ai làm bản phiên âm Quốc ngữ | toàn bộ cột `syllable` bắt nguồn từ bản phiên âm in kèm — nó là tác phẩm riêng, có bản quyền riêng |

## Giấy phép bản quét

⬜ **CHƯA ĐIỀN.** Phải xác định trước khi công bố:
tình trạng bản quyền của bản quét, và điều kiện bên lưu giữ cho phép.
"""


def datasheet(s: dict) -> str:
    fl = " · ".join(f"`{k}` {v:,} ({100*v/s['dong']:.2f}%)" for k, v in sorted(s["flag"].items()))
    return f"""# Datasheet

## Động cơ
Gán nhãn tự động ở mức ký tự cho văn bản chữ Nôm **chép tay bút lông, thể hành-thảo**,
bằng cách khai thác **bản phiên âm Quốc ngữ song song in kèm** làm giám sát yếu — thay cho
việc gán tay từng chữ.

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
3. **Chữ Nôm tự tạo có thể bị hụt**: chỉ **{_pct(s['ngoai_cjk_pct'])}%** ({s['ngoai_cjk']:,} ô)
   nằm ngoài khối CJK cơ bản, so với
   **4,21%** ở ngữ liệu NomNaOCR. Chưa rõ do pipeline bóc mất bộ thủ hay do Nôm Công
   giáo thế kỷ XIX vốn chuộng dạng giản.
4. **Chưa chuẩn hoá dị thể.** Cùng một chữ có thể xuất hiện dưới nhiều mã
   (`徳`/`德`, `别`/`別`). Ứng viên đã lọc ở `docs/UNG_VIEN_CHUAN_HOA_DI_THE.csv`,
   **chưa áp dụng**.
5. **{s['anh_hong']} ô có ảnh hỏng** (`crop_quality_flag` = `blank`/`truncated` trong
   `labels_trace.csv`) vẫn nằm trong bộ — nhãn có thể đúng, ảnh thì không dùng được.
6. **{s['cot_lech']} ô nằm trên {s['trang_cot_lech']} trang không đủ 9 cột**
   ({', '.join(s['ds_trang_lech']) or 'không có'} — suy từ `columns.csv`).
   Bố cục trang luôn 9 cột, nên thiếu cột nghĩa là phép ghép cột Nôm↔Quốc ngữ trên
   trang đó có thể đã trượt một nhịp. Chỉ nêu sự việc, không kết luận nhãn sai.
7. **Không có recall.** Bộ này chỉ chứa ô đã gán được nhãn; phần bị bỏ không nằm ở đây.
8. **Không chia train/val/test.** Bộ giao nộp không mang cột `split`; ai cần thì chia theo
   **trang** (`book` + `page`) bằng công thức ghi trong README, rồi tự tính lớp chữ có mặt
   trong train của phép chia đó.

## Khuyến nghị dùng
Dùng được: huấn luyện mô hình, thăm dò, làm điểm khởi đầu để chấm tay.
**Chưa dùng được**: trích dẫn như dữ liệu đã kiểm chứng, hoặc làm chuẩn đánh giá.

*Sinh tự động từ `labels.csv` ngày {s['ngay']} · commit `{s['commit']}`*
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
                       ("NGUON_THU_TICH.md", nguon(s))):
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
