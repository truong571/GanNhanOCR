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
    # ĐO chia tách từ CHÍNH DỮ LIỆU, không khẳng định theo thiết kế. Tài liệu từng nói
    # "rời nhau theo TRANG" trong khi đo được 360/444 trang nằm ở hai phía — vì nó chép
    # lại ý định của mã thay vì đọc tệp. Bộ sinh tài liệu KHÔNG được phép nói điều nó
    # chưa đo.
    _pg = collections.defaultdict(set)
    for r in rows:
        _pg[(r.get("book"), r.get("page"))].add(r.get("split", ""))
    n_leak = sum(1 for v in _pg.values() if len(v) > 1)
    unseen = sum(1 for r in rows if r.get("label_in_train") == "0")
    # ĐO tỷ lệ chữ ngoài khối CJK cơ bản. Trước 2026-08-25 con số này là chuỗi ghim cứng
    # "1,63%" nằm lọt giữa một f-string mà mọi số quanh nó đều động — đo lại được 2,25%.
    _ngoai = sum(1 for r in char if not ("\u4e00" <= r["label"] <= "\u9fff"))
    _lech = sum(1 for r in rows if r.get("page_cot_lech") == "1")
    _trang_lech = len({(r.get("book"), r.get("page")) for r in rows
                       if r.get("page_cot_lech") == "1"})
    co_cot_moi = "label_in_train" in (rows[0] if rows else {})
    return {
        "dong": len(rows), "nhan_ky_tu": len(char), "chu_giai_am": len(syl),
        "lop_chu": len({r["label"] for r in char}),
        "sach": sorted({r["book"] for r in rows}),
        "trang": len({(r["book"], r["page"]) for r in rows}),
        "cot": len({(r["book"], r["page"], r["column"]) for r in rows}),
        "am_qn": len({(r.get("syllable") or "").lower() for r in rows if r.get("syllable")}),
        "anh_hong": q.get("blank", 0) + q.get("truncated", 0),
        "flag": {k: v for k, v in q.items() if k},
        "cot_lech": _lech, "trang_cot_lech": _trang_lech,
        "ngoai_cjk": _ngoai,
        "ngoai_cjk_pct": (100 * _ngoai / len(char)) if char else 0.0,
        "n_leak": n_leak, "n_trang_split": len(_pg), "unseen": unseen,
        "co_cot_moi": co_cot_moi,
        "commit": subprocess.run(["git", "log", "-1", "--format=%h", "--", str(labels)],
                                 cwd=REPO, capture_output=True, text=True).stdout.strip() or "?",
        "ngay": datetime.date.fromtimestamp(labels.stat().st_mtime).isoformat(),
    }


def _pct(x: float) -> str:
    """Dấu thập phân kiểu Việt — để không đứng cạnh '4,21%' mà viết '2.25%'."""
    return f"{x:.2f}".replace(".", ",")


def _khoi_chia_tach(s: dict) -> str:
    """Mô tả chia tách theo SỐ ĐO, không theo ý định của mã."""
    if s["n_leak"] == 0:
        t = ["## Chia tách — rời nhau theo TRANG", "",
             f"Đo trên chính `labels.csv`: **0/{s['n_trang_split']} trang** nằm ở hai phía.", "",
             "Chọn mức trang chứ không phải cột vì hai cột cạnh nhau trên cùng một trang dùng",
             "chung nét bút, chung mực, chung lần quét — chia theo cột thì mô hình học được",
             "*diện mạo trang* rồi được chấm lại trên chính trang đó."]
        if s["co_cot_moi"]:
            t += ["", f"**Hệ quả phải biết:** {s['unseen']:,} ô có lớp chữ **không xuất hiện trong",
                  "`train`** — hệ quả của việc không bóp méo chia tách. Cột **`label_in_train`**",
                  "đánh dấu chúng: `1` có mặt, `0` không, **rỗng** với dòng tầng SYLLABLE (chúng",
                  "không có nhãn cấp ký tự nên câu hỏi không áp dụng).", "",
                  f"Khi đánh giá **cấp ký tự**, bỏ các ô `label_in_train == 0` — nếu không, "
                  f"{s['unseen']:,} ô đó",
                  "bị tính sai 100% dù mô hình chưa từng có cơ hội học lớp chữ ấy.", "",
                  "> ⚠️ Đừng viết `df[df.label_in_train == \"1\"]` để lọc cả bộ: điều kiện đó vứt",
                  f"> luôn {s['chu_giai_am']:,} dòng SYLLABLE có ô rỗng, tức {s['chu_giai_am'] + s['unseen']:,} dòng",
                  f"> chứ không phải {s['unseen']:,}. Lọc trong phạm vi `label_level == \"char\"`."]
        return "\n".join(t)
    return "\n".join([
        "## 🔴 Chia tách CÓ RÒ RỈ theo trang", "",
        f"Đo trên chính `labels.csv`: **{s['n_leak']}/{s['n_trang_split']} trang** có ô nằm ở",
        "**hai phía khác nhau**. Hai cột cạnh nhau trên cùng một trang dùng chung nét bút,",
        "chung mực, chung lần quét, nên mô hình huấn luyện trên `train` học được *diện mạo",
        "trang* rồi được chấm lại trên chính trang đó.", "",
        "**Mọi chỉ số đo bằng `split` sẵn có là CẬN TRÊN**, không phải hiệu năng thật trên",
        "trang chưa từng thấy. Muốn đánh giá trung thực thì tự chia lại theo `book` + `page`."])


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
| `split` / `split_group` | {"**rời nhau theo TRANG**" if s['n_leak']==0 else "🔴 **CÓ RÒ RỈ** — xem dưới"} |
{"| `label_in_train` | `0` = lớp chữ này **không có mặt trong `train`**. Đánh giá phải lọc theo cột này |" if s['co_cot_moi'] else ""}
{f"| `page_cot_lech` | `1` = trang này không đủ 9 cột ({s['cot_lech']} ô / {s['trang_cot_lech']} trang). Ghép cột Nôm↔Quốc ngữ có thể đã trượt |" if s['cot_lech'] else ""}

{_khoi_chia_tach(s)}

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
5. **{s['anh_hong']} ô có ảnh hỏng** (`usable_image=0`) vẫn nằm trong bộ — nhãn có thể
   đúng, ảnh thì không dùng được.
6. **{s['cot_lech']} ô nằm trên {s['trang_cot_lech']} trang không đủ 9 cột** (`page_cot_lech=1`).
   Bố cục trang luôn 9 cột, nên thiếu cột nghĩa là phép ghép cột Nôm↔Quốc ngữ trên
   trang đó có thể đã trượt một nhịp. Cờ chỉ nêu sự việc, không kết luận nhãn sai.

7. **Không có recall.** Bộ này chỉ chứa ô đã gán được nhãn; phần bị bỏ không nằm ở đây.
8. **Chia tách: {s['n_leak']}/{s['n_trang_split']} trang nằm ở hai phía.** {s['unseen']:,} ô có
   lớp chữ không mặt trong `train` — hệ quả của việc chia tách trung thực theo trang.
   Lọc bằng `label_in_train` khi đánh giá (cột này RỖNG ở tầng SYLLABLE, xem README).

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
