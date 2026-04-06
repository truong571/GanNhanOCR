#!/usr/bin/env python3
"""
prepare_data.py - Trích xuất và tổ chức dữ liệu Hán Nôm từ PDF

Xử lý PDF sách Nôm có cấu trúc xen kẽ:
  - Trang chẵn (PDF): Ảnh chữ Nôm viết tay (9 cột dọc)
  - Trang lẻ (PDF): Bản dịch Quốc ngữ (9 dòng đánh số)

Output:
  output_dir/
  ├── pages/                # Ảnh trang Nôm (PNG, high-res)
  │   ├── page_0012.png
  │   └── ...
  ├── transcriptions/       # Text QN, mỗi dòng = 1 cột, âm tiết cách bởi space
  │   ├── page_0012.txt
  │   └── ...
  └── manifest.json         # Index toàn bộ dataset

Usage:
  # CacThanhTruyen (text sạch, dùng trực tiếp):
  python prepare_data.py data/CacThanhTruyen2.pdf

  # SachThanhTruyen (cần re-OCR vì text gốc kém):
  python prepare_data.py data/SachThanhTruyen2.pdf --reocr

  # Nhiều file:
  python prepare_data.py data/CacThanhTruyen2.pdf data/CacThanhTruyen4.pdf --dpi 300
"""

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def remove_footnote_markers(text: str) -> str:
    """Xóa số chú thích dính liền sau chữ: 'Vít-vồ1' → 'Vít-vồ'"""
    # Số 1-2 chữ số ngay sau ký tự chữ cái (kể cả có dấu), trước dấu câu/space/cuối
    return re.sub(r"(?<=[a-zA-ZÀ-ỹ\u0300-\u036f])\d{1,2}(?=[\s.,;:!?\)\]\"\'»]|$)", "", text)


def remove_punctuation(text: str) -> str:
    """Xóa dấu câu, giữ lại chữ cái và khoảng trắng"""
    # Giữ chữ cái (kể cả tiếng Việt), dấu gạch nối, khoảng trắng
    return re.sub(r"[.,;:!?\"\'()\[\]{}«»…–—]", " ", text)


def normalize_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng"""
    return " ".join(text.split())


def split_to_syllables(text: str) -> list[str]:
    """Tách text thành danh sách âm tiết (mỗi âm tiết = 1 ký tự Nôm).

    - Tách bởi khoảng trắng trước
    - Sau đó tách dấu gạch nối trong từ: 'I-na-xu' → ['I', 'na', 'xu']
    - Bỏ phần tử rỗng (trailing hyphen: 'An-ti-' → ['An', 'ti'])
    """
    words = text.split()
    syllables = []
    for word in words:
        parts = word.split("-")
        for part in parts:
            part = part.strip()
            if part:  # bỏ chuỗi rỗng (do trailing hyphen)
                syllables.append(part)
    return syllables


def clean_ocr_artifacts(text: str) -> str:
    """Xóa ký tự nhiễu từ Tesseract OCR: |, ¬, _, °, `, ~, v.v."""
    # Xóa các ký tự OCR artifact thường gặp
    text = re.sub(r"[|¬_°`~©®™•§¶†‡]", "", text)
    # Xóa khoảng trắng thừa sau khi xóa artifact
    return normalize_whitespace(text)


def clean_line_text(text: str) -> str:
    """Pipeline làm sạch một dòng QN"""
    text = remove_footnote_markers(text)
    text = clean_ocr_artifacts(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_text_page(page: fitz.Page, dpi: int = 300, lang: str = "vie") -> str:
    """Re-OCR trang text QN bằng Tesseract (cho SachThanhTruyen).

    Render trang PDF thành ảnh rồi chạy Tesseract Vietnamese.
    Chất lượng tốt hơn nhiều so với OCR text nhúng trong PDF.
    """
    import pytesseract
    from PIL import Image
    import io

    # Render page thành ảnh
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))

    # Chạy Tesseract
    text = pytesseract.image_to_string(img, lang=lang, config="--psm 6")
    return text


def has_vietnamese_diacritics(text: str) -> bool:
    """Kiểm tra text có dấu tiếng Việt không (để phát hiện OCR kém)."""
    # Các ký tự có dấu đặc trưng tiếng Việt
    viet_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
    viet_chars |= set(c.upper() for c in viet_chars)
    text_lower = text.lower()
    count = sum(1 for c in text_lower if c in viet_chars)
    # Nếu < 2% ký tự có dấu → OCR kém
    total_alpha = sum(1 for c in text_lower if c.isalpha())
    if total_alpha == 0:
        return False
    return (count / total_alpha) > 0.05


# ---------------------------------------------------------------------------
# PDF page parsing
# ---------------------------------------------------------------------------

def is_image_page(page: fitz.Page) -> bool:
    """Phân loại trang: True nếu là trang ảnh Nôm, False nếu là trang text QN.

    Trang ảnh Nôm: text ngắn (chỉ số cột + số trang).
    Trang text QN: có nội dung text dài với dòng đánh số.

    Hoạt động cho cả CacThanhTruyen (text nhúng) và SachThanhTruyen (scan + OCR).
    """
    text = page.get_text().strip()

    # Trang text QN có dòng đánh số (1. xxx, 2. xxx) và text dài
    # Chấp nhận cả "N." và "N," (Tesseract artifact)
    has_numbered_lines = bool(re.search(r"^\d+[.,]\s", text, re.MULTILINE))

    if has_numbered_lines and len(text) > 200:
        return False  # Trang text QN

    # Trang ảnh: text ngắn (số cột + số trang, thường < 200 ký tự)
    return len(text) < 200


def extract_book_page_number(page: fitz.Page) -> int | None:
    """Trích xuất số trang sách từ nội dung trang PDF.

    CacThanhTruyen: số trang ở dòng đầu (VD: "12\\n\\t\\n 9\\t\\n8...")
    SachThanhTruyen: số trang ở dòng cuối (VD: "9\\n8\\n...1\\n!\\n12\\n")
    Text page: số trang ở dòng đầu hoặc cuối
    """
    text = page.get_text().strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None

    # Dòng đầu là số trang (CacThanhTruyen)
    match = re.match(r"^(\d+)$", lines[0])
    if match:
        num = int(match.group(1))
        if num > 9:  # Bỏ qua số cột (1-9)
            return num

    # Dòng cuối là số trang (SachThanhTruyen image page + Tesseract text page)
    match = re.match(r"^(\d+)$", lines[-1])
    if match:
        num = int(match.group(1))
        if num > 9:
            return num

    # Tìm trong vài dòng cuối (có thể có noise sau số trang)
    for line in reversed(lines[-5:]):
        match = re.match(r"^(\d+)$", line)
        if match:
            num = int(match.group(1))
            if num > 9:
                return num

    return None


def extract_nom_image(page: fitz.Page, output_path: Path, dpi: int = 300) -> dict:
    """Trích xuất ảnh Nôm từ trang PDF, lưu file PNG.

    Ưu tiên trích ảnh nhúng gốc (chất lượng cao hơn).
    Fallback: render toàn trang.

    Returns:
        dict với thông tin ảnh (width, height, dpi)
    """
    images = page.get_images()

    if images:
        # Trích ảnh nhúng gốc (giữ nguyên chất lượng)
        xref = images[0][0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        img_width = base_image["width"]
        img_height = base_image["height"]

        # Lưu file
        # Nếu ảnh gốc là JPEG, convert sang PNG để đồng nhất
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        img.save(str(output_path), "PNG")

        return {"width": img_width, "height": img_height, "source": "embedded"}
    else:
        # Fallback: render toàn trang
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        pix.save(str(output_path))

        return {"width": pix.width, "height": pix.height, "dpi": dpi, "source": "rendered"}


def parse_numbered_lines(text: str) -> dict[int, str]:
    """Parse text QN thành dict {số_dòng: nội_dung}.

    Format: "N.\\t text có thể xuống dòng\\n tiếp tục ở đây"
    Mỗi dòng đánh số tương ứng với 1 cột trong ảnh Nôm.

    Robust với Tesseract noise:
    - Chấp nhận "N." và "N," (Tesseract nhầm dấu chấm)
    - Cho phép noise (——, |, Đ) trước số dòng
    - Bỏ qua dòng chỉ chứa noise
    """
    lines = {}
    current_num = None
    current_content = []

    # Pattern: (noise tùy chọn) + số 1-9 + dấu chấm/phẩy + space + nội dung
    line_start_pattern = re.compile(r"^[^a-zA-ZÀ-ỹ]*?(\d)[.,]\s+(.*)")

    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue

        match = line_start_pattern.match(stripped)
        if match:
            num = int(match.group(1))
            if 1 <= num <= 9:
                # Lưu dòng trước
                if current_num is not None:
                    lines[current_num] = " ".join(current_content)
                current_num = num
                current_content = [match.group(2)]
                continue

        # Dòng tiếp tục (wrapped) hoặc noise
        if current_num is not None:
            # Bỏ dòng chỉ chứa noise (< 2 ký tự chữ cái)
            alpha_count = sum(1 for c in stripped if c.isalpha())
            if alpha_count >= 2:
                current_content.append(stripped)

    # Lưu dòng cuối
    if current_num is not None:
        lines[current_num] = " ".join(current_content)

    return lines


def extract_quocngu_text(page: fitz.Page) -> tuple[int | None, dict[int, str]]:
    """Trích xuất text QN từ trang PDF.

    Returns:
        (book_page_number, {line_num: raw_text})
    """
    text = page.get_text()
    book_page = extract_book_page_number(page)

    # Bỏ dòng đầu (số trang) trước khi parse
    text_without_page_num = re.sub(r"^\d+\s*\n", "", text.strip(), count=1)
    lines = parse_numbered_lines(text_without_page_num)

    return book_page, lines


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300,
    reocr: bool = False,
    ocr_lang: str = "vie",
    verbose: bool = True,
) -> list[dict]:
    """Xử lý 1 file PDF, trích xuất cặp (ảnh Nôm, text QN).

    Args:
        reocr: True để re-OCR trang text bằng Tesseract (cho SachThanhTruyen).
               False để dùng text nhúng trong PDF (cho CacThanhTruyen).
               "auto" sẽ tự phát hiện khi text kém.
        ocr_lang: Ngôn ngữ Tesseract (mặc định: "vie")

    Returns:
        List[dict] - thông tin từng trang đã xử lý
    """
    doc = fitz.open(str(pdf_path))
    pdf_name = pdf_path.stem

    pages_dir = output_dir / "pages"
    trans_dir = output_dir / "transcriptions"
    pages_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)

    results = []
    page_idx = 0
    total_pages = doc.page_count
    reocr_count = 0

    if verbose:
        mode = "re-OCR (Tesseract)" if reocr else "text nhúng PDF"
        print(f"\nXử lý: {pdf_path.name} ({total_pages} trang PDF)")
        print(f"Output: {output_dir}/")
        print(f"Chế độ text: {mode}")
        print("-" * 60)

    while page_idx < total_pages:
        page = doc[page_idx]

        if not is_image_page(page):
            if verbose:
                print(f"  [SKIP] PDF trang {page_idx}: không phải trang ảnh Nôm")
            page_idx += 1
            continue

        # Trang hiện tại là ảnh Nôm
        image_page = page
        book_page_from_image = extract_book_page_number(image_page)

        # Trang tiếp theo phải là text QN
        text_page_idx = page_idx + 1
        if text_page_idx >= total_pages:
            if verbose:
                print(f"  [WARN] PDF trang {page_idx}: trang ảnh cuối, không có trang text tương ứng")
            page_idx += 1
            continue

        text_page = doc[text_page_idx]

        if is_image_page(text_page):
            if verbose:
                print(f"  [WARN] PDF trang {page_idx}: trang tiếp theo ({text_page_idx}) cũng là ảnh, bỏ qua")
            page_idx += 1
            continue

        # Trích xuất text QN
        used_reocr = False
        embedded_text = text_page.get_text()

        if reocr or (reocr == "auto" and not has_vietnamese_diacritics(embedded_text)):
            # Re-OCR bằng Tesseract
            ocr_text = ocr_text_page(text_page, dpi=dpi, lang=ocr_lang)

            # Lấy số trang từ embedded text trước (tin cậy hơn)
            book_page_from_text = extract_book_page_number(text_page)

            # Nếu không có, thử tìm trong Tesseract output
            if book_page_from_text is None:
                # Tìm số trang ở cuối OCR output
                ocr_lines = [l.strip() for l in ocr_text.strip().split("\n") if l.strip()]
                for line in reversed(ocr_lines[-3:]):
                    m = re.match(r"^(\d+)$", line)
                    if m and int(m.group(1)) > 9:
                        book_page_from_text = int(m.group(1))
                        break

            # Xóa noise: số trang ở đầu/cuối
            ocr_text = ocr_text.strip()
            ocr_text = re.sub(r"\n\d+\s*$", "", ocr_text)
            ocr_lines_clean = ocr_text.strip().split("\n")
            if ocr_lines_clean and re.match(r"^\d+$", ocr_lines_clean[0].strip()):
                first_num = int(ocr_lines_clean[0].strip())
                if first_num > 9:
                    ocr_text = "\n".join(ocr_lines_clean[1:])

            raw_lines = parse_numbered_lines(ocr_text)
            used_reocr = True
            reocr_count += 1
        else:
            book_page_from_text, raw_lines = extract_quocngu_text(text_page)

        # Xác định số trang sách
        # Ưu tiên: image page number > (text page number - 1) > fallback
        if book_page_from_image:
            book_page = book_page_from_image
        elif book_page_from_text:
            # Trang text QN luôn = trang ảnh Nôm + 1 (chẵn/lẻ)
            book_page = book_page_from_text - 1
        else:
            # Fallback: tính từ vị trí trong PDF
            # Mỗi cặp = 2 trang PDF, trang đầu sách thường ≈ 10-12
            book_page = page_idx + 10
            if verbose:
                print(f"  [WARN] Không xác định được số trang sách, dùng fallback: {book_page}")

        # Lưu ảnh Nôm
        image_filename = f"page_{book_page:04d}.png"
        image_path = pages_dir / image_filename
        image_info = extract_nom_image(image_page, image_path, dpi)

        # Làm sạch text và tách âm tiết
        columns = []
        for line_num in sorted(raw_lines.keys()):
            raw_text = raw_lines[line_num]
            cleaned = clean_line_text(raw_text)
            syllables = split_to_syllables(cleaned)
            columns.append(
                {
                    "column": line_num,
                    "raw_text": raw_text,
                    "cleaned_text": cleaned,
                    "syllables": syllables,
                    "num_syllables": len(syllables),
                }
            )

        # Lưu transcription (1 dòng = 1 cột, âm tiết cách space)
        trans_filename = f"page_{book_page:04d}.txt"
        trans_path = trans_dir / trans_filename
        with open(trans_path, "w", encoding="utf-8") as f:
            for col in columns:
                f.write(" ".join(col["syllables"]) + "\n")

        # Lưu thêm bản gốc chi tiết (JSON)
        detail_filename = f"page_{book_page:04d}.json"
        detail_path = trans_dir / detail_filename
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(
                {"book_page": book_page, "columns": columns},
                f,
                ensure_ascii=False,
                indent=2,
            )

        total_syllables = sum(c["num_syllables"] for c in columns)

        text_source = "tesseract" if used_reocr else "embedded"
        page_result = {
            "book_page": book_page,
            "source_pdf": pdf_name,
            "image_file": f"pages/{image_filename}",
            "transcription_file": f"transcriptions/{trans_filename}",
            "detail_file": f"transcriptions/{detail_filename}",
            "num_columns": len(columns),
            "total_syllables": total_syllables,
            "syllable_counts": [c["num_syllables"] for c in columns],
            "image_info": image_info,
            "text_source": text_source,
        }
        results.append(page_result)

        if verbose:
            syllable_str = ", ".join(str(c["num_syllables"]) for c in columns)
            ocr_tag = " [OCR]" if used_reocr else ""
            print(
                f"  Trang {book_page:4d}: {len(columns)} cột, "
                f"{total_syllables} âm tiết [{syllable_str}]{ocr_tag}"
            )

        page_idx = text_page_idx + 1  # Nhảy qua cả 2 trang

    doc.close()

    if verbose and reocr_count > 0:
        print(f"\n  Re-OCR (Tesseract): {reocr_count} trang")

    return results


def validate_results(results: list[dict], verbose: bool = True) -> dict:
    """Kiểm tra tính nhất quán của kết quả."""
    stats = {
        "total_pages": len(results),
        "total_columns": sum(r["num_columns"] for r in results),
        "total_syllables": sum(r["total_syllables"] for r in results),
        "pages_with_9_columns": sum(1 for r in results if r["num_columns"] == 9),
        "pages_not_9_columns": [],
        "min_syllables_per_column": float("inf"),
        "max_syllables_per_column": 0,
    }

    for r in results:
        if r["num_columns"] != 9:
            stats["pages_not_9_columns"].append(
                {"page": r["book_page"], "columns": r["num_columns"]}
            )
        for count in r["syllable_counts"]:
            stats["min_syllables_per_column"] = min(stats["min_syllables_per_column"], count)
            stats["max_syllables_per_column"] = max(stats["max_syllables_per_column"], count)

    if stats["total_pages"] == 0:
        stats["min_syllables_per_column"] = 0

    if verbose:
        print("\n" + "=" * 60)
        print("THỐNG KÊ")
        print("=" * 60)
        print(f"  Tổng số trang Nôm  : {stats['total_pages']}")
        print(f"  Tổng số cột        : {stats['total_columns']}")
        print(f"  Tổng số âm tiết    : {stats['total_syllables']}")
        print(f"  Trang có đủ 9 cột  : {stats['pages_with_9_columns']}")
        if stats["pages_not_9_columns"]:
            print(f"  [WARN] Trang không đủ 9 cột:")
            for p in stats["pages_not_9_columns"]:
                print(f"    - Trang {p['page']}: {p['columns']} cột")
        if stats["total_pages"] > 0:
            print(f"  Âm tiết/cột (min)  : {stats['min_syllables_per_column']}")
            print(f"  Âm tiết/cột (max)  : {stats['max_syllables_per_column']}")
            avg = stats["total_syllables"] / max(stats["total_columns"], 1)
            print(f"  Âm tiết/cột (TB)   : {avg:.1f}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Trích xuất dữ liệu Hán Nôm từ PDF sách",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python prepare_data.py data/CacThanhTruyen2.pdf
  python prepare_data.py data/CacThanhTruyen2.pdf --output-dir output/vol2
  python prepare_data.py data/CacThanhTruyen2.pdf data/CacThanhTruyen4.pdf
  python prepare_data.py data/CacThanhTruyen2.pdf --dpi 400
        """,
    )
    parser.add_argument("pdf_files", nargs="+", type=Path, help="Đường dẫn file PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục output (mặc định: data/prepared/<tên_pdf>/)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Độ phân giải ảnh (mặc định: 300)")
    parser.add_argument(
        "--reocr",
        action="store_true",
        help="Re-OCR trang text bằng Tesseract (cần cho SachThanhTruyen)",
    )
    parser.add_argument(
        "--ocr-lang",
        default="vie",
        help="Ngôn ngữ Tesseract (mặc định: vie)",
    )
    parser.add_argument("--quiet", action="store_true", help="Không hiển thị chi tiết")

    args = parser.parse_args()

    all_results = []

    for pdf_path in args.pdf_files:
        if not pdf_path.exists():
            print(f"[ERROR] Không tìm thấy file: {pdf_path}", file=sys.stderr)
            continue

        # Xác định thư mục output
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = Path("data/prepared") / pdf_path.stem

        # Xử lý PDF
        results = process_pdf(
            pdf_path,
            output_dir,
            dpi=args.dpi,
            reocr=args.reocr,
            ocr_lang=args.ocr_lang,
            verbose=not args.quiet,
        )
        all_results.extend(results)

        # Validate
        stats = validate_results(results, verbose=not args.quiet)

        # Lưu manifest
        manifest = {
            "source_pdf": str(pdf_path),
            "dpi": args.dpi,
            "reocr": args.reocr,
            "ocr_lang": args.ocr_lang if args.reocr else None,
            "stats": stats,
            "pages": results,
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        if not args.quiet:
            print(f"\n  Manifest: {manifest_path}")

    # Tổng kết nếu xử lý nhiều file
    if len(args.pdf_files) > 1 and not args.quiet:
        print("\n" + "=" * 60)
        print("TỔNG KẾT TẤT CẢ")
        print("=" * 60)
        total_stats = validate_results(all_results, verbose=False)
        print(f"  Tổng file PDF      : {len(args.pdf_files)}")
        print(f"  Tổng trang Nôm     : {total_stats['total_pages']}")
        print(f"  Tổng âm tiết       : {total_stats['total_syllables']}")


if __name__ == "__main__":
    main()
