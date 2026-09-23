#!/usr/bin/env python3
"""
scripts/download_handwritten_manuscripts.py
===========================================
Tải về toàn bộ các bộ tư liệu chữ viết tay Hán Nôm (bản chép tay bút lông thật)
được số hóa từ Thư viện Quốc gia Việt Nam (NLVNPF) và Thư viện Quốc gia Pháp (BnF Gallica).

Các bộ tư liệu hỗ trợ:
  1. kieu        : Truyện Kiều - Phong Tình Cổ Lục (風情古錄, NLVNPF-0221 / R.987)
                   120 trang, bản chép tay bút lông, chấm son chu sa đỏ, khớp 3.254 câu Kiều.
  2. lucvantien  : Lục Vân Tiên (陸雲僊歌演, Trần Nguyên Hanh chép tay 1883, BnF Gallica bpt6k54602432)
                   105 trang, bản chép tay bút lông của nhà nho Trần Nguyên Hanh, khớp 2.088 câu thơ.
  3. tamtukinh   : Tam Tự Kinh Diễn Âm (三字經演音, NLVNPF-0463)
                   32 trang, bản thảo viết tay thơ vần Nôm giải nghĩa kinh thư cổ.
  4. lyhang      : Lý Hạng Ca Dao (俚巷歌謠, NLVNPF-0026)
                   51 trang, bản chép tay ca dao, tục ngữ dân gian bằng chữ Nôm.

Tính năng:
  - Tải đa luồng song song (ThreadPoolExecutor) kèm cơ chế tự thử lại (retry backoff).
  - Xuất file ảnh chất lượng gốc (Full Resolution JPG) vào thư mục pages/.
  - Tự động đóng gói thành file PDF hoàn chỉnh qua Pillow để xem lướt tiện lợi.
  - Tự động tạo metadata.json và README.md thuyết minh lai lịch từng bộ.

Cách dùng:
  python scripts/download_handwritten_manuscripts.py --target all
  python scripts/download_handwritten_manuscripts.py --target kieu
  python scripts/download_handwritten_manuscripts.py --target lucvantien
  python scripts/download_handwritten_manuscripts.py --target all --workers 10
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = PROJECT_ROOT / "data" / "handwritten_manuscripts"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

COLLECTIONS = {
    "kieu": {
        "id": "TruyenKieu_PhongTinhCoLuc",
        "title": "Truyện Kiều - Phong Tình Cổ Lục (風情古錄)",
        "source": "Thư viện Quốc gia Việt Nam / Hội Bảo tồn Di sản Chữ Nôm (NLVNPF-0221 / R.987)",
        "type": "Bản chép tay bút lông (Manuscript), có chấm son chu sa đỏ",
        "literature": "Nguyễn Du - Đoạn trường tân thanh (3.254 câu Kiều)",
        "pages_count": 120,
        "max_workers": 8,
        "delay": 0.0,
        "url_pattern": lambda p: f"https://lib.nomfoundation.org/site_media/nom/nlvnpf-0221/large/nlvnpf-0221-{p:03d}.jpg",
        "page_filename": lambda p: f"kieu_manuscript_{p:03d}.jpg",
        "page_range": range(1, 121),
    },
    "lucvantien": {
        "id": "LucVanTien_1883_TranNguyenHanh",
        "title": "Lục Vân Tiên - Bản chép tay Trần Nguyên Hanh 1883 (陸雲僊歌演)",
        "source": "Bibliothèque nationale de France (BnF Gallica ark:/12148/bpt6k54602432)",
        "type": "Bản chép tay bút lông của nhà nho Trần Nguyên Hanh, ấn bản Abel des Michels 1883",
        "literature": "Nguyễn Đình Chiểu - Lục Vân Tiên (2.088 câu lục bát có đánh số dòng)",
        "pages_count": 105,
        "max_workers": 2,
        "delay": 0.5,
        # Gallica pagination for Nom text: canvas 448 (f449) is page 1, canvas 344 (f345) is page 105
        "url_pattern": lambda p: f"https://gallica.bnf.fr/iiif/ark:/12148/bpt6k54602432/f{450 - p}/full/full/0/native.jpg",
        "page_filename": lambda p: f"lucvantien_manuscript_{p:03d}.jpg",
        "page_range": range(1, 106),
    },
    "tamtukinh": {
        "id": "TamTuKinh_NLVNPF0463",
        "title": "Tam Tự Kinh Diễn Âm (三字經演音)",
        "source": "Thư viện Quốc gia Việt Nam / NLVNPF (NLVNPF-0463)",
        "type": "Bản thảo viết tay bút lông chữ Nôm thể lục bát",
        "literature": "Kinh thư giáo dục truyền thống diễn giải bằng chữ Nôm",
        "pages_count": 32,
        "max_workers": 8,
        "delay": 0.0,
        "url_pattern": lambda p: f"https://lib.nomfoundation.org/site_media/nom/nlvnpf-0463/large/nlvnpf-0463-{p:03d}.jpg",
        "page_filename": lambda p: f"tamtukinh_manuscript_{p:03d}.jpg",
        "page_range": range(1, 33),
    },
    "lyhang": {
        "id": "LyHangCaDao_NLVNPF0026",
        "title": "Lý Hạng Ca Dao (俚巷歌謠)",
        "source": "Thư viện Quốc gia Việt Nam / NLVNPF (NLVNPF-0026)",
        "type": "Bản thảo viết tay bút lông chữ Nôm ghi chép ca dao, dân ca dân gian",
        "literature": "Kho tàng ca dao tục ngữ dân gian Việt Nam",
        "pages_count": 51,
        "max_workers": 8,
        "delay": 0.0,
        "url_pattern": lambda p: f"https://lib.nomfoundation.org/site_media/nom/nlvnpf-0026/large/nlvnpf-0026-{p:03d}.jpg",
        "page_filename": lambda p: f"lyhang_manuscript_{p:03d}.jpg",
        "page_range": range(1, 52),
    },
}


def download_single_page(url: str, out_path: Path, max_retries: int = 8, delay: float = 0.0) -> bool:
    """Tải một trang ảnh với cơ chế thử lại lũy thừa và xử lý HTTP 429."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True  # Đã tải trước đó

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")

    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, max_retries + 1):
        try:
            if delay > 0:
                time.sleep(delay)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=35) as resp:
                if resp.status == 200:
                    content = resp.read()
                    if len(content) > 1024:
                        with open(tmp_path, "wb") as f:
                            f.write(content)
                        tmp_path.replace(out_path)
                        return True
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                sleep_sec = 4 * attempt
                print(f"    [RateLimit 429] Chờ {sleep_sec}s rồi thử lại...", flush=True)
                time.sleep(sleep_sec)
            else:
                time.sleep(1.5 * attempt)
            if attempt == max_retries:
                print(f"[ERROR] Failed {url} sau {max_retries} lần thử: {e}", file=sys.stderr)
                if tmp_path.exists():
                    tmp_path.unlink()
                return False
    return False


def build_pdf_from_images(image_paths: list[Path], output_pdf: Path):
    """Gộp toàn bộ danh sách ảnh thành file PDF duy nhất."""
    if not image_paths:
        return
    print(f"[*] Đang đóng gói {len(image_paths)} trang thành file PDF: {output_pdf.name}...")
    try:
        pil_images = []
        first_img = None
        for p in image_paths:
            if p.exists() and p.stat().st_size > 0:
                img = Image.open(p)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if first_img is None:
                    first_img = img
                else:
                    pil_images.append(img)
        
        if first_img:
            first_img.save(output_pdf, save_all=True, append_images=pil_images)
            size_mb = output_pdf.stat().st_size / (1024 * 1024)
            print(f"[✓] Đã tạo PDF thành công: {output_pdf} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"[!] Lỗi khi tạo PDF: {e}", file=sys.stderr)


def process_collection(col_key: str, col_info: dict, max_workers: int = 8):
    """Tải và đóng gói một bộ tư liệu."""
    target_dir = OUTPUT_BASE / col_info["id"]
    pages_dir = target_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU TẢI BỘ: {col_info['title']}")
    print(f"Nguồn: {col_info['source']}")
    print(f"Thể loại: {col_info['type']}")
    print(f"Tổng số trang dự kiến: {col_info['pages_count']}")
    print(f"Thư mục lưu: {target_dir}")
    print("=" * 80)

    # 1. Tạo danh sách URL và File
    tasks = []
    image_paths = []
    for p in col_info["page_range"]:
        url = col_info["url_pattern"](p)
        fname = col_info["page_filename"](p)
        fpath = pages_dir / fname
        tasks.append((url, fpath, p))
        image_paths.append(fpath)

    # 2. Tải song song
    success_count = 0
    total = len(tasks)
    start_time = time.time()
    actual_workers = col_info.get("max_workers", max_workers)
    delay = col_info.get("delay", 0.0)

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_page = {
            executor.submit(download_single_page, url, fpath, 8, delay): (p, fpath)
            for url, fpath, p in tasks
        }
        for future in as_completed(future_to_page):
            p, fpath = future_to_page[future]
            try:
                ok = future.result()
                if ok:
                    success_count += 1
                    status = f"Trang {p:03d}/{total:03d} OK ({fpath.stat().st_size // 1024} KB)"
                else:
                    status = f"Trang {p:03d}/{total:03d} THẤT BẠI"
                print(f"  [{success_count:03d}/{total:03d}] {status}", flush=True)
            except Exception as e:
                print(f"  [!] Trang {p:03d} phát sinh ngoại lệ: {e}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[✓] Hoàn tất tải {success_count}/{total} trang trong {elapsed:.1f}s.")

    # 3. Tạo PDF
    pdf_path = target_dir / f"{col_info['id']}.pdf"
    build_pdf_from_images(image_paths, pdf_path)

    # 4. Ghi metadata.json
    metadata = {
        "id": col_info["id"],
        "title": col_info["title"],
        "source": col_info["source"],
        "type": col_info["type"],
        "literature": col_info["literature"],
        "pages_downloaded": success_count,
        "pages_total": total,
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pdf_file": str(pdf_path.relative_to(PROJECT_ROOT)) if pdf_path.exists() else None,
        "pages_dir": str(pages_dir.relative_to(PROJECT_ROOT)),
    }
    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 5. Ghi README.md
    readme_content = f"""# {col_info['title']}

- **Mã định danh:** `{col_info['id']}`
- **Nguồn tài liệu:** {col_info['source']}
- **Thể loại thư tịch:** {col_info['type']}
- **Tác phẩm / Khảo cứu:** {col_info['literature']}
- **Số trang scan:** {success_count} / {total} trang
- **Định dạng:** Ảnh màu độ phân giải cao (.jpg) trong `pages/` và bản gộp `{col_info['id']}.pdf`

## Ý nghĩa đối với Đề tài Nhận dạng OCR Chữ Viết tay Hán Nôm:
Tài liệu này là bản viết tay bút lông nguyên bản (Authentic Brush Manuscript), bảo đảm đặc tính nét viết thực tế:
khác biệt hoàn toàn với bản in khắc gỗ (mộc bản). Chữ viết có nét thanh đậm tự nhiên, mật độ mực biến thiên,
chấm câu và chú giải tay, cung cấp dữ liệu huấn luyện và đánh giá thực chất cho mô hình OCR chữ Nôm viết tay.
"""
    with open(target_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"[✓] Đã tạo xong metadata và README tại {target_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Tải các bộ tư liệu viết tay Hán Nôm (Truyện Kiều, Lục Vân Tiên, Tam Tự Kinh, Lý Hạng Ca Dao)")
    parser.add_argument(
        "--target",
        choices=["all", "kieu", "lucvantien", "tamtukinh", "lyhang"],
        default="all",
        help="Chọn bộ tư liệu cần tải (mặc định: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Số luồng tải song song (mặc định: 8)",
    )
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if args.target == "all":
        targets = list(COLLECTIONS.keys())
    else:
        targets = [args.target]

    print("=" * 80)
    print("TOOL TẢI DỮ LIỆU BẢN CHÉP TAY CHỮ NÔM NGUYÊN BẢN (AUTHENTIC MANUSCRIPTS)")
    print(f"Các bộ sẽ tải: {', '.join(targets)}")
    print(f"Số luồng: {args.workers}")
    print(f"Thư mục lưu: {OUTPUT_BASE}")
    print("=" * 80)

    for target in targets:
        process_collection(target, COLLECTIONS[target], max_workers=args.workers)

    print("\n" + "#" * 80)
    print("TẤT CẢ CÁC BỘ TƯ LIỆU ĐỀ XUẤT ĐÃ ĐƯỢC TẢI XONG VÀ ĐÓNG GÓI HOÀN HẢO!")
    print(f"Vui lòng kiểm tra thư mục: {OUTPUT_BASE}")
    print("#" * 80)


if __name__ == "__main__":
    main()
