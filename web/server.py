#!/usr/bin/env python3
"""
web/server.py — GanNhanOCR Web Presentation Server
Phục vụ trình diễn luận văn Thạc sĩ: Hệ thống gán nhãn tự động văn bản Hán Nôm cổ.
Sử dụng chuẩn thư viện Python (không cần cài thêm dependencies).
"""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent

# Cấu hình danh mục sách và vị trí dữ liệu
BOOKS_CONFIG = {
    "LucVanTien1883": {
        "id": "LucVanTien1883",
        "title": "Lục Vân Tiên (1883)",
        "subtitle": "Bản thạch bản khắc in năm Quý Mùi (1883)",
        "author": "Nguyễn Đình Chiểu",
        "layout": "lithograph",
        "layout_desc": "Thơ Lục Bát (10 cột/trang, 14 âm/cột)",
        "n_columns": 10,
        "n_pages": 105,
        "labels_path": REPO_ROOT / "dataset" / "LucVanTien1883" / "labels.csv",
        "pages_dir": REPO_ROOT / "prepared" / "LucVanTien1883" / "pages",
        "crops_root": REPO_ROOT / "dataset" / "LucVanTien1883",
        "sample_pages": ["page_0001", "page_0002", "page_0003", "page_0004", "page_0005"],
        "default_page": "page_0002",
    },
    "KimVanKieu1884": {
        "id": "KimVanKieu1884",
        "title": "Kim Vân Kiều (1884)",
        "subtitle": "Bản thạch bản Kim Vân Kiều Tân Truyện (Giáp Thân 1884)",
        "author": "Nguyễn Du",
        "layout": "lithograph",
        "layout_desc": "Thơ Lục Bát (10 cột/trang, 14 âm/cột - Chốt B1')",
        "n_columns": 10,
        "n_pages": 163,
        "labels_path": REPO_ROOT / "dataset" / "KimVanKieu1884" / "labels.csv",
        "pages_dir": REPO_ROOT / "prepared" / "KimVanKieu1884" / "pages",
        "crops_root": REPO_ROOT / "dataset" / "KimVanKieu1884",
        "sample_pages": ["page_0001", "page_0002", "page_0003", "page_0004", "page_0005"],
        "default_page": "page_0002",
    },
    "Chrestomathie1872": {
        "id": "Chrestomathie1872",
        "title": "Chrestomathie Annamite (1872)",
        "subtitle": "Bản giáo trình văn xuôi tiếng An Nam (1872)",
        "author": "Trương Vĩnh Ký / E. Luro",
        "layout": "prose",
        "layout_desc": "Văn xuôi lịch sử (7 cột/trang, số chữ linh hoạt)",
        "n_columns": 7,
        "n_pages": 65,
        "labels_path": REPO_ROOT / "dataset" / "Chrestomathie1872" / "labels.csv",
        "pages_dir": REPO_ROOT / "prepared" / "Chrestomathie1872" / "pages",
        "crops_root": REPO_ROOT / "dataset" / "Chrestomathie1872",
        "sample_pages": ["page_0001", "page_0002", "page_0003", "page_0004", "page_0005"],
        "default_page": "page_0002",
    },
    "SachThanhTruyen": {
        "id": "SachThanhTruyen",
        "title": "Sách Thánh Truyện (STT 2, 4, 11)",
        "subtitle": "Bản khắc gỗ Nôm tôn giáo cổ (TK XVII-XIX)",
        "author": "Khuyết danh / Dòng Tên",
        "layout": "woodblock",
        "layout_desc": "Bản khắc gỗ cổ (9 cột/trang, nét chữ đao khắc sâu)",
        "n_columns": 9,
        "n_pages": 300,
        "labels_path": REPO_ROOT / "dataset" / "labels.csv",
        "pages_dir": REPO_ROOT / "prepared" / "SachThanhTruyen11" / "pages",
        "crops_root": REPO_ROOT / "dataset",
        "sample_pages": ["page_0010", "page_0012", "page_0014", "page_0016"],
        "default_page": "page_0010",
    },
}

# Cache bộ nhãn trong RAM để tìm kiếm và render trang phản hồi < 5ms
LABELS_CACHE: dict[str, list[dict]] = {}
PAGE_INDEX: dict[str, dict[str, list[dict]]] = {}


def load_dataset_caches():
    """Tải trước bảng nhãn vào RAM để tăng tốc tối đa khi Hội đồng soi trang và tìm kiếm."""
    global LABELS_CACHE, PAGE_INDEX
    print("[server] Đang nạp dữ liệu nhãn từ dataset/ vào bộ nhớ...")
    for book_id, cfg in BOOKS_CONFIG.items():
        lp = cfg["labels_path"]
        if not lp.exists():
            print(f"  [!] Không thấy {lp}")
            continue
        rows = []
        pages: dict[str, list[dict]] = {}
        with open(lp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                pg = r.get("page", "").strip()
                if pg not in pages:
                    pages[pg] = []
                pages[pg].append(r)
        LABELS_CACHE[book_id] = rows
        PAGE_INDEX[book_id] = pages
        print(f"  [OK] {book_id}: {len(rows):,} dòng nhãn ({len(pages)} trang)")


class GanNhanHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        # Thêm CORS và Cache-Control cho API nội bộ
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_HEAD(self):
        url = urlparse(self.path)
        path = url.path
        if path.startswith("/api/") or path.startswith("/crops/") or path.startswith("/page_scans/"):
            self.do_GET()
        else:
            super().do_HEAD()

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        query = parse_qs(url.query)

        if path.startswith("/api/"):
            self.handle_api(path, query)
        elif path.startswith("/crops/"):
            self.serve_crop_image(path)
        elif path.startswith("/page_scans/"):
            self.serve_page_scan(path)
        else:
            # Phục vụ file tĩnh từ thư mục web/
            super().do_GET()

    def send_json(self, data: dict | list, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def handle_api(self, path: str, query: dict):
        if path == "/api/stats":
            self.get_stats()
        elif path == "/api/books":
            self.get_books()
        elif path == "/api/page":
            self.get_page(query)
        elif path == "/api/search":
            self.search_characters(query)
        elif path == "/api/pipeline_flow":
            self.get_pipeline_flow()
        elif path == "/api/benchmarks":
            self.get_benchmarks()
        else:
            self.send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    def get_stats(self):
        """Tổng hợp toàn bộ chỉ số của bộ dataset 107.786 ký tự."""
        books_summary = {}
        total_chars = 0
        total_gold = 0
        total_syl = 0
        total_text_only = 0

        for book_id, cfg in BOOKS_CONFIG.items():
            rows = LABELS_CACHE.get(book_id, [])
            n_gold = sum(1 for r in rows if r.get("tier") == "GOLD")
            n_syl = sum(1 for r in rows if r.get("tier") == "SYLLABLE")
            n_txt = sum(1 for r in rows if r.get("tier") == "GOLD_text_only")
            total = len(rows)

            total_chars += total
            total_gold += n_gold
            total_syl += n_syl
            total_text_only += n_txt

            books_summary[book_id] = {
                "title": cfg["title"],
                "subtitle": cfg["subtitle"],
                "layout": cfg["layout_desc"],
                "total": total,
                "gold": n_gold,
                "syllable": n_syl,
                "gold_text_only": n_txt,
                "gold_pct": round(n_gold / total * 100, 1) if total else 0,
                "pages": cfg["n_pages"],
            }

        data = {
            "title": "Hệ Thống Gán Nhãn Tự Động Văn Bản Hán Nôm Cổ (GanNhanOCR)",
            "subtitle": "Đề Tài Luận Văn Thạc Sĩ · Kết Quả Sinh Dữ Liệu Tự Động 100%",
            "impact_metrics": {
                "total_characters": total_chars,
                "total_gold": total_gold,
                "total_syllable": total_syl,
                "total_gold_text_only": total_text_only,
                "gold_rate_overall": round((total_gold + total_text_only) / total_chars * 100, 1) if total_chars else 0,
                "books_count": 4,
                "human_intervention": 0,
                "automation_rate": "100%",
                "code_invariants": "18/18 PASS",
            },
            "books": books_summary,
        }
        self.send_json(data)

    def get_books(self):
        """Trả về danh sách các sách và trang mẫu có sẵn."""
        result = []
        for b_id, cfg in BOOKS_CONFIG.items():
            pages = list(PAGE_INDEX.get(b_id, {}).keys())
            pages.sort()
            result.append({
                "id": b_id,
                "title": cfg["title"],
                "subtitle": cfg["subtitle"],
                "author": cfg["author"],
                "layout": cfg["layout"],
                "layout_desc": cfg["layout_desc"],
                "n_columns": cfg["n_columns"],
                "n_pages": cfg["n_pages"],
                "sample_pages": cfg["sample_pages"],
                "default_page": cfg["default_page"],
                "available_pages": pages[:40],  # 40 trang đầu để chọn nhanh
                "total_chars": len(LABELS_CACHE.get(b_id, [])),
            })
        self.send_json(result)

    def get_page(self, query: dict):
        """Trả về ảnh scan và danh sách toàn bộ bounding box của một trang sách."""
        book_id = query.get("book", ["LucVanTien1883"])[0]
        page_id = query.get("page", ["page_0002"])[0]

        cfg = BOOKS_CONFIG.get(book_id)
        if not cfg:
            self.send_json({"error": f"Không tìm thấy sách: {book_id}"}, status=HTTPStatus.NOT_FOUND)
            return

        pages_data = PAGE_INDEX.get(book_id, {})
        rows = pages_data.get(page_id, [])

        # Kiểm tra ảnh trang scan có sẵn trên đĩa không
        scan_file = cfg["pages_dir"] / f"{page_id}.png"
        img_width = 1896
        img_height = 3212
        has_image = scan_file.exists()

        if has_image:
            try:
                # Đọc kích thước ảnh thật nhanh qua header PNG
                with open(scan_file, "rb") as f:
                    f.seek(16)
                    w_bytes = f.read(4)
                    h_bytes = f.read(4)
                    img_width = int.from_bytes(w_bytes, "big")
                    img_height = int.from_bytes(h_bytes, "big")
            except Exception:
                pass

        # Xử lý bounding box và thông tin từng ô chữ
        chars = []
        for idx, r in enumerate(rows):
            bbox_raw = r.get("bbox", "[]")
            bbox = [0, 0, 0, 0]
            try:
                bbox = json.loads(bbox_raw)
            except Exception:
                pass

            # Tọa độ bbox: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = bbox if len(bbox) == 4 else (0, 0, 0, 0)
            chars.append({
                "index": idx + 1,
                "image_rel": r.get("image", ""),
                "crop_url": f"/crops/{book_id}/{r.get('image', '')}",
                "column": int(r.get("column", 0)),
                "ocr_char": r.get("ocr_char", ""),
                "syllable": r.get("syllable", ""),
                "label": r.get("label", ""),
                "unicode": r.get("unicode", ""),
                "tier": r.get("tier", "OTHER"),
                "rule": r.get("rule", ""),
                "bbox": [xmin, ymin, xmax, ymax],
                # Tính tỷ lệ phần trăm so với ảnh để render CSS tuyệt đối chính xác
                "rect": {
                    "left_pct": round((xmin / img_width) * 100, 3) if img_width else 0,
                    "top_pct": round((ymin / img_height) * 100, 3) if img_height else 0,
                    "width_pct": round(((xmax - xmin) / img_width) * 100, 3) if img_width else 0,
                    "height_pct": round(((ymax - ymin) / img_height) * 100, 3) if img_height else 0,
                },
            })

        # Sắp xếp các ô chữ theo thứ tự đọc tự nhiên Hán Nôm: Từ phải sang trái, từ trên xuống dưới
        chars.sort(key=lambda c: (-c["column"], c["rect"]["top_pct"]))

        response = {
            "book": book_id,
            "book_title": cfg["title"],
            "page": page_id,
            "scan_url": f"/page_scans/{book_id}/{page_id}.png" if has_image else None,
            "has_scan": has_image,
            "dimensions": {"width": img_width, "height": img_height},
            "char_count": len(chars),
            "tier_counts": {
                "GOLD": sum(1 for c in chars if c["tier"] == "GOLD"),
                "SYLLABLE": sum(1 for c in chars if c["tier"] == "SYLLABLE"),
                "GOLD_text_only": sum(1 for c in chars if c["tier"] == "GOLD_text_only"),
            },
            "characters": chars,
        }
        self.send_json(response)

    def search_characters(self, query: dict):
        """Tìm kiếm mẫu chữ theo âm Quốc ngữ, chữ Nôm, Unicode hoặc bậc chất lượng."""
        q = query.get("q", [""])[0].strip().lower()
        book_filter = query.get("book", ["all"])[0]
        tier_filter = query.get("tier", ["all"])[0].upper()
        limit = min(int(query.get("limit", [60])[0]), 200)

        results = []
        books_to_search = [book_filter] if book_filter in BOOKS_CONFIG else list(BOOKS_CONFIG.keys())

        for b_id in books_to_search:
            rows = LABELS_CACHE.get(b_id, [])
            for r in rows:
                if tier_filter != "ALL" and r.get("tier") != tier_filter:
                    continue

                nom = r.get("label", "").lower()
                syl = r.get("syllable", "").lower()
                ocr = r.get("ocr_char", "").lower()
                uni = r.get("unicode", "").lower()

                # So khớp từ khóa
                if q:
                    matched = (q in syl) or (q in nom) or (q in ocr) or (q in uni)
                    if not matched:
                        continue

                results.append({
                    "book": b_id,
                    "book_title": BOOKS_CONFIG[b_id]["title"],
                    "page": r.get("page", ""),
                    "column": r.get("column", ""),
                    "label": r.get("label", ""),
                    "syllable": r.get("syllable", ""),
                    "ocr_char": r.get("ocr_char", ""),
                    "unicode": r.get("unicode", ""),
                    "tier": r.get("tier", ""),
                    "rule": r.get("rule", ""),
                    "image": r.get("image", ""),
                    "crop_url": f"/crops/{b_id}/{r.get('image', '')}",
                })

                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        self.send_json({"query": q, "count": len(results), "results": results})

    def get_pipeline_flow(self):
        """Dữ liệu chi tiết về 6 bước của pipeline tự động."""
        steps = [
            {
                "step": 1,
                "id": "ingest",
                "name": "Thu Nhận & Tiền Xử Lý",
                "tag": "Ingest & Denoise",
                "input": "File PDF bản quét gốc (scan 300 DPI) + Phiên âm Quốc ngữ",
                "model": "OpenCV Sauvola Adaptive Threshold + Otsu / Stretch",
                "process": "Khử nhiễu nền giấy cổ ố vàng, bóc tách viền trang, phân đoạn cột văn bản tự động (9-10 cột với thơ lục bát, 7 cột với văn xuôi) và gọi OCR Hán Nôm bước đầu.",
                "output": "Ảnh trang sạch (pages/), toạ độ cột thô (detected/) và dữ liệu OCR thô (kim_raw/)",
                "evidence": "Tách biệt hoàn toàn cache kim_raw/ theo tham số; 0 gọi API lại khi đã cache.",
            },
            {
                "step": 2,
                "id": "detect",
                "name": "Phát Hiện Ký Tự Nôm",
                "tag": "CenterNet ResNet-34",
                "input": "Ảnh cột chữ dọc đã bóc tách từ trang sách",
                "model": "CenterNet ResNet-34 (Ảnh 1024px, hồi quy tâm Heatmap, ngưỡng 0.15)",
                "process": "Dự đoán tâm ký tự Nôm, phân tách hộp dính bằng Seam Carving và Pitch Decoding (giải mã nhịp cách đều), triệt tiêu hiện tượng hộp rỗng và cắt vào nét chữ mảnh.",
                "output": "Tập hộp bao ký tự [xmin, ymin, xmax, ymax] cho từng cột",
                "evidence": "Giảm tỷ lệ cắt vào thân chữ từ 10.6% xuống 2.6%; tỷ lệ đếm đúng số chữ trên cột tăng từ 59% lên >78%.",
            },
            {
                "step": 3,
                "id": "align",
                "name": "Gióng Hàng Đa Nguồn (DP)",
                "tag": "Banded Dynamic Programming",
                "input": "Hộp chữ Nôm phát hiện được + Chuỗi âm Quốc ngữ đối ứng",
                "model": "Banded Dynamic Programming 2 chiều + Từ điển Hán Nôm QuocNgu_SinoNom (104.177 từ)",
                "process": "Tìm đường đi tối ưu giữa chuỗi hộp ảnh và chuỗi âm thơ lục bát. Áp dụng ràng buộc cứng theo cấu trúc nhịp 6/8 (câu lục 6 chữ, câu bát 8 chữ) để ngăn lỗi lệch trôi xuyên tầng.",
                "output": "Nhãn sơ bộ cho từng hộp ký tự kèm xác suất posterior và quy tắc liên kết (labels.csv)",
                "evidence": "Tự động 100% — 0 ô can thiệp thủ công (assert_qd01 = 0).",
            },
            {
                "step": 4,
                "id": "remediate",
                "name": "Kiểm Kê & Sửa Lỗi Nhầm",
                "tag": "Census AE-1/F1 & Confusion Fix",
                "input": "Bảng nhãn sơ bộ sau bước gióng hàng",
                "model": "Kiểm kê tần suất ngữ nghĩa + Bảng tri thức sửa nhầm hệ thống (confusion_fixes.yaml)",
                "process": "Phát hiện các chữ bị nhầm lẫn phổ biến (đồng âm khác nghĩa, tự dạng gần giống). Phân hạng 4 bậc chất lượng: GOLD (chữ + âm chuẩn xác), SYLLABLE (giữ vị trí âm), REVIEW (nghi ngờ), QUARANTINE (cách ly).",
                "output": "Bảng nhãn đã hiệu chỉnh (labels_remediated.csv)",
                "evidence": "Ghi nhận mã băm sha256 vào CHECKSUMS.txt sau mỗi bước chuyển đổi.",
            },
            {
                "step": 5,
                "id": "rescue_gates",
                "name": "Giải Cứu Thông Minh & Cổng Cơ Chế",
                "tag": "Self-Training & Mechanism Gates",
                "input": "Các ô rơi vào vùng nghi vấn (REVIEW / lệch số lượng)",
                "model": "Enhanced SE-ResNet v3 (Self-Training nội bộ) & 4 Cổng cơ chế (a, b, c, d)",
                "process": "• STT: Mô hình nhận dạng Nôm nội vùng giải cứu các ô REVIEW đạt ngưỡng tin cậy >= 0.70 lên GOLD.\n• Sách mới: 4 cổng cơ chế (a: đối chiếu số lượng, b: kiểm tra nhịp pitch, c: ranh giới hộp, d: đối chiếu dị bản độc lập 1871/1916) gán nhãn an toàn GOLD_text_only.",
                "output": "Bảng nhãn công bố hoàn thiện (labels_final.csv / labels_gated.csv)",
                "evidence": "Giải cứu thành công hàng nghìn mẫu chữ lên hạng GOLD mà không làm tăng nhiễu nhãn.",
            },
            {
                "step": 6,
                "id": "export",
                "name": "Đóng Gói & Xuất Bản Dataset",
                "tag": "Final Dataset Export",
                "input": "labels_final.csv + ảnh crop gốc từ thư mục trung gian",
                "model": "export_final_dataset.py + make_dataset_docs + make_xlsx",
                "process": "Lọc sạch các ô đạt chuẩn (GOLD, SYLLABLE), bóc tách ảnh crop thật sang thư mục tự chứa dataset/, sinh bảng tính labels.xlsx chuẩn hóa, và tự động tạo tài liệu lai lịch thư tịch (DATASHEET, README).",
                "output": "Thư mục dataset/ chuẩn giao nộp: labels.csv (12 cột cố định), gold/, syllable/, labels.xlsx",
                "evidence": "Độc lập hoàn toàn, tự chứa 100%, không còn phụ thuộc vào dữ liệu trung gian.",
            },
        ]
        self.send_json(steps)

    def get_benchmarks(self):
        """Bảng đối chiếu và số liệu kiểm chứng khoa học."""
        benchmarks = {
            "title": "Bảng Đối Chiếu Số Liệu & Luận Chứng Khoa Học",
            "comparisons": [
                {
                    "metric": "Tỷ lệ đếm đúng số chữ trên cột (n_det == N)",
                    "baseline": "59.2% (Ngưỡng thô 0.20)",
                    "proposed": "78.4% - 90.0% (CenterNet + Pitch Decoder)",
                    "improvement": "+19.2% đến +30.8%",
                    "impact": "Triệt tiêu hiện tượng dính chữ và mất chữ trên các cột thạch bản nét mảnh.",
                },
                {
                    "metric": "Tỷ lệ cắt phạm vào thân chữ (Ink Cut Rate)",
                    "baseline": "10.6% (Bổ đôi hộp đều tuyến tính)",
                    "proposed": "2.6% (Pitch Decoding thích ứng)",
                    "improvement": "Giảm 4 lần (giảm 8.0%)",
                    "impact": "Bảo toàn nguyên vẹn cấu trúc nét của từng chữ Nôm trong ảnh crop.",
                },
                {
                    "metric": "Độ chính xác đối soát dị bản (Cross-Edition Agreement)",
                    "baseline": "64.4% (OCR đơn kênh Hán thông thường)",
                    "proposed": "86.3% (OCR kênh Nôm chuyên biệt + Cổng cơ chế)",
                    "improvement": "+21.9%",
                    "impact": "Khớp chính xác với các bản khắc độc lập thời Nguyễn.",
                },
                {
                    "metric": "Tính Tự Động Hoá (Human-in-the-loop Rules)",
                    "baseline": "Cần can thiệp người gán nhãn thủ công",
                    "proposed": "0 ô can thiệp thủ công (quyet_dinh_nguoi = 0)",
                    "improvement": "Tự động 100%",
                    "impact": "Đảm bảo tính khách quan tuyệt đối và khả năng nhân rộng trên hàng trăm cuốn sách cổ.",
                },
            ],
            "invariants": {
                "total_invariants": 18,
                "passed_invariants": 18,
                "status": "ALL PASS",
                "items": [
                    "Bảo toàn cấu trúc 12 cột cố định chuẩn giao nộp",
                    "Không có ô quyet_dinh_nguoi trong toàn bộ các tệp xuất bản",
                    "Toàn bộ ảnh crop GOLD/SYLLABLE có thật và khớp mã băm MD5",
                    "Toàn bộ nhãn Nôm nằm trong tập từ điển chuẩn",
                    "Bảo toàn chuỗi checksum sha256 cho từng mốc pipeline",
                ],
            },
        }
        self.send_json(benchmarks)

    def serve_crop_image(self, path: str):
        """Phục vụ ảnh crop ký tự từ thư mục dataset/."""
        # /crops/<book_id>/<rel_path>
        parts = path.strip("/").split("/", 2)
        if len(parts) < 3:
            self.send_error(HTTPStatus.NOT_FOUND, "Đường dẫn crop không hợp lệ")
            return

        book_id, rel_crop = parts[1], parts[2]
        cfg = BOOKS_CONFIG.get(book_id)
        if not cfg:
            self.send_error(HTTPStatus.NOT_FOUND, f"Không tìm thấy sách: {book_id}")
            return

        crop_path = (cfg["crops_root"] / rel_crop).resolve()
        # Bảo mật: không cho thoát khỏi thư mục REPO_ROOT
        if not str(crop_path).startswith(str(REPO_ROOT)) or not crop_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Không tìm thấy file ảnh crop: {rel_crop}")
            return

        self.serve_file(crop_path, "image/png")

    def serve_page_scan(self, path: str):
        """Phục vụ ảnh scan trang gốc từ thư mục prepared/."""
        # /page_scans/<book_id>/<filename>
        parts = path.strip("/").split("/", 2)
        if len(parts) < 3:
            self.send_error(HTTPStatus.NOT_FOUND, "Đường dẫn trang scan không hợp lệ")
            return

        book_id, filename = parts[1], parts[2]
        cfg = BOOKS_CONFIG.get(book_id)
        if not cfg:
            self.send_error(HTTPStatus.NOT_FOUND, f"Không tìm thấy sách: {book_id}")
            return

        scan_path = (cfg["pages_dir"] / filename).resolve()
        if not str(scan_path).startswith(str(REPO_ROOT)) or not scan_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Không tìm thấy ảnh trang: {filename}")
            return

        mime_type, _ = mimetypes.guess_type(str(scan_path))
        self.serve_file(scan_path, mime_type or "image/png")

    def serve_file(self, file_path: Path, content_type: str):
        try:
            stat = file_path.stat()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            if self.command != "HEAD":
                with open(file_path, "rb") as f:
                    while chunk := f.read(65536):
                        self.wfile.write(chunk)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))


def run_server(port=8080):
    load_dataset_caches()
    server_address = ("", port)
    try:
        httpd = ThreadingHTTPServer(server_address, GanNhanHandler)
    except OSError as e:
        if port == 8080:
            print(f"[server] Cổng 8080 đang bận ({e}), thử cổng 8088...")
            run_server(8088)
            return
        raise

    print("\n" + "=" * 65)
    print(f"  GanNhanOCR Web Presentation Server Đang Hoạt Động!")
    print(f"  Truy cập trực tiếp tại: http://localhost:{port}")
    print("=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Đã dừng máy chủ.")
        httpd.server_close()


if __name__ == "__main__":
    p = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        p = int(sys.argv[1])
    run_server(p)
