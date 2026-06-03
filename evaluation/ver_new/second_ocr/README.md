> ⏸️ **TẠM BỎ (2026-06):** đường S3 đang dùng là **model train**
> (`../nom_classifier/`, đã thắng nghiệm thu). Folder này GIỮ làm **so sánh**
> cho luận văn (DINOv2 vs Gemini vs classifier), KHÔNG nằm trong pipeline.

# second_ocr — Đồng thuận OCR-thứ-2 (so với classifier / DINOv2)

Ý tưởng: S1 = HCMUS SinoNom OCR. Một **OCR thứ 2 độc lập** đọc lại cùng crop →
nếu **đồng thuận** với S1/từ điển thì nâng nhãn — thay vì phải train classifier.

## "Có cái nào sẵn để call về chạy không?" — Khảo sát

| Lựa chọn | Sẵn sàng? | Đọc chữ Nôm? |
|---|---|---|
| **HCMUS SinoNom** | đang là **S1** | (chính nó) — không dùng làm OCR-2 được |
| **Gemini vision** (REST) | ✅ **CÓ `GEMINI_API_KEY` trong .env** → call-và-chạy, **không cần train** | ✅ đọc được cả Nôm Ext-B (model lớn, từ vựng mở) — **lựa chọn tốt nhất** |
| Tesseract `chi_tra` | binary có, cần `pytesseract` + model | ⚠️ trained **tiếng Trung hiện đại** → yếu trên Nôm mộc bản |
| PaddleOCR / EasyOCR / cnocr | phải `pip install` | ⚠️ như Tesseract — modern Chinese, thiếu glyph Nôm |

→ **Không có** OCR *chuyên Nôm* nào sẵn ngoài HCMUS. "Call về chạy thôi" thực tế =
**Gemini vision** (đã có key). OCR-CJK generic chỉ là cross-check yếu ở chữ Hán phổ thông.

## File
- `ocr_backends.py` — `GeminiOCR` (forced-choice `pick` + open `read`, REST, có
  backoff 429, **không lộ key**) · `TesseractOCR` (tuỳ chọn) · `get_backend()`.
- `run_consensus.py` — đo trên SAMPLE:
  - **mặc định (gold):** crop GOLD đã biết chữ → top-1 forced-4-choice = *Gemini đọc
    Nôm tốt không?* (so DINOv2 = 0%, classifier mục tiêu ≥80%).
  - **`--target review`:** crop unconfirmed (cắt từ trang) → tỉ lệ đồng thuận với
    `ocr_char` = *ước lượng % unconfirmed nâng được lên char-level bằng đồng thuận 2-OCR*.

## Chạy
```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
.venv/bin/python evaluation/ver_new/second_ocr/run_consensus.py --n 50 --sleep 4
.venv/bin/python evaluation/ver_new/second_ocr/run_consensus.py --target review --n 50
```

## ⚠️ Hạn mức (đã kiểm)
Key + model `gemini-2.0-flash` **hợp lệ, gọi được** (request tới API thành công).
NHƯNG **free-tier bị giới hạn RPM/RPD → trả 429**; script có backoff nhưng chạy
chậm. Để có số liệu đầy đủ:
- Dùng `--sleep 6..8` và `--n` nhỏ (free tier ~15 RPM), hoặc
- Bật **billing** cho key (paid tier) để chạy nhanh/đủ, hoặc
- Chạy trên Colab/Kaggle với key riêng.
Tôi đã verify *integration chạy được* nhưng chưa lấy được con số accuracy ở đây vì
quota free-tier — bạn chạy với pacing/paid key sẽ ra số để so với classifier.

## Đọc kết quả để quyết định
- Gemini top-1 (gold) **cao** (vd ≥80%) ⇒ **OCR-2 đồng thuận là hướng tốt, không cần
  train** → cắm `GeminiOCR.pick` làm S3 trong `../visual_signal.py`.
- **Thấp** ⇒ quay lại classifier Nôm (`../nom_classifier/`).
- So 3 đường trên **cùng tập test**: DINOv2 (0%) vs Gemini-2OCR vs classifier-train.
