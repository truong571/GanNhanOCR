# Web OCR APIs test — Chinese OCR services on full page

Service: OCR.space free API (`helloworld` demo key, very limited).
Hypothesis: web Chinese OCR might work better than Tesseract/Paddle local
because they use bigger models + full-page context (not tiny crop).

### SachThanhTruyen2 / page_0012 — OCR.space Simplified
- detected: 7 CJK chars (7 unique)
- overlap with Kimhannom: 3 chars
- raw output (first 200 chars):
  ```
  弄！声下
丨眉同云
12
  ```

### SachThanhTruyen2 / page_0012 — OCR.space Traditional
- detected: 3 CJK chars (2 unique)
- overlap with Kimhannom: 0 chars
- raw output (first 200 chars):
  ```
  4
8
一一詈1
  ```

### SachThanhTruyen2 / page_0100 — OCR.space Simplified
- detected: 0 CJK chars (0 unique)
- overlap with Kimhannom: 0 chars
- raw output (first 200 chars):
  ```
  100
  ```

### SachThanhTruyen2 / page_0100 — OCR.space Traditional
- detected: 0 CJK chars (0 unique)
- overlap with Kimhannom: 0 chars
- raw output (first 200 chars):
  ```
  1
LO
8
  ```

### SachThanhTruyen4 / page_0050 — OCR.space Simplified
- detected: 11 CJK chars (10 unique)
- overlap with Kimhannom: 2 chars
- raw output (first 200 chars):
  ```
  胄、未翁
翁岷邳
．0丿
、还绮寿主
  ```

### SachThanhTruyen4 / page_0050 — OCR.space Traditional
- detected: 0 CJK chars (0 unique)
- overlap with Kimhannom: 0 chars
- raw output (first 200 chars):
  ```
  8
1
  ```
