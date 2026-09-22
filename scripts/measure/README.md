# scripts/measure/ — bộ đo tái lập được (2026-09-21)

Mọi phép đo về dữ liệu mới (thạch bản Lục Vân Tiên 1883, Kim Vân Kiều 1884, Chrestomathie 1872) và về mã pipeline
được gói thành script chạy bằng `.venv` + `tesseract`, **0 token LLM**, ra `summary.json` ≤ 8 KB kèm `invariants`.

## QUY ƯỚC BẮT BUỘC

> **Mọi phiên nghiên cứu/agent về sau KHÔNG chạy lại phép đo bằng LLM; chỉ chạy `measure.py`, đọc
> `measure_out/SUMMARY.json` / `measure_out/REPORT.md`; nếu nghi ngờ thì đổi tham số script hoặc thêm invariant,
> KHÔNG đọc ảnh/CSV thô bằng LLM; đọc mã pipeline qua `docs/PIPELINE_FACTS.json` trước, chỉ mở tệp nguồn khi FACTS thiếu.**

Lý do: lần đo bằng 19 agent (21/09) tốn ≈2,65 M token cho các con số mà bộ này tái lập trong ≈10 phút CPU,
và số đo bằng script không phụ thuộc agent "đọc bằng mắt".

## Cách chạy

```bash
cd <thư mục gốc repo GanNhanOCR>
.venv/bin/python scripts/measure/measure.py --all                      # toàn bộ (≈10–15 phút lần đầu; ≈4 phút khi có cache OCR)
.venv/bin/python scripts/measure/measure.py --book LucVanTien1883      # 1 sách (layout + qn_ocr) — thêm --steps để chọn
.venv/bin/python scripts/measure/measure.py --all --limit 8            # chạy thử; invariants toàn sách ghi SKIP
.venv/bin/python scripts/measure/measure.py --all --report-only        # chỉ gom summary hiện có → SUMMARY/REPORT
.venv/bin/python scripts/measure/measure.py --all --dry-run            # in kế hoạch lệnh, không chạy
```

Tuỳ chọn: `--steps code_facts,layout,qn_ocr,chresto_map,detector_transfer` · `--workers N` · `--out DIR` ·
`--qn-engine vietocr` (chậm ~20 s/trang, chỉ để so sánh) · `--nomna-pages 0` (bỏ NomNaOCR, cần venv có TensorFlow) ·
`--det-pages 27 --stt-pages 3` (detector; 27 trang/sách → CI ≈ ±3,5 điểm; `detector_transfer.py --page-ids 'LucVanTien1883:010,030;KimVanKieu1884:0020'` chạy đúng một mẫu trang cố định để tái lập số cũ) · `--layout-detector` (CenterNet làm phương pháp 2 cho layout, ~0,5 s/trang).

Mã thoát của `measure.py`: **0** mọi invariant cứng PASS · **1** có invariant cứng FAIL · **2** có bước chạy lỗi/thiếu summary.
Invariant "mềm" (bảng `SOFT_INVARIANTS` trong `measure.py`, mỗi mục có lý do) chỉ cảnh báo.

## Các mô-đun

| mô-đun | đo gì | sách | thời gian đủ | đầu ra chính |
|---|---|---|---|---|
| `layout_lithograph.py` | tầng/cột/ô Nôm thạch bản, số câu in (tesseract + kNN tự huấn luyện), chuỗi câu Viterbi, số trang góc | LVT1883, KVK1884 | 60 s / 100 s | `layout_pages.csv`, `layout_columns.csv`, `verse_chain.csv`, `numbers.csv`, `pages_full.json` |
| `qn_print_ocr.py` | dòng thơ QN in (tesseract psm4 + số lề psm7), chuỗi số câu, phân lớp trang FR/QN, CER 44 dòng đọc mắt | LVT1883, KVK1884 | 40 s / 3 phút (cache: 1–2 s) | `verses.tsv` (verse_no theo số in + seq_no vật lý), `pages.csv`, `anchors.csv`, `pages_review.json` |
| `chresto_map.py` | 20 truyện QN ↔ cột Nôm Chrestomathie; ranh giới tự động vs bảng REF; NomNaOCR xếp hạng truyện | Chrestomathie1872 | 25 s (cache) | `qn_stories.csv`, `qn_lines.csv`, `nom_columns.csv`, `bang_truyen_trang.csv`, `nom_boundaries_auto.csv` |
| `detector_transfer.py` | CenterNet (`train_crop/detector_r34.best.pt`) trên thạch bản: raw/stretch/otsu × thr 0.2/0.3/0.4, đối chứng STT2/4/11 | LVT+KVK+STT | ≈1–2 phút | `detector_columns.csv`, `detector_pages.csv`, `detector_configs.csv` |
| `code_facts.py` | sự kiện mã pipeline bằng AST/grep: mọi ghim `9`, `expected_cols`, CLI từng bước, khoá JSON cache, config sách | (repo) | 2 s | `docs/PIPELINE_FACTS.json` |

Mỗi mô-đun cũng chạy độc lập: `--book`, `--out`, `--limit N`, `--workers N` (xem `--help`).

## Đầu ra

```
measure_out/
  SUMMARY.json            gom mọi summary: bước, rc, thời gian, invariants (PASS/FAIL/mềm/SKIP), chỉ số then chốt, tệp CSV
  REPORT.md               bảng markdown đọc được: bước → chỉ số chính + invariants + đường dẫn CSV
  logs/<bước>_<sách>.log  stdout/stderr từng bước
  <book>/layout/          <book>/qn_ocr/          Chrestomathie1872/chresto_map/
  detector_transfer/      code_facts/             _cache/ (OCR cache theo md5 ảnh — xoá được)
```

Quy ước thư mục: **mỗi phép đo ghi vào `measure_out/<book>/<phép đo>/`** (không đè `summary.json` của phép đo khác).
`measure_out/` nằm trong `.gitignore`; sinh lại bằng `measure.py --all`.

## Thêm phép đo / sách mới

1. Viết `scripts/measure/<ten>.py`: `REPO = Path(__file__).resolve().parents[2]`; argparse `--book --out --limit --workers`;
   chỉ phụ thuộc `.venv` + tesseract; ghi `summary.json` ≤ 8 KB với `invariants: [{name, expected, observed, pass}]`
   (`pass = null` = SKIP khi `--limit`); ≤ 5 ảnh debug; ≤ 40 dòng stdout; idempotent (seed cố định, không phụ thuộc thứ tự glob).
2. Hai phương pháp độc lập cho đại lượng then chốt (số câu: tesseract vs kNN; số cột: chiếu vs chuỗi run/detector) + `agreement`.
3. Đăng ký trong `measure.py`: `ALL_STEPS`, `plan_steps()`, `KEY_METRICS[<ten>]`; nếu có invariant FAIL do bản chất dữ liệu
   (không phải lỗi mô-đun) thì ghi vào `SOFT_INVARIANTS` kèm lý do.
4. Sách mới: thêm vào `BOOKS` của mô-đun tương ứng (đường dẫn `data/<book>/pages`, số cột kỳ vọng…) và `ALL_BOOKS` của `measure.py`.

## Sự thật đã đo (để tự kiểm khi chạy lại)

- LVT1883: 105 trang Nôm, 10 cột × 2 tầng, 1.044 cặp = 2.088 câu; 139 trang QN in, 2.088 dòng, 0 trang trôi.
- KVK1884: canvas chữ 4–166 (page = 167 − canvas), 1.628 cặp = 3.256 câu Nôm; QN 295 canvas lẻ, 3.251 dòng vật lý,
  số in cuối 3.253 (4 bất thường của chuỗi số in, vị trí theo verses.tsv: nhảy 1581→1583 (vol2 canvas 23), 2220→2222 (canvas 135),
  2232 lặp (canvas 137), 3053→3055 (canvas 269)) → KHÔNG khớp 1-1 theo chỉ số toàn sách; 3.256 − 3.251 = 5 câu chưa định vị.
- Chrestomathie1872: 20 truyện, 608 dòng QN, 419 cột Nôm ≈ 8.692 chữ, 7 ô/trang.
- Detector (27 trang/sách, 540 cột): nền xám 128 làm CenterNet mù (KVK raw 23,5 % cột đúng) → cần stretch/otsu + thr 0,2
  (LVT stretch 77,6 %, KVK otsu 77,6 %; mẫu 9 trang cũ của 20/09 tái lập đúng 82,2 / 80,3 bằng `--page-ids`). Lề trái chỉ cắt 2 % (6 % làm mất cột 10 KVK).
