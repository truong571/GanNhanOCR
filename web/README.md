# GanNhanOCR — ứng dụng web minh hoạ luận văn

Ứng dụng một trang (HTML/CSS/JS thuần + máy chủ Python chỉ dùng thư viện chuẩn) phục vụ buổi bảo vệ luận văn Thạc sĩ
*Hệ thống hỗ trợ gán nhãn tự động văn bản Hán Nôm cổ*. Giao diện chỉ trình bày **tầng nhãn** của bộ dữ liệu:
**GOLD** · **SYLLABLE** · **GOLD_text_only** (chỉ hiện khi có ô) · **REVIEW** · **QUARANTINE**.

**Không có con số nào gõ cứng**: số ký tự, số ô theo tầng, số trang… được máy chủ đọc trực tiếp từ dữ liệu của dự án;
tệp nguồn thiếu thì mục tương ứng không hiển thị (hoặc ghi "chưa có dữ liệu" cho bộ chưa dựng) thay vì bịa số.

## 1. Nội dung trình bày

Trang kiểu tài liệu học thuật: nền trắng, một màu nhấn, bảng kiểu booktabs có chú thích ("Bảng n.", "Hình 1."), chỉ dùng
phông hệ thống (chạy được ngoại tuyến). Màu tầng nhãn (bảng màu an toàn cho mù màu) chỉ dùng cho viền hộp và chú giải.

| Mục (`#…`) | Nội dung |
| --- | --- |
| Giới thiệu (`#gioi-thieu`) | Bảng 1: 10 bộ theo nhóm (giao nộp · đánh giá IHR · đánh giá Borg) — loại bản, số ô, GOLD, REVIEW; định nghĩa 5 tầng nhãn |
| Quy trình (`#quy-trinh`) | Bảng 2: 7 bước (đầu vào · phương pháp · đầu ra); Bảng 3: số đo ngắn đọc từ tệp đo (chỉ hiện khi có tệp) |
| Bản quét (`#ban-quet`) | Chọn bộ + trang, thu phóng; hộp tô viền theo tầng, bật/tắt từng tầng; bấm hộp → Bảng 4 (crop, chữ, âm, tầng, luật, Unicode, hộp) |
| Tra cứu (`#tra-cuu`) | Tìm theo âm (có/không dấu), chữ Nôm, Unicode; lọc bộ/nhóm và tầng; kết quả dạng bảng có crop, bấm dòng để mở trang |
| Số liệu (`#so-lieu`) | Bảng 6: số ô theo tầng của từng bộ + tổng; một dòng bất biến bộ đo (PASS/FAIL); cây thư mục `dataset/` |

### 10 bộ

| Nhóm | Bộ | Dữ liệu web đọc | Ảnh trang |
| --- | --- | --- | --- |
| Giao nộp | SachThanhTruyen 2 / 4 / 11 | `dataset/SachThanhTruyen/labels.csv` (cột `book` = stt2/stt4/stt11); vắng quyển nào thì lùi về bản xuất cũ `dataset/SachThanhTruyen{2,4,11}/` (ghi rõ "bản cũ") | `prepared/SachThanhTruyen{N}/pages/` |
| Giao nộp | LucVanTien1883, KimVanKieu1884, Chrestomathie1872 | `dataset/<Bộ>/labels.csv` | `prepared/<Bộ>/pages/` |
| Đánh giá (IHR-NomDB) | LucVanTien1916, TruyenKieu1872 | như trên (`evaluation_only`) | `prepared/<Bộ>/pages/` |
| Đánh giá (Borg, chép tay) | SachKinhThayCaBinh (Borg.tonch.18), SachDungLyHoThan (Borg.tonch.34) | như trên (`evaluation_only`) | `prepared/_auto/<Sách>/pages/` |

Vắng hẳn `dataset/<Bộ>/labels.csv` thì máy chủ lọc bộ đó từ bộ gộp `dataset/_ALL/labels.csv` (cột `book_set`).

Ô **REVIEW / QUARANTINE** không được đóng gói vào `labels.csv` (không giao ảnh), nên máy chủ lấy thêm từ bảng mọi tầng của bản dựng — `dataset_out/labels_final.csv` (STT, cột `book`) và `<ds_out>/labels_gated.csv` (sách khác), cùng nguồn với `scripts/bao_cao_tong_hop.py`: hiện hộp + nhãn trên bản quét, không có crop.

### Nguồn số liệu (đọc từ tệp)

`dataset/<Bộ>/labels.csv` (cột `tier`, chỉ đọc 11 cột cần) · `dataset/_ALL/SOURCES.json` (bộ gộp) ·
`measure_out/SUMMARY.json` (dòng bất biến) · `measure_out/box_ref/summary.json` (hộp legacy → pitch, giai đoạn 2) ·
`prepared/<LVT1883|KVK1884>/dataset_out/auto_precision_*/SUMMARY.json` (đối soát dị bản, giai đoạn 5) ·
`Dict/QuocNgu_SinoNom.csv` (số mục từ).

## 2. Chạy

### Cách 1 — máy chủ API (khuyên dùng khi trình bày trên máy có repo)

```bash
.venv/bin/python web/server.py 8088                  # http://localhost:8088
.venv/bin/python web/server.py 8088 --host 127.0.0.1 # chỉ máy này truy cập
GANNHANOCR_ROOT=/đường/dẫn/gốc .venv/bin/python web/server.py 8088   # hoặc --root <thư mục>: đọc dữ liệu ở gốc khác
```

- Máy chủ **chỉ đọc**; nạp nhãn vào RAM lúc khởi động (log liệt kê từng bộ: số dòng theo tầng, số trang, số ảnh trang).
- Bộ chưa có dữ liệu (đang dựng lại) không làm sập máy chủ: giao diện ghi "chưa có dữ liệu (đang dựng lại…)".
- Sau khi pipeline chạy xong: bấm **"Nạp lại dữ liệu"** ở chân trang (gọi `/api/reload`) — không cần khởi động lại.
- API: `/api/stats` · `/api/books` · `/api/page?book=&page=` · `/api/search?q=&book=&tier=&limit=` ·
  `/api/pipeline_flow` (mỗi bước: `name`, `input`, `method`, `output`, `metrics`) · `/api/benchmarks` · `/api/reload`. `book` nhận mã bộ, bí danh (`lvt1883`, `b18`…) hoặc nhóm
  (`all`, `giao_nop`, `danh_gia`); `tier` ∈ `GOLD|SYLLABLE|GOLD_text_only|REVIEW|QUARANTINE|all`.
- Tệp phục vụ (chỉ ảnh, chặn đường dẫn thoát ra ngoài): `/crops/<bộ>/…`, `/page_scans/<bộ>/…`.

### Cách 2 — mở trực tiếp `index.html` (máy trình chiếu không có Python)

- Chép thư mục `web/` (cần `index.html`, `style.css`, `app.js`, `sample_data.json`, `sample_data.js`) rồi mở `index.html`.
- Ứng dụng tự chuyển sang **chế độ dữ liệu mẫu**: 1 trang mẫu / bộ, ảnh trang thu nhỏ và crop của thẻ thư viện được nhúng
  sẵn; số liệu = đúng API của máy chủ lúc sinh bản mẫu.
- `sample_data.js` là bản bọc của `sample_data.json` (Chrome chặn `fetch()` tệp cục bộ khi mở bằng `file://`).

### Sinh lại `sample_data.json` (+ `sample_data.js`)

Chạy **sau khi** lượt dựng lại dữ liệu đã xong (script tự dừng nếu `clean_rebuild_all.sh` / `run_pipeline.sh` còn chạy):

```bash
.venv/bin/python web/build_sample_data.py                  # mặc định: ≤ 5 MB, ảnh trang rộng 720 px
.venv/bin/python web/build_sample_data.py --max-mb 3 --page-width 640 --gallery-per-book 12
.venv/bin/python web/build_sample_data.py --no-images      # không nhúng ảnh (chỉ đường dẫn ../dataset/…)
```

## 3. Cấu trúc `web/`

```text
web/
├── index.html            # 5 mục; co giãn tới màn hình điện thoại (một cột dưới ~760 px, không cuộn ngang trang)
├── style.css             # kiểu tài liệu học thuật, bảng booktabs; màu tầng nhãn chỉ cho viền hộp + chú giải
├── app.js                # logic: bảng bộ, quy trình, trình xem bản quét, tra cứu, bảng số liệu
├── server.py             # máy chủ HTTP + REST API (thư viện chuẩn), chỉ đọc
├── build_sample_data.py  # sinh sample_data.json/.js từ dữ liệu thật (dùng lại hàm API của server.py)
├── sample_data.json      # dữ liệu mẫu cho chế độ ngoại tuyến
└── sample_data.js        # bản bọc sample_data.json cho file:// (sinh cùng lúc)
```

## 4. Gợi ý thuyết minh

1. **Giới thiệu**: quy mô 10 bộ, tách giao nộp và tập đánh giá (`evaluation_only`); nhãn do luật gán, nhãn người chỉ để đo.
2. **Bản quét**: chọn một trang, tắt/bật từng tầng để thấy ô GOLD, SYLLABLE, REVIEW nằm ở đâu; bấm một ô để xem crop và luật.
3. **Tra cứu**: gõ một âm (ví dụ `nguoi`), lọc theo tầng, bấm dòng để mở đúng trang.
4. **Số liệu**: so tỉ lệ GOLD giữa các loại bản (chép tay, thạch bản, mộc bản, sách in).
