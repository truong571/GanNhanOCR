# GanNhanOCR — ứng dụng web minh hoạ luận văn

Ứng dụng một trang (HTML/CSS/JS thuần + máy chủ Python chỉ dùng thư viện chuẩn) phục vụ buổi bảo vệ luận văn Thạc sĩ
*Hệ thống hỗ trợ gán nhãn tự động văn bản Hán Nôm cổ*. Giao diện chỉ trình bày **tầng nhãn** của bộ dữ liệu:
**GOLD** · **SYLLABLE** · **GOLD_text_only** (chỉ hiện khi có ô) · **REVIEW** · **QUARANTINE**.

**Không có con số nào gõ cứng**: số ký tự, số ô theo tầng, số trang… được máy chủ đọc trực tiếp từ dữ liệu của dự án;
tệp nguồn thiếu thì mục tương ứng không hiển thị (hoặc ghi "chưa có dữ liệu" cho bộ chưa dựng) thay vì bịa số.

## 1. Nội dung trình bày

| Tab | Nội dung |
| --- | --- |
| 1. Tổng quan | KPI (tổng ký tự, GOLD, SYLLABLE, REVIEW, QUARANTINE, số bộ; kèm một dòng "Bất biến bộ đo: N PASS / M FAIL" nếu có `measure_out/SUMMARY.json`), danh mục **10 bộ** theo vai trò (giao nộp · đánh giá IHR · đánh giá Borg; thẻ sách ghi bộ giải mã hộp đang dùng, ví dụ "· hộp visual_dp"), quy ước các tầng nhãn kèm số ô |
| 2. Quy trình | **7 giai đoạn**: tiền xử lý & OCR → phát hiện & chọn hộp ký tự (CenterNet + bộ giải mã hộp theo bộ) → gióng hàng (Banded DP) → kiểm kê & xếp tầng → cổng cơ chế & đối soát dị bản → đóng gói `dataset/<Bộ>/` → gộp `dataset/_ALL`; chỉ số của từng bước đọc từ tệp (thiếu thì bỏ qua) |
| 3. Trực quan hoá bản quét | Ảnh trang + hộp ký tự **tô màu theo tầng nhãn**; chú giải = bộ lọc (bấm để ẩn/hiện từng tầng, kèm số ô trên trang); bảng chi tiết hiện crop, chữ Nôm, âm, tầng, Unicode, luật gán, toạ độ hộp |
| 4. Tra cứu | Tìm theo âm (có/không dấu), chữ Nôm, Unicode; lọc theo bộ/nhóm (giao nộp · tập đánh giá) và tầng nhãn; bấm thẻ để mở đúng trang |
| 5. Kết quả | Bảng **tầng nhãn theo bộ** (10 bộ + dòng tổng: tổng ô, GOLD, SYLLABLE, [GOLD_text_only], REVIEW, QUARANTINE, % GOLD) và cây thư mục `dataset/` |

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
`Dict/QuocNgu_SinoNom.csv` (số mục từ) · bộ giải mã hộp của từng bộ = `pipeline/tools/duong_chay.py` (tuỳ chọn, cần .venv;
thiếu thì thẻ sách không ghi phần "hộp …").

## 2. Chạy

### Cách 1 — máy chủ API (khuyên dùng khi trình bày trên máy có repo)

```bash
.venv/bin/python web/server.py 8088                  # http://localhost:8088
.venv/bin/python web/server.py 8088 --host 127.0.0.1 # chỉ máy này truy cập
GANNHANOCR_ROOT=/đường/dẫn/gốc .venv/bin/python web/server.py 8088   # hoặc --root <thư mục>: đọc dữ liệu ở gốc khác
```

- Máy chủ **chỉ đọc**; nạp nhãn vào RAM lúc khởi động (log liệt kê từng bộ: số dòng theo tầng, số trang, số ảnh trang).
- Bộ chưa có dữ liệu (đang dựng lại) không làm sập máy chủ: giao diện ghi "chưa có dữ liệu (đang dựng lại…)".
- Sau khi pipeline chạy xong: bấm **"Nạp lại dữ liệu"** trên thanh đầu trang (gọi `/api/reload`) — không cần khởi động lại.
- API: `/api/stats` · `/api/books` · `/api/page?book=&page=` · `/api/search?q=&book=&tier=&limit=` ·
  `/api/pipeline_flow` · `/api/benchmarks` · `/api/reload`. `book` nhận mã bộ, bí danh (`lvt1883`, `b18`…) hoặc nhóm
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
├── index.html            # giao diện (sáng/tối, co giãn tới màn hình điện thoại, không cuộn ngang)
├── style.css             # phong cách tài liệu khoa học; màu theo tầng nhãn
├── app.js                # logic: KPI, quy trình, trình soi (tô màu theo tầng), tra cứu, bảng tầng nhãn
├── server.py             # máy chủ HTTP + REST API (thư viện chuẩn), chỉ đọc
├── build_sample_data.py  # sinh sample_data.json/.js từ dữ liệu thật (dùng lại hàm API của server.py)
├── sample_data.json      # dữ liệu mẫu cho chế độ ngoại tuyến
└── sample_data.js        # bản bọc sample_data.json cho file:// (sinh cùng lúc)
```

## 4. Gợi ý thuyết minh

1. **Quy mô + tính tự động** (tab 1): KPI đọc từ dữ liệu; nhấn mạnh tách **giao nộp** và **tập đánh giá** (IHR, Borg,
   `evaluation_only`); mọi nhãn do luật gán, không có quyết định của người.
2. **Tầng nhãn trên trang thật** (tab 3): chọn một trang, bật/tắt từng tầng trong chú giải để thấy ô GOLD và SYLLABLE
   nằm ở đâu; bấm một ô để xem crop, âm và luật gán.
3. **Tra cứu** (tab 4): gõ một âm (ví dụ `nguoi`), lọc theo tầng, bấm thẻ để mở đúng trang.
4. **Kết quả** (tab 5): so tỉ lệ GOLD giữa các bộ (chép tay, thạch bản, mộc bản, sách in).
