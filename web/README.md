# GanNhanOCR — ứng dụng web minh hoạ luận văn

Ứng dụng một trang (HTML/CSS/JS thuần + máy chủ Python chỉ dùng thư viện chuẩn) phục vụ buổi bảo vệ luận văn Thạc sĩ
*Hệ thống hỗ trợ gán nhãn tự động văn bản Hán Nôm cổ*. **Không có con số nào gõ cứng**: mọi số liệu (số ký tự, tầng nhãn,
GOLD chính xác, độ chính xác, bất biến) được máy chủ đọc trực tiếp từ dữ liệu của dự án; tệp nguồn thiếu thì giao diện ghi
**"chưa có"** thay vì bịa số.

## 1. Nội dung trình bày

| Tab | Nội dung |
| --- | --- |
| 1. Tổng quan | KPI (ký tự, GOLD, SYLLABLE, **GOLD chính xác ok**, số bộ, bất biến), danh mục **10 bộ** theo vai trò, quy ước 3 tầng nhãn, **4 trạng thái GOLD chính xác** và 3 mức chứng cứ |
| 2. Quy trình | **9 giai đoạn**: tiền xử lý & OCR → phát hiện & **chọn hộp theo từng bộ** (legacy / pitch / visual_dp / visual_dp_hybrid — đọc đúng bảng ĐƯỜNG CHẠY của `run_pipeline.sh` + số đo TN6) → gióng hàng (Banded DP) → kiểm kê → cổng & dị bản → đóng gói → gộp `dataset/_ALL` → **GOLD chính xác (B8)** kèm **profile chữ viết tay + lần đọc thứ hai lt2 (STT)** → **công bố** `dataset/_ALL/cong_bo` (ảnh = ô ok); mỗi bước có chỉ số đọc từ `measure_out/` |
| 3. Trực quan hoá bản quét | Ảnh trang + hộp ký tự; **công tắc tô màu theo tầng nhãn HOẶC theo GOLD chính xác** (ok xanh · text_only vàng · uncertified xám · review đỏ · GOLD chưa có gold_exact tím nét đứt · ô không phải GOLD nét đứt mờ); chú giải = bộ lọc; bảng chi tiết hiện **crop gốc + crop chuẩn**, trạng thái, **lý do (tiếng Việt)**, **mức chứng cứ**, `cell_uid` |
| 4. Tra cứu | Tìm theo âm (có/không dấu), chữ Nôm, Unicode; lọc theo bộ/nhóm (giao nộp · tập đánh giá · Borg nhãn người), tầng nhãn, **trạng thái GOLD chính xác**; bấm thẻ để mở đúng trang |
| 5. Kết quả & đánh giá | Độ chính xác theo **mức chứng cứ** (ĐO trên nhãn người · ƯỚC LƯỢNG qua dị bản · SUY ĐOÁN), bảng GOLD chính xác 10 bộ, pitch decoding so với hộp legacy (box_ref), độ đúng OCR kim theo loại bản, bất biến, bộ crop nhãn người Borg, cây thư mục `dataset/`, danh sách nguồn còn thiếu |

### 10 bộ + 2 mục xem nhãn người

| Nhóm | Bộ | Dữ liệu web đọc | Ảnh trang |
| --- | --- | --- | --- |
| Giao nộp | SachThanhTruyen 2 / 4 / 11 | `dataset/SachThanhTruyen/labels.csv` (cột `book` = stt2/stt4/stt11); vắng quyển nào thì lùi về bản xuất cũ `dataset/SachThanhTruyen{2,4,11}/` (23/09, ghi rõ "bản cũ", không ghép gold_exact) | `prepared/SachThanhTruyen{N}/pages/` |
| Giao nộp | LucVanTien1883, KimVanKieu1884, Chrestomathie1872 | `dataset/<Bộ>/labels.csv` + `labels_trace.csv` | `prepared/<Bộ>/pages/` |
| Đánh giá (IHR-NomDB) | LucVanTien1916, TruyenKieu1872 | như trên (`evaluation_only`) | `prepared/<Bộ>/pages/` |
| Đánh giá (Borg, chép tay) | SachKinhThayCaBinh (Borg.tonch.18), SachDungLyHoThan (Borg.tonch.34) | như trên (`evaluation_only`) | `prepared/_auto/<Sách>/pages/` |
| Borg — nhãn người | 2 mục xem riêng | `dataset/_BORG_NHAN_NGUOI/labels.csv` (+ `crops/`, `crops_chuan/`), tô màu theo `keep_level` | `prepared/<Sách>/pages/` |

### GOLD chính xác (bước B8)

- Nguồn: `dataset/_ALL/gold_exact.csv` (vắng thì `dataset/<Bộ>/gold_exact.csv`), khoá **`cell_uid`**. Dòng của
  `dataset/<Bộ>/labels.csv` được gán `cell_uid` đúng như `pipeline/tools/merge_datasets._uid`
  (`<bộ>/<book>/<page>/c<cột>/n<nom_idx>/s<syl_idx>`, `nom_idx`/`syl_idx` lấy từ `labels_trace.csv`); dự phòng ghép theo
  (`book_set`, `image`). Nếu `gold_exact.csv` khác lượt dựng với `labels.csv`, trang ghi rõ "không khớp".
- Crop chuẩn của ô `ok`: `dataset/_ALL/crops_chuan/…` (URL `/crops_chuan/…`).
- Mức chứng cứ (`evidence_level`): **ĐO** ở LucVanTien1916, TruyenKieu1872, 2 bộ Borg (nhãn người) · **ƯỚC LƯỢNG** ở
  LucVanTien1883, KimVanKieu1884 (dị bản người) · **SUY ĐOÁN** ở STT và Chrestomathie1872.
- Lý do (`reason`) được dịch sang tiếng Việt trong `server.py` (`REASON_VI`); mã lạ hiển thị nguyên mã.

### Nguồn số liệu (đọc từ tệp, thiếu -> "chưa có")

`measure_out/SUMMARY.json` (bất biến) · `measure_out/gold_exact/summary.json` · `measure_out/<LVT1916|TK1872>/ihr_endtoend/summary.json`
· `measure_out/<Borg>/borg_endtoend/summary.json` · `measure_out/box_ref/summary.json` · `measure_out/borg_human/summary.json`
· `prepared/<LVT1883|KVK1884>/dataset_out/auto_precision_{verify,gated}/SUMMARY.json` · `dataset/_ALL/SOURCES.json`
· `dataset/_BORG_NHAN_NGUOI/BUILD_INFO.json` · `Dict/QuocNgu_SinoNom.csv` (số mục từ) · `measure_out/_tn6/table.json` (TN6: hộp trước/sau) · `measure_out/stt_lt2/summary.json` (hiệu chuẩn lt2) · `dataset/_ALL/cong_bo/EXCLUSIONS.json` (tập công bố) · đường chạy từng bộ = `pipeline/tools/duong_chay.py` (cần .venv; thiếu -> "chưa có") — cùng cách đọc với
`scripts/bao_cao_tong_hop.py`.

## 2. Chạy

### Cách 1 — máy chủ API (khuyên dùng khi trình bày trên máy có repo)

```bash
.venv/bin/python web/server.py 8088                  # http://localhost:8088
.venv/bin/python web/server.py 8088 --host 127.0.0.1 # chỉ máy này truy cập
GANNHANOCR_ROOT=/đường/dẫn/gốc .venv/bin/python web/server.py 8088   # hoặc --root <thư mục>: đọc dữ liệu ở gốc khác
```

- Máy chủ **chỉ đọc**; nạp nhãn vào RAM lúc khởi động (log liệt kê từng bộ: số dòng, số trang, số ô ghép gold_exact).
- Bộ chưa có dữ liệu (đang dựng lại) không làm sập máy chủ: giao diện ghi "chưa có dữ liệu (đang dựng lại…)".
- Sau khi pipeline chạy xong: bấm **"Nạp lại dữ liệu"** trên thanh đầu trang (gọi `/api/reload`) — không cần khởi động lại.
- API: `/api/stats` · `/api/books` · `/api/page?book=&page=` · `/api/search?q=&book=&tier=&gx=&limit=` ·
  `/api/pipeline_flow` · `/api/benchmarks` · `/api/reload`. `book` nhận mã bộ, bí danh (`lvt1883`, `b18`…) hoặc nhóm
  (`all`, `giao_nop`, `danh_gia`, `nhan_nguoi`); `gx` ∈ `ok|text_only|uncertified|review|chua_co`.
- Tệp phục vụ (chặn đường dẫn thoát ra ngoài): `/crops/<bộ>/…`, `/crops_chuan/…`, `/borg_human/…`, `/page_scans/<bộ>/…`.

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
├── style.css             # phong cách tài liệu khoa học; màu tầng nhãn + GOLD chính xác + mức chứng cứ
├── app.js                # logic: KPI, quy trình, trình soi (tô màu 2 chế độ), tra cứu, bảng số đo
├── server.py             # máy chủ HTTP + REST API (thư viện chuẩn), chỉ đọc
├── build_sample_data.py  # sinh sample_data.json/.js từ dữ liệu thật (dùng lại hàm API của server.py)
├── sample_data.json      # dữ liệu mẫu cho chế độ ngoại tuyến
└── sample_data.js        # bản bọc sample_data.json cho file:// (sinh cùng lúc)
```

## 4. Gợi ý thuyết minh

1. **Quy mô + tính tự động** (tab 1): KPI đọc từ dữ liệu; nhấn mạnh tách **giao nộp** và **tập đánh giá** (IHR, Borg có
   nhãn người, `evaluation_only`), nhãn người chỉ để đo.
2. **GOLD chính xác** (tab 3): chọn một trang, bật "Tô theo GOLD chính xác", chỉ vào ô `text_only`/`review` và đọc lý do;
   so crop gốc với crop chuẩn của ô `ok`.
3. **Chữ viết tay** (tab 3, Borg / STT; tab 2 bước 8): profile handwriting — bỏ cổng CNT, bộ kiểm viết tay LOBO theo sách;
   mở mục "Borg — nhãn người" để thấy hộp do người phiên.
4. **Độ chính xác trung thực** (tab 5): mỗi con số đi kèm mức chứng cứ; chỉ bộ có nhãn người mới là **ĐO**.
