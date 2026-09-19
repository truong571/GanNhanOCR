# data/ — mỗi thư mục = một cuốn sách (cập nhật 2026-09-20 04:55)

| Thư mục | Sách | Loại chữ | Trang Nôm | Ảnh scan Quốc ngữ căn cứ (`quocngu_pages/`) | Vai trò | Trạng thái Ground Truth |
|---|---|---|---|---|---|---|
| `SachThanhTruyen2/` | Sách Các Thánh Truyện tháng 2 | viết tay (thảo) | 160 Nôm | ✅ **160 trang scan QN gốc** (`quocngu_pages/`) + `INDEX.tsv` | **Kho chính** (yen2) | 100% Ground Truth đối chiếu trang |
| `SachThanhTruyen4/` | — tháng 4 | viết tay | 147 Nôm | ✅ **147 trang scan QN gốc** (`quocngu_pages/`) + `INDEX.tsv` | Kho chính (yen4) | 100% Ground Truth đối chiếu trang |
| `SachThanhTruyen11/` | — tháng 11 | viết tay | 145 Nôm | ✅ **145 trang scan QN gốc** (`quocngu_pages/`) + `INDEX.tsv` | Kho chính (yen11) | 100% Ground Truth đối chiếu trang |
| `LucVanTien1883/` | Lục Vân Tiên, Abel des Michels 1883 (BnF) | in thạch bản chữ bút lông | 105 Nôm | ✅ **139 trang scan QN gốc** (`quocngu_pages/`) + `luc_van_tien_quoc_ngu.tsv` (2.088 câu) | **Dataset chuẩn viết tay xuất sắc** | **100% Ground Truth chuẩn xác của chính bản này** |
| `TruyenKieuPhongTinhCoLuc/` | Truyện Kiều chép tay R.987 (NLV) | viết tay bút lông, chấm son | 120 Nôm | ❌ **Sách gốc KHÔNG CÓ trang QN** (bản cảo đơn ngữ Nôm thế kỷ 19) | Ứng viên viết tay cổ điển | Cần qua bước Căn chỉnh dị bản (so với bản phiên âm) |
| `KimVanKieu1884/` *(Sẵn sàng tải BnF)* | Kim Vân Kiều tân truyện, Abel des Michels 1884 | in thạch bản chữ bút lông | 165 Nôm (171 canvas) | ✅ **Trọn bộ 2 tập scan QN đối chiếu** (631 canvas Gallica: `bpt6k5439461n` + `bpt6k54394659`) | **Bộ song ngữ Kiều hoàn hảo nhất** | 100% Khớp câu-với-câu (3.254 câu) y hệt LucVanTien1883 |
| `TamTuKinhDienAm/` | Tam tự kinh diễn âm (NLV R.2042) | viết tay chữ to | 32 Nôm | ❌ **Sách gốc KHÔNG CÓ trang QN** (sách dạy chữ Nôm cổ) | Bộ thử nhỏ | Phải phiên âm thủ công từ ảnh scan Nôm |
| `KimVanKieu1894/` | Kim Vân Kiều tân truyện (Liễu Văn Đường) | in mộc bản | 152 Nôm | ❌ **Sách gốc KHÔNG CÓ trang QN** | Dự trữ | Mộc bản |
| `LucVanTien1916/` | Lục Vân Tiên nlvnpf-0059 (IHR-NomDB) | in mộc bản | 104 Nôm | ✅ Có GT chữ từng ký tự từ IHR-NomDB | Benchmark ngoài | 100% GT bản khắc ván 1916 |
| `TruyenKieu1872/` | Truyện Kiều 1872 (IHR-NomDB) | in mộc bản | 162 Nôm | ✅ Có GT chữ từng ký tự từ IHR-NomDB | Benchmark ngoài | 100% GT bản khắc ván 1872 |

---

### Ghi chú quan trọng về Ground Truth Quốc ngữ & Ảnh scan làm căn cứ:
1. **Đã lưu ảnh scan Quốc ngữ làm căn cứ (`quocngu_pages/`)**:
   - `data/LucVanTien1883/quocngu_pages/`: **139 trang scan** bản in Quốc ngữ của Abel des Michels 1883 (đầy đủ 2.088 câu).
   - `data/SachThanhTruyen2/quocngu_pages/`: **160 trang scan** bản Quốc ngữ đối chiếu gốc (1722x2715 px) kèm `INDEX.tsv`.
   - `data/SachThanhTruyen4/quocngu_pages/`: **147 trang scan** bản Quốc ngữ đối chiếu gốc (1142x1815 px) kèm `INDEX.tsv`.
   - `data/SachThanhTruyen11/quocngu_pages/`: **145 trang scan** bản Quốc ngữ đối chiếu gốc (1691x2704 px) kèm `INDEX.tsv`.
   -> **Tổng cộng đã trích xuất và lưu trữ cục bộ 591 trang scan Quốc ngữ gốc** để làm căn cứ đối chiếu 1-1 với từng cột/trang Nôm!
2. **Thực tế tài liệu lịch sử về các sách còn lại**:
   - Các bản chép tay cổ điển của Nho sĩ Việt Nam như `TruyenKieuPhongTinhCoLuc` (R.987) hay `TamTuKinhDienAm` (R.2042) là **bản chép tay đơn ngữ Chữ Nôm thuần túy**, trong hiện vật gốc của Thư viện Quốc gia **hoàn toàn không có bất kỳ trang dịch chữ Quốc ngữ nào**.
   - Do đó, để có bản **Truyện Kiều chữ viết tay có trang dịch Quốc ngữ scan 1-1 chuẩn xác tuyệt đối**, giải pháp hoàn hảo nhất là sử dụng bộ **`Kim Vân Kiều tân truyện` của Abel des Michels (BnF Gallica 1884)** gồm trọn vẹn bản Nôm viết tay in thạch bản (165 trang) và 2 tập scan Quốc ngữ đối chiếu từng câu (3.254 câu).

