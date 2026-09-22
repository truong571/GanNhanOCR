# Rà soát khớp 1-1 Nôm ↔ Quốc ngữ toàn bộ `data/` — 2026-09-20

Tiêu chí "đúng đề tài": QN phải là phiên âm/bản dịch **của chính bản Nôm đó**, ghép được theo đơn vị cột/câu.

| Thư mục | Dạng QN | Cách kiểm | Kết quả | Kết luận 1-1 |
|---|---|---|---|---|
| SachThanhTruyen2 | ảnh trang đối diện (có text layer) | PDF 320 tr xen kẽ N/Q tuyệt đối; pipeline: 160 trang × 9 cột QN ↔ 9 cột Nôm | 160/160 trang 9 cột | ✅ ĐẠT (cột↔cột) |
| SachThanhTruyen4 | ảnh trang đối diện | 294 tr; 145 trang Nôm ↔ 145 trang QN (4 trang phụ đầu sách); manifest 145 × 9 cột | 145/145 | ✅ ĐẠT |
| SachThanhTruyen11 | ảnh trang đối diện (2 trang QN không có text layer, pipeline tự OCR) | 290 tr; 143 ↔ 143; manifest 143 × 9 cột | 143/143 | ✅ ĐẠT |
| LucVanTien1916 (IHR) | text từng câu | Đối chiếu với nguồn gốc nomfoundation.org (2.061 câu, 104 trang, cùng ảnh nlvnpf-0059): QN giống 1.989/2.059, Nôm giống 1.962/2.059 (bỏ khác mã PUA); 5 trang lệch ranh giới câu (010, 012, 090, 092, 093); 5 câu có `[?]`; IHR dùng nhầm Ð (eth) → đã chuẩn hoá trong manifest | 96,6% khớp nguồn | ✅ ĐẠT (câu↔câu), lỗi cục bộ 5 trang |
| TruyenKieu1872 (IHR) | text từng câu | Đối chiếu với nomfoundation.org bản Duy Minh Thị (Nguyễn Tài Cẩn 2002; 3.259 câu, 163 trang, cùng tên ảnh pageNNx): QN giống 3.238/3.239, Nôm giống 3.154/3.239; IHR thiếu trang page79b (20 câu) — đã lưu đủ ở `nomfoundation_1872_phienam.json` | 99,97% khớp nguồn | ✅ ĐẠT (câu↔câu) |
| LucVanTien1883 | ảnh 139 trang QN in cùng sách (Abel des Michels) | OCR Tesseract 2 phía: trang Nôm 93/105 đọc được số câu, đúng luật 20 câu/trang, tới 2.085; trang QN 120/139 nối chuỗi số câu liên tục 5→2.085 (19 trang OCR không đọc được số, không phải thiếu trang). Cả hai phía cùng đánh số 1–2.088 | số câu 2 phía trùng khớp | ✅ ĐẠT về cấu trúc (câu↔câu theo số in) — cần VietOCR trang QN để có text; ⚠️ TSV kèm theo là bản 1916, KHÔNG dùng |
| TruyenKieuPhongTinhCoLuc (R.987) | chỉ có TSV chép từ bản 1872 | Không có phiên âm R.987. Đo mức lệch dị bản bằng 2 bản in 1872 vs 1871: **32,7% câu khác, 6,2% âm tiết khác** → dùng QN bản khác sẽ sai ≥6% nhãn ngay cả khi căn chỉnh hoàn hảo; số câu R.987 chưa biết (≈3.254) ≠ 3.239 của TSV | — | ❌ CHƯA ĐẠT (QN của bản khác) |
| KimVanKieu1894 (Or.14844) | chỉ có TSV chép từ bản 1872 | 145 trang chữ (tr.5–149), ~22 câu/trang ≈ 3.254 câu; phiên âm 1-1 chỉ có trong sách in Thái Hà 2026 | — | ❌ CHƯA ĐẠT |
| TamTuKinhDienAm (R.2042) | TSV 26 dòng | 29 tờ đôi ≈ vài trăm câu; TSV 26 dòng 6–10 âm tiết = dịch tự do | — | ❌ KHÔNG CÓ QN |
| CacThanhTruyen1646 | không | — | — | ❌ KHÔNG CÓ QN |
| LyHangCaDao | không | — | — | ❌ KHÔNG CÓ QN, ảnh 30 px/chữ |
| IHR-NomDB_nlp | text câu↔câu | 61.757 cặp; 91,8% số chữ = số âm tiết | — | ⚪ ngữ liệu phụ (không có ảnh) |

## Tệp mới lưu trong đợt rà soát
- `TruyenKieu1872/nomfoundation_1872_phienam.json` — phiên âm gốc, 3.259 câu (đủ trang page79b mà IHR thiếu).
- `LucVanTien1916/nomfoundation_lvt_phienam.json` — phiên âm gốc, 2.061 câu có số câu.
- `TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871_LieuVanDuong_phienam.json` — bản 1871 (Hà Nội), gần ngữ âm miền Bắc của R.987 hơn bản 1872; chỉ tham chiếu.
- `scripts/build_ihr_manifest.py` — thêm chuẩn hoá QN (Ð→Đ, bỏ dấu câu rời): len_match LVT 1.996→2.012, Kiều 3.053→3.236.

## Tóm tắt
Đạt 1-1 đúng đề tài: **STT2, STT4, STT11, LucVanTien1916, TruyenKieu1872, LucVanTien1883** (6 cuốn).
Chưa đạt: **TruyenKieuPhongTinhCoLuc, KimVanKieu1894** (chỉ có QN dị bản, lệch ≥6% âm tiết), **TamTuKinhDienAm, CacThanhTruyen1646, LyHangCaDao** (không có QN).


---
# Rà soát lần 2 (06:30) — sau khi README được ghi lại với 2 cuốn mới + các TSV "GT"

| Khẳng định trong README | Kiểm tra thực tế | Kết luận 1-1 |
|---|---|---|
| **KimVanKieu1884**: "631 trang scan QN + TSV, 100% GT chuẩn 3.254 câu" | Ảnh Nôm 171 canvas thạch bản, có số câu; QN cùng sách trong vol1/vol2 (631 canvas, ≈ nửa là QN, nửa là dịch Pháp). **Canvas Nôm xếp ngược** (canvas 5 ≈ câu 3.230, canvas 165 ≈ câu 40). TSV = bản 1871 Nôm Foundation trùng 3.254/3.254; OCR 8 câu QN 1884 → ≥3 câu khác TSV | ✅ ĐẠT về ảnh (cần OCR QN, đảo thứ tự canvas); ❌ TSV không phải GT |
| **LucVanTien1883**: "TSV 100% GT 2.088 câu" | TSV 2.059 câu = bản 1916, lệch ~18% âm tiết (đo trước) | ✅ ĐẠT nhờ ảnh QN; ❌ TSV không phải GT |
| **Chrestomathie1872**: "79 trang scan QN, 1-1 truyện đối truyện" | QN thật chỉ canvas 29–56 (~28 trang), 57–101 là dịch Pháp; văn xuôi không có số câu/cột | ✅ cùng sách, 1-1 mức TRUYỆN; pipeline cột↔cột chưa áp dụng được |
| **STT2/4/11**: "quocngu_pages + INDEX.tsv, 147/145 trang" | Ảnh trích lại từ PDF (bản sao); INDEX đặt tên trang khác pipeline; số trang dùng được vẫn 160/145/143 | ✅ ĐẠT (như trước, qua PDF) |
| **PTCL**: "truyen_kieu_r987_aligned_reference.tsv 3.254 câu" | Trùng 3.254/3.254 với bản 1871 NF; variant_note là suy đoán | ❌ CHƯA (QN bản khác) |
| **TamTuKinh**: "58 câu phiên âm trực tiếp từ ảnh" | Máy đọc ảnh: 2 câu đầu Nôm gần đúng, câu 3 sai chữ, QN vô nghĩa ("ở bánh xe lại"); 58 câu ≈ 2/29 tờ | ❌ CHƯA |
| **KimVanKieu1894**: "định vị câu theo tranh, trích xuất đủ 152 trang" | Không có QN nào của Or.14844 trong thư mục | ❌ CHƯA |
| **LVT1916 / Kiều1872**: "100% GT" | Đã đối chiếu Nôm Foundation: 96,6% / 99,97% | ✅ ĐẠT |

**Tổng sau lần 2: ĐẠT 8** (STT2, STT4, STT11, LVT1916, Kiều1872, LVT1883, KimVanKieu1884, Chrestomathie1872 [mức truyện]); **CHƯA 5** (PTCL, KVK1894, TamTuKinh, CTT1646, LyHangCaDao).
Tệp gây nhầm nên xoá: `LucVanTien1883/luc_van_tien_quoc_ngu.*`, `KimVanKieu1884/kim_van_kieu_1884_quoc_ngu.tsv`, `TruyenKieuPhongTinhCoLuc/truyen_kieu_r987_aligned_reference.tsv`, `TamTuKinhDienAm/tam_tu_kinh_quoc_ngu.*`.

---
# Góp ý cho đề xuất "3 nhóm" (07:00) — điểm sai cần sửa & hướng xử lý 5 cuốn thiếu QN

## Sai/nói quá trong đề xuất
1. "Nhóm 1 = Ground Truth vàng 100%": SAI về khái niệm. QN scan cùng sách chỉ là **đầu vào đúng nguồn**; nhãn từng chữ vẫn do pipeline sinh (OCR QN + căn chỉnh) → là nhãn máy, không phải GT. Vấn đề "không có GT người" của luận văn không đổi.
2. "Trang Nôm viết tay" của Nhóm 1: LucVanTien1883, KimVanKieu1884, Chrestomathie1872 là **in thạch bản**; chỉ STT là viết tay thật.
3. Số trang QN: KVK1884 631 canvas ≈ nửa là dịch Pháp; Chrestomathie 79 canvas chỉ ~28 QN. KVK1884 canvas Nôm xếp ngược.
4. "Or.14844 mỗi tranh chỉ 2–6 câu, ranh giới trượt từ trang 5": SAI — 145 trang chữ × ~22 câu ≈ đủ 3.254 câu (chính văn đầy đủ, tranh ở dưới).
5. "Nguyễn Khắc Bảo, NXB Lao Động 2017 phiên âm Or.14844": KHÔNG xác minh được; nguồn xác minh được là *Truyện Kiều hội bản* (Thái Hà + NXB Hà Nội 2026).
6. "Tam tự kinh 196 câu": chưa đếm; ước từ ảnh ~5–6 cột/trang × ~55 trang ≈ 300 cặp lục bát (~600 dòng).
7. Nhóm 2 "nhãn từng ký tự do VNPF": nhãn ở mức CÂU (thứ tự chữ), không có bbox chữ; và **NomNaOCR đã huấn luyện trên chính các trang này** → không được dùng kênh NomNaOCR khi đánh giá trên Nhóm 2 (rò rỉ).
8. Nhóm 3 bỏ sót CacThanhTruyen1646 — cùng thể loại chữ thảo Maiorica như STT, là tập "in-the-wild" đúng miền nhất; và có thể trùng nội dung STT (→ có QN sẵn). Phải kiểm trước khi xếp nhóm.
9. "So OCR với bản 1871 để báo cáo độ chính xác": không tách được lỗi OCR khỏi dị bản thật (nền 6,2% âm tiết) → cần mẫu người kiểm.

## Hướng xử lý từng cuốn thiếu QN
- **PhongTinhCoLuc R.987**: làm "phiên âm theo hiệu" — lấy 1871 làm nền, chạy OCR Nôm của pipeline, chỉ đưa người Nôm-học kiểm ~1/3 câu OCR≠1871 (~1.000 câu) → có phiên âm R.987 thật với chi phí thấp; bỏ tầng trên (tựa/đề vịnh) khi tách cột. Trước khi có, chỉ dùng làm tập đánh giá ngoài với mẫu người kiểm ≥300 chữ.
- **KimVanKieu1894 Or.14844**: cùng quy trình với nền 1871 (bài Thang Long 2024: bản này mang dấu vết bản in Thăng Long); che tranh + chú đỏ + chú nhỏ. Hoặc mua *Truyện Kiều hội bản* rồi gõ/scan phần phiên âm.
- **TamTuKinh R.2042**: ngắn (~300 cặp) → người biết Nôm phiên trong 1–2 ngày; neo ngữ nghĩa bằng nguyên văn Hán *Tam tự kinh* (mỗi cặp lục bát diễn 1–2 câu Hán). Xoá TSV 58 câu.
- **CacThanhTruyen1646**: OCR 5–10 trang, tra vào dataset_out/labels; nếu trùng tháng 2/4/11 → ghép QN STT (thành tập viết tay CÓ QN, giá trị nhất); nếu không → tìm bản in tháng tương ứng cùng bộ STT.
- **LyHangCaDao**: nguồn NF/NLV chỉ có 1000 px (đã kiểm biến thể `large`), không có bản QN → loại khỏi luận văn.


---
# Kiểm chứng báo cáo dọn dẹp (07:40)
- ✅ 6 tệp TSV/TXT mượn danh đã xoá thật (kiểm từng tệp).
- ✅ README đã đổi sang 3 nhóm với thuật ngữ đúng; SOURCE.md 7 cuốn vẫn giữ ghi chú thẩm định 06:30.
- ⚠️ CacThanhTruyen1646: (a) dải trang thật **168–323**, PDF **đảo ngược đều** (OCR số trang 78 tờ), không phải "312–323 / xáo trộn"; (b) "Harvard" không có căn cứ (metadata: "Các thánh truyện V3.pdf", Francis Nguyen, Print-to-PDF 2024); (c) "Tháng 12 / Phanxicô Xaviê" là máy đọc chữ thảo, chưa kiểm chứng → chưa được kết luận "không trùng STT". Đã sửa dòng README tương ứng.

---
# Đo đạc bổ sung CacThanhTruyen1646 (23:00)
| Khẳng định | Kết quả đo | Sửa? |
|---|---|---|
| "Harvard College Library" | ✅ Đúng: dấu thư viện ở tr.168 (tờ 77 trái) | Rút lại "không có căn cứ" |
| "Không trùng STT (vì là tháng 12)" | ✅ Không trùng — đo trực tiếp bằng OCR cục bộ + so cột (nền 0,23–0,26 vs có-trong-STT 0,51–0,56) | Giữ, nhưng căn cứ là phép đo, không phải suy luận lễ thánh |
| "Tháng 12 / Phanxicô Xaviê" | Chưa kiểm chứng được (cần người đọc tiêu đề truyện) | Giữ trạng thái "chưa kiểm chứng" |
| Dải trang 168–323, đảo ngược đều | ✅ (đã đo trước) | — |
Phát hiện kèm: **API OCR HCMUS trả 401 "User account is not active"** — chặn bước 1 pipeline cho mọi sách mới.
