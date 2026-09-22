# Lục Vân Tiên — bản nlvnpf-0059 (in mộc bản 1916), từ IHR-NomDB (Nov 2020)
- **Loại chữ: IN MỘC BẢN, không phải viết tay** (đã kiểm tra ảnh) — không được gọi là viết tay trong luận văn.
- 104 trang (≈494×763 px, ~35 px/chữ), 2.064 câu; 1.995 câu có patch cột; bbox chỉ ở mức CỘT (VoTT).
- Ground truth: `nom_text` theo thứ tự chữ + `qn_verse` câu-với-câu → **benchmark ngoài** để đo phương pháp gán nhãn.
- Trùng nguồn với NomNaOCR "Luc Van Tien" (2.007/2.053 câu nhãn giống hệt) — chỉ giữ MỘT bản này.
- Tệp: `manifest.tsv` (1 dòng = 1 câu = 1 patch), `pages/`, `patches/`, `patches_preprocessed/`, `train.json`, `val.json`.
- Khả năng ra dataset kiểu STT: CÓ (crop chữ + OCR Nôm + QN + Unicode + bbox chữ do pipeline sinh), kèm điểm accuracy so GT; rủi ro: độ phân giải thấp 3× so STT, trang gốc là tờ đôi xếp chồng.

## Xác minh nguồn (20/09 04:40)
- NLVNPF-0059 = **R.403 "Vân Tiên cổ tích tân truyện", Liễu Văn Đường tàng bản, khắc in Khải Định nguyên niên (1916)**, 53 tờ (Nôm Foundation volume/62). QN câu-với-câu trong manifest.tsv lấy từ trang dự án Lục Vân Tiên của Nôm Foundation (văn bản web, không có ảnh trang QN — sách gốc không có phần quốc ngữ).
