# Lục Vân Tiên — bản nlvnpf-0059 (in mộc bản 1916), từ IHR-NomDB (Nov 2020)
- **Loại chữ: IN MỘC BẢN, không phải viết tay** (đã kiểm tra ảnh) — không được gọi là viết tay trong luận văn.
- 104 trang (≈494×763 px, ~35 px/chữ), 2.064 câu; 1.995 câu có patch cột; bbox chỉ ở mức CỘT (VoTT).
- Ground truth: `nom_text` theo thứ tự chữ + `qn_verse` câu-với-câu → **benchmark ngoài** để đo phương pháp gán nhãn.
- Trùng nguồn với NomNaOCR "Luc Van Tien" (2.007/2.053 câu nhãn giống hệt) — chỉ giữ MỘT bản này.
- Tệp: `manifest.tsv` (1 dòng = 1 câu = 1 patch), `pages/`, `patches/`, `patches_preprocessed/`, `train.json`, `val.json`.
- Khả năng ra dataset kiểu STT: CÓ (crop chữ + OCR Nôm + QN + Unicode + bbox chữ do pipeline sinh), kèm điểm accuracy so GT; rủi ro: độ phân giải thấp 3× so STT, trang gốc là tờ đôi xếp chồng.
