# Bộ mẫu chung so sánh OCR (seed 20260911)

Nguồn: dataset_out/labels.csv (82780 ô). Trọng tài R(âm) = qn_to_nom[syllable] từ Dict/QuocNgu_SinoNom.csv
(nạp bằng core.text.dictionary.load_qn_to_nom, chuẩn hoá dấu như pipeline; 7464 âm).

| Tập | n | Định nghĩa |
|---|---|---|
| M1 | 600 | GOLD / s1_inter_s2_direct, ocr_char ∈ R(âm) — đối chứng; 200 ô/sách, rải đều trang (round-robin) |
| M2 | 600 | ocr_char ∉ R(âm), tier SILVER/SYLLABLE, R(âm) ≠ ∅ — PHÉP ĐO CHÍNH; 200 ô/sách, rải đều trang. (Pool 17249, trong đó 258 ô có âm ngoài từ điển → R rỗng, đã loại) |
| M3 | 300 | ocr_char = 㝵, syllable = người — phá sản; ngẫu nhiên từ 1183 ô. LƯU Ý R('người') chứa cả 㝵 lẫn 𠊚 nên R không phân xử được, phải đo phân bố |
| M4 | 20 | cột nguyên vẹn, cắt từ prepared/<sách>/pages/<trang>.png theo union bbox chữ trong cache kinhhannom (+8px); cột chọn ngẫu nhiên trong trang |

Cột `image` của M1-M3 là đường dẫn tương đối trong dataset_out/. `R_candidates` = các chữ trong R(âm), phân cách '|'.
M4: `image` tương đối trong thư mục này (cot/), `kinhhannom_text` = chuỗi kinhhannom đọc ra cho cột, `kinhhannom_n` = số chữ.
