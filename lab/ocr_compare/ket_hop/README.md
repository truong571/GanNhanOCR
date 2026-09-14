# ket_hop — kết hợp ba luồng OCR trên bộ mẫu chung

Đọc kết quả THÔ của hai engine (`../paddle/ket_qua/`, `../glm/out/`) + `ocr_char` kinhhannom trong `../mau/M*.csv`, đo các luật kết hợp (a) đồng thuận, (b) trượt R → lấy engine khác ∈R, (c) phủ quyết, (d) hợp tập ∩ R đếm cầu. Không gọi engine nào lại, không sửa gì ngoài thư mục này.

Tái lập:
```
# 1. luật + hoán vị 500 lần + ngoại suy (~60 s CPU); lần đầu chạy không có t2s_map.json thì các luật *v bằng luật thường
/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/.venv/bin/python ket_hop.py
# 2. bảng phồn→giản (OpenCC, venv paddle) rồi chạy lại bước 1 — t2s_map.json đã có sẵn trong thư mục
/tmp/venv_paddle/bin/python gen_t2s.py && /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/.venv/bin/python ket_hop.py
# 3. phân tích bổ sung (biến thể, cầu tự dạng, xác nhận K, M3)
/tmp/venv_paddle/bin/python phan_tich_them.py
```
Tệp: `ket_hop.py` (mã chính), `phan_tich_them.py`, `gen_t2s.py` + `t2s_map.json`, `ket_qua.json` (mọi số), `bang_luat.csv` (bảng luật × tập), `phan_tich_them.json`, `chi_tiet/{M1,M2,M3}_luong.csv` (ba luồng + luồng phụ từng ô), `chi_tiet/M2_luat.csv` (từng ô M2: luật nào bắn, đề xuất gì), `ket_qua.md` (báo cáo).
