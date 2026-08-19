# Nguồn đã GỘP vào hai file chính (2026-08-19)

Giữ ở đây làm xuất xứ. Đã gộp bằng `pipeline.tools.merge_dicts` (hợp, không thay thế);
bản trước khi gộp nằm ở `../_backup/*.pre_merge.csv`.

| file | đóng góp thực tế | ghi chú |
|---|---|---|
| `QuocNgu_SinoNom_TongHop3.xlsx` | **+2 cặp** (𭸓 → lăm, săn) | **NHỎ HƠN** bản CSV đang dùng: thiếu 4.355 cặp. Không được thay thế, chỉ lấy phần hợp |
| `SinoNom_Similar_HVThiVien.xlsx` | +3 cặp âm Hán-Việt (瓩 ngoã, 兙/兛 khắc) | cột tự dạng của nó là **tập con thật sự** của Đạt_v0 → 0 liên kết mới |
| `SinoNom_Similar_Đạt_v0.xlsx` | **+34.881 cặp tự dạng vô hướng** (+8,8%) | nguồn duy nhất có giá trị thật trong ba file |

16 cặp bị loại vì âm không hợp lệ (`is_plausible_qn_syllable`): pinyin lẫn trong cột âm
của HVThiVien (`ti4`, `tie1`, `tiao3`), mảnh vỡ (`c`, `gì;`, `chăng?`). Nếu không lọc thì
những âm rác này thành khoá tra cứu và mở lại đúng lỗ hổng đã bịt hồi 07-14.

Kiểm lại: `.venv/bin/python -m pipeline.tools.merge_dicts` (không có `--apply` = chỉ xem).
