# Script đã nghỉ hưu — GIỮ LẠI CÓ CHỦ Ý

Các script trong thư mục này **không còn chạy được nguyên trạng** và **không nằm trong
đường chạy nào** của pipeline. Chúng được giữ lại (và vẫn được git theo dõi) vì là
**bằng chứng thực nghiệm duy nhất** cho một kết luận trong luận văn.

## Vì sao không xoá

Chương 4 của luận văn có câu "đã thử và loại PP-OCRv5". Bốn script dưới đây là toàn bộ
mã đã dùng để đi tới kết luận đó. Xoá chúng thì câu kết luận mất chỗ dựa — git history
không thay thế được, vì khi hội đồng hỏi thì phải chỉ ra được file.

| Script | Việc nó làm |
|---|---|
| `test_ppocrv5_det.py` | Thử PP-OCRv5 **detection** trên trang Nôm |
| `test_ppocrv5_rec.py` | Thử PP-OCRv5 **recognition** trên crop |
| `rec_all_boxes.py` | Chạy recognition hàng loạt trên mọi box đã detect |
| `make_ppocrv5_pdf.py` | Xuất PDF đối chiếu kết quả để soi bằng mắt |

## Vì sao không chạy lại được

- Thư mục `PP-OCRv5/` và `scratch_ppocrv5_out/` đã bị xoá khỏi máy
- `.venv` hiện tại là Python 3.14, **không có `paddleocr`**
- 3/4 script chạy thẳng ở top-level với `json.load(glob(...)[0])` → `IndexError` ngay
  khi import nếu thư mục dữ liệu không tồn tại

Muốn chạy lại phải dựng môi trường Paddle riêng và tải lại dữ liệu — chỉ nên làm nếu
hội đồng yêu cầu chứng minh trực tiếp.

## Kết luận đã rút ra

PP-OCRv5 **không phù hợp** làm kênh nhận dạng cho chữ Nôm. Khớp với văn liệu: bài
fine-tune PaddleOCRv5 cho Hán-Nôm (arXiv 2510.04003, 2025) chỉ nâng exact accuracy
37,5% → 50,0%, và đó là model **mức dòng** 476 ký tự — khác granularity với bài toán
gán nhãn mức ký tự của luận văn.
