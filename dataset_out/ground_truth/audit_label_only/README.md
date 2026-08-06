# Mẻ audit NHÃN — chỉ một câu hỏi

**250 ô**, mẫu ngẫu nhiên đơn giản từ 50,031 hàng GOLD chưa từng được chấm
(255 ô đã chấm ở các mẻ trước đã bị loại).

## Chỉ có MỘT câu hỏi

> **Nhãn được gán có đúng là chữ viết trong ô này không?**

Ba lựa chọn, không có lựa chọn nào khác:

- **1 · nhãn ĐÚNG** — chữ trong ô đúng là chữ được gán
- **2 · nhãn SAI** — chữ trong ô là một chữ KHÁC
- **3 · không đọc được** — không đủ căn cứ để kết luận

## Điều đã đổi so với các mẻ trước — đọc kỹ

Trước đây có thêm lựa chọn "sai ảnh" cho crop cắt hỏng. **Lựa chọn đó đã bỏ.**

Lý do: kiểm tra lặp cho thấy phán đoán "crop có sạch không" của con người không tái lập
được (κ = 0,14 — chính bạn đảo verdict trên 14/40 ô). Nó cũng chính là thứ làm precision
nhảy từ 95,8% xuống 84,0% giữa hai buổi. Nay chất lượng crop được đo bằng hình học trên
toàn bộ 69.440 crop, và kết quả là chỉ **14 ô (0,02%)** hỏng kết cấu thật.

**Vì vậy: crop bẩn KHÔNG còn là một verdict.**

- Crop dính chút mực của chữ bên cạnh mà vẫn đọc ra chữ → **nhãn ĐÚNG** (nếu nhãn khớp).
- Crop cắt khó nhìn → dùng **ảnh ngữ cảnh** (khung đỏ trên trang gốc) để đọc chữ.
- Chỉ chọn **không đọc được** khi thật sự không đọc nổi chữ đó là gì, kể cả khi nhìn
  ảnh ngữ cảnh.

Đừng phạt một nhãn đúng chỉ vì khung cắt xấu. Câu hỏi ở đây là về **chữ**, không phải về
khung.

## Một lưu ý từ dữ liệu trước

Ở kiểm tra lặp, 5 trên 6 lần bạn bấm "sai nhãn" thì lần sau chính bạn đổi lại thành đúng —
tức xu hướng là **gọi quá tay**. Nếu lưỡng lự giữa "sai nhãn" và "đúng", hãy nhìn kỹ ảnh
ngữ cảnh và bộ ứng viên từ điển trước khi quyết. Vẫn lưỡng lự thì chọn **không đọc được**
chứ đừng chọn "sai nhãn" — ô không đọc được bị loại khỏi phép tính, còn "sai nhãn" thì bị
tính là lỗi.

## Cách chấm

1. Mở `audit_001.html` (và các phần sau).
2. Bấm phím **1** / **2** / **3**, hoặc bấm chuột. Tiến độ tự lưu.
3. Chấm hết MỌI phần rồi bấm **Download JSON** → lưu `verdicts_001.jsonl` ngay trong
   thư mục này.

Chấm hết theo đúng thứ tự, đừng bỏ ô khó — bỏ chọn lọc sẽ phá tính ngẫu nhiên của mẫu.

## Mẻ này siết được CI tới đâu (n = 250, độ tin cậy 95%)

| Số ô sai nhãn | Precision | CI95 | Cận dưới một phía |
|--------------:|----------:|------|------------------:|
| 0 | 100.0% | [98.5% · 100.0%] | 98.8% |
| 1 | 99.6% | [97.8% · 100.0%] | 98.1% |
| 2 | 99.2% | [97.1% · 99.9%] | 97.5% |
| 3 | 98.8% | [96.5% · 99.8%] | 96.9% |
| 5 | 98.0% | [95.4% · 99.3%] | 95.8% |
| 8 | 96.8% | [93.8% · 98.6%] | 94.3% |

So với mẻ trước (n=116, CI [93,9% · 99,8%]), mẻ này thu hẹp khoảng tin cậy khoảng một nửa.

## Sau khi chấm

```
.venv/bin/python -m pipeline.ground_truth --out dataset_out/ground_truth/audit_label_only estimate \
    --verdicts dataset_out/ground_truth/audit_label_only \
    --manifest dataset_out/ground_truth/audit_label_only/manifest.jsonl \
    --design srs --p0 0.97
```
