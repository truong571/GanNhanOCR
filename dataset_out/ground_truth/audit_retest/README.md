# Mẻ kiểm tra lặp — đo độ ổn định của người chấm

**40 ô**, khoảng 20 phút.

## Vì sao có mẻ này

Hai mẫu **ngẫu nhiên từ cùng một dân số GOLD**, bạn chấm cách nhau một ngày:

| | Tổng lỗi | riêng `wrong_image` |
|---|---|---|
| Mẻ 03/08 (n=119) | 4,2% | 0,8% |
| Mẻ 04/08 (n=25) | 16,0% | 12,0% |

Dữ liệu y hệt nhau, chỉ tiêu chí chấm khác. Suy ra precision GOLD là **95,8%** hay
**84,0%** tuỳ buổi — chênh 12 điểm phần trăm. Chừng nào chưa biết con số đó có ổn định
không thì không tuyên bố precision nào đứng vững được.

## ĐỌC KỸ — đây là điểm mấu chốt

**Toàn bộ ô trong mẻ này bạn ĐÃ TỪNG CHẤM.** Tôi nói thẳng để bạn không thấy bị gài.

Nhưng chính vì thế, cách chấm quyết định mẻ này có giá trị hay không:

- **Đừng cố nhớ lần trước bạn đã bấm gì.** Nếu bạn cố nhớ lại, kết quả chỉ đo trí nhớ
  của bạn chứ không đo tiêu chí chấm.
- **Đừng cố tỏ ra nhất quán.** Nếu hôm nay bạn thấy khác hôm qua, hãy bấm theo cái bạn
  thấy HÔM NAY. Chuyện đổi ý là dữ liệu quý, không phải lỗi.
- Nếu lỡ nhớ ra đáp án cũ mà giờ thấy nó sai, **cứ bấm theo cái bạn thấy bây giờ**.

Không đọc ra chữ thì chọn **không chắc** — đừng đoán.

## Câu hỏi cần bạn tự trả lời trước khi bắt đầu

Trước khi chấm, hãy tự chốt trong đầu: **bao nhiêu mực thừa của chữ bên cạnh thì tính là
`wrong_image`?** Một mẩu nét nhỏ ở mép có tính không? Nửa chữ hàng xóm thì sao?

Chênh lệch giữa hai mẻ nằm gần như trọn vẹn ở câu hỏi này. Cứ giữ nguyên một tiêu chí
từ đầu đến cuối mẻ, dù tiêu chí đó là gì.

## Cách chấm

1. Mở `audit_001.html` (hoặc `audit.html`).
2. Bấm: **correct** / **wrong_label** (crop đúng 1 chữ nhưng gán sai chữ) /
   **wrong_image** (crop cắt hỏng, dính chữ khác, nhầm ô) / **unsure**.
3. Xong bấm **Download JSON** → lưu `verdicts_001.jsonl` ngay trong thư mục này.

## Thành phần (bạn KHÔNG cần biết trước khi chấm — để đây cho hồ sơ)

          orig_batch orig_verdict  n
audit_confusion_奴_nó      correct 10
audit_confusion_奴_nó  wrong_image 10
audit_confusion_奴_nó  wrong_label  4
    audit_gold_human      correct 10
    audit_gold_human       unsure  3
    audit_gold_human  wrong_image  1
    audit_gold_human  wrong_label  2

Nhóm lỗi được lấy vượt tỷ lệ có chủ đích, để đo được cả hai chiều lật. **Vì vậy mẻ này
KHÔNG phải ước lượng precision** — `design_weight` để trống nhằm chặn việc dùng nhầm.

## Mẻ này quyết được gì

| Kết quả | Kết luận |
|---------|----------|
| đồng thuận cao, κ ≥ 0,8 | tiêu chí ổn định → chênh lệch hai mẻ là do n nhỏ; dùng 95,8% được |
| κ 0,4–0,8 | ổn định vừa → phải viết rubric rõ cho `wrong_image` rồi chấm lại |
| κ < 0,4 | **chưa con số precision nào bảo vệ được** — rubric là việc bắt buộc trước mọi thứ khác |
