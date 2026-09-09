# Nghiên cứu đang dở — tạm ngưng 2026-09-09

Hai mẻ nghiên cứu bị **tạm dừng giữa chừng** theo yêu cầu. Kết quả của các agent ĐÃ HOÀN THÀNH
được lưu ở đây; các agent chưa xong thì chưa có gì.

## Chạy tiếp

Mỗi mẻ resume được — agent nào đã xong sẽ lấy từ cache, không phải chạy lại.

```
Workflow({
  scriptPath: "<thư mục phiên>/workflows/scripts/toi-uu-chat-luong-khong-gioi-han-thoi-gian-wf_cea32642-fb0.js",
  resumeFromRunId: "wf_cea32642-fb0"
})

Workflow({
  scriptPath: "<thư mục phiên>/workflows/scripts/sinh-lai-kho-glyph-hanh-thao-wf_94b9b79f-6e1.js",
  resumeFromRunId: "wf_94b9b79f-6e1"
})
```

⚠️ `resumeFromRunId` chỉ dùng lại được cache **trong cùng một phiên**. Nếu mở phiên mới thì
hai mẻ này phải chạy lại từ đầu — script vẫn còn nguyên ở thư mục `workflows/scripts/`.

## Đã xong

### `nghien_cuu_chat_luong.json` — 4/8 agent
Mẻ "nâng chất lượng khi thời gian không còn là ràng buộc".

| | Hướng | Trạng thái |
|---|---|---|
| 2 | Trần trên của khối OCR đọc sai tự dạng | ✅ xong |
| 3 | Nâng tầng 2 lên "đã đo" (LOBO bỏ cả quyển sách) | ✅ xong |
| 4 | Cứu lại nhãn chữ cho 3.875 ô `nghia_consensus_tu_silver` | ✅ xong |
| 5 | Chất lượng ảnh crop xét lại cho chữ thảo | ✅ xong |
| 1 | Sinh kho tham chiếu theo chữ viết (`font_diffusion`) | ❌ chưa |
| 6 | Thí nghiệm hạ nguồn như một chương luận văn | ❌ chưa |
| — | 2 agent phán xử (kiểm số + xếp lộ trình) | ❌ chưa chạy |

### `sinh_lai_kho_glyph.json` — 3/8 agent
Mẻ "sáu kịch bản sinh lại kho glyph cho khớp nét bút hành-thảo".

| | Kịch bản | Trạng thái |
|---|---|---|
| 0 | Chẩn đoán: truy hồi hỏng ở đâu | ✅ xong |
| D | Không sinh ảnh, chỉ dùng crop thật | ✅ xong |
| E | Đổi bài toán: chọn trong `R(âm)` thay vì 4.700 lớp | ✅ xong |
| A | Sinh lại với giao thức phong cách tốt hơn | ❌ chưa |
| B | Không sinh lại, tăng cường ảnh kéo hai miền gần nhau | ❌ chưa |
| C | Crop lớp A làm phong cách, sinh cho lớp B | ❌ chưa |
| — | 2 agent phán xử (kiểm số + chốt phương án) | ❌ chưa chạy |

## Đọc kết quả thế nào

⚠️ **Chưa qua phản biện.** Hai agent phán xử của mỗi mẻ — vốn có nhiệm vụ chạy lại mọi phép đếm,
bắt lift ảo và bắt cộng dồn tập ô chồng nhau — đều CHƯA chạy. Trong các mẻ trước, chính vòng phản
biện này đã bác những khẳng định lớn: một luật khai cứu 4.651 ô hoá ra lift chỉ 1,16×; tám báo cáo
cộng dồn 21.111 ô nhưng hợp thực chỉ 9.893.

Vì vậy **đừng trích số nào từ hai tệp JSON này vào luận văn** cho tới khi chạy hết phán xử.
Chúng là ghi chép trung gian, không phải kết luận.
