# Khối C — mẻ chấm mù (16/09/2026)

**1279 ô** chia **5 phiên** (mỗi phiên một tệp HTML, mở offline, tiến độ tự
lưu trong trình duyệt). Ước tính **5–8 s/ô** → mỗi phiên ≈ 25–35 phút, cả mẻ ≈ 2,5 giờ.
Không chấm hơn một phiên liền — nghỉ giữa các phiên.

## Hướng dẫn 10 dòng

1. Mở `phien_1.html` bằng trình duyệt (Chrome/Safari). Chấm xong phiên 1 mới sang phiên 2.
2. Mỗi màn là MỘT ô, **hai pha**: pha 1 chỉ có crop phóng to · vị trí trên trang (khung đỏ) · ÂM hiển thị;
   **MÃ + glyph tham chiếu chỉ hiện SAU khi bạn trả lời Q1** (pha 2). Ô không có mã thì sau Q1 tự sang ô sau.
3. **Q1** (bắt buộc, trả lời KHI CHƯA THẤY MÃ): *Ô này cắt đúng MỘT chữ, và chữ đó đọc là ÂM hiển thị?* — phím **1·2·3·4**.
4. **Q2** (chỉ khi có mã, hiện sau Q1): *MÃ Unicode hiển thị có đúng chữ trên crop?* — phím **5·6·7**.
   Sau khi thấy mã bạn vẫn được đổi Q1, nhưng câu trả lời mù được giữ riêng và lần đổi được ghi lại.
5. Trả lời xong cả hai câu, tự chuyển ô sau. `←`/`→` xem lại; nút **Ô chưa chấm** nhảy tới ô còn trống.
6. Crop khó nhìn → nhìn khung đỏ trên ảnh trang. Vẫn không đủ căn cứ → **KHÔNG RÕ**, đừng đoán.
7. Đổi ý được; mọi lần đổi đều được ghi lại — đó là dữ liệu, không phải lỗi.
8. Chấm **hết** theo thứ tự, không bỏ ô khó (bỏ chọn lọc làm khoảng tin cậy mất hiệu lực).
9. Xong phiên → bấm **Xuất JSONL** → lưu tệp vào **chính thư mục này** (tên tệp có sẵn). Sau khi xuất, phiên bị khoá.
10. Không mở thư mục `_khoa/`. Không so ô này với ô khác, không tìm ô lặp.

## Q1 — ba lựa chọn có định nghĩa

| Phím | Lựa chọn | Định nghĩa |
|:---:|---|---|
| **1** | **ĐÚNG** | một chữ trọn vẹn, và chữ đó đọc đúng là ÂM hiển thị |
| **2** | **SAI CROP** | cắt cụt > 1/3 chữ · dính ≥ 2 chữ · hoặc không phải chữ |
| **3** | **SAI ÂM** | crop là MỘT chữ hoàn chỉnh, nhưng KHÔNG đọc là âm này |
| **4** | **KHÔNG RÕ** | không đủ căn cứ, kể cả sau khi xem ảnh ngữ cảnh |

Ranh giới cần nhớ: mất **dưới 1/3** chữ hoặc dính **một chút mực** của chữ bên cạnh mà vẫn đọc ra
trọn chữ → **không** phải SAI CROP. Dị thể (viết khác nét nhưng cùng chữ, cùng âm) → **ĐÚNG**.

## Q2 — chỉ khi có mã

| Phím | Lựa chọn | Định nghĩa |
|:---:|---|---|
| **5** | **MÃ ĐÚNG** | glyph tham chiếu đúng là chữ trên crop (dị thể chấp nhận) |
| **6** | **MÃ SAI** | chữ trên crop là một chữ KHÁC — ghi mã đúng nếu biết |
| **7** | **KHÔNG RÕ** | không đủ căn cứ |

## Phiên

| Phiên | Tệp | Số ô | Tệp xuất |
|:---:|---|---:|---|
| 1 | `phien_1.html` | 256 | `verdicts_KC20260916-S1.jsonl` |
| 2 | `phien_2.html` | 256 | `verdicts_KC20260916-S2.jsonl` |
| 3 | `phien_3.html` | 256 | `verdicts_KC20260916-S3.jsonl` |
| 4 | `phien_4.html` | 256 | `verdicts_KC20260916-S4.jsonl` |
| 5 | `phien_5.html` | 255 | `verdicts_KC20260916-S5.jsonl` |

Số liệu thiết kế (tầng, cỡ mẫu, trọng số, sha256 bộ nhãn): `plan.json`. Ánh xạ ô → nguồn chỉ nằm
trong `_khoa/` và chỉ dùng ở bước ước lượng (`pipeline.ground_truth.estimate_khoi_c`, C-3).

## Pilot (C-1) — chấm TRƯỚC các phiên chính

`pilot_50.html` — 50 lượt (45 ô + 5 lặp), ~5 phút. Xuất `verdicts_KC20260916-PILOT.jsonl` vào
chính thư mục này rồi chạy `.venv/bin/python -m pipeline.ground_truth.estimate_khoi_c --pilot` để xem
dwell / κ / mồi. Ô pilot KHÔNG trùng ô nào của các phiên chính. Hướng dẫn đầy đủ:
`docs/HUONG_DAN_CHAM_KHOI_C_2026-09-16.md`.
