# BÀN GIAO VIỆC CHẤM KHỐI C CHO NGƯỜI CHẤM NGOÀI

**Viết cho:** người nhận việc chấm (biết đọc chữ Nôm/Hán, không cần biết lập trình) và người bàn giao (chủ đề tài).
**Việc:** nhìn 1.279 ô ảnh chữ Nôm, trả lời 1–2 câu trắc nghiệm mỗi ô, xuất 6 tệp kết quả. Tổng ≈ **2,5–3 giờ**, rải
3–5 ngày, không cần mạng, không cần cài gì.

---

## PHẦN A — DÀNH CHO NGƯỜI BÀN GIAO (chủ đề tài)

### A1. Gói gửi đi (chỉ gửi đúng những tệp này)

Từ `dataset_out/human_audit/khoi_c_2026-09-16/`:

| Gửi | Tệp | Cỡ |
|---|---|---|
| ✅ | `pilot_50.html` | 1,3 MB |
| ✅ | `phien_1.html` … `phien_5.html` | 6–6,6 MB mỗi tệp |
| ✅ | `README.md` (hướng dẫn 10 dòng + định nghĩa) | |
| ✅ | `docs/HUONG_DAN_CHAM_KHOI_C_2026-09-16.md` (hướng dẫn đầy đủ, **có 6 ảnh ví dụ**) | |
| ❌ **KHÔNG** | `_khoa/` (ánh xạ ô → nguồn/đáp án — người chấm biết là hỏng mù) | |
| ❌ | `plan.json`, `plan_pilot.json` (thiết kế tầng — không cần, tránh gợi ý) | |
| ❌ | bất kỳ tệp nào khác trong repo (`labels*.csv`, `re-dataset/`, từ điển của pipeline) | |

Cách gửi: nén 7 tệp trên thành một zip (~33 MB), gửi qua Drive/USB. Mọi ảnh đã nhúng trong HTML → mở là chạy.

### A2. Yêu cầu với người chấm

- Đọc được chữ Nôm/Hán viết tay ở mức nhận mặt chữ (Q2 hỏi *mã có đúng chữ trên crop không* → phải biết chữ).
- Không phải là tác giả pipeline; chưa từng xem `labels.csv`, bảng QĐ-01, hay bất kỳ nhãn máy nào của bộ này.
- Có thể tra **từ điển Nôm bên ngoài** (giấy/web) khi cần — cho phép; **không** được tra tệp nào của đề tài.
- Tốt nhất: **một người chấm hết 5 phiên** (để đo độ nhất quán nội tại bằng 100 ô lặp ẩn). Nếu muốn có κ liên người:
  mời **người thứ hai chấm thêm đúng `phien_1.html`** (256 ô, ≈40 phút) và xuất với tên tệp thêm hậu tố `-R2`.

### A3. Nhận về và chạy ước lượng (chỉ chủ đề tài)

Nhận 6 tệp `verdicts_KC20260916-PILOT.jsonl`, `-S1` … `-S5.jsonl` (không sửa tay), chép vào
`dataset_out/human_audit/khoi_c_2026-09-16/`, rồi:

```
# sau pilot (báo dwell / κ / mồi — đạt ngưỡng mới cho chấm tiếp)
.venv/bin/python -m pipeline.ground_truth.estimate_khoi_c --pilot
#   -> dataset_out/human_audit/khoi_c_2026-09-16/pilot_report.md
# sau đủ 5 phiên (chạy được khi thiếu phiên — sẽ báo số ô chưa chấm)
.venv/bin/python -m pipeline.ground_truth.estimate_khoi_c
#   -> docs/KET_QUA_KHOI_C_<ngày>.md + .json
```

Ngưỡng chấp nhận một phiên (script tự báo động): mồi ≥ 90 % đúng · ô < 1,5 s không quá vài % · κ lặp ẩn Q1 ≥ 0,6.
Phiên rớt ngưỡng → chấm lại phiên đó (trình duyệt khác hoặc xoá dữ liệu trang), không "sửa" kết quả.
Kết quả chỉ vào `BANG_SO_LIEU` sau khi bạn duyệt `KET_QUA_KHOI_C` (bước A-13 `evidence()`).

---

## PHẦN B — DÀNH CHO NGƯỜI CHẤM

### B1. Bạn đang làm gì

Mỗi ô là một chữ Nôm viết tay được máy cắt ra từ trang sách và ghép với một **âm Quốc ngữ** (và có thể một **mã chữ**).
Bạn kiểm tra máy làm đúng không. Không có "chuẩn" nào khác ngoài mắt bạn — đừng đoán ý máy.

### B2. Ba điều cấm

1. **Không mở** thư mục `_khoa/` hay bất kỳ tệp nào ngoài 7 tệp được giao.
2. **Không bỏ ô khó.** Bỏ chọn lọc làm hỏng khoảng tin cậy. Không đủ căn cứ → chọn **KHÔNG RÕ**.
3. **Không so ô này với ô khác, không cố tìm ô lặp.** Một số ô xuất hiện hai lần — đó là cố ý để đo độ nhất quán;
   cứ chấm như ô mới.

### B3. Chuẩn bị (1 phút)

- Máy tính (không dùng điện thoại), trình duyệt **Chrome hoặc Safari**, màn hình ≥ 13". Không cần mạng.
- Dùng **một trình duyệt duy nhất** cho cả mẻ; đừng xoá dữ liệu trang/duyệt ẩn danh — tiến độ lưu trong trình duyệt.
- Ngồi chỗ sáng, nghỉ giữa các phiên; **không chấm hai phiên liền**.

### B4. Thứ tự làm

| Bước | Tệp | Số ô | Thời gian | Xuất ra |
|:---:|---|---:|---|---|
| 0 | `pilot_50.html` | 50 | ≈ 5 phút | `verdicts_KC20260916-PILOT.jsonl` → gửi ngay cho chủ đề tài, **chờ phản hồi** rồi mới chấm tiếp |
| 1 | `phien_1.html` | 256 | 25–35 phút | `verdicts_KC20260916-S1.jsonl` |
| 2 | `phien_2.html` | 256 | 25–35 phút | `…-S2.jsonl` |
| 3 | `phien_3.html` | 256 | 25–35 phút | `…-S3.jsonl` |
| 4 | `phien_4.html` | 256 | 25–35 phút | `…-S4.jsonl` |
| 5 | `phien_5.html` | 255 | 25–35 phút | `…-S5.jsonl` |

### B5. Màn hình một ô — HAI PHA

**Pha 1** (mù): bạn thấy **crop phóng to**, **vị trí trên trang** (khung đỏ, kèm 2 chữ trên/dưới), và **ÂM** hiển thị
(ví dụ *"người"*). Trả lời **Q1** bằng phím **1 / 2 / 3 / 4**.

**Pha 2** (chỉ với ô có mã): sau khi bấm Q1, màn hình hiện thêm **MÃ chữ** và **glyph tham chiếu** (chữ in mẫu của mã ấy).
Trả lời **Q2** bằng phím **5 / 6 / 7**. Ô không có mã → tự sang ô sau.

Sau khi thấy mã bạn **được phép đổi Q1** (câu trả lời mù vẫn được giữ riêng) — đổi ý là dữ liệu, không phải lỗi.
`←`/`→` để xem lại; nút **Ô chưa chấm** nhảy đến ô còn trống.

### B6. Q1 — *"Ô này cắt đúng MỘT chữ, và chữ đó đọc là ÂM hiển thị?"*

| Phím | Chọn | Khi nào |
|:---:|---|---|
| **1** | **ĐÚNG** | trong khung là **một chữ trọn vẹn**, và chữ đó đọc **đúng** là âm hiển thị (dị thể — viết khác nét nhưng cùng chữ, cùng âm — vẫn ĐÚNG) |
| **2** | **SAI CROP** | khung **cắt cụt hơn 1/3** chữ · **dính từ hai chữ** trở lên · hoặc **không phải chữ** (vệt mực, khoảng trống, cột bên) |
| **3** | **SAI ÂM** | khung là **một chữ hoàn chỉnh**, nhưng chữ đó **không đọc** là âm hiển thị (thường là chữ của ô trên/dưới bị ghép nhầm) |
| **4** | **KHÔNG RÕ** | không đủ căn cứ kể cả sau khi nhìn khung đỏ trên trang |

Ranh giới hay gặp:
- Mất **dưới 1/3** chữ, hoặc dính **một chút mực** của chữ bên cạnh mà vẫn nhận ra trọn chữ → **1 ĐÚNG** (không phải SAI CROP).
- Chữ đúng nhưng âm hiển thị là âm của chữ trên/dưới → **3 SAI ÂM** (đây chính là lỗi quan trọng nhất cần bắt).
- Vừa cắt cụt nặng vừa sai âm → **2 SAI CROP** (ưu tiên lỗi ảnh).
- Sáu ảnh ví dụ có chú giải: `HUONG_DAN_CHAM_KHOI_C_2026-09-16.md` §3.

### B7. Q2 — *"MÃ chữ hiển thị có đúng chữ trên crop?"* (chỉ khi ô có mã)

| Phím | Chọn | Khi nào |
|:---:|---|---|
| **5** | **MÃ ĐÚNG** | glyph tham chiếu đúng là chữ trên crop (dị thể/khác kiểu nét chấp nhận) |
| **6** | **MÃ SAI** | chữ trên crop là **một chữ khác** — nếu biết, gõ chữ đúng vào ô nhập |
| **7** | **KHÔNG RÕ** | không đủ căn cứ |

Lưu ý: Q2 hỏi **mã có khớp hình chữ** không, không hỏi âm. Một ô có thể Q1 = SAI ÂM mà Q2 = MÃ ĐÚNG (chữ đúng, ghép âm sai).

### B8. Xuất kết quả

Cuối mỗi phiên bấm **Xuất JSONL** → trình duyệt tải về tệp có sẵn tên (`verdicts_KC20260916-S1.jsonl` …). Sau khi
xuất, phiên bị **khoá** (không sửa được nữa — đúng ý). Gửi đúng tệp đó, **không mở/sửa/đổi tên**. Nếu lỡ đóng trình
duyệt giữa chừng: mở lại đúng tệp HTML, tiến độ vẫn còn (trừ khi đã xoá dữ liệu trang).

### B9. Sự cố

| Tình huống | Làm gì |
|---|---|
| Ảnh không hiện / trang trắng | mở bằng Chrome; kiểm tệp tải đủ (6 MB); không mở từ trong zip |
| Bấm nhầm | dùng `←` quay lại, chọn lại — lần đổi được ghi, không sao |
| Mất tiến độ | báo chủ đề tài kèm tên phiên; chấm lại phiên đó từ đầu |
| Nghi hai ô giống nhau | cứ chấm độc lập như ô mới |
| Không biết chữ đó là gì | tra từ điển ngoài được; vẫn không biết → KHÔNG RÕ |

---

## PHẦN C — SAU KHI CÓ KẾT QUẢ (chủ đề tài)

`estimate_khoi_c` sẽ in: precision Q1 (crop + thanh ghi) và Q2 (mã chữ) theo tầng với Wilson 95 %, gộp GOLD/USABLE
theo trọng số tầng (Horvitz–Thompson, N = 70.326); κ lặp ẩn (100 cặp, KTC bootstrap); độ chính xác trên 90 ô mồi;
cổng B-3 (T1 = 300 ô: cận trên Clopper–Pearson của tỉ lệ lỗi ≤ 3 % → đủ điều kiện thêm rule `visual_syl_gate`);
T2 chuỗi trượt (bao nhiêu chuỗi thật); T3 (19 ô B-2: âm cũ hay mới đúng; 30 ô B-5: hộp khoá đúng crop?); T6 40 ô
"người" chưa khoá (đầu vào QĐ-01a). Mọi số chỉ công bố sau khi bạn duyệt tệp `docs/KET_QUA_KHOI_C_<ngày>.md`.
