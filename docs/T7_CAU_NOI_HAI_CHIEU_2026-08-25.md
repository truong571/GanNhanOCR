# T7 — Cầu nối tự dạng HAI CHIỀU

**Phán quyết: GIỮ MỘT CHIỀU.** Không đủ bằng chứng thuận để nới một chốt chặn đặt có chủ ý.

Chạy lại: `.venv/bin/python -m pipeline.lab.t7_bridge_2way`
Đối chứng: `... --mot-chieu` (quần thể phải RỖNG)
600 ô đã xuất: `docs/T7_CAU_NGUOC_600_O.csv`

## Câu hỏi

`consensus.py:100` bắc cầu bằng `similar_dict.get(ocr_char)` — **một chiều**. `SinoNom_Similar.csv`
là bảng "20 chữ giống nhất" của mỗi chữ, tức danh sách k-láng-giềng, nên bất đối xứng: X có thể
nằm trong top-20 của Y mà Y không nằm trong top-20 của X. **600 ô** đang bị giữ lại có đúng một
cầu theo chiều ngược và không có cầu nào theo chiều xuôi.

## Không thể đo trực tiếp "đúng bao nhiêu"

Dự án không có ground truth người. Thứ đo được là: quần thể này có hành xử như tầng cầu nối
**đã được chấp nhận** hay không, dưới một kênh không tham gia gán nhãn — `sem_score`
(nghĩa Hán Unihan `kDefinition`). Kênh này **chỉ xác nhận, không bao giờ bác**: điểm > 0,05
chính xác 100%, nhưng 47,7% cặp đúng cũng cho điểm 0.

## Kết quả

Ở **mức ánh xạ** (chữ-OCR → chữ nhãn) — đơn vị phân tích đúng:

| quần thể | ánh xạ | chấm được | xác nhận | tỷ lệ | so đối chứng |
|---|---:|---:|---:|---:|---:|
| GOLD cầu XUÔI (mốc so sánh) | 1.003 | 818 | 100 | 12,2% | 1,31x |
| cầu NGƯỢC (đang xét) | 183 | 141 | 18 | **12,8%** | **1,91x** |

p = 0,059 (n hiệu dụng 141). Luật quyết định chốt trước: (a) ĐẠT · (b) **KHÔNG ĐẠT** (1,91 < 2,0)
· (c) ĐẠT → **GIỮ MỘT CHIỀU**.

Đáng ghi: cầu ngược (12,8%) **không kém** cầu xuôi (12,2%) — tầng vốn đã được giao như GOLD.
Điều không đạt là ngưỡng tuyệt đối so với đối chứng, và cả **hai** loại cầu đều tách yếu.
Đó là nhận xét về chính luật cầu nối, không riêng chiều ngược.

## Hai sai sót của chính tôi, đã sửa, ghi lại để không tái diễn

**1. Đơn vị phân tích sai — đây là chỗ lật kết luận.** Tính ở mức Ô cho 18,5% vs 7,9% = 2,34x,
p = 0,00014 → "NỚI". Nhưng ô **không độc lập**: 600 ô chỉ là 183 ánh xạ, 8 ánh xạ đầu chiếm 46%
số ô, và 84 ô được xác nhận chỉ thuộc **18** ánh xạ — riêng 3 ánh xạ đã chiếm 59/84. Đếm ở mức
ô là đếm cùng một bằng chứng vài chục lần. Tính lại ở mức ánh xạ: p = 0,059, kết luận lật về GIỮ.

**2. Cổng hợp lệ G2 đặt sai chỗ.** Bản đăng ký đầu đòi kênh chấm phải có lực trên tầng **cầu xuôi**
(mốc so sánh) và đã **KHÔNG QUA** (1,40x) — phán quyết đó giữ nguyên trong hồ sơ, không xoá.
Một cổng hợp lệ phải kiểm kênh trên quần thể ta **tin** (đối chứng dương = GOLD trực tiếp, 2,10x),
chứ không phải trên mốc so sánh; mốc so sánh yếu là **phát hiện về mốc**, không phải phép đo hỏng.
⚠️ Sai sót này được nhận ra **sau** khi đã thấy kết quả thuận, nên bản sửa là gợi ý, không phải
bằng chứng chốt.

**3. Bug trong chính lab**: `thu_thap()` ghim cứng `hai_chieu=True` khiến cờ `--mot-chieu` vô hiệu —
tức đối chứng của lab không chạy. Đã vá, đã có test hồi quy.

## Quan sát phụ đáng giữ

`SinoNom_Similar.csv` **không được sắp theo độ giống**: tương quan thứ hạng hai chiều rho = 0,24
trên 4.004 cặp. Nên thứ hạng trong danh sách không phải đại lượng độ giống và không được dùng làm
bằng chứng (cùng ràng buộc G3.1 của T5).

Ánh xạ ngược lặp nhiều nhất, để chấm tay:

| ánh xạ | ô | trang | âm | kênh nghĩa |
|---|---:|---:|---|---|
| 思 → 鬼 | 86 | 57 | quỷ | 0/86 (điểm 0 không nói lên gì) |
| 得 → 𣈜 | 54 | 50 | ngày | không chấm được (chữ Nôm riêng) |
| 至 → 圣 | 24 | 21 | thánh | **24/24 xác nhận** |
| 政 → 双 | 19 | 18 | song | **19/19 xác nhận** |
| 瑪 → 嗎 | 16 | 16 | ma | **16/16 xác nhận** |

## Việc còn để ngỏ

600 ô này chỉ người chấm mới phân định được. Chúng đã nằm sẵn ở `docs/T7_CAU_NGUOC_600_O.csv`
kèm chữ đề xuất, điểm nghĩa và số trang lặp. Nếu muốn đưa vào mẻ chấm thì đó là quyết định
riêng, không phải hệ quả của lab này.
