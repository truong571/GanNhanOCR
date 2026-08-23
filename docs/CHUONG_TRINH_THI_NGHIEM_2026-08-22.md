# CHƯƠNG TRÌNH THÍ NGHIỆM — TỐI ƯU PIPELINE KHÔNG CẦN NHÃN NGƯỜI

**Ngày**: 2026-08-22 · **Bối cảnh**: tạm gác chấm tay; tập trung nghiên cứu–thử nghiệm để đưa mọi
bước còn lại lên mức tốt nhất đo được. Phần chấm tay sẽ rút từ bộ dữ liệu sau và nạp ngược vào.

**Tài liệu nền**: `docs/KE_HOACH_TONG_THE_2026-08-22.md` (§0 giải thích vì sao mọi số precision cũ
đã bị huỷ). Tài liệu này thay thế phần thứ tự công việc của kế hoạch đó.

---

# §0 — HAI RÀNG BUỘC QUYẾT ĐỊNH CÁCH THÍ NGHIỆM

## 0.1 Cache OCR **không bao giờ tự vô hiệu** — vừa là cơ hội, vừa là bẫy

`core/ocr/ocr_api.py:533-537`: `ocr_page` chỉ kiểm **tệp cache có tồn tại hay không**. Trường
`image_hash` được ghi vào cache nhưng **không bao giờ được đối chiếu**.

**Cơ hội**: bạn được thử mọi biến thể tiền xử lý ảnh với **chi phí API bằng 0**, miễn là không xoá
`prepared/*/detected/*_ocr_cache.json`.

**Bẫy** (lỗi âm thầm thứ 6, chưa ai ghi): nếu bạn đổi `pages/*.png` mà giữ cache, hệ thống sẽ dùng
**bbox của ảnh cũ trên ảnh mới** — toạ độ lệch hoàn toàn, không một cảnh báo nào. Đây đúng là cơ
chế đã từng gây lỗi lệch 252px trước đây.

### Luật vàng của mọi thí nghiệm

```
prepared/<sách>/pages/*.png          ← ĐÓNG BĂNG TUYỆT ĐỐI. Là đầu vào của cache OCR.
prepared/<sách>/detected/*_ocr_cache.json  ← PRIMARY DATA. Sao lưu ra ngoài repo NGAY.
────────────────────────────────────────────────────────────────────────────
Mọi thứ dưới đây được tự do thay đổi, chi phí = CPU:
  pages_denoised/ · nhị phân hoá · dò cột · tách ký tự · tinh chỉnh crop
  ma trận chi phí align · luật tier · remediation · export
```

Chuẩn hoá DPI **phải làm dưới đường này**: giữ nguyên `pages/`, nhân tỉ lệ bbox OCR bằng hệ số
tương ứng. Không được resample `pages/`.

**Việc đầu tiên, làm ngay hôm nay**: `cp -r prepared/*/detected/ ~/backup_ocr_cache_2026-08-22/`
và ghi sha256. Mất cache = phải trả tiền API lại.

## 0.2 Bổ sung ngay một chốt chặn cho chính cái bẫy này

Thêm vào `ocr_page`: khi cache tồn tại, tính md5 ảnh hiện tại và so với `image_hash` trong cache.
Lệch thì **ném lỗi** kèm hướng dẫn, không tự OCR lại (tránh vô tình đốt tiền API). Đây là 5 dòng
code chặn được một lớp lỗi âm thầm cả đề tài.

---

# §1 — BỐN HỌ TIÊU CHÍ ĐO ĐƯỢC KHÔNG CẦN NHÃN NGƯỜI

Không có nhãn không có nghĩa là không đo được. Có bốn họ tiêu chí hợp lệ, xếp theo sức mạnh:

## A. Ngữ liệu tổng hợp có đáp án — **mạnh nhất, và đang bị bỏ quên**

Repo đã có **89.898 glyph FontDiffusion** và `font_diffusion/`. Dựng một **corpus tổng hợp** với
sự thật 100% biết trước:

1. Lấy một chuỗi (chữ Nôm, âm QN) có thật từ từ điển → dựng trang 9 cột bằng glyph;
2. Áp nhiễu mô phỏng bản khắc: đứt nét, nhòe mực, vân gỗ, nghiêng, co giãn — **hiệu chuẩn tham số
   nhiễu theo thống kê mực đo được trên ảnh thật** (`ink_pct`, `stray_ink` phân bố thực);
3. Chạy toàn pipeline lên trang tổng hợp;
4. Chấm điểm **tự động** ở mọi tầng: dò cột đúng không · tách ký tự đúng chỗ không · align ghép
   đúng cặp không · tier gán đúng không.

**Đây là ground truth thật, miễn phí, sinh được hàng nghìn trang.** Giới hạn phải khai báo thẳng:
nó kiểm **hành vi thuật toán**, không kiểm độ chính xác trên ảnh thật — chính khoảng cách miền này
đã giết S3. Nhưng để trả lời "cấu hình A hay B chịu nhiễu tốt hơn" thì nó là công cụ đúng.

## B. Nhiễu loạn có kiểm soát trên dữ liệu THẬT — **rẻ và trực tiếp nhất**

Không cần dựng ảnh. Lấy dữ liệu thật, tiêm hỏng hóc **biết trước**, đo khả năng phục hồi.

**Ô neo (anchor)** = ô có `rule = s1_inter_s2_direct` (chữ OCR nằm trong danh sách đọc âm của âm
tiết, chi phí 0,0). Đây là những cặp gần chắc chắn đúng. Đo được: trung vị **58%** ô mỗi cột là ô
neo (p10 = 42%, p90 = 73%) — quá đủ tín hiệu.

> Lưu ý thiết kế: **không** dùng "cột sạch tuyệt đối" làm chuẩn — toàn corpus chỉ có **6 cột** đạt
> 100% dict-confirmed (54 ô). Chuẩn đúng là **ô neo**, không phải cột sạch.

Quy trình: tiêm hỏng vào chuỗi âm QN hoặc chuỗi chữ Nôm của một cột (bỏ 1 âm · thêm 1 âm giả · đổi
chỗ 2 âm · sai thanh · thay chữ bằng chữ nhìn giống · bỏ 1 hộp ký tự · tách 1 hộp thành 2), rồi
căn chỉnh lại và đo: **các ô neo còn sống có giữ nguyên cặp ghép cũ không?**

Kết quả là một **đường cong độ bền**: tỉ lệ neo giữ nguyên theo từng loại hỏng × từng mức hỏng, cho
từng cấu hình chi phí. Đây chính là thứ biến việc chỉnh ma trận chi phí từ "đoán" thành "đo".

## C. Bỏ-ra-kiểm-chéo tài nguyên

- **Giữ lại một phần từ điển**: bỏ ngẫu nhiên 10% mục `(âm → chữ)`, chạy lại, đếm bao nhiêu ô GOLD
  vẫn được tìm ra bằng đường khác. Cho biết **bao nhiêu nhãn GOLD treo trên một mục từ điển duy
  nhất** — một đại lượng về độ giòn chưa ai đo.
- **LOBO theo sách**: cấu hình chọn trên 2 sách có giữ được trên sách thứ 3 không.

## D. Hình học và thống kê thuần

Đo trực tiếp, không cần bất kỳ nhãn nào: `border_ink` · `stray_ink` · tỉ lệ crop trắng · IoU chồng
lấn hai hộp kề · aspect-ratio ngoại lai · tỉ lệ crop bị cắt cụt · tỉ lệ trang ra đúng 9 cột · tỉ lệ
âm QN nằm trong từ điển · bất đồng beam-vs-greedy của VietOCR · số thao tác del/ins mỗi cột.

---

# §2 — BASELINE ĐO ĐƯỢC HÔM NAY

Chốt mốc xuất phát để mọi cải tiến về sau có gốc so sánh:

| thước đo | giá trị hiện tại | dư địa |
|---|---|---|
| Âm QN hợp lệ (`is_plausible_qn_syllable`) | **99,76%** | 197 ô rác |
| Âm QN **có trong từ điển** | **99,05%** | **778 ô** — top: `giu` 112 · `ay` 44 · `ga` 33 · `trấy` 32 · `truyen` 16 · `but` 12 |
| Rác của bộ bóc marker lọt vào nội dung | `1` (15 ô) · `0` (10) · `r1` (8) | ~33 ô, lỗi parser thuần |
| Trang ra đúng 9 cột | **439/445** | 6 trang (5 trang 8 cột, 1 trang 7 cột) |
| Mâu thuẫn tự thân (cùng md5 khác nhãn) | **1 / 62.876 crop** (2 dòng) | **đã bão hoà — không còn dư địa** |
| Ô / cột | trung vị 21 · min 1 · max 40 | biên độ rộng, nghi over-segmentation |
| Ô neo mỗi cột | trung vị **58%** | — (dùng làm chuẩn đo) |
| Chất lượng hình học của crop | **CHƯA ĐO BAO GIỜ** | `crop_quality.py` chưa từng chạy — **lỗ hổng đo lường lớn nhất** |
| Độ bền căn chỉnh | **CHƯA ĐO BAO GIỜ** | chưa có chuẩn nhiễu loạn |
| Ma trận chi phí align | **0 assertion** | chưa từng được quét thử |

## 2.1 Trần dư địa — nói thẳng để khỏi đặt kỳ vọng sai

| thành phần | REVIEW | ghi chú |
|---|---|---|
| `no_s1_inter_s2` | 10.934 | chữ OCR **không** là đọc âm của âm tiết. Muốn cứu cần: mở từ điển (đã bác — vòng tròn), tín hiệu thị giác (cần nhãn), hoặc một bộ đọc Nôm thứ hai |
| `confusion_fix` 㝵 | 1.924 | đang chờ mẻ chấm mới xác nhận |
| `diverged_column` | 1.659 | **đây mới là phần align có thể cứu được bằng thí nghiệm** |
| `unconfirmed_no_s3` | 38 | không đáng kể |

**Kết luận trung thực: sản lượng đã gần trần.** Không hứa tăng số dòng. Cái tăng được bằng thí
nghiệm không nhãn là **chất lượng crop, độ bền thuật toán, tính đúng đắn, và khả năng tái lập** —
cộng với ~1.659 ô `diverged_column` là phần sản lượng thực tế còn cứu được.

---

# §3 — T0: DỰNG BÀN THÍ NGHIỆM (3–4 ngày, làm trước mọi thứ)

Không có bàn thí nghiệm thì mỗi lần thử một cấu hình sẽ mất nửa ngày thao tác tay và không so sánh
được với nhau.

| # | hạng mục | mô tả |
|---|---|---|
| T0.1 | **Sao lưu cache OCR** ra ngoài repo + ghi sha256 | primary data, mất là mất tiền |
| T0.2 | **Chốt md5 cache** (§0.2) | 5 dòng, chặn một lớp lỗi âm thầm |
| T0.3 | Vá 4 lỗi chặn của GĐ 0 (48 ô 㝵 · join `yen*/stt*` · `raise` thay midpoint · `CHECKSUMS.txt`) | nếu không, mọi phép đo về sau bị nhiễu bởi chúng |
| T0.4 | **`pipeline/lab/runner.py`** — chạy một cấu hình vào thư mục riêng `lab/run_<hash>/`, không đụng `dataset_out/` | cấu hình mô tả bằng 1 YAML; hash cấu hình = tên thư mục |
| T0.5 | **`pipeline/lab/metrics.py`** — thư viện thước đo họ D, trả về 1 dòng CSV/lần chạy | nối `crop_quality.py` vào đây (lần đầu tiên nó được dùng) |
| T0.6 | **`pipeline/lab/perturb.py`** — sinh nhiễu loạn họ B trên dữ liệu thật | 7 loại hỏng × 3 mức |
| T0.7 | **`pipeline/lab/synth.py`** — sinh trang tổng hợp họ A từ glyph FontDiffusion | hiệu chuẩn nhiễu theo `ink_pct` thật |
| T0.8 | `lab/results.csv` — mỗi dòng = 1 cấu hình × mọi thước đo | mọi biểu đồ về sau đọc từ đây |

**Hoàn thành khi**: `python -m pipeline.lab.runner --config lab/configs/baseline.yaml` chạy hết và
ghi được 1 dòng đầy đủ vào `lab/results.csv`, tái lập byte-identical khi chạy lại.

---

# §4 — SÁU CHƯƠNG TRÌNH THÍ NGHIỆM

Thứ tự **bắt buộc từ thượng nguồn xuống**: đầu ra bước trước là đầu vào bước sau, chỉnh crop trên
hộp xấu là công cốc.

---

## T1 — ĐƯỜNG QUỐC NGỮ (tách dòng · OCR · bóc marker) · 3 ngày

**Câu hỏi**: nhãn của cả đề tài neo vào âm Quốc ngữ. Âm đó sai bao nhiêu, và giảm được không?

**Thước đo chính**: tỉ lệ âm **ngoài từ điển** (hiện 0,95% = 778 ô) — thấp hơn là tốt hơn.
**Thước đo phụ**: tỉ lệ bóc đủ 9 dòng · bất đồng beam-vs-greedy · tỉ lệ âm rác (`1`, `0`, `r1`).

| yếu tố | các mức thử |
|---|---|
| Bộ tách dòng | **`projection_deskew`** (hiện tại) · `projection` · `dbnet` (nếu cài được paddle) |
| Chiến lược 2-pass | beam→text (hiện tại) · greedy→text · **lấy bản có nhiều âm trong từ điển hơn** |
| Chuẩn hoá thanh điệu | sau OCR (hiện tại) · **trước khi khớp từ điển** · cả hai |
| Bóc marker | `parser_v5` hiện tại · **+ vá rò rỉ chữ số marker** |

3 × 3 × 3 × 2 = **54 cấu hình**. Không tốn API (cache QN cũng có sẵn 445 tệp).

**Luật chọn**: cấu hình có tỉ lệ ngoài-từ-điển thấp nhất **mà không làm giảm** tỉ lệ bóc đủ 9 dòng.
Hoà thì chọn cái ít lệch so với hiện trạng nhất.

**Dư địa cụ thể đã nhìn thấy**: `giu`→`giữ` (112 ô), `ay`→`ấy` (44), `but`→`bụt` (12) là nhóm A của
`fix_tone` — đưa chuẩn hoá **lên trước** khi khớp từ điển sẽ tự động thu chúng về, thay vì vá sau
bằng một tệp mồ côi. Cộng ~33 ô rác marker. Tổng ~200–300 ô, tức giảm ngoài-từ-điển từ 0,95% xuống
~0,6%.

---

## T2 — HÌNH HỌC TRANG NÔM (DPI · nhị phân hoá · dò cột) · 4 ngày

**Câu hỏi**: STT4 ở ~201 DPI trong khi hai sách kia ~302 DPI. Chuẩn hoá thế nào là đúng, và Otsu
hay Sauvola?

**Thước đo chính**: tỉ lệ trang ra đúng 9 cột (hiện 439/445) + chất lượng hộp ở T4.
**Thước đo phụ**: độ ổn định biên cột (độ lệch chuẩn của `pitch` trong trang) · tỉ lệ pixel mực.

| yếu tố | các mức thử |
|---|---|
| Chuẩn hoá DPI | **không** (hiện tại) · resample ảnh downstream về 300 DPI · chuẩn hoá theo `pitch` (chiều rộng cột) thay vì DPI |
| Nhị phân hoá | **Otsu** (hiện tại) · Sauvola k=0,2 w=25 · Sauvola k=0,3 w=35 · adaptive-Gaussian |
| Nguồn ảnh để dò cột | `pages_denoised/` (hiện tại) · `pages/` thô |
| Dò cột | `nom_detect_v3` hybrid (hiện tại) · projection thuần |

3 × 4 × 2 × 2 = **48 cấu hình**.

> **Không dùng** công thức `pad_px = pad_frac × pitch × (300/DPI)` của bản góp ý: `pitch` đo bằng
> pixel *của chính trang đó* nên `pad_frac × pitch` **vốn đã tự chuẩn hoá**; nhân thêm sẽ làm pad
> của STT4 rộng hơn 1,5 lần tương đối. Phương án "chuẩn hoá theo `pitch`" trong lưới trên mới là
> cách đúng — và đáng thử vì nó độc lập hoàn toàn với DPI.

**Ràng buộc cứng**: mọi cấu hình phải **giữ nguyên `pages/`**; bbox OCR được nhân tỉ lệ trong
`bbox_fix`, có assertion kiểm toạ độ sau biến đổi vẫn nằm trong khung ảnh.

**Mục tiêu phụ đáng giá**: 6 trang không ra 9 cột — soi từng trang, đây là 6 ca cụ thể sửa được.

---

## T3 — CĂN CHỈNH: CHUẨN NHIỄU LOẠN + QUÉT MA TRẬN CHI PHÍ · 5 ngày · **trọng tâm**

Đây là chương trình có giá trị khoa học cao nhất trong toàn bộ danh sách, vì nó tạo ra một **phép
đo chưa từng tồn tại** trong đề tài.

### T3.1 Dựng chuẩn nhiễu loạn (họ B)

Trên mỗi cột, ô neo (`s1_inter_s2_direct`, trung vị 58%/cột) là sự thật tham chiếu. Tiêm hỏng:

| loại hỏng | mô phỏng lỗi thật nào | mức thử |
|---|---|---|
| bỏ 1 âm QN | VietOCR nuốt âm | 1 · 2 · 3 âm |
| thêm 1 âm giả | VietOCR sinh thừa | 1 · 2 |
| đổi chỗ 2 âm kề | lỗi thứ tự dòng | 1 · 2 |
| sai thanh điệu | lỗi VietOCR nhóm A | 5% · 10% · 20% âm |
| thay chữ bằng chữ nhìn giống | S1 đọc nhầm tự dạng | 5% · 10% · 20% chữ |
| bỏ 1 hộp ký tự | detector sót | 1 · 2 |
| tách 1 hộp thành 2 | detector cắt đôi | 1 · 2 |

**Chỉ số**: `anchor_retention` = tỉ lệ ô neo còn sống giữ nguyên cặp ghép. Chạy trên toàn bộ 3.998
cột × 7 loại × 3 mức, lặp 5 seed → ~420.000 phép thử/cấu hình, chạy song song, hoàn toàn tất định.

### T3.2 Quét ma trận chi phí

| tham số | mức thử | hiện tại |
|---|---|---|
| `BAND_SLACK` | 2 · 3 · 4 · thích nghi (+3 khi \|m−n\| > 2) | 2 |
| `COST_SIMILAR` | 0,20 · 0,30 · 0,45 · 0,60 | 0,30 |
| `COST_NODICT` | 0,80 · 0,90 · 1,00 | 0,90 |
| `COST_DEL` = `COST_INS` | 0,60 · 0,70 · 0,85 | 0,70 |

**144 cấu hình** × chuẩn nhiễu loạn.

### T3.3 Luật chọn — và cái bẫy vòng tròn

> ⚠️ **Tuyệt đối không tối ưu theo "số ô GOLD" hay "tỉ lệ dict-confirm".** Đó chính là luật gán
> nhãn; cấu hình nới lỏng nhất sẽ luôn thắng trong khi sinh ra nhiều nhãn sai nhất.

Chọn theo **biên Pareto** hai trục:
- trục X: `anchor_retention` trung bình có trọng số theo tần suất lỗi thật (ưu tiên các loại hỏng
  đã quan sát được trong corpus);
- trục Y: sản lượng ô GOLD, **có ràng buộc** ≥ 48.000 để loại các cấu hình quá chặt.

Kiểm chéo bằng LOBO: cấu hình chọn trên STT2+STT11 phải giữ thứ hạng trên STT4.

### T3.4 Bổ sung bắt buộc

Thêm **≥ 10 assertion** cho `anchor_align` (hiện **0**): `COST_CONFIRM` là sàn tuyệt đối · băng
không cho ghép ngoài giới hạn · chi phí đối xứng · một ca hồi quy cho mỗi hằng số · tính tất định
khi đổi thứ tự đầu vào.

### T3.5 Mục tiêu sản lượng cụ thể

1.659 ô `diverged_column` là phần align thực sự cứu được. Nếu cấu hình tốt hơn thu hồi được ~30-50%
số đó, đấy là +500–800 ô có nhãn — và quan trọng hơn, **có bằng chứng đo được rằng chúng bền hơn**.

---

## T4 — TÁCH KÝ TỰ & CHẤT LƯỢNG CROP · 4 ngày

**Câu hỏi**: chất lượng hình học của 66.000 ảnh cắt hiện ra sao? (chưa ai đo bao giờ)

### T4.1 Việc đầu tiên: đo hiện trạng

Chạy `crop_quality.py` lên **toàn bộ** crop hiện có — đây là lần đầu tiên module 208 dòng này
được dùng. Kết quả là phân bố `border_ink` / `stray_ink` / `crop_quality_flag` trên 66.589 ảnh.
**Chỉ riêng bảng này đã là một mục trong luận văn**, vì nó lượng hoá được lớp lỗi `wrong_image` mà
trước đây phải nhờ mắt người (và κ = 0,14).

### T4.2 Lưới cấu hình

| yếu tố | mức thử |
|---|---|
| `pad_frac` | 0,12 · 0,15 · **0,18** · 0,22 |
| Ngưỡng siết hộp | cứng 128 (hiện tại) · Otsu per-crop · Sauvola |
| `carve_neighbor_ink` | bật (hiện tại) · tắt |
| `resolve_overlap` (crop_quality) | bật · **tắt** (hiện tại) |
| Bộ tách | CenterNet (hiện tại) · midpoint · valley |

4 × 3 × 2 × 2 × 3 = **144 cấu hình**.

### T4.3 Hai vòng

**Vòng A** — sàng toàn bộ 144 cấu hình bằng 6 thước đo họ D, gộp thành 1 điểm z-score, lấy top 5.
**Vòng B** — 5 cấu hình còn lại chạy trên **corpus tổng hợp họ A**, nơi có đáp án hộp chính xác →
đo IoU thật với hộp đúng. Đây là chỗ ngữ liệu tổng hợp trả công.

### T4.4 Cạm bẫy

Đổi crop = đổi ảnh dưới chân bộ nhãn. Chốt cấu hình **một lần**, trước khi rút mẻ chấm tay. Ghi
`crop_quality_flag` + `stray_ink` + `border_ink` vào `labels.csv` để mẻ chấm sau này **không phải
chấm chiều CROP bằng mắt** nữa.

---

## T5 — LUẬT TIER & ĐỘ GIÒN CỦA TỪ ĐIỂN · 3 ngày

**Câu hỏi**: bao nhiêu nhãn GOLD treo trên một mục từ điển duy nhất?

| thí nghiệm | cách làm | cho biết |
|---|---|---|
| Từ điển bỏ-ra 10% × 10 lần | bỏ ngẫu nhiên 10% mục `(âm→chữ)`, chạy lại | phân bố "độ giòn": ô nào mất nhãn khi mất đúng 1 mục |
| Ngưỡng `syllable_gate` | `≥5 lần / ≥3 trang / ≥60%` (hiện tại) vs 8 tổ hợp khác | đường cong sản lượng SYLLABLE vs độ nhất quán liên trang |
| `s1_inter_s2_similar` | bật/tắt · giới hạn top-K chữ nhìn giống (K = 1 · 3 · 5 · 20) | 3.875 ô GOLD đang dựa vào luật này — K = 20 hiện tại có quá rộng không |

**Thước đo**: sản lượng · độ giòn · tính nhất quán liên sách (một cặp (chữ, âm) xuất hiện ở ≥2 sách
đáng tin hơn cặp chỉ ở 1 sách — đo được, không cần nhãn).

**Đầu ra quan trọng**: một cột `fragility` trong `labels.csv` = số mục từ điển hậu thuẫn cho nhãn
này. Ô `fragility = 1` là ứng viên số một cho mẻ chấm tay sau này — **đây là cách bạn dùng thí
nghiệm hôm nay để làm mẻ chấm mai sau rẻ hơn nhiều lần.**

---

## T6 — TÁI LẬP & BẰNG CHỨNG · 2 ngày

| việc | tiêu chí |
|---|---|
| Chạy lại toàn pipeline 2 lần | `labels_final.csv` **byte-identical** |
| Đổi thứ tự sách, đổi seed | byte-identical |
| Chạy trên bản clone sạch (Linux/Docker) | ra cùng sha256 |
| `CHECKSUMS.txt` + `evidence()` | hash hiện hành có mặt trong `EVIDENCE_INDEX.md` |
| Selftest | 414 → **≥ 450** assertion (thêm T3.4 + `crop_quality` + `lab/metrics`) |

---

# §5 — LỊCH 5 TUẦN

| tuần | việc | sản phẩm |
|---|---|---|
| 1 | T0 (bàn thí nghiệm) + sao lưu cache + vá 4 lỗi chặn | `lab/runner.py` chạy được, `results.csv` có dòng baseline |
| 2 | T1 (54 cấu hình QN) + T2 (48 cấu hình hình học trang) | ngoài-từ-điển 0,95% → mục tiêu ~0,6% · 445/445 trang 9 cột |
| 3 | T3 (chuẩn nhiễu loạn + 144 cấu hình chi phí) | đường cong độ bền · cấu hình align chốt · +10 assertion |
| 4 | T4 (144 cấu hình crop) + bảng chất lượng 66.589 ảnh | `crop_quality_flag` vào labels · cấu hình crop chốt |
| 5 | T5 (độ giòn từ điển) + T6 (tái lập) + chạy lại toàn bộ | bộ dữ liệu mới + cột `fragility` + chuỗi hash khép kín |

Sau 5 tuần: một bộ dữ liệu **tốt hơn đo được ở 4 chiều** (âm QN sạch hơn, hộp chuẩn hơn, align bền
hơn, crop sạch hơn), **có cột `fragility` để rút mẻ chấm thông minh**, và **chuỗi bằng chứng khép
kín** — đúng lúc để bắt đầu chấm tay.

---

# §6 — CÁI GÌ VẪN PHẢI CHỜ CHẤM TAY (nói thẳng)

Ba câu hỏi này **không** chương trình nào ở trên trả lời được, đừng kỳ vọng:

1. **Precision thật của từng tier.** Mọi thứ ở trên tối ưu các đại lượng *thay thế*. Chúng tương
   quan với precision nhưng không phải precision.
2. **S3 có bắt được lỗi nhãn không.** Cần phán quyết trên tier SILVER. (Nhắc lại thiết kế rẻ: đo
   trên SILVER — tỉ lệ lỗi 14–28% — chứ không trên GOLD 2–3%; n = 300 SILVER cho ~50–80 ca lỗi,
   trong khi GOLD cần 2.000–6.000 ô.)
3. **Lớp 㝵/"người" có thật sự sai hệ thống không.** 1.924 ô đang bị hạ cấp dựa trên bằng chứng đã
   huỷ.

Nhưng ba chương trình T3, T4, T5 làm cho mẻ chấm sau này **rẻ hơn nhiều**: chiều CROP chuyển sang
máy đo (T4), ô đáng chấm nhất được xếp hạng sẵn bằng `fragility` (T5), và mọi cấu hình đã chốt nên
sẽ không phải chấm lại vì đổi crop.

---

# §7 — VIỆC LÀM NGAY HÔM NAY (theo thứ tự)

1. `cp -r prepared/*/detected/ ~/backup_ocr_cache_2026-08-22/` + ghi sha256 — **5 phút, không thể
   hoãn**;
2. Thêm chốt md5 cache vào `ocr_page` (§0.2) — 5 dòng;
3. Commit toàn bộ cây làm việc (`dataset_out/`, `docs/`, `fix_tone.py`) — mọi thí nghiệm phải có
   mốc để so;
4. Vá 48 ô 㝵 + join `yen*/stt*` + `raise` thay midpoint;
5. Bắt đầu T0.4 `lab/runner.py`.
