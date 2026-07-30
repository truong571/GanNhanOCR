# QUY TRÌNH TỔNG THỂ — CẢI THIỆN CHẤT LƯỢNG DATASET TRƯỚC BẢO VỆ

**Lập ngày**: 2026-07-29 · **Căn cứ**: 21 agent điều tra độc lập trong phiên 2026-07-28/29, mọi số đều đo trực tiếp trên code/dữ liệu thật, không trích từ tài liệu.

**Thay thế/bổ sung cho**: `THU_TU_THUC_HIEN_TONG_THE.md` (đã bị xoá — xem P0.1).

---

## NGUYÊN TẮC (giữ nguyên từ kế hoạch cũ, đã kiểm chứng vẫn đúng)

1. **Bằng chứng trước, mã sau.**
2. **Đo một lần, không đo hai lần** — mọi thay đổi ảnh hưởng mẫu số phải gộp vào MỘT mốc đóng băng.
3. **Mọi claim precision neo vào audit NGƯỜI**, không neo vào S3, không neo vào verdict AI.
4. Không tái cấu trúc lớn. Không thêm sách mới.

---

## VẤN ĐỀ CỐT LÕI — 2 CÂU HỎI, TRẠNG THÁI ĐO ĐƯỢC, VÀ BƯỚC SỬA TƯƠNG ỨNG

> Toàn bộ chất lượng dataset quy về đúng 2 câu. Mọi bước P0–P6 bên dưới tồn tại là để trả lời một trong hai.

### CÂU 1 — Ảnh có đúng là chữ Nôm đã OCR ra không? (ảnh ↔ nhãn)

#### Trạng thái đo được — câu 1

| Sự thật | Số đo | Nguồn |
|---|---|---|
| **94,3% GOLD chưa từng qua BẤT KỲ kiểm tra ảnh nào** | 46.185 / 48.969 dòng | `maybe_s3()` (`build_dataset.py:94-102`) thoát sớm khi `ocr_char` đã khớp từ điển → luật `s1_inter_s2_direct` quyết nhãn **hoàn toàn bằng OCR + từ điển đồng thuận**, không nhìn ảnh |
| Độ phủ tín hiệu ảnh `s3_cosine` | **31,7%** (21.968/69.344) | `labels_final.csv` |
| Chất lượng tín hiệu ảnh (AUC bắt-lỗi, neo verdict NGƯỜI) | **0,43** production · **0,58** retrain · p=0,093 | 826 verdict người |
| Precision NGƯỜI — GOLD | **97,1%** (801/825) | `verdicts_reanchored.csv` |
| Precision NGƯỜI — SILVER | **chưa từng đo (0 mẫu)** | — |

**Nghĩa là**: con số 98,00% của luận văn hiện đứng trên nền *"OCR và từ điển cùng nói một chữ"* — chứ **không phải** *"ảnh trông giống chữ đó"*. Đây là câu hội đồng sẽ hỏi thẳng.

#### Tại sao không sửa được bằng máy

Đã thử và đo: S3 với AUC 0,43–0,58 (p=0,093, không có ý nghĩa thống kê) **không đủ** để tự động xác nhận ảnh↔nhãn. Sub-center ArcFace + SAM + page-disjoint split chỉ đưa 0,566 → 0,580. Conformal cần 8.000–13.000 crop được audit mới dùng được. → **Chỉ người mới trả lời được câu 1.**

#### Bước sửa — câu 1

| Bước | Việc | Vai trò |
|---|---|---|
| **P1.1** | Vá `reanchor_verdicts.py` | Cứu chính con số 97,08%/98,00% đang hỏng âm thầm |
| **P3** | Công cụ audit **chọn-ép** (hiện *"chữ này là 㝵, đúng không?"* → đổi thành *"mực này là chữ nào?"*) | **Cơ chế đối chiếu ảnh↔chữ duy nhất đáng tin.** Hiện tại đang mắc thiên kiến xác nhận |
| **P2** | SILVER n≈295 (đang 0 mẫu, 15% dữ liệu) | Lấp lỗ hổng lớn nhất |
| P7 (GĐ4) | Chấm người: GOLD 450–600 · SILVER 295 · SYLLABLE 300 | Nguồn sự thật |

**Tài sản đã có, chưa dùng**: `dataset_out/fusion/s3_corpus.csv` — 62.631 dòng, đã chạy xong (5 phút, `score_s3.py --all`), phủ tín hiệu ảnh cho **toàn bộ** corpus. Dùng để **xếp hạng ưu tiên mẫu audit** và **chọn chữ gây nhiễu** cho dãy chọn-ép ở P3 — **không** dùng làm bằng chứng precision (vi phạm nguyên tắc 3).

---

### CÂU 2 — Ảnh có bao trọn chữ không? (cắt thiếu / dính chữ bên cạnh)

#### Trạng thái đo được — câu 2 (69.344 crop thật, đo bằng pixel)

| Khuyết tật | Ngưỡng chặt | Ngưỡng lỏng |
|---|---|---|
| **Cắt thiếu nửa chữ** | **0,71%** (476 crop) | 5,99% (3.985) |
| **Dính chữ hàng xóm** | **1,27%** (848 crop) | 11,06% (7.364) |

Chi tiết đo được:

- Cắt phía **trên** 4,46% — gấp **2,5 lần** phía dưới (1,79%). Nguyên nhân đã truy ra: `carve_neighbor_ink` chặn cứng tại `oy1` (`bbox_fix.py:175-183`) nên **không bao giờ** xoá được mực đã lọt vào trong hộp.
- Cờ hiện có `seg_flag` chỉ phủ **2,37% GOLD** → gần như không phát hiện được gì.
- `bbox` **không** đo được dính chữ: 99,5% dòng chồng lấn **do thiết kế** (`pitch*0.10`). Bắt buộc đọc pixel.
- Chữ Nôm ghép ⿱ vốn có 2 dải mực → khe tách dải phải ≥ `max(6px, 0.15·h)`, nếu không báo động giả hàng loạt.

#### Bước sửa — câu 2

| Bước | Việc | Kết quả |
|---|---|---|
| **P4/C1** | 3 cột pixel `stray_ink`, `border_ink`, `crop_quality_flag` | Lần đầu đo được, thay `seg_flag` |
| **P4/C2** | Tách seam cho hộp chồng lấn thật | Đã test 200 cặp xấu nhất: **147 sạch hẳn** (chồng lấn về 0), 53 từ chối tự sửa |
| **P4/C3** | Cổng demote → REVIEW | GOLD 48.969 → **45.960** (−6,14%) |

Câu 2 **tự động hoá được 100%**, không cần giờ công người.

---

### ⚠️ QUAN HỆ GIỮA 2 CÂU — ĐỪNG NHẦM

Sửa câu 2 **KHÔNG** giải quyết câu 1. Đo được trên chính 828 verdict người:

| | Tỷ lệ sai | Fisher p |
|---|---|---|
| Crop `seg_flag=tall` | 15,4% (2/13) | 0,052 |
| Crop sạch | 2,70% (22/815) | |
| Crop bị cờ cắt-thiếu (`vcut>0,08`) | 4,26% (2/47) | **0,641** (thiếu lực) |

**91,7% lỗi nhãn đã xác nhận (22/24) nằm ở crop HOÀN TOÀN SẠCH về hình học.** Tức lỗi thật chủ yếu đến từ OCR/từ điển/logic quyết nhãn, không phải độ khít của hộp cắt.

> **Hệ quả cho ưu tiên**: nếu chỉ đủ thời gian cho một việc → chọn **câu 1** (P3 + audit người). Câu 2 rẻ và nên làm, nhưng trần lợi ích chỉ khoảng **+0,3–0,5 điểm %** precision GOLD.
>
> Cảnh báo ngược lại: các số Fisher p=0,641 là **thiếu lực thống kê, không phải bằng chứng vô hại** — chỉ 47 crop bị cờ từng được người chấm; cần 400–900 mới kết luận được. Đừng viết vào luận văn rằng cắt-thiếu không ảnh hưởng.

---

## P0 — CỨU HỎA (hôm nay, ~1 giờ) 🔴 CHẶN TẤT CẢ

### P0.1 — 4 tài liệu kế hoạch đã BỊ XOÁ khỏi đĩa

Commit `513e5b4850 "update code"` (2026-07-22) đã xoá 3.635 dòng tài liệu:

| File | Trạng thái |
|---|---|
| `THU_TU_THUC_HIEN_TONG_THE.md` | MẤT khỏi đĩa |
| `DE_XUAT_HOAN_THIEN_LUAN_VAN_2026-07-20.md` | MẤT khỏi đĩa |
| `FLOW_TONG_THE_CHOT_2026-07-14.md` | MẤT khỏi đĩa |
| `KIEM_KE_FILE_VA_LO_TRINH_2026-07-20.html` | MẤT khỏi đĩa |

**Tin tốt**: lần này git CÓ giữ (khác lần mất `FLOW_TONG_THE_CHOT` trước đây phải phục dựng từ transcript). Khôi phục:

```bash
git show 513e5b4850~1:THU_TU_THUC_HIEN_TONG_THE.md > THU_TU_THUC_HIEN_TONG_THE.md
git show 513e5b4850~1:DE_XUAT_HOAN_THIEN_LUAN_VAN_2026-07-20.md > DE_XUAT_HOAN_THIEN_LUAN_VAN_2026-07-20.md
git show 513e5b4850~1:FLOW_TONG_THE_CHOT_2026-07-14.md > FLOW_TONG_THE_CHOT_2026-07-14.md
git show 513e5b4850~1:KIEM_KE_FILE_VA_LO_TRINH_2026-07-20.html > KIEM_KE_FILE_VA_LO_TRINH_2026-07-20.html
```

> Nếu việc xoá là **cố ý**, bỏ qua bước này — nhưng phải ghi vào `EVIDENCE_INDEX.md` rằng kế hoạch gốc nằm ở commit `513e5b4850~1`, vì luận văn sẽ cần trích.

### P0.2 — 51.211 crop `stt*` đang sống KHÔNG có bản sao lưu nào

Trên đĩa `dataset_out/gold/` hiện có **hai thế hệ** crop:

| Thế hệ | Số PNG | Trạng thái |
|---|---|---|
| `yen*` (cũ, chết) | 51.874 | Mồ côi — **đây mới là thứ nằm trong 3 gói backup lạnh** |
| `stt*` (đang dùng) | 51.211 | **Không có backup** |

Sao lưu ngay `dataset_out/{gold,silver,syllable}` thế hệ `stt*` ra ngoài repo TRƯỚC mọi thao tác build/xoá.

### P0.3 — 3 file chưa commit

`ArcFace/eval_human_verdicts.py`, `ArcFace/nom-embed-arcface/`, `dataset_out/fusion/s3_corpus.csv` (62.631 dòng — bằng chứng cho P2.1).

---

## P1 — VÁ ĐIỂM GÃY ĐANG CHẢY MÁU (1 ngày, không cần rebuild)

### P1.1 🔴 `reanchor_verdicts.py` đang hỏng 100% âm thầm — làm mất con số đầu bảng của luận văn

Đổi tên sách `yen→stt` đã **được commit** (`f71e4ca9c4`) và đã ngấm vào mọi artifact dữ liệu. Nhưng phía audit thì không:

| File | Book code |
|---|---|
| `labels_final.csv`, `release/`, mọi crop đang dùng | **100% `stt`** |
| `ground_truth/manifest.jsonl`, `sample_srs`, `sample_stratified`, `labels_ranked.csv` | **100% `yen`** |
| `verdicts_reanchored.csv` (846 dòng) | **828 `yen`, 0 `stt`** |

`reanchor_verdicts.py:94,104` khớp theo khoá `(book, page)` → `"yen4" != "stt4"` → **0 ứng viên cho mọi dòng** → toàn bộ 846 verdict rơi thành `orphan`, không ném lỗi nào.

**File này là nguồn DUY NHẤT của con số 97,08% / 98,00%** ghi tại `docs/BANG_SO_LIEU_CHINH_THUC.md:48-53` — tức con số đầu bảng của luận văn.

Vá: ~8 LOC, chuẩn hoá book code hai phía trước khi khớp. **Đây là việc có tỷ lệ giá trị/công cao nhất trong toàn bộ danh sách.**

### P1.2 Demote `s3_head_bank_consensus` — 0 chi phí tính toán

Luật này sinh 1.519 dòng SILVER và đo được là bệnh lý:

| Chỉ số | Giá trị |
|---|---|
| head argmax **trùng** nhãn được phát | chỉ **48,9%** |
| trung vị `head_margin` | **âm** (−0,009) |
| `bank_cos` dưới ngưỡng p10 của GOLD | **78,0%** số dòng |
| Precision (AI chấm) | **0,619** — thấp nhất trong 3 luật SILVER |

Demote 1.519 dòng này về REVIEW nâng precision SILVER (theo bằng chứng AI) từ 0,730 → **0,748**, loại bỏ 14% rủi ro nặng nhất, **không cần rebuild**.

### P1.3 Đổi tên cột `quality_flag` → `crop_quality_flag`

`suspicion.py:129` **đã** định nghĩa `quality_flag` và dùng nó làm *tầng phân tầng* (`sampling.py:25`). Nếu thêm cột cùng tên, `add_suspicion` sẽ ghi đè âm thầm.

---

## P2 — LỖ HỔNG LỚN NHẤT: SILVER CHƯA TỪNG CÓ NGƯỜI KIỂM 🔴

### Sự thật đo được

| | GOLD | SILVER |
|---|---|---|
| Số dòng | 48.969 | **10.854** (15,0% dữ liệu có nhãn) |
| Verdict NGƯỜI | 825 | **0** |
| Precision người | **97,1%** (801/825) | **chưa từng đo** |
| Precision AI (tham khảo) | — | **0,730** [0,696–0,762] |
| Kiểm định vs p0=0,90 | — | **BÁC BỎ** (cận dưới 0,701) |

98,3% dòng SILVER là **OCR bị S3 lật nhãn**. Mà S3 (checkpoint production `nom-embed/best.pt`) đo được **AUC bắt-lỗi = 0,43 trên verdict người — nghịch tương quan, tệ hơn ngẫu nhiên**.

Ước tính ~2.900 dòng SILVER sai, trong đó ~1.650 là `wrong_image`.

### Việc phải làm

Bổ sung **SILVER vào GĐ4** với cỡ mẫu tính sẵn (p=0,27):

| Độ chính xác mong muốn | n cần (đã hiệu chỉnh FPC) |
|---|---|
| ±5% | **295** |
| ±4% | 454 |
| ±3% | 781 |

Phân tầng có ép tối thiểu ~100 mẫu cho `s3_head_bank_consensus`.

> Đây là câu hỏi hội đồng chắc chắn sẽ hỏi: *"15% dataset dựa vào tín hiệu nào, đã kiểm chưa?"* — hiện chưa có câu trả lời.

---

## P3 — SỬA CÔNG CỤ AUDIT TRƯỚC KHI TIÊU 1 GIỜ CÔNG NÀO

Giao diện audit hiện tại (`audit_gold/audit_001.html`) hiển thị **sẵn nhãn** rồi hỏi `1 đúng · 2 sai-nhãn · 3 sai-ảnh · 4 không-chắc`. Đây là format dễ tổn thương nhất trước **thiên kiến xác nhận**: người chấm nhìn thấy đáp án rồi đi tìm lý do xác nhận; chữ gần giống lọt lưới.

Kế hoạch cũ ghi grid *"đã dựng sẵn"* như một lợi thế — **giờ nó là gánh nặng**.

Sửa (~40 LOC `audit_grid.py`):
- Hiện **dãy glyph**: chữ được gán + 4 chữ cạnh tranh, **xáo trộn vị trí** theo `item_id` → hỏi *"mực này là chữ nào?"*
- Dùng S3 để **chọn** chữ gây nhiễu (chỉ dùng để chọn, không hiển thị giá trị)
- Tăng `out_w` vùng scan 260 → 420px để người thấy rõ crop có bị cắt/dính không
- Thêm field `quality` **trực giao** (`clean|cut_off|neighbor_ink`), **giữ nguyên 4 giá trị verdict cũ** (5 nơi downstream đang phụ thuộc, thêm giá trị mới sẽ NaN âm thầm)

⚠️ **Đổi công cụ giữa chừng làm 2 đợt audit không gộp được** → phải chốt công cụ TRƯỚC khi bắt đầu GĐ4, và báo cáo đợt 1 (846 verdict) riêng.

---

## P4 — MỘT LẦN REBUILD DUY NHẤT (không được hai lần)

### Thứ tự bắt buộc

```
C2 (tách seam)  →  C1 (đo pixel)  →  C3 (cổng demote)
      ↓                                    ↓
   md5 đổi  →  census  →  remediation  →  confusion_fix  →  publish  →  băm lại
                                              ↓
                                    RỒI MỚI rút mẫu GĐ4
```

**Lý do từng cạnh**: C1 đo pixel nên pixel phải cuối cùng trước đã (C2 trước C1); C3 tiêu thụ cột của C1; C2 đổi `image_md5` nên census/dedup phải chạy lại.

### Nội dung

| Mã | Việc | LOC | Rủi ro |
|---|---|---|---|
| C1 | 3 cột chất lượng pixel (`stray_ink`, `border_ink`, `crop_quality_flag`) | ~45 | Trung bình |
| C2 | Tách seam cho box chồng lấn thật (prototype đã test 147/200) | ~92 | **Cao** ⚠️ |
| C3 | Cổng demote → REVIEW | ~6 | Thấp |

⚠️ **C2 rủi ro cao**: guard tại `bbox_fix.py:174-183` được thêm vào chính vì lần trước seam cắt xuyên giữa chữ, gây `ink_pct=0.0` hàng loạt. C2 cố ý mở lại vùng đó. **Bắt buộc**: chạy trên bản sao, đo `ink_pct` trước/sau, xem tay ≥50 ảnh trước khi ghi đè thật.

### Tác động phải khai trong luận văn

| | Trước | Sau |
|---|---|---|
| GOLD | 48.969 | **45.960** (−6,14%, 3.009 dòng) |
| Lớp ký tự | 1.559 | 1.536 (−23, đều là singleton) |

Đánh đổi: giảm 6,14% GOLD để loại một họ lỗi mà **hiện không cơ chế nào phát hiện được**. Phải trình bày chủ động, không né.

### Sau rebuild bắt buộc

- Viết lại toàn bộ sha256 trong `EVIDENCE_INDEX.md` §3.1/§3.2 (hiện `labels.csv` và `labels_remediated.csv` **đã lệch**, không khớp bất kỳ giá trị nào đã ghi)
- Cập nhật `BANG_SO_LIEU_CHINH_THUC.md` §1/§5/§6
- Re-baseline assertion census trong `ground_truth/selftest.py:131,133,136,140` và `remediation/selftest.py:140,143,146,149,167`
- Tag `dataset-frozen-final` → **sau tag này mọi thay đổi số liệu bị từ chối**

### Bẫy khi rút mẫu GĐ4

`ground_truth/cli.py:24` mặc định `DEFAULT_LABELS = dataset_out/labels.csv` (**không phải** `labels_final.csv`), và `cli.py:50-52` trả về `labels_ranked.csv` **cache cũ** (13/07, thế hệ `yen`, GOLD 51.200) mà không kiểm tra hạn. Bắt buộc gọi:

```bash
--labels dataset_out/labels_final.csv --force
```

---

## P5 — KHÔNG LÀM (đã đo, có bằng chứng bác bỏ)

| Việc | Lý do bác bỏ (số đo) |
|---|---|
| Vá `maybe_s3` để chạy S3 toàn bộ | **Vi phạm code-freeze** + buộc rebuild. Dùng `score_s3.py --all` ngoài luồng (đã chạy: 62.631 dòng, 5 phút) |
| Chấm điểm S3 cho SYLLABLE | **Tiền đề sai**: 6.809 dòng SYLLABLE có `label=""` theo thiết kế → không có lớp để chấm |
| Cổng conformal | Chỉ 24 mẫu sai → sàn phân giải 4% FAR; kiểm định chia đôi: **bảo đảm sai 50% số lần**. Cần 8.000–13.000 crop được audit |
| Đổi cosine → Energy/MLS | ΔAUC **+0,008**, CI [−0,003, +0,019] — không phân biệt được |
| Train S3 thêm | 0,566 → 0,580 sau Sub-center ArcFace + SAM + page-disjoint. p=0,093 (không có ý nghĩa thống kê) |
| Swap ArcFace ckpt vào production | Bị cấm kép: p=0,093 + buộc đo lại toàn bộ |
| Hồi sinh PPI / fusion Path B | Nằm trong DANH SÁCH CẤM của kế hoạch gốc |
| Dùng `bbox` để đo dính chữ | 99,5% dòng chồng lấn **do thiết kế** (`pitch*0.10`) — không mang tín hiệu |

---

## P6 — LỊCH TRÌNH: KẾ HOẠCH CŨ KHÔNG CÒN VỪA 🔴

| | Tuần |
|---|---|
| Quỹ còn (29/07 → cửa sổ bảo vệ) | **7,6 – 11,9** |
| Đường găng cần (GĐ4 ∥ GĐ5, + GĐ6/7/8) | **11 – 14** |

Mốc `HẠN CHÓT CỨNG cuối tuần 7` của kế hoạch cũ (07/09/2026) **đã không thể đạt** — GĐ4 cần 4–5 tuần và chưa khởi động.

**Chỉ mốc 3 tháng còn khả thi**, và chỉ khi kích hoạt phương án lùi **ngay bây giờ** thay vì đợi tuần 5:

1. Kích hoạt ngay dòng 122 kế hoạch cũ: *SILVER n=300 với CI rộng hơn, khai rõ giới hạn*
2. Cắt GĐ5 xuống sàn 2 tuần, **một bảng duy nhất** (kế hoạch gốc đã cho phép)
3. Bỏ hẳn tuỳ chọn nộp JOHD data paper
4. **Bắt đầu P3 (sửa công cụ audit) hôm nay** — nó chặn toàn bộ GĐ4

---

## THỨ TỰ THỰC HIỆN — TÓM TẮT MỘT TRANG

| Ưu tiên | Việc | Thời gian | Cần rebuild? |
|---|---|---|---|
| 🔴 1 | P0.1 khôi phục 4 tài liệu · P0.2 backup crop `stt*` · P0.3 commit | 1 giờ | Không |
| 🔴 2 | P1.1 vá `reanchor_verdicts.py` → cứu số 98,00% | 1 giờ | Không |
| 🔴 3 | P1.2 demote `s3_head_bank_consensus` (1.519 dòng) | 30 phút | Không |
| 🟠 4 | P3 sửa công cụ audit sang chọn-ép | 1 ngày | Không |
| 🟠 5 | P4 rebuild DUY NHẤT: C2→C1→C3→publish→băm lại→tag | 2–3 ngày | **Có** |
| 🟠 6 | Rút mẫu GĐ4 trên `labels_final` đã đóng băng (`--force`) | 1 giờ | Không |
| 🔴 7 | **GĐ4 chấm người**: GOLD n≈450–600 · SILVER n≈295 · SYLLABLE n≈300 | 4–5 tuần | Không |
| 🟡 8 | GĐ5 downstream (song song) → GĐ6 ablation → GĐ7 viết → GĐ8 bảo vệ | còn lại | Không |

**Ba ràng buộc bất di bất dịch**

1. P0 xong **trước mọi lệnh** `build` / `rm` / `git checkout`
2. P3 (công cụ audit) chốt **trước** khi chấm dòng verdict đầu tiên — đổi giữa chừng làm 2 đợt không gộp được
3. P4 (rebuild) xong **trước** khi rút mẫu GĐ4 — demote 3.009 dòng sau khi rút mẫu làm hỏng mẫu số, audit thành vô hiệu

---

## PHỤ LỤC — 4 KẾT QUẢ ÂM ĐÁNG GIÁ CHO CHƯƠNG THẢO LUẬN

Đây là tài sản, không phải thất bại. Literature Nôm gần như trống (NomNaOCR chỉ gán nhãn mức câu, không có cơ chế xác minh ảnh↔nhãn mức ký tự nào được công bố) — nên các kết quả âm đo được đàng hoàng đều là đóng góp.

1. **S3 là ranker, không phải error-gate.** Retrieval@1 ≈ 0,89 nhưng AUC bắt-lỗi 0,43 (production) / 0,58 (retrain), p=0,093. Hai metric đo hai việc khác nhau — đây là điểm phương pháp luận cốt lõi.
2. **Conformal không cứu được khi thiếu mẫu âm.** Chạy thật: 24 mẫu âm → sàn 4% FAR, bảo đảm sai 50% số lần. Định lượng được nhu cầu 200–400 mẫu âm.
3. **bbox không đo được chất lượng crop.** 99,5% chồng lấn do thiết kế — cảnh báo phương pháp cho mọi ai định dùng IoU làm chỉ số chất lượng cắt.
4. **Cảnh báo giả "76% cắt dính" của VLM.** Chữ Nôm ghép ⿱ vốn có 2 dải mực tự nhiên → ngưỡng tách dải phải ≥0,15·h, nếu không báo động giả hàng loạt.
