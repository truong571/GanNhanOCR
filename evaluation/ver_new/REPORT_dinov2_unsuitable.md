# Báo cáo: DINOv2 KHÔNG phù hợp để so khớp 2 ảnh chữ Nôm

**Ngày:** 2026-06-01 · **Phạm vi:** tín hiệu thị giác S3 trong pipeline gán nhãn
(`core/ranking/dinov2_ranker.py`, model `dinov2_vitb14_reg`, ngưỡng "match" 0.75).

---

## 1. Kết luận (TL;DR)

**DINOv2 (zero-shot) KHÔNG dùng được để xác định một crop chữ Nôm là chữ nào**, vì
embedding của nó **không phân biệt được các chữ Nôm khác nhau**: cosine giữa hai
chữ *khác nhau* xấp xỉ cosine giữa hai ảnh *cùng một chữ*. Hệ quả trực tiếp: tầng
**SILVER** (dựa S3) và **tier-3 visual matching** của pipeline gốc cho nhãn ≈ ngẫu
nhiên. **Phải thay** bằng model embedding **huấn luyện riêng** cho glyph Nôm.

---

## 2. Bối cảnh

Pipeline gán nhãn dùng 3 tín hiệu cho mỗi crop: **S1** chữ OCR, **S2** từ điển
QN↔Nôm, **S3** khớp thị giác (DINOv2 cosine giữa crop và glyph tham chiếu
FontDiffusion). S3 được kỳ vọng "phá thế" khi S1/S2 không thống nhất — nhưng chỉ
đúng nếu cosine của DINOv2 **phân biệt được chữ này với chữ kia**. Báo cáo này
kiểm định chính giả định đó.

Quy ước điểm: cosine ánh xạ về [0,1] bằng `(cos+1)/2` (đúng như
`dinov2_ranker.py`). Hai ảnh "giống hệt" → 1.0; "không liên quan" → ~0.5.

---

## 3. Phương pháp

Ba thí nghiệm độc lập, chạy bằng chính `DINOv2Ranker._embed` của project (MPS,
`dinov2_vitb14_reg`). Script: `evaluation/ver_new/` (reproduce: xem §6).

- **T1 — Glyph SẠCH (best case).** 150 chữ ngẫu nhiên từ cache FontDiffusion.
  - *Cùng chữ:* cosine(FD[X], augment(FD[X])) — augment = xoay ±5°, co giãn
    0.9–1.05, nhiễu Gauss (mô phỏng biến thiên nội lớp).
  - *Khác chữ:* cosine(FD[X], FD[Y≠X]).
- **T2 — Crop WOODBLOCK THẬT.** Crop GOLD đã gán nhãn (đã sửa bbox).
  - *Cùng chữ:* 2 crop khác nhau của **cùng một** chữ (120 chữ có ≥2 mẫu).
  - *Khác chữ:* crop của 2 chữ khác nhau.
- **T3 — Retrieval top-1 (chính là việc S3 phải làm).** Gallery 500 glyph FD;
  với mỗi crop GOLD (200 mẫu), lấy glyph gần nhất theo cosine → có đúng chữ không.

---

## 4. Kết quả

| Thí nghiệm | Cùng chữ | Khác chữ | Khoảng cách | Đọc |
|---|---|---|---|---|
| **T1** glyph FD sạch | **0.950** | **0.900** | 0.050 | min(cùng)=0.812 **<** max(khác)=0.973 → **dải trùng nhau** |
| **T2** crop woodblock thật | **0.889** | **0.877** | **0.012** | **gần như bằng nhau** → không tách được |
| **T3** retrieval top-1 | — | — | — | **0.0%** đúng (0/200), chance = 0.2%, gallery 500 |

**Diễn giải:**

1. **Ngay trên glyph SẠCH (T1)** — điều kiện dễ nhất — DINOv2 đã không tách được:
   một cặp *khác chữ* có thể đạt 0.973, cao hơn cả một cặp *cùng chữ* 0.812. Phân
   bố cùng/khác **chồng lên nhau**, không có ngưỡng nào tách được.
2. **Trên crop THẬT (T2)** thì sụp hoàn toàn: cùng-chữ 0.889 ≈ khác-chữ 0.877.
   Chênh 0.012 là **vô nghĩa** — DINOv2 không biết 2 crop có phải cùng một chữ hay
   không.
3. **T3 là bằng chứng đắt nhất:** đặt đúng bài toán của S3 (tìm glyph khớp crop)
   thì độ chính xác **0.0%** — gần như **ngẫu nhiên** (0.2%). Trong 200 lần, glyph
   đúng **không một lần nào** là láng giềng gần nhất.

> Với ngưỡng "match" 0.75 mà pipeline dùng: vì **mọi** cặp (kể cả khác chữ) đều
> ~0.88–0.91 > 0.75, **mọi ứng viên đều "đạt"** → S3 chọn argmax giữa các điểm
> gần bằng nhau = **chọn bừa**.

---

## 4.1 Hình minh chứng

**Hình 1 — Phân bố cosine trên crop THẬT (cùng chữ vs khác chữ).** Hai phân bố
**chồng khít** lên nhau, cùng dồn quanh 0.85–0.95 và **đều nằm bên phải ngưỡng
0.75** → không có ngưỡng nào tách được "cùng chữ" khỏi "khác chữ".

![Hình 1: phân bố cosine cùng/khác chữ trùng nhau](figures/fig1_hist_real_crops.png)

**Hình 2 — Heatmap cosine giữa 14 glyph FD.** Đường chéo (cùng chữ) = 1.00, nhưng
**các ô ngoài đường chéo (khác chữ) hầu hết 0.84–0.97 — gần như ĐẬM bằng đường
chéo**. Không hề có mẫu "đường chéo sáng / ngoài tối" mà một embedding phân biệt
được phải có ⇒ DINOv2 ánh xạ mọi glyph về gần một điểm.

![Hình 2: heatmap cosine gần như đồng nhất](figures/fig2_cosine_heatmap.png)

**Hình 3 — Retrieval thất bại.** Với mỗi crop (ô đỏ, trái), 5 glyph **gần nhất**
theo DINOv2 đều là **chữ SAI** (cosine ~0.84–0.92); glyph **ĐÚNG** (ô xanh, phải)
bị xếp tận **hạng #492–#995 / 800**, không bao giờ lọt top.

![Hình 3: glyph đúng không nằm trong top-5 gần nhất](figures/fig3_retrieval_fail.png)

---

## 5. Vì sao DINOv2 thất bại ở đây

- DINOv2 là model **tự giám sát trên ảnh tự nhiên** (ImageNet-scale), tối ưu để
  nắm **đặc trưng ngữ nghĩa thô** ("đây là khối nét đen kiểu chữ Hán trên nền
  trắng"), **không** để phân biệt **khác biệt nét nhỏ** giữa hai chữ — mà đó mới
  là thứ định danh chữ Nôm.
- Toàn bộ glyph CJK rơi vào **một vùng embedding chật hẹp** (mọi cặp ~0.9) → mất
  khả năng phân biệt nội bộ (low intra-class resolution).
- Đây là hạn chế **fine-grained** đã biết của embedding zero-shot: tốt cho
  "phân loại thô", kém cho "phân biệt cá thể trong cùng một loại".

Lưu ý: model **không hỏng** — nó vẫn tách được "glyph vs ảnh không phải glyph".
Vấn đề **đặc thù** là **phân biệt chữ-Nôm-với-chữ-Nôm**, đúng việc S3 cần.

---

## 6. Tác động & khuyến nghị

- **Tác động:** tầng SILVER (504 nhãn ở bản trước) và tier-3 visual của pipeline
  gốc **không đáng tin**. Đã **tắt SILVER**; GOLD (S1∩S2, không dùng thị giác)
  **không bị ảnh hưởng** và vẫn là sàn an toàn.
- **Khuyến nghị:** thay DINOv2 zero-shot bằng **embedding-model huấn luyện riêng
  trên glyph Nôm** (contrastive/metric-learning trên cặp *crop GOLD ↔ glyph FD
  cùng chữ*). Xem `evaluation/ver_new/TRAIN_nom_embedding.md`.
- **Tiêu chí nghiệm thu** (chạy lại đúng 3 test này): T3 top-1 phải lên
  **~80–95%**; T2 phải **tách rõ** cùng/khác (chênh ≥ ~0.2); T1 đạt gần hoàn hảo.

---

## 7. Reproduce

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
.venv/bin/python evaluation/ver_new/dinov2_proof.py   # số liệu T1/T2/T3 -> results/dinov2_proof.json
.venv/bin/python evaluation/ver_new/make_figs.py      # 3 hình -> evaluation/ver_new/figures/
```
Số liệu thô: `T1_clean {same 0.95, diff 0.90, same_min 0.812, diff_max 0.973}`,
`T2_real {same 0.889, diff 0.877, n=120/118}`,
`T3_retrieval {top1 0.0%, n=200, gallery 500, chance 0.002}`.
