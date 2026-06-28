# Thiết kế S3 tốt nhất — Tín hiệu thị giác cho gán nhãn Hán-Nôm

> Bản thiết kế tích hợp cho **chương S3** của luận văn, bám số đo thật (445 trang, 3
> sách, 1591 lớp, 4.736 quyết định VAL-GOLD: bank retrieval@1 **0.894**, head r@1
> **0.907**, AURC **0.029**, đuôi hiếm **0.465** vs phổ biến **0.915**, IDS phân rã
> **94.7%**) và code thật trong `evaluation/ver_new/`. Mỗi lựa chọn có trích dẫn đã
> kiểm chứng đứng sau; các claim chưa-kiểm (SCoRE, "48% kuzushiji", AdaFace-trên-CJK)
> đã được **làm mềm/hạ xuống future-work**.

**Đổi trục tư duy:** đóng góp của chương S3 **KHÔNG phải một encoder tốt hơn** —
encoder đã ổn (89–91% r@1, rò rỉ LOBO ~1%). Khoảng trống bảo-vệ-được là: con số
precision 95% hiện tại là **ước lượng điểm grid-search trên GOLD dễ, không có giá
trị thống kê**. Sửa điều đó (Mục 2) là **tâm điểm**; encoder/đuôi-hiếm/hợp-nhất là
nâng cấp hỗ trợ.

---

## 1. KIẾN TRÚC KHUYẾN NGHỊ (end-to-end)

### GIỮ (đã chạy tốt — đừng đụng xương sống)
- **Khung metric-retrieval** (rerank trên candidate set, KHÔNG phải softmax
  1591-lớp) — đúng chuẩn SOTA cho CJK cổ suy biến (*Oracle Bone Inscriptions
  Retrieval, ICDAR 2024*). 89% + rò rỉ 1% xác nhận; đừng lùi về classifier.
- **Bank tham chiếu per-class** (`crop` prototype → `simfont` → `fd` trong
  `visual_signal._ref_bank`). Tier crop thật (zero domain-gap) là **lift lớn nhất** —
  giữ làm tier 1.
- **Candidate set** R = {OCR} ∪ {cách đọc từ điển}, `tighten_box`, embedding 256-D,
  hợp đồng checkpoint → `infer.py`/`visual_signal.py` **không cần đổi**.
- **Isotonic per-tier → P(match)**: giữ làm *phép biến đổi điểm*, nhưng *ngưỡng* bên
  trên bị thay (Mục 2).
- **"Glyph guard"** trong `decide()` (winner thắng chỉ nhờ crop-prototype → abstain):
  ý tưởng tốt — gập vào điểm hợp-nhất (Mục 4), đừng xoá.

### ĐỔI (có mục tiêu, có đo, mỗi cái có trích dẫn thật)

**Loss encoder — Sub-center ArcFace + AdaFace (giữ backbone).**
Thay vanilla ArcFace trong `nom_classifier/kaggle_train.py` + `model.py` bằng
**Sub-center ArcFace K=3** (*Deng et al., ECCV 2020*) **+ margin AdaFace thích nghi
chất lượng** (*Kim et al., CVPR 2022 Oral, arXiv 2204.00964*). ~50 dòng, cùng data,
cùng compute, checkpoint giữ nguyên.
- *Vì sao sub-center:* một lớp Nôm trải nhiều bản khắc + người viết + glyph FD; một
  center duy nhất gộp thành prototype mờ — đúng cơ chế gây sụp đuôi-hiếm 45 điểm.
  K=3 cho các mode crop-thật và mode glyph-tổng-hợp **cùng tồn tại** (đúng cầu nối
  crop↔fd mà bank dựa vào).
- *Vì sao AdaFace:* crop rách/lem là regime "chất-lượng-biến-thiên" mà AdaFace nhắm;
  nó hạ trọng số crop norm-thấp (suy biến) thay vì để chúng kéo prototype → **cải
  thiện tính trung thực của calibration** (encoder ngừng overfit crop không-định-
  danh-được, điểm open-set phản ánh độ khó thật).
- ⚠️ **Trung thực:** **chưa có kết quả AdaFace-trên-CJK công bố** — đây là suy diễn
  hợp lý từ regime face/TinyFace. Coi là **giả thuyết A/B** trong `kaggle_train.py`,
  báo delta, chỉ giữ nếu thắng. 2 lỗi code thật: (a) head phải thấy norm **thô** →
  chuyển `F.normalize` ra **sau** head; (b) K=2–3 và **chỉ áp cho lớp ≥3 crop** (tránh
  vỡ vụn lớp nhỏ). **Sub-center là fix biến-thiên lớp-phổ-biến, KHÔNG phải fix
  đuôi-hiếm** — đừng quảng cáo nhầm.

**Backbone — hoãn, đừng dẫn đầu bằng nó.** ConvNeXt-V2 + pretrain FCMAE in-domain
(*Woo et al., CVPR 2023, arXiv 2301.00808*) có *trần* cao nhất NHƯNG tiền-đề (dư
crop chưa-nhãn lớn) **chưa kiểm trên data bạn** (`index.csv` chỉ có 51.195 crop nhãn
+ 9.493 font + 1.591 FD, **không có pool crop chưa-nhãn riêng**) và delta trên glyph
160px nhị-phân là **ngoại suy**. **Giữ ResNet34** cho luận văn; FCMAE = future work,
chỉ chạy nếu A/B cho thấy backbone là nút thắt sau khi đổi head.

**Inference — GeM pooling + TTA nhẹ (gần như miễn phí).** Thay global-avg-pool cuối
bằng **GeM** trainable (p≈3) (*Radenović et al., arXiv 1811.00202*); lúc suy luận
trung bình embedding L2-norm qua vài augmentation nhẹ (xoay/scale nhỏ). GeM nhấn vùng
nét phân biệt (tốt cho dị bản Nôm gần giống); TTA giảm phương sai framing mà
`tighten_box` không bỏ hết. ~1 ngày, đo ở quick-eval + `s3_risk_coverage.py`.

---

## 2. REJECT CÓ BẢO ĐẢM PRECISION — TÂM ĐIỂM CHƯƠNG S3 ⭐

**Vấn đề, nói thẳng.** `calibrate_s3.py` (dòng 164–181) grid-search (τ,δ) trên 48×5
điểm để max coverage với `precision ≥ target` **trên VAL-GOLD**. Đó là bẫy
multiple-testing / winner's-curse: con số 0.9517 là **ước lượng điểm lệch lạc-quan,
không bảo đảm**, lại đo trên regime GOLD *dễ* trong khi SILVER bắn ở crop *khó hơn*.
Luận văn hiện **không bảo vệ được** "precision SILVER ≥ 90%".

**Cách sửa — đặt tên đối tượng, rồi chặn nó.** Precision-trong-số-chấp-nhận =
**1 − FSR**, với **FSR(τ) = P(Ŷ≠Y | chấp nhận)** = False Selection Rate. *Zhao & Su,
arXiv 2311.03811* đặt đúng đối tượng này, **chứng minh FSR đơn điệu giảm theo ngưỡng**
(nên ngưỡng tồn tại & **fixed-sequence testing hợp lệ**), và cho thấy ngưỡng trên
max-of-scores (đúng kiểu max-over-tiers P(match) của S3) là luật đúng. (⚠️ chỉ trích
cho *khung FSR = 1−precision* + đơn điệu, KHÔNG cho "bảo đảm exchangeability".)

**Cỗ máy — Learn-Then-Test (LTT) fixed-sequence (CHÍNH).** *Angelopoulos, Bates,
Candès, Jordan, Lei — arXiv 2110.01052.* Thay grid-search bằng:
1. Loss mỗi quyết định `L_i = 1[chấp nhận VÀ top_char ≠ true]`.
2. Quét τ từ cao (an toàn) xuống; mỗi τ tính `FSR_hat = lỗi_chấp_nhận / số_chấp_nhận`
   và **p-value Hoeffding–Bentkus**
   `p_HB = min(exp(−n·h1(min(FSR_hat,α),α)), e·P(Bin(n,α) ≤ ⌈n·FSR_hat⌉))`,
   `h1(a,b)=a·log(a/b)+(1−a)·log((1−a)/(1−b))`.
3. Vì FSR đơn điệu → duyệt **fixed-sequence**, chấp nhận khi `p_HB ≤ δ`, xuất τ thấp
   nhất còn giữ (max coverage). Cho **P(precision SILVER thật ≥ 1−α) ≥ 1−δ**, không
   phạt Bonferroni (FWER = δ đúng cho risk vô hướng đơn điệu).

~**30–40 dòng numpy, không GPU, dưới 1 giây**, **thay grid-search nguyên chỗ** — ghi
τ,α,δ vào `s3_calibration.json`; `decide()` giữ nguyên, **kế thừa bảo đảm**.

**Bound chặt hơn (khuyến nghị):** ở n≈1–4k, Hoeffding lỏng. Dùng **Waudby-Smith-Ramdas
betting / empirical-Bernstein UCB** (*RCPS, Bates et al., JACM 2021, arXiv 2101.02703*;
impl `aangelopoulos/rcps`) — cùng wrapper fixed-sequence, thích nghi phương sai →
**chấp nhận nhiều crop hơn ở cùng precision bảo đảm**. Giữ HB làm fallback an toàn.

**Làm cho bảo đảm KHÔNG vòng tròn (con số luận văn đang thiếu).** LTT chỉ chuyển được
dưới **exchangeability**. Calibrate trên GOLD rồi bắn trên SILVER = subpopulation
shift → **vô hiệu hoá bound**. Vì vậy:
- Lấy **~300–600 crop SILVER-ELIGIBLE** (regime S1/S2 *chưa* đồng ý, nơi reject thật
  sự bắn) qua pipeline `measure_precision.py` / `crop_audit.pdf`, **soát người**, rồi
  chạy LTT **trên stratum đó**.
- Báo α-bảo-đảm-LTT trên stratum SILVER đã-soát làm **con số headline**. Đây là con
  số precision phi-vòng-tròn DUY NHẤT luận văn đang thiếu. Đường GOLD trong
  `s3_risk_coverage.py` (AURC 0.029) giữ làm *chẩn đoán*, ghi rõ "lạc quan".

**Code:** file mới `evaluation/ver_new/conformal_reject.py` (chọn ngưỡng LTT/RCPS);
`calibrate_s3.py` bỏ grid-search; `s3_risk_coverage.py` vẽ **điểm vận hành bảo đảm**
(kèm UCB) trên đường risk-coverage. `decide()` không đổi.

**Trung thực ghi trong chương:** (a) kiểm **FSR có đơn điệu thật** trên mảng
`decisions` trước khi tin fixed-sequence (không thì lùi về Bonferroni-LTT); (b) ở n
nhỏ HB/WSR có thể ép α bảo thủ — chạy `conformal_reject.py` một lần để xác nhận α
bảo-đảm còn cạnh tranh với 0.90 hiện tại, đừng giả định. **Bỏ SCoRE/e-value** khỏi
headline (arXiv id chưa kiểm) — chỉ nêu future-work.

---

## 3. ĐUÔI HIẾM — tuyến cấu trúc/zero-shot đóng gap 45 điểm

Đo được: lớp hiếm (<5 crop) retrieval **0.465** vs phổ biến **0.915**; **44/46 lỗi
hiếm có bộ thủ khác nhau** (`structural_potential` 0.957); cjkvi-ids phân rã 94.7% lớp
và 797/831 đuôi hiếm. **Bộ phận có sẵn và phân biệt được** cho gần như mọi lỗi hiếm.
IDS đã ở `ids_data/`, tooling ở `ids_coverage.py`.

**Tuyến: thêm head phụ IDS/bộ-thủ trên embedding chung**, giám sát bằng phân rã
cjkvi-ids, train chung với metric loss trong `kaggle_train.py`. Đây là cơ chế mọi
zero-shot CJK SOTA dùng: *Hi-GITA (arXiv 2505.24837)* ~+19% zero-shot char so
CCR-CLIP bằng căn cấp bậc nét/bộ-thủ; *kuzushiji "character parts" (Pattern
Recognition vol.148, art.110181, 2024)* nhận lớp 0-mẫu qua loss nhất-quán
font→cổ-văn — **đúng setup crop-thật + glyph-tổng-hợp của bạn**. Biến cầu nối FD từ
"kéo embedding cả-glyph" thành "**chia sẻ bộ thủ**" — tuyến duy nhất nhận được lớp
0 crop thật.

Hai bổ sung cụ thể:
- **Head bộ-thủ phụ:** multi-label tập-bộ-thủ (hoặc GRU sinh chuỗi IDS) nhẹ trên
  embedding, giám sát từ cjkvi-ids. Nhỏ; chi phí = cùng forward backbone.
- **Loss nhất-quán mức-bộ-phận:** MSE kéo đặc trưng crop về đặc trưng glyph-FD **ở
  mức bộ phận** (cơ chế kuzushiji-parts), không chỉ ArcFace kéo cả-glyph.

**Tích hợp — CỔNG CHỈ-ĐUÔI-HIẾM, không override toàn cục.** Trộn điểm cấu trúc với
cosine trong `decide()` **chỉ khi candidate có <5 crop** (gate theo stratum tần-suất
của `eval_rare_tail.py`). Lớp phổ biến 0.915 đã mạnh; IDS mơ hồ cho dị-bản gần-giống,
override toàn cục sẽ hại. Dùng làm tín hiệu cứu **chỉ cho đuôi**.

**Reject đuôi hiếm — Clustered/class-conditional conformal** (*Ding et al., NeurIPS
2023, arXiv 2306.09335*). Với 1591 lớp + đuôi <5 crop, bảo-đảm per-class là bất khả
(quantile = +∞ khi n_class < 1/α−1). Gom lớp đuôi (theo cấu trúc IDS hoặc profile
điểm) thành nhóm ~20–75 mẫu → **1 ngưỡng bảo đảm/cụm**; lớp dưới ngưỡng lùi về ngưỡng
LTT biên. Chặn việc bảo-đảm-biên giấu hết lỗi vào đuôi. *(per-cluster, ε-xấp-xỉ
per-class — ghi rõ.)*

---

## 4. HỢP NHẤT TÍN HIỆU — học, không chỉnh tay

Năm tín hiệu: **chữ OCR**, **cách-đọc từ điển (prior âm tiết)**, **bank cosine**
(0.894), **head 1591-lớp chưa dùng** (0.907; head+bank đồng thuận → 0.93 precision @
0.944 coverage, cứu ~6.9k REVIEW theo `head_rescue.json`), **điểm cấu trúc** (Mục 3).
`decide()` hiện chỉ hợp nhất các tier bank qua isotonic + glyph-guard chỉnh tay.

**Hợp nhất nguyên lý: hiệu chuẩn MỖI tín hiệu về P(match) so-sánh-được, rồi để tầng
conformal đặt luật chấp nhận — không ngưỡng chỉnh tay.**
1. **Hiệu chuẩn mỗi tín hiệu → P(match).** Giữ isotonic per-tier (`calibrate_tier`);
   thêm **head** (softmax-trên-R, hiệu chuẩn cùng cách) và **cấu trúc**. Giờ bank,
   head, cấu trúc là xác suất so-sánh-được → "head VÀ bank đồng thuận" thành AND đã
   hiệu chuẩn, không phải luật ad-hoc.
2. **Hợp nhất → hiệu chuẩn điểm-hợp-nhất → ngưỡng bằng LTT.** Điểm hợp nhất đơn giản
   bảo vệ được (max/mean-trên-tín-hiệu, hoặc logistic nhỏ trên các tín hiệu đã hiệu
   chuẩn, fit trên VAL) → hiệu chuẩn lại → đặt 1 ngưỡng bằng LTT/FSR (Mục 2). Luật
   quyết định **học end-to-end và có bound chứng minh**, thay (τ,δ)+glyph-guard. Giữ
   *ý tưởng* glyph-guard nhưng diễn đạt thành bất-đồng bank-vs-glyph đã-hiệu-chuẩn
   nuôi vào điểm hợp nhất, không phải margin 0.10 phép thuật.
3. **Khai thác prior âm tiết tường minh.** R đã được cắt về cách-đọc-từ-điển của âm
   tiết — *đó là prior*. Mạnh thêm: trọng số mỗi candidate theo **tần suất cách đọc**
   (prior Bayes đúng nghĩa trên R) → chữ OCR hiếm phải thắng glyph-match mạnh mới
   được. Miễn phí, bám corpus.
4. **Nhất quán xuyên-trang (đòn bẩy cao, rẻ).** Cùng Unicode tái diễn khắp 445 trang.
   Sau lượt 1, gộp crop tin-cậy-cao của mỗi lớp vào bank (tự-huấn-luyện tier `crop`),
   và để ánh xạ âm↔chữ xác nhận ở trang này nâng prior ở trang khác. Trực tiếp **nuôi
   lớn tier crop zero-gap cho đuôi hiếm**. Cài như lượt 2 trong `align_production.py`
   nạp crop xác nhận về `_load_or_build_protos`.

---

## 5. KẾ HOẠCH ƯU TIÊN (đòn bẩy cao trước)

| # | Việc | Ở đâu | Công sức | Kaggle/Local | Lợi ích | Loại |
|---|------|-------|----------|--------------|---------|------|
| **1** | **Conformal reject LTT/RCPS (FSR)** — thay grid-search; ghi τ bảo-đảm | mới `conformal_reject.py`, sửa `calibrate_s3.py`/`s3_risk_coverage.py` | **S** (~40 dòng) | Local/CPU | **Tuyên bố "P(precision ≥ 1−α) ≥ 1−δ"** — đòn bẩy cao nhất | Chương NC (tâm điểm) |
| **2** | **Bộ soát người SILVER-eligible** (~300–600) → chạy #1 trên đó | `measure_precision.py`, `crop_audit.pdf` | **M** (người, không GPU) | Local | Con số precision phi-vòng-tròn; làm #1 *trung thực* | Chương NC |
| **3** | **Head Sub-center AdaFace** (K=3) | `kaggle_train.py`, `model.py` | **S/M** (~50 dòng, retrain 40ep) | Kaggle (~2–4h) | Calibration trung thực hơn + đuôi hiếm; A/B-gate | Quick win → chương |
| **4** | **Thêm head 1591-lớp làm tín hiệu hợp nhất đã hiệu chuẩn** (đã train, đang bỏ phí) | `head_rescue.py`→`decide`, `calibrate_s3.py` | **S** | Local | head+bank → ~0.93 precision, cứu ~6.9k REVIEW; gần miễn phí | Quick win |
| **5** | **GeM + TTA nhẹ** | `model.py`, `visual_signal.compute` | **S** (~1 ngày) | Kaggle+Local | Tăng retrieval@1 rẻ; giảm AURC | Quick win |
| **6** | **Nhất quán xuyên-trang / tự-huấn-luyện prototype** | `align_production.py`, `_load_or_build_protos` | **M** | Local | Nuôi lớn tier crop zero-gap; giúp đuôi | Quick win → chương |
| **7** | **Head phụ IDS/bộ-thủ + loss nhất-quán bộ-phận**, hợp nhất chỉ-đuôi | `kaggle_train.py`, `ids_coverage.py`, `decide()` | **M/L** (~5–8 ngày) | Kaggle | Đóng gap đuôi-hiếm 45 điểm (94.7% phân rã) | Chương NC |
| **8** | **Clustered conformal** cho bảo-đảm per-cụm đuôi hiếm | mở rộng `conformal_reject.py` | **M** | Local/CPU | Bảo-đảm precision per-stratum; chặn số-biên giấu lỗi đuôi | Chương NC |
| **9** | ConvNeXt-V2 + FCMAE in-domain | `model.py`, script pretrain | **L** | Kaggle | Trần cao nhất nhưng dư-crop chưa kiểm; A/B trước | Future work |

**Đường tới hạn cho luận văn:** **1 → 2 → 4 → 3**, rồi **7 → 8** là các chương NC.
Mục 1,4,5,6 là quick-win vài ngày; **#2 là nút cổ chai con người — bắt đầu NGAY**.

---

## 6. KHÔNG NÊN LÀM

- **Không VLM / CLIP hai-tháp đầy đủ** (Hi-GITA/CCR-CLIP full). Bạn cần *cơ chế
  chia-sẻ-bộ-phận*, không phải tháp ảnh-text thứ 2. Head phụ IDS nhẹ trên embedding
  sẵn có lấy ~toàn bộ lợi ích với chi phí nhỏ. CLIP hai-tháp là dự án NC, không phải
  chương P100.
- **Không DINOv2 / SSL off-the-shelf** — đã bác trên Nôm (retrieval 0%,
  `REPORT_dinov2_unsuitable.md`). Bài học là *train in-domain*; đừng tranh lại.
- **Đừng dẫn đầu bằng đổi backbone.** ConvNeXt-V2+FCMAE trần cao nhưng tiền-đề
  (dư crop chưa-nhãn) chưa kiểm, delta per-glyph ngoại suy. Đổi head (#3) là thắng
  an toàn cùng-compute; backbone = future work sau A/B.
- **Đừng để SCoRE/e-value làm reject headline** — hứa hẹn cho hợp-nhất tín-hiệu phụ-
  thuộc nhưng là preprint mới, arXiv id chưa kiểm — trích LTT làm bảo đảm, e-value
  chỉ future-work.
- **Đừng tuyên bố AdaFace là thắng-CJK đã-chứng-minh** — chưa có kết quả CJK công bố;
  trình bày như transfer được-A/B-xác-nhận.
- **Đừng đuổi Partial-FC / Unicom triệu-lớp** — chỉ liên quan nếu số lớp vọt xa 1591.
- **Đừng over-augment TTA / vỡ vụn sub-center** — augmentation nhẹ; K≤3 & chỉ lớp ≥3
  crop.
- **Đừng tiếp tục báo precision GOLD-only làm headline** — đó là chẩn-đoán lạc-quan;
  con số headline là LTT-trên-SILVER-đã-soát.

---

## 7. TÀI LIỆU THAM KHẢO THEN CHỐT

1. **Learn Then Test** — Angelopoulos, Bates, Candès, Jordan, Lei; FnT ML 2021/2025 —
   https://arxiv.org/abs/2110.01052 — *Bảo đảm reject: HB p-value + fixed-sequence →
   P(precision SILVER ≥ 1−α) ≥ 1−δ. Tâm điểm chương.*
2. **RCPS (Risk-Controlling Prediction Sets)** — Bates et al.; JACM 2021 —
   https://arxiv.org/abs/2101.02703 — *Phiên bản 1-ngưỡng UCB + bound WSR betting
   chặt hơn ở n≈1–4k.*
3. **Controlling FSR in Selective Classification** — Zhao & Su; arXiv 2023 —
   https://arxiv.org/abs/2311.03811 — *Đặt FSR = 1−precision, chứng minh đơn điệu
   (hợp lệ hoá fixed-sequence). Khung hình thức.*
4. **Class-Conditional / Clustered Conformal** — Ding et al.; NeurIPS 2023 —
   https://arxiv.org/abs/2306.09335 — *Bảo-đảm precision per-cụm khi per-class bất khả
   (đuôi hiếm).*
5. **A Gentle Introduction to Conformal Prediction** — Angelopoulos & Bates; FnT ML
   2023 — https://arxiv.org/abs/2107.07511 — *Cookbook + group-balanced (per-book/
   per-stratum); phần toán để viết chương.*
6. **AdaFace: Quality Adaptive Margin** — Kim et al.; CVPR 2022 Oral —
   https://arxiv.org/abs/2204.00964 — *Margin per-sample theo norm: hạ trọng số crop
   suy biến → calibration trung thực.*
7. **Sub-center ArcFace** — Deng et al.; ECCV 2020 —
   https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf — *K
   sub-center cho mode crop + glyph cùng tồn tại; đánh sụp đuôi-hiếm.*
8. **Hi-GITA (Zero-Shot CCR, multi-granularity)** — arXiv 2025 (xây trên CCR-CLIP
   ICCV'23) — https://arxiv.org/abs/2505.24837 — *Bằng chứng mạnh nhất: phân rã bộ
   thủ/nét là ĐÒN BẨY đuôi-hiếm (~+19% zero-shot). (số có thể đổi — trích cơ chế.)*
9. **Japanese Historical Char Recognition by Character Parts** — Ishikawa, Miyazaki,
   Omachi; Pattern Recognition 148, 2024 —
   https://www.sciencedirect.com/science/article/pii/S0031320323008786 — *Đúng setup
   crop-thật + glyph-tổng-hợp; loss nhất-quán bộ-phận nhận lớp 0-mẫu. (số 48% xấp xỉ.)*
10. **ConvNeXt V2 (FCMAE)** — Woo et al.; CVPR 2023 —
    https://arxiv.org/abs/2301.00808 — *Backbone + SSL in-domain future-work; lý do
    SSL off-the-shelf thất bại là MIỀN, không phải kiến trúc.*
11. **GeM Pooling for Image Retrieval** — Radenović et al.; 2018 —
    https://arxiv.org/abs/1811.00202 — *Tăng retrieval gần-miễn-phí, nhấn vùng nét
    phân biệt.*
12. **Oracle Bone Inscriptions Retrieval (Metric Learning)** — ICDAR 2024 —
    https://dl.acm.org/doi/10.1007/978-3-031-70543-4_10 — *Xác nhận metric-retrieval
    (không phải classification) là chuẩn SOTA cho CJK cổ suy biến.*

> ⚠️ **Lưu ý kiểm chứng:** 2 nhánh research (encoder, conformal) thành công nhưng
> verify-agent gặp lỗi API 529 (chưa kiểm-chéo lần 2). Các trích dẫn lõi (LTT, RCPS,
> Sub-center, AdaFace, ConvNeXt-V2, GeM, Clustered-Conformal, Zhao&Su) là paper thật
> nổi tiếng; các cái mới/mềm (Hi-GITA, kuzushiji-48%, SCoRE) đã hạ xuống future-work
> và ghi rõ "số xấp xỉ". Kiểm lại URL trước khi đưa vào luận văn.
