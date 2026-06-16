# Hướng hoàn thiện luận văn: Gán nhãn tự động kho ngữ liệu Hán-Nôm từ bản dịch Quốc ngữ

> Tài liệu định hướng, bám sát code hiện có (`evaluation/ver_new/`, `core/`) và đối
> chiếu literature quốc tế đã **kiểm chứng trích dẫn**. Trọng tâm: 2 chỗ đang bí —
> (A) **cắt/segment chữ** và (B) **so khớp 2 ảnh (S3)** + đánh giá trung thực.
> Mọi code mới đề xuất đặt dưới `evaluation/`.

Ngày: 2026-06-14. Trạng thái dữ liệu: 445 trang · 82.268 cặp · GOLD 51.195 ·
SILVER 9.247 · SYLLABLE 6.188 · REVIEW 15.638 · 1.591 lớp chữ · 3 sách
(yen2/yen4/yen11). Encoder S3: ResNet34 + ArcFace, 160px, 256-D.

---

## 1. ĐỊNH VỊ ĐỀ TÀI (để biết phải khẳng định gì và né đòn gì)

**Bối cảnh.** Bộ dữ liệu Hán-Nôm OCR công khai duy nhất hiện nay là **NomNaOCR**
(Đặng & Nguyễn, RIVF 2022; 2.953 trang → 38.318 patch **dòng/chuỗi**, DBNet phát
hiện + CRNN/CTC & SC-CNN+Transformer nhận dạng). Bộ này cũng được dựng bằng cách
**ánh xạ phiên âm bán-tự-động lên bản dịch Quốc ngữ** của Hội Bảo tồn Di sản Nôm —
tức **đã** dùng tương ứng Nôm↔QN. **IHR-NomDB** (Vũ, Lê, Beurton-Aimar, ICDAR 2021;
~260 trang, 13.254 chữ, 101.621 chuỗi tổng hợp) có **lưu** bản dịch tiếng Việt
hiện đại nhưng gán nhãn **thủ công**.

➡️ **Hệ quả quan trọng:** bản thân ý tưởng "dùng bản dịch QN để gán nhãn" **KHÔNG
mới**. Đừng khẳng định điều này là đóng góp — hội đồng/người phản biện sẽ chỉ ngay
ra NomNaOCR. Đóng góp thật của bạn nằm ở chỗ khác:

**Ba đóng góp THẬT (chỉ khẳng định đúng 3 cái này):**
1. **Độ hạt mịn (granularity).** Crop **từng chữ**, đã siết ink, gán nhãn Unicode,
   ở **quy mô lớn** (60k+ usable). Các corpus Nôm công khai dừng ở mức dòng/chuỗi.
   Đây là khoảng trống dữ liệu thật sự.
2. **Cỗ máy gán nhãn dựa căn chỉnh + đồng thuận đa tín hiệu** (S1 OCR ∩ S2 từ điển
   ∩ S3 thị giác → GOLD/SILVER/SYLLABLE/REVIEW) — chứ không phải "có bản dịch".
   Tiền lệ gần nhất: **Toselli et al., "Digital Editions as Distant Supervision,"
   ICDAR 2021** (lưu ý: họ khai thác markup TEI/EpiDoc, không phải forced-alignment
   theo nghĩa đen — nói "lấy cảm hứng", đừng nói "giống hệt"). Cái này hợp thức hoá
   khái niệm "phiên âm song song = distant supervision".
3. **Quy trình đánh giá trung thực, open-set, kiểm soát rò rỉ** cho một corpus chữ
   cổ gán nhãn tự động — **NẾU** bạn thực sự làm xong Mục 3/4 dưới đây. NomNaOCR báo
   CER mức chuỗi; **chưa ai trong mảng Nôm** báo precision per-char đã hiệu chuẩn,
   có người soát, phân tầng. Đây là chỗ cắm cờ sạch nhất.

**Bốn đòn người phản biện sẽ đánh (phải phòng trước):**
- **Đánh giá S3 vòng tròn.** Calibrate trên GOLD (chỗ dễ, nơi S1∩S2 đã đồng ý từ
  trước), prototype lấy từ **cùng 3 sách** với query, **chưa bao giờ** được người
  soát trên regime SILVER — nơi S3 thực sự gánh việc. Chính docstring
  `calibrate_s3.py` của bạn đã thú nhận.
- **Rò rỉ trong-sách (within-book leakage).** Chỉ 3 sách. Crop-prototype và crop
  query **chung máy scan, mực, ván khắc** → `retrieval@1 ≈ 89%` là **cận trên**,
  không phải số khái quát hoá. Với 3 sách, split sạch = leave-one-book-out (train 2,
  test 1, xoay vòng 3 lần).
- **REVIEW là "thùng rác im lặng".** 15.638 chữ (~19% cặp) bị bỏ mà **không phân
  loại lý do** (segment lỗi vs gap căn chỉnh thật vs S3 dưới ngưỡng). Người đọc sẽ
  hiểu "19% mất không giải thích" = trần năng lực bạn đang giấu.
- **Điểm vận hành trông như chưa tinh chỉnh.** Một điểm `(τ=0.5, δ=0)` không kèm
  đường risk-coverage mời gọi đòn "anh overfit 1 ngưỡng" — dù thực ra nó **chính
  đáng** (xem 3.B1a). Sửa bằng cách vẽ cả đường cong.

> Bạn **không** cần thắng NomNaOCR về độ chính xác nhận dạng. Đóng góp của bạn là
> **dữ liệu + phương pháp + đánh giá trung thực**, không phải một recognizer SOTA
> mới. Khung luận văn theo đúng tinh thần đó và ngừng "phòng thủ" những con số nhận
> dạng mà bạn vốn không tuyên bố.

---

## 2. CHỖ BÍ (A) — CẮT / SEGMENT CHỮ

Mảng này đã **dứt khoát từ bỏ** projection-valley/connected-component cho CJK dọc
dày. **Đừng** tinh chỉnh thêm heuristic trung-điểm `_reseg_column` — về bản chất nó
**không thể** sửa được khi **đếm sai số chữ** (chữ dính/chữ thiếu phá vỡ mọi cách
cắt theo số). Đòn bẩy độc nhất bạn nắm: **sau căn chỉnh Nôm↔QN, bạn BIẾT số chữ
thật `N` của mỗi cột** (= số âm tiết QN), kể cả khi OCR đếm sai. Điều này biến
segmentation từ bài toán mở thành bài toán **có ràng buộc**.

### 2.0. Việc nên làm TRƯỚC TIÊN (rẻ, bắt buộc): đo phân rã REVIEW

Trước khi xây bất cứ thứ gì, trả lời: **"Bao nhiêu phần của REVIEW thật sự là lỗi
segment?"**. Trong `consensus.py`, lý do REVIEW **đã được tính sẵn**
(`diverged_column` / `below_visual_threshold` / `unconfirmed_no_s3`); trong
`align_production.py::_pair_new`, `n_review_gap` (số op `ins` = Nôm-OCR **bỏ sót**)
đã có. Chỉ cần **gộp các con số này vào `summary.json`**.

- **Công sức: S (nhỏ). Lợi ích: cao.** Biến "19% bị bỏ im lặng" thành "X%
  cứu-được-bằng-segment, Y% gap căn chỉnh thật, Z% S3-chưa-chắc" — vừa **biện minh**
  cho việc đầu tư bên dưới, vừa **tự thân là một bảng trong luận văn**.
- **Cách dùng kết quả:** nếu REVIEW chủ yếu là `diverged_column` do lệch số → A đáng
  đầu tư nặng. Nếu chủ yếu là gap căn chỉnh thật (chèn/xoá thật) → segment cứu được
  ít, và bạn **nên nói thẳng điều đó** (cũng là một kết luận khoa học).

### 2.1. PHƯƠNG ÁN A1 (CHỌN TRƯỚC) — Detector chữ anchorless, RÀNG BUỘC theo số `N`

**Phương pháp:** heatmap tâm-điểm kiểu CenterNet + hồi quy kích thước —
**HRCenterNet** (Tang et al., *IEEE Big Data 2020*, arXiv 2012.05739, code
github.com/Tverous/HRCenterNet; IoU trung bình ~0.81 trên ván khắc MTHv2). Pretrain
trên **TKH/MTHv2** (github.com/HCIILAB/TKH_MTH_Datasets_Release — CJK ván khắc có
**box mức chữ**; lưu ý đây là chữ **Hán in**, lệch miền so với Nôm viết tay → **bắt
buộc fine-tune**).

**Mẹo cốt lõi — ràng buộc số:** chạy detector trên từng cột, rồi **giữ đúng `N`
tâm** với `N` = số âm tiết QN. Nếu detector ra `>N` → gộp 2 box kề có confidence
thấp nhất; nếu `<N` → tách box rộng nhất. Đây chính là thứ làm cột "diverged" trở
nên **re-segment được** thay vì rơi vào REVIEW, và được "khoá" bằng độ tin cậy căn
chỉnh mà bạn đã có.

**Đặt ở đâu trong `evaluation/`:** thư mục mới `evaluation/ver_new/char_detector/`.
**Box huấn luyện lấy MIỄN PHÍ từ crop GOLD đã xác nhận** — chúng đã mang `bbox`
toạ-độ-trang-gốc trong `labels.csv` (cột cuối). **Bắt buộc:** suy lại box trên **toạ
độ ẢNH GỐC** (ghi chú MEMORY của bạn + các lời gọi `frame_offset`/`correct_columns`
trong `align_production.py` xác nhận box OCR từng lệch ~1.7 cột ở toạ độ frame-crop;
detector train trên box chưa-sửa-offset sẽ **kế thừa lỗi lệch**). Hook suy luận:
trong `align_production.py::_pair_new`, khi cột diverged → gọi detector đề xuất đúng
`N` box thay cho trung-điểm của `_reseg_column`.

**Kaggle:** 1 detector, batch 8, thừa sức P100/T4. **Công sức: L. Lợi ích: cao** —
đây là cách sửa chuẩn-mực của ngành, trực tiếp cứu phần lệch-số của REVIEW. Đo
**tỉ lệ cột cứu được + IoU per-char** so với segmenter trung-điểm hiện tại trên tập
trang held-out; dùng **độ nhất quán S3** của crop cứu-được làm proxy chất lượng.

### 2.2. PHƯƠNG ÁN A2 (bổ trợ) — Re-segment yếu chỉ-từ-phiên-âm (mẹo confidence CRAFT)

**Phương pháp:** **CRAFT** (Baek et al., *CVPR 2019*, arXiv 1904.01941). Công thức
khai thác pseudo-box theo confidence của CRAFT **đúng y tình huống bạn**:
`s_conf = [l − min(l, |l − lᶜ|)] / l`, với `l` = số chữ đã biết (= số âm tiết QN),
`lᶜ` = số chữ detector dự đoán; cột confidence thấp thì chia-đều và đặt `s_conf=0.5`.
Pretrain một mạng region/center nhỏ trên **dòng Nôm dọc tổng hợp** (box miễn phí khi
tự sinh từ `gannhanocr-fd`), rồi mine pseudo-box trên cột thật, **trọng số theo
`s_conf`** đó.

**Đặt ở đâu:** `evaluation/ver_new/seg_craft/` — `synth_lines.py` (dựng dòng dọc từ
kho glyph `gannhanocr-fd`, box đã biết) → `train_region.py` (Kaggle) → `mine_boxes.py`
(xuất crop + cột `s_conf` đưa vào tiering). **Cổng bắt buộc:** chỉ mine từ cột thuộc
tầng căn chỉnh GOLD/SILVER — `l` sai từ một căn chỉnh kém sẽ sinh pseudo-box sai.

**Công sức: L. Lợi ích: trung-cao.** Bổ trợ cho A1 (CRAFT cho pretrain dòng-tổng-hợp
+ confidence có gradation thay vì bỏ cứng). **Vẫn chọn A1 trước** vì A1 có sẵn
trọng số pretrain và ràng buộc-số dễ kiểm chứng hơn là watershed-split trên mực ván
khắc dính nặng.

### 2.3. Nên BIẾT nhưng KHÔNG nên làm bây giờ
- **PageNet** (Peng et al., *IJCV 2022*, arXiv 2207.14807): khớp kiến trúc đẹp nhất
  (chỉ-phiên-âm → box chữ + thứ tự đọc), **nhưng KHÔNG mở code train** (chỉ inference
  + weight tiếng Trung). Đừng hứa "drop-in"; reimplement là mục tiêu mở rộng, không
  phải deliverable.
- **Recognition by Segmentation, Segment-Annotation-Free** (Dezhi Peng, Lianwen Jin
  et al., **IEEE TMM 2022**, arXiv 2207.14801): nhận-dạng-bằng-segment **không cần
  nhãn segment**; gần về ý niệm nhưng **xuất điểm-cắt CTC, không phải box sát** → vẫn
  phải ghép với `tighten_box`. A1 tới đích với ít kỹ thuật hơn.
- **Detector cột học sâu (SegHist, ICDAR 2024):** chỉ làm **nếu** bạn **đo được**
  rằng phát hiện cột (chứ không phải đếm-sai-trong-cột) là nguồn lỗi thật. Divergence
  của bạn là đếm-sai-trong-cột → khả năng ROI thấp. **Đo trước (2.0) rồi quyết.**

---

## 3. CHỖ BÍ (B) — SO KHỚP ẢNH S3 + ĐÁNH GIÁ TRUNG THỰC

**Thiết kế S3 của bạn vốn ĐÃ đúng hướng SOTA — hãy nói rõ điều đó và trích dẫn.**
Encoder train + reference-bank per-class + cosine + reject **chính là**
**Cross-Modal Prototype Learning** (Ao, Zhang, Liu, *Pattern Recognition* 2022:
prototype chữ in + nearest-neighbor, chứng minh khái quát hoá hiện-đại→cổ) và
**nhận dạng open-set bằng label-to-prototype có cột reject** (Liu et al., *Pattern
Recognition* 2023, arXiv 2203.05179). Việc DINOv2 hỏng mà bạn tìm ra được **củng cố
độc lập** bởi **Raven, Matei, Fink, "SSL ViT for Writer Retrieval," ICDAR 2024 WS**
(ViT ImageNet off-the-shelf "kém hơn đáng kể"; encoder train theo miền thắng) — biến
"giai thoại" của bạn thành kết quả **được kỳ vọng & trích dẫn được**. Quan trọng:
đòn bẩy nằm ở **trung thực trong đánh giá, luật reject, và phần đuôi dài** — **không
phải** ở một loss "xịn hơn" (**Musgrave et al., "A Metric Learning Reality Check,"
ECCV 2020**: ArcFace là ổn; lợi thế các loss mới biến mất khi đánh giá công bằng).

### 3.B1 — Sửa điểm vận hành & rò rỉ (lõi của tính "bảo vệ được")

**(a) Vẽ TOÀN BỘ đường risk-coverage / AURC, không phải 1 điểm.** **Traub et al.,
NeurIPS 2024 Spotlight** (arXiv 2407.01032) chứng minh báo cáo 1-ngưỡng gây hiểu
lầm và đề xuất **AUGRC**. `calibrate_s3.py` của bạn **đã** dựng mảng per-decision
`recs = (P_win, margin, correct)` — bạn chỉ cách đường cong **một vòng lặp**: sort
theo `P_win`, quét, vẽ selective-risk vs coverage, tích phân ra AURC/AUGRC, và đặt
`(τ,δ)` như **MỘT điểm có nhãn trên đường cong**. Thay đổi nhỏ này **biến "τ=0.5 suy
biến" thành "điểm coverage-cực-đại tại P≥0.90"** — đúng là thứ vòng sweep của bạn
tính, và **hoàn toàn chính đáng** một khi vẽ ra đường cong. **Công sức: S.** *Nhưng*:
risk trên đường cong rốt cuộc phải đo trên **nhãn người** (B2), nếu không vẫn vòng
tròn.

**(b) Split TÁCH-SÁCH và TÁCH-CHỮ.** Bạn có cột `book` và chỉ 3 sách → làm
**leave-one-book-out (3-fold)**: dựng lại prototype từ 2 sách, đánh giá trên sách
thứ 3, xoay vòng. Báo cáo retrieval@1 **trong-sách vs xuyên-sách** RIÊNG và **kỳ vọng
số xuyên-sách tụt** — chính sự trung thực đó là đóng góp. `eval_char_disjoint.py` đã
làm trục tách-chữ (và tự thú nhận caveat cận-trên: backbone vẫn train trên chữ
held-out). Thêm chế độ `--book-holdout` dựa cột `book` sẵn có. Muốn số tách-chữ
**thật sạch** thì retrain Kaggle loại các lớp holdout; nếu không, báo số within-backbone
như **cận trên đã nêu rõ**. **Công sức: M. Lợi ích: con số headline bảo vệ được.**

**(c) Đổi điểm reject từ isotonic-trên-cosine sang Max-Logit / p-norm.** Isotonic
**overfit khi <~1000 điểm calibrate**, mà split GOLD-val per-tier của bạn mỏng — đây
**rất có thể là lý do** vòng sweep co về ngưỡng dễ dãi nhất. **Vaze et al., ICLR 2022
Oral** (Maximum Logit Score là scorer open-set hàng đầu) + **Cattelan & Silva, UAI
2024** (max-logit chuẩn-hoá p-norm, thuần post-hoc, **không cần retrain**). **NHƯNG:**
`infer.py` hiện chỉ nạp `ck["backbone"]` và **bỏ ArcFace head** → MLS cần **lưu &
nạp thêm trọng số head + danh sách lớp** lúc inference. Đó là thay đổi nhỏ, thật,
trong `infer.py` + `NomEncoder` (và thêm key `"arc"` khi save ở `kaggle_train.py`).
Ghép với **reject phi-tham-số kNN-distance** (**Sun et al., ICML 2022 Spotlight**,
arXiv 2204.06507; reference-bank của bạn **chính là** một chỉ mục kNN): abstain khi
crop **xa MỌI prototype** — bắt luôn crop rác/cắt-lỗi từ (A) **trước khi** chúng
thành nhãn SILVER sai, thay cho cổng ad-hoc `ink<0.03/>0.97`. **Công sức: M (MLS, cần
ship head) + S (kNN).**

### 3.B2 — Quy trình soát người (bạn đã xây nửa chừng — HÃY HOÀN TẤT)

Bạn đã có `export_eval_sample.py` (sampler phân tầng, `verify.csv`, `review.html`)
và `measure_precision.py` (**đã** tính Wilson CI — chọn đúng; Wald under-cover ở n
nhỏ). **Đây là con số phi-vòng-tròn.** Để "chống phản biện":
- **Phân tầng theo tier × dải-tần-suất** (đã làm) và **ưu tiên tầng SILVER
  `s2_inter_s3_corrected`** — regime chưa bao giờ được kiểm. Định trước `n` từ sai số
  mục tiêu (n≈200 → ±6%, n≈400 → ±4.5%).
- **Đồng thuận đa-người-soát.** ≥2 người biết chữ Nôm soát phần chồng nhau; báo
  **Cohen's κ**; theo **Northcutt et al., "Pervasive Label Errors," NeurIPS D&B 2021**
  (5 thợ, sai nếu <3/5 đồng ý; **chỉ ~51% candidate bị flag là thật sự sai** → **đừng
  bao giờ tự-động "sửa" theo flag**). Nếu chỉ có 1 người soát → nêu rõ như hạn chế +
  làm re-test nội-soát (intra-annotator).
- **Dùng taxonomy lỗi CER-HV** (Al-azzawi et al., arXiv 2601.16713 — **là preprint**,
  ĐỪNG trích như "đã đăng ICDAR 2026"): transcription / **segmentation** / orientation
  / script-mismatch / non-text. Lớp "segmentation" cho phép bạn **báo cáo lỗi của
  bài toán A như một chỉ số hạng nhất**.
- **Quét Cleanlab** (Northcutt et al., "Confident Learning," JAIR 2021) trên crop sẽ
  release để có **con số headline** mà một dataset paper cần ("residual label error
  ước lượng = X%, CI 95% …"), giới hạn ở lớp có ≥k mẫu (đuôi 1-shot không tin cậy).
  Đưa flag vào cùng hàng đợi soát.

**Công sức: S–M. Lợi ích: ĐÂY là deliverable quyết định độ tin cậy.** Không có
precision SILVER đã-soát-người kèm CI thì **bài toán B coi như chưa giải**.

### 3.B3 — Tham chiếu CẤU TRÚC cho đuôi hiếm / 0-crop (cách sửa rò rỉ nguyên lý nhất)

Lớp yếu nhất của bạn là đuôi singleton — nơi tham chiếu duy nhất là **1 glyph
FontDiffuser**, mà chính calibration của bạn xác nhận tier `fd` tách rất kém
(P-range ~0.002–0.49 trong `s3_calibration.json`). Hướng cấu trúc cho tham chiếu
**nội tại với chữ và không thể rò rỉ xuyên sách**:
- **CCR-CLIP** (Yu et al., *ICCV 2023*, arXiv 2309.01083, code mở): căn chỉnh ảnh
  crop với **IDS** (phân rã bộ thủ) của chữ bằng contrastive; điểm = cosine(crop,
  IDS-embedding). Dùng làm **re-rank/tiebreaker cho ứng viên có <2 crop thật**,
  KHÔNG dùng làm oracle duy nhất.
- **HierCode** (Zhang et al., *Pattern Recognition* 2024, arXiv 2403.13761; backbone
  ResNet18/34 nhẹ ~90MB, hợp T4) là biến thể codebook.
- Dữ liệu IDS: **cjkvi-ids / CHISE** (GPL, `ids.txt`). **Bước đầu bắt buộc: audit độ
  phủ IDS trên 1.591 lớp của bạn** — chữ Nôm tự tạo (CJK Ext-B/C) có thể **thiếu**
  IDS, mà đó đúng là phần đuôi bạn muốn phủ. Giữ glyph FD làm fallback.

Thêm tier `"ids"` vào `visual_signal.py::_ref_bank()` cạnh `crop`/`simfont`/`fd`;
toàn bộ `decide()`/calibration **giữ nguyên** — chỉ đổi embedding tham chiếu. **Công
sức: L. Lợi ích: trung bình, dồn vào đuôi dài.** Coi như **chương mở rộng nghiên
cứu**, không phải lõi.

### 3.B4 — Nâng cấp reference-bank & encoder rẻ tiền (làm cái S, bỏ cái suy đoán)
- **Đa-font + đa-FD (S, nên làm):** render mỗi lớp bằng vài font CJK thật (NomNaTong
  + một Minh/Tống + một Khải) và sinh K kiểu FontDiffuser → mỗi lớp có 3–6 prototype
  tổng hợp. `_build_fd_index` + cắt-tỉa `PROTO_TOPK` đã hỗ trợ; một tham chiếu lỗi
  không thể áp đảo. **FontDiffuser** (Yang et al., *AAAI 2024*) đúng là **tài sản
  offline** — bỏ tuyên bố tốc độ chưa-kiểm; "quá chậm cho vòng online, render trước"
  thì đúng. Augment ink/degradation mạnh (KuzushijiDiffuser, *MMM 2025*) vì glyph
  tổng hợp sạch hơn mực ván khắc.
- **Cumulative-Class-Prototype (S):** theo paper Siamese chữ-cổ đã kiểm (*Sci.
  Reports* 2022, PMC9436983): **chọn crop đại-diện-nhất mỗi lớp** thay vì lấy trung
  bình — trung bình làm nhoè phương sai ván khắc và bị crop trong-cùng-sách (nguồn
  rò rỉ) chi phối.
- **Sub-center ArcFace (M, gate sau đánh giá trung thực):** dung nạp phương sai nét
  đứt/mực trong-lớp; trong `pytorch-metric-learning`, retrain Kaggle dễ. **Chỉ ship
  nếu** nó cải thiện AURC xuyên-sách.
- **Pooling theo token-mực/foreground (M, chỉ để ablation):** thắng lợi này thuộc về
  ViT+VLAD trong writer-retrieval, **không** phải ArcFace-ResNet — coi như giả thuyết,
  global-embedding là baseline.

### 3.B5 — KHÔNG nên làm
- **VLM-reader làm S1′** (CHURRO / GOT-OCR2.0 / DeepSeek-OCR — đều có thật): hấp dẫn
  nhưng số accuracy của chúng đo trên Latin/CJK-hiện-đại/Âu-in; **chưa cái nào được
  benchmark trên Nôm ván khắc dọc**, bạn sẽ phải tự benchmark lại từ đầu. Không đáng
  cho một luận văn ThS trong tầm thời gian này; **chỉ nêu như future work**.
- **Chạy theo một loss metric-learning "xịn hơn"** (Reality Check: không nhúc nhích).

---

## 4. ĐÁNH GIÁ & MẠCH KỂ LUẬN VĂN

Mạch kể: *"Chúng tôi khai thác nhãn Unicode per-char từ căn chỉnh Nôm↔QN qua đồng
thuận 3 tín hiệu; và báo cáo chất lượng của nó **trung thực** dưới đánh giá open-set,
kiểm soát rò rỉ, có người soát."* **Trung thực CHÍNH LÀ đóng góp** — mỗi con số tụt
khi bỏ rò rỉ là một điểm **cộng** cho bạn, không phải regression.

**Bảng phải có:**
1. **Bảng năng suất tier** — đếm GOLD/SILVER/SYLLABLE/REVIEW (đã có) + **phân rã
   REVIEW theo lý do** (segment-lệch / gap-căn-chỉnh / S3-dưới-ngưỡng) — từ việc 2.0.
2. **Precision per tier × dải-tần-suất, đã soát người, kèm Wilson 95% CI** (n mỗi
   stratum) + **κ** đa-người. Headline: precision SILVER `s2_inter_s3_corrected` kèm
   CI — con số mà bài toán B tồn tại để tạo ra.
3. **Bảng rò rỉ S3:** retrieval@1 và AURC cho {trong-sách vs xuyên-sách (LOBO)} ×
   {chữ-đã-thấy vs tách-chữ}. Ô xuyên-sách/tách-chữ là **số triển khai trung thực**;
   ô trong-sách cho thấy **khoảng rò rỉ**.
4. **Bảng segmentation:** IoU/F1 per-char và tỉ lệ cứu-REVIEW, trung-điểm cổ điển vs
   detector ràng-buộc-số (A1), trên tập trang held-out, lấy TKH/MTHv2 làm mốc ngoài.
5. **Ablation:** S1 / S1+S2 (=GOLD) / +S3 (=SILVER) / +tier IDS cấu trúc — năng suất
   usable-char và precision đã-soát ở từng bước. Cộng ablation luật-reject (isotonic
   vs MLS vs kNN) trên AURC xuyên-sách, và glyph-guard bật/tắt (`eval_char_disjoint.py`
   đã chạy được).

**Hình:** đường risk-coverage/AUGRC có đánh dấu `(τ,δ)` (dập đòn "điểm suy biến" ngay
lập tức); reliability diagram (ECE/Brier) cho calibration; panel trước/sau của cột
được cứu (định tính).

**Tuyên bố đóng góp (chỉ 3 cái này):** (1) bộ dữ liệu crop Hán-Nôm **per-char,
Unicode, phân tầng, siết-ink đầu tiên** dựng bằng weak-supervision dựa căn chỉnh;
(2) phương pháp gán nhãn đồng thuận 3-tín-hiệu có **abstention open-set đã hiệu
chuẩn**; (3) quy trình đánh giá **trung thực, kiểm soát rò rỉ, có người soát** kèm
residual label error. Release kèm phụ lục **Datasheets for Datasets** (Gebru et al.,
*CACM* 2021) ghi rõ xuất xứ trong-sách + loại trừ REVIEW, và giữ metadata
**Croissant** (MLCommons, NeurIPS D&B 2024) + Frictionless mà bạn đã xuất. **ĐỪNG**
khẳng định ý tưởng căn-chỉnh-QN là chưa từng có (IHR-NomDB lưu bản dịch; NomNaOCR
ánh xạ tới nó).

---

## 5. ROADMAP ƯU TIÊN (đòn bẩy cao nhất trước)

| # | Việc | Vì sao | Ở đâu (`evaluation/`) | Công sức | Lợi ích |
|---|------|--------|----------------------|----------|---------|
| 1 | **Đo phân rã REVIEW** → `summary.json` | Biết A có đáng đầu tư nặng không; tự thân là 1 bảng LV | `consensus.py` reasons + `align_production.py::n_review_gap` gộp ở `build_dataset.py` | **S** | **Cao** |
| 2 | **Đường risk-coverage / AURC** từ mảng `recs` có sẵn | Dập đòn "τ suy biến"; chuẩn báo cáo selective-classification (Traub 2024) | `calibrate_s3.py` / `make_s3_report.py` | **S** | **Cao** |
| 3 | **Hoàn tất soát người SILVER** 2 người + κ + Wilson CI + taxonomy CER-HV | Con số phi-vòng-tròn mà B tồn tại để tạo; thiếu nó LV chưa xong | `export_eval_sample.py` (ưu tiên stratum SILVER) → `measure_precision.py` | **S–M** | **Quyết định** |
| 4 | **Eval LOBO + tách-chữ, báo AURC mỗi ô** | Dập đòn rò rỉ; headline trung thực | mở rộng `eval_char_disjoint.py` thêm `--book-holdout` | **M** | **Cao** |
| 5 | **Detector ràng-buộc-số (A1, HRCenterNet→Nôm)** | Cứu phần lệch-số của REVIEW; chuẩn ngành; nâng trần dữ liệu | `evaluation/ver_new/char_detector/`, hook `align_production.py::_pair_new` | **L** | **Cao** |
| 6 | **Nâng reference-bank rẻ:** đa-font + đa-FD + Cumulative-Class-Prototype | Giảm phụ thuộc crop trong-sách rò rỉ & tier FD đơn yếu | `visual_signal.py::_build_fd_index`/`_ref_bank`/proto | **S** | **TB** |
| 7 | **Ship ArcFace head lúc inference + MLS/p-norm + kNN reject** | Thay isotonic overfit; abstention nguyên lý | `infer.py` + `NomEncoder` (+ save `"arc"` ở `kaggle_train.py`), `visual_signal.py::decide` | **M** | **TB** |
| 8 | **Quét residual-error Cleanlab** trên bản release | Con số headline cho dataset paper | mới `evaluation/ver_new/cleanlab_audit.py` (5-fold OOF, Kaggle) | **M** | **TB** |
| 9 | **Tier IDS cấu trúc (B3)** cho đuôi hiếm — **sau** audit độ phủ IDS | Phủ chữ 0-crop không rò rỉ xuyên sách | `visual_signal.py::_ref_bank` thêm tier `ids` | **L** | **TB (đuôi)** |
| 10 | **Datasheet + đồng bộ số đã-soát vào Croissant/Frictionless** | Chuẩn release trung thực | `to_standard.py` + phụ lục | **S** | **TB** |

> A2 (CRAFT) là **dự phòng** nếu ràng-buộc-số của A1 tách-sai quá nhiều.
> PageNet / segment-annotation-free FCN / mọi VLM-reader: **ngoài phạm vi** (code
> train không có / chưa benchmark trên Nôm).

---

## 6. TÀI LIỆU THAM KHẢO THEN CHỐT (chỉ những cái đã kiểm chứng)

- **NomNaOCR: The First Dataset for OCR on Han-Nom Script** — RIVF 2022 —
  https://github.com/ds4v/NomNaOCR — bộ Nôm OCR có trước duy nhất; mức chuỗi, dựng
  trên cùng ánh xạ QN → làm rõ delta per-char của bạn + nguồn test ngoài.
- **IHR-NomDB** — ICDAR 2021 —
  https://link.springer.com/chapter/10.1007/978-3-030-86334-0_6 — DB gần nhất có lưu
  bản dịch QN nhưng gán nhãn thủ công; hợp thức hoá augment glyph tổng hợp.
- **HRCenterNet** — IEEE Big Data 2020 — https://arxiv.org/abs/2012.05739 — detector
  tâm-điểm anchorless cho CJK cổ dày (IoU 0.81 ván khắc MTHv2); fix segmentation A1.
- **CRAFT** — CVPR 2019 — https://arxiv.org/abs/1904.01941 — mine pseudo-box từ số
  chữ đã biết (công thức `s_conf`); công thức re-segment A2.
- **TKH/MTHv2 Datasets** — HCIILAB —
  https://github.com/HCIILAB/TKH_MTH_Datasets_Release — CJK ván khắc có box chữ;
  pretrain + benchmark segment ngoài (Hán in → bắt buộc fine-tune).
- **PageNet** — IJCV 2022 — https://arxiv.org/abs/2207.14807 — phát hiện char mức
  trang chỉ-từ-phiên-âm + thứ tự đọc; kiến trúc tham chiếu (KHÔNG mở code train).
- **Recognition of HCT by Segmentation (Segment-Annotation-Free)** — Peng, Jin et
  al., IEEE **TMM** 2022 — https://arxiv.org/abs/2207.14801 — nhận-dạng-bằng-segment
  chỉ-từ-phiên-âm; phương án A ý niệm (xuất anchor, không box sát).
- **Cross-Modal Prototype Learning (CMPL)** — Pattern Recognition 2022 —
  https://ui.adsabs.harvard.edu/abs/2022PatRe.13108859A — prototype-in +
  nearest-neighbor, khái quát hiện-đại→cổ; nền SOTA hợp thức hoá thiết kế S3.
- **Towards Open-Set Text Recognition via Label-to-Prototype Learning** — Pattern
  Recognition 2023 — https://arxiv.org/abs/2203.05179 — prototype + cột reject =
  GOLD/SILVER/REVIEW của bạn; bán kính reject nguyên lý.
- **A Good Closed-Set Classifier is All You Need?** — ICLR 2022 Oral —
  https://arxiv.org/abs/2110.06207 — Max-Logit-Score abstention; nâng cấp reject B1(c).
- **How to Fix a Broken Confidence Estimator (MaxLogit-pNorm)** — UAI 2024 —
  https://arxiv.org/abs/2305.15508 — chuẩn-hoá logit p-norm post-hoc; sửa
  isotonic-collapse không cần retrain.
- **OOD Detection with Deep Nearest Neighbors (kNN)** — ICML 2022 —
  https://arxiv.org/abs/2204.06507 — reject kNN-distance phi-tham-số; bank của bạn đã
  là chỉ mục kNN, bắt crop cắt-lỗi.
- **Overcoming Common Flaws in Evaluation of Selective Classification (AUGRC)** —
  NeurIPS 2024 Spotlight — https://arxiv.org/abs/2407.01032 — báo cáo risk-coverage/
  AUGRC; xương sống đánh giá S3 trung thực.
- **A Metric Learning Reality Check** — ECCV 2020 —
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf — ArcFace
  là ổn; kỷ luật đánh giá MAP@R không rò rỉ.
- **Pervasive Label Errors (Confident Learning + crowd)** — NeurIPS D&B 2021 —
  https://arxiv.org/abs/2103.14749 — quy trình soát người (~51% flag là lỗi thật);
  template cho audit SILVER + Cleanlab.
- **CCR-CLIP (ảnh→IDS)** — ICCV 2023 — https://arxiv.org/abs/2309.01083 — tham chiếu
  cấu trúc không rò rỉ cho đuôi 0-crop (dùng số liệu accuracy chỉ định tính).
- **HierCode** — Pattern Recognition 2024 — https://arxiv.org/abs/2403.13761 —
  recognizer codebook bộ-thủ nhẹ (ResNet18/34), hợp T4; phủ đuôi.
- **FontDiffuser** — AAAI 2024 — https://arxiv.org/abs/2312.12142 — sinh font
  one-shot diffusion; xuất xứ/trích dẫn cho glyph tham chiếu (tài sản offline).
- **Datasheets for Datasets** — CACM 2021 — https://arxiv.org/abs/1803.09010 — artefact
  tài liệu hoá release cho corpus weak-label trung thực.

---

### Phụ lục — vài chỉnh nhỏ về độ chính xác (để khỏi bị bắt lỗi)
- HRCenterNet: **đừng** khẳng định backbone là HRNet (paper nói "parallelized/
  anchorless"); số trang TKH/MTH là **xấp xỉ** (TKH≈1.000 + MTH≈500 ảnh).
- Segment-Annotation-Free: đúng tác giả/venue là **Dezhi Peng, Lianwen Jin et al.,
  IEEE TMM 2022** (KHÔNG phải "Wang/Yin", KHÔNG phải TPAMI).
- CCR-CLIP: **đừng** trích con số "+28%/~21.7%" (chưa kiểm) — dùng định tính.
- FontDiffuser: bỏ "~6s/char" và "tương đương HCCR" (chưa kiểm).
- CER-HV taxonomy (arXiv 2601.16713): là **preprint**, đừng trích như đã đăng ICDAR 2026.
