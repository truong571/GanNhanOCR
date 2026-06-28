# Hướng ĐỘT PHÁ cho luận văn (nâng tầm, không chỉ vá thêm)

> Nghiên cứu biên giới có kiểm chứng (lọc hype). Xếp theo **tác động × khả thi** cho
> ThS trên Kaggle P100/T4. Các cải tiến tăng-dần (encoder, conformal, head, IDS,
> detector) đã làm — đây là tầng *nâng câu chuyện đề tài*.

---

## ⭐ TOP PICK — ĐÓNG VÒNG LẶP: train recognizer Nôm trên nhãn tự-sinh, chứng minh nó **VƯỢT** SinoNom OCR đã tạo ra nó

**Đây là nước đi tác-động-cao nhất và khả thi nhất. Làm cái này.**

### Vì sao nâng tầm đề tài
Chuyển tuyên bố đóng góp từ *"xây được dataset gán nhãn tự động (một sản phẩm)"* →
*"giám sát yếu sinh ra một recognizer giỏi hơn chính OCR-thầy đã bootstrap nó"* — một
narrative tác-động-cao đã được công nhận (*Pseudo Label Is Better Than Human Label*,
Interspeech 2022; giải thích cơ chế bằng *Born-Again / self-distillation = label
smoothing theo từng mẫu*, NeurIPS 2020). **Chưa công trình Hán-Nôm nào tuyên bố điều
này** (NomNaOCR, Scius-Bertrand, IHR-NomDB đều ở mức dòng/phát-hiện). Trả lời thẳng
câu hỏi sát thủ của hội đồng: *"dataset này rốt cuộc dùng để làm gì?"*

### ⚠️ Scoping TRUNG THỰC (phần phải làm đúng — verifier đánh dấu must_fix)
**ĐỪNG hứa "vượt thầy" ở mức tổng.** Chữ phổ biến thì SinoNom OCR mạnh; student của
bạn sẽ **hoà hoặc thua** ở đó. Nói thẳng từ đầu. Chiến thắng **bảo vệ được, gần như
bất khả công, và bạn ĐÃ có dữ liệu để chứng minh** là **hẹp**:
- **Tầng teacher-OOV** — chữ mà charset kimhannom **không thể xuất ra**, nhưng căn-
  chỉnh-theo-QN + từ điển cấp đúng Unicode. *Bất kỳ* nhận đúng nào ở đây = thắng tuyệt
  đối, do giám sát yếu, không phải rò rỉ nhãn. **Kết quả sạch nhất.**
- **Đuôi hiếm / lỗi-dư** — nơi alignment cấp nhãn mà thầy sai.
- Câu an toàn (không thắng chỗ nào cũng không sao): *"khớp thầy với 0 nhãn tay ở chữ
  phổ biến, và VƯỢT HẲN ở tầng teacher-OOV / đuôi hiếm."* → câu này chống đạn.

### Kế hoạch cụ thể (`evaluation/ver_new/nom_recognizer/`)
1. **`train_recognizer.py`** — tái dùng backbone ResNet34/ArcFace thành classifier
   softmax 1591-lớp (bạn đã train đúng backbone này; ~1–3h P100). Train trên GOLD
   (~51k) + SILVER sạch.
2. **Bảng so sánh 3-chiều** (headline, theo protocol Interspeech'22):
   - (A) student train trên **nhãn đồng thuận tự-sinh**
   - (B) student train trên **nhãn OCR thô** (cùng crop, nhãn của thầy) — control
   - (C) **chính SinoNom thầy**
   - Tất cả chấm trên **ground-truth người-soát, split tách-sách (LOBO)**. Tái dùng
     `eval_book_disjoint.py`, `export_eval_sample.py`, `measure_precision.py`. Delta
     (A − C) trên tầng OOV/hiếm = số headline, kèm **Wilson CI** (n nhỏ ~150–300).
3. **Bắt buộc tự đo (must_fix):** accuracy per-char của kimhannom **chưa công bố** →
   phải tự đo trên cùng tập LOBO đã-soát, và **kiểm charset kimhannom** để CHỨNG MINH
   các lớp OOV thật sự không xuất được. Thiếu cái này thì headline vô căn cứ.
4. **Bộ khử-nhiễu-nhãn + ablation** (`train_robust.py`): Cleanlab 5-fold OOF prune
   trên SILVER/SYLLABLE, rồi ablate {nhãn thô} → {nhãn đồng thuận} → {+ train chống
   nhiễu} đối chiếu gold người. Đây là **giải thích cơ chế** *vì sao* student vượt
   thầy (soft target khử nhiễu = label smoothing theo mẫu) — biến "số may mắn" thành
   phát hiện có kiểm soát. Tập trung ablation vào SILVER/SYLLABLE (nơi nhiễu thật sự).

### Công sức / rủi ro
**M** (backbone + split + harness soát đã có; mới = so sánh 3-chiều + tầng OOV soát-
người). Rủi ro **TB, giảm hết được bằng cách phát biểu chính xác**: kiểm rò rỉ bằng
LOBO, chỉ tuyên thắng ở tầng OOV/hiếm, báo CI.

### Trích dẫn (đã kiểm chứng)
- *Pseudo Label Is Better Than Human Label* — Interspeech 2022 — isca-archive.org/interspeech_2022/hwang22c_interspeech.html — tiền lệ hợp thức hoá headline.
- *CTC Transcription Alignment of the Bullinger Letters* — arXiv 2508.07904 (2025) — **analog gần nhất**: HTR train trên nhãn tự-căn cải thiện CER ~1.1pp + cải thiện alignment ("model yếu cho alignment tốt hơn → lặp được").
- *Born-Again / Self-Distillation as Instance-Specific Label Smoothing* — ICML 2018 / NeurIPS 2020 — *vì sao* student-vượt-thầy không nghịch lý.
- *Confident Learning (Cleanlab)* — JAIR 2021 — **arXiv 1911.00068** (URL bundle sai) — bước khử nhiễu.
- *Transcription Alignment of Historical Vietnamese Manuscripts* — Scius-Bertrand et al., Applied Sciences 2021 — **prior art Nôm gần nhất, PHẢI trích + phân biệt** (họ làm phát-hiện+căn-chỉnh, KHÔNG có recognizer per-char vượt thầy).
- *Sino-Vietnamese OCR via PaddleOCRv5* — arXiv 2510.04003 (2025) — điểm so sánh cùng miền (trần ~50% exact). **Làm mềm: split/eval của họ riêng tư → trích là "kết quả gần đây", đừng nói "ta beat benchmark của họ".**

---

## RUNNER-UP A — NomNaBench: benchmark Hán-Nôm **mức ký tự** đầu tiên + leaderboard VLM/LMM
- **ROI công bố cao nhất** (dataset+benchmark paper ở NeurIPS D&B / LREC / ICDAR không
  cần tuyên SOTA-recognizer), và biến **đánh giá kiểm-soát-rò-rỉ trung thực** của bạn
  thành *điểm bán hàng* thay vì footnote phòng thủ. Theo mẫu **OBI-Bench (ICLR 2025)**
  từng đưa một chữ ngách lên top-venue. NomNaOCR/IHR-NomDB chỉ ở mức dòng → per-char
  Unicode + split chữ-chưa-thấy là "đầu tiên" thật.
- **Vì sao #2:** (a) "VLM tối tân thất bại trên Nôm ván khắc dọc" là **giả thuyết phải
  ĐO**, đừng khẳng định; (b) phản biện về quy mô (3 sách/1591 lớp/445 trang) là thật →
  scope "đầu tiên" trung thực, lý tưởng mở rộng ảnh dòng công khai NomNaOCR thành
  per-char. **Kết hợp mạnh nhất: làm TOP PICK trước, rồi release NomNaBench với
  recognizer của bạn làm baseline tham chiếu.** Hai cái cộng hưởng. Dựng ở
  `evaluation/ver_new/nombench/`, tái dùng `eval_char_disjoint.py`, `eval_rare_tail.py`,
  pipeline verify.csv + `label_error_candidates.csv` (→ task phụ NomNaBench-LE: phát
  hiện lỗi nhãn, theo *Pervasive Label Errors*, NeurIPS D&B 2021).

## RUNNER-UP B — Vòng re-label lặp ràng-buộc-số (1 thế hệ kỷ luật)
- **Tài sản thật sự mới:** căn-chỉnh-QN cho **số ký tự đúng N mỗi cột** — ràng buộc cấu
  trúc cứng mà *chưa self-training OCR nào có* (vì không ai có phiên-âm-song-song). Dùng
  posterior recognizer-v1 chạy lại banded-DP, **cứu phần REVIEW**, train v2. Bullinger
  (ICCVW 2025) de-risk: recognizer yếu cho alignment tốt hơn (+~1.1pp CER), lặp được.
- **must_fix — bỏ hype:** EM **KHÔNG** có "bảo đảm hội tụ chứng minh được" → nói *"vòng
  có ràng-buộc, theo dõi empirically"*. Giữ 1–2 thế hệ (ngân sách P100), báo đường cong
  per-generation.

---

## ❌ KHÔNG ĐÁNG cho ThS này (frontier nhưng hype/không khả thi)
- **VLM làm labeler / "VLM tự OCR"** (mọi kiểu). AncientDoc (ICLR 2025): VLM off-the-
  shelf **yếu** (CER ~32). Fine-tune VLM bilingual-prompt = canh bạc LoRA 1–2 tuần, lợi
  ích so với fusion dict+visual+head **hiện có** gần như biên ở chữ phổ biến — đúng chỗ
  bạn KHÔNG cần. (Bài "Risk-Controlled Generative OCR" bị verifier bác: không phải
  conformal/LTT; `conformal_reject.py` của bạn còn chặt hơn.) → Tối đa: **VLM-as-verifier
  training-free CHỈ trên tập REVIEW**, gated, làm tín hiệu thứ 5. Không phải trụ cột.
- **CalliReader (glyph-in-LLM adapter):** cần A6000 + 742k ảnh. Future work.
- **DAN / PageNet full-page:** PageNet **không có code train**; DAN data-hungry (445
  trang quá ít), CER của DAN là chữ Latin. Chỉ trích related-work/future-work, **đừng
  hứa retrain.**

## Câu chuyện đề tài SAU khi làm TOP PICK
Từ *"dataset + phương pháp gán nhãn trung thực"* → **"bánh đà giám sát yếu tự cải
thiện"**: căn chỉnh QN → nhãn đồng thuận → **recognizer vượt thầy ở tầng OOV/hiếm** →
quay lại làm tín hiệu thứ 4/5 → cứu thêm GOLD. Đóng góp: (1) dataset per-char + (2)
recognizer chứng minh vượt OCR-thầy dưới eval kiểm-soát-rò-rỉ + (3) vòng phản hồi đóng.

## Kết luận thẳng
**Có hướng đột phá thật, và bạn đã có sẵn mọi tài sản cho nó (TOP PICK).** Không phải
mơ mộng VLM — mà là *dùng đúng 60k nhãn đã sinh* để train một recognizer rồi chứng minh
nó vượt thầy ở chỗ thầy mù. Nếu thời gian eo hẹp: chỉ riêng TOP PICK (bảng 3-chiều +
tầng OOV soát-người) đã đủ nâng đề tài; NomNaBench là phần thưởng nếu nhắm công bố.
