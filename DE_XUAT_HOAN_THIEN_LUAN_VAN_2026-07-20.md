# ĐỀ XUẤT TỔNG THỂ HOÀN THIỆN LUẬN VĂN THẠC SĨ CNTT
## Đề tài: Gán nhãn tự động cho kho ngữ liệu Hán Nôm viết tay từ bản dịch quốc ngữ
### Ngày lập: 2026-07-20 — tổng hợp từ khảo sát toàn bộ mã nguồn (11 phân hệ) + khảo sát SOTA quốc tế 2020–2026 + 3 vòng phản biện mô phỏng hội đồng

---

## 0. KẾT LUẬN TỔNG THỂ (đọc trước)

**Đề tài ĐỦ tầm — thậm chí vượt mặt bằng luận văn ThS trong nước.** Sản phẩm đã chạy thật: 82.274 cặp ký tự từ 445 trang / 3 sách, 48.969 crop GOLD với precision đo bằng audit người ~98%, phát hành theo chuẩn quốc tế Croissant/Frictionless/Datasheet. Khảo sát SOTA xác nhận **khoảng trống nghiên cứu là thật**: chưa có công trình nào gán nhãn chữ Nôm MỨC KÝ TỰ tự động bằng đối sánh bản dịch QUỐC NGỮ (khác hệ chữ) — NomNaOCR bán tự động mức dòng từ transcription Nôm sẵn có; IHR-NomDB gán tay ~5k patch; công trình gần nhất (Applied Sciences 2021, nhóm Fribourg) align transcript Nôm **cùng hệ chữ**, không phải quốc ngữ.

**Điểm yếu chí mạng KHÔNG phải khối lượng hay novelty, mà là TÍNH HỢP LỆ CỦA SỐ LIỆU.** Có 4 vấn đề phải xử lý trước bảo vệ (chi tiết §2), tất cả đều vá được trong 2–3 tháng bằng công cụ đã có sẵn trong repo:

1. **SILVER 72,98% là AI chấm, không phải người** (750/750 verdict `source='ai_vision'`) — nếu viết "human audit" là sai lệch phương pháp.
2. **GOLD 98% là số post-hoc**: plan acceptance gốc (n=846, accept ≤17) thực tế FAIL (24–25 defect); 98,0% tính lại trên chính mẫu đã dùng để phát hiện lỗi 㝵/người → cần mẫu SRS xác nhận MỚI.
3. **Circularity chưa được trình bày thành hệ thống** + các số lạc quan cũ đã bị bác vẫn sống trong docstring/README (95,9%/97,6%, s3_pass=0.29, AUC 0.938).
4. **Chưa có thí nghiệm downstream nào** (train model trên nhãn tự sinh) — 84% dataset paper được nhận đều có benchmark; đây là lỗ hổng thực nghiệm lớn nhất.

---

## 1. ĐÓNG GÓP KHOA HỌC NÊN TUYÊN BỐ + CÂU HỎI NGHIÊN CỨU

Luận văn hiện là "mô tả hệ thống". Cần chuyển sang khung nghiên cứu với 3 đóng góp + 3 RQ đo được:

### 3 đóng góp (theo thứ tự mạnh dần về novelty)

- **Đ1 — Căn chỉnh khác-hệ-chữ (cross-script character-level alignment)**: sinh nhãn ký tự Nôm từ bản dịch quốc ngữ (Latin) qua từ điển âm tiết↔ký tự + banded-DP neo từ điển. Toàn bộ văn liệu alignment hiện có (PageNet IJCV 2022, CHDAC/NSR 2023, Bullinger ICCV-W 2025, transcript-mapping PR 2013, Applied Sciences 2021) đều căn transcript **cùng hệ chữ** — đây là điểm mới rõ nhất.
- **Đ2 — Khung weak-supervision 3 tín hiệu phân tầng tin cậy** (S1 OCR / S2 từ điển / S3 thị giác → GOLD/SILVER/SYLLABLE/REVIEW): một thể hiện Snorkel-style cho OCR chữ khối vuông — văn liệu còn trống mảng này.
- **Đ3 — Quy trình kiểm định thống kê + sửa chữa nhãn tự động**: audit mù SRS + acceptance sampling (đúng khung Klie ACL 2024) → phát hiện lỗi hệ thống (㝵/người chiếm ~32% lỗi) → confusion-mining + demote có cấu hình → đo lại. NomNaOCR và cả HisDoc1B đều KHÔNG công bố error-rate có khoảng tin cậy — đây là điểm "vượt chuẩn hiện hành".

### 3 câu hỏi nghiên cứu

- **RQ1**: Căn chỉnh khác-hệ-chữ từ bản dịch QN sinh nhãn ký tự Nôm chính xác bao nhiêu? → trả lời bằng audit người có CI (per-tier).
- **RQ2**: Từng tín hiệu S1/S2/S3 và cổng syllable đóng góp gì vào precision/coverage? → ablation (tính được từ cột `rule` + verdict audit, gần như 0 compute).
- **RQ3**: Model train trên nhãn tự sinh có dùng được không? GOLD+SILVER có hơn GOLD-only? → thí nghiệm downstream (hiện CHƯA có).

---

## 2. BỐN VIỆC BẮT BUỘC TRƯỚC BẢO VỆ (theo mức nghiêm trọng)

### 2.1. Trung thực hóa nguồn verdict (nặng nhất — rủi ro "hỏng buổi bảo vệ")
- SILVER: 100% verdict trong `audit_SILVER/verdicts_ai.jsonl` là `ai_vision` (confidence 0.4–0.7); 2 AI còn bất đồng lớn (verdicts_ai 505/692 correct vs Qwen 293/308). SYLLABLE 6.751 crop: **0 verdict** (chỉ có triage theo S3 — mà S3 đã tự đo là non-discriminative, AUC 0.566–0.572).
- **Việc cần làm**: người chấm 750 item SILVER + 300 SYLLABLE trên grid HTML **đã dựng sẵn** (~3–4 ngày công, 0 compute). Tối thiểu: 200–300 item SILVER để hiệu chuẩn AI-auditor (báo agreement/kappa AI-vs-người) và trình bày verdicts_ai như "AI second opinion" minh bạch.

### 2.2. Hợp lệ hóa con số GOLD (câu hỏi thống kê chắc chắn bị hỏi)
- Sự thật trong repo: plan n=846/accept≤17 → 24–25 defect → **FAIL**; demote 㝵/người rút từ chính mẫu đó; 98,00% (784/800) tính lại trên cùng mẫu = data snooping.
- **Việc cần làm**: rút mẫu SRS xác nhận MỚI trên `labels_final.csv` (n≈450–600, plan acceptance mới — `pipeline/ground_truth` đã tự động hóa toàn bộ). Trình bày trình tự "audit → phát hiện → sửa → xác nhận lại bằng mẫu mới" như một CASE STUDY phương pháp (97,08% → 98,0%) — đây là điểm mạnh nếu chủ động, điểm chết nếu bị phát hiện.
- Kèm: người thứ 2 chấm chung ~200 item → Cohen's kappa (hiện 1 auditor, 0 kappa); phân tích độ nhạy `unsure` (3 cận: unsure=đúng / loại / =sai).

### 2.3. Một nguồn số liệu duy nhất + trình bày circularity chủ động
- Hiện 3 bản labels cùng tồn tại (labels.csv 51.211 GOLD / labels_remediated 50.365 / labels_final 48.969); metadata.csv + croissant.json + release/ đều build từ bản CŨ (lệch ~2.770 dòng usable); sha256 trỏ tới crops.csv đã mất; README các module ghi số run cũ (quarantine 2.299 vs 8 hiện hành).
- **Việc cần làm**: chốt `labels_final.csv` là bản chính thức; rebuild metadata/croissant/datapackage/release từ nó; nối đủ 8 bước FLOW §5 vào `run_pipeline.sh` (hiện chỉ có 0–3); lập file `BANG_SO_LIEU_CHINH_THUC.md` — mọi chương luận văn chỉ trích từ đây.
- Circularity: xóa/chú thích các số đã bị bác còn trong docstring (consensus.py 95,9%/97,6%; channels.py s3_pass=0.29; "AUC 0.938" là retrieval-metric, không phải error-detection). Nguyên tắc trình bày: **mọi claim công bố neo vào audit NGƯỜI; S3 là ranker (retrieval@1 0,89) chứ không phải gate bắt lỗi (error-AUC 0,57) — hai metric đo hai việc khác nhau**; mọi so sánh NomNaOCR dùng LOBO (đã tự chứng minh double-circular); trích Cross-PPI (PNAS 2024) làm hướng khắc phục.

### 2.4. Thí nghiệm downstream (RQ3)
- Train classifier ký tự (tái dùng ResNet18+ArcFace + notebook Kaggle sẵn có) trên 3 cấu hình: **GOLD-only / GOLD+SILVER / +SYLLABLE**, đánh giá trên (a) test subset đã human-audit, (b) 1 cấu hình LOBO theo sách. Kaggle T4 đủ, ~1–2 tuần lịch. Một bảng duy nhất, không cầu toàn SOTA — mục tiêu: "nhãn tự động train được model; SILVER đóng góp X điểm".

---

## 3. ĐỊNH VỊ SOTA — BẢN ĐỒ RELATED WORK (đã kiểm chứng URL)

### 3.1. Dataset chữ Nôm / Hán cổ (Chương 1, mục 1.2)

| Dataset | Quy mô | Mức nhãn | Cách gán | Venue |
|---|---|---|---|---|
| **NomNaOCR** | 2.953 trang, 38.318 patch | dòng | bán tự động (PPOCRLabel + transcription VNPF) | NICS 2022 |
| **IHR-NomDB** | 260 trang, >5.000 patch (+101.621 synthetic) | dòng | tay | ICDAR 2021 |
| **CASIA-AHCDB** | 11.937 trang, 2,2M ký tự, 10.350 lớp | ký tự | tay có kiểm soát | ICDAR 2019 |
| **TKH/MTHv2** | 2.200 ảnh, ~520k ký tự | dòng+ký tự bbox | tay | 2018–2020 |
| **HisDoc1B** | >3M trang, >1 TỶ ký tự | ký tự | **bán tự động + kiểm định** | Sci Data 2025 |
| **Luận văn này** | 445 trang, 82.274 cặp, 66.576 usable, ~1.564 lớp | **ký tự** | **tự động cross-script + audit CI** | — |

Điểm bán: HisDoc1B chứng minh venue lớn chấp nhận nhãn (bán) tự động **miễn có kiểm định chất lượng minh bạch**; luận văn còn hơn ở chỗ công bố error-rate có CI per-tier — điều cả NomNaOCR lẫn HisDoc1B không có.

### 3.2. Sinh nhãn từ transcript (mục 1.3 — định vị Đ1)
- **PageNet** (IJCV 2022): transcript cấp dòng → tự sinh bbox+nhãn ký tự (matching–updating–optimization) — baseline khái niệm gần nhất, cùng hệ chữ.
- **CHDAC/ICDAR 2022** (NSR 2023): chuẩn hóa paradigm "weakly supervised pseudo-label updating" cho Hán cổ (91,43% Norm).
- **Bullinger CTC alignment** (VisionDocs@ICCV 2025): forced alignment tự SỬA nhãn (−1,1 CER); phát hiện "model yếu căn chỉnh tốt hơn" + vòng lặp align→retrain — trích để biện minh vòng audit→remediation.
- **Transcript mapping chữ Hán** (Pattern Recognition 2013): tổ tiên trực tiếp (Bayesian + DP).
- **Applied Sciences 2021** (nhóm Fischer): align văn bản Unicode với ảnh Nôm không cần nhãn tay — **tiền lệ gần nhất cho chính chữ Nôm, PHẢI trích và phân biệt**: họ dùng transcript Nôm cùng hệ chữ, mức trang/dòng; luận văn dùng bản dịch quốc ngữ khác hệ chữ, mức ký tự, có kiểm định thống kê.
- **ICCIES 2025**: xây corpus song song Hán–Việt từ OCR + phrase-matching âm Hán-Việt (Đại Nam Thực Lục) — mẫu về chọn baseline so sánh.
- Mốc tham chiếu: forced alignment với transcript "hiện đại hóa" vẫn đạt ~97,4% word assignment accuracy — con số để đối chiếu alignment accuracy của banded-DP.

### 3.3. Weak supervision + chất lượng nhãn (mục 1.4 — định vị Đ2, Đ3)
- **Snorkel** (VLDB-J 2020): khung lý thuyết cho "nhiều nguồn nhãn nhiễu tương quan → nhãn xác suất" — chưa ai áp dụng cho OCR ideographic.
- **KESAR** (AAAI 2024): tri thức miền (Abductive Learning) sửa pseudo-label — khung học thuật cho việc dùng từ điển + luật consensus làm knowledge base.
- **Klie et al.** (ACL 2024, có giải): CI + acceptance sampling cho annotation error — **nền tảng lý thuyết trực tiếp của Giai đoạn 0**; kèm Klie CL 2023 (18 phương pháp AED) và CL 50(3) 2024 ("đa số dataset quản lý chất lượng nhãn kém" — motivation).
- **PPI** (Science 2023) + **Cross-PPI** (PNAS 2024, chống circularity) — trích khi giải thích PPI auto-skip (S3 coverage 30%) và future work.
- Label noise: Impact of GT Quality (arXiv 2312.09037), DivideMix/RAFNI — luận cứ "lỗi hệ thống + tương quan thì ensemble không cứu được, phải confusion-mining".

### 3.4. VLM trên chữ CJK cổ (biện minh kiến trúc consensus, mục 3.4)
- Benchmark 2025–2026 nhất quán: VLM tổng quát **không đọc được** chữ khối vuông cổ — AncientDoc (CER 32–75%), Chronicles-OCR 2026 (28 model đời mới nhất, fine-grained ≤27,1%), MCS-Bench (ACL 2025), OBI-Bench (ICLR 2025). Khớp đo nội bộ (Qwen-Flash wrong>right).
- Kỹ thuật verifier có số đo: **Consensus Entropy** (AUC 0.9226, hơn VLM-as-Judge 42%; escalate 7,3% mẫu khó); MCQA position-bias (hoán vị đáp án làm giảm 17–30%); sycophancy (không đưa nhãn đương nhiệm vào prompt); over-historicization (post-OCR correction bằng LLM làm GIẢM chất lượng → demote-only là đúng).
- **Khoảng trống công bố được**: chưa tồn tại benchmark VLM nào cho chữ Nôm — một bảng đánh giá nhỏ trên ~200 crop có ground truth audit (protocol OBI-Bench) là kết quả mới, rẻ; làm nếu dư thời gian.

### 3.5. Chuẩn công bố + venue
- Croissant bắt buộc tại NeurIPS từ 2025; nên thêm trường Croissant-RAI (nguồn gốc nhãn, bias 㝵/người đã biết); khai `$schema` Data Package v2.0 + `frictionless validate` vào CI gate; map datasheet sang template **Datasheets for Digital Cultural Heritage** (JOHD 2023, template v2 7/2025) vì đây là dataset di sản.
- Lộ trình công bố thực tế (LREC 2026/ICDAR 2026 đã qua hạn): **JOHD data paper** (rolling, ~1.000 từ, review chỉ kiểm mô tả + lưu trữ mở — rào cản thấp, datasheet JOHD đã build sẵn) → ICDAR 2027/HIP cho phần phương pháp → NeurIPS E&D 2027 nếu muốn top-tier. Trong nước: NICS/KSE/RIVF (nơi NomNaOCR đã đăng).
- Pháp lý: TT 23/2021 — công bố KHÔNG bắt buộc để tốt nghiệp ThS; là lợi thế mềm. Cần xác nhận với khoa: định hướng nghiên cứu (luận văn) hay ứng dụng (đề án).

---

## 4. CẤU TRÚC LUẬN VĂN ĐỀ XUẤT (50–75 trang, chuẩn VN)

- **MỞ ĐẦU**: lý do (trích survey Vietnamese DAR 2025: "thiếu dataset gán nhãn quy mô lớn" là gap số 1), RQ1–3, 3 đóng góp, phạm vi (3 sách/445 trang, 1 nét chữ).
- **CHƯƠNG 1 — Tổng quan**: 1.1 chữ Nôm và bài toán số hóa; 1.2 dataset Nôm/Hán cổ (bảng §3.1); 1.3 sinh nhãn từ transcript và khoảng trống cross-script (§3.2); 1.4 weak supervision + kiểm định chất lượng nhãn (§3.3).
- **CHƯƠNG 2 — Phương pháp gán nhãn tự động**: 2.1 kiến trúc tổng thể; 2.2 trích xuất song ngữ (Kimhannom + VietOCR + chuẩn hóa QN, parse 9 cột); 2.3 căn chỉnh banded-DP neo từ điển (**Đ1**); 2.4 consensus 3 tín hiệu + phân tầng (**Đ2**); 2.5 mô hình thị giác (CenterNet + ràng buộc N + seam carving; encoder ArcFace; lý do bác DINOv2).
- **CHƯƠNG 3 — Kiểm định và sửa chữa chất lượng nhãn (Đ3)**: 3.1 thiết kế audit (SRS/stratified/acceptance, Wilson/CP — trích Klie); 3.2 kết quả audit + kappa + độ nhạy unsure; 3.3 phân tích lỗi hệ thống (case study 㝵/người: phát hiện → mine → demote → xác nhận lại 97,08→98,0); 3.4 giới hạn S3 + fusion (negative results có phương pháp); 3.5 phát hành chuẩn quốc tế.
- **CHƯƠNG 4 — Thực nghiệm**: 4.1 số liệu dataset cuối (labels_final); 4.2 ablation tín hiệu + alignment cũ/mới; 4.3 downstream + LOBO (RQ3); 4.4 so sánh NomNaOCR/PaddleOCRv5 (khai rõ khác granularity); 4.5 thảo luận hạn chế (mất lớp "người", 1 bộ sách, API ngoài).
- **KẾT LUẬN + hướng phát triển**: mở dataset + shared task (chưa có cuộc thi ICDAR nào cho Nôm); radical/IDS-based verification cho cặp nhầm lẫn; Cross-PPI; cứu 1.923 crop 㝵 bằng human glyph-anchor.
- **PHỤ LỤC**: config, selftest (223 assertions), datasheet, ví dụ audit grid, Q&A.

### Trình bày negative results như tài sản (điểm cộng nếu chủ động)
1. DINOv2 non-discriminative trên Nôm (có bảng đo).
2. S3 là ranker chứ không phải error-gate (retrieval 0,89 vs error-AUC 0,57).
3. Fusion Path A chưa áp (train-AUC 0,54, 25 negative, không held-out) — trình bày là THIẾT KẾ + negative result, gỡ nhánh PROMOTE.
4. NomNaOCR double-circular → LOBO; Qwen-Flash wrong>right → VLM demote-only.

---

## 5. KẾ HOẠCH 12 TUẦN

| Tuần | Việc | Ghi chú |
|---|---|---|
| **1** | **Đóng băng & commit** (rủi ro mất trắng cao nhất): commit 7 file untracked + confusion_fixes.yaml + labels_final + toàn bộ ground_truth artifacts; merge branch vào main; .gitmodules cho gannhanocr-fd/nom-embed; bật `--strict`; khôi phục verdicts_001–006.jsonl nếu còn; **CODE-FREEZE tính năng** | Đang có 7 file .py + toàn bộ artifact audit chưa track trên branch chưa merge |
| **1–2** | **Một nguồn số liệu**: nối đủ 8 bước vào run_pipeline.sh; chạy end-to-end từ cache → labels_final + release mới; xóa metadata cũ; sửa README stale; lập BANG_SO_LIEU_CHINH_THUC.md; rút mẫu SRS xác nhận GOLD mới n≈450–600 | Chặn ~1/3 câu hỏi hội đồng, 0 compute |
| **2–5** | **Audit người đợt 2** (~30–40 giờ chấm, bắt đầu sớm): GOLD confirm ~450 → SILVER 750 (tối thiểu 300 + hiệu chuẩn AI) → SYLLABLE 300 → người thứ 2 chấm 200 (kappa); chạy lại estimate → report "human" | Grid HTML đã dựng sẵn hết |
| **4–6** | **RQ3 downstream** (song song audit): GOLD vs GOLD+SILVER vs +SYLLABLE; test human-audit + LOBO | Kaggle T4, hạ tầng sẵn |
| **5–6** | **Ablation**: banded-DP vs `_pair_old` vs Levenshtein cũ; 3 chế độ reseg; có/không S3; có/không syllable_gate; đo độ phủ từ điển (7.522 key) trên corpus | Chạy từ cache, không cần API |
| **6–8** | Audit ≥10 crop/cặp cho ~10 cặp confusion lớn (đời→代 195, mới→買 183, còn→群 173…) → mở rộng confusion_fixes.yaml có kiểm chứng; audit ~150 crop REVIEW để có số recall/bỏ-sót đầu tiên | |
| **7–10** | **Viết luận văn** theo mục lục §4; mọi số trace về BANG_SO_LIEU_CHINH_THUC.md | |
| **11–12** | Buffer + bảo vệ thử: slide, demo run 1 sách từ cache, 1 trang Q&A (§6); (tùy chọn) nộp JOHD data paper | |

**Rủi ro trễ & phòng ngừa**: (1) audit người không ai làm → bắt đầu tuần 2, phương án lùi SILVER n=300 + CI rộng; (2) Kimhannom API chết → KHÔNG re-OCR, đóng băng cache như primary data, ghi rõ trong luận văn; (3) sa đà quay lại fusion/kênh nặng ("gần xong") → đã đo AUC 0,54, không đổi được điểm bảo vệ; (4) rebuild làm lệch verdict đã neo → sau lần chạy end-to-end tuần 2 thì KHÓA dataset_out.

### Cắt khỏi luận văn (dead weight)
Fusion Path B + kênh nặng (kraken, Qwen-235B full, nna_lobo runtime); nhánh PROMOTE của fuse_stage; PPI (chưa chạy được, coverage 30%); Kish n_eff + blind-MCQ → trình bày là THIẾT KẾ có trích dẫn; MTH 4.7GB + kkanji2 549MB + dataset/ rỗng → xóa hoặc ghi "không dùng"; step2/3/4 retired → 1 đoạn đối chứng DINOv2; scripts/ PP-OCRv5 → 1 câu "bằng chứng loại Paddle".

---

## 6. Q&A DỰ KIẾN CHO BUỔI BẢO VỆ

1. **"Nhãn máy tạo thì tin được không?"** → bảng audit người per-tier có CI + acceptance sampling (Klie ACL 2024) + trình tự phát hiện–sửa–xác nhận lại.
2. **"Khác gì NomNaOCR / công cụ CLC?"** → bảng so sánh mức nhãn: dòng-bán-tự-động vs **ký-tự-tự-động-cross-script-có-kiểm-định**; CLC làm chuyển tự text→text, không gán nhãn ảnh.
3. **"Hệ thống tự xác nhận nhãn của chính mình?"** → mọi claim neo audit người độc lập; S3 chỉ là ranker; LOBO cho mọi so sánh NomNaOCR; Cross-PPI là hướng khắc phục đã trích dẫn.
4. **"Vì sao 98% mà không phải 97,08%?"** → khai trình tự demote + mẫu SRS xác nhận mới (nếu chưa chạy mẫu mới thì đây là câu chết — xem §2.2).
5. **"Consensus 3 tín hiệu mà tier lớn nhất chỉ dùng 2?"** → GOLD-direct là phát hiện của chính luận văn: audit thống kê bắt được lớp lỗi consensus không tự thấy (㝵/người); confusion-mining là cơ chế sửa — trình bày thành mục "phân tích lỗi hệ thống".
6. **"Mất lớp 'người' thì dataset còn đại diện không?"** → known-bias khai trong datasheet + kế hoạch cứu bằng human glyph-anchor (future work).
7. **"Tái lập thế nào khi dùng API ngoài?"** → cache OCR đóng băng là primary data + 8 bước tự động 1 lệnh + .gitmodules + selftest 223 assertions.
8. **"Nguồn gốc từ điển và bản quyền tư liệu?"** → khai QuocNgu_SinoNom_TongHop3.csv / SinoNom_Similar_Dic (nghi nhóm CLC-HCMUS — phải xác minh và xin trích dẫn) + bản quyền 3 PDF sách tôn giáo ngay trong C2 + datasheet.

---

## 7. DANH MỤC THAM KHẢO LÕI (đã kiểm chứng tồn tại)

**Nôm/CJK datasets**: NomNaOCR (NICS 2022, ieeexplore 10013842); IHR-NomDB (ICDAR 2021); CASIA-AHCDB (ICDAR 2019); TKH/MTH (HCIILAB); HisDoc1B (Sci Data 2025); Kuzushiji/CODH; survey historical datasets (IJDAR 2022, arXiv 2203.08504); survey Vietnamese DAR (arXiv 2506.05061).

**Alignment/weak supervision**: PageNet (arXiv 2207.14807); CHDAC (NSR 2023); Bullinger CTC (arXiv 2508.07904); transcript mapping (PR 2013); Applied Sciences 2021 (mdpi 2076-3417/11/11/4894); ICCIES 2025 (CCIS 2585); Snorkel (VLDB-J 2020); KESAR (AAAI 2024); self-training HTR (arXiv 2206.03149); StrDA (arXiv 2410.09913); GT4HistOCR (arXiv 1809.05501); approximate GT (IJDAR 2024).

**Chất lượng nhãn/thống kê**: Klie ACL 2024 (aclanthology 2024.acl-long.837); Klie CL 49(1) 2023; CL 50(3) 2024; PPI (Science 2023); Cross-PPI (PNAS 2024); Impact of GT Quality (arXiv 2312.09037).

**VLM**: AncientDoc (arXiv 2509.09731); Chronicles-OCR (arXiv 2605.11960); MCS-Bench (ACL 2025); OBI-Bench (ICLR 2025); Consensus Entropy (arXiv 2504.11101); MCQA bias (arXiv 2509.16805); sycophancy (arXiv 2408.11261); over-historicization (arXiv 2510.06743); CHURRO (arXiv 2509.19768); PaddleOCRv5 Sino-Vietnamese (arXiv 2510.04003).

**Chuẩn công bố**: Croissant (arXiv 2403.19546); Croissant-RAI (arXiv 2407.16883); Data Package v2.0; Datasheets (Gebru, CACM 2021); Datasheets for DCH (JOHD 10.5334/johd.124); TT 23/2021/TT-BGDĐT; SMT Nôm→QN (ACM ICIIT 2022).
