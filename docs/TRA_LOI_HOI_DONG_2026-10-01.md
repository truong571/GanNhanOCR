# Trả lời góp ý: đánh giá thực nghiệm, tổng quan tài liệu, so khớp ảnh (01/10/2026)

Mọi con số dưới đây lấy từ bộ đo tái lập của dự án (`scripts/measure/`, `measure_out/`) hoặc từ script trong
`lab/hoi_dong_2026-10-01/`. Mỗi số ghi mức chứng cứ: **ĐO** (so với nhãn người), **ƯỚC LƯỢNG** (so với dị bản do người số
hoá, là cận dưới), **SUY ĐOÁN** (không có nhãn người).

> **Làm rõ một tiền đề.** "Kinh Hán Nôm" (kim) trong dự án là **dịch vụ OCR bên ngoài** của HCMUS
> (`kimhannom.clc.hcmus.edu.vn`, gọi qua `core/ocr/ocr_api.py`). Dự án **không huấn luyện** mô hình OCR đó và không biết dữ
> liệu huấn luyện của nó. Đóng góp của đề tài là **pipeline gán nhãn tự động** đặt quanh kim: gióng chữ kim đọc với bản phiên
> Quốc ngữ, cắt và kiểm ảnh từng chữ, xếp tầng tin cậy. Các mô hình dự án tự huấn luyện là mô hình **phụ trợ** (mục 1.4).

---

## 1. Đánh giá thực nghiệm

### 1.1 Trước và sau pipeline (ĐO trên nhãn người)

"Trước" = chữ kim đọc thô tại ô. "Sau" = nhãn GOLD của pipeline. Độ đúng GOLD chỉ tính trên ô GOLD, nên luôn đi kèm
**độ phủ** (tỉ lệ ô được gán GOLD).

| Bộ (loại bản) | Ô có chữ người | kim thô đúng (strict) | GOLD đúng (strict / dị thể V1+) | Độ phủ GOLD |
|---|---:|---:|---:|---:|
| Lục Vân Tiên 1916 (mộc bản, IHR-NomDB) | 13.743 | 95,23 % | **97,95 %** / 99,68 % | 93,8 % |
| Truyện Kiều 1872 (mộc bản, IHR-NomDB) | 21.369 | 97,46 % | **98,60 %** / 100 % | 97,4 % |
| Borg.tonch.18 (chép tay Công giáo) | 75.670 | 50,49 % | **90,96 %** / 94,40 % | 90,3 % |
| Borg.tonch.34 (chép tay Công giáo) | 13.530 | 38,03 % | **93,84 %** / 96,70 % | 68,7 % |

Nguồn: `measure_out/<bộ>/{ihr_endtoend,borg_endtoend}/summary.json`; CI 95 % bootstrap cụm trang trong các tệp đó.

- **Chữ viết tay:** pipeline **sửa** nhãn (chỉ 56 % / 44 % nhãn GOLD trùng chữ kim), nhờ bộ chọn chữ bằng ảnh trong tập ứng
  viên R(âm). Độ đúng tăng từ 38–50 % lên 91–94 % (strict).
- **Mộc bản:** kim vốn đã tốt; pipeline **lọc** chứ không sửa (100 % nhãn GOLD trùng chữ kim), độ đúng tăng 1–3 điểm, đổi lại
  bỏ trống 2,6–6,2 % ô.
- **Sau bước kiểm ảnh + chữ (gold_exact = ok):** đúng cả chữ lẫn vị trí ảnh 99,21 % (L16), 99,88 % (TK), 98,33 % (B18), 97,70 %
  (B34), độ phủ 22 / 51 / 8 / 13 % ô (`measure_out/gold_exact/summary.json`).
- Bộ chưa có nhãn người: KVK, LVT1883 **ƯỚC LƯỢNG** (trùng dị bản 87,4 % / 79,8 %, cận dưới); STT, Chrestomathie **SUY ĐOÁN**
  (STT ước 93–97,5 %, `measure_out/_tn9/KET_QUA.md`).

### 1.2 CER, WER, Top-1/Top-5

**CER mức trang** (Levenshtein / số chữ người; `lab/hoi_dong_2026-10-01/cer_before_after.py`):

| Bộ | kim thô | nhãn sau pipeline (mọi tầng) | lỗi THAY: kim → pipeline |
|---|---:|---:|---:|
| L16 | 3,41 % | 5,02 % | 2,79 → 1,67 % |
| TK | 2,62 % | 2,86 % | 2,20 → 1,36 % |
| Borg.18 | 64,36 % | 36,74 % | 38,7 → 5,9 % |
| Borg.34 | 79,39 % | 48,00 % | 54,7 → 4,2 % |

CER của pipeline tính cả ô bỏ trống (REVIEW không mang nhãn) là lỗi xoá, nên ở mộc bản CER nhích lên dù lỗi **thay** giảm.
Với bộ gán nhãn có chọn lọc, chỉ số phù hợp là (độ đúng, độ phủ) ở mục 1.1. **WER** không tách riêng được: mỗi chữ Nôm là một
âm tiết, WER mức âm tiết chính là CER; nhãn người không có ranh giới từ.

**Top-1 / Top-5 của bộ chọn chữ bằng ảnh** (chọn trong ~25 ứng viên R(âm) ∪ {kim}; học trên sách kia — LOBO;
`lab/hoi_dong_2026-10-01/topk_chooser.py`):

| Bộ thử ← bộ học | kim top-1 | Bộ chọn top-1 / top-5 | Trần (đáp án có trong ứng viên) |
|---|---:|---:|---:|
| Borg.18 ← Borg.34 | 53,9 % | **92,6 % / 96,1 %** | 97,8 % |
| Borg.34 ← Borg.18 | 40,2 % | **92,9 % / 98,1 %** | 98,8 % |
| L16 ← TK | 95,5 % | **96,2 % / 98,2 %** | 98,7 % |
| TK ← L16 | 97,7 % | **98,0 % / 99,3 %** | 99,5 % |

### 1.3 Ablation (từng thành phần)

| Thành phần | Bộ | Không có → có | Mức | Nguồn |
|---|---|---|---|---|
| kim đọc chế độ Nôm thay Hán | Borg | precision kim 39,6 → 53,7 % | ĐO | `measure_out/_tn7_kim/KET_QUA.md` |
| Gửi kim cả trang thay từng ảnh câu | L16 / TK | 49,2 / 42,3 → 97,2 / 98,2 % | ĐO | `docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` §10.2 |
| Luật GOLD "chữ kim ∈ R(âm)" | L16 / TK | 95,23 → 97,95 % / 97,46 → 98,59 % | ĐO | `ihr_endtoend/summary.json` |
| Hộp ảnh pitch → visual_dp (DP thị giác) | Borg.18 / .34 | đúng vị trí 61,5 → 99,4 % / 58,6 → 99,8 % | ƯỚC LƯỢNG* | `docs/HOP_ANH_TN6_2026-09-28.md` |
| Bộ chọn chữ bằng ảnh + sửa cổng crop | L16 / TK | độ phủ 84,2 → 93,8 % / 86,8 → 97,4 %, độ đúng giữ 98 % | ĐO | `measure_out/_tn8/KET_QUA.md` |
| Bộ chọn chữ bằng ảnh | Borg.18 / .34 | độ phủ 13,1 → 90,3 % / 1,9 → 68,7 %; đúng V1+ 91,3 → 94,4 % / 91,8 → 96,7 % | ĐO | như trên |
| Kiểm ảnh + chữ (gold_exact) | L16 / TK / Borg | đúng 97,95 → 99,21 / 98,60 → 99,88 / 94,40 → 98,33 % (giảm độ phủ) | ĐO | `gold_exact/summary.json` |
| Đọc 2 lượt kim Hán+Nôm cho STT (l1skel_l2) | STT | GOLD 57,6–60,4 → 65,2–71,0 % | SUY ĐOÁN | `measure_out/_tn9/KET_QUA.md` |

\* Vị trí "đúng" đo bằng hộp chữ người do máy gióng nên hơi lạc quan.
Đã thử và **bác bỏ** bằng số đo: engine OCR khác (12–14 % trên IHR), đọc nhiều lượt / phóng ảnh / khử nhiễu / gửi từng dòng
cho kim, ghép hai lượt đọc theo toạ độ (GOLD −1,9 đến −4,9 điểm), phóng ×2 trước khi dò hộp, DINOv2 (không phân biệt được chữ
Nôm). Nguồn: `docs/DIEU_HUONG_PIPELINE_2026-09-28.md`, `measure_out/_tn7_kim`, `_tn9`, `_tn10`.

### 1.4 Dữ liệu và các mô hình phụ trợ

**Dữ liệu xử lý** (`data/*/SOURCE.md`, `docs/BAO_CAO_TONG_HOP_2026-10-01.md`): 10 bộ, 275.136 ô chữ, khoảng 1.689 trang Nôm.

| Bộ | Nguồn | Loại chữ | Trang | Nhãn người |
|---|---|---|---:|---|
| Sách Thánh Truyện Q2/Q4/Q11 | PDF trong `data/` (kho lưu chưa ghi rõ) | chép tay | 160/145/143 | không |
| Lục Vân Tiên 1883 | BnF Gallica | thạch bản | 105 | không (dị bản) |
| Kim Vân Kiều 1884 | BnF Gallica | thạch bản | 163 | không (dị bản 1871/1872) |
| Chrestomathie 1872 | BnF Gallica | thạch bản văn xuôi | ~66 | không |
| Lục Vân Tiên 1916, Truyện Kiều 1872 | IHR-NomDB | mộc bản | 104 / 162 | có (theo câu) |
| Borg.tonch.18 / .34 | Thư viện Vatican | chép tay Công giáo | 529 / 112 | có (bản phiên Excel) |

**Tiền xử lý:** lấy ảnh 300 dpi từ PDF/IIIF, khử nhiễu (chia nền, kéo dải 2–98 %), cắt khung trang; OCR Quốc ngữ (VietOCR /
tesseract, hoặc bản phiên người ở IHR, Borg).
**Gán nhãn (tự động 100 %):** kim OCR → dò hộp chữ (CenterNet) → gióng chữ kim với âm Quốc ngữ bằng quy hoạch động + từ điển
R(âm) (~104.177 cặp) → cổng kiểm → bộ chọn chữ bằng ảnh → xếp tầng GOLD / SYLLABLE / GOLD_text_only / REVIEW / QUARANTINE.
**Chia tập:** bộ dữ liệu giao nộp **không chia** train/val/test (quyết định A-10, 16/09); bộ có nhãn người chỉ mang cờ
`evaluation_only`. Khi huấn luyện mô hình phụ trợ, dự án chia rời trang hoặc học chéo sách (bảng dưới).

| Mô hình phụ trợ | Kiến trúc | Dữ liệu học | Nhãn | Chia | Kết quả | Nguồn |
|---|---|---|---|---|---|---|
| Detector v1 | CenterNet ResNet34+FPN | pretrain 3.199 trang chữ Hán (MTH/TKH) → 445 trang STT, 66.630 hộp | máy | rời trang (44 trang val) | F1 0,844 (IoU 0,5) | `train_crop/README.md` |
| Detector v2 | khởi tạo từ v1 | 215 trang thạch bản + 356 trang STT | máy | rời trang theo sách | cắt vào thân chữ 9,2 → 0,2 % (L83) | `docs/VONG9_DETECTOR_V2_APDUNG_2026-09-23.md` |
| Encoder v1 (nom-embed) | ResNet18 + ArcFace, 1.591 lớp | 51.195 crop STT + glyph font/FD | máy | **không rời trang** (rò trang) | val top-1 0,785 | checkpoint |
| Encoder v2 (ArcFace) | ResNet18 + sub-center ArcFace K=3, SAM | 59.823 crop STT + 1.564 glyph FD | máy | rời trang 80/10/10 | val top-1 0,806 | `ArcFace/` |
| Bộ kiểm ảnh↔chữ vft / hand | ResNet nhỏ 64×64, CosFace | crop người Borg (4/5 khối trang) + IHR (3/4 trang) + glyph | **người** | LOBO theo sách | AUC 0,95–0,97 | `models/gold_exact/MANIFEST.json` |
| Bộ chọn chữ (chon_chu) | logistic 2 tầng, 46 đặc trưng | Borg, IHR | **người** | LOBO + CV 5 khối trang | mục 1.2 | `models/chon_chu/MANIFEST.json` |
| CNN self-training (rescue STT) | SE-ResNet, 1.331 lớp | 54.094 crop STT | máy | ngẫu nhiên theo dòng | val top-1 0,799 | `lab/enhanced_self_training_v3` |

**Cần nói rõ khi bảo vệ:** nhãn người IHR và Borg **có được dùng để huấn luyện** bộ kiểm tra và bộ chọn chữ (luôn học chéo sách
khi đo trên chính các sách đó), nên câu "nhãn người chỉ để đo" và "Borg không vào tập huấn luyện" (`docs/GOLD_CHINH_XAC_2026-09-27.md`
§11.7) phải sửa thành "không vào **bộ dữ liệu giao nộp**".

### 1.5 Mô hình gốc so với mô hình học trên dữ liệu chuyên biệt

| So sánh | Gốc | Học trên dữ liệu dự án | Nguồn |
|---|---|---|---|
| Embedding chung DINOv2 vs encoder v1 | cùng chữ / khác chữ cách nhau +0,012; truy hồi 0/200 | cách nhau +0,29; truy hồi 76,5 % | `pipeline/align_engine/visual_signal.py:5-10` |
| Hộp hình học (pitch) vs gióng bằng encoder (visual_dp) | đúng vị trí 61,5 / 58,6 % | 99,4 / 99,8 % | `docs/HOP_ANH_TN6_2026-09-28.md` |
| Detector v1 vs v2 (thạch bản) | cắt vào thân chữ 9,2 / 2,5 % | 0,2 / 0,1 % | `docs/VONG9_DETECTOR_V2_APDUNG_2026-09-23.md` |
| Không kiểm vs bộ kiểm vft | đúng hai vế 96,4 / 98,2 % | 99,5 / 99,9 % (giữ 24 / 53 %) | `lab/thu_nghiem_anh_chu/TN1_cong_kiem/KET_QUA.md` |
| Bộ chọn chữ học trên sách Borg kia | kim top-1 53,9 / 40,2 % | 92,6 / 92,9 % | mục 1.2 |
| NomNaOCR bản công bố trên STT | ~9 % | (bản tự tinh chỉnh bị loại vì vòng tròn nhãn) | `NomNaOCR/HUONG_DAN_SU_DUNG.md` |

---

## 2. Tổng quan tài liệu và tính mới

Bản đầy đủ: **`docs/KHAO_SAT_TAI_LIEU_2026-10-01.md`**. Gồm 66 công trình, mỗi công trình đều mở được trang gốc (arXiv / DOI /
hội nghị), 31 công trình kiểm chéo qua Crossref. Tài liệu không xác minh được thì xếp riêng ở §6 và không dùng.

**Các nghiên cứu trước đã làm gì:**
- Nhận dạng chữ Nôm: tách chữ hình học + phân lớp, rồi CNN, rồi CRNN theo dòng.
  Bộ dữ liệu công khai (IHR-NomDB, NomNaOCR) chỉ có nhãn mức dòng/cột.
- Gióng ảnh với bản phiên (DTW/HMM, CTC, PageNet): đều làm **trong cùng một hệ chữ**.
- Gần nhất với đề tài: Scius-Bertrand và cộng sự (Applied Sciences 2021, DOI 10.3390/app11114894). Nhóm này gióng bản chép tay
  Nôm với văn bản Nôm Unicode không cần nhãn người, dùng YOLOv5 học trên trang sinh từ phông chữ.
- VLM tổng quát còn yếu trên chữ cổ: tốt nhất 31,9 % trên Chronicles-OCR.

**Điểm mới có thể khẳng định** (dùng "theo khảo sát của chúng tôi"):
1. Gióng ảnh chữ Nôm với văn bản **Quốc ngữ (khác hệ chữ)** ở mức từng chữ: một âm tiết ứng với nhiều chữ Nôm, giải bằng từ điển
   R(âm) + quy hoạch động.
2. Gióng hộp bằng quy hoạch động thị giác: so ảnh hộp chỉ với glyph font/FontDiffuser của các chữ mà từ điển cho phép cho âm đó.
   Trên Borg, đúng vị trí tăng từ 61,5 lên 99,4 %.
3. Bộ dữ liệu crop xếp tầng tin cậy, trong đó cổng kiểm chỉ hạ tầng, không sửa nhãn. Kèm phép đo trên **hai thể loại** có nhãn
   người: mộc bản và chép tay Công giáo.
4. Kết quả âm có số đo: engine OCR khác, VLM và DINOv2 không bổ trợ được kim.

**Không được nhận là mới:**
- huấn luyện mô hình OCR, Transformer OCR;
- dùng LMM trong pipeline (mã Qwen-VL có nhưng `run_pipeline.sh` không gọi);
- huấn luyện diffusion (chỉ dùng glyph FontDiffuser sinh sẵn);
- "hoàn toàn không dùng nhãn người" (xem mục 1.4).

**Trích dẫn cần sửa trong repo:**
- `pipeline/consensus_fusion/README.md`:
  - dòng 76 ghi "VLM < 25 % char-F1" — bài gốc nói tốt nhất 31,9 %;
  - dòng 64 gắn arXiv 2508.07904 với kraken, nhưng bài đó là căn chỉnh CTC thư Bullinger.
- `pipeline/consensus_fusion/qwen_verifier.py:8`: arXiv 2410.15393 là CalibraEval (thiên lệch lựa chọn của LLM chấm điểm),
  không phải "position bias" của VLM.
- Trích NomNaOCR đúng là Dang et al., RIVF 2022 (DOI 10.1109/RIVF55975.2022.10013842).
- IHR-NomDB có chữ "Handwritten" trong tên bài, nhưng LVT1916 là **mộc bản**.

---

## 3. Hiện trạng so khớp ảnh trong mã

- **Nhãn được quyết ở mức văn bản:** chữ kim đọc ra phải thuộc R(âm Quốc ngữ).
- **Ảnh được so bằng embedding học sâu kiểu metric learning:** ResNet18 + ArcFace / sub-center ArcFace, cùng CNN kiểm tra học
  bằng CosFace.
- **Không dùng** SIFT/ORB/so khớp mẫu/pHash (0 kết quả grep trong `pipeline/`, `core/`, `train_crop/`, `scripts/`).
- **Không có** Siamese contrastive/triplet trong luồng chạy, **không có** ViT/VLM.

| Bước | Cơ chế | Mã | Trạng thái |
|---|---|---|---|
| Dò chữ | CenterNet (heatmap tâm chữ) | `train_crop/infer_centernet.py` | mặc định |
| Gióng văn bản | Needleman–Wunsch có dải; chữ kim ∈ R(âm) qua từ điển, dị thể V1+ | `pipeline/align_engine/anchor_align.py`, `gold_exact/common.py` | mặc định |
| Gióng hộp thị giác (visual_dp) | DP Viterbi + forward–backward; phát xạ = cos(embedding crop, glyph font/FD của chữ ∈ R(âm)) | `pipeline/align_engine/visual_dp.py` | mặc định (STT, Borg, TK, Chr; L16 lai) |
| Cắt ảnh | hình học (nới lề, seam-carve chữ láng giềng); crop chuẩn vuông (Otsu) | `build_dataset.py`, `gold_exact/crop_chuan.py` | mặc định |
| STT hai lượt kim | ghép lượt Hán + Nôm theo dòng (Needleman–Wunsch, khớp dị thể) | `pipeline/stt_hai_luot/` | mặc định cho STT |
| Bộ chọn chữ bằng ảnh | 35–46 đặc trưng: cos crop–glyph, kNN với nguyên mẫu cùng sách (khối trang khác), nguyên mẫu người (LOBO), điểm CNN kiểm → logistic | `pipeline/chon_chu/` | bật cho sách in và Borg; tắt cho STT |
| Kiểm ảnh + chữ (gold_exact) | CNN vft/hand (CosFace), cos với glyph nhãn so với chữ cạnh tranh, M-OCR, hình học hộp kim; chỉ hạ, không đổi nhãn | `pipeline/gold_exact/signals_*.py` | mặc định sau gộp |
| Trùng ảnh | md5 chính xác (không cảm nhận) | `remediation/census.py` | mặc định |
| Bộ chữ người Borg | DP gióng chữ người ↔ hộp với encoder v1/v2; Paddle PP-OCRv6 làm phép kiểm độc lập (hồ sơ đóng băng) | `pipeline/borg_human/` | chạy riêng |

FontDiffuser chỉ dùng **ảnh glyph sinh sẵn** (89.898 tệp); pipeline không chạy mô hình diffusion lúc gán nhãn.

---

## 4. Giới hạn và việc nên làm thêm

- Độ đúng chỉ **ĐO** được trên 4 bộ có nhãn người (IHR, Borg). Ở các bộ này Quốc ngữ do người cung cấp, nên số là **cận trên**.
  STT (bộ giao nộp chính) và Chrestomathie chỉ suy đoán.
- kim là hộp đen: không loại trừ được việc dữ liệu huấn luyện của kim trùng với sách đánh giá.
- Encoder học trên nhãn máy của chính pipeline (có tính vòng tròn); encoder v1 bị rò trang.
- Chưa có thí nghiệm hạ nguồn: huấn luyện một bộ nhận dạng trên GOLD rồi thử trên sách khác. Đây là cách trực tiếp nhất để chứng
  minh giá trị của bộ dữ liệu.
- Chưa có ablation riêng cho phần đóng góp của glyph FontDiffuser.
- Chưa có cận thống kê cho độ đúng GOLD ở sách không có nhãn người (ví dụ prediction-powered inference).

---

## 5. Bộ nào có thể nằm trong dữ liệu huấn luyện của kim? Xếp mức độ khó

### 5.1 Chồng lấn với dữ liệu huấn luyện của kim

kim là hộp đen; công bố chỉ nói chung: khoảng 200.000 ảnh gán nhãn để huấn luyện OCR, đối tác **Nom Foundation**, Viện Trần Nhân
Tông ([HCMUS, 19/07/2026](https://hcmus.edu.vn/ai-giai-ma-di-san-han-nom-mo-canh-cua-dua-kho-tri-thuc-nghin-nam-den-gan-cong-chung/)),
thêm 5 TB dữ liệu thư viện Đại học Columbia năm 2025 ([Digitizing Vietnam](https://www.digitizingvietnam.com/vi/tools/digital-humanities-tools/kim-han-nom)).
Không có danh sách sách. Bằng chứng tự đo (0 API, `lab/hoi_dong_2026-10-01/`):

| Phép kiểm | LVT 1916 | Truyện Kiều 1872 | Ý nghĩa |
|---|---|---|---|
| kim trả **đúng mã PUA riêng của Nôm Foundation** ở ô nhãn người là PUA | 439/518 (84,7 %) | 1.104/1.155 (95,6 %) | Mã PUA do Nôm Foundation tự gán, OCR không thể tự đoán ra → kim chắc chắn học dữ liệu mã hoá theo Nôm Foundation (`kim_pua_signature.py`) |
| Câu có trong NomNaOCR (bộ công khai, nguồn Nôm Foundation) | 99,5 % | 99,8 % | Cả hai cuốn nằm trong dữ liệu công khai thường dùng để huấn luyện (`data/<Bộ>/manifest.tsv`) |
| kim đúng ở câu train − câu val của IHR-NomDB | −0,7 điểm [−2,9; +1,3] | −0,7 điểm [−1,8; +0,2] | Không có lợi thế ở phần train → không phải chỉ học phần train của IHR; không loại được việc học cả cuốn (`kim_train_val_ihr.py`) |
| Chỗ nhãn NomNaOCR ≠ IHR: kim theo bên nào | 2/2 theo NomNaOCR | 6 theo IHR, 4 theo NomNaOCR | Quá ít mẫu (14 vị trí) để kết luận (`kim_nomnaocr_signature.py`) |

**Chốt:**
- **LVT 1916, Truyện Kiều 1872 (IHR-NomDB): nhiều khả năng nằm trong, hoặc trùng nguồn với, dữ liệu huấn luyện của kim**
  (nguồn Nôm Foundation, đối tác dữ liệu của kim; dấu vết PUA; có trong NomNaOCR). Vì vậy độ đúng kim 95–97 % và GOLD
  98–99 % trên hai bộ này là **lạc quan**, không dùng làm bằng chứng tổng quát hoá.
- **Borg.tonch.18/34 (Vatican), Sách Thánh Truyện: không có dấu hiệu.** kim chỉ đúng 38–56 %, và kim không nằm trong nguồn
  công bố của các bản này. Đây là thước đo thật cho sách ngoài miền dữ liệu của kim.
- **LVT 1883, Kim Vân Kiều 1884, Chrestomathie (BnF Gallica): không xác định** được. Không có danh sách để đối chiếu, và các
  bộ này không có nhãn người.
- kim trả mã PUA ở mọi bộ (0,2–5,5 % số chữ đọc ra), nên bộ từ vựng của kim chắc chắn theo bảng mã Nôm Foundation.

### 5.2 Xếp mức độ khó

Thước đo chung cho cả 10 bộ là tỉ lệ ô mà chữ kim là một cách đọc của âm Quốc ngữ, theo từ điển (`do_kho_10_bo.py`). Kèm
theo là độ đúng kim thật (bộ có nhãn người) và loại chữ.

| Mức | Bộ | Loại chữ | kim khớp âm | kim đúng (nhãn người) | GOLD hiện tại | Dùng trong luận văn |
|---|---|---|---:|---:|---:|---|
| **Dễ** | Truyện Kiều 1872 | mộc bản | 98,7 % | 97,5 % | 97,6 % | trần trên; **không** dùng chứng minh tổng quát hoá (có thể kim đã học) |
| **Dễ** | Lục Vân Tiên 1916 | mộc bản | 97,4 % | 95,2 % | 95,4 % | như trên |
| **Trung bình** | Kim Vân Kiều 1884 | thạch bản | 92,3 % | — (dị bản 87 %) | 90,5 % | sách in ngoài IHR; độ đúng ước lượng |
| **Trung bình** | Chrestomathie 1872 | thạch bản văn xuôi | 80,8 % | — | 74,2 % | văn xuôi: gióng khó hơn thơ |
| **Trung bình** | Lục Vân Tiên 1883 | thạch bản | 79,6 % | — (dị bản 80 %) | 75,4 % | âm Quốc ngữ mượn ấn bản khác |
| **Khó** | Sách Thánh Truyện Q2/Q4/Q11 | chép tay | 69,6–75,6 % | — | 59,9–66,3 % | bộ giao nộp chính; độ đúng suy đoán |
| **Khó** | Borg.tonch.18 | chép tay Công giáo | 55,9 % | 50,5 % | 90,3 %* | thước đo THẬT cho chữ viết tay |
| **Rất khó** | Borg.tonch.34 | chép tay Công giáo | 34,7 % | 38,0 % | 68,7 % | 24,6 % ô có âm giữ chỗ (không có chữ) |

\* Nhờ bộ chọn chữ bằng ảnh học trên Borg.34 (học chéo sách), nên GOLD cao dù kim yếu.

**Cách trình bày:** báo kết quả theo 3 mức. "Dễ" dùng để kiểm tính đúng của pipeline. "Trung bình" và "Khó" dùng để chứng minh
pipeline hoạt động ngoài miền dữ liệu của kim. Chỉ số đo thật cho mức khó lấy từ Borg.
