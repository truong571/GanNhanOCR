# NomNaOCR làm Recognizer #3 (thay cho Paddle arXiv 2510.04003)

Recognizer Nôm mức-chuỗi **đọc cột dọc trực tiếp** (432×48, sequence theo chiều cao →
đọc top→bottom), map **1-1 với cột của kinhhannom, KHÔNG cần xoay ảnh**. Vốn từ **7.479
chữ Nôm thật** (Truyện Kiều, Lục Vân Tiên, ĐVSKTT…) — khác hẳn Paddle arXiv (dòng ngang,
domain synthetic Trung văn) đã fail.

---

## 0. Mình ĐÃ làm sẵn (không phải làm lại)

| Việc | Vị trí | Trạng thái |
|---|---|---|
| Clone repo ds4v/NomNaOCR | `NomNaOCR/ds4v_repo/` | ✅ |
| Tải + giải nén weights (679MB zip) | `NomNaOCR/weights/NomNaOCR_CRNNxCTC.h5` (51MB) + `NomNaOCR_SC-CRNNxCTC.h5` | ✅ |
| Code inference | `nomnaocr_rec.py` | ✅ đã verify: build lại đúng kiến trúc, load được weights, chạy ra `(1,25,7482)` |
| Code build vocab | `build_vocab.py` | ✅ |
| Code test 10 trang vs kim | `run_nomnaocr_consensus.py` | ✅ |

**Chỉ còn THIẾU 1 thứ để chạy thật: `vocab.txt`** (bảng chữ↔chỉ số) — vì CTC xuất ra chỉ
số, phải có đúng danh sách 7.479 chữ theo đúng thứ tự lúc train mới giải mã ra chữ đúng.
Danh sách đó dựng từ file transcript `All.txt` của dataset (trên Kaggle).

---

## 1. Môi trường (paddle/TF không có wheel Python 3.14 → dùng venv 3.10 riêng)

```bash
# 1 lần, tạo venv Python 3.10 riêng cho TensorFlow (numpy<2)
/opt/homebrew/bin/python3.10 -m venv ~/nomna_tf
~/nomna_tf/bin/pip install "tensorflow==2.15.1" pillow
```
> Lưu ý: env mình dựng lúc test nằm trong thư mục scratchpad tạm của phiên → sẽ bị xoá.
> Hãy tạo env cố định như trên. Code + weights trong `NomNaOCR/` thì cố định, không mất.

## 2. Lấy `vocab.txt` — bước quyết định (2 cách)

### Cách A — Nhanh (chỉ cần file text All.txt)
1. Vào Kaggle dataset **`quandang/nomnaocr`** → tải file `Datasets/Patches/All.txt`
   (hoặc dùng Kaggle API: `kaggle datasets download quandang/nomnaocr -f Datasets/Patches/All.txt`
   — cần `~/.kaggle/kaggle.json` từ Kaggle → Account → Create API Token).
2. Dựng vocab:
   ```bash
   ~/nomna_tf/bin/python build_vocab.py /đường/dẫn/All.txt
   ```
   Script tự kiểm tra `== 7479 chữ`. Nếu đúng 7479 → vocab CHÍNH XÁC ✅.

### Cách B — Chính xác tuyệt đối (nếu Cách A báo lệch số)
Tải TOÀN BỘ dataset (ảnh + All.txt), rồi chạy đúng `DataImporter` của repo để dựng vocab
(loại bỏ hệt các patch mà training đã loại). Xem `ds4v_repo/Text recognition/loader.py`.
Chỉ cần khi Cách A không ra đúng 7479 (do khác phiên bản dataset).

## 3. Chạy test 10 trang (kim vs NomNaOCR, ép 9 cột)

```bash
cd NomNaOCR
~/nomna_tf/bin/python run_nomnaocr_consensus.py --book SachThanhTruyen4 --n 10
```
In ra: mỗi trang kim_chars / nna_chars / match% / sub/del/ins, và bản cạnh nhau từng cột ở
`NomNaOCR/out/nna_<page>.txt`. So sánh trực tiếp với kết quả Qwen ở `evaluation/qwen_test/`.

## 4. Ghép vào consensus (theo đúng thiết kế đã chốt)

- **Tọa độ**: kinhhannom (9 cột) — gốc, không đổi.
- **Recognizer #3**: NomNaOCR đọc từng cột → `NomNaRecognizer.recognize([col_img])` → chuỗi chữ
  → align vào cột kim (đã có sẵn trong `run_nomnaocr_consensus.py`).
- **Vote**: mỗi vị trí lấy đa số trong {kim, NomNaOCR, (Qwen)}; cả 3 khác → fallback kim / REVIEW.
- **Qwen**: giữ làm CỜ bất đồng (không phải phiếu GOLD).
- **Segmentation**: cột kim đếm-lệch với NomNaOCR → nghi box sai → reseg/REVIEW.

Muốn nối thẳng vào engine chính (`pipeline/align_engine`), export kết quả NomNaOCR per-page
ra JSON (giống cache Qwen) rồi cho `decide_label` đọc thêm cột `nna_char` — vì engine chạy ở
venv 3.14, còn NomNaOCR ở venv 3.10, nên chạy **offline precompute** rồi engine đọc file
(KHÔNG import chéo env).

## 5. Kỳ vọng & cảnh báo thật

- **Domain**: NomNaOCR train chủ yếu trên **mộc bản in văn học/sử** (Kiều, LVT, ĐVSKTT). Sách
  của bạn (Sách Thánh Truyện Công giáo) cũng là **mộc bản in** → gần domain, khả năng khá hơn
  Paddle arXiv nhiều. Nhưng **chưa kiểm chứng** trên đúng loại chữ này → phải chạy §3 để biết.
- **Phủ từ**: 7.479 chữ — nếu Sách Thánh Truyện có chữ Nôm hiếm ngoài tập này, model trả `[UNK]`.
- **Nếu kém**: fine-tune (KHÔNG train from scratch) trên vài trăm–nghìn cột của bạn — dùng
  `CRNNxCTC_finetune.ipynb` (pretrain synthetic IHR-NomDB → fine-tune dữ liệu bạn).
- Muốn char-accuracy/CER cao hơn sequence-accuracy: dùng `SC-CNNxTransformer` (nặng hơn).

## 6. File trong thư mục này

| File | Vai trò |
|---|---|
| `nomnaocr_rec.py` | Recognizer CRNN×CTC (build arch + load `.h5` + CTC decode). Smoke: `python nomnaocr_rec.py weights/NomNaOCR_CRNNxCTC.h5` |
| `build_vocab.py` | Dựng `vocab.txt` từ `All.txt` |
| `run_nomnaocr_consensus.py` | Test 10 trang vs kim (ép 9 cột) |
| `ds4v_repo/` | Repo gốc (kiến trúc/layers được import từ đây) |
| `weights/*.h5` | Trọng số fine-tuned (đã giải nén). `weights/nomnaocr_weights.zip` chứa cả model khác (Transformer…) — xoá được sau khi lấy đủ |
| `huong-dan-NomNaOCR.md` | Hướng dẫn gốc của bạn |
| `prepare_finetune_data.py` | Tạo data fine-tune (ảnh cột-chunk + nhãn chuỗi) từ GOLD/SILVER |
| `finetune_data/` + `finetune_data.zip` | **Data fine-tune ĐÃ TẠO SẴN** (17.5k patch) — upload cái .zip |

---

## 7. Fine-tune NomNaOCR trên chữ Sách (data đã tạo sẵn ✅)

**Vì sao phải fine-tune:** NomNaOCR pretrained đọc style của nó (Kiều/ĐVSKTT) 100% nhưng
Sách Thánh Truyện chỉ ~9% — lệch domain. Đã kiểm chứng: pipeline đúng, chỉ thiếu train
trên đúng nét chữ của bạn.

**Data đã dựng sẵn** (`prepare_finetune_data.py`, chạy 100% local):
- Cắt mỗi cột kim thành **chunk chữ LIÊN TỤC được gán nhãn** (GOLD/SILVER), ≤10 chữ/chunk
  (giới hạn 27 timestep CTC). Chữ REVIEW ở giữa → **cắt đứt run** để crop KHÔNG dính chữ chưa gán.
- Neo theo **OCR cache** (cột đầy đủ) + khớp `ocr_char` → chỉ số cột chính xác → run = các chỉ số **liền kề**.
- Kết quả: **17.572 patch** (train 14.226 / val 1.665 / test 1.681), 62k chữ, vocab 1591,
  đúng format notebook: `Datasets/Patches/<book>/*.jpg` + `All.txt`/`Validate.txt`/`Test.txt`.

**Tạo lại / tuỳ chỉnh:**
```bash
.venv/bin/python NomNaOCR/prepare_finetune_data.py --tiers GOLD,SILVER --maxlen 10
# --tiers GOLD        (nhãn sạch nhất, ít data hơn) | --maxlen 8 (chunk ngắn hơn)
```

**Fine-tune trên Kaggle GPU — script `finetune_kaggle.py` (ĐÃ TEST chạy được):**

Upload **1 file duy nhất**: **`nomna_finetune_all.zip`** (đã gộp sẵn — Kaggle tự giải nén).
Tạo Kaggle Dataset tên `nomna-finetune`. Bên trong có sẵn: `Datasets/Patches/`,
`NomNaOCR_CRNNxCTC.h5`, `Text recognition/`, `finetune_kaggle.py`.

Bật **GPU** trong notebook rồi chạy 1 lệnh (chỉ 1 đường dẫn input):
```bash
!python "/kaggle/input/nomna-finetune/finetune_kaggle.py" \
   --dataset_dir /kaggle/input/nomna-finetune/Datasets/Patches \
   --pretrained  /kaggle/input/nomna-finetune/NomNaOCR_CRNNxCTC.h5 \
   --repo        "/kaggle/input/nomna-finetune/Text recognition" \
   --epochs 30 --batch 64 --out /kaggle/working
```
(Nếu tên dataset khác `nomna-finetune`, sửa lại cho khớp đường dẫn `/kaggle/input/<tên>/`.)
Nó tự: rebuild head 1591 lớp (phủ hết chữ Sách; head gốc chỉ 82%), **transfer CNN+BiGRU**
từ bản gốc (`by_name+skip_mismatch`), train CTC, EarlyStopping. Ra 2 file ở `/kaggle/working`:
`finetuned_CRNNxCTC.h5` + `finetuned_vocab.txt` → **tải cả 2 về**.

3. Ráp lại: đặt 2 file đó vào `NomNaOCR/weights/`, rồi
   ```python
   rec = NomNaRecognizer("weights/finetuned_CRNNxCTC.h5", "weights/finetuned_vocab.txt")
   ```
   Chạy lại `run_nomnaocr_consensus.py --chunk 8` để đo độ khớp mới với kim.

> Đã verify local (tf_env, CPU): load transfer OK, CTC loss giảm, lưu checkpoint OK. Trên
> Kaggle GPU sẽ nhanh hơn nhiều. Muốn chuỗi train dài hơn (ít fragment) → điền `ocr_char`
> cho cả vị trí REVIEW trong `prepare_finetune_data.py` (đổi độ sạch nhãn lấy độ dài).
