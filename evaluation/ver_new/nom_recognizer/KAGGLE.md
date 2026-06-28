# Train recognizer Nôm trên Kaggle (ĐỘT PHÁ #1: student vượt thầy ở OOV/lỗi)

Train ResNet34 → softmax 1591-lớp trên ~60k nhãn tự-sinh, chứng minh nó **vượt SinoNom
OCR** ở **tầng teacher-OOV** (chữ kimhannom không xuất được) + **lỗi của thầy**. Đẩy HF.

## 0. Chuẩn bị (local)
```bash
.venv/bin/python evaluation/ver_new/nom_recognizer/pack_for_kaggle.py
#   -> kaggle_rec_pkg/  (gold/ + silver/ crops + labels.csv + train_recognizer.py + encoder_best.pt)
```

## 1. Upload `kaggle_rec_pkg/` làm 1 Kaggle Dataset (vd `nom-recognizer-data`).

## 2. Notebook (import `train_recognizer_kaggle.ipynb`)
- **Add Input** → dataset trên. **Settings: GPU T4 x2** (KHÔNG P100) · **Internet On**.
- **Add-ons → Secrets → `HF_TOKEN`** (token write) nếu muốn đẩy HF; điền `HF_REPO='mdnt571/nom-recognizer'`.
- **Run All.** Mỗi epoch in:
  `TEST student X teacher Y | OOV student Z (n=..) | teacher-wrong recovery W (n=..)`

## 3. HEADLINE — đọc đúng (điều kiện sống còn)
| Chỉ số | Tuyên bố |
|---|---|
| **OOV student > 0** (teacher = 0) | ✅ THẮNG TUYỆT ĐỐI — nhận chữ Nôm-thuần mà OCR Hán không xuất được |
| **teacher-wrong recovery** | ✅ student sửa được % lỗi của thầy |
| student vs teacher (tổng) | thầy thường ≥ (GOLD thiên lệch về chỗ thầy đúng) — **ĐỪNG tuyên thắng tổng** |

Câu chống đạn: *"khớp/thua thầy ở chữ phổ biến, nhưng vượt hẳn ở chữ Nôm-thuần OCR-Hán không xuất được + ở lỗi của thầy — với 0 nhãn tay."*

## 4. Bảng 3-chiều A/B/C (cho luận văn)
- **A** = student trên nhãn ĐỒNG THUẬN (`--target consensus`, cell A)
- **B** = student trên nhãn OCR THÔ (`--target ocr`, cell B) — control
- **C** = teacher (ocr_char, không train)
So 3 trên cùng test tách-sách: **A−C** trên tầng OOV/lỗi = headline; **A>B** chứng minh *nhãn đồng thuận khử nhiễu > nhãn thô* (cơ chế Born-Again/self-distillation).

## 5. Kéo model về + dùng
```bash
huggingface-cli download mdnt571/nom-recognizer recognizer.best.pt \
  --local-dir evaluation/ver_new/nom_recognizer
# eval lại local (đối chiếu proxy head 80%): so trong eval_teacher_vs_student.py
```

## 6. Bánh đà (giảm REVIEW) — bước sau
Dùng recognizer.best.pt chấm lại REVIEW (tín hiệu S3 mạnh hơn cosine bank) → cứu thêm
nhãn → giảm REVIEW. (Hook: thêm recognizer làm tier `s4_recognizer` trong consensus,
hoặc chấm posterior để re-run banded-DP.)

---
**Tài nguyên:** ResNet34 @160px batch 128 vừa T4 16GB; warm-start từ encoder (`--init
encoder_best.pt`) → hội tụ nhanh (~30 epoch ≈ 1–2h). Smoke local đã verify build+train+
eval. Trích dẫn: Pseudo-Label>Human (Interspeech'22), Born-Again/self-distillation
(NeurIPS'20), Confident Learning (JAIR'21), Scius-Bertrand (Applied Sci 2021, prior art Nôm).
