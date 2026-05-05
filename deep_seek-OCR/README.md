# DeepSeek-OCR experiment flow

Thu muc nay chua flow thu nghiem DeepSeek-OCR rieng cho repo GanNhanOCR.
Code doc lai output san co tu pipeline chinh (`prepared/<book>/detected` va
`prepared/<book>/labeled`) roi chay model `deepseek-ai/DeepSeek-OCR` de nhan
dang tung crop ky tu hoac OCR ca trang.

Mac dinh output duoc ghi vao:

```text
DeepSeek-OCR/results/<book>/
```

## Kiem tra moi truong

```bash
.venv/bin/python DeepSeek-OCR/check_environment.py
```

DeepSeek-OCR chinh thuc khuyen nghi CUDA + torch/flash-attn. Neu may khong co
GPU, script van co the load bang Transformers o CPU/MPS khi thu vien ho tro,
nhung se rat cham va co the khong du RAM.

## Dry-run du lieu

Lenh nay khong load model, chi kiem tra detection/crop/label dau vao:

```bash
.venv/bin/python DeepSeek-OCR/run_deepseek_ocr_flow.py \
  --book CacThanhTruyen2 \
  --pages page_0012 \
  --max-crops 10 \
  --dry-run
```

## Chay DeepSeek-OCR tren crop ky tu

```bash
.venv/bin/python DeepSeek-OCR/run_deepseek_ocr_flow.py \
  --book CacThanhTruyen2 \
  --pages page_0012 \
  --max-crops 10 \
  --mode crops
```

Neu chi muon kiem tra cache local va khong cho Hugging Face download:

```bash
.venv/bin/python DeepSeek-OCR/run_deepseek_ocr_flow.py \
  --book CacThanhTruyen2 \
  --pages page_0012 \
  --max-crops 1 \
  --mode crops \
  --offline
```

Ket qua chinh:

```text
DeepSeek-OCR/results/CacThanhTruyen2/deepseek_ocr/labels_rec.csv
DeepSeek-OCR/results/CacThanhTruyen2/deepseek_ocr/evaluation.csv
DeepSeek-OCR/results/CacThanhTruyen2/deepseek_ocr/evaluation.json
DeepSeek-OCR/results/CacThanhTruyen2/deepseek_ocr/page_0012.json
```

## OCR ca trang

```bash
.venv/bin/python DeepSeek-OCR/run_deepseek_ocr_flow.py \
  --book CacThanhTruyen2 \
  --pages page_0012 \
  --mode pages
```

Text tung trang se nam trong:

```text
DeepSeek-OCR/results/CacThanhTruyen2/page_ocr/page_0012.txt
```

## Ghi chu

- `--prepare auto` se dung output san co neu detection da ton tai. Neu chua co,
  script moi goi step 1 va step 2 cua pipeline chinh.
- So sanh trong `evaluation.*` dung `prepared/<book>/labeled/labels.csv` lam
  reference gan nhan hien co, khong phai ground truth doc lap.
- Model mac dinh la `deepseek-ai/DeepSeek-OCR`; co the doi bang `--model`.
