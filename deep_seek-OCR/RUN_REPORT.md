# Run report

Ngay chay: 2026-05-05

## Moi truong hien tai

```text
Python:       3.13.11
Platform:     macOS-26.4.1-arm64-arm-64bit-Mach-O
torch:        2.11.0
torchvision:  0.26.0
transformers: 4.46.3
tokenizers:   0.20.3
Pillow:       12.2.0
opencv:       4.13.0
hf hub:       0.36.2
CUDA:         False
MPS:          False
Model cache:  6.2G, deepseek-ai/DeepSeek-OCR downloaded
```

## Dry-run

Lenh:

```bash
.venv/bin/python DeepSeek-OCR/run_deepseek_ocr_flow.py \
  --book CacThanhTruyen2 \
  --pages page_0012 \
  --max-crops 10 \
  --dry-run
```

Ket qua:

```json
{
  "book": "CacThanhTruyen2",
  "pages": ["page_0012", "page_0014", "page_0016", "page_0018", "page_0020"],
  "mode": "crops",
  "max_crops": null,
  "dry_run": true,
  "crop_count": 935,
  "label_reference": true
}
```

## Thu inference that

Lenh:

```bash
.venv/bin/python DeepSeek-OCR/run_deepseek_ocr_flow.py \
  --book CacThanhTruyen2 \
  --mode crops \
  --force
```

Ket qua: model va weight da tai xong, nhung inference that khong chay duoc tren
may hien tai vi code remote cua `deepseek-ai/DeepSeek-OCR` goi `.cuda()` truc
tiep trong `model.infer(...)`.

Loi dung tai crop dau tien:

```text
AssertionError: Torch not compiled with CUDA enabled
```

Status JSON da luu tai:

```text
DeepSeek-OCR/results/CacThanhTruyen2/deepseek_ocr/run_status.json
```
