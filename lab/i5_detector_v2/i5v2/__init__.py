"""i5v2 — gói TỰ CHỨA để huấn luyện/đánh giá detector CenterNet v2 (thạch bản + STT) trên Kaggle.

Không import gì từ repo GanNhanOCR: chỉ cần torch, torchvision, numpy, opencv. Các mô-đun:
  model.py        bản sao nguyên văn train_crop/model_centernet.py (state_dict tương thích pipeline)
  loss.py         Focal + L1 size/offset (train_crop/train_centernet.py)
  decode.py       heatmap -> hộp; lớp Detector (boxes_for_image) như infer_centernet.CenterNetDetector
  data.py         Dataset nhãn yếu + ignore mask + augment thạch bản; sampler cân bằng miền
  pitch_decode.py bản sao pipeline/align_engine/char_detector/pitch_decode.py (ink_cut_cells cho ô tham chiếu)
  evalref.py      đánh giá trung thực: ok50 / miss / extra / |dy| / % tầng n==N / cắt thân chữ / STT F1
  trainer.py      vòng huấn luyện: init v1, EMA, AMP, cosine-warmup, resume last.pt, guard STT, early stop
  hub.py          đẩy/kéo checkpoint lên Hugging Face hub (tuỳ chọn, như ArcFace/hub.py)
"""
STRIDE = 4
__version__ = "2026.09.22"
