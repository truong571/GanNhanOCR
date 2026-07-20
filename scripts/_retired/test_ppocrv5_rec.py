"""Full pipeline det+rec PP-OCRv5 trên 1 trang -> in text từng vùng."""
import os, sys, json
MODEL_DET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PP-OCRv5"))

img = sys.argv[1] if len(sys.argv) > 1 else "prepared/SachThanhTruyen11/pages/page_0010.png"
out_dir = "scratch_ppocrv5_out"
os.makedirs(out_dir, exist_ok=True)

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_detection_model_dir=MODEL_DET,
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

res = ocr.predict(img)
for r in res:
    texts = r["rec_texts"]; scores = r["rec_scores"]
    print(f"\n=== {len(texts)} vùng đọc được ===")
    for i, (t, s) in enumerate(zip(texts, scores)):
        print(f"[{i:2d}] conf={s:.2f}  {t}")
    r.save_to_img(out_dir)
    r.save_to_json(out_dir)
print("\nĐã lưu ảnh + json vào", out_dir)
