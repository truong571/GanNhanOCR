"""Rec từng box detect được (mkldnn off để tránh segfault trên Mac ARM)."""
import os, cv2, json, glob, numpy as np

IMG = "prepared/SachThanhTruyen11/pages/page_0010.png"
OUT = "scratch_ppocrv5_out"
res = json.load(open(glob.glob(f"{OUT}/*_res.json")[0]))
img = cv2.imread(IMG)

from paddleocr import TextRecognition
rec = TextRecognition(model_name="PP-OCRv5_mobile_rec", enable_mkldnn=False)

rows = []
for i, poly in enumerate(res["dt_polys"]):
    p = np.array(poly).astype(int)
    x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    kind = "COT" if h > 500 else "nho"
    tmp = f"{OUT}/_crop_{i}.png"
    cv2.imwrite(tmp, crop)
    o = list(rec.predict(tmp))[0]
    rows.append((i, kind, h, o["rec_text"], o["rec_score"]))
    os.remove(tmp)

print(f"{'#':>2} {'loai':4} {'cao':>5} {'conf':>5}  text")
print("-" * 70)
for i, kind, h, t, s in rows:
    print(f"{i:>2} {kind:4} {h:>5} {s:>5.2f}  {t}")

# vẽ text lên ảnh (dùng font hệ thống qua PIL)
from PIL import Image, ImageDraw, ImageFont
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil)
try:
    font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 26)
except Exception:
    font = ImageFont.load_default()
for i, poly in enumerate(res["dt_polys"]):
    p = np.array(poly).astype(int)
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    draw.rectangle([x0, y0, p[:, 0].max(), p[:, 1].max()], outline=(255, 0, 0), width=2)
    draw.text((x0, y0 - 28), rows[i][3][:12], fill=(0, 0, 255), font=font)
pil.save(f"{OUT}/page_0010_rec_vis.png")
print("\nĐã lưu", f"{OUT}/page_0010_rec_vis.png")
