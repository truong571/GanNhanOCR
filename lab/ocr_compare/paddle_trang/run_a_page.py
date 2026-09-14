"""Đường (a): PP-OCRv6 det+rec CẢ TRANG (PaddleOCR pipeline chuẩn, return_word_box=True).
Không dùng bất kỳ hộp nào của kinhhannom. Lưu đầu ra thô một JSON/trang vào raw_a/.
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True /tmp/venv_paddle/bin/python run_a_page.py [--version PP-OCRv6]
"""
import os, sys, json, time, argparse
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import numpy as np, pandas as pd
from PIL import Image
REPO = "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"
HERE = f"{REPO}/lab/ocr_compare/paddle_trang"
BOOK_DIR = {"stt11": "SachThanhTruyen11", "stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4"}

def page_set():
    pages = set()
    for s in ["M1", "M2", "M3"]:
        m = pd.read_csv(f"{REPO}/lab/ocr_compare/mau/{s}.csv", dtype=str, keep_default_na=False)
        pages |= set(zip(m.book, m.page))
    return sorted(pages)

def tolist(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (list, tuple)): return [tolist(v) for v in x]
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    return x

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="PP-OCRv6"); ap.add_argument("--out", default="raw_a")
    a = ap.parse_args()
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="ch", ocr_version=a.version, device="cpu", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    out = f"{HERE}/{a.out}"; os.makedirs(out, exist_ok=True)
    pages = page_set(); print("pages", len(pages), flush=True)
    for k, (b, p) in enumerate(pages):
        f = f"{out}/{b}_{p}.json"
        if os.path.exists(f): continue
        img = np.array(Image.open(f"{REPO}/prepared/{BOOK_DIR[b]}/pages/{p}.png").convert("RGB"))[:, :, ::-1].copy()
        t = time.time(); res = list(ocr.predict(img, return_word_box=True)); dt = time.time() - t
        r = res[0]
        d = dict(book=b, page=p, sec=dt, H=img.shape[0], W=img.shape[1],
                 rec_texts=list(r["rec_texts"]), rec_scores=tolist(r["rec_scores"]), rec_polys=tolist(r["rec_polys"]),
                 text_word=tolist(r.get("text_word")), text_word_boxes=tolist(r.get("text_word_boxes")))
        json.dump(d, open(f, "w"), ensure_ascii=False)
        if k % 20 == 0: print(k, b, p, "boxes", len(d["rec_texts"]), "sec", round(dt, 1), flush=True)

if __name__ == "__main__":
    main()
