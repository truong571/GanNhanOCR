"""Dump Qwen-VL readings on the SAME reference frame as kim (frame-cropped image).

Key fix: kim OCRs crop_to_frame(page) (9-column region, no margin numbers/border).
Qwen must read the SAME image, else it sees the 1-9 column markers / page number /
border that kim never sees → misreads them as text and mis-aligns. This feeds Qwen
the identical frame-crop → one reference frame for all models.

Runs in the MAIN .venv (uses core.image.frame_detector + stdlib HTTP).
  .venv/bin/python evaluation/tri_consensus/dump_qwen.py --book SachThanhTruyen4 --n 10
Output: evaluation/tri_consensus/qwen_cache/<book>_<page>.json = {"text": ...}
"""
from __future__ import annotations
import argparse, base64, io, json, sys, urllib.request
from pathlib import Path

import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from core.image.frame_detector import crop_to_frame            # noqa: E402

QWEN_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen3-vl-flash"
PROMPT = (
    "Đây là một trang văn bản chữ Hán-Nôm khắc gỗ (ĐÃ cắt sát khung, chỉ còn 9 cột chữ), "
    "viết theo CỘT DỌC, đọc từ PHẢI sang TRÁI, mỗi cột từ TRÊN xuống DƯỚI. Hãy phiên chính "
    "xác TẤT CẢ các chữ Hán-Nôm trong ảnh theo đúng thứ tự đọc đó.\n"
    "QUY TẮC: MỖI CỘT MỘT DÒNG (cột phải nhất là dòng đầu). CHỈ xuất ký tự Hán-Nôm; TUYỆT ĐỐI "
    "không phiên âm Quốc Ngữ, không số, không dấu câu, không giải thích. Không bỏ sót/bịa thêm."
)


def _env(k):
    for line in (REPO / ".env").read_text().splitlines():
        s = line.strip()
        if s.startswith(k + "=") or s.startswith(k + " ="):
            return s.partition("=")[2].strip().strip("'").strip('"')
    return ""


def frame_pil(page_png, pad=12):
    bgr = cv2.imread(str(page_png))
    if bgr is None:
        return None
    crop = crop_to_frame(bgr, pad=pad)                          # SAME crop kim uses
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def call_qwen(pil_img, cache_path, key, model=QWEN_MODEL):
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["text"]
    im = pil_img
    if max(im.size) > 2048:
        r = 2048 / max(im.size); im = im.resize((int(im.width * r), int(im.height * r)))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 2048,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
    }).encode()
    req = urllib.request.Request(QWEN_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    cache_path.write_text(json.dumps({"text": text, "usage": data.get("usage", {})},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen4")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pad", type=int, default=12)
    ap.add_argument("--model", default="qwen3-vl-flash")
    ap.add_argument("--out", default="qwen_cache", help="cache subdir under this folder")
    args = ap.parse_args()
    key = _env("Qwen3-VL-Flash")
    assert key, "no Qwen3-VL-Flash key in .env"
    pg = REPO / "prepared" / args.book / "pages_denoised"
    det = REPO / "prepared" / args.book / "detected"
    out = HERE / args.out; out.mkdir(exist_ok=True)
    print(f"[model={args.model}]  out={out}")
    stems = [f.stem for f in sorted(pg.glob("page_*.png"))
             if (det / f"{f.stem}_ocr_cache.json").exists()][: args.n]
    for stem in stems:
        pil = frame_pil(pg / f"{stem}.png", args.pad)
        if pil is None:
            print(stem, "SKIP (read fail)"); continue
        text = call_qwen(pil, out / f"{args.book}_{stem}.json", key, args.model)
        cjk = sum(1 for c in text if 0x3400 <= ord(c) <= 0x9FFF or 0x20000 <= ord(c) <= 0x2A6DF)
        print(f"{stem}: frame-crop {pil.size} -> qwen {cjk} CJK chars")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
