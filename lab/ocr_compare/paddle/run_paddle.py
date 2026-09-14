"""Chạy PaddleOCR 3.7 (PP-OCRv6 / PP-OCRv5) trên bộ mẫu chung M1-M4.

venv riêng: /tmp/venv_paddle (python3.10, paddlepaddle 3.3.1, paddleocr 3.7.0, paddlex 3.7.2, CPU).
Dùng:
  /tmp/venv_paddle/bin/python run_paddle.py rec  --set M1 --model PP-OCRv6_medium_rec --prep raw|pad|up3|padup3
  /tmp/venv_paddle/bin/python run_paddle.py pipe --set M1 --version PP-OCRv6 --prep padup3      (det+rec)
  /tmp/venv_paddle/bin/python run_paddle.py m4   --version PP-OCRv6                               (cột dọc)
Kết quả: ket_qua/<set>_<cfg>.csv  (sample_id, pred, score, sec[, n_boxes])
"""
import os, sys, time, json, argparse
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import numpy as np
import pandas as pd
from PIL import Image

REPO = "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"
MAU = f"{REPO}/lab/ocr_compare/mau"
OUT = f"{REPO}/lab/ocr_compare/paddle/ket_qua"
os.makedirs(OUT, exist_ok=True)


def load_bgr(path):
    im = Image.open(path).convert("RGB")
    return np.array(im)[:, :, ::-1].copy()


def prep(img, mode):
    """img: BGR uint8. pad = viền trắng 25% mỗi phía; up3 = phóng 3x (bicubic)."""
    if "pad" in mode:
        h, w = img.shape[:2]
        ph, pw = int(0.25 * h), int(0.25 * w)
        canvas = np.full((h + 2 * ph, w + 2 * pw, 3), 255, np.uint8)
        canvas[ph:ph + h, pw:pw + w] = img
        img = canvas
    if "up3" in mode:
        img = np.array(Image.fromarray(img[:, :, ::-1]).resize((img.shape[1] * 3, img.shape[0] * 3), Image.BICUBIC))[:, :, ::-1].copy()
    if "up2" in mode:
        img = np.array(Image.fromarray(img[:, :, ::-1]).resize((img.shape[1] * 2, img.shape[0] * 2), Image.BICUBIC))[:, :, ::-1].copy()
    return img


def run_rec(args):
    from paddleocr import TextRecognition
    model = TextRecognition(model_name=args.model, device="cpu")
    m = pd.read_csv(f"{MAU}/{args.set}.csv", dtype=str, keep_default_na=False)
    rows = []
    t_all = time.time()
    for i, r in m.iterrows():
        img = prep(load_bgr(f"{REPO}/dataset_out/{r.image}"), args.prep)
        t = time.time()
        res = list(model.predict(img, batch_size=1))
        dt = time.time() - t
        txt = res[0]["rec_text"] if res else ""
        sc = float(res[0]["rec_score"]) if res else 0.0
        rows.append(dict(sample_id=r.sample_id, pred=txt.strip(), score=round(sc, 4), sec=round(dt, 4)))
        if i % 100 == 0:
            print(f"{args.set} {args.model} {args.prep} {i}/{len(m)} '{txt}' {sc:.2f} {dt*1000:.0f}ms", flush=True)
    cfg = f"{args.model}_{args.prep}"
    pd.DataFrame(rows).to_csv(f"{OUT}/{args.set}_{cfg}.csv", index=False)
    print("done", args.set, cfg, "wall", round(time.time() - t_all, 1), "s")


def make_pipe(version, textline_ori=False, lang="ch"):
    from paddleocr import PaddleOCR
    return PaddleOCR(lang=lang, ocr_version=version, device="cpu",
                     use_doc_orientation_classify=False, use_doc_unwarping=False,
                     use_textline_orientation=textline_ori)


def run_pipe(args):
    ocr = make_pipe(args.version)
    m = pd.read_csv(f"{MAU}/{args.set}.csv", dtype=str, keep_default_na=False)
    rows = []
    for i, r in m.iterrows():
        img = prep(load_bgr(f"{REPO}/dataset_out/{r.image}"), args.prep)
        t = time.time()
        res = list(ocr.predict(img))
        dt = time.time() - t
        texts = res[0]["rec_texts"] if res else []
        scores = res[0]["rec_scores"] if res else []
        rows.append(dict(sample_id=r.sample_id, pred="".join(texts).strip(), score=round(float(np.mean(scores)), 4) if len(scores) else 0.0,
                         sec=round(dt, 4), n_boxes=len(texts)))
        if i % 100 == 0:
            print(f"{args.set} pipe {args.version} {args.prep} {i}/{len(m)} {texts} {dt*1000:.0f}ms", flush=True)
    cfg = f"pipe-{args.version}_{args.prep}"
    pd.DataFrame(rows).to_csv(f"{OUT}/{args.set}_{cfg}.csv", index=False)
    print("done", args.set, cfg)


def run_m4(args):
    """Cột dọc nguyên vẹn: (a) det+rec nguyên cột; (b) det+rec + textline orientation;
    (c) rec-only cột xoay 90° CCW (như pipeline làm với hộp dọc) và 90° CW; (d) det+rec trên cột xoay 90° CCW."""
    from paddleocr import TextRecognition
    m = pd.read_csv(f"{MAU}/M4.csv", dtype=str, keep_default_na=False)
    rec_name = {"PP-OCRv6": "PP-OCRv6_medium_rec", "PP-OCRv5": "PP-OCRv5_server_rec"}[args.version]
    rec = TextRecognition(model_name=rec_name, device="cpu")
    pipe = make_pipe(args.version, textline_ori=False)
    pipe_ori = make_pipe(args.version, textline_ori=True)
    rows = []
    for i, r in m.iterrows():
        img = load_bgr(f"{MAU}/{r.image}")
        rec_out = dict(sample_id=r.sample_id, kinh_n=int(r.kinhhannom_n), kinh_text=r.kinhhannom_text)
        for name, fn in (("pipe", lambda: pipe.predict(img)), ("pipe_ori", lambda: pipe_ori.predict(img)),
                         ("pipe_rot90ccw", lambda: pipe.predict(np.ascontiguousarray(np.rot90(img)))),
                         ("pipe_up2", lambda: pipe.predict(prep(img, "up2")))):
            t = time.time(); res = list(fn()); dt = time.time() - t
            texts = res[0]["rec_texts"] if res else []
            polys = res[0]["rec_polys"] if res else []
            rec_out[f"{name}_text"] = "|".join(texts); rec_out[f"{name}_n_boxes"] = len(texts)
            rec_out[f"{name}_sec"] = round(dt, 3)
            rec_out[f"{name}_boxes"] = json.dumps([np.asarray(p).astype(int).tolist() for p in polys])
        for name, arr in (("rec_rot90ccw", np.rot90(img)), ("rec_rot90cw", np.rot90(img, -1))):
            t = time.time(); res = list(rec.predict(np.ascontiguousarray(arr), batch_size=1)); dt = time.time() - t
            rec_out[f"{name}_text"] = res[0]["rec_text"] if res else ""; rec_out[f"{name}_sec"] = round(dt, 3)
        rows.append(rec_out)
        print(i, r.sample_id, "kinh", r.kinhhannom_n, "| pipe", rec_out["pipe_n_boxes"], rec_out["pipe_text"][:40],
              "| ori", rec_out["pipe_ori_text"][:40], "| rot", rec_out["pipe_rot90ccw_text"][:40],
              "| recCCW", rec_out["rec_rot90ccw_text"][:30], "| recCW", rec_out["rec_rot90cw_text"][:30], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/M4_{args.version}.csv", index=False)
    print("done M4", args.version)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rec", "pipe", "m4"])
    ap.add_argument("--set", default="M1")
    ap.add_argument("--model", default="PP-OCRv6_medium_rec")
    ap.add_argument("--version", default="PP-OCRv6")
    ap.add_argument("--prep", default="raw")
    a = ap.parse_args()
    {"rec": run_rec, "pipe": run_pipe, "m4": run_m4}[a.mode](a)
