"""PaddleOCR-VL-1.6-0.9B (backend native, CPU fp32) trên bộ mẫu chung. ~3 s/ô khi CPU rảnh (45-160 s/ô khi tranh chấp).
  /tmp/venv_paddle/bin/python run_vl.py M2 M3 M4 M1
Ghi dần: ket_qua/<set>_PaddleOCR-VL-1.6.csv. Prompt "OCR:" (prompt nhận dạng văn bản chuẩn của PaddleOCR-VL).
M4: ảnh cột nguyên vẹn (không xoay) — VL là VLM, tự xử lý bố cục."""
import os, sys, time, csv
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle")
from run_paddle import load_bgr, MAU, REPO, OUT
from paddlex import create_model
vl = create_model("PaddleOCR-VL-1.6-0.9B", device="cpu")
for s in sys.argv[1:]:
    m = pd.read_csv(f"{MAU}/{s}.csv", dtype=str, keep_default_na=False)
    path = f"{OUT}/{s}_PaddleOCR-VL-1.6.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "pred", "score", "sec"]); f.flush()
        for _, r in m.iterrows():
            img = load_bgr(f"{MAU}/{r.image}" if s == "M4" else f"{REPO}/dataset_out/{r.image}")
            t = time.time()
            try:
                res = list(vl.predict({"image": img, "query": "OCR:"}))
                txt = str(res[0].get("result", "")).replace("\n", " ").strip()
            except Exception as e:
                txt = f"ERR {e!r}"[:100]
            dt = time.time() - t
            w.writerow([r.sample_id, txt, "", round(dt, 1)]); f.flush()
            print(s, r.sample_id, "->", repr(txt)[:80], f"{dt:.0f}s", flush=True)
print("VL_DONE")
