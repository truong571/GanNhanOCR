"""Chế độ CỘT cho M1-M3: đọc NGUYÊN CỘT chứa ô (cắt từ trang theo boxes_raw của cache kinhhannom, +8px,
xoay 90° CCW như pipeline PaddleOCR làm với hộp dọc), lấy vị trí CTC từng chữ (return_word_box) rồi
gióng về ô theo toạ độ y. Đây là cách dùng PaddleOCR như một S1 thay thế trên cùng hộp cột.

  /tmp/venv_paddle/bin/python run_paddle_col.py --set M2 --model PP-OCRv6_medium_rec
Kết quả: ket_qua/<set>_col-<model>.csv (sample_id, pred = chữ gióng vào ô theo y, pred_seq = gióng theo chuỗi (NW với
                                       chuỗi kinhhannom của cột — THIÊN VỊ về phía kinhhannom), pred_mono = gióng đơn điệu
                                       theo toạ độ (DP, không dùng chuỗi kinhhannom), pred_all = mọi chữ rơi vào ô,
                                       col_text, col_n, kinh_n, sec = giây/cột)
"""
import os, sys, json, time, argparse
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import numpy as np
import pandas as pd
from PIL import Image

REPO = "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"
MAU = f"{REPO}/lab/ocr_compare/mau"
OUT = f"{REPO}/lab/ocr_compare/paddle/ket_qua"
BOOK_DIR = {"stt11": "SachThanhTruyen11", "stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4"}
PAD = 8


def nw_align(a, b, match=2, mismatch=-1, gap=-1):
    """Needleman-Wunsch a (chuỗi paddle) ↔ b (chuỗi kinhhannom). Trả về map j(b) -> i(a) hoặc None."""
    n, m = len(a), len(b)
    S = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): S[i][0] = i * gap
    for j in range(1, m + 1): S[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i][j] = max(S[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch), S[i - 1][j] + gap, S[i][j - 1] + gap)
    i, j, mp = n, m, {}
    while i > 0 and j > 0:
        if S[i][j] == S[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            mp[j - 1] = i - 1; i -= 1; j -= 1
        elif S[i][j] == S[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    return mp


def read_column(rec, r, pad):
    cf = f"{REPO}/prepared/{BOOK_DIR[r.book]}/detected/{r.page}_ocr_cache.json"
    d = json.load(open(cf))
    ci = int(r.column) - 1
    if ci >= len(d["columns"]) or ci >= len(d["boxes_raw"]):  # cache lệch cột (vd page_0010_p0028: 8 cột/11 hộp)
        return dict(text="", chars=[], ys=[], sec=0.0, kinh_cells=[], kinh_n=0, seqmap={})
    pts = d["boxes_raw"][ci]["points"]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    page = Image.open(f"{REPO}/{d['image']}").convert("RGB")
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    crop = page.crop((cx0, cy0, min(page.width, x1 + pad), min(page.height, y1 + pad)))
    img = np.array(crop)[:, :, ::-1].copy()
    rot = np.ascontiguousarray(np.rot90(img))  # CCW: x_rot = y_col
    t = time.time()
    res = list(rec.predict(rot, batch_size=1))
    dt = time.time() - t
    rt = res[0]["rec_text"]
    if isinstance(rt, tuple):
        text = rt[0]; T = rt[1][0]; chars = rt[1][1][0] if rt[1][1] else []; pos = rt[1][2][0] if rt[1][2] else []
    else:
        text, T, chars, pos = rt, 1, [], []
    Hc = img.shape[0]
    ys_page = [cy0 + (p + 0.5) / T * Hc for p in pos]
    return dict(text=text, chars=list(chars), ys=ys_page, sec=dt, kinh_cells=d["columns"][ci], kinh_n=len(d["columns"][ci]))


def mono_align(ys, centers, cell_h, gap=0.75):
    """Gióng đơn điệu chữ đọc ra (toạ độ y) ↔ ô (tâm y), chỉ dùng toạ độ, KHÔNG dùng chuỗi kinhhannom.
    Chi phí khớp = |y - c| / cell_h; bỏ qua chữ hoặc ô = gap. Trả về map j(ô) -> i(chữ)."""
    n, m = len(ys), len(centers)
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): S[i][0] = i * gap
    for j in range(1, m + 1): S[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i][j] = min(S[i - 1][j - 1] + abs(ys[i - 1] - centers[j - 1]) / cell_h, S[i - 1][j] + gap, S[i][j - 1] + gap)
    i, j, mp = n, m, {}
    while i > 0 and j > 0:
        if abs(S[i][j] - (S[i - 1][j - 1] + abs(ys[i - 1] - centers[j - 1]) / cell_h)) < 1e-9:
            mp[j - 1] = i - 1; i -= 1; j -= 1
        elif abs(S[i][j] - (S[i - 1][j] + gap)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="M2")
    ap.add_argument("--model", default="PP-OCRv6_medium_rec")
    a = ap.parse_args()
    from paddlex import create_model
    rec = create_model(a.model, device="cpu", return_word_box=True)
    m = pd.read_csv(f"{MAU}/{a.set}.csv", dtype=str, keep_default_na=False)
    cache_cols = {}
    rows = []
    for i, r in m.iterrows():
        key = (r.book, r.page, int(r.column))
        if key not in cache_cols:
            cache_cols[key] = read_column(rec, r, PAD)
        c = cache_cols[key]
        if "seqmap" not in c:
            c["seqmap"] = nw_align(c["chars"], [k["char"] for k in c["kinh_cells"]])
            cents = [(k["bbox"][1] + k["bbox"][3]) / 2 for k in c["kinh_cells"]]
            ch = (c["kinh_cells"][0]["bbox"][3] - c["kinh_cells"][0]["bbox"][1]) if c["kinh_cells"] else 1
            c["monomap"] = mono_align(c["ys"], cents, max(ch, 1))
        # ô = hộp chữ trong cache gần y_center của bbox labels nhất
        bb = json.loads(r.bbox); yc = (bb[1] + bb[3]) / 2
        cell = min(c["kinh_cells"], key=lambda k: abs((k["bbox"][1] + k["bbox"][3]) / 2 - yc)) if c["kinh_cells"] else None  # y_center trong cache là toạ độ trong cột; bbox là toạ độ trang
        if cell is None:
            rows.append(dict(sample_id=r.sample_id, pred="", pred_seq="", pred_mono="", pred_all="", col_text=c["text"], col_n=len(c["chars"]),
                             kinh_n=c["kinh_n"], sec=round(c["sec"], 4), cell_char="", note="no_cell"))
            continue
        kidx = c["kinh_cells"].index(cell)
        pred_seq = c["chars"][c["seqmap"][kidx]] if kidx in c["seqmap"] else ""
        pred_mono = c["chars"][c["monomap"][kidx]] if kidx in c["monomap"] else ""
        cy0_, cy1_ = cell["bbox"][1], cell["bbox"][3]; ccy = (cy0_ + cy1_) / 2
        inside = [(abs(y - ccy), ch) for ch, y in zip(c["chars"], c["ys"]) if cy0_ <= y <= cy1_]
        inside.sort()
        pred = inside[0][1] if inside else ""
        if not inside and c["chars"]:
            # không chữ nào rơi vào ô: lấy chữ gần tâm ô nhất nếu cách < nửa chiều cao ô
            j = int(np.argmin([abs(y - ccy) for y in c["ys"]]))
            if abs(c["ys"][j] - ccy) < (cy1_ - cy0_) / 2:
                pred = c["chars"][j]
        rows.append(dict(sample_id=r.sample_id, pred=pred, pred_seq=pred_seq, pred_mono=pred_mono, pred_all="".join(ch for _, ch in inside), col_text=c["text"],
                         col_n=len(c["chars"]), kinh_n=c["kinh_n"], sec=round(c["sec"], 4), cell_char=cell["char"],
                         note="ok" if cell["char"] == r.ocr_char else "cell_char_mismatch"))
        if i % 100 == 0:
            print(a.set, i, r.sample_id, r.ocr_char, r.syllable, "->", repr(pred), "| col", c["text"][:30], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{a.set}_col-{a.model}.csv", index=False)
    print("done", a.set, "cols", len(cache_cols), "mismatch", (df.note != "ok").sum(),
          "sec/col", round(np.mean([c["sec"] for c in cache_cols.values()]), 3))


if __name__ == "__main__":
    main()
