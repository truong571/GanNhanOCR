#!/usr/bin/env python
"""r02_candidates.py — chấm crop của MỌI ô với từng ứng viên x ∈ R(âm) ∪ {kim} (+ gt/ref chỉ để chẩn đoán).

Kênh điểm (cosine trên encoder v1+v2, embedding r01):
  s_font   : glyph font (NomNaTong → Plangothic P1/P2), 1 bản render (như m_ocr/h02)
  s_faug   : trung bình 5 bản render (gốc, đậm, mảnh, nhoè, thu nhỏ) — bắt chước nét khắc/in
  s_oth    : nguyên mẫu crop GOLD của các bộ KHÔNG phải IHR và KHÁC sách đang chấm
             (STT, LVT1883, KVK1884, Chresto; nhãn pipeline — chỉ làm khuôn tham chiếu, không làm truth)
             nguồn: gold_img_audit/m_ocr/out/E_crops.f16.npy + inputs.pkl (cùng tiền xử lý); cần ≥3 crop
  s_hum    : nguyên mẫu crop IHR có NHÃN NGƯỜI của sách IHR KIA (LOBO: chấm LVT1916 dùng mẫu TK1872 và ngược lại;
             chấm LVT1883/KVK dùng cả hai), chỉ ô slot_ok=1, cần ≥2 crop
  s_head   : cos với trọng số đầu ArcFace v1 (1.591 lớp)
view A (crop pipeline) và view B (cắt bbox): lưu cả hai cho s_font/s_faug.
Ra: out/cand.pkl (dạng dài: 1 hàng / (ô, ứng viên)), out/glyph_meta.pkl, out/r02_meta.json
"""
import hashlib, json, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SCR = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
sys.path.insert(0, str(SCR / "gold_img_audit" / "m_ocr"))
sys.path.insert(0, str(SCR / "kim_bottleneck" / "harness"))
import sv_lib as sv
OUT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/measure_out/_thu_nghiem_anh_chu/TN4/v1/rerank_new")
IHR = ("LucVanTien1916", "TruyenKieu1872")


def augs(im):
    k = np.ones((3, 3), np.uint8)
    out = [im, cv2.erode(im, k, 1), cv2.dilate(im, k, 1)]           # nền trắng: erode = nét ĐẬM, dilate = MẢNH
    b = cv2.GaussianBlur(im, (5, 5), 1.5)
    out.append(np.where(b < 170, 0, 255).astype(np.uint8))
    s = cv2.resize(im, (84, 84), interpolation=cv2.INTER_AREA)
    c = np.full_like(im, 255); c[14:98, 14:98] = s; out.append(c)
    return out


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "mps"
    T0 = time.time()
    d = pd.read_csv(SCR / "kim_bottleneck/harness/cells_eval.csv", dtype=str, keep_default_na=False)
    idx = pd.read_csv(OUT / "cells_idx.csv", dtype=str)
    assert (idx.cell_uid.values == d.cell_uid.values).all()
    EA = np.load(OUT / "E_A.f16.npy").astype(np.float32)
    EB = np.load(OUT / "E_B.f16.npy").astype(np.float32)
    N = len(d)
    # ---- ứng viên
    cands = []
    need = set()
    for i in range(N):
        r = d.iloc[i] if False else None
    Rl = [list(dict.fromkeys(x)) for x in d.R]      # R là chuỗi ký tự
    for i in range(N):
        c = set(Rl[i])
        k = d.ocr_char.iat[i]
        diag = set()
        if len(k) == 1: c.add(k)
        for x in (d.gt_char.iat[i], d.gt_img.iat[i], d.ref_char.iat[i], d.ref2_char.iat[i]):
            if len(x) == 1 and x not in c: diag.add(x)
        cands.append((sorted(c), sorted(diag)))
        need |= c | diag
    # ---- glyph font
    gl = sv.Glyphs()
    fch, fimg, ffont = [], [], []
    for ch in sorted(need):
        for k, (f, cmap) in enumerate(gl.fonts):
            if ord(ch) in cmap:
                fch.append(ch); fimg.append(gl.font(ch)); ffont.append(k); break
    enc = sv.MultiEnc(["v1", "v2"], dev, True)
    FE = enc.embed(fimg, norm=False)
    allaug = [a for im in fimg for a in augs(im)]
    FA = enc.embed(allaug, norm=False).reshape(len(fimg), 5, -1).mean(1)
    FA /= np.linalg.norm(FA, axis=1, keepdims=True)
    fidx = {c: j for j, c in enumerate(fch)}
    md5 = {c: hashlib.md5(im.tobytes()).hexdigest() for c, im in zip(fch, fimg)}
    blankmd5 = hashlib.md5(np.full((112, 112), 255, np.uint8).tobytes()).hexdigest()
    print(f"[glyph] need {len(need)} font {len(fch)} blank {sum(v==blankmd5 for v in md5.values())} {time.time()-T0:.0f}s", flush=True)
    # ---- nguyên mẫu 'oth' (GOLD pipeline các bộ không IHR), theo book_set
    g = pd.read_pickle(SCR / "gold_img_audit/m_ocr/out/inputs.pkl")
    EG = np.load(OUT / "E_crops_new.f16.npy").astype(np.float32)
    assert len(EG) == len(g)
    osum, on = {}, {}
    for bs in ("SachThanhTruyen", "LucVanTien1883", "KimVanKieu1884", "Chrestomathie1872"):
        m = (g.book_set == bs).values
        lab = g.label.values[m]; E = EG[m]
        df = pd.DataFrame({"l": lab}); grp = df.groupby("l").indices
        osum[bs] = {l: E[ix].sum(0) for l, ix in grp.items()}
        on[bs] = {l: len(ix) for l, ix in grp.items()}
    # ---- nguyên mẫu 'hum' (nhãn người IHR, ô slot_ok=1, view A)
    hsum, hn = {}, {}
    for b in IHR:
        m = ((d.book == b) & (d.slot_ok == "1") & (d.gt_char.str.len() == 1) & (d.gt_img == d.gt_char)).values
        lab = d.gt_char.values[m]; E = EA[m]
        grp = pd.DataFrame({"l": lab}).groupby("l").indices
        hsum[b] = {l: E[ix].sum(0) for l, ix in grp.items()}
        hn[b] = {l: len(ix) for l, ix in grp.items()}

    def proto(sets_sum, sets_n, sets, ch, minn):
        v, n = 0, 0
        for s in sets:
            if ch in sets_n[s]:
                v = v + sets_sum[s][ch]; n += sets_n[s][ch]
        if n < minn: return None, n
        return v / (np.linalg.norm(v) + 1e-9), n

    cache = {}
    def P(kind, book, ch):
        key = (kind, book, ch)
        if key not in cache:
            if kind == "oth":
                sets = [s for s in osum if s != book]
                cache[key] = proto(osum, on, sets, ch, 3)
            else:
                sets = [s for s in IHR if s != book]
                cache[key] = proto(hsum, hn, sets, ch, 2)
        return cache[key]

    W = enc.Wn; lab2idx = enc.lab2idx
    rows = []
    books = d.book.values
    for i in range(N):
        R = set(Rl[i]); k = d.ocr_char.iat[i]
        eA, eB = EA[i], EB[i]
        cs, diag = cands[i]
        for x, isdiag in [(x, 0) for x in cs] + [(x, 1) for x in diag]:
            j = fidx.get(x)
            po, npo = P("oth", books[i], x)
            ph, nph = P("hum", books[i], x)
            rows.append((i, x, isdiag, int(x in R), int(x == k),
                         float(eA @ FE[j]) if j is not None else np.nan,
                         float(eA @ FA[j]) if j is not None else np.nan,
                         float(eB @ FE[j]) if j is not None else np.nan,
                         float(eB @ FA[j]) if j is not None else np.nan,
                         float(eA @ po) if po is not None else np.nan, npo,
                         float(eA @ ph) if ph is not None else np.nan, nph,
                         float(eA @ W[lab2idx[x]]) if x in lab2idx else np.nan,
                         md5.get(x, "")))
        if i % 10000 == 0:
            print(f"  {i}/{N} {time.time()-T0:.0f}s", flush=True)
    cand = pd.DataFrame(rows, columns=["i", "ch", "diag", "inR", "is_kim", "s_font", "s_faug", "s_fontB", "s_faugB",
                                       "s_oth", "n_oth", "s_hum", "n_hum", "s_head", "gmd5"])
    cand["gmd5"] = cand.gmd5.replace(blankmd5, "BLANK")
    cand.to_pickle(OUT / "cand.pkl")
    meta = dict(n_cells=N, n_rows=len(cand), n_need=len(need), n_font=len(fch),
                font_by_file={str(k): ffont.count(k) for k in range(3)},
                frac_rows_font=float(cand.s_font.notna().mean()), frac_rows_oth=float(cand.s_oth.notna().mean()),
                frac_rows_hum=float(cand.s_hum.notna().mean()), frac_rows_head=float(cand.s_head.notna().mean()),
                t_total=round(time.time() - T0))
    json.dump(meta, open(OUT / "r02_meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
