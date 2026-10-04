"""v_pb_truy_nguon_07 (04/10) — PHẢN BIỆN: kiểm lại kết luận 'đồng bộ hệ quy chiếu (HQC2) xấp xỉ 0' của người kiểm chứng bằng phép đo
n lớn hơn p05 (~16-20 ô) trên 4 bộ có nhãn người, ĐÚNG hàm của p05 (render_font_raw / extract_book_paper_model / blend_on_paper sao nguyên văn),
CPU (torch device cpu, 3 luồng, MPS tắt), 0 API. Chỉ HQC1 (nền trắng) và HQC2 (mẫu giấy sách) — HQC3 bỏ vì p05 cắt crop sai carve.
Giao thức top-3 theo f_vW như p05 (giữ ô chữ đúng trong top-3); chọn ngẫu nhiên N ô mục tiêu bằng seed CỐ ĐỊNH.
Ra: measure_out/_tn11/verify/pb_truy_nguon/hqc_nho_<bộ>.json
    VT_N=400 .venv/bin/python v_pb_truy_nguon_07_hqc_nho.py L16 TK B34 B18
"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import cv2, numpy as np, pandas as pd, torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
torch.set_num_threads(int(os.environ.get("VT_THREADS", "3")))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"; OUT.mkdir(parents=True, exist_ok=True)
EMB = T.OUT / "emb"; CAND = T.OUT / "cand"
FONT_P05 = str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")
N = int(os.environ.get("VT_N", "400")); SEED = 20261004; BATCH = 128

def nrm(M): return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)

def render_font_raw(char, size=128):  # sao nguyên văn p05:58-70
    img = Image.new("L", (size, size), 255); draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_P05, int(size * 0.75)); bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]; x = (size - w) // 2 - bbox[0]; y = (size - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
    except Exception:
        draw.rectangle([(10, 10), (size - 10, size - 10)], outline=128, width=2)
    return np.array(img)

def extract_book_paper_model(book):  # sao nguyên văn p05:73-124
    cfg = T.BOOKS[book]; prep = T.REPO / cfg["prep"]; sample_pages = sorted((prep / "pages").glob("*.png"))
    sample_page = sample_pages[min(10, len(sample_pages) - 1)]; page_img = cv2.imread(str(sample_page), cv2.IMREAD_GRAYSCALE)
    T_val, _ = cv2.threshold(page_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper_mask = page_img >= T_val; paper_pixels = page_img[paper_mask]; ink_pixels = page_img[~paper_mask]
    bg_median = float(np.median(paper_pixels)) if len(paper_pixels) else 240.0
    ink_median = float(np.median(ink_pixels)) if len(ink_pixels) else 60.0
    bg_std = float(np.std(paper_pixels)) if len(paper_pixels) else 10.0
    h, w = page_img.shape; best_patch = None; min_ink = 999999
    for y in range(20, h - 148, 32):
        for x in range(20, w - 148, 32):
            sub = page_img[y:y + 128, x:x + 128]; n_ink = np.sum(sub < T_val)
            if n_ink < min_ink: min_ink = n_ink; best_patch = sub.copy()
            if min_ink == 0: break
        if min_ink == 0: break
    synth = False
    if best_patch is None or min_ink > 200:
        rng = np.random.default_rng(42); synth = True
        best_patch = np.clip(rng.normal(bg_median, max(bg_std, 4.0), (128, 128)), 0, 255).astype(np.uint8)
    return {"bg_median": round(bg_median, 1), "ink_median": round(ink_median, 1), "bg_std": round(bg_std, 1), "min_ink_patch": int(min_ink), "patch_tong_hop": synth, "paper_patch": best_patch}

def blend_on_paper(raw_glyph, pm):  # sao nguyên văn p05:127-132
    patch = pm["paper_patch"].astype(np.float32); alpha = (255.0 - raw_glyph.astype(np.float32)) / 255.0
    return np.clip((1.0 - alpha) * patch + alpha * pm["ink_median"], 0, 255).astype(np.uint8)

def embed_many(enc, imgs):
    out = [np.asarray(enc.embed(imgs[s:s + BATCH], norm=False), np.float32) for s in range(0, len(imgs), BATCH)]
    return nrm(np.concatenate(out)) if out else np.zeros((0, 512), np.float32)

def boot_diff(a, b, pages, B=2000, seed=20260930):
    d = np.asarray(a, float) - np.asarray(b, float); up, inv = np.unique(pages, return_inverse=True)
    k = np.bincount(inv, weights=d, minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(up), size=(B, len(up))); r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return [round(100 * float(d.mean()), 2), round(100 * float(np.quantile(r, 0.025)), 2), round(100 * float(np.quantile(r, 0.975)), 2)]

def run(b, enc, cmap):
    t0 = time.time(); D = T.load_base(b); E = nrm(np.load(EMB / f"{b}_enc.npy").astype(np.float32))
    F = pd.read_pickle(CAND / f"{b}.pkl")[["i", "c", "y", "f_vW"]]
    hy = F.groupby("i").y.max(); target = hy[hy == 1].index.values
    rng = np.random.default_rng(SEED); cells = np.sort(rng.choice(target, size=min(N, len(target)), replace=False))
    S = F[F.i.isin(cells)].copy(); S["rk"] = S.f_vW.fillna(-9)
    S = S.sort_values(["i", "rk"], ascending=[True, False], kind="stable").groupby("i").head(3)
    keep = S.groupby("i").y.max(); valid = np.array(sorted(keep[keep == 1].index)); S = S[S.i.isin(valid)].copy()
    pm = extract_book_paper_model(b); chars = sorted(set(S.c))
    present = [c for c in chars if len(c) == 1 and ord(c) in cmap]; blank = np.full((128, 128), 255, np.uint8)
    raw = {c: render_font_raw(c) for c in present}
    V = {}
    for nm, fn in (("1", lambda x: x), ("2", lambda x: blend_on_paper(x, pm))):
        ims = [fn(raw[c]) for c in present] + [fn(blank)]; em = embed_many(enc, ims)
        V[nm] = (dict(zip(present, em[:-1])), em[-1])
    for nm in ("1", "2"):
        d, bl = V[nm]; S["H" + nm] = [float(E[i] @ d.get(c, bl)) for i, c in zip(S.i.values, S.c.values)]
    def top1(col):
        x = S[["i", col, "y"]]; return x.loc[x.groupby("i", sort=False)[col].idxmax()].set_index("i").y.reindex(valid).values.astype(float)
    pg = D.page.values[valid]; t1, t2 = top1("H1"), top1("H2")
    res = dict(bo=b, N_mau=int(len(cells)), n_giu_lai=int(len(valid)), paper={k: v for k, v in pm.items() if k != "paper_patch"},
               HQC1=round(100 * float(t1.mean()), 2), HQC2=round(100 * float(t2.mean()), 2), HQC2_tru_HQC1_diem_ci=boot_diff(t2, t1, pg),
               so_o_HQC2_dung_HQC1_sai=int(((t2 == 1) & (t1 == 0)).sum()), so_o_HQC1_dung_HQC2_sai=int(((t1 == 1) & (t2 == 0)).sum()),
               thieu_glyph_p05_pct=round(100 * float((~S.c.isin(present)).mean()), 2), giay=round(time.time() - t0, 1))
    (OUT / f"hqc_nho_{b}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False), flush=True)

def main():
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    S_ = SI.Scorers(Assets(), "cpu", EMB, False, lambda m: None); enc = S_.enc(); cmap = set(TTFont(FONT_P05).getBestCmap())
    for b in sys.argv[1:] or ["L16", "TK", "B34", "B18"]:
        run(b, enc, cmap)
if __name__ == "__main__": main()
