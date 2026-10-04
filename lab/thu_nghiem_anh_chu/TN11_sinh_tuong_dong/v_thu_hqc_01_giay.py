"""v_thu_hqc_01_giay.py — ước lượng MÔ HÌNH GIẤY của từng sách từ >= 10 trang (16 trang cách đều) của ẢNH GỐC (đúng nguồn crop sản xuất):
  * mực (lõi nét) = trung vị p3 của điểm xám trong hộp chữ; giấy = trung vị p90 trong hộp chữ (như build_dataset._paper_bg);
  * mảng giấy: 32 cửa sổ ngẫu nhiên 112×112 trong khối chữ mỗi trang, giữ 2 cửa sổ ít mực nhất (tỉ lệ mực+nở <= 0,70) (mặt nạ mực = xám < giấy-0,35·tương_phản,
    nở 5×5), điền điểm mực bằng điểm giấy sạch rút ngẫu nhiên trong chính cửa sổ -> bể mảng giấy THẬT;
  * K = 3 mảng (3 trang khác nhau, hạt giống cố định) cho mỗi sách (+ bộ ba thứ hai để đo độ nhạy theo mảng);
  * P_font / P_fd = cỡ khung glyph theo độ phân giải gốc của crop sách: hộp chữ của glyph ≈ L_med (trung vị cạnh dài hộp chữ crop).
Ra: measure_out/_tn11/verify/thu_hqc/giay/<bộ>.npz + .json ; đối chiếu số "mẫu giấy" của K09 (p05_full_corpus_he_quy_chieu.extract_book_paper_model).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

T = H.T
M_PAGES = 16
WIN = 112
N_CAND = 32
KEEP = 2
INK_FRAC_MAX = 0.70


def glyph_tight_long(imgs):
    L = []
    for g in imgs:
        ys, xs = np.nonzero(g < 128)
        if len(ys):
            L.append(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    return float(np.median(L)), len(L)


def k09_paper_stats(b):
    """Gọi đúng hàm của K09 + chạy lại vòng quét mảng để biết có rơi vào nhánh 'giấy tổng hợp' (min_ink > 200) không."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("p05fc", H.REPO / "lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p05_full_corpus_he_quy_chieu.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    pm = m.extract_book_paper_model(b)
    prep = T.REPO / T.BOOKS[b]["prep"]
    pages = sorted((prep / "pages").glob("*.png"))
    page = cv2.imread(str(pages[min(10, len(pages) - 1)]), cv2.IMREAD_GRAYSCALE)
    Tv, _ = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = page.shape; min_ink = 10 ** 9
    for y in range(20, h - 148, 32):
        for x in range(20, w - 148, 32):
            n = int(np.sum(page[y:y + 128, x:x + 128] < Tv)); min_ink = min(min_ink, n)
            if min_ink == 0:
                break
        if min_ink == 0:
            break
    return dict(page=pm["sample_page"], bg_median=pm["bg_median"], ink_median=pm["ink_median"], contrast=pm["contrast"],
                min_ink_patch=int(min_ink), fallback_synthetic_gauss=bool(min_ink > 200),
                patch_mean=float(pm["paper_patch"].mean()), patch_std=float(pm["paper_patch"].std()), n_pages_used=1)


def build(b, SI, gl):
    from pipeline.align_engine.build_dataset import load_original_page
    cfg = T.BOOKS[b]; prep = T.REPO / cfg["prep"]
    D = T.load_base(b)
    pages = sorted(D.page.unique())
    sel = [pages[int(round(x))] for x in np.linspace(0, len(pages) - 1, M_PAGES)]
    rng = np.random.default_rng(H.SEED + 11 + H.BI[b])
    ink_core, paper90, boxes_n = [], [], 0
    pool, pool_page, pool_stat = [], [], []
    for pg in sel:
        P = cv2.imread(str(prep / "pages" / f"{pg}.png"), cv2.IMREAD_COLOR)
        O = load_original_page(prep, pg, shape_like=P.shape[:2])
        g = cv2.cvtColor(O, cv2.COLOR_BGR2GRAY)
        Hh, Ww = g.shape
        bbs = [bb for bb in (T.bbox_of(x) for x in D[D.page == pg].bbox) if bb]
        bbs = np.array(bbs)
        x1, y1 = max(0, int(bbs[:, 0].min())), max(0, int(bbs[:, 1].min()))
        x2, y2 = min(Ww, int(bbs[:, 2].max())), min(Hh, int(bbs[:, 3].max()))
        pi, pp = [], []
        for bb in bbs[rng.permutation(len(bbs))[:80]]:
            a, c, e, f = [int(v) for v in bb]
            reg = g[max(0, c):max(0, f), max(0, a):max(0, e)]
            if reg.size < 64:
                continue
            pi.append(np.percentile(reg, 3)); pp.append(np.percentile(reg, 90))
        ink_core += pi; paper90 += pp; boxes_n += len(pi)
        I_page = float(np.median(pi)); P_page = float(np.median(pp))
        cands = []
        for _ in range(N_CAND):
            xs = int(rng.integers(x1, max(x1 + 1, x2 - WIN + 1))); ys = int(rng.integers(y1, max(y1 + 1, y2 - WIN + 1)))
            win = g[ys:ys + WIN, xs:xs + WIN]
            if win.shape != (WIN, WIN):
                continue
            ref = float(np.percentile(win, 90))
            thr = ref - 0.35 * max(ref - I_page, 20.0)
            mask = cv2.dilate((win < thr).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            cands.append((float(mask.mean()), win, mask))
        cands.sort(key=lambda t: t[0])
        for fr, win, mask in cands[:KEEP]:
            if fr > INK_FRAC_MAX:
                continue
            clean = win[~mask]
            if clean.size < 1200:
                continue
            w2 = win.copy()
            if mask.any():
                w2[mask] = rng.choice(clean, size=int(mask.sum()))
            pool.append(w2); pool_page.append(pg)
            pool_stat.append(dict(page=pg, ink_frac=fr, clean_mean=float(clean.mean()), clean_std=float(clean.std()),
                                  frac255=float((clean == 255).mean())))
    pool = np.stack(pool).astype(np.uint8)
    pool_page = np.array(pool_page)
    # ---- 2 bộ ba (3 trang khác nhau)
    up = sorted(set(pool_page))
    def triple(seed):
        r = np.random.default_rng(seed)
        pgs = r.choice(len(up), size=3, replace=len(up) < 3)
        ids = []
        for j in pgs:
            cand = np.nonzero(pool_page == up[j])[0]
            ids.append(int(r.choice(cand)))
        return ids
    ids_a, ids_b = triple(H.SEED + 21 + H.BI[b]), triple(H.SEED + 31 + H.BI[b])
    # ---- độ phân giải gốc của crop
    meta = np.load(T.OUT / "emb" / f"{b}_crop_meta.npy")
    ok = meta[:, 3] > 0
    L_med = float(np.median(np.maximum(meta[ok, 0], meta[ok, 1])))
    Lf, nf = glyph_tight_long([gl.font(c) for c in [chr(x) for x in range(0x4E00, 0x4E00 + 3000)] if gl.font(c) is not None])
    Ld, nd = glyph_tight_long([gl.fdimg(c) for c in gl.fd])
    P_font = int(round(L_med * 112.0 / Lf)); P_fd = int(round(L_med * 96.0 / Ld))
    st = dict(book=b, pages_used=sel, n_pages=len(sel), n_boxes_used=boxes_n,
              ink_level_p3_median=float(np.median(ink_core)), paper_level_p90_median=float(np.median(paper90)),
              contrast=float(np.median(paper90) - np.median(ink_core)),
              paper_noise_std_median=float(np.median([s["clean_std"] for s in pool_stat])),
              paper_clipped255_frac_median=float(np.median([s["frac255"] for s in pool_stat])),
              pool_n=int(len(pool)), pool_pages=len(up), pool_ink_frac_median=float(np.median([s["ink_frac"] for s in pool_stat])),
              ink=float(np.median(ink_core)), patch_ids_a=ids_a, patch_ids_b=ids_b,
              patch_pages_a=[str(pool_page[i]) for i in ids_a], patch_pages_b=[str(pool_page[i]) for i in ids_b],
              L_med_crop_long_side=L_med, glyph_tight_long_font=Lf, glyph_tight_long_fd=Ld, P_font=P_font, P_fd=P_fd,
              window=WIN, n_cand_windows_per_page=N_CAND, keep_per_page=KEEP, ink_frac_max=INK_FRAC_MAX,
              params=dict(ink_mask="gray < ref - 0.35*max(ref-I_page,20), ref = p90(window); dilate 5x5", fill="iid draw from clean pixels of the window",
                          ink_level="median over boxes of p3", paper_level="median over boxes of p90"))
    st["k09_paper_stats"] = k09_paper_stats(b)
    np.savez(H.OUT / "giay" / f"{b}.npz", patches=pool, pages=pool_page)
    H.jdump(st, H.OUT / "giay" / f"{b}.json")
    k = st["k09_paper_stats"]
    H.log(f"{b}: giấy p90 {st['paper_level_p90_median']:.0f}, mực p3 {st['ink_level_p3_median']:.0f}, tương phản {st['contrast']:.0f}, nhiễu giấy σ {st['paper_noise_std_median']:.1f}, "
          f"bão hoà 255 {st['paper_clipped255_frac_median']:.2f}; bể mảng {st['pool_n']} từ {st['pool_pages']} trang; L_med {L_med:.0f} -> P_font {P_font}, P_fd {P_fd} | "
          f"K09: nền {k['bg_median']}/mực {k['ink_median']}/tương phản {k['contrast']} (1 trang {k['page']}, min_ink {k['min_ink_patch']}, giấy tổng hợp: {k['fallback_synthetic_gauss']})")


def main():
    (H.OUT / "giay").mkdir(parents=True, exist_ok=True)
    from pipeline.gold_exact import signals_img as SI
    gl = SI.Glyphs()
    for b in H.BOOKS:
        build(b, SI, gl)


if __name__ == "__main__":
    main()
