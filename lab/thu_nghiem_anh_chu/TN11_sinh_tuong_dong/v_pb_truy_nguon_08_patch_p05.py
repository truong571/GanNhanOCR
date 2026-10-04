"""v_pb_truy_nguon_08 (04/10) — PHẢN BIỆN: dựng lại 'tấm giấy' (paper_patch) mà p05 dùng cho HQC2 ở 10 bộ: thật (min_ink==0 / <=200) hay TỔNG HỢP
(np.clip(normal(bg, std)) khi min_ink>200), và ink_target có phải giá trị dự phòng 60.0. 0 API, CPU, chỉ đọc ảnh bằng cv2.
Ra: measure_out/_tn11/verify/pb_truy_nguon/patch_p05.json
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"
P05 = {r["book"]: r for r in json.loads((REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
res = {}
for b in T.ORDER:
    cfg = T.BOOKS[b]; pgs = sorted((T.REPO / cfg["prep"] / "pages").glob("*.png")); sp = pgs[min(10, len(pgs) - 1)]
    im = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE); Tv, _ = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = im >= Tv; pp = im[mask]; ip = im[~mask]
    bg = float(np.median(pp)) if len(pp) else 240.0; ink = float(np.median(ip)) if len(ip) else 60.0; sd = float(np.std(pp)) if len(pp) else 10.0
    h, w = im.shape; best = None; mi = 999999
    for y in range(20, h - 148, 32):
        for x in range(20, w - 148, 32):
            sub = im[y:y + 128, x:x + 128]; n = int(np.sum(sub < Tv))
            if n < mi: mi = n; best = sub.copy()
            if mi == 0: break
        if mi == 0: break
    synth = best is None or mi > 200
    if synth: best = np.clip(np.random.default_rng(42).normal(bg, max(sd, 4.0), (128, 128)), 0, 255).astype(np.uint8)
    # trang thực: trung vị / độ lệch chuẩn của vùng nền SẠCH (điểm >= Otsu) trong 8 trang khác (để so với canvas tổng hợp)
    r = dict(trang=sp.name, otsu=float(Tv), n_gia_tri=int(len(np.unique(im))), ink_median=ink, ink_la_du_phong_60=bool(len(ip) == 0), bg_median=bg,
             bg_std_cua_giay=round(sd, 2), min_ink_patch=int(mi) if mi != 999999 else None, patch_tong_hop=bool(synth),
             patch_mean=round(float(best.mean()), 1), patch_std=round(float(best.std()), 1), patch_min=int(best.min()), patch_max=int(best.max()),
             p05_paper_stats=P05[b]["paper_stats"], p05_top1=[P05[b]["hqc1_white_canvas"]["top1"], P05[b]["hqc2_book_paper_canvas"]["top1"], P05[b]["hqc3_canonical_normalized"]["top1"]],
             n_cells=P05[b]["n_cells"])
    res[b] = r; print(b, {k: r[k] for k in ("ink_median", "ink_la_du_phong_60", "bg_median", "bg_std_cua_giay", "min_ink_patch", "patch_tong_hop", "patch_mean", "patch_std", "p05_top1")}, flush=True)
(OUT / "patch_p05.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
