"""v_pb_truy_nguon_03 (04/10) — PHẢN BIỆN: dựng lại ĐÚNG hồ sơ mẫu giấy của p05 (trang thứ 11, Otsu) cho 10 bộ, ghi lại
những bộ mà ink_median = 60.0 (giá trị DỰ PHÒNG khi không có điểm ảnh mực), trang có phải hằng số / nhị phân không.
0 API, CPU, chỉ đọc ảnh bằng cv2 (không LLM). Ra: measure_out/_tn11/verify/pb_truy_nguon/ho_so_giay_p05.json
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"
res = {}
for b in T.ORDER:
    cfg = T.BOOKS[b]; pages_dir = T.REPO / cfg["prep"] / "pages"
    pgs = sorted(pages_dir.glob("*.png"))
    if not pgs: res[b] = dict(loi="khong co trang"); continue
    sp = pgs[min(10, len(pgs) - 1)]
    im = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
    Tv, _ = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = im >= Tv; pp = im[mask]; ip = im[~mask]
    uniq = np.unique(im)
    r = dict(trang=sp.name, kich_thuoc=list(im.shape), n_trang_png=len(pgs), otsu=float(Tv), n_gia_tri_khac_nhau=int(len(uniq)),
             min=int(im.min()), max=int(im.max()), n_mực=int(len(ip)), ty_le_muc=round(float(len(ip)) / im.size, 4),
             bg_median=float(np.median(pp)) if len(pp) else None, ink_median=(float(np.median(ip)) if len(ip) else "DU_PHONG_60.0"),
             bg_std=float(np.std(pp)) if len(pp) else None)
    # tỉ lệ pixel ở đúng 255 / 0
    r["ty_le_255"] = round(float((im == 255).mean()), 4); r["ty_le_0"] = round(float((im == 0).mean()), 4)
    # trang này có mực nào không (n_ink < Otsu) và patch tìm kiếm có tìm được min_ink=0 không
    h, w = im.shape; min_ink = 10**9
    for y in range(20, h - 148, 32):
        for x in range(20, w - 148, 32):
            n = int((im[y:y+128, x:x+128] < Tv).sum())
            if n < min_ink: min_ink = n
        if min_ink == 0: break
    r["min_ink_patch"] = None if min_ink == 10**9 else int(min_ink)
    r["dung_patch_tong_hop"] = bool(min_ink == 10**9 or min_ink > 200)
    res[b] = r
    print(b, {k: r[k] for k in ("trang","kich_thuoc","otsu","n_gia_tri_khac_nhau","min","max","ty_le_muc","bg_median","ink_median","bg_std","ty_le_255","min_ink_patch","dung_patch_tong_hop")}, flush=True)
(OUT / "ho_so_giay_p05.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
