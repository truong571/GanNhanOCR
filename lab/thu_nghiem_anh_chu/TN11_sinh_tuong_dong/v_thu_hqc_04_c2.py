"""v_thu_hqc_04_c2.py — C2 = HQC3: nhị phân hoá Otsu CẢ crop LẪN glyph (lọc đốm < 6 điểm, hộp bao mực, thu 80 % khung = lề 10 %, căn giữa 128×128),
trên mẫu con <= 3000 ô/bộ (hạt giống cố định, mau_<bộ>.npz). Crop cắt lại đúng hình học sản xuất (t02_embed.page_crops, đã kiểm == E sản xuất ở bước 0),
KHÔNG ép crop thành vuông trước khi nhị phân (khác p05_full_corpus_he_quy_chieu.py:236-238).
Đối chứng hình học (chạy với --geom): giữ XÁM, chỉ chuẩn hoá hộp bao mực + lề 10 % cho cả hai phía (tách tác dụng hình học khỏi nhị phân).
Ra: emb/c2_crops_<bộ>.npy, emb/c2_canon_font.npz, emb/c2_canon_fd.npz (hoặc c2g_* cho --geom)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402
import numpy as np  # noqa: E402

OUT = H.OUT


def get_crops(b, Bk, S):
    """crop cắt lại của mẫu (cache pickle: dùng chung cho C2 và đối chứng hình học)."""
    import pickle
    f = OUT / "emb" / f"cropimgs_{b}.pkl"
    have = pickle.load(open(f, "rb")) if f.exists() else {}
    need = np.array([int(i) for i in S if int(i) not in have])
    if len(need):
        have.update(H.recut_crops(b, Bk["D"], need))
        pickle.dump(have, open(f, "wb"))
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--books", nargs="*", default=H.BOOKS)
    ap.add_argument("--recut-only", action="store_true")
    a = ap.parse_args()
    pre = "c2g" if a.geom else "c2"
    kc = "crops_geom" if a.geom else "crops_canon"
    kg = "geom" if a.geom else "canon"
    pool = None if a.recut_only else H.make_pool()
    allfont = {}
    fdchars = set()
    for b in a.books:
        Bk = H.load_eval_book(b, with_E=False)
        F = Bk["F"]
        S = np.load(OUT / f"mau_{b}.npz")["s3000"][: a.n]
        crops = get_crops(b, Bk, S)
        H.log(f"{b}: {sum(1 for i in S if int(i) in crops)}/{len(S)} crop đã cắt lại")
        if a.recut_only:
            continue
        f = OUT / "emb" / f"{pre}_crops_{b}.npy"
        ids = np.array([int(i) for i in S if int(i) in crops])
        if not f.exists() or len(np.load(OUT / "emb" / f"{pre}_crops_{b}_ids.npy")) != len(ids):
            imgs = [crops[int(i)] for i in ids]
            H.log(f"{b}: nhúng {len(ids)} crop ({kc})")
            E = H.embed_images(pool, kc, imgs)
            np.save(f, E); np.save(OUT / "emb" / f"{pre}_crops_{b}_ids.npy", ids)
        m = F.i.isin(set(int(x) for x in S)).values
        for c in F.c.astype(str)[m & F.f_font.notna().values].unique():
            allfont[c] = 1
        fdchars |= set(F.c.astype(str)[m & F.f_fd.notna().values])
    if a.recut_only:
        return
    fonts = sorted(allfont)
    H.embed_chars(pool, kg, "font", fonts, OUT / "emb" / f"{pre}_canon_font.npz", tag=f"{pre}.font")
    H.embed_chars(pool, kg, "fd", sorted(fdchars), OUT / "emb" / f"{pre}_canon_fd.npz", tag=f"{pre}.fd")
    pool.close(); pool.join()
    H.log("XONG")


if __name__ == "__main__":
    main()
