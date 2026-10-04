"""r_goi_du_lieu_08_kiem_bang_PUA.py — HƯỚNG 3 (tiếp): chữ PUA mặt phẳng 15 của danh sách (306 cặp) có thật sự là CHỮ mà kim/từ điển muốn nói khi vẽ bằng NomNaTong không?

Gói vẽ chữ PUA bằng NomNaTong (tiền đề của khâu quy mô: "kim và từ điển R(âm) xuất mã PUA theo bảng của họ NomNaTong", chưa kiểm hình). Kiểm bằng hình, 0 nhãn người:
  với ô neo tự động sạch (hồ neo_khuyen_nghi, sách in L16/TK/KVK/L83/Chr) có nhãn = chữ kim là PUA: so bản đồ mật độ mực 24x24 của crop chuẩn 128 với glyph NomNaTong (ttf2im của gói) của CHÍNH
  nhãn đó và của 300 chữ PUA ngẫu nhiên khác; thống kê hạng. Đối chứng cùng phương pháp trên chữ không PUA (CJK) của cùng sách.
Nếu bảng PUA khớp: tỉ lệ hạng 1/hạng ≤ 10 của PUA gần đối chứng và ≫ ngẫu nhiên (1/301, 10/301). Nếu lệch bảng: PUA ≈ ngẫu nhiên.

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_08_kiem_bang_PUA.py
Ra: 08_kiem_bang_PUA.json.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import importlib.util
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402

import cv2  # noqa: E402


def dens(ink: np.ndarray, n: int = 24):
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return None
    c = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    d = cv2.resize(c, (n, n), interpolation=cv2.INTER_AREA).ravel()
    d = d - d.mean()
    nr = float(np.linalg.norm(d))
    return d / nr if nr > 0 else None


def crop_ink(path: Path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = bw < 128
    if ink.mean() > 0.5:
        ink = ~ink
    return ink


def main():
    t0 = time.time()
    inv = Lb.Inv()
    res: dict = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = Lb.PACK / "code" / "font_diffusion" / "src" / "tools" / "utils.py"
    spec = importlib.util.spec_from_file_location("fd_utils_goi", p)
    U = importlib.util.module_from_spec(spec); spec.loader.exec_module(U)
    fdir = Lb.PACK / "code" / "font_diffusion" / "fonts"
    font = U.load_ttf(str(fdir / "NomNaTong-Regular.ttf"))
    plan = json.loads((Lb.PACK / "plan.json").read_text(encoding="utf-8"))
    pool_pua = sorted({c for b in plan["books"].values() for c in b["chars"] if Lb.is_pua(c)})
    nn_chars = {c for b in plan["books"].values() for c, i in zip(b["chars"], b["font_idx"]) if plan["font_names"][i] == "NomNaTong"}
    pool_cjk = sorted(c for c in nn_chars if not Lb.is_pua(c) and 0x4E00 <= ord(c) <= 0x9FFF)
    gcache = {}

    def gl(c):
        if c not in gcache:
            buf = io.StringIO()
            with redirect_stdout(buf):
                im = U.ttf2im(font=font, char=c)
            gcache[c] = None if im is None else dens(np.asarray(im)[:, :, 0] < 128)
        return gcache[c]
    rng = np.random.default_rng(20261004)
    out = {}
    for kind, is_p in (("PUA", True), ("CJK_doi_chung", False)):
        rows = []
        for b in ("L16", "TK", "KVK", "L83", "Chr"):
            N = Lb.rd(Lb.HTD / "neo_khuyen_nghi" / f"{b}.csv", usecols=["label", "crop_chuan_cache_128", "cache_ton_tai", "page", "column", "nom_idx", "syl_idx"])
            N = N[(N.cache_ton_tai == "True") & (N.label.map(lambda c: Lb.is_pua(c) == is_p)) & (N.label.map(lambda c: c in nn_chars))]
            if not is_p:
                N = N[N.label.map(lambda c: 0x4E00 <= ord(c) <= 0x9FFF)]
            N = N.sample(n=min(220, len(N)), random_state=20261004)
            rows += [(b, r.label, r.crop_chuan_cache_128) for r in N.itertuples()]
        pool = pool_pua if is_p else pool_cjk
        ranks, per = [], {}
        for b, lab, path in rows:
            ink = crop_ink(Lb.REPO / path)
            dc = dens(ink) if ink is not None else None
            dg = gl(lab)
            if dc is None or dg is None:
                continue
            others = [c for c in rng.choice(pool, size=min(300, len(pool)), replace=False) if c != lab]
            sims = np.array([float(dc @ gl(c)) for c in others if gl(c) is not None])
            t = float(dc @ dg)
            r = int(1 + (sims >= t).sum())
            ranks.append(r); per.setdefault(b, []).append(r)
        ranks = np.array(ranks)
        n_other = 300
        out[kind] = dict(n=int(len(ranks)), top1_pct=round(100 * float((ranks == 1).mean()), 1), top5_pct=round(100 * float((ranks <= 5).mean()), 1), top10_pct=round(100 * float((ranks <= 10).mean()), 1),
                         hang_trung_vi=int(np.median(ranks)), ngau_nhien_top1_pct=round(100 / (n_other + 1), 2), ngau_nhien_top10_pct=round(1000 / (n_other + 1), 2),
                         theo_sach={b: dict(n=len(v), top10_pct=round(100 * float((np.array(v) <= 10).mean()), 1), hang_trung_vi=int(np.median(v))) for b, v in per.items()})
    res["ket_qua"] = out
    # bootstrap độ lệch giữa PUA và đối chứng (top10)
    res["ghi_chu"] = ("so khớp thô (mật độ mực 24x24) giữa crop in và glyph font: không phải nhận dạng chữ; chỉ dùng để phân biệt 'đúng bảng' (≫ ngẫu nhiên, gần đối chứng) với 'lệch bảng' (≈ ngẫu nhiên). "
                      "Chữ PUA là chữ Nôm hiếm nên nhiều ô neo có thể thuộc ít chữ khác nhau (xem n).")
    res["so_chu_PUA_khac_nhau_trong_mau"] = None
    inv.check("PUA_top10_>=_5x_ngau_nhien (bảng PUA không lệch hẳn)", out["PUA"]["top10_pct"] >= 5 * out["PUA"]["ngau_nhien_top10_pct"], out["PUA"])
    inv.check("PUA_top10_>=_70%_cua_doi_chung_CJK (PUA khớp hình cỡ như chữ thường)", out["PUA"]["top10_pct"] >= 0.7 * out["CJK_doi_chung"]["top10_pct"], (out["PUA"]["top10_pct"], out["CJK_doi_chung"]["top10_pct"]))
    res["bat_bien"] = inv.summary(); res["bat_bien_chi_tiet"] = inv.rows; res["giay"] = round(time.time() - t0, 1)
    Lb.jdump(res, "08_kiem_bang_PUA.json")
    print(f"[08] xong {res['giay']}s; bất biến {res['bat_bien']['dat']}/{res['bat_bien']['tong']} rớt={res['bat_bien']['rot']}")


if __name__ == "__main__":
    main()
