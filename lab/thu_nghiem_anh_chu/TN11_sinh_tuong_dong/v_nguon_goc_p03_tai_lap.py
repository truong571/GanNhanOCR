"""TN11 v_nguon_goc_p03_tai_lap (03/10) — TÁI LẬP p03 từ ảnh sinh đã lưu (không sinh lại, CPU, 0 API) và kiểm hai nhánh nền.

Sao chép logic p03 dòng 66-95 (chọn ô, seed 42 tất định, top-3 theo f_vW, lọc chữ đúng ∈ top-3) và 155-256 (nhúng + chấm), đọc ảnh
FD sinh từ measure_out/_tn11/p03_direct/gen_book_style/b34_style/ (24 tệp) rồi so với ket_qua_p03.json:
  1. tập ô chọn có trùng 10 ô trong JSON không; cos_font / cos_fd / cos_gen từng ô có khớp tới 4 chữ số không (=> bookkeeping của p03 trung thực?)
  2. 'FD chung THẬT': dùng pipeline.gold_exact.signals_img.Glyphs.fdimg (đọc ArcFace/data/glyphs + gannhanocr-fd, bỏ con trỏ LFS < 1 KB) cho
     chữ nào có ảnh thật; đếm số ô mà CẢ 3 ứng viên đều có ảnh FD thật (mới chấm được baseline 2 đúng nghĩa) và Top-1 trên các ô đó.
  3. gen không dilate (t0) so với dilate 2 (t2) trên cùng 10 ô (p03 chỉ báo t2).
Ra: measure_out/_tn11/verify/nguon_goc/p03_tai_lap.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

M11 = REPO / "measure_out/_tn11"
OUT = M11 / "verify/nguon_goc"
EMB_DIR = T.OUT / "emb"
CAND_DIR = T.OUT / "cand"
FONT_PATH = str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def render_font_image(char: str, font_path: str, size: int = 128) -> np.ndarray:   # p03:45-58 nguyên văn
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, int(size * 0.75))
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - w) // 2 - bbox[0]
        y = (size - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
    except Exception:
        draw.rectangle([(10, 10), (size - 10, size - 10)], outline=128, width=2)
    return np.array(img)


def main():
    torch.set_num_threads(3)
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    book = "B34"
    D = T.load_base(book)
    E = nrm(np.load(EMB_DIR / f"{book}_enc.npy").astype(np.float32))
    F = pd.read_pickle(CAND_DIR / f"{book}.pkl")
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    rare_idx = tru.index[tru.n_self.fillna(0) < 2]
    rng = np.random.default_rng(42)
    selected_cells = np.sort(rng.choice(rare_idx, size=12, replace=False))
    S = F[F.i.isin(selected_cells)].copy()
    S["rk"] = S.f_vW.fillna(-9)
    S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(3)
    has_truth = S.groupby("i").y.max()
    valid_cells = has_truth[has_truth == 1].index
    S = S[S.i.isin(valid_cells)].copy()
    cells_list = sorted(set(S.i))
    chars = sorted(set(S.c))
    J = json.loads((M11 / "p03_direct/ket_qua_p03.json").read_text(encoding="utf-8"))
    jd = {r["cell_id"]: r for r in J["details"]}
    res = {"o_chon_giong_JSON": sorted(int(c) for c in cells_list) == sorted(jd), "n_o": len(cells_list), "n_chu": len(chars), "n_chon_12": int(len(selected_cells))}
    print(f"[tái lập p03] chọn {len(selected_cells)} ô, còn {len(cells_list)} sau lọc; trùng JSON: {res['o_chon_giong_JSON']}", flush=True)

    enc = SI.Scorers(Assets(), "cpu", EMB_DIR, False, lambda m: None).enc()
    gdir = M11 / "p03_direct/gen_book_style/b34_style"
    k3 = np.ones((3, 3), np.uint8)
    font_imgs, fd_imgs, gen2, gen0 = [], [], [], []
    for c in chars:
        f_im = render_font_image(c, FONT_PATH, 128)
        font_imgs.append(f_im)
        p = REPO / "gannhanocr-fd" / f"{ord(c):02X}"[:2] / f"U+{ord(c):04X}.png"
        fd_im = f_im if not p.exists() else cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if fd_im is None:
            fd_im = f_im
        fd_imgs.append(fd_im)
        gp = gdir / f"U+{ord(c):04X}.png"
        if gp.exists():
            g0 = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
            gen0.append(g0); gen2.append(cv2.dilate(g0, k3, iterations=2))
        else:
            gen0.append(fd_im); gen2.append(fd_im)
    Vf, Vd, V2, V0 = (dict(zip(chars, nrm(np.asarray(enc.embed(X, norm=False), np.float32)))) for X in (font_imgs, fd_imgs, gen2, gen0))
    # FD chung THẬT (bỏ con trỏ LFS): Glyphs.fdimg
    G = SI.Glyphs()
    real = {c: G.fdimg(c) for c in chars}
    have = [c for c in chars if real[c] is not None]
    Vr = dict(zip(have, nrm(np.asarray(enc.embed([real[c] for c in have], norm=False), np.float32)))) if have else {}

    rows, mism = [], []
    for cell in cells_list:
        sub = S[S.i == cell]
        gt = sub[sub.y == 1].iloc[0].c
        cv = E[cell]
        sc = {nm: {c: float(cv @ V[c]) for c in sub.c} for nm, V in (("font", Vf), ("fd", Vd), ("gen", V2), ("gen_t0", V0))}
        r = {"cell_id": int(cell), "gt": gt}
        for nm in sc:
            t = max(sc[nm], key=sc[nm].get)
            w = [c for c in sub.c if c != gt]
            r[f"acc_{nm}"] = int(t == gt); r[f"cos_{nm}"] = round(sc[nm][gt], 4); r[f"margin_{nm}"] = round(sc[nm][gt] - max(sc[nm][x] for x in w), 4)
        allreal = all(c in Vr for c in sub.c)
        r["ca_3_ung_vien_co_anh_FD_that"] = bool(allreal)
        if allreal:
            s_r = {c: float(cv @ Vr[c]) for c in sub.c}
            r["acc_fd_that"] = int(max(s_r, key=s_r.get) == gt); r["cos_fd_that"] = round(s_r[gt], 4)
        j = jd.get(int(cell))
        if j:
            for nm, jn in (("font", "font"), ("fd", "fd"), ("gen", "gen")):
                if abs(r[f"cos_{nm}"] - j[f"cos_{jn}"]) > 1.5e-4 or r[f"acc_{nm}"] != j[f"acc_{jn}"] or abs(r[f"margin_{nm}"] - j[f"margin_{jn}"]) > 1.5e-4:
                    mism.append((int(cell), nm, r[f"cos_{nm}"], j[f"cos_{jn}"]))
        rows.append(r)
    n = len(rows)
    A = lambda k: float(np.mean([r[k] for r in rows]) * 100)
    full = [r for r in rows if r["ca_3_ung_vien_co_anh_FD_that"]]
    res.update({"tai_lap_khop_JSON_4_chu_so": len(mism) == 0, "so_khong_khop": mism,
                "top1_tai_lap": {"font": round(A("acc_font"), 1), "fd(roi_ve_phong)": round(A("acc_fd"), 1), "gen_t2": round(A("acc_gen"), 1), "gen_t0_khong_dilate": round(A("acc_gen_t0"), 1)},
                "top1_JSON": J["metrics"]["top1_accuracy"],
                "mean_cos_tai_lap": {"font": round(float(np.mean([r["cos_font"] for r in rows])), 4), "gen_t2": round(float(np.mean([r["cos_gen"] for r in rows])), 4), "gen_t0": round(float(np.mean([r["cos_gen_t0"] for r in rows])), 4)},
                "mean_margin_tai_lap": {"font": round(float(np.mean([r["margin_font"] for r in rows])), 4), "gen_t2": round(float(np.mean([r["margin_gen"] for r in rows])), 4), "gen_t0": round(float(np.mean([r["margin_gen_t0"] for r in rows])), 4)},
                "FD_that": {"n_chu_co_anh_that": len(have), "n_chu": len(chars), "n_o_ca_3_ung_vien_co_anh_that": len(full),
                            "top1_FD_that_tren_cac_o_do": (round(float(np.mean([r["acc_fd_that"] for r in full]) * 100), 1) if full else None),
                            "top1_font_tren_cac_o_do": (round(float(np.mean([r["acc_font"] for r in full]) * 100), 1) if full else None),
                            "top1_gen_t2_tren_cac_o_do": (round(float(np.mean([r["acc_gen"] for r in full]) * 100), 1) if full else None)},
                "chi_tiet": rows})
    print(f"[tái lập p03] khớp JSON (cos/margin/acc, sai số 1,5e-4): {res['tai_lap_khop_JSON_4_chu_so']} {mism[:3]}; Top-1 tái lập {res['top1_tai_lap']} (JSON {J['metrics']['top1_accuracy']})")
    print(f"              cos TB {res['mean_cos_tai_lap']} | margin TB {res['mean_margin_tai_lap']}")
    print(f"              FD thật: {len(have)}/{len(chars)} chữ có ảnh; ô có cả 3 ứng viên có ảnh thật: {len(full)}/{n}; Top-1 trên các ô đó: FD thật {res['FD_that']['top1_FD_that_tren_cac_o_do']} font {res['FD_that']['top1_font_tren_cac_o_do']} gen {res['FD_that']['top1_gen_t2_tren_cac_o_do']}")
    (OUT / "p03_tai_lap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Đã ghi {OUT / 'p03_tai_lap.json'}")


if __name__ == "__main__":
    main()
