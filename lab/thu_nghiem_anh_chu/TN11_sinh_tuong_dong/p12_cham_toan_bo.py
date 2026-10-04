"""TN11 p12 (04/10) — CHẤM "ảnh sinh theo phong cách TỪNG cuốn" có giúp ích không, sau khi chạy gói Kaggle đủ 10 cuốn. 0 API, CPU.

Đầu vào: thư mục kết quả của bộ chạy (HF `hf download ...` hoặc unbundle): <ket_qua>/shards/<cuốn>/<mô hình>_s<j>/<NNNNN>.tar (mỗi tar chứa U+XXXX.png + _meta.json).
Hoặc, để thử: --gen-dir <dir> có sẵn <cuốn>/<mô hình>_s<j>/U+XXXX.png (đã giải nén).
Chấm trên 4 cuốn CÓ nhãn người (B18, B34, L16, TK; nhãn người CHỈ để chấm) bằng bảng ứng viên TN8 + nhúng crop sản xuất (measure_out/_tn8/{cand,emb}; cùng ứng viên với dữ liệu 01/10):
  extract  giải nén tar -> measure_out/_tn11/full/cham/gen/<cuốn>/<mô hình>_s<j>/U+XXXX.png (kiểm _meta, bỏ lô hỏng)
  single   Top-1 MỘT tín hiệu trên CÙNG tập ứng viên có ảnh sinh: f_gen vs f_font (phông render, sản xuất) — mọi ô và nhóm chữ hiếm; Δ + CI95 cụm trang; độ phủ ảnh sinh
  chooser  thêm f_gen vào bộ chọn chữ TN8 (LOBO, cùng t06_eval): (a) f_gen có NaN (mang cả "có ảnh" = tiên nghiệm do danh sách sinh chọn theo tần suất), (b) f_gen_imp (NaN → trung bình ô,
           chỉ còn tín hiệu ảnh). So với pred 'goc' của TN12 (hiệu chuẩn == TN8). Tiêu chí đăng ký trước: "có ích" nếu Δ Top-1 bộ chọn >= +0,30 điểm (CI cận dưới > 0) ở >= 2/4 bộ.
Với --pack <thư mục gói>: (a) loại khỏi chấm các ô có chữ đúng == chữ dùng làm ảnh phong cách của cuốn đó (ảnh sinh của chính chữ ấy được điều kiện bằng ảnh thật của nó ≈ sao chép; đọc từ <gói>_bao_cao.json),
(b) báo thêm riêng nhóm ứng viên vẽ bằng phông NomNaTong (bỏ ứng viên phông dự phòng; font_idx trong plan.json) vì ảnh nội dung phông dự phòng khác kiểu.
Ra: measure_out/_tn11/full/cham/ket_qua.json.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p12_cham_toan_bo.py --ket-qua measure_out/_tn11/full/kaggle_ket_qua --stage all
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

SRC = T.OUT
W = REPO / "measure_out" / "_tn11" / "full" / "cham"
HUMAN = {"B18": "B18", "B34": "B34", "L16": "L16", "TK": "TK"}
PAIRS = [("B34", "B18"), ("B18", "B34"), ("TK", "L16"), ("L16", "TK")]
GOC_PRED = REPO / "measure_out" / "_tn11" / "tn12_fd_day_du" / "pred_goc"
MODEL = "base_s0"


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def load_meta(pack):
    """{cuốn: dict(fallback=tập chữ vẽ bằng phông dự phòng, style=chữ ảnh phong cách 0)} từ gói (plan.json + _bao_cao.json); {} nếu không có --pack."""
    if not pack:
        return {}
    pk = Path(pack)
    plan = json.loads((pk / "plan.json").read_text(encoding="utf-8"))
    rep_p = Path(str(pk) + "_bao_cao.json")
    rep = json.loads(rep_p.read_text(encoding="utf-8")) if rep_p.exists() else {}
    meta = {}
    for b, v in plan["books"].items():
        fb = {c for c, k in zip(v["chars"], v["font_idx"]) if k != 0}
        st = None
        try:
            st = rep["sach"][b]["phong_cach"][0]["chu"]
        except (KeyError, IndexError):
            pass
        meta[b] = dict(fallback=fb, style=st)
    return meta


def style_cells(F, style):
    """tập ô có chữ đúng (V1+ thu gọn: hàng y == 1) == chữ ảnh phong cách."""
    if not style:
        return set()
    return set(F[(F.y == 1) & (F.c == style)].i.unique())


def rowdot(E, idx_e, M, idx_m, chunk=500000):
    out = np.full(len(idx_e), np.nan, np.float32)
    ok = np.nonzero(idx_m >= 0)[0]
    for s in range(0, len(ok), chunk):
        k = ok[s:s + chunk]
        out[k] = np.einsum("ij,ij->i", E[idx_e[k]], M[idx_m[k]])
    return out


def stage_extract(a):
    root = Path(a.ket_qua) / "shards"
    n_tar = n_img = n_bad = 0
    for tp in sorted(root.rglob("*.tar")):
        b, ms = tp.parent.parent.name, tp.parent.name
        try:
            with tarfile.open(tp) as tf:
                meta = json.loads(tf.extractfile("_meta.json").read())
                d = W / "gen" / b / ms
                d.mkdir(parents=True, exist_ok=True)
                for m in tf.getmembers():
                    if m.name.startswith("U+"):
                        (d / m.name).write_bytes(tf.extractfile(m).read()); n_img += 1
            n_tar += 1
        except Exception:  # noqa: BLE001
            n_bad += 1
    print(f"[p12] extract: {n_tar} tar, {n_img} ảnh, {n_bad} tar hỏng", flush=True)


def gen_dir_of(a, b):
    return Path(a.gen_dir) / b / MODEL if a.gen_dir else W / "gen" / b / MODEL


def glyph_vectors(a, b, enc):
    d = gen_dir_of(a, b)
    files = sorted(d.glob("U+*.png"))
    ims, chars = [], []
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            ims.append(im); chars.append(chr(int(f.name[2:-4], 16)))
    V = []
    for s in range(0, len(ims), 1024):
        V.append(np.asarray(enc.embed(ims[s:s + 1024], norm=False), np.float32))
    return dict(zip(chars, nrm(np.concatenate(V)))) if V else {}


def build_cand(a):
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    (W / "cand").mkdir(parents=True, exist_ok=True)
    (W / "emb_cache").mkdir(parents=True, exist_ok=True)
    Sc = SI.Scorers(Assets(), "cpu", W / "emb_cache", False, lambda m: None)
    enc = Sc.enc()
    info = {}
    for b in HUMAN:
        if not gen_dir_of(a, b).exists():
            print(f"[p12] {b}: chưa có ảnh sinh — bỏ qua", flush=True); continue
        G = glyph_vectors(a, b, enc)
        F = pd.read_pickle(SRC / "cand" / f"{b}.pkl")
        E = nrm(np.load(SRC / "emb" / f"{b}_enc.npy").astype(np.float32))
        chars = sorted(G)
        cid = {c: j for j, c in enumerate(chars)}
        V = np.stack([G[c] for c in chars]) if chars else np.zeros((0, 512), np.float32)
        ci = F.i.to_numpy(np.int64)
        jg = np.array([cid.get(c, -1) for c in F.c.to_numpy(object)], np.int64)
        F["f_gen"] = rowdot(E, ci, V, jg)
        m = F.groupby("i").f_gen.transform("mean")
        F["f_gen_imp"] = F.f_gen.fillna(m).fillna(0.0).astype(np.float32)
        F.to_pickle(W / "cand" / f"{b}.pkl")
        cp = W / "cand" / f"{b}_cells.pkl"
        if not cp.exists():
            os.symlink(SRC / "cand" / f"{b}_cells.pkl", cp)
        hy = F.groupby("i").y.max(); cells = hy[hy == 1].index
        Fc = F[F.i.isin(cells)]
        info[b] = dict(n_chu_co_anh=len(chars), phu_chu_dung=round(float(Fc[Fc.y == 1].f_gen.notna().mean()) * 100, 1),
                       phu_moi_ung_vien=round(float(Fc.f_gen.notna().mean()) * 100, 1))
        print(f"[p12] {b}: {len(chars)} chữ có ảnh sinh; phủ chữ đúng {info[b]['phu_chu_dung']} %, phủ mọi ứng viên {info[b]['phu_moi_ung_vien']} %", flush=True)
    for d in ("base", "emb"):
        if not (W / d).exists():
            os.symlink(SRC / d, W / d)
    return info


def top1_cell(F, col):
    x = F[["i", col, "y"]]
    return x.loc[x.groupby("i")[col].idxmax()].set_index("i").y.astype(float)


def single_one(F, b, pages, excl, drop_chars=None):
    """Top-1 phông vs sinh trên cùng tập ứng viên có ảnh; excl = ô loại; drop_chars = ứng viên loại (phông dự phòng)."""
    hy = F.groupby("i").y.max(); cells = hy[hy == 1].index
    M = F[F.i.isin(cells) & F.f_gen.notna() & F.f_font.notna() & ~F.i.isin(excl)]
    if drop_chars:
        M = M[~M.c.isin(drop_chars)]
    k = M.groupby("i").size(); ht = M.groupby("i").y.max()
    keep = k[(k >= 2) & (ht.reindex(k.index) == 1)].index
    M = M[M.i.isin(keep)]
    if len(keep) == 0:
        return None
    t_fo, t_ge = top1_cell(M, "f_font"), top1_cell(M, "f_gen")
    d = (t_ge - t_fo).to_numpy()
    _, lo, hi = T.boot_ci_pages(d, pages[t_ge.index.to_numpy()])
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    rare = np.isin(t_ge.index.to_numpy(), tru.index[tru.n_self.fillna(0) < 2].to_numpy())
    dr = d[rare]
    return dict(n=int(len(keep)), uv_o=round(float(k.loc[keep].mean()), 1), phong=round(float(t_fo.mean()) * 100, 1), sinh=round(float(t_ge.mean()) * 100, 1),
                delta=round(float(d.mean()) * 100, 2), ci=[round(lo * 100, 2), round(hi * 100, 2)], n_hiem=int(rare.sum()),
                delta_hiem=round(float(dr.mean()) * 100, 2) if len(dr) else None)


def stage_single(a, info, meta):
    out = {}
    print(f"{'bộ':4s} | {'nhóm':14s} | cùng tập ứng viên có ảnh sinh: n ô (ƯV/ô) | phông | sinh | Δ sinh−phông [CI95 cụm trang] | chữ hiếm Δ")
    for b in info:
        F = pd.read_pickle(W / "cand" / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font", "f_gen"]]
        pages = T.load_base(b).page.to_numpy()
        m = meta.get(b, {})
        excl = style_cells(F, m.get("style"))
        out[b] = dict(loai_o_chu_phong_cach=dict(chu=m.get("style"), so_o=len(excl)))
        for nhom, drop in (("tat_ca", None), ("chi_NomNaTong", m.get("fallback") or None)):
            if nhom == "chi_NomNaTong" and not m.get("fallback"):
                continue
            r = single_one(F, b, pages, excl, drop)
            out[b][nhom] = r
            if r:
                print(f"{b:4s} | {nhom:14s} | n={r['n']:6d} ({r['uv_o']} ƯV/ô) | {r['phong']:5.1f} | {r['sinh']:5.1f} | {r['delta']:+6.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['delta_hiem']}", flush=True)
    return out


def stage_chooser(info, meta):
    import tn8model as M
    import t06_eval as E
    base_sims = list(M.SIMS)
    res = {}
    for run, col in (("gen", "f_gen"), ("gen_imp", "f_gen_imp")):
        M.SIMS[:] = base_sims + [col]
        T.OUT = W
        E.PRED = W / f"pred_{run}"
        for tr, te in PAIRS:
            if te in info and tr in info:
                E.run_pair([tr], te, tag=f"{te}__from_{tr}")
        M.SIMS[:] = base_sims
    T.OUT = SRC
    n_ok = 0
    for tr, te in PAIRS:
        if te not in info or tr not in info:
            continue
        F = pd.read_pickle(SRC / "cand" / f"{te}.pkl")
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        rare = set(tru.index[tru.n_self.fillna(0) < 2])
        pages = T.load_base(te).page.to_numpy()
        base = pd.read_pickle(GOC_PRED / f"{te}__from_{tr}.pkl"); base = base[base.any_y == 1].set_index("i").y1.astype(float)
        ex = style_cells(pd.read_pickle(SRC / "cand" / f"{te}.pkl"), meta.get(te, {}).get("style"))
        base = base[~base.index.isin(ex)]
        r = {"goc": round(float(base.mean()) * 100, 2)}
        for run in ("gen", "gen_imp"):
            v = pd.read_pickle(W / f"pred_{run}" / f"{te}__from_{tr}.pkl"); v = v[v.any_y == 1].set_index("i").y1.astype(float).reindex(base.index)
            d = (v - base).to_numpy()
            _, lo, hi = T.boot_ci_pages(d, pages[base.index.to_numpy()])
            rr = base.index.isin(rare)
            r[run] = dict(top1=round(float(v.mean()) * 100, 2), delta=round(float(d.mean()) * 100, 2), ci=[round(lo * 100, 2), round(hi * 100, 2)],
                          delta_hiem=round(float(d[rr].mean()) * 100, 2))
            print(f"{te}<-{tr}: gốc {r['goc']} -> {run} {r[run]['top1']} (Δ {r[run]['delta']:+.2f} [{r[run]['ci'][0]:+.2f}, {r[run]['ci'][1]:+.2f}]; hiếm {r[run]['delta_hiem']:+.2f})", flush=True)
        res[f"{te}<-{tr}"] = r
    cap = {k: v for k, v in res.items() if "<-" in k}
    for run in ("gen", "gen_imp"):
        n_ok = sum(1 for v in cap.values() if v[run]["delta"] >= 0.30 and v[run]["ci"][0] > 0)
        res[f"co_ich_{run}"] = dict(so_bo_dat=n_ok, tren=len(cap), tieu_chi="Δ >= +0,30 điểm và CI cận dưới > 0 ở >= 2 bộ", dat=bool(n_ok >= 2))
    return res


def main(argv=None):
    global W
    ap = argparse.ArgumentParser()
    ap.add_argument("--ket-qua", help="thư mục kết quả bộ chạy (có shards/)")
    ap.add_argument("--gen-dir", help="(thử) thư mục đã giải nén: <cuốn>/base_s0/U+XXXX.png")
    ap.add_argument("--stage", choices=["extract", "single", "chooser", "all"], default="all")
    ap.add_argument("--pack", help="thư mục gói (plan.json + <gói>_bao_cao.json) để loại ô chữ phong cách và tách phông dự phòng")
    ap.add_argument("--work", default=str(W), help="thư mục làm việc/đầu ra (thử nghiệm giả dùng thư mục khác để không lẫn kết quả thật)")
    a = ap.parse_args(argv)
    W = Path(a.work)
    W.mkdir(parents=True, exist_ok=True)
    res = {}
    if a.stage in ("extract", "all") and a.ket_qua:
        stage_extract(a)
    if a.stage in ("single", "chooser", "all"):
        res["phu"] = build_cand(a)
        if not res["phu"]:
            raise SystemExit("không có ảnh sinh của cuốn nào có nhãn người — kiểm --ket-qua/--gen-dir")
    meta = load_meta(a.pack)
    if a.stage in ("single", "all"):
        res["single"] = stage_single(a, res["phu"], meta)
    if a.stage in ("chooser", "all"):
        res["chooser"] = stage_chooser(res["phu"], meta)
    prev = json.loads((W / "ket_qua.json").read_text()) if (W / "ket_qua.json").exists() else {}
    prev.update(res)
    (W / "ket_qua.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
