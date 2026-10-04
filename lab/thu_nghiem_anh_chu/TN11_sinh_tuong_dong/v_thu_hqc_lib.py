"""v_thu_hqc_lib.py (03/10) — thư viện chung cho phép KIỂM ĐÚNG QUY MÔ "HQC" (đồng bộ hệ quy chiếu quang học glyph <-> crop).

Mọi suy luận mô hình chạy trên CPU (không dùng MPS: GPU đang bận huấn luyện FontDiffuser), 0 API, không sửa mã/tệp có sẵn.
Đầu ra: measure_out/_tn11/verify/thu_hqc/. Quy ước giao thức đo == p01 (TN11) / t05_feats (TN8):
  * ô tính = ô có chữ đúng trong ứng viên (any y==1); Top-1 = argmax điểm của MỘT tín hiệu trên TOÀN BỘ ứng viên của ô,
    NaN -> -9 (idxmax lấy hàng đầu khi hoà) — y đúng như p01.top1; nhóm "chữ hiếm" = n_self(chữ đúng) < 2.
  * CI 95 % = bootstrap CỤM TRANG (T.boot_ci_pages: B=2000, seed 20260930); chênh lệch cặp = bootstrap cặp cùng ma trận cụm.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "thu_hqc"
BOOKS = ["B18", "B34", "L16", "TK"]
SEED = 20261003                       # hạt giống gốc (mọi mẫu con / mảng giấy / tuning suy ra từ đây, ghi vào ket_qua.json)
BI = {b: i for i, b in enumerate(BOOKS)}
NWORK = int(os.environ.get("HQC_WORKERS", "3"))
BOOT_B = T.BOOT_B                      # 2000
BOOT_SEED = T.BOOT_SEED                # 20260930


def jdump(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================================== dữ liệu
def load_eval_book(b: str, with_E=True, cols=("i", "c", "y", "n_self", "f_font", "f_fd")):
    """Bảng ứng viên của các ô CÓ sự thật (any y==1) + nhúng crop sản xuất + trang. Giữ đúng thứ tự hàng của cand/<b>.pkl."""
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[list(cols)]
    hy = F.groupby("i").y.max()
    cells = hy[hy == 1].index.values
    F = F[F.i.isin(cells)].reset_index(drop=True)
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    rare = (tru.n_self.fillna(0) < 2)                       # Series (index = ô) — y như p01
    F["c"] = F["c"].astype("category")
    D = T.load_base(b)
    out = dict(F=F, cells=cells, rare=rare, page=D.page.values, D=D)
    if with_E:
        out["E"] = np.load(T.OUT / "emb" / f"{b}_enc.npy").astype(np.float32)
    return out


def rowdot(E, ei, G, gi, chunk=200000):
    """s[k] = E[ei[k]] · G[gi[k]] (gi < 0 -> nan); float32 như t05_feats.rowdot."""
    out = np.full(len(ei), np.nan, np.float32)
    ok = np.nonzero(gi >= 0)[0]
    for s in range(0, len(ok), chunk):
        k = ok[s:s + chunk]
        out[k] = np.einsum("ij,ij->i", E[ei[k]], G[gi[k]])
    return out


def score_rows(Bk, emb: dict, E=None):
    """emb: dict ký tự -> vec 512 (NaN nếu không có glyph). Trả mảng điểm theo hàng Bk['F'] (cos với crop)."""
    F = Bk["F"]
    E = Bk["E"] if E is None else E
    cats = list(F.c.cat.categories)
    G = np.zeros((len(cats), 512), np.float32)
    has = np.zeros(len(cats), bool)
    for j, ch in enumerate(cats):
        v = emb.get(ch)
        if v is not None and not np.isnan(v[0]):
            G[j] = v; has[j] = True
    codes = F.c.cat.codes.values.astype(np.int64)
    gi = np.where(has[codes], codes, -1)
    return rowdot(E, F.i.values.astype(np.int64), G, gi)


# ============================================================================================== chỉ số
def cell_metrics(i, s, y):
    """Mỗi ô: ok (Top-1 đúng, NaN->-9, hoà lấy hàng đầu), cos_true (max điểm trong hàng y==1), margin (cos_true - max điểm hàng y==0).
    Trả DataFrame index = ô (đã sắp)."""
    s = np.asarray(s, np.float32)
    s9 = np.where(np.isnan(s), np.float32(-9.0), s).astype(np.float32)
    df = pd.DataFrame({"i": i, "s": s9, "y": y})
    idx = df.groupby("i", sort=True)["s"].idxmax()
    ok = df.y.values[idx.values].astype(np.int8)
    cells = idx.index.values
    st = pd.Series(np.where(y == 1, s, np.nan)).groupby(i, sort=True).max()
    sw = pd.Series(np.where(y == 0, s, np.nan)).groupby(i, sort=True).max()
    return pd.DataFrame({"ok": ok, "cos_true": st.reindex(cells).values, "margin": (st - sw).reindex(cells).values}, index=cells)


def boot_idx(pages, B=BOOT_B, seed=BOOT_SEED):
    """Ma trận chỉ số cụm trang (dùng chung cho mọi so sánh cặp) -> (inv, n_pages, idx)."""
    up, inv = np.unique(pages, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(up), size=(B, len(up)))
    return inv, len(up), idx


def boot_mean(v, pages, B=BOOT_B, seed=BOOT_SEED):
    """trung bình v + CI95 % bootstrap cụm trang (v float, NaN bị bỏ)."""
    v = np.asarray(v, float); pages = np.asarray(pages)
    m = ~np.isnan(v)
    if m.sum() == 0:
        return (float("nan"),) * 3
    v, pages = v[m], pages[m]
    inv, npg, idx = boot_idx(pages, B, seed)
    k = np.bincount(inv, weights=v, minlength=npg); n = np.bincount(inv, minlength=npg).astype(float)
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return (float(v.mean()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))


def boot_diff(a, b, pages, B=BOOT_B, seed=BOOT_SEED):
    """Δ = mean(a) - mean(b) theo cặp ô (NaN ở một trong hai -> bỏ ô), CI95 % bootstrap cụm trang. Đơn vị: giữ nguyên (a,b ∈ [0,1] -> Δ ∈ [-1,1])."""
    a = np.asarray(a, float); b = np.asarray(b, float); pages = np.asarray(pages)
    m = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[m]; pg = pages[m]
    if len(d) == 0:
        return (float("nan"),) * 3 + (0,)
    inv, npg, idx = boot_idx(pg, B, seed)
    k = np.bincount(inv, weights=d, minlength=npg); n = np.bincount(inv, minlength=npg).astype(float)
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return (float(d.mean()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)), int(len(d)))


# ============================================================================================== biến đổi glyph
K3 = np.ones((3, 3), np.uint8)


def thick(g, d):
    """d > 0: làm nét ĐẬM thêm d điểm mỗi bên (erode xám = min-filter lan vùng tối); d < 0: mảnh đi."""
    if d == 0:
        return g
    return cv2.erode(g, K3, iterations=d) if d > 0 else cv2.dilate(g, K3, iterations=-d)


def hqc2_img(g, patch, ink, P=None, d=0, sigma_c=0.0):
    """HQC2: ghép alpha glyph (nền trắng 255, mực 0, vuông S×S) lên MẢNG GIẤY THẬT (uint8 ≥ P×P, cắt giữa P×P).
    Tuỳ chọn khớp quang học: độ đậm nét d, nhoè Gauss σ_c (đơn vị điểm của khung glyph), thu về P×P (độ phân giải gốc của crop).
    out = (1-α)·giấy + α·màu_mực  với α = (255-g)/255. Trả uint8 P×P."""
    g = thick(g, d)
    if sigma_c > 0:
        g = cv2.GaussianBlur(g, (0, 0), float(sigma_c))
    S = g.shape[0]
    P = int(P or S)
    if P != S:
        g = cv2.resize(g, (P, P), interpolation=cv2.INTER_AREA)
    a = (255.0 - g.astype(np.float32)) / 255.0
    h = patch.shape[0]; o = (h - P) // 2
    pp = patch[o:o + P, o:o + P].astype(np.float32)
    out = pp * (1.0 - a) + float(ink) * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def canon_bin(img, size=128, margin=0.10, min_area=6):
    """HQC3 (đúng binarize_canonical của p05_full_corpus_he_quy_chieu.py:135-163, nhưng KHÔNG ép crop thành vuông trước):
    Otsu (mực = tối) -> bỏ đốm < min_area điểm -> hộp bao mực -> thu về 80 % khung (lề 10 %) giữ tỉ lệ -> căn giữa size×size -> nền 255."""
    _, inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n > 1:
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
        inv = np.where(keep[lab], 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(inv)
    if len(ys) == 0:
        return np.full((size, size), 255, np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ch = inv[y0:y1, x0:x1]
    h, w = ch.shape
    scale = int(size * (1 - 2 * margin)) / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    sc = cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_NEAREST)
    can = np.zeros((size, size), np.uint8)
    sy, sx = (size - nh) // 2, (size - nw) // 2
    can[sy:sy + nh, sx:sx + nw] = sc
    return 255 - can


def geom_only(img, size=128, margin=0.10):
    """ĐỐI CHỨNG hình học (không nhị phân): cắt hộp bao mực (ngưỡng Otsu chỉ để tìm hộp), giữ MÀU XÁM, thu 80 % khung, nền = p95."""
    _, inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(inv)
    if len(ys) == 0:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ch = img[y0:y1, x0:x1]
    h, w = ch.shape
    scale = int(size * (1 - 2 * margin)) / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    sc = cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_AREA)
    bg = int(np.percentile(img, 95))
    can = np.full((size, size), bg, np.uint8)
    sy, sx = (size - nh) // 2, (size - nw) // 2
    can[sy:sy + nh, sx:sx + nw] = sc
    return can


# ============================================================================================== mô hình giấy (đã dựng ở v_thu_hqc_01)
_PM = {}


def paper_model(b: str) -> dict:
    """measure_out/_tn11/verify/thu_hqc/giay/<b>.npz + <b>.json -> dict(patches uint8 [n,112,112], ink, P_font, P_fd, stats)."""
    if b not in _PM:
        z = np.load(OUT / "giay" / f"{b}.npz")
        st = json.loads((OUT / "giay" / f"{b}.json").read_text(encoding="utf-8"))
        _PM[b] = dict(patches=z["patches"], pages=z["pages"], **st)
    return _PM[b]


# ============================================================================================== worker (mỗi tiến trình 1 luồng CPU)
_W = {}


def _winit():
    import torch
    torch.set_num_threads(1)
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    A = Assets(verify=False)
    _W["SI"] = SI
    _W["enc"] = SI.MultiEnc([A.ext(SI.ENC_FILES["v1"]), A.ext(SI.ENC_FILES["v2"])], "cpu", True)
    _W["gl"] = SI.Glyphs()


def _src(src, ch):
    gl = _W["gl"]
    return gl.font(ch) if src == "font" else gl.fdimg(ch)


EMB_BS = int(os.environ.get("HQC_BS", "32"))     # lô nhỏ: activations ResNet18@128 ~ 5 MB/ảnh -> giữ RAM mỗi worker < ~300 MB (máy đang tràn swap)


def _embed(enc, imgs, norm):
    outs = [enc.embed(imgs[s:s + EMB_BS], norm=norm) for s in range(0, len(imgs), EMB_BS)]
    return np.concatenate(outs) if outs else np.zeros((0, 512), np.float32)


def _norm_rows(E):
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def _wtask(t):
    """Một tác vụ: trả (id, E[n,512] float32; hàng NaN = không có glyph)."""
    enc = _W["enc"]
    kind = t["kind"]
    if kind in ("white", "paper", "canon", "geom"):
        src, chars = t["src"], t["chars"]
        base = [_src(src, ch) for ch in chars]
        has = [g is not None for g in base]
        K = 1
        imgs = []
        if kind == "white":
            imgs = [g for g in base if g is not None]; norm = False
        elif kind == "canon":
            imgs = [canon_bin(g) for g in base if g is not None]; norm = False
        elif kind == "geom":
            imgs = [geom_only(g) for g in base if g is not None]; norm = False
        else:
            cfg = t["cfg"]
            pm = paper_model(t["book"])
            K = len(cfg["patch_ids"]); norm = bool(cfg["norm"])
            P = cfg["P"] if cfg.get("P") not in (None, 0) else None
            for g in base:
                if g is None:
                    continue
                for k in cfg["patch_ids"]:
                    if cfg.get("white_bg"):
                        patch = np.full((112, 112), 255, np.uint8); ink = 0
                    else:
                        patch = pm["patches"][k]; ink = pm["ink"]
                    imgs.append(hqc2_img(g, patch, ink, P, cfg.get("d", 0), cfg.get("sigma", 0.0)))
        out = np.full((len(chars), 512), np.nan, np.float32)
        if imgs:
            E = _embed(enc, imgs, norm)
            if K > 1:
                E = E.reshape(-1, K, 512).mean(1)
            if kind == "paper":
                E = _norm_rows(E)
            out[np.array(has)] = E
        return t["id"], out
    if kind in ("crops_std", "crops_canon", "crops_geom"):
        imgs = t["imgs"]
        if kind == "crops_canon":
            imgs = [canon_bin(g) for g in imgs]; norm = False
        elif kind == "crops_geom":
            imgs = [geom_only(g) for g in imgs]; norm = None
        else:
            norm = None                                        # chuẩn hoá sản xuất của crop (Enc.norm = True)
        return t["id"], _embed(enc, imgs, norm)
    raise ValueError(kind)


def make_pool(n=None):
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(n or NWORK, initializer=_winit)


def embed_chars(pool, kind, src, chars, store: Path, book=None, cfg=None, chunk=64, save_every=40, tag=""):
    """Nhúng danh sách ký tự (K mảng giấy trung bình nếu kind='paper'); lưu/đọc lại từ `store` (.npz keys+E) để chạy tiếp.
    Trả dict ký tự -> vec (NaN nếu không có glyph)."""
    store = Path(store)
    done = {}
    if store.exists():
        z = np.load(store, allow_pickle=False)
        done = {str(k): v for k, v in zip(z["keys"], z["E"])}
    todo = [c for c in dict.fromkeys(chars) if c not in done]
    log(f"[{tag or store.stem}] {len(chars)} ký tự, đã có {len(chars) - len(todo)}, cần nhúng {len(todo)}")
    if not todo:
        return done
    tasks = []
    for s in range(0, len(todo), chunk):
        t = dict(id=s, kind=kind, src=src, chars=todo[s:s + chunk])
        if kind == "paper":
            t["book"] = book; t["cfg"] = cfg
        tasks.append(t)

    def save():
        ks = list(done)
        tmp = store.with_suffix(".tmp.npz")
        store.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tmp, keys=np.array(ks), E=np.stack([done[k] for k in ks]).astype(np.float32))
        tmp.replace(store)
    t0 = time.time()
    nimg = 0
    for n, (tid, E) in enumerate(pool.imap(_wtask, tasks, chunksize=1)):
        for ch, v in zip(todo[tid:tid + chunk], E):
            done[ch] = v
        nimg += int(np.sum(~np.isnan(E[:, 0])))
        if (n + 1) % save_every == 0:
            save()
            el = time.time() - t0
            log(f"  [{tag or store.stem}] {min((n + 1) * chunk, len(todo))}/{len(todo)} ký tự, {el:.0f}s, "
                f"~{(len(todo) - (n + 1) * chunk) * el / max(1, (n + 1) * chunk) / 60:.1f} phút còn lại")
    save()
    log(f"[{tag or store.stem}] xong {len(todo)} ký tự trong {time.time() - t0:.0f}s")
    return done


def embed_images(pool, kind, imgs, chunk=64):
    """Nhúng danh sách ảnh xám (crop): kind 'crops_std' | 'crops_canon' | 'crops_geom'. Trả mảng [n,512] đúng thứ tự."""
    tasks = [dict(id=s, kind=kind, imgs=imgs[s:s + chunk]) for s in range(0, len(imgs), chunk)]
    out = np.zeros((len(imgs), 512), np.float32)
    for tid, E in pool.imap(_wtask, tasks, chunksize=1):
        out[tid:tid + len(E)] = E
    return out


# ============================================================================================== cắt lại crop đúng hình học sản xuất
def recut_crops(b: str, D: pd.DataFrame, want: np.ndarray, workers=4):
    """Cắt lại crop (ảnh xám của tệp giao nộp) cho các ô `want` bằng đúng t02_embed.page_crops (cần mọi ô của trang để carve hàng xóm)."""
    from concurrent.futures import ThreadPoolExecutor
    from t02_embed import page_crops
    cfg = T.BOOKS[b]
    prep = T.REPO / cfg["prep"]
    want = set(int(x) for x in want)
    jobs = []
    for pg, g in D.groupby("page", sort=True):
        if not (set(g.index) & want):
            continue
        recs = [(i, c, T.bbox_of(bb)) for i, c, bb in zip(g.index, g.column, g.bbox)]
        jobs.append((b, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs))
    res = {}
    with ThreadPoolExecutor(workers) as ex:
        for lst in ex.map(page_crops, jobs):
            for i, g, m in lst:
                if int(i) in want and g is not None:
                    res[int(i)] = g
    return res


# ============================================================================================== mẫu con cố định (hạt giống numpy)
def sample_cells(b: str, cells: np.ndarray, n: int = 3000):
    """Hoán vị cố định của các ô có sự thật (np.random.default_rng(SEED + 100*chỉ_số_bộ)) -> n ô đầu. Các mẫu con lồng nhau:
    cal = [:250], s1000 = [:1000], s3000 = [:3000]."""
    rng = np.random.default_rng(SEED + 100 * BI[b])
    perm = rng.permutation(np.sort(np.asarray(cells)))
    return perm[:n]
