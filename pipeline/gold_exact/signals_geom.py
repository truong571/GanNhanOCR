"""signals_geom.py — tín hiệu HÌNH HỌC của ô GOLD (CPU, 0 API, ≤ 4 worker).

  crop chuẩn v2   crop_chuan.canon trên trang (bản ghi build mọi tầng làm láng giềng cột, như TN4 s01_crop) -> cờ blank/cut/two,
                  tall_new, bleed/truncated của crop_quality (pipeline) đo trên crop chặt MỚI; ảnh vuông + 128×128 ghi vào cache.
  ink             tỉ lệ mực/(p·w) so với trung vị cùng (bộ, nhãn) (lớp < 5 ô: trung vị bộ, khoảng rộng) — TN4 s02_geom.ink_flag.
  core_loss       CHỐT AN TOÀN MỚI (đăng ký trước, config/gold_exact.yaml `core_loss`): mực của CROP CŨ (luật save_crop,
                  ngưỡng Otsu cục bộ của crop chuẩn) trong vùng lõi = |y − tâm hộp cũ| ≤ 0,3·p × dải cột [xa, xb) của crop chuẩn;
                  core_loss = phần mực lõi đó KHÔNG thuộc mặt nạ mực của crop chuẩn; > 0,25 -> text_only.
  B0_box          geo_f_dup_bbox ∪ geo_f_ov_heavy — gold_img_audit/crop_geometry/geom_flags.column_geometry trên bản ghi thô
                  (STT dataset_out/labels_final.csv, sách khác prepared/<S>/dataset_out/labels_gated.csv), mọi tầng.
  CNT             số chữ OCR của cột ≠ số âm QN (dataset/_ALL/columns.csv) — geom_flags f_cnt_ocr_ne_qn.
  dy/dx kim       crop_geometry/kim_vs_box: tâm hộp giao so với hộp chữ kim đã đọc (align_production._detect, 0 API) theo bước cột.
  int_foreign     ô rescue mà tệp ảnh là crop build của ô KHÁC (integrity_viewer a_integrity, RESCUE_FILE_OTHER_CELL).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import crop_chuan as CC
from .common import ORIGINAL, PREP_DIR, REPO, md5_bytes, page_dir, rd, uid_path

BUILD = {"SachThanhTruyen": REPO / "dataset_out/labels.csv",
         **{s: REPO / "prepared" / s / "dataset_out/labels.csv" for s in ORIGINAL}}
RAW = {"SachThanhTruyen": REPO / "dataset_out/labels_final.csv",
       **{s: REPO / f"prepared/{s}/dataset_out/labels_gated.csv" for s in ORIGINAL}}
LITHO4 = {"LucVanTien1883", "KimVanKieu1884", "LucVanTien1916", "TruyenKieu1872"}


def _stat_sig(paths):
    out = []
    for p in paths:
        p = Path(p)
        out.append((str(p), p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else (str(p), -1, -1))
    return out


def code_sig() -> str:
    """md5 mã crop chuẩn + hàm trang + PARAMS: đổi mã = cache trang vô hiệu."""
    src = inspect.getsource(CC) + inspect.getsource(_page_worker) + inspect.getsource(core_loss_of) + json.dumps(CC.PARAMS)
    return hashlib.md5(src.encode()).hexdigest()


# ================================================================================================ chốt core_loss
def core_loss_of(G, bbox, res, oc, band=0.3):
    """ĐĂNG KÝ TRƯỚC (xem config core_loss): tỉ lệ mực lõi của crop CŨ bị crop chuẩn bỏ đi.
    mực cũ = (G < T) trong khung crop cũ (save_crop) trừ vùng carve; T = ngưỡng Otsu cục bộ của crop chuẩn (như TN4 đo khe người);
    lõi = hàng y với |y − cy_hộp| ≤ band·p (p = bước cột crop chuẩn) và cột x ∈ [xa, xb) (dải cột crop chuẩn);
    giữ = điểm ảnh lõi thuộc mặt nạ mực M của crop chuẩn. Trả (n_core, n_kept, loss) ; loss = 0 khi lõi không có mực."""
    T = res["_T"]
    wx0, wy0, M = res["M_win"]
    ox0, oy0, ox1, oy1 = (int(v) for v in oc["rect"])
    O = (G[oy0:oy1, ox0:ox1] < T) & ~oc["carved"]
    ys, xs = np.nonzero(O)
    ys = ys + oy0; xs = xs + ox0
    cyb = (float(bbox[1]) + float(bbox[3])) / 2.0
    p = float(res["p"]); xa, xb = (float(v) for v in res["band_x"])
    core = (np.abs(ys - cyb) <= band * p) & (xs >= xa) & (xs < xb)
    n_core = int(core.sum())
    if n_core == 0:
        return 0, 0, 0.0
    yy, xx = ys[core] - wy0, xs[core] - wx0
    inside = (yy >= 0) & (yy < M.shape[0]) & (xx >= 0) & (xx < M.shape[1])
    kept = int(M[yy[inside], xx[inside]].sum())
    return n_core, kept, 1.0 - kept / n_core


# ================================================================================================ trang -> crop chuẩn
_BD = None


def _init():
    global _BD
    cv2.setNumThreads(1)
    from pipeline.align_engine import build_dataset as bd
    _BD = bd


def _cq(gray):
    from pipeline.align_engine.crop_quality import measure
    return measure(gray)


def _page_worker(task):
    """Một trang: crop chuẩn mọi ô GOLD + crop cũ dựng lại (core_loss); ghi PNG vào cache; trả bản ghi từng ô."""
    bs, bk, page, pdir, use_orig, rows, colrows, sq_root, s128_root, band = task
    bd = _BD
    img = cv2.imread(str(Path(pdir) / "pages" / f"{page}.png"), cv2.IMREAD_COLOR)
    if img is None:
        return [dict(cell_uid=r["uid"], status="NO_PAGE") for r in rows]
    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src, src_kind = None, "processed"
    if use_orig:
        src = bd.load_original_page(Path(pdir), page, shape_like=img.shape[:2])
    if src is not None:
        src_kind = "original"
    else:
        src = img
    colboxes = {c: list(bb) for c, bb in colrows.items()}
    col_cx = {c: float(np.median([(b[0] + b[2]) / 2.0 for b in bb])) for c, bb in colboxes.items() if bb}
    pitches = [CC.col_pitch(CC.distinct_col_boxes(bb)) for bb in colboxes.values() if bb]
    pitches = [p for p in pitches if p]
    page_pitch = float(np.median(pitches)) if pitches else None
    out = []
    for r in rows:
        o = dict(cell_uid=r["uid"], src_kind=src_kind)
        bbox, col = r["bbox"], r["column"]
        try:
            res = CC.canon(G, src, bbox, colboxes.get(col, []), [v for c, v in col_cx.items() if c != col], page_pitch)
        except Exception as e:  # noqa: BLE001
            o["status"] = f"ERR:{type(e).__name__}:{e}"[:120]; out.append(o); continue
        for k in ("p", "p_src", "wc", "xc", "top_kind", "bot_kind", "T", "n_comp", "n_own", "n_cut_comp", "n_tip", "n_line",
                  "mass", "ink_norm", "cut_ratio", "edge", "flags", "status", "tall", "E", "Wd", "ink_cx", "ink_cy",
                  "foreign_erased_px"):
            if k in res:
                o[k] = res[k]
        for k in ("ink_box", "sq", "win", "band_x"):
            if k in res:
                o[k] = json.dumps([float(v) for v in res[k]])
        if res.get("status") == "ok":
            q = _cq(res["tight_img"])
            o.update({f"new_{k}": v for k, v in q.items()})
            rel = uid_path(r["uid"])
            ok1, b1 = cv2.imencode(".png", res["sq_img"]); ok2, b2 = cv2.imencode(".png", res["sq128"])
            f1 = Path(sq_root) / rel; f2 = Path(s128_root) / rel
            f1.parent.mkdir(parents=True, exist_ok=True); f2.parent.mkdir(parents=True, exist_ok=True)
            f1.write_bytes(b1.tobytes()); f2.write_bytes(b2.tobytes())
            o["sq_md5"] = md5_bytes(b1.tobytes()); o["sq128_md5"] = md5_bytes(b2.tobytes())
            oc = CC.old_crop(img, G, bbox, r["prev_b"], r["next_b"])
            if oc is not None:
                n_core, kept, loss = core_loss_of(G, bbox, res, oc, band)
                o.update(core_n=n_core, core_kept=kept, core_loss=round(loss, 5))
                o["old_tall"] = bool(oc["gray"].shape[0] > 1.8 * max(oc["gray"].shape[1], 1))
        out.append(o)
    return out


def load_build(bs):
    B = rd(BUILD[bs], usecols=["book", "page", "column", "bbox"])
    return B[B.bbox != ""]


def run_crop_chuan(G: pd.DataFrame, cache_dir: Path, workers=4, recompute=False, band=0.3, log=print) -> pd.DataFrame:
    root = Path(cache_dir) / "crop_chuan"
    sq_root, s128_root, pg_root = root / "sq", root / "sq128", root / "pages"
    csig = code_sig()
    tasks, cached = [], []
    for bs, sub in G.groupby("book_set", sort=False):
        B = load_build(bs)
        Bg = {k: g for k, g in B.groupby(["book", "page"])}
        for (bk, pg), g in sub.groupby(["book", "page"], sort=False):
            pdir = page_dir(bs, bk)
            colrows = defaultdict(list)
            bp = Bg.get((bk, pg))
            if bp is not None:
                for c, b in zip(bp.column, bp.bbox):
                    colrows[c].append(json.loads(b))
            rows = []
            for u, bb, col in zip(g.cell_uid, g.bbox, g.column):
                bb = json.loads(bb)
                pv, nx = CC.neighbours_by_rule(colrows.get(col, []), bb)
                rows.append(dict(uid=u, bbox=bb, column=col, prev_b=pv, next_b=nx))
            origf = []
            if bs in ORIGINAL:
                try:
                    from pipeline.align_engine.build_dataset import _orig_index
                    s_ = _orig_index(Path(pdir))["pages"].get(pg)
                    origf = [s_] if s_ else []
                except Exception:  # noqa: BLE001
                    origf = []
            key = hashlib.sha256(json.dumps([csig, band, _stat_sig([pdir / "pages" / f"{pg}.png"] + origf), rows,
                                             {c: v for c, v in sorted(colrows.items())}], default=str).encode()).hexdigest()
            pf = pg_root / bs / bk / f"{pg}.pkl"
            if not recompute and pf.exists():
                try:
                    z = pickle.load(open(pf, "rb"))
                    if z["key"] == key and all((sq_root / uid_path(r["cell_uid"])).exists() for r in z["recs"]
                                               if r.get("status") == "ok"):
                        cached.extend(z["recs"]); continue
                except Exception:  # noqa: BLE001
                    pass
            tasks.append(((bs, bk, pg, str(pdir), bs in ORIGINAL, rows, dict(colrows), str(sq_root), str(s128_root), band), key, pf))
    log(f"crop chuẩn: {len(tasks)} trang cần tính, {len(cached)} ô lấy từ cache")
    recs = list(cached)
    if tasks:
        with Pool(workers, initializer=_init) as pool:
            for k, (res, (t, key, pf)) in enumerate(zip(pool.imap(_page_worker, [t[0] for t in tasks], chunksize=2), tasks)):
                pf.parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(dict(key=key, recs=res), open(pf, "wb"))
                recs.extend(res)
                if k % 200 == 0:
                    log(f"  trang {k}/{len(tasks)}")
    D = pd.DataFrame(recs)
    return D


def ink_flag(D: pd.DataFrame, key="set8"):
    """== TN4 s02_geom.ink_flag (tham số cclib_v2.PARAMS)."""
    P = CC.PARAMS
    ok = D.status == "ok"
    med_lab = D[ok].groupby([key, "label"]).ink_norm.agg(["median", "size"])
    med_set = D[ok].groupby(key).ink_norm.median()
    j = D[[key, "label"]].merge(med_lab, left_on=[key, "label"], right_index=True, how="left")
    big = (j["size"].fillna(0).values >= P["ink_nmin"])
    ref = np.where(big, j["median"].values, D[key].map(med_set).values)
    r = D.ink_norm.values.astype(float) / np.maximum(ref.astype(float), 1e-9)
    lo = np.where(big, P["ink_lo"], P["ink_lo_w"]); hi = np.where(big, P["ink_hi"], P["ink_hi_w"])
    fl = ok.values & ((r < lo) | (r > hi))
    return r, fl


def crop_flags(G: pd.DataFrame, D: pd.DataFrame, core_thr=0.25) -> pd.DataFrame:
    """Ghép bản ghi crop chuẩn vào G (thứ tự G) + cờ one_char_ok / tall_new / core_loss (định nghĩa TN4 t00_freeze)."""
    X = G[["cell_uid", "set8", "label"]].merge(D, on="cell_uid", how="left", validate="1:1")
    st = X.status.fillna("MISSING").values
    fl = X["flags"].fillna("").astype(str).values
    has = lambda t: np.array([t in f.split("|") for f in fl])
    X["f_blank"] = has("blank") | (st != "ok")
    X["f_cut"] = has("cut"); X["f_two"] = has("two")
    X["ink_norm"] = pd.to_numeric(X.ink_norm, errors="coerce")
    r, fink = ink_flag(X.assign(status=st), "set8")
    X["ink_ratio"], X["f_ink"] = r, fink
    ncq = X.new_crop_quality_flag.fillna("").values
    X["bleed_new"], X["trunc_new"] = ncq == "bleed", ncq == "truncated"
    X["tall_new"] = X.tall.fillna(False).astype(bool).values
    X["one_char_ok"] = ((st == "ok") & ~X.f_blank & ~X.f_cut & ~X.f_two & ~X.f_ink & ~X.bleed_new & ~X.trunc_new)
    X["core_loss"] = pd.to_numeric(X.get("core_loss"), errors="coerce")
    X["core_loss_flag"] = (X.core_loss > core_thr).fillna(False).values & (st == "ok")
    return X


# ================================================================================================ cờ hình học thô (geom_flags)
def load_raw() -> pd.DataFrame:
    parts = []
    for bs, p in RAW.items():
        d = rd(p); d["book_set"] = bs; parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["key"] = (d.book_set + "/" + d.book + "/" + d.page + "/c" + d.column.astype(str) + "/n" + d.nom_idx.astype(str)
                + "/s" + d.syl_idx.astype(str))
    return d


def _pb(s):
    try:
        v = json.loads(s) if isinstance(s, str) and s.startswith("[") else None
        return [float(x) for x in v[:4]] if v and len(v) >= 4 else None
    except Exception:  # noqa: BLE001
        return None


def column_geometry(raw: pd.DataFrame, OV_HEAVY=0.5) -> pd.DataFrame:
    """== geom_flags.column_geometry, chỉ giữ cờ dùng ở đây: f_dup_bbox, f_ov_heavy, pitch (làm tròn 1 như flags.csv)."""
    bb = raw.bbox.map(_pb)
    ok = bb.notna()
    r = raw[ok].copy()
    arr = np.array(bb[ok].tolist())
    r["x1"], r["y1"], r["x2"], r["y2"] = arr.T
    r["cy"] = (r.y1 + r.y2) / 2
    r["h"] = r.y2 - r.y1
    r["ni"] = r.nom_idx.astype(int); r["si"] = r.syl_idx.astype(int)
    by_nom = r.count_source.isin(["equal_ocr", "pitch_ocr", "pitch_rule", "conflict", "legacy"])
    r["ai"] = np.where(by_nom, r.ni, r.si)
    r["bbkey"] = r.book_set + "|" + r.book + "|" + r.page + "|" + r.bbox
    dup = (r.groupby("bbkey").key.transform("count") > 1).astype(int)
    recs, cols = [], {}
    for (bs, bk, pg, col), g in r.groupby(["book_set", "book", "page", "column"], sort=False):
        g = g.sort_values(["ai", "si"])
        cy, ai = g.cy.values, g.ai.values
        dy, da = np.diff(cy), np.diff(ai)
        cross = ((ai[:-1] <= 5) & (ai[1:] >= 6)) if bs in LITHO4 else np.zeros(len(dy), bool)
        steps = [(dy[k] / da[k]) for k in range(len(dy)) if da[k] >= 1 and not cross[k] and dy[k] > 0]
        pitch = float(np.median(steps)) if len(steps) >= 2 else np.nan
        recs.append((bs, bk, pg, col, pitch)); cols[(bs, bk, pg, col)] = (g, pitch)
    pp = defaultdict(list)
    for bs, bk, pg, col, p in recs:
        if p == p:
            pp[(bs, bk, pg)].append(p)
    rows = []
    for (bs, bk, pg, col), (g, pitch) in cols.items():
        if not (pitch == pitch):
            v = pp.get((bs, bk, pg))
            pitch = float(np.median(v)) if v else np.nan
        n = len(g)
        y1, y2, h = g.y1.values, g.y2.values, g.h.values
        ovh = np.zeros(n, int)
        for k in range(n - 1):
            ov = max(0.0, min(y2[k], y2[k + 1]) - max(y1[k], y1[k + 1])) / max(1.0, min(h[k], h[k + 1]))
            if ov > OV_HEAVY:
                ovh[k] = ovh[k + 1] = 1
        pr = round(pitch, 1) if pitch == pitch else np.nan
        for k_, o in zip(g.key.values, ovh):
            rows.append((k_, pr, o))
    geo = pd.DataFrame(rows, columns=["key", "pitch", "f_ov_heavy"]).set_index("key")
    geo["f_dup_bbox"] = pd.Series(dup.values, index=r.key.values)
    return geo


def cnt_flag(G: pd.DataFrame, all_dir: Path) -> np.ndarray:
    """f_cnt_ocr_ne_qn = n_ocr ≠ n_qn của cột (NaN ≠ NaN -> True, đúng như geom_flags)."""
    c = rd(Path(all_dir) / "columns.csv")
    c["ckey"] = c.book_set + "|" + c.book + "|" + c.page + "|" + c.column
    c = c.set_index("ckey")
    ck = G.book_set + "|" + G.book + "|" + G.page + "|" + G.column
    no = pd.to_numeric(ck.map(c.n_ocr), errors="coerce"); nq = pd.to_numeric(ck.map(c.n_qn), errors="coerce")
    return (no != nq).values


# ================================================================================================ hộp kim (kim_vs_box)
_KG = {}


def _kim_init(qn_path):
    cv2.setNumThreads(1)
    from core.text.dictionary import load_qn_to_nom
    _KG["qn"] = set(load_qn_to_nom(qn_path).keys())


def _kim_page(args):
    bname, page, bdict = args
    from pipeline.align_engine.align_production import _detect
    from pipeline.align_engine.book_layout import book_layout
    d = _detect(page, REPO / "prepared" / bname, _KG["qn"], layout=book_layout(bdict))
    if d is None:
        return None
    cols, qn_lines, iter_pairs, binary, ok = d
    return {int(line_id): [(c.get("char") or c.get("text") or "", c.get("bbox")) for c in cols[ci]["chars"]]
            for ci, line_id in iter_pairs}


def kim_vs_box(G: pd.DataFrame, raw: pd.DataFrame, geo: pd.DataFrame, cfg_map: dict, cache_dir: Path, workers=4,
               recompute=False, log=print) -> pd.DataFrame:
    """dy_kim = (cy_hộp − cy_kim)/bước, dx_kim = (cx_hộp − cx_kim)/bề ngang hộp kim — cho mọi ô GOLD (bbox = bản ghi thô)."""
    from pipeline.align_engine.build_dataset import _book_code
    from pipeline.step0_setup import load_config
    base = load_config(str(REPO / "config/pipeline.yaml"))
    qn_path = str(REPO / base["paths"]["qn_to_nom_dict"])
    code2 = {}
    jobs = []
    need = G.groupby(["book_set", "book"]).page.apply(lambda s: sorted(set(s))).to_dict()
    root = Path(cache_dir) / "kim_pages"
    res = {}
    for bs, cfgf in cfg_map.items():
        cfg = load_config(str(REPO / cfgf))
        for b in cfg["books"]:
            code = _book_code(b["name"]); code2[(bs, code)] = b["name"]
            for pg in need.get((bs, code), []):
                dd = REPO / "prepared" / b["name"]
                sig = _stat_sig(sorted(p for sub in ("pages", "pages_denoised", "detected", "transcriptions", "labeled")
                                       for p in (dd / sub).glob(f"{pg}*")))
                key = hashlib.sha256(json.dumps([sig, b, cfgf], default=str).encode()).hexdigest()
                pf = root / b["name"] / f"{pg}.pkl"
                if not recompute and pf.exists():
                    z = pickle.load(open(pf, "rb"))
                    if z["key"] == key:
                        res[(bs, b["name"], pg)] = z["out"]; continue
                jobs.append(((b["name"], pg, b), (bs, b["name"], pg), key, pf))
    log(f"hộp kim: {len(jobs)} trang cần _detect, {len(res)} trang từ cache")
    if jobs:
        with Pool(workers, initializer=_kim_init, initargs=(qn_path,)) as pool:
            for k, (out, (_, rk, key, pf)) in enumerate(zip(pool.imap(_kim_page, [j[0] for j in jobs], chunksize=4), jobs)):
                pf.parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(dict(key=key, out=out), open(pf, "wb"))
                res[rk] = out
                if k % 200 == 0:
                    log(f"  _detect {k}/{len(jobs)}")
    rk = raw.set_index("key")
    pit = geo.pitch
    dys, dxs, oks = [], [], []
    for u, bs, bk in zip(G.cell_uid, G.book_set, G.book):
        dy = dx = np.nan; cok = np.nan
        if u in rk.index:
            r = rk.loc[u]
            cols = res.get((bs, code2.get((bs, bk)), r.page))
            chars = cols.get(int(r.column)) if cols else None
            b = _pb(r.bbox); ni = int(r.nom_idx)
            if chars and ni < len(chars) and b is not None and chars[ni][1]:
                kc, kb = chars[ni]
                kb = [float(v) for v in kb[:4]]
                p = pit.get(u, np.nan)
                if not (p == p):
                    ys = sorted(((c[1][1] + c[1][3]) / 2) for c in chars if c[1])
                    p = float(np.median(np.diff(ys))) if len(ys) > 2 else np.nan
                dy = ((b[1] + b[3]) / 2 - (kb[1] + kb[3]) / 2) / p if p == p and p > 0 else np.nan
                dx = ((b[0] + b[2]) / 2 - (kb[0] + kb[2]) / 2) / max(1.0, (kb[2] - kb[0]))
                dy = round(dy, 3) if dy == dy else dy; dx = round(dx, 3)
                cok = int(kc == r.ocr_char)
        dys.append(dy); dxs.append(dx); oks.append(cok)
    return pd.DataFrame(dict(cell_uid=G.cell_uid.values, dy_kim=dys, dx_kim=dxs, kim_char_ok=oks))


# ================================================================================================ ảnh của ô khác (rescue)
def int_foreign(G: pd.DataFrame, all_dir: Path) -> np.ndarray:
    """== integrity_viewer a_integrity §5: ô rescue (GOLD) có tên tệp trùng một crop build 'gold/' của ô KHÁC
    (nom_idx hoặc bbox khác) -> RESCUE_FILE_OTHER_CELL."""
    T = rd(Path(all_dir) / "labels_trace.csv", usecols=["cell_uid", "nom_idx"])
    nidx = dict(zip(T.cell_uid, T.nom_idx))
    bl = rd(REPO / "dataset_out/labels.csv", usecols=["image", "nom_idx", "bbox"])
    bg = bl[bl.image.str.startswith("gold/")]
    bmap = dict(zip(bg.image, zip(bg.nom_idx, bg.bbox)))
    out = np.zeros(len(G), bool)
    for i, (rule, img, u, bb) in enumerate(zip(G.rule, G.image, G.cell_uid, G.bbox)):
        if rule != "self_training_rescue":
            continue
        rel = img.replace("crops/SachThanhTruyen/", "")
        if rel in bmap:
            ni, bbx = bmap[rel]
            out[i] = not (ni == nidx.get(u, "") and bbx == bb)
    return out
