"""crops.py — cắt ảnh cho ô được giao (mức trong params.CROP_LEVELS: keep trở lên), mỗi trang một tác vụ (≤ 3 worker, CPU).

  crops/        = CHÍNH pipeline.align_engine.build_dataset.save_crop (như a08_crops.py): pad 0,12, carve mực láng giềng
                  prev/next cùng cột theo y (láng giềng = mọi ô người có hộp trong cột, như bản ghi build), tighten_box;
                  hình học trên prepared/pages, ĐIỂM ẢNH jpg gốc (crop_source original).
  crops_chuan/  = crop chuẩn v2 (pipeline.gold_exact.crop_chuan.canon, PARAMS đóng băng): ảnh VUÔNG điểm ảnh gốc;
                  cờ blank/cut/two + đo crop_quality (bleed/truncated) trên crop chặt mới -> one_char_ok như gold_exact.
Tên tệp: <book>/<page>_c<col:02d>_<idx:04d>.png (== a08).
"""
from __future__ import annotations

import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .params import BOOKS, REPO, SAVE_CROP_PAD

_M = {}


def _init():
    cv2.setNumThreads(1)


def crop_name(book, page, col, idx):
    return f"{book}/{page}_c{int(col):02d}_{int(idx):04d}.png"


def _page(task):
    from pipeline.align_engine.build_dataset import save_crop
    from pipeline.align_engine.crop_quality import measure
    from pipeline.gold_exact import crop_chuan as CC
    book, pn, src_file, rows, want, out_dir = task
    out_dir = Path(out_dir)
    img = cv2.imread(str(REPO / "prepared" / book / "pages" / f"{pn}.png"), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    org = cv2.imread(str(REPO / BOOKS[book]["data"] / src_file), cv2.IMREAD_COLOR)
    if org is None or org.shape[:2] != img.shape[:2]:
        raise RuntimeError(f"{book}/{pn}: ảnh gốc {src_file} thiếu hoặc khác kích thước")
    df = pd.DataFrame(rows, columns=["idx", "col", "x1", "y1", "x2", "y2"])
    colboxes = {c: [[int(r.x1), int(r.y1), int(r.x2), int(r.y2)] for r in g.itertuples()] for c, g in df.groupby("col")}
    col_cx = {c: float(np.median([(b[0] + b[2]) / 2.0 for b in bb])) for c, bb in colboxes.items() if bb}
    pitches = [CC.col_pitch(CC.distinct_col_boxes(bb)) for bb in colboxes.values() if bb]
    pitches = [p for p in pitches if p]
    page_pitch = float(np.median(pitches)) if pitches else None
    out = []
    for c, gc in df.groupby("col"):
        gc = gc.assign(cy=(gc.y1 + gc.y2) / 2).sort_values("cy")
        bb = [[int(r.x1), int(r.y1), int(r.x2), int(r.y2)] for r in gc.itertuples()]
        for t, r in enumerate(gc.itertuples()):
            if int(r.idx) not in want:
                continue
            rel = crop_name(book, pn, c, r.idx)
            o = dict(idx=int(r.idx), image=f"crops/{rel}")
            q = save_crop(img, gray, bb[t], SAVE_CROP_PAD, out_dir / "crops" / rel, tighten=True,
                          prev_bbox=bb[t - 1] if t > 0 else None, next_bbox=bb[t + 1] if t + 1 < len(bb) else None,
                          img_orig=org)
            if q is None:
                o.update(image="", image_md5="", crop_w=0, crop_h=0, crop_ink=np.nan)
            else:
                o.update(image_md5=hashlib.md5((out_dir / "crops" / rel).read_bytes()).hexdigest(), crop_w=q["w"], crop_h=q["h"],
                         crop_ink=q["ink"], crop_source=q["crop_source"])
            try:
                res = CC.canon(gray, org, bb[t], colboxes.get(c, []), [v for k, v in col_cx.items() if k != c], page_pitch)
            except Exception as e:  # noqa: BLE001
                o.update(cc_status=f"ERR:{type(e).__name__}"[:60], crop_chuan="", crop_chuan_md5="")
                out.append(o)
                continue
            o.update(cc_status=res.get("status", ""), cc_flags=res.get("flags", ""), cc_ink_norm=res.get("ink_norm", np.nan),
                     cc_tall=bool(res.get("tall", False)))
            if res.get("status") == "ok":
                o["cc_quality_flag"] = measure(res["tight_img"]).get("crop_quality_flag", "")
                ok, b = cv2.imencode(".png", res["sq_img"])
                f = out_dir / "crops_chuan" / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b.tobytes())
                o.update(crop_chuan=f"crops_chuan/{rel}", crop_chuan_md5=hashlib.md5(b.tobytes()).hexdigest())
            else:
                o.update(crop_chuan="", crop_chuan_md5="", cc_quality_flag="")
            out.append(o)
    return book, pn, out


def run(A: pd.DataFrame, sel: np.ndarray, out_dir: Path, source_file: dict, workers=3, log=print) -> pd.DataFrame:
    """A = mọi ô người; sel = mặt nạ ô cần ảnh. Láng giềng cột = mọi ô có hộp (kind != skip) cùng trang/cột."""
    has = A[A.kind != "skip"]
    want = A[sel]
    wmap = {k: set(g.idx.astype(int)) for k, g in want.groupby(["book", "page"])}
    tasks = []
    for (b, p), g in has.groupby(["book", "page"], sort=True):
        if (b, p) not in wmap:
            continue
        rows = list(zip(g.idx.astype(int), g.col.astype(int), g.x1, g.y1, g.x2, g.y2))
        tasks.append((b, p, source_file[(b, p)], rows, wmap[(b, p)], str(out_dir)))
    recs = []
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init) as pool:
        for k, (b, p, out) in enumerate(pool.imap(_page, tasks, chunksize=2)):
            for o in out:
                o.update(book=b, page=p)
            recs.extend(out)
            if k % 100 == 0:
                log(f"  crop: trang {k}/{len(tasks)}")
    R = pd.DataFrame(recs)
    log(f"  crop: {len(R)} ô, save_crop ok {int((R.image != '').sum())}, crop chuẩn ok {int((R.crop_chuan != '').sum())}")
    return R


def one_char_ok(R: pd.DataFrame, key="book") -> pd.Series:
    """== gold_exact.signals_geom.crop_flags/ink_flag (TN4): status ok ∧ ¬blank ∧ ¬cut ∧ ¬two ∧ mực trong khoảng
    [0,4; 2,5]× trung vị cùng (sách, chữ) (lớp < 5 ô: trung vị sách, [0,25; 4]) ∧ crop_quality ≠ bleed/truncated."""
    from pipeline.gold_exact.crop_chuan import PARAMS as P
    st = R.cc_status.fillna("MISSING").astype(str).values
    fl = R.cc_flags.fillna("").astype(str).values
    has = lambda t: np.array([t in f.split("|") for f in fl])  # noqa: E731
    ok = st == "ok"
    ink = pd.to_numeric(R.cc_ink_norm, errors="coerce")
    X = pd.DataFrame({key: R[key].values, "label": R.char.values, "ink": ink.values, "ok": ok})
    med_lab = X[X.ok].groupby([key, "label"]).ink.agg(["median", "size"])
    med_set = X[X.ok].groupby(key).ink.median()
    j = X[[key, "label"]].merge(med_lab, left_on=[key, "label"], right_index=True, how="left")
    big = j["size"].fillna(0).values >= P["ink_nmin"]
    ref = np.where(big, j["median"].values, X[key].map(med_set).values)
    r = X.ink.values.astype(float) / np.maximum(ref.astype(float), 1e-9)
    lo = np.where(big, P["ink_lo"], P["ink_lo_w"])
    hi = np.where(big, P["ink_hi"], P["ink_hi_w"])
    f_ink = ok & ((r < lo) | (r > hi))
    cq = R.cc_quality_flag.fillna("").astype(str).values
    good = ok & ~has("blank") & ~has("cut") & ~has("two") & ~f_ink & (cq != "bleed") & (cq != "truncated")
    return pd.Series(good, index=R.index)


def page_source_files() -> dict:
    out = {}
    for b in BOOKS:
        for p in json.load(open(REPO / "prepared" / b / "manifest.json", encoding="utf-8"))["pages"]:
            out[(b, p["page_name"])] = p["source_file"]
    return out
