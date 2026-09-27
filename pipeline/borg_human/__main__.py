"""python -m pipeline.borg_human — dựng BỘ CROP NHÃN NGƯỜI từ 2 bản chép tay Borg.tonch (0 API, không tải gì).

Các bước (--stage all = tất cả theo thứ tự; hoặc liệt kê, vd --stage keep,paddle,export):
  detect    CenterNet v1 trên prepared/<b>/pages (+ kiểm prepared == stretch(jpg gốc))       -> WORK/det/
  units     đơn vị (hộp thật + ô ảo) + nhúng encoder v1+v2 (MPS)                               -> WORK/units/
  glyphs    nhúng glyph font/FD cho từ vựng 2 sách                                             -> WORK/units/glyph_emb.pkl
  align     6 lượt DP: font -> proto -> proto cho encoder v1 và v2                              -> WORK/align/
  keep      luật keep/keep_high (a12)                                                          -> WORK/cells_keep.csv.gz
  paddle    lọc vòng 5 trên hồ sơ Paddle ĐÓNG BĂNG: keep_paddle_ok, chuẩn hoá, keep_v5, θ       -> WORK/cells.csv.gz, paddle.json
  crops     crop save_crop + crop chuẩn v2 cho ô keep trở lên (CROP_LEVELS)                       -> OUT/crops, OUT/crops_chuan
  validate  θ bằng encoder v2 trên crop cuối (T1 glyph font, T2 nguyên mẫu khối khác; a07)       -> WORK/validate.json
  repro     so với số gốc r4/r5 (measure_out/_audit_2026-09-26)                                 -> WORK/repro.json
  export    labels.csv + README/DATASHEET + BUILD_INFO.json + CHECKSUMS.txt                     -> OUT/
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import align as AL
from . import crops as CR
from . import export as EX
from . import paddle as PD
from .geom import crop_gray, stretch, units_for_page
from .params import (BOOKS, CHECK, CROP_LEVELS, DETECTOR, FINAL, KEEP, OUT, PADDLE, PASSES, R4, R5, REF, REPO, SRC, WORK)

STAGES = ["detect", "units", "glyphs", "align", "keep", "paddle", "crops", "validate", "repro", "export"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def page_keys():
    out = []
    for b in BOOKS:
        for p in json.load(open(REPO / "prepared" / b / "manifest.json", encoding="utf-8"))["pages"]:
            out.append((b, p["page_name"], p["source_file"]))
    return out


def jdump(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(o, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


def read_csv(p, **k):
    return pd.read_csv(p, dtype={"char": str, "syllable": str}, keep_default_na=False, **k)


# ------------------------------------------------------------------------------------------------ detect (a01)
def st_detect(work: Path, dev: str):
    import cv2
    sys.path.insert(0, str(REPO / "train_crop"))
    from infer_centernet import CenterNetDetector
    det = CenterNetDetector(str(REPO / DETECTOR["ckpt"]), img=DETECTOR["img"], thr=DETECTOR["thr"], resize=DETECTOR["resize"],
                            device=dev)
    res, chk = {}, []
    t0 = time.time()
    for book, pn, src in page_keys():
        png = cv2.imread(str(REPO / "prepared" / book / "pages" / f"{pn}.png"), cv2.IMREAD_GRAYSCALE)
        jpg = cv2.imread(str(REPO / BOOKS[book]["data"] / src))
        ok = jpg is not None and jpg.shape[:2] == png.shape
        mad = float(np.abs(stretch(cv2.cvtColor(jpg, cv2.COLOR_BGR2GRAY)).astype(int) - png.astype(int)).mean()) if ok else -1
        chk.append(dict(book=book, page=pn, src=src, shape_ok=ok, mad=round(mad, 3)))
        bx = det.boxes_for_page(cv2.cvtColor(png, cv2.COLOR_GRAY2BGR))
        res[f"{book}/{pn}"] = [[round(float(v), 1) for v in b[:4]] + [round(float(b[4]), 4)] for b in bx]
    (work / "det").mkdir(parents=True, exist_ok=True)
    with gzip.open(work / "det" / "boxes_v1.json.gz", "wt") as f:
        json.dump(res, f)
    jdump(chk, work / "det" / "page_map_check.json")
    m = np.array([c["mad"] for c in chk])
    log(f"detect: {len(res)} trang, shape_ok {sum(c['shape_ok'] for c in chk)}, mad max {m.max()} ({time.time() - t0:.0f} s)")


def load_boxes(work):
    with gzip.open(work / "det" / "boxes_v1.json.gz", "rt") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ units (a04)
def st_units(work: Path, dev: str):
    import cv2
    from .encoders import MultiEnc
    B = load_boxes(work)
    enc = MultiEnc(["v1", "v2"], dev, True)
    allU, meta, embs = [], {}, []
    t0 = time.time()
    for key, bx in B.items():
        book, pn = key.split("/")
        U, m = units_for_page(bx)
        meta[key] = m
        g = cv2.imread(str(REPO / "prepared" / book / "pages" / f"{pn}.png"), cv2.IMREAD_GRAYSCALE)
        embs.append(enc.embed([crop_gray(g, u) for u in U]).astype(np.float16))
        for u in U:
            u["book"] = book
            u["page"] = pn
        allU += U
    E = np.concatenate(embs)
    (work / "units").mkdir(parents=True, exist_ok=True)
    np.save(work / "units" / "emb_v1.npy", E)
    pickle.dump(dict(units=allU, meta=meta), open(work / "units" / "units_v1.pkl", "wb"))
    log(f"units: {len(allU)} đơn vị, ảo {sum(u['virtual'] for u in allU)}, emb {E.shape} ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------------------------------------ glyphs (a04b)
def st_glyphs(work: Path, dev: str):
    import unicodedata
    from .encoders import Glyphs, MultiEnc
    voc = set()
    for book in BOOKS:
        for f in (REPO / "prepared" / book / "transcriptions").glob("*.json"):
            for s in json.load(open(f, encoding="utf-8"))["sentences"]:
                voc |= {c for c in s["nom_clean"] if unicodedata.category(c) == "Lo"}
    voc = sorted(voc)
    G = Glyphs()
    enc = MultiEnc(["v1", "v2"], dev, True)
    fonts, fds, fchars, dchars = [], [], [], []
    for c in voc:
        im = G.font(c)
        if im is not None:
            fonts.append(im); fchars.append(c)
        im2 = G.fdimg(c)
        if im2 is not None:
            fds.append(im2); dchars.append(c)
    Ef = enc.embed(fonts) if fonts else np.zeros((0, 512))
    Ed = enc.embed(fds) if fds else np.zeros((0, 512))
    pickle.dump(dict(font=dict(zip(fchars, Ef)), fd=dict(zip(dchars, Ed)), voc=voc), open(work / "units" / "glyph_emb.pkl", "wb"))
    log(f"glyphs: từ vựng {len(voc)}, font {len(fchars)}, fd {len(dchars)}")


# ------------------------------------------------------------------------------------------------ align (a05)
def st_align(work: Path, workers: int):
    prev = {}
    for tag, emis, enc, pv in PASSES:
        df = AL.run_pass(work, tag, emis, enc, prev.get(pv), workers=workers, log=log)
        prev[tag] = df


# ------------------------------------------------------------------------------------------------ keep (a06 + a12)
def st_keep(work: Path):
    A = read_csv(work / "align" / f"{FINAL}.csv.gz")
    B = read_csv(work / "align" / f"{CHECK}.csv.gz")[["book", "page", "idx", "unit", "post"]].rename(
        columns={"unit": "unit2", "post": "post2"})
    ps = A.groupby(["book", "page"]).kind.apply(lambda x: (x == "skip").mean()).rename("page_skip")
    A = A.merge(B, on=["book", "page", "idx"]).merge(ps, left_on=["book", "page"], right_index=True)
    A = A.sort_values(["book", "page", "idx"]).reset_index(drop=True)
    A["agree2"] = (A.unit == A.unit2).astype(int)
    base = (A.kind == "real") & (A.det_score >= KEEP["det_min"]) & (A.page_skip < KEEP["page_skip_max"])
    A["keep"] = (base & (A.post >= KEEP["post_v1"]) & (A.agree2 == 1) & (A.post2 >= KEEP["post_v2"])).astype(int)
    A["keep_high"] = (base & (A.post >= KEEP["post_v1"]) & (A.agree2 == 1) & (A.post2 >= KEEP["post_v2_high"])).astype(int)
    A["conf"] = np.where(A.agree2 == 1, np.minimum(A.post, A.post2), 0.0).round(5)
    A.loc[A.kind == "skip", "conf"] = 0.0
    A.to_csv(work / "cells_keep.csv.gz", index=False)
    log(f"keep: {len(A)} ô, keep {int(A.keep.sum())}, keep_high {int(A.keep_high.sum())}, {A.kind.value_counts().to_dict()}")


# ------------------------------------------------------------------------------------------------ paddle (r5 q06/q07/q14)
def st_paddle(work: Path):
    A = read_csv(work / "cells_keep.csv.gz")
    prof = PD.Prof()
    U = pickle.load(open(work / "units" / "units_v1.pkl", "rb"))["units"]
    um = PD.units_match(prof, U)
    log(f"paddle: đơn vị hồ sơ {um['n_units_profile']} / hiện tại {um['n_units_now']}, lệch {um['mismatch']}")
    A, st = PD.apply(A, prof, log=log)
    if um["mismatch"]:
        bad = A.book + "/" + A.page
        m = bad.isin(um["pages_mismatch"]).to_numpy()
        A.loc[m, "keep_v5"] = 0          # an toàn: trang có đơn vị không khớp hồ sơ -> không chứng nhận keep_v5
        st["keep_v5_after_unit_guard"] = int(A.keep_v5.sum())
    st["units_match"] = um
    A = A.sort_values(["book", "page", "idx"]).reset_index(drop=True)
    A.to_csv(work / "cells.csv.gz", index=False)
    jdump(st, work / "paddle.json")


# ------------------------------------------------------------------------------------------------ crops (a08 + crop chuẩn v2)
def st_crops(work: Path, out: Path, workers: int):
    A = read_csv(work / "cells.csv.gz")
    for sub in ("crops", "crops_chuan"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    sel = np.zeros(len(A), bool)
    for lv in CROP_LEVELS:                       # keep_v5 ⊂ keep ⊂ keep_high: cột cờ cùng tên
        sel |= (A[lv] == 1).to_numpy()
    R = CR.run(A, sel, out, CR.page_source_files(), workers=workers, log=log)
    R = R.merge(A[["book", "page", "idx", "char"]], on=["book", "page", "idx"], how="left")
    R["one_char_ok"] = CR.one_char_ok(R).astype(int)
    R.to_csv(work / "crops_rec.csv.gz", index=False)
    log(f"crops: one_char_ok {int(R.one_char_ok.sum())}/{len(R)}")


# ------------------------------------------------------------------------------------------------ validate (a07, encoder v2)
def st_validate(work: Path, out: Path, dev: str):
    import cv2
    from .encoders import Enc
    A = read_csv(work / "cells.csv.gz").sort_values(["book", "page", "idx"]).reset_index(drop=True)
    R = read_csv(work / "crops_rec.csv.gz")[["book", "page", "idx", "image"]]
    A = A.merge(R, on=["book", "page", "idx"], how="left")
    A["image"] = A.image.fillna("")
    D = 3
    OFF = list(range(-D, D + 1))
    win = np.full((len(A), 2 * D + 1), "", object)
    for _, g in A.groupby(["book", "page"]):
        ch = g.char.tolist(); ix = g.index.to_numpy()
        for t, i in enumerate(ix):
            for k, d in enumerate(OFF):
                if 0 <= t + d < len(ch):
                    win[i, k] = ch[t + d]
    win_ok = np.array([all(w) and len(set(w)) == 2 * D + 1 for w in win])
    kp = ((A.keep == 1) & (A.kind == "real") & (A.image != "")).to_numpy()
    enc = Enc("v2", dev, True)
    rows = np.nonzero(kp)[0]
    E = np.zeros((len(A), 256), np.float32)
    for s in range(0, len(rows), 4096):
        rr = rows[s:s + 4096]
        gs = [cv2.imread(str(out / A.image.iat[i]), cv2.IMREAD_GRAYSCALE) for i in rr]
        e = (enc.embed(gs) / np.sqrt(2)).astype(np.float16).astype(np.float32)   # == a08: ghép [v1,v2]/√2 lưu float16
        E[rr] = e
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    GL = pickle.load(open(work / "units" / "glyph_emb.pkl", "rb"))
    FONT = {}
    for c in GL["voc"]:
        vs = [AL.nrm(np.asarray(GL[k][c], np.float32)[256:512]) for k in ("font", "fd") if c in GL[k]]
        if vs:
            FONT[c] = AL.nrm(np.mean(vs, 0))
    sums, cnt = {}, {}
    for r in rows:
        key = (int(A.fold.iat[r]), A.char.iat[r])
        sums[key] = sums.get(key, 0) + E[r]
        cnt[key] = cnt.get(key, 0) + 1
    PRO = {}
    for f in range(5):
        for c in set(c for _, c in sums):
            n = sum(cnt.get((g, c), 0) for g in range(5) if g != f)
            if n >= 3:
                PRO[(f, c)] = AL.nrm(sum(sums[(g, c)] for g in range(5) if g != f and (g, c) in sums))

    def offs(getter, m):
        o, grp = [], []
        for i in np.nonzero(m)[0]:
            refs = [getter(int(A.fold.iat[i]), c) for c in win[i]]
            if any(v is None for v in refs):
                continue
            o.append(int(np.argmax(np.stack(refs) @ E[i])) - D)
            grp.append(A.book.iat[i] + "/" + A.page.iat[i])
        return np.array(o), np.array(grp)
    res = {}
    for lvl, m in (("keep", kp & win_ok), ("keep_v5", kp & win_ok & (A.keep_v5 == 1).to_numpy())):
        for name, getter in (("T1_v2", lambda f, c: FONT.get(c)), ("T2_v2", lambda f, c: PRO.get((f, c)))):
            o, g = offs(getter, m)
            res[f"{name}|{lvl}"] = PD.summarize(o, g, B=400, seed=0)
    jdump(res, work / "validate.json")
    log("validate: " + " · ".join(f"{k} θ={v['theta']} n={v['n']}" for k, v in res.items()))


# ------------------------------------------------------------------------------------------------ repro (so với r4/r5)
def st_repro(work: Path):
    rep = {}
    B = load_boxes(work)
    with gzip.open(R4 / "det" / "boxes_v1.json.gz", "rt") as f:
        B0 = json.load(f)
    rep["boxes_identical_pages"] = int(sum(B.get(k) == v for k, v in B0.items()))
    rep["boxes_pages_ref"] = len(B0)
    for tag in [p[0] for p in PASSES]:
        f0 = R4 / "align" / f"{tag}.csv.gz"
        if not f0.exists():
            continue
        a = read_csv(work / "align" / f"{tag}.csv.gz")[["book", "page", "idx", "char", "kind", "unit", "post"]]
        b = read_csv(f0)[["book", "page", "idx", "char", "kind", "unit", "post"]]
        m = a.merge(b, on=["book", "page", "idx"], suffixes=("", "_r4"))
        rep[f"align|{tag}"] = dict(n=len(a), n_ref=len(b), same_char=int((m.char == m.char_r4).sum()),
                                   same_unit_kind=int(((m.unit == m.unit_r4) & (m.kind == m.kind_r4)).sum()),
                                   post_absdiff_max=round(float((m.post - m.post_r4).abs().max()), 5))
    A = read_csv(work / "cells.csv.gz")
    C0 = read_csv(R4 / "borg_cells.csv.gz", usecols=["book", "page", "idx", "keep", "keep_high"])
    m = A.merge(C0, on=["book", "page", "idx"], suffixes=("", "_r4"))
    rep["keep"] = dict(now=int(A.keep.sum()), r4=int(C0.keep.sum()), ref=REF["keep"],
                       sym_diff=int((m.keep != m.keep_r4).sum()))
    rep["keep_high"] = dict(now=int(A.keep_high.sum()), r4=int(C0.keep_high.sum()), ref=REF["keep_high"],
                            sym_diff=int((m.keep_high != m.keep_high_r4).sum()))
    K5 = read_csv(R5 / "out" / "clean_keep.csv.gz", usecols=["book", "page", "idx", "keep_paddle_ok", "keep_v5"])
    m5 = A.merge(K5, on=["book", "page", "idx"], how="left", suffixes=("", "_r5")).fillna({"keep_v5_r5": 0, "keep_paddle_ok_r5": 0})
    rep["keep_paddle_ok"] = dict(now=int(A.keep_paddle_ok.sum()), r5=int(K5.keep_paddle_ok.sum()), ref=REF["keep_paddle_ok"],
                                 sym_diff=int((m5.keep_paddle_ok != m5.keep_paddle_ok_r5).sum()))
    rep["keep_v5"] = dict(now=int(A.keep_v5.sum()), r5=int(K5.keep_v5.sum()), ref=REF["keep_v5"],
                          sym_diff=int((m5.keep_v5 != m5.keep_v5_r5).sum()))
    P = json.load(open(work / "paddle.json"))
    q06 = json.load(open(R5 / "out" / "q06_borg_slip.json"))["D|borg_keep|mid+delay"]
    rep["theta_paddle_keep"] = dict(now=P["theta"]["keep"], r5=dict(n=q06["n"], theta=q06["theta"], ci=q06["theta_ci"]))
    rep["removed_by_char"] = dict(now=P["removed_by_char"], ref=REF["removed_by_char"])
    vf = work / "validate.json"
    if vf.exists():
        V = json.load(open(vf))
        r4v = json.load(open(R4 / "align" / "validate_q2_v1_final_keep.json"))
        rep["theta_encoder"] = {k: dict(now=dict(n=V[k]["n"], theta=V[k]["theta"]), r4=dict(n=r4v[k]["n"], theta=r4v[k]["theta"]))
                                for k in ("T1_v2|keep", "T2_v2|keep") if k in V and k in r4v}
    ok = (rep["keep"]["sym_diff"] == 0 and rep["keep_high"]["sym_diff"] == 0 and rep["keep_v5"]["sym_diff"] == 0
          and rep["keep_paddle_ok"]["sym_diff"] == 0
          and rep["theta_paddle_keep"]["now"]["n"] == q06["n"] and abs(rep["theta_paddle_keep"]["now"]["theta"] - q06["theta"]) < 1e-4)
    rep["exact_reproduction"] = bool(ok)
    jdump(rep, work / "repro.json")
    log(f"repro: keep {rep['keep']['now']} (Δ {rep['keep']['sym_diff']}), keep_v5 {rep['keep_v5']['now']} (Δ {rep['keep_v5']['sym_diff']}), "
        f"θ {rep['theta_paddle_keep']['now']['theta']} vs {q06['theta']} -> {'TRÙNG' if ok else 'KHÁC'}")


# ------------------------------------------------------------------------------------------------ export
def code_sha() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "pipeline" / "borg_human").glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def st_export(work: Path, out: Path):
    A = read_csv(work / "cells.csv.gz")
    R = read_csv(work / "crops_rec.csv.gz")
    ok1 = R.one_char_ok.astype(bool)
    L = EX.labels_frame(A, R.drop(columns=["one_char_ok"]), ok1)
    # ô keep nào thiếu ảnh -> hạ về mức có ảnh? Không: ghi rõ, invariant bắt.
    out.mkdir(parents=True, exist_ok=True)
    L.to_csv(out / "labels.csv", index=False)
    P = json.load(open(work / "paddle.json"))
    V = json.load(open(work / "validate.json")) if (work / "validate.json").exists() else {}
    rep = json.load(open(work / "repro.json")) if (work / "repro.json").exists() else {}
    chk = json.load(open(work / "det" / "page_map_check.json"))
    info = dict(
        created=datetime.now().isoformat(timespec="seconds"), code_sha256=code_sha(),
        tool="python -m pipeline.borg_human --stage all", params_file="pipeline/borg_human/params.py",
        sources=dict(r4=str(R4.relative_to(REPO)), r5=str(R5.relative_to(REPO)), frozen=str(SRC.relative_to(REPO)),
                     paddle_profile_sha256=PADDLE["sha256"]),
        page_map=dict(n=len(chk), shape_ok=int(sum(c["shape_ok"] for c in chk)), mad_max=max(c["mad"] for c in chk)),
        counts=EX.counts(L), paddle={k: P[k] for k in ("keep", "keep_paddle_flagged", "keep_paddle_ok", "removed_normalised",
                                                        "keep_v5", "removed_by_char", "n_real_tested", "paddle_han_all")},
        units_match=P.get("units_match"),
        theta=dict(paddle=P["theta"], encoder=V,
                   note="θ Paddle trên keep là số chính (bộ đọc độc lập); trên keep_v5 lệch lạc quan vì ô Paddle báo lệch đã bị bỏ; "
                        "θ encoder v2 lệch lạc quan vì v2 tham gia chọn keep."),
        repro=rep, crop_levels=list(CROP_LEVELS),
        size=dict(crops=EX.du(out / "crops"), crops_chuan=EX.du(out / "crops_chuan")),
    )
    info["size"]["total_mb"] = round(sum(v["bytes"] for v in info["size"].values()) / 1e6 + (out / "labels.csv").stat().st_size / 1e6, 1)
    (out / "BUILD_INFO.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "README.md").write_text(EX.readme(info), encoding="utf-8")
    (out / "DATASHEET.md").write_text(EX.datasheet(info), encoding="utf-8")
    EX.write_checksums(out)
    c = info["counts"]["cumulative"]["tong"]
    log(f"export: {len(L)} dòng, keep_v5 {c['keep_v5']}, keep {c['keep']}, keep_high {c['keep_high']}, ảnh {c['with_image']}, "
        f"{info['size']['total_mb']} MB -> {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help=f"all hoặc danh sách con của {STAGES}")
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="auto", help="auto (mps nếu có) | cpu | mps")
    a = ap.parse_args(argv)
    stages = STAGES if a.stage == "all" else [s.strip() for s in a.stage.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"bước không biết: {bad}")
    if a.workers > 3:
        ap.error("≤ 3 worker (máy 16 GB dùng chung)")
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    from .encoders import device
    dev = device(a.device) if any(s in stages for s in ("detect", "units", "glyphs", "validate")) else "cpu"
    t0 = time.time()
    for s in STAGES:
        if s not in stages:
            continue
        log(f"== {s}")
        if s == "detect":
            st_detect(work, dev)
        elif s == "units":
            st_units(work, dev)
        elif s == "glyphs":
            st_glyphs(work, dev)
        elif s == "align":
            st_align(work, a.workers)
        elif s == "keep":
            st_keep(work)
        elif s == "paddle":
            st_paddle(work)
        elif s == "crops":
            st_crops(work, out, a.workers)
        elif s == "validate":
            st_validate(work, out, dev)
        elif s == "repro":
            st_repro(work)
        elif s == "export":
            st_export(work, out)
    log(f"xong {', '.join(stages)} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
