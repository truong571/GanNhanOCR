"""align.py — GIÓNG chuỗi chữ NGƯỜI (nom_clean, chữ Lo) của trang với chuỗi đơn vị (hộp detector + ô ảo),
thứ tự cột phải->trái, trên->dưới: DP đơn điệu (Viterbi) + forward-backward (hậu nghiệm ghép).

Bản chép NGUYÊN hành vi r4/borg_align/scripts/a05_align.py (tham số params.DP / params.PROTO):
  chuyển tiếp : ghép hộp thật / ô ảo / gộp 2 hộp liền cùng cột với 1 chữ; bỏ hộp; bỏ chữ
  phát xạ     : z(i,j) = cos chuẩn hoá theo hàng giữa emb(đơn vị i) và tham chiếu của chữ c_j
                --emis font : glyph font/FD (độc lập với căn chỉnh)
                --emis proto: nguyên mẫu = trung bình emb các ô post>=0,9 của lượt trước, CHỈ từ các khối trang KHÁC
                              (5 khối trang liền nhau mỗi sách) + trộn glyph font (w = n/(n+3))
Mỗi worker nạp trạng thái từ tệp (spawn-an-toàn); ≤ 3 worker.
"""
from __future__ import annotations

import json
import pickle
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from .geom import human_seq
from .params import BOOKS, DP as P, PROTO as PR

SL = {"cat": slice(0, 512), "v1": slice(0, 256), "v2": slice(256, 512)}
_S = {}


def nrm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def load_state(work: Path, enc: str):
    D = pickle.load(open(work / "units" / "units_v1.pkl", "rb"))
    U, META = D["units"], D["meta"]
    E = np.load(work / "units" / "emb_v1.npy").astype(np.float32)
    E = np.ascontiguousarray(E[:, SL[enc]])
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    GL = pickle.load(open(work / "units" / "glyph_emb.pkl", "rb"))
    FONT = {}
    for c in GL["voc"]:
        vs = [np.asarray(GL[k][c], np.float32)[SL[enc]] for k in ("font", "fd") if c in GL[k]]
        if vs:
            FONT[c] = nrm(np.mean([nrm(np.asarray(v, np.float32)) for v in vs], 0))
    page_units = {}
    for i, u in enumerate(U):
        page_units.setdefault(f"{u['book']}/{u['page']}", []).append(i)
    keys = sorted(page_units)
    fold = {}
    for book in BOOKS:
        ks = [k for k in keys if k.startswith(book + "/")]
        for r, k in enumerate(ks):
            fold[k] = r * PR["n_folds"] // len(ks)
    return dict(U=U, META=META, E=E, FONT=FONT, page_units=page_units, keys=keys, fold=fold)


def build_proto(st: dict, prev: pd.DataFrame) -> dict:
    """Nguyên mẫu (khối f, chữ c) từ ô real post>=post_min của lượt trước, chỉ lấy khối g != f (a05 --emis proto)."""
    E, FONT, fold = st["E"], st["FONT"], st["fold"]
    prev = prev[(prev.post >= PR["post_min"]) & (prev.kind == "real")]
    pf = [fold[f"{b}/{p}"] for b, p in zip(prev.book, prev.page)]
    sums, cnt = {}, {}
    for f, c, ui in zip(pf, prev.char, prev.unit):
        sums.setdefault((f, c), np.zeros(E.shape[1], np.float32))
        sums[(f, c)] += E[ui]
        cnt[(f, c)] = cnt.get((f, c), 0) + 1
    PROTO = {}
    chars = set(c for (_, c) in sums)
    for f in range(PR["n_folds"]):
        for c in chars:
            s = sum((sums[(g, c)] for g in range(PR["n_folds"]) if g != f and (g, c) in sums), np.zeros(E.shape[1], np.float32))
            n = sum(cnt.get((g, c), 0) for g in range(PR["n_folds"]) if g != f)
            if n >= PR["n_min"]:
                w = n / (n + PR["shrink"])
                base = FONT.get(c)
                v = nrm(s / n)
                PROTO[(f, c)] = nrm(w * v + (1 - w) * base) if base is not None else v
    return PROTO


def _init(work, enc, emis, proto_path, kim_path="", kim_beta=0.0):
    st = load_state(Path(work), enc)
    st["emis"] = emis
    st["PROTO"] = pickle.load(open(proto_path, "rb")) if proto_path else None
    st["KIM"] = pickle.load(open(kim_path, "rb")) if kim_path and kim_beta else None   # TN10: mốc kim (tắt mặc định)
    st["kim_beta"] = float(kim_beta or 0.0)
    _S.update(st)


def _ref(c, f):
    PROTO, FONT = _S["PROTO"], _S["FONT"]
    if PROTO is not None and (f, c) in PROTO:
        return PROTO[(f, c)]
    return FONT.get(c)


def align_page(key):
    U, E, META = _S["U"], _S["E"], _S["META"]
    book, pn = key.split("/")
    idx = _S["page_units"][key]
    M = len(idx)
    chars, syls, sid, dot = human_seq(book, pn)
    N = len(chars)
    uu = [U[i] for i in idx]
    f = _S["fold"][key]
    pitch = META[key]["pitch"]
    Eu = E[idx]
    if _S["emis"] == "none":
        Z = np.zeros((M, N))
        S = np.zeros((M, N))
    else:
        dc = sorted(set(chars))
        R = []
        for c in dc:
            r = _ref(c, f)
            R.append(r if r is not None else np.zeros(E.shape[1], np.float32))
        R = np.stack(R)
        S0 = Eu @ R.T
        mu = S0.mean(1, keepdims=True)
        sd = S0.std(1, keepdims=True) + 1e-6
        Z0 = (S0 - mu) / sd
        col = {c: k for k, c in enumerate(dc)}
        jj = np.array([col[c] for c in chars])
        Z = Z0[:, jj] * P["alpha"]
        S = S0[:, jj]
    if _S.get("KIM") and _S.get("kim_beta"):
        Z = np.array(Z, dtype=float, copy=True)
        for i, j in _S["KIM"].get(key, ()):   # TN10: kim đọc đúng chữ người j tại đơn vị i -> cộng beta (mốc vị trí)
            Z[i, j] += _S["kim_beta"]
    virt = np.array([u["virtual"] for u in uu])
    sc = np.array([u["score"] for u in uu])
    colid = np.array([u["col"] for u in uu])
    mb = np.where(virt == 1, P["b_virt"], P["b_real"])[:, None] + Z
    sb = np.where(virt == 1, P["skipv"], P["skipr0"] + P["skipr1"] * sc)
    can = np.zeros(M, bool)
    for i in range(M - 1):
        if virt[i] == 0 and virt[i + 1] == 0 and colid[i] == colid[i + 1] and \
                (uu[i + 1]["y2"] - uu[i]["y1"]) <= P["merge_max"] * pitch:
            can[i] = True
    mg = np.full((M, N), -1e9)
    if M > 1:
        mg[:-1] = np.where(can[:-1, None], P["b_merge"] + np.maximum(Z[:-1], Z[1:]), -1e9)
    scc = P["skipc"]
    NEG = -1e18
    V = np.full((M + 1, N + 1), NEG)
    V[0] = scc * np.arange(N + 1)
    BP = np.zeros((M + 1, N + 1), np.int8)   # 1 ghép, 2 bỏ hộp, 3 bỏ chữ, 4 gộp
    ar = np.arange(N + 1)
    for i in range(1, M + 1):
        cand_skip = V[i - 1] + sb[i - 1]
        cand_m = np.full(N + 1, NEG)
        cand_m[1:] = V[i - 1, :-1] + mb[i - 1]
        cand_g = np.full(N + 1, NEG)
        if i >= 2:
            cand_g[1:] = V[i - 2, :-1] + mg[i - 2]
        T = np.maximum(np.maximum(cand_skip, cand_m), cand_g)
        tb = np.where(T == cand_m, 1, np.where(T == cand_g, 4, 2))
        A = T - scc * ar
        cm = np.maximum.accumulate(A)
        row = cm + scc * ar
        V[i] = row
        BP[i] = np.where(row > T + 1e-9, 3, tb)
    i, j = M, N
    asg = {}
    while i > 0 or j > 0:
        t = BP[i, j] if i > 0 else 3
        if t == 1:
            asg[j - 1] = ("m", i - 1); i -= 1; j -= 1
        elif t == 4:
            asg[j - 1] = ("g", i - 2); i -= 2; j -= 1
        elif t == 2:
            i -= 1
        else:
            j -= 1

    def fwd(mb, mg, sb):
        F = np.full((M + 1, N + 1), NEG)
        F[0] = scc * ar
        for i in range(1, M + 1):
            a = F[i - 1] + sb[i - 1]
            b = np.full(N + 1, NEG)
            b[1:] = F[i - 1, :-1] + mb[i - 1]
            g = np.full(N + 1, NEG)
            if i >= 2:
                g[1:] = F[i - 2, :-1] + mg[i - 2]
            T = np.logaddexp(np.logaddexp(a, b), g)
            F[i] = np.logaddexp.accumulate(T - scc * ar) + scc * ar
        return F
    F = fwd(mb, mg, sb)
    Bk = fwd(mb[::-1, ::-1], np.vstack([mg[:-1][::-1, ::-1], mg[-1:]]) if M > 1 else mg, sb[::-1])
    Zt = F[M, N]

    def post_m(i, j):
        return float(np.exp(F[i, j] + mb[i, j] + Bk[M - i - 1, N - j - 1] - Zt))

    def post_g(i, j):
        return float(np.exp(F[i, j] + mg[i, j] + Bk[M - i - 2, N - j - 1] - Zt))
    rows = []
    for j in range(N):
        a = asg.get(j)
        r = dict(book=book, page=pn, idx=j, char=chars[j], syllable=syls[j], sent=sid[j], fold=f)
        if a is None:
            r.update(kind="skip", unit=-1, col=-1, row=-1, x1=0, y1=0, x2=0, y2=0, det_score=0, z=np.nan, s=np.nan, post=0.0,
                     zprev=np.nan, znext=np.nan)
        else:
            t, i = a
            ui = idx[i]
            u = U[ui]
            if t == "g":
                u2 = U[idx[i + 1]]
                x1, y1, x2, y2 = min(u["x1"], u2["x1"]), u["y1"], max(u["x2"], u2["x2"]), u2["y2"]
                kind = "merge"; p = post_g(i, j); zz = max(Z[i, j], Z[i + 1, j])
            else:
                x1, y1, x2, y2 = u["x1"], u["y1"], u["x2"], u["y2"]
                kind = "virtual" if u["virtual"] else "real"; p = post_m(i, j); zz = Z[i, j]
            r.update(kind=kind, unit=ui, col=u["col"], row=u["row"], x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1),
                     y2=round(y2, 1), det_score=u["score"], z=round(float(zz), 3), s=round(float(S[i, j]), 4), post=round(p, 4),
                     zprev=round(float(Z[i, j - 1]), 3) if j > 0 else np.nan,
                     znext=round(float(Z[i, j + 1]), 3) if j + 1 < N else np.nan)
        rows.append(r)
    nsb = M - sum(1 if a[0] == "m" else 2 for a in asg.values())
    info = dict(key=key, M=M, N=N, n_real=int((virt == 0).sum()), matched=len(asg), skip_char=N - len(asg),
                skip_unit=nsb, logZ=float(Zt), vit=float(V[M, N]), pitch=pitch)
    return rows, info


def run_pass(work: Path, tag: str, emis: str, enc: str, prev: pd.DataFrame | None, workers: int = 3, log=print,
             kim_path: str = "", kim_beta: float = 0.0):
    """Một lượt căn toàn bộ 641 trang -> DataFrame (mỗi chữ người 1 dòng, thứ tự trang sắp xếp, rồi j) + info trang."""
    t0 = time.time()
    proto_path = ""
    if emis == "proto":
        st = load_state(work, enc)
        PROTO = build_proto(st, prev)
        proto_path = str(work / "align" / f"_proto_{tag}.pkl")
        pickle.dump(PROTO, open(proto_path, "wb"))
        keys = st["keys"]
        log(f"  {tag}: nguyên mẫu {len(PROTO)} cặp (khối, chữ)")
        del st
    else:
        D = pickle.load(open(work / "units" / "units_v1.pkl", "rb"))
        keys = sorted({f"{u['book']}/{u['page']}" for u in D["units"]})
        del D
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(str(work), enc, emis, proto_path, kim_path, kim_beta)) as pool:
        res = pool.map(align_page, keys, chunksize=4)
    if proto_path:
        Path(proto_path).unlink(missing_ok=True)
    df = pd.DataFrame([r for rr, _ in res for r in rr])
    infos = [i for _, i in res]
    (work / "align").mkdir(parents=True, exist_ok=True)
    df.to_csv(work / "align" / f"{tag}.csv.gz", index=False)
    json.dump(infos, open(work / "align" / f"{tag}_pages.json", "w"))
    k = df.kind.value_counts().to_dict()
    log(f"  {tag}: {len(df)} ô {k} post>=0.999 {int((df.post >= 0.999).sum())} ({time.time() - t0:.0f} s)")
    return df
