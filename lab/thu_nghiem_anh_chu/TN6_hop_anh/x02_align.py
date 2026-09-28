"""TN6 x02 — thử nghiệm NGOẠI TUYẾN: gióng CHUỖI ÂM QN của trang (thứ tự cột của adapter) vào đơn vị ảnh (hộp detector +
ô ảo, cột phải->trái) bằng DP đơn điệu + phát xạ THỊ GIÁC theo tập ứng viên R(âm) (∪ chữ kim) — KHÔNG dùng chữ người.
Đo: tâm hộp chọn ∈ hộp người của khe (tn6lib), so với hộp pitch hiện hành trên CÙNG tập ô.

  .venv/bin/python lab/thu_nghiem_anh_chu/TN6_hop_anh/x02_align.py --variant base
"""
import argparse, json, pickle, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import tn6lib as T
REPO = T.REPO
X01 = REPO / "measure_out/_tn6/x01"
OUT = REPO / "measure_out/_tn6/x02"; OUT.mkdir(parents=True, exist_ok=True)
SL = {"cat": slice(0, 512), "v1": slice(0, 256), "v2": slice(256, 512)}
BASE = dict(b_real=1.5, b_virt=0.2, b_merge=0.3, skipc=-2.5, skipv=-0.05, skipr0=-0.3, skipr1=-3.0, merge_max=1.7,
            alpha=1.0, enc="v1", kim=0.0, lam_col=0.0, col_margin=0.5, itemnorm=0, emis="R", proto_w=1.0, lam_y=0.0)
_S = {}


def nrm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def _init(P, proto_path):
    D = pickle.load(open(REPO / "measure_out/_borg_human/units/units_v1.pkl", "rb"))
    E = np.load(REPO / "measure_out/_borg_human/units/emb_v1.npy").astype(np.float32)[:, SL[P["enc"]]]
    E = nrm(E)
    GL = pickle.load(open(X01 / "glyph_R.pkl", "rb"))
    FONT = {}
    for c in GL["voc"]:
        vs = [nrm(np.asarray(GL[k][c], np.float32)[SL[P["enc"]]]) for k in ("font", "fd") if c in GL[k]]
        if vs:
            FONT[c] = nrm(np.mean(vs, 0))
    pu = {}
    for i, u in enumerate(D["units"]):
        pu.setdefault(f"{u['book']}/{u['page']}", []).append(i)
    _S.update(U=D["units"], META=D["meta"], E=E, FONT=FONT, pu=pu, SEQ=pickle.load(open(X01 / "seq.pkl", "rb")), P=P,
              PROTO=pickle.load(open(proto_path, "rb")) if proto_path else None)


def emission(key, items, Eu, P):
    """S[u, j]: cos giữa đơn vị u và tham chiếu tốt nhất của mục j (max trên R(âm) ∪ kim; hoặc nguyên mẫu âm)."""
    FONT, PROTO = _S["FONT"], _S["PROTO"]
    fold = _S["fold"][key] if PROTO is not None else None
    M, N = len(Eu), len(items)
    S = np.full((M, N), np.nan, np.float32)
    cache = {}
    for j, it in enumerate(items):
        cands = list(dict.fromkeys(list(it["R"]) + ([it["kim"]] if P["kim"] > 0 and it.get("kim") else [])))
        ck = (tuple(cands), it["syl"])
        if ck not in cache:
            G = [FONT[c] for c in cands if c in FONT]
            s = (Eu @ np.stack(G).T).max(1) if G else np.full(M, np.nan, np.float32)
            if PROTO is not None:
                pv = PROTO.get((fold, it["syl"].lower()))
                if pv is not None:
                    sp = Eu @ pv
                    s = np.where(np.isnan(s), sp, np.maximum(s, P["proto_w"] * sp)) if P["emis"] == "Rproto" else sp
            cache[ck] = s
        S[:, j] = cache[ck]
    # mục không có tham chiếu -> trung bình hàng (trung tính)
    rowm = np.nanmean(S, 1, keepdims=True) if N else np.zeros((M, 1))
    S = np.where(np.isnan(S), rowm, S)
    if P["itemnorm"]:
        S = S - S.mean(0, keepdims=True)
    mu = S.mean(1, keepdims=True); sd = S.std(1, keepdims=True) + 1e-6
    return (S - mu) / sd, S


def dp_align(uu, Z, pitch, P, prior=None):
    """DP đơn điệu (chép hành vi pipeline.borg_human.align.align_page): ghép/ảo/gộp 2 hộp/bỏ hộp/bỏ mục + hậu nghiệm."""
    M, N = Z.shape
    virt = np.array([u["virtual"] for u in uu]); sc = np.array([u["score"] for u in uu]); colid = np.array([u["col"] for u in uu])
    Zp = Z * P["alpha"] + (prior if prior is not None else 0.0)
    mb = np.where(virt == 1, P["b_virt"], P["b_real"])[:, None] + Zp
    sb = np.where(virt == 1, P["skipv"], P["skipr0"] + P["skipr1"] * sc)
    can = np.zeros(M, bool)
    for i in range(M - 1):
        if virt[i] == 0 and virt[i + 1] == 0 and colid[i] == colid[i + 1] and (uu[i + 1]["y2"] - uu[i]["y1"]) <= P["merge_max"] * pitch:
            can[i] = True
    mg = np.full((M, N), -1e9)
    if M > 1:
        mg[:-1] = np.where(can[:-1, None], P["b_merge"] + np.maximum(Zp[:-1], Zp[1:]), -1e9)
    scc = P["skipc"]; NEG = -1e18
    V = np.full((M + 1, N + 1), NEG); V[0] = scc * np.arange(N + 1)
    BP = np.zeros((M + 1, N + 1), np.int8); ar = np.arange(N + 1)
    for i in range(1, M + 1):
        cs = V[i - 1] + sb[i - 1]
        cm = np.full(N + 1, NEG); cm[1:] = V[i - 1, :-1] + mb[i - 1]
        cg = np.full(N + 1, NEG)
        if i >= 2:
            cg[1:] = V[i - 2, :-1] + mg[i - 2]
        Tt = np.maximum(np.maximum(cs, cm), cg)
        tb = np.where(Tt == cm, 1, np.where(Tt == cg, 4, 2))
        row = np.maximum.accumulate(Tt - scc * ar) + scc * ar
        V[i] = row; BP[i] = np.where(row > Tt + 1e-9, 3, tb)
    i, j = M, N; asg = {}
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
        F = np.full((M + 1, N + 1), NEG); F[0] = scc * ar
        for i in range(1, M + 1):
            a = F[i - 1] + sb[i - 1]
            b = np.full(N + 1, NEG); b[1:] = F[i - 1, :-1] + mb[i - 1]
            g = np.full(N + 1, NEG)
            if i >= 2:
                g[1:] = F[i - 2, :-1] + mg[i - 2]
            Tt = np.logaddexp(np.logaddexp(a, b), g)
            F[i] = np.logaddexp.accumulate(Tt - scc * ar) + scc * ar
        return F
    F = fwd(mb, mg, sb)
    Bk = fwd(mb[::-1, ::-1], np.vstack([mg[:-1][::-1, ::-1], mg[-1:]]) if M > 1 else mg, sb[::-1])
    Zt = F[M, N]
    out = []
    for j in range(N):
        a = asg.get(j)
        if a is None:
            out.append(None); continue
        t, i = a
        u = uu[i]
        if t == "g":
            u2 = uu[i + 1]
            box = [min(u["x1"], u2["x1"]), u["y1"], max(u["x2"], u2["x2"]), u2["y2"]]
            p = float(np.exp(F[i, j] + mg[i, j] + Bk[M - i - 2, N - j - 1] - Zt)); kind = "merge"
        else:
            box = [u["x1"], u["y1"], u["x2"], u["y2"]]
            p = float(np.exp(F[i, j] + mb[i, j] + Bk[M - i - 1, N - j - 1] - Zt)); kind = "virtual" if u["virtual"] else "real"
        out.append((kind, i, box, p))
    return out


def col_prior(uu, items, cols, P, wmed):
    """Tiên nghiệm hình học mềm: đơn vị nằm ngoài dải x của cột kim của mục -> −lam_col (thang phát xạ z)."""
    if P["lam_col"] <= 0:
        return None
    cx = np.array([(u["x1"] + u["x2"]) / 2 for u in uu])
    pr = np.zeros((len(uu), len(items)), np.float32)
    for j, it in enumerate(items):
        xr = cols[it["col"]]["x_range"]
        if not xr:
            continue
        m = P["col_margin"] * wmed
        out = (cx < xr[0] - m) | (cx > xr[1] + m)
        pr[out, j] = -P["lam_col"]
    return pr


def run_page(key):
    P = _S["P"]
    idx = _S["pu"].get(key, [])
    sq = _S["SEQ"].get(key)
    if not idx or sq is None or not sq["items"]:
        return key, []
    items = sq["items"]
    # chữ kim của mục: theo thứ tự trong cột (xấp xỉ ghép thứ tự — chỉ dùng khi P['kim'] > 0)
    for it in items:
        k = sq["cols"][it["col"]]["kim"]
        it["kim"] = k[it["si"]] if it["si"] < len(k) else ""
    uu = [_S["U"][i] for i in idx]
    Eu = _S["E"][idx]
    meta = _S["META"][key]
    Z, S = emission(key, items, Eu, P)
    if P["emis"] == "none":
        Z = np.zeros_like(Z)
    pr = col_prior(uu, items, sq["cols"], P, meta["wmed"])
    res = dp_align(uu, Z, meta["pitch"], P, pr)
    rows = []
    for it, r in zip(items, res):
        if r is None:
            rows.append(dict(page=key.split("/")[1], column=it["col"], syl_idx=it["si"], bbox="", kind="skip", post=0.0,
                             unit=-1, J=it["J"]))
        else:
            kind, i, box, p = r
            rows.append(dict(page=key.split("/")[1], column=it["col"], syl_idx=it["si"], bbox=json.dumps([round(v, 1) for v in box]),
                             kind=kind, post=round(p, 4), unit=idx[i], J=it["J"]))
    return key, rows


def run_all(P, keys, workers=3, proto_path=None, fold=None):
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init2, initargs=(P, proto_path, fold)) as pool:
        res = pool.map(run_page, keys, chunksize=4)
    out = {}
    for k, rows in res:
        out[k] = rows
    return out


def _init2(P, proto_path, fold):
    _init(P, proto_path)
    _S["fold"] = fold or {}


def folds(keys, n=5):
    f = {}
    for b in T.BOOKS:
        ks = sorted(k for k in keys if k.startswith(b + "/"))
        for r, k in enumerate(ks):
            f[k] = r * n // len(ks)
    return f


def build_proto(res, P, fold, post_min=0.9, n_min=2, shrink=3.0):
    """Nguyên mẫu (khối f, âm) = trung bình nhúng đơn vị THẬT có post ≥ post_min ở lượt trước, CHỈ từ khối trang KHÁC."""
    E = nrm(np.load(REPO / "measure_out/_borg_human/units/emb_v1.npy").astype(np.float32)[:, SL[P["enc"]]])
    SEQ = pickle.load(open(X01 / "seq.pkl", "rb"))
    sums, cnt = {}, {}
    for k, rows in res.items():
        f = fold[k]
        syls = {(it["col"], it["si"]): it["syl"].lower() for it in SEQ[k]["items"]}
        for r in rows:
            if r["kind"] != "real" or r["post"] < post_min:
                continue
            s = syls[(r["column"], r["syl_idx"])]
            key = (f, s)
            sums[key] = sums.get(key, 0) + E[r["unit"]]
            cnt[key] = cnt.get(key, 0) + 1
    PROTO = {}
    syl_all = {s for (_, s) in sums}
    for f in range(5):
        for s in syl_all:
            n = sum(cnt.get((g, s), 0) for g in range(5) if g != f)
            if n >= n_min:
                v = sum(sums[(g, s)] for g in range(5) if g != f and (g, s) in sums)
                PROTO[(f, s)] = nrm(v / n)
    return PROTO


def evaluate(res, tag):
    out = {}
    for book in T.BOOKS:
        L = pd.DataFrame([r for k, rows in res.items() if k.startswith(book + "/") for r in rows])
        J = T.join(book, L)
        # tập so sánh: ô có trong labels_gated của pitch (cùng (trang, cột, syl_idx))
        Pj = pd.read_pickle(REPO / f"measure_out/_tn6/diag/{book}_pitch_join.pkl")
        Pj["column"] = Pj.column.astype(int); Pj["syl_idx"] = Pj.syl_idx.astype(int)
        J["column"] = J.column.astype(int); J["syl_idx"] = J.syl_idx.astype(int)
        m = J.merge(Pj[["page", "column", "syl_idx", "slot_ok", "tier"]].rename(columns={"slot_ok": "slot_pitch"}),
                    on=["page", "column", "syl_idx"], how="inner")
        k = m[m.has_h & m.keep_level.isin(T.KEEP_OK)]
        g = k[k.tier == "GOLD"]
        out[book] = dict(all_items=T.slot_block(J), same_set_n=int(len(k)), new=round(float(k.slot_ok.mean()), 4),
                         pitch=round(float(k.slot_pitch.astype(bool).mean()), 4),
                         gold_n=int(len(g)), gold_new=round(float(g.slot_ok.mean()), 4),
                         gold_pitch=round(float(g.slot_pitch.astype(bool).mean()), 4),
                         kinds=J.kind.value_counts().to_dict(),
                         post999_slot=round(float(k[k.post >= 0.999].slot_ok.mean()), 4) if (k.post >= 0.999).any() else None,
                         post999_frac=round(float((k.post >= 0.999).mean()), 4))
    print(tag, json.dumps(out, ensure_ascii=False, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--set", nargs="*", default=[], help="k=v ghi đè tham số")
    ap.add_argument("--proto", type=int, default=0, help="số lượt nguyên mẫu âm sau lượt font")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    P = dict(BASE)
    for kv in a.set:
        k, v = kv.split("=")
        P[k] = type(BASE[k])(v) if not isinstance(BASE[k], str) else v
    SEQ = pickle.load(open(X01 / "seq.pkl", "rb"))
    keys = sorted(SEQ)
    t0 = time.time()
    res = run_all(P, keys, a.workers)
    allr = {"pass0": evaluate(res, f"{a.variant}/p0")}
    fold = folds(keys)
    for it in range(a.proto):
        PROTO = build_proto(res, P, fold)
        pp = OUT / f"_proto_{a.variant}.pkl"; pickle.dump(PROTO, open(pp, "wb"))
        P2 = dict(P); P2["emis"] = P["emis"] if P["emis"] in ("Rproto", "proto") else "Rproto"
        res = run_all(P2, keys, a.workers, str(pp), fold)
        allr[f"proto{it + 1}"] = evaluate(res, f"{a.variant}/q{it + 1}")
    pickle.dump(res, open(OUT / f"res_{a.variant}.pkl", "wb"))
    json.dump(dict(P=P, res=allr, sec=round(time.time() - t0)), open(OUT / f"{a.variant}.json", "w"), ensure_ascii=False, indent=1,
              default=str)


if __name__ == "__main__":
    main()
