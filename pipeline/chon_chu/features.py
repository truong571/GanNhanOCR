"""features.py — bảng ỨNG VIÊN (ô × chữ) + đặc trưng — bản chép lab/thu_nghiem_kim/TN8_chon_chu/t05_feats.py + t05c_knn.py.

Ứng viên của ô: R(âm) (Dict/QuocNgu_SinoNom.csv, gold_exact.common.R_of) ∪ {chữ kim} ∪ {chữ lt2 (STT)}.
Đặc trưng (không đặc trưng nào nhìn nhãn người của sách đang chạy):
  f_font, f_fd   crop (MultiEnc) · glyph font / FD         f_head   crop · head v1
  f_self, f_selfk  nguyên mẫu CÙNG SÁCH (trung bình / 3 láng giềng) từ ô NEO (rule s1_inter_s2_direct*, nhãn = kim ∈ R) chỉ ở
                 khối trang KHÁC (5 khối liền theo thứ tự tên trang)
  f_hum, f_humk  nguyên mẫu NGƯỜI Borg (models/chon_chu/hum_<B18|B34>.pt) — sách do config chỉ định (LOBO)
  f_vW, f_vP     CNN kiểm ảnh↔chữ do config chỉ định (bảng W, P của models/gold_exact)
  is_kim, is_lt2, inR, lp_syl (log P(chữ|âm) từ ô neo khối khác), lp_glob, n_self, n_hum
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd

K_NN = 3
CH = 1500


def page_blocks(pages: np.ndarray, k: int = 5) -> np.ndarray:
    """== tn8lib.page_blocks: k khối liền theo thứ tự tên trang."""
    up = np.array(sorted(set(pages)))
    blk = {p: min(k - 1, i * k // max(1, len(up))) for i, p in enumerate(up)}
    return np.array([blk[p] for p in pages])


def rowdot(E, idx_e, M, idx_m, chunk=500000):
    out = np.full(len(idx_e), np.nan, np.float32)
    ok = np.nonzero(idx_m >= 0)[0]
    for s in range(0, len(ok), chunk):
        k = ok[s:s + chunk]
        out[k] = np.einsum("ij,ij->i", E[idx_e[k]], M[idx_m[k]])
    return out


def knn_col(E, ci, cc, cblk, R, Rlab, Rblk, exclude_same_block: bool):
    """top-K cos trung bình của E[ci] với hàng R có nhãn cc (khối ≠ cblk nếu exclude). == t05c_knn.knn_col."""
    out = np.full(len(ci), np.nan, np.float32)
    by = {}
    for j, l in enumerate(Rlab):
        by.setdefault(l, []).append(j)
    by = {l: np.array(v) for l, v in by.items()}
    order = np.argsort(ci, kind="stable")
    ci_s, cc_s, cb_s = ci[order], cc[order], cblk[order]
    cells = np.unique(ci_s)
    for s in range(0, len(cells), CH):
        cs = cells[s:s + CH]
        lo, hi = np.searchsorted(ci_s, cs[0]), np.searchsorted(ci_s, cs[-1], side="right")
        rowmap = {c: r for r, c in enumerate(cs)}
        S = E[cs] @ R.T
        df = pd.DataFrame({"c": cc_s[lo:hi], "r": [rowmap[x] for x in ci_s[lo:hi]], "b": cb_s[lo:hi], "pos": np.arange(lo, hi)})
        keys = ["c", "b"] if exclude_same_block else ["c"]
        for key, g in df.groupby(keys, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            c, b = key[0], (key[1] if exclude_same_block else None)
            idx = by.get(c)
            if idx is None:
                continue
            if exclude_same_block:
                idx = idx[Rblk[idx] != b]
            if len(idx) == 0:
                continue
            sub = S[np.ix_(g.r.values, idx)]
            k = min(K_NN, sub.shape[1])
            top = -np.partition(-sub, k - 1, axis=1)[:, :k] if sub.shape[1] > k else sub
            out[order[g.pos.values]] = top.mean(1)
    return out


def build(D: pd.DataFrame, E: np.ndarray, Z: dict, meta: np.ndarray, lt2: np.ndarray, *, C, scorers, assets,
          verifiers: list[str], human: list[tuple[np.ndarray, np.ndarray]], log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D: bảng ô (page, syllable, ocr_char, label, rule); E: (N,512) MultiEnc; Z: {net: (N,256)}; meta: (N,4) w,h,ink,ok;
    lt2: (N,) chữ lt2 hoặc ''; C = gold_exact.common (sau set_lexicon); scorers = signals_img.Scorers (glyph, head);
    human = [(H (n,512), chars (n,))] đã chọn theo LOBO. Trả (F ứng viên sắp theo (i, c), cells)."""
    t0 = time.time()
    N = len(D)
    kim = D.ocr_char.values.astype(object)
    syl = D.syllable.values.astype(object)
    Rm = {s: C.R_of(s) for s in set(syl)}
    blk = page_blocks(D.page.values, 5)
    lab = D.label.values.astype(object)
    anc = np.array([str(r).startswith("s1_inter_s2_direct") for r in D.rule.values]) & (lab != "") & (lab == kim)
    anc &= np.array([l in Rm[s] for l, s in zip(lab, syl)])
    anc &= meta[:, 3] > 0
    ci, cc = [], []
    for i, (s, k, l2) in enumerate(zip(syl, kim, lt2)):
        cs = set(Rm[s])
        if k and len(k) == 1:
            cs.add(k)
        if l2 and len(l2) == 1:
            cs.add(l2)
        for c in sorted(cs):
            ci.append(i); cc.append(c)
    ci = np.array(ci, np.int64); cc = np.array(cc, dtype=object)
    chars = sorted(set(cc))
    cid = {c: j for j, c in enumerate(chars)}
    cj = np.array([cid[c] for c in cc], np.int64)
    log(f"[chon_chu] {N:,} ô, {len(ci):,} cặp ứng viên ({len(ci) / max(1, N):.1f}/ô), {len(chars):,} chữ, neo {int(anc.sum()):,}")
    F = pd.DataFrame({"i": ci, "c": cc})
    # ---- glyph font / FD + head v1
    FE, _, DE, _ = scorers.glyph_emb(chars, with_aug=False, with_fd=True)
    Mf = np.zeros((len(chars), 512), np.float32); jf = np.full(len(chars), -1)
    Md = np.zeros((len(chars), 512), np.float32); jd = np.full(len(chars), -1)
    for c, j in cid.items():
        if c in FE:
            Mf[j] = FE[c]; jf[j] = j
        if c in DE:
            Md[j] = DE[c]; jd[j] = j
    F["f_font"] = rowdot(E, ci, Mf, jf[cj])
    F["f_fd"] = rowdot(E, ci, Md, jd[cj])
    enc = scorers.enc()
    Wn = enc.Wn.astype(np.float32)
    jh = np.array([enc.lab2idx.get(c, -1) for c in chars])
    F["f_head"] = rowdot(E, ci, Wn, jh[cj])
    # ---- nguyên mẫu cùng sách (khối khác) + tiên nghiệm
    ai = np.nonzero(anc)[0]
    alab = lab[ai]; ablk = blk[ai]; asyl = np.array([C.R_key(s) for s in syl[ai]], dtype=object)
    tot = defaultdict(lambda: np.zeros(512, np.float64)); totn = defaultdict(int)
    bs = defaultdict(lambda: np.zeros(512, np.float64)); bn = defaultdict(int)
    for i, l, k in zip(ai, alab, ablk):
        tot[l] += E[i]; totn[l] += 1; bs[(k, l)] += E[i]; bn[(k, l)] += 1
    sy_tot = defaultdict(int); sy_blk = defaultdict(int); syc_tot = defaultdict(int); syc_blk = defaultdict(int)
    for l, k, s in zip(alab, ablk, asyl):
        sy_tot[s] += 1; sy_blk[(k, s)] += 1; syc_tot[(s, l)] += 1; syc_blk[(k, s, l)] += 1
    keys = sorted(set(zip(blk[ci].tolist(), cc.tolist())))
    kid = {k: j for j, k in enumerate(keys)}
    Ps = np.zeros((len(keys), 512), np.float32); js = np.full(len(keys), -1); ns = np.zeros(len(keys), np.int32)
    for (k, c), j in kid.items():
        n = totn.get(c, 0) - bn.get((k, c), 0)
        ns[j] = n
        if n >= 2:
            v = tot[c] - bs.get((k, c), 0)
            Ps[j] = v / max(np.linalg.norm(v), 1e-9); js[j] = j
    kk = np.array([kid[(k, c)] for k, c in zip(blk[ci].tolist(), cc.tolist())])
    F["f_self"] = rowdot(E, ci, Ps, js[kk])
    F["n_self"] = ns[kk]
    rk = np.array([C.R_key(s) for s in syl], dtype=object)
    nsy = np.array([sy_tot.get(rk[i], 0) - sy_blk.get((blk[i], rk[i]), 0) for i in ci])
    nsc = np.array([syc_tot.get((rk[i], c), 0) - syc_blk.get((blk[i], rk[i], c), 0) for i, c in zip(ci, cc)])
    nR = np.array([max(1, len(Rm[s])) for s in syl])[ci]
    F["lp_syl"] = np.log((nsc + 0.5) / (nsy + 0.5 * nR))
    F["n_syl_c"] = nsc
    F["lp_glob"] = np.log1p(ns[kk])
    # ---- nguyên mẫu người (LOBO)
    if human:
        hs = defaultdict(lambda: np.zeros(512, np.float64)); hn = defaultdict(int)
        for H, hc in human:
            for c, v in zip(hc, H):
                hs[c] += v; hn[c] += 1
        Ph = np.zeros((len(chars), 512), np.float32); jh2 = np.full(len(chars), -1); nh2 = np.zeros(len(chars), np.int32)
        for c, j in cid.items():
            nh2[j] = hn.get(c, 0)
            if hn.get(c, 0) >= 2:
                v = hs[c]; Ph[j] = v / max(np.linalg.norm(v), 1e-9); jh2[j] = j
        F["f_hum"] = rowdot(E, ci, Ph, jh2[cj]); F["n_hum"] = nh2[cj]
    else:
        F["f_hum"] = np.nan; F["n_hum"] = 0
    # ---- CNN kiểm ảnh↔chữ
    U = assets.load("vft_universe.pt")["U"]
    uid = {c: j for j, c in enumerate(U)}
    ju = np.array([uid.get(c, -1) for c in chars])
    vW, vP = [], []
    for net in verifiers:
        tb = assets.load(net.replace("vft_", "vft_tables_").replace("hand_", "hand_tables_") + ".pt")
        W = np.asarray(tb["W"], np.float32); P = np.asarray(tb["P"], np.float32); nh = np.asarray(tb["nh"])
        vW.append(rowdot(Z[net], ci, W, ju[cj]))
        jp = np.where((ju >= 0) & (nh[np.maximum(ju, 0)] >= 3), ju, -1)
        vP.append(rowdot(Z[net], ci, P, jp[cj]))
    import warnings
    with warnings.catch_warnings():            # nanmean của hàng toàn NaN (chữ ngoài vũ trụ U) -> NaN, không cảnh báo
        warnings.simplefilter("ignore", RuntimeWarning)
        F["f_vW"] = np.nanmean(np.stack(vW), 0) if len(vW) > 1 else vW[0]
        F["f_vP"] = np.nanmean(np.stack(vP), 0) if len(vP) > 1 else vP[0]
    # ---- cờ kim / lt2 / R
    veq = C.var_eq_plus
    F["is_kim"] = np.array([bool(kim[i]) and (c == kim[i] or veq(c, kim[i])) for i, c in zip(ci, cc)], np.int8)
    F["is_lt2"] = np.array([bool(lt2[i]) and (c == lt2[i] or veq(c, lt2[i])) for i, c in zip(ci, cc)], np.int8)
    F["inR"] = np.array([c in Rm[s] for c, s in zip(cc, syl[ci])], np.int8)
    # ---- láng giềng gần (t05c)
    if human:
        H = np.concatenate([h for h, _ in human]); Hl = np.concatenate([np.asarray(c, dtype=object) for _, c in human])
        F["f_humk"] = knn_col(E, ci, cc, blk[ci], H, Hl, np.zeros(len(Hl), int), False)
    else:
        F["f_humk"] = np.nan
    aik = np.nonzero(anc)[0]
    F["f_selfk"] = knn_col(E, ci, cc, blk[ci], E[aik], lab[aik].astype(object), blk[aik], True)
    for col in F.columns:
        if F[col].dtype == np.float64:
            F[col] = F[col].astype(np.float32)
    F = F.sort_values(["i", "c"], kind="stable").reset_index(drop=True)
    cells = pd.DataFrame(dict(anchor=anc, blk=blk, crop_w=meta[:, 0], crop_h=meta[:, 1], ink=meta[:, 2], crop_ok=meta[:, 3],
                              nR=[len(Rm[s]) for s in syl], lt2=lt2))
    log(f"[chon_chu] đặc trưng xong ({time.time() - t0:.0f}s)")
    return F, cells
