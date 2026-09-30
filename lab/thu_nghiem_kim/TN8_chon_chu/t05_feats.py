"""t05_feats.py — bảng ỨNG VIÊN (ô × chữ) cho bộ chọn chữ bằng ảnh. 0 API; đọc nhúng t02/t04 (+ glyph font/FD).

Ứng viên của ô: R(âm) (Dict/QuocNgu_SinoNom, như tier_v3) ∪ {chữ kim} ∪ {chữ lt2 (STT)}.
Đặc trưng (cos trên nhúng; không đặc trưng nào nhìn nhãn người của CHÍNH sách đang chấm):
  f_font, f_fd   crop (MultiEnc v1+v2) · glyph font / FD của chữ
  f_head         crop · trọng số head v1 (nom-embed, 1.591 lớp)
  f_self         crop · nguyên mẫu CÙNG SÁCH = trung bình nhúng ô "neo" (kim ∈ R(âm), nhãn = kim — tự động, không người)
                 có nhãn = chữ, CHỈ từ 4 khối trang KHÁC (5 khối liền) -> không tự khẳng định; n_self
  f_hum          crop · nguyên mẫu NGƯỜI Borg của SÁCH KIA (LOBO: B18 <- B34, B34 <- B18; STT <- cả hai); n_hum
  f_vW, f_vP     CNN kiểm ảnh↔chữ SẠCH với bộ đang chấm (B18: hand_*_DungLy; B34/STT: hand_*_Kinh; L16: vft_T; TK: vft_L;
                 Chr/L83/KVK: trung bình vft_T, vft_L): z·W[chữ], z·P[chữ] (P chỉ khi nh ≥ 3)
  is_kim, is_lt2, inR, lp_syl (tiên nghiệm chữ|âm từ ô neo khối khác), lp_glob (log1p số ô neo của chữ, khối khác)
Ra: measure_out/_tn8/cand/<bộ>.pkl (hàng = cặp ô×ứng viên) + cand/<bộ>_cells.pkl (thông tin ô).
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402

EMB = T.OUT / "emb"
CAND = T.OUT / "cand"
VER = {"B18": ["hand_T_DungLy", "hand_L_DungLy"], "B34": ["hand_T_Kinh", "hand_L_Kinh"],
       "stt2": ["hand_T_Kinh", "hand_L_Kinh"], "stt4": ["hand_T_Kinh", "hand_L_Kinh"], "stt11": ["hand_T_Kinh", "hand_L_Kinh"],
       "L16": ["vft_T"], "TK": ["vft_L"], "Chr": ["vft_T", "vft_L"], "L83": ["vft_T", "vft_L"], "KVK": ["vft_T", "vft_L"]}
HUM = {"B18": ["B34"], "B34": ["B18"], "stt2": ["B18", "B34"], "stt4": ["B18", "B34"], "stt11": ["B18", "B34"]}
ANCHOR_RULES = ("s1_inter_s2_direct",)       # tiền tố: direct, direct_lowp, direct_am_sua_dau


def tables(name):
    tb = torch.load(T.REPO / "models/gold_exact" / (name.replace("vft_", "vft_tables_").replace("hand_", "hand_tables_") + ".pt"),
                    map_location="cpu", weights_only=False)
    return np.asarray(tb["W"], np.float32), np.asarray(tb["P"], np.float32), np.asarray(tb["nh"])


def norm_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-9)


def rowdot(E, idx_e, M, idx_m, chunk=500000):
    """s[k] = E[idx_e[k]] · M[idx_m[k]] (idx_m < 0 -> nan)."""
    out = np.full(len(idx_e), np.nan, np.float32)
    ok = np.nonzero(idx_m >= 0)[0]
    for s in range(0, len(ok), chunk):
        k = ok[s:s + chunk]
        out[k] = np.einsum("ij,ij->i", E[idx_e[k]], M[idx_m[k]])
    return out


def build(b: str):
    t0 = time.time()
    C = T.lex()
    D = T.load_base(b)
    N = len(D)
    E = np.load(EMB / f"{b}_enc.npy").astype(np.float32)
    meta = np.load(EMB / f"{b}_crop_meta.npy")
    kim = D.ocr_char.values
    lt2 = np.array([""] * N, dtype=object)
    if b.startswith("stt"):
        L2 = pd.read_pickle(T.OUT / "stt_lt2" / f"{b}.pkl")
        lt2 = L2.lt2.fillna("").values.astype(object)
    Rm = {s: C.R_of(s) for s in set(D.syllable)}
    blk = T.page_blocks(D.page.values, 5)
    # ---- ô neo (tự động): kim ∈ R(âm), nhãn = kim
    anc = np.array((D.rule.str.startswith(ANCHOR_RULES[0]) & (D.label != "") & (D.label == D.ocr_char)).values, bool)
    anc &= np.array([l in Rm[s] for l, s in zip(D.label, D.syllable)])
    anc &= meta[:, 3] > 0
    # ---- ứng viên
    ci, cc = [], []
    for i, (s, k, l2) in enumerate(zip(D.syllable, kim, lt2)):
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
    print(f"[t05] {b}: {N} ô, {len(ci)} cặp ứng viên ({len(ci) / N:.1f}/ô), {len(chars)} chữ, neo {int(anc.sum())}", flush=True)
    F = pd.DataFrame({"i": ci, "c": cc})
    # ---- glyph font / FD + head v1
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    A = Assets()
    S = SI.Scorers(A, "mps" if torch.backends.mps.is_available() else "cpu", EMB, False, lambda m: None)
    FE, _, DE, _ = S.glyph_emb(chars, with_aug=False, with_fd=True)
    Mf = np.zeros((len(chars), 512), np.float32); jf = np.full(len(chars), -1)
    Md = np.zeros((len(chars), 512), np.float32); jd = np.full(len(chars), -1)
    for c, j in cid.items():
        if c in FE:
            Mf[j] = FE[c]; jf[j] = j
        if c in DE:
            Md[j] = DE[c]; jd[j] = j
    F["f_font"] = rowdot(E, ci, Mf, jf[cj])
    F["f_fd"] = rowdot(E, ci, Md, jd[cj])
    enc = S.enc()
    Wn = enc.Wn.astype(np.float32)
    jh = np.array([enc.lab2idx.get(c, -1) for c in chars])
    F["f_head"] = rowdot(E, ci, Wn, jh[cj])
    # ---- nguyên mẫu cùng sách (khối khác) + tiên nghiệm
    ai = np.nonzero(anc)[0]
    alab = D.label.values[ai]; ablk = blk[ai]; asyl = np.array([C.R_key(s) for s in D.syllable.values[ai]], dtype=object)
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
    rk = np.array([C.R_key(s) for s in D.syllable.values], dtype=object)
    nsy = np.array([sy_tot.get(rk[i], 0) - sy_blk.get((blk[i], rk[i]), 0) for i in ci])
    nsc = np.array([syc_tot.get((rk[i], c), 0) - syc_blk.get((blk[i], rk[i], c), 0) for i, c in zip(ci, cc)])
    nR = np.array([max(1, len(Rm[s])) for s in D.syllable.values])[ci]
    F["lp_syl"] = np.log((nsc + 0.5) / (nsy + 0.5 * nR))
    F["n_syl_c"] = nsc
    F["lp_glob"] = np.log1p(ns[kk])
    # ---- nguyên mẫu người (LOBO)
    if b in HUM:
        hs = defaultdict(lambda: np.zeros(512, np.float64)); hn = defaultdict(int)
        for hb in HUM[b]:
            Mh = pd.read_pickle(EMB / f"hum_{hb}_meta.pkl"); Eh = np.load(EMB / f"hum_{hb}_enc.npy").astype(np.float32)
            for c, v in zip(Mh.char.values, Eh):
                hs[c] += v; hn[c] += 1
        Ph = np.zeros((len(chars), 512), np.float32); jh2 = np.full(len(chars), -1); nh2 = np.zeros(len(chars), np.int32)
        for c, j in cid.items():
            nh2[j] = hn.get(c, 0)
            if hn.get(c, 0) >= 2:
                v = hs[c]; Ph[j] = v / max(np.linalg.norm(v), 1e-9); jh2[j] = j
        F["f_hum"] = rowdot(E, ci, Ph, jh2[cj]); F["n_hum"] = nh2[cj]
    else:
        F["f_hum"] = np.nan; F["n_hum"] = 0
    # ---- CNN kiểm ảnh↔chữ (sạch với bộ này)
    U = torch.load(T.REPO / "models/gold_exact/vft_universe.pt", weights_only=False)["U"]
    uid = {c: j for j, c in enumerate(U)}
    ju = np.array([uid.get(c, -1) for c in chars])
    vW, vP = [], []
    for net in VER[b]:
        Z = np.load(EMB / f"{b}_{net}.npy").astype(np.float32)
        W, P, nh = tables(net)
        vW.append(rowdot(Z, ci, W, ju[cj]))
        jp = np.where((ju >= 0) & (nh[np.maximum(ju, 0)] >= 3), ju, -1)
        vP.append(rowdot(Z, ci, P, jp[cj]))
    F["f_vW"] = np.nanmean(np.stack(vW), 0) if len(vW) > 1 else vW[0]
    with np.errstate(all="ignore"):
        F["f_vP"] = np.nanmean(np.stack(vP), 0) if len(vP) > 1 else vP[0]
    # ---- kim / lt2 / R
    veq = C.var_eq_plus
    F["is_kim"] = np.array([bool(kim[i]) and (c == kim[i] or veq(c, kim[i])) for i, c in zip(ci, cc)], np.int8)
    F["is_lt2"] = np.array([bool(lt2[i]) and (c == lt2[i] or veq(c, lt2[i])) for i, c in zip(ci, cc)], np.int8)
    F["inR"] = np.array([c in Rm[s] for c, s in zip(cc, D.syllable.values[ci])], np.int8)
    # ---- sự thật (chỉ để ĐO / học LOBO)
    gt = D.gt_char.values
    F["y"] = np.array([bool(gt[i]) and (c == gt[i] or veq(c, gt[i])) for i, c in zip(ci, cc)], np.int8)
    ref = D.ref.values
    F["y_ref"] = np.array([any(r and (c == r or veq(c, r)) for r in ref[i].split("|")) if ref[i] else False
                           for i, c in zip(ci, cc)], np.int8)
    CAND.mkdir(parents=True, exist_ok=True)
    for col in F.columns:
        if F[col].dtype == np.float64:
            F[col] = F[col].astype(np.float32)
    F.to_pickle(CAND / f"{b}.pkl")
    cells = pd.DataFrame(dict(anchor=anc, blk=blk, crop_w=meta[:, 0], crop_h=meta[:, 1], ink=meta[:, 2], crop_ok=meta[:, 3],
                              nR=[len(Rm[s]) for s in D.syllable.values], lt2=lt2))
    cells.to_pickle(CAND / f"{b}_cells.pkl")
    print(f"[t05] {b}: xong [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    for b in (sys.argv[1:] or T.ORDER):
        build(b)
