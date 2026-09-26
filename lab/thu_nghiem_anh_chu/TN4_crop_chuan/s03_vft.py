#!/usr/bin/env python
"""s03_vft.py — điểm bộ kiểm ảnh verifier_ft (p_wood_T, p_wood_L) trên CROP CHUẨN, bộ kiểm GIỮ NGUYÊN (CNN, nguyên mẫu W,
nguyên mẫu crop người P, logistic 'wood' hiệu chuẩn trên đặc trưng crop cũ của sách tune phần B — như r4/verifier_ft/s02).
Chỉ đổi ẢNH ĐẦU VÀO: khung vuông crop chuẩn (TN4/crops, TN4/aux) qua CÙNG tiền xử lý 64×64 (p01_prep.prep).
MPS cho CNN (vài chục giây), còn lại CPU. 0 API, repo chỉ đọc.
Invariant: (1) nhúng lại ảnh IHR cũ (hand_lobo/out/img_ihr.npy = p01 prep, thứ tự meta_ihr) trùng emb_<T>_ihr đã lưu;
           (2) logistic tái lập đúng p_wood_T/L cũ của gold_scores.csv (sai ≤ 1e-4).
Ra: measure_out/_thu_nghiem_anh_chu/TN4/vft_{gold,ihr}.pkl, lab/.../s03_vft.json
"""
import json, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
VF = SP / 'r4/verifier_ft'; OUTV = VF / 'out'
sys.path.insert(0, str(HERE))
from ver import BIG, LABOUT  # noqa
sys.path.insert(0, str(VF)); sys.path.insert(0, str(SP / 'kim_bottleneck/harness'))
from vflib import Net, to_t  # noqa
from harness_lib import R_of, variants_of, simp  # noqa
T0 = time.time()
INV = {}
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
TUNE = {'T': 'TruyenKieu1872', 'L': 'LucVanTien1916'}


def log(s):
    print(f'[{time.time()-T0:5.0f}s] {s}', flush=True)


import os  # noqa
FRAME = os.environ.get('TN4_FRAME', 'sq')      # 'sq' = khung vuông lề 10 % (chính); 'tight' = ĐỘ NHẠY hậu kiểm: hộp mực + 4 px
SUF = '' if FRAME == 'sq' else '_' + FRAME


def reframe_tight(gray, m=4):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(gray < t)
    if len(ys) < 8:
        return gray
    return gray[max(0, ys.min() - m):ys.max() + 1 + m, max(0, xs.min() - m):xs.max() + 1 + m]


def prep(gray, SZ=64):   # == verifier_ft/p01_prep.prep
    if FRAME == 'tight' and gray is not None and gray.size:
        gray = reframe_tight(gray)
    if gray is None or gray.size == 0:
        gray = np.full((8, 8), 255, np.uint8)
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    if hi - lo > 10:
        gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    h, w = gray.shape
    s = max(h, w)
    can = np.full((s, s), 255, np.uint8)
    can[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = gray
    return cv2.resize(can, (SZ, SZ), interpolation=cv2.INTER_AREA)


def uid_path(uid):
    a = uid.split('/')
    return '/'.join(a[:3]) + '/' + '_'.join(a[3:]) + '.png'


def load_imgs(paths):
    def one(p):
        return prep(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p else None)
    with ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(one, paths, chunksize=256)))


def load_net(TAG):
    ck = torch.load(OUTV / f'model_{TAG}.pt', map_location='cpu')
    net = Net(widths=tuple(ck['widths']))
    net.load_state_dict({k: v.float() for k, v in ck['net'].items()})
    return net.to(DEV).eval()


@torch.no_grad()
def emb(net, X, bs=2048):
    out = []
    for s in range(0, len(X), bs):
        out.append(net.crop(to_t(X[s:s + bs]).to(DEV)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


# ------------------------------------------------------------------ features (CHÉP NGUYÊN verifier_ft/s01_features.py)
U = json.load(open(OUTV / 'glyph_chars.json')); uid = {c: i for i, c in enumerate(U)}
SIM5 = json.load(open(OUTV / 'simtop5.json'))
simp_id = {}
SID = np.array([simp_id.setdefault(simp(c), len(simp_id)) for c in U])
by_sid = defaultdict(list)
for k, s in enumerate(SID):
    by_sid[s].append(k)
_vs = {}


def var_set(k):
    if k not in _vs:
        o = set(by_sid[SID[k]]) | {uid[v] for v in variants_of(U[k]) if v in uid}
        o.add(k); _vs[k] = o
    return _vs[k]


def pages_split(pages):
    ps = sorted(set(pages)); rank = {p: i for i, p in enumerate(ps)}
    return np.array([rank[p] % 4 == 3 for p in pages])


def features(Z, W, P, nh, rows, cands, syls, nbps, nbns):
    n = len(rows)
    out = {k: np.full(n, np.nan, np.float32) for k in ('sw', 'mR', 'mS', 'mN', 'mRSN', 'mG', 'se', 'meR', 'meRSN')}
    out['n_hum'] = np.zeros(n, np.int32); out['in_univ'] = np.zeros(n, np.int8)
    out['g_top'] = np.array([''] * n, dtype=object); out['riv_top'] = np.array([''] * n, dtype=object)
    Rcache = {}
    for s0 in range(0, n, 4096):
        idx = np.arange(s0, min(n, s0 + 4096))
        Zb = Z[rows[idx]]
        S = Zb @ W.T
        top = np.argpartition(-S, 12, axis=1)[:, :12]
        SP_ = Zb @ P.T if P is not None else None
        for j, i in enumerate(idx):
            x = cands[i]
            k = uid.get(x)
            if k is None:
                continue
            out['in_univ'][i] = 1
            vs = var_set(k)
            s = S[j]
            sx = s[k]; out['sw'][i] = sx
            tt = sorted(top[j], key=lambda t: -s[t])
            for t in tt:
                if t not in vs:
                    out['mG'][i] = sx - s[t]; out['g_top'][i] = U[t]; break
            syl = syls[i]
            if syl not in Rcache:
                Rcache[syl] = [uid[c] for c in R_of(syl) if c in uid]
            groups = {'mR': Rcache[syl], 'mS': [uid[c] for c in SIM5.get(x, []) if c in uid],
                      'mN': [uid[c] for c in (nbps[i], nbns[i]) if c and c in uid]}
            allr = []
            for gname, gl in groups.items():
                gl = [t for t in gl if t not in vs]
                if gl:
                    out[gname][i] = sx - s[gl].max(); allr += gl
            if allr:
                t = max(allr, key=lambda t: s[t]); out['mRSN'][i] = sx - s[t]; out['riv_top'][i] = U[t]
            out['n_hum'][i] = nh[k]
            if SP_ is not None and nh[k] >= 3:
                sp = SP_[j]; out['se'][i] = sp[k]
                rr = [t for t in groups['mR'] if t not in vs and nh[t] >= 3]
                if rr:
                    out['meR'][i] = sp[k] - sp[rr].max()
                ra = [t for t in allr if nh[t] >= 3]
                if ra:
                    out['meRSN'][i] = sp[k] - sp[ra].max()
    return pd.DataFrame(out)


# ------------------------------------------------------------------ X_of + Logit (CHÉP NGUYÊN s02_eval.py)
def X_of(F):
    def fill(c, v):
        a = F[c].values.astype(np.float64); m = np.isnan(a)
        a = np.where(m, v, a); return a, m.astype(np.float64)
    sw, _ = fill('sw', 0.0)
    mG, _ = fill('mG', 0.0)
    mR, nR = fill('mR', 0.3); mS, nS = fill('mS', 0.3); mN, nN = fill('mN', 0.3); mA, nA = fill('mRSN', 0.3)
    se, ne = fill('se', 0.0); meA, nmeA = fill('meRSN', 0.3)
    lh = np.log1p(F.n_hum.values.astype(np.float64))
    cols = [sw, mG, np.minimum(mG, 0), mR, mS, mN, mA, np.minimum(mA, 0), nR, nN, se, meA, np.minimum(meA, 0), ne, nmeA, lh]
    X = np.stack(cols, 1)
    X[F.in_univ.values == 0] = 0
    return X


from scipy.optimize import minimize  # noqa


class Logit:
    def fit(self, X, y, w=None, l2=1e-2):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        w = np.ones(len(y)) if w is None else w
        def f(b):
            z = Z @ b; p = 1 / (1 + np.exp(-z))
            ll = -(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() / w.sum() + l2 * (b[:-1] ** 2).sum()
            g = Z.T @ (w * (p - y)) / w.sum(); g[:-1] += 2 * l2 * b[:-1]
            return ll, g
        self.b = minimize(f, np.zeros(Z.shape[1]), jac=True, method='L-BFGS-B').x
        return self

    def p(self, X):
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        return 1 / (1 + np.exp(-(Z @ self.b)))


def main():
    from harness_lib import var_eq  # noqa
    MB = pd.read_pickle(OUTV / 'meta_borg.pkl')
    MI = pd.read_pickle(OUTV / 'meta_ihr.pkl'); MI['partB'] = pages_split(MI.page.values)
    MG = pd.read_pickle(OUTV / 'meta_gold.pkl')
    key = dict(zip(zip(MI.page, MI.book, MI.column, MI.nom_idx.astype(int)), MI.ocr_char))
    MI['nbp'] = [key.get((p, b, c, n - 1), '') for p, b, c, n in zip(MI.page, MI.book, MI.column, MI.nom_idx.astype(int))]
    MI['nbn'] = [key.get((p, b, c, n + 1), '') for p, b, c, n in zip(MI.page, MI.book, MI.column, MI.nom_idx.astype(int))]
    MI['cand'] = np.where(MI.label != '', MI.label, MI.ocr_char)
    gold_uids = set(MG.cell_uid)
    # ---- ảnh crop chuẩn
    def new_path(u):
        p = BIG / ('crops' if u in gold_uids else 'aux') / uid_path(u)
        return p if p.exists() else None
    pg = [new_path(u) for u in MG.cell_uid]
    pi = [new_path(u) for u in MI.cell_uid]
    INV['gold_new_missing'] = int(sum(p is None for p in pg)); INV['ihr_new_missing'] = int(sum(p is None for p in pi))
    XG = load_imgs(pg); XI = load_imgs(pi)
    log(f'imgs gold {XG.shape} ihr {XI.shape} missing {INV["gold_new_missing"]}/{INV["ihr_new_missing"]}')
    XI_old = np.load(SP / 'r5/hand_lobo/out/img_ihr.npy')
    res_g = pd.DataFrame({'cell_uid': MG.cell_uid.values, 'book_set': MG.book_set.values})
    res_i = pd.DataFrame({'cell_uid': MI.cell_uid.values, 'book': MI.book.values})
    GS = pd.read_csv(VF / 'gold_scores.csv', usecols=['cell_uid', 'p_wood_T', 'p_wood_L'])
    assert (GS.cell_uid.values == MG.cell_uid.values).all()
    for TAG in ('T', 'L'):
        net = load_net(TAG)
        EI_old_re = emb(net, XI_old)
        EI = np.load(OUTV / f'emb_{TAG}_ihr.f16.npy').astype(np.float32)
        INV[f'reembed_ihr_{TAG}_maxabs'] = float(np.abs(EI_old_re.astype(np.float32) - EI).max())
        INV[f'reembed_ihr_{TAG}_cos_min'] = float((EI_old_re.astype(np.float32) * EI).sum(1).min())
        ZG = emb(net, XG).astype(np.float32); ZI = emb(net, XI).astype(np.float32)
        np.save(BIG / f'emb_{TAG}_gold_new{SUF}.f16.npy', ZG.astype(np.float16)); np.save(BIG / f'emb_{TAG}_ihr_new{SUF}.f16.npy', ZI.astype(np.float16))
        del net
        W = np.load(OUTV / f'W_{TAG}.f16.npy').astype(np.float32)
        EB = np.load(OUTV / f'emb_{TAG}_borg.f16.npy').astype(np.float32)
        EGo = np.load(OUTV / f'emb_{TAG}_gold.f16.npy').astype(np.float32)
        acc = np.zeros((len(U), W.shape[1]), np.float32); nh = np.zeros(len(U), np.int32)
        for r in np.nonzero((MB.keep.values == 1) & (MB.fold.values != 4))[0]:
            k = uid.get(MB.char.iat[r])
            if k is not None:
                acc[k] += EB[r]; nh[k] += 1
        for r, row in enumerate(MI.itertuples()):
            if row.book != TUNE[TAG] or row.partB:
                continue
            t = row.gt_char if (row.slot_ok == '1' and row.gt_char) else (row.gt_img if row.slot_ok == '0' else '')
            k = uid.get(t) if t else None
            if k is not None:
                acc[k] += EI[r]; nh[k] += 1
        P = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-9)
        # đặc trưng cũ (đã lưu) -> hiệu chuẩn; đặc trưng mới
        FIo = pd.read_pickle(OUTV / f'feat_{TAG}_ihr.pkl')
        FGo = pd.read_pickle(OUTV / f'feat_{TAG}_gold.pkl')
        # invariant: tính lại đặc trưng cũ của GOLD từ emb cũ == đã lưu
        FGo_re = features(EGo, W, P, nh, np.arange(len(MG)), MG.label.values, MG.syllable.values, MG.nb_prev.values, MG.nb_next.values)
        INV[f'feat_gold_old_reproduce_{TAG}'] = float(np.nanmax(np.abs(FGo_re.mRSN.values - FGo.mRSN.values)))
        FIn = features(ZI, W, P, nh, np.arange(len(MI)), MI.cand.values, MI.syllable.values, MI.nbp.values, MI.nbn.values)
        FGn = features(ZG, W, P, nh, np.arange(len(MG)), MG.label.values, MG.syllable.values, MG.nb_prev.values, MG.nb_next.values)
        slot_det = MI.slot_ok.isin(['0', '1'])
        y_both = np.array([var_eq(a, b) for a, b in zip(MI.cand, MI.gt_char)]) & (MI.slot_ok == '1').values
        cal = ((MI.book == TUNE[TAG]) & MI.partB & slot_det & (MI.gt_char != '') & (MI.cand.str.len() == 1)).values
        m = Logit().fit(X_of(FIo)[cal], y_both[cal].astype(float))
        pgo = m.p(X_of(FGo))
        INV[f'p_wood_{TAG}_old_reproduce_maxabs'] = float(np.abs(np.round(pgo, 5) - GS[f'p_wood_{TAG}'].values).max())
        res_g[f'p_wood_{TAG}_old'] = pgo; res_g[f'p_wood_{TAG}_new'] = m.p(X_of(FGn))
        res_i[f'p_wood_{TAG}_old'] = m.p(X_of(FIo)); res_i[f'p_wood_{TAG}_new'] = m.p(X_of(FIn))
        for c in ('sw', 'mRSN', 'mG', 'se'):
            res_g[f'{c}_{TAG}_new'] = FGn[c].values; res_g[f'{c}_{TAG}_old'] = FGo[c].values
        log(f'{TAG} xong')
    res_g.to_pickle(BIG / f'vft_gold{SUF}.pkl'); res_i.to_pickle(BIG / f'vft_ihr{SUF}.pkl')
    INV['ok_reembed'] = all(INV[f'reembed_ihr_{t}_maxabs'] < 2e-2 for t in 'TL')
    INV['ok_p_wood_old'] = all(INV[f'p_wood_{t}_old_reproduce_maxabs'] < 1e-4 for t in 'TL')
    INV['sec'] = round(time.time() - T0)
    json.dump(INV, open(LABOUT / f's03_vft{SUF}.json', 'w'), indent=1)
    log(json.dumps(INV))


if __name__ == '__main__':
    main()
