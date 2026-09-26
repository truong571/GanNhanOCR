#!/usr/bin/env python3
"""s3_eval.py — TN2 bước 3: VÒNG LẶP SỬA HỘP (một lượt) + chấm bằng MÔ HÌNH KHE NGƯỜI (IHR) + LOBO hai chiều.

Vòng lặp (định trước, xem KET_QUA.md §1):
  ô bị cờ = GOLD có geo_f_kim_off=1. Ứng viên {prev, next, kim, det} (s1b), điểm m = sw(nhãn) − max sw(đối thủ) (s2).
  best = argmax m. NHẬN thô khi m(best) ≥ θ VÀ m(best) > m(crop đang giao). Sau đó, trên TRẠNG THÁI CUỐI của cột:
    - không trùng hộp: IoU(hộp mới, hộp của mọi ô giao nộp khác còn giữ hộp) < 0,5
    - đơn điệu dọc: tâm y hộp mới nằm giữa tâm y của ô còn giữ có nom_idx liền trước / liền sau
  vi phạm -> huỷ nhận (ô bị HẠ), lặp đến khi ổn định. Không nhận -> HẠ. Nhãn KHÔNG BAO GIỜ đổi.
Bộ chọn (s2) KHÔNG dùng để chấm. Chấm = slotlib (hộp cột người vẽ + chữ GT người):
  ảnh đúng khe = slot_ok==1 (khe chưa xác định tính SAI, như báo cáo v2); phụ: ảnh=nhãn (gt_img ~V1+ nhãn), hai vế.
Mô hình chọn: sách L16 chấm bằng T, TK chấm bằng L (không mô hình nào thấy sách nó chấm). θ chọn trên sách kia
(tiêu chí đăng ký trước: θ nhỏ nhất trên lưới mà độ đúng khe của ô được sửa ≥ độ đúng khe của GOLD không bị cờ
ở sách tune, và giữ được ở mọi θ lớn hơn trên lưới).
Chr/STT: không có sự thật -> chỉ đếm đề xuất (hai mô hình cùng chọn một ứng viên và cùng qua θ của mình).
Ra: lab/.../TN2_vong_sua_hop/{s3_summary.json, s3_lobo_table.csv, s3_theta_curve.csv}, measure_out/.../TN2/decisions.csv
"""
import json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR')
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
OUT = REPO / 'measure_out/_thu_nghiem_anh_chu/TN2'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(SP / 'r5/policy_v2/scripts'))
from slotlib import SlotModel  # noqa
from vlib import var_eq_plus  # noqa
IHR = ['LucVanTien1916', 'TruyenKieu1872']
MODEL_FOR = {'LucVanTien1916': 'T', 'TruyenKieu1872': 'L'}
OTHER = {'LucVanTien1916': 'TruyenKieu1872', 'TruyenKieu1872': 'LucVanTien1916'}
GRID = [round(x, 2) for x in np.arange(-0.20, 0.801, 0.05)]
FIXC = ['prev', 'next', 'kim', 'det']
B_BOOT = 2000
SEED = 20260926


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def cy(b):
    return (b[1] + b[3]) / 2.0


def nom_of(uid):
    return int(re.search(r'/n(\d+)/', uid).group(1))


def load_all():
    S = pd.read_csv(OUT / 'scores.csv', dtype=str, keep_default_na=False)
    for c in ['m_T', 'm_L', 'mN_T', 'mN_L', 'sw_T', 'sw_L']:
        S[c] = pd.to_numeric(S[c], errors='coerce')
    L = pd.read_csv(REPO / 'dataset/_ALL/labels.csv', dtype=str, keep_default_na=False,
                    usecols=['cell_uid', 'book_set', 'book', 'page', 'column', 'label', 'tier', 'bbox'])
    L = L[L.book_set.isin(IHR + ['Chrestomathie1872', 'SachThanhTruyen'])]
    return S, L


def column_state(L):
    """(book_set, book, page, column) -> list of [cell_uid, nom_idx, bbox] (mọi tầng giao nộp)."""
    cols = defaultdict(list)
    for u, bs, bk, pg, c, bb in zip(L.cell_uid, L.book_set, L.book, L.page, L.column, L.bbox):
        if not bb:
            continue
        cols[(bs, bk, pg, c)].append([u, nom_of(u), [int(v) for v in json.loads(bb)]])
    return cols


def propose(S, tag, theta):
    """-> dict cell_uid -> (cand, bbox) cho ô NHẬN THÔ (trước kiểm trùng/đơn điệu)."""
    m = f'm_{tag}'
    out = {}
    for u, g in S.groupby('cell_uid', sort=False):
        dv = g[g.cand == 'deliv'][m]
        m0 = float(dv.iloc[0]) if len(dv) and dv.iloc[0] == dv.iloc[0] else -np.inf
        f = g[g.cand.isin(FIXC) & g[m].notna()]
        if not len(f):
            continue
        j = f[m].idxmax()
        if f.at[j, m] >= theta and f.at[j, m] > m0:
            out[u] = (f.at[j, 'cand'], [int(v) for v in json.loads(f.at[j, 'bbox'])], float(f.at[j, m]))
    return out


def propose_kim(S):
    """Biến thể THAM CHIẾU (không bộ kiểm, không ngưỡng): mọi ô bị cờ nhận ứng viên 'kim' nếu có."""
    k = S[S.cand == 'kim']
    return {u: ('kim', [int(v) for v in json.loads(b)], float('nan')) for u, b in zip(k.cell_uid, k.bbox)}


SHARE = 'strict'   # 'strict': ô bị HẠ vẫn giữ hộp cũ (vẫn nằm trong bộ ở tầng thấp) | 'lenient': chỉ ô không bị hạ giữ hộp


def resolve(raw, flagged, cols, key_of):
    """Kiểm trùng hộp + đơn điệu trên trạng thái cuối; trả tập ô được nhận cuối và lý do huỷ.
    Trùng hộp: so với mọi ô GIỮ hộp (SHARE='strict': mọi ô giao nộp của cột, kể cả ô bị hạ với hộp cũ;
    'lenient': chỉ ô không bị hạ). Đơn điệu: so với ô không bị hạ liền trước/liền sau theo nom_idx.
    Cặp hai ô cùng dời vào trùng nhau -> hạ cả hai (hộp của ô vừa bị huỷ vẫn tính tới hết lượt quét)."""
    acc = dict(raw)
    why = {}
    changed = True
    while changed:
        changed = False
        for k, members in cols.items():
            mv = [x for x in members if x[0] in acc]
            if not mv:
                continue
            active = [x for x in members if (x[0] not in flagged) or (x[0] in acc)]
            holders = members if SHARE == 'strict' else active
            box = {x[0]: (acc[x[0]][1] if x[0] in acc else x[2]) for x in members}
            for u, ni, _ in mv:
                if u not in acc:
                    continue
                b = box[u]
                bad = None
                for v, nj, _ in holders:
                    if v != u and iou(b, box[v]) >= 0.5:
                        bad = 'trung_hop'; break
                if bad is None:
                    lo = [x for x in active if x[1] < ni and x[0] != u]
                    hi = [x for x in active if x[1] > ni and x[0] != u]
                    if lo:
                        p = max(lo, key=lambda x: x[1])
                        if cy(box[p[0]]) >= cy(b):
                            bad = 'khong_don_dieu'
                    if bad is None and hi:
                        q = min(hi, key=lambda x: x[1])
                        if cy(box[q[0]]) <= cy(b):
                            bad = 'khong_don_dieu'
                if bad:
                    why[u] = bad
                    del acc[u]
                    changed = True
    return acc, why


def boot_ci(pages, vals, B=B_BOOT, seed=SEED):
    """vals: list of arrays (một mảng/ chính sách) theo ô, NaN = ô không ở GOLD. CI cụm theo trang."""
    rng = np.random.default_rng(seed)
    up = np.unique(pages)
    idx = {p: np.nonzero(pages == p)[0] for p in up}
    res = [[] for _ in vals]
    for _ in range(B):
        samp = rng.choice(up, len(up), replace=True)
        ii = np.concatenate([idx[p] for p in samp])
        for k, v in enumerate(vals):
            x = v[ii]; x = x[~np.isnan(x)]
            res[k].append(x.mean() if len(x) else np.nan)
    return [np.nanpercentile(r, [2.5, 97.5]).round(4).tolist() for r in res], [np.array(r) for r in res]




def main():
    global SHARE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--share', default='strict', choices=['strict', 'lenient'])
    SHARE = ap.parse_args().share
    SUF = '' if SHARE == 'strict' else '_lenient'
    t0 = time.time()
    S, L = load_all()
    cols_all = column_state(L)
    ce = pd.read_csv(SP / 'kim_bottleneck/harness/cells_eval.csv', dtype=str, keep_default_na=False,
                     usecols=['cell_uid', 'book', 'page', 'column', 'syl_idx', 'label', 'gt_char', 'gt_img', 'slot_ok', 'bbox',
                              'delivered_tier', 'geo_status'])
    summary = dict(date='2026-09-26', api_calls=0, grid=GRID, invariants={},
                   dinh_nghia=dict(
                       anh_nhan='CHÍNH: ảnh = nhãn — chữ GT người ở khe mà crop nằm (2 mô hình khe cùng chỉ) tương đương V1+ với nhãn',
                       khe='phụ: slot_ok==1 (crop ở đúng khe syl_idx của cột GT)',
                       hai_ve='phụ: nhãn ~V1+ chữ GT tại syl_idx VÀ slot_ok==1 (định nghĩa báo cáo v2)',
                       mau_so='GOLD trên trang có GT người (geo_status ok); khe chưa xác định tính SAI'))
    inv = summary['invariants']
    # ------------------------------------------------ sự thật khe cho mọi ứng viên IHR
    truth = {}
    for book in IHR:
        M = SlotModel(book)
        Sb = S[S.book_set == book]
        cer = ce[ce.book == book].set_index('cell_uid')
        n_eq = n = 0
        for r in Sb.itertuples():
            o = M.slot(r.page, r.column, r.syl_idx, json.loads(r.bbox))
            truth[(r.cell_uid, r.cand)] = o
            if r.cand in ('old', 'deliv'):
                n += 1; n_eq += (str(o['slot_ok']) == cer.at[r.cell_uid, 'slot_ok'] and o['gt_img'] == cer.at[r.cell_uid, 'gt_img'])
        inv[f'{book}_old_slot_gtimg_reproduce_cells_eval'] = bool(n_eq == n)
    S['slot_new'] = [str(truth.get((u, c), {}).get('slot_ok', '')) for u, c in zip(S.cell_uid, S.cand)]
    S['gtimg_new'] = [truth.get((u, c), {}).get('gt_img', '') for u, c in zip(S.cell_uid, S.cand)]
    S['gttmpl_new'] = [truth.get((u, c), {}).get('gt_tmpl', '') for u, c in zip(S.cell_uid, S.cand)]
    # ------------------------------------------------ bảng GOLD IHR + cờ (mẫu số = trang có GT người)
    gs = pd.read_csv(SP / 'gold_img_audit/measure/gold_suspicion.csv', dtype=str, keep_default_na=False,
                     usecols=['cell_uid', 'is_gold', 'geo_f_kim_off'])
    flag_all = set(gs[(gs.is_gold == '1') & (gs.geo_f_kim_off == '1')].cell_uid)
    per_book = {}
    for book in IHR:
        G = ce[(ce.book == book) & (ce.delivered_tier == 'GOLD')].copy()
        inv[f'{book}_flagged_all_have_candidates'] = int(G.cell_uid.isin(flag_all).sum()) == int(
            S[(S.book_set == book) & (S.cand == 'old')].cell_uid.nunique())
        summary.setdefault('ngoai_mau_so', {})[book] = dict(gold_trang_khong_gt=int((G.geo_status != 'ok').sum()),
                                                            co_trong_do=int((G.cell_uid.isin(flag_all) & (G.geo_status != 'ok')).sum()))
        G = G[G.geo_status == 'ok'].reset_index(drop=True)
        G['flag'] = G.cell_uid.isin(flag_all)
        G['A0'] = (G.slot_ok == '1').astype(float)
        G['B0'] = [float(var_eq_plus(a, b)) for a, b in zip(G.gt_img, G.label)]
        G['Bdet0'] = (G.gt_img != '').astype(int)
        G['C0'] = [float(var_eq_plus(a, b) and s == '1') for a, b, s in zip(G.gt_char, G.label, G.slot_ok)]
        per_book[book] = G

    def run_policy(book, tag, theta, raw=None):
        G = per_book[book]
        Sb = S[S.book_set == book]
        flagged = set(Sb.cell_uid)                      # mọi ô bị cờ của sách (cả trang không GT) giữ/nhả hộp trong cột
        if raw is None:
            raw = propose(Sb, tag, theta)
        cols = {k: v for k, v in cols_all.items() if k[0] == book and any(x[0] in raw for x in v)}
        acc, why = resolve(raw, flagged, cols, None)
        sn = Sb.set_index(['cell_uid', 'cand'])
        A1, B1, C1, cls, clsA, Bt = [], [], [], [], [], []
        for u, f, a0, b0, bd0, c0, lab, gch in zip(G.cell_uid, G.flag, G.A0, G.B0, G.Bdet0, G.C0, G.label, G.gt_char):
            if not f:
                A1.append(a0); B1.append(b0); C1.append(c0); cls.append('khong_co'); clsA.append('khong_co'); continue
            before = 'dung' if b0 == 1 else ('sai' if bd0 else 'chua_xd')
            if u in acc:
                cand = acc[u][0]
                s_new = sn.at[(u, cand), 'slot_new']; gi = sn.at[(u, cand), 'gtimg_new']
                a1 = float(s_new == '1'); b1 = float(var_eq_plus(gi, lab)); c1 = float(var_eq_plus(gch, lab) and s_new == '1')
                A1.append(a1); B1.append(b1); C1.append(c1)
                Bt.append(float(var_eq_plus(sn.at[(u, cand), 'gttmpl_new'], lab)))
                after = 'dung' if b1 == 1 else ('sai' if gi != '' else 'chua_xd')
                if before == 'dung':
                    cls.append('doi_van_dung' if after == 'dung' else 'lam_hong')
                else:
                    cls.append({'dung': 'sua_dung', 'sai': 'sua_sai', 'chua_xd': 'sua_chua_xd'}[after])
                clsA.append(('sua_dung' if a1 else 'sua_sai') if a0 == 0 else ('doi_van_dung' if a1 else 'lam_hong'))
            else:
                A1.append(np.nan); B1.append(np.nan); C1.append(np.nan)
                cls.append('ha_o_dung' if before == 'dung' else 'ha_o_sai')
                clsA.append('ha_o_dung' if a0 == 1 else 'ha_o_sai')
        return dict(A1=np.array(A1), B1=np.array(B1), C1=np.array(C1), cls=np.array(cls), clsA=np.array(clsA),
                    acc=acc, why=why, raw=raw, Btmpl=np.array(Bt))

    FIXED = ['sua_dung', 'sua_sai', 'sua_chua_xd', 'doi_van_dung', 'lam_hong']

    def metrics(book, R):
        G = per_book[book]
        c = Counter(R['cls']); cA = Counter(R['clsA'])
        fixed = np.isin(R['cls'], FIXED)
        pf = float(np.nanmean(R['B1'][fixed])) if fixed.any() else None
        pfA = float(np.nanmean(R['A1'][fixed])) if fixed.any() else None
        return dict(n_gold=len(G), n_flag=int(G.flag.sum()), n_raw_accept=int(sum(1 for u in R['raw'] if u in set(G.cell_uid))),
                    n_accept=int(fixed.sum()),
                    huy_trung_hop=sum(1 for u, v in R['why'].items() if v == 'trung_hop' and u in set(G.cell_uid)),
                    huy_khong_don_dieu=sum(1 for u, v in R['why'].items() if v == 'khong_don_dieu' and u in set(G.cell_uid)),
                    sua_dung=c['sua_dung'], sua_sai=c['sua_sai'], sua_chua_xd=c['sua_chua_xd'], lam_hong_o_dung=c['lam_hong'],
                    doi_hop_van_dung=c['doi_van_dung'], ha_o_sai=c['ha_o_sai'], ha_o_dung=c['ha_o_dung'],
                    prec_fix=round(pf, 4) if pf is not None else None,
                    khe_sua_dung=cA['sua_dung'], khe_sua_sai=cA['sua_sai'], khe_lam_hong=cA['lam_hong'],
                    khe_doi_van_dung=cA['doi_van_dung'], khe_ha_o_sai=cA['ha_o_sai'], khe_ha_o_dung=cA['ha_o_dung'],
                    prec_fix_khe=round(pfA, 4) if pfA is not None else None,
                    prec_fix_chi_mau=round(float(np.mean(R['Btmpl'])), 4) if len(R['Btmpl']) else None)

    # ------------------------------------------------ đường cong θ trên từng sách (mô hình ngoài-sách của nó)
    curve = []
    for book in IHR:
        tag = MODEL_FOR[book]
        G = per_book[book]
        Pu = float(G.B0[~G.flag].mean())
        for th in GRID:
            mm = metrics(book, run_policy(book, tag, th))
            curve.append(dict(book=book, model=tag, theta=th, P_unflagged=round(Pu, 4), **mm))
    CV = pd.DataFrame(curve)
    CV.to_csv(HERE / f's3_theta_curve{SUF}.csv', index=False)

    def choose_theta(book):
        c = CV[CV.book == book].sort_values('theta').reset_index(drop=True)
        Pu = c.P_unflagged.iloc[0]
        good = [(r.n_accept == 0) or (r.prec_fix is not None and r.prec_fix == r.prec_fix and r.prec_fix >= Pu)
                for r in c.itertuples()]
        for i in range(len(c)):
            if all(good[i:]) and c.n_accept.iloc[i] > 0:
                return float(c.theta.iloc[i])
        return None

    # ------------------------------------------------ LOBO hai chiều
    rows, dec_rows = [], []
    for test in IHR:
        tune = OTHER[test]
        th = choose_theta(tune)
        tag = MODEL_FOR[test]
        G = per_book[test]
        pages = G.page.values
        fl = G.flag.values
        if th is None:
            R = run_policy(test, tag, np.inf)
        else:
            R = run_policy(test, tag, th)
        mm = metrics(test, R)
        B0 = G.B0.values.astype(float); A0 = G.A0.values.astype(float); C0 = G.C0.values.astype(float)
        B1 = np.where(fl, np.nan, B0); A1 = np.where(fl, np.nan, A0); C1 = np.where(fl, np.nan, C0)
        cis, reps = boot_ci(pages, [B0, B1, R['B1'], A0, A1, R['A1'], C0, C1, R['C1']])
        d21 = reps[2] - reps[1]
        row = dict(test=test, tune=tune, model=tag, theta=th, **mm,
                   n_P0=len(G), n_P1=int(np.sum(~np.isnan(B1))), n_P2=int(np.sum(~np.isnan(R['B1']))))
        for nm, i0 in (('anh_nhan', 0), ('khe', 3), ('hai_ve', 6)):
            for k, p in enumerate(['P0', 'P1', 'P2']):
                v = [B0, B1, R['B1'], A0, A1, R['A1'], C0, C1, R['C1']][i0 + k]
                row[f'{nm}_{p}'] = round(float(np.nanmean(v)), 4); row[f'{nm}_{p}_ci'] = cis[i0 + k]
        fx = np.isin(R['cls'], FIXED)
        if fx.any():
            vv = np.where(fx, R['B1'], np.nan)
            row['prec_fix_ci'] = boot_ci(pages, [vv])[0][0]
        Rk = run_policy(test, tag, None, raw=propose_kim(S[S.book_set == test]))
        mk = metrics(test, Rk)
        row['kimonly'] = dict(n_accept=mk['n_accept'], sua_dung=mk['sua_dung'], sua_sai=mk['sua_sai'], sua_chua_xd=mk['sua_chua_xd'],
                              lam_hong=mk['lam_hong_o_dung'], doi_van_dung=mk['doi_hop_van_dung'], ha=mk['ha_o_sai'] + mk['ha_o_dung'],
                              prec_fix=mk['prec_fix'], anh_nhan_P2=round(float(np.nanmean(Rk['B1'])), 4),
                              anh_nhan_P2_ci=boot_ci(pages, [Rk['B1']])[0][0], n_P2=int(np.sum(~np.isnan(Rk['B1']))))
        row['anh_nhan_P2_minus_P1'] = round(row['anh_nhan_P2'] - row['anh_nhan_P1'], 4)
        row['anh_nhan_P2_minus_P1_ci'] = np.nanpercentile(d21, [2.5, 97.5]).round(4).tolist()
        rows.append(row)
        acc = R['acc']
        for u, c, cA in zip(G.cell_uid, R['cls'], R['clsA']):
            if c == 'khong_co':
                continue
            dec_rows.append(dict(cell_uid=u, book_set=test, decision='sua' if u in acc else 'ha', cls=c, cls_khe=cA,
                                 cand=acc[u][0] if u in acc else '', new_bbox=json.dumps(acc[u][1]) if u in acc else '',
                                 m_new=round(acc[u][2], 4) if u in acc else '', huy=R['why'].get(u, ''), theta=th, model=tag))
    pd.DataFrame(rows).to_csv(HERE / f's3_lobo_table{SUF}.csv', index=False)
    summary['lobo'] = rows
    summary['theta_chosen_on'] = {b: choose_theta(b) for b in IHR}
    # ------------------------------------------------ Chr / STT: đề xuất (ƯỚC LƯỢNG, chưa kiểm được)
    thT = choose_theta('LucVanTien1916')   # T là mô hình ngoài-sách của L16 -> θ của T chọn trên L16
    thL = choose_theta('TruyenKieu1872')   # L là mô hình ngoài-sách của TK -> θ của L chọn trên TK
    unl = {}
    for bs in ['Chrestomathie1872', 'SachThanhTruyen']:
        Sb = S[S.book_set == bs]
        flagged = set(Sb.cell_uid)
        rT = propose(Sb, 'T', thT) if thT is not None else {}
        rL = propose(Sb, 'L', thL) if thL is not None else {}
        raw = {u: rT[u] for u in rT if u in rL and rL[u][0] == rT[u][0]}
        cols = {k: v for k, v in cols_all.items() if k[0] == bs and any(x[0] in raw for x in v)}
        acc, why = resolve(raw, flagged, cols, None)
        gold_n = int((L[(L.book_set == bs)].tier == 'GOLD').sum())
        unl[bs] = dict(n_gold=gold_n, n_flag=len(flagged), raw_T=len(rT), raw_L=len(rL), raw_both_same_cand=len(raw),
                       de_xuat_sua=len(acc), de_xuat_ha=len(flagged) - len(acc),
                       huy_trung_hop=sum(1 for v in why.values() if v == 'trung_hop'),
                       huy_khong_don_dieu=sum(1 for v in why.values() if v == 'khong_don_dieu'),
                       cand_counts=dict(Counter(v[0] for v in acc.values())), theta_T=thT, theta_L=thL,
                       muc_chac='UOC_LUONG_CHUA_KIEM')
        for u in flagged:
            dec_rows.append(dict(cell_uid=u, book_set=bs, decision='sua' if u in acc else 'ha', cls='chua_kiem', cls_khe='',
                                 cand=acc[u][0] if u in acc else '', new_bbox=json.dumps(acc[u][1]) if u in acc else '',
                                 m_new=round(acc[u][2], 4) if u in acc else '', huy=why.get(u, ''), theta=f'{thT}/{thL}', model='T&L'))
    summary['unlabelled'] = unl
    pd.DataFrame(dec_rows).to_csv(OUT / f'decisions{SUF}.csv', index=False)
    inv['label_never_changed'] = True   # vòng lặp chỉ trả (cand, bbox); nhãn đọc thẳng từ labels.csv khi chấm
    inv['selector_not_used_for_scoring'] = True
    summary['sec'] = round(time.time() - t0)
    summary['share_rule'] = SHARE
    (HERE / f's3_summary{SUF}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    keys = ['test', 'tune', 'model', 'theta', 'n_flag', 'n_accept', 'sua_dung', 'sua_sai', 'sua_chua_xd', 'lam_hong_o_dung',
            'doi_hop_van_dung', 'ha_o_sai', 'ha_o_dung', 'anh_nhan_P0', 'anh_nhan_P1', 'anh_nhan_P2', 'anh_nhan_P2_ci',
            'anh_nhan_P2_minus_P1', 'anh_nhan_P2_minus_P1_ci', 'n_P1', 'n_P2', 'prec_fix', 'prec_fix_ci', 'prec_fix_chi_mau', 'kimonly']
    for r in rows:
        print(json.dumps({k: r.get(k) for k in keys}, ensure_ascii=False))
    print(json.dumps(summary['theta_chosen_on']), json.dumps(unl, ensure_ascii=False))
    print(json.dumps(inv))


if __name__ == '__main__':
    main()
