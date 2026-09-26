#!/usr/bin/env python
"""s06a_masks.py — mặt nạ H1/H2/H4 của TN3 làm lại với CROP CHUẨN (phiên bản TN4_VER). CPU, 0 API.

H1_new = ¬(A0_new ∪ B0_new ∪ CNT ∪ BC)    (A0_new, B0_new theo t00_freeze prereg; CNT, BC giữ như TN3)
SIMG_new: công thức TN3 với điểm tính trên crop mới (p_wood_T/L: s03; viss_T/L/X: rerank_new r03 'viss'; STT: s05 cert q=0,00015
          ∧ lobo_nh ≥ 3). Ngưỡng CHỌN LẠI đúng thủ tục TN1 (sel_C5/sel_C2, τ = 0,995, sách tune phần B, ô slot_det, y = V1(var_eq)
          ∧ khe = 1 — khe của CROP MỚI, lọc r2 = ¬(A0_new ∪ BC)), LOBO: T chọn trên TK -> dùng cho L16; L chọn trên L16 -> dùng cho TK.
Invariant: cùng hàm chọn ngưỡng trên điểm CŨ + cờ CŨ tái lập đúng ngưỡng TN1 (t01_summary thresholds T_0.995/L_0.995).
Ra: TN4/<ver>/masks_new.pkl (cell_uid, H1n, SIMGn, slot_new), lab/.../out_<ver>/s06a.json"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ver import BIG, LABOUT, VER  # noqa
REPO = HERE.parents[2]
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
TN1 = REPO / 'lab/thu_nghiem_anh_chu/TN1_cong_kiem'
T0 = time.time()
INV = {}
MIN_ACC = 50
TGT = 0.995


def pick_t(s, y, tgt):
    o = np.argsort(-s, kind='stable'); cp = np.cumsum(y[o]) / np.arange(1, len(y) + 1)
    ok = np.nonzero(cp >= tgt)[0]; ok = ok[ok >= MIN_ACC - 1]
    return float(s[o][ok.max()]) if len(ok) else np.inf


def sel_C5(pw, vs, y, tu, r2, tgt):
    pw, vs, y, r2 = pw[tu], vs[tu], y[tu].astype(float), r2[tu]
    qs = np.unique(np.quantile(pw, np.linspace(0, 0.8, 41))); qv = np.unique(np.quantile(vs, np.linspace(0, 0.8, 41)))
    best = None
    for t1 in qs:
        a1 = r2 & (pw >= t1)
        for t2 in qv:
            a = a1 & (vs >= t2)
            if a.sum() < MIN_ACC:
                continue
            if y[a].mean() >= tgt and (best is None or a.mean() > best[2]):
                best = (float(t1), float(t2), a.mean())
    return (round(best[0], 5), round(best[1], 5)) if best else (np.inf, np.inf)


def sel_C2(pw, y, tu, r2, tgt):
    s = np.where(r2[tu], pw[tu], -np.inf)
    return round(pick_t(s, y[tu].astype(float), tgt), 5)


def load_viss(dirp, prefix=''):
    v_T = pd.read_pickle(dirp / 'pred_viss_Tru2Luc.pkl')[['cell_uid', 'p_kim']].rename(columns={'p_kim': prefix + 'viss_T'})
    v_L = pd.read_pickle(dirp / 'pred_viss_Luc2Tru.pkl')[['cell_uid', 'p_kim']].rename(columns={'p_kim': prefix + 'viss_L'})
    v_X = pd.read_pickle(dirp / 'pred_viss_LITHO.pkl')
    v_X = v_X[v_X.role == 'test'][['cell_uid', 'p_kim']].rename(columns={'p_kim': prefix + 'viss_X'})
    return v_T, v_L, v_X


def main():
    X = pd.read_pickle(SP / 'r6/policy_v3/out/base_v2.pkl')
    n = len(X)
    G4 = pd.read_pickle(BIG / 'gold_tn4.pkl').set_index('cell_uid').reindex(X.cell_uid)
    VG = pd.read_pickle(BIG / 'vft_gold.pkl').set_index('cell_uid').reindex(X.cell_uid)
    HS = pd.read_pickle(BIG / 'hand_stt.pkl').set_index('cell_uid')
    vT, vL, vX = load_viss(BIG / 'rerank_new', 'n_')
    Xn = X[['cell_uid']].merge(vT, on='cell_uid', how='left').merge(vL, on='cell_uid', how='left').merge(vX, on='cell_uid', how='left')
    # invariant: viss cũ của base_v2 tái lập từ pred cũ
    oT, oL, oX = load_viss(SP / 'kim_bottleneck/rerank/out')
    Xo = X[['cell_uid']].merge(oT, on='cell_uid', how='left').merge(oL, on='cell_uid', how='left').merge(oX, on='cell_uid', how='left')
    INV['viss_old_join_eq'] = bool(np.allclose(Xo.viss_T.fillna(-1).values, X.viss_T.fillna(-1).values) and
                                   np.allclose(Xo.viss_L.fillna(-1).values, X.viss_L.fillna(-1).values) and
                                   np.allclose(Xo.viss_X.fillna(-1).values, X.viss_X.fillna(-1).values))
    AB = {'Chrestomathie1872': 'Chr', 'LucVanTien1883': 'L83', 'KimVanKieu1884': 'KVK', 'LucVanTien1916': 'L16', 'TruyenKieu1872': 'TK'}
    bk = X.cell_uid.str.split('/').str[1]
    S8 = np.where(X.book_set.values == 'SachThanhTruyen', bk, X.book_set.map(AB).fillna('').values).astype(object)
    S = X.book_set.values
    b = lambda c: X[c].fillna(False).astype(bool).values
    cqf = X.crop_quality_flag.fillna('').values
    A0 = b('int_foreign') | b('blank') | b('truncated') | b('geo_f_dup_bbox')
    B0 = ((cqf == 'blank') | b('blank') | (cqf == 'truncated') | b('truncated') | b('geo_f_dup_bbox') | b('geo_f_ov_heavy')
          | b('f_tight_nb') | (cqf == 'bleed') | (X.seg_flag.fillna('').astype(str).values == 'tall'))
    CNT = b('geo_f_cnt_ocr_ne_qn')
    ady, adx, visz = X.ady.values.astype(float), X.adx.values.astype(float), X.vis_z.values.astype(float)
    visf = lambda v: np.nan_to_num(visz, nan=np.inf) <= v + 1e-12
    BC = {'T': (np.nan_to_num(ady, nan=-1) > 0.6) | visf(-1.0),
          'L': (np.nan_to_num(ady, nan=-1) > 0.5) | (np.nan_to_num(adx, nan=-1) > 0.5) | visf(-0.5)}
    gb = lambda c: G4[c].fillna(False).astype(bool).values
    A0n = b('int_foreign') | b('geo_f_dup_bbox') | gb('f_blank') | gb('f_cut')
    B0n = gb('B0_box') | ~gb('one_char_ok') | gb('tall_new')
    isL16, isTK = S8 == 'L16', S8 == 'TK'
    BCd = np.where(isL16, BC['T'], BC['L'])
    H1o = ~(A0 | B0 | CNT | BCd)
    H1n = ~(A0n | B0n | CNT | BCd)
    # sự thật cho chọn ngưỡng (quy ước p02b/TN1): y = y_var ∧ khe=1 trên ô slot_det, sách tune phần B
    gt = X.gt_char.fillna('').values
    HAS = np.isin(S, ['LucVanTien1916', 'TruyenKieu1872']) & (gt != '')
    PARTB = b('partB')
    yvar = b('y_var')
    slot_o = X.slot_ok.fillna('').astype(str).values
    slot_n = G4.slot_new.fillna('').astype(str).values
    DIRS = {'T': dict(tune='TruyenKieu1872', pw='p_wood_T', vs='viss_T'), 'L': dict(tune='LucVanTien1916', pw='p_wood_L', vs='viss_L')}
    TH = {}
    for tag, slot, A0_, pwsrc in (('old', slot_o, A0, 'old'), ('new', slot_n, A0n, 'new')):
        y = yvar & (slot == '1')
        sdet = np.isin(slot, ['0', '1'])
        for d, c in DIRS.items():
            tu = (S == c['tune']) & HAS & sdet & PARTB
            r2 = ~(A0_ | BC[d])
            pw = (X[c['pw']].values.astype(float) if tag == 'old' else np.round(VG[f'{c["pw"]}_new'].values.astype(float), 5))
            vs = (X[c['vs']] if tag == 'old' else Xn['n_' + c['vs']]).fillna(0).values.astype(float)
            TH[(tag, d)] = dict(C5=sel_C5(pw, vs, y, tu, r2, TGT), C2=sel_C2(pw, y, tu, r2, TGT), n_tune=int(tu.sum()))
    T1S = json.load(open(TN1 / 't01_summary.json'))['thresholds']
    INV['thresholds_old_reproduce_TN1'] = bool(np.allclose(TH[('old', 'T')]['C5'], T1S['T_0.995']['C5']) and
                                               np.allclose(TH[('old', 'L')]['C5'], T1S['L_0.995']['C5']) and
                                               abs(TH[('old', 'T')]['C2'] - T1S['T_0.995']['C2']) < 1e-9 and
                                               abs(TH[('old', 'L')]['C2'] - T1S['L_0.995']['C2']) < 1e-9)
    # ---- SIMG
    LITHO, CHR, STT = np.isin(S8, ['L83', 'KVK']), S8 == 'Chr', np.isin(S8, ['stt2', 'stt4', 'stt11'])

    def simg(tag):
        if tag == 'old':   # quy ước r4: p_wood làm tròn 5 chữ số (gold_scores.csv)
            pwT, pwL = X.p_wood_T.values.astype(float), X.p_wood_L.values.astype(float)
        else:
            pwT, pwL = (np.round(VG[f'p_wood_{t}_new'].values.astype(float), 5) for t in 'TL')
        src = X if tag == 'old' else Xn.rename(columns={'n_viss_T': 'viss_T', 'n_viss_L': 'viss_L', 'n_viss_X': 'viss_X'})
        vsT, vsL, vX = (src[c].fillna(0).values.astype(float) for c in ('viss_T', 'viss_L', 'viss_X'))
        C5T, C5L = TH[(tag, 'T')]['C5'], TH[(tag, 'L')]['C5']
        C2T, C2L = TH[(tag, 'T')]['C2'], TH[(tag, 'L')]['C2']
        if tag == 'old':
            hand = (X.lobo_cert_00015.fillna(0).astype(int).values == 1) & (X.lobo_nh.fillna(0).values >= 3)
        else:
            hc = HS.cert_new.reindex(X.cell_uid).fillna(0).astype(int).values
            hand = (hc == 1) & (X.lobo_nh.fillna(0).values >= 3)
        s = np.zeros(n, bool)
        s[isL16] = ((pwT >= C5T[0]) & (vsT >= C5T[1]))[isL16]
        s[isTK] = ((pwL >= C5L[0]) & (vsL >= C5L[1]))[isTK]
        s[LITHO] = ((pwT >= C5T[0]) & (pwL >= C5L[0]) & (vX >= max(C5T[1], C5L[1])))[LITHO]
        s[CHR] = ((pwT >= C2T) & (pwL >= C2L))[CHR]
        s[STT] = hand[STT]
        return s
    SIo, SIn = simg('old'), simg('new')
    TA_SETS = np.isin(S8, ['TK', 'KVK', 'L83'])
    TA_OK = ~TA_SETS | (X.ta.fillna('na').values == 'attested')
    # invariant: H2/H4 cũ = TN1 projection (d)/(e)
    PJ = pd.read_csv(TN1 / 'projection.csv')
    for bs, a in [('SachThanhTruyen', 'STT'), ('Chrestomathie1872', 'Chr'), ('LucVanTien1883', 'L83'), ('KimVanKieu1884', 'KVK'),
                  ('LucVanTien1916', 'L16'), ('TruyenKieu1872', 'TK')]:
        mm = S == bs
        INV[f'H2_old_eq_TN1_d_{a}'] = int((H1o & SIo & mm).sum()) == int(PJ[(PJ.sach == a) & (PJ.config == 'd')].giu.iloc[0])
        INV[f'H4_old_eq_TN1_e_{a}'] = int((H1o & SIo & TA_OK & mm).sum()) == int(PJ[(PJ.sach == a) & (PJ.config == 'e')].giu.iloc[0])
    out = pd.DataFrame(dict(cell_uid=X.cell_uid.values, set8=S8, H1o=H1o, H1n=H1n, SIMGo=SIo, SIMGn=SIn, TA_OK=TA_OK,
                            slot_old=slot_o, slot_new=slot_n))
    out.to_pickle(BIG / 'masks_new.pkl')
    cnt = {}
    for s in ['stt2', 'stt4', 'stt11', 'Chr', 'L83', 'KVK', 'L16', 'TK']:
        mm = S8 == s
        cnt[s] = dict(n=int(mm.sum()), H1_cu=int((H1o & mm).sum()), H1_moi=int((H1n & mm).sum()),
                      H2_cu=int((H1o & SIo & mm).sum()), H2_moi=int((H1n & SIn & mm).sum()),
                      H4_cu=int((H1o & SIo & TA_OK & mm).sum()), H4_moi=int((H1n & SIn & TA_OK & mm).sum()),
                      SIMG_cu=int((SIo & mm).sum()), SIMG_moi=int((SIn & mm).sum()))
    res = dict(ver=VER, invariants=INV, thresholds={f'{k[0]}_{k[1]}': v for k, v in TH.items()}, counts=cnt,
               sec=round(time.time() - T0))
    json.dump(res, open(LABOUT / 's06a.json', 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, default=str)[:3500])


if __name__ == '__main__':
    main()
