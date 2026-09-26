#!/usr/bin/env python
"""s10b_hand_retrain_eval.py — ĐỘ NHẠY hậu kiểm (v2): bộ kiểm chữ viết tay HỌC LẠI trên crop chuẩn (h01_train_tn4.py, cùng công thức,
cùng hạt giống, cùng tập ô học: Borg Kinh keep fold 0–3 + IHR sách tune phần A) — đặc trưng (chép s01/h02), hiệu chuẩn (chép h03:
logistic đầy đủ trên Kinh fold 4; ngưỡng q = 0,00015 = lượng tử cross-fit hai nửa trang trên âm) — rồi đo trên Borg DungLy GIỮ NGOÀI
(nhãn người) và đếm chứng nhận STT. So với: bộ kiểm cũ trên crop cũ, và bộ kiểm cũ trên crop chuẩn (s05, "drop-in").
CPU, 0 API. Ra: lab/.../out_v2/s10b_hand_retrain.json, TN4/v2/hand_retrain/hand_stt_rt.pkl"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault('TN4_VER', 'v2')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s03_vft as S3  # noqa
import s05_hand as S5  # noqa
from ver import BIG, LABOUT  # noqa
HL = S3.SP / 'r5/hand_lobo/out'
RT = BIG / 'hand_retrain'
Q = 0.00015
T0 = time.time()


def half_of(keys):
    u = sorted(set(keys)); h = {k: i % 2 for i, k in enumerate(u)}
    return np.array([h[k] for k in keys])


def main():
    MA = pd.read_pickle(HL / 'meta_all.pkl'); MA['pkey'] = MA.book + '/' + MA.page
    MS = pd.read_pickle(HL / 'meta_stt.pkl')
    MI = pd.read_pickle(S3.OUTV / 'meta_ihr.pkl'); MI['partB'] = S3.pages_split(MI.page.values)
    bpath = [BIG / 'aux_borg' / f'{b}/{p}/c{int(c):02d}_{int(i):04d}.png' for b, p, c, i in zip(MA.book, MA.page, MA.col, MA.idx)]
    hasB = np.array([p.exists() for p in bpath])
    single = MA.single.values == 1
    P_T, P_L, PS, R = {}, {}, {}, {}
    thr = {}
    borg = {}
    for t in 'TL':
        ft = f'{t}_Kinh'
        W = np.load(RT / f'W_{ft}.f16.npy').astype(np.float32)
        EA = np.load(RT / f'emb_{ft}_all.f16.npy').astype(np.float32)
        EI = np.load(RT / f'emb_{ft}_ihr.f16.npy').astype(np.float32)
        ES = np.load(RT / f'emb_{ft}_stt.f16.npy').astype(np.float32)
        acc = np.zeros((len(S3.U), W.shape[1]), np.float32); nh = np.zeros(len(S3.U), np.int32)
        for r in np.nonzero((MA.book.values == 'SachKinhThayCaBinh') & (MA.keep.values == 1) & (MA.fold.values != 4) & single)[0]:
            k = S3.uid.get(MA.char.iat[r])
            if k is not None:
                acc[k] += EA[r]; nh[k] += 1
        for r, row in enumerate(MI.itertuples()):
            if row.book != S3.TUNE[t] or row.partB:
                continue
            tt = row.gt_char if (row.slot_ok == '1' and row.gt_char) else (row.gt_img if row.slot_ok == '0' else '')
            k = S3.uid.get(tt) if tt else None
            if k is not None:
                acc[k] += EI[r]; nh[k] += 1
        P = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-9)
        Fc0 = pd.read_pickle(HL / f'feat_{ft}_cal.pkl')
        Fc = S3.features(EA, W, P, nh, Fc0.row.values, Fc0.cand.values, Fc0.syllable.values, Fc0.nbp.values, Fc0.nbn.values)
        keep_c = Fc.in_univ.values == 1
        yc = (Fc0.kind.values == 'pos').astype(float); w = S5.wprior(yc)
        ckey = MA.pkey.values[Fc0.row.values]; chalf = half_of(ckey)
        Xc = S3.X_of(Fc)
        pcf = np.zeros(len(Xc))
        for h in (0, 1):
            tr = (chalf != h) & keep_c
            m = S3.Logit().fit(Xc[tr], yc[tr], w=w[tr]); pcf[chalf == h] = m.p(Xc[chalf == h])
        mfull = S3.Logit().fit(Xc[keep_c], yc[keep_c], w=w[keep_c])
        thr[t] = float(np.quantile(pcf[(yc == 0) & keep_c], 1 - Q))
        Ft0 = pd.read_pickle(HL / f'feat_{ft}_test.pkl')
        sel = hasB[Ft0.row.values] & (Ft0.in_univ.values == 1)
        Ft0 = Ft0[sel].reset_index(drop=True)
        Ft = S3.features(EA, W, P, nh, Ft0.row.values, Ft0.cand.values, Ft0.syllable.values, Ft0.nbp.values, Ft0.nbn.values)
        borg[t] = dict(row=Ft0.row.values, kind=Ft0.kind.values, p=mfull.p(S3.X_of(Ft)))
        Fs = S3.features(ES, W, P, nh, np.arange(len(MS)), MS.label.values, MS.syllable.values, MS.nb_prev.values, MS.nb_next.values)
        PS[t] = mfull.p(S3.X_of(Fs))
        print(f'[{time.time()-T0:.0f}s] {t} thr {thr[t]:.5f}', flush=True)
    assert (borg['T']['row'] == borg['L']['row']).all()
    kd = borg['T']['kind']; kv5 = MA.keep_v5.values[borg['T']['row']] == 1
    c = (borg['T']['p'] >= thr['T']) & (borg['L']['p'] >= thr['L'])
    mn = np.minimum(borg['T']['p'], borg['L']['p'])
    J = {}
    for sub, mm in (('keep_or_high', np.ones(len(kd), bool)), ('keep_v5', kv5)):
        J[sub] = dict(n_pos=int(((kd == 'pos') & mm).sum()), accept_pos=round(float(c[(kd == 'pos') & mm].mean()), 5),
                      far_hom=round(float(c[(kd == 'hom') & mm].mean()), 6), far_slip=round(float(c[(kd == 'slip') & mm].mean()), 6),
                      auc_min_pos_vs_slip=round(S5.auc(mn[(kd == 'pos') & mm], mn[(kd == 'slip') & mm]), 5),
                      auc_min_pos_vs_hom=round(S5.auc(mn[(kd == 'pos') & mm], mn[(kd == 'hom') & mm]), 5))
    cert = (PS['T'] >= thr['T']) & (PS['L'] >= thr['L'])
    out = pd.DataFrame(dict(cell_uid=MS.cell_uid.values, p_T_rt=PS['T'], p_L_rt=PS['L'], cert_rt=cert.astype(int)))
    out.to_pickle(RT / 'hand_stt_rt.pkl')
    # số ô H2/H4 STT nếu dùng bộ kiểm học lại (H1 v2, n_hum ≥ 3 như TN3)
    X = pd.read_pickle(S3.SP / 'r6/policy_v3/out/base_v2.pkl')[['cell_uid', 'lobo_nh']]
    MN = pd.read_pickle(BIG / 'masks_new.pkl')[['cell_uid', 'set8', 'H1n', 'H1o', 'SIMGo', 'SIMGn']]
    Z = MN.merge(X, on='cell_uid').merge(out[['cell_uid', 'cert_rt']], on='cell_uid', how='left')
    Z['simg_rt'] = (Z.cert_rt.fillna(0) == 1) & (Z.lobo_nh.fillna(0) >= 3)
    cnt = {s: dict(H4_cu=int((g.H1o & g.SIMGo).sum()), H4_v2_dropin=int((g.H1n & g.SIMGn).sum()), H4_v2_hoc_lai=int((g.H1n & g.simg_rt).sum()))
           for s, g in Z[Z.set8.isin(['stt2', 'stt4', 'stt11'])].groupby('set8')}
    old = json.load(open(LABOUT / 's05_hand.json'))
    res = dict(q=Q, thr_hoc_lai=thr, thr_cu=json.load(open(HL / 'h03_dir_Kinh.json'))['thr'],
               borg_joint=dict(hoc_lai=J, cu=dict(keep_v5=old['borg_joint']['keep_v5|cu'], keep_or_high=old['borg_joint']['keep_or_high|cu']),
                               dropin=dict(keep_v5=old['borg_joint']['keep_v5|moi'], keep_or_high=old['borg_joint']['keep_or_high|moi'])),
               stt_cert_rate=dict(cu=old['stt_cert_rate']['old'], dropin=old['stt_cert_rate']['new'], hoc_lai=round(float(cert.mean()), 4)),
               stt_H4=cnt, sec=round(time.time() - T0))
    json.dump(res, open(LABOUT / 's10b_hand_retrain.json', 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, default=float)[:3000])


if __name__ == '__main__':
    main()
