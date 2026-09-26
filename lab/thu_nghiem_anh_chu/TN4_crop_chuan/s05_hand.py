#!/usr/bin/env python
"""s05_hand.py — bộ kiểm chữ VIẾT TAY LOBO-sách (r5/hand_lobo, mô hình + nguyên mẫu + hiệu chuẩn CHỈ từ Borg Kinh) chấm
CROP CHUẨN: (a) 52.707 ô GOLD STT -> chứng nhận q = 0,00015 (lobo_cert_00015) + n_hum; (b) Borg DungLy (sách GIỮ NGOÀI, nhãn
NGƯỜI) -> AUC dương vs đồng âm / trượt, trước (crop cũ) ↔ sau (crop chuẩn), trên ô DungLy có crop chuẩn (keep ∨ keep_high)
và tập con keep_v5. Bộ kiểm GIỮ NGUYÊN; chỉ đổi ảnh. MPS cho CNN. 0 API, repo chỉ đọc.
Invariant: đặc trưng cũ đã lưu -> p_T/p_L tái lập stt_cert_Kinh.pkl (≤ 1e-6); nhúng lại img_stt cũ trùng emb_*_Kinh_stt.
Ra: TN4/<ver>/hand_stt.pkl, lab/.../out_<ver>/s05_hand.json"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ver import BIG, LABOUT  # noqa
import s03_vft as S3  # noqa  (prep, load_imgs, features, X_of, Logit, uid_path, emb)
SP = S3.SP
HL = SP / 'r5/hand_lobo/out'
T0 = time.time()
INV = {}
Q = '0.00015'


def wprior(y):
    return np.where(y == 1, 1.0, 0.05 / 0.95 * (y == 1).sum() / max((y == 0).sum(), 1))


def load_net(ft):
    from vflib import Net  # noqa
    ck = torch.load(HL / f'model_{ft}.pt', map_location='cpu')
    net = Net(widths=tuple(ck['widths']))
    net.load_state_dict({k: v.float() for k, v in ck['net'].items()})
    return net.to(S3.DEV).eval()


def auc(pos, neg):
    from scipy.stats import rankdata
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    MA = pd.read_pickle(HL / 'meta_all.pkl')
    MS = pd.read_pickle(HL / 'meta_stt.pkl')
    MI = pd.read_pickle(S3.OUTV / 'meta_ihr.pkl'); MI['partB'] = S3.pages_split(MI.page.values)
    TH = json.load(open(HL / 'h03_dir_Kinh.json'))['thr']
    old_cert = pd.read_pickle(HL / 'stt_cert_Kinh.pkl')
    assert (old_cert.cell_uid.values == MS.cell_uid.values).all()
    # ảnh mới: STT GOLD + Borg (ô có crop chuẩn)
    XS = S3.load_imgs([BIG / 'crops' / S3.uid_path(u) if (BIG / 'crops' / S3.uid_path(u)).exists() else None for u in MS.cell_uid])
    bpath = [BIG / 'aux_borg' / f'{b}/{p}/c{int(c):02d}_{int(i):04d}.png' for b, p, c, i in zip(MA.book, MA.page, MA.col, MA.idx)]
    hasB = np.array([p.exists() for p in bpath])
    INV['borg_rows_with_new_crop'] = int(hasB.sum())
    XB = S3.load_imgs([p if h else None for p, h in zip(bpath, hasB)])
    XS_old = np.load(HL / 'img_stt.npy')
    out = pd.DataFrame({'cell_uid': MS.cell_uid.values})
    RB = {}
    for t in 'TL':
        ft = f'{t}_Kinh'
        net = load_net(ft)
        e_old = S3.emb(net, XS_old[:2000]).astype(np.float32)
        E_stt_saved = np.load(HL / f'emb_{ft}_stt.f16.npy').astype(np.float32)
        INV[f'reembed_stt_{t}_maxabs'] = float(np.abs(e_old - E_stt_saved[:2000]).max())
        ZS = S3.emb(net, XS).astype(np.float32); ZB = S3.emb(net, XB).astype(np.float32)
        del net
        W = np.load(HL / f'W_{ft}.f16.npy').astype(np.float32)
        EA = np.load(HL / f'emb_{ft}_all.f16.npy').astype(np.float32)
        EI = np.load(HL / f'emb_{ft}_ihr.f16.npy').astype(np.float32)
        single = MA.single.values == 1
        inb = (MA.book.values == 'SachKinhThayCaBinh')
        acc = np.zeros((len(S3.U), W.shape[1]), np.float32); nh = np.zeros(len(S3.U), np.int32)
        for r in np.nonzero(inb & (MA.keep.values == 1) & (MA.fold.values != 4) & single)[0]:
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
        Fc = pd.read_pickle(HL / f'feat_{ft}_cal.pkl')
        keep_c = Fc.in_univ.values == 1
        yc = (Fc.kind.values == 'pos').astype(float); w = wprior(yc)
        m = S3.Logit().fit(S3.X_of(Fc)[keep_c], yc[keep_c], w=w[keep_c])
        Fs_old = pd.read_pickle(HL / f'feat_{ft}_stt.pkl')
        p_old = m.p(S3.X_of(Fs_old))
        INV[f'p_{t}_old_reproduce_maxabs'] = float(np.abs(p_old - old_cert[f'p_{t}'].values).max())
        Fs_new = S3.features(ZS, W, P, nh, np.arange(len(MS)), MS.label.values, MS.syllable.values, MS.nb_prev.values, MS.nb_next.values)
        out[f'p_{t}_old'] = p_old; out[f'p_{t}_new'] = m.p(S3.X_of(Fs_new))
        out[f'nh_{t}_new'] = Fs_new.n_hum.values; out[f'nh_{t}_old'] = Fs_old.n_hum.values
        # ---- Borg DungLy (giữ ngoài): cặp như h02 test, chỉ hàng có crop chuẩn
        Ft = pd.read_pickle(HL / f'feat_{ft}_test.pkl')
        sel = hasB[Ft.row.values] & (Ft.in_univ.values == 1)
        Ftn = Ft[sel].reset_index(drop=True)
        ZBr = np.zeros((len(MA), ZB.shape[1]), np.float32); ZBr[hasB] = ZB[hasB]
        Fn = S3.features(ZBr, W, P, nh, Ftn.row.values, Ftn.cand.values, Ftn.syllable.values, Ftn.nbp.values, Ftn.nbn.values)
        po, pn = m.p(S3.X_of(Ftn)), m.p(S3.X_of(Fn))
        kd = Ftn.kind.values; kv5 = MA.keep_v5.values[Ftn.row.values] == 1
        for sub, mm in (('keep_or_high', np.ones(len(Ftn), bool)), ('keep_v5', kv5)):
            for nm, pp in (('cu', po), ('moi', pn)):
                RB[f'{t}|{sub}|{nm}'] = dict(
                    n_pos=int(((kd == 'pos') & mm).sum()),
                    auc_pos_vs_hom=round(auc(pp[(kd == 'pos') & mm], pp[(kd == 'hom') & mm]), 5),
                    auc_pos_vs_slip=round(auc(pp[(kd == 'pos') & mm], pp[(kd == 'slip') & mm]), 5),
                    accept_pos=round(float((pp[(kd == 'pos') & mm] >= TH[t][Q]).mean()), 5),
                    accept_hom=round(float((pp[(kd == 'hom') & mm] >= TH[t][Q]).mean()), 6),
                    accept_slip=round(float((pp[(kd == 'slip') & mm] >= TH[t][Q]).mean()), 6))
        RB[f'{t}|rows'] = Ftn.row.values.tolist()[:0]
        out.attrs[f'borg_{t}'] = dict(row=Ftn.row.values, kind=kd, kv5=kv5, po=po, pn=pn)
        print(f'[{time.time()-T0:.0f}s] {t}', flush=True)
    for s in ('old', 'new'):
        out[f'cert_{s}'] = ((out[f'p_T_{s}'] >= TH['T'][Q]) & (out[f'p_L_{s}'] >= TH['L'][Q])).astype(int)
    INV['cert_old_eq_saved'] = float((out.cert_old.values == old_cert[f'cert_{Q}'].values).mean())
    # joint cả hai mô hình trên Borg (cert = T ∧ L ở ngưỡng q)
    J = {}
    bt, bl = out.attrs['borg_T'], out.attrs['borg_L']
    assert (bt['row'] == bl['row']).all()
    for sub, mm in (('keep_or_high', np.ones(len(bt['row']), bool)), ('keep_v5', bt['kv5'])):
        for nm, a, b in (('cu', bt['po'], bl['po']), ('moi', bt['pn'], bl['pn'])):
            c = (a >= TH['T'][Q]) & (b >= TH['L'][Q])
            kd = bt['kind']
            J[f'{sub}|{nm}'] = dict(n_pos=int(((kd == 'pos') & mm).sum()),
                                    accept_pos=round(float(c[(kd == 'pos') & mm].mean()), 5),
                                    far_hom=round(float(c[(kd == 'hom') & mm].mean()), 6),
                                    far_slip=round(float(c[(kd == 'slip') & mm].mean()), 6),
                                    auc_min_pos_vs_slip=round(auc(np.minimum(a, b)[(kd == 'pos') & mm], np.minimum(a, b)[(kd == 'slip') & mm]), 5),
                                    auc_min_pos_vs_hom=round(auc(np.minimum(a, b)[(kd == 'pos') & mm], np.minimum(a, b)[(kd == 'hom') & mm]), 5))
    out.attrs = {}
    out.to_pickle(BIG / f'hand_stt{S3.SUF}.pkl')
    res = dict(invariants=INV, borg_per_model={k: v for k, v in RB.items() if not k.endswith('|rows')}, borg_joint=J,
               stt_cert_rate=dict(old=round(float(out.cert_old.mean()), 4), new=round(float(out.cert_new.mean()), 4)),
               sec=round(time.time() - T0))
    json.dump(res, open(LABOUT / f's05_hand{S3.SUF}.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False)[:3000])


if __name__ == '__main__':
    main()
