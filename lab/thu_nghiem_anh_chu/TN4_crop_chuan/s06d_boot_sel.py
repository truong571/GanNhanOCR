#!/usr/bin/env python
"""s06d_boot_sel.py — ĐỘ NHẠY hậu kiểm: số ô H4 giữ ở L16/TK dao động bao nhiêu chỉ vì CHỌN NGƯỠNG (lưới 41×41 trên ~2.800–4.700 ô
tune phần B, rất ít lỗi)? Lặp B = 300 lần: lấy lại (bootstrap) các TRANG phần B của sách tune -> chọn lại C5 (đúng thủ tục TN1/s06a)
-> áp lên sách thử -> đếm ô H4 giữ và độ chính xác hai vế (khe của crop tương ứng). Cùng một mẫu trang cho cũ / v1 / v2 (ghép cặp).
CPU, 0 API. Ra: lab/.../out_v2/boot_sel.json (và bảng tóm tắt in ra)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s06a_masks as A  # noqa
REPO = HERE.parents[2]
SP = A.SP
BIGR = REPO / 'measure_out/_thu_nghiem_anh_chu/TN4'
sys.path.insert(0, str(SP / 'r6/policy_v3/scripts'))
from vlib import var_eq_plus  # noqa
B = 300
T0 = time.time()


def main():
    X = pd.read_pickle(SP / 'r6/policy_v3/out/base_v2.pkl')
    S = X.book_set.values
    b = lambda c: X[c].fillna(False).astype(bool).values
    gt = X.gt_char.fillna('').values
    HAS = np.isin(S, ['LucVanTien1916', 'TruyenKieu1872']) & (gt != '')
    y_lab = np.zeros(len(X), bool)
    y_lab[HAS] = [var_eq_plus(a, c) for a, c in zip(X.label.values[HAS], gt[HAS])]
    PARTB = b('partB'); yvar = b('y_var'); page = X.page.fillna('').values
    ady, adx, visz = X.ady.values.astype(float), X.adx.values.astype(float), X.vis_z.values.astype(float)
    visf = lambda v: np.nan_to_num(visz, nan=np.inf) <= v + 1e-12
    BC = {'T': (np.nan_to_num(ady, nan=-1) > 0.6) | visf(-1.0),
          'L': (np.nan_to_num(ady, nan=-1) > 0.5) | (np.nan_to_num(adx, nan=-1) > 0.5) | visf(-0.5)}
    cqf = X.crop_quality_flag.fillna('').values
    A0o = b('int_foreign') | b('blank') | b('truncated') | b('geo_f_dup_bbox')
    V = {}
    for ver in ('v1', 'v2'):
        MN = pd.read_pickle(BIGR / ver / 'masks_new.pkl')
        G4 = pd.read_pickle(BIGR / ver / 'gold_tn4.pkl').set_index('cell_uid').reindex(X.cell_uid)
        VG = pd.read_pickle(BIGR / ver / 'vft_gold.pkl').set_index('cell_uid').reindex(X.cell_uid)
        vT, vL, _ = A.load_viss(BIGR / ver / 'rerank_new')
        vs = X[['cell_uid']].merge(vT, on='cell_uid', how='left').merge(vL, on='cell_uid', how='left')
        gb = lambda c: G4[c].fillna(False).astype(bool).values
        V[ver] = dict(H1=MN.H1n.values, slot=MN.slot_new.astype(str).values,
                      A0=b('int_foreign') | b('geo_f_dup_bbox') | gb('f_blank') | gb('f_cut'),
                      pw={'T': np.round(VG.p_wood_T_new.values.astype(float), 5), 'L': np.round(VG.p_wood_L_new.values.astype(float), 5)},
                      vs={'T': vs.viss_T.fillna(0).values, 'L': vs.viss_L.fillna(0).values}, TA=MN.TA_OK.values)
    MN = pd.read_pickle(BIGR / 'v2' / 'masks_new.pkl')
    V['cu'] = dict(H1=MN.H1o.values, slot=MN.slot_old.astype(str).values, A0=A0o,
                   pw={'T': X.p_wood_T.values.astype(float), 'L': X.p_wood_L.values.astype(float)},
                   vs={'T': X.viss_T.fillna(0).values, 'L': X.viss_L.fillna(0).values}, TA=MN.TA_OK.values)
    DIRS = {'T': ('TruyenKieu1872', 'LucVanTien1916'), 'L': ('LucVanTien1916', 'TruyenKieu1872')}
    rng = np.random.default_rng(20260926)
    OUT = {}
    for d, (tune, test) in DIRS.items():
        base_tu = (S == tune) & HAS & PARTB
        pages = np.unique(page[base_tu])
        te = (S == test) & HAS
        res = {k: [] for k in V}
        for bi in range(B + 1):
            if bi == 0:
                w = np.ones(len(X))          # lượt 0 = mẫu gốc (phải trùng s06a)
            else:
                cnt = pd.Series(rng.integers(0, len(pages), len(pages))).value_counts()
                pw_ = dict(zip(pages[cnt.index.values], cnt.values))
                w = np.array([pw_.get(p, 0) for p in page], float)
            for k, v in V.items():
                y = yvar & (v['slot'] == '1')
                tu = base_tu & np.isin(v['slot'], ['0', '1']) & (w > 0)
                # nhân bản theo trọng số trang (bootstrap)
                idx = np.repeat(np.nonzero(tu)[0], w[tu].astype(int))
                m = np.zeros(len(X), bool)
                pwv, vsv, yv, r2 = v['pw'][d][idx], v['vs'][d][idx], y[idx], ~(v['A0'] | BC[d])[idx]
                qs = np.unique(np.quantile(pwv, np.linspace(0, 0.8, 41))); qv = np.unique(np.quantile(vsv, np.linspace(0, 0.8, 41)))
                best = None
                for t1 in qs:
                    a1 = r2 & (pwv >= t1)
                    for t2 in qv:
                        a = a1 & (vsv >= t2)
                        if a.sum() < A.MIN_ACC:
                            continue
                        if yv[a].mean() >= A.TGT and (best is None or a.mean() > best[2]):
                            best = (round(float(t1), 5), round(float(t2), 5), a.mean())
                if best is None:
                    res[k].append((0, np.nan)); continue
                keep = te & v['H1'] & (v['pw'][d] >= best[0]) & (v['vs'][d] >= best[1]) & v['TA']
                ok = y_lab & (v['slot'] == '1')
                res[k].append((int(keep.sum()), float(ok[keep].mean()) if keep.any() else np.nan))
        o = {}
        for k, r in res.items():
            a = np.array(r, float)
            o[k] = dict(goc_giu=int(a[0, 0]), goc_cx=round(a[0, 1], 5), giu_trung_vi=float(np.median(a[1:, 0])),
                        giu_p5=float(np.percentile(a[1:, 0], 5)), giu_p95=float(np.percentile(a[1:, 0], 95)),
                        cx_trung_vi=round(float(np.nanmedian(a[1:, 1])), 5), cx_p5=round(float(np.nanpercentile(a[1:, 1], 5)), 5))
        for k in ('v1', 'v2'):
            dk = np.array(res[k])[1:, 0] - np.array(res['cu'])[1:, 0]
            o[f'{k}_tru_cu'] = dict(trung_vi=float(np.median(dk)), p5=float(np.percentile(dk, 5)), p95=float(np.percentile(dk, 95)),
                                   ti_le_nhieu_hon=round(float((dk > 0).mean()), 3))
        OUT[test] = o
    OUT['B'] = B; OUT['sec'] = round(time.time() - T0)
    json.dump(OUT, open(HERE / 'out_v2' / 'boot_sel.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(OUT, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
