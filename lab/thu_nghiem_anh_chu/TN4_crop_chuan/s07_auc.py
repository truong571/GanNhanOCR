#!/usr/bin/env python
"""s07_auc.py — bộ kiểm ảnh trên IHR (nhãn NGƯỜI, CHẮC CHẮN THEO MÁY) trước ↔ sau crop chuẩn: AUC tách
  (i) crop đúng khe / sai khe (y = khe = 1 trên ô khe xác định; khe của CHÍNH crop được chấm: cũ = tâm bbox, mới = tâm hộp mực),
  (ii) nhãn đúng / sai (y = V1+(nhãn, chữ người) trên ô khe = 1),
  (iii) hai vế (y = (ii) ∧ khe = 1 trên ô khe xác định).
Điểm: p_wood hướng LOBO (L16 <- mô hình T, TK <- mô hình L) và viss hướng LOBO (L16 <- Tru2Luc, TK <- Luc2Tru).
Tập ô: GOLD, và MỌI tier (cand = nhãn nếu có, không thì chữ OCR — như s02_eval).
CI 95 % bootstrap cụm theo trang (B = 500) cho HIỆU AUC (mới − cũ), ghép cặp cùng ô. CPU, 0 API.
Ra: lab/.../out_<ver>/auc_ihr.csv, s07_auc.json"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ver import BIG, LABOUT, VER  # noqa
from hslot import Slot  # noqa
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
sys.path.insert(0, str(SP / 'r6/policy_v3/scripts'))
from vlib import var_eq_plus  # noqa
import os  # noqa
SUF = '' if os.environ.get('TN4_FRAME', 'sq') == 'sq' else '_' + os.environ['TN4_FRAME']
T0 = time.time()
B = 500
rng = np.random.default_rng(20260926)


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float('nan')
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    d = pd.read_csv(SP / 'kim_bottleneck/harness/cells_eval.csv', dtype=str, keep_default_na=False)
    d = d[d.eval_set == 'ihr_human'].reset_index(drop=True)
    V = pd.read_pickle(BIG / f'vft_ihr{SUF}.pkl')
    assert (V.cell_uid.values == d.cell_uid.values).all()
    # tâm crop mới: GOLD (s02) + tier khác (cells_aux)
    G4 = pd.read_pickle(BIG / 'gold_tn4.pkl')[['cell_uid', 'slot_new']]
    CA = pd.read_pickle(BIG / 'cells_aux.pkl').rename(columns={'key': 'cell_uid'})[['cell_uid', 'ink_cx', 'ink_cy', 'status']]
    d = d.merge(G4, on='cell_uid', how='left').merge(CA, on='cell_uid', how='left')
    for book in ('LucVanTien1916', 'TruyenKieu1872'):
        S = Slot(book)
        m = (d.book == book) & (d.tier != 'GOLD') & (d.status == 'ok')
        sub = d[m]
        d.loc[m, 'slot_new'] = [S.slot(p, c, k, a, q) for p, c, k, a, q in zip(sub.page, sub.column, sub.syl_idx, sub.ink_cx, sub.ink_cy)]
    d['slot_new'] = d.slot_new.fillna('').astype(str)
    d['cand'] = np.where(d.label != '', d.label, d.ocr_char)
    d['y_lab'] = [var_eq_plus(a, g) if g else False for a, g in zip(d.cand, d.gt_char)]
    # viss LOBO
    def vis(dirp):
        vT = pd.read_pickle(dirp / 'pred_viss_Tru2Luc.pkl').set_index('cell_uid').p_kim
        vL = pd.read_pickle(dirp / 'pred_viss_Luc2Tru.pkl').set_index('cell_uid').p_kim
        return vT, vL
    oT, oL = vis(SP / 'kim_bottleneck/rerank/out'); nT, nL = vis(BIG / 'rerank_new')
    isL = (d.book == 'LucVanTien1916').values
    d['pw_old'] = np.where(isL, V.p_wood_T_old, V.p_wood_L_old); d['pw_new'] = np.where(isL, V.p_wood_T_new, V.p_wood_L_new)
    d['vs_old'] = np.where(isL, d.cell_uid.map(oT), d.cell_uid.map(oL)); d['vs_new'] = np.where(isL, d.cell_uid.map(nT), d.cell_uid.map(nL))
    rows = []
    for book, bk in (('LucVanTien1916', 'L16'), ('TruyenKieu1872', 'TK')):
        for cset in ('GOLD', 'moi_tier'):
            base = (d.book == book) & (d.gt_char != '')
            if cset == 'GOLD':
                base &= d.tier == 'GOLD'
            for score in ('pw', 'vs'):
                for task in ('khe', 'nhan', 'hai_ve'):
                    res = {}
                    masks, ys, ss = {}, {}, {}
                    for tag, slotc in (('cu', 'slot_ok'), ('moi', 'slot_new')):
                        sl = d[slotc].astype(str).values
                        if task == 'khe':
                            m = base.values & np.isin(sl, ['0', '1']); y = sl == '1'
                        elif task == 'nhan':
                            m = base.values & (sl == '1'); y = d.y_lab.values
                        else:
                            m = base.values & np.isin(sl, ['0', '1']); y = d.y_lab.values & (sl == '1')
                        masks[tag], ys[tag], ss[tag] = m, y, d[f'{score}_{"old" if tag == "cu" else "new"}'].values.astype(float)
                        res[tag] = auc(ss[tag][m], y[m]); res[f'n_{tag}'] = int(m.sum()); res[f'neg_{tag}'] = int((m & ~y).sum())
                    # bootstrap cụm trang cho hiệu
                    pg = d.page.values
                    up = np.unique(pg[base.values]); pidx = {p: i for i, p in enumerate(up)}
                    diffs = []
                    groups = {p: np.nonzero((pg == p) & base.values)[0] for p in up}
                    for _ in range(B):
                        pick = rng.integers(0, len(up), len(up))
                        idx = np.concatenate([groups[up[k]] for k in pick])
                        a1 = auc(ss['cu'][idx][masks['cu'][idx]], ys['cu'][idx][masks['cu'][idx]])
                        a2 = auc(ss['moi'][idx][masks['moi'][idx]], ys['moi'][idx][masks['moi'][idx]])
                        diffs.append(a2 - a1)
                    diffs = np.array(diffs)
                    rows.append(dict(bo=bk, tap=cset, diem=score, viec=task, auc_cu=round(res['cu'], 4), auc_moi=round(res['moi'], 4),
                                     hieu=round(res['moi'] - res['cu'], 4), hieu_lo=round(float(np.nanpercentile(diffs, 2.5)), 4),
                                     hieu_hi=round(float(np.nanpercentile(diffs, 97.5)), 4),
                                     n_cu=res['n_cu'], am_cu=res['neg_cu'], n_moi=res['n_moi'], am_moi=res['neg_moi']))
    R = pd.DataFrame(rows)
    R.to_csv(LABOUT / f'auc_ihr{SUF}.csv', index=False)
    json.dump(dict(ver=VER, B=B, sec=round(time.time() - T0)), open(LABOUT / f's07_auc{SUF}.json', 'w'))
    print(R.to_string())


if __name__ == '__main__':
    main()
