"""t03_errors.py — soi các ô NHẬN SỬA mà khe mới sai (θ=0, mô hình ngoài-sách): loại ứng viên, khe trước/sau, khoảng cách tới kim."""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s3_eval as E
from slotlib import SlotModel
S, L = E.load_all(); cols_all = E.column_state(L)
gs = pd.read_csv(E.SP/'gold_img_audit/measure/gold_suspicion.csv', dtype=str, keep_default_na=False, usecols=['cell_uid','is_gold','geo_f_kim_off','geo_dy_kim','geo_dx_kim'])
th = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
for book, tag in [('LucVanTien1916','T'),('TruyenKieu1872','L')]:
    M = SlotModel(book); Sb = S[S.book_set==book]
    flagged = set(Sb.cell_uid)
    raw = E.propose(Sb, tag, th)
    cols = {k: v for k, v in cols_all.items() if k[0]==book and any(x[0] in raw for x in v)}
    acc, why = E.resolve(raw, flagged, cols, None)
    sn = Sb.set_index(['cell_uid','cand'])
    print('==', book, 'theta', th, 'acc', len(acc))
    for u, (c, bb, m) in acc.items():
        r = sn.loc[(u,'deliv')]
        b0 = M.slot(r.page, r.column, r.syl_idx, json.loads(r.old_bbox))
        b1 = M.slot(r.page, r.column, r.syl_idx, bb)
        if b1['slot_ok'] != 1:
            g = gs[gs.cell_uid==u].iloc[0]
            print(f"  {u.split('/',2)[2]:28s} cand={c:4s} before={b0['slot_ok']!s:1s}(t{b0['slot_tmpl']},k{b0['slot_kim']}) after={b1['slot_ok']!s:1s}(t{b1['slot_tmpl']},k{b1['slot_kim']}) syl={r.syl_idx} dKim={sn.loc[(u,c)].d_kim_pitch} dOld={sn.loc[(u,c)].d_old_pitch} m_new={m:.2f} m_old={float(r[f'm_{tag}']):.2f} dy_kim={g.geo_dy_kim} dx_kim={g.geo_dx_kim} lab={r.label} gt_old={b0['gt_img']} gt_new={b1['gt_img']}")
