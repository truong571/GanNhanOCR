#!/usr/bin/env python
"""s02b_cov_core.py — ĐỘ NHẠY của phép đo khe người (thêm sau khi thấy s02: L16 crop mới mất > 25 % "độ phủ mực khe người" ở
~150 ô). Khe người là khe MẪU (vị trí trung vị theo cả sách, hộp cột người vẽ) nên mép khe có thể lệch vài px và mép cột có
nét kẻ. Đo lại độ phủ trên LÕI khe: bỏ 15 % trên + 15 % dưới chiều cao khe và 10 % mỗi bên bề ngang cột. Nếu mất mát nằm ở mép
(mực chữ kề/nét kẻ rơi vào khe mẫu) thì độ phủ LÕI của crop mới ≈ crop cũ; nếu crop mới cắt thật vào chữ thì độ phủ lõi cũng giảm.
Chỉ L16/TK, GOLD, khe cũ = 1. Tính lại crop (cclib v1/v2 theo TN4_VER) trên trang — không ghi ảnh. CPU, 4 worker, 0 API.
Ra: lab/.../out_<ver>/cov_core.json"""
import json, sys, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ver import BIG, LABOUT, LIBNAME  # noqa
import s01_crop as S1  # noqa
REPO = S1.REPO
_CL = None


def _init():
    global _CL
    import importlib, cclib
    cv2.setNumThreads(1)
    _CL = importlib.import_module(LIBNAME)
    _CL.old_crop = cclib.old_crop


def core(r, fy=0.15, fx=0.10):
    x0, y0, x1, y1 = r
    h, w = y1 - y0, x1 - x0
    return (x0 + fx * w, y0 + fy * h, x1 - fx * w, y1 - fy * h)


def do_page(t):
    book, page, rows, colrows = t
    from pipeline.align_engine import build_dataset as bd
    pdir = REPO / 'prepared' / book
    img = cv2.imread(str(pdir / 'pages' / f'{page}.png'), cv2.IMREAD_COLOR)
    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src = bd.load_original_page(pdir, page, shape_like=img.shape[:2])
    src = img if src is None else src
    col_cx = {c: float(np.median([(b[0] + b[2]) / 2.0 for b in bb])) for c, bb in colrows.items() if bb}
    pit = [_CL.col_pitch(_CL.distinct_col_boxes(bb)) for bb in colrows.values() if bb]
    pit = [p for p in pit if p]
    pp = float(np.median(pit)) if pit else None
    out = []
    for r in rows:
        res = _CL.canon(G, src, r['bbox'], colrows.get(r['column'], []), [v for c, v in col_cx.items() if c != r['column']], pp)
        oc = _CL.old_crop(img, G, r['bbox'], r['prev_b'], r['next_b'])
        if res.get('status') != 'ok' or oc is None:
            continue
        T = res['_T']
        px0, py0, Mp = res['M_page']
        ys, xs = np.nonzero(Mp); ys = ys + py0; xs = xs + px0
        ox0, oy0, ox1, oy1 = (int(v) for v in oc['rect'])
        ok_ = (G[oy0:oy1, ox0:ox1] < T) & ~oc['carved']
        ys2, xs2 = np.nonzero(ok_); ys2 = ys2 + oy0; xs2 = xs2 + ox0
        o = dict(key=r['key'])
        for nm, rr in (('full', r['hrect']), ('core', core(r['hrect']))):
            hx0, hy0, hx1, hy1 = (int(round(v)) for v in rr)
            hx0, hy0 = max(0, hx0), max(0, hy0)
            hx1, hy1 = min(G.shape[1], hx1), min(G.shape[0], hy1)
            ink_h = int((G[hy0:hy1, hx0:hx1] < T).sum()) if hx1 > hx0 and hy1 > hy0 else 0
            inn = int(((xs >= hx0) & (xs < hx1) & (ys >= hy0) & (ys < hy1)).sum())
            inn2 = int(((xs2 >= hx0) & (xs2 < hx1) & (ys2 >= hy0) & (ys2 < hy1)).sum())
            o.update({f'{nm}_ink': ink_h, f'{nm}_new': inn, f'{nm}_old': inn2})
        out.append(o)
    return out


def main():
    T0 = time.time()
    D = pd.read_pickle(BIG / 'gold_tn4.pkl')
    D = D[D.set8.isin(['L16', 'TK']) & (D.slot_old_bbox == '1')]
    HS = S1.human_slots()
    tasks = []
    for bs in ('LucVanTien1916', 'TruyenKieu1872'):
        B = S1.load_build(bs)
        Bg = {k: g for k, g in B.groupby(['book', 'page'])}
        for (bk, pg), g in D[D.book_set == bs].groupby(['book', 'page']):
            colrows = defaultdict(list)
            bp = Bg.get((bk, pg))
            if bp is not None:
                for c, b in zip(bp.column, bp.bbox):
                    colrows[c].append(json.loads(b))
            rows = []
            for r in g.itertuples():
                bb = json.loads(r.bbox)
                pv, nx = S1.neighbours_by_rule(colrows.get(r.column, []), bb)
                hr = S1.hslot_rect(HS, bs, pg, r.column, r.syl_idx)
                if hr is not None:
                    rows.append(dict(key=r.cell_uid, bbox=bb, column=r.column, prev_b=pv, next_b=nx, hrect=hr))
            tasks.append((bs, pg, rows, dict(colrows)))
    with Pool(4, initializer=_init) as pool:
        res = [x for ch in pool.imap_unordered(do_page, tasks, chunksize=2) for x in ch]
    R = pd.DataFrame(res).merge(D[['cell_uid', 'set8']].rename(columns={'cell_uid': 'key'}), on='key')
    R['core_drop'] = (R.core_ink > 0) & ((R.core_old - R.core_new) / R.core_ink.clip(lower=1) > 0.25)
    R.to_pickle(BIG / 'cov_core.pkl')
    OUT = {}
    for s, g in R.groupby('set8'):
        o = {}
        for nm in ('full', 'core'):
            ok = g[f'{nm}_ink'] > 0
            cn = g[f'{nm}_new'][ok] / g[f'{nm}_ink'][ok]; co = g[f'{nm}_old'][ok] / g[f'{nm}_ink'][ok]
            o[nm] = dict(n=int(ok.sum()), phu_cu=round(float(co.mean()), 4), phu_moi=round(float(cn.mean()), 4),
                         lt80_cu=int((co < 0.8).sum()), lt80_moi=int((cn < 0.8).sum()),
                         giam_gt25=int(((co - cn) > 0.25).sum()))
        OUT[s] = o
    OUT['sec'] = round(time.time() - T0)
    json.dump(OUT, open(LABOUT / 'cov_core.json', 'w'), indent=1)
    print(json.dumps(OUT))


if __name__ == '__main__':
    main()
