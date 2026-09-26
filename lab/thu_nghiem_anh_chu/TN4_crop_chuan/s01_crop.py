#!/usr/bin/env python
"""s01_crop.py — áp CROP CHUẨN (cclib.canon, thiết kế đóng băng) cho từng ô + dựng lại crop CŨ (luật save_crop) để đo
trên cùng thang. 0 API, CPU, ≤ 4 worker. Repo CHỈ ĐỌC; ghi measure_out/_thu_nghiem_anh_chu/TN4/.

Phần (--part):
  gold : mọi ô GOLD của dataset/_ALL/labels.csv (8 bộ). Ghi crops/<cell_uid>.png (khung vuông, nền giấy gốc) và
         crops128/<cell_uid>.png (128×128).
  aux  : ô KHÔNG phải GOLD của 4 sách trong harness cells_eval.csv (L16, TK, L83, KVK) — cần cho kênh ảnh viss
         (xếp hạng lại dựng trên MỌI tier) và hiệu chuẩn. Ghi aux/<cell_uid>.png.
  borg : ô Borg 'real' có crop và (keep ∨ keep_high) (nhãn NGƯỜI) — ghi aux_borg/<book>/<page>/c<col>_<idx>.png.
Láng giềng cột: bản ghi build mọi tầng (prepared/<Book>/dataset_out/labels.csv; STT: dataset_out/labels.csv), như
PASS 2 / b_recrop.py. Crop cũ: prev/next theo đúng luật b_recrop (xếp tâm y, khớp bbox) — kiểm md5 byte trên mẫu.
Khe NGƯỜI (chỉ L16/TK có trang GT): hộp cột người vẽ + mẫu vị trí khe (build_summary mau_khe_t) -> đo mực ngoài khe
người / độ phủ mực khe người cho crop cũ và crop mới (cùng ngưỡng T của ô).
Ra: TN4/cells_<part>.pkl (1 hàng/ô), TN4/s01_<part>.json
Chạy: .venv/bin/python lab/thu_nghiem_anh_chu/TN4_crop_chuan/s01_crop.py --part gold --workers 4 [--sets L16,TK] [--limit-pages N]
"""
import argparse, csv, hashlib, json, os, sys, tempfile, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
from ver import BIG, BIGROOT, LABOUT, LIBNAME, FREEZE, VER  # noqa
PREP = {'stt2': 'SachThanhTruyen2', 'stt4': 'SachThanhTruyen4', 'stt11': 'SachThanhTruyen11'}
ORIGINAL = {'LucVanTien1883', 'KimVanKieu1884', 'Chrestomathie1872', 'LucVanTien1916', 'TruyenKieu1872'}
BUILD = {'SachThanhTruyen': REPO / 'dataset_out/labels.csv', **{s: REPO / 'prepared' / s / 'dataset_out/labels.csv' for s in ORIGINAL}}
AB = {'Chrestomathie1872': 'Chr', 'LucVanTien1883': 'L83', 'KimVanKieu1884': 'KVK', 'LucVanTien1916': 'L16', 'TruyenKieu1872': 'TK'}
IHR = ('LucVanTien1916', 'TruyenKieu1872')
BORG_DATA = {'SachKinhThayCaBinh': REPO / 'data/SachKinhThayCaBinh', 'SachDungLyHoThan': REPO / 'data/SachDungLyHoThan'}
rd = lambda p, **k: pd.read_csv(p, dtype=str, keep_default_na=False, **k)


def uid_path(uid):
    a = uid.split('/')
    return '/'.join(a[:3]) + '/' + '_'.join(a[3:]) + '.png'


# ----------------------------------------------------------------------------- khe người (L16/TK)
def human_slots():
    S = json.load(open(SP / 'kim_bottleneck/harness/build_summary.json'))
    out = {}
    for book in IHR:
        mau = S[book]['mau_khe_t']
        pm = {p['page_name']: p['page_id'] for p in json.loads((REPO / 'prepared' / book / 'manifest.json').read_text())['pages']}
        gt = {}
        for r in csv.DictReader(open(REPO / 'data' / book / 'manifest.tsv', encoding='utf-8'), delimiter='\t'):
            if r['col_index'] == '' or r['part'] == '':
                continue
            gt[(r['page_id'], int(r['col_index']))] = tuple(map(float, r['col_bbox_xywh'].split(',')))
        lo = [(mau[0] - (mau[1] - mau[0]) / 2) if k == 0 else (mau[k - 1] + mau[k]) / 2 for k in range(14)]
        hi = [(mau[13] + (mau[13] - mau[12]) / 2) if k == 13 else (mau[k] + mau[k + 1]) / 2 for k in range(14)]
        out[book] = dict(pm=pm, gt=gt, lo=lo, hi=hi)
    return out


def hslot_rect(HS, book, page, column, syl_idx):
    if book not in HS:
        return None
    h = HS[book]
    pid = h['pm'].get(page)
    box = h['gt'].get((pid, int(column) - 1)) if pid else None
    k = int(syl_idx) if syl_idx not in ('', None) else -1
    if box is None or not (0 <= k < 14):
        return None
    x, y, w, hh = box
    return (x, y + hh * h['lo'][k], x + w, y + hh * h['hi'][k])


# ----------------------------------------------------------------------------- worker
_BD = _CL = None


def _init():
    global _BD, _CL
    cv2.setNumThreads(1)
    from pipeline.align_engine import build_dataset as bd
    import importlib, cclib
    _BD, _CL = bd, cclib
    _CL = importlib.import_module(LIBNAME)
    _CL.old_crop = cclib.old_crop


def _cq(gray):
    from pipeline.align_engine.crop_quality import measure
    return measure(gray)


def do_page(task):
    part, book_set, book, page, pdir, orig_path, rows, colrows, write_root, write128, md5_uids = task
    bd, CL = _BD, _CL
    img = cv2.imread(str(Path(pdir) / 'pages' / f'{page}.png'), cv2.IMREAD_COLOR)
    if img is None:
        return [dict(key=r['key'], status='NO_PAGE') for r in rows]
    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src, src_kind = None, 'processed'
    if orig_path == 'LOAD':
        src = bd.load_original_page(Path(pdir), page, shape_like=img.shape[:2])
    elif orig_path:
        o = cv2.imread(orig_path, cv2.IMREAD_COLOR)
        if o is not None and o.shape[:2] == img.shape[:2]:
            src = o
    if src is not None:
        src_kind = 'original'
    else:
        src = img
    # tâm cột + hộp cột
    colboxes = {c: [json.loads(b) if isinstance(b, str) else b for b in bb] for c, bb in colrows.items()}
    col_cx = {c: float(np.median([(b[0] + b[2]) / 2.0 for b in bb])) for c, bb in colboxes.items() if bb}
    pitches = [CL.col_pitch(CL.distinct_col_boxes(bb)) for bb in colboxes.values() if bb]
    pitches = [p for p in pitches if p]
    page_pitch = float(np.median(pitches)) if pitches else None
    out = []
    tmpd = None
    for r in rows:
        o = dict(key=r['key'], src_kind=src_kind)
        bbox = r['bbox']
        col = r['column']
        cb = colboxes.get(col, [])
        others_cx = [v for c, v in col_cx.items() if c != col]
        try:
            res = CL.canon(G, src, bbox, cb, others_cx, page_pitch)
        except Exception as e:  # noqa
            o.update(status=f'ERR:{type(e).__name__}:{e}'[:120]); out.append(o); continue
        for k in ('p', 'p_src', 'wc', 'xc', 'top_kind', 'bot_kind', 'T', 'n_comp', 'n_own', 'n_cut_comp', 'n_tip', 'n_line',
                  'mass', 'ink_norm', 'cut_ratio', 'edge', 'flags', 'status', 'tall', 'E', 'Wd', 'ink_cx', 'ink_cy',
                  'foreign_erased_px'):
            if k in res:
                o[k] = res[k]
        for k in ('ink_box', 'sq', 'win', 'band_x'):
            if k in res:
                o[k] = json.dumps([float(v) for v in res[k]])
        if res.get('status') == 'ok':
            q = _cq(res['tight_img'])
            o.update({f'new_{k}': v for k, v in q.items()})
            fn = write_root / uid_path(r['key']) if part != 'borg' else write_root / r['relpath']
            fn.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fn), res['sq_img'])
            o['new_file'] = str(fn.relative_to(BIG))
            if write128:
                f2 = write128 / uid_path(r['key'])
                f2.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(f2), res['sq128'])
        # ---- crop cũ (luật save_crop), chỉ khi có láng giềng theo luật b_recrop
        if part != 'borg':
            oc = CL.old_crop(img, G, bbox, r.get('prev_b'), r.get('next_b'))
            if oc is not None:
                q = _cq(oc['gray'])
                o.update({f'old_{k}': v for k, v in q.items()})
                o['old_tall'] = bool(oc['gray'].shape[0] > 1.8 * max(oc['gray'].shape[1], 1))
                o['old_rect'] = json.dumps([float(v) for v in oc['rect']])
                o['old_h'], o['old_w'] = oc['gray'].shape
            # ---- khe người
            hr = r.get('hrect')
            if hr is not None and res.get('status') == 'ok' and oc is not None:
                T = res['_T']
                hx0, hy0, hx1, hy1 = (int(round(v)) for v in hr)
                hx0, hy0 = max(0, hx0), max(0, hy0)
                hx1, hy1 = min(G.shape[1], hx1), min(G.shape[0], hy1)
                ink_h = int((G[hy0:hy1, hx0:hx1] < T).sum()) if hx1 > hx0 and hy1 > hy0 else 0
                px0, py0, Mp = res['M_page']
                ys, xs = np.nonzero(Mp); ys = ys + py0; xs = xs + px0
                inn = (xs >= hx0) & (xs < hx1) & (ys >= hy0) & (ys < hy1)
                ox0, oy0, ox1, oy1 = (int(v) for v in oc['rect'])
                ok_ = (G[oy0:oy1, ox0:ox1] < T) & ~oc['carved']
                ys2, xs2 = np.nonzero(ok_); ys2 = ys2 + oy0; xs2 = xs2 + ox0
                inn2 = (xs2 >= hx0) & (xs2 < hx1) & (ys2 >= hy0) & (ys2 < hy1)
                o.update(h_ink=ink_h, new_in=int(inn.sum()), new_tot=int(len(xs)), old_in=int(inn2.sum()), old_tot=int(len(xs2)))
            # ---- md5 byte của crop cũ (mẫu)
            if r['key'] in md5_uids and r.get('file'):
                if tmpd is None:
                    tmpd = Path(tempfile.mkdtemp(prefix='tn4_', dir=str(BIGROOT / '_tmp')))
                tp = tmpd / 'x.png'
                io = bd.load_original_page(Path(pdir), page, shape_like=img.shape[:2]) if book_set in ORIGINAL else None
                qq = bd.save_crop(img, G, bbox, 0.12, tp, tighten=True, prev_bbox=r.get('prev_b'), next_bbox=r.get('next_b'),
                                  img_orig=io)
                m1 = hashlib.md5(tp.read_bytes()).hexdigest() if qq else ''
                m2 = hashlib.md5(Path(r['file']).read_bytes()).hexdigest() if Path(r['file']).exists() else '-'
                o['md5_eq'] = bool(m1 and m1 == m2)
        out.append(o)
    if tmpd is not None:
        for f in tmpd.iterdir():
            f.unlink()
        tmpd.rmdir()
    return out


# ----------------------------------------------------------------------------- dựng nhiệm vụ
def load_build(bs):
    B = rd(BUILD[bs], usecols=['book', 'page', 'column', 'bbox'])
    return B[B.bbox != '']


def neighbours_by_rule(col_bboxes, bbox):
    ys = sorted(((b[1] + b[3]) / 2.0, b) for b in col_bboxes)
    pos = [i for i, (_, b) in enumerate(ys) if list(b) == list(bbox)]
    if not pos:
        return None, None
    i = pos[0]
    return (ys[i - 1][1] if i > 0 else None), (ys[i + 1][1] if i < len(ys) - 1 else None)


def tasks_cells(part, cells, write_root, write128, md5_uids, HS, limit_pages=None):
    tasks = []
    for bs, sub in cells.groupby('book_set'):
        B = load_build(bs)
        Bg = {k: g for k, g in B.groupby(['book', 'page'])}
        for (bk, pg), g in sub.groupby(['book', 'page']):
            pdir = REPO / 'prepared' / PREP.get(bk, bs)
            bp = Bg.get((bk, pg))
            colrows = defaultdict(list)
            if bp is not None:
                for c, b in zip(bp.column, bp.bbox):
                    colrows[c].append(json.loads(b))
            rows = []
            for r in g.itertuples():
                bb = json.loads(r.bbox)
                pv, nx = neighbours_by_rule(colrows.get(r.column, []), bb)
                d = dict(key=r.cell_uid, bbox=bb, column=r.column, prev_b=pv, next_b=nx,
                         file=str(REPO / 'dataset/_ALL' / r.image) if getattr(r, 'image', '') else '')
                if bs in IHR:
                    d['hrect'] = hslot_rect(HS, bs, pg, r.column, r.syl_idx)
                rows.append(d)
            tasks.append((part, bs, bk, pg, str(pdir), 'LOAD' if bs in ORIGINAL else '', rows, dict(colrows),
                          write_root, write128, md5_uids))
    if limit_pages:
        tasks = tasks[:limit_pages]
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['gold', 'aux', 'borg'])
    ap.add_argument('--sets', default='')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit-pages', type=int, default=0)
    ap.add_argument('--md5-per-set', type=int, default=150)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    T0 = time.time()
    BIG.mkdir(parents=True, exist_ok=True); (BIGROOT / '_tmp').mkdir(exist_ok=True); LABOUT.mkdir(exist_ok=True)
    fr = json.load(open(FREEZE))
    md5_lib = hashlib.md5((HERE / f'{LIBNAME}.py').read_bytes()).hexdigest()
    inv = dict(design_frozen=md5_lib == fr['cclib_md5'])
    assert inv['design_frozen'], 'cclib.py đã đổi sau khi đóng băng'
    HS = human_slots()
    meta = dict(part=a.part)
    if a.part in ('gold', 'aux'):
        if a.part == 'gold':
            L = rd(REPO / 'dataset/_ALL/labels.csv')
            L = L[L.tier == 'GOLD'].copy()
            L['syl_idx'] = L.cell_uid.str.split('/').str[-1].str[1:]
            root, r128 = BIG / 'crops', BIG / 'crops128'
        else:
            L = rd(SP / 'kim_bottleneck/harness/cells_eval.csv')
            L = L[(L.tier != 'GOLD') & (L.bbox != '')].copy()
            L['book_set'] = L.cell_uid.str.split('/').str[0]
            L['book'] = L.cell_uid.str.split('/').str[1]
            L['image'] = ''
            root, r128 = BIG / 'aux', None
        L['set8'] = np.where(L.book_set == 'SachThanhTruyen', L.book, L.book_set.map(AB).fillna(''))
        if a.sets:
            L = L[L.set8.isin(a.sets.split(','))]
        rng = np.random.default_rng(20260926)
        md5_uids = set()
        if a.part == 'gold':
            for s, g in L.groupby('set8'):
                md5_uids |= set(g.cell_uid.values[rng.permutation(len(g))[:a.md5_per_set]])
        tasks = tasks_cells(a.part, L, root, r128, md5_uids, HS, a.limit_pages or None)
        meta['n_cells'] = int(sum(len(t[6]) for t in tasks)); meta['n_pages'] = len(tasks)
    else:
        bc = rd(SP / 'r4/borg_align/borg_cells.csv')
        nb = bc[bc.kind != 'skip']
        sel = bc[(bc.kind == 'real') & (bc.crop_path != '') & ((bc.keep == '1') | (bc.keep_high == '1'))]
        tasks = []
        for (bk, pg), g in sel.groupby(['book', 'page']):
            man = {p['page_name']: p for p in json.load(open(REPO / 'prepared' / bk / 'manifest.json'))['pages']}
            orig = str(BORG_DATA[bk] / man[pg]['source_file'])
            gb = nb[(nb.book == bk) & (nb.page == pg)]
            colrows = defaultdict(list)
            for c, b in zip(gb.col, gb.bbox):
                colrows[c].append(json.loads(b))
            rows = [dict(key=f'{bk}/{pg}/{r.col}/{r.idx}', bbox=json.loads(r.bbox), column=r.col,
                         relpath=f'{bk}/{pg}/c{int(r.col):02d}_{int(r.idx):04d}.png') for r in g.itertuples()]
            tasks.append(('borg', bk, bk, pg, str(REPO / 'prepared' / bk), orig, rows, dict(colrows), BIG / 'aux_borg', None, set()))
        if a.limit_pages:
            tasks = tasks[:a.limit_pages]
        meta['n_cells'] = int(sum(len(t[6]) for t in tasks)); meta['n_pages'] = len(tasks)
    print(f'[{time.time()-T0:.0f}s] {a.part}: {meta["n_cells"]} ô / {meta["n_pages"]} trang', flush=True)
    res = []
    with Pool(a.workers, initializer=_init) as pool:
        for k, chunk in enumerate(pool.imap_unordered(do_page, tasks, chunksize=2)):
            res.extend(chunk)
            if k % 200 == 0:
                print(f'[{time.time()-T0:.0f}s] trang {k}/{len(tasks)} ô {len(res)}', flush=True)
    D = pd.DataFrame(res)
    tag = a.part + (f'_{a.tag}' if a.tag else '')
    D.to_pickle(BIG / f'cells_{tag}.pkl')
    meta['status'] = D.status.value_counts().to_dict() if 'status' in D else {}
    if 'md5_eq' in D:
        meta['md5_sample'] = dict(n=int(D.md5_eq.notna().sum()), eq=int(D.md5_eq.fillna(False).astype(bool).sum()))
    meta['sec'] = round(time.time() - T0)
    meta['invariants'] = inv
    json.dump(meta, open(LABOUT / f's01_{tag}.json', 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(meta, ensure_ascii=False, default=str)[:1500])


if __name__ == '__main__':
    main()
