#!/usr/bin/env python3
"""s1b_candidates.py — TN2 bước 1b: sinh hộp ỨNG VIÊN cho mỗi ô GOLD bị cờ trượt và CẮT LẠI crop bằng ĐÚNG save_crop.

Ứng viên (định trước, tối đa 5 crop/ô):
  old  : hộp hiện tại (cắt lại để kiểm tái lập: md5 phải trùng tệp giao nộp với sách crop_source=original)
  prev : hộp kề TRÊN trong cột (dịch chỉ số −1; hộp phân biệt kề theo thứ tự y trong bản ghi build mọi tầng)
  next : hộp kề DƯỚI (dịch chỉ số +1)
  kim  : hộp cũ TỊNH TIẾN cho tâm trùng tâm hộp kim của CHÍNH chữ nhãn (chars[nom_idx], kim đọc ra nhãn)
  det  : hộp detector THÔ trong cột gần tâm hộp kim nhất (chỉ nhận khi cách ≤ 0,5 bước cột)
Cắt: pipeline.align_engine.build_dataset.save_crop(img, gray, bbox, pad 0,12, tighten, prev/next = hộp phân biệt kề
theo y của CHÍNH hộp ứng viên trong cột (hộp cũ của ô được bỏ ra, hộp trùng IoU≥0,5 với ứng viên bị bỏ qua),
img_orig = ảnh gốc khi sách khai crop_source original) — y hệt cách gọi của gold_img_audit/integrity_viewer/b_recrop.py.
Chỉ ĐỌC repo; ghi measure_out/_thu_nghiem_anh_chu/TN2/{candidates.csv, crops/<set>/*.png, s1b_summary.json}.
Chạy: .venv/bin/python s1b_candidates.py [--workers 4]
"""
import argparse, hashlib, json, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np, pandas as pd

REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR')
sys.path.insert(0, str(REPO))
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
OUT = REPO / 'measure_out/_thu_nghiem_anh_chu/TN3/tn2_l83kvk'   # TN3: thư mục đầu ra riêng
SETS = ['LucVanTien1883', 'KimVanKieu1884']   # TN3
ORIGINAL = {'LucVanTien1883', 'KimVanKieu1884'}   # TN3: cả hai khai crop_source: original
PREP = {'stt2': 'SachThanhTruyen2', 'stt4': 'SachThanhTruyen4', 'stt11': 'SachThanhTruyen11'}
PAD = 0.12


def build_path(bs):
    return REPO / 'dataset_out/labels.csv' if bs == 'SachThanhTruyen' else REPO / 'prepared' / bs / 'dataset_out/labels.csv'


def prep_dir(bs, book):
    return REPO / 'prepared' / PREP.get(book, bs)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def cyc(b):
    return (b[1] + b[3]) / 2.0


def cxc(b):
    return (b[0] + b[2]) / 2.0


_BD = None


def _init():
    global _BD
    from pipeline.align_engine import build_dataset as bd
    _BD = bd


def neighbours(col_boxes, cand, skip_box=None):
    """prev/next = hộp phân biệt kề theo y của `cand` trong cột; bỏ skip_box (hộp cũ của ô) và hộp IoU≥0,5 với cand."""
    L = []
    removed = False
    for b in col_boxes:
        if skip_box is not None and not removed and list(b) == list(skip_box):
            removed = True; continue
        if iou(b, cand) >= 0.5:
            continue
        L.append(b)
    c = cyc(cand)
    up = [b for b in L if cyc(b) < c]; dn = [b for b in L if cyc(b) >= c]
    prev_b = max(up, key=cyc) if up else None
    next_b = min(dn, key=cyc) if dn else None
    return prev_b, next_b


def do_page(task):
    bs, book, page, cells, col_boxes, kim = task
    bd = _BD
    pdir = prep_dir(bs, book)
    img = cv2.imread(str(pdir / 'pages' / f'{page}.png'), cv2.IMREAD_COLOR)
    if img is None:
        return [dict(cell_uid=c['cell_uid'], cand='old', status='NO_PAGE') for c in cells]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    img_orig = bd.load_original_page(pdir, page, shape_like=img.shape[:2]) if bs in ORIGINAL else None
    rows = []
    cdir = OUT / 'crops' / bs
    cdir.mkdir(parents=True, exist_ok=True)
    for c in cells:
        b0 = [int(v) for v in json.loads(c['bbox'])]
        cb = col_boxes.get(c['column'], [])
        uniq = []
        for b in sorted(cb, key=cyc):
            if not any(list(b) == list(u) for u in uniq):
                uniq.append(b)
        ys = sorted(cyc(b) for b in uniq)
        pitch = float(np.median(np.diff(ys))) if len(ys) > 2 else float(b0[3] - b0[1])
        base = dict(cell_uid=c['cell_uid'], book_set=bs, book=book, page=page, column=c['column'],
                    nom_idx=c['nom_idx'], syl_idx=c['syl_idx'], label=c['label'], ocr_char=c['ocr_char'],
                    syllable=c['syllable'], pitch=round(pitch, 2), old_bbox=json.dumps(b0))
        # ---- kim của chính chữ nhãn + chữ kề (đối thủ)
        kcol = (kim or {}).get(str(int(c['column'])))
        ni = int(c['nom_idx'])
        kbox = None; kchar = ''
        nb = {}
        if kcol:
            ch = kcol['chars']
            if 0 <= ni < len(ch):
                kchar, kbox = ch[ni][0], ch[ni][1]
            for d in (-2, -1, 1, 2):
                j = ni + d
                nb[d] = ch[j][0] if 0 <= j < len(ch) else ''
        base.update(kim_char=kchar, kim_char_ok=int(kchar == c['ocr_char']), kim_bbox=json.dumps(kbox) if kbox else '',
                    nb_m2=nb.get(-2, ''), nb_m1=nb.get(-1, ''), nb_p1=nb.get(1, ''), nb_p2=nb.get(2, ''))
        cands = [('old', b0)]
        # (a) dịch chỉ số ±1 = hộp phân biệt kề theo y
        others = [b for b in uniq if iou(b, b0) < 0.5]
        up = [b for b in others if cyc(b) < cyc(b0)]; dn = [b for b in others if cyc(b) > cyc(b0)]
        if up:
            cands.append(('prev', list(max(up, key=cyc))))
        if dn:
            cands.append(('next', list(min(dn, key=cyc))))
        if kbox and base['kim_char_ok']:
            kx, ky = cxc(kbox), cyc(kbox)
            w, h = b0[2] - b0[0], b0[3] - b0[1]
            x1 = int(round(kx - w / 2)); y1 = int(round(ky - h / 2))
            x1 = min(max(0, x1), W - w); y1 = min(max(0, y1), H - h)
            cands.append(('kim', [x1, y1, x1 + w, y1 + h]))
            det = [d[:4] for d in (kcol.get('det') or [])]
            if det:
                dd = [((cxc(d) - kx) ** 2 + (cyc(d) - ky) ** 2) ** 0.5 for d in det]
                j = int(np.argmin(dd))
                if dd[j] <= 0.5 * pitch:
                    cands.append(('det', [int(v) for v in det[j]]))
                base['det_dist_pitch'] = round(dd[j] / pitch, 3) if pitch > 0 else ''
        for name, bb in cands:
            o = dict(base, cand=name, bbox=json.dumps([int(v) for v in bb]))
            o['d_old_pitch'] = round((cyc(bb) - cyc(b0)) / pitch, 3) if pitch > 0 else ''
            if kbox:
                o['d_kim_pitch'] = round((cyc(bb) - cyc(kbox)) / pitch, 3) if pitch > 0 else ''
            if name == 'old':
                # cắt lại y hệt b_recrop: láng giềng = hộp kề của chính hộp cũ trong danh sách build
                ysb = sorted(cb, key=cyc)
                pos = [i for i, b in enumerate(ysb) if list(b) == list(b0)]
                pb = nbx = None
                if pos:
                    i = pos[0]
                    pb = ysb[i - 1] if i > 0 else None
                    nbx = ysb[i + 1] if i < len(ysb) - 1 else None
            else:
                pb, nbx = neighbours(cb, bb, skip_box=b0)
            key = hashlib.md5(f"{c['cell_uid']}|{name}".encode()).hexdigest()[:16]
            p = cdir / f'{key}.png'
            q = bd.save_crop(img, gray, bb, PAD, p, tighten=True, prev_bbox=pb, next_bbox=nbx,
                             img_orig=img_orig, bin_path=None)
            if q:
                o.update(crop=str(p.relative_to(OUT)), md5=hashlib.md5(p.read_bytes()).hexdigest(), status='ok',
                         crop_w=q['w'], crop_h=q['h'], ink=q['ink'])
            else:
                o.update(crop='', md5='', status='CROP_FAIL')
            if name == 'old':
                f = REPO / 'dataset/_ALL' / c['image']
                o['deliv_md5'] = hashlib.md5(f.read_bytes()).hexdigest() if f.exists() else ''
                o['old_exact'] = int(o.get('md5', '') == o['deliv_md5'])
            rows.append(o)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--sets', default=','.join(SETS))
    a = ap.parse_args()
    g = pd.read_csv(SP / 'gold_img_audit/measure/gold_suspicion.csv', dtype=str, keep_default_na=False,
                    usecols=['cell_uid', 'is_gold', 'book_set', 'book', 'page', 'column', 'nom_idx', 'syl_idx', 'label',
                             'ocr_char', 'syllable', 'bbox', 'image', 'geo_f_kim_off'])
    F = g[(g.is_gold == '1') & (g.geo_f_kim_off == '1')]
    tasks = []
    summ = {}
    for bs in a.sets.split(','):
        kim = json.loads((OUT / 'kimdet' / f'{bs}.json').read_text())
        B = pd.read_csv(build_path(bs), dtype=str, keep_default_na=False, usecols=['book', 'page', 'column', 'bbox', 'nom_idx'])
        B = B[B.bbox != '']
        Fs = F[F.book_set == bs]
        summ[bs] = dict(flagged=len(Fs))
        Bg = {k: v for k, v in B.groupby(['book', 'page'])}
        for (bk, pg), sub in Fs.groupby(['book', 'page']):
            Bp = Bg.get((bk, pg))
            colb = {}
            if Bp is not None:
                for cc, bb in zip(Bp.column, Bp.bbox):
                    colb.setdefault(cc, []).append([int(v) for v in json.loads(bb)])
            tasks.append((bs, bk, pg, sub.to_dict('records'), colb, kim.get(f'{bk}|{pg}')))
    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(do_page, tasks, chunksize=2)):
            rows += r
            if i % 100 == 0:
                print(f'  {i}/{len(tasks)} trang {time.time()-t0:.0f}s', flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / 'candidates.csv', index=False)
    for bs, x in D.groupby('book_set'):
        o = x[x.cand == 'old']
        summ[bs].update(cells=int(x.cell_uid.nunique()), cand_counts=dict(Counter(x.cand)),
                        crop_fail=int((x.status != 'ok').sum()),
                        old_exact=int(o.old_exact.sum()), old_n=len(o),
                        kim_char_ok=int(o.kim_char_ok.sum()))
    summ['invariants'] = {bs: dict(old_exact_all=(summ[bs]['old_exact'] == summ[bs]['old_n'])) for bs in summ if bs in ORIGINAL}
    summ['sec'] = round(time.time() - t0)
    (OUT / 's1b_summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == '__main__':
    main()
