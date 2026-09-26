#!/usr/bin/env python3
"""s1a_kimdet.py — TN2 bước 1a: dựng lại HỘP KIM từng chữ + HỘP DETECTOR THÔ của mọi cột trên các trang có ô GOLD bị cờ trượt.

Chỉ ĐỌC repo, 0 API:
  * hộp kim: chính hàm engine pipeline.align_engine.align_production._detect (đọc cache OCR + ảnh trang),
    như gold_img_audit/crop_geometry/kim_vs_box.py; cột nhãn `column` == line_id của iter_pairs.
  * hộp detector thô: DetectorInfer (ckpt/thr/resize THEO SÁCH như build_dataset) -> boxes_for_page ->
    raw_column_boxes(x_range, det_xmargin) — hộp detector thô của cột, chưa ép đếm (không qua pitch decode).
Ô bị cờ trượt (định trước): ô GOLD có geo_f_kim_off = 1 (tâm hộp ảnh cách tâm hộp kim của CHÍNH chữ nhãn > 0,5 bước
cột theo y hoặc > 0,5 bề rộng theo x) trong gold_img_audit/measure/gold_suspicion.csv.

Chạy: .venv/bin/python s1a_kimdet.py [--sets LucVanTien1916,TruyenKieu1872,Chrestomathie1872,SachThanhTruyen] [--limit-pages N]
Ra:   measure_out/_thu_nghiem_anh_chu/TN2/kimdet/<set>.json
"""
import argparse, json, sys, time
from pathlib import Path
import cv2
import pandas as pd

REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR')
sys.path.insert(0, str(REPO))
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
OUT = REPO / 'measure_out/_thu_nghiem_anh_chu/TN2'
CFG = {'SachThanhTruyen': 'config/pipeline.yaml', 'Chrestomathie1872': 'config/pipeline_Chrestomathie1872.yaml',
       'LucVanTien1916': 'config/pipeline_LucVanTien1916.yaml', 'TruyenKieu1872': 'config/pipeline_TruyenKieu1872.yaml'}


def flagged_cells():
    g = pd.read_csv(SP / 'gold_img_audit/measure/gold_suspicion.csv', dtype=str, keep_default_na=False,
                    usecols=['cell_uid', 'is_gold', 'book_set', 'book', 'page', 'column', 'geo_f_kim_off'])
    return g[(g.is_gold == '1') & (g.geo_f_kim_off == '1')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default=','.join(CFG))
    ap.add_argument('--limit-pages', type=int, default=0)
    a = ap.parse_args()
    from pipeline.step0_setup import load_config
    from pipeline.align_engine.build_dataset import _book_code
    from pipeline.align_engine.book_layout import book_layout, resolve_detector_ckpt
    from pipeline.align_engine.align_production import _detect
    from pipeline.align_engine.char_detector.detector_infer import DetectorInfer
    from core.text.dictionary import load_qn_to_nom
    F = flagged_cells()
    (OUT / 'kimdet').mkdir(parents=True, exist_ok=True)
    for bs in a.sets.split(','):
        cfg = load_config(str(REPO / CFG[bs]))
        qn = set(load_qn_to_nom(str(REPO / cfg['paths']['qn_to_nom_dict'])).keys())
        s2 = cfg.get('step2', {}) or {}
        res, t0 = {}, time.time()
        n_pages = 0
        for b in cfg['books']:
            code = _book_code(b['name'])
            lay = book_layout(b)
            pages = sorted(F[(F.book_set == bs) & (F.book == code)].page.unique())
            if a.limit_pages:
                pages = pages[:a.limit_pages]
            ck = resolve_detector_ckpt(lay.detector_ckpt, REPO)
            thr = lay.det_thr if lay.det_thr is not None else float(s2.get('det_thr', 0.2))
            xm = lay.det_xmargin if lay.det_xmargin is not None else float(s2.get('det_xmargin', 0.25))
            det = DetectorInfer(ckpt=ck, thr=thr, resize=lay.detector_resize)
            assert det.trained, f'detector không nạp được ({ck})'
            ddir = REPO / 'prepared' / b['name']
            for pg in pages:
                d = _detect(pg, ddir, qn, layout=lay)
                if d is None:
                    res[f'{code}|{pg}'] = None
                    continue
                cols, qn_lines, iter_pairs, binary, ok = d
                page_bgr = cv2.imread(str(ddir / 'pages' / f'{pg}.png'), cv2.IMREAD_COLOR)
                pbox = det.boxes_for_page(page_bgr)
                out = {}
                for ci, line_id in iter_pairs:
                    cl = cols[ci]
                    xr = cl.get('x_range')
                    chars = [[c.get('char') or c.get('text') or '', [float(v) for v in c['bbox'][:4]] if c.get('bbox') else None]
                             for c in cl['chars']]
                    raw = det.raw_column_boxes(pbox, xr, xm) if xr else []
                    out[str(int(line_id))] = dict(x_range=[float(v) for v in xr] if xr else None, chars=chars, det=raw)
                res[f'{code}|{pg}'] = out
                n_pages += 1
                if n_pages % 25 == 0:
                    print(f'  [{bs}] {n_pages} trang {time.time()-t0:.0f}s', flush=True)
            meta = dict(ckpt=ck, thr=thr, xmargin=xm, resize=lay.detector_resize)
            res.setdefault('_meta', {})[code] = meta
        (OUT / 'kimdet' / f'{bs}.json').write_text(json.dumps(res, ensure_ascii=False))
        print(f'[{bs}] {n_pages} trang, {sum(v is None for k, v in res.items() if k != "_meta")} trang _detect=None, '
              f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
