#!/usr/bin/env python
"""s02_geom.py — crop CŨ ↔ crop CHUẨN trên 8 bộ GOLD: cờ "một chữ", mực lạ, cờ bleed/tall/truncated, ô B0 được cứu,
và (L16/TK, nhãn người) khe của crop mới theo mô hình khe người + mực ngoài khe người / độ phủ mực khe người.
CPU, 0 API, không mở ảnh (chỉ đọc bảng s01). Định nghĩa theo t00_freeze.json (prereg).
Ra: measure_out/_thu_nghiem_anh_chu/TN4/gold_tn4.pkl (1 hàng/ô GOLD, mọi cột cần cho s05/s06/s07),
    lab/.../geom_table.csv, geom_ihr.csv, geom_borg.json, s02_geom.json
Chạy: .venv/bin/python lab/thu_nghiem_anh_chu/TN4_crop_chuan/s02_geom.py"""
import json, sys, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
from ver import BIG, LABOUT, LIBNAME, FREEZE, VER  # noqa
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
from hslot import Slot  # noqa
import cclib  # noqa
T0 = time.time()
P = cclib.PARAMS
SETS8 = ['stt2', 'stt4', 'stt11', 'Chr', 'L83', 'KVK', 'L16', 'TK']
AB = {'Chrestomathie1872': 'Chr', 'LucVanTien1883': 'L83', 'KimVanKieu1884': 'KVK', 'LucVanTien1916': 'L16', 'TruyenKieu1872': 'TK'}
INV = {}


def ink_flag(D, key):
    """cờ ink (lượt 2): r = ink_norm / trung vị cùng (key, nhãn) nếu lớp ≥ 5 ô; lớp nhỏ so trung vị cả key (khoảng rộng)."""
    ok = D.status == 'ok'
    med_lab = D[ok].groupby([key, 'label']).ink_norm.agg(['median', 'size'])
    med_set = D[ok].groupby(key).ink_norm.median()
    j = D[[key, 'label']].merge(med_lab, left_on=[key, 'label'], right_index=True, how='left')
    big = (j['size'].fillna(0).values >= P['ink_nmin'])
    ref = np.where(big, j['median'].values, D[key].map(med_set).values)
    r = D.ink_norm.values / np.maximum(ref, 1e-9)
    lo = np.where(big, P['ink_lo'], P['ink_lo_w']); hi = np.where(big, P['ink_hi'], P['ink_hi_w'])
    fl = ok.values & ((r < lo) | (r > hi))
    return r, fl


def main():
    fr = json.load(open(FREEZE))
    import hashlib
    INV['design_frozen'] = hashlib.md5((HERE / f'{LIBNAME}.py').read_bytes()).hexdigest() == fr['cclib_md5']
    parts = [pd.read_pickle(BIG / f'cells_gold_{s}.pkl') for s in SETS8]
    C = pd.concat(parts, ignore_index=True).rename(columns={'key': 'cell_uid'})
    INV['n_cells'] = len(C); INV['uid_unique'] = bool(C.cell_uid.is_unique)
    X = pd.read_pickle(SP / 'r6/policy_v3/out/base_v2.pkl').drop(columns=['status'])
    L = pd.read_csv(REPO / 'dataset/_ALL/labels.csv', dtype=str, keep_default_na=False)
    L = L[L.tier == 'GOLD'][['cell_uid', 'book', 'page', 'column', 'image', 'bbox']]
    X = X.merge(L, on='cell_uid', how='left', suffixes=('', '_l'), validate='1:1')
    D = X.merge(C, on='cell_uid', how='left', validate='1:1').copy()
    INV['all_gold_have_crop_record'] = bool(D.status.notna().all())
    bk = D.cell_uid.str.split('/').str[1]
    D['set8'] = np.where(D.book_set == 'SachThanhTruyen', bk, D.book_set.map(AB))
    D['syl_idx'] = D.cell_uid.str.split('/').str[-1].str[1:]
    b = lambda c: D[c].fillna(False).astype(bool).values
    cqf = D.crop_quality_flag.fillna('').values
    tall_old = D.seg_flag.fillna('').astype(str).values == 'tall'
    D['B0_crop_old'] = ((cqf == 'blank') | b('blank') | (cqf == 'truncated') | b('truncated') | (cqf == 'bleed')
                        | b('f_tight_nb') | tall_old)
    D['B0_box'] = b('geo_f_dup_bbox') | b('geo_f_ov_heavy')
    D['bleed_old'] = cqf == 'bleed'; D['trunc_old'] = (cqf == 'truncated') | b('truncated'); D['tall_old'] = tall_old
    D['blank_old'] = (cqf == 'blank') | b('blank'); D['tightnb_old'] = b('f_tight_nb')
    # ---- invariant: cờ pipeline tính lại trên crop cũ dựng lại == cờ trong bảng (trừ ô 64×64 rescue / không trùng md5)
    rescue64 = D.rule.fillna('').values == 'self_training_rescue'
    m = ~rescue64 & D.old_crop_quality_flag.notna().values
    INV['old_cqf_recomputed_eq'] = round(float((D.old_crop_quality_flag.values[m] == cqf[m]).mean()), 5)
    INV['old_tall_recomputed_eq'] = round(float((D.old_tall.values[m].astype(bool) == tall_old[m]).mean()), 5)
    # ---- crop chuẩn: cờ
    fl = D['flags'].fillna('').values
    has = lambda t: np.array([t in f.split('|') for f in fl])
    D['f_blank'], D['f_cut'], D['f_two'] = has('blank') | (D.status.values != 'ok'), has('cut'), has('two')
    r, fink = ink_flag(D, 'set8')
    D['ink_ratio'], D['f_ink'] = r, fink
    ncq = D.new_crop_quality_flag.fillna('').values
    D['bleed_new'], D['trunc_new'] = ncq == 'bleed', ncq == 'truncated'
    D['tall_new'] = D.tall.fillna(False).astype(bool).values
    D['one_char_ok'] = ((D.status.values == 'ok') & ~D.f_blank & ~D.f_cut & ~D.f_two & ~D.f_ink & ~D.bleed_new & ~D.trunc_new)
    D['clean_new'] = D.one_char_ok & ~D.tall_new
    D['B0_new'] = D.B0_box | ~D.one_char_ok | D.tall_new
    D['A0_new'] = b('int_foreign') | b('geo_f_dup_bbox') | D.f_blank.values | D.f_cut.values
    D['rescued'] = D.B0_crop_old & D.clean_new
    D['newly_flagged'] = ~D.B0_crop_old & ~D.clean_new
    # ---- IHR: khe crop mới (tâm hộp mực) và khe cũ (tâm bbox, = harness)
    for c in ('slot_old_bbox', 'slot_new', 'slot_oldrect'):
        D[c] = ''
    for book, s8 in (('LucVanTien1916', 'L16'), ('TruyenKieu1872', 'TK')):
        S = Slot(book)
        mm = (D.set8 == s8).values
        sub = D[mm]
        bb = np.array([json.loads(v) for v in sub.bbox])
        cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
        D.loc[mm, 'slot_old_bbox'] = [S.slot(p, c, k, a, q) for p, c, k, a, q in zip(sub.page, sub.column, sub.syl_idx, cx, cy)]
        D.loc[mm, 'slot_new'] = [S.slot(p, c, k, a, q) if st == 'ok' else '' for p, c, k, a, q, st in
                                 zip(sub.page, sub.column, sub.syl_idx, sub.ink_cx.values, sub.ink_cy.values, sub.status)]
        orc = [json.loads(v) if isinstance(v, str) else None for v in sub.old_rect]
        D.loc[mm, 'slot_oldrect'] = [S.slot(p, c, k, (o[0] + o[2]) / 2, (o[1] + o[3]) / 2) if o else '' for p, c, k, o in
                                     zip(sub.page, sub.column, sub.syl_idx, orc)]
        INV[f'slot_old_bbox_eq_harness_{s8}'] = round(float((D.loc[mm, 'slot_old_bbox'].values ==
                                                           D.loc[mm, 'slot_ok'].fillna('').astype(str).values).mean()), 5)
    ihr = D.set8.isin(['L16', 'TK']).values
    D['damaged'] = ihr & (D.slot_old_bbox.values == '1') & (D.slot_new.values != '1')
    D['slot_fixed'] = ihr & (D.slot_old_bbox.values != '1') & (D.slot_new.values == '1')
    hm = ihr & D.h_ink.notna().values & (D.h_ink.fillna(0).values > 0)
    D['cont_old'] = np.where(hm, 1 - D.old_in / D.old_tot.clip(lower=1), np.nan)
    D['cont_new'] = np.where(hm, 1 - D.new_in / D.new_tot.clip(lower=1), np.nan)
    D['cov_old'] = np.where(hm, D.old_in / D.h_ink.clip(lower=1), np.nan)
    D['cov_new'] = np.where(hm, D.new_in / D.h_ink.clip(lower=1), np.nan)
    D['cov_drop'] = hm & ((D.cov_old.fillna(0) - D.cov_new.fillna(0)) > 0.25)
    D.to_pickle(BIG / 'gold_tn4.pkl')
    # ---- bảng 8 bộ
    rows = []
    for s in SETS8:
        g = D[D.set8 == s]
        n = len(g)
        rr = dict(bo=s, n=n,
                  sach_cu=int((~g.B0_crop_old).sum()), sach_moi=int(g.clean_new.sum()),
                  mot_chu_ok_moi=int(g.one_char_ok.sum()),
                  B0_cu=int(g.B0_crop_old.sum()), bleed_cu=int(g.bleed_old.sum()), tall_cu=int(g.tall_old.sum()),
                  trunc_cu=int(g.trunc_old.sum()), tightnb_cu=int(g.tightnb_old.sum()), blank_cu=int(g.blank_old.sum()),
                  bleed_moi=int(g.bleed_new.sum()), tall_moi=int(g.tall_new.sum()), trunc_moi=int(g.trunc_new.sum()),
                  cut_moi=int(g.f_cut.sum()), two_moi=int(g.f_two.sum()), ink_moi=int(g.f_ink.sum()), blank_moi=int(g.f_blank.sum()),
                  cuu=int(g.rescued.sum()), co_moi=int(g.newly_flagged.sum()),
                  stray_cu_tb=round(float(g.old_stray_ink.mean()), 4), stray_moi_tb=round(float(g.new_stray_ink.mean()), 4),
                  stray_cu_gt008=int((g.old_stray_ink > 0.08).sum()), stray_moi_gt008=int((g.new_stray_ink > 0.08).sum()),
                  tach_nen_moi=int((g.foreign_erased_px.fillna(0) > 0).sum()),
                  src_goc=int((g.src_kind == 'original').sum()),
                  md5_mau=f"{int(g.md5_eq.fillna(False).astype(bool).sum())}/{int(g.md5_eq.notna().sum())}")
        rows.append(rr)
    T = pd.DataFrame(rows)
    T.to_csv(LABOUT / 'geom_table.csv', index=False)
    # md5 lệch chỉ ở ô rescue 64×64?
    mm5 = D.md5_eq.notna() & ~D.md5_eq.fillna(True).astype(bool)
    INV['md5_mismatch_n'] = int(mm5.sum()); INV['md5_mismatch_all_rescue64'] = bool(rescue64[mm5.values].all())
    # ---- IHR
    R = []
    for s in ('L16', 'TK'):
        g = D[D.set8 == s]
        h = g[g.cov_old.notna()]
        h1 = h[h.slot_old_bbox == '1']
        R.append(dict(bo=s, n=len(g),
                      khe_cu_1=int((g.slot_old_bbox == '1').sum()), khe_cu_0=int((g.slot_old_bbox == '0').sum()),
                      khe_cu_chua=int((g.slot_old_bbox == '').sum()),
                      khe_moi_1=int((g.slot_new == '1').sum()), khe_moi_0=int((g.slot_new == '0').sum()),
                      khe_moi_chua=int((g.slot_new == '').sum()),
                      khe_rect_cu_1=int((g.slot_oldrect == '1').sum()),
                      hong=int(g.damaged.sum()), hong_pct=round(100 * g.damaged.sum() / max(1, (g.slot_old_bbox == '1').sum()), 3),
                      sua_khe=int(g.slot_fixed.sum()), phu_giam=int(g.cov_drop.sum()),
                      n_khe_nguoi=len(h1),
                      muc_ngoai_khe_cu=round(float(h1.cont_old.mean()), 4), muc_ngoai_khe_moi=round(float(h1.cont_new.mean()), 4),
                      ngoai_khe_gt10_cu=int((h1.cont_old > 0.10).sum()), ngoai_khe_gt10_moi=int((h1.cont_new > 0.10).sum()),
                      ngoai_khe_gt25_cu=int((h1.cont_old > 0.25).sum()), ngoai_khe_gt25_moi=int((h1.cont_new > 0.25).sum()),
                      phu_cu=round(float(h1.cov_old.mean()), 4), phu_moi=round(float(h1.cov_new.mean()), 4),
                      phu_lt80_cu=int((h1.cov_old < 0.80).sum()), phu_lt80_moi=int((h1.cov_new < 0.80).sum()),
                      B0cu_ngoai_khe_cu=round(float(h1[h1.B0_crop_old].cont_old.mean()), 4),
                      B0cu_ngoai_khe_moi=round(float(h1[h1.B0_crop_old].cont_new.mean()), 4),
                      cuu_va_khong_hong=int((g.rescued & ~g.damaged).sum()),
                      hong_trong_sach_moi=int((g.damaged & g.clean_new).sum())))
    pd.DataFrame(R).to_csv(LABOUT / 'geom_ihr.csv', index=False)
    # ---- Borg (chỉ crop mới: cờ)
    Bg = pd.read_pickle(BIG / 'cells_borg.pkl')
    kv = pd.read_csv(SP / 'r5/borg_quality/out/clean_keep.csv', dtype=str, keep_default_na=False)
    Bg['book'] = Bg.key.str.split('/').str[0]; Bg['page'] = Bg.key.str.split('/').str[1]; Bg['idx'] = Bg.key.str.split('/').str[3]
    Bg = Bg.merge(kv[['book', 'page', 'idx', 'keep_v5']], on=['book', 'page', 'idx'], how='left')
    bc = pd.read_csv(SP / 'r4/borg_align/borg_cells.csv', dtype=str, keep_default_na=False, usecols=['book', 'page', 'idx', 'char'])
    Bg = Bg.merge(bc, on=['book', 'page', 'idx'], how='left')
    Bg['label'] = Bg.char
    r, fink = ink_flag(Bg, 'book')
    flb = Bg['flags'].fillna('').values
    hb = lambda t: np.array([t in f.split('|') for f in flb])
    okc = (Bg.status.values == 'ok') & ~hb('blank') & ~hb('cut') & ~hb('two') & ~fink & \
          ~Bg.new_crop_quality_flag.isin(['bleed', 'truncated']).values
    Bg['one_char_ok'] = okc; Bg['clean_new'] = okc & ~Bg.tall.fillna(False).astype(bool).values
    Bg[['key', 'book', 'page', 'idx', 'keep_v5', 'status', 'flags', 'one_char_ok', 'clean_new', 'new_stray_ink',
        'new_crop_quality_flag', 'tall', 'ink_cx', 'ink_cy', 'sq', 'new_file']].to_pickle(BIG / 'borg_tn4.pkl')
    GB = {}
    for kvv, g in Bg.groupby(Bg.keep_v5.fillna('')):
        GB[f'keep_v5={kvv}'] = dict(n=len(g), one_char_ok=int(g.one_char_ok.sum()), clean_new=int(g.clean_new.sum()),
                                    cut=int(np.array(['cut' in f for f in g['flags'].fillna('')]).sum()),
                                    two=int(np.array(['two' in f for f in g['flags'].fillna('')]).sum()),
                                    tall=int(g.tall.fillna(False).astype(bool).sum()), bleed_new=int((g.new_crop_quality_flag == 'bleed').sum()))
    json.dump(GB, open(LABOUT / 'geom_borg.json', 'w'), ensure_ascii=False, indent=1)
    INV['sec'] = round(time.time() - T0)
    json.dump(INV, open(LABOUT / 's02_geom.json', 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(INV, ensure_ascii=False, default=str))
    print(T.to_string()); print(pd.DataFrame(R).T.to_string()); print(json.dumps(GB))


if __name__ == '__main__':
    main()
