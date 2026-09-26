#!/usr/bin/env python
"""s10a_hand_retrain_prep.py — ĐỘ NHẠY hậu kiểm (chỉ v2): HỌC LẠI bộ kiểm chữ viết tay LOBO-sách (Kinh) trên CROP CHUẨN để tách
"crop chuẩn làm mất thông tin" khỏi "bộ kiểm học trên crop cũ nên lạ khung mới". Bước 1: dựng mảng ảnh 64×64 (tiền xử lý p01) theo
đúng thứ tự của r5/hand_lobo (meta_all 138.885 ô Borg, meta_ihr 36.259 ô IHR, meta_stt 52.707 ô STT); ô có crop chuẩn v2 dùng crop
chuẩn, ô không có (Borg không keep/keep_high, ô rỗng) giữ ảnh CŨ (các ô này không vào tập học, chỉ vào tập con 'nonkeep' — bỏ khi báo).
Bước 2: sinh h01_train_tn4.py = bản chép h01_train.py, CHỈ đổi thư mục OUT. Ra: measure_out/_thu_nghiem_anh_chu/TN4/v2/hand_retrain/"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault('TN4_VER', 'v2')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s03_vft as S3  # noqa
from ver import BIG  # noqa
HL = S3.SP / 'r5/hand_lobo'
OUT = BIG / 'hand_retrain'; OUT.mkdir(exist_ok=True)
MA = pd.read_pickle(HL / 'out/meta_all.pkl'); MS = pd.read_pickle(HL / 'out/meta_stt.pkl')
MI = pd.read_pickle(S3.OUTV / 'meta_ihr.pkl')
gold = set(pd.read_pickle(S3.OUTV / 'meta_gold.pkl').cell_uid)
inv = {}
def build(paths, old):
    has = np.array([p is not None and Path(p).exists() for p in paths])
    X = old.copy()
    idx = np.nonzero(has)[0]
    X[idx] = S3.load_imgs([paths[i] for i in idx])
    return X, int(has.sum())
pb = [BIG / 'aux_borg' / f'{b}/{p}/c{int(c):02d}_{int(i):04d}.png' for b, p, c, i in zip(MA.book, MA.page, MA.col, MA.idx)]
XA, inv['borg_new'] = build(pb, np.load(HL / 'out/img_all.npy'))
pi = [BIG / ('crops' if u in gold else 'aux') / S3.uid_path(u) for u in MI.cell_uid]
XI, inv['ihr_new'] = build(pi, np.load(HL / 'out/img_ihr.npy'))
ps = [BIG / 'crops' / S3.uid_path(u) for u in MS.cell_uid]
XS, inv['stt_new'] = build(ps, np.load(HL / 'out/img_stt.npy'))
np.save(OUT / 'img_all.npy', XA); np.save(OUT / 'img_ihr.npy', XI); np.save(OUT / 'img_stt.npy', XS)
shutil.copyfile(HL / 'out/meta_all.pkl', OUT / 'meta_all.pkl')
tr = (MA.keep.values == 1) & (MA.fold.values != 4) & (MA.book.values == 'SachKinhThayCaBinh')
inv['train_rows_kinh'] = int(tr.sum()); inv['train_rows_with_new'] = int((tr & np.array([Path(p).exists() for p in pb])).sum())
s = (HL / 'h01_train.py').read_text()
t = s.replace("from hlib import OUT  # noqa", f"from hlib import OUT  # noqa\nOUT = Path('{OUT}')  # TN4: thư mục học lại trên crop chuẩn v2")
t = t.replace("sys.path.insert(0, str(Path(__file__).resolve().parent))", f"sys.path.insert(0, '{HL}')")
t = t.replace("sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'r4/verifier_ft'))", f"sys.path.insert(0, '{S3.SP / 'r4/verifier_ft'}')")
assert t != s
(HERE / 'h01_train_tn4.py').write_text(t)
json.dump(inv, open(HERE / 'out_v2' / 's10a.json', 'w'), indent=1)
print(inv)
