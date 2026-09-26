#!/usr/bin/env python
"""s04b_embed_sv.py — nhúng CROP CHUẨN bằng encoder v1+v2 (sv_lib.MultiEnc, norm=True — y hệt r01/m_ocr) để chạy lại kênh
xếp hạng lại (viss) với view A = crop chuẩn. MPS. 0 API, repo chỉ đọc.
  E_crops_new.f16.npy : 118.954 ô GOLD (thứ tự m_ocr/inputs.pkl = labels GOLD) — nguyên mẫu 'oth' của r02
  E_A.f16.npy         : 73.483 ô harness (thứ tự cells_eval) — GOLD lấy từ E_crops_new, tier khác nhúng từ TN4/aux
  E_B.f16.npy         : CHÉP NGUYÊN view B cũ (cắt bbox, không đổi)
  cells_idx.csv       : như r01
Invariant: nhúng lại 400 crop GOLD CŨ (dataset/_ALL) -> cos với E_crops cũ ≥ 0,999.
Ra: measure_out/_thu_nghiem_anh_chu/TN4/rerank_new/, lab/.../s04b.json"""
import json, shutil, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
sys.path.insert(0, str(HERE))
from ver import BIG, LABOUT  # noqa
OUT = BIG / 'rerank_new'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SP / 'gold_img_audit/m_ocr'))
import sv_lib as sv  # noqa
T0 = time.time()


def uid_path(uid):
    a = uid.split('/')
    return '/'.join(a[:3]) + '/' + '_'.join(a[3:]) + '.png'


def rd(p):
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p is not None and Path(p).exists() else None
    return im if im is not None and im.size else np.full((8, 8), 255, np.uint8)


def embed_paths(enc, paths, chunk=8192):
    out = []
    for s in range(0, len(paths), chunk):
        out.append(enc.embed([rd(p) for p in paths[s:s + chunk]]).astype(np.float16))
        print(f'  {s + chunk}/{len(paths)} {time.time()-T0:.0f}s', flush=True)
    return np.concatenate(out)


def main():
    inv = {}
    enc = sv.MultiEnc(['v1', 'v2'], 'mps', True)
    g = pd.read_pickle(SP / 'gold_img_audit/m_ocr/out/inputs.pkl')
    # invariant: nhúng lại crop cũ
    Eo = np.load(SP / 'gold_img_audit/m_ocr/out/E_crops.f16.npy', mmap_mode='r')
    idx = np.random.default_rng(7).permutation(len(g))[:400]
    e = enc.embed([rd(REPO / 'dataset/_ALL' / p) for p in g.image.values[idx]])
    cs = (e * np.asarray(Eo[idx], np.float32)).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(np.asarray(Eo[idx], np.float32), axis=1))
    inv['reembed_old_cos_min'] = float(cs.min()); inv['ok_reembed'] = bool(cs.min() >= 0.999)
    print(inv, flush=True)
    paths = [BIG / 'crops' / uid_path(u) for u in g.cell_uid]
    inv['gold_missing'] = int(sum(not p.exists() for p in paths))
    EG = embed_paths(enc, paths)
    np.save(OUT / 'E_crops_new.f16.npy', EG)
    d = pd.read_csv(SP / 'kim_bottleneck/harness/cells_eval.csv', dtype=str, keep_default_na=False, usecols=['cell_uid', 'tier'])
    pos = {u: i for i, u in enumerate(g.cell_uid)}
    EA = np.zeros((len(d), EG.shape[1]), np.float16)
    isg = d.cell_uid.map(pos)
    m = isg.notna().values
    EA[m] = EG[isg[m].astype(int).values]
    inv['harness_gold_from_crops'] = int(m.sum())
    ap = [BIG / 'aux' / uid_path(u) for u in d.cell_uid.values[~m]]
    inv['aux_missing'] = int(sum(not p.exists() for p in ap))
    EA[~m] = embed_paths(enc, ap)
    np.save(OUT / 'E_A.f16.npy', EA)
    shutil.copyfile(SP / 'kim_bottleneck/rerank/out/E_B.f16.npy', OUT / 'E_B.f16.npy')
    okA = np.ones(len(d), int)
    pd.DataFrame(dict(cell_uid=d.cell_uid, okA=okA, okB=okA)).to_csv(OUT / 'cells_idx.csv', index=False)
    inv['sec'] = round(time.time() - T0)
    json.dump(inv, open(LABOUT / 's04b.json', 'w'), indent=1)
    print(json.dumps(inv))


if __name__ == '__main__':
    main()
