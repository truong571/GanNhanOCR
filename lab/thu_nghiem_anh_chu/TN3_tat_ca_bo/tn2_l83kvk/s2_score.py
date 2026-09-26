#!/usr/bin/env python3
"""s2_score.py — TN2 bước 2: CHẤM mọi crop ứng viên bằng bộ kiểm ảnh↔chữ học trên NHÃN NGƯỜI (r4/verifier_ft).

Hai mô hình: T (học TK1872 phần A + Borg), L (học LVT1916 phần A + Borg). Không mô hình nào học Chr/STT.
Với crop z và nhãn x (KHÔNG đổi):
  sw(x)  = cos(f(z), W_x)                                    (nguyên mẫu học của chữ x)
  m      = sw(nhãn) − max sw(đối thủ); đối thủ = chữ kim kề (nom_idx±1, ±2) ∪ đồng âm R(âm) ∪ top-5 gần hình của nhãn,
           bỏ mọi chữ tương đương dị thể với nhãn (lớp simp + Unihan, như s01_features)
  mN     = như m nhưng đối thủ chỉ là chữ kim kề (±1, ±2)
Crop 'deliv' = tệp GOLD đang giao (dataset/_ALL/<image>) — điểm TRƯỚC khi sửa.
Tiền xử lý 64×64 y hệt verifier_ft/p01_prep.py (kéo giãn p2–p98, đệm vuông trắng, INTER_AREA).
Ra: measure_out/_thu_nghiem_anh_chu/TN2/scores.csv
Chạy: .venv/bin/python s2_score.py
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F

REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR')
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
V = SP / 'r4/verifier_ft'
sys.path.insert(0, str(V)); sys.path.insert(0, str(SP / 'kim_bottleneck/harness'))
from vflib import Net, to_t, load_json  # noqa
from harness_lib import R_of, variants_of, simp  # noqa
OUT = REPO / 'measure_out/_thu_nghiem_anh_chu/TN3/tn2_l83kvk'   # TN3: thư mục đầu ra riêng
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
SZ = 64


def prep(gray):
    if gray is None or gray.size == 0:
        gray = np.full((8, 8), 255, np.uint8)
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    if hi - lo > 10:
        gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    h, w = gray.shape
    s = max(h, w)
    can = np.full((s, s), 255, np.uint8)
    can[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = gray
    return cv2.resize(can, (SZ, SZ), interpolation=cv2.INTER_AREA)


def main():
    t0 = time.time()
    D = pd.read_csv(OUT / 'candidates.csv', dtype=str, keep_default_na=False)
    D = D[D.status == 'ok'].reset_index(drop=True)
    # thêm crop GIAO NỘP làm ứng viên 'deliv'
    g = pd.read_csv(REPO / 'dataset/_ALL/labels.csv', dtype=str, keep_default_na=False, usecols=['cell_uid', 'image'])
    img_of = dict(zip(g.cell_uid, g.image))
    old = D[D.cand == 'old'].copy()
    old['cand'] = 'deliv'
    old['crop'] = [str(REPO / 'dataset/_ALL' / img_of[u]) for u in old.cell_uid]
    D = pd.concat([D, old], ignore_index=True)
    paths = [p if p.startswith('/') else str(OUT / p) for p in D.crop]
    X = np.stack([prep(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in paths])
    print(f'[s2] {len(X)} crop, {time.time()-t0:.0f}s', flush=True)
    U = load_json(V / 'out/glyph_chars.json'); uid = {c: i for i, c in enumerate(U)}
    SIM5 = load_json(V / 'out/simtop5.json')
    simp_id = {}
    SID = np.array([simp_id.setdefault(simp(c), len(simp_id)) for c in U])
    by_sid = defaultdict(list)
    for k, s in enumerate(SID):
        by_sid[s].append(k)

    def var_set(k):
        o = set(by_sid[SID[k]]) | {uid[v] for v in variants_of(U[k]) if v in uid}
        o.add(k)
        return o
    # đối thủ theo ô
    lab_id = np.full(len(D), -1)
    riv_all, riv_nb = [], []
    for i, r in enumerate(D.itertuples()):
        k = uid.get(r.label, -1)
        lab_id[i] = k
        if k < 0:
            riv_all.append([]); riv_nb.append([]); continue
        vs = var_set(k)
        nb = [uid[c] for c in (r.nb_m2, r.nb_m1, r.nb_p1, r.nb_p2) if c in uid and uid[c] not in vs]
        hom = [uid[c] for c in R_of(r.syllable) if c in uid and uid[c] not in vs] if r.syllable else []
        sim = [uid[c] for c in SIM5.get(r.label, []) if c in uid and uid[c] not in vs]
        riv_nb.append(sorted(set(nb)))
        riv_all.append(sorted(set(nb) | set(hom) | set(sim)))
    Xt = to_t(X)
    for tag in ('T', 'L'):
        ck = torch.load(V / f'out/model_{tag}.pt', map_location='cpu')
        net = Net(widths=tuple(ck['widths']))
        net.load_state_dict({k: v.float() for k, v in ck['net'].items()})
        net.eval().to(DEV)
        Wn = torch.from_numpy(np.load(V / f'out/W_{tag}.f16.npy').astype(np.float32))
        Z = []
        with torch.no_grad():
            for s in range(0, len(Xt), 512):
                Z.append(net.crop(Xt[s:s + 512].to(DEV)).float().cpu())
        Z = torch.cat(Z)
        sw = np.full(len(D), np.nan); m = np.full(len(D), np.nan); mN = np.full(len(D), np.nan)
        rival = [''] * len(D)
        for i in range(len(D)):
            k = lab_id[i]
            if k < 0:
                continue
            z = Z[i]
            s0 = float(z @ Wn[k]); sw[i] = s0
            ra = riv_all[i]
            if ra:
                sr = (Wn[ra] @ z).numpy(); j = int(np.argmax(sr))
                m[i] = s0 - float(sr[j]); rival[i] = U[ra[j]]
            rn = riv_nb[i]
            if rn:
                mN[i] = s0 - float((Wn[rn] @ z).max())
        D[f'sw_{tag}'] = np.round(sw, 4); D[f'm_{tag}'] = np.round(m, 4); D[f'mN_{tag}'] = np.round(mN, 4)
        D[f'rival_{tag}'] = rival
        del net
        print(f'[s2] model {tag} xong {time.time()-t0:.0f}s', flush=True)
    D['in_univ'] = (lab_id >= 0).astype(int)
    D.to_csv(OUT / 'scores.csv', index=False)
    print(f'[s2] {len(D)} hàng, in_univ {int((lab_id>=0).sum())}, {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
