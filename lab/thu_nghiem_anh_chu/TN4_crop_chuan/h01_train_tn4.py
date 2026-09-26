#!/usr/bin/env python
"""h01_train.py — (hand_lobo) BẢN LOBO-SÁCH của verifier_ft/t02_train.py: y hệt công thức, CHỈ khác tập Borg học =
keep fold 0–3 của MỘT sách (TRAIN_BOOK); sách Borg kia KHÔNG được thấy (giữ ngoài theo SÁCH). Fold 4 của sách học = hiệu chuẩn.
argv: TRAIN_BOOK (Kinh|DungLy) TUNE_BOOK STEPS TAG.   Ra: out/model_<TAG>_<BK>.pt, out/emb_<TAG>_<BK>_{all,ihr,stt}.f16.npy, out/W_<TAG>_<BK>.f16.npy
(bản gốc) t02_train.py — bộ kiểm ảnh↔chữ dạng NGUYÊN MẪU học (metric CNN + bảng nguyên mẫu theo chữ), CHỈ nhãn NGƯỜI + glyph font. 0 API.

(t01_train.py = bản cặp crop↔glyph hai nhánh: 1,1 s/bước trên MPS, quá chậm cho ngân sách 25 phút → thay bằng bản này:
 nhánh glyph được thay bằng một NGUYÊN MẪU học W_x cho mỗi chữ x trong vũ trụ ứng viên; nguyên mẫu của chữ không có crop
 người được học từ bản render font biến dạng. Điểm cặp s(crop, x) = cos(f(crop), W_x).)

argv: TUNE_BOOK (LucVanTien1916|TruyenKieu1872)  STEPS  [TAG]
  Crop người: Borg keep=1 fold 0–3; IHR sách TUNE phần A trang (thứ hạng trang %4 != 3) — slot_ok=1&gt_char → gt_char,
              slot_ok=0&gt_img → gt_img. Sách IHR kia KHÔNG dùng (LOBO). Borg fold 4 + TUNE phần B = giữ ngoài.
  Font     : mọi chữ có font trong vũ trụ ứng viên (p01), render font 0/1 biến dạng mạnh + mực chữ kề.
  Lớp      : toàn vũ trụ chữ có font (+ chữ người không font). Mất mát = CosFace (s=30, m=0.2) toàn lớp, che lớp biến thể
             của đích + 0.5 × CE trong tập ứng viên cứng (đồng âm R(âm), top-5 gần hình, chữ kề = trượt).
Ra: out/model_<TAG>.pt (fp16), out/emb_<TAG>_{borg,ihr,gold}.f16.npy, out/W_<TAG>.f16.npy, out/classes_<TAG>.json, out/t02_<TAG>.json
"""
import json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad/r5/hand_lobo')
sys.path.insert(0, '/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad/r4/verifier_ft')
from vflib import OUT as VOUT, Net, to_t, affine, stroke, photometric, neighbour_ink, load_json  # noqa
from hlib import OUT  # noqa
OUT = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/measure_out/_thu_nghiem_anh_chu/TN4/v2/hand_retrain')  # TN4: thư mục học lại trên crop chuẩn v2
from harness_lib import R_of, variants_of, simp  # noqa

BKS = {'Kinh': 'SachKinhThayCaBinh', 'DungLy': 'SachDungLyHoThan'}
BK = sys.argv[1]; TRAIN_BOOK = BKS[BK]
TUNE = sys.argv[2]
STEPS = int(sys.argv[3])
TAG = sys.argv[4]
FT = f'{TAG}_{BK}'
SEED = 0
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
WID = (32, 64, 128, 256)
NB_, NI_, NS_ = 112, 96, 48
KC = 24          # tối đa ứng viên cứng / crop
S_, M_ = 30.0, 0.2
T0 = time.time()


def log(s):
    print(f'[{time.time()-T0:5.0f}s] {s}', flush=True)


U = load_json(VOUT / 'glyph_chars.json'); uid = {c: i for i, c in enumerate(U)}
GL = np.load(VOUT / 'glyph.npy'); NF = np.load(VOUT / 'glyph_nf.npy')
SIM5 = load_json(VOUT / 'simtop5.json')
simp_id = {}
SID = np.array([simp_id.setdefault(simp(c), len(simp_id)) for c in U])
by_sid = {}
for k, s in enumerate(SID):
    by_sid.setdefault(s, []).append(k)


def var_set(k):
    out = set(by_sid[SID[k]])
    out |= {uid[v] for v in variants_of(U[k]) if v in uid}
    out.discard(k)
    return out


def hard(chars, k, vs):
    return [uid[c] for c in dict.fromkeys(chars) if c in uid and uid[c] != k and uid[c] not in vs]


def pages_split(pages):
    ps = sorted(set(pages)); rank = {p: i for i, p in enumerate(ps)}
    return np.array([rank[p] % 4 == 3 for p in pages])


XB = np.load(OUT / 'img_all.npy'); MB = pd.read_pickle(OUT / 'meta_all.pkl')
XI = np.load(OUT / 'img_ihr.npy'); MI = pd.read_pickle(VOUT / 'meta_ihr.pkl')
items = {'borg': [], 'ihr': []}
skip = Counter()


def pack(k, cand, vs):
    c = np.full(KC, -1, np.int64)
    cand = [x for x in cand][:KC - 1]
    c[0] = k; c[1:1 + len(cand)] = cand
    v = np.full(16, -1, np.int64); vl = list(vs)[:16]; v[:len(vl)] = vl
    return c, v


for r, (c, key, pv, nx, kp, fo, bk_) in enumerate(zip(MB.char, MB.R_key, MB.nb_prev, MB.nb_next, MB.keep, MB.fold, MB.book)):
    if kp != 1 or fo == 4 or bk_ != TRAIN_BOOK or len(c) != 1:
        continue
    k = uid.get(c)
    if k is None:
        skip['borg_nouniv'] += 1; continue
    vs = var_set(k)
    Rl = list(R_of(key)) if key else []
    rng.shuffle(Rl)
    cand = hard([pv, nx] + SIM5.get(c, [])[:5] + Rl, k, vs)
    items['borg'].append((r, *pack(k, cand, vs)))
MI['partB'] = pages_split(MI.page.values)
for r, row in enumerate(MI.itertuples()):
    if row.book != TUNE or row.partB:
        continue
    if row.slot_ok == '1' and row.gt_char:
        t = row.gt_char; nb = [row.gt_prev, row.gt_next]
    elif row.slot_ok == '0' and row.gt_img:
        t = row.gt_img; nb = [row.gt_char, row.gt_prev, row.gt_next]
    else:
        continue
    k = uid.get(t)
    if k is None:
        skip['ihr_nouniv'] += 1; continue
    vs = var_set(k)
    Rl = list(R_of(row.syllable)); rng.shuffle(Rl)
    cand = hard(nb + SIM5.get(t, [])[:5] + Rl, k, vs)
    items['ihr'].append((r, *pack(k, cand, vs)))
syn_pool = np.nonzero(NF > 0)[0]
syn_c = []
for k in syn_pool:
    vs = var_set(k)
    syn_c.append(pack(k, hard(SIM5.get(U[k], [])[:5], k, vs), vs))
log(f"items borg {len(items['borg'])} ihr {len(items['ihr'])} syn {len(syn_pool)} classes {len(U)} skip {dict(skip)}")


def stack(lst):
    rows = np.array([x[0] for x in lst]); C = np.stack([x[1] for x in lst]); V = np.stack([x[2] for x in lst])
    return rows, torch.from_numpy(C), torch.from_numpy(V)


RB, CB, VB = stack(items['borg']); RI, CI, VI = stack(items['ihr'])
CS = torch.from_numpy(np.stack([x[0] for x in syn_c])); VS = torch.from_numpy(np.stack([x[1] for x in syn_c]))
XBt = to_t(XB); XIt = to_t(XI); GLt = torch.from_numpy(GL)
del XB, XI

net = Net(WID).to(DEV)
Wc = nn.Parameter(torch.randn(len(U), 256, device=DEV) * 0.05)
params = list(net.parameters()) + [Wc]
opt = torch.optim.AdamW([{'params': net.parameters(), 'weight_decay': 5e-4}, {'params': [Wc], 'weight_decay': 0.0}], lr=3e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=STEPS, pct_start=0.1)
log(f'train borg={TRAIN_BOOK} ihr={TUNE} steps {STEPS} dev {DEV}')
tl, ta, th, tn = 0.0, 0, 0, 0
for step in range(STEPS):
    net.train()
    ib = rng.integers(0, len(RB), NB_); ii = rng.integers(0, len(RI), NI_); js = rng.integers(0, len(syn_pool), NS_)
    real = torch.cat([XBt[RB[ib]], XIt[RI[ii]]], 0).to(DEV)
    fv = torch.from_numpy(rng.integers(0, 2, NS_))
    syn = GLt[torch.from_numpy(syn_pool[js]), fv].float().div(255).sub(0.5).div(0.5)[:, None].to(DEV)
    C = torch.cat([CB[ib], CI[ii], CS[js]], 0).to(DEV); V = torch.cat([VB[ib], VI[ii], VS[js]], 0).to(DEV)
    real = photometric(stroke(affine(real, rot=4, scale=0.08, shift=0.06), 0.15, 0.15), 0.3, 0.05)
    other = syn[torch.randperm(NS_, device=DEV)]
    syn = photometric(stroke(neighbour_ink(affine(syn, rot=8, scale=0.2, shift=0.12, sx=0.12), other, 0.4), 0.4, 0.3), 0.6, 0.12)
    z = net.crop(torch.cat([real, syn], 0))
    Wn = F.normalize(Wc, dim=1)
    cos = z @ Wn.T                                            # (n, |U|)
    y = C[:, 0]
    n = len(y); ar = torch.arange(n, device=DEV)
    # che lớp biến thể của đích
    vm = torch.zeros_like(cos, dtype=torch.bool)
    vv = V.clamp(min=0); vm[ar[:, None].expand_as(vv), vv] = (V >= 0)
    lg = S_ * cos
    lg = lg.masked_fill(vm, -1e4)
    lg[ar, y] = S_ * (cos[ar, y] - M_)
    loss_all = F.cross_entropy(lg, y)
    # CE trong tập ứng viên cứng (cột 0 = đích)
    cc = C.clamp(min=0)
    lh = torch.gather(lg, 1, cc).masked_fill(C < 0, -1e4)
    loss_h = F.cross_entropy(lh, torch.zeros(n, dtype=torch.long, device=DEV))
    loss = loss_all + 0.5 * loss_h
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        tl += float(loss) * n; ta += int((cos.masked_fill(vm, -9).argmax(1) == y).sum())
        th += int((lh.argmax(1) == 0).sum()); tn += n
    if (step + 1) % 250 == 0 or step == STEPS - 1:
        log(f'step {step+1} loss {tl/tn:.3f} top1_all {ta/tn:.3f} top1_hard {th/tn:.3f}')
        tl, ta, th, tn = 0.0, 0, 0, 0
net.eval()


@torch.no_grad()
def emb(X, bs=2048):
    out = []
    for s in range(0, len(X), bs):
        out.append(net.crop(X[s:s + bs].to(DEV)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


torch.save({'net': {k: v.half() for k, v in net.state_dict().items()}, 'W': Wc.detach().half().cpu(), 'widths': WID,
            'classes_file': 'glyph_chars.json'}, OUT / f'model_{FT}.pt')
np.save(OUT / f'W_{FT}.f16.npy', F.normalize(Wc.detach(), dim=1).cpu().numpy().astype(np.float16))
np.save(OUT / f'emb_{FT}_all.f16.npy', emb(XBt))
np.save(OUT / f'emb_{FT}_ihr.f16.npy', emb(XIt))
del XBt, XIt
XG = to_t(np.load(OUT / 'img_stt.npy'))
np.save(OUT / f'emb_{FT}_stt.f16.npy', emb(XG))
json.dump(dict(train_book=TRAIN_BOOK, tune=TUNE, steps=STEPS, tag=TAG, widths=WID, n_borg=len(RB), n_ihr=len(RI), n_syn=int(len(syn_pool)),
               classes=len(U), skip=dict(skip), sec=round(time.time() - T0), batch=[NB_, NI_, NS_], s=S_, m=M_, seed=SEED),
          open(OUT / f'h01_{FT}.json', 'w'), indent=1)
log('done')
