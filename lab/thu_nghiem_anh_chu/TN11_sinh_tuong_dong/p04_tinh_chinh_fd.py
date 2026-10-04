"""TN11 p04 (03/10) — TINH CHỈNH FontDiffuser trên ô neo TỰ ĐỘNG của chính cuốn, rồi sinh chữ hiếm và so ảnh (0 API).

Theo tài liệu (Gui et al., ICDAR 2023; Ao et al., ICASSP 2024): ảnh sinh chỉ thay được mẫu thật khi bộ sinh đã HỌC trên chữ thật
cùng miền (lớp đã thấy) rồi khái quát sang lớp chưa thấy. p02 cho thấy FontDiffuser có sẵn + một ảnh phong cách (one-shot)
KHÔNG đủ. Ở đây: cặp (glyph phông NomNaTong của nhãn kim, crop ô neo) của sách --book (kim ∈ R, nhãn = kim; KHÔNG nhãn người).
Loại khỏi tập học: chữ có < 3 ô neo và mọi CHỮ ĐÚNG của 200 ô đo p02 (đo khả năng sinh chữ CHƯA THẤY; chặt hơn cần thiết).
Học: MSE dự đoán nhiễu (như huấn luyện gốc), bỏ điều kiện 10 % (giữ classifier-free guidance), AdamW lr 1e-5, UNet + bộ mã
phong cách (bộ mã nội dung đóng băng). Sinh lại 1.019 chữ của p02 với phong cách style_0 -> p02_<bộ>/gen_ft/s0/, đo bằng
p02 --stage eval --gen-sub gen_ft --styles 1.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p04_tinh_chinh_fd.py --book B34 --stage data|train|gen [--steps 1500]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "font_diffusion"))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
from p02_fd_phong_cach_sach import anchors, style_image  # noqa: E402

CK = REPO / "font_diffusion/ckpt/PROD"


FIXED = False      # --fixed: đường chuẩn use_fst=False (fd_wrapper_fix.py)


def gen_obj(out):
    if FIXED:
        from fd_wrapper_fix import FixedFontDiffusionGenerator as FontDiffusionGenerator
    else:
        from core.ranking.fontdiffusion_gen import FontDiffusionGenerator
    g = FontDiffusionGenerator(ckpt_dir=str(CK), phase1_ckpt_dir=str(CK), font_path=str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf"),
                               cache_dir=str(out), batch_size=8)
    g._load_pipeline()
    return g


def data(a, w):
    b = a.book
    D = T.load_base(b); meta = np.load(T.OUT / "emb" / f"{b}_crop_meta.npy")
    an = np.nonzero(anchors(b, D, meta))[0]
    lab = D.label.values[an]
    cnt = Counter(lab)
    P = pd.read_pickle(REPO / "measure_out/_tn11" / f"p02_{b}" / "plan.pkl")
    S = P["S"]; held = set(S[S.y == 1].c)
    keep = np.array([cnt[l] >= 3 and l not in held for l in lab])
    an = an[keep]
    if a.max_pairs and len(an) > a.max_pairs:
        an = np.sort(np.random.default_rng(0).choice(an, a.max_pairs, replace=False))
    from t02_embed import page_crops
    cfg = T.BOOKS[b]; prep = T.REPO / cfg["prep"]
    want = set(an.tolist()); X, L = [], []
    for pg in sorted(set(D.page.values[an])):
        allp = D[D.page == pg]
        recs = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox)]
        for ii, gg, _ in page_crops((b, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs)):
            if ii in want and gg is not None and min(gg.shape) >= 8:
                X.append(style_image(gg)); L.append(D.label.values[ii])
    np.savez_compressed(w / "pairs.npz", X=np.stack(X), L=np.array(L))
    print(f"[p04] {b}: {len(X)} cặp học ({len(set(L))} chữ); loại {len(held)} chữ đúng của mẫu đo + chữ < 3 neo")


def train(a, w):
    from inference.sample_optimized import get_content_transform, get_style_transform
    from src.tools.utils import ttf2im
    Z = np.load(w / "pairs.npz"); X, L = Z["X"], Z["L"]
    g = gen_obj(w / "_tmp")
    m = g.pipe.model; dev = g.device; args = g.args
    from src.builders.build import build_ddpm_scheduler
    acp = build_ddpm_scheduler(args).alphas_cumprod.to(torch.float32)
    font = g.font_manager.get_font(g.font_manager.get_font_names()[0])
    ct = get_content_transform(args.content_image_size); stt = get_style_transform(args.style_image_size)
    cont = {}
    for c in set(L):
        try:
            im = ttf2im(font=font, char=c)
        except Exception:
            im = None
        if im is not None and np.asarray(im).min() < 200:
            cont[c] = ct(im)
    ok = np.array([l in cont for l in L]); X, L = X[ok], L[ok]
    tgt = torch.stack([stt(Image.fromarray(x).convert("RGB")) for x in X])
    print(f"[p04] học {len(L)} cặp, {len(cont)} chữ có glyph nội dung; {a.steps} bước × lô {a.bs}", flush=True)
    for p in m.content_encoder.parameters():
        p.requires_grad_(False)
    params = [p for p in list(m.unet.parameters()) + list(m.style_encoder.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    m.train()
    rng = np.random.default_rng(0); torch.manual_seed(0)
    t0 = time.time(); run = []
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, len(L), a.bs)
        sidx = np.array([rng.choice(np.nonzero(L != L[i])[0][:5000]) for i in idx])   # phong cách: ô neo chữ KHÁC, cùng sách
        x0 = tgt[idx]; st = tgt[sidx].clone(); ci = torch.stack([cont[L[i]] for i in idx])
        drop = torch.rand(a.bs) < 0.1
        ci[drop] = 1; st[drop] = 1
        ts = torch.randint(0, 1000, (a.bs,))
        eps = torch.randn_like(x0)
        ab = acp[ts].view(-1, 1, 1, 1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        pred = m(xt.to(dev), ts.to(dev), [ci.to(dev), st.to(dev)], content_encoder_downsample_size=args.content_encoder_downsample_size,
                 version="V3")
        loss = torch.nn.functional.mse_loss(pred.float(), eps.to(dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        run.append(float(loss))
        if step % 100 == 0:
            print(f"[p04] bước {step}: loss {np.mean(run[-100:]):.4f} [{time.time() - t0:.0f}s]", flush=True)
    m.eval()
    torch.save({"unet": m.unet.state_dict(), "style_encoder": m.style_encoder.state_dict()}, w / "ft.pt")
    print(f"[p04] lưu {w / 'ft.pt'}", flush=True)


def gen(a, w):
    out = REPO / "measure_out/_tn11" / f"p02_{a.book}"
    P = pd.read_pickle(out / "plan.pkl")
    g = gen_obj(out / "gen_ft")
    sd = torch.load(w / "ft.pt", map_location="cpu")
    g.pipe.model.unet.load_state_dict(sd["unet"]); g.pipe.model.style_encoder.load_state_dict(sd["style_encoder"])
    g.pipe.model.eval(); g.pipe.guidance_scale = 2.0
    torch.manual_seed(0)
    todo = [c for c in P["chars"] if not (out / "gen_ft" / "s0" / f"U+{ord(c):04X}.png").exists()]
    t0 = time.time()
    if todo:
        g.generate(todo, str(out / "style_0.png"), style_name="s0")
    print(f"[p04] sinh {len(todo)} chữ bằng mô hình tinh chỉnh [{time.time() - t0:.0f}s]", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="B34")
    ap.add_argument("--stage", choices=["data", "train", "gen"], required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--fixed", action="store_true", help="dùng đường FontDiffuser chuẩn (use_fst=False); thư mục làm việc p04_<bộ>_fix")
    ap.add_argument("--max-pairs", type=int, default=0, help="giới hạn số cặp học (0 = tất cả), lấy ngẫu nhiên seed 0")
    a = ap.parse_args(argv)
    global FIXED
    FIXED = a.fixed
    w = REPO / "measure_out/_tn11" / f"p04_{a.book}{'_fix' if a.fixed else ''}"
    w.mkdir(parents=True, exist_ok=True)
    {"data": data, "train": train, "gen": gen}[a.stage](a, w)


if __name__ == "__main__":
    main()
