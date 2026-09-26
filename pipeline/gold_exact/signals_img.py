"""signals_img.py — bộ kiểm ảnh↔chữ của cấu hình "lai": CHẤM CROP CŨ (tệp giao nộp dataset/_ALL/<image>), 0 API.

Bản chép NGUYÊN các bước đã đo (thư mục thử nghiệm), chỉ đổi nơi đọc tài sản:
  p_wood_T/L  verifier_ft: CNN (r4/verifier_ft vflib.Net) trên ảnh 64×64 (p01_prep.prep) -> đặc trưng cặp (s01_features)
              -> logistic 'wood' (s02_eval). Nhúng làm tròn qua float16 như emb_*.f16.npy; p làm tròn 5 chữ số (gold_scores).
  lobo cert   STT: CNN chữ viết tay LOBO-sách (r5/hand_lobo, mô hình + nguyên mẫu + hiệu chuẩn chỉ Kinh) -> logistic 'hand'
              đầy đủ; chứng nhận = p_T ≥ t_T(q) ∧ p_L ≥ t_L(q) (h03), n_hum(nhãn) ≥ 3 (lobo_nh).
  vis_z       score_visual (gold_img_audit): encoder v1+v2 (nom-embed + ArcFace, repo) trên crop -> m_win_glyph = s_glyph(nhãn)
              − max s_glyph(chữ ô lân cận n±1..2 cùng cột); vis_z = (m − t_mod)/(t_mod − t_cons) theo bộ (measure/calib.json).
  m_hom       M-OCR (m_ocr/h02_score): cos(crop, font nhãn) − max cos(crop, font đối thủ R(âm) ∪ chữ dị bản); cờ t50 theo bộ.
  viss_T/L/X  xếp hạng lại (kim_bottleneck/rerank r02/r02b/r03 'viss'): logit có điều kiện trên R(âm) ∪ {kim}, p_kim.
Mọi nhúng được cache theo md5 ảnh + sha mô hình (common.EmbCache).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import REPO, EmbCache, R_of, f16, md5_bytes, simp, variants_of

# ================================================================================================ tiền xử lý 64×64
SZ = 64


def prep64(gray):
    """== r4/verifier_ft/p01_prep.prep (cũng là hand_lobo/hlib.prep)."""
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


# ================================================================================================ CNN (vflib.Net)
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class Net(nn.Module):
    def __init__(self, widths=(48, 96, 192, 384), dim=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers += [Block(cin, w, 1 if i == 0 else 2), Block(w, w, 1)]
            cin = w
        self.body = nn.Sequential(*layers)
        self.crop_proj = nn.Sequential(nn.Linear(cin, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, dim))
        self.glyph_proj = nn.Sequential(nn.Linear(cin, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, dim))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def trunk(self, x):
        return F.adaptive_avg_pool2d(self.body(self.stem(x)), 1).flatten(1)

    def crop(self, x):
        return F.normalize(self.crop_proj(self.trunk(x)), dim=1)


def to_t(x_uint8):
    return torch.from_numpy(np.ascontiguousarray(x_uint8)).float().div(255.0).sub(0.5).div(0.5)[:, None]


def load_net(ck, dev):
    net = Net(widths=tuple(ck["widths"]))
    net.load_state_dict({k: v.float() for k, v in ck["net"].items()})
    return net.to(dev).eval()


@torch.no_grad()
def net_embed(net, X, dev, bs=2048):
    out = []
    for s in range(0, len(X), bs):
        out.append(net.crop(to_t(X[s:s + bs]).to(dev)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 256), np.float32)


# ================================================================================================ đặc trưng cặp (s01)
class Universe:
    def __init__(self, U, SIM5):
        self.U = list(U); self.uid = {c: i for i, c in enumerate(self.U)}; self.SIM5 = SIM5
        simp_id = {}
        self.SID = np.array([simp_id.setdefault(simp(c), len(simp_id)) for c in self.U])
        self.by_sid = defaultdict(list)
        for k, s in enumerate(self.SID):
            self.by_sid[s].append(k)
        self._vs = {}

    def var_set(self, k):
        if k not in self._vs:
            o = set(self.by_sid[self.SID[k]]) | {self.uid[v] for v in variants_of(self.U[k]) if v in self.uid}
            o.add(k); self._vs[k] = o
        return self._vs[k]


def features(uni: Universe, Z, W, P, nh, rows, cands, syls, nbps, nbns):
    """== s01_features.features (sw, mR, mS, mN, mRSN, mG, se, meR, meRSN, n_hum, in_univ)."""
    uid, U, SIM5 = uni.uid, uni.U, uni.SIM5
    n = len(rows)
    out = {k: np.full(n, np.nan, np.float32) for k in ("sw", "mR", "mS", "mN", "mRSN", "mG", "se", "meR", "meRSN")}
    out["n_hum"] = np.zeros(n, np.int32); out["in_univ"] = np.zeros(n, np.int8)
    Rcache = {}
    for s0 in range(0, n, 4096):
        idx = np.arange(s0, min(n, s0 + 4096))
        Zb = Z[rows[idx]]
        S = Zb @ W.T
        top = np.argpartition(-S, 12, axis=1)[:, :12]
        SP_ = Zb @ P.T if P is not None else None
        for j, i in enumerate(idx):
            x = cands[i]
            k = uid.get(x)
            if k is None:
                continue
            out["in_univ"][i] = 1
            vs = uni.var_set(k)
            s = S[j]
            sx = s[k]; out["sw"][i] = sx
            tt = sorted(top[j], key=lambda t: -s[t])
            for t in tt:
                if t not in vs:
                    out["mG"][i] = sx - s[t]; break
            syl = syls[i]
            if syl not in Rcache:
                Rcache[syl] = [uid[c] for c in R_of(syl) if c in uid]
            groups = {"mR": Rcache[syl], "mS": [uid[c] for c in SIM5.get(x, []) if c in uid],
                      "mN": [uid[c] for c in (nbps[i], nbns[i]) if c and c in uid]}
            allr = []
            for gname, gl in groups.items():
                gl = [t for t in gl if t not in vs]
                if gl:
                    out[gname][i] = sx - s[gl].max(); allr += gl
            if allr:
                t = max(allr, key=lambda t: s[t]); out["mRSN"][i] = sx - s[t]
            out["n_hum"][i] = nh[k]
            if SP_ is not None and nh[k] >= 3:
                sp = SP_[j]; out["se"][i] = sp[k]
                rr = [t for t in groups["mR"] if t not in vs and nh[t] >= 3]
                if rr:
                    out["meR"][i] = sp[k] - sp[rr].max()
                ra = [t for t in allr if nh[t] >= 3]
                if ra:
                    out["meRSN"][i] = sp[k] - sp[ra].max()
    return pd.DataFrame(out)


def X_of(F_):
    """== s02_eval.X_of."""
    def fill(c, v):
        a = F_[c].values.astype(np.float64); m = np.isnan(a)
        return np.where(m, v, a), m.astype(np.float64)
    sw, _ = fill("sw", 0.0); mG, _ = fill("mG", 0.0)
    mR, nR = fill("mR", 0.3); mS, nS = fill("mS", 0.3); mN, nN = fill("mN", 0.3); mA, nA = fill("mRSN", 0.3)
    se, ne = fill("se", 0.0); meA, nmeA = fill("meRSN", 0.3)
    lh = np.log1p(F_.n_hum.values.astype(np.float64))
    X = np.stack([sw, mG, np.minimum(mG, 0), mR, mS, mN, mA, np.minimum(mA, 0), nR, nN, se, meA, np.minimum(meA, 0), ne,
                  nmeA, lh], 1)
    X[F_.in_univ.values == 0] = 0
    return X


class Logit:
    """== s02_eval.Logit (clip=False) / h03_eval.Logit (clip=True: kẹp z ở ±50)."""

    def __init__(self, mu=None, sd=None, b=None, clip=False):
        self.mu, self.sd, self.b, self.clip = mu, sd, b, clip

    def fit(self, X, y, w=None, l2=1e-2):
        from scipy.optimize import minimize
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        w = np.ones(len(y)) if w is None else w

        def f(b):
            z = Z @ b; p = 1 / (1 + np.exp(-z))
            ll = -(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() / w.sum() + l2 * (b[:-1] ** 2).sum()
            g = Z.T @ (w * (p - y)) / w.sum(); g[:-1] += 2 * l2 * b[:-1]
            return ll, g
        self.b = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def p(self, X):
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        z = Z @ self.b
        if self.clip:
            z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))


# ================================================================================================ encoder v1+v2 (sv_lib)
MEAN = 0.5
FONT_FILES = ("fonts/NomNaTong-Regular.ttf", "fonts/PlangothicP1-Regular.ttf", "fonts/PlangothicP2-Regular.ttf")
FD_DIRS = ("ArcFace/data/glyphs", "gannhanocr-fd")
ENC_FILES = {"v1": "nom-embed/best.pt", "v2": "ArcFace/checkpoints/best.pt"}


class Enc:
    def __init__(self, path, device, norm):
        from pipeline.align_engine.nom_classifier.model import NomEmbedder
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.size = ck.get("img", 128)
        self.dev = torch.device(device)
        self.net = NomEmbedder(ck.get("embed_dim", 256), pretrained=False, arch=ck.get("arch") or "resnet18")
        self.net.load_state_dict(ck["backbone"])
        self.net.eval().to(self.dev)
        W = ck["head"]["W"].float()
        self.Wn = F.normalize(W, dim=1).numpy()
        cl = ck["classes"]
        self.lab2idx = dict(cl) if isinstance(cl, dict) else {c: i for i, c in enumerate(cl)}
        self.norm = norm

    def prep(self, gray):
        if self.norm:
            lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
            if hi - lo > 10:
                gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        h, w = gray.shape
        s = max(h, w)
        can = np.full((s, s), 255, np.uint8)
        can[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = gray
        g = cv2.resize(can, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return (g.astype(np.float32) / 255.0 - MEAN) / 0.5

    @torch.no_grad()
    def embed(self, grays, bs=256):
        out = []
        for i in range(0, len(grays), bs):
            x = np.stack([self.prep(g) for g in grays[i:i + bs]])
            x = torch.from_numpy(np.repeat(x[:, None], 3, axis=1)).to(self.dev)
            out.append(self.net(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 256), np.float32)


class MultiEnc:
    """[e1, e2]/sqrt(2): tích vô hướng = trung bình cosine hai encoder; head = head encoder ĐẦU (v1)."""

    def __init__(self, paths, device, norm=True):
        self.encs = [Enc(p, device, norm) for p in paths]
        self.k = len(self.encs)
        self.lab2idx = self.encs[0].lab2idx
        W0 = self.encs[0].Wn
        pad = np.zeros((W0.shape[0], W0.shape[1] * (self.k - 1)), np.float32)
        self.Wn = np.concatenate([W0, pad], 1) * np.sqrt(self.k)

    def embed(self, grays, norm=None):
        outs = []
        for e in self.encs:
            keep = e.norm
            if norm is not None:
                e.norm = norm
            outs.append(e.embed(grays))
            e.norm = keep
        return (np.concatenate(outs, 1) / np.sqrt(self.k)).astype(np.float32)


class Glyphs:
    """== sv_lib.Glyphs: font NomNaTong → Plangothic P1/P2 (render 112×112, cỡ 84) + ảnh FD (bỏ con trỏ LFS < 1 KB)."""

    def __init__(self):
        from PIL import ImageFont
        from fontTools.ttLib import TTFont
        self.fonts = []
        for f in FONT_FILES:
            p = REPO / f
            if p.exists():
                self.fonts.append((ImageFont.truetype(str(p), 84), set(TTFont(str(p)).getBestCmap())))
        self.fd = {}
        for d in FD_DIRS:
            d = REPO / d
            if not d.exists():
                continue
            for root, _, fs in os.walk(d):
                for fn in fs:
                    if fn.startswith("U+") and fn.endswith(".png"):
                        p = Path(root) / fn
                        if p.stat().st_size < 1024:
                            continue
                        try:
                            ch = chr(int(fn[2:-4], 16))
                        except ValueError:
                            continue
                        self.fd.setdefault(ch, str(p))
        self._font = {}

    def font(self, ch):
        if ch not in self._font:
            from PIL import Image, ImageDraw
            r = None
            for f, cmap in self.fonts:
                if ord(ch) in cmap:
                    im = Image.new("L", (112, 112), 255)
                    ImageDraw.Draw(im).text((56, 56), ch, font=f, fill=0, anchor="mm")
                    r = np.array(im); break
            self._font[ch] = r
        return self._font[ch]

    def fdimg(self, ch):
        p = self.fd.get(ch)
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE) if p else None


def augs(im):
    """== rerank/r02_candidates.augs (gốc, đậm, mảnh, nhoè, thu nhỏ)."""
    k = np.ones((3, 3), np.uint8)
    out = [im, cv2.erode(im, k, 1), cv2.dilate(im, k, 1)]
    b = cv2.GaussianBlur(im, (5, 5), 1.5)
    out.append(np.where(b < 170, 0, 255).astype(np.uint8))
    s = cv2.resize(im, (84, 84), interpolation=cv2.INTER_AREA)
    c = np.full_like(im, 255); c[14:98, 14:98] = s; out.append(c)
    return out


# ================================================================================================ nạp ảnh + nhúng (cache)
def read_crop(path):
    b = Path(path).read_bytes() if Path(path).exists() else b""
    return md5_bytes(b), b


def decode_gray(b):
    if not b:
        return None
    x = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
    return x if x is not None and x.size else None


class Scorers:
    """Nạp mô hình + bảng một lần; cache nhúng theo md5 ảnh."""

    def __init__(self, assets, device, cache_dir, recompute, log):
        self.A, self.log = assets, log
        self.dev = torch.device(device)
        self.cache_dir, self.recompute = cache_dir, recompute
        self._enc = None; self._gl = None

    def enc(self):
        if self._enc is None:
            self._enc = MultiEnc([self.A.ext(ENC_FILES["v1"]), self.A.ext(ENC_FILES["v2"])], self.dev.type, True)
            for f in FONT_FILES:
                self.A.ext(f)
            for d in FD_DIRS:
                self.A.ext(d)
        return self._enc

    def enc_sha(self):
        return hashlib.sha256((self.A.sha(ENC_FILES["v1"]) + self.A.sha(ENC_FILES["v2"])).encode()).hexdigest()

    def glyphs(self):
        if self._gl is None:
            self._gl = Glyphs()
        return self._gl

    # ---------------------------------------------------------------- nhúng crop giao nộp (6 mô hình, một lượt đọc ảnh)
    def embed_crops(self, keys, paths, need_enc=True, need_vft=True, need_hand_mask=None):
        """keys = md5 tệp; trả dict tên -> mảng (N, d) theo thứ tự keys (float32, CHƯA làm tròn f16)."""
        C = {}
        specs = []
        if need_vft:
            for t in "TL":
                C[f"vft_{t}"] = EmbCache(self.cache_dir, f"vft_{t}", self.A.sha(f"vft_model_{t}.pt"), recompute=self.recompute)
                specs.append((f"vft_{t}", f"vft_model_{t}.pt", None))
        if need_hand_mask is not None and need_hand_mask.any():
            for t in "TL":
                C[f"hand_{t}"] = EmbCache(self.cache_dir, f"hand_{t}_Kinh", self.A.sha(f"hand_model_{t}_Kinh.pt"), recompute=self.recompute)
                specs.append((f"hand_{t}", f"hand_model_{t}_Kinh.pt", need_hand_mask))
        if need_enc:
            C["enc"] = EmbCache(self.cache_dir, "enc_v1v2_norm", self.enc_sha(), recompute=self.recompute)
        keys = list(keys)
        kset = {}
        for name, cache in C.items():
            if name.startswith("hand_"):
                sub = [k for k, m in zip(keys, need_hand_mask) if m]
                kset[name] = set(cache.missing(sub))
            else:
                kset[name] = set(cache.missing(keys))
        todo = sorted(set().union(*kset.values())) if kset else []
        self.log(f"nhúng crop: {len(set(keys))} ảnh, cần tính {len(todo)} (" +
                 ", ".join(f"{k}:{len(v)}" for k, v in kset.items()) + ")")
        if todo:
            path_of = dict(zip(keys, paths))
            nets = {}
            for name, fn, _ in specs:
                if kset[name]:
                    nets[name] = load_net(self.A.load(fn), self.dev)
            enc = self.enc() if need_enc and kset.get("enc") else None
            CH = 8192
            for s in range(0, len(todo), CH):
                part = todo[s:s + CH]
                with ThreadPoolExecutor(8) as ex:
                    grays = list(ex.map(lambda k: decode_gray(Path(path_of[k]).read_bytes()) if Path(path_of[k]).exists() else None,
                                        part))
                g8 = [x if x is not None else np.full((8, 8), 255, np.uint8) for x in grays]
                need64 = [n for n in nets if any(k in kset[n] for k in part)]
                if need64:
                    X64 = np.stack([prep64(x) for x in grays])
                    for n in need64:
                        sel = [i for i, k in enumerate(part) if k in kset[n]]
                        C[n].put([part[i] for i in sel], net_embed(nets[n], X64[sel], self.dev))
                if enc is not None:
                    sel = [i for i, k in enumerate(part) if k in kset["enc"]]
                    if sel:
                        C["enc"].put([part[i] for i in sel], enc.embed([g8[i] for i in sel]))
                self.log(f"  nhúng {min(s + CH, len(todo))}/{len(todo)}")
            for c in C.values():
                c.save()
        out = {}
        for name, cache in C.items():
            if name.startswith("hand_"):
                sub = [k for k, m in zip(keys, need_hand_mask) if m]
                out[name] = cache.get(sub)
            else:
                out[name] = cache.get(keys)
        return out

    # ---------------------------------------------------------------- glyph font / FD (nhúng norm=False)
    def glyph_emb(self, chars, with_aug=False, with_fd=False):
        gl = self.glyphs(); enc = None
        tag = "glyph_font"
        cache = EmbCache(self.cache_dir, tag, self.enc_sha(), recompute=self.recompute)
        FE, FA, DE, gmd5 = {}, {}, {}, {}
        items = []
        for ch in chars:
            im = gl.font(ch)
            if im is None:
                continue
            h = md5_bytes(im.tobytes())
            gmd5[ch] = h
            items.append((ch, im, h))
        need = []
        for ch, im, h in items:
            if h + ":n" not in cache.d:
                need.append((h + ":n", im))
            if with_aug:
                for j, a in enumerate(augs(im)):
                    if f"{h}:a{j}" not in cache.d:
                        need.append((f"{h}:a{j}", a))
        fdk = {}
        if with_fd:
            for ch in chars:
                p = gl.fd.get(ch)
                if p:
                    im = gl.fdimg(ch)
                    if im is None:
                        continue
                    h = "fd:" + md5_bytes(Path(p).read_bytes())
                    fdk[ch] = h
                    if h not in cache.d:
                        need.append((h, im))
        if need:
            enc = self.enc()
            self.log(f"nhúng glyph: {len(need)} ảnh")
            for s in range(0, len(need), 4096):
                part = need[s:s + 4096]
                cache.put([k for k, _ in part], enc.embed([im for _, im in part], norm=False))
            cache.save()
        for ch, im, h in items:
            FE[ch] = cache.d[h + ":n"]
            if with_aug:
                v = np.stack([cache.d[f"{h}:a{j}"] for j in range(5)]).mean(0)
                FA[ch] = v / np.linalg.norm(v)
        for ch, h in fdk.items():
            DE[ch] = cache.d[h]
        return FE, FA, DE, gmd5


# ================================================================================================ p_wood (verifier_ft)
def score_vft(S: Scorers, G, E, uni, log):
    """E[t] = nhúng thô (N, 256) của crop GOLD; trả p_wood_T, p_wood_L (làm tròn 5) + n_hum_T."""
    res = {}
    for t in "TL":
        tb = S.A.load(f"vft_tables_{t}.pt")
        W = np.asarray(tb["W"], np.float32); P = np.asarray(tb["P"], np.float32); nh = np.asarray(tb["nh"])
        Z = f16(E[f"vft_{t}"])
        Fg = features(uni, Z, W, P, nh, np.arange(len(G)), G.label.values, G.syllable.values, G.nb_prev.values,
                      G.nb_next.values)
        m = Logit(tb["wood"]["mu"], tb["wood"]["sd"], tb["wood"]["b"], clip=False)
        res[f"p_wood_{t}"] = np.round(m.p(X_of(Fg)), 5)
        res[f"n_hum_vft_{t}"] = Fg.n_hum.values
        log(f"p_wood_{t} xong")
    return res


def score_hand(S: Scorers, G, E, uni, q, log):
    """STT: chứng nhận chữ viết tay LOBO (Kinh) ở mức q; trả lobo_pT, lobo_pL, lobo_nh (= n_hum mô hình T), cert."""
    ps, nh_T = {}, None
    thr = {}
    for t in "TL":
        tb = S.A.load(f"hand_tables_{t}_Kinh.pt")
        W = np.asarray(tb["W"], np.float32); P = np.asarray(tb["P"], np.float32); nh = np.asarray(tb["nh"])
        Z = f16(E[f"hand_{t}"])
        Fs = features(uni, Z, W, P, nh, np.arange(len(G)), G.label.values, G.syllable.values, G.nb_prev.values,
                      G.nb_next.values)
        m = Logit(tb["hand"]["mu"], tb["hand"]["sd"], tb["hand"]["b"], clip=True)
        ps[t] = m.p(X_of(Fs))
        thr[t] = float(tb["thr"][str(q)])
        if t == "T":
            nh_T = Fs.n_hum.values
        log(f"hand_{t} xong")
    cert = (ps["T"] >= thr["T"]) & (ps["L"] >= thr["L"])
    return dict(lobo_pT=ps["T"], lobo_pL=ps["L"], lobo_nh=nh_T, lobo_cert=cert.astype(np.int8), thr=thr)


# ================================================================================================ vis_z (score_visual)
def build_ctx(G):
    """(book, page, column, nom_idx) -> (ocr_char, label) từ bảng build ĐẦY ĐỦ (mọi tầng) — sv_lib.load_context."""
    FULLSEQ = {"SachThanhTruyen": REPO / "dataset_out" / "labels.csv"}
    for s in ("LucVanTien1883", "KimVanKieu1884", "Chrestomathie1872", "LucVanTien1916", "TruyenKieu1872"):
        FULLSEQ[s] = REPO / "prepared" / s / "dataset_out" / "labels.csv"
    ctx = {}
    for s in sorted(G.book_set.unique()):
        t = pd.read_csv(FULLSEQ[s], dtype=str, keep_default_na=False, usecols=["book", "page", "column", "nom_idx", "ocr_char", "label"])
        for b, p, c, n, o, l in zip(t["book"], t["page"], t["column"], t["nom_idx"], t["ocr_char"], t["label"]):
            try:
                ctx[(b, p, str(int(float(c))), int(float(n)))] = (o, l)
            except ValueError:
                pass
    return ctx


def score_vis(S: Scorers, G, Eenc, calib, log, WIN=2):
    ctx = build_ctx(G)
    ok_ctx = sum(1 for b, p, c, n, o in zip(G.book, G.page, G.column, G.nom_i, G.ocr_char) if ctx.get((b, p, c, n), ("#",))[0] == o)
    wins = []
    for b, p, c, n in zip(G.book, G.page, G.column, G.nom_i):
        w = set()
        for k in range(n - WIN, n + WIN + 1):
            if k == n:
                continue
            o, l = ctx.get((b, p, c, k), ("", ""))
            for ch in (o, l):
                if len(ch) == 1:
                    w.add(ch)
        wins.append(w)
    need = set(G.label) | set().union(*wins)
    FE, _, DE, _ = S.glyph_emb(sorted(need), with_aug=False, with_fd=True)
    E = Eenc
    m = np.full(len(G), np.nan)
    cache_g = {}
    for i, (L, W) in enumerate(zip(G.label.values, wins)):
        e = E[i]

        def sg(ch):
            v = []
            if ch in FE:
                v.append(float(e @ FE[ch]))
            if ch in DE:
                v.append(float(e @ DE[ch]))
            return float(np.mean(v)) if v else np.nan
        sl = sg(L)
        if np.isnan(sl):
            continue
        best, bc = -9.0, ""
        for c in W:
            if c == L:
                continue
            v = sg(c)
            if not np.isnan(v) and v > best:
                best, bc = v, c
        if bc:
            m[i] = sl - best
    bs = G.book_set.values
    tm = np.array([calib[b]["t_mod"] for b in bs]); tc = np.array([calib[b]["t_cons"] for b in bs])
    vis_z = (m - tm) / (tm - tc)
    log(f"vis_z xong (ctx khớp ocr_char {ok_ctx}/{len(G)})")
    return dict(vis_m_win_glyph=m, vis_z=vis_z, ctx_ok=ok_ctx)


# ================================================================================================ M-OCR (m_ocr h02)
def score_mocr(S: Scorers, G, Eenc, refs, t50, sets, log):
    """m_hom cho ô thuộc `sets` (bộ có M-OCR trong cấu hình gate_v2); cờ t50 = m_hom ≤ t50[bộ]."""
    sel = np.isin(G.book_set.values, list(sets))
    idx = np.nonzero(sel)[0]
    Rl = [R_of(s) for s in G.syllable.values[idx]]
    need = set(G.label.values[idx])
    for R in Rl:
        need |= R
    for i in idx:
        need |= set(refs.get(G.cell_uid.iat[i], []))
    need |= {c for c in G.ocr_char.values[idx] if len(c) == 1}
    FE, _, _, rmd5 = S.glyph_emb(sorted(need), with_aug=False, with_fd=False)
    m = np.full(len(G), np.nan)
    for j, i in enumerate(idx):
        L = G.label.iat[i]
        C0 = (Rl[j] | set(refs.get(G.cell_uid.iat[i], []))) - {L}
        mL = rmd5.get(L)
        same = {c for c in C0 if mL is not None and rmd5.get(c) == mL}
        cf = [c for c in C0 - same if c in FE]
        if cf and L in FE:
            e = f16(Eenc[i])                         # h02 đọc lại E_crops.f16.npy (t_emb = 0)
            sL = float(e @ FE[L])
            sc = np.stack([FE[c] for c in cf]) @ e
            m[i] = sL - float(sc.max())
    bs = G.book_set.values
    t = np.array([t50.get(b, np.nan) for b in bs])
    flag = ~np.isnan(m) & (m <= t)
    log(f"m_hom xong ({int(sel.sum())} ô, cờ t50 {int(flag.sum())})")
    return dict(m_hom=m, mocr_cons=flag)


# ================================================================================================ viss (rerank r02/r02b/r03)
VISS_FEATS = ["s_font", "s_faug", "s_fontB", "s_faugB", "s_oth0", "has_oth", "s_head0", "has_head", "has_font",
              "d_faug", "d_faugB", "d_oth", "r1_faug", "lp_oth", "s_self0", "has_self", "d_self", "lp_self"]
VISS_DIR = {"LucVanTien1916": ("viss_T", "TruyenKieu1872"), "TruyenKieu1872": ("viss_L", "LucVanTien1916"),
            "LucVanTien1883": ("viss_X", "IHR2"), "KimVanKieu1884": ("viss_X", "IHR2")}
OTH_SETS = ("SachThanhTruyen", "LucVanTien1883", "KimVanKieu1884", "Chrestomathie1872")


def _second_max(v, g):
    """theo nhóm g: (max, max thứ hai theo np.sort(x)[-2] hoặc −1 nếu nhóm 1 phần tử) — bản vector hoá của r03 build."""
    o = np.lexsort((v, g))
    gs, vs = g[o], v[o]
    last = np.r_[gs[1:] != gs[:-1], True]
    first = np.r_[True, gs[1:] != gs[:-1]]
    gid = np.cumsum(first) - 1
    mx = vs[last]
    size = np.bincount(gid)
    pos_last = np.nonzero(last)[0]
    s2 = np.where(size > 1, vs[np.maximum(pos_last - 1, 0)], -1.0)
    inv = np.empty_like(o); inv[o] = np.arange(len(o))
    return mx[gid][inv], s2[gid][inv]


def _rel(v_raw, g):
    v = np.where(np.isnan(v_raw), -1.0, v_raw)
    mx, s2 = _second_max(v, g)
    dd = v - np.where(v >= mx, s2, mx)
    dd[np.isnan(v_raw)] = 0.0
    return np.clip(dd, -0.5, 0.5)


def viss_features(c: pd.DataFrame) -> pd.DataFrame:
    """== r03_model.build (bỏ phần sự thật/học)."""
    c = c.copy()
    g = c.i.values
    c["has_font"] = c.s_font.notna().astype(float)
    for k in ("font", "faug", "fontB", "faugB"):
        col = f"s_{k}"
        c[col] = c[col].fillna(c.groupby("i")[col].transform("min")).fillna(0.0)
    c["has_oth"] = c.s_oth.notna().astype(float); c["s_oth0"] = c.s_oth.fillna(0.0)
    c["has_head"] = c.s_head.notna().astype(float); c["s_head0"] = c.s_head.fillna(0.0)
    c["lp_oth"] = np.log1p(c.n_oth.fillna(0).astype(float))
    for k, col in (("faug", "s_faug"), ("faugB", "s_faugB"), ("oth", "s_oth")):
        c[f"d_{k}"] = _rel(c[col].values.astype(float), g)
    c["has_self"] = c.s_self.notna().astype(float); c["s_self0"] = c.s_self.fillna(0.0)
    c["lp_self"] = np.log1p(c.n_self.astype(float))
    c["d_self"] = _rel(c.s_self.values.astype(float), g)
    mxf, _ = _second_max(c.s_faug.values.astype(float), g)
    c["r1_faug"] = (c.s_faug.values >= mxf).astype(float)
    return c


def seg_logsoftmax(logit, seg, nseg):
    mx = torch.full((nseg,), -1e9).scatter_reduce(0, seg, logit, "amax")
    e = torch.exp(logit - mx[seg])
    z = torch.zeros(nseg).index_add(0, seg, e)
    return logit - mx[seg] - torch.log(z[seg])


def viss_predict(prm, c):
    """== r03_model.predict (torch float32, softmax trong từng ô)."""
    X = (torch.tensor(c[prm["feats"]].values, dtype=torch.float32) - torch.as_tensor(prm["mu"])) / torch.as_tensor(prm["sd"])
    cells, seg = np.unique(c.i.values, return_inverse=True)
    lp = seg_logsoftmax(X @ torch.as_tensor(prm["w"]), torch.tensor(seg), len(cells))
    return np.exp(lp.detach().numpy())


def viss_pkim(c, p, kim_md5):
    """== r03_model.summarize: p_kim = tổng p các ứng viên cùng HÌNH glyph với kim (hoặc chính kim)."""
    same = np.array([(gm != "" and kim_md5.get(i, "") == gm) for i, gm in zip(c.i.values, c.gmd5.values)]) | (c.is_kim.values == 1)
    return pd.Series(np.where(same, p, 0.0)).groupby(c.i.values).sum()


def score_viss(S: Scorers, G, Eenc, EencB, params, log):
    """G: mọi ô GOLD (cần cả bộ 'oth'); EencB: dict idx -> nhúng view B (cắt bbox trang đã xử lý) cho 4 sách."""
    enc = S.enc()
    books4 = list(VISS_DIR)
    EA = f16(Eenc)                                   # r01/r02 đọc E_A.f16.npy, m_ocr E_crops.f16.npy
    # ---- nguyên mẫu 'oth' (GOLD các bộ không IHR), theo book_set
    osum, on = {}, {}
    for bs in OTH_SETS:
        m = np.nonzero(G.book_set.values == bs)[0]
        grp = pd.DataFrame({"l": G.label.values[m]}).groupby("l").indices
        osum[bs] = {l: EA[m[ix]].sum(0) for l, ix in grp.items()}
        on[bs] = {l: len(ix) for l, ix in grp.items()}
    pcache = {}

    def P_oth(book, ch):
        key = (book, ch)
        if key not in pcache:
            v, n = 0, 0
            for s in OTH_SETS:
                if s == book:
                    continue
                if ch in on[s]:
                    v = v + osum[s][ch]; n += on[s][ch]
            pcache[key] = (None, n) if n < 3 else (v / (np.linalg.norm(v) + 1e-9), n)
        return pcache[key]
    # ---- nguyên mẫu 'self' (GOLD cùng sách, rời trang)
    Sd, Td = {}, {}
    for i in np.nonzero(np.isin(G.book_set.values, books4) & (G.label.str.len().values == 1))[0]:
        k = (G.book_set.iat[i], G.page.iat[i], G.label.iat[i])
        s, n = Sd.get(k, (0, 0)); Sd[k] = (s + EA[i], n + 1)
    for (b, p, ch), (s, n) in Sd.items():
        s0, n0 = Td.get((b, ch), (0, 0)); Td[(b, ch)] = (s0 + s, n0 + n)
    # ---- ứng viên
    idx = np.nonzero(np.isin(G.book_set.values, books4))[0]
    cands = []
    need = set()
    for i in idx:
        c = set(R_of(G.syllable.iat[i])); k = G.ocr_char.iat[i]
        if len(k) == 1:
            c.add(k)
        cands.append(sorted(c)); need |= c
    FE, FA, _, gmd5 = S.glyph_emb(sorted(need), with_aug=True, with_fd=False)
    blank = md5_bytes(np.full((112, 112), 255, np.uint8).tobytes())
    W, lab2idx = enc.Wn, enc.lab2idx
    rows = []
    selfc = {}
    for j, i in enumerate(idx):
        b, pgn, kim = G.book_set.iat[i], G.page.iat[i], G.ocr_char.iat[i]
        R = set(R_of(G.syllable.iat[i]))
        eA = EA[i]; eB = f16(EencB[i])
        for x in cands[j]:
            fe, fa = FE.get(x), FA.get(x)
            po, npo = P_oth(b, x)
            key = (b, pgn, x)
            if key not in selfc:
                s, n = Td.get((b, x), (0, 0)); sp, npg = Sd.get(key, (0, 0))
                s, n = s - sp, n - npg
                selfc[key] = ((s / (np.linalg.norm(s) + 1e-9)), n) if n >= 3 else (None, n)
            vs, ns_ = selfc[key]
            gm = gmd5.get(x, "")
            rows.append((i, x, int(x in R), int(x == kim),
                         float(eA @ fe) if fe is not None else np.nan, float(eA @ fa) if fa is not None else np.nan,
                         float(eB @ fe) if fe is not None else np.nan, float(eB @ fa) if fa is not None else np.nan,
                         float(eA @ po) if po is not None else np.nan, npo,
                         float(eA @ W[lab2idx[x]]) if x in lab2idx else np.nan,
                         "BLANK" if gm == blank else gm,
                         float(eA @ vs) if vs is not None else np.nan, ns_))
    c = pd.DataFrame(rows, columns=["i", "ch", "inR", "is_kim", "s_font", "s_faug", "s_fontB", "s_faugB", "s_oth", "n_oth",
                                    "s_head", "gmd5", "s_self", "n_self"])
    log(f"viss: {len(c)} hàng ứng viên / {c.i.nunique()} ô")
    c = viss_features(c)
    c["book"] = G.book_set.values[c.i.values]
    out = {k: np.full(len(G), np.nan) for k in ("viss_T", "viss_L", "viss_X")}
    kim_md5 = c[c.is_kim == 1].set_index("i").gmd5.to_dict()
    for book, (col, tune) in VISS_DIR.items():
        cc = c[c.book == book]
        if not len(cc):
            continue
        p = viss_predict(params[tune], cc)
        pk = viss_pkim(cc, p, kim_md5)
        out[col][pk.index.values] = pk.values
    log("viss xong")
    return out
