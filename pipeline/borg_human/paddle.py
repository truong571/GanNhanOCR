"""paddle.py — làm sạch vòng 5 bằng bộ đọc ĐỘC LẬP Paddle PP-OCRv6 (không học trên STT/Borg, không encoder),
trên HỒ SƠ THỜI GIAN ĐÓNG BĂNG measure_out/_borg_human_src/prof_borgx.pkl (không chạy lại Paddle: 0 API, không tải gì).

Bản chép NGUYÊN hành vi r5/borg_quality: qlib.Prof/est/summarize/boot_theta, cửa sổ mid+delay (q06/q07/q14):
  cửa sổ y của đơn vị = [trung điểm tâm với hộp kề trên, trung điểm với hộp kề dưới] ∩ hộp, dời +delay·h (delay đo trên IHR)
  s(u, c) = max_{t trong cửa sổ} P_Paddle[t, c]
  keep_paddle_ok (q07) = keep ∧ ¬(thử được cửa sổ 7 chữ khác nhau ∧ argmax_d s(u, c_{j+d}) ≠ 0)
  chuẩn hoá người phiên (q14) = chữ người N ∈ 11 cặp thăm dò ∧ s(u, L) >= 0,5 (Paddle thấy dạng Hán L)
  keep_v5 = keep_paddle_ok ∧ ¬chuẩn hoá
  θ trượt ±1 (q06 'D|borg_keep|mid+delay') = (P(±1) − 2·far)/(a − far), far = TB P(±2, ±3), a = 1 − 6·far; CI bootstrap cụm trang
"""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd

from .params import PADDLE, PROBE_PAIRS, SRC

D = PADDLE["D"]
OFF = list(range(-D, D + 1))


def sha256(p, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def est(P):
    P = np.asarray(P)
    n = len(P)
    cnt = {d: float((P == d).sum()) / n for d in OFF}
    far = np.mean([cnt[-2], cnt[2], cnt[-3], cnt[3]])
    a = 1 - 6 * far
    return (cnt[-1] + cnt[1] - 2 * far) / max(a - far, 1e-6), cnt


def boot_theta(offs, grp, B=300, seed=0):
    rng = np.random.default_rng(seed)
    ks = pd.Series(np.arange(len(grp))).groupby(grp).apply(lambda s: s.to_numpy()).tolist()
    v = [est(offs[np.concatenate([ks[i] for i in rng.integers(0, len(ks), len(ks))])])[0] for _ in range(B)]
    return [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]


def summarize(offs, grp, B=300, seed=0):
    th, cnt = est(offs)
    return dict(n=int(len(offs)), theta=round(float(th), 4), theta_ci=boot_theta(offs, grp, B, seed) if len(offs) > 50 else None,
                p0=round(cnt[0], 4), m1=round(cnt[-1], 4), p1=round(cnt[1], 4),
                far=round(float(np.mean([cnt[d] for d in (-3, -2, 2, 3)])), 4))


class Prof:
    def __init__(self, path=None, check_sha=True):
        path = path or (SRC / PADDLE["profile"])
        if check_sha:
            got = sha256(path)
            if got != PADDLE["sha256"]:
                raise RuntimeError(f"hồ sơ Paddle {path}: sha256 {got[:12]} ≠ đóng băng {PADDLE['sha256'][:12]}")
        o = pickle.load(open(path, "rb"))
        self.pagechars = o["pagechars"]
        self.segs = o["segs"]
        self.CI = {k: {c: i for i, c in enumerate(v)} for k, v in self.pagechars.items()}
        self.U = {}
        for si, s in enumerate(self.segs):
            for uid, y1, y2, v in s["units"]:
                self.U[uid] = (si, y1, y2)

    def scores(self, uid, chars, lo_hi):
        si, y1, y2 = self.U[uid]
        s = self.segs[si]
        a, b = lo_hi
        m = (s["ty"] >= a) & (s["ty"] <= b)
        if not m.any():
            return None
        ci = self.CI[s["key"]]
        M = s["M"]
        out = []
        for c in chars:
            j = ci.get(c)
            if j is None:
                out.append(np.nan)
                continue
            v = M[m, j].astype(np.float32)
            out.append(np.nan if np.isnan(v).all() else float(np.nanmax(v)))
        return out

    def mid_windows(self, shift):
        Wd = {}
        for s in self.segs:
            us = sorted(s["units"], key=lambda u: (u[1] + u[2]) / 2)
            cs = [(u[1] + u[2]) / 2 for u in us]
            for i, u in enumerate(us):
                h = u[2] - u[1]
                lo = (cs[i - 1] + cs[i]) / 2 if i > 0 else u[1]
                hi = (cs[i] + cs[i + 1]) / 2 if i + 1 < len(us) else u[2]
                lo, hi = max(lo, u[1]), min(hi, u[2])
                Wd[u[0]] = (lo + shift * h, hi + shift * h)
        return Wd


def units_match(prof: Prof, units: list) -> dict:
    """Hồ sơ gắn với đơn vị theo uid (chỉ số trong units_v1.pkl của r4): kiểm TRÙNG (uid, y1, y2, ảo) từng đơn vị."""
    n = bad = 0
    bad_keys = set()
    for s in prof.segs:
        for uid, y1, y2, v in s["units"]:
            n += 1
            u = units[uid] if 0 <= uid < len(units) else None
            if u is None or u["y1"] != y1 or u["y2"] != y2 or int(u["virtual"]) != int(v) or \
                    f"{u['book']}/{u['page']}" != s["key"]:
                bad += 1
                bad_keys.add(s["key"])
    return dict(n_units_profile=n, n_units_now=len(units), mismatch=bad, pages_mismatch=sorted(bad_keys))


def window7(seq, pos):
    if pos - D < 0 or pos + D >= len(seq):
        return None
    w = seq[pos - D:pos + D + 1]
    if not all(isinstance(x, str) and x for x in w) or len(set(w)) != 2 * D + 1:
        return None
    return w


def apply(A: pd.DataFrame, prof: Prof, log=print) -> tuple[pd.DataFrame, dict]:
    """A = mọi ô người (book, page, idx, char, kind, unit, keep). Thêm cột paddle_test ('', 'ok', 'lech'),
    paddle_han (dạng Hán L Paddle thấy, cho 11 cặp thăm dò), keep_paddle_ok, keep_v5; trả (A, thống kê)."""
    A = A.copy()
    A["key"] = A.book + "/" + A.page
    A = A.sort_values(["key", "idx"]).reset_index(drop=True)
    A["pos"] = A.groupby("key").cumcount()
    seq = {k: g.char.fillna("").tolist() for k, g in A.groupby("key")}
    Wd = prof.mid_windows(PADDLE["delay_h"])
    real = (A.kind == "real") & (A.unit >= 0)
    test = np.array([""] * len(A), dtype=object)
    offs, grp = [], []
    for i, (key, pos, uid, r) in enumerate(zip(A.key, A.pos, A.unit, real)):
        if not r or uid not in Wd:
            continue
        w = window7(seq[key], pos)
        if w is None:
            continue
        sc = prof.scores(uid, w, lo_hi=Wd[uid])
        if sc is None or np.isnan(sc).any():
            continue
        o = int(np.argmax(sc)) - D
        test[i] = "ok" if o == 0 else f"lech{o:+d}"
    A["paddle_test"] = test
    kp = A.keep.to_numpy() == 1
    flag = np.array([t.startswith("lech") for t in test])
    A["keep_paddle_ok"] = (kp & ~flag).astype(int)
    han = np.array([""] * len(A), dtype=object)
    for i, (ch, uid, key, r) in enumerate(zip(A.char, A.unit, A.key, real)):
        L = PROBE_PAIRS.get(ch)
        if not L or not r or uid not in Wd or L not in prof.CI.get(key, {}):
            continue
        sc = prof.scores(uid, [L], lo_hi=Wd[uid])
        if sc is not None and not np.isnan(sc[0]) and sc[0] >= PADDLE["tau_norm"]:
            han[i] = L
    A["paddle_han"] = han
    norm = (A.keep_paddle_ok.to_numpy() == 1) & (han != "")
    A["keep_v5"] = (A.keep_paddle_ok.to_numpy() == 1) & ~norm
    A["keep_v5"] = A.keep_v5.astype(int)
    # θ trượt ±1 (q06): trên keep, cửa sổ mid+delay, thứ tự (key, idx), bootstrap cụm trang seed cố định
    th = {}
    kh = A.keep_high.to_numpy() == 1
    for name, m in (("keep", kp), ("keep_v5", A.keep_v5.to_numpy() == 1), ("keep_high", kh),
                    ("keep_high_tru_keep", kh & ~kp), ("khong_real", real.to_numpy() & ~kh)):
        o, g = [], []
        for key, t in zip(A.key[m], test[m]):
            if t:
                o.append(0 if t == "ok" else int(t[4:]))
                g.append(key)
        th[name] = summarize(np.array(o), np.array(g), B=PADDLE["boot_B"], seed=PADDLE["boot_seed"])
    st = dict(keep=int(kp.sum()), keep_paddle_flagged=int((kp & flag).sum()), keep_paddle_ok=int(A.keep_paddle_ok.sum()),
              removed_normalised=int(norm.sum()), keep_v5=int(A.keep_v5.sum()),
              removed_by_char={k: int(v) for k, v in A[norm].char.value_counts().items()},
              n_real_tested=int((test != "").sum()), theta=th,
              paddle_han_all={f"{n}~{PROBE_PAIRS[n]}": int(((A.char == n) & (han != "")).sum()) for n in PROBE_PAIRS})
    log(f"  paddle: keep {st['keep']} − lệch {st['keep_paddle_flagged']} − chuẩn hoá {st['removed_normalised']} = keep_v5 {st['keep_v5']}; "
        f"θ(keep) {th['keep']['theta']} {th['keep']['theta_ci']} (n={th['keep']['n']})")
    return A.drop(columns=["key", "pos"]), st
