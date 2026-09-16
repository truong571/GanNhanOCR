"""THỰC NGHIỆM RÀ SOÁT GÁN NHÃN NÔM ↔ QUỐC NGỮ — 2026-09-13
Tái sinh mọi con số của docs/RA_SOAT_GAN_NHAN_2026-09-13.md.

    PY=.venv/bin/python
    $PY lab/gan_nhan_2026-09-13/thuc_nghiem.py rebuild     # dựng chuỗi (chữ OCR, âm, DP) từ prepared/ -> cols.pkl
    $PY lab/gan_nhan_2026-09-13/thuc_nghiem.py posterior   # xác suất hậu nghiệm thanh ghi + benchmark có đáp án
    $PY lab/gan_nhan_2026-09-13/thuc_nghiem.py rules       # từng luật từ điển: độ phủ × lift so với null lệch-1
    $PY lab/gan_nhan_2026-09-13/thuc_nghiem.py calib       # ma trận chi phí production vs hiệu chuẩn theo likelihood
    $PY lab/gan_nhan_2026-09-13/thuc_nghiem.py recursive   # đệ quy: neo ngữ liệu từ trang khác nạp lại DP
    $PY lab/gan_nhan_2026-09-13/thuc_nghiem.py v3          # mô phỏng phân hạng v3 trên dữ liệu thật + benchmark
    $PY lab/gan_nhan_2026-09-13/thuc_nghiem.py classes     # bảng lớp (sách, âm, nhãn thiểu số) để phân xử
    $PY lab/gan_nhan_2026-09-13/thuc_nghiem.py geo         # (cần detector) hộp ảnh trong cột rụng chữ
    $PY lab/gan_nhan_2026-09-13/thuc_nghiem.py all         # tất cả trừ geo

Chỉ đọc repo; đầu ra duy nhất là cols.pkl (cache) cạnh tệp này. Tất định (seed cố định).
"""
from __future__ import annotations

import glob
import io
import json
import math
import pickle
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from core.align.run_full import nom_cols_hybrid                                  # noqa: E402
from core.text.dictionary import load_qn_to_nom, load_similarity_dict             # noqa: E402
from core.text.text_utils import (is_plausible_qn_syllable, normalize_tone_marks,  # noqa: E402
                                  strip_all, strip_tone)
from pipeline.align_engine import anchor_align as aa                              # noqa: E402
from pipeline.align_engine.anchor_align import is_confirmed, realign_column, substitution_cost  # noqa: E402
from pipeline.align_engine.syllable_normalize import build_readings, normalize_column  # noqa: E402
from pipeline.step2_align import _get_qn_lines                                    # noqa: E402

BOOKS = {"SachThanhTruyen2": "stt2", "SachThanhTruyen4": "stt4", "SachThanhTruyen11": "stt11"}
CACHE = HERE / "cols.pkl"

qn = load_qn_to_nom(str(REPO / "dict/QuocNgu_SinoNom.csv"))
sim = load_similarity_dict(str(REPO / "dict/SinoNom_Similar.csv"))
qs = set(qn)
Rset = {s: set(v) for s, v in qn.items()}

PROD = dict(COST_CONFIRM=0.0, COST_SIMILAR=0.3, COST_DICTMISS=1.0, COST_NODICT=0.9, COST_DEL=0.7, COST_INS=0.7)
# −log tỉ số khả năng đo trên ngữ liệu (mục §3 báo cáo): confirmed 0 · similar 2,5 · nodict 5,1 · miss 6,7 · khe 8,6
CALIB = dict(COST_CONFIRM=0.0, COST_SIMILAR=2.5, COST_DICTMISS=6.7, COST_NODICT=5.1, COST_DEL=8.6, COST_INS=8.6)
# Ma trận ĐANG ở engine (từ A-4 = CALIB): rebuild dựng cols.pkl bằng ma trận này và mọi
# cmd reset về nó — PROD chỉ còn là mốc so sánh lịch sử, không phải mặc định.
ENGINE = {k: getattr(aa, k) for k in PROD}
# Trần chi phí neo ngữ liệu (flow N4b) — đọc từ engine, 2,0 nat thay vì COST_CONFIRM=0
ANCHOR_CAP = getattr(aa, "ANCHOR_CAP", 2.0)


def set_matrix(M):
    for k, v in M.items():
        setattr(aa, k, v)


def h(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# =========================================================================== rebuild
def cmd_rebuild():
    import pandas as pd
    readings = build_readings(qn)
    df = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
    df["column"] = df["column"].astype(int)
    df["row"] = range(len(df))
    lab = {k: g for k, g in df.groupby(["book", "page", "column"], sort=False)}
    cols, bad = [], 0
    set_matrix(ENGINE)
    for book, code in BOOKS.items():
        dd = REPO / "prepared" / book
        for tf in sorted(glob.glob(str(dd / "transcriptions" / "page_*.json"))):
            if tf.endswith("_qn_ocr_cache.json"):
                continue
            page = Path(tf).stem
            ocr_path = dd / "detected" / f"{page}_ocr_cache.json"
            if not ocr_path.exists():
                continue
            od = json.load(open(ocr_path, encoding="utf-8"))
            lines, _ = _get_qn_lines(dd, page, qs)
            keys = sorted(lines)
            cs = nom_cols_hybrid(od["columns"], min_len=4)
            if len(cs) != 9:
                cs = nom_cols_hybrid(od["columns"], min_len=1)
            for i in range(min(len(cs), len(keys))):
                syl = lines[keys[i]]
                if not syl:
                    continue
                syl, _ = normalize_column(cs[i]["chars"], syl, qs, readings)
                chars = [c["char"] for c in cs[i]["chars"]]
                ops = realign_column(cs[i]["chars"], syl, qn, sim)
                mp = [o for o in ops if o["op"] == "match"]
                g = lab.get((code, page, keys[i]))
                rec = dict(book=code, page=page, column=keys[i], chars=chars, syl=syl, ops=ops,
                           tiers=None, rules=None, rows=None, matrix=dict(ENGINE))
                if g is not None and len(g) == len(mp):
                    rec.update(tiers=list(g.tier), rules=list(g.rule), rows=list(g.row))
                else:
                    bad += 1
                cols.append(rec)
    pickle.dump(cols, open(CACHE, "wb"))
    print(f"cột: {len(cols)} · không join được với labels_final: {bad} · cache -> {CACHE}")


def load_cols():
    if not CACHE.exists():
        cmd_rebuild()
    return pickle.load(open(CACHE, "rb"))


# =========================================================================== posterior
def _lse(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def posterior_matches(chars, syls, T=0.35):
    """{(i,j): P(ghép i↔j)} bằng forward–backward trên lưới NW có băng, trọng số exp(−cost/T)."""
    m, n = len(chars), len(syls)
    if m == 0 or n == 0:
        return {}
    band = abs(m - n) + max(1, aa.BAND_SLACK)
    NEG = -math.inf
    cache = {}

    def sc(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = -substitution_cost(chars[i], syls[j], qn, sim) / T
        return cache[(i, j)]
    d, ins = -aa.COST_DEL / T, -aa.COST_INS / T
    F = [[NEG] * (n + 1) for _ in range(m + 1)]
    F[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            if abs(i - j) > band or (i == 0 and j == 0):
                continue
            v = NEG
            if i > 0 and j > 0:
                v = _lse(v, F[i - 1][j - 1] + sc(i - 1, j - 1))
            if i > 0:
                v = _lse(v, F[i - 1][j] + d)
            if j > 0:
                v = _lse(v, F[i][j - 1] + ins)
            F[i][j] = v
    B = [[NEG] * (n + 1) for _ in range(m + 1)]
    B[m][n] = 0.0
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if abs(i - j) > band or (i == m and j == n):
                continue
            v = NEG
            if i < m and j < n:
                v = _lse(v, B[i + 1][j + 1] + sc(i, j))
            if i < m:
                v = _lse(v, B[i + 1][j] + d)
            if j < n:
                v = _lse(v, B[i][j + 1] + ins)
            B[i][j] = v
    Z = F[m][n]
    return {(i, j): math.exp(F[i][j] + sc(i, j) + B[i + 1][j + 1] - Z)
            for i in range(m) for j in range(n)
            if abs(i - j) <= band and F[i][j] > NEG and B[i + 1][j + 1] > NEG}


def viterbi_pairs(chars, syls):
    return [(o["nom_idx"], o["syl_idx"]) for o in realign_column(chars, syls, qn, sim) if o["op"] == "match"]


def perturb(chars, syls, kind, rng, all_chars):
    """(chars2, syls2, truth: chỉ số OCR -> chỉ số âm đúng | None)."""
    m = len(chars)
    k = rng.randrange(2, m - 2)
    if kind == "drop_char":
        c2 = chars[:k] + chars[k + 1:]
        return c2, syls, {i: (i if i < k else i + 1) for i in range(len(c2))}
    if kind == "drop2_char":
        c2 = chars[:k] + chars[k + 2:]
        return c2, syls, {i: (i if i < k else i + 2) for i in range(len(c2))}
    if kind == "split_char":
        c2 = chars[:k + 1] + [rng.choice(["一", "丨", "口", "十"])] + chars[k + 1:]
        return c2, syls, {i: (i if i <= k else (None if i == k + 1 else i - 1)) for i in range(len(c2))}
    if kind == "subst_char":
        c2 = list(chars)
        c2[k] = rng.choice(all_chars)
        return c2, syls, {i: i for i in range(len(c2))}
    if kind == "drop_syl":
        s2 = syls[:k] + syls[k + 1:]
        return chars, s2, {i: (i if i < k else (None if i == k else i - 1)) for i in range(m)}
    if kind == "none":
        return chars, syls, {i: i for i in range(m)}
    raise ValueError(kind)


def local_offset(chars, syls, i, j, w=2):
    def conf(delta):
        return sum(1 for k in range(-w, w + 1)
                   if 0 <= i + k < len(chars) and 0 <= j + k + delta < len(syls) and chars[i + k]
                   and chars[i + k] in Rset.get(syls[j + k + delta].lower(), ()))
    c0 = conf(0)
    return c0, max(conf(dl) for dl in (-2, -1, 1, 2))


def cmd_posterior():
    cols = load_cols()
    set_matrix(PROD)
    all_chars = [c for col in cols for c in col["chars"] if c]
    eq = [c for c in cols if len(c["chars"]) == len(c["syl"]) and len(c["chars"]) >= 10]
    rng = random.Random(2026)
    sample = rng.sample(eq, 500)
    h("A. BENCHMARK CÓ ĐÁP ÁN — 500 cột khớp số, mỗi cột gây 1 hỏng")
    for kind in ["drop_char", "drop2_char", "split_char", "drop_syl", "subst_char"]:
        n = wrong = wrong_conf = 0
        pw, pr = [], []
        off_w = off_r = 0
        for col in sample:
            c2, s2, truth = perturb(col["chars"], col["syl"], kind, rng, all_chars)
            post = posterior_matches(c2, s2)
            for (i, j) in viterbi_pairs(c2, s2):
                n += 1
                ok = truth.get(i) == j
                p = post.get((i, j), 0.0)
                c0, best = local_offset(c2, s2, i, j)
                if ok:
                    pr.append(p); off_r += best - c0 >= 2
                else:
                    wrong += 1; pw.append(p); off_w += best - c0 >= 2
                    wrong_conf += bool(c2[i] and c2[i] in Rset.get(s2[j].lower(), ()))
        pw, pr = np.array(pw), np.array(pr)
        rs = rng.sample(list(pr), min(3000, len(pr)))
        auc = np.mean([1.0 if a < b else (0.5 if a == b else 0.0) for a in pw for b in rs])
        print(f"\n{kind:11s} cặp {n} · ghép SAI {wrong} ({wrong/n:.1%}) · sai mà từ điển vẫn xác nhận {wrong_conf}")
        print(f"   posterior: đúng p50={np.median(pr):.3f} · sai p50={np.median(pw):.3f} · AUC bắt sai={auc:.3f}")
        for th in (0.5, 0.8, 0.9):
            print(f"   p<{th}: bắt {np.mean(pw < th):.0%} cặp sai · rớt oan {np.mean(pr < th):.1%} cặp đúng")
        print(f"   kiểm lệch cục bộ (w=2, margin≥2): bắt {off_w/max(1,wrong):.0%} sai · oan {off_r/max(1,len(pr)):.1%}")

    h("B. DỮ LIỆU THẬT — posterior thanh ghi theo tier hiện hành")
    by = defaultdict(list)
    for col in cols:
        if col["tiers"] is None:
            continue
        post = posterior_matches(col["chars"], col["syl"])
        for (i, j), t, r in zip(viterbi_pairs(col["chars"], col["syl"]), col["tiers"], col["rules"]):
            key = t if t != "GOLD" else ("GOLD/" + ("direct" if r == "s1_inter_s2_direct" else
                                                     ("quyet_dinh" if r.startswith("quyet") else "bridge/L1/L3")))
            by[key].append(post.get((i, j), 0.0))
    for k in sorted(by):
        v = np.array(by[k])
        print(f"{k:22s} n={len(v):6d}  p50={np.median(v):.3f}  p<0.5: {np.mean(v<0.5):5.1%}  p<0.9: {np.mean(v<0.9):5.1%}")

    h("C. CỘT KHỚP SỐ (m=n) — cặp ghép LỆCH đường chéo do DP")
    off = Counter(); tot = 0; ncol = 0; cols_gap = 0
    for col in cols:
        if len(col["chars"]) != len(col["syl"]) or col["tiers"] is None:
            continue
        ncol += 1
        mp = [o for o in col["ops"] if o["op"] == "match"]
        k = 0
        for o, t, r in zip(mp, col["tiers"], col["rules"]):
            tot += 1
            if o["nom_idx"] != o["syl_idx"]:
                off[(t, r)] += 1; k += 1
        cols_gap += k > 0
    print(f"cột m=n: {ncol} · cặp {tot} · lệch chéo {sum(off.values())} ({sum(off.values())/tot:.2%}) · cột có khe {cols_gap}")
    for k, v in off.most_common(8):
        print("  ", k, v)


# =========================================================================== rules
def _external():
    """Hán-Việt (han_raw.txt) + Unihan kVietnamese từ cache của bộ kiểm độc lập (nếu có)."""
    C = REPO / "re-dataset/check/_cache"
    HV, KV = {}, defaultdict(set)
    if (C / "han_raw.txt").exists():
        code = 0x4E00
        for line in open(C / "han_raw.txt", encoding="utf-8"):
            line = line.strip()
            if line:
                HV[chr(code)] = {normalize_tone_marks(x.strip().lower()) for x in line.split(",") if x.strip()}
            code += 1
    if (C / "Unihan.zip").exists():
        with zipfile.ZipFile(C / "Unihan.zip") as z:
            for name in z.namelist():
                if "Readings" in name:
                    for ln in io.TextIOWrapper(z.open(name), encoding="utf-8"):
                        if "\tkVietnamese\t" in ln:
                            cp, _, vals = ln.rstrip("\n").split("\t")
                            for v in vals.split():
                                KV[chr(int(cp[2:], 16))].add(normalize_tone_marks(v.lower()))
    return HV, KV


class Corpus:
    """Thống kê ngữ liệu theo TRANG để loại-một-trang khi đánh giá (không tự khẳng định)."""

    def __init__(self, cols):
        self.pair_pages = defaultdict(set)
        self.bigram_pages = defaultdict(set)
        for col in cols:
            pg = (col["book"], col["page"])
            mp = [o for o in col["ops"] if o["op"] == "match"]
            for o in mp:
                if o["ocr_char"]:
                    self.pair_pages[(o["ocr_char"], o["syllable"].lower())].add(pg)
            for a, b in zip(mp, mp[1:]):
                if a["nom_idx"] + 1 == b["nom_idx"] and a["syl_idx"] + 1 == b["syl_idx"]:
                    self.bigram_pages[(a["ocr_char"], b["ocr_char"], a["syllable"].lower(), b["syllable"].lower())].add(pg)
        self.by_tone, self.by_all = defaultdict(set), defaultdict(set)
        for s in qn:
            self.by_tone[strip_tone(s)].add(s)
            self.by_all[strip_all(s)].add(s)
        self.HV, self.KV = _external()

    def bigrams(self, chars, syls, i, j):
        c, s = chars[i], syls[j].lower()
        out = []
        if i > 0 and j > 0:
            out.append((chars[i - 1], c, syls[j - 1].lower(), s))
        if i + 1 < len(chars) and j + 1 < len(syls):
            out.append((c, chars[i + 1], s, syls[j + 1].lower()))
        return out

    def feats(self, chars, syls, i, j, pg):
        c, s = chars[i], syls[j].lower()
        R = Rset.get(s, set())
        bg = self.bigrams(chars, syls, i, j)
        other = self.pair_pages.get((c, s), set()) - {pg}
        simc = set(sim.get(c, []))
        mutual = {b for b in simc & R if c in sim.get(b, ())}
        return dict(
            direct=c in R,
            sim_unique=len(simc & R) == 1,
            sim_any=bool(simc & R),
            sim_rev_unique=sum(1 for r in R if r != c and c in sim.get(r, ())) == 1,
            sim_mutual_unique=len(mutual) == 1,
            hop2_unique=len(({x for a in simc for x in sim.get(a, [])} | simc) - {c} & R) == 1,
            tone=any(c in Rset.get(v, ()) for v in self.by_tone.get(strip_tone(s), set()) - {s}),
            dia=any(c in Rset.get(v, ()) for v in self.by_all.get(strip_all(s), set()) - {s}),
            hanviet=s in self.HV.get(c, ()),
            hanviet_tone=strip_tone(s) in {strip_tone(x) for x in self.HV.get(c, ())},
            unihan=s in self.KV.get(c, ()),
            corpus2=len(other) >= 2, corpus4=len(other) >= 4,
            bigram=any(self.bigram_pages.get(k, set()) - {pg} for k in bg),
            bigram2=any(len(self.bigram_pages.get(k, set()) - {pg}) >= 2 for k in bg),
            bigram_both=len(bg) == 2 and all(self.bigram_pages.get(k, set()) - {pg} for k in bg),
        )


RULES = [
    ("direct  c∈R(s)", lambda f: f["direct"]),
    ("sim_unique (GOLD cầu xuôi hiện hành)", lambda f: f["sim_unique"]),
    ("sim_any", lambda f: f["sim_any"]),
    ("sim_rev_unique (L2)", lambda f: f["sim_rev_unique"]),
    ("sim_mutual_unique", lambda f: f["sim_mutual_unique"]),
    ("hop2_unique", lambda f: f["hop2_unique"]),
    ("tone_direct c∈R(s~thanh)", lambda f: f["tone"]),
    ("dia_direct c∈R(s~dấu)", lambda f: f["dia"]),
    ("hanviet HV(c)=s", lambda f: f["hanviet"]),
    ("hanviet~thanh", lambda f: f["hanviet_tone"]),
    ("unihan kVietnamese", lambda f: f["unihan"]),
    ("corpus_pair ≥2 trang khác", lambda f: f["corpus2"]),
    ("corpus_pair ≥4 trang khác", lambda f: f["corpus4"]),
    ("bigram ≥1 trang khác", lambda f: f["bigram"]),
    ("bigram ≥2 trang khác", lambda f: f["bigram2"]),
    ("bigram CẢ HAI phía", lambda f: f["bigram_both"]),
    ("sim_unique & bigram", lambda f: f["sim_unique"] and f["bigram"]),
    ("sim_unique & corpus≥2", lambda f: f["sim_unique"] and f["corpus2"]),
    ("sim_unique & tone/dia", lambda f: f["sim_unique"] and (f["tone"] or f["dia"])),
    ("tone_direct & corpus≥2", lambda f: f["tone"] and f["corpus2"]),
]


def cmd_rules():
    cols = load_cols()
    set_matrix(PROD)
    C = Corpus(cols)
    h("LUẬT KHAI THÁC TỪ ĐIỂN — độ phủ × lift so với NULL LỆCH THANH GHI 1 Ô (cột khớp số)")
    n = n_un = 0
    fire, null, cov, covn = Counter(), Counter(), Counter(), Counter()
    for col in cols:
        if len(col["chars"]) != len(col["syl"]) or col["tiers"] is None:
            continue
        pg = (col["book"], col["page"])
        for o in col["ops"]:
            if o["op"] != "match":
                continue
            i, j = o["nom_idx"], o["syl_idx"]
            c, s = col["chars"][i], col["syl"][j].lower()
            if not c or not is_plausible_qn_syllable(s) or j + 1 >= len(col["syl"]):
                continue
            f0 = C.feats(col["chars"], col["syl"], i, j, pg)
            f1 = C.feats(col["chars"], col["syl"], i, j + 1, pg)
            n += 1
            un = not f0["direct"]
            n_un += un
            for name, fn in RULES:
                a, b = bool(fn(f0)), bool(fn(f1))
                fire[name] += a; null[name] += b
                if un:
                    cov[name] += a; covn[name] += b
    print(f"cặp: {n:,} · trong đó ocr_char ∉ R(âm): {n_un:,}\n")
    print(f"{'luật':38s} {'bắn/đúng':>9s} {'bắn/lệch':>9s} {'lift':>6s} | {'phủ thêm':>9s} {'null':>6s} {'lift':>6s}")
    for name, _ in RULES:
        f, z = fire[name] / n, null[name] / n
        lift = f / z if z else float("inf")
        lift2 = (cov[name] / max(1, covn[name]))
        print(f"{name:38s} {f:9.2%} {z:9.2%} {lift:6.0f} | {cov[name]:9,d} {covn[name]:6,d} {lift2:6.0f}")


# =========================================================================== calib / recursive
def _dp_with(sub, chars, syls):
    m, n = len(chars), len(syls)
    band = abs(m - n) + max(1, aa.BAND_SLACK)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    bt = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            if abs(i - j) > band or (i == 0 and j == 0):
                continue
            best, op = INF, None
            if i > 0 and j > 0 and dp[i - 1][j - 1] < INF:
                c = dp[i - 1][j - 1] + sub(i - 1, j - 1)
                if c < best:
                    best, op = c, "M"
            if i > 0 and dp[i - 1][j] + aa.COST_DEL < best:
                best, op = dp[i - 1][j] + aa.COST_DEL, "D"
            if j > 0 and dp[i][j - 1] + aa.COST_INS < best:
                best, op = dp[i][j - 1] + aa.COST_INS, "I"
            dp[i][j], bt[i][j] = best, op
    i, j = m, n
    pairs, ins = {}, []
    while i > 0 or j > 0:
        op = bt[i][j] or ("D" if i > 0 else "I")
        if op == "M":
            i -= 1; j -= 1; pairs[i] = j
        elif op == "D":
            i -= 1
        else:
            j -= 1; ins.append(j)
    return pairs, ins


def cmd_calib():
    cols = load_cols()
    all_chars = [c for col in cols for c in col["chars"] if c]
    eq = [c for c in cols if len(c["chars"]) == len(c["syl"]) and len(c["chars"]) >= 10 and c["tiers"] is not None]
    h("MA TRẬN CHI PHÍ — production vs hiệu chuẩn theo likelihood")
    for name, M in (("production (0/0,3/0,9/1,0 · khe 0,7)", PROD), ("calibrated (0/2,5/5,1/6,7 · khe 8,6)", CALIB)):
        set_matrix(M)
        off = tot = gaps = 0
        for col in eq:
            mp = [o for o in realign_column(col["chars"], col["syl"], qn, sim) if o["op"] == "match"]
            tot += len(mp)
            k = sum(1 for o in mp if o["nom_idx"] != o["syl_idx"])
            off += k; gaps += k > 0
        print(f"\n## {name}\n  cột m=n: cặp lệch chéo {off}/{tot} ({off/tot:.2%}) · cột có khe {gaps}/{len(eq)}")
        rng = random.Random(7)
        sample = rng.sample(eq, 500)
        for kind in ["drop_char", "split_char", "drop_syl", "subst_char"]:
            n = wrong = gap_ok = 0
            for col in sample:
                c2, s2, truth = perturb(col["chars"], col["syl"], kind, rng, all_chars)
                ops = realign_column(c2, s2, qn, sim)
                for o in ops:
                    if o["op"] == "match":
                        n += 1; wrong += truth.get(o["nom_idx"]) != o["syl_idx"]
                if kind == "drop_char":
                    ins = [o["syl_idx"] for o in ops if o["op"] == "ins"]
                    tg = [i for i in range(len(c2)) if truth[i] != i]
                    gap_ok += len(ins) == 1 and bool(tg) and ins[0] == tg[0]
            extra = f" · khe đúng chỗ {gap_ok}/500" if kind == "drop_char" else ""
            print(f"  {kind:10s} ghép sai {wrong}/{n} ({wrong/n:.2%}){extra}")
    # ô giao nộp đổi âm ghép khi đổi ma trận
    set_matrix(CALIB)
    changed, tot = Counter(), Counter()
    for col in cols:
        if col["tiers"] is None:
            continue
        cmap = {o["nom_idx"]: o["syl_idx"] for o in realign_column(col["chars"], col["syl"], qn, sim) if o["op"] == "match"}
        for o, t in zip([o for o in col["ops"] if o["op"] == "match"], col["tiers"]):
            tot[t] += 1
            changed[t] += cmap.get(o["nom_idx"]) != o["syl_idx"]
    print("\nô giao nộp ĐỔI âm ghép khi dùng ma trận hiệu chuẩn:")
    for t in tot:
        print(f"  {t:20s} {changed[t]:5d}/{tot[t]} ({changed[t]/tot[t]:.2%})")
    set_matrix(ENGINE)


def cmd_recursive():
    cols = load_cols()
    all_chars = [c for col in cols for c in col["chars"] if c]
    C = Corpus(cols)
    eq = [c for c in cols if len(c["chars"]) == len(c["syl"]) and len(c["chars"]) >= 10 and c["tiers"] is not None]
    h("ĐỆ QUY — neo ngữ liệu (cặp/bigram thấy ở TRANG KHÁC) nạp lại vào DP")
    for mname, M in (("production", PROD), ("calibrated", CALIB)):
        for mode in ("dict", "+corpus≥2", "+corpus+bigram"):
            set_matrix(M)
            rng = random.Random(3)
            sample = rng.sample(eq, 600)
            res = Counter()
            for kind in ("none", "drop_char", "drop_syl"):
                for col in sample:
                    pg = (col["book"], col["page"])
                    c2, s2, truth = perturb(col["chars"], col["syl"], kind, rng, all_chars)

                    def sub(i, j, c2=c2, s2=s2, pg=pg):
                        c, s = c2[i], s2[j].lower()
                        base = substitution_cost(c, s, qn, sim)
                        if mode == "dict":
                            return base
                        if len(C.pair_pages.get((c, s), set()) - {pg}) >= 2:
                            return min(base, ANCHOR_CAP)
                        if mode == "+corpus+bigram" and any(C.bigram_pages.get(k, set()) - {pg} for k in C.bigrams(c2, s2, i, j)):
                            return min(base, ANCHOR_CAP)
                        return base
                    pairs, ins = _dp_with(sub, c2, s2)
                    res[kind + "_n"] += len(pairs)
                    res[kind + "_w"] += sum(1 for i, j in pairs.items() if truth.get(i) != j)
                    if kind == "drop_char":
                        tg = [i for i in range(len(c2)) if truth[i] != i]
                        res["gap"] += len(ins) == 1 and bool(tg) and ins[0] == tg[0]
            print(f"{mname:11s} {mode:15s} | none: sai {res['none_w']/res['none_n']:.2%} | drop_char: sai "
                  f"{res['drop_char_w']/res['drop_char_n']:.2%}, khe đúng {res['gap']/600:.0%} | drop_syl: sai {res['drop_syl_w']/res['drop_syl_n']:.2%}")
    set_matrix(ENGINE)


# =========================================================================== v3
def tier_v3(f, p, P_HI=0.8, P_MID=0.5):
    ctx = f["bigram"] or f["corpus2"] or f["tone"]
    if f["direct"] and p >= P_HI:
        return "CHAR_A"
    if f["direct"]:
        return "CHAR_B"
    if f["sim_unique"] and ctx and p >= P_HI:
        return "CHAR_B"
    if p >= P_MID and (f["bigram"] or f["corpus4"] or f["tone"]):
        return "SYL"
    return "REVIEW"


def _v3_crosstab(cols, C, recursive):
    """tier hiện hành × v3. recursive=True: ma trận hiệu chuẩn + neo cặp lặp ≥2 trang khác trong DP."""
    import pandas as pd
    cross = Counter()
    for col in cols:
        if col["tiers"] is None:
            continue
        pg = (col["book"], col["page"])
        chars, syls = col["chars"], col["syl"]
        prod = {o["nom_idx"]: t for o, t in zip([o for o in col["ops"] if o["op"] == "match"], col["tiers"])}
        if not recursive:
            set_matrix(PROD)
            pairs = dict(viterbi_pairs(chars, syls))
            post = posterior_matches(chars, syls)
        else:
            set_matrix(CALIB)

            def sub(i, j):
                base = substitution_cost(chars[i], syls[j], qn, sim)
                if len(C.pair_pages.get((chars[i], syls[j].lower()), set()) - {pg}) >= 2:
                    return min(base, aa.COST_CONFIRM)
                return base
            pairs, _ = _dp_with(sub, chars, syls)
            # posterior dùng cùng chi phí có neo (vá tạm substitution_cost ở mức module)
            global substitution_cost
            orig = substitution_cost
            substitution_cost = lambda c, s, q, sm, _sub=orig: (min(_sub(c, s, q, sm), aa.COST_CONFIRM)
                                                                if len(C.pair_pages.get((c, s.lower()), set()) - {pg}) >= 2 else _sub(c, s, q, sm))
            try:
                post = posterior_matches(chars, syls, T=1.0)
            finally:
                substitution_cost = orig
        for i, j in pairs.items():
            f = C.feats(chars, syls, i, j, pg)
            cross[(prod.get(i, "(khe)"), tier_v3(f, post.get((i, j), 0.0)))] += 1
    set_matrix(ENGINE)
    tab = pd.Series(cross).unstack(fill_value=0)
    print(tab.to_string()); print("tổng v3:", tab.sum(axis=0).to_dict())


def cmd_v3():
    cols = load_cols()
    set_matrix(PROD)
    C = Corpus(cols)
    all_chars = [c for col in cols for c in col["chars"] if c]
    h("MÔ PHỎNG PHÂN HẠNG v3 (posterior + luật có lift) — dữ liệu thật, ma trận production")
    _v3_crosstab(cols, C, recursive=False)
    h("MÔ PHỎNG v3 — ma trận hiệu chuẩn + đệ quy neo ngữ liệu")
    _v3_crosstab(cols, C, recursive=True)
    h("BENCHMARK — tỉ lệ GHÉP SAI theo tier, luật hiện hành vs v3 (600 cột khớp số)")
    eq = [c for c in cols if len(c["chars"]) == len(c["syl"]) and len(c["chars"]) >= 10 and c["tiers"] is not None]
    rng = random.Random(2026)
    sample = rng.sample(eq, 600)
    for kind in ("none", "drop_char", "drop2_char", "split_char", "drop_syl"):
        stat = defaultdict(lambda: [0, 0])
        for col in sample:
            pg = (col["book"], col["page"])
            c2, s2, truth = perturb(col["chars"], col["syl"], kind, rng, all_chars)
            pairs = viterbi_pairs(c2, s2)
            post = posterior_matches(c2, s2)
            conf = [is_confirmed(c2[i], s2[j], qn, sim) for (i, j) in pairs]
            matched = len(c2) == len(s2)
            for k, (i, j) in enumerate(pairs):
                f = C.feats(c2, s2, i, j, pg)
                anchored = (k > 0 and conf[k - 1]) or (k + 1 < len(conf) and conf[k + 1])
                cur = "GOLD" if f["direct"] or ((matched or anchored) and f["sim_unique"]) else "REVIEW"
                ok = truth.get(i) == j
                for name in ("cur:" + cur, "v3:" + tier_v3(f, post.get((i, j), 0.0))):
                    stat[name][0] += 1; stat[name][1] += not ok
        print(f"\n{kind}:")
        for name in sorted(stat):
            n, w = stat[name]
            print(f"  {name:12s} n={n:6d}  ghép sai {w:4d} ({w/max(1,n):.2%})")


# =========================================================================== classes
def cmd_classes():
    import pandas as pd
    df = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
    chk_path = REPO / "re-dataset/check/labels.xlsx"
    if chk_path.exists():
        chk = pd.read_excel(chk_path, sheet_name="Nhan", dtype=str)
        df = df.merge(chk[["image", "ket_luan", "dung_sai"]], on="image", how="left")
        df["nghi_sai"] = df.dung_sai.astype(str).str.lower().isin(["0", "0.0", "false"])
    else:
        df["nghi_sai"] = False
    g = df[df.tier == "GOLD"]
    h("BỘ KIỂM ĐỘC LẬP (re-dataset/check) — tỉ lệ 'nghi sai' theo luật GOLD")
    print(g.groupby("rule").agg(n=("image", "size"), nghi_sai=("nghi_sai", "mean")).sort_values("n", ascending=False).round(3).to_string())
    h("LỚP (sách, âm, nhãn THIỂU SỐ ≥3 ô) — ứng viên phân xử theo lớp")
    rows = []
    for (b, s), grp in g.groupby(["book", "syllable"]):
        c = Counter(grp.label)
        if len(c) < 2:
            continue
        M, nM = c.most_common(1)[0]
        for L, n in c.items():
            if L == M or n < 3:
                continue
            sub = grp[grp.label == L]
            rows.append(dict(book=b, syllable=s, majority=M, n_major=nM, minority=L, n_minor=n,
                             lookalike=(L in sim.get(M, [])) or (M in sim.get(L, [])),
                             chk_suspect=round(sub.nghi_sai.mean(), 2),
                             direct=round((sub.rule == "s1_inter_s2_direct").mean(), 2)))
    R = pd.DataFrame(rows)
    print(f"lớp: {len(R)} · ô: {R.n_minor.sum()} · nhìn giống nhãn đa số: {R.lookalike.sum()} lớp / {R[R.lookalike].n_minor.sum()} ô")
    print(R[R.lookalike].sort_values("n_minor", ascending=False).head(25).to_string(index=False))
    h("LỚP (chữ OCR, âm) ∉ R — Pareto để phân xử/mở rộng từ điển ngữ liệu")
    d = df[df.ocr_char != ""]
    un = d[[c not in Rset.get(s, ()) for c, s in zip(d.ocr_char, d.syllable)]]
    pc = Counter(zip(un.ocr_char, un.syllable)).most_common()
    print(f"ô: {len(un):,} · cặp: {len(pc):,}")
    for N in (50, 100, 300, 500, 1000):
        print(f"  top {N:4d} cặp phủ {sum(v for _, v in pc[:N]):6,d} ô")


# =========================================================================== geo
def cmd_geo():
    """Hộp ảnh cho chữ OCR khi cột lệch 1 đơn vị — THAM SỐ HOÁ theo engine (A-6, N4d):
    thr = ap.DETECTOR_THR, hộp thô = det.raw_column_boxes(±ap.DETECTOR_XMARGIN·w), điều
    kiện lấy mẫu |G| == |Q| (in độ phủ), gán hộp bằng ap.assign_boxes (cùng hàm sản xuất).
    Kịch bản INS = rụng 1 chữ OCR (n_ocr = |Q| − 1, nhánh a: hộp theo syl_idx), DEL = rụng
    1 âm QN (n_qn = |Q| − 1, nhánh b: hộp theo nom_idx). Ma trận = engine (CALIB)."""
    import cv2
    from pipeline.align_engine import align_production as ap
    from pipeline.align_engine.char_detector.detector_infer import DetectorInfer
    readings = build_readings(qn)
    det = DetectorInfer(thr=ap.DETECTOR_THR)
    rng = random.Random(11)
    pages = []
    for b in BOOKS:
        ts = [t for t in sorted(glob.glob(str(REPO / "prepared" / b / "transcriptions" / "page_*.json"))) if not t.endswith("_qn_ocr_cache.json")]
        pages += [(b, Path(t).stem) for t in rng.sample(ts, 14)]
    set_matrix(ENGINE)
    tot = Counter()
    cov = Counter()
    for b, page in pages:
        dd = REPO / "prepared" / b
        od = json.load(open(dd / "detected" / f"{page}_ocr_cache.json", encoding="utf-8"))
        lines, _ = _get_qn_lines(dd, page, qs)
        keys = sorted(lines)
        cs = nom_cols_hybrid(od["columns"], 4)
        if len(cs) != 9:
            cs = nom_cols_hybrid(od["columns"], 1)
        img = cv2.imread(str(dd / "pages" / f"{page}.png"))
        pb = det.boxes_for_page(img)
        for i in range(min(len(cs), len(keys))):
            cl = cs[i]
            chars = [c["char"] for c in cl["chars"]]
            syl, _ = normalize_column(cl["chars"], lines[keys[i]], qs, readings)
            if len(chars) != len(syl) or len(chars) < 10:
                continue
            cov["cột OCR=QN, ≥10 chữ"] += 1
            G = det.raw_column_boxes(pb, cl["x_range"], ap.DETECTOR_XMARGIN)
            cov["|G| == |Q|" if len(G) == len(syl) else ("|G| < |Q|" if len(G) < len(syl) else "|G| > |Q|")] += 1
            if len(G) != len(syl):
                continue
            x1, x2 = cl["x_range"]
            y_top = min(c["bbox"][1] for c in cl["chars"]); y_bot = max(c["bbox"][3] for c in cl["chars"]); H = y_bot - y_top
            k = rng.randrange(2, len(chars) - 2)
            for kind in ("INS", "DEL"):
                if kind == "INS":                                   # rụng chữ k: n_ocr = |Q| − 1
                    c2 = chars[:k] + chars[k + 1:]; s2 = syl
                    truth_box = {t: (t if t < k else t + 1) for t in range(len(c2))}
                    truth_syl = dict(truth_box)
                else:                                               # rụng âm k: n_qn = |Q| − 1
                    c2 = chars; s2 = syl[:k] + syl[k + 1:]
                    truth_box = {t: t for t in range(len(c2))}
                    truth_syl = {t: (t if t < k else (None if t == k else t - 1)) for t in range(len(c2))}
                n2 = len(c2)
                # tâm TỔNG HỢP như ocr_api dựng (chỉ nhánh c dùng; ở đây |G| ∈ {n_qn, n_ocr})
                cl2 = {"chars": [{"char": c2[t], "bbox": [x1, int(y_top + t * H / n2), x2, int(y_top + (t + 1) * H / n2)]} for t in range(n2)],
                       "x_range": cl["x_range"]}
                ops = realign_column(c2, s2, qn, sim)
                boxes, src, count_source = ap.assign_boxes(G, ops, n2, len(s2), cluster=cl2, det=det)
                tot[f"{kind}:{count_source}"] += 1
                gidx = {tuple(g[:4]): gi for gi, g in enumerate(G)}
                for o in ops:
                    if o["op"] != "match":
                        continue
                    t, j = o["nom_idx"], o["syl_idx"]
                    tot[f"{kind}:n"] += 1
                    q = gidx.get(tuple(boxes[t][:4])) if boxes[t] is not None else None
                    tot[f"{kind}:box_ok"] += q == truth_box[t]
                    tot[f"{kind}:box_wrong"] += q is not None and q != truth_box[t]
                    tot[f"{kind}:box_none"] += q is None
                    tot[f"{kind}:syl_ok"] += j == truth_syl[t]
    h(f"HỘP ẢNH CHO CHỮ OCR KHI CỘT LỆCH 1 (thr {ap.DETECTOR_THR}, ±{ap.DETECTOR_XMARGIN}w, ma trận engine; {len(pages)} trang)")
    print(f"  độ phủ: {dict(cov)}")
    for kind, mota in (("INS", "rụng 1 chữ OCR — nhánh (a) hộp theo syl_idx"), ("DEL", "rụng 1 âm QN — nhánh (b) hộp theo nom_idx")):
        n = tot[f"{kind}:n"] or 1
        srcs = {k.split(":")[1]: v for k, v in tot.items() if k.startswith(kind + ":") and k.split(":")[1] in ("equal_qn", "equal_ocr", "conflict")}
        print(f"  {kind} ({mota}): {tot[kind + ':n']} ô · count_source {srcs}")
        print(f"     hộp đúng {tot[kind + ':box_ok'] / n:.1%} · hộp của chữ KHÁC {tot[kind + ':box_wrong'] / n:.1%} · không hộp/midpoint {tot[kind + ':box_none'] / n:.1%} · ghép đúng âm {tot[kind + ':syl_ok'] / n:.1%}")
    set_matrix(ENGINE)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"rebuild": cmd_rebuild, "posterior": cmd_posterior, "rules": cmd_rules, "calib": cmd_calib,
             "recursive": cmd_recursive, "v3": cmd_v3, "classes": cmd_classes, "geo": cmd_geo}
    if cmd == "all":
        for k in ("rebuild", "posterior", "rules", "calib", "recursive", "v3", "classes"):
            steps[k]()
    else:
        steps[cmd]()
