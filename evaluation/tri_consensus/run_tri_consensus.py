"""Tri-model consensus — kim (coord base) + Qwen + NomNaOCR → vote per position,
with 2 optional levers:
  --variants : treat VARIANT forms as agreement (崑=昆, 别=別). variant-equiv(a,b|syl)
               = both ∈ dict(syllable) AND visually-similar (SinoNom_Similar_Dic). The
               "both ∈ dict(same syllable)" guard blocks look-alikes that are different
               chars (未/末 share no syllable → NOT merged).
  --dict     : when no majority, DICT ADJUDICATION — the char ∈ dict(syllable) wins
               (unique valid → DICT; several different valid → still REVIEW).

Syllable per position comes from dataset_out/labels.csv (has syllable for ALL tiers),
aligned to the kim-cache char sequence by ocr_char.

Runs in MAIN .venv (reads caches; no TF/paddle/network).
  .venv/bin/python evaluation/tri_consensus/run_tri_consensus.py --n 5 \
      --qwen_dir qwen_cache_235b --variants --dict
"""
from __future__ import annotations
import argparse, ast, csv, json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
QWEN_FRAME = HERE / "qwen_cache"
QWEN_FULLPAGE = REPO / "evaluation" / "qwen_test" / "cache"
NNA_CACHE = HERE / "nomnaocr_cache"
BOOKCODE = {"SachThanhTruyen2": "yen2", "SachThanhTruyen4": "yen4", "SachThanhTruyen11": "yen11"}


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2EBEF or 0x2F800 <= o <= 0x2FA1F)


def _bt(a, b):
    """Needleman-Wunsch dp + backtrace of aligned (i,j) pairs (matches/subs)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs


def align_map(a, b):
    """for each a[i] -> aligned b char or None."""
    res = [None] * len(a)
    for i, j in _bt(a, b):
        res[i] = b[j]
    return res


def align_to(a, b_vals, b_keys=None):
    """for each a[i] -> the b VALUE (b_vals[j]) it aligns to, or None."""
    res = [None] * len(a)
    keys = b_keys if b_keys is not None else b_vals
    for i, j in _bt(a, keys):
        res[i] = b_vals[j]
    return res


def load_qn_dict():
    d = defaultdict(set)
    with open(REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[r["QuocNgu"].strip().lower()].add(r["SinoNom"].strip())
    return d


def load_similar(k=20):
    sim = {}
    with open(REPO / "Dict" / "SinoNom_Similar_Dic_v2.csv", encoding="utf-8-sig") as f:
        rd = csv.reader(f); next(rd)
        for row in rd:
            if len(row) >= 2:
                try:
                    sim[row[0]] = set(ast.literal_eval(row[1])[:k])
                except Exception:
                    pass
    return sim


def load_syllables(book):
    """(page) -> (ocr_seq, syl_seq) in kim reading order (col asc = R→L, then y)."""
    rows = defaultdict(list)
    code = BOOKCODE[book]
    with open(REPO / "dataset_out" / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["book"] != code or not r.get("ocr_char"):
                continue
            try:
                y = ast.literal_eval(r["bbox"])[1]
            except Exception:
                continue
            rows[r["page"]].append((int(r["column"]), y, r["ocr_char"], r.get("syllable", "")))
    out = {}
    for page, lst in rows.items():
        lst.sort(key=lambda x: (x[0], x[1]))
        out[page] = ([o for _, _, o, _ in lst], [s for _, _, _, s in lst])
    return out


def load_syl_rows(book):
    """(page) -> list[(col_int, y, ocr_char, syllable)] (raw, chưa order)."""
    rows = defaultdict(list); code = BOOKCODE[book]
    with open(REPO / "dataset_out" / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["book"] != code or not r.get("ocr_char"):
                continue
            try:
                y = ast.literal_eval(r["bbox"])[1]
            except Exception:
                continue
            rows[r["page"]].append((int(r["column"]), y, r["ocr_char"], r.get("syllable", "")))
    return rows


def smap_for(kim_cols, rows, method):
    """Syllable per kim position. method='page' = 1 lần align cả trang (dễ TRÔI);
    'percol' = align TỪNG CỘT với cột labels khớp nhất (chắc hơn, chống trôi chéo cột)."""
    kim_flat = [c["char"] for col in kim_cols for c in col]
    if not rows:
        return [""] * len(kim_flat)
    if method == "page":
        s = sorted(rows, key=lambda r: (r[0], r[1]))
        return [x or "" for x in align_to(kim_flat, [r[3] for r in s], [r[2] for r in s])]
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    for k in by:
        by[k].sort(key=lambda r: r[1])
    smap = []
    for col in kim_cols:
        kc = [c["char"] for c in col]
        best, bh = None, -1
        for cn, lr in by.items():                       # chọn cột labels khớp NHIỀU ký tự nhất
            am = align_map(kc, [r[2] for r in lr])
            h = sum(1 for i, ch in enumerate(am) if ch == kc[i])
            if h > bh:
                bh, best = h, cn
        if best is None:
            smap += [""] * len(kc); continue
        lr = by[best]
        smap += [x or "" for x in align_to(kc, [r[3] for r in lr], [r[2] for r in lr])]
    return smap


def make_decider(qn, sim, use_variants, use_dict):
    def variant_eq(a, b, syl):
        if not (use_variants and syl):
            return False
        if a in qn.get(syl, ()) and b in qn.get(syl, ()):
            return b in sim.get(a, ()) or a in sim.get(b, ())
        return False

    def agree(a, b, syl):
        return a == b or variant_eq(a, b, syl)

    def decide(kim, qw, nn, syl):
        votes = [v for v in (kim, qw, nn) if v]
        present3 = bool(kim and qw and nn)
        # best cluster (variant-aware); tie -> prefer kim
        best_char, best_n = kim, 0
        for cand in votes:
            n = sum(1 for v in votes if agree(cand, v, syl))
            if n > best_n or (n == best_n and cand == kim):
                best_n, best_char = n, cand
        if present3 and best_n == 3:
            return best_char, "CONSENSUS3"
        if best_n >= 2:
            return best_char, "MAJORITY"
        # no majority -> dict adjudication
        if use_dict and syl:
            valid = [v for v in votes if v in qn.get(syl, ())]
            uniq = list(dict.fromkeys(valid))
            if len(uniq) == 1:
                return uniq[0], "DICT"
            if len(uniq) >= 2 and all(agree(uniq[0], v, syl) for v in uniq):
                return uniq[0], "DICT"
        return kim, "REVIEW"
    return decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen4")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--qwen", choices=["frame", "fullpage"], default="frame")
    ap.add_argument("--qwen_dir", default="")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--dict", dest="use_dict", action="store_true")
    ap.add_argument("--syl", choices=["page", "percol"], default="percol",
                    help="align syllable: 'percol' = từng cột (chắc), 'page' = cả trang (cũ, dễ trôi)")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="bỏ Qwen ở trang có tỉ lệ kim==nna >= gate (0 = luôn dùng Qwen). Vd 0.7")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    QWEN_CACHE = (HERE / args.qwen_dir) if args.qwen_dir else (QWEN_FRAME if args.qwen == "frame" else QWEN_FULLPAGE)

    lever = args.variants or args.use_dict
    qn = load_qn_dict() if lever else {}
    sim = load_similar() if args.variants else {}
    syl_rows = load_syl_rows(args.book) if lever else {}
    decide = make_decider(qn, sim, args.variants, args.use_dict)

    det = REPO / "prepared" / args.book / "pages_denoised"
    cdir = REPO / "prepared" / args.book / "detected"
    stems = [f.stem for f in sorted(det.glob("page_*.png"))
             if (cdir / f"{f.stem}_ocr_cache.json").exists()][: args.n]

    tiers = Counter(); n_pos = n_ovr = n_syl = n_kim_in = 0
    q_used = q_skip = 0
    out = HERE / "out"; out.mkdir(exist_ok=True)
    overruled = [("book", "page", "col", "syl", "kim", "qwen", "nna", "label", "tier")]
    for stem in stems:
        cache = json.loads((cdir / f"{stem}_ocr_cache.json").read_text(encoding="utf-8"))
        kim_cols = [c for c in cache.get("columns", []) if c]
        kim_flat = [c["char"] for col in kim_cols for c in col]
        smap = smap_for(kim_cols, syl_rows.get(stem, []), args.syl)
        n_syl += sum(1 for s in smap if s)
        n_kim_in += sum(1 for i, s in enumerate(smap) if s and kim_flat[i] in qn.get(s, ()))
        # nna (per-column align)
        nf = NNA_CACHE / f"{args.book}_{stem}.json"
        nmap = []
        if nf.exists():
            nc = json.loads(nf.read_text(encoding="utf-8")).get("columns", [])
            if len(nc) == len(kim_cols):
                for col, ns in zip(kim_cols, nc):
                    nmap += align_map([c["char"] for c in col], list(ns))
        if len(nmap) != len(kim_flat):
            nmap = [None] * len(kim_flat)
        # GATE: bỏ Qwen ở trang mà kim==nna đủ cao
        qf = QWEN_CACHE / f"{args.book}_{stem}.json"
        npres = sum(1 for x in nmap if x)
        kn_agree = (sum(1 for i in range(len(kim_flat)) if nmap[i] and kim_flat[i] == nmap[i]) / npres) if npres else 0.0
        skip_q = args.gate > 0 and kn_agree >= args.gate
        if skip_q or not qf.exists():
            qmap = [None] * len(kim_flat)
            if qf.exists():
                q_skip += 1
        else:
            qmap = align_map(kim_flat, [c for c in json.loads(qf.read_text(encoding="utf-8")).get("text", "") if is_cjk(c)])
            q_used += 1

        lines = [f"# {stem} pos={len(kim_flat)} syl={args.syl} gate={args.gate} "
                 f"kim=nna {100*kn_agree:.0f}% {'[QWEN SKIP]' if skip_q else ''}"]
        gi = 0
        for ci, col in enumerate(kim_cols):
            for c in col:
                kim, qw, nn, syl = c["char"], qmap[gi], nmap[gi], smap[gi]
                lbl, tier = decide(kim, qw, nn, syl)
                tiers[tier] += 1
                ovr = bool(qw and nn and lbl != kim and (qw == lbl or nn == lbl))
                if ovr:
                    n_ovr += 1
                    overruled.append((args.book, stem, ci, syl, kim, qw or "·", nn or "·", lbl, tier))
                lines.append(f"  {gi:>4} {kim:>3} {qw or '·':>4} {nn or '·':>3} {syl or '·':>7} "
                             f"{'->':>2} {lbl:>4} {tier}{' *OVR' if ovr else ''}")
                gi += 1
        (out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_pos += len(kim_flat)

    with open(out / "kim_overruled.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(overruled)

    d = n_pos or 1
    sd = n_syl or 1
    usable = tiers["CONSENSUS3"] + tiers["MAJORITY"] + tiers["DICT"]
    tag = args.tag or f"syl={args.syl} gate={args.gate} qwen={QWEN_CACHE.name}"
    print(f"===== [{tag}] {n_pos} vị trí ({args.n} trang) =====")
    print(f"  align syllable: có syl {100*n_syl/d:.0f}%  |  kim∈dict(syl) {100*n_kim_in/sd:.0f}% "
          f"(cao = align syllable CHẮC)")
    print(f"  Qwen calls: dùng {q_used}, BỎ {q_skip}/{q_used+q_skip} trang (tiết kiệm {100*q_skip/max(1,q_used+q_skip):.0f}%)")
    print(f"  CONSENSUS3 : {tiers['CONSENSUS3']:5}  ({100*tiers['CONSENSUS3']/d:5.1f}%)")
    print(f"  MAJORITY   : {tiers['MAJORITY']:5}  ({100*tiers['MAJORITY']/d:5.1f}%)")
    print(f"  DICT       : {tiers['DICT']:5}  ({100*tiers['DICT']/d:5.1f}%)   (từ điển phân xử)")
    print(f"  REVIEW     : {tiers['REVIEW']:5}  ({100*tiers['REVIEW']/d:5.1f}%)")
    print(f"  -> USABLE : {usable} ({100*usable/d:.1f}%) | REVIEW {100*tiers['REVIEW']/d:.1f}% | overrule {n_ovr}")


if __name__ == "__main__":
    main()
