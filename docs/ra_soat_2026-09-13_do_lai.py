"""Tái sinh MỌI con số trong docs/RA_SOAT_TOAN_BO_2026-09-13.md.

Chỉ đọc: prepared/, dict/, dataset_out/labels_final.csv, dataset_out/labels.csv, git.
Không ghi gì vào repo. Chạy ~1 phút (không cần torch/detector).

    .venv/bin/python docs/ra_soat_2026-09-13_do_lai.py
"""
from __future__ import annotations

import glob
import io
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.align.run_full import nom_cols_hybrid                      # noqa: E402
from core.text.dictionary import load_qn_to_nom, load_similarity_dict  # noqa: E402
from core.text.text_utils import is_plausible_qn_syllable            # noqa: E402
from pipeline.align_engine.anchor_align import realign_column         # noqa: E402
from pipeline.step2_align import _get_qn_lines                        # noqa: E402

BOOKS = {"SachThanhTruyen2": "stt2", "SachThanhTruyen4": "stt4", "SachThanhTruyen11": "stt11"}


def h(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main():
    qn = load_qn_to_nom(str(REPO / "dict/QuocNgu_SinoNom.csv"))
    sim = load_similarity_dict(str(REPO / "dict/SinoNom_Similar.csv"))
    qs = set(qn)

    # ---------------------------------------------------------------- A. cột
    h("A. Số chữ OCR Nôm vs số âm tiết QN theo CỘT (đầu vào của căn chỉnh)")
    colinfo = {}
    n_ocr_tot = n_qn_tot = 0
    parser_src = Counter()
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
            lines, src = _get_qn_lines(dd, page, qs)
            parser_src[src] += 1
            keys = sorted(lines)
            cols = nom_cols_hybrid(od["columns"], min_len=4)
            if len(cols) != 9:
                cols = nom_cols_hybrid(od["columns"], min_len=1)
            for i in range(min(len(cols), len(keys))):
                a, b = len(cols[i]["chars"]), len(lines[keys[i]])
                colinfo[(code, page, keys[i])] = (a, b)
                n_ocr_tot += a
                n_qn_tot += b
    diffs = Counter(a - b for a, b in colinfo.values())
    tot = sum(diffs.values())
    print(f"cột: {tot} · khớp số: {diffs[0]} ({diffs[0]/tot:.1%}) · OCR<QN: "
          f"{sum(v for k, v in diffs.items() if k < 0)} · OCR>QN: {sum(v for k, v in diffs.items() if k > 0)}")
    print(f"|lệch|>=2: {sum(v for k, v in diffs.items() if abs(k) >= 2)} cột")
    print(f"tổng chữ OCR {n_ocr_tot} · tổng âm tiết {n_qn_tot} · cặp phát ra 82.780 "
          f"-> chữ OCR không có âm (del) {n_ocr_tot-82780} · âm không có hộp (ins) {n_qn_tot-82780}")
    print(f"nguồn bóc dòng QN: {dict(parser_src)}  <- trang dùng fallback 'numbered' nhận NGUYÊN DÒNG làm 1 âm tiết (BUG)")

    # ------------------------------------------------------------- B. nhãn
    h("B. Bộ nhãn hiện hành theo lớp cột")
    df = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
    df["column"] = df["column"].astype(int)
    ab = [colinfo.get((r.book, r.page, r.column), (None, None)) for r in df.itertuples()]
    df["dif"] = [None if a is None else a - b for a, b in ab]
    df["mism"] = df["dif"].map(lambda d: "na" if d is None else ("eq" if d == 0 else ("ocr<qn" if d < 0 else "ocr>qn")))
    print(pd.crosstab(df.mism, df.tier, margins=True))
    usable = df[df.tier.isin(["GOLD", "SYLLABLE"])]
    print("\nusable (GOLD+SYLLABLE) theo lớp cột:", usable.mism.value_counts().to_dict())

    # hộp midpoint (tổng hợp) — nhận diện bằng x-range trùng nhau nguyên cột
    df["bb"] = df.bbox.apply(json.loads)
    df["xr"] = df.bb.apply(lambda b: (b[0], b[2]))
    mid = Counter(); cells = Counter()
    for _, g in df.groupby(["book", "page", "column"]):
        if len(g) < 4:
            continue
        mode = Counter(g.xr).most_common(1)[0]
        cls = g.mism.iloc[0]
        mid[cls] += mode[1] if mode[1] >= 3 else 0
        cells[cls] += len(g)
    print("\nhộp có dấu hiệu MIDPOINT (x-range trùng nguyên cột):")
    for k in ("eq", "ocr<qn", "ocr>qn"):
        print(f"  {k:7s} {mid[k]/cells[k]:.1%} / {cells[k]} ô")

    # ------------------------------------------------- C. neo từ điển giả
    h("C. Xác nhận từ điển GIẢ khi lệch thanh ghi 1 ô (chỉ cột khớp số)")
    df["cy"] = df.bb.apply(lambda b: (b[1] + b[3]) / 2)
    df = df.sort_values(["book", "page", "column", "cy"])
    random.seed(0)
    allsyl = list(df.syllable)
    n = direct = shifted = shifted_sim = rnd = 0
    for _, g in df[df.mism == "eq"].groupby(["book", "page", "column"]):
        chars, syls = list(g.ocr_char), list(g.syllable)
        if len(chars) < 6:
            continue
        for i, (c, s) in enumerate(zip(chars, syls)):
            if not c:
                continue
            n += 1
            direct += c in qn.get(s, [])
            s2 = syls[i + 1] if i + 1 < len(syls) else syls[i - 1]
            if c in qn.get(s2, []):
                shifted += 1
            elif any(x in qn.get(s2, []) for x in sim.get(c, [])):
                shifted_sim += 1
            rnd += c in qn.get(random.choice(allsyl), [])
    print(f"ô: {n} · direct đúng thanh ghi {direct/n:.1%} · direct khi LỆCH 1: {shifted/n:.2%} "
          f"· cầu tự dạng khi lệch 1: {shifted_sim/n:.2%} · direct với âm NGẪU NHIÊN: {rnd/n:.2%}")

    ok = tot2 = 0
    for _, g in list(df[df.mism == "eq"].groupby(["book", "page", "column"]))[:400]:
        chars, syls = list(g.ocr_char), list(g.syllable)
        if len(chars) < 8:
            continue
        k = len(chars) // 2
        ops = realign_column(chars[:k] + chars[k + 1:], syls, qn, sim)
        ins = [o for o in ops if o["op"] == "ins"]
        tot2 += 1
        ok += len(ins) == 1 and ins[0]["syl_idx"] == k
    print(f"DP đặt khe đúng chỗ khi OCR rụng 1 chữ giữa cột: {ok}/{tot2} = {ok/tot2:.1%}")

    # ------------------------------------------------- D. GOLD: dị thể / nhầm hệ thống
    h("D. GOLD — nhãn thiểu số NHÌN GIỐNG nhãn đa số của cùng (sách, âm)")
    g = df[df.tier == "GOLD"]
    R = g.syllable.map(lambda s: len(qn.get(s, [])))
    print("|R(âm)| trên ô GOLD — phân vị:", {k: int(v) for k, v in R.quantile([.1, .25, .5, .75, .9]).items()})
    sus = 0; tot_min = 0; pairs = Counter()
    for (b, s), grp in g.groupby(["book", "syllable"]):
        c = Counter(grp.label)
        if len(c) < 2:
            continue
        M = c.most_common(1)[0][0]
        for L, k in c.items():
            if L == M:
                continue
            tot_min += k
            if L in sim.get(M, []) or M in sim.get(L, []):
                sus += k; pairs[(s, M, L)] += k
    print(f"ô GOLD mang nhãn thiểu số: {tot_min} · trong đó nhìn giống nhãn đa số: {sus} ({sus/len(g):.1%} GOLD)")
    print("top:", pairs.most_common(12))

    # ------------------------------------------------- E. L1 sửa âm
    h("E. L1 `s1_inter_s2_direct_am_sua_dau` — âm gốc có phải từ có thật?")
    raw = pd.read_csv(REPO / "dataset_out/labels.csv", dtype=str, keep_default_na=False)
    l1 = raw[raw.rule == "s1_inter_s2_direct_am_sua_dau"]
    old = subprocess.run(["git", "-C", str(REPO), "show", "2f0b9dc116:dataset_out/labels.csv"],
                         capture_output=True, text=True).stdout
    odf = pd.read_csv(io.StringIO(old), dtype=str, keep_default_na=False)
    key = ["book", "page", "column", "bbox"]
    m = l1.merge(odf[key + ["syllable"]], on=key, suffixes=("", "_goc"), how="inner")
    real = m.syllable_goc.map(lambda s: s in qn)
    print(f"L1: {len(l1)} ô · khớp thế hệ trước: {len(m)} · âm GỐC là từ có trong từ điển: {real.sum()} ({real.mean():.0%})")
    print("top (âm gốc -> âm mới, chữ):", Counter(zip(m[real].syllable_goc, m[real].syllable, m[real].ocr_char)).most_common(10))

    # ------------------------------------------------- F. REVIEW
    h("F. REVIEW — bằng chứng thanh ghi cục bộ và cổng độ thuần")
    df["gd"] = df.rule == "s1_inter_s2_direct"
    grp = df.groupby(["book", "page", "column"]).gd
    df["prev"] = grp.shift(1).fillna(False).astype(bool)
    df["next"] = grp.shift(-1).fillna(False).astype(bool)
    r = df[df.tier == "REVIEW"]
    print(f"REVIEW {len(r)} · kẹp giữa 2 ô GOLD-trực-tiếp: {(r.prev & r.next).sum()} "
          f"(trong cột khớp số: {((r.prev & r.next) & (r.mism == 'eq')).sum()})")
    s = df[df.tier == "SILVER_uncalibrated"]
    print(f"SILVER_uncalibrated {len(s)} · trong cột khớp số: {(s.mism == 'eq').sum()}")
    pool = df[df.tier.isin(["REVIEW", "SILVER_uncalibrated"]) & (df.ocr_char != "")]
    cnt = defaultdict(Counter); pages = defaultdict(lambda: defaultdict(set))
    for row in pool.itertuples():
        cnt[row.ocr_char][row.syllable] += 1
        pages[row.ocr_char][row.syllable].add((row.book, row.page))
    blocked = 0; bp = []
    for ch, c in cnt.items():
        t = sum(c.values())
        for syl, k in c.items():
            if k >= 5 and len(pages[ch][syl]) >= 3 and k / t < 0.6:
                blocked += k; bp.append((ch, syl, k, round(k / t, 2)))
    bp.sort(key=lambda x: -x[2])
    print(f"cặp đủ (>=5 ô, >=3 trang) nhưng RỚT cổng độ thuần 0,6: {blocked} ô · ví dụ: {bp[:8]}")
    r2 = r[r.rule == "no_s1_inter_s2"]
    pc = Counter(zip(r2.ocr_char, r2.syllable))
    print(f"no_s1_inter_s2: {len(r2)} ô · {len(pc)} cặp · đơn lẻ {sum(1 for v in pc.values() if v == 1)}")

    # ------------------------------------------------- G. QN ngoài từ điển
    h("G. Âm tiết QN ngoài từ điển")
    ood = df[~df.syllable.isin(qs)]
    print(f"{len(ood)} ô ({len(ood)/len(df):.2%}) · tier: {ood.tier.value_counts().to_dict()}")
    print("token >12 ký tự (nguyên dòng lọt làm 1 âm):", (df.syllable.str.len() > 12).sum(),
          "· trang:", df[df.syllable.str.len() > 12].groupby(["book", "page"]).size().to_dict())


if __name__ == "__main__":
    main()
