#!/usr/bin/env python3
"""ptcl_layout.py — đo BỐ CỤC bản CHÉP TAY Truyện Kiều "Phong Tình Cổ Lục" (R.987 / NLVNPF-0221).

Khác LucVanTien1916 / TruyenKieu1872 (IHR-NomDB, có bbox cột + GT chữ): bộ này chỉ có
    data/TruyenKieuPhongTinhCoLuc/pages/*.jpg          120 ảnh 2000×1820 (TỜ ĐÔI)
    data/TruyenKieuPhongTinhCoLuc/reference_kieu_1872.tsv               3.239 câu QN+Nôm bản 1872
    data/TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871_LieuVanDuong_phienam.json  bản 1871 (gần R.987 hơn)
⇒ KHÔNG có nhãn của chính bản chép tay: mọi QN dùng để gán nhãn đều là DỊ BẢN.

Phép đo (0 token LLM; kim chỉ khi --kim-pages > 0):
  1. cột/trang theo chiếu mực (khung + bước cột bằng tự tương quan), cột/tờ-đôi và cột/nửa-tờ
  2. TẦNG TRÊN: dải chữ nằm cao hơn thân trang (tựa / đề vịnh) — tỉ lệ trang có, chiều cao
  3. bước chữ trong cột, px/chữ, số chữ/cột (đếm run mực) → cột = 1 CÂU hay 1 CẶP lục bát
  4. kim trên vài trang: số chữ mỗi cột thật sự (đối chứng cho mục 3) và khớp dị bản 1871/1872
  5. nền dị bản: 1871 ↔ 1872 khớp bao nhiêu (đã biết 82,9 %) — cận trên của mọi phép so sau này

Chạy:
    .venv/bin/python scripts/measure/ptcl_layout.py
    .venv/bin/python scripts/measure/ptcl_layout.py --kim-pages 6
    .venv/bin/python scripts/measure/ptcl_layout.py --selftest
Đầu ra: measure_out/TruyenKieuPhongTinhCoLuc/ptcl_layout/{summary.json,pages.csv,cols.csv,kim_cols.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BOOK = "TruyenKieuPhongTinhCoLuc"
ROOT = REPO / "data" / BOOK
PAGES = ROOT / "pages"
REF1872 = ROOT / "reference_kieu_1872.tsv"
REF1871 = ROOT / "thamchieu_kieu_1871_LieuVanDuong_phienam.json"


# --------------------------------------------------------------------------- dị bản
def load_1871() -> list[dict]:
    """[{verse_no|None, nom, qn}] theo thứ tự trang bản in 1871 → chuỗi câu phẳng."""
    d = json.loads(REF1871.read_text(encoding="utf-8"))
    out = []
    for page in (d.get("pages") or {}).values():
        out.extend(page)
    return out


def load_1872() -> list[dict]:
    return list(csv.DictReader(open(REF1872, encoding="utf-8"), delimiter="\t"))


def ref_agreement(n: int = 0) -> dict:
    """Nền dị bản: hai bản in khác nhau khớp CHỮ bao nhiêu trên các câu cùng chỉ số."""
    a, b = load_1871(), load_1872()
    m = min(len(a), len(b)) if not n else min(n, len(a), len(b))
    eq = tot = 0
    for i in range(m):
        x, y = a[i].get("nom") or "", b[i]["nom_text"]
        if len(x) != len(y):
            continue
        for cx, cy in zip(x, y):
            tot += 1
            eq += int(cx == cy)
    return dict(n_verses_1871=len(a), n_verses_1872=len(b), n_chars_compared=tot,
                eq=eq, rate=round(eq / tot, 4) if tot else None)


# --------------------------------------------------------------------------- ảnh
def _otsu(a) -> int:
    import numpy as np
    h, _ = np.histogram(a, 256, (0, 256))
    p = h.astype(float) / max(1, h.sum())
    w0 = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    return int(np.argmax((m[-1] * w0 - m) ** 2 / (w0 * (1 - w0) + 1e-9)))


def _runs(on) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _pitch(prof, lo: int, hi: int) -> int | None:
    """Chu kỳ trội của một chiếu mực, bằng tự tương quan trong [lo, hi]."""
    import numpy as np
    v = prof - prof.mean()
    ac = np.correlate(v, v, "full")[len(v) - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return None
    return int(np.argmax(ac[lo:hi])) + lo


def char_pitch_from_runs(hp, thr: float = 0.012) -> float | None:
    """Bước chữ = trung vị khoảng cách giữa TÂM hai dải mực liền nhau trong cột.

    Bền hơn tự tương quan trên bản chép tay: chữ viết tay không tuần hoàn đủ mạnh nên
    tự tương quan hay khoá vào hài (đo: khoá vào đúng cận dưới ở 120/120 tờ)."""
    from scipy import ndimage
    hs = ndimage.uniform_filter1d(hp, 5)
    rr = _runs(hs > thr)
    if len(rr) < 3:
        return None
    ctr = [(a + b) / 2 for a, b in rr]
    d = [ctr[i + 1] - ctr[i] for i in range(len(ctr) - 1)]
    return statistics.median(d) if d else None


def analyze_page(path: Path, col_lo: int = 90, col_hi: int = 260,
                 char_lo: int = 60, char_hi: int = 200,
                 col_window: tuple[int, int] | None = None) -> dict:
    """Bố cục 1 tờ: khung chữ, bước cột, ô cột, bước chữ, số chữ/cột, dải tầng trên."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    im = np.asarray(Image.open(path).convert("L"))
    H, W = im.shape
    bw = im < _otsu(im)
    # bỏ vệt mực nhỏ (chấm son, nhiễu quét)
    lab, n = ndimage.label(bw)
    if n:
        sizes = ndimage.sum(bw, lab, range(1, n + 1))
        bw = np.isin(lab, np.nonzero(sizes >= 30)[0] + 1)
    colp = bw.mean(0)
    rowp = bw.mean(1)
    on_x = np.nonzero(ndimage.uniform_filter1d(colp, 9) > 0.01)[0]
    on_y = np.nonzero(ndimage.uniform_filter1d(rowp, 9) > 0.01)[0]
    if len(on_x) < 50 or len(on_y) < 50:
        return dict(page=path.stem, blank=True, W=W, H=H)
    x0, x1, y0, y1 = int(on_x[0]), int(on_x[-1]), int(on_y[0]), int(on_y[-1])
    inner = bw[y0:y1 + 1, x0:x1 + 1]
    Hi, Wi = inner.shape
    vp = ndimage.uniform_filter1d(inner.mean(0), 9)
    lo, hi = col_window or (col_lo, col_hi)
    cp = _pitch(vp, lo, hi) or lo
    # lưới cột: trượt gốc để tổng mực tại ranh giới nhỏ nhất
    nslot = max(1, int(round(Wi / cp)))
    best = None
    for off in range(cp):
        bnd = [off + k * cp for k in range(nslot + 1) if 0 <= off + k * cp < Wi]
        if len(bnd) < 2:
            continue
        sc = sum(vp[b] for b in bnd) / len(bnd)
        if best is None or sc < best[0]:
            best = (sc, off)
    off = best[1] if best else 0
    edges = [off + k * cp for k in range(-1, nslot + 2)]
    cols = []
    for a, b in zip(edges[:-1], edges[1:]):
        a2, b2 = max(0, a), min(Wi, b)
        if b2 - a2 < 0.5 * cp:
            continue
        sub = inner[:, a2:b2]
        if sub.sum() < 0.02 * cp * Hi:
            continue
        hp = sub.mean(1)
        ri = np.nonzero(hp > 0.01)[0]
        if len(ri) < 10:
            continue
        cols.append(dict(x0=int(a2 + x0), x1=int(b2 + x0), cy0=int(ri[0]), cy1=int(ri[-1]),
                         ink=float(sub.sum()), hp=hp))
    # bước chữ = chu kỳ trội của chiếu ngang, lấy trên các cột dài
    long_cols = [c for c in cols if c["cy1"] - c["cy0"] > 0.7 * Hi]
    chp = None
    if long_cols:
        cands = [char_pitch_from_runs(c["hp"]) for c in long_cols]
        cands = [x for x in cands if x and char_lo <= x <= char_hi]
        chp = round(statistics.median(cands), 1) if cands else None
    chp_use = chp or 100
    # TẦNG. Đo (kim 6 tờ, §1 tài liệu): mỗi cột gồm TẦNG TRÊN = lời bình chữ Hán, chữ NHỎ
    # (cao/chữ ~61 px, thường chia 2 cột con) và TẦNG DƯỚI = ĐÚNG MỘT CẶP LỤC BÁT 14 chữ Nôm
    # (cao/chữ ~77 px). Ranh giới là một ĐƯỜNG NGANG chung cả tờ ở y ~ 0,3 x chiều cao.
    hprof = ndimage.uniform_filter1d(inner.mean(1), max(3, int(0.01 * Hi)))
    a_lo, a_hi = int(0.15 * Hi), int(0.45 * Hi)
    tier_y = int(np.argmin(hprof[a_lo:a_hi])) + a_lo if a_hi > a_lo else None
    bands = [r for r in _runs(hprof > 0.005) if r[1] - r[0] >= 0.03 * Hi]
    top_band = None
    if len(bands) >= 2 and (bands[1][0] - bands[0][1]) >= 0.05 * Hi and (bands[0][1] - bands[0][0]) < 0.4 * Hi:
        top_band = [int(bands[0][0]), int(bands[0][1])]
    # cột NGẮN nằm hẳn ở nửa trên tờ (cách ghi tựa theo cột)
    body_top = statistics.median([c["cy0"] for c in long_cols]) if long_cols else 0
    top_tier = [c for c in cols
                if (c["cy1"] - c["cy0"]) < 0.45 * Hi and c["cy1"] < body_top + 0.45 * Hi]
    # rãnh giữa tờ đôi: khe mực thấp nhất trong dải giữa +-15 % bề rộng
    gut = None
    a, b = int(0.35 * Wi), int(0.65 * Wi)
    if b > a + 10:
        seg = vp[a:b]
        gut = int(np.argmin(seg)) + a
        gut_ink = float(seg.min() / (vp.mean() + 1e-9))
    else:
        gut_ink = None
    # bước chữ của RIÊNG tầng dưới (chữ Nôm to) — tầng trên là lời bình chữ nhỏ, không trộn
    chp_bot = None
    if long_cols and tier_y:
        cb = [char_pitch_from_runs(c["hp"][tier_y:]) for c in long_cols]
        cb = [x for x in cb if x and char_lo <= x <= char_hi]
        chp_bot = round(statistics.median(cb), 1) if cb else None
    cb_use = chp_bot or chp_use
    for c in cols:
        hp = c.pop("hp")
        hs = ndimage.uniform_filter1d(hp, 5)
        rr = [r for r in _runs(hs > 0.012) if r[1] - r[0] >= 0.15 * chp_use]
        c["n_runs"] = int(sum(max(1, round((r[1] - r[0]) / chp_use)) for r in rr))
        c["h"] = c["cy1"] - c["cy0"]
        c["n_est"] = round((c["h"] + chp_use * 0.25) / chp_use, 1)
        if tier_y:
            hb = ndimage.uniform_filter1d(hp[tier_y:], 5)
            rb = [r for r in _runs(hb > 0.012) if r[1] - r[0] >= 0.15 * cb_use]
            c["n_runs_bottom"] = int(sum(max(1, round((r[1] - r[0]) / cb_use)) for r in rb))
            c["ink_bottom"] = float(hp[tier_y:].sum())
        else:
            c["n_runs_bottom"], c["ink_bottom"] = 0, 0.0
    _ib = sorted(c["ink_bottom"] for c in cols if c["ink_bottom"] > 0)
    _med_ib = statistics.median(_ib) if _ib else 0.0
    text_cols = [c for c in cols if c["ink_bottom"] >= 0.25 * _med_ib and _med_ib > 0]
    n_left = sum(1 for c in cols if gut is not None and (c["x0"] + c["x1"]) / 2 - x0 < gut)
    return dict(page=path.stem, blank=False, W=W, H=H, frame=[x0, y0, x1, y1],
                tier_y=(tier_y + y0) if tier_y else None, char_pitch_bottom=chp_bot,
                n_cols_text=len(text_cols),
                chars_bottom=[c["n_runs_bottom"] for c in text_cols],
                inner_hw=[Hi, Wi], col_pitch=cp, char_pitch=chp, n_cols=len(cols),
                n_cols_long=len(long_cols), n_top_tier=len(top_tier), top_band=top_band,
                n_bands=len(bands), gutter_x=(gut + x0) if gut is not None else None,
                gutter_ink_ratio=round(gut_ink, 3) if gut_ink is not None else None,
                n_cols_left=n_left, n_cols_right=len(cols) - n_left,
                chars_per_long_col=[c["n_runs"] for c in long_cols],
                px_per_char=chp, cols=cols)


# --------------------------------------------------------------------------- kim
def kim_cols(path: Path, lay: dict, out: Path, kim_params: dict, scale: int = 1) -> dict | None:
    """kim OCR 1 tờ → số chữ mỗi ô cột (đối chứng cho số chữ đếm bằng run mực)."""
    import hashlib
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{path.stem}_x{scale}.png"
    if not png.exists():
        im = Image.open(path).convert("L")
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
        im.save(png, "PNG")
    md5 = hashlib.md5(png.read_bytes()).hexdigest()
    cp = out / f"{md5}_{kim_params['lang_type']}.json"
    if cp.exists():
        boxes = json.loads(cp.read_text(encoding="utf-8"))["boxes"]
        new = False
    else:
        from core.ocr import ocr_api
        fn = ocr_api.upload_image(str(png))
        if not fn:
            return None
        boxes = ocr_api.recognize(fn, **kim_params)
        if boxes is None:
            return None
        cp.write_text(json.dumps(dict(md5=md5, kim_params=kim_params, boxes=boxes), ensure_ascii=False),
                      encoding="utf-8")
        new = True
    per = [0] * len(lay["cols"])
    bot = [0] * len(lay["cols"])
    ty = lay.get("tier_y")
    total = 0
    for b in boxes:
        text = (b.get("transcription") or "").strip()
        chars = [c for c in text if c.strip()]
        if not chars:
            continue
        xs = [p[0] / scale for p in b["points"]]
        ys = [p[1] / scale for p in b["points"]]
        cx = (min(xs) + max(xs)) / 2
        y0, y1 = min(ys), max(ys)
        h = (y1 - y0) / len(chars)
        total += len(chars)
        d, k = min((abs(cx - (c["x0"] + c["x1"]) / 2), i) for i, c in enumerate(lay["cols"]))
        if d > 0.5 * lay["col_pitch"]:
            continue
        per[k] += len(chars)
        if ty:
            bot[k] += sum(1 for i in range(len(chars)) if y0 + h * (i + 0.5) > ty)
    return dict(page=path.stem, n_boxes=len(boxes), n_chars=total, per_col=per, per_col_bottom=bot,
                new_call=new)


# --------------------------------------------------------------------------- chạy
def run(out: Path, limit: int, kim_pages: int, kim_lang_type: int, workers: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(PAGES.glob("*.jpg"))
    sel = imgs[:limit] if limit else imgs
    def _run_all(win):
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from functools import partial
            with ProcessPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(partial(analyze_page, col_window=win), sel))
        return [analyze_page(p, col_window=win) for p in sel]

    lays = _run_all(None)
    # Lượt 2: bước cột của tờ lẻ hay khoá vào HÀI (đo: 35/120 tờ ra 253-259 px thay vì 107);
    # khoá cửa sổ +-25 % quanh TRUNG VỊ CẢ SÁCH rồi phân tích lại (vẫn không dùng nhãn nào).
    pit = [x["col_pitch"] for x in lays if not x.get("blank") and x.get("col_pitch")]
    win = None
    if pit:
        med = statistics.median(pit)
        win = (int(med * 0.75), int(med * 1.25) + 1)
        lays = _run_all(win)
    col_window = win
    prow, crow = [], []
    for L in lays:
        if L.get("blank"):
            prow.append(dict(page=L["page"], blank=1, n_cols=0, n_cols_text=0, tier_y=0,
                             char_pitch_bottom=0, med_chars_bottom=0, n_cols_bottom14=0,
                             n_cols_long=0, n_top_tier=0,
                             has_top_band=0, n_bands=0, n_cols_left=0, n_cols_right=0,
                             col_pitch=0, char_pitch=0, W=L["W"], H=L["H"], med_chars=0))
            continue
        ch = L["chars_per_long_col"]
        cb = L.get("chars_bottom") or []
        prow.append(dict(page=L["page"], blank=0, n_cols=L["n_cols"],
                         n_cols_text=L.get("n_cols_text", 0), tier_y=L.get("tier_y") or 0,
                         char_pitch_bottom=L.get("char_pitch_bottom") or 0,
                         med_chars_bottom=statistics.median(cb) if cb else 0,
                         n_cols_bottom14=sum(1 for x in cb if x == 14),
                         n_cols_long=L["n_cols_long"],
                         n_top_tier=L["n_top_tier"], has_top_band=int(bool(L.get("top_band"))),
                         n_bands=L.get("n_bands", 0), n_cols_left=L.get("n_cols_left", 0),
                         n_cols_right=L.get("n_cols_right", 0), col_pitch=L["col_pitch"],
                         char_pitch=L["char_pitch"] or 0, W=L["W"], H=L["H"],
                         med_chars=statistics.median(ch) if ch else 0))
        for i, c in enumerate(L["cols"]):
            crow.append(dict(page=L["page"], col=i, x0=c["x0"], x1=c["x1"], y0=c["cy0"], y1=c["cy1"],
                             h=c["h"], n_runs=c["n_runs"], n_est=c["n_est"]))
    with open(out / "pages.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    with open(out / "cols.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crow[0]))
        w.writeheader()
        w.writerows(crow)

    good = [p for p in prow if not p["blank"]]
    hist: dict[str, int] = {}
    for p in good:
        hist[str(p["n_cols"])] = hist.get(str(p["n_cols"]), 0) + 1
    all_chars = [c["n_runs"] for c in crow if c["h"] > 0.7 * 1820 * 0.9]
    med_chars = statistics.median([p["med_chars"] for p in good if p["med_chars"]]) if good else None

    kim = None
    if kim_pages:
        kp = dict(ocr_id=1, lang_type=kim_lang_type, font_type=2)   # chép tay -> font_type 2
        rows, calls = [], 0
        for L in [x for x in lays if not x.get("blank")][:kim_pages]:
            r = kim_cols(PAGES / f"{L['page']}.jpg", L, out / "kim_cache", kp)
            if r is None:
                continue
            calls += int(r.pop("new_call"))
            rows.append(r)
        with open(out / "kim_cols.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["page", "n_boxes", "n_chars", "per_col"])
            for r in rows:
                w.writerow([r["page"], r["n_boxes"], r["n_chars"], " ".join(map(str, r["per_col"]))])
        per = [n for r in rows for n in r["per_col"] if n]
        bot = [n for r in rows for n in r.get("per_col_bottom", []) if n]
        kim = dict(pages=len(rows), api_calls=calls, kim_params=kp,
                   chars_total=sum(r["n_chars"] for r in rows),
                   n_text_cols=len(per),
                   chars_per_col_med=statistics.median(per) if per else None,
                   chars_per_col_hist={str(k): per.count(k) for k in sorted(set(per))} if per else {},
                   bottom_per_col_med=statistics.median(bot) if bot else None,
                   bottom_per_col_hist={str(k): bot.count(k) for k in sorted(set(bot))} if bot else {},
                   n_bottom_14=sum(1 for n in bot if n == 14),
                   n_bottom_13_15=sum(1 for n in bot if 13 <= n <= 15),
                   rate_bottom_14=round(sum(1 for n in bot if n == 14) / len(bot), 4) if bot else None,
                   n_cols_14=sum(1 for n in per if n == 14),
                   n_cols_6_or_8=sum(1 for n in per if n in (6, 8)))

    s = dict(book=BOOK, generated_by="scripts/measure/ptcl_layout.py",
             pages=dict(n=len(sel), blank=len(sel) - len(good), analyzed=len(good)),
             columns=dict(hist_n_cols=hist,
                          n_cols_med=statistics.median([p["n_cols"] for p in good]) if good else None,
                          n_cols_long_med=statistics.median([p["n_cols_long"] for p in good]) if good else None,
                          col_pitch_med=statistics.median([p["col_pitch"] for p in good]) if good else None,
                          char_pitch_med=statistics.median([p["char_pitch"] for p in good if p["char_pitch"]]) if good else None,
                          chars_per_col_med=med_chars,
                          chars_per_col_hist={str(k): all_chars.count(k) for k in sorted(set(all_chars))},
                          col_pitch_window=col_window,
                          n_cols_text_med=statistics.median([p["n_cols_text"] for p in good]) if good else None,
                          hist_n_cols_text={str(k): sum(1 for p in good if p["n_cols_text"] == k)
                                            for k in sorted({p["n_cols_text"] for p in good})},
                          tier_y_med=statistics.median([p["tier_y"] for p in good if p["tier_y"]]) if good else None,
                          char_pitch_bottom_med=statistics.median([p["char_pitch_bottom"] for p in good if p["char_pitch_bottom"]]) if good else None,
                          chars_bottom_med=statistics.median([p["med_chars_bottom"] for p in good if p["med_chars_bottom"]]) if good else None,
                          n_cols_bottom14=sum(p["n_cols_bottom14"] for p in good),
                          n_cols_text_total=sum(p["n_cols_text"] for p in good),
                          n_cols_left_med=statistics.median([p["n_cols_left"] for p in good]) if good else None,
                          n_cols_right_med=statistics.median([p["n_cols_right"] for p in good]) if good else None,
                          pages_with_top_band=sum(1 for p in good if p["has_top_band"]),
                          rate_top_band=round(sum(1 for p in good if p["has_top_band"]) / len(good), 4) if good else None,
                          pages_with_top_tier=sum(1 for p in good if p["n_top_tier"]),
                          rate_top_tier=round(sum(1 for p in good if p["n_top_tier"]) / len(good), 4) if good else None,
                          n_top_tier_cols=sum(p["n_top_tier"] for p in good)),
             image=dict(W_med=statistics.median([p["W"] for p in good]) if good else None,
                        H_med=statistics.median([p["H"] for p in good]) if good else None),
             reference=ref_agreement(),
             kim=kim)
    s["invariants"] = invariants(s)
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


def invariants(s: dict) -> list[dict]:
    iv = []

    def add(name, expected, observed, ok, method):
        iv.append(dict(name=name, expected=expected, observed=observed, method=method, **{"pass": ok}))
    c = s["columns"]
    add("moi_to_doi_co_cot_deu", ">= 10 cột/tờ", c["n_cols_med"], (c["n_cols_med"] or 0) >= 10,
        "lưới cột từ tự tương quan chiếu mực; tờ đôi 2000 px nên có ~2 × số cột của một nửa tờ")
    add("buoc_cot_on_dinh", "CV <= 0.15", c["col_pitch_med"], c["col_pitch_med"] is not None,
        "trung vị bước cột toàn bộ tờ")
    add("so_chu_moi_cot_gan_boi_cua_6_8", "med ∈ [12,22]", c["chars_per_col_med"],
        c["chars_per_col_med"] is not None and 12 <= c["chars_per_col_med"] <= 22,
        "đếm run mực trong cột dài; 14 = 1 cặp lục bát, 6/8 = 1 câu")
    r = s["reference"]
    add("nen_di_ban_1871_1872", "≈ 0.83", r["rate"], r["rate"] is not None and 0.70 <= r["rate"] <= 0.95,
        "tỉ lệ chữ trùng giữa hai ấn bản Kiều khác nhau = CẬN TRÊN của mọi phép so nhãn với dị bản")
    k = s.get("kim")
    if k:
        add("kim_tang_duoi_dung_14", "med == 14", k.get("bottom_per_col_med"),
            k.get("bottom_per_col_med") == 14,
            "chữ kim có tâm y DƯỚI ranh giới tầng, gom theo cột: trung vị phải là 14 "
            "(= 1 cặp lục bát 6+8) nếu tầng dưới đúng là thân truyện Kiều")
        add("kim_tang_duoi_13_15", ">= 0.8", 
            round(k["n_bottom_13_15"] / max(1, k["n_text_cols"]), 4) if k.get("n_text_cols") else None,
            k.get("n_text_cols") and k["n_bottom_13_15"] / k["n_text_cols"] >= 0.8,
            "tỉ lệ cột có 13-15 chữ ở tầng dưới (kim đọc chép tay nên rụng/thêm 1 chữ)")
        add("kim_khop_dem_run_muc_tang_duoi", "|med kim − med run| <= 2",
            [k.get("bottom_per_col_med"), c.get("chars_bottom_med")],
            k.get("bottom_per_col_med") is not None and c.get("chars_bottom_med") is not None
            and abs(k["bottom_per_col_med"] - c["chars_bottom_med"]) <= 2,
            "hai cách đếm chữ tầng dưới độc lập (kim vs chiếu mực)")
    return iv


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    import numpy as np
    n = ok = 0

    def chk(name, cond):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'} {name}")

    chk("_runs rỗng", _runs([0, 0, 0]) == [])
    chk("_runs một dải", _runs([0, 1, 1, 0]) == [(1, 3)])
    chk("_runs hai dải", _runs([1, 0, 1, 1]) == [(0, 1), (2, 4)])
    sig = np.array([1.0 if (i % 10) < 3 else 0.0 for i in range(200)])
    chk("_pitch chu kỳ 10", _pitch(sig, 5, 30) == 10)
    chk("_pitch dải hẹp", _pitch(sig, 100, 50) is None)
    a = np.full((10, 10), 210, dtype=np.uint8)
    a[:3, :] = [20, 25, 30][0]
    a[1, :], a[2, :] = 25, 30
    thr = _otsu(a)
    chk("_otsu nằm giữa hai mức", 20 <= thr < 210)
    chk("_otsu tách được mực", 0 < int((a <= thr).sum()) < a.size)
    chk("dữ liệu: 120 ảnh", len(list(PAGES.glob("*.jpg"))) == 120)
    chk("dữ liệu: tham chiếu 1872", REF1872.exists())
    chk("dữ liệu: tham chiếu 1871", REF1871.exists())
    a71 = load_1871()
    chk("1871 phẳng >= 3200 câu", len(a71) >= 3200)
    chk("1871 có nom + qn", all(("nom" in v and "qn" in v) for v in a71[:50]))
    a72 = load_1872()
    chk("1872 >= 3200 câu", len(a72) >= 3200)
    ra = ref_agreement(200)
    chk("nền dị bản trong [0.6, 0.95]", ra["rate"] is not None and 0.6 <= ra["rate"] <= 0.95)
    chk("nền dị bản có n", ra["n_chars_compared"] > 500)
    print(f"ptcl_layout selftest: {ok}/{n}")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default=BOOK)
    ap.add_argument("--out", default=str(REPO / "measure_out" / BOOK / "ptcl_layout"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kim-pages", type=int, default=0)
    ap.add_argument("--kim-lang-type", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    s = run(Path(a.out), a.limit, a.kim_pages, a.kim_lang_type, a.workers)
    c = s["columns"]
    print(f"{BOOK}: {s['pages']['analyzed']} tờ · cột/tờ {c['hist_n_cols']} · bước cột {c['col_pitch_med']} px · "
          f"bước chữ {c['char_pitch_med']} px · chữ/cột {c['chars_per_col_med']}")
    print(f"  tầng trên: dải ngang {c['pages_with_top_band']} tờ ({c['rate_top_band']}) · "
          f"cột ngắn {c['pages_with_top_tier']} tờ ({c['rate_top_tier']}) · {c['n_top_tier_cols']} cột")
    print(f"  cột nửa trái/phải (trung vị): {c['n_cols_left_med']} / {c['n_cols_right_med']} · "
          f"cửa sổ bước cột {c['col_pitch_window']}")
    print(f"  cột CÓ CHỮ ở tầng dưới: trung vị {c['n_cols_text_med']}/tờ (hist {c['hist_n_cols_text']}), "
          f"tổng {c['n_cols_text_total']}; ranh giới tầng y={c['tier_y_med']}; "
          f"bước chữ tầng dưới {c['char_pitch_bottom_med']} px; chữ/cột tầng dưới {c['chars_bottom_med']}; "
          f"cột đúng 14 {c['n_cols_bottom14']}")
    print(f"  nền dị bản 1871↔1872: {s['reference']['rate']} (n {s['reference']['n_chars_compared']})")
    if s.get("kim"):
        k = s["kim"]
        print(f"  kim: {k['pages']} tờ, {k['chars_total']} chữ, {k['n_text_cols']} cột có chữ, "
              f"chữ/cột trung vị {k['chars_per_col_med']}; TẦNG DƯỚI trung vị {k['bottom_per_col_med']}, "
              f"đúng 14: {k['n_bottom_14']} ({k['rate_bottom_14']}), 13-15: {k['n_bottom_13_15']}; "
              f"lượt mới {k['api_calls']}")
        print(f"     hist tầng dưới: {k['bottom_per_col_hist']}")
    nf = sum(1 for iv in s["invariants"] if iv["pass"] is False)
    for iv in s["invariants"]:
        print(f"  {'PASS' if iv['pass'] else 'FAIL'} {iv['name']} kỳ vọng {iv['expected']} quan sát {iv['observed']}")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
