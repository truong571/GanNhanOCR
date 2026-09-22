#!/usr/bin/env python
"""Phép đo: Chrestomathie cochinchinoise 1872 — bản đồ 20 truyện QN ↔ 65 trang Nôm.

Gộp scripts/measure_wf_2026-09-21/chresto/{qn_ocr_tsv,qn_stories,nom_layout2,nom_segments,
run_nomna,match_stories,match_window}.py + chresto_review/* thành 1 lệnh, không đường dẫn cứng.

(a) QN: tesseract -l vie --psm 4 (TSV, cache ở --out/ocr_qn_tsv) trên canvas 29–56 →
    dòng có toạ độ → trang tiếng Việt (tỉ lệ âm tiết trong lexicon ≥ 0,5) → tiêu đề truyện
    bằng 2 cách: (1) neo tham chiếu (20 vị trí đã kiểm bằng mắt 2026-09-21, cờ cần người rà)
    (2) tự phát hiện theo bố cục (dòng ngắn, căn giữa thân, kề trước dòng thân dài, có
    thể có số La Mã ngay trên) → agreement. Bảng 20 truyện: title, canvas, n_lines, n_syll,
    opening, OOV so với dict/QuocNgu_SinoNom.csv.
(b) Nôm: 65 trang canvas 106–170: khung (mở hình thái 160px), pitch cột (tự tương quan),
    lưới 7 ô, số chữ/cột bằng 2 cách (pitch chữ tự tương quan vs run mực có tách run dài)
    → agreement; ô trống nội bộ, cột ngắn, dấu khuyên.
(c) Bảng truyện ↔ trang Nôm: tái dùng bảng tham chiếu (agent dựng 2026-09-21, cờ cần
    người rà), tính lại số cột/chữ từ (b); tự phát hiện ranh giới (ô trống nội bộ hoặc đầu
    trang sau trang kết cột ngắn/thiếu cột) → precision/recall so với bảng.
(d) NomNaOCR (tuỳ chọn, chỉ khi --tf-env tồn tại): cắt ~5 chữ/đoạn 3 trang đầu, nhận dạng
    2 bộ trọng số trong tiến trình con (log TF tắt), LCS cửa sổ 150 âm tiết với 20 truyện
    qua từ điển → hạng của truyện kỳ vọng, đối chứng ngẫu nhiên.

Chạy:  .venv/bin/python scripts/measure/chresto_map.py
Thử:   .venv/bin/python scripts/measure/chresto_map.py --limit 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata as ud
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BOOK = "Chrestomathie1872"
DATA = REPO / "data" / BOOK
QN_DIR, NOM_DIR = DATA / "quocngu_pages", DATA / "nom_pages"
DICT = REPO / "dict" / "QuocNgu_SinoNom.csv"
NOMNA = REPO / "NomNaOCR"
TF_ENV_DEFAULT = Path(os.environ.get("GN_TF_ENV", str(REPO / "tf_env")))  # R-05: không ghim scratchpad phiên; đặt GN_TF_ENV nếu venv TF ở nơi khác
QN_PAGES = list(range(29, 57))
NOM_PAGES = list(range(106, 171))

# Tham chiếu do agent đọc ảnh 2026-09-21 (KHÔNG phải người kiểm) — mọi dòng mang cờ can_nguoi_ra.
# (số, canvas QN, y dòng số La Mã, tiêu đề, canvas Nôm đầu, ô cột đầu, kiểu ranh giới)
REF = [
    (1, 29, 720, "Con gái cầu chồng đại vương", 106, 1, "đầu trang"),
    (2, 30, 1456, "Thằng cày, con trâu lại với con cọp", 110, 0, "đầu trang"),
    (3, 32, 300, "Ông huyện thanh liêm", 114, 0, "đầu trang"),
    (4, 33, 625, "Con chồn cáo với con cọp", 117, 0, "đầu trang"),
    (5, 34, 919, "Ông thầy già nghe nghĩa mà ăn khín bánh của học trò", 120, 2, "sau ô cột trống"),
    (6, 35, 695, "Con Thỏ cứu cá, mà ra khỏi nơm", 122, 5, "sau ô cột trống"),
    (7, 36, 690, "Con thỏ cứu con voi mà nhát cọp", 125, 3, "sau ô cột trống"),
    (8, 37, 1070, "Thầy trừ chồn cáo", 129, 0, "đầu trang"),
    (9, 38, 518, "Thầy dạy ăn trộm thử học trò", 131, 0, "đầu trang"),
    (10, 40, 825, "Thằng đi làm rể", 137, 0, "đầu trang"),
    (11, 41, 869, "Học phép hà tiện", 139, 4, "sau ô cột trống"),
    (12, 42, 260, "Láo dinh Láo quê", 141, 0, "đầu trang"),
    (13, 42, 1619, "Anh học trò sửa liễn triều đình đặt", 143, 0, "đầu trang"),
    (14, 43, 1658, "Thầy bói: «Bụng làm, Dạ chịu.»", 146, 0, "đầu trang"),
    (15, 45, 250, "Đại-trượng-phu với Quân-tử", 149, 0, "đầu trang"),
    (16, 47, 542, "Bạn học người đậu người không", 155, 0, "đầu trang"),
    (17, 48, 290, "Hang Từ-thức", 157, 0, "đầu trang"),
    (18, 49, 1400, "Ông huyện xử tội con ruồi", 161, 4, "sau ô cột trống"),
    (19, 50, 559, "Con beo nhờ ông già cứu, rồi đòi ăn thịt ông già đi", 163, 0, "đầu trang"),
    (20, 52, 310, "Con chồn với con cọp", 167, 5, "sau ô cột trống"),
]
TONES = {"̀": "2", "́": "1", "̃": "3", "̉": "4", "̣": "5"}
WORD_RE = re.compile(r"[A-Za-zÀ-ỹ]+")


# ------------------------------------------------------------------------- QN OCR
def canon(s):
    s = ud.normalize("NFD", s.lower()); t = "0"; out = []
    for ch in s:
        if ch in TONES:
            t = TONES[ch]
        else:
            out.append(ch)
    return ud.normalize("NFC", "".join(out)) + t


def load_dict():
    """rev: chữ Nôm → tập âm QN (canon); lex: tập âm QN (NFC thường, không tách thanh)."""
    rev, lex = defaultdict(set), set()
    with open(DICT, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            q, h = row["QuocNgu"].strip(), row["SinoNom"].strip()
            if q and h:
                rev[h].add(canon(q)); lex.add(ud.normalize("NFC", q.lower()))
    return rev, lex


def tesseract_page(page, tsv_dir):
    out = tsv_dir / f"canvas_{page:04d}"
    if not (out.with_suffix(".tsv")).exists():
        subprocess.run(["tesseract", str(QN_DIR / f"canvas_{page:04d}.jpg"), str(out), "-l", "vie", "--psm", "4", "tsv"],
                       check=True, capture_output=True)
    rows = list(csv.DictReader(open(out.with_suffix(".tsv"), encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    lines = defaultdict(list)
    for r in rows:
        if int(r["level"]) == 5 and r["text"].strip():
            lines[(int(r["block_num"]), int(r["par_num"]), int(r["line_num"]))].append(r)
    res = []
    for key in sorted(lines):
        ws = lines[key]
        x0 = min(int(w["left"]) for w in ws); x1 = max(int(w["left"]) + int(w["width"]) for w in ws)
        y0 = min(int(w["top"]) for w in ws); y1 = max(int(w["top"]) + int(w["height"]) for w in ws)
        conf = [float(w["conf"]) for w in ws if float(w["conf"]) >= 0]
        res.append({"page": page, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "w": x1 - x0, "cx": (x0 + x1) / 2,
                    "text": " ".join(w["text"] for w in ws), "conf": round(sum(conf) / len(conf), 1) if conf else -1})
    return res


def qn_ocr(pages, tsv_dir, workers):
    tsv_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        per = list(ex.map(lambda p: tesseract_page(p, tsv_dir), pages))
    return [l for ls in per for l in ls]


def page_lexicon_rate(lines, lex):
    ws = [ud.normalize("NFC", w.lower()) for l in lines for w in WORD_RE.findall(l["text"])]
    return (sum(1 for w in ws if w in lex) / len(ws) if ws else 0.0), len(ws)


ROMAN_RE = re.compile(r"^[IVXLivxl1lI|Ị.,:;·\s]{1,8}$")


def detect_titles_auto(lines_page, W=1500):
    """Tiêu đề = dòng ngắn (w < 0,62·thân), tâm gần tâm dòng thân (±120px), y ≥ 200 (bỏ tiêu đề
    chạy), dòng kế tiếp trong 200px là dòng thân dài, không mở bằng « — (đối thoại), có ≥ 2 từ.
    Số La Mã (≤ 8 ký tự dạng IVXL/lỗi OCR) ngay trên (≤ 260px) là bằng chứng phụ."""
    ls = sorted(lines_page, key=lambda l: l["y0"])
    body = [l for l in ls if l["w"] > 850 and l["y0"] >= 200]
    if not body:
        return []
    bw = float(np.median([l["w"] for l in body])); bcx = float(np.median([l["cx"] for l in body]))
    out = []
    for i, l in enumerate(ls):
        if l["y0"] < 200 or l["w"] >= 0.62 * bw or abs(l["cx"] - bcx) > 120:
            continue
        txt = l["text"].strip()
        if txt[:1] in "«—-_" and not txt.lstrip("«—-_ .").strip():
            continue
        words = WORD_RE.findall(txt)
        if len(words) < 2:
            continue
        nxt = next((m for m in ls[i + 1:] if m["y0"] > l["y0"]), None)
        if not nxt or nxt["y0"] - l["y0"] > 200 or nxt["w"] < 0.8 * bw:
            continue
        # dòng ngay trước: nếu là dòng thân dài mà không kết câu → tiêu đề giả (ngắt đoạn); chấp nhận nếu kết bằng dấu câu
        roman = any(ROMAN_RE.match(m["text"].strip()) and 0 < l["y0"] - m["y0"] <= 260 for m in ls[max(0, i - 3):i])
        out.append({"page": l["page"], "y": l["y0"], "text": txt, "roman_above": roman, "w": l["w"]})
    return out


def build_stories(lines, anchors, lex, vn_pages):
    """anchors: [(so, page, y_roman, title)] → 20 truyện với dòng thân (w>600, y≥200, trang VN)."""
    def is_vn_line(l):                       # loại cước chú Pháp (tr.29 …): dòng ≥ 5 từ mà < 0,4 trong lexicon
        ws = [ud.normalize("NFC", w.lower()) for w in WORD_RE.findall(l["text"])]
        return len(ws) < 5 or sum(1 for w in ws if w in lex) / len(ws) >= 0.4
    body = sorted((l for l in lines if l["w"] > 600 and l["y0"] >= 200 and l["page"] in vn_pages and is_vn_line(l)),
                  key=lambda l: (l["page"], l["y0"]))
    stories = []
    for i, (n, p, y, t) in enumerate(anchors):
        nxt = anchors[i + 1] if i + 1 < len(anchors) else (None, 10 ** 6, 0, None)
        ls = [l for l in body if (p, y + 60) < (l["page"], l["y0"]) < (nxt[1], nxt[2])]
        if ls and ls[0]["w"] < 750 and ls[0]["y0"] < y + 200 and any(w in ls[0]["text"].lower() for w in t.lower().split()[:2]):
            ls = ls[1:]                                      # dòng tiêu đề dài lọt vào thân
        txt = " ".join(l["text"] for l in ls)
        words = [ud.normalize("NFC", w.lower()) for w in WORD_RE.findall(txt)]
        oov = sum(1 for w in words if w not in lex)
        stories.append({"so": n, "tieu_de": t, "qn_canvas": p, "qn_y": y, "qn_trang": sorted({l["page"] for l in ls}),
                        "n_lines": len(ls), "n_syll": len(words), "oov": oov,
                        "oov_rate": round(oov / len(words), 3) if words else None,
                        "conf_med": round(float(np.median([l["conf"] for l in ls])), 1) if ls else None,
                        "opening": " ".join(l["text"] for l in ls[:2])[:160], "syl_canon": [canon(w) for w in words]})
    return stories


# ------------------------------------------------------------------------ Nôm layout
def otsu_thr(a):
    h, _ = np.histogram(a, 256, (0, 256)); p = h.astype(float) / h.sum()
    w0 = np.cumsum(p); m = np.cumsum(p * np.arange(256)); mt = m[-1]
    return int(np.argmax((mt * w0 - m) ** 2 / (w0 * (1 - w0) + 1e-9)))


def groups(idx, gap=8):
    if len(idx) == 0:
        return []
    g = [[idx[0]]]
    for i in idx[1:]:
        (g[-1].append(i) if i - g[-1][-1] <= gap else g.append([i]))
    return g


def runs(on, minlen=1):
    r, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i >= minlen:
                r.append((i, j))
            i = j
        else:
            i += 1
    return r


def count_runs(hp, cpitch):
    """Đếm chữ bằng run mực: mượt 5px, ngưỡng 0,012, gộp khe < 5px, bỏ run < 0,15·pitch,
    run dài (chữ dính) tính round(len/pitch)."""
    from scipy import ndimage
    hs = ndimage.uniform_filter1d(hp, 5)
    mr = []
    for r in runs(hs > 0.012):
        if mr and r[0] - mr[-1][1] < 5:
            mr[-1] = (mr[-1][0], r[1])
        else:
            mr.append(r)
    return int(sum(max(1, round((r[1] - r[0]) / cpitch)) for r in mr if r[1] - r[0] >= 0.15 * cpitch))


def analyze_nom(page):
    """Bố cục 1 trang Nôm (thuật toán nom_layout2.py)."""
    from PIL import Image
    from scipy import ndimage
    path = NOM_DIR / f"canvas_{page:04d}.jpg"
    im = np.asarray(Image.open(path).convert("L"))
    bw = im < otsu_thr(im)
    H, W = bw.shape
    bw3 = ndimage.binary_dilation(bw, structure=np.ones((3, 3)))
    lv = ndimage.binary_opening(bw3, structure=np.ones((160, 1)))
    lh = ndimage.binary_opening(bw3, structure=np.ones((1, 160)))
    vg, hg = groups(np.nonzero(lv.sum(0) > 100)[0]), groups(np.nonzero(lh.sum(1) > 100)[0])
    lefts = [g for g in vg if g[-1] < W / 2]; rights = [g for g in vg if g[0] > W / 2]
    tops = [g for g in hg if g[-1] < H / 2]; bots = [g for g in hg if g[0] > H / 2]
    frame_ok = bool(lefts and rights and tops and bots)
    text = bw & ~ndimage.binary_dilation(lv | lh, structure=np.ones((7, 7)))
    lab, n = ndimage.label(text)
    if n:
        sizes = ndimage.sum(text, lab, range(1, n + 1))
        text = np.isin(lab, np.nonzero(sizes >= 15)[0] + 1)
    if frame_ok:
        top, bot, left, right = tops[-1][-1] + 6, bots[0][0] - 6, lefts[-1][-1] + 6, rights[0][0] - 6
    else:
        ys, xs = np.nonzero(text)
        if len(ys) < 100:
            return {"page": page, "frame_ok": False, "blank": True, "n_cols": 0, "cols": []}
        top, bot, left, right = (int(np.percentile(ys, 0.5)), int(np.percentile(ys, 99.5)),
                                 int(np.percentile(xs, 0.5)), int(np.percentile(xs, 99.5)))
    inner = text[top:bot, left:right]
    Hi, Wi = inner.shape
    vps = ndimage.uniform_filter1d(inner.mean(0), 15)
    v = vps - vps.mean(); ac = np.correlate(v, v, "full")[len(v) - 1:]; ac = ac / (ac[0] + 1e-9)
    pitch = int(np.argmax(ac[100:180])) + 100
    nslot = int(round(Wi / pitch))
    best = None
    for off in range(pitch):
        b = [off + k * pitch for k in range(nslot + 1) if off + k * pitch < Wi]
        sc = sum(vps[min(x, Wi - 1)] for x in b) / len(b)
        if best is None or sc < best[0]:
            best = (sc, off)
    off = best[1]
    bounds = [b for b in [max(0, off - pitch)] + [off + k * pitch for k in range(nslot + 1)] if -pitch // 2 < b < Wi + pitch // 2]
    cols, narrow = [], 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        a2, b2 = max(0, a), min(Wi, b)
        if b2 - a2 < pitch * 0.5:
            continue
        sub = inner[:, a2:b2]
        ink = int(sub.sum())
        if ink < 400:
            continue
        hp = sub.mean(1)
        ri = np.nonzero(hp > 0.01)[0]; y0, y1 = int(ri[0]), int(ri[-1])
        cp = sub.mean(0); cx = np.nonzero(cp > 0.005)[0]; x0, x1 = int(cx[0]), int(cx[-1])
        if x1 - x0 < 0.3 * pitch:                  # vệt mực hẹp (dư đường khung / dấu) ≠ cột chữ
            narrow += 1; continue
        cols.append({"slot": int(round(a2 / pitch)), "x0": a2 + x0, "x1": a2 + x1, "slot_x0": int(a2), "slot_x1": int(b2),
                     "y0": y0, "y1": y1, "ink_h": y1 - y0, "ink": ink, "hp": hp})
    cpitch = None
    longcols = [c for c in cols if c["ink_h"] > 0.8 * Hi]
    if longcols:
        hpm = np.mean([c["hp"] for c in longcols], axis=0); h = hpm - hpm.mean()
        ac2 = np.correlate(h, h, "full")[len(h) - 1:]; ac2 = ac2 / (ac2[0] + 1e-9)
        cpitch = int(np.argmax(ac2[70:120])) + 70
    cp_use = cpitch or 90
    for c in cols:
        hp = c.pop("hp")
        c["n_runs"] = count_runs(hp, cp_use)
        c["n_est"] = round((c["ink_h"] + cp_use * 0.25) / cp_use, 1)
        c["short"] = bool(c["ink_h"] < 0.8 * Hi)
        c["indent"] = bool(c["y0"] > 60)
        sub = inner[:, c["slot_x0"]:c["slot_x1"]]
        lab2, n2 = ndimage.label(sub); circ = 0
        if n2:
            for k2, sl in enumerate(ndimage.find_objects(lab2)):
                hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
                if 9 <= hh <= 30 and 9 <= ww <= 30 and abs(hh - ww) <= 7:
                    m = lab2[sl] == k2 + 1
                    if ndimage.binary_fill_holes(m).sum() > m.sum() * 1.25:
                        circ += 1
        c["circles"] = circ
    slots = {c["slot"] for c in cols}
    empty_internal = sorted(s for s in range(min(slots), max(slots) + 1) if s not in slots) if slots else []
    return {"page": page, "frame_ok": frame_ok, "blank": False, "frame": [int(top), int(bot), int(left), int(right)],
            "inner_hw": [int(Hi), int(Wi)], "col_pitch": pitch, "nslot": nslot, "char_pitch": cpitch,
            "n_cols": len(cols), "n_narrow_skipped": narrow, "empty_internal_slots": empty_internal, "cols": cols}


def boundaries_auto(layouts):
    """Ứng viên ranh giới truyện: (trang, ô) = ô đầu tiên sau ô trống nội bộ; hoặc ô đầu trang khi
    trang trước kết bằng cột ngắn hoặc THIẾU CỘT Ở CUỐI (ô cuối cùng có chữ < nslot−1).
    Không dùng n_cols < nslot: ô trống NỘI BỘ (ranh giới giữa trang) cũng làm n_cols < nslot và
    sinh 6 ứng viên giả ở đầu trang kế (121/123/126/140/162/168, đo 21/09)."""
    cands = []
    prev = None
    for L in layouts:
        if L["blank"] or not L["cols"]:
            prev = L; continue
        cols = sorted(L["cols"], key=lambda c: c["slot"])
        if prev is not None and not prev["blank"] and prev["cols"]:
            pc = sorted(prev["cols"], key=lambda c: c["slot"])
            if pc[-1]["short"] or pc[-1]["slot"] < prev["nslot"] - 1:
                cands.append((L["page"], cols[0]["slot"], "đầu trang"))
        for s in L["empty_internal_slots"]:
            nxt = next((c for c in cols if c["slot"] > s), None)
            if nxt:
                cands.append((L["page"], nxt["slot"], "sau ô cột trống"))
        prev = L
    return sorted(set(cands))


def story_page_table(layouts, stories):
    """Gắn cột Nôm cho mỗi truyện theo REF (trang, ô đầu) → tính lại số cột/chữ."""
    by_page = {L["page"]: L for L in layouts}
    starts = [(r[4], r[5]) for r in REF]
    rows = []
    for i, r in enumerate(REF):
        so, _, _, title, p0, s0, kind = r
        end = starts[i + 1] if i + 1 < len(starts) else (10 ** 6, 0)
        n_cols = 0; n_est = 0.0; n_runs = 0; pages = set(); missing = False
        for p in range(p0, min(end[0], max(by_page) if by_page else p0) + 1):
            L = by_page.get(p)
            if L is None:
                missing = True; continue
            for c in L["cols"]:
                if (p, c["slot"]) >= (p0, s0) and (p, c["slot"]) < end:
                    n_cols += 1; n_est += c["n_est"]; n_runs += c["n_runs"]; pages.add(p)
        st = next((s for s in stories if s["so"] == so), None)
        rows.append({"so": so, "tieu_de": title, "qn_canvas": r[1], "qn_n_lines": st["n_lines"] if st else None,
                     "qn_n_syll": st["n_syll"] if st else None, "nom_canvas_dau": p0, "nom_slot_dau": s0,
                     "nom_trang": f"{min(pages)}-{max(pages)}" if pages else None, "nom_n_cols": n_cols,
                     "nom_chu_uoc_pitch": round(n_est, 1), "nom_chu_uoc_runs": n_runs,
                     "ti_le_chu_am": round(n_est / st["n_syll"], 3) if st and st["n_syll"] and n_cols else None,
                     "ranh_gioi": kind, "can_nguoi_ra": True, "layout_thieu_trang": missing})
    return rows


# ---------------------------------------------------------------------- NomNaOCR
NOMNA_RUNNER = r'''
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging; logging.disable(logging.WARNING)
nomna, segdir, meta_path, out_path = sys.argv[1:5]
sys.path.insert(0, nomna)
from nomnaocr_rec import NomNaRecognizer
from PIL import Image
meta = json.load(open(meta_path))
imgs = [Image.open(os.path.join(segdir, m["file"])) for m in meta]
res = {}
for name, w, v in [("finetuned", f"{nomna}/weights/finetuned_CRNNxCTC.h5", f"{nomna}/weights/finetuned_vocab.txt"),
                   ("orig", f"{nomna}/weights/NomNaOCR_CRNNxCTC.h5", f"{nomna}/vocab.txt")]:
    if not (os.path.exists(w) and os.path.exists(v)):
        continue
    rec = NomNaRecognizer(w, v); r = []
    for i in range(0, len(imgs), 16):
        r += rec.recognize(imgs[i:i + 16])
    res[name] = r
json.dump(res, open(out_path, "w"), ensure_ascii=False)
'''


def segment_pages(layouts, pages, seg_dir, target=5):
    from PIL import Image
    seg_dir.mkdir(parents=True, exist_ok=True)
    by_page = {L["page"]: L for L in layouts}
    meta = []
    for p in pages:
        L = by_page.get(p)
        if not L or L["blank"]:
            continue
        top, bot, left, right = L["frame"]; cp = L["char_pitch"] or 90
        im = Image.open(NOM_DIR / f"canvas_{p:04d}.jpg").convert("L"); a = np.asarray(im); bw = a < otsu_thr(a)
        for c in sorted(L["cols"], key=lambda c: c["slot"]):
            x0, x1 = left + c["x0"] - 6, left + c["x1"] + 6; y0, y1 = top + c["y0"], top + c["y1"]
            hp = bw[y0:y1 + 1, x0:x1].mean(1); n = len(hp)
            gaps = [(i + j) // 2 for i, j in runs(hp < 0.004) if j - i >= 3]
            cuts, start = [0], 0
            while True:
                tgt = start + target * cp
                if tgt >= n - 0.6 * cp:
                    break
                cand = [g for g in gaps if abs(g - tgt) <= 0.6 * cp and g > start + cp]
                cut = min(cand, key=lambda g: abs(g - tgt)) if cand else int(tgt)
                cuts.append(cut); start = cut
            cuts.append(n)
            for k, (ya, yb) in enumerate(zip(cuts[:-1], cuts[1:])):
                if yb - ya < cp * 0.5:
                    continue
                fn = f"p{p}_s{c['slot']}_k{k}.png"
                im.crop((x0, y0 + ya, x1, y0 + yb)).save(seg_dir / fn)
                meta.append({"page": p, "slot": c["slot"], "k": k, "file": fn, "h": int(yb - ya), "est_chars": round((yb - ya) / cp, 1)})
    return meta


def lcs(chars, syls, rev):
    m = len(syls); prev = [0] * (m + 1)
    for ch in chars:
        rs = rev.get(ch, set()); cur = [0] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if syls[j - 1] in rs else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def window_match(chars, stories, rev, W=150, step=25):
    best = {}
    for s in stories:
        syl = s["syl_canon"]; b = (0, 0)
        for st in range(0, max(1, len(syl) - W + 1), step):
            l = lcs(chars, syl[st:st + W], rev)
            if l > b[0]:
                b = (l, st)
        best[s["so"]] = b
    return best


def run_nomna(layouts, stories, rev, pages, out, tf_env):
    seg_dir = out / "segs"
    meta = segment_pages(layouts, pages, seg_dir)
    if not meta:
        return {"status": "skipped", "reason": "không cắt được đoạn"}
    meta_path = out / "segs_meta.json"; ocr_path = out / "segs_ocr.json"
    json.dump(meta, open(meta_path, "w"), ensure_ascii=False)
    runner = out / "_nomna_runner.py"; runner.write_text(NOMNA_RUNNER)
    r = subprocess.run([str(tf_env / "bin" / "python"), str(runner), str(NOMNA), str(seg_dir), str(meta_path), str(ocr_path)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not ocr_path.exists():
        return {"status": "failed", "stderr_tail": r.stderr[-600:]}
    ocr = json.load(open(ocr_path))
    random.seed(0)
    allchars = [c for v in ocr.values() for s in v for c in s]
    expected = {}
    for row in REF:
        expected[row[4]] = row[0]
    res = {"status": "ok", "n_segments": len(meta), "seg_est_chars_med": float(np.median([m["est_chars"] for m in meta])),
           "models": list(ocr), "pages": {}}
    for p in pages:
        idx = [i for i, m in enumerate(meta) if m["page"] == p]
        if not idx:
            continue
        exp_story = max((k for k in expected if k <= p), default=None)
        exp_story = expected.get(exp_story)
        pr = {"expected_story": exp_story}
        for model, seqs in ocr.items():
            chars = [c for i in idx for c in seqs[i]]
            best = window_match(chars, stories, rev)
            rank = sorted(best, key=lambda n: -best[n][0])
            ctrl = []
            for _ in range(3):
                rnd = [random.choice(allchars) for _ in chars]
                ctrl.append(max(v[0] for v in window_match(rnd, stories, rev).values()))
            pr[model] = {"n_chars": len(chars), "in_dict": sum(1 for c in chars if c in rev), "best_story": rank[0],
                         "best_lcs": best[rank[0]][0], "best_offset": best[rank[0]][1], "second_lcs": best[rank[1]][0],
                         "rank_of_expected": rank.index(exp_story) + 1 if exp_story in rank else None, "ctrl_max": max(ctrl)}
        res["pages"][str(p)] = pr
    return res


# ------------------------------------------------------------------------- debug
def draw_debug(out, lines, stories, auto_titles, layouts, bounds_auto, qn_pages, nom_pages):
    from PIL import Image, ImageDraw
    n = 0
    for p in qn_pages[:2]:
        f = QN_DIR / f"canvas_{p:04d}.jpg"
        if not f.exists():
            continue
        im = Image.open(f).convert("RGB"); d = ImageDraw.Draw(im)
        for l in lines:
            if l["page"] == p:
                d.rectangle((l["x0"], l["y0"], l["x1"], l["y1"]), outline=(0, 160, 255), width=1)
        for s in stories:
            if s["qn_canvas"] == p:
                d.line((0, s["qn_y"], im.width, s["qn_y"]), fill=(255, 0, 0), width=3)
        for t in auto_titles:
            if t["page"] == p:
                d.line((0, t["y"] - 5, im.width, t["y"] - 5), fill=(0, 200, 0), width=3)
        im.resize((im.width // 2, im.height // 2)).save(out / f"debug_qn_{p:04d}.jpg", quality=80); n += 1
    for p in nom_pages[:3]:
        L = next((x for x in layouts if x["page"] == p), None)
        if not L or L["blank"]:
            continue
        im = Image.open(NOM_DIR / f"canvas_{p:04d}.jpg").convert("RGB"); d = ImageDraw.Draw(im)
        top, bot, left, right = L["frame"]
        d.rectangle((left, top, right, bot), outline=(255, 160, 0), width=2)
        for c in L["cols"]:
            d.rectangle((left + c["x0"], top + c["y0"], left + c["x1"], top + c["y1"]), outline=(0, 160, 255), width=2)
            d.text((left + c["x0"], top + c["y0"] - 14), f"s{c['slot']} {c['n_est']}/{c['n_runs']}", fill=(255, 0, 0))
        for (bp, bs, kind) in bounds_auto:
            if bp == p:
                x = left + bs * L["col_pitch"]
                d.line((x, top, x, bot), fill=(255, 0, 0), width=4)
        im.resize((im.width // 2, im.height // 2)).save(out / f"debug_nom_{p:04d}.jpg", quality=80); n += 1
    return n


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default=BOOK)
    ap.add_argument("--out", default=None, help=f"mặc định measure_out/{BOOK}/chresto_map/ (quy ước: mỗi phép đo một thư mục con)")
    ap.add_argument("--limit", type=int, default=0, help="chạy thử: N trang QN + N trang Nôm đầu (0 = tất cả)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nomna-pages", type=int, default=3, help="số trang Nôm đầu chạy NomNaOCR (0 = tắt)")
    ap.add_argument("--tf-env", default=str(TF_ENV_DEFAULT), help="venv có tensorflow; vắng → bỏ qua NomNaOCR")
    a = ap.parse_args()
    assert a.book == BOOK, f"script này chỉ đo {BOOK}"
    out = Path(a.out) if a.out else REPO / "measure_out" / BOOK / "chresto_map"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    qn_pages = QN_PAGES[:a.limit] if a.limit else QN_PAGES
    nom_pages = NOM_PAGES[:a.limit] if a.limit else NOM_PAGES
    rev, lex = load_dict()

    # (a) QN
    lines = qn_ocr(qn_pages, out / "ocr_qn_tsv", a.workers)
    page_rate = {p: page_lexicon_rate([l for l in lines if l["page"] == p], lex) for p in qn_pages}
    vn_pages = {p for p, (r, n) in page_rate.items() if r >= 0.5 and n >= 30}
    anchors = [(r[0], r[1], r[2], r[3]) for r in REF if r[1] in vn_pages]
    stories = build_stories(lines, anchors, lex, vn_pages)
    auto_titles = [t for p in sorted(vn_pages) for t in detect_titles_auto([l for l in lines if l["page"] == p])]
    matched = 0
    for s in stories:
        hit = [t for t in auto_titles if t["page"] == s["qn_canvas"] and -20 <= t["y"] - s["qn_y"] <= 260]
        s["auto_confirmed"] = bool(hit); matched += bool(hit)
        s["can_nguoi_ra"] = not hit
    auto_fp = sum(1 for t in auto_titles if not any(t["page"] == s["qn_canvas"] and -20 <= t["y"] - s["qn_y"] <= 260 for s in stories))
    with open(out / "qn_stories.csv", "w", newline="") as f:
        keys = [k for k in stories[0].keys() if k != "syl_canon"] if stories else ["so"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for s in stories:
            w.writerow({**s, "qn_trang": ";".join(map(str, s["qn_trang"]))})
    with open(out / "qn_lines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0].keys())); w.writeheader(); w.writerows(lines)
    with open(out / "qn_titles_auto.csv", "w", newline="") as f:
        if auto_titles:
            w = csv.DictWriter(f, fieldnames=list(auto_titles[0].keys())); w.writeheader(); w.writerows(auto_titles)

    # (b) Nôm
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        layouts = list(ex.map(analyze_nom, nom_pages))
    layouts.sort(key=lambda L: L["page"])
    col_rows = [{"page": L["page"], "frame_ok": L["frame_ok"], "col_pitch": L.get("col_pitch"), "char_pitch": L.get("char_pitch"),
                 "nslot": L.get("nslot"), **{k: v for k, v in c.items()}} for L in layouts for c in L["cols"]]
    if col_rows:
        with open(out / "nom_columns.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(col_rows[0].keys())); w.writeheader(); w.writerows(col_rows)
    text_pages = [L for L in layouts if not L["blank"]]
    full_cols = [c for L in text_pages for c in L["cols"] if not c["short"]]
    agree_chars = [abs(c["n_est"] - c["n_runs"]) <= 2 for c in full_cols]

    # (c) bảng truyện ↔ trang
    bounds_auto = boundaries_auto(layouts)
    ref_starts = {(r[4], r[5]) for r in REF if r[4] in set(nom_pages)}
    ref_starts_eval = {k for k in ref_starts if k != (106, 1)}          # đầu sách không có tín hiệu ranh giới
    auto_set = {(p, s) for p, s, _ in bounds_auto}
    tp = len(auto_set & ref_starts)
    table = story_page_table(layouts, stories)
    for row in table:
        row["ranh_gioi_auto"] = (row["nom_canvas_dau"], row["nom_slot_dau"]) in auto_set
    with open(out / "bang_truyen_trang.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    with open(out / "nom_boundaries_auto.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["page", "slot", "kind", "in_ref"])
        w.writerows([(p, s, k, (p, s) in ref_starts) for p, s, k in bounds_auto])

    # (d) NomNaOCR
    tf_env = Path(a.tf_env)
    if a.nomna_pages > 0 and (tf_env / "bin" / "python").exists() and stories:
        nomna = run_nomna(layouts, stories, rev, nom_pages[:a.nomna_pages], out, tf_env)
    else:
        nomna = {"status": "skipped", "reason": "tf_env vắng" if not (tf_env / "bin" / "python").exists() else "--nomna-pages 0"}

    n_dbg = draw_debug(out, lines, stories, auto_titles, layouts, bounds_auto, sorted(vn_pages), nom_pages)

    # ---- invariants
    inv = []

    def add(name, expected, observed, ok, method=""):
        inv.append({"name": name, "expected": expected, "observed": observed, "pass": bool(ok), "method": method})

    n_exp_st = len(anchors)
    add("qn_n_stories", 20 if not a.limit else f"{n_exp_st} (limit)", len(stories), len(stories) == (20 if not a.limit else n_exp_st), "neo REF trên trang VN")
    if stories:
        add("qn_every_story_has_lines", "min n_lines >= 5", min(s["n_lines"] for s in stories),
            min(s["n_lines"] for s in stories) >= 5, "tesseract dòng thân")
        allw = sum(s["n_syll"] for s in stories); alloov = sum(s["oov"] for s in stories)
        add("qn_oov_rate_vs_dict", "<= 0.08", round(alloov / allw, 4) if allw else None, allw and alloov / allw <= 0.08, "âm tiết ngoài dict/QuocNgu_SinoNom")
        add("qn_titles_auto_vs_ref_recall", f">= {int(0.7 * len(stories))}/{len(stories)}", f"{matched}/{len(stories)} (dư {auto_fp})",
            matched >= 0.7 * len(stories), "2 phương pháp: neo REF vs bố cục tự động")
        if not a.limit:
            add("qn_total_lines_range", "560..660 (đo trước 610)", sum(s["n_lines"] for s in stories),
                560 <= sum(s["n_lines"] for s in stories) <= 660)
    add("qn_vn_pages", "29..53 là VN; 54,55,56 không" if not a.limit else "trang VN ≥ 1",
        f"{min(vn_pages) if vn_pages else None}-{max(vn_pages) if vn_pages else None} n={len(vn_pages)}",
        (vn_pages == set(range(29, 54))) if not a.limit else len(vn_pages) >= 1, "tỉ lệ âm tiết trong lexicon ≥ 0,5")
    add("nom_n_text_pages", len(nom_pages), len(text_pages), len(text_pages) == len(nom_pages), "canvas 106–170 đều có chữ")
    add("nom_frame_detected", f"{len(text_pages)}/{len(text_pages)}", sum(1 for L in text_pages if L["frame_ok"]),
        all(L["frame_ok"] for L in text_pages), "mở hình thái 160px")
    add("nom_nslot_7", "mọi trang", sorted({L["nslot"] for L in text_pages}), all(L["nslot"] == 7 for L in text_pages))
    add("nom_n_cols_le_nslot", "mọi trang", max((L["n_cols"] for L in text_pages), default=0),
        all(L["n_cols"] <= L["nslot"] for L in text_pages))
    cps = [L["col_pitch"] for L in text_pages]
    add("nom_col_pitch_px", "130..150", [min(cps), max(cps)] if cps else None, cps and 130 <= min(cps) and max(cps) <= 150, "tự tương quan")
    if full_cols:
        med = float(np.median([c["n_est"] for c in full_cols]))
        add("nom_chars_per_full_col_med", "15..24 (đếm mắt 17–18)", round(med, 1), 15 <= med <= 24, "pitch chữ")
        add("nom_chars_agreement_pitch_vs_runs", ">= 0.7 cột đầy lệch ≤ 2", round(float(np.mean(agree_chars)), 3),
            np.mean(agree_chars) >= 0.7, "2 phương pháp độc lập")
    if not a.limit:
        tot = sum(L["n_cols"] for L in text_pages)
        add("nom_total_cols", "400..440 (đo trước 419)", tot, 400 <= tot <= 440)
    if ref_starts_eval:
        rec = len(auto_set & ref_starts_eval) / len(ref_starts_eval)
        prec = tp / len(auto_set) if auto_set else 0
        add("nom_boundary_auto_recall", ">= 0.8 (bỏ đầu sách)", round(rec, 2), rec >= 0.8, "ô trống nội bộ / đầu trang sau cột ngắn")
        add("nom_boundary_auto_precision", ">= 0.6", round(prec, 2), prec >= 0.6)
    tab_ok = [r for r in table if r["ti_le_chu_am"] is not None and (r["qn_n_lines"] or 0) >= 15]
    if tab_ok and not a.limit:
        ratios = [r["ti_le_chu_am"] for r in tab_ok]
        out_r = [r["so"] for r in table if r["ti_le_chu_am"] is not None and not 0.85 <= r["ti_le_chu_am"] <= 1.35]
        add("map_chars_per_syllable_ratio", "0.85..1.35 (truyện ≥ 15 dòng QN)", f"[{min(ratios)}, {max(ratios)}] ngoài: {out_r}",
            0.85 <= min(ratios) and max(ratios) <= 1.35, "chữ Nôm ước / âm tiết QN OCR; truyện ngắn sai số lớn")
        add("map_nom_cols_monotone", "trang Nôm đầu tăng dần", True, all(REF[i][4] <= REF[i + 1][4] for i in range(19)))
    if nomna.get("status") == "ok":
        ranks = [pr[m]["rank_of_expected"] for pr in nomna["pages"].values() for m in nomna["models"] if m in pr]
        above = [pr[m]["best_lcs"] > pr[m]["ctrl_max"] for pr in nomna["pages"].values() for m in nomna["models"] if m in pr]
        add("nomna_rank_of_expected_story", "1 mọi trang/model", ranks, all(r == 1 for r in ranks), "LCS cửa sổ 150 âm")
        add("nomna_lcs_above_random_ctrl", "mọi trang/model", sum(above), all(above))

    summary = {
        "measure": "chresto_map", "book": BOOK, "date": time.strftime("%Y-%m-%d"),
        "cli": " ".join(["scripts/measure/chresto_map.py"] + [x for x in sys.argv[1:] if not x.startswith("/")]),
        "runtime_s": round(time.time() - t0, 1),
        "qn": {"method": "tesseract 5 -l vie --psm 4 TSV; trang VN = lexicon ≥ 0,5", "pages_run": [qn_pages[0], qn_pages[-1]],
               "vn_pages": sorted(vn_pages), "n_lines_total": len(lines),
               "n_body_lines": sum(s["n_lines"] for s in stories), "n_syll": sum(s["n_syll"] for s in stories),
               "oov_rate": round(sum(s["oov"] for s in stories) / max(1, sum(s["n_syll"] for s in stories)), 4),
               "conf_med": round(float(np.median([l["conf"] for l in lines if l["conf"] >= 0])), 1) if lines else None,
               "titles_auto": {"n": len(auto_titles), "matched_ref": matched, "extra": auto_fp, "method": "dòng ngắn căn giữa kề dòng thân"},
               "stories": [{"so": s["so"], "canvas": s["qn_canvas"], "n_lines": s["n_lines"], "n_syll": s["n_syll"],
                            "auto": s["auto_confirmed"], "title": s["tieu_de"][:40]} for s in stories]},
        "nom": {"method": "khung mở hình thái; pitch cột tự tương quan 100–180; lưới 7 ô; chữ/cột = pitch chữ (70–120) vs run mực",
                "n_pages": len(text_pages), "n_cols": sum(L["n_cols"] for L in text_pages),
                "cols_per_page_hist": {str(k): sum(1 for L in text_pages if L["n_cols"] == k) for k in sorted({L["n_cols"] for L in text_pages})},
                "col_pitch_med": float(np.median(cps)) if cps else None,
                "char_pitch_med": float(np.median([L["char_pitch"] for L in text_pages if L["char_pitch"]])) if text_pages else None,
                "chars_per_full_col_med_pitch": round(float(np.median([c["n_est"] for c in full_cols])), 1) if full_cols else None,
                "chars_per_full_col_med_runs": float(np.median([c["n_runs"] for c in full_cols])) if full_cols else None,
                "agreement_pitch_vs_runs_le2": round(float(np.mean(agree_chars)), 3) if agree_chars else None,
                "total_chars_est": round(sum(c["n_est"] for L in text_pages for c in L["cols"])),
                "n_short_cols": sum(1 for L in text_pages for c in L["cols"] if c["short"]),
                "n_narrow_skipped": sum(L["n_narrow_skipped"] for L in text_pages),
                "n_empty_internal_slots": sum(len(L["empty_internal_slots"]) for L in text_pages),
                "n_circles": sum(c["circles"] for L in text_pages for c in L["cols"])},
        "map": {"source": "REF: agent đọc ảnh 2026-09-21, chưa người rà (cột can_nguoi_ra=True)", "n_stories": len(table),
                "boundary_auto": {"n_cands": len(auto_set), "tp": tp, "ref_n": len(ref_starts),
                                  "false": sorted(auto_set - ref_starts)[:12], "missed": sorted(ref_starts - auto_set)},
                "table": [{"so": r["so"], "qn": r["qn_canvas"], "nom": r["nom_trang"], "slot0": r["nom_slot_dau"], "cols": r["nom_n_cols"],
                           "chu": r["nom_chu_uoc_pitch"], "ratio": r["ti_le_chu_am"], "auto": r["ranh_gioi_auto"]} for r in table]},
        "nomna": nomna,
        "invariants": inv,
        "files": ["qn_stories.csv", "qn_lines.csv", "qn_titles_auto.csv", "nom_columns.csv", "bang_truyen_trang.csv",
                  "nom_boundaries_auto.csv", f"debug_*.jpg ×{n_dbg}", "ocr_qn_tsv/ (cache)"],
    }
    txt = json.dumps(summary, ensure_ascii=False, indent=1)
    if len(txt.encode()) > 8000:
        txt = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    if len(txt.encode()) > 8000:
        summary["qn"].pop("stories"); summary["qn"]["stories"] = "xem qn_stories.csv"
        txt = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    if len(txt.encode()) > 8000:
        summary["map"]["table"] = "xem bang_truyen_trang.csv"
        txt = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    (out / "summary.json").write_text(txt, encoding="utf-8")
    q, nm = summary["qn"], summary["nom"]
    print(f"QN: trang VN {q['vn_pages'][:1]}..{q['vn_pages'][-1:]} | {len(stories)} truyện, {q['n_body_lines']} dòng, {q['n_syll']} âm, OOV {q['oov_rate']}, "
          f"tiêu đề tự động khớp {matched}/{len(stories)} (+{auto_fp} dư)")
    print(f"NÔM: {nm['n_pages']} trang, {nm['n_cols']} cột, pitch cột {nm['col_pitch_med']}, chữ/cột đầy {nm['chars_per_full_col_med_pitch']} (pitch) "
          f"vs {nm['chars_per_full_col_med_runs']} (run), khớp≤2: {nm['agreement_pitch_vs_runs_le2']}, tổng chữ ≈ {nm['total_chars_est']}")
    print(f"MAP: ranh giới tự động {len(auto_set)} ứng viên, đúng {tp}/{len(ref_starts)} | NomNaOCR: {nomna.get('status')}"
          + (f" ({nomna.get('reason')})" if nomna.get("reason") else ""))
    for i in inv:
        print(f"  [{'PASS' if i['pass'] else 'FAIL'}] {i['name']}: expected {i['expected']} | observed {i['observed']}")
    print(f"saved {out}/summary.json ({len(txt.encode())} B), {summary['runtime_s']}s")


if __name__ == "__main__":
    main()
