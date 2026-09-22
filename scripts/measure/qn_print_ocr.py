#!/usr/bin/env python
"""qn_print_ocr.py — OCR trang Quốc ngữ IN (Abel des Michels 1883/1884) thành bảng câu thơ có số câu.

Gộp từ scripts/measure_wf_2026-09-21/qn_ocr/{run_tess_all, extract_verses, classify_kvk_full, chain_analysis,
margin_number_ocr}.py + sửa từ qn_ocr_critic. Chỉ phụ thuộc .venv (numpy, PIL) + tesseract CLI (vie).

Chức năng (--book LucVanTien1883 | KimVanKieu1884):
  1. KVK: phân loại từng canvas quocngu_pages/{vol1,vol2} thành QN / FR (dịch Pháp) / other (bìa, tựa, khảo cứu).
  2. tesseract -l vie --psm 4 TSV toàn trang, cache theo md5 ảnh (measure_out/_cache/qn_print_ocr/tess4/<md5>.json).
  3. Tách vùng: tiêu đề chạy | dòng thơ | cước chú Pháp (extract_verses) ; loại dòng Pháp lọt lưới + dòng rác.
  4. Neo số câu theo VỊ TRÍ mod 5 (số in ở lề mỗi 5 câu) + GIÁ TRỊ đọc được; chuỗi toàn sách s_{k+1} = s_k + n_k, đặt lại khi
     đa số neo bảo khác ("trôi"); trang chỉ 1 neo lệch = "mơ hồ".
  5. Phương pháp độc lập thứ 2 cho số câu neo: cắt dải lề trái của từng dòng thơ tại vị trí mod 5, phóng 3x,
     tesseract --psm 7 chỉ chữ số ("margin_digits") -> agreement với chuỗi.
  6. Cổng: tổng câu == 2.088 / 3.253, đơn điệu, parity 6/8, danh sách trang cần người xem.
  7. Xuất verses.tsv (verse_no, page, line_text, n_syll, anchor_source, confidence) = đầu vào cho adapter ingest.

CLI:  python scripts/measure/qn_print_ocr.py --book LucVanTien1883 [--limit 6] [--start 26] [--workers 4] [--engine tesseract|vietocr]
Đầu ra (measure_out/<book>/qn_ocr/): summary.json (<= 8 KB, mọi số kèm n + phương pháp + "invariants"), verses.tsv, pages.csv (mọi canvas
+ lớp + đặc trưng), anchors.csv (số đọc inline vs lề theo dòng), pages_review.json (trang trôi/mơ hồ/không neo), debug/*.jpg (<= 5).
verses.tsv: verse_no = số câu THEO SỐ IN (chuỗi neo); seq_no = thứ tự vật lý 1..N (dùng khi ấn bản đánh số bất thường — KVK có 4 chỗ);
anchor_source ∈ {ocr_num, margin_digits, position_mod5, chain_interp}; confidence = tesseract conf trung bình/100.
Idempotent: thứ tự trang cố định (sort theo tên), không ngẫu nhiên, cache theo md5 ảnh (measure_out/_cache/qn_print_ocr/).
Thời gian (M-series, 4 worker): LVT 139 trang ≈ 40 s (tess 15 s + số lề 25-40 s); KVK 631 canvas ≈ 3,5 phút; chạy lại từ cache ≈ 1-2 s.
--engine vietocr ≈ 15-25 s/trang QN (LVT ≈ 40 phút, KVK ≈ 1,5-2 giờ), CER cao hơn tesseract (0,19 vs 0,07 trên 44 dòng đọc mắt).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CACHE_ROOT = REPO / "measure_out" / "_cache" / "qn_print_ocr"
TESS_VERSION = "tess-vie-psm4-tsv-v1"
MARGIN_VERSION = "margin-psm7-digits-x3-v1"
VIETOCR_VERSION = "vietocr-projection_deskew-v1"

BOOKS = {
    "LucVanTien1883": dict(
        qn_dirs=[("", DATA / "LucVanTien1883" / "quocngu_pages")],
        tag_prefix="lvt",
        classify=False,
        expected_lines=2088,          # 18 + 20×103 + 10 (đo từ Nôm 105 trang)
        printed_last=2088,
        known_drift_pages={},
        lines_per_page=(4, 20),
        expected_qn_pages=139,
        parity_flip_at=None,
        expected_pages_per_vol=None,
    ),
    "KimVanKieu1884": dict(
        qn_dirs=[("vol1", DATA / "KimVanKieu1884" / "quocngu_pages" / "vol1"),
                 ("vol2", DATA / "KimVanKieu1884" / "quocngu_pages" / "vol2")],
        tag_prefix="kvk",
        classify=True,
        # số DÒNG thật = số in cuối 3253 − 3 số bị bỏ qua + 1 dòng không số = 3251. Bất thường đánh số của ấn bản (đã soi dòng OCR thô):
        #   1584 không in (vol2 canvas 21→23), 2224 không in (canvas 133→135), 2235 in sớm một dòng (canvas 137: 11 dòng giữa 2230 và 2240),
        #   3054 không in (canvas 267→269). Chuỗi verse_no THEO SỐ IN; seq_no trong verses.tsv = thứ tự vật lý 1..N.
        expected_lines=3251,
        printed_last=3253,            # "Mua vui cũng được một vài trống canh" = 3250 + 3 (vol2 canvas 303, xem bằng mắt)
        known_drift_pages={"kvk_vol2_canvas_0023": 1, "kvk_vol2_canvas_0135": 1, "kvk_vol2_canvas_0137": -1, "kvk_vol2_canvas_0269": 1},
        lines_per_page=(2, 18),       # vol2 canvas 301: 2 câu rồi tới khối trích dẫn Hán + dịch Pháp
        expected_qn_pages=295,        # vol1 146 + vol2 149, chỉ canvas lẻ
        parity_flip_at=1585,          # từ số in 1585 câu lẻ = 8 âm tiết
        expected_pages_per_vol={"vol1": 146, "vol2": 149},
        expected_canvases=631,
    ),
}

# ----------------------------------------------------------------------------------------------------------------
# 1. Tesseract TSV -> dòng có toạ độ (run_ocr_pages.tess_lines)
# ----------------------------------------------------------------------------------------------------------------

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tess_lines(img_path: Path, psm: int = 4) -> list[dict]:
    r = subprocess.run(["tesseract", str(img_path), "-", "-l", "vie", "--psm", str(psm), "tsv"],
                       capture_output=True, text=True)
    rows = [ln.split("\t") for ln in r.stdout.splitlines()[1:]]
    groups: dict = {}
    for row in rows:
        if len(row) < 12 or row[0] != "5":
            continue
        key = (int(row[2]), int(row[3]), int(row[4]))
        l, t, w, h, conf, txt = int(row[6]), int(row[7]), int(row[8]), int(row[9]), float(row[10]), row[11]
        if not txt.strip():
            continue
        g = groups.setdefault(key, {"words": [], "x1": 10**9, "y1": 10**9, "x2": 0, "y2": 0, "hs": [], "confs": []})
        g["words"].append(txt); g["hs"].append(h); g["confs"].append(conf)
        g["x1"] = min(g["x1"], l); g["y1"] = min(g["y1"], t); g["x2"] = max(g["x2"], l + w); g["y2"] = max(g["y2"], t + h)
    lines = []
    for key in sorted(groups, key=lambda k: (groups[k]["y1"], groups[k]["x1"])):
        g = groups[key]
        lines.append(dict(x1=g["x1"], y1=g["y1"], x2=g["x2"], y2=g["y2"], h=int(np.median(g["hs"])),
                          conf=round(float(np.mean(g["confs"])), 1), text=" ".join(g["words"]), src=f"tess{psm}"))
    lines.sort(key=lambda d: (d["y1"], d["x1"]))
    return lines


def ocr_page_cached(item: tuple[str, Path], engine: str) -> dict:
    """Trả về {'tag','path','size','md5','lines'}; cache theo md5 ảnh + phiên bản engine."""
    tag, path = item
    digest = md5_file(path)
    sub = "tess4" if engine == "tesseract" else "vietocr"
    cdir = CACHE_ROOT / sub
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"{digest}.json"
    version = TESS_VERSION if engine == "tesseract" else VIETOCR_VERSION
    if cpath.exists():
        try:
            d = json.loads(cpath.read_text(encoding="utf-8"))
            if d.get("version") == version:
                d.update(tag=tag, path=str(path), fresh=False)
                return d
        except Exception:
            pass
    with Image.open(path) as im:
        size = list(im.size)
    t0 = time.time()
    if engine == "tesseract":
        lines = tess_lines(path, 4)
    else:
        lines = vietocr_lines(path)
    d = dict(version=version, md5=digest, size=size, lines=lines, t=round(time.time() - t0, 2))
    cpath.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    d.update(tag=tag, path=str(path), fresh=True)
    return d


_VIET_PRED = None


def vietocr_lines(img_path: Path) -> list[dict]:
    """Đường đi VietOCR như core/ocr/qn_ocr.py (projection_deskew) nhưng giữ hộp dòng để lọc thơ/cước chú."""
    global _VIET_PRED
    import cv2  # noqa
    sys.path.insert(0, str(REPO))
    from core.ocr.line_detector import estimate_skew_angle, _binary, _rotate, detect_line_boxes  # noqa
    from core.ocr.qn_ocr import _get_predictor, _predict_with_conf  # noqa
    if _VIET_PRED is None:
        _VIET_PRED = _get_predictor()
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    angle = estimate_skew_angle(_binary(img_rgb))
    work = img_rgb
    if abs(angle) >= 0.1:
        work = _rotate(img_rgb, angle, border_value=(255, 255, 255))
    out = []
    for (x1, y1, x2, y2) in detect_line_boxes(work):
        crop = work[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue
        text, conf = _predict_with_conf(_VIET_PRED, Image.fromarray(crop))
        if text:
            out.append(dict(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), h=int(y2 - y1),
                            conf=round(float(conf) * 100, 1), text=text, src="vietocr"))
    return out


# ----------------------------------------------------------------------------------------------------------------
# 2. Tách vùng + trích dòng thơ (extract_verses.py)
# ----------------------------------------------------------------------------------------------------------------
VI_SPEC = set("ăđơưạảãắằẳẵặấầẩẫậẹẻẽếềểễệỉịọỏõốồổỗộớờởỡợụủũứừửữựỳỷỹỵý")
FR_STOP = set(("le la les de des du que qui est et un une pour dans ce cette ces sur par au aux il elle ils elles "
               "ne pas plus son sa ses leur mais avec comme lui nous vous je tu on se en ou où à est sont été "
               "fait faire dit peut être cet mot sens litt").split())
TITLE_WORDS = {"luc", "van", "tien", "kim", "kieu", "tan", "truyen"}
DIGIT_MAP = {"ö": "5", "õ": "5", "ð": "5", "Ö": "5", "O": "0", "o": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "|": "1",
             "!": "1", "ï": "1", "¡": "1", "i": "1"}
NUM_TOKEN_RE = re.compile(r'^\s*[«‹"\'\-_:;,.]*\s*([0-9öõðÖOoQDIl|!ï¡iSsBZzGT]{1,4})(?=\s+\S|\s*$)')
PUNCT = "«»‹›\"'!?;:,.()[]{}—–-_*"
FOOTNOTE_PAREN_RE = re.compile(r"^\s*[«‹]?\s*\d{1,2}\s*[\)°º]\s")
FOOTNOTE_MARK_RE = re.compile(r"^\s*([«‹]?\s*(\d{1,2}\s*[\)\.°º]\s|L[iíìl]tt)|»)", re.I)
ONLY_NUM_RE = re.compile(r"[\s\d«»\-_.:;,]*")


_QN_DICT: set | None = None


def qn_dict() -> set:
    """Tập âm tiết QN (dict/QuocNgu_SinoNom.csv, cột 1) — chỉ làm proxy 'âm tiết hợp lệ' để nhận diện dòng rác (trích dẫn Hán
    OCR thành ký tự vô nghĩa), KHÔNG dùng gán nhãn. Chuẩn hoá vị trí dấu thanh qua core.text.text_utils nếu có."""
    global _QN_DICT
    if _QN_DICT is not None:
        return _QN_DICT
    try:
        sys.path.insert(0, str(REPO))
        from core.text.text_utils import normalize_tone_marks as _nt  # noqa
    except Exception:
        _nt = lambda t: unicodedata.normalize("NFC", t)  # noqa: E731
    words = set()
    path = REPO / "dict" / "QuocNgu_SinoNom.csv"
    if path.exists():
        with open(path, encoding="utf-8-sig") as fh:
            rd = csv.reader(fh); next(rd, None)
            for row in rd:
                if row and row[0].strip():
                    words.add(_nt(row[0].strip().lower()))
    _QN_DICT = words
    globals()["_norm_tone"] = _nt
    return _QN_DICT


_norm_tone = None
_OK_CHARS_RE = re.compile(r"^[^\W\d_]+$|^\d+$")


def weird_token(tok: str) -> bool:
    """Token 'lạ' = có ký tự ngoài chữ/số/dấu câu, trộn chữ-số, IN HOA >= 2 ký tự, hoặc thường-rồi-HOA (mB, s`1, L1, TT], †)."""
    t = tok.strip(PUNCT + "«»‹›")
    if not t:
        return tok.strip() not in ("", "«", "»", "‹", "›") and not all(c in PUNCT for c in tok)
    if not _OK_CHARS_RE.match(t):
        return True
    if t.isdigit():
        return False
    if len(t) >= 2 and t.isupper():
        return True
    return t[0].islower() and any(c.isupper() for c in t[1:])


def is_garbage_line(text: str) -> bool:
    """Dòng trích dẫn Hán OCR ra ký tự vô nghĩa rồi « lời dịch QN. Luật rất bảo thủ vì dòng thơ in nghiêng nát chữ
    (LVT: «9ð cö0n hai chữ «&kÖoø kÿ») phải được GIỮ: chỉ khi có token lạ và MỌI token (>= 2) trước dấu « đều ngoài từ điển.
    Khối trích dẫn Hán chủ yếu được bắt bằng hình học (bước dòng sít, chữ thấp) trong split_zones."""
    d = qn_dict()
    toks = text.split()
    n_weird = sum(1 for t in toks if weird_token(t))
    syl = [t.lower() for t in syllables(text)]
    hit = sum(1 for t in syl if _norm_tone(t) in d) / len(syl) if syl else 0.0
    m = re.search(r"[«‹]", text)
    if m and m.start() > 0 and n_weird >= 1:
        pre = [t.lower() for t in syllables(text[:m.start()])]
        if len(pre) >= 2 and d and not any(_norm_tone(t) in d for t in pre):
            return True
    return False


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower())


def is_french(text: str) -> bool:
    toks = tokens(text)
    if len(toks) < 5:
        return False
    n_fr = sum(1 for t in toks if t in FR_STOP)
    n_vi = sum(1 for t in toks if any(c in VI_SPEC for c in t))
    fr_ratio, vi_ratio = n_fr / len(toks), n_vi / len(toks)
    return (fr_ratio >= 0.25 and vi_ratio < 0.35) or fr_ratio >= 0.4 or (n_fr >= 3 and n_vi <= 1)


def is_header(text: str) -> bool:
    toks = tokens(text)
    hits = sum(1 for t in toks if strip_accents(t) in TITLE_WORDS)
    letters = [c for c in text if c.isalpha()]
    upper_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
    only_num = ONLY_NUM_RE.fullmatch(text) is not None
    # tiêu đề chạy in HOA ("LỤC VÂN TIÊN." / "KIM VÂN KIỀU TÂN TRUYỆN") kể cả khi OCR nát chữ ("LỤO VÂN 'HÊN.")
    caps = upper_ratio >= 0.8 and len(letters) >= 3 and len(toks) <= 6
    return (hits >= 2 and upper_ratio >= 0.6) or caps or only_num or (len(toks) <= 1 and not any(c in VI_SPEC for c in text))


def eff_height(line: dict) -> float:
    return line["h"] - 10 if line["src"] == "vietocr" else line["h"]


def is_junk(text: str) -> bool:
    letters = sum(1 for c in text if c.isalpha())
    toks = tokens(text)
    if ONLY_NUM_RE.fullmatch(text) is not None:
        return False
    return letters < 3 or (len(toks) <= 2 and not any(c in VI_SPEC for c in text) and not any(c.isdigit() for c in text))


def merge_margin_numbers(lines: list[dict]) -> list[dict]:
    """psm 4 đôi khi tách số câu ở lề thành 'dòng' riêng; gộp vào dòng thơ chồng lấn dọc >= 50% bên phải."""
    out = [dict(l) for l in lines]
    nums = [l for l in out if re.fullmatch(r"\s*\d{1,4}\s*", l["text"])]
    for n in nums:
        best, best_ov = None, 0.0
        for l in out:
            if l is n or re.fullmatch(r"\s*\d{1,4}\s*", l["text"]) or l["x1"] <= n["x1"]:
                continue
            ov = min(l["y2"], n["y2"]) - max(l["y1"], n["y1"])
            ov = ov / max(1, min(l["y2"] - l["y1"], n["y2"] - n["y1"]))
            if ov > best_ov:
                best, best_ov = l, ov
        if best is not None and best_ov >= 0.5 and not re.match(r"^\s*[«‹]?\s*\d{2,4}\s", best["text"]):
            best["text"] = n["text"].strip() + " " + best["text"]
            best["x1"] = min(best["x1"], n["x1"])
            best["y1"] = min(best["y1"], n["y1"]); best["y2"] = max(best["y2"], n["y2"])
            n["text"] = ""
    return [l for l in out if l["text"] != ""]


def merge_same_row(lines: list[dict]) -> list[dict]:
    """psm 4 đôi khi tách MỘT dòng thành 2 khối cùng hàng ("Vôi và" | "ỘI vàng quày ngựa trở ra").
    Gộp các mảnh chồng lấn dọc >= 50% (theo mảnh thấp hơn) thành một dòng, nối text theo x."""
    lines = sorted(lines, key=lambda l: (l["y1"], l["x1"]))
    groups: list[list[dict]] = []
    for l in lines:
        for g in groups:
            gy1, gy2 = min(x["y1"] for x in g), max(x["y2"] for x in g)
            ov = (min(gy2, l["y2"]) - max(gy1, l["y1"])) / max(1, min(gy2 - gy1, l["y2"] - l["y1"]))
            if ov >= 0.5:
                g.append(l); break
        else:
            groups.append([l])
    out = []
    for g in groups:
        g.sort(key=lambda x: x["x1"])
        main = max(g, key=lambda x: x["x2"] - x["x1"])
        # mảnh thấp hơn 60% mảnh chính = vết bẩn / mảnh dấu ("SH =) t Le)", "» ` là s") -> bỏ, không nối vào dòng thơ
        g = [x for x in g if x is main or x["h"] >= 0.6 * main["h"]]
        out.append(dict(x1=min(x["x1"] for x in g), y1=min(x["y1"] for x in g), x2=max(x["x2"] for x in g), y2=max(x["y2"] for x in g),
                        h=main["h"], conf=round(float(np.mean([x["conf"] for x in g])), 1), text=" ".join(x["text"].strip() for x in g),
                        src=main["src"], n_pieces=len(g)))
    return out


def split_zones(lines: list[dict], page_h: int, page_w: int, p_ref: float | None = None, h_ref: float | None = None):
    """(header_lines, verse_zone_lines, footnote_lines). p_ref/h_ref = bước dòng & chiều cao chữ thơ toàn sách (ổn định ±5%),
    dùng thay ước lượng trong trang khi trang có ít dòng thơ (trang chỉ 2 câu rồi tới khối trích dẫn Hán)."""
    lines = merge_margin_numbers(lines)
    lines = merge_same_row(lines)
    lines = sorted((l for l in lines if not is_junk(l["text"]) and l["y1"] >= 0.03 * page_h), key=lambda l: (l["y1"], l["x1"]))
    header, body = [], []
    started = False
    for l in lines:
        if not started and l["y1"] < 0.40 * page_h and is_header(l["text"]):
            header.append(l)
            continue
        started = True
        body.append(l)
    if not body:
        return header, [], []
    H = h_ref or float(np.median([eff_height(l) for l in body[:5]]))
    # mảnh nhiễu (vết bẩn, mảnh dấu: "gà ~ 1A" cao 8 px) thấp hơn 45% chữ thơ -> loại trước khi chấm điểm cước chú
    body = [l for l in body if eff_height(l) >= 0.45 * H]
    if not body:
        return header, [], []
    pitches = [body[i + 1]["y1"] - body[i]["y1"] for i in range(min(4, len(body) - 1))]
    P = p_ref or (float(np.median(pitches)) if pitches else 100.0)
    X = float(np.median([l["x1"] for l in body[:5]]))
    h_thr = (0.85 if body[0]["src"] == "vietocr" else 0.78) * H
    scores, strong = [], []
    for i, l in enumerate(body):
        if re.fullmatch(r"\s*\d{1,4}\s*", l["text"]):
            scores.append(0); strong.append(0)
            continue
        s_h = int(eff_height(l) < h_thr)
        s_p = int(i > 0 and (l["y1"] - body[i - 1]["y1"]) < 0.6 * P)
        s_f = int(is_french(l["text"]) or FOOTNOTE_MARK_RE.match(l["text"]) is not None)
        if FOOTNOTE_PAREN_RE.match(l["text"]):
            s_f = 2          # "1) Voy. le ..." — dấu mở cước chú không thể nhầm với số câu
        s_x = int(l["x1"] < X - 0.08 * page_w) if l["src"] != "vietocr" else 0   # lệch trái (yếu: dòng có số câu cũng lệch)
        s_xr = int(l["x1"] > X + 0.10 * page_w)          # thụt vào phải = khối trích dẫn Hán / thơ dịch, không phải câu thơ (mạnh)
        # lệch trái mà KHÔNG bắt đầu bằng số câu ("(n4) feuille de papier;») = mảnh cước chú -> mạnh
        s_xs = int(s_x and parse_number(l["text"])[0] is None)
        s_len = int(len(tokens(l["text"])) >= 10)      # dòng thơ tối đa 8 âm tiết (+số câu); cước chú thường >= 10 từ
        scores.append(s_h + s_p + s_f + s_x + s_xr + s_len)
        # với VietOCR chiều cao crop phụ thuộc dấu thanh (dòng ngắn mất dấu) -> không dùng h làm tín hiệu mạnh
        strong.append(s_f + s_len + s_xr + s_xs if l["src"] == "vietocr" else s_h + s_f + s_len + s_xr + s_xs)
    # khối dòng sít (>= 2 bước liên tiếp < 0,75 bước thơ) = cước chú / trích dẫn Hán xen giữa, dù từng dòng chưa đủ điểm
    tight = [i > 0 and (body[i]["y1"] - body[i - 1]["y1"]) < 0.75 * P for i in range(len(body))]
    fn_start = len(body)
    for i in range(len(body)):
        nxt = scores[i + 1] if i + 1 < len(body) else 0
        if scores[i] >= 2 or (strong[i] >= 1 and nxt >= 2) or \
                (i + 2 < len(body) and tight[i + 1] and tight[i + 2] and eff_height(body[i]) < 0.8 * H):
            fn_start = i
            break
    return header, body[:fn_start], body[fn_start:]


def parse_number(text: str):
    m = NUM_TOKEN_RE.match(text)
    if not m:
        return None, None, text
    raw = m.group(1)
    if not any(c.isdigit() for c in raw):
        return None, raw, text
    conv = "".join(DIGIT_MAP.get(c, c) for c in raw)
    if not conv.isdigit():
        return None, raw, text
    return int(conv), raw, text[m.end():].strip()


def syllables(text: str) -> list[str]:
    t = text.replace("-", " ").replace("—", " ").replace("–", " ")
    out = []
    for tok in t.split():
        tok = tok.strip(PUNCT)
        if tok and any(c.isalpha() for c in tok):
            out.append(tok)
    return out


def page_metrics(lines: list[dict], page_h: int, page_w: int) -> tuple[float, float] | None:
    """(bước dòng, chiều cao chữ) của vùng thơ một trang, None nếu < 4 dòng thơ — để lấy trung vị toàn sách."""
    r = analyse_page(lines, page_h, page_w)
    v = r["verses"]
    if len(v) < 4:
        return None
    return float(np.median([v[i + 1]["y1"] - v[i]["y1"] for i in range(len(v) - 1)])), float(np.median([x["h"] for x in v]))


def analyse_page(lines: list[dict], page_h: int, page_w: int, p_ref: float | None = None, h_ref: float | None = None) -> dict:
    header, vz, fn = split_zones(lines, page_h, page_w, p_ref, h_ref)
    verses = []
    for l in vz:
        num, raw, rest = parse_number(l["text"])
        syl = syllables(rest)
        verses.append(dict(x1=l["x1"], y1=l["y1"], x2=l["x2"], y2=l["y2"], h=l["h"], conf=l["conf"], num=num, num_raw=raw,
                           text=rest, raw_text=l["text"], n_syl=len(syl), french=is_french(rest), src=l["src"]))
    leaks = [v for v in verses if v["french"]]
    # hình học vùng thơ: chữ cao H, mép trái X (dòng không số) -> dòng ngắn (< 3 âm tiết) nhưng đúng hàng, đúng cỡ = câu thơ mất chữ
    Hv = float(np.median([v["h"] for v in verses])) if verses else 0.0
    xs = [v["x1"] for v in verses if v["num"] is None]
    Xv = float(np.median(xs)) if xs else (float(np.median([v["x1"] for v in verses])) if verses else 0.0)
    for v in verses:
        v["garbage"] = (not v["french"]) and is_garbage_line(v["text"])
        v["text_lost"] = (v["n_syl"] == 0 and v["num"] is not None)
        # mép trái: dòng có số câu lệch trái ~0,07W ("2085 Hư không" -> OCR mất số) -> chấp nhận [Xv-0,10W, Xv+0,03W]
        v["short_verse"] = (v["n_syl"] < 3 and not v["text_lost"] and not v["garbage"] and v["h"] >= 0.8 * Hv
                            and -0.10 * page_w <= v["x1"] - Xv <= 0.03 * page_w and (v["x2"] - v["x1"]) >= 0.15 * page_w)
    junk = [v for v in verses if not v["french"] and (v["garbage"] or (v["n_syl"] < 3 and not v["text_lost"] and not v["short_verse"]))]
    kept = [v for v in verses if not v["french"] and not v["garbage"] and (v["n_syl"] >= 3 or v["text_lost"] or v["short_verse"])]
    anchors = [(i, v["num"]) for i, v in enumerate(kept) if v["num"] is not None]
    return dict(header=header, verse_zone=vz, footnote=fn, verses=kept, leaks=leaks, junk=junk, anchors=anchors,
                n_kept=len(kept), n_leak=len(leaks), n_junk=len(junk), n_footnote=len(fn))


# ----------------------------------------------------------------------------------------------------------------
# 3. Phân loại canvas KVK (classify_kvk_full.py)
# ----------------------------------------------------------------------------------------------------------------

def hdr_pos(header_lines: list[dict]) -> str:
    for l in header_lines:
        toks = l["text"].split()
        if len(toks) < 2:
            continue
        idx = [i for i, w in enumerate(toks) if strip_accents(re.sub(r"[^\w]", "", w.lower())) in TITLE_WORDS]
        if not idx:
            continue
        first, last = idx[0], idx[-1]
        if first > 0 and re.search(r"\d", " ".join(toks[:first])):
            return "L"
        if last < len(toks) - 1:
            return "R"
    return "?"


def classify_page(r: dict) -> tuple[str, dict]:
    """QN: vùng thơ >= 3 dòng và tỷ lệ token Việt cao (kết hợp vị trí số trang: 'L' = trang chẵn = QN).
    FR: không phải QN, toàn trang (trừ tiêu đề) >= 3 dòng, hư từ Pháp >= 12% token và token Việt < 30%.
    other: bìa, tựa, mục lục, trang trắng, khảo cứu ngắn."""
    toks = [t for v in r["verses"] for t in tokens(v["text"])]
    vi = sum(1 for t in toks if any(c in VI_SPEC for c in t)) / len(toks) if toks else 0.0
    n_kept, hp = r["n_kept"], hdr_pos(r["header"])
    body = r["verse_zone"] + r["footnote"]
    all_toks = [t for l in body for t in tokens(l["text"])]
    fr_all = sum(1 for t in all_toks if t in FR_STOP) / len(all_toks) if all_toks else 0.0
    vi_all = sum(1 for t in all_toks if any(c in VI_SPEC for c in t)) / len(all_toks) if all_toks else 0.0
    if n_kept >= 3 and ((hp == "L" and vi >= 0.30) or (hp != "R" and vi >= 0.45)):
        cls = "QN"
    elif n_kept >= 2 and hp == "L" and vi >= 0.30 and vi_all >= 0.15:
        cls = "QN"          # trang chỉ còn 2 câu rồi tới khối trích dẫn Hán (KVK vol2 canvas 301)
    elif len(body) >= 3 and fr_all >= 0.12 and vi_all < 0.30:
        cls = "FR"
    else:
        cls = "other"
    return cls, dict(vi_tok=round(vi, 3), hdr=hp, fr_all=round(fr_all, 3), vi_all=round(vi_all, 3))


# ----------------------------------------------------------------------------------------------------------------
# 4. Chuỗi số câu toàn sách (chain_analysis.py)
# ----------------------------------------------------------------------------------------------------------------

def run_chain(pages: list[dict], use_margin: bool = True) -> tuple[list[dict], Counter]:
    """pages: [{tag, kept:[verse...], margin:{idx: 'digits'}}] theo thứ tự đọc. Gán start/end/drift/flag cho từng trang.
    Neo = (idx, giá trị, nguồn) từ 2 nguồn độc lập: số đầu dòng do tess4 đọc (inline) và số lề đọc riêng (margin, --psm 7).
    Quy tắc:  (1) VỊ TRÍ: đa số neo phải nằm ở dòng có (s+idx) % 5 == 0 -> dịch s trong [-2, 2] nếu cần.
              (2) GIÁ TRỊ: >= 2 neo cùng ngụ ý s' (s' - s chia hết 5, |s'-s| <= 25) -> đặt lại s = s'.
                  Lệch không chia hết 5 = đọc sai số (5->6, 2->3), giữ s, ghi 'value_conflict'.
              (3) 1 neo lệch = 'ambiguous'; không neo nào = 'no_anchor'."""
    s = 1
    st: Counter = Counter()
    calibrated = False      # chưa gặp trang có >= 2 neo đồng thuận: cho phép nhảy xa (chạy thử với --start giữa sách)
    for p in pages:
        kept = p["kept"]
        n = len(kept)
        inline = [(i, v["num"], "inline") for i, v in enumerate(kept) if v["num"] is not None]
        margin = []
        if use_margin:
            for i, g in sorted(p.get("margin", {}).items()):
                if g and 2 <= len(g) <= 4:
                    margin.append((i, int(g), "margin"))
        anchors = inline + margin
        # hai nguồn cùng dòng cùng giá trị -> 1 neo có 2 phiếu (đếm ở Counter), giữ nguyên
        drift = None
        flag = ""
        if anchors:
            pos = Counter((-i - s) % 5 for i, _, _ in anchors)
            d_mod, cnt_pos = pos.most_common(1)[0]
            d = d_mod if d_mod <= 2 else d_mod - 5
            if d != 0 and cnt_pos >= max(1, (len(anchors) + 1) // 2):
                # 1 neo duy nhất chỉ được dịch chuỗi nếu giá trị đọc được xác nhận vị trí mới ("8o"->80 giả không được dịch)
                if len(anchors) >= 2 or anchors[0][1] == s + d + anchors[0][0]:
                    drift = d; s += d
                else:
                    st["ambiguous_pages"] += 1; flag = "ambiguous"
            implied = Counter(num - i for i, num, _ in anchors)
            best, cnt = implied.most_common(1)[0]
            if best != s and cnt >= 2 and (abs(best - s) <= 25 or not calibrated) and (best - s) % 5 == 0:
                drift = (drift or 0) + (best - s); s = best
            if cnt >= 2 and best == s:
                calibrated = True
            elif best != s and cnt >= 2:
                st["value_conflict_pages"] += 1; flag = "value_conflict"
            elif best != s and len(anchors) == 1:
                st["ambiguous_pages"] += 1; flag = "ambiguous"
            if drift:
                st["drift_pages"] += 1; st["drift_abs"] += abs(drift); flag = (flag + ";" if flag else "") + "drift"
        elif n:
            st["no_anchor_pages"] += 1; flag = "no_anchor"
        p.update(start=s, n=n, end=s + n - 1, drift=drift, flag=flag,
                 anchors=[(i, num) for i, num, _ in inline], margin_anchors=[(i, num) for i, num, _ in margin])
        for src, lst in (("inline", inline), ("margin", margin)):
            st[f"{src}_anchors"] += len(lst)
            st[f"{src}_ok"] += sum(1 for i, num, _ in lst if num == s + i)
            st[f"{src}_pos_ok"] += sum(1 for i, num, _ in lst if (s + i) % 5 == 0)
        both = {i for i, _, _ in inline} & {i for i, _, _ in margin}
        st["both_read"] += len(both)
        st["both_agree"] += sum(1 for i in both if kept[i]["num"] == int(p["margin"][i]))
        st["both_agree_and_ok"] += sum(1 for i in both if kept[i]["num"] == int(p["margin"][i]) == s + i)
        st["anchors"] += len(anchors)
        st["anchors_ok"] += sum(1 for i, num, _ in anchors if num == s + i)
        st["anchors_pos_ok"] += sum(1 for i, num, _ in anchors if (s + i) % 5 == 0)
        st["expected_anchor_lines"] += sum(1 for k in range(s, s + n) if k % 5 == 0)
        st["anchor_lines_covered"] += len({i for i, _, _ in anchors if (s + i) % 5 == 0})
        st["lines"] += n
        s += n
    st["total_chain"] = s - 1
    return pages, st


def expect_syl(verse_no: int, flip_at: int | None) -> int:
    phase = 1 if (flip_at is not None and verse_no >= flip_at) else 0
    return 6 if (verse_no + phase) % 2 == 1 else 8


# ----------------------------------------------------------------------------------------------------------------
# 5. Phương pháp 2: đọc số câu ở lề bằng tesseract --psm 7 chỉ chữ số (margin_number_ocr.py)
# ----------------------------------------------------------------------------------------------------------------

def margin_digits_page(p: dict, tmp_dir: Path, scale: int = 3) -> dict[int, str]:
    """Đọc dải lề trái của MỌI dòng thơ (độc lập với số đầu dòng tess4): {idx: chuỗi chữ số}. Cache theo md5 ảnh + (y1,h)."""
    cdir = CACHE_ROOT / "margin"
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"{p['md5']}.json"
    cache = {}
    if cpath.exists():
        try:
            cache = json.loads(cpath.read_text(encoding="utf-8"))
            if cache.get("version") != MARGIN_VERSION:
                cache = {}
        except Exception:
            cache = {}
    W, H = p["size"]
    kept = p["kept"]
    xs = [l["x1"] for l in p["lines"] if not re.match(r"^\s*[«‹]?\s*\d", l["text"]) and 0.2 * W < l["x1"] < 0.45 * W]
    x_text = float(np.median(xs)) if xs else 0.30 * W
    out: dict[int, str] = {}
    dirty = False
    im = None
    for i, v in enumerate(kept):
        key = f"{v['y1']}:{v['h']}"
        if key in cache:
            out[i] = cache[key]; continue
        if im is None:
            im = Image.open(p["path"]).convert("L")
        y1, y2 = v["y1"] - 8, v["y1"] + int(v["h"] * 1.8) + 4
        strip = im.crop((int(0.10 * W), max(0, y1), int(x_text) - 6, min(H, y2)))
        got = ""
        if strip.width >= 8 and strip.height >= 8:
            strip = ImageOps.autocontrast(strip).resize((strip.width * scale, strip.height * scale), Image.LANCZOS)
            sp = tmp_dir / f"strip_{p['md5'][:8]}_{i}.png"
            strip.save(sp)
            r = subprocess.run(["tesseract", str(sp), "-", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789"],
                               capture_output=True, text=True)
            sp.unlink(missing_ok=True)
            got = re.sub(r"\D", "", r.stdout)
        out[i] = got; cache[key] = got; dirty = True
    if dirty:
        cache["version"] = MARGIN_VERSION
        cpath.write_text(json.dumps(cache), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------------------------------------------
# 6. Kiểm tra CER trên 3 trang đọc mắt (qn_visual_truth.json, 44 dòng)
# ----------------------------------------------------------------------------------------------------------------

def _norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t.lower())
    t = re.sub(r"[«»‹›\"'!?;:,.()\[\]—–\-_*]", " ", t)
    return " ".join(t.split())


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def visual_cer(pages_by_tag: dict[str, dict]) -> dict | None:
    vt_path = Path(__file__).with_name("qn_visual_truth.json")
    if not vt_path.exists():
        return None
    vt = json.loads(vt_path.read_text(encoding="utf-8"))
    edits = chars = n_lines = matched = start_ok = n_pages = 0
    for tag, rec in vt.items():
        if tag.startswith("_") or tag not in pages_by_tag:
            continue
        p = pages_by_tag[tag]
        n_pages += 1
        start_ok += int(p["start"] == rec["start"])
        truth = rec["lines"]
        ocr = [v["text"] for v in p["kept"]]
        n_lines += len(truth)
        for i, t in enumerate(truth):
            if i < len(ocr):
                matched += 1
                edits += _lev(_norm(ocr[i]), _norm(t)); chars += len(_norm(t))
    if n_pages == 0:
        return None
    return dict(pages=n_pages, lines=n_lines, matched=matched, start_ok=start_ok, edits=edits, chars=chars,
                cer=round(edits / max(1, chars), 4))


# ----------------------------------------------------------------------------------------------------------------
# 7. Ảnh debug
# ----------------------------------------------------------------------------------------------------------------

def draw_debug(p: dict, out_path: Path, max_w: int = 1200):
    im = Image.open(p["path"]).convert("RGB")
    dr = ImageDraw.Draw(im)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=max(18, im.width // 60))
    except Exception:
        font = None
    for l in p["analysis"]["header"]:
        dr.rectangle([l["x1"], l["y1"], l["x2"], l["y2"]], outline=(0, 0, 255), width=3)
    for l in p["analysis"]["footnote"]:
        dr.rectangle([l["x1"], l["y1"], l["x2"], l["y2"]], outline=(220, 0, 0), width=3)
    for l in p["analysis"]["leaks"] + p["analysis"]["junk"]:
        dr.rectangle([l["x1"], l["y1"], l["x2"], l["y2"]], outline=(255, 140, 0), width=3)
    for i, v in enumerate(p["kept"]):
        dr.rectangle([v["x1"], v["y1"], v["x2"], v["y2"]], outline=(0, 160, 0), width=3)
        dr.text((max(0, v["x1"] - 90), v["y1"]), str(p["start"] + i), fill=(0, 120, 0), font=font)
    dr.text((10, 10), f"{p['tag']} cls={p.get('cls','QN')} start={p.get('start')} n={len(p['kept'])} flag={p.get('flag','')}",
            fill=(0, 0, 0), font=font)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    im.save(out_path, quality=80)


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------

def list_pages(cfg: dict) -> list[tuple[str, str, Path]]:
    """[(vol, tag, path)] theo thứ tự đọc cố định (sort theo tên)."""
    items = []
    for vol, d in cfg["qn_dirs"]:
        for f in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")):
            tag = f"{cfg['tag_prefix']}_{vol + '_' if vol else ''}{f.stem}"
            items.append((vol, tag, f))
    items.sort(key=lambda t: (t[0], t[2].name))
    return items


def canvas_no(path: Path) -> int | None:
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True, choices=sorted(BOOKS))
    ap.add_argument("--out", default=None, help="mặc định measure_out/<book>/qn_ocr/ (quy ước: mỗi phép đo một thư mục con)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ xử lý N trang (chạy thử)")
    ap.add_argument("--start", type=int, default=0, help="bỏ qua K trang đầu trước khi áp --limit (KVK: canvas 1..26 là bìa/tựa)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--engine", default="tesseract", choices=["tesseract", "vietocr"])
    ap.add_argument("--no-margin", action="store_true", help="bỏ phương pháp 2 (đọc số lề)")
    ap.add_argument("--debug-pages", type=int, default=5)
    args = ap.parse_args()

    cfg = BOOKS[args.book]
    out = Path(args.out) if args.out else REPO / "measure_out" / args.book / "qn_ocr"
    out.mkdir(parents=True, exist_ok=True)
    tmp_dir = CACHE_ROOT / "tmp"; tmp_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    items = list_pages(cfg)
    n_all = len(items)
    if args.start:
        items = items[args.start:]
    if args.limit:
        items = items[: args.limit]
    print(f"[qn_print_ocr] book={args.book} engine={args.engine} pages={len(items)}/{n_all} out={out}")

    # --- OCR (cache). Luôn chạy tesseract (rẻ, để phân loại + tham chiếu bước dòng); VietOCR chỉ trên trang QN ---
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        recs = list(ex.map(lambda it: ocr_page_cached((it[1], it[2]), "tesseract"), items))
    t_ocr = time.time() - t0
    n_fresh = sum(1 for r in recs if r.get("fresh"))

    # --- pass 1: bước dòng & cỡ chữ thơ toàn sách (trung vị các trang >= 4 dòng thơ) ---
    mets = [m for rec in recs for m in [page_metrics(rec["lines"], rec["size"][1], rec["size"][0])] if m]
    p_ref = float(np.median([m[0] for m in mets])) if len(mets) >= 3 else None
    h_ref = float(np.median([m[1] for m in mets])) if len(mets) >= 3 else None

    # --- pass 2: phân tích trang + phân loại ---
    pages = []
    for (vol, tag, path), rec in zip(items, recs):
        W, H = rec["size"]
        r = analyse_page(rec["lines"], H, W, p_ref, h_ref)
        if cfg["classify"]:
            cls, feats = classify_page(r)
        else:
            cls, feats = "QN", dict(vi_tok=None, hdr=None, fr_all=None, vi_all=None)
        pages.append(dict(vol=vol, tag=tag, path=str(path), md5=rec["md5"], size=rec["size"], lines=rec["lines"],
                          analysis=r, kept=r["verses"], cls=cls, canvas=canvas_no(path), **feats))

    cls_counter = Counter(p["cls"] for p in pages)
    qn_pages = [p for p in pages if p["cls"] == "QN"]

    # --- --engine vietocr: nhận dạng lại các trang QN bằng VietOCR (core/ocr/qn_ocr.py), giữ phân loại của tesseract ---
    n_fresh_v = 0
    if args.engine == "vietocr":
        t0 = time.time()
        for p in qn_pages:
            rec = ocr_page_cached((p["tag"], Path(p["path"])), "vietocr")
            n_fresh_v += int(bool(rec.get("fresh")))
            W, H = rec["size"]
            r = analyse_page(rec["lines"], H, W, p_ref, h_ref)
            p.update(lines=rec["lines"], analysis=r, kept=r["verses"])
        t_ocr += time.time() - t0
        n_fresh += n_fresh_v

    # --- phương pháp 2: số lề (mọi dòng thơ, song song, cache) ---
    margin = Counter()
    if not args.no_margin:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            for p, got in zip(qn_pages, ex.map(lambda q: margin_digits_page(q, tmp_dir), qn_pages)):
                p["margin"] = got
        t_margin = time.time() - t0
    else:
        t_margin = 0.0
        for p in qn_pages:
            p["margin"] = {}

    # --- chuỗi số câu (neo = inline ∪ margin) ---
    qn_pages, st = run_chain(qn_pages, use_margin=not args.no_margin)
    margin_rows = []
    for p in qn_pages:
        for i, v in enumerate(p["kept"]):
            g = p["margin"].get(i, "")
            truth = p["start"] + i
            if truth % 5 == 0:
                margin["expected"] += 1
                margin["read"] += int(bool(g))
                margin["ok"] += int(bool(g) and int(g) == truth)
                margin["inline_ok"] += int(v["num"] == truth)
                margin["both_ok"] += int(v["num"] == truth and bool(g) and int(g) == truth)
                margin["either_ok"] += int(v["num"] == truth or (bool(g) and int(g) == truth))
            else:
                margin["nonanchor_lines"] += 1
                margin["false_pos"] += int(len(g) >= 2)
            if v["num"] is not None or g:
                margin_rows.append((p["tag"], truth, v["num"] if v["num"] is not None else "", g))

    # --- verses.tsv ---
    verses_path = out / "verses.tsv"
    n_par_ok = n_any68 = 0
    conf_vals = []
    with open(verses_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["verse_no", "page", "line_text", "n_syll", "anchor_source", "confidence", "seq_no", "expect_syll", "num_read", "margin_read",
                    "line_flag", "page_flag"])
        seq = 0
        for p in qn_pages:
            for i, v in enumerate(p["kept"]):
                seq += 1
                vn = p["start"] + i
                mg = p.get("margin", {}).get(i, "")
                if v["num"] == vn:
                    src = "ocr_num"
                elif mg and int(mg) == vn:
                    src = "margin_digits"
                elif vn % 5 == 0:
                    src = "position_mod5"
                else:
                    src = "chain_interp"
                exp = expect_syl(vn, cfg["parity_flip_at"])
                n_par_ok += int(v["n_syl"] == exp); n_any68 += int(v["n_syl"] in (6, 8))
                conf = round(v["conf"] / 100.0, 3)
                conf_vals.append(conf)
                lflag = "text_lost" if v["text_lost"] else ("short" if v.get("short_verse") else "")
                w.writerow([vn, p["tag"], v["text"], v["n_syl"], src, conf, seq, exp, v["num"] if v["num"] is not None else "", mg, lflag, p["flag"]])

    # --- pages.csv + review list ---
    with open(out / "pages.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "vol", "canvas", "cls", "vi_tok", "hdr", "fr_all", "vi_all", "n_kept", "n_footnote", "n_leak", "n_junk", "start", "end", "n_anchor",
                    "n_anchor_ok", "drift", "flag", "md5"])
        for p in pages:
            anc = p.get("anchors", [])
            w.writerow([p["tag"], p["vol"], p["canvas"], p["cls"], p["vi_tok"], p["hdr"], p["fr_all"], p["vi_all"], p["analysis"]["n_kept"], p["analysis"]["n_footnote"],
                        p["analysis"]["n_leak"], p["analysis"]["n_junk"], p.get("start", ""), p.get("end", ""), len(anc),
                        sum(1 for i, n in anc if n == p.get("start", -1) + i), p.get("drift", ""), p.get("flag", ""), p["md5"]])
    review = [dict(tag=p["tag"], start=p["start"], n=p["n"], drift=p["drift"], flag=p["flag"], anchors=p["anchors"])
              for p in qn_pages if p["flag"]]
    (out / "pages_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    if margin_rows:
        with open(out / "anchors.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh); w.writerow(["tag", "verse_no_chain", "inline_read", "margin_read"]); w.writerows(margin_rows)

    # --- ảnh debug ---
    dbg_dir = out / "debug"; dbg_dir.mkdir(exist_ok=True)
    for f in dbg_dir.glob("*.jpg"):
        f.unlink()
    prio = {"drift": 0, "value_conflict": 1, "ambiguous": 2, "no_anchor": 3}
    flagged = sorted((p for p in qn_pages if p["flag"]), key=lambda p: min(prio.get(f, 9) for f in p["flag"].split(";")))
    chosen = (flagged[:3] + [p for p in qn_pages if not p["flag"]][:1] + [p for p in pages if p["cls"] != "QN"][:1])[: args.debug_pages]
    if len(chosen) < args.debug_pages:
        chosen += [p for p in qn_pages if p not in chosen][: args.debug_pages - len(chosen)]
    for p in chosen:
        p.setdefault("start", 0)
        draw_debug(p, dbg_dir / f"{p['tag']}.jpg")

    # --- CER trên trang đọc mắt ---
    vis = visual_cer({p["tag"]: p for p in qn_pages})

    # --- invariants ---
    full = args.limit is None
    n_lines = st["lines"]
    inv = []

    def add(name, expected, observed, ok, note=""):
        d = dict(name=name, expected=expected, observed=observed, **{"pass": (None if not full and note == "book" else bool(ok))})
        if not full and note == "book":
            d["note"] = "bỏ qua khi --limit"
        inv.append(d)

    if cfg["classify"]:
        add("n_canvases", cfg["expected_canvases"], n_all, n_all == cfg["expected_canvases"], "book")
        add("n_qn_pages", cfg["expected_qn_pages"], len(qn_pages), len(qn_pages) == cfg["expected_qn_pages"], "book")
        for vol, ev in cfg["expected_pages_per_vol"].items():
            nv = sum(1 for p in qn_pages if p["vol"] == vol)
            add(f"n_qn_pages_{vol}", ev, nv, nv == ev, "book")
        odd = sum(1 for p in qn_pages if p["canvas"] is not None and p["canvas"] % 2 == 1)
        add("qn_canvases_all_odd", "100%", f"{odd}/{len(qn_pages)}", odd == len(qn_pages))
    else:
        add("n_qn_pages", cfg["expected_qn_pages"], len(qn_pages), len(qn_pages) == cfg["expected_qn_pages"], "book")
    add("lines_total_vs_expected", cfg["expected_lines"], n_lines, n_lines == cfg["expected_lines"], "book")
    add("chain_total_vs_printed_last", cfg["printed_last"], st["total_chain"], st["total_chain"] == cfg["printed_last"], "book")
    last_anchor = max((num for p in qn_pages for _, num in p["anchors"] + p["margin_anchors"] if num <= cfg["printed_last"]), default=None)
    add("last_anchor_value", cfg["printed_last"] - cfg["printed_last"] % 5, last_anchor, last_anchor == cfg["printed_last"] - cfg["printed_last"] % 5, "book")
    seq = [p["start"] + i for p in qn_pages for i in range(p["n"])]
    n_nonmono = sum(1 for a, b in zip(seq, seq[1:]) if b <= a)
    n_gap = sum(1 for a, b in zip(seq, seq[1:]) if b > a + 1)
    known = cfg["known_drift_pages"]
    k_gap = sum(1 for d in known.values() if d > 0); k_ovl = sum(1 for d in known.values() if d < 0)
    add("verse_no_strictly_increasing", f"{k_ovl} chỗ lùi/lặp (bất thường đánh số đã biết của ấn bản)", n_nonmono, n_nonmono == k_ovl)
    add("verse_no_no_gap", f"{k_gap} chỗ nhảy số (số in bỏ qua, đã biết)", n_gap, n_gap == k_gap)
    drift_pages = {p["tag"]: p["drift"] for p in qn_pages if p["drift"]}
    add("drift_pages_only_known", sorted(known.items()), sorted(drift_pages.items()),
        (drift_pages == known) if full else set(drift_pages.items()) <= set(known.items()))
    lo, hi = cfg["lines_per_page"]
    add("lines_per_page_range", f"{lo}..{hi}", f"min={min((p['n'] for p in qn_pages), default=0)} max={max((p['n'] for p in qn_pages), default=0)}",
        all(lo <= p["n"] <= hi for p in qn_pages))
    add("anchor_position_mod5", ">= 0.95", round(st["anchors_pos_ok"] / max(1, st["anchors"]), 3), st["anchors_pos_ok"] >= 0.95 * st["anchors"])
    add("anchor_value_inline_ok", ">= 0.65", round(st["anchors_ok"] / max(1, st["anchors"]), 3), st["anchors_ok"] >= 0.65 * st["anchors"])
    add("inline_anchor_lines_found", ">= 0.75 của số dòng lẽ ra có số", round(st["inline_anchors"] / max(1, st["expected_anchor_lines"]), 3),
        st["inline_anchors"] >= 0.75 * st["expected_anchor_lines"])
    add("parity_6_8_by_verse_no", ">= 0.85", round(n_par_ok / max(1, n_lines), 3), n_par_ok >= 0.85 * n_lines)
    add("any_6_or_8", ">= 0.90", round(n_any68 / max(1, n_lines), 3), n_any68 >= 0.90 * n_lines)
    n_leak = sum(p["analysis"]["n_leak"] for p in qn_pages)
    add("french_leaks_in_verse_zone", "<= 2 (đếm rồi loại, không vào verses.tsv)", n_leak, n_leak <= 2)
    if not args.no_margin:
        add("two_methods_agree_rate", ">= 0.60 (inline == margin cùng dòng; mỗi bên đúng ~75-80%)", f"{st['both_agree']}/{st['both_read']}",
            st["both_agree"] >= 0.60 * max(1, st["both_read"]))
        add("agreed_value_is_correct", ">= 0.97 (hai bên khớp nhau => khớp chuỗi)", f"{st['both_agree_and_ok']}/{st['both_agree']}",
            st["both_agree_and_ok"] >= 0.97 * max(1, st["both_agree"]))
        add("anchor_lines_covered_by_either_method", ">= 0.85 của số dòng lẽ ra có số", round(st["anchor_lines_covered"] / max(1, st["expected_anchor_lines"]), 3),
            st["anchor_lines_covered"] >= 0.85 * st["expected_anchor_lines"])
        add("margin_false_positive_on_nonanchor_lines", "<= 0.03", round(margin["false_pos"] / max(1, margin["nonanchor_lines"]), 4),
            margin["false_pos"] <= 0.03 * max(1, margin["nonanchor_lines"]))
    if vis:
        add("visual_truth_cer", "<= 0.10 (tess) / <= 0.20 (vietocr)", vis["cer"], vis["cer"] <= (0.10 if args.engine == "tesseract" else 0.20))
        add("visual_truth_start_ok", vis["pages"], vis["start_ok"], vis["start_ok"] == vis["pages"])

    # --- summary.json ---
    n_flag = Counter()
    for p in qn_pages:
        for f in filter(None, p["flag"].split(";")):
            n_flag[f] += 1
    per_vol = {vol: dict(qn=sum(1 for p in qn_pages if p["vol"] == vol), lines=sum(p["n"] for p in qn_pages if p["vol"] == vol))
               for vol, _ in cfg["qn_dirs"]} if cfg["classify"] else None
    summary = dict(
        book=args.book, engine=args.engine, run_at=time.strftime("%Y-%m-%d %H:%M:%S"), limit=args.limit,
        pages=dict(listed=n_all, processed=len(items), qn=len(qn_pages), by_class=dict(cls_counter), per_vol=per_vol,
                   verse_pitch_px=p_ref, verse_height_px=h_ref, n_pages_for_ref=len(mets),
                   method="tess4 TSV -> tách vùng -> (vị trí số trang trong tiêu đề L/R + tỷ lệ token Việt trong vùng thơ)"),
        ocr=dict(engine=args.engine, version=TESS_VERSION if args.engine == "tesseract" else VIETOCR_VERSION,
                 pages_fresh=n_fresh, pages_cached=len(items) - n_fresh, t_ocr_s=round(t_ocr, 1),
                 s_per_fresh_page=round(t_ocr / max(1, n_fresh), 2) if n_fresh else None, cache_dir=str(CACHE_ROOT)),
        verses=dict(n=n_lines, expected=cfg["expected_lines"], chain_total=st["total_chain"],
                    last_anchor_value=last_anchor,  # neo lớn nhất ≤ số in cuối (loại đọc sai kiểu 2085→9085)
                    last_anchor_raw_max=max((num for p in qn_pages for _, num in p["anchors"]), default=None),
                    parity_ok=n_par_ok, parity_rate=round(n_par_ok / max(1, n_lines), 4),
                    any68=n_any68, any68_rate=round(n_any68 / max(1, n_lines), 4),
                    conf_mean=round(float(np.mean(conf_vals)), 3) if conf_vals else None,
                    conf_p10=round(float(np.percentile(conf_vals, 10)), 3) if conf_vals else None,
                    n_text_lost=sum(1 for p in qn_pages for v in p["kept"] if v["text_lost"]),
                    n_leak=sum(p["analysis"]["n_leak"] for p in qn_pages), n_junk=sum(p["analysis"]["n_junk"] for p in qn_pages),
                    n_footnote_lines=sum(p["analysis"]["n_footnote"] for p in qn_pages),
                    method="chuỗi s_{k+1}=s_k+n_k, đặt lại theo đa số neo (vị trí mod 5 trước, giá trị sau, |delta|<=25)"),
        anchors=dict(n=st["anchors"], expected_lines=st["expected_anchor_lines"], covered_lines=st["anchor_lines_covered"],
                     value_ok=st["anchors_ok"], position_ok=st["anchors_pos_ok"],
                     value_ok_rate=round(st["anchors_ok"] / max(1, st["anchors"]), 4), position_ok_rate=round(st["anchors_pos_ok"] / max(1, st["anchors"]), 4),
                     inline=dict(n=st["inline_anchors"], value_ok=st["inline_ok"], position_ok=st["inline_pos_ok"]),
                     margin=dict(n=st["margin_anchors"], value_ok=st["margin_ok"], position_ok=st["margin_pos_ok"]),
                     method="inline: token đầu dòng tess4 (DIGIT_MAP ö/õ/ð->5, O->0, l/I->1); margin: --psm 7 chữ số; position: (s+idx)%5==0"),
        margin_digits=(dict(n_expected=margin["expected"], read=margin["read"], ok=margin["ok"], inline_ok=margin["inline_ok"],
                            both_ok=margin["both_ok"], either_ok=margin["either_ok"], false_pos=margin["false_pos"],
                            nonanchor_lines=margin["nonanchor_lines"], t_s=round(t_margin, 1),
                            agreement=dict(both_read=st["both_read"], both_agree=st["both_agree"], both_agree_and_ok=st["both_agree_and_ok"],
                                           rate=round(st["both_agree"] / max(1, st["both_read"]), 4)),
                            method="dải lề [0.10W, x_text-6] × [y1-8, y1+1.8h+4] mỗi dòng thơ, autocontrast, x3, tesseract --psm 7 whitelist 0-9; neo nếu 2-4 chữ số")
                       if not args.no_margin else "skipped"),
        flags=dict(drift_pages=st.get("drift_pages", 0), drift_abs=st.get("drift_abs", 0), ambiguous_pages=st.get("ambiguous_pages", 0),
                   value_conflict_pages=st.get("value_conflict_pages", 0), no_anchor_pages=st.get("no_anchor_pages", 0),
                   review_list=[r["tag"] for r in review][:40], review_file="pages_review.json"),
        visual_truth=vis or "skipped (3 trang mẫu không nằm trong tập xử lý)",
        invariants=inv,
        outputs=dict(verses="verses.tsv", pages="pages.csv", anchors="anchors.csv", review="pages_review.json", debug=[p.name for p in sorted(dbg_dir.glob("*.jpg"))]),
        t_total_s=round(time.time() - t_start, 1),
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- stdout <= 40 dòng ---
    print(f"pages: {dict(cls_counter)}  qn={len(qn_pages)}  ocr {t_ocr:.0f}s (fresh {n_fresh})")
    print(f"verses: n={n_lines} expected={cfg['expected_lines']} chain_total={st['total_chain']} parity={n_par_ok/max(1,n_lines):.1%} any68={n_any68/max(1,n_lines):.1%}")
    print(f"anchors: {st['anchors']}/{st['expected_anchor_lines']} value_ok={st['anchors_ok']} pos_ok={st['anchors_pos_ok']}  flags={dict(n_flag)}")
    if not args.no_margin:
        print(f"margin_digits ({t_margin:.0f}s): anchor lines expected={margin['expected']} margin_ok={margin['ok']} inline_ok={margin['inline_ok']} "
              f"either={margin['either_ok']} false_pos={margin['false_pos']}/{margin['nonanchor_lines']} both_read={st['both_read']} agree={st['both_agree']}")
    if vis:
        print(f"visual_truth: CER={vis['cer']} start_ok={vis['start_ok']}/{vis['pages']}")
    for d in inv:
        mark = "PASS" if d["pass"] else ("skip" if d["pass"] is None else "FAIL")
        print(f"  [{mark}] {d['name']}: expected={d['expected']} observed={d['observed']}")
    print(f"summary: {out / 'summary.json'} ({(out / 'summary.json').stat().st_size} B)  total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
