#!/usr/bin/env python3
"""stt_lt2_eval.py — LẦN ĐỌC THỨ HAI của kim ở chế độ NÔM (lang_type=2, "lt2") cho bộ STT (2026-09-28). 0 API, CPU ~10 s.

Bối cảnh: STT (SachThanhTruyen2/4/11, 448 trang chép tay) được OCR bằng kim lang_type=1 (Hán, "lt1"):
prepared/<Sách>/detected/<page>_ocr_cache.json. Nhãn GOLD của STT = chữ lt1 tại nom_idx của cột Nôm (ocr_char).
pipeline/tools/stt_reocr_lt2.py OCR lại ĐÚNG ảnh, ĐÚNG quy trình, chỉ đổi lang_type=2 →
prepared/<Sách>/kim_raw_lt2/<page>_lt2.json (cùng lược đồ + kim_params). Script này CHỈ ĐỌC các tệp đó (không ghi prepared/).

Hộp chữ lt1 của ô: cột kim dựng lại bằng align_production._detect như gold_exact/signals_geom.kim_vs_box (đọc cache
prepared/_gold_exact/kim_pages/<Sách>/<page>.pkl; thiếu thì gọi _detect; `--detect-check N` gọi lại _detect trên N trang
để kiểm cache) → chars[nom_idx]; kiểm chữ == ocr_char của ô (invariant).

Tìm chữ lt2 CÙNG VỊ TRÍ — hai phương pháp + một đối chứng (kim trả HỘP DÒNG, hộp chữ = chia đều hộp dòng theo số chữ
đọc được, core.ocr.ocr_api.boxes_to_columns; lt2 hay đọc ÍT chữ hơn lt1 trên CÙNG hộp dòng ⇒ hộp chữ trôi):
  line  (chính) dòng kim chứa ô ↔ dòng lt2 có IoU hộp dòng ≥ 0,5 (hộp dòng hai lần đọc gần như trùng: cùng bộ dò), rồi căn
        chuỗi chữ hai dòng (Needleman–Wunsch có dải |j − i·n2/n1| ≤ |n1−n2|+2; khớp V1+ +2, lệch −1, khoảng −1) → chữ lt2 cặp
        với vị trí của ô; ô rơi vào khoảng trống = "gap" (lt2 không đọc/gộp chữ đó). Căn chuỗi ưu tiên khớp ⇒ tỉ lệ trùng
        nghiêng LÊN (cận trên).
  geom  (theo đặc tả ban đầu) chữ lt2 CÙNG CỘT theo x (chồng ngang ≥ 50 % bề ngang hẹp hơn), GẦN NHẤT theo tâm y, nhận nếu
        chồng dọc ≥ 50 % chiều cao chữ lt1. Không nhìn nội dung chữ nhưng trôi khi số chữ/dòng khác ⇒ nghiêng XUỐNG (cận dưới).
  same_count (đối chứng) chỉ các dòng hai lần đọc CÙNG số chữ → cặp theo chỉ số (tương ứng chắc chắn, không căn).
  Đối chứng tự thân: ghép hộp lt1 với chính chữ lt1 của trang phải cho 100 % ghép + 100 % trùng.

Đo (theo stt2/stt4/stt11 + tổng):
  1. độ phủ cache lt2 (trang), tỉ lệ ô ghép được chữ lt2;
  2. lt1 == lt2 (strict, V1+ = pipeline/gold_exact/common.var_eq_plus) theo tầng, luật, trạng thái gold_exact;
  3. R(âm) như tier_v3 (common.R_of = core.text.dictionary.load_qn_to_nom + normalize_tone_marks): ô lt1 ∉ R, lt2 ∈ R
     (ứng viên cứu thêm) · ô lt1 ∈ R, lt2 ∈ R, lt2 ≠ lt1 V1+ (xung đột trong R — nghi nhãn sai);
  4. tỉ lệ chữ Nôm riêng (CJK Ext-B..I, PUA) lt1 vs lt2 (ô ghép + cả trang); 20 cặp lt1→lt2 lệch hay gặp nhất;
  5. HIỆU CHUẨN trên bộ có CẢ lt1 và lt2 cùng trang + chữ tham chiếu: LucVanTien1883 / KimVanKieu1884 (THẠCH BẢN in;
     kim_raw/<page>.json = lt1, kim_raw/<page>_lt2.json = lt2; tham chiếu = chữ DỊ BẢN measure_out/auto_precision/cross;
     ô chuẩn = hộp lt2 của build hiện hành, tìm chữ lt1 bằng CÙNG hàm ghép) → P(lt1 đúng | lt1 == lt2) vs
     P(lt1 đúng | lt1 ≠ lt2) (V1+, Wilson + bootstrap cụm trang). IHR (LVT1916/TK1872, nhãn người) và Borg chỉ có lt2 →
     KHÔNG hiệu chuẩn được trên nhãn người / chữ viết tay. Đề xuất cổng "gold_exact ok chỉ khi lt1 == lt2 (V1+)": số ô ok
     còn lại theo bộ + độ chính xác ƯỚC LƯỢNG (thạch bản so dị bản, cận dưới) — với chữ viết tay STT chỉ là SUY ĐOÁN.

Chạy:
  .venv/bin/python scripts/measure/stt_lt2_eval.py                       # phần cache lt2 đang có ("MỘT PHẦN" nếu < 448 trang)
  .venv/bin/python scripts/measure/stt_lt2_eval.py --require-full        # FAIL nếu cache lt2 chưa đủ 448 trang
  .venv/bin/python scripts/measure/stt_lt2_eval.py --match geom          # dùng phương pháp geom làm chính
  .venv/bin/python scripts/measure/stt_lt2_eval.py --book stt2 --limit 5 # chạy thử 5 trang/bộ (invariant toàn bộ = SKIP)
  .venv/bin/python scripts/measure/stt_lt2_eval.py --selftest
Đầu ra: measure_out/stt_lt2/{summary.json (≤ 8 KB), breakdown.json, cells.csv, calib_cells.csv}.
Mã thoát: 0 mọi invariant cứng PASS · 1 có FAIL · 2 thiếu dữ liệu đầu vào.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BOOKS = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}
N_PAGES_STT = 448
LT2_DIR = "kim_raw_lt2"
PKL_ROOT = REPO / "prepared" / "_gold_exact" / "kim_pages"
STT_CFG = "config/pipeline.yaml"
# bộ hiệu chuẩn: có kim_raw/<page>.json (lt1) + kim_raw/<page>_lt2.json (lt2) + chữ dị bản (auto_precision/cross)
CALIB = {"LucVanTien1883": "config/pipeline_LucVanTien1883.yaml",
         "KimVanKieu1884": "config/pipeline_KimVanKieu1884_b1.yaml"}
CROSS = REPO / "measure_out" / "auto_precision" / "cross"
DEFAULT_OUT = REPO / "measure_out" / "stt_lt2"
X_OV, Y_OV, LINE_IOU = 0.5, 0.5, 0.5
NOM_RANGES = [(0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF),
              (0x2EBF0, 0x2EE5F), (0x30000, 0x3134F), (0x31350, 0x323AF),            # Ext-B..I
              (0xE000, 0xF8FF), (0xF0000, 0x10FFFD)]                                   # PUA (BMP + vùng 15/16)
GE_ORDER = ["ok", "text_only", "uncertified", "review", "-"]   # "-" = ô không GOLD (SYLLABLE)
_C = {}


# ------------------------------------------------------------------------------------------------ tiện ích
def rd(path, **k) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **k)


def rate(k, n) -> dict:
    return dict(n=int(n), k=int(k), p=(round(k / n, 4) if n else None))


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(float(max(0.0, (c - r) / d)), 4), round(float(min(1.0, (c + r) / d)), 4)]


def nom_rieng(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch[0])
    return any(a <= o <= b for a, b in NOM_RANGES)


def lexicon():
    if "C" not in _C:
        from pipeline.gold_exact import common as C
        C.set_lexicon(C.Assets(verify=True))
        _C["C"] = C
    return _C["C"]


@lru_cache(maxsize=None)
def veq(a: str, b: str) -> bool:
    """V1+ (gold_exact.common.var_eq_plus), có nhớ đệm."""
    return bool(a) and bool(b) and (a == b or lexicon().var_eq_plus(a, b))


def sha_files(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def lt1_caches(book: str) -> list:
    return sorted((REPO / "prepared" / BOOKS[book] / "detected").glob("*_ocr_cache.json"))


def page_of(p: Path) -> str:
    return p.name[: -len("_ocr_cache.json")]


# ------------------------------------------------------------------------------------------------ hình học dòng / chữ kim
def raw_lines(boxes) -> list:
    """Hộp dòng kim thô -> [{x1,y1,x2,y2,s:[chữ]}] (cùng quy ước toạ độ của core.ocr.ocr_api.boxes_to_columns)."""
    out = []
    for box in boxes or []:
        valid = [ch for ch in (box.get("transcription") or "").strip() if ch.strip()]
        if not valid:
            continue
        p = box["points"]
        out.append(dict(x1=p[0][0], y1=p[0][1], x2=p[1][0], y2=p[2][1], s=valid))
    return out


def sub_box(L: dict, i: int) -> list:
    """Hộp chữ thứ i của dòng = chia đều (y_top + h·i ... ) như boxes_to_columns."""
    h = (L["y2"] - L["y1"]) / len(L["s"])
    return [L["x1"], int(L["y1"] + h * i), L["x2"], int(L["y1"] + h * (i + 1))]


def chars_from_raw(boxes) -> list:
    return [(L["s"][i], sub_box(L, i)) for L in raw_lines(boxes) for i in range(len(L["s"]))]


def flat_cols(columns) -> list:
    return [(c.get("char") or "", list(c["bbox"][:4])) for col in columns or [] for c in col if c.get("bbox")]


def match_box(ref, C: np.ndarray):
    """PHƯƠNG PHÁP geom. ref [x1,y1,x2,y2]; C (n,4). Cùng cột theo x (chồng ngang ≥ X_OV · bề ngang hẹp hơn) → gần nhất
    theo tâm y → nhận nếu chồng dọc ≥ Y_OV · chiều cao chữ ref. Trả (idx | None, lý_do)."""
    if C is None or len(C) == 0:
        return None, "no_x"
    x1, y1, x2, y2 = (float(v) for v in ref)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    ox = np.minimum(C[:, 2], x2) - np.maximum(C[:, 0], x1)
    same = ox >= X_OV * np.minimum(w, np.maximum(1.0, C[:, 2] - C[:, 0]))
    if not same.any():
        return None, "no_x"
    idx = np.nonzero(same)[0]
    j = idx[np.argmin(np.abs((C[idx, 1] + C[idx, 3]) / 2 - (y1 + y2) / 2))]
    oy = max(0.0, min(C[j, 3], y2) - max(C[j, 1], y1))
    return (int(j), "ok") if oy / h >= Y_OV else (None, "low_overlap")


def locate(ch: str, bbox, lines: list):
    """Dòng + chỉ số của chữ `ch` có hộp `bbox`: dòng chồng ngang ≥ 50 % và chứa tâm y; chỉ số = vị trí chữ `ch` gần ước
    lượng hình học nhất (≤ 2,5 chữ). STT: hộp = đúng ô chia đều ⇒ chính xác; thạch bản: hộp cắt lại theo tầng ⇒ neo bằng chữ."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cy, best = (y1 + y2) / 2, None
    for li, L in enumerate(lines):
        ox = min(x2, L["x2"]) - max(x1, L["x1"])
        if ox < 0.5 * max(1.0, min(x2 - x1, L["x2"] - L["x1"])) or not (L["y1"] - 1 <= cy <= L["y2"] + 1):
            continue
        n = len(L["s"]); est = (cy - L["y1"]) / ((L["y2"] - L["y1"]) / n) - 0.5
        cands = [k for k, c in enumerate(L["s"]) if c == ch]
        if not cands:
            continue
        k = min(cands, key=lambda k: abs(k - est))
        if abs(k - est) <= 2.5 and (best is None or abs(k - est) < best[2]):
            best = (li, k, abs(k - est))
    return best[:2] if best else None


def best_line(L: dict, lines: list):
    """Dòng bên kia có IoU hộp dòng lớn nhất (≥ LINE_IOU)."""
    a = (L["x2"] - L["x1"]) * (L["y2"] - L["y1"])
    bi, bv = None, 0.0
    for i, M in enumerate(lines):
        iw = min(L["x2"], M["x2"]) - max(L["x1"], M["x1"]); ih = min(L["y2"], M["y2"]) - max(L["y1"], M["y1"])
        if iw <= 0 or ih <= 0:
            continue
        inter = iw * ih
        v = inter / (a + (M["x2"] - M["x1"]) * (M["y2"] - M["y1"]) - inter)
        if v > bv:
            bi, bv = i, v
    return bi if bv >= LINE_IOU else None


def align(s1: list, s2: list, eq=veq) -> list:
    """Needleman–Wunsch có dải: khớp(eq) +2, lệch −1, khoảng −1; ưu tiên chéo khi hoà. -> map[i] = j | None."""
    n1, n2 = len(s1), len(s2)
    if n1 == 0:
        return []
    band = abs(n1 - n2) + 2
    NEG = -1e9
    S = np.full((n1 + 1, n2 + 1), NEG)
    B = np.zeros((n1 + 1, n2 + 1), np.int8)       # 1 chéo, 2 lên (gap ở s2), 3 trái (gap ở s1)
    S[0, 0] = 0.0
    for i in range(n1 + 1):
        c = i * n2 / n1
        for j in range(max(0, int(np.floor(c - band))), min(n2, int(np.ceil(c + band))) + 1):
            if i == 0 and j == 0:
                continue
            best, mv = NEG, 0
            if i > 0 and j > 0 and S[i - 1, j - 1] > NEG:
                v = S[i - 1, j - 1] + (2.0 if eq(s1[i - 1], s2[j - 1]) else -1.0)
                if v > best:
                    best, mv = v, 1
            if i > 0 and S[i - 1, j] > NEG and S[i - 1, j] - 1.0 > best:
                best, mv = S[i - 1, j] - 1.0, 2
            if j > 0 and S[i, j - 1] > NEG and S[i, j - 1] - 1.0 > best:
                best, mv = S[i, j - 1] - 1.0, 3
            S[i, j], B[i, j] = best, mv
    m = [None] * n1
    i, j = n1, n2
    if S[i, j] <= NEG:                               # ngoài dải (không xảy ra với dải trên) -> cặp theo chỉ số
        return [k if k < n2 else None for k in range(n1)]
    while i > 0 or j > 0:
        mv = B[i, j]
        if mv == 1:
            m[i - 1] = j - 1; i, j = i - 1, j - 1
        elif mv == 2:
            i -= 1
        else:
            j -= 1
    return m


class PagePair:
    """Một trang: bên CHUẨN (hộp ô đã biết) và bên KIA (lần đọc còn lại). Hai phương pháp tìm chữ bên kia."""

    def __init__(self, ref_boxes, oth_boxes):
        self.ref, self.oth = raw_lines(ref_boxes), raw_lines(oth_boxes)
        self.oflat = [(L["s"][i], sub_box(L, i)) for L in self.oth for i in range(len(L["s"]))]
        self.OC = np.array([b for _, b in self.oflat], float) if self.oflat else np.zeros((0, 4))
        self._aln = {}

    def by_line(self, ch, bbox) -> dict:
        loc = locate(ch, bbox, self.ref)
        if loc is None:
            return dict(st_line="no_loc")
        li, k = loc
        L = self.ref[li]
        if li not in self._aln:
            oi = best_line(L, self.oth)
            self._aln[li] = (oi, align(L["s"], self.oth[oi]["s"]) if oi is not None else None)
        oi, m = self._aln[li]
        if oi is None:
            return dict(st_line="no_line", n1=len(L["s"]))
        O = self.oth[oi]
        same = len(O["s"]) == len(L["s"])
        d = dict(n1=len(L["s"]), n2=len(O["s"]), same_count=int(same), oth_idx=(O["s"][k] if same else ""))
        if m[k] is None:
            d["st_line"] = "gap"
            return d
        d.update(st_line="ok", oth_line=O["s"][m[k]], oth_line_box=sub_box(O, m[k]))
        return d

    def by_geom(self, bbox) -> dict:
        j, st = match_box(bbox, self.OC)
        return dict(st_geom=st, oth_geom=(self.oflat[j][0] if j is not None else ""),
                    oth_geom_box=(self.oflat[j][1] if j is not None else None))


# ------------------------------------------------------------------------------------------------ cột kim (_detect)
class KimCols:
    """Cột kim {line_id: [(chữ, bbox)]} của 1 trang = _detect (qua cache gold_exact nếu có). Chỉ ĐỌC prepared/."""

    def __init__(self, recompute=False):
        self.recompute = recompute
        self._ok = False
        self.n_pkl = self.n_detect = 0
        self._bdict = {}

    def _setup(self):
        if self._ok:
            return
        from pipeline.step0_setup import load_config
        from pipeline.gold_exact import signals_geom as SG
        base = load_config(str(REPO / STT_CFG))
        SG._kim_init(str(REPO / base["paths"]["qn_to_nom_dict"]))
        self.SG, self.load_config, self._ok = SG, load_config, True

    def detect(self, name: str, page: str, cfg_file: str):
        self._setup()
        if name not in self._bdict:
            cfg = self.load_config(str(REPO / cfg_file))
            self._bdict[name] = next(b for b in cfg["books"] if b["name"] == name)
        self.n_detect += 1
        return self.SG._kim_page((name, page, self._bdict[name]))

    def get(self, name: str, page: str, cfg_file: str):
        pf = PKL_ROOT / name / f"{page}.pkl"
        if not self.recompute and pf.exists():
            try:
                z = pickle.load(open(pf, "rb"))
                if isinstance(z, dict) and "out" in z:
                    self.n_pkl += 1
                    return z["out"]
            except Exception:  # noqa: BLE001
                pass
        return self.detect(name, page, cfg_file)


# ------------------------------------------------------------------------------------------------ STT
def lt2_pages(books, limit=0):
    """-> {book: {"lt1": [page], "lt2": {page: (rec_lt2, rec_lt1)}, "bad": [(page, lý do)]}}; trang lt2 hợp lệ =
    lang_type 2, cùng image_hash với lt1, toạ độ fullpage (cả hai)."""
    out = {}
    for b in books:
        caches = lt1_caches(b)
        lt2, bad = {}, []
        for cf in caches:
            pg = page_of(cf)
            f = REPO / "prepared" / BOOKS[b] / LT2_DIR / f"{pg}_lt2.json"
            if not f.exists():
                continue
            if limit and len(lt2) >= limit:
                break
            try:
                r = json.loads(f.read_text(encoding="utf-8")); l1 = json.loads(cf.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                bad.append((pg, f"đọc lỗi {e}")); continue
            why = [w for w, bad_ in (("lang_type≠2", (r.get("kim_params") or {}).get("lang_type") != 2),
                                     ("ảnh khác lt1", r.get("image_hash") != l1.get("image_hash")),
                                     ("toạ độ không fullpage", r.get("coords_space") != "fullpage"
                                      or l1.get("coords_space") != "fullpage")) if bad_]
            if why:
                bad.append((pg, ",".join(why))); continue
            lt2[pg] = (r, l1)
        out[b] = dict(lt1=[page_of(p) for p in caches], lt2=lt2, bad=bad)
    return out


def eval_stt(cov, kim: KimCols, all_dir: Path, primary: str, detect_check: int):
    C = lexicon()
    L = rd(all_dir / "labels.csv", usecols=["cell_uid", "book_set", "book", "page", "column", "ocr_char", "syllable",
                                             "label", "tier", "rule"])
    L = L[(L.book_set == "SachThanhTruyen") & L.book.isin(list(cov))].copy()
    ge_path = all_dir / "gold_exact.csv"
    GE = rd(ge_path, usecols=["cell_uid", "gold_exact"]) if ge_path.exists() else pd.DataFrame(columns=["cell_uid",
                                                                                                        "gold_exact"])
    L["gold_exact"] = L.cell_uid.map(dict(zip(GE.cell_uid, GE.gold_exact))).fillna("-").replace("", "-")
    L["nom_idx"] = L.cell_uid.str.extract(r"/n(\d+)/s\d+$")[0].astype(int)
    L["rule_g"] = L.rule.str.split(":").str[0]
    L = L.sort_values(["book", "page", "cell_uid"]).reset_index(drop=True)

    rows = []
    page_nom = defaultdict(lambda: [0, 0, 0, 0])   # book -> [chữ lt1, Nôm riêng lt1, chữ lt2, Nôm riêng lt2] (cả trang)
    self_ctl = [0, 0, 0]                           # ô có hộp lt1 · tự ghép được (geom + line) · trùng chữ
    for (b, pg), g in L.groupby(["book", "page"], sort=True):
        pair = cov[b]["lt2"].get(pg)
        if pair is None:
            rows += [{**r._asdict(), "status": "no_lt2_page"} for r in g.itertuples(index=False)]
            continue
        r2, r1 = pair
        pp = PagePair(r1.get("boxes_raw"), r2.get("boxes_raw"))
        me = PagePair(r1.get("boxes_raw"), r1.get("boxes_raw"))   # đối chứng tự thân
        pn = page_nom[b]
        pn[0] += len(me.oflat); pn[1] += sum(nom_rieng(c) for c, _ in me.oflat)
        pn[2] += len(pp.oflat); pn[3] += sum(nom_rieng(c) for c, _ in pp.oflat)
        out = kim.get(BOOKS[b], pg, STT_CFG) or {}
        for r in g.itertuples(index=False):
            d = {**r._asdict()}
            chars = out.get(int(r.column)); ni = int(r.nom_idx)
            if not chars or ni >= len(chars) or not chars[ni][1]:
                d["status"] = "no_lt1_box"; rows.append(d); continue
            kc, kb = chars[ni][0], [float(v) for v in chars[ni][1][:4]]
            d["lt1_box"] = json.dumps([int(v) for v in kb])
            if kc != r.ocr_char:
                d["status"] = "lt1_mismatch"; rows.append(d); continue
            s_l, s_g = me.by_line(kc, kb), me.by_geom(kb)
            self_ctl[0] += 1
            self_ctl[1] += int(s_l.get("st_line") == "ok" and s_g["st_geom"] == "ok")
            self_ctl[2] += int(s_l.get("oth_line") == kc and s_g["oth_geom"] == kc)
            dl, dg = pp.by_line(kc, kb), pp.by_geom(kb)
            d.update(st_line=dl.get("st_line"), lt2_line=dl.get("oth_line", ""), n1=dl.get("n1"), n2=dl.get("n2"),
                     same_count=dl.get("same_count"), lt2_idx=dl.get("oth_idx", ""),
                     st_geom=dg["st_geom"], lt2_geom=dg["oth_geom"],
                     ab_same=int(dl.get("st_line") == "ok" and dg["st_geom"] == "ok"
                                 and dl["oth_line_box"] == [int(v) for v in dg["oth_geom_box"]]))
            d["status"], d["lt2"] = ((d["st_line"], d["lt2_line"]) if primary == "line" else (d["st_geom"], d["lt2_geom"]))
            if d["status"] == "ok" and d["lt2"]:
                bb = dl["oth_line_box"] if primary == "line" else dg["oth_geom_box"]
                d["lt2_box"] = json.dumps([int(v) for v in bb])
            rows.append(d)
    D = pd.DataFrame(rows)
    for c in ("n1", "n2", "same_count", "ab_same"):
        if c not in D:
            D[c] = np.nan
    for c in ("lt2", "lt2_line", "lt2_geom", "lt2_idx", "lt1_box", "lt2_box", "st_line", "st_geom"):
        D[c] = D[c].fillna("") if c in D else ""
    ok = D.status == "ok"
    R = {s: C.R_of(s) for s in D.syllable.unique()}
    f = lambda a, c, m: [int(veq(x, y)) if z else np.nan for x, y, z in zip(a, c, m)]
    D["eq_strict"] = [int(x == y) if z else np.nan for x, y, z in zip(D.ocr_char, D.lt2, ok)]
    D["eq_v1p"] = f(D.ocr_char, D.lt2, ok)
    D["eq_v1p_geom"] = f(D.ocr_char, D.lt2_geom, D.st_geom == "ok")
    D["eq_v1p_line"] = f(D.ocr_char, D.lt2_line, D.st_line == "ok")
    D["eq_v1p_same_count"] = f(D.ocr_char, D.lt2_idx, D.same_count.fillna(0) == 1)
    D["lab_eq_v1p"] = f(D.label, D.lt2, ok & (D.label != ""))
    D["lt1_inR"] = [int(a in R[s]) for a, s in zip(D.ocr_char, D.syllable)]
    D["lt2_inR"] = [int(c in R[s]) if z else np.nan for c, s, z in zip(D.lt2, D.syllable, ok)]
    D["rescue_cand"] = np.where(ok, ((D.lt1_inR == 0) & (D.lt2_inR == 1)).astype(int), np.nan)
    D["conflict_inR"] = np.where(ok, ((D.lt1_inR == 1) & (D.lt2_inR == 1) & (D.eq_v1p == 0)).astype(int), np.nan)
    D["lt1_nom"] = D.ocr_char.map(nom_rieng).astype(int)
    D["lt2_nom"] = np.where(ok, D.lt2.map(nom_rieng).astype(int), np.nan)

    # kiểm cache pickle bằng _detect trên vài trang (rải đều, cố định)
    chk = [(b, pg) for b in cov for pg in sorted(cov[b]["lt2"])]
    if chk and detect_check > 0:
        chk = [chk[i] for i in sorted(set(np.linspace(0, len(chk) - 1, min(detect_check, len(chk))).astype(int)))]
    else:
        chk = []
    n_eq = sum(int((kim.detect(BOOKS[b], pg, STT_CFG) or {}) == (kim.get(BOOKS[b], pg, STT_CFG) or {})) for b, pg in chk)
    return D, dict(page_nom=dict(page_nom), self_ctl=self_ctl, detect_check=[n_eq, len(chk)])


def _p(s: pd.Series):
    s = s.dropna()
    return rate(int(s.sum()), len(s))


def stats(d: pd.DataFrame) -> dict:
    ok = d[d.status == "ok"]
    return dict(n=len(d), matched=len(ok), eq_strict=_p(ok.eq_strict)["p"], eq_v1p=_p(ok.eq_v1p)["p"],
                eq_v1p_geom=_p(d.eq_v1p_geom)["p"], lab_eq_v1p=_p(ok.lab_eq_v1p)["p"],
                rescue_cand=int(ok.rescue_cand.sum()), conflict_inR=int(ok.conflict_inR.sum()))


def book_block(d: pd.DataFrame, n_lt1: int, n_lt2: int, n_bad: int, pn) -> dict:
    on = d[d.status != "no_lt2_page"]
    ok = d[d.status == "ok"]
    st = Counter(on.status)
    return dict(
        pages_lt1=n_lt1, pages_lt2=n_lt2, pages_lt2_bad=n_bad, cells=len(d), cells_on_lt2_pages=len(on), matched=len(ok),
        match_rate=round(len(ok) / len(on), 4) if len(on) else None,
        unmatched={k: v for k, v in st.items() if k != "ok"},
        eq_strict=_p(ok.eq_strict), eq_v1p=_p(ok.eq_v1p),
        methods=dict(line=dict(match=_p(on.st_line == "ok")["p"], eq_v1p=_p(on.eq_v1p_line)["p"]),
                     geom=dict(match=_p(on.st_geom == "ok")["p"], eq_v1p=_p(on.eq_v1p_geom)["p"]),
                     same_count=dict(cells=_p(on.same_count.fillna(0) == 1)["p"], eq_v1p=_p(on.eq_v1p_same_count)["p"]),
                     line_geom_same_pick=_p(on.ab_same[(on.st_line == "ok") & (on.st_geom == "ok")])["p"]),
        rescue_cand=int(ok.rescue_cand.sum()), conflict_inR=int(ok.conflict_inR.sum()),
        nom_rieng_cells=dict(lt1=_p(ok.lt1_nom)["p"], lt2=_p(ok.lt2_nom)["p"]),
        nom_rieng_pages=dict(lt1=rate(pn[1], pn[0])["p"], lt2=rate(pn[3], pn[2])["p"], chars_lt1=pn[0], chars_lt2=pn[2])
        if pn else None,
    )


def breakdowns(d: pd.DataFrame) -> dict:
    on = d[d.status != "no_lt2_page"]
    out = {}
    for key, col in (("tier", "tier"), ("rule", "rule_g"), ("gold_exact", "gold_exact")):
        g = {v: stats(s) for v, s in on.groupby(col)}
        out[key] = {k: g[k] for k in GE_ORDER if k in g} if key == "gold_exact" else g
    return out


# ------------------------------------------------------------------------------------------------ hiệu chuẩn (thạch bản)
def calibrate(kim: KimCols, primary: str, limit=0):
    """Bảng ô cho bộ có lt1 + lt2 cùng trang + dị bản. Ô chuẩn = hộp lt2 (cột _detect của build hiện hành, lt2);
    tìm chữ lt1 cùng vị trí bằng CÙNG PagePair (vai trò đảo so với STT)."""
    C = lexicon()
    rows, meta = [], {}
    for book, cfgf in CALIB.items():
        cf, kd = CROSS / book / "cells.csv", REPO / "prepared" / book / "kim_raw"
        if not cf.exists() or not kd.is_dir():
            meta[book] = dict(skip="thiếu measure_out/auto_precision/cross/<Sách>/cells.csv hoặc kim_raw/"); continue
        X = rd(cf, usecols=["page", "column", "nom_idx", "syl_idx", "syllable", "ref", "ref_nom", "ocr_char", "match_tier"])
        X = X[X.ref_nom != ""]
        refs = X.groupby(["page", "column", "nom_idx", "syl_idx"], sort=True).agg(
            syllable=("syllable", "first"), ocr_char=("ocr_char", "first"),
            exact=("match_tier", lambda s: int((s == "exact").any())), refs=("ref_nom", list)).reset_index()
        pages = sorted(refs.page.unique())[: (limit or None)]
        n_chk = n_eq = n_img = n_pg = 0
        for pg in pages:
            f1, f2 = kd / f"{pg}.json", kd / f"{pg}_lt2.json"
            if not (f1.exists() and f2.exists()):
                continue
            r1 = json.loads(f1.read_text(encoding="utf-8")); r2 = json.loads(f2.read_text(encoding="utf-8"))
            if (r1.get("kim_params") or {}).get("lang_type", 1) != 1 or (r2.get("kim_params") or {}).get("lang_type") != 2:
                continue
            n_pg += 1; n_img += int(r1.get("image_hash") == r2.get("image_hash"))
            pp = PagePair(r2.get("boxes"), r1.get("boxes"))       # chuẩn = lt2, bên kia = lt1
            out = kim.get(book, pg, cfgf) or {}
            for r in refs[refs.page == pg].itertuples(index=False):
                chars = out.get(int(r.column)); ni = int(r.nom_idx)
                if not chars or ni >= len(chars) or not chars[ni][1]:
                    continue
                c2, b2 = chars[ni][0], chars[ni][1]
                n_chk += 1
                if c2 != r.ocr_char:
                    continue
                n_eq += 1
                dl, dg = pp.by_line(c2, b2), pp.by_geom(b2)
                st, c1 = ((dl.get("st_line"), dl.get("oth_line", "")) if primary == "line"
                          else (dg["st_geom"], dg["oth_geom"]))
                c1 = c1 if st == "ok" else ""
                R = C.R_of(r.syllable)
                rows.append(dict(book=book, page=pg, column=r.column, nom_idx=ni, syllable=r.syllable, exact=r.exact,
                                 lt1=c1, lt2=c2, status=st, matched=int(st == "ok"),
                                 same_count=int(dl.get("same_count") or 0), agree=int(veq(c1, c2)),
                                 agree_geom=int(dg["st_geom"] == "ok" and veq(dg["oth_geom"], c2)),
                                 lt1_inR=int(c1 in R), lt2_inR=int(c2 in R),
                                 lt1_ok=int(any(veq(c1, x) for x in r.refs)), lt2_ok=int(any(veq(c2, x) for x in r.refs)),
                                 refs="|".join(r.refs)))
        meta[book] = dict(pages=n_pg, lt2_char_eq_cross=rate(n_eq, n_chk), same_image=rate(n_img, n_pg))
    return pd.DataFrame(rows), meta


def boot_diff(d: pd.DataFrame, B=1000, seed=0):
    """Bootstrap cụm trang: CI 95 % của P(lt1 đúng | đồng ý), P(lt1 đúng | lệch) và hiệu."""
    if d.empty:
        return None
    arr = np.array([[s.agree.sum(), (s.agree * s.lt1_ok).sum(), (1 - s.agree).sum(), ((1 - s.agree) * s.lt1_ok).sum()]
                    for _, s in d.groupby(["book", "page"])], float)
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(B):
        a = arr[rng.integers(0, len(arr), len(arr))].sum(0)
        pa = a[1] / a[0] if a[0] else np.nan
        pdis = a[3] / a[2] if a[2] else np.nan
        res.append((pa, pdis, pa - pdis))
    res = np.array(res)
    q = lambda k: [round(float(np.nanpercentile(res[:, k], 2.5)), 4), round(float(np.nanpercentile(res[:, k], 97.5)), 4)]
    return dict(agree=q(0), disagree=q(1), diff=q(2), n_pages=len(arr), B=B)


def cond(d: pd.DataFrame) -> dict:
    A, N = d[d.agree == 1], d[d.agree == 0]
    x = dict(n=len(d), agree=rate(len(A), len(d))["p"], p_lt1_ok=rate(int(d.lt1_ok.sum()), len(d))["p"],
             p_ok_agree=rate(int(A.lt1_ok.sum()), len(A)), p_ok_disagree=rate(int(N.lt1_ok.sum()), len(N)),
             p_lt2_ok_disagree=rate(int(N.lt2_ok.sum()), len(N))["p"])
    x["p_ok_agree"]["wilson"] = wilson(x["p_ok_agree"]["k"], x["p_ok_agree"]["n"])
    return x


def calib_block(d: pd.DataFrame) -> dict:
    m = d[d.matched == 1]
    blk = dict(cells=len(d), matched=rate(len(m), len(d))["p"], all=cond(m), lt1_inR=cond(m[m.lt1_inR == 1]),
               same_count_lt1_inR=cond(m[(m.same_count == 1) & (m.lt1_inR == 1)]),
               agree_geom=rate(int(d.agree_geom.sum()), len(d))["p"])
    return blk


# ------------------------------------------------------------------------------------------------ main
def run(a) -> int:
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    all_dir = Path(a.all_dir)
    books = list(BOOKS) if a.book == "all" else [a.book]
    if not (all_dir / "labels.csv").exists():
        print(f"[stt_lt2] thiếu {all_dir / 'labels.csv'}"); return 2
    sha0 = {b: sha_files(lt1_caches(b)) for b in books}
    try:
        prev = json.loads((out / "summary.json").read_text(encoding="utf-8")).get("lt1_cache_sha16", {}) or {}
    except Exception:  # noqa: BLE001
        prev = {}
    cov = lt2_pages(books, a.limit)
    n_lt1 = sum(len(c["lt1"]) for c in cov.values()); n_lt2 = sum(len(c["lt2"]) for c in cov.values())
    full = (n_lt2 == n_lt1 == N_PAGES_STT) if (a.book == "all" and not a.limit) else False
    partial = not full
    print(f"[stt_lt2] cache lt2: {n_lt2}/{n_lt1} trang" + ("" if full else " — MỘT PHẦN")
          + "".join(f" · {b} {len(cov[b]['lt2'])}/{len(cov[b]['lt1'])}" for b in books) + f" · phương pháp chính: {a.match}")
    lexicon()
    kim = KimCols(recompute=a.recompute_detect)
    D, ex = (eval_stt(cov, kim, all_dir, a.match, a.detect_check) if n_lt2
             else (pd.DataFrame(), dict(page_nom={}, self_ctl=[0, 0, 0], detect_check=[0, 0])))

    per_book, bd, top = {}, {}, []
    if not D.empty:
        for b in books:
            per_book[b] = book_block(D[D.book == b], len(cov[b]["lt1"]), len(cov[b]["lt2"]), len(cov[b]["bad"]),
                                     ex["page_nom"].get(b))
            bd[b] = breakdowns(D[D.book == b])
        pn = [sum(ex["page_nom"].get(b, [0] * 4)[i] for b in books) for i in range(4)]
        per_book["total"] = book_block(D, n_lt1, n_lt2, sum(len(cov[b]["bad"]) for b in books), pn)
        bd["total"] = breakdowns(D)
        ok = D[D.status == "ok"]
        dis = ok[ok.eq_v1p == 0]
        top = [[x, y, n] for (x, y), n in Counter(zip(dis.ocr_char, dis.lt2)).most_common(20)]

    # hiệu chuẩn
    if a.no_calib:
        K, kmeta, calib = pd.DataFrame(), {}, dict(skip="--no-calib")
    else:
        K, kmeta = calibrate(kim, a.match, a.limit)
        calib = dict(kind="THẠCH BẢN IN so DỊ BẢN — ƯỚC LƯỢNG, cận dưới (dị bản khác chữ hợp lệ; nền 1871↔1872 = 82,9 %)",
                     books=kmeta, ihr_human="KHÔNG: LucVanTien1916/TruyenKieu1872 chỉ có kim_raw/*_lt2.json (0 lt1)",
                     handwriting="KHÔNG: Borg chỉ có lt2; áp số thạch bản cho STT chép tay = SUY ĐOÁN",
                     chrestomathie="có lt1+lt2 nhưng không có chữ tham chiếu -> không dùng")
        if not K.empty:
            calib["per_book"] = {b: calib_block(K[K.book == b]) for b in sorted(K.book.unique())}
            calib["pooled"] = calib_block(K)
            calib["pooled"]["boot_page_lt1_inR"] = boot_diff(K[(K.matched == 1) & (K.lt1_inR == 1)])

    # đề xuất cổng: gold_exact ok chỉ khi lt1 == lt2 (V1+)
    gate = {}
    if not D.empty:
        pool = (calib.get("pooled") or {}).get("lt1_inR") or {}
        for b in books + ["total"]:
            db = D if b == "total" else D[D.book == b]
            okc = db[db.gold_exact == "ok"]
            on = okc[okc.status != "no_lt2_page"]
            keep = on[(on.status == "ok") & (on.eq_v1p == 1)]
            kr = round(len(keep) / len(on), 4) if len(on) else None
            gate[b] = dict(ok=len(okc), ok_on_lt2_pages=len(on), ok_matched=int((on.status == "ok").sum()),
                           ok_keep=len(keep), keep_rate=kr,
                           ok_keep_extrapolated=(round(len(okc) * kr) if (kr is not None and partial) else None),
                           ok_label_ne_lt1=int((okc.label != okc.ocr_char).sum()))
        gate["_est"] = dict(
            p_ok_given_agree_inR=(pool.get("p_ok_agree") or {}).get("p"), wilson=(pool.get("p_ok_agree") or {}).get("wilson"),
            p_ok_given_disagree_inR=(pool.get("p_ok_disagree") or {}).get("p"), p_ok_inR_no_gate=pool.get("p_lt1_ok"),
            muc_chac="ƯỚC LƯỢNG trên thạch bản so dị bản (cận dưới); cho chữ viết tay STT = SUY ĐOÁN",
            unmatched="ô không ghép được chữ lt2 bị loại (bảo thủ)")

    # ghi tệp
    if not D.empty:
        cols = ["cell_uid", "book", "page", "column", "nom_idx", "tier", "rule", "gold_exact", "syllable", "ocr_char",
                "label", "lt2", "status", "st_line", "lt2_line", "st_geom", "lt2_geom", "n1", "n2", "same_count", "lt2_idx",
                "ab_same", "eq_strict", "eq_v1p", "eq_v1p_line", "eq_v1p_geom", "eq_v1p_same_count", "lab_eq_v1p",
                "lt1_inR", "lt2_inR", "rescue_cand", "conflict_inR", "lt1_nom", "lt2_nom", "lt1_box", "lt2_box"]
        D.reindex(columns=cols).rename(columns={"ocr_char": "lt1"}).to_csv(out / "cells.csv", index=False)
    if not K.empty:
        K.to_csv(out / "calib_cells.csv", index=False)
    (out / "breakdown.json").write_text(json.dumps(dict(by_book=bd, top_pairs_lt1_lt2=top, calib=calib.get("per_book")),
                                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # invariants
    sha1 = {b: sha_files(lt1_caches(b)) for b in books}
    inv = []

    def add(name, exp, obs, p, method=""):
        inv.append(dict(name=name, expected=exp, observed=obs, **{"pass": p}, **({"method": method} if method else {})))

    add("lt1_cache_khong_doi_trong_luot_do", "sha256 đầu == cuối", {b: sha1[b][:16] for b in books},
        all(sha0[b] == sha1[b] for b in books), "sha256(tên + nội dung) mọi detected/*_ocr_cache.json")
    same_prev = [b for b in books if b in prev]
    add("lt1_cache_bang_lan_do_truoc", "sha16 == summary.json lần trước", {b: prev[b] for b in same_prev} or None,
        (all(prev[b] == sha1[b][:16] for b in same_prev) if same_prev else None))
    add("so_trang_lt2_le_448_va_le_lt1", f"≤ {N_PAGES_STT} và ≤ trang lt1 từng bộ", {b: len(cov[b]["lt2"]) for b in books},
        n_lt2 <= N_PAGES_STT and all(len(cov[b]["lt2"]) <= len(cov[b]["lt1"]) for b in books))
    add("so_trang_lt1_STT", N_PAGES_STT, n_lt1, (n_lt1 == N_PAGES_STT) if a.book == "all" else None)
    add("trang_lt2_hop_le", "0 trang lt2 sai lang_type/ảnh/toạ độ", [x for b in books for x in cov[b]["bad"]][:5],
        all(not cov[b]["bad"] for b in books))
    if not D.empty:
        n_ok = int((D.status == "ok").sum())
        add("o_ghep_le_o_STT", "ô ghép ≤ ô STT", [n_ok, len(D)], n_ok <= len(D))
        hb = int(D.status.isin(["ok", "gap", "no_line", "no_loc", "no_x", "low_overlap", "lt1_mismatch"]).sum())
        mm = int((D.status == "lt1_mismatch").sum())
        add("chu_lt1_tai_nom_idx_khop_ocr_char", "≥ 0,999 ô có hộp lt1", rate(hb - mm, hb),
            (hb - mm) >= 0.999 * hb if hb else None, "cột _detect (kim_vs_box)")
        sc = ex["self_ctl"]
        add("doi_chung_lt1_ghep_voi_chinh_no", "100 % ghép + 100 % trùng (line & geom)", sc,
            sc[0] == sc[1] == sc[2] if sc[0] else None)
    dc = ex["detect_check"]
    add("cache_pickle_khop_detect", "N/N trang _detect == pickle", dc, (dc[0] == dc[1]) if dc[1] else None)
    add("phu_du_448_trang", f"{N_PAGES_STT}/{N_PAGES_STT} trang lt2", n_lt2,
        full if a.require_full else (True if full else None), "cứng khi --require-full; không cờ: SKIP khi một phần")
    for b, m in kmeta.items():
        if "skip" in m:
            continue
        e = m["lt2_char_eq_cross"]
        add(f"calib_{b}_lt2_khop_cross", "≥ 0,99", e, (e["p"] or 0) >= 0.99 if e["n"] else None)
        add(f"calib_{b}_cung_anh", "lt1 & lt2 cùng image_hash", m["same_image"],
            (m["same_image"]["k"] == m["same_image"]["n"]) if m["same_image"]["n"] else None)

    tot = per_book.get("total", {})
    summary = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"), script="scripts/measure/stt_lt2_eval.py",
        api_calls=0, partial_run=partial, scope=("MỘT PHẦN" if partial else "ĐỦ 448 trang"), limit=a.limit or None,
        books=books, primary_method=a.match,
        coverage=dict(pages_lt1=n_lt1, pages_lt2=n_lt2, cells_stt=len(D), cells_on_lt2_pages=tot.get("cells_on_lt2_pages"),
                      matched=tot.get("matched"), match_rate=tot.get("match_rate"),
                      kim_cols=dict(pickle=kim.n_pkl, detect=kim.n_detect)),
        per_book=per_book,
        by_gold_exact=(bd.get("total") or {}).get("gold_exact"),
        top_pairs_lt1_lt2=top[:20],
        calibration=({k: v for k, v in calib.items() if k != "per_book"} if "skip" not in calib else calib),
        gate_ok_iff_lt1_eq_lt2=gate,
        lt1_cache_sha16={b: sha1[b][:16] for b in books},
        invariants=inv, runtime_s=round(time.time() - t0, 1),
        detail="breakdown.json (theo tầng/luật/gold_exact từng bộ, hiệu chuẩn từng sách), cells.csv, calib_cells.csv",
    )
    enc = lambda s: json.dumps(s, ensure_ascii=False, separators=(",", ":"), default=str)
    for cut in ("top_pairs_lt1_lt2", "by_gold_exact"):      # giữ ≤ 8 KB: rút bảng lớn (đã có trong breakdown.json)
        if len(enc(summary).encode()) <= 8192:
            break
        summary[cut] = (top[:8] if cut == "top_pairs_lt1_lt2" else "xem breakdown.json")
    sj = enc(summary)
    (out / "summary.json").write_text(sj, encoding="utf-8")

    # stdout ngắn
    nf = sum(iv["pass"] is False for iv in inv)
    print(f"[stt_lt2] ô STT {len(D)} · trên trang có lt2 {tot.get('cells_on_lt2_pages')} · ghép {tot.get('matched')} "
          f"({tot.get('match_rate')}) · cột kim: pickle {kim.n_pkl} / _detect {kim.n_detect}")
    if tot:
        mt = tot["methods"]
        print(f"[stt_lt2] lt1==lt2 V1+ line {mt['line']['eq_v1p']} (ghép {mt['line']['match']}) · geom "
              f"{mt['geom']['eq_v1p']} (ghép {mt['geom']['match']}) · cùng-số-chữ {mt['same_count']['eq_v1p']} "
              f"(ô {mt['same_count']['cells']}) · strict chính {tot['eq_strict']['p']}")
        print(f"[stt_lt2] ứng viên cứu (lt1∉R, lt2∈R) {tot['rescue_cand']} · xung đột trong R {tot['conflict_inR']} · "
              f"Nôm riêng ô lt1/lt2 {tot['nom_rieng_cells']['lt1']}/{tot['nom_rieng_cells']['lt2']} · cả trang "
              f"{tot['nom_rieng_pages']['lt1']}/{tot['nom_rieng_pages']['lt2']}")
        for k, v in ((bd.get("total") or {}).get("gold_exact") or {}).items():
            print(f"    gold_exact={k:12s} n={v['n']:6d} ghép {v['matched']:6d} V1+ {v['eq_v1p']} (geom {v['eq_v1p_geom']})")
    pool = calib.get("pooled") or {}
    if pool:
        a_, i_ = pool["all"], pool["lt1_inR"]
        print(f"[stt_lt2] hiệu chuẩn thạch bản (dị bản): ghép {pool['matched']} · đồng ý {a_['agree']} · P(lt1 đúng|đồng ý) "
              f"{a_['p_ok_agree']['p']} vs |lệch {a_['p_ok_disagree']['p']} · lt1∈R: {i_['p_ok_agree']['p']} "
              f"{i_['p_ok_agree']['wilson']} vs {i_['p_ok_disagree']['p']} (n {i_['p_ok_agree']['n']}/{i_['p_ok_disagree']['n']})")
    if gate.get("total"):
        g = gate["total"]
        print(f"[stt_lt2] cổng ok⇔lt1==lt2: ok {g['ok']} · trên trang lt2 {g['ok_on_lt2_pages']} · giữ {g['ok_keep']} "
              f"({g['keep_rate']})" + (f" · ngoại suy đủ bộ ≈ {g['ok_keep_extrapolated']}" if g["ok_keep_extrapolated"] else ""))
    print(f"[stt_lt2] invariants PASS {sum(iv['pass'] is True for iv in inv)} FAIL {nf} SKIP "
          f"{sum(iv['pass'] is None for iv in inv)} · {summary['scope']} · {time.time() - t0:.0f} s · "
          f"{out / 'summary.json'} ({len(sj.encode())} B)")
    for iv in inv:
        if iv["pass"] is False:
            print(f"  FAIL {iv['name']}: kỳ vọng {iv['expected']} · quan sát {iv['observed']}")
    return 1 if nf else 0


# ------------------------------------------------------------------------------------------------ selftest
def selftest() -> int:
    ok = 0
    # 1) geom
    C2 = np.array([[100, 0, 150, 50], [100, 50, 150, 100], [100, 100, 150, 150], [40, 0, 90, 50]], float)
    assert match_box([100, 52, 150, 98], C2) == (1, "ok"); ok += 1
    assert match_box([200, 0, 250, 50], C2)[1] == "no_x"; ok += 1
    assert match_box([100, 160, 150, 200], C2)[1] == "low_overlap"; ok += 1
    assert match_box([45, 10, 95, 45], C2)[0] == 3; ok += 1
    # 2) chia hộp dòng == core.ocr.ocr_api.boxes_to_columns
    from core.ocr.ocr_api import boxes_to_columns
    bx = [{"transcription": "天地人", "points": [[100, 10], [150, 10], [150, 160], [100, 160]]},
          {"transcription": "𠀧", "points": [[30, 20], [80, 20], [80, 70], [30, 70]]}]
    a = sorted((c, tuple(b)) for c, b in chars_from_raw(bx))
    b = sorted((c, tuple(b)) for c, b in flat_cols(boxes_to_columns(bx)))
    assert a == b, (a, b); ok += 1
    # 3) căn chuỗi: bỏ 1 chữ giữa dòng -> khoảng đúng chỗ, phần còn lại khớp; cùng số chữ -> chéo
    eq = lambda x, y: x == y
    assert align(list("ABCDEF"), list("ABDEF"), eq) == [0, 1, None, 2, 3, 4]; ok += 1
    assert align(list("ABC"), list("XYZ"), eq) == [0, 1, 2]; ok += 1
    assert align(list("祓日公翁君依"), list("百翁垂依"), eq)[3] == 1 and align(list("祓日公翁君依"), list("百翁垂依"), eq)[5] == 3; ok += 1
    # 4) PagePair: lt2 đọc ít chữ hơn trên CÙNG hộp dòng -> line tìm đúng, geom trôi
    r1 = [{"transcription": "ABCDEFGHIJ", "points": [[0, 0], [50, 0], [50, 500], [0, 500]]}]
    r2 = [{"transcription": "ABCDEFGIJ", "points": [[0, 0], [50, 0], [50, 500], [0, 500]]}]
    pp = PagePair(r1, r2)
    d = pp.by_line("I", sub_box(raw_lines(r1)[0], 8))
    assert d["st_line"] == "ok" and d["oth_line"] == "I"; ok += 1
    assert pp.by_line("H", sub_box(raw_lines(r1)[0], 7))["st_line"] == "gap"; ok += 1
    # 5) chữ Nôm riêng
    assert nom_rieng("𠀧") and nom_rieng("") and not nom_rieng("人") and not nom_rieng(""); ok += 1
    # 6) V1+ + R(âm)
    assert veq("人", "人") and not veq("人", "天") and not veq("", ""); ok += 1
    assert len(lexicon().R_of("người")) > 0; ok += 1
    # 7) bootstrap cụm trang: đồng ý luôn đúng, lệch luôn sai
    d = pd.DataFrame(dict(book="x", page=[f"p{i % 7}" for i in range(70)], agree=[i % 2 for i in range(70)],
                          lt1_ok=[i % 2 for i in range(70)]))
    bt = boot_diff(d, B=200)
    assert bt["agree"] == [1.0, 1.0] and bt["disagree"] == [0.0, 0.0]; ok += 1
    # 8) đối chứng tự thân trên 1 trang STT thật: mọi chữ lt1 tự ghép (line + geom) đúng chính nó
    cf = next(iter(lt1_caches("stt2")), None)
    if cf is not None:
        r = json.loads(cf.read_text(encoding="utf-8"))
        me = PagePair(r["boxes_raw"], r["boxes_raw"])
        hit = sum(me.by_line(c, bb).get("oth_line") == c and me.by_geom(bb)["oth_geom"] == c for c, bb in me.oflat)
        assert hit == len(me.oflat), (hit, len(me.oflat)); ok += 1
    # 9) thư mục ra mặc định nằm ngoài prepared/ (phép đo chỉ đọc prepared/)
    assert "prepared" not in DEFAULT_OUT.relative_to(REPO).parts; ok += 1
    print(f"[stt_lt2] selftest: {ok}/{ok} PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="all", choices=[*BOOKS, "all"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--all-dir", default=str(REPO / "dataset" / "_ALL"))
    ap.add_argument("--match", default="line", choices=["line", "geom"], help="phương pháp ghép chính (mặc định line)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ N trang lt2 đầu mỗi bộ (chạy thử; invariant toàn bộ = SKIP)")
    ap.add_argument("--workers", type=int, default=1, help="(giữ giao diện chung; phép đo 1 luồng)")
    ap.add_argument("--require-full", action="store_true", help="FAIL nếu cache lt2 chưa đủ 448 trang")
    ap.add_argument("--recompute-detect", action="store_true", help="bỏ cache pickle gold_exact, gọi _detect mọi trang")
    ap.add_argument("--detect-check", type=int, default=4, help="số trang gọi lại _detect để kiểm cache pickle")
    ap.add_argument("--no-calib", action="store_true", help="bỏ phần hiệu chuẩn LVT1883/KVK1884")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    return selftest() if a.selftest else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
