#!/usr/bin/env python3
"""
web/server.py — máy chủ trình diễn GanNhanOCR (luận văn Thạc sĩ: gán nhãn tự động văn bản Hán Nôm cổ).

Giao diện CHỈ trình bày TẦNG NHÃN (GOLD · SYLLABLE · GOLD_text_only · REVIEW · QUARANTINE) của 10 bộ tự động.
CHỈ dùng thư viện chuẩn Python (ThreadingHTTPServer). CHỈ ĐỌC dữ liệu dưới gốc dự án, không bao giờ ghi:

  dataset/<Bộ>/labels.csv (vắng quyển STT nào -> bản xuất cũ dataset/SachThanhTruyen{2,4,11}/; vắng hẳn -> dataset/_ALL)
  dataset/_ALL/SOURCES.json · prepared/<Bộ>/pages/ (Borg: prepared/_auto/<Sách>/pages/) · measure_out/SUMMARY.json
  measure_out/box_ref/summary.json · prepared/<LVT1883|KVK1884>/dataset_out/auto_precision_*/SUMMARY.json
  Dict/QuocNgu_SinoNom.csv (số mục từ)

Mọi con số trả về đều ĐỌC từ tệp; tệp thiếu -> mục đó không hiển thị (không bịa số).

Gốc dữ liệu: --root <thư mục> > biến môi trường GANNHANOCR_ROOT > thư mục cha của web/.

Chạy:   .venv/bin/python web/server.py [cổng] [--root DIR] [--host 0.0.0.0]
API:    /api/stats · /api/books · /api/page?book=&page= · /api/search?q=&book=&tier=&limit=
        /api/pipeline_flow · /api/benchmarks · /api/reload (nạp lại dữ liệu sau khi pipeline chạy xong)
Tệp:    /crops/<bộ>/<ảnh> · /page_scans/<bộ>/<tệp>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
import sys
import threading
import time
import unicodedata
from collections import Counter
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

csv.field_size_limit(1 << 30)

WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parent
NA = "chưa có"


def default_root() -> Path:
    env = os.environ.get("GANNHANOCR_ROOT", "").strip()
    return Path(env).expanduser().resolve() if env else REPO_ROOT


# =====================================================================================================================
# Tầng nhãn + danh mục 10 bộ (chỉ là SIÊU DỮ LIỆU mô tả — mọi con số đọc từ tệp)
# =====================================================================================================================
TIER_ORDER = ["GOLD", "SYLLABLE", "GOLD_text_only", "REVIEW", "QUARANTINE"]
TIER_ALWAYS = ("GOLD", "SYLLABLE", "REVIEW", "QUARANTINE")          # GOLD_text_only chỉ hiện khi có ô
TIER_VI = {"GOLD": "GOLD", "SYLLABLE": "SYLLABLE", "GOLD_text_only": "GOLD_text_only",
           "REVIEW": "REVIEW", "QUARANTINE": "QUARANTINE"}
ROLE_VI = {"giao_nop": "Giao nộp", "danh_gia_ihr": "Đánh giá (IHR-NomDB)", "danh_gia_borg": "Đánh giá (Borg, Vatican)"}
STATUS_VI = {"ok": "Có dữ liệu", "ban_cu": "Bản xuất cũ theo quyển (23/09)",
             "tu_bo_gop": "Đọc từ bộ gộp dataset/_ALL",
             "chua_co": "Chưa có dữ liệu (đang dựng lại hoặc chưa chạy pipeline)", "loi": "Lỗi đọc dữ liệu"}


def _book(id_, set8, book_set, title, subtitle, author, script, genre, n_columns, role, pages, default_page=None,
          book_key=None, legacy=None, ds_out=None):
    return dict(id=id_, set8=set8, book_set=book_set, title=title, subtitle=subtitle, author=author, script=script,
                genre=genre, n_columns=n_columns, role=role, pages_dirs=pages, default_page=default_page,
                book_key=book_key, legacy=legacy, ds_out=ds_out,
                layout_desc=f"{genre} · {script} · {n_columns} cột/trang" if isinstance(n_columns, int)
                else f"{genre} · {script} · số cột theo trang")


STT_AUTHOR = "Khuyết danh (văn bản Công giáo)"
BOOKS = [
    _book("SachThanhTruyen2", "stt2", "SachThanhTruyen", "Sách Thánh Truyện (Quyển 2)",
          "Bản chép tay chữ Nôm Công giáo — quyển 2", STT_AUTHOR, "viết tay", "Văn xuôi", 9, "giao_nop",
          ["SachThanhTruyen2/pages"], "page_0012", book_key="stt2", legacy="SachThanhTruyen2"),
    _book("SachThanhTruyen4", "stt4", "SachThanhTruyen", "Sách Thánh Truyện (Quyển 4)",
          "Bản chép tay chữ Nôm Công giáo — quyển 4", STT_AUTHOR, "viết tay", "Văn xuôi", 9, "giao_nop",
          ["SachThanhTruyen4/pages"], "page_0012", book_key="stt4", legacy="SachThanhTruyen4"),
    _book("SachThanhTruyen11", "stt11", "SachThanhTruyen", "Sách Thánh Truyện (Quyển 11)",
          "Bản chép tay chữ Nôm Công giáo — quyển 11", STT_AUTHOR, "viết tay", "Văn xuôi", 9, "giao_nop",
          ["SachThanhTruyen11/pages"], "page_0010", book_key="stt11", legacy="SachThanhTruyen11"),
    _book("LucVanTien1883", "L83", "LucVanTien1883", "Lục Vân Tiên (1883)",
          "Bản thạch bản năm Quý Mùi (1883)", "Nguyễn Đình Chiểu", "thạch bản", "Thơ lục bát", 10, "giao_nop",
          ["LucVanTien1883/pages"], "page_0002", ds_out="prepared/LucVanTien1883/dataset_out"),
    _book("KimVanKieu1884", "KVK", "KimVanKieu1884", "Kim Vân Kiều (1884)",
          "Bản thạch bản Kim Vân Kiều tân truyện (Giáp Thân 1884)", "Nguyễn Du", "thạch bản", "Thơ lục bát", 10,
          "giao_nop", ["KimVanKieu1884/pages"], "page_0002", ds_out="prepared/KimVanKieu1884/dataset_out"),
    _book("Chrestomathie1872", "Chr", "Chrestomathie1872", "Chrestomathie Annamite (1872)",
          "Giáo trình văn xuôi tiếng An Nam (1872)", "Trương Vĩnh Ký / E. Luro", "sách in", "Văn xuôi", 7,
          "giao_nop", ["Chrestomathie1872/pages"], "page_0002", ds_out="prepared/Chrestomathie1872/dataset_out"),
    _book("LucVanTien1916", "L16", "LucVanTien1916", "Lục Vân Tiên (1916)",
          "Mộc bản 1916 — tập đánh giá IHR-NomDB", "Nguyễn Đình Chiểu", "mộc bản", "Thơ lục bát", 10,
          "danh_gia_ihr", ["LucVanTien1916/pages"], "page_0002", ds_out="prepared/LucVanTien1916/dataset_out"),
    _book("TruyenKieu1872", "TK", "TruyenKieu1872", "Truyện Kiều (1872)",
          "Mộc bản Duy Minh Thị (1872) — tập đánh giá IHR-NomDB", "Nguyễn Du", "mộc bản",
          "Thơ lục bát", 10, "danh_gia_ihr", ["TruyenKieu1872/pages"], "page_0002",
          ds_out="prepared/TruyenKieu1872/dataset_out"),
    _book("SachKinhThayCaBinh", "B18", "SachKinhThayCaBinh", "Sách kinh Thầy cả Bỉnh (Borg.tonch.18)",
          "Bản chép tay Vatican — chữ viết tay Công giáo, tập đánh giá", "Philiphê Bỉnh",
          "viết tay", "Văn xuôi", "auto", "danh_gia_borg",
          ["_auto/SachKinhThayCaBinh/pages", "SachKinhThayCaBinh/pages"],
          ds_out="prepared/_auto/SachKinhThayCaBinh/dataset_out"),
    _book("SachDungLyHoThan", "B34", "SachDungLyHoThan", "Sách Dũng Lý Hộ Thần (Borg.tonch.34)",
          "Bản chép tay Vatican — chữ viết tay Công giáo, tập đánh giá", "Khuyết danh",
          "viết tay", "Văn xuôi", "auto", "danh_gia_borg",
          ["_auto/SachDungLyHoThan/pages", "SachDungLyHoThan/pages"],
          ds_out="prepared/_auto/SachDungLyHoThan/dataset_out"),
]
BOOK_BY_ID = {b["id"]: b for b in BOOKS}
AUTO_IDS = [b["id"] for b in BOOKS]
BOOK_GROUPS = {
    "all": AUTO_IDS,
    "giao_nop": [b["id"] for b in BOOKS if b["role"] == "giao_nop"],
    "danh_gia": [b["id"] for b in BOOKS if b["role"] in ("danh_gia_ihr", "danh_gia_borg")],
}
ALIASES = {
    "lvt1883": "LucVanTien1883", "l83": "LucVanTien1883", "lvt1916": "LucVanTien1916", "l16": "LucVanTien1916",
    "kieu1884": "KimVanKieu1884", "kvk": "KimVanKieu1884", "kieu1872": "TruyenKieu1872", "tk": "TruyenKieu1872",
    "chresto1872": "Chrestomathie1872", "chr": "Chrestomathie1872", "stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4",
    "stt11": "SachThanhTruyen11", "sachthanhtruyen": "SachThanhTruyen11", "b18": "SachKinhThayCaBinh",
    "borg18": "SachKinhThayCaBinh", "b34": "SachDungLyHoThan", "borg34": "SachDungLyHoThan",
}


def resolve_book_id(bid: str | None) -> str:
    """Chuẩn hoá mã sách (không phân biệt hoa thường, hỗ trợ bí danh)."""
    if not bid:
        return "LucVanTien1883"
    b = bid.strip()
    if b in BOOK_BY_ID or b in BOOK_GROUPS:
        return b
    low = b.lower()
    for k in BOOK_BY_ID:
        if k.lower() == low:
            return k
    return ALIASES.get(low, b)


def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt (tra cứu gõ không dấu)."""
    if not s:
        return ""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


def tier_order(found) -> list[str]:
    """Thứ tự cột tầng: GOLD, SYLLABLE, (GOLD_text_only nếu có), REVIEW, QUARANTINE, rồi tầng lạ có ô."""
    found = Counter(found)
    cols = [t for t in TIER_ORDER if t in TIER_ALWAYS or found.get(t)]
    return cols + sorted(t for t, n in found.items() if n and t not in TIER_ORDER)


# =====================================================================================================================
# Định dạng số kiểu Việt Nam
# =====================================================================================================================
def fmt_int(x) -> str:
    if x is None:
        return NA
    try:
        return f"{int(x):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(x)


def fmt_pct(x, d: int = 2, scale: float = 100.0) -> str:
    """tỉ lệ 0..1 (scale 100) hoặc đã là % (scale 1) -> '98,05 %'."""
    if x is None:
        return NA
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NA
    if math.isnan(v):
        return NA
    return f"{scale * v:.{d}f} %".replace(".", ",")


def dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# =====================================================================================================================
# Đọc tệp an toàn
# =====================================================================================================================
def rel_to(p: Path, root: Path) -> str:
    try:
        return str(Path(p).relative_to(root))
    except ValueError:
        return str(p)


def read_json(p: Path):
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — tệp đang ghi dở / hỏng -> coi như chưa có
        return None
    return None


def read_csv_cols(p: Path, cols: list[str]) -> tuple[list[tuple], list[str]]:
    """Đọc CHỈ các cột cần (nhanh, ít RAM). Cột thiếu -> ''. Trả (danh sách tuple, danh sách cột có mặt)."""
    with open(p, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd, None) or []
        idx = {c: i for i, c in enumerate(hdr)}
        pos = [idx.get(c) for c in cols]
        out = []
        for row in rd:
            n = len(row)
            out.append(tuple((row[i] if i is not None and i < n else "") for i in pos))
    return out, [c for c in cols if c in idx]


def image_size(p: Path) -> tuple[int, int] | None:
    """(rộng, cao) từ header PNG/JPEG — không cần thư viện ảnh."""
    try:
        with open(p, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":
                        b = f.read(1)
                    while b == b"\xff":
                        b = f.read(1)
                    if not b:
                        return None
                    marker = b[0]
                    if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                        continue
                    ln = int.from_bytes(f.read(2), "big")
                    if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                        f.read(1)
                        h = int.from_bytes(f.read(2), "big")
                        w = int.from_bytes(f.read(2), "big")
                        return w, h
                    f.seek(ln - 2, 1)
    except OSError:
        return None
    return None


def safe_join(base: Path, rel: str) -> Path | None:
    """base/rel nếu (sau khi resolve symlink) vẫn nằm trong base và là tệp; ngược lại None."""
    rel = unquote(rel or "").lstrip("/")
    if not rel or "\x00" in rel:
        return None
    parts = Path(rel).parts
    if any(p == ".." for p in parts):
        return None
    try:
        b = base.resolve()
        p = (b / rel).resolve()
    except (OSError, RuntimeError):
        return None
    if (p == b or b in p.parents) and p.is_file():
        return p
    return None


def parse_bbox(s: str):
    try:
        v = json.loads(s) if s else None
        if isinstance(v, list) and len(v) == 4:
            return [float(x) for x in v]
    except (ValueError, TypeError):
        pass
    return None


# =====================================================================================================================
# Kho dữ liệu nạp vào RAM (ảnh chụp bất biến; /api/reload thay nguyên khối)
# =====================================================================================================================
class Row:
    __slots__ = ("image", "page", "column", "ocr_char", "syllable", "syl_norm", "label", "unicode", "tier", "rule",
                 "bbox")

    def __init__(self, image, page, column, ocr_char, syllable, label, unicode_, tier, rule, bbox):
        self.image, self.page, self.column = image, page, column
        self.ocr_char, self.syllable, self.label, self.unicode = ocr_char, syllable, label, unicode_
        self.syl_norm = strip_accents(syllable)
        self.tier, self.rule, self.bbox = tier, rule, bbox


class BookData:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rows: list[Row] = []
        self.pages: dict[str, list[int]] = {}
        self.status = "chua_co"
        self.source = None
        self.note = ""
        self.tiers: Counter = Counter()
        self.crop_bases: list[Path] = []
        self.eval_marked = False
        self.scan_pages: list[str] = []
        self.pages_dir: Path | None = None
        self.best_page: str | None = None     # trang minh hoạ tốt nhất (nhiều ô GOLD nhất)
        self.review_source: str | None = None  # nguồn ô REVIEW/QUARANTINE (bảng mọi tầng của bản dựng)


def _scan_pages(root: Path, cfg: dict) -> tuple[Path | None, list[str]]:
    for rel in cfg["pages_dirs"]:
        d = root / "prepared" / rel
        if d.is_dir():
            try:
                names = sorted(p.stem for p in d.iterdir()
                               if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.stem.startswith("page_"))
            except OSError:
                names = []
            if names:
                return d, names
    return None, []


def _finish(bd: BookData, root: Path):
    score: dict[str, list] = {}
    for i, r in enumerate(bd.rows):
        bd.pages.setdefault(r.page, []).append(i)
        bd.tiers[r.tier] += 1
        s = score.setdefault(r.page, [0, 0])
        s[0] += r.tier == "GOLD"
        s[1] += 1
    if score:
        bd.best_page = max(sorted(score), key=lambda pg: tuple(score[pg]))
    bd.pages_dir, bd.scan_pages = _scan_pages(root, bd.cfg)


LABEL_COLS = ["image", "book", "page", "column", "ocr_char", "syllable", "label", "unicode", "tier", "rule", "bbox"]
ALL_COLS = LABEL_COLS + ["book_set", "evaluation_only"]


def load_book(root: Path, cfg: dict, cache: dict) -> BookData:
    bd = BookData(cfg)
    ds = root / "dataset"
    bs, key = cfg["book_set"], cfg.get("book_key")
    cands = [("ok", ds / bs, key)]
    if cfg.get("legacy"):
        cands.append(("ban_cu", ds / cfg["legacy"], None))
    for status, d, k in cands:
        lp = d / "labels.csv"
        if not (lp.is_file() and lp.stat().st_size > 0):
            continue
        ck = str(d)
        if ck not in cache:
            cache[ck] = read_csv_cols(lp, LABEL_COLS)[0]    # tuple theo LABEL_COLS (ít RAM hơn dict từng dòng)
        L = cache[ck]
        if k and not any(r[1] == k for r in L):
            continue                      # bản gộp 3 quyển chưa có quyển này -> thử bản theo quyển
        for img, book, page, col, ocr, syl, lab, uni, tier, rule, bbox in L:
            if k and book != k:
                continue
            bd.rows.append(Row(img, page.strip(), col, ocr, syl, lab, uni, tier, rule, bbox))
        bd.status, bd.source = status, rel_to(lp, root)
        bd.crop_bases = [d, ds / "_ALL" / "crops" / bs]
        bd.eval_marked = (d / "evaluation_only.json").is_file()
        if status == "ban_cu":
            bd.note = f"Đang hiển thị bản xuất cũ theo quyển; dataset/{bs}/ chưa có quyển này."
        if not key:
            cache.pop(ck, None)           # bộ một sách: giải phóng ngay (STT giữ cho 3 quyển)
        break
    else:
        # dự phòng: bộ gộp dataset/_ALL/labels.csv (ảnh 'crops/<bs>/…' tương đối dataset/_ALL)
        la = ds / "_ALL" / "labels.csv"
        if la.is_file():
            ck = str(la) + "#cols"
            if ck not in cache:
                cache[ck] = read_csv_cols(la, ALL_COLS)[0]
            for (img, book, page, col, ocr, syl, lab, uni, tier, rule, bbox, rbs, ev) in cache[ck]:
                if rbs != bs or (key and book != key):
                    continue
                bd.rows.append(Row(img, page.strip(), col, ocr, syl, lab, uni, tier, rule, bbox))
                bd.eval_marked = bd.eval_marked or ev == "1"
            if bd.rows:
                bd.status, bd.source = "tu_bo_gop", rel_to(la, root)
                bd.crop_bases = [ds / "_ALL"]
    add_review_rows(root, cfg, bd)
    _finish(bd, root)
    return bd


FULL_TIERS_EXTRA = ("REVIEW", "QUARANTINE")   # tầng KHÔNG được đóng gói vào dataset/<Bộ>/labels.csv (không giao ảnh)


def full_tier_file(root: Path, cfg: dict) -> Path:
    """Bảng MỌI tầng của bản dựng (cùng nguồn với scripts/bao_cao_tong_hop.py): STT dataset_out/labels_final.csv (cột book),
    sách khác <ds_out>/labels_gated.csv."""
    if cfg.get("book_key"):
        return root / "dataset_out" / "labels_final.csv"
    return root / (cfg.get("ds_out") or f"prepared/{cfg['book_set']}/dataset_out") / "labels_gated.csv"


def add_review_rows(root: Path, cfg: dict, bd: BookData) -> None:
    """Thêm ô REVIEW / QUARANTINE (chỉ hộp + nhãn, KHÔNG crop) để web hiện đủ kết quả GOLD / REVIEW như bản đầu."""
    if bd.status not in ("ok", "tu_bo_gop"):
        return
    fp = full_tier_file(root, cfg)
    if not fp.is_file():
        bd.note = (bd.note + " " if bd.note else "") + f"Chưa có {rel_to(fp, root)} — không hiện được ô REVIEW/QUARANTINE."
        return
    rows, _ = read_csv_cols(fp, LABEL_COLS)
    key = cfg.get("book_key")
    n = 0
    for img, book, page, col, ocr, syl, lab, uni, tier, rule, bbox in rows:
        if tier not in FULL_TIERS_EXTRA or (key and book != key):
            continue
        bd.rows.append(Row("", page.strip(), col, ocr, syl, lab, uni, tier, rule, bbox))
        n += 1
    bd.review_source = rel_to(fp, root) if n else None


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.books: dict[str, BookData] = {}
        self.dict_entries = None
        self.dims: dict[str, tuple] = {}
        self.dims_lock = threading.Lock()
        self.box_decoder: dict[str, str] = {}


def load_box_decoders(log=print) -> dict[str, str]:
    """Bộ giải mã hộp (box_decoder) từng bộ = bảng ĐƯỜNG CHẠY của run_pipeline.sh (pipeline.tools.duong_chay, đọc config
    hiện hành). Tuỳ chọn: cần .venv (PyYAML + pipeline/); thiếu thì thẻ sách không ghi phần "hộp …"."""
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from pipeline.tools.duong_chay import ALL10, routes  # noqa: PLC0415 — tuỳ chọn, không bắt buộc
        rows = routes(ALL10)[0]
        out = {str(r[0]): str(r[3]) for r in rows if len(r) > 3 and r[3]}
    except Exception as e:  # noqa: BLE001
        log(f"  [!!] bộ giải mã hộp: không đọc được ({type(e).__name__}: {e})")
        return {}
    log(f"  [OK] bộ giải mã hộp: " + ", ".join(f"{k}={v}" for k, v in out.items()))
    return out


def code_file(root: Path, rel: str) -> Path:
    """Tệp MÃ/TỪ ĐIỂN (Dict/): lấy dưới gốc dữ liệu nếu có, không thì dưới repo."""
    p = root / rel
    return p if p.exists() else REPO_ROOT / rel


def load_store(root: Path, log=print) -> Store:
    t0 = time.time()
    st = Store(root)
    log(f"[server] Nạp dữ liệu từ {root} ...")
    cache: dict = {}
    for cfg in BOOKS:
        try:
            bd = load_book(root, cfg, cache)
        except Exception as e:  # noqa: BLE001 — một bộ hỏng không được làm sập máy chủ
            bd = BookData(cfg)
            bd.status, bd.note = "loi", f"Lỗi đọc dữ liệu: {type(e).__name__}: {e}"
            _finish(bd, root)
        st.books[cfg["id"]] = bd
        mark = "OK" if bd.rows else "!!"
        tiers = " · ".join(f"{t} {bd.tiers[t]:,}" for t in tier_order(bd.tiers) if bd.tiers.get(t))
        log(f"  [{mark}] {cfg['id']}: {len(bd.rows):,} dòng ({tiers or '—'}) · {len(bd.pages)} trang nhãn · "
            f"{len(bd.scan_pages)} ảnh trang · {STATUS_VI[bd.status]}")
    cache.clear()
    st.box_decoder = load_box_decoders(log)
    try:
        dpath = code_file(root, "Dict/QuocNgu_SinoNom.csv")
        if dpath.is_file():
            with open(dpath, encoding="utf-8", newline="") as f:
                st.dict_entries = max(0, sum(1 for _ in f) - 1)
    except OSError:
        pass
    log(f"[server] Nạp xong trong {time.time() - t0:.1f} s")
    return st


STORE: Store | None = None
RELOAD_LOCK = threading.Lock()


# =====================================================================================================================
# Đường dẫn URL <-> tệp
# =====================================================================================================================
def crop_url(bid: str, image: str) -> str | None:
    if not image:
        return None
    return f"/crops/{bid}/{quote(image)}"


def file_for_url(st: Store, url: str) -> Path | None:
    """URL tệp của máy chủ -> tệp trên đĩa (cùng luật với bộ phục vụ; None nếu không có / ngoài vùng cho phép)."""
    path = urlparse(url).path
    parts = path.strip("/").split("/", 2)
    if len(parts) < 3:
        return None
    kind = parts[0]
    if kind == "crops":
        bd = st.books.get(resolve_book_id(unquote(parts[1])))
        if not bd:
            return None
        for base in bd.crop_bases:
            p = safe_join(base, parts[2])
            if p:
                return p
        return None
    if kind == "page_scans":
        cfg = BOOK_BY_ID.get(resolve_book_id(unquote(parts[1])))
        if not cfg:
            return None
        for rel in cfg["pages_dirs"]:
            p = safe_join(st.root / "prepared" / rel, parts[2])
            if p:
                return p
    return None


def scan_file(st: Store, bd: BookData, page: str) -> Path | None:
    for rel in bd.cfg["pages_dirs"]:
        for ext in (".png", ".jpg", ".jpeg"):
            p = st.root / "prepared" / rel / f"{page}{ext}"
            if p.is_file():
                return p
    return None


def page_dims(st: Store, p: Path) -> tuple[int, int] | None:
    k = str(p)
    with st.dims_lock:
        if k in st.dims:
            return st.dims[k]
    d = image_size(p)
    with st.dims_lock:
        st.dims[k] = d
    return d


# =====================================================================================================================
# Số đo đọc từ tệp (thiếu -> None, mục tương ứng không hiển thị)
# =====================================================================================================================
def invariants_summary(root: Path) -> dict | None:
    """measure_out/SUMMARY.json -> số bất biến PASS / FAIL của bộ đo tái lập."""
    s = read_json(root / "measure_out" / "SUMMARY.json")
    t = (s or {}).get("totals") or {}
    if t.get("inv_pass") is None:
        return None
    return dict(n_pass=t.get("inv_pass"), n_fail=t.get("inv_fail") or 0, generated_at=s.get("generated_at"),
                text=f"Bất biến bộ đo: {fmt_int(t.get('inv_pass'))} PASS / {fmt_int(t.get('inv_fail') or 0)} FAIL")


def sources_all(root: Path) -> tuple[dict | None, str]:
    p = root / "dataset" / "_ALL" / "SOURCES.json"
    return read_json(p), rel_to(p, root)


def cross_summary(root: Path, book_id: str) -> tuple[dict | None, str | None]:
    """Đối soát dị bản của bước cổng (prepared/<Bộ>/dataset_out/auto_precision_*/SUMMARY.json)."""
    cfg = BOOK_BY_ID[book_id]
    ds = root / (cfg.get("ds_out") or f"prepared/{book_id}/dataset_out")
    for sub in ("auto_precision_verify", "auto_precision_gated", "auto_precision"):
        p = ds / sub / "SUMMARY.json"
        d = read_json(p)
        if d is not None:
            return dig(d, "cross", "books", book_id), rel_to(p, root)
    return None, None


# =====================================================================================================================
# Tải trọng API (hàm thuần — build_sample_data.py dùng lại)
# =====================================================================================================================
def book_summary(st: Store, bd: BookData) -> dict:
    cfg = bd.cfg
    total = len(bd.rows)
    t = bd.tiers
    gold = t.get("GOLD", 0)
    dec = st.box_decoder.get(cfg["id"])
    layout = cfg["layout_desc"] + (f" · hộp {dec}" if dec else "")
    return dict(id=cfg["id"], title=cfg["title"], subtitle=cfg["subtitle"], author=cfg["author"], layout=layout,
                role=cfg["role"], role_vi=ROLE_VI[cfg["role"]], script=cfg["script"], genre=cfg["genre"],
                set8=cfg["set8"], book_set=cfg["book_set"], status=bd.status, status_vi=STATUS_VI[bd.status],
                note=bd.note, source=bd.source, review_source=bd.review_source, total=total, tiers=dict(t), gold=gold,
                syllable=t.get("SYLLABLE", 0), gold_text_only=t.get("GOLD_text_only", 0),
                review=t.get("REVIEW", 0), quarantine=t.get("QUARANTINE", 0),
                gold_pct=round(gold / total * 100, 1) if total else 0,
                pages=len(bd.pages), n_scans=len(bd.scan_pages),
                evaluation_only=bd.eval_marked or cfg["role"] != "giao_nop")


def api_stats(st: Store) -> dict:
    books, tiers = {}, Counter()
    by_role = {"giao_nop": 0, "danh_gia": 0}
    n_with = 0
    for bid in AUTO_IDS:
        bd = st.books.get(bid)
        if not bd:
            continue
        b = book_summary(st, bd)
        books[bid] = b
        n_with += bool(b["total"])
        tiers.update(bd.tiers)
        by_role["giao_nop" if b["role"] == "giao_nop" else "danh_gia"] += b["total"]
    total = sum(tiers.values())
    inv = invariants_summary(st.root)
    im = dict(
        total_characters=total, total_gold=tiers.get("GOLD", 0), total_syllable=tiers.get("SYLLABLE", 0),
        total_gold_text_only=tiers.get("GOLD_text_only", 0), total_review=tiers.get("REVIEW", 0),
        total_quarantine=tiers.get("QUARANTINE", 0), tier_totals=dict(tiers), tier_columns=tier_order(tiers),
        gold_rate=round(tiers.get("GOLD", 0) / total * 100, 1) if total else None,
        books_count=len(AUTO_IDS), books_with_data=n_with,
        deliver_characters=by_role["giao_nop"], eval_characters=by_role["danh_gia"],
        invariants_available=inv is not None, invariants_text=(inv or {}).get("text"),
        invariants_pass=(inv or {}).get("n_pass"), invariants_fail=(inv or {}).get("n_fail"),
        invariants_generated_at=(inv or {}).get("generated_at"), dict_entries=st.dict_entries,
    )
    src_all, src_all_path = sources_all(st.root)
    all_info = None
    if src_all:
        all_info = dict(n_dong=src_all.get("n_dong"), n_tep_crop=src_all.get("n_tep_crop"),
                        eval_only_dong=src_all.get("eval_only_dong"), n_bo=src_all.get("n_bo"),
                        ngay=src_all.get("ngay"), source=src_all_path)
    return dict(
        title="Hệ thống hỗ trợ gán nhãn tự động văn bản Hán Nôm cổ (GanNhanOCR)",
        subtitle="Trực quan hoá tầng nhãn của bộ dữ liệu thực nghiệm — luận văn Thạc sĩ",
        data_root=str(st.root), loaded_at=st.loaded_at, impact_metrics=im, books=books, all_dataset=all_info,
        tier_labels=TIER_VI,
    )


def api_books(st: Store) -> list[dict]:
    out = []
    for cfg in BOOKS:
        bd = st.books.get(cfg["id"])
        if not bd:
            continue
        pages = sorted(bd.pages) if bd.pages else list(bd.scan_pages)
        default = bd.best_page if bd.best_page in pages else cfg.get("default_page")   # trang nhiều ô GOLD nhất
        if default not in pages:
            default = pages[0] if pages else None
        b = book_summary(st, bd)
        b.update(n_columns=cfg["n_columns"], layout_desc=cfg["layout_desc"], available_pages=pages,
                 sample_pages=pages[:5], default_page=default, total_chars=len(bd.rows))
        out.append(b)
    return out


def char_payload(bd: BookData, r: Row) -> dict:
    return dict(image_rel=r.image, column=r.column, ocr_char=r.ocr_char, syllable=r.syllable, label=r.label,
                unicode=r.unicode, tier=r.tier, rule=r.rule, page=r.page,
                crop_url=crop_url(bd.cfg["id"], r.image) if r.tier != "GOLD_text_only" else None)  # text_only: không có crop


def api_page(st: Store, query: dict) -> tuple[dict, int]:
    bid = resolve_book_id((query.get("book") or ["LucVanTien1883"])[0])
    bd = st.books.get(bid)
    if not bd:
        return {"error": f"Không tìm thấy sách: {bid}"}, 404
    cfg = bd.cfg
    pages = sorted(bd.pages) if bd.pages else list(bd.scan_pages)
    page = (query.get("page") or [None])[0] or (cfg.get("default_page") if cfg.get("default_page") in bd.pages else None) \
        or bd.best_page or (pages[0] if pages else "page_0001")
    idx = bd.pages.get(page, [])
    sf = scan_file(st, bd, page)
    dims = page_dims(st, sf) if sf else None
    chars = []
    boxes = [(r, parse_bbox(r.bbox)) for r in (bd.rows[i] for i in idx)]
    boxes = [(r, b) for r, b in boxes if b]
    if not dims:
        mx = max((b[2] for _, b in boxes), default=0)
        my = max((b[3] for _, b in boxes), default=0)
        dims = (int(mx * 1.04) or 1896, int(my * 1.03) or 3212)
    W, H = dims

    def colkey(r):
        try:
            return int(float(r.column))
        except (TypeError, ValueError):
            return 9999

    boxes.sort(key=lambda rb: (colkey(rb[0]), rb[1][1]))
    for n, (r, (x0, y0, x1, y1)) in enumerate(boxes, 1):
        c = char_payload(bd, r)
        c.update(index=n, bbox=[x0, y0, x1, y1],
                 rect=dict(left_pct=round(x0 / W * 100, 3), top_pct=round(y0 / H * 100, 3),
                           width_pct=round((x1 - x0) / W * 100, 3), height_pct=round((y1 - y0) / H * 100, 3)))
        try:
            c["column"] = int(float(r.column))
        except (TypeError, ValueError):
            pass
        chars.append(c)
    msg = ""
    if not bd.rows:
        msg = STATUS_VI[bd.status] + (f" — {bd.note}" if bd.note else "")
    elif not idx:
        msg = "Trang này không có ô chữ nào trong bảng nhãn."
    return dict(book=bid, book_title=cfg["title"], page=page, status=bd.status, status_vi=STATUS_VI[bd.status],
                note=bd.note, message=msg, scan_url=f"/page_scans/{bid}/{quote(sf.name)}" if sf else None,
                has_scan=bool(sf), dimensions=dict(width=W, height=H), char_count=len(chars), n_rows_page=len(idx),
                tier_counts=dict(Counter(c["tier"] for c in chars)), characters=chars), 200


def api_search(st: Store, query: dict) -> dict:
    q = (query.get("q") or [""])[0].strip().lower()
    qn = strip_accents(q)
    bsel = resolve_book_id((query.get("book") or ["all"])[0])
    tier = (query.get("tier") or ["all"])[0]
    try:
        limit = max(1, min(int((query.get("limit") or [60])[0]), 300))
    except ValueError:
        limit = 60
    ids = BOOK_GROUPS.get(bsel) or ([bsel] if bsel in st.books else AUTO_IDS)
    tier_all = tier.lower() == "all"
    per_book: list[tuple[BookData, list[Row]]] = []
    n_total = 0
    for bid in ids:
        bd = st.books.get(bid)
        if not bd or not bd.rows:
            continue
        hits = []
        for r in bd.rows:
            if not tier_all and r.tier != tier and r.tier.upper() != tier.upper():
                continue
            if q and not ((q in r.syllable.lower()) or (qn and qn in r.syl_norm) or (q in r.label.lower())
                          or (q in r.ocr_char.lower()) or (q in r.unicode.lower())):
                continue
            hits.append(r)
        n_total += len(hits)
        if hits:
            per_book.append((bd, hits))
    # chọn đều giữa các sách và rải đều trong mỗi sách (không chỉ lấy đầu bảng)
    picked: list[tuple[BookData, Row]] = []
    if per_book:
        quota = max(1, math.ceil(limit / len(per_book)))
        seen = set()
        for bd, hits in per_book:
            step = max(1, len(hits) // quota)
            for r in hits[::step][:quota]:
                picked.append((bd, r))
                seen.add(id(r))
        for bd, hits in per_book:              # còn chỗ -> lấp bằng kết quả chưa chọn
            if len(picked) >= limit:
                break
            for r in hits:
                if len(picked) >= limit:
                    break
                if id(r) not in seen:
                    picked.append((bd, r))
                    seen.add(id(r))
        picked = picked[:limit]
    results = []
    for bd, r in picked:
        c = char_payload(bd, r)
        c.update(book=bd.cfg["id"], book_title=bd.cfg["title"], image=r.image)
        results.append(c)
    return dict(query=q, book=bsel, tier=tier, count=len(results), total_matches=n_total, results=results)


def api_pipeline_flow(st: Store) -> list[dict]:
    """7 bước, mỗi bước một dòng: tên · đầu vào · phương pháp · đầu ra (+ số đo ngắn đọc từ tệp, thiếu thì bỏ)."""
    root = st.root
    de = f"từ điển Quốc ngữ ↔ Hán Nôm ({fmt_int(st.dict_entries)} mục)" if st.dict_entries else "từ điển Quốc ngữ ↔ Hán Nôm"

    def m(label, value, source=""):
        return dict(label=label, value=value, source=source)

    # Bước 2: hộp legacy -> pitch trên hộp tham chiếu tự động (chỉ khi có measure_out/box_ref/summary.json)
    box_p = root / "measure_out" / "box_ref" / "summary.json"
    box, box_src = read_json(box_p), rel_to(box_p, root)
    det_metrics = []
    for bid in ("LucVanTien1883", "KimVanKieu1884"):
        cv = dig(box, "per_book", bid, "cells_verified") or {}
        a, b = (cv.get("legacy@0.15_prepared") or {}), (cv.get("pitch_prepared") or {})
        for key, label in (("cut_glyph_pct", "hộp cắt vào chữ"), ("ok_iou50_pct", "hộp khớp tham chiếu (IoU ≥ 0,5)")):
            if a.get(key) is not None and b.get(key) is not None:
                det_metrics.append(m(f"{BOOK_BY_ID[bid]['title']}: {label}, legacy → pitch",
                                     f"{fmt_pct(a[key], 1, 1)} → {fmt_pct(b[key], 1, 1)}", box_src))

    # Bước 5: tỉ lệ GOLD trùng chữ dị bản (cận dưới)
    gate_metrics = []
    for bid in ("LucVanTien1883", "KimVanKieu1884"):
        c, src = cross_summary(root, bid)
        for ref, blk in ((c or {}).get("refs") or {}).items():
            at = (blk or {}).get("all_tiers") or {}
            if "GOLD_eq_pct" in at:
                gate_metrics.append(m(f"{BOOK_BY_ID[bid]['title']}: GOLD trùng chữ dị bản {ref} (cận dưới)",
                                      fmt_pct(at["GOLD_eq_pct"], 1, 1) + f" (n = {fmt_int(at.get('n_GOLD'))})", src))

    # Bước 7: bộ gộp dataset/_ALL
    src_all, src_all_path = sources_all(root)
    merge_metrics = []
    if src_all:
        bb = src_all.get("bat_bien") or {}
        merge_metrics = [m("Số dòng dataset/_ALL/labels.csv", fmt_int(src_all.get("n_dong")), src_all_path),
                         m("Trong đó evaluation_only", fmt_int(src_all.get("eval_only_dong")), src_all_path)]
        if bb:
            merge_metrics.append(m("Bất biến bước gộp", f"{sum(1 for v in bb.values() if v is True)}/{len(bb)} PASS",
                                   src_all_path))

    def step(n, id_, name, input_, method, output, metrics=()):
        return dict(step=n, id=id_, name=name, input=input_, method=method, output=output, metrics=list(metrics))

    return [
        step(1, "ingest", "Tiền xử lý, OCR", "Ảnh quét; bản Quốc ngữ",
             "Chuẩn hoá ảnh; OCR chữ Nôm và Quốc ngữ (lưu bộ đệm)", "Ảnh trang, kết quả OCR"),
        step(2, "detect", "Phát hiện hộp chữ", "Ảnh cột chữ",
             "CenterNet (ResNet-34) + bộ giải mã hộp", "Hộp bao từng chữ", det_metrics),
        step(3, "align", "Gióng hàng", "Hộp chữ; chuỗi âm Quốc ngữ",
             f"Quy hoạch động dải hẹp; {de}", "Nhãn sơ bộ, mã luật"),
        step(4, "remediate", "Kiểm kê, xếp tầng", "Nhãn sơ bộ",
             "Kiểm trùng; sửa nhầm lẫn có hệ thống", "Tầng nhãn của từng ô"),
        step(5, "gates", "Cổng kiểm tra", "Ô nghi vấn",
             "Cổng cơ chế; so với dị bản độc lập", "labels_gated.csv", gate_metrics),
        step(6, "export", "Đóng gói theo bộ", "Nhãn; ảnh crop",
             "Xuất bảng nhãn và crop theo tầng", "dataset/<Bộ>/"),
        step(7, "merge", "Gộp các bộ", "10 bộ",
             "Gộp theo khoá cell_uid; kiểm bất biến", "dataset/_ALL/", merge_metrics),
    ]


def tier_table(st: Store) -> dict:
    """Bảng tầng nhãn theo bộ: 10 bộ + dòng tổng (đếm trực tiếp trên labels.csv đã nạp)."""
    rows, tot = [], Counter()
    for bid in AUTO_IDS:
        bd = st.books.get(bid)
        if not bd:
            continue
        tot.update(bd.tiers)
        n = len(bd.rows)
        rows.append(dict(book=bid, book_title=bd.cfg["title"], role=bd.cfg["role"], role_vi=ROLE_VI[bd.cfg["role"]],
                         status=bd.status, status_vi=STATUS_VI[bd.status], total=n, counts=dict(bd.tiers),
                         gold_pct=round(bd.tiers.get("GOLD", 0) / n * 100, 1) if n else None, source=bd.source))
    n_all = sum(tot.values())
    cols = tier_order(tot)
    return dict(columns=cols, labels={t: TIER_VI.get(t, t) for t in cols}, rows=rows,
                totals=dict(total=n_all, counts=dict(tot),
                            gold_pct=round(tot.get("GOLD", 0) / n_all * 100, 1) if n_all else None),
                empty_tiers=[t for t in cols if not tot.get(t)])


def api_benchmarks(st: Store) -> dict:
    return dict(title="Kết quả — tầng nhãn theo bộ (đếm trên labels.csv)", data_root=str(st.root),
                loaded_at=st.loaded_at, tier_table=tier_table(st))


# =====================================================================================================================
# HTTP
# =====================================================================================================================
DATA_PREFIXES = ("/crops/", "/page_scans/")


class GanNhanHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def log_message(self, fmt, *args):  # gọn log: bỏ ảnh
        if self.path.startswith(DATA_PREFIXES):
            return
        super().log_message(fmt, *args)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path.startswith(("/api/",) + DATA_PREFIXES):
            self.do_GET()
        else:
            super().do_HEAD()

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        try:
            if path.startswith("/api/"):
                self.handle_api(path, parse_qs(url.query))
            elif path.startswith(DATA_PREFIXES):
                self.serve_data_file(path)
            else:
                super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001 — không để một yêu cầu làm sập máy chủ
            try:
                self.send_json({"error": f"{type(e).__name__}: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception:  # noqa: BLE001
                pass

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_text(self, status, msg: str):
        """Như send_error nhưng thông điệp UTF-8 nằm trong thân (send_error mã hoá latin-1 -> lỗi với tiếng Việt)."""
        body = msg.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def handle_api(self, path: str, query: dict):
        st = STORE
        if st is None:
            self.send_json({"error": "Máy chủ đang nạp dữ liệu"}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if path == "/api/stats":
            self.send_json(api_stats(st))
        elif path == "/api/books":
            self.send_json(api_books(st))
        elif path == "/api/page":
            data, code = api_page(st, query)
            self.send_json(data, status=code)
        elif path == "/api/search":
            self.send_json(api_search(st, query))
        elif path == "/api/pipeline_flow":
            self.send_json(api_pipeline_flow(st))
        elif path == "/api/benchmarks":
            self.send_json(api_benchmarks(st))
        elif path == "/api/reload":
            self.send_json(reload_store())
        else:
            self.send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    def serve_data_file(self, path: str):
        st = STORE
        f = file_for_url(st, path) if st else None
        if f and f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"):
            f = None                      # vùng dữ liệu chỉ phục vụ ẢNH (không lộ labels.csv, cache…)
        if not f:
            self.send_text(HTTPStatus.NOT_FOUND, "Không tìm thấy tệp (hoặc đường dẫn ngoài vùng cho phép).")
            return
        mime, _ = mimetypes.guess_type(str(f))
        try:
            size = f.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            if self.command != "HEAD":
                with open(f, "rb") as h:
                    while chunk := h.read(65536):
                        self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def reload_store() -> dict:
    global STORE
    if not RELOAD_LOCK.acquire(blocking=False):
        return {"ok": False, "message": "Đang nạp lại, thử lại sau."}
    try:
        root = STORE.root if STORE else default_root()
        new = load_store(root)
        STORE = new
        return {"ok": True, "loaded_at": new.loaded_at, "books": {k: len(v.rows) for k, v in new.books.items()}}
    finally:
        RELOAD_LOCK.release()


def run_server(port: int = 8080, host: str = "", root: Path | None = None):
    global STORE
    STORE = load_store(root or default_root())
    try:
        httpd = ThreadingHTTPServer((host, port), GanNhanHandler)
    except OSError as e:
        if port == 8080:
            print(f"[server] Cổng 8080 đang bận ({e}), thử cổng 8088...")
            httpd = ThreadingHTTPServer((host, 8088), GanNhanHandler)
            port = 8088
        else:
            raise
    httpd.daemon_threads = True
    print("\n" + "=" * 65)
    print("  GanNhanOCR — máy chủ trình diễn đang chạy")
    print(f"  Gốc dữ liệu: {STORE.root}")
    print(f"  Truy cập: http://localhost:{port}   (nạp lại dữ liệu: /api/reload)")
    print("=" * 65 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Đã dừng máy chủ.")
    finally:
        httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Máy chủ trình diễn GanNhanOCR (chỉ đọc dữ liệu).")
    ap.add_argument("port", nargs="?", type=int, default=8080)
    ap.add_argument("--root", default=None, help="gốc dữ liệu (mặc định: $GANNHANOCR_ROOT hoặc thư mục cha của web/)")
    ap.add_argument("--host", default="", help="địa chỉ lắng nghe (mặc định mọi giao diện; 127.0.0.1 = chỉ máy này)")
    a = ap.parse_args(argv)
    root = Path(a.root).expanduser().resolve() if a.root else default_root()
    run_server(a.port, a.host, root)


if __name__ == "__main__":
    main()
