#!/usr/bin/env python3
"""Thu hoạch Google Cloud Vision (DOCUMENT_TEXT_DETECTION) cho các trang của 10 cuốn, GHÉP 3–4 TRANG/ẢNH để vừa trần 950 yêu cầu/tháng,
lưu về đĩa (ký hiệu + hộp + độ tin cậy, đã quy về TOẠ ĐỘ TRANG GỐC) để đối chứng với kim (vision/vision_crosscheck.py).

Bản 2 (sau phản biện độc lập 04/10): sổ cái có KHOÁ TỆP + đếm TRƯỚC khi gửi (hoàn lại khi chắc chắn không tính phí), nằm NGOÀI repo; khoá chạy một tiến trình;
bộ ngắt sau chuỗi lỗi; trần điểm ảnh mỗi ảnh ghép (--cap-mp, mặc định 5 MP — vùng đã kiểm chứng với Vision thật); toạ độ phát lại từ layout đã lưu;
kiểm bất biến kích thước/toạ độ sau mỗi phản hồi và DỪNG ở ảnh bất thường; mặc định mỗi lần chạy tối đa 25 yêu cầu; không bao giờ in khoá/URL (kể cả đường lỗi).

Lệnh (mặc định KHÔNG gọi mạng):
  python3 vision/vision_harvest.py plan   [--n 4 --layout grid --hints zh-Hant --books KVK,L83]
  python3 vision/vision_harvest.py pilot  --stage 1 --yes     # 20 yêu cầu trên 10 trang cũ: n=1 · n=4 không trần MP · n=4 trần 5 MP · n=3 trần 5 MP
  python3 vision/vision_harvest.py pilot  --stage 2 --yes     # 6 yêu cầu: gợi ý zh-Hant · không gợi ý
  python3 vision/vision_harvest.py run    --yes [--max-requests 0]
  python3 vision/vision_harvest.py status
  python3 vision/vision_harvest.py import-legacy              # đổi 5 phản hồi cache cũ sang định dạng mới (0 yêu cầu)
Chỉ cần Pillow; thêm google-auth + requests khi dùng khoá dịch vụ (hoặc --api-key-env, không cần thư viện).
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent                                           # .../GanNhanOCR/vision
REPO = Path(os.environ.get("VISION_REPO_ROOT", HERE.parent))
ROOT = Path(os.environ.get("VISION_CACHE_ROOT", HERE / "cache"))                  # phản hồi thô + trang đã quy toạ độ (KHÔNG commit)
LEDGER_PATH = Path(os.environ.get("VISION_LEDGER", HERE / "ledger" / "vision_ledger.json"))
LEDGER_MIRROR = os.environ.get("VISION_LEDGER_MIRROR", str(Path.home() / ".cache" / "gannhanocr" / "vision_ledger.json"))   # bản sao ngoài repo; nạp lại lấy MAX từng tháng
LEGACY_LEDGER = Path(os.environ.get("VISION_LEGACY_LEDGER", HERE / "legacy" / "cache" / "google_vision" / "usage_ledger.json"))
LEGACY_DIR = Path(os.environ.get("VISION_LEGACY_DIR", HERE / "legacy" / "cache" / "google_vision" / "stitched"))
ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
MAX_JSON_BYTES = 9_500_000          # trần JSON của API là 10 MB
HARD_CAP = 990                      # không cho --cap vượt trừ khi --allow-over-free (miễn phí 1.000/tháng)
BOOKS = [   # (khoá, thư mục) — sách in trước (giá trị đối chứng cao), viết tay sau
    ("KVK", "KimVanKieu1884"), ("L83", "LucVanTien1883"), ("TK", "TruyenKieu1872"), ("L16", "LucVanTien1916"), ("Chr", "Chrestomathie1872"),
    ("stt11", "SachThanhTruyen11"), ("stt2", "SachThanhTruyen2"), ("stt4", "SachThanhTruyen4"), ("B34", "SachDungLyHoThan"), ("B18", "SachKinhThayCaBinh"),
]
LEGACY10 = [    # 10 trang của kịch bản cũ (mỗi cuốn một trang) — tập thử nghiệm so sánh cách ghép
    ("stt2", "page_0312"), ("stt4", "page_0060"), ("stt11", "page_0016"), ("L83", "page_0048"), ("KVK", "page_0075"),
    ("Chr", "page_0021"), ("TK", "page_0042"), ("L16", "page_0020"), ("B18", "page_0088"), ("B34", "page_0081"),
]
STOP_FLAG = {"v": False}


class PreSendError(Exception):
    """Lỗi xảy ra TRƯỚC khi yêu cầu rời máy (DNS, xác thực): không tính phí."""


class Ambiguous(Exception):
    """Mất kết nối/hết giờ sau khi có thể đã gửi: giả định ĐÃ tính phí."""


class Stop(Exception):
    pass


def scrub(s: str) -> str:
    s = re.sub(r"(key=)[^&\s'\"]+", r"\1***", str(s))
    return re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1***", s)


def current_month() -> str:
    """Kỳ thanh toán của Google theo giờ Pacific."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m")
    except Exception:  # noqa: BLE001
        return time.strftime("%Y-%m", time.gmtime())


# ───────────────────────── cấu hình ─────────────────────────
@dataclass(frozen=True)
class Cfg:
    n: int = 4                     # số trang mỗi ảnh ghép
    layout: str = "grid"           # row | grid
    hints: tuple = ("zh-Hant",)    # languageHints; () = tự nhận
    cell_h: float = 80.0           # cao ô trung vị mong muốn sau co giãn (px)
    smin: float = 0.35
    smax: float = 3.0
    sep: int = 80
    jpeg_q: int = 90
    cap_mp: float = 5.0            # trần điểm ảnh (triệu) của MỖI ảnh ghép: hạ hệ số cả nhóm nếu vượt

    def params(self) -> dict:
        d = asdict(self)
        d["hints"] = list(self.hints)
        return d

    def cid(self) -> str:
        h = "+".join(x.replace("-", "") for x in self.hints) or "auto"
        tag = hashlib.sha256(json.dumps(self.params(), sort_keys=True).encode()).hexdigest()[:6]
        return f"n{self.n}-{self.layout}-{h}-h{int(self.cell_h)}-{tag}"


@dataclass
class PageRec:
    book: str
    bdir: str
    page: str
    path: str
    w: int
    h: int
    md5: str = ""
    orphan: bool = False           # trang có ảnh nhưng KHÔNG có ô trong labels.csv (bìa/trống): xếp nhóm riêng ở cuối để không làm lệch cặp đã chạy


# ───────────────────────── dữ liệu trang ─────────────────────────
_MED_CACHE: dict = {}


def _label_rows(bdir: str):
    with open(REPO / "dataset" / bdir / "labels.csv", encoding="utf-8-sig", newline="") as fh:
        yield from csv.DictReader(fh)


def med_cell_height(bdir: str) -> float:
    if bdir not in _MED_CACHE:
        hs = []
        for row in _label_rows(bdir):
            b = row.get("bbox")
            if b:
                try:
                    v = json.loads(b)
                    hs.append(v[3] - v[1])
                except Exception:  # noqa: BLE001
                    pass
        _MED_CACHE[bdir] = float(statistics.median(hs)) if hs else 80.0
    return _MED_CACHE[bdir]


def list_pages(book: str, bdir: str) -> list[PageRec]:
    """MỌI trang có ảnh trong prepared/<Bộ>/pages_denoised (kể cả trang không có ô → orphan=True)."""
    withcells = {row["page"] for row in _label_rows(bdir) if row.get("page")}
    pdir = REPO / "prepared" / bdir / "pages_denoised"
    out = []
    for f in sorted(pdir.glob("page_*.png")):
        with Image.open(f) as im:
            out.append(PageRec(book, bdir, f.stem, str(f), im.width, im.height, orphan=f.stem not in withcells))
    return out


def file_md5(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def select_pages(books: list[str] | None, subset: str | None) -> dict:
    """{khoá sách: [PageRec]} theo thứ tự ưu tiên; subset='legacy10' chỉ lấy 10 trang cũ."""
    bd = dict(BOOKS)
    res = {}
    for k, _ in BOOKS:
        if books and k not in books:
            continue
        ps = list_pages(k, bd[k])
        if subset == "legacy10":
            want = {p for kk, p in LEGACY10 if kk == k}
            ps = [r for r in ps if r.page in want]
        if ps:
            res[k] = ps
    return res


def plan_groups(pages: dict, n: int, mix: bool = False) -> list[list[PageRec]]:
    """Nhóm n trang liên tiếp theo từng cuốn. Trang orphan (không ô) được nhóm RIÊNG sau cùng của cuốn → cặp của các trang có ô không đổi khi thêm/bớt orphan."""
    def chunks(ps):
        return [ps[i:i + n] for i in range(0, len(ps), n)]
    if mix:
        flat = [r for ps in pages.values() for r in ps]
        return chunks([r for r in flat if not r.orphan]) + chunks([r for r in flat if r.orphan])
    out = []
    for ps in pages.values():
        out += chunks([r for r in ps if not r.orphan]) + chunks([r for r in ps if r.orphan])
    return out


# ───────────────────────── hệ số co giãn (đóng băng theo cấu hình) ─────────────────────────
def get_scales(cfg: Cfg, bdirs, store: "Store | None" = None, write: bool = True) -> dict:
    """Hệ số theo cuốn = cao ô mong muốn / cao ô trung vị (kẹp). ĐÓNG BĂNG vào scales.json của cấu hình ở lần đầu để phát lại/khoá nhóm không đổi khi labels.csv được dựng lại."""
    d = {}
    p = store.dir / "scales.json" if store else None
    if p and p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
    changed = False
    for b in bdirs:
        if b not in d:
            d[b] = round(min(cfg.smax, max(cfg.smin, cfg.cell_h / med_cell_height(b))), 4)
            changed = True
    if p and changed and write:
        write_json(p, d)
    return d


def group_id(group: list[PageRec], cfg: Cfg, scales: dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(cfg.params(), sort_keys=True).encode())
    for r in group:
        h.update(f"|{r.book}/{r.page}/{r.md5 or file_md5(r.path)}/{scales[r.bdir]}".encode())
    return h.hexdigest()


# ───────────────────────── ghép ảnh ─────────────────────────
def arrange(sizes: list, cfg: Cfg):
    """→ (vị trí [(x,y)], rộng, cao) của ảnh ghép; lưới 2×2 (n=3,4) hoặc một hàng."""
    n = len(sizes)
    cols = n if cfg.layout == "row" else math.ceil(math.sqrt(n))
    pos, y, cw = [], 0, 0
    for i in range(0, n, cols):
        row = sizes[i:i + cols]
        x, rh = 0, max(h for _, h in row)
        for w, h in row:
            pos.append((x, y))
            x += w + cfg.sep
        cw = max(cw, x - cfg.sep)
        y += rh + cfg.sep
    return pos, cw, y - cfg.sep


def effective_scales(group: list[PageRec], cfg: Cfg, scales: dict):
    """Hệ số từng trang sau khi hạ CẢ NHÓM để ảnh ghép ≤ cap_mp (không bao giờ phóng thêm)."""
    base = [scales[r.bdir] for r in group]
    g = 1.0
    for _ in range(10):
        sizes = [(max(1, round(r.w * s * g)), max(1, round(r.h * s * g))) for r, s in zip(group, base)]
        _, W, H = arrange(sizes, cfg)
        mp = W * H / 1e6
        if mp <= cfg.cap_mp:
            break
        g *= math.sqrt(cfg.cap_mp / mp) * 0.995
    return [s * g for s in base], g


def canvas_dims(group: list[PageRec], cfg: Cfg, scales: dict):
    es, g = effective_scales(group, cfg, scales)
    sizes = [(max(1, round(r.w * s)), max(1, round(r.h * s))) for r, s in zip(group, es)]
    _, W, H = arrange(sizes, cfg)
    return W, H, g


def to_l(im: Image.Image) -> Image.Image:
    if im.mode.startswith("I"):                       # 16 bit: convert('L') sẽ kẹp trắng/đen → đưa về 8 bit trước
        return im.point(lambda p: p * (1 / 256.0)).convert("L")
    return im.convert("L")


def stitch(group: list[PageRec], cfg: Cfg, scales: dict):
    """→ (ảnh RGB, [placement], g); placement = {rec, x0, y0, w, h, sx, sy} trong toạ độ ảnh ghép."""
    es, g = effective_scales(group, cfg, scales)
    ims = []
    for rec, s in zip(group, es):
        w, h = max(1, round(rec.w * s)), max(1, round(rec.h * s))
        with Image.open(rec.path) as im:
            ims.append((rec, to_l(im).resize((w, h), Image.Resampling.LANCZOS), w, h))
    pos, cw, ch = arrange([(w, h) for _, _, w, h in ims], cfg)
    canvas = Image.new("L", (cw, ch), 255)
    place = []
    for (rec, im, w, h), (x, y) in zip(ims, pos):
        canvas.paste(im, (x, y))
        place.append(dict(rec=rec, x0=x, y0=y, w=w, h=h, sx=w / rec.w, sy=h / rec.h))
    return canvas.convert("RGB"), place, g


class TooBig(Exception):
    pass


def encode_jpeg(canvas: Image.Image, cfg: Cfg) -> bytes:
    for q in (cfg.jpeg_q, 85, 80, 75, 70):
        buf = io.BytesIO()
        canvas.save(buf, "JPEG", quality=q)
        b = buf.getvalue()
        if len(b) * 4 / 3 + 2000 <= MAX_JSON_BYTES:
            return b
    raise TooBig(f"ảnh ghép {canvas.size} vẫn > {MAX_JSON_BYTES / 1e6:.1f} MB sau khi hạ chất lượng — giảm --n hoặc --cap-mp")


def build_payload(jpeg: bytes, hints: tuple) -> bytes:
    req = {"image": {"content": base64.b64encode(jpeg).decode("ascii")}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}
    if hints:
        req["imageContext"] = {"languageHints": list(hints)}
    return json.dumps({"requests": [req]}).encode("utf-8")


# ───────────────────────── phản hồi → trang gốc ─────────────────────────
def _symbols(resp: dict):
    for pg in resp.get("fullTextAnnotation", {}).get("pages", []):
        for b in pg.get("blocks", []):
            for pa in b.get("paragraphs", []):
                for w in pa.get("words", []):
                    yield from w.get("symbols", [])


def parse_symbols(resp: dict) -> list:
    """→ [(ch, conf, x0, y0, x1, y1, ang)] trong toạ độ ảnh gửi đi (đỉnh thiếu x/y nghĩa là 0); ang = góc cạnh v0→v1 (độ)."""
    out = []
    for s in _symbols(resp):
        v = s.get("boundingBox", {}).get("vertices", [])
        if len(v) < 3:
            continue
        xs = [p.get("x", 0) for p in v]
        ys = [p.get("y", 0) for p in v]
        ang = round(math.degrees(math.atan2(ys[1] - ys[0], xs[1] - xs[0])), 1)
        out.append((s.get("text", ""), s.get("confidence"), min(xs), min(ys), max(xs), max(ys), ang))
    return out


def no_vertex_count(resp: dict) -> int:
    return sum(1 for s in _symbols(resp) if len(s.get("boundingBox", {}).get("vertices", [])) < 3)


def vision_dims(resp: dict):
    pg = resp.get("fullTextAnnotation", {}).get("pages", [])
    return (pg[0].get("width"), pg[0].get("height")) if pg else (None, None)


def demux(symbols: list, place: list, canvas_wh: tuple, vwh: tuple):
    """Chia ký hiệu về từng trang (theo TÂM hộp), đổi sang toạ độ trang gốc. → (per, dropped, info).
    Nếu Vision báo kích thước xử lý khác ảnh gửi: chỉ quy đổi khi toạ độ nằm trong khung Vision báo; ngược lại coi là khung ảnh gửi và gắn cờ."""
    fx = fy = 1.0
    frame = "canvas"
    if vwh[0] and vwh[1] and (abs(vwh[0] - canvas_wh[0]) > 1 or abs(vwh[1] - canvas_wh[1]) > 1):
        mx = max((s[4] for s in symbols), default=0)
        my = max((s[5] for s in symbols), default=0)
        if mx <= vwh[0] + 1 and my <= vwh[1] + 1:
            fx, fy = canvas_wh[0] / vwh[0], canvas_wh[1] / vwh[1]
            frame = "rescaled"
        else:
            frame = "anomaly_dims"
    per = {i: [] for i in range(len(place))}
    dropped = 0
    for ch, conf, x0, y0, x1, y1, ang in symbols:
        x0, x1, y0, y1 = x0 * fx, x1 * fx, y0 * fy, y1 * fy
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for i, p in enumerate(place):
            if p["x0"] <= cx < p["x0"] + p["w"] and p["y0"] <= cy < p["y0"] + p["h"]:
                per[i].append([ch, conf, round((x0 - p["x0"]) / p["sx"], 1), round((y0 - p["y0"]) / p["sy"], 1),
                               round((x1 - p["x0"]) / p["sx"], 1), round((y1 - p["y0"]) / p["sy"], 1), ang])
                break
        else:
            dropped += 1
    return per, dropped, dict(frame=frame, fx=fx, fy=fy)


# ───────────────────────── đĩa ─────────────────────────
def atomic_write(path: Path, data: bytes, fsync: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
        if fsync:
            dfd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: Path, obj, fsync: bool = False):
    atomic_write(path, json.dumps(obj, ensure_ascii=False).encode("utf-8"), fsync)


class Store:
    def __init__(self, cfg_id: str):
        self.dir = ROOT / cfg_id

    def page(self, book, page): return self.dir / "pages" / book / f"{page}.json"
    def raw(self, gid): return self.dir / "raw" / f"{gid[:20]}.json.gz"
    def layout(self, gid): return self.dir / "layout" / f"{gid[:20]}.json"
    def stitched(self, gid): return self.dir / "stitched" / f"{gid[:20]}.jpg"

    def page_ok(self, rec: PageRec) -> bool:
        f = self.page(rec.book, rec.page)
        if not f.exists():
            return False
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("src_md5") == (rec.md5 or file_md5(rec.path))
        except Exception:  # noqa: BLE001
            return False


class Ledger:
    """Sổ cái THEO THÁNG (giờ Pacific) có KHOÁ TỆP (flock), nằm ngoài repo, có .bak. Đếm TRƯỚC khi gửi (reserve); hoàn lại (refund) khi chắc chắn không tính phí
    (429/4xx/5xx/chưa gửi); giữ khi mơ hồ (hết giờ, ngắt) và khi HTTP 200 (kể cả lỗi theo ảnh)."""

    def __init__(self, path: Path | None = None, month: str | None = None, readonly: bool = False, mirror: str | None = None):
        self.path = Path(path) if path else LEDGER_PATH
        self.mirror = Path(mirror) if mirror else (Path(LEDGER_MIRROR) if (path is None and LEDGER_MIRROR) else None)   # đường dẫn tường minh (test) → không dùng bản sao
        self.month = month or current_month()
        self.readonly = readonly
        self._lockp = self.path.with_name(self.path.name + ".lock")
        with self._locked():
            self.data = self._load()
            if not self.data.get("legacy_imported") and LEGACY_LEDGER.exists():
                try:
                    lg = json.loads(LEGACY_LEDGER.read_text(encoding="utf-8"))
                    m = time.strftime("%Y-%m", time.gmtime(LEGACY_LEDGER.stat().st_mtime))
                    self.data["months"].setdefault(m, {"calls": 0, "seq": 0, "history": []})["calls"] += int(lg.get("total_api_calls", 0))
                    self.data["legacy_imported"] = True
                    if not readonly:
                        self._save()
                except Exception:  # noqa: BLE001
                    pass

    @contextlib.contextmanager
    def _locked(self):
        if self.readonly:
            yield
            return
        self._lockp.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._lockp, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(Exception):
                fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()

    def _load(self) -> dict:
        bak = self.path.with_name(self.path.name + ".bak")
        for p, tag in ((self.path, ""), (bak, " (.bak)")):
            if p.exists():
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    if tag:
                        print(f"⚠ sổ cái chính hỏng — dùng bản dự phòng{tag}", flush=True)
                    d.setdefault("months", {})
                    return self._merge_mirror(d)
                except Exception:  # noqa: BLE001
                    continue
        if self.path.exists() or bak.exists():
            sys.exit(f"Sổ cái {self.path} và bản dự phòng đều hỏng — KHÔNG tự đặt về 0 (sẽ vượt hạn mức). Kiểm tay rồi chạy lại.")
        return self._merge_mirror({"version": 3, "months": {}, "legacy_imported": False})

    def _merge_mirror(self, d: dict) -> dict:
        """Gộp với bản sao ngoài repo: mỗi tháng lấy số yêu cầu LỚN HƠN (mất một trong hai bản không làm sổ cái về 0)."""
        if not self.mirror or not self.mirror.exists():
            return d
        try:
            m = json.loads(self.mirror.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return d
        for mo, v in m.get("months", {}).items():
            cur = d["months"].get(mo)
            if cur is None or int(v.get("calls", 0)) > int(cur.get("calls", 0)):
                d["months"][mo] = v
        d["legacy_imported"] = bool(d.get("legacy_imported") or m.get("legacy_imported"))
        return d

    def _save(self):
        if self.path.exists():
            with contextlib.suppress(Exception):
                shutil.copyfile(self.path, self.path.with_name(self.path.name + ".bak"))
        write_json(self.path, self.data, fsync=True)
        if self.mirror:
            with contextlib.suppress(Exception):
                write_json(self.mirror, self.data, fsync=True)

    def used(self) -> int:
        with self._locked():
            self.data = self._load() if self.path.exists() else self.data
            return int(self.data["months"].get(self.month, {}).get("calls", 0))

    def reserve(self, cid: str, gid: str, cap: int):
        with self._locked():
            if self.path.exists():
                self.data = self._load()
            m = self.data["months"].setdefault(self.month, {"calls": 0, "seq": 0, "history": []})
            if m["calls"] + 1 > cap:
                return None
            m["calls"] += 1
            m["seq"] = m.get("seq", 0) + 1
            m["history"].append({"id": m["seq"], "ts": int(time.time()), "cfg": cid, "gid": gid[:12], "note": "pending"})
            m["history"] = m["history"][-3000:]
            self._save()
            return (self.month, m["seq"])

    def _entry(self, tok):
        m = self.data["months"].get(tok[0], {})
        return m, next((e for e in reversed(m.get("history", [])) if e.get("id") == tok[1]), None)

    def settle(self, tok, note: str):
        with self._locked():
            self.data = self._load()
            m, e = self._entry(tok)
            if e is not None:
                e["note"] = scrub(note)[:200]
                self._save()

    def refund(self, tok, why: str):
        with self._locked():
            self.data = self._load()
            m, e = self._entry(tok)
            if e is not None and not str(e.get("note", "")).startswith("refunded"):
                m["calls"] = max(0, m["calls"] - 1)
                e["note"] = f"refunded:{scrub(why)[:100]}"
                self._save()


# ───────────────────────── xác thực + gọi mạng ─────────────────────────
class Auth:
    """Khoá dịch vụ (google-auth, làm mới mỗi 40 phút) hoặc khoá API trong biến môi trường. Không in khoá/URL."""

    def __init__(self, key_path: str | None, api_key: str | None):
        self.key_path, self.api_key, self._tok, self._t = key_path, api_key, None, 0.0
        if api_key and not re.fullmatch(r"[A-Za-z0-9_\-]{20,80}", api_key):
            sys.exit("Khoá API sai định dạng (có ký tự lạ/khoảng trắng) — không in giá trị. Dán lại khoá.")

    def headers(self) -> dict:
        if self.api_key:
            return {}
        if self._tok is None or time.time() - self._t > 2400:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(self.key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(Request())
            self._tok, self._t = creds.token, time.time()
        return {"Authorization": f"Bearer {self._tok}"}

    def url(self) -> str:
        return ENDPOINT + (f"?key={self.api_key}" if self.api_key else "")

    def invalidate(self):
        self._tok = None


def need_google_auth():
    try:
        import google.auth  # noqa: F401
        import google.oauth2.service_account  # noqa: F401
        from google.auth.transport.requests import Request  # noqa: F401
    except ImportError:
        sys.exit("Thiếu google-auth/requests trong interpreter này. Dùng đúng interpreter bạn đã chạy kịch bản cũ, hoặc "
                 "`python3 -m venv ~/.venvs/vision && ~/.venvs/vision/bin/pip install google-auth requests pillow` rồi chạy bằng interpreter đó "
                 "(hoặc dùng --api-key-env, không cần thư viện). KHÔNG cài vào .venv của dự án.")


def find_key(arg: str | None) -> str | None:
    if arg:
        if not Path(arg).is_file():
            sys.exit(f"--key trỏ tới tệp không tồn tại: {Path(arg).name}")
        return str(Path(arg).resolve())
    cands = [os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")]
    g = Path.home() / ".config" / "gcloud"
    cands += sorted(str(p) for p in g.glob("vision-ocr*.json")) if g.exists() else []
    cands += sorted(str(p) for p in REPO.glob("vision-ocr-*.json"))
    for c in cands:
        if c and Path(c).is_file():
            return str(Path(c).resolve())
    return None


def key_guard(path: str):
    """Từ chối khoá nằm TRONG repo mà git chưa ignore; cảnh báo quyền tệp."""
    p = Path(path).resolve()
    with contextlib.suppress(Exception):
        if p.stat().st_mode & 0o077:
            print(f"⚠ khoá {p.name} cho nhóm/người khác đọc được — chạy: chmod 600 '{p}'", flush=True)
    try:
        p.relative_to(REPO.resolve())
    except ValueError:
        return
    r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(p)], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"TỪ CHỐI: khoá dịch vụ {p.name} nằm trong repo và git CHƯA ignore (sẽ bị `git add -A` bắt). Thêm vào .gitignore hoặc chuyển ra ngoài repo.")
    print(f"⚠ khoá nằm trong repo (đã được git ignore; nhớ COMMIT .gitignore) — nên chuyển ra ~/.config/gcloud/ : {p.name}", flush=True)


def describe_key(path: str):
    """In project_id + client_email (KHÔNG in private_key) để phát hiện nhầm dự án."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        print(f"khoá dịch vụ: dự án {d.get('project_id')} · {d.get('client_email')}", flush=True)
        if d.get("type") != "service_account":
            sys.exit("Tệp khoá không phải loại service_account.")
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        sys.exit("Không đọc được tệp khoá JSON.")


def make_transport(auth: Auth):
    def transport(body: bytes):
        try:
            hdr = {"Content-Type": "application/json", **auth.headers()}
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            raise PreSendError(f"xác thực lỗi ({type(e).__name__})") from None
        try:
            req = urllib.request.Request(auth.url(), data=body, headers=hdr)
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 401:
                auth.invalidate()
            return e.code, e.read()
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.gaierror, ConnectionRefusedError)):
                raise PreSendError(type(e.reason).__name__) from None
            raise Ambiguous(type(e.reason).__name__) from None
        except (TimeoutError, socket.timeout):
            raise Ambiguous("timeout") from None
        except Exception as e:  # noqa: BLE001
            raise Ambiguous(type(e).__name__) from None
    return transport


# ───────────────────────── xử lý một nhóm ─────────────────────────
def call_with_retries(transport, payload: bytes, ledger: Ledger, cid: str, gid: str, cap: int, sleep=time.sleep):
    """→ (resp_json | None, ghi_chú). Đếm TRƯỚC khi gửi; hoàn lại khi chắc chắn không tính phí; giữ khi mơ hồ/HTTP 200."""
    ambiguous = 0
    for attempt in range(6):
        tok = ledger.reserve(cid, gid, cap)
        if tok is None:
            raise Stop(f"chạm trần {cap}/tháng")
        try:
            status, body = transport(payload)
        except PreSendError as e:
            ledger.refund(tok, "chưa gửi")
            return None, f"lỗi trước khi gửi: {scrub(e)}"
        except Ambiguous as e:
            ledger.settle(tok, f"không rõ ({e}) — giữ tính phí")
            ambiguous += 1
            if ambiguous >= 3:
                return None, "mất kết nối nhiều lần"
            sleep(2 ** attempt)
            continue
        except BaseException as e:   # noqa: BLE001  (KeyboardInterrupt, SystemExit…)
            ledger.settle(tok, f"ngắt ({type(e).__name__}) — giữ tính phí")
            raise
        if status == 200:
            ledger.settle(tok, "ok")
            try:
                return json.loads(body.decode("utf-8")), "ok"
            except Exception:  # noqa: BLE001
                return None, "phản hồi không phải JSON"
        ledger.refund(tok, f"HTTP {status}")
        if status in (429, 500, 502, 503, 504, 401):
            sleep(min(60, 2 ** (attempt + 1)))
            continue
        try:
            msg = json.loads(body.decode("utf-8")).get("error", {}).get("message", "")[:200]
        except Exception:  # noqa: BLE001
            msg = body[:200].decode("utf-8", "replace")
        return None, f"HTTP {status}: {scrub(msg)}"
    return None, "hết số lần thử"


def write_pages(store: Store, cfg_id: str, gid: str, place: list, per: dict, meta: dict):
    for i, p in enumerate(place):
        r = p["rec"]
        write_json(store.page(r.book, r.page), dict(
            book=r.book, page=r.page, cfg=cfg_id, gid=gid[:20], src_md5=r.md5, orig_w=r.w, orig_h=r.h, sx=round(p["sx"], 6), sy=round(p["sy"], 6),
            cols=["ch", "conf", "x0", "y0", "x1", "y1", "ang"], sym=per[i], **meta))


def _place_from_layout(lay: dict, group: list[PageRec]):
    by = {(r.book, r.page): r for r in group}
    return [dict(rec=by[(p["book"], p["page"])], x0=p["x0"], y0=p["y0"], w=p["w"], h=p["h"], sx=p["sx"], sy=p["sy"]) for p in lay["placements"]]


def process_group(group, cfg: Cfg, store: Store, ledger: Ledger, transport, cap: int, scales: dict, keep_stitched=False, sleep=time.sleep):
    """→ (trạng thái, số ký hiệu, cờ_bất_thường). trạng thái ∈ cached | raw | api | fail:<lý do>. Có thể ném Stop khi chạm trần."""
    for r in group:
        r.md5 = r.md5 or file_md5(r.path)
    if all(store.page_ok(r) for r in group):
        return "cached", 0, ""
    gid = group_id(group, cfg, scales)
    rawp, layp = store.raw(gid), store.layout(gid)
    state = "raw"
    if rawp.exists():
        resp = json.loads(gzip.decompress(rawp.read_bytes()).decode("utf-8"))
        if layp.exists():                                  # phát lại bằng bố cục ĐÃ LƯU lúc gửi (không phụ thuộc labels.csv hiện tại)
            lay = json.loads(layp.read_text(encoding="utf-8"))
            place, canvas_wh, g = _place_from_layout(lay, group), tuple(lay["canvas"]), lay.get("g", 1.0)
        else:
            canvas, place, g = stitch(group, cfg, scales)
            canvas_wh = canvas.size
            write_json(layp, dict(gid=gid, cfg=cfg.params(), canvas=list(canvas_wh), g=g, rebuilt=True,
                                  placements=[dict(book=p["rec"].book, page=p["rec"].page, x0=p["x0"], y0=p["y0"], w=p["w"], h=p["h"], sx=p["sx"], sy=p["sy"],
                                                   md5=p["rec"].md5) for p in place]))
    else:
        canvas, place, g = stitch(group, cfg, scales)
        canvas_wh = canvas.size
        try:
            jpeg = encode_jpeg(canvas, cfg)
        except TooBig as e:
            return f"fail:{e}", 0, ""
        resp, note = call_with_retries(transport, build_payload(jpeg, cfg.hints), ledger, store.dir.name, gid, cap, sleep)
        if resp is None:
            return f"fail:{note}", 0, ""
        r0 = (resp.get("responses") or [{}])[0]
        if "error" in r0:
            return f"fail:lỗi ảnh {r0['error'].get('code')}: {scrub(str(r0['error'].get('message')))[:120]} (đã tính 1 yêu cầu, KHÔNG lưu cache)", 0, ""
        atomic_write(rawp, gzip.compress(json.dumps(resp, ensure_ascii=False).encode("utf-8"), 6))
        if keep_stitched:
            atomic_write(store.stitched(gid), jpeg)
        state = "api"
        write_json(layp, dict(gid=gid, cfg=cfg.params(), canvas=list(canvas_wh), g=g,
                              placements=[dict(book=p["rec"].book, page=p["rec"].page, x0=p["x0"], y0=p["y0"], w=p["w"], h=p["h"], sx=p["sx"], sy=p["sy"],
                                               md5=p["rec"].md5) for p in place]))
    r0 = (resp.get("responses") or [{}])[0]
    syms = parse_symbols(r0)
    vwh = vision_dims(r0)
    per, dropped, info = demux(syms, place, canvas_wh, vwh)
    nov = no_vertex_count(r0)
    anomaly = ""
    if info["frame"] != "canvas":
        anomaly = f"Vision báo {vwh[0]}×{vwh[1]} ≠ ảnh gửi {canvas_wh[0]}×{canvas_wh[1]} ({info['frame']})"
    if nov:
        anomaly = (anomaly + "; " if anomaly else "") + f"{nov} ký hiệu không có đỉnh (bị bỏ)"
    meta = dict(canvas_w=canvas_wh[0], canvas_h=canvas_wh[1], vision_w=vwh[0], vision_h=vwh[1], dropped_in_gap=dropped, frame=info["frame"], no_vertices=nov,
                scale_g=round(g, 4), hints=list(cfg.hints))
    write_pages(store, store.dir.name, gid, place, per, meta)
    return state, len(syms), anomaly


def run_groups(groups, cfg: Cfg, transport, ledger: Ledger, cap: int, max_requests: int = 0, keep_stitched=False, sleep=time.sleep, log=print, every=10,
               max_consec_fail: int = 3, stop_on_anomaly: bool = True, store: Store | None = None):
    store = store or Store(cfg.cid())
    scales = get_scales(cfg, sorted({r.bdir for g in groups for r in g}), store)
    write_json(store.dir / "manifest.json", dict(cfg=cfg.params(), cid=cfg.cid(), created=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())))
    n_api = n_cached = n_raw = nsym = consec = 0
    fails, anomalies = [], []
    t0 = time.time()
    stopped = ""
    for gi, g in enumerate(groups, 1):
        if STOP_FLAG["v"]:
            stopped = "nhận tín hiệu dừng (Ctrl-C/SIGTERM) — đã xong nhóm đang chạy"
            break
        if max_requests and n_api >= max_requests:
            stopped = f"đạt --max-requests {max_requests}"
            break
        try:
            st, ns, anomaly = process_group(g, cfg, store, ledger, transport, cap, scales, keep_stitched, sleep)
        except Stop as e:
            stopped = str(e)
            break
        except Exception as e:  # noqa: BLE001   (tệp raw hỏng, ...) — không sập cả lượt
            st, ns, anomaly = f"fail:{type(e).__name__}: {scrub(e)[:120]}", 0, ""
        if st == "api":
            n_api += 1
            consec = 0
        elif st == "cached":
            n_cached += 1
        elif st == "raw":
            n_raw += 1
            consec = 0
        else:
            fails.append((f"{g[0].book}:{g[0].page}…", st[5:]))
            log(f"  ✗ {g[0].book} {g[0].page}: {st[5:]}")
            consec += 1
            if consec >= max_consec_fail:
                stopped = f"{consec} nhóm lỗi liên tiếp — dừng để không đốt hạn mức (xem lỗi ở trên)"
                break
        nsym += ns
        if anomaly:
            anomalies.append((f"{g[0].book}:{g[0].page}…", anomaly))
            log(f"  ⚠ {g[0].book} {g[0].page}: {anomaly}")
            if stop_on_anomaly and "Vision báo" in anomaly:
                stopped = "ảnh đầu tiên có kích thước/toạ độ bất thường — dừng để bạn xem (--no-stop-on-anomaly để bỏ qua)"
                break
        if n_api and n_api % every == 0 and st == "api":
            log(f"  [{gi}/{len(groups)}] {n_api} yêu cầu mới · {nsym:,} ký hiệu · sổ cái {ledger.used()}/{cap} · {(time.time() - t0) / 60:.1f} phút")
    return dict(cfg=store.dir.name, groups=len(groups), api=n_api, cached=n_cached, from_raw=n_raw, fails=fails, anomalies=anomalies, symbols=nsym, stopped=stopped,
                ledger=ledger.used())


# ───────────────────────── nhập sổ cũ ─────────────────────────
def import_legacy(log=print):
    """Đổi 5 phản hồi cache của stitched_vision_audit.py sang định dạng mới (cfg 'legacy-n2')."""
    lg = json.loads(LEGACY_LEDGER.read_text(encoding="utf-8"))
    by_pair = {}
    for h in lg["history"]:
        d = h.get("description", "")
        if d.startswith("stitched_pair_"):
            by_pair[int(d.split("_")[2].split(".")[0])] = h["hash"]
    bd = dict(BOOKS)
    cid = "legacy-n2-row-zhHant+vi-H1800"
    store = Store(cid)
    write_json(store.dir / "manifest.json", dict(cid=cid, note="nhập từ cache stitched_vision_audit.py (cao 1800, ngăn 30, languageHints zh-Hant+vi)"))
    TH, SEP = 1800, 30
    done = 0
    for pair in range(1, 6):
        f = LEGACY_DIR / f"{by_pair[pair]}.json"
        if not f.exists():
            continue
        resp = json.loads(f.read_text(encoding="utf-8"))["responses"][0]
        items = LEGACY10[(pair - 1) * 2:(pair - 1) * 2 + 2]
        place, x = [], 0
        for k, page in items:
            path = REPO / "prepared" / bd[k] / "pages_denoised" / f"{page}.png"
            with Image.open(path) as im:
                s = TH / im.height
                w = int(im.width * s)
                rec = PageRec(k, bd[k], page, str(path), im.width, im.height, file_md5(str(path)))
            place.append(dict(rec=rec, x0=x, y0=0, w=w, h=TH, sx=w / rec.w, sy=TH / rec.h))
            x += w + SEP
        vwh = vision_dims(resp)
        per, dropped, info = demux(parse_symbols(resp), place, (x - SEP, TH), vwh)
        meta = dict(canvas_w=x - SEP, canvas_h=TH, vision_w=vwh[0], vision_h=vwh[1], dropped_in_gap=dropped, frame=info["frame"], no_vertices=no_vertex_count(resp),
                    scale_g=1.0, hints=["zh-Hant", "vi"])
        write_pages(store, cid, by_pair[pair], place, per, meta)
        done += 1
    log(f"đã nhập {done}/5 ảnh ghép cũ → {store.dir}")
    return cid


# ───────────────────────── giao diện dòng lệnh ─────────────────────────
def cfg_from_args(a) -> Cfg:
    hints = () if a.hints.lower() in ("none", "auto", "") else tuple(x.strip() for x in a.hints.split(",") if x.strip())
    return Cfg(n=a.n, layout=a.layout, hints=hints, cell_h=a.cell_h, cap_mp=a.cap_mp)


def fmt_plan(cfg: Cfg, pages: dict, groups: list, ledger: Ledger, cap: int, mix: bool, store: Store, render: bool = True):
    scales = get_scales(cfg, sorted({r.bdir for g in groups for r in g}), store, write=False)
    print(f"cấu hình {cfg.cid()} | {cfg.n} trang/ảnh ({'hàng' if cfg.layout == 'row' else 'lưới'}) | cao ô mục tiêu {cfg.cell_h:g}px | trần {cfg.cap_mp:g} MP | gợi ý {list(cfg.hints) or 'tự nhận'}")
    print(f"{'sách':6s} {'trang':>6s} {'nhóm':>5s} {'đã có':>6s} {'scale':>6s}  điểm ảnh ghép (MP trung vị / lớn nhất; JPEG của nhóm lớn nhất)")
    by_book: dict = {}
    for g in groups:
        by_book.setdefault(g[0].book if not mix else "(trộn)", []).append(g)
    tot_g = tot_done = 0
    for k, gs in by_book.items():
        pg = sum(len(g) for g in gs)
        done = sum(1 for g in gs if all(store.page_ok(r) for r in g))
        dims = [canvas_dims(g, cfg, scales) for g in gs]
        mps = [w * h / 1e6 for w, h, _ in dims]
        big = max(range(len(gs)), key=lambda i: mps[i])
        sz = f"{statistics.median(mps):.1f} / {mps[big]:.1f} MP ({dims[big][0]}×{dims[big][1]}, hạ ×{dims[big][2]:.2f})"
        if render:
            try:
                cv, _, _ = stitch(gs[big], cfg, scales)
                sz += f", JPEG {len(encode_jpeg(cv, cfg)) / 1e6:.2f} MB"
            except TooBig as e:
                sz += f" QUÁ LỚN: {e}"
        print(f"{k:6s} {pg:6d} {len(gs):5d} {done:6d} {scales[gs[0][0].bdir]:6.2f}  {sz}")
        tot_g += len(gs)
        tot_done += done
    todo = tot_g - tot_done
    used = ledger.used()
    print(f"TỔNG {tot_g} nhóm = {tot_g} yêu cầu ({tot_done} đã có → cần gọi ≤ {todo}) | sổ cái {ledger.month}: {used}/{cap} → sau khi chạy ≤ {used + todo}/{cap}"
          f" {'✓ vừa trần' if used + todo <= cap else '✗ VƯỢT TRẦN'}")
    print("lưu ý: sổ cái chỉ thấy việc dùng của repo này; hạn mức miễn phí tính theo TÀI KHOẢN THANH TOÁN, tháng giờ Pacific — xem Console → Vision API → Metrics / Billing → Reports "
          "nếu có dùng nơi khác. Giá: 1.000 đơn vị/tháng miễn phí, 1,50 USD/1.000 sau đó (1 ảnh = 1 đơn vị).")


def install_signal_handlers():
    def h(signum, frame):
        STOP_FLAG["v"] = True
        print("\n⚠ nhận tín hiệu dừng — hoàn tất yêu cầu đang bay rồi thoát", flush=True)
    signal.signal(signal.SIGINT, h)
    signal.signal(signal.SIGTERM, h)


@contextlib.contextmanager
def run_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    fh = open(ROOT / "run.lock", "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        sys.exit("Đang có một tiến trình run/pilot khác (vision_cache/run.lock). Chỉ chạy MỘT tại một thời điểm.")
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["plan", "run", "pilot", "status", "import-legacy"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--layout", choices=["row", "grid"], default="grid")
    ap.add_argument("--hints", default="zh-Hant", help="ví dụ zh-Hant | zh-Hant,vi | none")
    ap.add_argument("--cell-h", type=float, default=80.0)
    ap.add_argument("--cap-mp", type=float, default=5.0, help="trần điểm ảnh (triệu) mỗi ảnh ghép")
    ap.add_argument("--books", default="", help="khoá sách, cách nhau dấu phẩy (mặc định: tất cả, sách in trước)")
    ap.add_argument("--pages", choices=["all", "legacy10"], default="all")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1, help="pilot: 1 = cách ghép/kích thước (20 yêu cầu), 2 = gợi ý ngôn ngữ (6 yêu cầu)")
    ap.add_argument("--cap", type=int, default=950)
    ap.add_argument("--allow-over-free", action="store_true", help="cho phép --cap > 990 (có thể phát sinh phí)")
    ap.add_argument("--max-requests", type=int, default=None, help="tối đa yêu cầu MỚI mỗi lần chạy (mặc định 25 cho run; 0 = không giới hạn)")
    ap.add_argument("--no-stop-on-anomaly", action="store_true")
    ap.add_argument("--yes", action="store_true", help="bắt buộc để gọi mạng (run/pilot)")
    ap.add_argument("--key", default=None)
    ap.add_argument("--api-key-env", default=None, help="tên biến môi trường chứa khoá API (thay cho khoá dịch vụ)")
    ap.add_argument("--keep-stitched", action="store_true")
    ap.add_argument("--no-render", action="store_true", help="plan: không dựng ảnh mẫu")
    a = ap.parse_args(argv)
    cap = a.cap if a.allow_over_free else min(a.cap, HARD_CAP)
    books = [b for b in a.books.split(",") if b] or None
    readonly = a.cmd in ("plan", "status")
    ledger = Ledger(readonly=readonly)

    if a.cmd == "status":
        print(f"sổ cái {ledger.month}: {ledger.used()}/{cap}; các tháng: { {m: v['calls'] for m, v in ledger.data['months'].items()} } ({ledger.path})")
        for d in sorted(ROOT.glob("**/manifest.json")):
            n = len(list((d.parent / "pages").glob("*/*.json")))
            print(f"  {d.parent.relative_to(ROOT)}: {n} trang đã có")
        return 0
    if a.cmd == "import-legacy":
        import_legacy()
        return 0

    if a.cmd == "pilot":
        pages = select_pages(None, "legacy10")
        hz = ("zh-Hant", "vi")
        if a.stage == 1:
            plans = [(Cfg(n=1, layout="row", hints=hz), False), (Cfg(n=4, layout="grid", hints=hz, cap_mp=1e9), True), (Cfg(n=4, layout="grid", hints=hz, cap_mp=a.cap_mp), True),
                     (Cfg(n=3, layout="grid", hints=hz, cap_mp=a.cap_mp), True)]
        else:
            plans = [(Cfg(n=4, layout="grid", hints=("zh-Hant",), cap_mp=a.cap_mp, cell_h=a.cell_h), True), (Cfg(n=4, layout="grid", hints=(), cap_mp=a.cap_mp, cell_h=a.cell_h), True)]
        stores = {id(c): Store("_pilot/" + c.cid()) for c, _ in plans}
    else:
        pages = select_pages(books, a.pages)
        plans = [(cfg_from_args(a), a.pages == "legacy10")]
        stores = {id(c): Store(c.cid()) for c, _ in plans}

    if a.cmd == "plan":
        for cfg, mix in plans:
            fmt_plan(cfg, pages, plan_groups(pages, cfg.n, mix), ledger, cap, mix, stores[id(cfg)], render=not a.no_render)
        return 0

    # run / pilot — GỌI MẠNG
    if not a.yes:
        for cfg, mix in plans:
            fmt_plan(cfg, pages, plan_groups(pages, cfg.n, mix), ledger, cap, mix, stores[id(cfg)], render=False)
        print(f"\n(xem trước, KHÔNG gọi mạng) — thêm --yes để thực sự gọi Google Vision.")
        return 0
    key = find_key(a.key) if not a.api_key_env else None
    api_key = os.environ.get(a.api_key_env, "") if a.api_key_env else None
    if not key and not api_key:
        sys.exit("Không tìm thấy khoá: dùng --key <json> hoặc GOOGLE_APPLICATION_CREDENTIALS hoặc --api-key-env <TÊN_BIẾN>.")
    if key:
        need_google_auth()
        key_guard(key)
        describe_key(key)
    auth = Auth(key, api_key)
    transport = make_transport(auth)
    max_req = a.max_requests if a.max_requests is not None else (25 if a.cmd == "run" else 0)
    install_signal_handlers()
    with run_lock():
        for cfg, mix in plans:
            store = stores[id(cfg)]
            groups = plan_groups(pages, cfg.n, mix)
            todo = sum(1 for g in groups if not all(store.page_ok(r) for r in g))
            print(f"▶ {store.dir.relative_to(ROOT)}: {len(groups)} nhóm, cần gọi ≤ {todo}; sổ cái {ledger.used()}/{cap}"
                  f"{'; tối đa ' + str(max_req) + ' yêu cầu lần này (--max-requests 0 để chạy hết)' if max_req else ''}", flush=True)
            res = run_groups(groups, cfg, transport, ledger, cap, max_req, a.keep_stitched or a.cmd == "pilot", log=print,
                             stop_on_anomaly=(a.cmd == "run" and not a.no_stop_on_anomaly), store=store)
            print(f"  xong: {res['api']} yêu cầu mới · {res['cached']} đã có · {res['from_raw']} dựng lại từ phản hồi thô · {res['symbols']:,} ký hiệu · lỗi {len(res['fails'])}"
                  f" · cờ bất thường {len(res['anomalies'])}{' · DỪNG: ' + res['stopped'] if res['stopped'] else ''} · sổ cái {res['ledger']}/{cap}", flush=True)
            if res["stopped"] and "đạt --max-requests" not in res["stopped"]:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
