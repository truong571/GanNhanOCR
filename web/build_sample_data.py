#!/usr/bin/env python3
"""web/build_sample_data.py — sinh lại web/sample_data.json (+ sample_data.js) cho chế độ MỞ index.html TRỰC TIẾP.

Dùng CHÍNH các hàm API của web/server.py trên dữ liệu thật (chỉ đọc), nên số liệu trong bản mẫu trùng với máy chủ:
  stats · books · pipeline_flow · benchmarks (bảng tầng nhãn) · 1 trang mẫu / bộ (10 bộ) · thư viện ký tự mẫu
  (rải đều theo tầng nhãn).

  .venv/bin/python web/build_sample_data.py                    # -> web/sample_data.json + web/sample_data.js
  .venv/bin/python web/build_sample_data.py --max-mb 3 --page-width 640
  .venv/bin/python web/build_sample_data.py --no-images        # không nhúng ảnh (chỉ đường dẫn tương đối ../dataset/…)
  .venv/bin/python web/build_sample_data.py --root <gốc dữ liệu> --out /tmp/x.json --force

Ảnh: ảnh trang mẫu thu nhỏ (JPEG) + crop của các thẻ thư viện (PNG ≤ 72 px) được NHÚNG (data URI) để thư mục web/ chép
sang máy khác vẫn xem được; crop của các ô còn lại trên trang mẫu giữ đường dẫn tương đối (chỉ hiện khi mở trong repo).
Cần Pillow cho phần nhúng (có trong .venv); thiếu Pillow -> tự chuyển sang --no-images.

sample_data.js = cùng nội dung, gán `window.GANNHANOCR_SAMPLE` — vì Chrome chặn fetch() tệp cục bộ khi mở file://.

Chốt an toàn: KHÔNG chạy khi lượt dựng lại (scripts/clean_rebuild_all.sh / run_pipeline.sh) còn đang chạy — dữ liệu đang
bị xoá/sinh lại (bỏ chốt bằng --force). Chỉ ghi đúng 2 tệp đầu ra (ghi tạm rồi đổi tên).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB_DIR))
import server as S  # noqa: E402

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None


def rebuild_running() -> list[str]:
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    me = str(os.getpid())
    return [ln.strip() for ln in out.splitlines()
            if ("clean_rebuild_all.sh" in ln or "run_pipeline.sh" in ln or "run_pipeline.frozen.sh" in ln)
            and not ln.strip().startswith(me) and "grep" not in ln]


def data_uri_page(p: Path, width: int, quality: int) -> str | None:
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.width > width:
                im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def data_uri_crop(p: Path, side: int = 72) -> str | None:
    try:
        with Image.open(p) as im:
            im = im.convert("L") if im.mode not in ("L", "LA") else im
            im.thumbnail((side, side), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


class Builder:
    def __init__(self, st: S.Store, a):
        self.st, self.a = st, a
        self.embed = bool(Image) and not a.no_images
        self.n_embed_crop = self.n_embed_page = 0

    def rel_url(self, url: str | None) -> str | None:
        """URL máy chủ -> đường dẫn tương đối từ web/ (mở trong repo)."""
        if not url or url.startswith("data:"):
            return url
        f = S.file_for_url(self.st, url)
        return os.path.relpath(f, WEB_DIR).replace(os.sep, "/") if f else None

    def crop_uri(self, url: str | None) -> str | None:
        if not url:
            return None
        if self.embed:
            f = S.file_for_url(self.st, url)
            if f:
                d = data_uri_crop(f)
                if d:
                    self.n_embed_crop += 1
                    return d
        return self.rel_url(url)

    def pick_gallery(self, chars: list[dict], k: int) -> list[dict]:
        """Chọn k ô có crop, đa dạng tầng nhãn: vòng tròn qua các tầng (GOLD, SYLLABLE, REVIEW, QUARANTINE, khác)."""
        groups: OrderedDict = OrderedDict((g, []) for g in S.TIER_ORDER + ["other"])
        for c in chars:
            if not c.get("crop_url"):
                continue
            groups[c["tier"] if c.get("tier") in groups else "other"].append(c)
        out = []
        while len(out) < k and any(groups.values()):
            for g in list(groups):
                if groups[g] and len(out) < k:
                    lst = groups[g]
                    out.append(lst.pop(len(lst) // 2 if len(out) % 2 else 0))
        return out

    def build(self, page_width: int, quality: int, max_chars: int) -> dict:
        st = self.st
        self.n_embed_crop = self.n_embed_page = 0
        stats = S.api_stats(st)
        stats["data_root"] = "(bản mẫu — sinh từ gốc dự án)"
        books = S.api_books(st)
        flow = S.api_pipeline_flow(st)
        bm = S.api_benchmarks(st)
        bm["data_root"] = stats["data_root"]
        pages, gallery = {}, []
        for b in books:
            if not b.get("default_page") or not b.get("total_chars"):
                b["available_pages"], b["sample_pages"] = [], []
                continue
            pg = b["default_page"]
            p, code = S.api_page(st, {"book": [b["id"]], "page": [pg]})
            if code != 200:
                continue
            chars = p["characters"][:max_chars]
            p["characters"] = chars
            p["char_count"] = len(chars)
            # ảnh trang
            if p.get("scan_url"):
                f = S.file_for_url(st, p["scan_url"])
                uri = data_uri_page(f, page_width, quality) if (self.embed and f and page_width > 0) else None
                if uri:
                    self.n_embed_page += 1
                p["scan_url"] = uri or self.rel_url(p["scan_url"])
            # thư viện ký tự mẫu: lấy từ CHÍNH trang mẫu (bấm thẻ -> mở đúng trang)
            picks = self.pick_gallery(chars, self.a.gallery_per_book)
            picked_ids = {id(c) for c in picks}
            for c in chars:
                c["crop_url"] = self.crop_uri(c.get("crop_url")) if id(c) in picked_ids else self.rel_url(c.get("crop_url"))
            for c in picks:
                g = dict(c)
                g.pop("rect", None)
                g.update(book=b["id"], book_title=b["title"])
                gallery.append(g)
            pages[b["id"]] = p
            b["available_pages"], b["sample_pages"] = [pg], [pg]
        return OrderedDict(
            format="gannhanocr-web-v2", generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            generated_by="web/build_sample_data.py",
            note="Bản mẫu cho chế độ mở index.html trực tiếp: 1 trang / bộ; số liệu = API của web/server.py lúc sinh.",
            images=dict(embedded_pages=self.n_embed_page, embedded_crops=self.n_embed_crop, page_width=page_width),
            stats=stats, books=books, pipeline_flow=flow, benchmarks=bm, pages=pages, gallery=gallery)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="gốc dữ liệu (mặc định: $GANNHANOCR_ROOT hoặc thư mục cha của web/)")
    ap.add_argument("--out", default=str(WEB_DIR / "sample_data.json"))
    ap.add_argument("--max-mb", type=float, default=5.0, help="trần dung lượng sample_data.json (MB)")
    ap.add_argument("--page-width", type=int, default=720, help="bề rộng ảnh trang nhúng (px)")
    ap.add_argument("--quality", type=int, default=62, help="chất lượng JPEG ảnh trang nhúng")
    ap.add_argument("--max-chars", type=int, default=400, help="số ô tối đa mỗi trang mẫu")
    ap.add_argument("--gallery-per-book", type=int, default=10)
    ap.add_argument("--no-images", action="store_true", help="không nhúng ảnh")
    ap.add_argument("--no-js", action="store_true", help="không ghi sample_data.js")
    ap.add_argument("--force", action="store_true", help="bỏ chốt 'lượt dựng lại đang chạy'")
    a = ap.parse_args(argv)

    running = rebuild_running()
    if running and not a.force:
        print("[build_sample_data] DỪNG: lượt dựng lại dữ liệu đang chạy — dataset/ đang bị xoá/sinh lại:")
        for ln in running[:5]:
            print("   ", ln[:160])
        print("  Chạy lại lệnh này khi lượt dựng lại đã xong (hoặc --force nếu chắc chắn).")
        return 2
    if not Image and not a.no_images:
        print("[build_sample_data] Không có Pillow -> không nhúng ảnh (--no-images).")
    root = Path(a.root).expanduser().resolve() if a.root else S.default_root()
    st = S.load_store(root)
    B = Builder(st, a)
    width, quality, max_chars = a.page_width, a.quality, a.max_chars
    limit = int(a.max_mb * 1e6)
    for attempt in range(8):
        data = B.build(width, quality, max_chars)
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        size = len(body.encode("utf-8"))
        if size <= limit:
            break
        print(f"[build_sample_data] {size / 1e6:.2f} MB > {a.max_mb} MB -> thu nhỏ (lần {attempt + 1})")
        if width > 360 and B.embed:
            width, quality = int(width * 0.75), max(40, quality - 8)
        elif B.embed:
            B.embed = False
        else:
            max_chars = max(60, int(max_chars * 0.6))
    else:
        print(f"[build_sample_data] LỖI: không đưa được dưới {a.max_mb} MB ({size / 1e6:.2f} MB).")
        return 1
    out = Path(a.out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(out)
    js_out = None
    if not a.no_js:
        js_out = out.with_suffix(".js")
        tmpj = js_out.with_name(js_out.name + ".tmp")
        tmpj.write_text("/* Sinh bởi web/build_sample_data.py — bản bọc của sample_data.json cho chế độ file:// */\n"
                        "window.GANNHANOCR_SAMPLE = " + body + ";\n", encoding="utf-8")
        tmpj.replace(js_out)
    im = data["stats"]["impact_metrics"]
    print(f"[build_sample_data] -> {out} ({size / 1e6:.2f} MB){' + ' + str(js_out) if js_out else ''}")
    print(f"  {len(data['pages'])} trang mẫu · {len(data['gallery'])} thẻ thư viện · ảnh nhúng: "
          f"{data['images']['embedded_pages']} trang (rộng {data['images']['page_width']} px), {data['images']['embedded_crops']} crop")
    print(f"  tổng {im['total_characters']:,} ký tự · " + " · ".join(
        f"{t} {im['tier_totals'].get(t, 0):,}" for t in im["tier_columns"])
          + (f" · {im['invariants_text']}" if im.get("invariants_text") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
