"""NGỮ LIỆU TỔNG HỢP CÓ ĐÁP ÁN 100% — trang Nôm 9 cột dựng từ kho glyph FontDiffusion.

VÌ SAO
------
Đề tài có 0 phán quyết người, nên không có hộp ký tự nào được người xác nhận. Nhưng nếu
ta TỰ DỰNG trang thì biết chính xác từng hộp ở đâu và là chữ gì — ground truth miễn phí,
sinh được hàng nghìn trang. Dùng để chấm điểm tách cột / tách ký tự / căn chỉnh ở vòng B
của chương trình T4.

GIỚI HẠN PHẢI KHAI THẲNG
------------------------
Ngữ liệu tổng hợp kiểm **HÀNH VI THUẬT TOÁN**, KHÔNG kiểm độ chính xác trên ảnh khắc gỗ
thật. Chính khoảng cách miền glyph-phẳng ↔ khắc-gỗ là thứ đã làm hỏng tín hiệu S3
(docs/NGHIEN_CUU_UNIHAN_KDEFINITION §5, docs/PHAN_HOI_GOP_Y §2.4). Nên:

    ĐƯỢC dùng để trả lời "cấu hình A hay B chịu nhiễu tốt hơn".
    KHÔNG được dùng để công bố bất kỳ con số chất lượng nào của bộ dữ liệu thật.

NHIỄU HIỆU CHUẨN THEO ẢNH THẬT
------------------------------
Tham số nhiễu mặc định lấy từ phân bố ĐO ĐƯỢC trên 56.776 crop thật (2026-08-23):
ink_pct p50=0,175 (p95 0,253) · stray_ink p50=0 (p95 0,136) · aspect p50=0,794.
`calibrate()` đọc lại các số này từ một dòng `lab/results.csv` để không phải ghim cứng.
"""
from __future__ import annotations

import glob
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FD = REPO / "gannhanocr-fd"

# Bố cục đo trên trang thật: 9 cột dọc, chữ xếp từ trên xuống, phải sang trái.
DEFAULT_LAYOUT = {
    "cols": 9, "cell": 96, "pad": 14, "margin": 40,
    "rows_min": 14, "rows_max": 24,
}
DEFAULT_NOISE = {
    "break_p": 0.06,      # đứt nét: xoá ngẫu nhiên các mảng nhỏ
    "blot_p": 0.05,       # nhoè mực: bồi thêm mảng
    "grain": 0.10,        # vân gỗ: sọc dọc mờ
    "jitter": 3,          # lệch vị trí ô (px)
    "scale_jit": 0.06,    # co giãn từng chữ
}


def glyph_index(limit: int = 0) -> dict[str, str]:
    """{ký tự: đường dẫn glyph}. `limit>0` để chạy test nhanh."""
    out: dict[str, str] = {}
    for p in sorted(glob.glob(str(FD / "*" / "U+*.png"))):
        name = Path(p).stem                      # 'U+4E8C'
        try:
            ch = chr(int(name[2:], 16))
        except ValueError:
            continue
        try:
            with open(p, "rb") as f:
                if f.read(8) != b"\x89PNG\r\n\x1a\n":
                    continue
        except OSError:
            continue
        out[ch] = p
        if limit and len(out) >= limit:
            break
    return out


def calibrate(results_csv: Path | str = REPO / "lab" / "results.csv",
              name: str = "baseline") -> dict:
    """Lấy phân bố mực THẬT từ một dòng results.csv để hiệu chuẩn nhiễu."""
    import csv
    p = Path(results_csv)
    if not p.exists():
        return {}
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("name") == name]
    if not rows:
        return {}
    r = rows[-1]
    out = {}
    for k in ("ink_pct_p50", "ink_pct_p95", "stray_ink_p95", "aspect_p50"):
        try:
            out[k] = float(r[k])
        except (KeyError, TypeError, ValueError):
            pass
    return out


def make_page(chars_by_col: list[list[str]], glyphs: dict[str, str],
              rng: random.Random, layout: dict | None = None,
              noise: dict | None = None):
    """Dựng 1 trang tổng hợp. Trả (ảnh xám numpy, list hộp đáp án).

    Mỗi hộp: {'col': i, 'idx': j, 'char': c, 'bbox': (x1,y1,x2,y2)} — TOẠ ĐỘ ĐÚNG,
    không phải ước lượng.
    """
    import numpy as np
    from PIL import Image

    L = {**DEFAULT_LAYOUT, **(layout or {})}
    N = {**DEFAULT_NOISE, **(noise or {})}
    cell, pad, m = L["cell"], L["pad"], L["margin"]
    ncol = len(chars_by_col)
    nrow = max(len(c) for c in chars_by_col)
    W = m * 2 + ncol * (cell + pad)
    H = m * 2 + nrow * (cell + pad)
    page = np.full((H, W), 255, dtype=np.uint8)
    boxes = []

    for i, col in enumerate(chars_by_col):
        # cột đọc từ PHẢI sang TRÁI như bản khắc
        cx = W - m - (i + 1) * (cell + pad)
        for j, ch in enumerate(col):
            gp = glyphs.get(ch)
            if not gp:
                continue
            with Image.open(gp) as im:
                g = im.convert("L")
                s = 1.0 + rng.uniform(-N["scale_jit"], N["scale_jit"])
                sz = max(8, int(cell * s))
                g = g.resize((sz, sz))
                a = np.asarray(g)
            x = cx + rng.randint(-N["jitter"], N["jitter"])
            y = m + j * (cell + pad) + rng.randint(-N["jitter"], N["jitter"])
            x, y = max(0, min(x, W - sz)), max(0, min(y, H - sz))
            region = page[y:y + sz, x:x + sz]
            page[y:y + sz, x:x + sz] = np.minimum(region, a[:region.shape[0], :region.shape[1]])
            ink = np.argwhere(a < 128)
            if ink.size:
                y1, x1 = ink.min(0); y2, x2 = ink.max(0)
                boxes.append({"col": i, "idx": j, "char": ch,
                              "bbox": (int(x + x1), int(y + y1),
                                       int(x + x2) + 1, int(y + y2) + 1)})

    page = _degrade(page, rng, N)
    return page, boxes


def _degrade(page, rng: random.Random, N: dict):
    """Đứt nét / nhoè mực / vân gỗ — mô phỏng bản khắc, tham số hiệu chuẩn theo ảnh thật."""
    import numpy as np
    H, W = page.shape
    r = np.random.default_rng(rng.getrandbits(32))
    ink = page < 128
    if N["break_p"] > 0:                       # đứt nét: xoá bớt pixel mực
        page[ink & (r.random((H, W)) < N["break_p"])] = 255
    if N["blot_p"] > 0:                        # nhoè: bồi mực quanh nét
        from scipy import ndimage
        grow = ndimage.binary_dilation(page < 128, iterations=1)
        page[grow & (r.random((H, W)) < N["blot_p"])] = 0
    if N["grain"] > 0:                         # vân gỗ: sọc dọc mờ
        stripes = (r.random((1, W)) < N["grain"]).repeat(H, axis=0)
        page[stripes] = np.minimum(page[stripes], 235)
    return page


def make_corpus(n_pages: int, qn_to_nom: dict, glyphs: dict[str, str],
                seed: int = 0, layout: dict | None = None, noise: dict | None = None):
    """Sinh `n_pages` trang kèm đáp án (chữ, âm, hộp). Tất định theo `seed`."""
    from pipeline.lab.perturb import _seed
    pairs = [(ch, syl) for syl, chs in sorted(qn_to_nom.items())
             for ch in chs if ch in glyphs]
    if not pairs:
        return []
    L = {**DEFAULT_LAYOUT, **(layout or {})}
    pages = []
    for p in range(n_pages):
        rng = random.Random(_seed(seed, p, "synth"))
        cols, syls = [], []
        for _ in range(L["cols"]):
            k = rng.randint(L["rows_min"], L["rows_max"])
            picked = [pairs[rng.randrange(len(pairs))] for _ in range(k)]
            cols.append([c for c, _ in picked])
            syls.append([s for _, s in picked])
        img, boxes = make_page(cols, glyphs, rng, layout, noise)
        pages.append({"page": f"synth_{p:04d}", "image": img, "boxes": boxes,
                      "chars_by_col": cols, "syllables_by_col": syls})
    return pages
