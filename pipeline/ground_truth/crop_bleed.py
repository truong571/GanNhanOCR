"""Đo KHÁCH QUAN lượng mực của chữ bên cạnh lọt vào crop (`bleed_frac`).

VÌ SAO PHẢI ĐO CHỨ KHÔNG HỎI NGƯỜI
----------------------------------
Kiểm tra lặp ngày 2026-08-04 đo được κ = 0,14 cho phán đoán "crop có sạch không": chính
người chấm, trên chính những ô đó, đảo verdict 8 lần gắn mới / 6 lần gỡ bỏ trên 40 ô. Tiêu
chí `wrong_image` ("crop cắt lỗi/dính glyph hàng xóm") KHÔNG có ngưỡng, và với chữ Nôm thảo
thì "bao nhiêu mực hàng xóm là hỏng" không có câu trả lời tự nhiên — nó trôi khi xem thêm
ví dụ (tỷ lệ gán lỗi qua 3 buổi: 4,2% → 16,0% → 35,0%).

Vì vậy đại lượng này được ĐỊNH NGHĨA chứ không bỏ phiếu, và **cố ý không hiệu chỉnh theo
verdict người** — hiệu chỉnh theo một cái đích là nhiễu thì chỉ học được nhiễu.

ĐỊNH NGHĨA
----------
Với mỗi crop đã lưu, mọi điểm ảnh đều truy ngược được về toạ độ trang. Một điểm mực bị coi
là NGOẠI LAI nếu toạ độ trang của nó nằm trong bbox của MỘT CHỮ KHÁC trên cùng trang (bbox
do chính detector sinh ra, mọi tier) và nằm ngoài bbox của chính nó.

    bleed_frac = (số điểm mực ngoại lai) / (tổng số điểm mực trong crop)

Không có ngưỡng nào được chọn tuỳ ý ở đây: "chữ khác" là đầu ra của detector, không phải
phán đoán. Ai muốn đặt ngưỡng chấp nhận thì đặt ở tầng trên, và phải nói rõ ngưỡng đó.

VÌ SAO KHÔNG DÙNG PROFILE CHIẾU NGANG
-------------------------------------
Cách hiển nhiên hơn — tìm dải trắng vắt ngang rồi coi mảnh tách rời là mực lạ — ĐÃ THỬ và
HỎNG: chữ Nôm cấu trúc trên-dưới (⿱) vốn có sẵn dải trắng giữa hai bộ phận, nên chỉ số bắn
đều trên cả ô sạch (trung vị 0,17 ở mọi nhóm verdict). Đây đúng là cái bẫy "⿱ single char"
đã ghi nhận từ trước.

GIỚI HẠN (phải nói rõ khi trích dẫn)
------------------------------------
Chỉ thấy được phần mực thuộc chữ mà detector CÓ khoanh. Nếu detector bỏ sót hẳn một chữ
hàng xóm thì phần mực của nó không bị tính. Nên con số này là **cận dưới** của mức nhiễm.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline.align_engine.bbox_fix import carve_neighbor_ink, tighten_box

REPO = Path(__file__).resolve().parents[2]

__all__ = ["CropGeom", "reproduce_crop", "bleed_fraction", "measure_page", "measure_corpus"]


@dataclass(frozen=True)
class CropGeom:
    """Crop tái lập + phép ánh xạ ngược về toạ độ trang."""
    crop: np.ndarray
    x0: int          # toạ độ trang của cột đầu tiên trong crop
    y0: int          # toạ độ trang của hàng đầu tiên trong crop


def _parse_bbox(v) -> list[int] | None:
    try:
        arr = json.loads(v) if isinstance(v, str) else v
        if arr is None:
            return None
        return [int(round(float(x))) for x in arr]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def reproduce_crop(img, gray_full, bbox, pad: float, prev_bbox, next_bbox,
                   tighten: bool = True) -> CropGeom | None:
    """Tái lập ĐÚNG phép cắt của `build_dataset.save_crop`, kèm gốc toạ độ.

    Phải bám sát save_crop từng bước; tính đúng đắn được CHỨNG MINH bằng cách so md5 của
    crop tái lập với `image_md5` đã ghi trong labels.csv (xem measure_corpus).
    """
    if img is None or not bbox:
        return None
    H, W = img.shape[:2]
    ox1, oy1, ox2, oy2 = (int(v) for v in bbox)
    pw, ph = int((ox2 - ox1) * pad), int((oy2 - oy1) * pad)
    x1, y1 = max(0, ox1 - pw), max(0, oy1 - ph)
    x2, y2 = min(W, ox2 + pw), min(H, oy2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = crop.copy()
    if gray_full is not None and (prev_bbox is not None or next_bbox is not None):
        crop = carve_neighbor_ink(crop, gray_full, x1, y1, x2, y2, (oy1, oy2),
                                  prev_bbox, next_bbox)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ox, oy = x1, y1
    if tighten:
        tb = tighten_box(gray)
        if tb is not None:
            a, c, b, d = tb
            crop = crop[c:d, a:b]
            ox, oy = x1 + a, y1 + c
    if crop.size == 0:
        return None
    return CropGeom(crop=crop, x0=ox, y0=oy)


def bleed_fraction(geom: CropGeom, own_bbox, others: list[list[int]]) -> tuple[float, int, int]:
    """Trả về (bleed_frac, số điểm mực ngoại lai, tổng điểm mực)."""
    crop = geom.crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    dark = gray < 128
    total = int(dark.sum())
    if total == 0:
        return 0.0, 0, 0
    h, w = dark.shape
    ox1, oy1, ox2, oy2 = own_bbox

    foreign = np.zeros((h, w), dtype=bool)
    for bx1, by1, bx2, by2 in others:
        # giao của bbox chữ khác với khung crop, quy về toạ độ crop
        cx1 = max(bx1 - geom.x0, 0)
        cy1 = max(by1 - geom.y0, 0)
        cx2 = min(bx2 - geom.x0, w)
        cy2 = min(by2 - geom.y0, h)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        foreign[cy1:cy2, cx1:cx2] = True
    if not foreign.any():
        return 0.0, 0, total

    # trừ đi phần thuộc bbox của CHÍNH NÓ (vùng chồng lấn thì chữ này được ưu tiên)
    sx1 = max(ox1 - geom.x0, 0)
    sy1 = max(oy1 - geom.y0, 0)
    sx2 = min(ox2 - geom.x0, w)
    sy2 = min(oy2 - geom.y0, h)
    if sx2 > sx1 and sy2 > sy1:
        foreign[sy1:sy2, sx1:sx2] = False

    n_foreign = int((dark & foreign).sum())
    return n_foreign / total, n_foreign, total


def detached_fraction(geom: CropGeom, own_bbox) -> float:
    """Tỷ lệ mực KHÔNG liên thông với phần mực nằm trong bbox của chính chữ đó.

    Bổ sung cho `bleed_fraction`, và không có điểm mù của nó: ở đây không cần hàng xóm
    phải được detector khoanh. Bù lại chỉ bắt được ca RÕ RỆT — mảnh mực tách hẳn khỏi
    thân chữ. Nét của chính chữ vươn ra ngoài bbox vẫn liên thông nên không bị tính oan;
    chữ cấu trúc ⿱ cũng không bị tính oan vì cả hai bộ phận đều nằm trong bbox.

    = 1.0 nghĩa là KHÔNG còn chút mực nào dính với chữ được khoanh: crop hỏng hoàn toàn
    (thường do carve xoá sạch thân chữ, chỉ còn lại vụn).
    """
    c = geom.crop
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
    dark = (gray < 128).astype(np.uint8)
    total = int(dark.sum())
    if total == 0:
        return 0.0
    h, w = dark.shape
    ox1, oy1, ox2, oy2 = own_bbox
    inbox = np.zeros((h, w), dtype=bool)
    x1, y1 = max(ox1 - geom.x0, 0), max(oy1 - geom.y0, 0)
    x2, y2 = min(ox2 - geom.x0, w), min(oy2 - geom.y0, h)
    if x2 > x1 and y2 > y1:
        inbox[y1:y2, x1:x2] = True
    n_lab, lab = cv2.connectedComponents(dark, 8)
    keep = set(np.unique(lab[dark.astype(bool) & inbox])) - {0}
    if not keep:
        return 1.0
    foreign = dark.astype(bool) & ~np.isin(lab, list(keep))
    return float(foreign.sum()) / total


def measure_page(page_png: str, recs: list[dict], pad: float,
                 tighten: bool = True, carve: bool = True) -> list[dict]:
    """Đo mọi bản ghi CÓ ẢNH của một trang. `recs` phải gồm CẢ hàng không có ảnh
    (tier REVIEW) vì bbox của chúng vẫn đánh dấu chỗ mực hàng xóm."""
    img = cv2.imread(page_png, cv2.IMREAD_COLOR)
    if img is None:
        return []
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if carve else None

    # hàng xóm trước/sau trong CÙNG CỘT theo thứ tự y — giống hệt build_dataset
    by_col = defaultdict(list)
    for r in recs:
        if r["_bbox"]:
            by_col[r["column"]].append(r)
    for col_recs in by_col.values():
        col_recs.sort(key=lambda r: (r["_bbox"][1] + r["_bbox"][3]) / 2.0)
        for i, r in enumerate(col_recs):
            r["_prev"] = col_recs[i - 1]["_bbox"] if i > 0 else None
            r["_next"] = col_recs[i + 1]["_bbox"] if i < len(col_recs) - 1 else None

    all_boxes = [(r["_bbox"], id(r)) for r in recs if r["_bbox"]]
    out = []
    for r in recs:
        # `image` rỗng ở tier REVIEW đọc từ CSV ra là NaN (float), mà `not nan` là False —
        # nên phải kiểm KIỂU, nếu không các hàng không có ảnh sẽ lọt vào phép đo.
        img_rel = r.get("image")
        if not isinstance(img_rel, str) or not img_rel or not r["_bbox"]:
            continue
        geom = reproduce_crop(img, gray_full, r["_bbox"], pad,
                              r.get("_prev"), r.get("_next"), tighten=tighten)
        if geom is None:
            continue
        others = [b for b, k in all_boxes if k != id(r)]
        frac, n_f, n_tot = bleed_fraction(geom, r["_bbox"], others)
        det = detached_fraction(geom, r["_bbox"])
        ok, enc = cv2.imencode(".png", geom.crop)
        md5 = hashlib.md5(enc.tobytes()).hexdigest()[:12] if ok else ""
        out.append({
            "image": r["image"], "bleed_frac": frac, "detached_frac": det,
            "bleed_px": n_f, "ink_px": n_tot,
            "repro_md5": md5, "md5_match": bool(md5 and md5 == str(r.get("image_md5") or "")),
        })
    return out


def measure_corpus(labels: pd.DataFrame, prepared_dir: Path, pad: float = 0.12,
                   tighten: bool = True, carve: bool = True,
                   limit_pages: int = 0, progress: bool = True) -> pd.DataFrame:
    """Đo toàn corpus. Trả về khung (image, bleed_frac, bleed_px, ink_px, md5_match)."""
    from .audit_grid import book_to_scan_dir

    df = labels.copy()
    df["_bbox"] = df["bbox"].map(_parse_bbox)
    groups = list(df.groupby(["book", "page"], sort=True))
    if limit_pages:
        groups = groups[:limit_pages]

    rows = []
    for i, ((book, page), g) in enumerate(groups):
        png = prepared_dir / book_to_scan_dir(str(book)) / "pages" / f"{page}.png"
        if not png.exists():
            continue
        recs = g.to_dict("records")
        rows.extend(measure_page(str(png), recs, pad, tighten=tighten, carve=carve))
        if progress and (i + 1) % 50 == 0:
            print(f"   {i + 1}/{len(groups)} trang, {len(rows):,} crop", flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.crop_bleed")
    ap.add_argument("--labels", default=str(REPO / "dataset_out" / "labels_remediated.csv"))
    ap.add_argument("--prepared", default=str(REPO / "prepared"))
    ap.add_argument("--out", default=str(REPO / "dataset_out" / "crop_bleed.csv"))
    ap.add_argument("--pad", type=float, default=0.12,
                    help="PHẢI khớp --pad của build_dataset (mặc định 0.12); lưu ý "
                         "config `crop_pad_frac: 0.18` KHÔNG được build_dataset đọc")
    ap.add_argument("--limit-pages", type=int, default=0, dest="limit_pages")
    args = ap.parse_args(argv)

    labels = pd.read_csv(args.labels, dtype={"image_md5": str})
    print(f"[bleed] đo trên {len(labels):,} hàng, pad={args.pad}")
    res = measure_corpus(labels, Path(args.prepared), pad=args.pad,
                         limit_pages=args.limit_pages)
    if res.empty:
        print("[bleed] không đo được crop nào")
        return 1

    match = float(res["md5_match"].mean())
    print(f"[bleed] md5 tái lập khớp: {int(res['md5_match'].sum()):,}/{len(res):,} "
          f"({match:.2%})")
    if match < 1.0:
        print("[bleed] CẢNH BÁO: phép tái lập KHÔNG khớp hoàn toàn — mọi con số dưới đây "
              "đo trên một phép cắt KHÁC với crop đã lưu. Kiểm --pad trước khi dùng.")

    res.to_csv(args.out, index=False)
    print(f"[bleed] -> {args.out}")

    j = labels.merge(res, on="image", how="inner")
    for tier, g in j.groupby("tier"):
        q = g["bleed_frac"]
        d = g["detached_frac"]
        print(f"  {tier:9s} n={len(g):6,} | bleed trung vị={q.median():.4f} p99={q.quantile(.99):.4f}"
              f" | detached trung vị={d.median():.4f} p99={d.quantile(.99):.4f}"
              f" >50%: {(d > 0.5).sum():4d} ({(d > 0.5).mean():.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
