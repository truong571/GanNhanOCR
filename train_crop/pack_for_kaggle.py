"""Đóng gói code + dữ liệu để TRAIN TRÊN KAGGLE (GPU T4/P100).

Gói gồm:
  • 5 module CenterNet mới (data/model/train/infer + make_report_pdf)
  • images/        — trang đã downscale (chữ ~40–60px, đủ để định vị; gói nhẹ)
  • detect_manifest.json — đường dẫn TƯƠNG ĐỐI + box đã scale

`detect_manifest.json` gốc trỏ đường dẫn TUYỆT ĐỐI full-res. Để lên Kaggle ta
downscale ảnh + scale box + ghi lại manifest đường-dẫn-tương-đối. Để nhanh và
KHÔNG đụng CPU với phiên train local, mặc định ta TÁI SỬ DỤNG ảnh đã downscale
trong `evaluation/ver_new/char_detector/kaggle_det_pkg/` nếu có.

Chạy:
  .venv/bin/python test/pack_for_kaggle.py          # -> test/kaggle_pkg/
  # rồi upload test/kaggle_pkg/ lên Kaggle dưới dạng 1 Dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_FULL_MANIFEST = REPO / "evaluation/ver_new/char_detector/detect_manifest.json"
DEFAULT_REUSE = REPO / "evaluation/ver_new/char_detector/kaggle_det_pkg"

MODULES = ["data_centernet.py", "model_centernet.py", "train_centernet.py",
           "infer_centernet.py", "make_report_pdf.py", "build_manifest.py",
           "build_mth_pretrain.py", "kaggle_train.ipynb", "README.md", "KAGGLE.md"]


def _downscale_from_full(full_manifest, out, max_side):
    import cv2
    man = json.load(open(full_manifest, encoding="utf-8"))
    (out / "images").mkdir(parents=True, exist_ok=True)
    new_man, skipped = [], 0
    for i, it in enumerate(man):
        im = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        if im is None:
            skipped += 1; continue
        H, W = im.shape[:2]
        s = min(1.0, max_side / max(H, W))
        if s < 1.0:
            im = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        fn = f"{it.get('book','p')}_{it.get('page', i)}.png"
        cv2.imwrite(str(out / "images" / fn), im)
        boxes = [[round(x1 * s, 1), round(y1 * s, 1), round(x2 * s, 1), round(y2 * s, 1)]
                 for x1, y1, x2, y2 in it["boxes"]]
        new_man.append({"image": f"images/{fn}", "boxes": boxes, "n_boxes": len(boxes)})
        if (i + 1) % 100 == 0:
            print(f"  ... downscale {i+1}/{len(man)}", flush=True)
    json.dump(new_man, open(out / "detect_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    return len(new_man), skipped


def _reuse_existing(reuse_dir, out):
    """Sao chép images/ + detect_manifest.json đã downscale sẵn."""
    shutil.copytree(reuse_dir / "images", out / "images")
    man = json.load(open(reuse_dir / "detect_manifest.json", encoding="utf-8"))
    slim = [{"image": it["image"], "boxes": it["boxes"], "n_boxes": it.get("n_boxes", len(it["boxes"]))}
            for it in man]
    json.dump(slim, open(out / "detect_manifest.json", "w", encoding="utf-8"), ensure_ascii=False)
    return len(slim), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "kaggle_pkg"))
    ap.add_argument("--reuse", default=str(DEFAULT_REUSE),
                    help="thư mục kaggle_det_pkg có images/ đã downscale (để tái dùng)")
    ap.add_argument("--manifest", default=str(DEFAULT_FULL_MANIFEST),
                    help="manifest full-res (fallback khi không tái dùng được)")
    ap.add_argument("--max", type=int, default=1280, help="cạnh dài tối đa khi downscale")
    ap.add_argument("--force-downscale", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    reuse = Path(args.reuse)
    if (not args.force_downscale and (reuse / "images").is_dir()
            and (reuse / "detect_manifest.json").exists()):
        print(f"[pack] tái dùng ảnh downscale từ {reuse}")
        n, skipped = _reuse_existing(reuse, out)
    else:
        print(f"[pack] downscale từ manifest full-res {args.manifest} (max {args.max}px)")
        n, skipped = _downscale_from_full(args.manifest, out, args.max)

    for m in MODULES:
        src = HERE / m
        if src.exists():
            shutil.copy(src, out / m)

    size_mb = sum(p.stat().st_size for p in (out / "images").glob("*")) / 1e6
    print(f"\n[pack] -> {out}")
    print(f"  trang {n} (bỏ {skipped}) | ảnh {size_mb:.0f} MB | modules {[m for m in MODULES if (out/m).exists()]}")
    print("\nBƯỚC TIẾP:")
    print(f"  1) Upload thư mục {out} lên Kaggle = 1 Dataset (vd. 'nom-char-det-r34').")
    print( "  2) Mở Kaggle Notebook GPU (T4/P100, Internet ON), attach dataset đó.")
    print( "  3) Import test/kaggle_train.ipynb -> Run All  (hoặc xem test/KAGGLE.md).")


if __name__ == "__main__":
    main()
