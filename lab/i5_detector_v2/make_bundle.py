#!/usr/bin/env python
"""Gom bundle Kaggle cho detector v2: ảnh trang (thu về long side 1536, PNG xám không mất mát), nhãn yếu
+ ignore + split (quy đổi theo scale), ckpt v1, gói mã i5v2/ tự chứa, notebook, README → zip.

Nguồn (chỉ ĐỌC, không sửa): train_crop/data_lithograph/manifest_all.json (LVT1883 105 + KVK1884 163 trang thạch bản
+ STT 445 trang, split page-disjoint theo sách), prepared/<book>/pages/*.png, prepared/Chrestomathie1872 (kim cache +
transcriptions → ô tham chiếu cùng cách, chỉ để ĐO — sách không có trang nào trong train), train_crop/detector_r34.best.pt.

Chạy:  .venv/bin/python lab/i5_detector_v2/make_bundle.py            # → lab/i5_detector_v2/bundle/ + i5v2_bundle.zip
       .venv/bin/python lab/i5_detector_v2/make_bundle.py --limit 3  # thử nhanh (3 trang/sách)
Định dạng ảnh: PNG xám (đo 22/09: 1536 PNG ≈ JPEG q90 về dung lượng nhưng không mất mát); --fmt jpg nếu muốn.
Nếu tổng > 1,5 GB tự hạ long side 1536 → 1280 (không xảy ra với dữ liệu hiện tại ≈ 0,2 GB).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from i5v2 import pitch_decode as PD    # noqa: E402  (bản sao tự chứa; make_bundle không import pipeline/)

INK_MIN, BORDER_MAX, HP_LO, HP_HI = 0.05, 0.20, 0.6, 1.5   # = train_crop/build_lithograph_manifest.py


def _binarize(gray):
    import cv2
    _, b = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (b > 0).astype(np.uint8)


def _cell_ok(binimg, c, p):
    x1, y1, x2, y2 = [int(v) for v in c[:4]]
    sub = binimg[max(0, y1):y2, max(0, x1):x2]
    if sub.size == 0 or sub.shape[0] < 4:
        return False
    ink = float(sub.mean()); border = float(max(sub[:2].mean(), sub[-2:].mean())); hp = (y2 - y1) / p
    return ink >= INK_MIN and border <= BORDER_MAX and HP_LO <= hp <= HP_HI


def chresto_items(limit=0):
    """Chrestomathie1872 (văn xuôi, 1 tầng/cột): ô tham chiếu = hộp kim cột cắt N−1 khe mực (N = num_syllables
    của dòng QN cùng chỉ số), verified = n_chars_kim == N. Chỉ để ĐO (domain 'chresto', split 'chresto')."""
    import cv2
    pdir = REPO / "prepared" / "Chrestomathie1872"
    pages = sorted(p.stem for p in (pdir / "transcriptions").glob("page_*.json"))
    if limit:
        pages = pages[:limit]
    items, st = [], Counter()
    for page in pages:
        img = pdir / "pages" / f"{page}.png"; cache_p = pdir / "detected" / f"{page}_ocr_cache.json"
        if not img.exists() or not cache_p.exists():
            continue
        gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        cache = json.load(open(cache_p, encoding="utf-8"))
        trans = json.load(open(pdir / "transcriptions" / f"{page}.json", encoding="utf-8"))
        tcols = trans.get("columns") or []
        pitch = cache.get("col_pitch")
        binimg = _binarize(gray)
        boxes, ignore, tiers = [], [], []
        for j, chars in enumerate(cache.get("columns") or []):
            if not chars or j >= len(tcols):
                continue
            tc = tcols[j]
            n = int(tc.get("num_syllables") or 0)
            if n <= 0:
                continue
            x1, y0, x2, y1 = PD.tier_box(chars, list(range(len(chars))))
            if pitch:
                xc = (x1 + x2) / 2.0
                x1, x2 = int(max(x1, xc - 0.5 * pitch)), int(min(x2, xc + 0.5 * pitch))
            st["cols"] += 1
            if len(chars) != n:
                st["cols_unverified"] += 1; ignore.append([x1, y0, x2, y1]); continue
            cells = PD.ink_cut_cells(gray, x1, x2, y0, y1, n)
            p = (y1 - y0) / n
            # Văn xuôi in dày (bước ~80 px, chữ chạm nhau) → tiêu chí "mực chạm mép" loại phần lớn ô; vì chỉ để ĐO
            # (không train) nên GIỮ mọi cột verified: % cột n==N và cắt-thân-chữ không phụ thuộc ô; ok50 là proxy yếu hơn.
            n_ok = sum(_cell_ok(binimg, c, p) for c in cells)
            st["cells"] += len(cells); st["cells_ok"] += n_ok
            st["cols_kept"] += 1
            tiers.append([x1, y0, x2, y1, n, len(boxes)])
            boxes += [[int(v) for v in c] for c in cells]
        if not tiers:
            continue
        items.append({"image": str(img), "book": "Chrestomathie1872", "page": page, "domain": "chresto",
                      "boxes": boxes, "labels": [""] * len(boxes), "n_boxes": len(boxes), "ignore_boxes": ignore,
                      "tiers": tiers, "split": "chresto"})
    print(f"[chresto] {len(items)} trang, {sum(i['n_boxes'] for i in items)} ô tham chiếu (chỉ đo; ô đạt lọc chất lượng "
          f"{st['cells_ok']}/{st['cells']}), {dict(st)}", flush=True)
    return items


def _scale_item(it, s, rel_image):
    def sb(b):
        return [round(float(v) * s, 1) for v in b[:4]]
    out = {"image": rel_image, "book": it["book"], "page": it["page"], "domain": it["domain"], "split": it.get("split", ""),
           "boxes": [sb(b) for b in it.get("boxes") or []],
           "ignore_boxes": [sb(b) for b in it.get("ignore_boxes") or []],
           "tiers": [sb(t) + [int(t[4]), int(t[5])] for t in it.get("tiers") or []],
           "n_boxes": len(it.get("boxes") or []), "scale": round(s, 6)}
    return out


def build(a):
    import cv2
    bundle = Path(a.out)
    if bundle.exists() and a.clean:
        shutil.rmtree(bundle)
    (bundle / "images").mkdir(parents=True, exist_ok=True)
    man = json.load(open(REPO / "train_crop" / "data_lithograph" / "manifest_all.json", encoding="utf-8"))
    if a.limit:
        keep, cnt = [], Counter()
        for it in man:
            if cnt[it["book"]] < a.limit:
                keep.append(it); cnt[it["book"]] += 1
        man = keep
    items = man + chresto_items(a.limit)
    rows, out_items, n_bytes, t0 = [], [], 0, time.time()
    ext = ".png" if a.fmt == "png" else ".jpg"
    for k, it in enumerate(items):
        src = Path(it["image"])
        if not src.is_absolute():
            src = REPO / src
        g = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"  bỏ (không đọc được) {src}", flush=True); continue
        H, W = g.shape[:2]
        s = min(1.0, a.long_side / max(H, W))
        nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
        r = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA) if s < 1.0 else g
        rel = f"images/{it['book']}/{it['page']}{ext}"
        dst = bundle / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        if a.fmt == "png":
            cv2.imwrite(str(dst), r, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(str(dst), r, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        n_bytes += dst.stat().st_size
        rows.append({"book": it["book"], "page": it["page"], "domain": it["domain"], "split": it.get("split", ""),
                     "orig_image": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src),
                     "orig_w": W, "orig_h": H, "new_w": nw, "new_h": nh, "scale": round(s, 6), "bundle_image": rel})
        out_items.append(_scale_item(it, s, rel))
        if k % 100 == 0:
            print(f"  ảnh {k}/{len(items)} ({n_bytes / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
    with open(bundle / "scale_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    by_split = Counter()
    for sp in ("train", "val", "test", "chresto"):
        sub = [i for i in out_items if i["split"] == sp]
        json.dump(sub, open(bundle / f"manifest_{sp}.json", "w", encoding="utf-8"), ensure_ascii=False)
        for i in sub:
            by_split[f"{i['domain']}/{sp}"] += 1
    json.dump(out_items, open(bundle / "manifest_all.json", "w", encoding="utf-8"), ensure_ascii=False)
    # v1 ckpt + mã
    (bundle / "v1").mkdir(exist_ok=True)
    shutil.copy2(REPO / "train_crop" / "detector_r34.best.pt", bundle / "v1" / "detector_r34.best.pt")
    if (bundle / "i5v2").exists():
        shutil.rmtree(bundle / "i5v2")
    shutil.copytree(HERE / "i5v2", bundle / "i5v2", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("train_kaggle.py", "kaggle_train_i5_v2.ipynb"):
        if (HERE / f).exists():
            shutil.copy2(HERE / f, bundle / f)
    stats = {"date": time.strftime("%Y-%m-%d %H:%M"), "long_side": a.long_side, "fmt": a.fmt, "quality": a.quality,
             "n_images": len(rows), "by_domain_split": dict(by_split), "image_bytes": n_bytes,
             "boxes": {d: sum(i["n_boxes"] for i in out_items if i["domain"] == d) for d in ("litho", "stt", "chresto")},
             "ignore": {d: sum(len(i["ignore_boxes"]) for i in out_items if i["domain"] == d) for d in ("litho", "stt", "chresto")},
             "source_manifest": "train_crop/data_lithograph/manifest_all.json", "v1_ckpt": "train_crop/detector_r34.best.pt"}
    json.dump(stats, open(bundle / "bundle_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    (bundle / "README.txt").write_text(README_TXT.format(**stats, mb=n_bytes / 1e6), encoding="utf-8")
    return stats


README_TXT = """i5v2 bundle — huấn luyện detector CenterNet v2 (thạch bản + STT) trên Kaggle · {date}

NỘI DUNG
  images/<book>/<page>.{fmt}   {n_images} trang, long side {long_side} px, xám ({mb:.0f} MB): LucVanTien1883, KimVanKieu1884,
                              yen2/yen4/yen11 (= SachThanhTruyen 2/4/11), Chrestomathie1872 (chỉ để đo, không train)
  manifest_train/val/test.json nhãn yếu (boxes) + ignore_boxes + tiers [x1,y0,x2,y1,N,start] ở toạ độ ẢNH BUNDLE
  manifest_chresto.json         Chrestomathie1872 held-out (domain chresto)
  scale_table.csv               book,page → orig_w/h, new_w/h, scale (bbox_gốc = bbox_bundle / scale)
  v1/detector_r34.best.pt       checkpoint v1 (init + mốc hồi quy STT)
  i5v2/ + train_kaggle.py       mã tự chứa (torch/torchvision/numpy/opencv), không cần repo
  kaggle_train_i5_v2.ipynb      notebook Run All

UPLOAD LÊN KAGGLE (1 lần)
  1. kaggle.com → Datasets → New Dataset → kéo tệp i5v2_bundle.zip (Kaggle tự giải nén) → Title: "i5v2 bundle" → Create.
     (hoặc CLI: kaggle datasets create -p bundle/ --dir-mode zip  sau khi có dataset-metadata.json)
  2. Notebooks → New Notebook → File → Import Notebook → kaggle_train_i5_v2.ipynb.
  3. Add Input → dataset vừa tạo. Settings: Accelerator GPU T4 (x2 cũng được, dùng 1), Internet On (chỉ cần nếu đẩy HF),
     Persistence: Files only (giữ /kaggle/working khi restart).
  4. Run All. ≈ 1–1,5 giờ (20 epoch × 2–3 phút + eval). Kết quả: /kaggle/working/{{best.pt,last.pt,metrics.csv,report.md}}.
  5. Tải best.pt (Output tab) về máy → lab/i5_detector_v2/apply_v2.sh best.pt

MẤT KẾT NỐI / RESET
  Run All lại: trainer tự resume từ /kaggle/working/last.pt (Persistence Files only) hoặc từ HF hub nếu khai HF_REPO + Secret HF_TOKEN.
"""


def verify_import(bundle: Path) -> bool:
    """`python -c 'import i5v2'` với PYTHONPATH = chỉ bundle, cwd = thư mục tạm (không thấy repo)."""
    import os, tempfile
    tmp = tempfile.mkdtemp(prefix="i5v2_verify_")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["PYTHONPATH"] = str(bundle.resolve())
    env["PYTHONNOUSERSITE"] = "1"
    code = ("import i5v2, i5v2.model, i5v2.data, i5v2.loss, i5v2.decode, i5v2.evalref, i5v2.trainer, i5v2.pitch_decode, i5v2.hub;"
            "import sys; assert all(('GanNhanOCR/train_crop' not in (m.__file__ or '') and 'pipeline' not in (m.__name__)) "
            "for n, m in sys.modules.items() if n.startswith('i5v2')); print('import i5v2 OK', i5v2.__version__)")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-400:])
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "bundle"))
    ap.add_argument("--zip", default=str(HERE / "i5v2_bundle.zip"))
    ap.add_argument("--long-side", type=int, default=1536)
    ap.add_argument("--fmt", default="png", choices=["png", "jpg"])
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0, help="số trang mỗi sách (thử)")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-clean", dest="clean", action="store_false")
    ap.add_argument("--max-gb", type=float, default=1.5)
    a = ap.parse_args()
    st = build(a)
    if st["image_bytes"] > a.max_gb * 1e9 and a.long_side > 1280:
        print(f"[bundle] ảnh {st['image_bytes'] / 1e9:.2f} GB > {a.max_gb} GB → dựng lại ở long side 1280", flush=True)
        a.long_side = 1280; st = build(a)
    ok = verify_import(Path(a.out))
    total = sum(p.stat().st_size for p in Path(a.out).rglob("*") if p.is_file())
    print(f"[bundle] {a.out}: {st['n_images']} ảnh ({st['image_bytes'] / 1e6:.0f} MB) + v1 ckpt + mã = {total / 1e6:.0f} MB "
          f"| {st['by_domain_split']} | boxes {st['boxes']} | import i5v2 {'OK' if ok else 'LỖI'}", flush=True)
    if not a.no_zip:
        zp = Path(a.zip)
        if zp.exists():
            zp.unlink()
        t0 = time.time()
        subprocess.run(["zip", "-qr", str(zp), "."], cwd=a.out, check=True)
        print(f"[bundle] {zp} = {zp.stat().st_size / 1e6:.0f} MB ({time.time() - t0:.0f}s)", flush=True)
        # (2026-09-23) GÓI MÃ RIÊNG, ~50 KB. Khi CHỈ mã đổi (vd guard STT / best_litho.pt) thì
        # không phải đẩy lại 243 MB ảnh: tạo/cập nhật một Kaggle dataset chỉ chứa zip này và
        # Add Input thêm nó; cell (2) của notebook ưu tiên thư mục input có train_kaggle.py mà
        # KHÔNG có bundle_stats.json, nên ảnh vẫn lấy từ bundle cũ.
        cz = zp.with_name("i5v2_code.zip")
        if cz.exists():
            cz.unlink()
        subprocess.run(["zip", "-qr", str(cz), "i5v2", "train_kaggle.py"], cwd=a.out, check=True)
        print(f"[bundle] {cz} = {cz.stat().st_size / 1e3:.0f} KB (chỉ mã — đẩy riêng khi chỉ sửa mã)", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
