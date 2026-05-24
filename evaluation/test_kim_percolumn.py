"""Test: feed Kim Han-Nom OCR EACH COLUMN separately (vs current full-page).

Motivation: on p12, full-page Kim mất 2 cột (`与丄`, `石丄` = 2 char rác)
because the column is at the edge / low contrast. Hypothesis: cropping each
column tightly and submitting individually gives Kim a cleaner image with
no neighbour interference → higher per-column recall.

Procedure
─────────
For pages p12 + p14 of SachThanhTruyen2:
  1. Get the 9 column bboxes (from existing aligned/_aligned.json char bboxes).
  2. Crop each column with horizontal padding (so Kim sees marker zone too).
  3. Upload + recognize each column image separately.
  4. Compare against full-page OCR cache (detected/<page>_ocr_cache.json).

Outputs to evaluation/percol_kim_test/:
  - report.md                  side-by-side per-column comparison
  - col_crops/<page>/colNN.png cropped column images sent to API
  - col_ocr/<page>/colNN.json  raw API response
  - summary.json               machine-readable metrics
"""
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(ROOT))

from core.ocr.ocr_api import upload_image, recognize  # noqa: E402

BOOK_DIR = ROOT / "prepared" / "SachThanhTruyen2"
OUT_DIR = ROOT / "evaluation" / "percol_kim_test"
CROP_DIR = OUT_DIR / "col_crops"
OCR_DIR = OUT_DIR / "col_ocr"
for d in (OUT_DIR, CROP_DIR, OCR_DIR):
    d.mkdir(parents=True, exist_ok=True)

PAGES = ["page_0012", "page_0014"]
PAD_X = 30  # extra pixels left/right around column bbox
PAD_Y = 60  # extra pixels top/bottom (Kim needs context for marker strip)


def get_col_bboxes(page: str) -> dict[int, tuple[int, int, int, int]]:
    """Compute (x1,y1,x2,y2) per column from char bboxes in aligned file."""
    al = json.load(open(BOOK_DIR / "aligned" / f"{page}_aligned.json"))
    cols: dict[int, list[list[int]]] = {}
    for r in al:
        if r.get("type") != "match":
            continue
        ci = r.get("char") or {}
        bb = ci.get("bbox")
        col = r.get("column")
        if bb and col:
            cols.setdefault(col, []).append(bb)
    out = {}
    for c, bbs in cols.items():
        x1 = min(b[0] for b in bbs)
        y1 = min(b[1] for b in bbs)
        x2 = max(b[2] for b in bbs)
        y2 = max(b[3] for b in bbs)
        out[c] = (x1, y1, x2, y2)
    return out


def crop_and_save(page: str, col: int, bbox: tuple[int, int, int, int]) -> Path:
    img = cv2.imread(str(BOOK_DIR / "pages" / f"{page}.png"))
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - PAD_X); y1 = max(0, y1 - PAD_Y)
    x2 = min(W, x2 + PAD_X); y2 = min(H, y2 + PAD_Y)
    crop = img[y1:y2, x1:x2]
    out_dir = CROP_DIR / page
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"col{col:02d}.png"
    cv2.imwrite(str(p), crop)
    return p


def call_kim(image_path: Path, cache_file: Path) -> dict:
    """Upload + recognize. Cached so reruns don't re-bill API."""
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    file_name = upload_image(str(image_path))
    if not file_name:
        return {"error": "upload_failed", "boxes": []}
    boxes = recognize(file_name) or []
    out = {"file_name": file_name, "boxes": boxes}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def text_from_boxes(boxes: list[dict]) -> tuple[str, int]:
    """Flatten transcription, sort top-to-bottom by y of first point."""
    if not boxes:
        return "", 0
    items = []
    for b in boxes:
        txt = (b.get("transcription") or "").strip()
        if not txt:
            continue
        y = b.get("points", [[0, 0]])[0][1]
        items.append((y, txt))
    items.sort()
    joined = "".join(t for _, t in items)
    # Count visible chars (ignore whitespace)
    n_chars = sum(1 for c in joined if not c.isspace())
    return joined, n_chars


def full_page_kim_per_col(page: str) -> dict[int, str]:
    """Replicate hybrid filter on full-page Kim cache to get per-col text."""
    cache = json.load(open(BOOK_DIR / "detected" / f"{page}_ocr_cache.json"))
    clusters = []
    for cl in cache.get("columns", []):
        if len(cl) < 4:
            continue
        cx = sum((c["bbox"][0] + c["bbox"][2]) / 2 for c in cl) / len(cl)
        cs = sorted(cl, key=lambda c: c["bbox"][1])
        clusters.append((cx, "".join(c["char"] for c in cs)))
    clusters.sort(key=lambda x: -x[0])
    return {i + 1: txt for i, (_, txt) in enumerate(clusters)}


def expected_count_per_col(page: str) -> dict[int, int]:
    """How many chars the alignment ended up using per column."""
    al = json.load(open(BOOK_DIR / "aligned" / f"{page}_aligned.json"))
    out: dict[int, int] = {}
    for r in al:
        if r.get("type") != "match":
            continue
        out[r["column"]] = out.get(r["column"], 0) + 1
    return out


def main():
    summary = {"pages": {}, "totals": {
        "full_page_chars": 0, "per_col_chars": 0,
        "full_page_missing_cols": 0, "per_col_missing_cols": 0,
        "calls_made": 0,
    }}

    for page in PAGES:
        print(f"\n=== {page} ===")
        bboxes = get_col_bboxes(page)
        full = full_page_kim_per_col(page)
        expected = expected_count_per_col(page)

        per_col_results: dict[int, dict] = {}
        for col in sorted(bboxes):
            crop_path = crop_and_save(page, col, bboxes[col])
            cache_file = OCR_DIR / page / f"col{col:02d}.json"
            t0 = time.time()
            api = call_kim(crop_path, cache_file)
            elapsed = time.time() - t0
            if "boxes" not in api or not api.get("boxes"):
                if not cache_file.exists():
                    summary["totals"]["calls_made"] += 0
                txt, n = "", 0
            else:
                txt, n = text_from_boxes(api["boxes"])
                if not cache_file.read_text().startswith('{"file_name"'):
                    pass
            if not cache_file.parent.exists() or not cache_file.exists():
                summary["totals"]["calls_made"] += 1
            full_txt = full.get(col, "")
            full_n = len(full_txt)
            exp_n = expected.get(col, 0)
            per_col_results[col] = {
                "expected": exp_n,
                "full_page": {"n": full_n, "text": full_txt},
                "per_col":   {"n": n, "text": txt, "elapsed_s": round(elapsed, 2)},
                "delta_n":   n - full_n,
            }
            print(f"  col{col:>2}: exp={exp_n:>3}  full={full_n:>3}  per-col={n:>3}  "
                  f"Δ={n-full_n:+}  ({elapsed:.1f}s)")
            print(f"    full : {full_txt}")
            print(f"    perCol: {txt}")

        # Aggregates
        page_summary = {
            "n_cols": len(per_col_results),
            "full_page_total_chars": sum(r["full_page"]["n"] for r in per_col_results.values()),
            "per_col_total_chars":   sum(r["per_col"]["n"]   for r in per_col_results.values()),
            "full_page_missing_cols": sum(1 for r in per_col_results.values()
                                          if r["full_page"]["n"] < r["expected"] - 2),
            "per_col_missing_cols":   sum(1 for r in per_col_results.values()
                                          if r["per_col"]["n"] < r["expected"] - 2),
            "expected_total": sum(r["expected"] for r in per_col_results.values()),
            "per_col": per_col_results,
        }
        summary["pages"][page] = page_summary
        summary["totals"]["full_page_chars"] += page_summary["full_page_total_chars"]
        summary["totals"]["per_col_chars"] += page_summary["per_col_total_chars"]
        summary["totals"]["full_page_missing_cols"] += page_summary["full_page_missing_cols"]
        summary["totals"]["per_col_missing_cols"] += page_summary["per_col_missing_cols"]

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    # Markdown report
    lines = ["# Per-column Kim OCR test — SachThanhTruyen2",
             "",
             "Cắt từng cột (đã pad ±30px ngang, ±60px dọc) gửi riêng cho Kim, "
             "so với cách gửi cả trang.",
             "",
             "## Tổng hợp", "",
             "| Trang | n_cols | expected | full-page chars | per-col chars | "
             "full mất cột | per-col mất cột |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for p, ps in summary["pages"].items():
        lines.append(
            f"| {p} | {ps['n_cols']} | {ps['expected_total']} | "
            f"{ps['full_page_total_chars']} | {ps['per_col_total_chars']} | "
            f"{ps['full_page_missing_cols']} | {ps['per_col_missing_cols']} |"
        )
    t = summary["totals"]
    lines += [
        "",
        f"**Tổng 2 trang**: full-page = **{t['full_page_chars']}** char, "
        f"per-col = **{t['per_col_chars']}** char "
        f"(Δ = {t['per_col_chars']-t['full_page_chars']:+}).  ",
        f"Số cột mất ( < expected−2 char): full = **{t['full_page_missing_cols']}**, "
        f"per-col = **{t['per_col_missing_cols']}**.",
        "",
        "## Chi tiết per-col",
    ]

    for p, ps in summary["pages"].items():
        lines += ["", f"### {p}", "",
                  "| col | exp | full n | per-col n | Δ | full text | per-col text |",
                  "|---:|---:|---:|---:|---:|---|---|"]
        for col, r in sorted(ps["per_col"].items()):
            lines.append(
                f"| {col} | {r['expected']} | {r['full_page']['n']} | "
                f"{r['per_col']['n']} | {r['delta_n']:+} | "
                f"`{r['full_page']['text']}` | `{r['per_col']['text']}` |"
            )

    (OUT_DIR / "report.md").write_text("\n".join(lines))
    print(f"\nDone. See: {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()
