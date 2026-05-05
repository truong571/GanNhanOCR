"""A/B/C compare 3 character-crop cleaning approaches.

Approaches:
    A — Current  : full CharacterCleaner pipeline (Sauvola + bg_norm + morph
                   open + stroke_norm + CC noise removal + border line removal)
    B — Simple   : Sauvola binarize → trim → center 64×64 (Mức 2 đề xuất —
                   keep Sauvola for old-paper, drop everything that mutilates
                   strokes)
    C — NEW4     : Adaptive Gaussian threshold → trim → center 64×64
                   (NEW4 paper, 10-line approach)

Output:
    test_crop/results/samples/sample_NNN/
        original.png             (raw crop, varying size)
        A_current.png            (64×64)
        B_simple_sauvola.png     (64×64)
        C_new4_adaptive.png      (64×64)
        info.json                (metadata)
    test_crop/results/grid.png   (visual side-by-side grid)
    test_crop/results/summary.md (textual summary)

Run from repo root:
    PATH="$PWD/.venv/bin:$PATH" python test_crop/run_compare.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.image.crop_cleaner import CharacterCleaner

# --------- config ---------
N_PER_BOOK = 6                      # samples per book
BOOKS = [
    "CacThanhTruyen2",
    "CacThanhTruyen4",
    "CacThanhTruyen11",
    "SachThanhTruyen2",
    "SachThanhTruyen4",
    "SachThanhTruyen11",
]
TARGET = 64                         # output size
PAD = 4
RNG_SEED = 42


# --------- cleaners ---------
def cleaner_current(raw_bgr: np.ndarray) -> np.ndarray | None:
    """A: full CharacterCleaner."""
    cleaner = CharacterCleaner(target_size=TARGET, padding=PAD)
    output, _ = cleaner.clean(raw_bgr)
    return output


def _to_gray(raw_bgr: np.ndarray) -> np.ndarray:
    if len(raw_bgr.shape) == 3:
        return cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    return raw_bgr


def _trim_resize_center(binary: np.ndarray, white_bg: bool) -> np.ndarray | None:
    """Common output stage: trim to ink, aspect-resize, center on canvas.

    `binary`: ink = 255 (white), background = 0 (black) — standard cv2 inverted form.
    `white_bg`: True → final image white background + black ink (visual standard).
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w == 0 or h == 0:
        return None

    char_crop = binary[y:y + h, x:x + w]
    side = TARGET - 2 * PAD
    s = min(side / w, side / h)
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(char_crop, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((TARGET, TARGET), dtype=np.uint8)
    yo = (TARGET - new_h) // 2
    xo = (TARGET - new_w) // 2
    canvas[yo:yo + new_h, xo:xo + new_w] = resized
    return cv2.bitwise_not(canvas) if white_bg else canvas


def cleaner_simple(raw_bgr: np.ndarray) -> np.ndarray | None:
    """B: Sauvola binarize → trim → center. No morph / stroke / CC operations."""
    gray = _to_gray(raw_bgr)
    w_size = max(15, min(51, min(gray.shape) // 3)) | 1
    k, R = 0.2, 128.0

    gray_f = gray.astype(np.float64)
    mean = cv2.boxFilter(gray_f, -1, (w_size, w_size))
    sqmean = cv2.boxFilter(gray_f ** 2, -1, (w_size, w_size))
    variance = np.maximum(sqmean - mean ** 2, 0)
    std = np.sqrt(variance)
    threshold = mean * (1.0 + k * (std / R - 1.0))

    binary = np.zeros_like(gray)
    binary[gray_f < threshold] = 255  # ink = white in inverted form
    return _trim_resize_center(binary, white_bg=True)


def cleaner_new4(raw_bgr: np.ndarray) -> np.ndarray | None:
    """C: NEW4 paper exact — Adaptive Gaussian threshold."""
    gray = _to_gray(raw_bgr)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 8,
    )
    inv = cv2.bitwise_not(binary)  # ink = white
    return _trim_resize_center(inv, white_bg=True)


# --------- sampling ---------
def sample_from_book(book: str, n: int, rng: random.Random) -> list[dict]:
    """Try labeled dataset.json first. If absent (book not labeled yet),
    fall back to scanning aligned/*.json — gives info but without nom_char."""
    p_labeled = REPO / "prepared" / book / "labeled" / "dataset.json"
    if p_labeled.exists():
        with open(p_labeled) as f:
            data = json.load(f)
        matches = [
            d for d in data
            if d.get("type") == "match"
            and d.get("matched")
            and d.get("crop_file")
            and d.get("nom_char")
        ]
        if not matches:
            return []
        return matches if len(matches) <= n else rng.sample(matches, n)

    # Fallback: aligned data (no labels yet — book hasn't run step 3)
    aligned_dir = REPO / "prepared" / book / "aligned"
    aligned_files = sorted(aligned_dir.glob("page_*_aligned.json"))
    if not aligned_files:
        print(f"  [warn] {book}: no labeled/dataset.json and no aligned/ files",
              file=sys.stderr)
        return []
    pool: list[dict] = []
    for af in aligned_files:
        with open(af) as f:
            alignment = json.load(f)
        page_name = af.stem.replace("_aligned", "")
        for pair in alignment:
            if pair.get("type") != "match":
                continue
            ch = pair.get("char", {})
            if not ch:
                continue
            crop_file = ch.get("crop_file")
            if not crop_file:
                continue
            pool.append({
                "page": page_name,
                "column": pair.get("column"),
                "syllable": pair.get("syllable", "-"),
                "nom_char": "?",        # unknown, no label yet
                "ocr_char": ch.get("ocr_char"),
                "tier": "-",
                "crop_file": crop_file,
            })
    if not pool:
        return []
    return pool if len(pool) <= n else rng.sample(pool, n)


# --------- main ---------
def main():
    rng = random.Random(RNG_SEED)
    out_dir = Path(__file__).resolve().parent / "results"
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # collect samples
    all_samples: list[tuple[str, dict]] = []
    for book in BOOKS:
        chosen = sample_from_book(book, N_PER_BOOK, rng)
        all_samples.extend((book, lab) for lab in chosen)
        print(f"  {book}: sampled {len(chosen)}/{N_PER_BOOK}")

    # process each
    records: list[dict] = []
    for idx, (book, label) in enumerate(all_samples):
        crop_file = label["crop_file"]
        raw_path = REPO / "prepared" / book / "detected" / crop_file
        if not raw_path.exists():
            print(f"  [skip] sample_{idx:03d}: {raw_path} missing", file=sys.stderr)
            continue

        raw = cv2.imread(str(raw_path))
        if raw is None:
            print(f"  [skip] sample_{idx:03d}: cannot load image", file=sys.stderr)
            continue

        sample_dir = samples_dir / f"sample_{idx:03d}"
        sample_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(sample_dir / "original.png"), raw)

        out_a = cleaner_current(raw)
        out_b = cleaner_simple(raw)
        out_c = cleaner_new4(raw)
        if out_a is not None:
            cv2.imwrite(str(sample_dir / "A_current.png"), out_a)
        if out_b is not None:
            cv2.imwrite(str(sample_dir / "B_simple_sauvola.png"), out_b)
        if out_c is not None:
            cv2.imwrite(str(sample_dir / "C_new4_adaptive.png"), out_c)

        info = {
            "idx": idx,
            "book": book,
            "page": label.get("page"),
            "column": label.get("column"),
            "syllable": label.get("syllable"),
            "nom_char": label.get("nom_char"),
            "ocr_char": label.get("ocr_char"),
            "tier": label.get("tier"),
            "crop_file": crop_file,
            "raw_shape": list(raw.shape),
            "A_ok": out_a is not None,
            "B_ok": out_b is not None,
            "C_ok": out_c is not None,
        }
        with open(sample_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        records.append(info)

    # ------ grid PNG (visual side-by-side) ------
    if records:
        _build_grid(records, samples_dir, out_dir / "grid.png")

    # ------ summary.md ------
    _build_summary(records, out_dir / "summary.md")

    print(f"\nDone. {len(records)} samples processed.")
    print(f"  Open: {out_dir / 'grid.png'}")
    print(f"  Read: {out_dir / 'summary.md'}")


def _build_grid(records: list[dict], samples_dir: Path, out_path: Path) -> None:
    """4-col grid: original | A | B | C, one row per sample."""
    cell = 96               # cell pixel size
    label_w = 240           # left side label area
    header_h = 36
    pad = 6
    n = len(records)

    width = label_w + 4 * cell
    height = header_h + n * (cell + pad) + pad
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Try Nom font for chars, fallback to default
    nom_font_path = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
    try:
        font_nom = ImageFont.truetype(str(nom_font_path), 22)
    except Exception:
        font_nom = ImageFont.load_default()
    try:
        font_h = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_s = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font_h = ImageFont.load_default()
        font_s = ImageFont.load_default()

    headers = ["original (raw)", "A: current full", "B: Sauvola only", "C: NEW4 adaptive"]
    for j, h in enumerate(headers):
        x = label_w + j * cell + 4
        draw.text((x, 8), h, fill="black", font=font_h)

    for i, info in enumerate(records):
        y = header_h + i * (cell + pad) + pad
        sample_dir = samples_dir / f"sample_{info['idx']:03d}"

        # left labels
        draw.text((6, y + 4), f"#{info['idx']:02d}  {info['nom_char']}",
                  fill="black", font=font_nom)
        draw.text((6, y + 32), f"syl={info['syllable']}", fill="#444", font=font_s)
        draw.text((6, y + 48), f"book={info['book'][:18]}", fill="#444", font=font_s)
        draw.text((6, y + 62), f"page={info['page']} col={info['column']} tier={info['tier']}",
                  fill="#888", font=font_s)
        draw.text((6, y + 78), f"ocr_char={info.get('ocr_char') or '-'}",
                  fill="#888", font=font_s)

        # 4 thumbnails
        files = [
            "original.png",
            "A_current.png",
            "B_simple_sauvola.png",
            "C_new4_adaptive.png",
        ]
        for j, fn in enumerate(files):
            p = sample_dir / fn
            x = label_w + j * cell + 4
            # cell border
            draw.rectangle([x - 1, y - 1, x + cell - 8, y + cell - 8], outline="#ddd")
            if not p.exists():
                draw.text((x + 8, y + cell // 2 - 8), "(none)", fill="red", font=font_s)
                continue
            thumb = Image.open(p).convert("L")
            thumb = thumb.resize((cell - 12, cell - 12), Image.LANCZOS)
            img.paste(thumb, (x + 2, y + 2))

    img.save(out_path)


def _build_summary(records: list[dict], out_path: Path) -> None:
    n = len(records)
    n_a = sum(1 for r in records if r["A_ok"])
    n_b = sum(1 for r in records if r["B_ok"])
    n_c = sum(1 for r in records if r["C_ok"])
    by_book: dict = {}
    for r in records:
        by_book.setdefault(r["book"], []).append(r)

    lines = [
        "# Crop cleaning A/B/C comparison",
        "",
        f"Total samples: **{n}**  (per-book: {', '.join(f'{b}={len(rs)}' for b,rs in by_book.items())})",
        "",
        "## Approaches",
        "",
        "| Code | Approach | Description |",
        "|---|---|---|",
        "| **A** | Current full | Sauvola + bg_norm + morph open + stroke_norm + CC noise + border line removal |",
        "| **B** | Sauvola only | Sauvola → trim → center 64×64. No morph / stroke / CC operations |",
        "| **C** | NEW4 paper | Adaptive Gaussian (block 21, C 8) → trim → center 64×64 |",
        "",
        "## Counts",
        "",
        f"- A produced output: {n_a}/{n}",
        f"- B produced output: {n_b}/{n}",
        f"- C produced output: {n_c}/{n}",
        "",
        "## How to evaluate",
        "",
        "Open `grid.png` — each row is one sample, columns are original / A / B / C.",
        "",
        "Look for:",
        "- **Stroke fidelity**: are thin strokes preserved? Are dots from radicals (心 ⺗, 氵 etc.) intact?",
        "- **Distortion**: does any version make the character look thicker/thinner than the original?",
        "- **Background noise**: is paper texture removed without damaging ink?",
        "- **Border lines**: are vertical/horizontal ruling lines (mép sách) removed cleanly?",
        "",
        "## Per-sample details",
        "",
        "| # | book | page | col | nom | syl | tier | OCR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['idx']:02d} | {r['book'][:18]} | {r['page']} | {r['column']} | "
            f"{r['nom_char']} | {r['syllable']} | {r['tier']} | "
            f"{r.get('ocr_char') or '-'} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
