"""Adapter ingest hai bộ IHR-NomDB có NHÃN NGƯỜI (LucVanTien1916, TruyenKieu1872) → `prepared/<book>/`.

Vì sao có adapter riêng: hai bộ này KÈM SẴN bố cục do người vẽ (VoTT `pages/bboxes.json`,
tag "Column") và QN câu-với-câu (`pages/annotation.json` → `translation`), nên KHÔNG phải
dò bố cục như thạch bản (ingest_lithograph_book đọc measure_out/<book>/layout/*) và KHÔNG
cần số câu in để neo. Đã đo (scripts/measure/ihr_layout.py, docs/CHAY_3_BO_CON_LAI_2026-09-23.md §1):

    mỗi cột = ĐÚNG MỘT CẶP LỤC BÁT 6⧺8 = 14 chữ (LVT1916 997/998 cột, TK1872 1.529/1.530);
    10 cột/trang (LVT 100/104, TK 159/162); cột 0 = PHẢI nhất ở 103/103 và 162/162 trang;
    px/chữ 37,6 (LVT) · 40,0 (TK) — nhỏ hơn thạch bản 4×, nhưng kim KHÔNG cần phóng to:
    ×1 đúng GT 97,2 % / 98,2 %; ×3 97,3 % / 98,5 % (chênh trong nhiễu) ⇒ mặc định --scale 1.

⚠️ `manifest.tsv`/`annotation.json` CÓ nhãn chữ Nôm (`nom_text` / `hn_text`). Adapter này
**KHÔNG BAO GIỜ đọc** nhãn ấy: chỉ đọc ô cột (bboxes.json) và QN (`translation`). Nhãn người
chỉ dùng SAU KHI chạy, ở `scripts/measure/ihr_endtoend_eval.py`, để ĐO độ đúng end-to-end.
Hai bộ này là **TẬP ĐÁNH GIÁ**, không trộn vào tập huấn luyện (xem `evaluation_only` trong manifest).

Đầu vào:
  data/<book>/pages/images/*.jpg       ảnh trang
  data/<book>/pages/bboxes.json        VoTT v2.2, region tag "Column"
  data/<book>/pages/annotation.json    [{img, annotations:[{hn_text, translation}]}]

Đầu ra (`--out prepared` → prepared/<book>/) — ĐÚNG hợp đồng engine như thạch bản:
  pages/page_XXXX.png · pages_denoised/page_XXXX.png
  detected/page_XXXX_ocr_cache.json    coords_space=fullpage, columns phải→trái, tier_split
  transcriptions/page_XXXX.json + .txt 10 dòng = câu lẻ ⧺ câu chẵn (14 âm)
  kim_raw/page_XXXX[_lt2].json         hộp thô kim (cache theo md5 ảnh + tham số)
  manifest.json                        cổng từng trang + ánh xạ page_XXXX ↔ page_id gốc

Chạy:
  .venv/bin/python -m pipeline.tools.ingest_ihr_book --book LucVanTien1916 --limit 5 --ocr kim
  .venv/bin/python -m pipeline.tools.ingest_ihr_book --book TruyenKieu1872 --ocr kim --qn-count-rule
Test: .venv/bin/python -m pipeline.tools.ingest_ihr_selftest
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.tools.ingest_lithograph_book import (  # noqa: E402  (lõi dùng chung)
    EXPECT_TIER, box_rect, expand_box_chars, kim_boxes, kim_cache_suffix, kim_params_of,
    book_kim_params, make_column_texts, stretch_gray, otsu_gray, _rel)

BOOKS = {
    "LucVanTien1916": dict(
        root=REPO / "data/LucVanTien1916",
        images="pages/images",
        edition="nlvnpf-0059 = R.403 Vân Tiên cổ tích tân truyện, mộc bản 1916 (IHR-NomDB)",
    ),
    "TruyenKieu1872": dict(
        root=REPO / "data/TruyenKieu1872",
        images="pages/images",
        edition="Kiều Duy Minh Thị 1872, mộc bản (IHR-NomDB)",
    ),
}
N_COLUMNS = 10                    # cột/trang kỳ vọng (đo: LVT 100/104, TK 159/162)
N_PER_COL = sum(EXPECT_TIER)      # 14
COL_TOL = 0.5                     # × bước cột: |tâm hộp kim − tâm cột| tối đa để gán
TIER_MARGIN = 0.35                # × bước chữ: biên nhận chữ ngoài dải cột theo y


# ---------------------------------------------------------------------------
# 1. Bố cục + QN (KHÔNG đọc hn_text / nom_text)
# ---------------------------------------------------------------------------
def load_bboxes(book: str) -> dict[str, dict]:
    """bboxes.json (VoTT) → {tên ảnh: {W, H, cols sắp PHẢI→TRÁI}}."""
    data = json.loads((BOOKS[book]["root"] / "pages/bboxes.json").read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for a in (data.get("assets") or {}).values():
        asset = a.get("asset") or {}
        name = asset.get("name")
        if not name:
            continue
        sz = asset.get("size") or {}
        cols = []
        for r in a.get("regions") or []:
            if "Column" not in (r.get("tags") or []):
                continue
            bb = r.get("boundingBox") or {}
            x0, y0 = float(bb["left"]), float(bb["top"])
            cols.append(dict(x0=x0, y0=y0, x1=x0 + float(bb["width"]), y1=y0 + float(bb["height"])))
        cols.sort(key=lambda c: -(c["x0"] + c["x1"]) / 2)
        out[name] = dict(W=int(sz.get("width") or 0), H=int(sz.get("height") or 0), cols=cols)
    return out


def load_annotation(book: str) -> list[dict]:
    """annotation.json → [{img, verses:[QN từng câu]}] theo thứ tự tệp; KHÔNG lấy hn_text."""
    data = json.loads((BOOKS[book]["root"] / "pages/annotation.json").read_text(encoding="utf-8"))
    out = []
    for rec in data:
        img = Path(str(rec.get("img") or "")).name
        verses = [" ".join(str(t) for t in (a.get("translation") or [])).strip()
                  for a in (rec.get("annotations") or [])]
        out.append(dict(img=img, verses=verses))
    return out


def page_seq_step(n_verses: int) -> int:
    """Bước tiến số câu của MỘT trang = 2 * ceil(số câu / 2) (2026-09-23, sửa A11/A12).

    `verse_pairs_for_page` gán cột k ↔ (first_seq + 2k, +1) nên nó GIẢ ĐỊNH `first_seq` LẺ:
    mỗi cột là một cặp lục-bát trọn vẹn. Công thức cũ `seq += len(verses)` cộng số câu THÔ,
    nên một trang có số câu LẺ (annotation thiếu câu cuối / trang bìa) đảo parity cho MỌI
    trang sau đó — đo được 61/988 cột (LucVanTien1916) và 1.078/1.610 cột (TruyenKieu1872)
    có `verse_odd` là số CHẴN, và mỗi lần đảo còn làm một số câu bị dùng hai lần (A12).
    Cộng theo SỐ CỘT (= ceil(n/2) cặp) giữ bất biến "câu lẻ luôn ở tầng trên".
    Không đổi phép ghép ô↔chữ (ghép theo trang/cột/vị trí), chỉ đổi `verse_no` công bố."""
    return 2 * math.ceil(max(int(n_verses), 0) / 2)


def load_pages(book: str) -> list[dict]:
    """Trang theo THỨ TỰ TÊN ẢNH → page_0001…; gộp ô cột + QN; đánh số câu chạy suốt sách.

    Số câu chạy suốt sách tiến theo `page_seq_step` (CẶP câu, không phải câu thô) để parity
    câu lẻ/chẵn không bị đảo sau một trang có số câu lẻ — xem `page_seq_step`."""
    bbx = load_bboxes(book)
    ann = {a["img"]: a["verses"] for a in load_annotation(book)}
    src = BOOKS[book]["root"] / BOOKS[book]["images"]
    names = sorted(p.name for p in src.glob("*.jpg"))
    recs, seq = [], 1
    for i, name in enumerate(names, start=1):
        b = bbx.get(name) or dict(W=0, H=0, cols=[])
        verses = ann.get(name) or []
        recs.append(dict(page=i, page_id=Path(name).stem, img=name, file=src / name,
                         W=b["W"], H=b["H"], cols=b["cols"], verses=verses, first_seq=seq,
                         verses_odd=bool(len(verses) % 2)))
        seq += page_seq_step(len(verses))
    return recs


def page_gate(rec: dict) -> tuple[bool, list[str]]:
    """Cổng cứng TRƯỚC khi ghi: số ô cột phải == ceil(số câu / 2) (mỗi cột = 1 cặp)."""
    flags = []
    n_cols, n_v = len(rec["cols"]), len(rec["verses"])
    want = math.ceil(n_v / 2)
    if n_cols == 0 or n_v == 0:
        flags.append(f"no_layout:cols={n_cols}:verses={n_v}")
    elif n_cols != want:
        flags.append(f"cols={n_cols}!=ceil(verses/2)={want}")
    if n_cols and n_cols != N_COLUMNS:
        flags.append(f"n_cols={n_cols}!={N_COLUMNS}")
    if n_v % 2:
        # trang có số câu LẺ: nguồn của lỗi parity cũ (page_seq_step). Nay chỉ GHI CỜ —
        # số câu của trang sau đã tiến theo cặp nên parity không còn đảo.
        flags.append(f"verses_odd={n_v}")
    ok = not any(f.startswith(("no_layout", "cols=")) for f in flags)
    return ok, flags


def verse_pairs_for_page(rec: dict) -> list[tuple[int, int]]:
    """Cột k (0-based, phải→trái) ↔ (câu lẻ, câu chẵn) = first_seq + 2k, +1."""
    n = len(rec["cols"])
    return [(rec["first_seq"] + 2 * k, rec["first_seq"] + 2 * k + 1) for k in range(n)]


def verse_rows(rec: dict) -> dict[int, dict]:
    """{verse_no: dòng QN} theo hợp đồng của make_column_texts; thiếu câu → dòng rỗng."""
    out = {}
    for j, vno in enumerate(range(rec["first_seq"], rec["first_seq"] + 2 * len(rec["cols"]))):
        text = rec["verses"][j] if j < len(rec["verses"]) else ""
        out[vno] = dict(verse_no=vno, page=rec["page_id"], line_text=text, n_syll=0,
                        anchor_source="ihr_annotation", confidence="", seq_no=vno,
                        expect_syll=EXPECT_TIER[j % 2], num_read="", margin_read="",
                        line_flag="" if text else "qn_missing",
                        page_flag="", qn_source="ihr_annotation")
    return out


# ---------------------------------------------------------------------------
# 2. Hộp kim → (cột, tầng)
# ---------------------------------------------------------------------------
def scaled_cols(rec: dict, scale: int) -> list[dict]:
    return [dict(x0=c["x0"] * scale, y0=c["y0"] * scale,
                 x1=c["x1"] * scale, y1=c["y1"] * scale) for c in rec["cols"]]


def col_tiers(col: dict) -> tuple[float, float]:
    """(y ranh giới tầng, bước chữ) của một cột: 14 chữ chia đều chiều cao ô cột,
    ranh giới sau chữ thứ 6 (câu lục 6 chữ ở TRÊN, câu bát 8 chữ ở DƯỚI)."""
    h = (col["y1"] - col["y0"]) / N_PER_COL
    return col["y0"] + h * EXPECT_TIER[0], h


def col_pitch_of(cols: list[dict]) -> float:
    xc = [(c["x0"] + c["x1"]) / 2 for c in cols]
    if len(xc) < 2:
        return float("inf")
    d = sorted(abs(xc[i] - xc[i + 1]) for i in range(len(xc) - 1))
    return d[len(d) // 2]


def assign_boxes_to_columns(boxes: list[dict], cols: list[dict]) -> tuple[list[list[dict]], dict]:
    """Chữ kim → (cột theo tâm x, tầng theo tâm y trong CHÍNH ô cột ấy). Không ép về 6/8."""
    pitch = col_pitch_of(cols)
    per: dict[tuple[int, int], list[dict]] = {}
    stats = dict(n_boxes=len(boxes), n_chars=0, n_unassigned_col=0, n_unassigned_tier=0,
                 n_chars_nonpair=0, n_number_boxes=0)
    for box in boxes:
        chars = expand_box_chars(box)
        stats["n_chars"] += len(chars)
        if not chars:
            continue
        x0, _, x1, _ = box_rect(box)
        cx = (x0 + x1) / 2
        d, k = min((abs(cx - (c["x0"] + c["x1"]) / 2), i) for i, c in enumerate(cols))
        if d > COL_TOL * pitch:
            stats["n_unassigned_col"] += len(chars)
            continue
        ysplit, chp = col_tiers(cols[k])
        for ch in chars:
            cy = ch["y_center"]
            if cy < cols[k]["y0"] - TIER_MARGIN * chp or cy > cols[k]["y1"] + TIER_MARGIN * chp:
                stats["n_unassigned_tier"] += 1
                continue
            per.setdefault((k, 0 if cy < ysplit else 1), []).append(ch)
    columns, per_col = [], []
    for k in range(len(cols)):
        top = sorted(per.get((k, 0), []), key=lambda c: c["y_center"])
        bot = sorted(per.get((k, 1), []), key=lambda c: c["y_center"])
        columns.append(top + bot)
        per_col.append((len(top), len(bot)))
    stats["chars_per_tier"] = per_col
    return columns, stats


def rebalance_tiers(columns: list[list[dict]], tier_split: list[tuple[int, int]]
                    ) -> tuple[list[tuple[int, int]], int]:
    """Cột đủ 14 chữ nhưng chia tầng ≠ 6/8 → cắt lại theo LUẬT lục bát (giữ nguyên thứ tự chữ).

    Ranh giới hình học 6/14 chiều cao ô cột chỉ đúng khi chữ dàn đều; khi khoảng cách chữ
    lệch, một chữ có thể rơi nhầm tầng. Luật 6/8 là bất biến của bản in nên thắng hình học
    (cùng lập luận book_layout.expected_tier_counts / CHOT_KENH §5.2 bước 6)."""
    out, n_fixed = [], 0
    for col, (nt, nb) in zip(columns, tier_split):
        if nt + nb == N_PER_COL and (nt, nb) != EXPECT_TIER:
            out.append(EXPECT_TIER)
            n_fixed += 1
        else:
            out.append((nt, nb))
    return out, n_fixed


# ---------------------------------------------------------------------------
# 3. Ảnh
# ---------------------------------------------------------------------------
def prepare_image(src: Path, dst: Path, contrast: str, scale: int):
    """JPG → PNG mode L (phóng ×scale nếu khai); nền kéo về 255 theo `contrast`."""
    import numpy as np
    from PIL import Image
    im = Image.open(src).convert("L")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    gray = np.asarray(im)
    if contrast == "stretch":
        out = stretch_gray(gray)
    elif contrast == "otsu":
        out = otsu_gray(gray)
    else:
        out = gray
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst, "PNG")
    return gray, out


# ---------------------------------------------------------------------------
# 4. Chạy
# ---------------------------------------------------------------------------
def ingest(book: str, pages: list[int] | None, limit: int, ocr: str, force: bool, out_root: Path,
           contrast: str, scale: int, kim: dict | None, qn_rule: bool, verbose: bool = True) -> dict:
    from core.ocr.ocr_api import _file_md5, _pixel_hash, verify_cache_image
    import cv2
    from core.image.image_processing import denoise_image

    kim = kim_params_of(kim)
    kim_sfx = kim_cache_suffix(kim)
    recs = load_pages(book)
    if pages:
        recs = [r for r in recs if r["page"] in pages]
    if limit:
        recs = recs[:limit]
    out = out_root / book
    dirs = {k: out / k for k in ("pages", "pages_denoised", "detected", "transcriptions", "kim_raw")}
    for k, d in dirs.items():
        if k != "kim_raw" or ocr == "kim":
            d.mkdir(parents=True, exist_ok=True)

    results, flagged = [], {}
    qn_fix_stats: dict = {}
    g = dict(n_pages=0, n_pages_skipped=0, n_pages_10cols=0, n_cols=0, n_cols_syll14=0,
             n_cols_kim14=0, n_cols_kim_6_8=0, n_tier_rebalanced=0, n_chars_unassigned=0,
             n_cache_ok=0, ocr_calls=0, cols_per_page={}, pages_skipped=[])
    for rec in recs:
        name = f"page_{rec['page']:04d}"
        ok, flags = page_gate(rec)
        if not ok:
            g["n_pages_skipped"] += 1
            g["pages_skipped"].append(rec["page_id"])
            flagged[name] = flags
            if verbose:
                print(f"  {name} ({rec['page_id']}): BỎ — {flags}", flush=True)
            continue
        cols = scaled_cols(rec, scale)
        vpairs = verse_pairs_for_page(rec)
        vrows = verse_rows(rec)

        png = dirs["pages"] / f"{name}.png"
        if force or not png.exists():
            prepare_image(rec["file"], png, contrast, scale)
        den = dirs["pages_denoised"] / f"{name}.png"
        if force or not den.exists():
            cv2.imwrite(str(den), denoise_image(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)))
        img_hash = _file_md5(str(png))

        stats, boxes_raw, box_source = {}, [], "layout"
        if ocr == "kim":
            raw_path = dirs["kim_raw"] / f"{name}{kim_sfx}.json"
            had = raw_path.exists() and not force
            boxes = kim_boxes(png, raw_path, img_hash, force, kim=kim)
            if not had and boxes is not None:
                g["ocr_calls"] += 1
            if boxes is None:
                flags.append("kim_failed")
                columns = [[] for _ in cols]
                stats = dict(chars_per_tier=[(0, 0)] * len(cols))
            else:
                boxes_raw = boxes
                columns, stats = assign_boxes_to_columns(boxes, cols)
                box_source = "kim"
                g["n_chars_unassigned"] += stats["n_unassigned_col"] + stats["n_unassigned_tier"]
        else:
            columns = [[] for _ in cols]
            stats = dict(chars_per_tier=[(0, 0)] * len(cols))

        tier_split, n_fix = rebalance_tiers(columns, stats["chars_per_tier"])
        g["n_tier_rebalanced"] += n_fix
        if n_fix:
            flags.append(f"tier_rebalanced={n_fix}")
        for k, (nt, nb) in enumerate(tier_split, start=1):
            if (nt, nb) == EXPECT_TIER:
                g["n_cols_kim_6_8"] += 1
            elif ocr == "kim":
                flags.append(f"col{k}:kim={nt}+{nb}!=6+8")
            if nt + nb == N_PER_COL:
                g["n_cols_kim14"] += 1

        cols_txt, tflags = make_column_texts(vpairs, vrows, qn_rule=qn_rule, fix_stats=qn_fix_stats)
        flags += tflags

        cache_path = dirs["detected"] / f"{name}_ocr_cache.json"
        cache = dict(image=_rel(png), image_hash=img_hash, pixel_hash=_pixel_hash(str(png)),
                     framed=False, frame_pad=0, coords_space="fullpage",
                     n_columns=len(columns), columns=columns, boxes_raw=boxes_raw,
                     layout="lithograph", box_source=box_source,
                     tiers=[[int(min(c["y0"] for c in cols)), int(max(col_tiers(c)[0] for c in cols))],
                            [int(min(col_tiers(c)[0] for c in cols)), int(max(c["y1"] for c in cols))]],
                     tier_split=[list(t) for t in tier_split], verse_pairs=vpairs,
                     source="ihr", page_id=rec["page_id"], scale=scale,
                     col_boxes=[[int(c["x0"]), int(c["y0"]), int(c["x1"]), int(c["y1"])] for c in cols])
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        vstat = verify_cache_image(str(cache_path), str(png))
        if vstat == "ok":
            g["n_cache_ok"] += 1
        else:
            flags.append(f"verify_cache={vstat}")

        (dirs["transcriptions"] / f"{name}.txt").write_text(
            "".join(" ".join(c["syllables"]) + "\n" for c in cols_txt), encoding="utf-8")
        (dirs["transcriptions"] / f"{name}.json").write_text(json.dumps(dict(
            book_page=rec["page"], page_id=rec["page_id"], columns=cols_txt,
            qn_line_confidences=[], qn_page_confidence=None, layout="lithograph",
            first_seq=rec["first_seq"], verse_pairs=vpairs, verse_map="ihr_annotation",
            verse_map_content=None), ensure_ascii=False, indent=1), encoding="utf-8")

        n14 = sum(1 for c in cols_txt if c["num_syllables"] == N_PER_COL)
        g["n_pages"] += 1
        g["n_pages_10cols"] += int(len(cols) == N_COLUMNS)
        g["n_cols"] += len(cols)
        g["n_cols_syll14"] += n14
        g["cols_per_page"][name] = len(cols)
        if flags:
            flagged[name] = flags
        results.append(dict(book_page=rec["page"], page_name=name, page_id=rec["page_id"],
                            source_file=rec["img"], num_columns=len(cols),
                            total_syllables=sum(c["num_syllables"] for c in cols_txt),
                            ocr_chars=sum(len(c) for c in columns), first_seq=rec["first_seq"],
                            n_cols_syll14=n14, box_source=box_source, kim_stats=stats or None,
                            flags=flags))
        if verbose:
            print(f"  {name} ({rec['page_id']}): {len(cols)} cột, syl14={n14}, "
                  f"chữ={sum(len(c) for c in columns)}, src={box_source}, cờ={len(flags)}", flush=True)

    g["qn_count_fix"] = qn_fix_stats or None
    manifest = dict(book=book, pdf=None, source="ihr-nomdb", layout="lithograph",
                    n_columns=N_COLUMNS, tiers=2, qn_per_column="couplet", contrast=contrast,
                    scale=scale, ocr=ocr, verse_map="ihr_annotation", kim_params=kim,
                    kim_cache_suffix=kim_sfx, qn_count_rule=qn_rule,
                    edition=BOOKS[book]["edition"],
                    evaluation_only=True,
                    evaluation_note="Bộ ĐỐI CHỨNG NGOÀI có nhãn người (IHR-NomDB). Nhãn pipeline sinh ra "
                                    "từ bộ này CHỈ để ĐO độ đúng end-to-end, KHÔNG trộn vào tập huấn "
                                    "luyện và KHÔNG ghi đè tệp nhãn gốc. Adapter chỉ đọc ô cột (VoTT) "
                                    "và bản dịch quốc ngữ; không đọc trường nhãn chữ Nôm nào.",
                    gt_file=str(Path("data") / book / "manifest.tsv"),
                    pages=results, total_pages=len(results),
                    total_syllables=sum(r["total_syllables"] for r in results),
                    gates=dict(**g, pages_flagged=flagged, n_pages_flagged=len(flagged)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def _parse_pages(s: str | None) -> list[int] | None:
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True, choices=sorted(BOOKS))
    ap.add_argument("--pages", default=None, help="vd 1,2,5-8 (số trang page_XXXX)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ocr", default="kim", choices=["kim", "none"])
    ap.add_argument("--out", default="prepared")
    ap.add_argument("--contrast", default="none", choices=["none", "stretch", "otsu"])
    ap.add_argument("--scale", type=int, default=1,
                    help="phóng ảnh trang ×N trước khi ghi/gọi kim (đo: ×3 không hơn ×1)")
    ap.add_argument("--qn-count-rule", action="store_true", help="sửa số đếm âm QN theo luật 6/8")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--kim-config", default=None, help="config lấy books[].kim_* (mặc định config/pipeline_<book>.yaml)")
    ap.add_argument("--kim-lang-type", type=int, default=None, choices=[0, 1, 2])
    a = ap.parse_args(argv)
    kim = book_kim_params(a.book, cli=dict(lang_type=a.kim_lang_type),
                          config=Path(a.kim_config) if a.kim_config else None)
    print(f"[ingest_ihr] {a.book} · ocr={a.ocr} · scale=×{a.scale} · contrast={a.contrast} · kim={kim}")
    m = ingest(a.book, _parse_pages(a.pages), a.limit, a.ocr, a.force, Path(a.out),
               a.contrast, a.scale, kim, a.qn_count_rule)
    gt = m["gates"]
    print(f"[ingest_ihr] {m['total_pages']} trang ghi, {gt['n_pages_skipped']} bỏ ({gt['pages_skipped']}); "
          f"{gt['n_cols']} cột, 14 âm {gt['n_cols_syll14']}, kim 6+8 {gt['n_cols_kim_6_8']}, "
          f"kim 14 {gt['n_cols_kim14']}, tầng cân lại {gt['n_tier_rebalanced']}, "
          f"lượt kim {gt['ocr_calls']}, cache ok {gt['n_cache_ok']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
