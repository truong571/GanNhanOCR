"""Build a blinded, self-contained HTML audit tool + a separate answer-key manifest.

Each item shows the auditor exactly three things — the crop, a context view of the
scan with the bbox drawn, and the font-rendered reference glyph for the proposed
label — plus the proposed Nôm character and its dictionary readings. Everything that
could bias the auditor (tier, rule, book, S3 score, stratum, suspicion) is withheld
and written only to the manifest, which estimate.py re-joins after labelling.

Hai bộ lựa chọn, chọn qua tham số `mode` (xem CHOICE_SETS):

  mode="full" (4 mức, nguyên bản)
    1 correct        ảnh là một glyph sạch VÀ nhãn khớp
    2 wrong_label    glyph sạch NHƯNG nhãn sai ký tự
    3 wrong_image    crop cắt lỗi / dính / glyph hàng xóm
    4 unsure         không đọc được / biến thể mơ hồ

  mode="label_only" (3 mức) — NÊN DÙNG cho mọi mẻ đo precision nhãn
    1 correct  2 wrong_label  3 unsure

Vì sao tách: bộ 4 mức trộn HAI câu hỏi khác hẳn nhau vào một lần bấm. Kiểm tra lặp
2026-08-04 đo được κ=0,14 cho chiều "crop có sạch không" (người chấm tự đảo verdict
8 gắn mới / 6 gỡ bỏ trên 40 ô), trong khi chiều nhãn ổn định theo hướng không báo động
giả (0/20). Chính sự trộn lẫn làm precision GOLD dao động 95,8% ↔ 84,0% giữa hai buổi.
Chất lượng crop nay được ĐO bằng hình học ở `crop_bleed.py` thay vì bỏ phiếu.

Output HTML is a local file (no external requests); verdicts are kept in localStorage
and exported as verdicts.jsonl by the auditor.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

__all__ = ["build_audit", "book_to_scan_dir"]

_HIDDEN_FIELDS = (
    "tier", "rule", "book", "page", "column", "label", "unicode", "syllable",
    "s3_cosine", "s3_val", "stratum", "suspicion", "design_weight", "risk_reason",
    "split", "image", "image_md5", "bbox",
    # tầng của mẻ hai-tầng — người chấm KHÔNG được thấy, nếu không mất tính mù
    "audit_batch", "risk_stratum",
    # mẻ kiểm tra lặp: verdict CŨ phải vào manifest để so, nhưng lộ ra HTML thì cả mẻ
    # mất giá trị — người chấm sẽ chỉ chép lại đáp án cũ.
    "orig_verdict", "orig_batch", "orig_group",
    # 6 tín hiệu S3 corpus-wide: mang thẳng vào manifest để bước B đo AUC trên verdict
    # người mà không phải join lại (join lại dễ lệch thế hệ dữ liệu)
    "s3_head_cos", "s3_head_prob", "s3_head_margin", "s3_head_isarg",
    "s3_bank_cos", "s3_mls",
)


def book_to_scan_dir(book: str) -> str:
    """stt2 -> SachThanhTruyen2 (labels.csv book code -> prepared/ directory).

    Keys off the trailing digits, so it works for either the new 'stt2' code or
    the legacy 'yen2'."""
    m = re.search(r"(\d+)", str(book))
    if not m:
        raise ValueError(f"cannot map book code {book!r} to a scan directory")
    return f"SachThanhTruyen{m.group(1)}"


def _data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _load_crop(dataset_dir: Path, image_rel: str) -> Image.Image | None:
    p = dataset_dir / image_rel
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def _reference_glyph(
    fd_dir: Path, unicode_str: str, font: ImageFont.FreeTypeFont | None, size: int = 120
) -> Image.Image | None:
    """Font-diffusion PNG for the codepoint, else render from the fallback font."""
    try:
        cp = int(str(unicode_str).replace("U+", ""), 16)
    except Exception:
        return None
    sub = f"{cp:04X}"[:2]
    p = fd_dir / sub / f"U+{cp:04X}.png"
    if p.exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            pass
    if font is not None:
        img = Image.new("RGB", (size, size), "white")
        d = ImageDraw.Draw(img)
        ch = chr(cp)
        try:
            bb = d.textbbox((0, 0), ch, font=font)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            d.text(((size - w) / 2 - bb[0], (size - h) / 2 - bb[1]), ch,
                   fill="black", font=font)
            return img
        except Exception:
            return None
    return None


def _context_crop(
    scan: Image.Image, bbox: list[int], pad_ratio: float = 2.2, out_w: int = 260
) -> Image.Image | None:
    """Region of the scan around bbox with the bbox drawn; downscaled to out_w."""
    try:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    W, H = scan.size
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = scan.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    d = ImageDraw.Draw(crop)
    d.rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], outline=(220, 30, 30), width=3)
    if crop.width > out_w:
        scale = out_w / crop.width
        crop = crop.resize((out_w, max(1, int(crop.height * scale))))
    return crop


def _parse_bbox(v) -> list[int] | None:
    try:
        arr = json.loads(v) if isinstance(v, str) else v
        return [int(round(float(x))) for x in arr]
    except Exception:
        return None


def build_audit(
    sample: pd.DataFrame,
    dataset_dir: str | Path,
    prepared_dir: str | Path,
    fd_dir: str | Path,
    out_html: str | Path,
    out_manifest: str | Path,
    qn_dict: dict[str, list[str]] | None = None,
    font_path: str | Path | None = None,
    with_context: bool = True,
    title: str = "Audit ground-truth · Hán-Nôm",
    batch_size: int | None = 150,
    mode: str = "full",
) -> dict:
    """Render the blinded audit HTML + manifest. Returns a small stats summary.

    If batch_size is set and the sample is larger, the HTML is split into
    audit_001.html, audit_002.html, ... (each a manageable file for the browser)
    sharing the single manifest. Verdicts from every batch merge by item_id.
    """
    dataset_dir = Path(dataset_dir)
    prepared_dir = Path(prepared_dir)
    fd_dir = Path(fd_dir)
    out_html = Path(out_html)
    out_manifest = Path(out_manifest)

    font = None
    if font_path and Path(font_path).exists():
        try:
            font = ImageFont.truetype(str(font_path), 96)
        except Exception:
            font = None

    sample = sample.sort_values("audit_order")
    # open each scan page once
    scans: dict[tuple[str, str], Image.Image | None] = {}

    items: list[dict] = []
    manifest_lines: list[str] = []
    n_no_crop = n_no_ref = n_no_ctx = 0

    for _, r in sample.iterrows():
        item_id = str(r["item_id"])
        crop = _load_crop(dataset_dir, str(r["image"]))
        if crop is None:
            n_no_crop += 1
            continue
        ref = _reference_glyph(fd_dir, r.get("unicode", ""), font)
        if ref is None:
            n_no_ref += 1

        ctx_uri = ""
        if with_context:
            key = (str(r["book"]), str(r["page"]))
            if key not in scans:
                sdir = book_to_scan_dir(r["book"])
                sp = prepared_dir / sdir / "pages" / f"{r['page']}.png"
                try:
                    scans[key] = Image.open(sp).convert("RGB") if sp.exists() else None
                except Exception:
                    scans[key] = None
            scan = scans[key]
            bbox = _parse_bbox(r.get("bbox"))
            if scan is not None and bbox is not None:
                ctx = _context_crop(scan, bbox)
                if ctx is not None:
                    ctx_uri = _data_uri(ctx)
                else:
                    n_no_ctx += 1
            else:
                n_no_ctx += 1

        syl = str(r.get("syllable", "") or "")
        label = str(r.get("label", "") or "")
        cands = qn_dict.get(syl, []) if (qn_dict and syl) else []
        cand_str = " ".join(cands[:12])

        items.append({
            "id": item_id,
            "crop": _data_uri(crop),
            "ref": _data_uri(ref) if ref is not None else "",
            "ctx": ctx_uri,
            "label": label,
            "syl": syl,
            "cands": cand_str,
        })

        manifest = {"item_id": item_id}
        for f in _HIDDEN_FIELDS:
            if f in r.index:
                v = r[f]
                manifest[f] = None if pd.isna(v) else (
                    float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                    else str(v) if not isinstance(v, (str, bool, int, float)) else v
                )
        manifest_lines.append(json.dumps(manifest, ensure_ascii=False))

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # split into browser-friendly batches sharing one manifest
    all_ids = [it["id"] for it in items]
    if batch_size and len(items) > batch_size:
        n_batches = (len(items) + batch_size - 1) // batch_size
        html_files = []
        for b in range(n_batches):
            chunk = items[b * batch_size:(b + 1) * batch_size]
            fp = out_html.with_name(f"{out_html.stem}_{b + 1:03d}{out_html.suffix}")
            fp.write_text(
                _render_html(chunk, f"{title} — phần {b + 1}/{n_batches}", all_ids, mode),
                encoding="utf-8")
            html_files.append(str(fp))
        html_out = html_files
    else:
        out_html.write_text(_render_html(items, title, all_ids, mode), encoding="utf-8")
        html_out = str(out_html)

    return {
        "items": len(items),
        "skipped_no_crop": n_no_crop,
        "missing_reference_glyph": n_no_ref,
        "missing_context": n_no_ctx,
        "html": html_out,
        "manifest": str(out_manifest),
    }


# Bộ lựa chọn của công cụ chấm.
#   full        4 mức nguyên bản — TRỘN hai câu hỏi khác hẳn nhau (nhãn có đúng chữ /
#               crop có sạch) vào một lần bấm. Kiểm tra lặp 2026-08-04 đo được κ=0,14 cho
#               chiều crop, và chính sự trộn lẫn đó làm precision dao động 95,8% ↔ 84,0%.
#   label_only  CHỈ hỏi về nhãn. Chiều này đã đo được là ổn định theo hướng không báo động
#               giả (0/20 ô đúng bị gọi mới là sai). Chất lượng crop nay đo bằng hình học
#               trong `crop_bleed.py`, không bỏ phiếu nữa.
CHOICE_SETS = {
    "full": [
        (1, "correct", "đúng"),
        (2, "wrong_label", "sai nhãn"),
        (3, "wrong_image", "sai ảnh"),
        (4, "unsure", "không chắc"),
    ],
    "label_only": [
        (1, "correct", "nhãn ĐÚNG"),
        (2, "wrong_label", "nhãn SAI"),
        (3, "unsure", "không đọc được"),
    ],
}


def _batch_key(all_ids: list[str]) -> str:
    """Khoá localStorage của MỘT mẻ audit — dẫn xuất từ chính tập item của mẻ."""
    return hashlib.sha1(",".join(sorted(all_ids)).encode()).hexdigest()[:12]


def _render_html(items: list[dict], title: str, all_ids: list[str] | None = None,
                 mode: str = "full") -> str:
    data_json = json.dumps(items, ensure_ascii=False)
    ids = list(all_ids) if all_ids is not None else [it["id"] for it in items]
    all_ids_json = json.dumps(ids)
    batch_key = _batch_key(ids)
    if mode not in CHOICE_SETS:
        raise ValueError(f"mode phải thuộc {sorted(CHOICE_SETS)}, nhận {mode!r}")
    choices = CHOICE_SETS[mode]
    choices_json = json.dumps([[k, v, lbl] for k, v, lbl in choices], ensure_ascii=False)
    hint = " · ".join(f"{k} {lbl}" for k, _, lbl in choices) + " · ←/→ chuyển"
    t = html.escape(title)
    # NOTE: this is a standalone local file (not an Artifact) — inline JS is intentional.
    return f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<style>
  :root {{ --bg:#f7f6f2; --card:#fff; --ink:#222; --line:#ddd; --accent:#35558a; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Heiti SC",Arial,sans-serif; }}
  header {{ position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line);
    padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; z-index:10; }}
  header b {{ font-size:15px; }}
  .bar {{ flex:1; height:10px; background:#eee; border-radius:6px; overflow:hidden; min-width:160px; }}
  .bar > i {{ display:block; height:100%; background:var(--accent); width:0; }}
  button {{ font:inherit; padding:6px 12px; border:1px solid var(--line); background:#fff;
    border-radius:6px; cursor:pointer; }}
  button:hover {{ background:#eef; }}
  main {{ max-width:760px; margin:0 auto; padding:16px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:18px; margin-bottom:16px; }}
  .imgs {{ display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }}
  .imgs figure {{ margin:0; text-align:center; }}
  .imgs img {{ max-width:200px; max-height:230px; border:1px solid var(--line); background:#fff; }}
  figcaption {{ font-size:11px; color:#777; text-transform:uppercase; letter-spacing:.06em; margin-top:4px; }}
  .lab {{ font-size:56px; line-height:1; margin:6px 0; }}
  .meta {{ color:#555; font-size:14px; margin:4px 0 12px; }}
  .cands {{ color:#888; font-size:13px; }}
  .choices {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }}
  .choices button {{ font-size:14px; }}
  .c1 {{ border-color:#2e6e4c; }} .c1.sel {{ background:#2e6e4c; color:#fff; }}
  .c2 {{ border-color:#9a6b10; }} .c2.sel {{ background:#9a6b10; color:#fff; }}
  .c3 {{ border-color:#a03b32; }} .c3.sel {{ background:#a03b32; color:#fff; }}
  .c4 {{ border-color:#777; }}    .c4.sel {{ background:#777; color:#fff; }}
  .done {{ outline:2px solid #2e6e4c33; }}
  .idx {{ color:#aaa; font-size:12px; float:right; }}
  .hint {{ color:#888; font-size:12px; }}
</style></head>
<body>
<header>
  <b>Audit ground-truth</b>
  <span class="hint">phím {hint}</span>
  <div class="bar"><i id="prog"></i></div>
  <span id="count" style="font-variant-numeric:tabular-nums">0/0</span>
  <button id="export">Xuất verdicts.jsonl</button>
</header>
<main id="main"></main>
<script>
const ITEMS = {data_json};
// Khoá localStorage phải RIÊNG cho từng mẻ. Trước đây dùng chung hằng "gt_audit_verdicts"
// nên mẻ sau thừa hưởng verdict của mẻ trước: bản xuất kéo theo item lạ, và tệ hơn — ảnh
// nào trùng giữa hai mẻ sẽ hiện ra như ĐÃ CHẤM, người chấm bỏ qua, verdict cũ bị nhập lại
// một cách âm thầm. Khoá dẫn xuất từ chính tập item của mẻ nên mỗi mẻ một không gian riêng.
const KEY = "gt_audit_verdicts::{batch_key}";
// Danh sách id của TOÀN mẻ (mọi phần audit_001/002/...), không chỉ phần đang mở — nhờ vậy
// các phần dùng CHUNG một kho và bấm Xuất ở phần nào cũng ra đủ verdict của cả mẻ.
const _ALLOWED = new Set({all_ids_json});
const CHOICES = {choices_json};
const VLABEL = Object.fromEntries(CHOICES.map(c => [c[0], c[1]]));
const VKEYS = CHOICES.map(c => String(c[0]));
let store = JSON.parse(localStorage.getItem(KEY) || "{{}}");
for (const k of Object.keys(store)) if (!_ALLOWED.has(k)) delete store[k];

function save() {{ localStorage.setItem(KEY, JSON.stringify(store)); refresh(); }}
function refresh() {{
  const done = Object.keys(store).length, total = ITEMS.length;
  document.getElementById("count").textContent = done + "/" + total;
  document.getElementById("prog").style.width = (total? 100*done/total : 0) + "%";
}}
function choose(id, v) {{
  store[id] = {{verdict: VLABEL[v], v: v, ts: Date.now()}};
  const card = document.getElementById("card-"+id);
  card.querySelectorAll(".choices button").forEach(b=>b.classList.remove("sel"));
  card.querySelector(".c"+v).classList.add("sel");
  card.classList.add("done");
  save();
}}
function render() {{
  const m = document.getElementById("main");
  ITEMS.forEach((it, i) => {{
    const d = document.createElement("div");
    d.className = "card"; d.id = "card-"+it.id;
    const prev = store[it.id];
    d.innerHTML =
      '<span class="idx">#'+(i+1)+'</span>' +
      '<div class="imgs">' +
        '<figure><img src="'+it.crop+'"><figcaption>crop được gán</figcaption></figure>' +
        (it.ctx? '<figure><img src="'+it.ctx+'"><figcaption>vị trí trên scan</figcaption></figure>':'') +
        (it.ref? '<figure><img src="'+it.ref+'"><figcaption>glyph tham chiếu</figcaption></figure>':'') +
      '</div>' +
      '<div class="lab">'+ (it.label||'?') +'</div>' +
      '<div class="meta">âm: <b>'+ (it.syl||'—') +'</b></div>' +
      (it.cands? '<div class="cands">ứng viên: '+it.cands+'</div>':'') +
      '<div class="choices">' +
        '<button class="c1'+(prev&&prev.v==1?" sel":"")+'" onclick="choose(\\''+it.id+'\\',1)">1 · đúng</button>' +
        '<button class="c2'+(prev&&prev.v==2?" sel":"")+'" onclick="choose(\\''+it.id+'\\',2)">2 · sai nhãn</button>' +
        '<button class="c3'+(prev&&prev.v==3?" sel":"")+'" onclick="choose(\\''+it.id+'\\',3)">3 · sai ảnh</button>' +
        '<button class="c4'+(prev&&prev.v==4?" sel":"")+'" onclick="choose(\\''+it.id+'\\',4)">4 · không chắc</button>' +
      '</div>';
    if (prev) d.classList.add("done");
    m.appendChild(d);
  }});
}}
let cur = 0;
function focusCard(i) {{
  cur = Math.max(0, Math.min(ITEMS.length-1, i));
  const el = document.getElementById("card-"+ITEMS[cur].id);
  if (el) el.scrollIntoView({{behavior:"smooth", block:"center"}});
}}
document.addEventListener("keydown", e => {{
  if (VKEYS.includes(e.key)) {{ choose(ITEMS[cur].id, +e.key); focusCard(cur+1); }}
  else if (e.key === "ArrowRight") focusCard(cur+1);
  else if (e.key === "ArrowLeft") focusCard(cur-1);
}});
document.getElementById("export").onclick = () => {{
  const lines = Object.entries(store).map(([id,v]) =>
    JSON.stringify({{item_id:id, verdict:v.verdict, ts:v.ts}}));
  const blob = new Blob([lines.join("\\n")+"\\n"], {{type:"application/x-ndjson"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "verdicts.jsonl"; a.click();
}};
render(); refresh();
</script>
</body></html>
"""
