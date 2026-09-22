"""Dựng bộ mẫu NGƯỜI KIỂM MÙ cho một sách bất kỳ (LVT1883, KVK1884, STT…).

Khác `pipeline.ground_truth grid` (hiện thẳng nhãn + glyph tham chiếu, cần cột
`stratum`/S3 của STT): ở đây người kiểm KHÔNG thấy nhãn máy. Mỗi ô hiện crop (phóng
`--zoom`×), ảnh ngữ cảnh cột (khung đỏ trên trang), âm Quốc ngữ và 3–5 ứng viên Nôm
lấy từ `dict/QuocNgu_SinoNom.csv` cho âm đó, xáo trộn; nhãn máy nằm trong đó nhưng
không được đánh dấu. Người kiểm chọn: một ứng viên / "không có trong danh sách" /
"crop sai chữ" / "không chắc". Verdict lưu localStorage, xuất CSV bằng JS thuần.

Đầu vào : <dataset>/labels.csv (+ labels_trace.csv nếu có, để lấy nom_idx) và crops
          <dataset>/<tier>/…png ; ảnh trang <pages>/page_XXXX.png (tuỳ chọn, cho ngữ cảnh).
Đầu ra  : <out>/index.html      lưới mù (không chứa tier/nhãn máy/đường dẫn gốc)
          <out>/items.csv       KHOÁ — có machine_label, tier: người kiểm KHÔNG mở
          <out>/img/            crop + ngữ cảnh + glyph ứng viên (tên trung tính K001…)
          <out>/README.md       cách chấm + cách chạy score

Lấy mẫu: ngẫu nhiên phân tầng theo TRANG, phân bổ tỷ lệ (largest remainder) → tự trọng
số, nên tỷ lệ đúng trên mẫu là ước lượng không chệch cho tổng thể tier. Seed cố định.

Chạy (từ gốc repo):
  .venv/bin/python pipeline/tools/kiem_nguoi_grid.py --dataset dataset_LucVanTien1883 \
      --pages prepared/LucVanTien1883/pages --seed 20260921 --n-gold 300 --n-second 100
Chấm xong → pipeline/tools/kiem_nguoi_score.py --items <out>/items.csv --verdicts <csv>
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.text.text_utils import normalize_tone_marks  # noqa: E402
from pipeline.ground_truth.audit_grid import (  # noqa: E402
    _font_for,
    _parse_bbox,
    build_font_chain,
)

SPECIAL = {"KHONG_CO": "Không có trong danh sách",
           "CROP_SAI": "Crop sai chữ (cắt lệch / dính / không phải chữ)",
           "KHONG_CHAC": "Không chắc"}


# --------------------------------------------------------------------------- helpers
def syl_key(s: str) -> str:
    return normalize_tone_marks(unicodedata.normalize("NFC", (s or "").strip().lower()))


def load_dict(path: Path) -> dict[str, list[str]]:
    d: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0].strip().lower() == "quocngu":
                continue
            k, v = syl_key(row[0]), row[1].strip()
            # bỏ chữ PUA (U+E000–F8FF, 287 dòng trong dict) / chuỗi nhiều chữ: người không đọc được
            if is_cjk(v) and v not in d[k]:
                d[k].append(v)
    return d


def load_similar(path: Path | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not path or not path.exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if len(row) < 2:
                continue
            try:
                out[row[0].strip()] = [c for c in json.loads(row[1].replace("'", '"')) if c]
            except Exception:
                continue
    return out


def is_cjk(ch: str) -> bool:
    if len(ch) != 1:
        return False
    cp = ord(ch)
    return (0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF
            or 0x20000 <= cp <= 0x3134F)


def stratified_by_page(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Phân bổ tỷ lệ theo trang (largest remainder), rút SRS trong từng trang."""
    if n <= 0 or not rows:
        return []
    n = min(n, len(rows))
    by_page: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_page[r["page"]].append(r)
    pages = sorted(by_page)
    total = len(rows)
    quota = {p: n * len(by_page[p]) / total for p in pages}
    alloc = {p: int(quota[p]) for p in pages}
    rem = n - sum(alloc.values())
    order = sorted(pages, key=lambda p: (-(quota[p] - alloc[p]), rng.random()))
    for p in order[:rem]:
        alloc[p] += 1
    out = []
    for p in pages:
        pool = sorted(by_page[p], key=lambda r: r["image"])
        k = min(alloc[p], len(pool))
        out.extend(rng.sample(pool, k))
    return out


def pick_candidates(machine: str, pool: list[str], similar: dict[str, list[str]],
                    all_chars: list[str], n_cands: int, rng: random.Random) -> tuple[list[str], dict]:
    """Ứng viên = nhãn máy (nếu là 1 chữ) + chữ cùng âm trong từ điển; đệm bằng chữ giống
    hình (SinoNom_Similar) rồi chữ ngẫu nhiên nếu vẫn < 3. Trả (danh sách đã xáo, meta)."""
    cands: list[str] = []
    if is_cjk(machine):
        cands.append(machine)
    others = [c for c in pool if c not in cands]
    n_take = min(len(others), max(0, n_cands - len(cands)))
    cands += rng.sample(others, n_take)
    n_from_dict = len(cands)
    src = {c: "dict" for c in cands}
    if is_cjk(machine):
        src[machine] = "machine"
    pad_sim = pad_rand = 0
    if len(cands) < 3:
        for c in similar.get(machine, []):
            if len(cands) >= 3:
                break
            if is_cjk(c) and c not in cands:
                cands.append(c); src[c] = "similar"; pad_sim += 1
    while len(cands) < 3 and all_chars:
        c = rng.choice(all_chars)
        if c not in cands:
            cands.append(c); src[c] = "random"; pad_rand += 1
    rng.shuffle(cands)
    meta = {"n_dict_cands": len(pool), "truncated": int(len(others) > n_take),
            "pad_similar": pad_sim, "pad_random": pad_rand,
            "sources": "|".join(src[c] for c in cands), "n_from_dict": n_from_dict}
    return cands, meta


def context_image(page_png: Path, bbox: list[int], out_w: int = 240,
                  pad_x: float = 1.1, pad_y: float = 2.4) -> Image.Image | None:
    try:
        scan = Image.open(page_png)
    except Exception:
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    W, H = scan.size
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = max(0, int(x1 - bw * pad_x)), max(0, int(y1 - bh * pad_y))
    cx2, cy2 = min(W, int(x2 + bw * pad_x)), min(H, int(y2 + bh * pad_y))
    crop = scan.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    ImageDraw.Draw(crop).rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
                                   outline=(220, 30, 30), width=3)
    if crop.width > out_w:
        s = out_w / crop.width
        crop = crop.resize((out_w, max(1, int(crop.height * s))))
    return crop


def glyph_png(ch: str, fonts: list[Path], out: Path, size: int = 72) -> bool:
    if out.exists():
        return True
    font = _font_for(ord(ch), fonts, int(size * 0.8))
    if font is None:
        return False
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    bb = d.textbbox((0, 0), ch, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((size - w) / 2 - bb[0], (size - h) / 2 - bb[1]), ch, fill="black", font=font)
    img.save(out)
    return True


def guess_pages_dir(book: str) -> Path | None:
    prep = REPO / "prepared"
    if not prep.exists():
        return None
    for d in prep.iterdir():
        if d.is_dir() and d.name.lower() == book.lower() and (d / "pages").exists():
            return d / "pages"
    return None


# --------------------------------------------------------------------------- HTML
INSTRUCTIONS = [
    "1. Nhìn ẢNH CROP (trái, phóng to) và ảnh NGỮ CẢNH (khung đỏ trên trang) để biết ô đang chấm là chữ nào trên bản gốc.",
    "2. Đọc ÂM Quốc ngữ được ghép cho ô này, rồi xem các ứng viên chữ Nôm bên dưới (thứ tự ngẫu nhiên, không ứng viên nào được ưu tiên).",
    "3. Bấm ứng viên KHỚP ĐÚNG với chữ trong crop. Nếu chữ trong crop không nằm trong danh sách → bấm \"Không có trong danh sách\".",
    "4. Nếu crop cắt lệch, dính 2 chữ, cắt nửa chữ hoặc không phải chữ (vết mực, khung) → bấm \"Crop sai chữ\". Không đọc được → \"Không chắc\".",
    "5. Kết quả tự lưu trong trình duyệt; chấm xong bấm \"Xuất CSV\" và gửi tệp. Không mở items.csv trong lúc chấm.",
]


def render_html(items: list[dict], title: str, batch_key: str, zoom: int) -> str:
    cards = []
    for it in items:
        cand_html = []
        for i, c in enumerate(it["candidates"], 1):
            g = it["glyph_files"][i - 1]
            gimg = f'<img class="g" src="{g}" alt="">' if g else '<span class="g noglyph">?</span>'
            cand_html.append(
                f'<button class="cand" data-v="{html.escape(c)}" data-i="{i}" title="ứng viên {i}">'
                f'<span class="k">{i}</span>{gimg}<span class="ch">{html.escape(c)} '
                f'<small>U+{ord(c):04X}</small></span></button>')
        spec = "".join(
            f'<button class="cand sp" data-v="{code}" data-i="{k}">'
            f'<span class="k">{k}</span>{html.escape(lbl)}</button>'
            for k, (code, lbl) in zip(("0", "9", "8"), SPECIAL.items()))
        ctx = (f'<img class="ctx" loading="lazy" src="{it["ctx_file"]}" alt="ngữ cảnh">'
               if it.get("ctx_file") else '<div class="ctx none">không có ảnh trang</div>')
        # phóng `zoom`× nhưng không vượt 560 px cao / 460 px rộng (giữ tỷ lệ)
        sc = min(zoom, 560 / max(1, it["crop_h"]), 460 / max(1, it["crop_w"]))
        w, h = int(round(it["crop_w"] * sc)), int(round(it["crop_h"] * sc))
        cards.append(f'''
<section class="card" id="{it["item_id"]}" data-id="{it["item_id"]}">
 <header><b>{it["item_id"]}</b> <span class="st"></span></header>
 <img class="crop" loading="lazy" src="{it["img_file"]}" width="{w}" height="{h}" alt="crop">
 {ctx}
 <div class="side">
  <div class="syl">âm: <b>{html.escape(it["syllable"])}</b></div>
  <div class="cands">{"".join(cand_html)}</div>
  <div class="specials">{spec}</div>
 </div>
</section>''')
    n = len(items)
    ids_json = json.dumps([it["item_id"] for it in items])
    instr = "".join(f"<li>{html.escape(s[3:])}</li>" for s in INSTRUCTIONS)
    return f'''<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--bg:#f6f4ee;--card:#fff;--ink:#222;--acc:#1f6f8b;--ok:#2e7d32;--warn:#b26a00;--bad:#b3261e;--line:#ddd}}
body{{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}}
.top{{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;z-index:5}}
.top h1{{font-size:18px;margin:0 0 6px}}
.top ol{{margin:0 0 8px;padding-left:22px;font-size:13.5px;line-height:1.45}}
.bar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:14px}}
.bar input{{padding:4px 6px;border:1px solid var(--line);border-radius:4px}}
.bar button{{padding:5px 10px;border:1px solid var(--acc);background:var(--acc);color:#fff;border-radius:4px;cursor:pointer}}
.bar button.sec{{background:#fff;color:var(--acc)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,1000px),1fr));gap:14px;padding:14px 16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;scroll-margin-top:190px;
 display:grid;grid-template-columns:auto 250px minmax(260px,1fr);grid-template-areas:"h h h" "crop ctx side";gap:6px 14px;align-items:start}}
.card header{{grid-area:h}} .card .crop{{grid-area:crop}} .card .ctx{{grid-area:ctx}} .card .side{{grid-area:side}}
.card.active{{outline:2px solid var(--acc)}}
.card.done{{border-color:var(--ok)}}
.card header{{display:flex;justify-content:space-between;font-size:13px;color:#555;margin-bottom:6px}}
.crop{{border:1px solid #bbb;image-rendering:auto;max-width:100%;height:auto;background:#fff}}
.ctx{{border:1px solid #bbb;max-width:240px;height:auto}}
.ctx.none{{width:240px;padding:20px 0;text-align:center;color:#888;font-size:12px}}
.syl{{margin:0 0 8px;font-size:22px}}
.cands{{display:flex;flex-direction:column;gap:6px}}
.cand{{display:flex;align-items:center;gap:6px;border:1px solid #bbb;background:#fafafa;border-radius:6px;padding:4px 8px;cursor:pointer;font-size:15px}}
.cand .g{{width:56px;height:56px;border:1px solid #eee;background:#fff}}
.cand .noglyph{{display:inline-flex;align-items:center;justify-content:center;color:#999}}
.cand .ch{{font-family:"Nom Na Tong","HAN NOM A","HAN NOM B","HanaMinA","HanaMinB","Noto Serif CJK TC","Songti TC",serif;font-size:26px}}
.cand .ch small{{font-size:10px;color:#777;font-family:system-ui,sans-serif}}
.cand .k{{font-size:11px;color:#777;border:1px solid #ccc;border-radius:3px;padding:0 4px}}
.cand.sel{{border-color:var(--acc);background:#e3f1f6;box-shadow:0 0 0 2px var(--acc) inset}}
.specials{{margin-top:8px;display:flex;flex-wrap:wrap;gap:8px}}
.specials .cand{{font-size:13px;background:#fff}}
.specials .cand.sel{{background:#fff3e0;border-color:var(--warn);box-shadow:0 0 0 2px var(--warn) inset}}
.hidden{{display:none}}
@media (max-width:900px){{.card{{grid-template-columns:1fr 1fr;grid-template-areas:"h h" "crop ctx" "side side"}}}}
@media (max-width:640px){{.grid{{grid-template-columns:1fr;padding:10px}}.card{{grid-template-columns:1fr;grid-template-areas:"h" "crop" "ctx" "side"}}.crop{{max-width:100%}}}}
</style></head>
<body>
<div class="top">
 <h1>{html.escape(title)} — {n} ô</h1>
 <ol>{instr}</ol>
 <div class="bar">
  <label>Người kiểm: <input id="who" placeholder="tên/viết tắt" size="12"></label>
  <span id="prog"></span>
  <button id="exp">Xuất CSV</button>
  <button id="imp" class="sec">Nhập CSV đã chấm</button><input type="file" id="impf" accept=".csv" class="hidden">
  <button id="onlytodo" class="sec">Chỉ hiện ô chưa chấm</button>
  <button id="next" class="sec">Tới ô chưa chấm kế</button>
  <span style="color:#777;font-size:12px">Phím: 1–5 chọn ứng viên · 0 không có · 9 crop sai · 8 không chắc · N ô kế</span>
 </div>
</div>
<main class="grid">{"".join(cards)}</main>
<script>
(function(){{
const KEY='kiem_nguoi:{batch_key}:';const IDS={ids_json};const N=IDS.length;
const $=(s,r=document)=>r.querySelector(s);const $$=(s,r=document)=>Array.from(r.querySelectorAll(s));
let active=null,onlyTodo=false;
function get(id){{try{{const s=localStorage.getItem(KEY+id);return s?JSON.parse(s):null}}catch(e){{return null}}}}
function set(id,obj){{try{{localStorage.setItem(KEY+id,JSON.stringify(obj))}}catch(e){{}}}}
function who(){{return ($('#who').value||'').trim()}}
try{{$('#who').value=localStorage.getItem(KEY+'__who')||''}}catch(e){{}}
$('#who').addEventListener('change',()=>{{try{{localStorage.setItem(KEY+'__who',who())}}catch(e){{}}}});
function paint(card){{const id=card.dataset.id,v=get(id);
 $$('.cand',card).forEach(b=>b.classList.toggle('sel',!!v&&b.dataset.v===v.verdict));
 card.classList.toggle('done',!!v);$('.st',card).textContent=v?('đã chấm '+(v.verdict.length===1?'ứng viên '+v.choice_index:v.verdict)):'chưa chấm';
 if(onlyTodo)card.classList.toggle('hidden',!!v);}}
function prog(){{const d=IDS.filter(i=>get(i)).length;$('#prog').textContent='đã chấm '+d+'/'+N;}}
function choose(card,btn){{const id=card.dataset.id;const cur=get(id);
 if(cur&&cur.verdict===btn.dataset.v){{try{{localStorage.removeItem(KEY+id)}}catch(e){{}}}}
 else set(id,{{verdict:btn.dataset.v,choice_index:btn.dataset.i,reviewer:who(),ts:new Date().toISOString()}});
 paint(card);prog();}}
$$('.card').forEach(card=>{{paint(card);
 card.addEventListener('click',e=>{{setActive(card);const b=e.target.closest('.cand');if(b)choose(card,b);}});}});
function setActive(card){{if(active)active.classList.remove('active');active=card;if(card)card.classList.add('active');}}
function nextTodo(){{const from=active?IDS.indexOf(active.dataset.id):-1;
 for(let k=1;k<=N;k++){{const id=IDS[(from+k)%N];if(!get(id)){{const c=document.getElementById(id);setActive(c);c.scrollIntoView({{behavior:'smooth',block:'start'}});return;}}}}
 alert('Đã chấm hết '+N+' ô. Bấm Xuất CSV.');}}
document.addEventListener('keydown',e=>{{if(e.target.tagName==='INPUT')return;
 if(e.key==='n'||e.key==='N'){{nextTodo();return;}}
 if(!active)return;const b=$$('.cand',active).find(x=>x.dataset.i===e.key);if(b){{choose(active,b);e.preventDefault();}}}});
$('#next').addEventListener('click',nextTodo);
$('#onlytodo').addEventListener('click',()=>{{onlyTodo=!onlyTodo;$('#onlytodo').textContent=onlyTodo?'Hiện tất cả':'Chỉ hiện ô chưa chấm';$$('.card').forEach(c=>c.classList.toggle('hidden',onlyTodo&&!!get(c.dataset.id)));}});
function csvq(s){{s=String(s==null?'':s);return /[",\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}}
$('#exp').addEventListener('click',()=>{{const rows=[['item_id','verdict','choice_index','reviewer','ts']];
 IDS.forEach(id=>{{const v=get(id);if(v)rows.push([id,v.verdict,v.choice_index,v.reviewer||who(),v.ts]);}});
 const blob=new Blob(['\\ufeff'+rows.map(r=>r.map(csvq).join(',')).join('\\n')],{{type:'text/csv;charset=utf-8'}});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);
 a.download='verdicts_'+(who()||'nguoikiem')+'_'+new Date().toISOString().slice(0,10)+'.csv';a.click();}});
$('#imp').addEventListener('click',()=>$('#impf').click());
$('#impf').addEventListener('change',ev=>{{const f=ev.target.files[0];if(!f)return;const rd=new FileReader();
 rd.onload=()=>{{const lines=rd.result.replace(/^\\ufeff/,'').split(/\\r?\\n/).filter(Boolean);let n=0;
  for(const ln of lines.slice(1)){{const m=ln.match(/^([^,]*),("(?:[^"]|"")*"|[^,]*),([^,]*),([^,]*),(.*)$/);if(!m)continue;
   const id=m[1],verdict=m[2].replace(/^"|"$/g,'').replace(/""/g,'"');if(IDS.indexOf(id)<0)continue;
   set(id,{{verdict:verdict,choice_index:m[3],reviewer:m[4],ts:m[5]}});n++;}}
  $$('.card').forEach(paint);prog();alert('Đã nhập '+n+' verdict.');}};rd.readAsText(f,'utf-8');}});
prog();
}})();
</script>
</body></html>'''


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="thư mục có labels.csv + crops (vd dataset_LucVanTien1883)")
    ap.add_argument("--pages", default=None, help="thư mục ảnh trang page_XXXX.png (mặc định: prepared/<book>/pages)")
    ap.add_argument("--dict", default=str(REPO / "dict" / "QuocNgu_SinoNom.csv"))
    ap.add_argument("--similar", default=str(REPO / "dict" / "SinoNom_Similar.csv"))
    ap.add_argument("--font", default=str(REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"))
    ap.add_argument("--out", default=None, help="mặc định <dataset>/kiem_nguoi")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--n-gold", type=int, default=300)
    ap.add_argument("--n-second", type=int, default=100)
    ap.add_argument("--second-tier", default="auto", help="auto|REVIEW|SYLLABLE|NONE (auto: REVIEW nếu có crop, không thì SYLLABLE)")
    ap.add_argument("--n-cands", type=int, default=5, help="số ứng viên tối đa (3–5)")
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)

    ds = Path(a.dataset).resolve()
    labels_csv = ds / "labels.csv"
    if not labels_csv.exists():
        sys.exit(f"không thấy {labels_csv}")
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8", newline="")))
    if not rows:
        sys.exit("labels.csv rỗng")
    for r in rows:
        r["_crop"] = ds / r["image"] if r.get("image") else None
        r["_has_crop"] = bool(r["_crop"] and r["_crop"].exists())
    book = rows[0].get("book", ds.name)
    trace = {}
    if (ds / "labels_trace.csv").exists():
        trace = {t["image"]: t for t in csv.DictReader(open(ds / "labels_trace.csv", encoding="utf-8", newline=""))}

    tiers = Counter(r["tier"] for r in rows)
    gold = [r for r in rows if r["tier"] == "GOLD" and r["_has_crop"]]
    second = a.second_tier.upper()
    if second == "AUTO":
        rv = [r for r in rows if r["tier"] == "REVIEW" and r["_has_crop"]]
        second = "REVIEW" if len(rv) >= max(10, a.n_second // 2) else "SYLLABLE"
    second_rows = [] if second == "NONE" else [r for r in rows if r["tier"] == second and r["_has_crop"]]

    rng = random.Random(a.seed)
    sample = [(r, "GOLD") for r in stratified_by_page(gold, a.n_gold, rng)]
    sample += [(r, second) for r in stratified_by_page(second_rows, a.n_second, rng)]
    if not sample:
        sys.exit("không rút được ô nào (thiếu crop?)")
    rng.shuffle(sample)  # trộn tier để mù

    qn_dict = load_dict(Path(a.dict))
    all_chars = sorted({c for v in qn_dict.values() for c in v if is_cjk(c)})
    similar = load_similar(Path(a.similar))
    fonts = build_font_chain(a.font)
    pages_dir = Path(a.pages) if a.pages else guess_pages_dir(book)
    if pages_dir and not pages_dir.exists():
        pages_dir = None

    out = Path(a.out) if a.out else ds / "kiem_nguoi"
    img_dir = out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    items, key_rows = [], []
    n_ctx = n_noglyph = 0
    for k, (r, tier) in enumerate(sample, 1):
        item_id = f"K{k:03d}"
        machine = r["label"] if tier == "GOLD" else r.get("ocr_char", "")
        machine_kind = "label" if tier == "GOLD" else "ocr_char"
        pool = qn_dict.get(syl_key(r["syllable"]), [])
        irng = random.Random(f"{a.seed}:{r['image']}")
        cands, meta = pick_candidates(machine, pool, similar, all_chars, a.n_cands, irng)
        # crop
        crop_file = f"img/{item_id}.png"
        shutil.copyfile(r["_crop"], out / crop_file)
        with Image.open(r["_crop"]) as im:
            cw, ch = im.size
        # ngữ cảnh
        ctx_file = None
        bbox = _parse_bbox(r.get("bbox"))
        if pages_dir and bbox:
            pp = pages_dir / f"{r['page']}.png"
            if pp.exists():
                ci = context_image(pp, bbox)
                if ci is not None:
                    ctx_file = f"img/{item_id}_ctx.jpg"
                    ci.save(out / ctx_file, quality=82)
                    n_ctx += 1
        # glyph ứng viên
        gfiles = []
        for c in cands:
            gf = f"img/g_U{ord(c):04X}.png"
            if glyph_png(c, fonts, out / gf):
                gfiles.append(gf)
            else:
                gfiles.append(None); n_noglyph += 1
        tr = trace.get(r["image"], {})
        nom_idx = tr.get("nom_idx", "")
        items.append({"item_id": item_id, "img_file": crop_file, "ctx_file": ctx_file, "crop_w": cw,
                      "crop_h": ch, "syllable": r["syllable"], "candidates": cands, "glyph_files": gfiles})
        key_rows.append({
            "item_id": item_id, "image": r["image"], "book": book, "page": r["page"], "column": r["column"],
            "nom_idx": nom_idx, "syllable": r["syllable"], "tier": tier, "rule": r.get("rule", ""),
            "machine_label": machine, "machine_label_kind": machine_kind,
            "machine_in_candidates": int(machine in cands),
            "machine_in_dict": int(machine in pool),  # SYLLABLE: chữ OCR có cùng âm trong từ điển?
            "candidates": "|".join(cands), "cand_sources": meta["sources"],
            "n_dict_cands": meta["n_dict_cands"], "cands_truncated": meta["truncated"],
            "pad_similar": meta["pad_similar"], "pad_random": meta["pad_random"],
            "bbox": r.get("bbox", ""), "image_md5": r.get("image_md5", ""), "has_ctx": int(ctx_file is not None),
        })

    title = a.title or f"Kiểm mù nhãn Nôm — {book}"
    batch_key = f"{book}-{a.seed}-{len(items)}"
    (out / "index.html").write_text(render_html(items, title, batch_key, a.zoom), encoding="utf-8")
    with open(out / "items.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader(); w.writerows(key_rows)

    n_by_tier = Counter(k["tier"] for k in key_rows)
    pages_by_tier = {t: len({k["page"] for k in key_rows if k["tier"] == t}) for t in n_by_tier}
    summary = {
        "book": book, "seed": a.seed, "dataset": str(ds), "pages_dir": str(pages_dir) if pages_dir else None,
        "population": dict(tiers), "population_with_crop": {"GOLD": len(gold), second: len(second_rows)},
        "second_tier": second, "sampled": dict(n_by_tier), "pages_covered": pages_by_tier,
        "n_items": len(items), "n_with_context": n_ctx, "n_candidates_without_glyph": n_noglyph,
        "machine_in_candidates": sum(k["machine_in_candidates"] for k in key_rows),
        "cands_truncated": sum(k["cands_truncated"] for k in key_rows),
        "n_cands_dist": dict(Counter(len(it["candidates"]) for it in items)),
        "design": "stratified by page, proportional allocation (largest remainder), SRS within page; display order shuffled",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "README.md").write_text(readme(summary, out), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("book", "second_tier", "sampled", "pages_covered", "n_with_context",
                                               "machine_in_candidates", "cands_truncated", "n_cands_dist",
                                               "n_candidates_without_glyph")}, ensure_ascii=False))
    print(f"→ {out / 'index.html'}\n→ {out / 'items.csv'} (KHOÁ, không đưa người kiểm)")
    return 0


def readme(s: dict, out: Path) -> str:
    return f"""# Bộ mẫu người kiểm mù — {s['book']}

- Sinh bởi `pipeline/tools/kiem_nguoi_grid.py`, seed {s['seed']}, lấy mẫu phân tầng theo trang
  (tỷ lệ, tự trọng số). Mẫu: {s['sampled']} trên tổng thể có crop {s['population_with_crop']}.
- `index.html` — mở bằng trình duyệt (không cần server). Người kiểm chỉ nhận thư mục này
  **trừ `items.csv`/`summary.json`** (khoá có nhãn máy + tier).
- Cách chấm: xem 5 dòng hướng dẫn đầu trang. Mỗi ô chọn 1 trong: ứng viên Nôm / Không có
  trong danh sách / Crop sai chữ / Không chắc. Bấm lại để bỏ chọn. Phím 1–5, 0, 9, 8, N.
- Xuất: nút **Xuất CSV** → `verdicts_<tên>_<ngày>.csv` (item_id, verdict, choice_index, reviewer, ts).
  Verdict = chữ Nôm đã chọn hoặc mã `KHONG_CO` / `CROP_SAI` / `KHONG_CHAC`.
- Chấm điểm: `.venv/bin/python pipeline/tools/kiem_nguoi_score.py --items {out.name}/items.csv --verdicts <csv|thư mục>`
  → precision GOLD + Wilson 95 % CI, tách theo lý do sai; tier {s['second_tier']}: tỷ lệ xác nhận âm.
- Giới hạn: chỉ hiện tối đa 5 ứng viên (ô bị cắt bớt: {s['cands_truncated']}/{s['n_items']}) — chữ đúng
  có thể nằm ngoài danh sách; khi đó "Không có trong danh sách" vẫn tính là nhãn máy SAI (đúng cho precision),
  nhưng không định vị được chữ đúng.
"""


if __name__ == "__main__":
    sys.exit(main())
