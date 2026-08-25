"""Trang ĐỐI CHIẾU NÔM ↔ HÁN — dựng để trả lời đúng MỘT câu hỏi.

CÂU HỎI
-------
Trong cùng một bộ văn bản, cùng một âm được viết vừa bằng chữ Nôm riêng vừa bằng chữ
Hán. Ví dụ âm "làm": 𡖋 (80 ô) đứng cạnh 而 (453 ô), 白 (113 ô), 以 (46 ô)…

  * nếu NÉT TRONG ẢNH của hai nhánh giống nhau  -> OCR đọc nhầm, chữ đúng là chữ Nôm,
    và 3.296 ô có đường sửa hàng loạt;
  * nếu nét KHÁC nhau  -> đó là hai chữ thật khác nhau cùng đọc một âm, giả thuyết
    OCR-thay-chữ bị bác, phải đi hướng mượn nghĩa.

Cả hai kết quả đều có giá trị. Đây là phép thử phân giải được, không phải khảo sát mơ hồ.

VÌ SAO KHÔNG DÙNG MẺ AUDIT THƯỜNG
---------------------------------
Mẻ audit hỏi "nhãn này đúng không" trên từng ô rời rạc — người chấm không thấy được
rằng 而 và 𡖋 đang tranh nhau cùng một âm. Trang này xếp hai nhánh CẠNH NHAU theo âm,
nên câu trả lời rút ra từ SO SÁNH chứ không từ trí nhớ.

Trang cũng cố ý KHÔNG giấu chữ: đây không phải phép đo precision (mù là bắt buộc), mà
là phép đối chiếu hình thái (phải thấy cả hai bên mới so được). Verdict xuất ra ở đây
KHÔNG được đưa vào bất kỳ phép tính precision nào.

    .venv/bin/python -m pipeline.ground_truth.make_lookalike_page
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from . import audit_grid
from .cli import _load_config, _paths

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GAP = REPO / "dataset_out" / "dict_gap_syllable.csv"
DEFAULT_OUT = REPO / "dataset_out" / "human_audit" / "lookalike"


def pick_samples(labels: pd.DataFrame, char: str, syl: str, n: int) -> pd.DataFrame:
    """n ô của cặp (chữ, âm), rải trên NHIỀU TRANG nhất có thể.

    Lấy n ô liền nhau trên một trang thì cả n ô cùng chịu một nét bút, một vệt mực, một
    lần quét — nhìn giống nhau là chuyện đương nhiên, không nói lên điều gì. Rải trang
    thì nét chung giữa hai nhánh mới là nét của CHỮ.
    """
    g = labels[(labels["ocr_char"] == char) & (labels["syllable"].str.lower() == syl)]
    if g.empty:
        return g
    first = g.groupby(["book", "page"], sort=False).head(1)
    if len(first) >= n:
        return first.head(n)
    return pd.concat([first, g.drop(first.index)]).head(n)


def build(gap: pd.DataFrame, labels: pd.DataFrame, paths: dict, out_dir: Path,
          per_char: int, min_rows: int) -> dict:
    fonts = audit_grid.build_font_chain(paths["font"])
    dataset_dir = Path(paths["dataset_dir"])
    prepared_dir = Path(paths["prepared_dir"])

    rival = gap[gap["nom_rival"].fillna("") != ""]
    syllables = (rival.groupby("syllable")["n_rows"].sum()
                 .sort_values(ascending=False).index.tolist())

    scans: dict[tuple[str, str], object] = {}

    def cell(row) -> str:
        crop = audit_grid._load_crop(dataset_dir, str(row["image"]))
        if crop is None:
            return ""
        ctx = ""
        key = (str(row["book"]), str(row["page"]))
        if key not in scans:
            sp = (prepared_dir / audit_grid.book_to_scan_dir(row["book"])
                  / "pages" / f"{row['page']}.png")
            try:
                from PIL import Image
                scans[key] = Image.open(sp).convert("RGB") if sp.exists() else None
            except Exception:
                scans[key] = None
        bbox = audit_grid._parse_bbox(row.get("bbox"))
        if scans[key] is not None and bbox is not None:
            c = audit_grid._context_crop(scans[key], bbox)
            if c is not None:
                ctx = audit_grid._data_uri(c)
        return ('<figure class="cell">'
                f'<img class="crop" src="{audit_grid._data_uri(crop)}">'
                + (f'<img class="ctx" src="{ctx}">' if ctx else '')
                + f'<figcaption>{html.escape(str(row["book"]))} '
                  f'{html.escape(str(row["page"]))}</figcaption></figure>')

    sections, items, n_cells = [], [], 0
    for syl in syllables:
        part = gap[gap["syllable"] == syl]
        nom = part[part["han_meaning"].fillna("") == ""]
        han = part[part["han_meaning"].fillna("") != ""]
        han = han[han["n_rows"].astype(int) >= min_rows]
        if nom.empty or han.empty:
            continue

        def branch(rows, kind: str) -> str:
            out = []
            for r in rows.itertuples():
                ref = audit_grid._reference_glyph(
                    paths["fd_dir"], f"U+{ord(r.ocr_char):04X}", fonts)
                samples = pick_samples(labels, r.ocr_char, syl, per_char)
                cells = "".join(cell(s) for _, s in samples.iterrows())
                nonlocal n_cells
                n_cells += len(samples)
                iid = f"{syl}:{r.ocr_char}"
                if kind == "han":
                    items.append(iid)
                meaning = html.escape(getattr(r, "han_meaning", "") or "")
                out.append(
                    f'<div class="branch {kind}" id="row-{html.escape(iid)}">'
                    f'<div class="head"><span class="glyph">{html.escape(r.ocr_char)}</span>'
                    + (f'<img class="ref" src="{audit_grid._data_uri(ref)}">' if ref is not None else '')
                    + f'<span class="n">{r.n_rows} ô</span>'
                    + (f'<span class="mean">{meaning}</span>' if meaning else
                       '<span class="mean nom">chữ Nôm riêng — Unihan không định nghĩa</span>')
                    + (f'<span class="vote" data-id="{html.escape(iid)}">'
                       '<button data-v="1">1 · CÙNG một chữ</button>'
                       '<button data-v="2">2 · KHÁC chữ</button>'
                       '<button data-v="3">3 · không chắc</button></span>'
                       if kind == "han" else '')
                    + '</div><div class="cells">' + cells + '</div></div>')
            return "".join(out)

        sections.append(
            f'<section><h2>âm <b>{html.escape(syl)}</b>'
            f'<span class="sum">{int(han["n_rows"].astype(int).sum())} ô ở nhánh Hán</span></h2>'
            + '<div class="lab">chữ Nôm riêng</div>' + branch(nom, "nom")
            + '<div class="lab">chữ Hán cùng âm</div>' + branch(han, "han")
            + '</section>')

    for s in scans.values():
        if s is not None:
            s.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    chars = set(gap.loc[gap["syllable"].isin(syllables), "ocr_char"])
    faces, stack = _embed_font(chars, fonts)
    (out_dir / "compare.html").write_text(
        _render(sections, items, faces, stack), encoding="utf-8")
    return {"syllables": len(sections), "han_rows_judged": len(items), "cells": n_cells}


def _embed_font(chars: set[str], fonts: list[Path]) -> tuple[str, str]:
    """Cắt gọn font chỉ chứa các chữ dùng trong trang rồi nhúng base64.

    Không nhúng thì chữ Nôm ngoài BMP (𡖋, 𦤳…) hiện thành ô vuông trên máy không có
    font Hán-Nôm — đúng lớp lỗi đã bịt ở phiếu chấm, không việc gì lặp lại ở đây. Cắt
    gọn nên chỉ tốn vài chục KB thay vì kéo theo cả font 20 MB.
    """
    import base64
    import io
    faces, names = [], []
    remaining = set(chars)
    for i, fp in enumerate(fonts):
        take = {c for c in remaining if ord(c) in audit_grid._cmap_of(fp)}
        if not take:
            continue
        remaining -= take
        try:
            from fontTools import subset
            from fontTools.ttLib import TTFont
            ft = TTFont(str(fp), fontNumber=0)
            o = subset.Options()
            o.layout_features = ["*"]
            o.notdef_outline = True
            o.drop_tables += ["FFTM"]
            s = subset.Subsetter(options=o)
            s.populate(text="".join(take))
            s.subset(ft)
            buf = io.BytesIO()
            ft.save(buf)
        except Exception:
            continue
        mime = "font/otf" if fp.suffix.lower() == ".otf" else "font/ttf"
        name = f"NomSub{i}"
        names.append(name)
        faces.append(f"@font-face {{ font-family:'{name}'; "
                     f"src:url(data:{mime};base64,"
                     f"{base64.b64encode(buf.getvalue()).decode()}); }}")
        if not remaining:
            break
    return "\n".join(faces), ", ".join(f"'{n}'" for n in names)


def _render(sections: list[str], items: list[str], faces: str = "", stack: str = "") -> str:
    ids = json.dumps(items, ensure_ascii=False)
    stack = (stack + ", ") if stack else ""
    body = "\n".join(sections)
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>Đối chiếu Nôm ↔ Hán</title>
<style>
{faces}
:root {{ --ink:#111; --mut:#667; --line:#dcdfe4; --nom:#1a6b45; --han:#a3243b; }}
body {{ font-family:-apple-system,'Helvetica Neue',Arial,sans-serif; color:var(--ink);
        margin:0; padding:24px 28px 80px; background:#fbfbfc; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
.intro {{ color:var(--mut); font-size:13px; max-width:70ch; line-height:1.5; margin:0 0 20px; }}
section {{ background:#fff; border:1px solid var(--line); border-radius:8px;
           padding:14px 16px; margin:0 0 22px; }}
h2 {{ font-size:15px; margin:0 0 10px; font-weight:600; }}
h2 b {{ font-size:20px; }}
.sum {{ color:var(--mut); font-weight:400; font-size:12px; margin-left:10px; }}
.lab {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase;
        color:var(--mut); margin:12px 0 6px; }}
.branch {{ border-left:3px solid var(--line); padding:6px 0 6px 12px; margin:0 0 10px; }}
.branch.nom {{ border-left-color:var(--nom); }}
.branch.han {{ border-left-color:var(--han); }}
.head {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:6px; }}
.glyph {{ font-size:30px; line-height:1; font-family:{stack}serif; }}
.ref {{ height:34px; width:34px; object-fit:contain; border:1px solid var(--line);
        background:#fff; }}
.n {{ font-size:12px; color:var(--mut); }}
.mean {{ font-size:12px; color:var(--mut); font-style:italic; }}
.mean.nom {{ color:var(--nom); font-style:normal; }}
.vote {{ margin-left:auto; display:flex; gap:6px; }}
.vote button {{ font:inherit; font-size:12px; padding:3px 9px; cursor:pointer;
                border:1px solid var(--line); background:#fff; border-radius:4px; }}
.vote button.sel {{ background:var(--ink); color:#fff; border-color:var(--ink); }}
.cells {{ display:flex; gap:10px; overflow-x:auto; padding-bottom:4px; }}
.cell {{ margin:0; flex:0 0 auto; text-align:center; }}
.crop {{ height:76px; display:block; border:1px solid var(--line); background:#fff; }}
.ctx {{ height:74px; display:block; margin-top:3px; opacity:.85; }}
figcaption {{ font-size:10px; color:var(--mut); margin-top:2px; }}
#bar {{ position:fixed; left:0; right:0; bottom:0; background:#fff;
        border-top:1px solid var(--line); padding:8px 28px; display:flex;
        align-items:center; gap:14px; font-size:13px; }}
#bar button {{ font:inherit; padding:5px 12px; cursor:pointer; }}
</style></head><body>
<h1>Đối chiếu Nôm ↔ Hán</h1>
<p class="intro">Mỗi mục là một âm được viết bằng <b>hai loại chữ</b> trong cùng bộ văn bản.
Hàng trên là chữ Nôm riêng, hàng dưới là các chữ Hán mà OCR đọc ra cho cùng âm đó.
So <b>nét trong ảnh</b> của hai hàng: giống nhau nghĩa là OCR đọc nhầm và chữ đúng là
chữ Nôm ở trên; khác nhau nghĩa là hai chữ thật sự khác nhau. Ảnh mỗi chữ lấy từ nhiều
trang khác nhau để nét chung không phải do cùng một vệt bút.</p>
{body}
<div id="bar"><span id="count">0/0</span>
<button id="export">Xuất verdict (.jsonl)</button>
<span style="color:#667">Verdict ở đây là ĐỐI CHIẾU HÌNH THÁI — không đưa vào phép tính precision.</span></div>
<script>
const IDS = {ids};
const KEY = "gt_lookalike_verdicts";
let store = JSON.parse(localStorage.getItem(KEY) || "{{}}");
for (const k of Object.keys(store)) if (!IDS.includes(k)) delete store[k];
const LABEL = {{1:"same_char", 2:"different_char", 3:"unsure"}};
function refresh() {{
  document.getElementById("count").textContent =
    Object.keys(store).length + "/" + IDS.length;
}}
document.querySelectorAll(".vote button").forEach(b => b.onclick = () => {{
  const id = b.parentElement.dataset.id, v = +b.dataset.v;
  store[id] = {{verdict: LABEL[v], v: v, ts: Date.now()}};
  localStorage.setItem(KEY, JSON.stringify(store));
  b.parentElement.querySelectorAll("button").forEach(x => x.classList.remove("sel"));
  b.classList.add("sel");
  refresh();
}});
for (const [id, r] of Object.entries(store)) {{
  const row = document.getElementById("row-" + id);
  if (row) {{
    const b = row.querySelector('.vote button[data-v="' + r.v + '"]');
    if (b) b.classList.add("sel");
  }}
}}
document.getElementById("export").onclick = () => {{
  const lines = Object.entries(store).map(([id, r]) => {{
    const [syllable, ocr_char] = id.split(":");
    return JSON.stringify({{syllable, ocr_char, verdict: r.verdict, ts: r.ts}});
  }}).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines + "\\n"], {{type: "application/x-ndjson"}}));
  a.download = "lookalike_verdicts.jsonl";
  a.click();
}};
refresh();
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_lookalike_page")
    ap.add_argument("--gap", default=str(DEFAULT_GAP))
    ap.add_argument("--labels", default=str(REPO / "dataset_out" / "labels_final.csv"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--per-char", type=int, default=5, help="số ảnh mẫu mỗi chữ")
    ap.add_argument("--min-rows", type=int, default=10,
                    help="bỏ nhánh Hán ít hơn ngần này ô")
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = ap.parse_args(argv)

    gap = pd.read_csv(args.gap, dtype=str).fillna("")
    if "nom_rival" not in gap.columns:
        raise SystemExit("bảng gap chưa có cột nom_rival — chạy pipeline.tools.gapcheck trước")
    gap["n_rows"] = gap["n_rows"].astype(int)
    labels = pd.read_csv(args.labels, dtype=str, low_memory=False)
    labels = labels[labels["ocr_char"].notna() & labels["image"].notna()]

    paths = _paths(_load_config(Path(args.config)))
    stat = build(gap, labels, paths, Path(args.out), args.per_char, args.min_rows)
    print(f"[đối chiếu] {stat['syllables']} âm tiết · {stat['han_rows_judged']} nhánh Hán "
          f"cần phán · {stat['cells']} ảnh")
    print(f"[đối chiếu] -> {Path(args.out) / 'compare.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
