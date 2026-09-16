"""K5 · 20 crop ngẫu nhiên qua cổng B-3 (p ≥ 0,9) → KhoiB/v3/b3_sample.html để người nhìn nhanh (không chấm, không đổi nhãn).

Crop cắt từ prepared/<sách>/pages/<page>.png theo bbox v3 (đệm 12% như build) + dải ngữ cảnh ±1 ô (bbox ô kề cùng cột).
Seed 2026, rút từ visual_syl_candidates.csv (gate_09 == 1).
    .venv/bin/python KhoiB/v3/b3_sample_html.py [--n 20] [--seed 2026]
"""
import argparse
import base64
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
BOOKDIR = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}


def b64(im, scale=2):
    im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG"); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    C = pd.read_csv(REPO / "KhoiB/v3/visual_syl_candidates.csv", dtype=str, keep_default_na=False)
    L = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
    O = pd.read_csv(REPO / "KhoiB/v3/p_visual_oof_v3_results/p_visual_oof_v3.csv", dtype=str, keep_default_na=False)
    L["top5"] = O.top5.values
    g = C[C.gate_09 == "1"].sample(n=a.n, random_state=a.seed).sort_values(["book", "page", "column"])
    cards = []
    for _, r in g.iterrows():
        page = Image.open(REPO / "prepared" / BOOKDIR[r.book] / "pages" / f"{r.page}.png").convert("L")
        col = L[(L.book == r.book) & (L.page == r.page) & (L["column"] == r["column"])].copy()
        col["si"] = col.syl_idx.astype(int); col = col.sort_values("si")
        me = col[col.nom_idx == r.nom_idx].iloc[0]
        x0, y0, x1, y1 = json.loads(me.bbox); w, h = x1 - x0, y1 - y0
        crop = page.crop((max(0, x0 - int(0.12 * w)), max(0, y0 - int(0.12 * h)), x1 + int(0.12 * w), y1 + int(0.12 * h)))
        si = int(me.syl_idx)
        nb = col[(col.si >= si - 2) & (col.si <= si + 2)]
        ys = [json.loads(b) for b in nb.bbox]
        ctx = page.crop((min(b[0] for b in ys) - 8, min(b[1] for b in ys) - 8, max(b[2] for b in ys) + 8, max(b[3] for b in ys) + 8))
        ctx_txt = " · ".join(f"{'<b>' if int(q.syl_idx) == si else ''}{q.ocr_char}/{q.syllable}{'</b>' if int(q.syl_idx) == si else ''}" for _, q in nb.iterrows())
        cards.append(f"""<div class="card"><div class="img"><img src="{b64(crop, 2)}" title="crop bbox v3"></div>
<div class="ctx"><img src="{b64(ctx, 1)}" title="ngữ cảnh ±2 ô"></div>
<div class="txt"><div class="syl">{r.syllable} <span class="p">p={float(r.p_syl_v3):.3f}</span></div>
<div>OCR: <b>{r.ocr_char}</b> · {r.book}/{r.page}/c{r['column']}/nom {r.nom_idx} (syl {r.syl_idx}) · {r.tier}/{r.rule} · fold {r.fold}</div>
<div>top-5: {', '.join(f'{s} {p:.2f}' for s, p in json.loads(me.top5))}</div>
<div class="c">ngữ cảnh: {ctx_txt}</div></div></div>""")
    html = f"""<!doctype html><meta charset="utf-8"><title>B-3 mẫu {a.n} ô qua cổng p≥0,9</title>
<style>body{{font-family:system-ui;margin:16px;max-width:1100px}} .card{{display:flex;gap:14px;border:1px solid #ccc;padding:8px;margin:8px 0;align-items:center}}
.img img{{height:160px;border:1px solid #999}} .ctx img{{height:200px;border:1px dashed #aaa}} .syl{{font-size:26px;font-weight:700}} .p{{font-size:14px;color:#060;font-weight:400}}
.c{{color:#444;font-size:13px}} b{{color:#a00}}</style>
<h1>B-3 · {a.n} ô REVIEW ngẫu nhiên qua cổng thị giác (argmax == âm ghép ∧ p ≥ 0,9)</h1>
<p>Rút seed {a.seed} từ <code>KhoiB/v3/visual_syl_candidates.csv</code> (gate_09 = 1, {int((C.gate_09 == '1').sum()):,} ô). Ảnh trái: crop bbox v3 (đệm 12%); ảnh giữa: ngữ cảnh ±2 ô cùng cột; chữ đỏ = ô đang xét. Chỉ để nhìn nhanh — KHÔNG phải mẻ chấm (Khối C: tầng 100–150 ô, 0/150 lỗi). Không đổi nhãn.</p>
{''.join(cards)}
<p>Sinh bởi <code>KhoiB/v3/b3_sample_html.py</code>.</p>"""
    (REPO / "KhoiB/v3/b3_sample.html").write_text(html, encoding="utf-8")
    print(f"-> KhoiB/v3/b3_sample.html ({len(cards)} ô, {len(html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
