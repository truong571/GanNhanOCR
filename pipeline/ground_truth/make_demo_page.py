"""Trang DEMO để chiếu trước hội đồng — 9 thẻ mẫu, mở tức thì.

KHÁC HẲN công cụ chấm thật, và khác có chủ đích:

  công cụ chấm (`audit.html`)     trang demo (`demo_audit.html`)
  ------------------------------  ---------------------------------------------
  860 thẻ, ~43 MB                 9 thẻ, < 2 MB — mở là hiện ngay
  GIẤU tier/rule/điểm S3          HIỆN chúng, đóng khung "người chấm KHÔNG thấy"
  bấm được, lưu verdict           chỉ trưng bày, không bấm được
  mục tiêu: đo                    mục tiêu: giải thích cách đo

Thẻ demo rút từ các hàng **không** nằm trong mẻ chấm thật (loại mọi ảnh đã có trong
`manifest.jsonl` bất kỳ), nên chiếu bao nhiêu lần cũng không làm bẩn mẫu đang đo.

Rút NGẪU NHIÊN có hạt giống cố định, và nói thẳng điều đó trên trang — để không ai hỏi
được "thầy chọn mấy ô đẹp phải không". Thẻ nào render xấu thì đổi `--seed` rồi dựng lại.

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.make_demo_page
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageFont

from . import audit_grid
from .cli import _load_config, _paths
from .make_confusion_batch import audited_images

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO / "dataset_out" / "labels_final.csv"
GT_DIR = REPO / "dataset_out" / "ground_truth"
DEFAULT_OUT = GT_DIR / "demo_audit.html"

TIERS = ("GOLD", "SILVER", "SYLLABLE")

TIER_NOTE = {
    "GOLD": "Đồng thuận mạnh nhất (S1 ∩ S2 trực tiếp). Chiếm 74 % tập có nhãn.",
    "SILVER": "Đồng thuận yếu hơn, phải qua hiệu chỉnh. Thẻ trông y hệt GOLD — "
              "và đó chính là điều cần chứng minh.",
    "SYLLABLE": "Nhãn là ÂM TIẾT Quốc ngữ, không có chữ Nôm đề xuất → không có glyph "
                "tham chiếu. Đây là tier DUY NHẤT người chấm nhận ra được.",
}


def pick(labels: pd.DataFrame, per_tier: int, seed: int, exclude: set[str]) -> pd.DataFrame:
    """Rút per_tier hàng mỗi tier, tất định theo seed, tránh mọi ô đang được chấm thật."""
    out = []
    for tier in TIERS:
        pool = labels[(labels["tier"] == tier) & (~labels["image"].astype(str).isin(exclude))]
        if len(pool) < per_tier:
            raise SystemExit(f"{tier}: chỉ còn {len(pool)} hàng ngoài mẻ chấm thật")
        s = int(hashlib.sha1(f"{seed}:demo:{tier}".encode()).hexdigest()[:8], 16) % (2**31)
        idx = np.random.default_rng(s).choice(pool.index.to_numpy(), size=per_tier,
                                              replace=False)
        g = pool.loc[idx].copy()
        g["demo_tier"] = tier
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_cards(sample: pd.DataFrame, paths: dict, qn_dict: dict | None) -> list[dict]:
    """Dựng dữ liệu thẻ: 3 ảnh + nhãn hiển thị + các trường BỊ GIẤU khi chấm thật."""
    font = None
    fp = paths["font"]
    if fp and Path(fp).exists():
        try:
            font = ImageFont.truetype(str(fp), 96)
        except OSError:
            font = None

    cards = []
    for _, r in sample.iterrows():
        crop = audit_grid._load_crop(paths["dataset_dir"], str(r["image"]))
        if crop is None:
            continue
        ref = audit_grid._reference_glyph(paths["fd_dir"], r.get("unicode", ""), font)

        ctx_uri = ""
        sp = (paths["prepared_dir"] / audit_grid.book_to_scan_dir(r["book"])
              / "pages" / f"{r['page']}.png")
        bbox = audit_grid._parse_bbox(r.get("bbox"))
        if sp.exists() and bbox is not None:
            from PIL import Image
            with Image.open(sp) as scan:
                ctx = audit_grid._context_crop(scan.convert("RGB"), bbox)
            if ctx is not None:
                ctx_uri = audit_grid._data_uri(ctx)

        def _txt(v) -> str:
            return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()

        syl = _txt(r.get("syllable"))
        label = _txt(r.get("label"))
        cands = " ".join((qn_dict.get(syl, []) if (qn_dict and syl) else [])[:12])
        s3 = r.get("s3_cosine")
        cards.append({
            "tier": r["demo_tier"],
            "crop": audit_grid._data_uri(crop),
            "ctx": ctx_uri,
            "ref": audit_grid._data_uri(ref) if ref is not None else "",
            "label": label,
            "syl": syl,
            "cands": cands,
            "hidden": {
                "tier": str(r["tier"]),
                "luật sinh nhãn": str(r.get("rule") or "—"),
                "sách · trang": f"{r['book']} · {r['page']}",
                "điểm S3 (cosine)": ("—" if s3 is None or pd.isna(s3) else f"{float(s3):.3f}"),
            },
        })
    return cards


def render(cards: list[dict], seed: int, n_pop: dict) -> str:
    e = html.escape
    blocks = []
    for tier in TIERS:
        group = [c for c in cards if c["tier"] == tier]
        items = []
        for c in group:
            imgs = f'<figure><img src="{c["crop"]}" alt=""><figcaption>crop được gán</figcaption></figure>'
            if c["ctx"]:
                imgs += (f'<figure><img src="{c["ctx"]}" alt="">'
                         f'<figcaption>vị trí trên bản scan</figcaption></figure>')
            if c["ref"]:
                imgs += (f'<figure><img src="{c["ref"]}" alt="">'
                         f'<figcaption>glyph tham chiếu</figcaption></figure>')
            else:
                imgs += ('<figure class="none"><div class="nobox">không có</div>'
                         '<figcaption>glyph tham chiếu</figcaption></figure>')
            if c["label"]:
                lab = (f'<div class="lab">{e(c["label"])}</div>'
                       f'<div class="meta">âm: <b>{e(c["syl"]) or "—"}</b></div>')
            else:
                lab = (f'<div class="lab">{e(c["syl"]) or "?"}</div>'
                       f'<div class="meta">nhãn ở ô này là <b>ÂM TIẾT</b>, '
                       f'không có chữ Nôm đề xuất</div>')
            cand = (f'<div class="cands">ứng viên từ điển: {e(c["cands"])}</div>'
                    if c["cands"] else "")
            hid = "".join(f"<li><span>{e(k)}</span><b>{e(v)}</b></li>"
                          for k, v in c["hidden"].items())
            items.append(f"""
      <article class="card">
        <div class="imgs">{imgs}</div>
        {lab}{cand}
        <div class="choices"><span class="c1">1 · nhãn ĐÚNG</span>
          <span class="c2">2 · nhãn SAI</span>
          <span class="c3">3 · không đọc được</span></div>
        <div class="hidden"><p class="hid-h">✗ Người chấm KHÔNG thấy những trường này —
          chúng chỉ nằm trong <code>manifest.jsonl</code>, ghép lại sau khi chấm xong</p>
          <ul>{hid}</ul></div>
      </article>""")
        blocks.append(f"""
  <section>
    <h2>{tier} <span class="pop">{n_pop.get(tier, 0):,} crop</span></h2>
    <p class="tnote">{e(TIER_NOTE[tier])}</p>
    <div class="grid">{"".join(items)}</div>
  </section>""")

    return f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Công cụ kiểm định nhãn — thẻ mẫu</title>
<style>
  :root {{ --ink:#141414; --mut:#6b6b6b; --line:#d8d5cd; --bg:#faf9f6; --card:#fff;
          --hid:#8a1c1c; --hidbg:#fdf3f2; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font-size:17px; line-height:1.55;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Heiti SC",Arial,sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:32px 24px 64px; }}
  h1 {{ font-size:30px; margin:0 0 6px; letter-spacing:-.01em; }}
  .sub {{ color:var(--mut); margin:0 0 24px; }}
  .legend {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:18px 22px; margin-bottom:32px; }}
  .legend h3 {{ margin:0 0 10px; font-size:16px; text-transform:uppercase;
    letter-spacing:.08em; color:var(--mut); }}
  .legend ol {{ margin:0; padding-left:22px; }}
  .legend li {{ margin:5px 0; }}
  h2 {{ font-size:22px; margin:34px 0 4px; border-top:2px solid var(--ink);
    padding-top:14px; }}
  .pop {{ font-size:15px; font-weight:400; color:var(--mut); margin-left:8px; }}
  .tnote {{ color:var(--mut); margin:0 0 16px; max-width:70ch; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:18px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:18px; }}
  .imgs {{ display:flex; gap:14px; align-items:flex-start; flex-wrap:wrap; }}
  .imgs figure {{ margin:0; text-align:center; }}
  .imgs img {{ max-width:110px; max-height:150px; border:1px solid var(--line); background:#fff; }}
  .nobox {{ width:82px; height:82px; border:1px dashed var(--line); border-radius:6px;
    display:flex; align-items:center; justify-content:center; color:#bbb; font-size:12px; }}
  figcaption {{ font-size:10.5px; color:var(--mut); text-transform:uppercase;
    letter-spacing:.06em; margin-top:5px; }}
  .lab {{ font-size:52px; line-height:1.1; margin:12px 0 2px; }}
  .meta {{ color:#444; font-size:15px; }}
  .cands {{ color:var(--mut); font-size:14px; margin-top:6px; }}
  .choices {{ display:flex; gap:8px; flex-wrap:wrap; margin:14px 0 0; }}
  .choices span {{ font-size:13.5px; padding:5px 11px; border-radius:6px; border:1px solid; }}
  .c1 {{ border-color:#2e6e4c; color:#2e6e4c; }}
  .c2 {{ border-color:#9a6b10; color:#9a6b10; }}
  .c3 {{ border-color:#a03b32; color:#a03b32; }}
  .hidden {{ margin-top:16px; background:var(--hidbg); border:1px solid #f0d9d6;
    border-radius:8px; padding:11px 13px; }}
  .hid-h {{ margin:0 0 7px; font-size:12.5px; color:var(--hid); font-weight:600; }}
  .hidden ul {{ list-style:none; margin:0; padding:0; font-size:13.5px; }}
  .hidden li {{ display:flex; justify-content:space-between; gap:12px; padding:2px 0; }}
  .hidden span {{ color:var(--mut); }}
  .hidden b {{ font-weight:600; }}
  code {{ font-size:12.5px; background:#0000000d; padding:1px 5px; border-radius:4px; }}
  footer {{ margin-top:40px; padding-top:18px; border-top:1px solid var(--line);
    color:var(--mut); font-size:14.5px; }}
  @media print {{ body {{ background:#fff; }} .card {{ break-inside:avoid; }} }}
</style></head>
<body><div class="wrap">
  <h1>Công cụ kiểm định nhãn bằng người — thẻ mẫu</h1>
  <p class="sub">9 thẻ rút ngẫu nhiên (hạt giống {seed}) từ các ô <b>ngoài</b> mẻ chấm thật.
     Trang trưng bày, không bấm chấm được.</p>

  <div class="legend">
    <h3>Người chấm thấy đúng bốn thứ</h3>
    <ol>
      <li><b>Crop được gán</b> — chính ảnh sẽ đi vào bộ dữ liệu.</li>
      <li><b>Vị trí trên bản scan</b> — khung đỏ trên trang gốc, để đọc được chữ khi khung
          cắt xấu.</li>
      <li><b>Glyph tham chiếu</b> — chữ được gán, render từ font, để so bằng mắt.</li>
      <li><b>Nhãn đề xuất + âm + ứng viên từ điển</b>.</li>
    </ol>
    <p style="margin:12px 0 0"><b>Một câu hỏi duy nhất:</b> <i>nhãn hiện trên thẻ có đúng
       là chữ viết trong ô không?</i> — ba lựa chọn, không có lựa chọn thứ tư. Mọi trường
       có thể gây thiên lệch đều bị giấu (khung đỏ nhạt dưới mỗi thẻ).</p>
  </div>
{"".join(blocks)}
  <footer>
    Sinh bởi <code>pipeline.ground_truth.make_demo_page</code> ·
    nguồn <code>dataset_out/labels_final.csv</code> ·
    công cụ chấm thật: <code>dataset_out/ground_truth/audit_combined/audit.html</code>
    (860 ô) · thiết kế mẫu và lý do: <code>docs/KE_HOACH_CHAM_TAY.md</code>
  </footer>
</div></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_demo_page")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--per-tier", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = ap.parse_args(argv)

    labels = pd.read_csv(args.labels, dtype={"image_md5": str})
    exclude = audited_images(GT_DIR)
    print(f"[demo] loại {len(exclude):,} ảnh đang nằm trong các mẻ chấm thật")

    sample = pick(labels, args.per_tier, args.seed, exclude)
    cfg = _load_config(Path(args.config))
    paths = _paths(cfg)
    qn_dict = None
    try:
        from core.text.dictionary import load_qn_to_nom
        if paths["qn_dict"].exists():
            qn_dict = load_qn_to_nom(str(paths["qn_dict"]))
    except (ImportError, OSError) as exc:
        print(f"[demo] bỏ qua từ điển QN ({exc})")

    cards = build_cards(sample, paths, qn_dict)
    n_pop = labels["tier"].value_counts().to_dict()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(cards, args.seed, n_pop), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"[demo] {len(cards)} thẻ ({', '.join(t + ' ' + str(sum(1 for c in cards if c['tier'] == t)) for t in TIERS)}) "
          f"-> {out} ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
