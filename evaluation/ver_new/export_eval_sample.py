"""Bước 3 (1/2) — draw a STRATIFIED human-verification sample for SILVER/GOLD/SYLLABLE
precision, and render a self-contained review packet.

Strata (target counts, capped by availability), NO ink/FD pre-filter so mislabels
are not hidden:
  GOLD  s1_inter_s2_direct     (dict-direct; the floor)
  GOLD  s1_inter_s2_similar    (similar-bridge; the WEAKER GOLD path)
  SILVER s2_inter_s3_corrected (vision overruled OCR; the calibrated tier to prove)
  SILVER s1_inter_s3_out_of_dict
  SYLLABLE nghia_consensus     (verify the SYLLABLE reading, not the char)
  REVIEW (control)             (should be mostly wrong -> sanity that we reject well)
Within each stratum rows are split by class-frequency band (rare <5 vs common) so
rare characters — where mislabels concentrate — are covered.

Outputs into evaluation/ver_new/eval_sample/:
  verify.csv   one row per sample with BLANK `human_correct` (1/0) + `human_label`
  review.html  self-contained table: crop image | proposed label+unicode | syllable
               | tier/rule | reference glyph — open in a browser, fill verify.csv.
  imgs/        copied crop + reference PNGs (so the packet is shareable)

Then a human fills `human_correct` (and `human_label` when wrong) and runs
measure_precision.py.

Run:
  .venv/bin/python evaluation/ver_new/export_eval_sample.py --n 430 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import html
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FD_DIR = REPO / "gannhanocr-fd"

# (tier, rule) -> target count
STRATA = [
    ("GOLD", "s1_inter_s2_direct", 100),
    ("GOLD", "s1_inter_s2_similar", 80),
    ("SILVER", "s2_inter_s3_corrected", 120),
    ("SILVER", "s1_inter_s3_out_of_dict", 40),
    ("SYLLABLE", "nghia_consensus", 60),
    ("REVIEW", "*", 30),
]


def fd_path(ch: str):
    """Reference similar-font glyph path for a char (gannhanocr-fd/<HH>/U+<HEX>.png)."""
    if not ch or len(ch) != 1:
        return None
    hx = f"{ord(ch):X}"
    p = FD_DIR / hx[:2] / f"U+{hx}.png"
    if p.exists():
        return p
    hits = list(FD_DIR.rglob(f"U+{hx}.png"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--out", default=str(HERE / "eval_sample"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=430, help="(advisory) total; strata caps apply")
    args = ap.parse_args()
    random.seed(args.seed)

    D = Path(args.dataset)
    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))
    # class-frequency band keyed by char-label (or syllable for SYLLABLE rows)
    cnt = Counter(r["label"] for r in rows if r["label_level"] == "char" and r["label"])

    def band(r):
        return "rare" if cnt.get(r["label"], 0) < 5 else "common"

    pool = defaultdict(list)
    for r in rows:
        if not r["image"] and r["tier"] != "REVIEW":
            continue
        pool[(r["tier"], r["rule"])].append(r)

    out = Path(args.out)
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    picked = []
    for tier, rule, target in STRATA:
        cands = []
        for (t, ru), rs in pool.items():
            if t == tier and (rule == "*" or ru == rule):
                cands += rs
        # split by band, sample ~half/half to guarantee rare coverage
        rare = [r for r in cands if band(r) == "rare"]
        common = [r for r in cands if band(r) == "common"]
        random.shuffle(rare); random.shuffle(common)
        half = target // 2
        sel = rare[:half] + common[:target - len(rare[:half])]
        if len(sel) < target:                         # top up from whatever remains
            extra = [r for r in (rare[half:] + common[target - half:]) if r not in sel]
            sel += extra[:target - len(sel)]
        for r in sel:
            picked.append((tier, rule, band(r), r))
    random.shuffle(picked)

    fields = ["sample_id", "book", "page", "column", "tier", "rule", "freq_band",
              "ocr_char", "syllable", "label", "unicode", "s3_cosine", "ref_char",
              "image", "human_correct", "human_label", "notes"]
    vrows = []
    cards = []
    for i, (tier, rule, fb, r) in enumerate(picked):
        sid = f"S{i:04d}"
        ref_char = r["label"] or r["ocr_char"]
        # copy crop + reference into the packet
        crop_src = D / r["image"] if r["image"] else None
        crop_dst = f"imgs/{sid}_crop.png"
        if crop_src and crop_src.exists():
            shutil.copy(crop_src, out / crop_dst)
        else:
            crop_dst = ""
        ref_src = fd_path(ref_char)
        ref_dst = f"imgs/{sid}_ref.png"
        if ref_src:
            shutil.copy(ref_src, out / ref_dst)
        else:
            ref_dst = ""
        vrows.append({"sample_id": sid, "book": r["book"], "page": r["page"],
                      "column": r["column"], "tier": tier, "rule": rule, "freq_band": fb,
                      "ocr_char": r["ocr_char"], "syllable": r["syllable"],
                      "label": r["label"], "unicode": r["unicode"],
                      "s3_cosine": r["s3_cosine"], "ref_char": ref_char,
                      "image": r["image"], "human_correct": "", "human_label": "", "notes": ""})
        target_txt = (f"<b style='font-size:34px'>{html.escape(r['label'])}</b> "
                      f"<span style='color:#888'>{r['unicode']}</span>") if r["label"] else \
                     (f"<i>syllable-only:</i> <b>{html.escape(r['syllable'])}</b>")
        cards.append(f"""<tr>
<td>{sid}</td>
<td>{'<img src=\"'+crop_dst+'\" height=90>' if crop_dst else '—'}</td>
<td>{target_txt}<br><small>âm: {html.escape(r['syllable'])} · ocr: {html.escape(r['ocr_char'])}</small></td>
<td><span class=tag>{tier}</span><br><small>{rule}</small><br><small>P={r['s3_cosine']} · {fb}</small></td>
<td>{'<img src=\"'+ref_dst+'\" height=90>' if ref_dst else '—'}<br><small>{html.escape(ref_char)}</small></td>
<td style='font-size:24px;color:#aaa'>☐</td>
</tr>""")

    with open(out / "verify.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(vrows)

    html_doc = f"""<!doctype html><meta charset=utf-8>
<style>body{{font-family:sans-serif;margin:20px}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:6px;vertical-align:middle}} .tag{{background:#eee;padding:2px 6px;border-radius:4px;font-weight:bold}}
th{{background:#f5f5f5;position:sticky;top:0}}</style>
<h2>Bước 3 — soi tay precision ({len(vrows)} mẫu)</h2>
<p>Với mỗi dòng: nhìn <b>ảnh crop</b> (trái) và <b>glyph tham chiếu</b> (phải). Hỏi:
crop CÓ ĐÚNG là chữ/âm đề xuất (cột giữa) không?
Mở <code>verify.csv</code>, điền <code>human_correct</code> = 1 (đúng) / 0 (sai); nếu sai, ghi chữ đúng vào <code>human_label</code>.
SYLLABLE: chỉ cần đúng ÂM TIẾT. REVIEW: nhóm đối chứng (kỳ vọng phần lớn sai/không chắc).</p>
<table><tr><th>id</th><th>crop (woodblock)</th><th>đề xuất</th><th>tier/rule</th><th>tham chiếu (similar-font)</th><th>✓?</th></tr>
{''.join(cards)}
</table>"""
    open(out / "review.html", "w", encoding="utf-8").write(html_doc)

    comp = Counter((t, ru) for t, ru, _, _ in picked)
    bnd = Counter(fb for _, _, fb, _ in picked)
    print(f"sample -> {out}  ({len(vrows)} rows)")
    for (t, ru), n in comp.items():
        print(f"  {t:9s} {ru:24s} {n}")
    print(f"  freq band: {dict(bnd)}")
    print(f"\n  open {out}/review.html, fill {out}/verify.csv (human_correct/human_label),")
    print("  then: .venv/bin/python evaluation/ver_new/measure_precision.py")


if __name__ == "__main__":
    main()
