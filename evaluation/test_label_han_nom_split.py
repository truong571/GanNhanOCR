"""Test A — Phân loại nhãn cuối theo Hán shared vs Nôm-leaning.

Phân loại heuristic theo Unicode block của codepoint:

  bucket            range                  ý nghĩa
  ----------------- ---------------------- ----------------------------------
  HAN_BASIC         U+4E00 – U+9FFF        CJK Unified — share Hán-Việt
                                           (chữ Hán dùng chung với Trung)
  CJK_EXT_A         U+3400 – U+4DBF        Ext A — rare CJK + một số Nôm
  NOM_EXT_B_PLUS    U+20000 – U+2FFFF      Ext B/C/D/E/F — chủ yếu Nôm thuần
  NOM_PUA           U+E000 – U+F8FF +      Private-Use Area — Nôm legacy
                    U+F0000 – U+10FFFF
  OTHER             ngoài các vùng trên    ký tự lạ / lỗi

Quan trọng: "HAN_BASIC" KHÔNG khẳng định nhãn là Hán-Việt thuần — chỉ là chữ
nằm trong vùng Unicode share. Tuy nhiên trong corpus tiếng Việt cổ, các chữ
ở vùng này đa số được dùng theo nghĩa Hán-Việt. Ext B+ và PUA gần như chắc
chắn là Nôm tự tạo.

Chạy trên:
  1. 2 trang debug (page_0012, page_0014) — sample nhỏ để cross-check
  2. dataset/all/labels.csv — toàn bộ 83k mẫu

Xuất:
  evaluation/label_han_nom_split/
    ├── summary.json
    └── report.md
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "evaluation" / "label_han_nom_split"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOK_DIR = ROOT / "prepared" / "SachThanhTruyen2"
ALL_LABELS = ROOT / "dataset" / "all" / "labels.csv"


def classify_codepoint(ch: str) -> str:
    if not ch:
        return "EMPTY"
    cp = ord(ch[0])
    if 0x4E00 <= cp <= 0x9FFF:
        return "HAN_BASIC"
    if 0x3400 <= cp <= 0x4DBF:
        return "CJK_EXT_A"
    if 0x20000 <= cp <= 0x2FFFF:
        return "NOM_EXT_B_PLUS"
    if (0xE000 <= cp <= 0xF8FF) or (0xF0000 <= cp <= 0x10FFFF):
        return "NOM_PUA"
    return "OTHER"


def classify_records(records, get_char, get_tier, get_matched):
    """records iterable → counters per bucket × tier."""
    by_bucket = Counter()
    by_bucket_tier: dict[tuple[str, int], int] = Counter()
    by_bucket_matched: dict[tuple[str, bool], int] = Counter()
    n_total = 0
    n_with_char = 0
    for r in records:
        n_total += 1
        ch = get_char(r)
        if not ch:
            continue
        n_with_char += 1
        b = classify_codepoint(ch)
        t = get_tier(r) or 0
        m = bool(get_matched(r))
        by_bucket[b] += 1
        by_bucket_tier[(b, t)] += 1
        by_bucket_matched[(b, m)] += 1
    return n_total, n_with_char, by_bucket, by_bucket_tier, by_bucket_matched


def stats_block(name: str, n_total, n_char, by_bucket,
                by_bucket_tier, by_bucket_matched):
    out = {
        "name": name,
        "n_total_records": n_total,
        "n_with_nom_char": n_char,
        "by_bucket": dict(by_bucket),
        "by_bucket_pct": {k: round(100 * v / max(1, n_char), 2)
                          for k, v in by_bucket.items()},
        "by_bucket_tier": {f"{b}|T{t}": v
                           for (b, t), v in by_bucket_tier.items()},
        "by_bucket_matched_true": {b: by_bucket_matched.get((b, True), 0)
                                   for b in by_bucket},
        "by_bucket_matched_false": {b: by_bucket_matched.get((b, False), 0)
                                    for b in by_bucket},
    }
    return out


# ── (1) Debug 2 trang
ds = json.load(open(BOOK_DIR / "labeled" / "dataset.json"))
debug = [r for r in ds if r["page"] in ("page_0012", "page_0014")]
res_debug = stats_block(
    "debug_p12_p14",
    *classify_records(
        debug,
        get_char=lambda r: r.get("nom_char"),
        get_tier=lambda r: r.get("tier"),
        get_matched=lambda r: r.get("matched"),
    ),
)

# ── (2) Toàn bộ dataset/all/labels.csv
all_rows = list(csv.DictReader(open(ALL_LABELS, encoding="utf-8")))
res_all = stats_block(
    "dataset_all",
    *classify_records(
        all_rows,
        get_char=lambda r: r.get("nom_char"),
        get_tier=lambda r: int(r["tier"]) if (r.get("tier") or "").isdigit() else 0,
        get_matched=lambda r: (r.get("matched") or "").strip().lower() == "true",
    ),
)

summary = {"debug_p12_p14": res_debug, "dataset_all": res_all}
(OUT_DIR / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2)
)

# ── Markdown report
order = ["HAN_BASIC", "CJK_EXT_A", "NOM_EXT_B_PLUS", "NOM_PUA", "OTHER"]
labels = {
    "HAN_BASIC":      "Hán-share (U+4E00–U+9FFF)",
    "CJK_EXT_A":      "Ext A (U+3400–U+4DBF) — rare CJK + một phần Nôm",
    "NOM_EXT_B_PLUS": "Ext B+ (U+20000+) — gần như chắc là Nôm thuần",
    "NOM_PUA":        "PUA — Nôm legacy",
    "OTHER":          "Khác / không CJK",
}


def render_block(res):
    lines = [
        f"### {res['name']}", "",
        f"- Tổng record: **{res['n_total_records']}**",
        f"- Có `nom_char`: **{res['n_with_nom_char']}**",
        "",
        "| Bucket | Mô tả | n | % | matched True | matched False | T1 | T2 | T3 | T0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in order:
        n = res["by_bucket"].get(b, 0)
        if n == 0:
            continue
        pct = res["by_bucket_pct"].get(b, 0.0)
        mt = res["by_bucket_matched_true"].get(b, 0)
        mf = res["by_bucket_matched_false"].get(b, 0)
        t1 = res["by_bucket_tier"].get(f"{b}|T1", 0)
        t2 = res["by_bucket_tier"].get(f"{b}|T2", 0)
        t3 = res["by_bucket_tier"].get(f"{b}|T3", 0)
        t0 = res["by_bucket_tier"].get(f"{b}|T0", 0)
        lines.append(
            f"| `{b}` | {labels[b]} | {n} | {pct}% | {mt} | {mf} "
            f"| {t1} | {t2} | {t3} | {t0} |"
        )
    return lines


md = [
    "# Test A — Phân loại nhãn cuối theo Hán shared vs Nôm thuần",
    "",
    "Bucket dựa trên Unicode block của `nom_char`:",
    "",
    "- `HAN_BASIC` (U+4E00–U+9FFF) — CJK Unified, dùng chung với Trung. "
    "Trong corpus tiếng Việt cổ thường mang nghĩa Hán-Việt.",
    "- `CJK_EXT_A` (U+3400–U+4DBF) — Ext A. Có cả CJK hiếm và một số Nôm.",
    "- `NOM_EXT_B_PLUS` (U+20000+) — Ext B/C/D/E/F. Đa số là Nôm thuần.",
    "- `NOM_PUA` — Private-Use Area, Nôm legacy.",
    "",
    "## (1) Sample 2 trang debug (page_0012 + page_0014)",
    "",
] + render_block(res_debug) + [
    "",
    "## (2) Toàn bộ dataset/all/labels.csv (84k record)",
    "",
] + render_block(res_all) + ["", "## Đọc số liệu", ""]


def add_reading(res):
    n = res["n_with_nom_char"]
    han = res["by_bucket"].get("HAN_BASIC", 0)
    nom_b = res["by_bucket"].get("NOM_EXT_B_PLUS", 0)
    nom_pua = res["by_bucket"].get("NOM_PUA", 0)
    ext_a = res["by_bucket"].get("CJK_EXT_A", 0)
    pure_nom = nom_b + nom_pua
    return (
        f"- **{res['name']}**: "
        f"Hán shared **{han}** ({100*han/max(1,n):.1f}%),  "
        f"Ext A {ext_a} ({100*ext_a/max(1,n):.1f}%),  "
        f"**Nôm thuần** (Ext B+ & PUA) **{pure_nom}** "
        f"({100*pure_nom/max(1,n):.1f}%)."
    )


md += [add_reading(res_debug), add_reading(res_all)]

# Tier composition note
md += ["", "## Cấu trúc Tier theo bucket — dataset toàn phần", "",
       "Mục tiêu: xem các bucket Nôm thuần (Ext B+ / PUA) thường được gán "
       "bằng tầng nào — nếu chủ yếu T3 → cảnh báo độ tin cậy.",
       ""]


def tier_dist_per_bucket(res):
    lines = ["| Bucket | n | T1% | T2% | T3% | matched% |", "|---|---:|---:|---:|---:|---:|"]
    for b in order:
        n = res["by_bucket"].get(b, 0)
        if n == 0:
            continue
        t1 = res["by_bucket_tier"].get(f"{b}|T1", 0)
        t2 = res["by_bucket_tier"].get(f"{b}|T2", 0)
        t3 = res["by_bucket_tier"].get(f"{b}|T3", 0)
        mt = res["by_bucket_matched_true"].get(b, 0)
        lines.append(
            f"| `{b}` | {n} | {100*t1/n:.1f}% | {100*t2/n:.1f}% "
            f"| {100*t3/n:.1f}% | {100*mt/n:.1f}% |"
        )
    return lines


md += tier_dist_per_bucket(res_all)

(OUT_DIR / "report.md").write_text("\n".join(md))

# ── Self-test assertions
for res in (res_debug, res_all):
    total = sum(res["by_bucket"].values())
    assert total == res["n_with_nom_char"], (
        f"bucket sum {total} ≠ n_with_nom_char {res['n_with_nom_char']}"
    )

print("✅ Self-test pass")
print(f"\n=== {res_debug['name']} ({res_debug['n_with_nom_char']} chars) ===")
for b in order:
    n = res_debug["by_bucket"].get(b, 0)
    if n:
        print(f"  {b:<16} {n:>5}  ({100*n/res_debug['n_with_nom_char']:.1f}%)")

print(f"\n=== {res_all['name']} ({res_all['n_with_nom_char']} chars) ===")
for b in order:
    n = res_all["by_bucket"].get(b, 0)
    if n:
        print(f"  {b:<16} {n:>6}  ({100*n/res_all['n_with_nom_char']:.1f}%)")

print(f"\nOutput: {OUT_DIR}/")
