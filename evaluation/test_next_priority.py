"""Scale evaluation cho 4 đề xuất tiếp theo trên TOÀN BỘ dataset (84k records).

Test A — Q1 Diacritic restore ở scale
    Đếm số syllable trong dataset đang sai/thiếu dấu so với QN dict.
    Mô phỏng: nếu khôi phục, có bao nhiêu cặp T2/T3 promote được lên T1?

Test B — F2' Bucket-aware T3 threshold simulation
    Apply ngưỡng HAN_BASIC≥0.85, NOM_*≥0.90 trên dataset.json 3 sách.
    Đếm trade-off: demote vs giữ recall.

Test C — V2 sqpad candidate analysis
    Phát hiện crop có aspect ratio ngoài [0.5, 2.0] (extreme_aspect).
    Đếm per sách/tier/matched để biết workload thực sự.

Test D — Per-col ensemble cost/impact projection
    Project chi phí API + số promotion dự kiến lên 3 sách dựa trên
    rate quan sát từ 2 trang debug (12 promotion / 373 record).

Mọi test KHÔNG gọi API, KHÔNG load DINOv2 — chỉ phân tích dữ liệu sẵn.
"""
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(ROOT))

from core.text.dictionary import (  # noqa: E402
    load_qn_to_nom, build_nom_to_qn,
)
from core.ranking.ranker import tier1_dictionary_lookup  # noqa: E402

OUT_DIR = ROOT / "evaluation" / "next_priority_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QN_DICT = ROOT / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"
BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]
LABELS_ALL = ROOT / "dataset" / "all" / "labels.csv"

T3_BUCKET_THR = {
    "HAN_BASIC": 0.85,
    "CJK_EXT_A": 0.87,
    "NOM_EXT_B_PLUS": 0.90,
    "NOM_PUA": 0.90,
    "OTHER": 0.90,
}


# ─────────────────────── helpers ───────────────────────
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if unicodedata.category(c) != "Mn").lower()


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


def build_qn_syl_set() -> set[str]:
    out = set()
    with open(QN_DICT, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qn = (row.get("QuocNgu") or "").strip()
            if qn:
                out.add(qn)
                out.add(qn.lower())
    return out


def build_accent_free_index(qn_set: set[str]) -> dict[str, list[str]]:
    """{accent-stripped: [variants ...]} for O(1) restore lookup."""
    idx = defaultdict(list)
    for q in qn_set:
        idx[strip_accents(q)].append(q)
    return idx


# ─────────────────────── TEST A ───────────────────────
def test_a_diacritic_scale(qn_to_nom):
    print("\n[Test A] Q1 Diacritic restore — scale test trên dataset/all/labels.csv ...")
    qn_set = build_qn_syl_set()
    af_idx = build_accent_free_index(qn_set)

    rows = list(csv.DictReader(open(LABELS_ALL, encoding="utf-8")))
    n_total = len(rows)
    n_match_type = 0
    n_syl_in_dict = 0
    n_syl_not_in_dict = 0
    n_restorable = 0
    n_unique_restorable = 0
    n_would_promote_t1 = 0

    seen_restorable: set[tuple[str, str]] = set()
    restorations: dict[tuple[str, str], int] = Counter()
    promotions_examples = []

    for r in rows:
        syl = (r.get("syllable") or "").strip()
        if not syl:
            continue
        n_match_type += 1
        if syl in qn_set or syl.lower() in qn_set:
            n_syl_in_dict += 1
            continue
        n_syl_not_in_dict += 1
        cands = af_idx.get(strip_accents(syl), [])
        if not cands:
            continue
        n_restorable += 1
        # Pick first candidate as restoration
        new_syl = cands[0]
        key = (syl, new_syl)
        if key not in seen_restorable:
            seen_restorable.add(key)
            n_unique_restorable += 1
        restorations[key] += 1

        # Promotion check: previously NOT matched OR tier ≥ 2; check if
        # ocr_char ∈ qn_to_nom[new_syl] would produce a Tier 1 match.
        ocr = (r.get("ocr_char") or "").strip()
        prev_tier = int(r["tier"]) if (r.get("tier") or "").isdigit() else 0
        prev_matched = (r.get("matched") or "").strip().lower() == "true"
        if not ocr or (prev_tier == 1 and prev_matched):
            continue
        candidates_new = qn_to_nom.get(new_syl.lower()) or qn_to_nom.get(new_syl) or []
        if ocr in candidates_new:
            n_would_promote_t1 += 1
            if len(promotions_examples) < 30:
                promotions_examples.append({
                    "syl_from": syl, "syl_to": new_syl,
                    "ocr": ocr, "prev_nom": r.get("nom_char"),
                    "prev_tier": prev_tier,
                    "source": r.get("source"), "page": r.get("page"),
                })

    return {
        "n_total": n_total,
        "n_with_syllable": n_match_type,
        "n_syl_in_dict": n_syl_in_dict,
        "n_syl_not_in_dict": n_syl_not_in_dict,
        "n_restorable_within_1_diacritic": n_restorable,
        "n_unique_restorable_pairs": n_unique_restorable,
        "n_would_promote_to_T1": n_would_promote_t1,
        "top_restorations": restorations.most_common(20),
        "promotion_examples": promotions_examples[:30],
    }


# ─────────────────────── TEST B ───────────────────────
def test_b_bucket_threshold():
    print("[Test B] F2' bucket threshold — simulation trên dataset.json 3 sách ...")
    by_book = {}
    for book in BOOKS:
        ds_path = ROOT / "prepared" / book / "labeled" / "dataset.json"
        if not ds_path.exists():
            continue
        ds = json.load(open(ds_path))
        n = len(ds)
        demote_per_bucket = Counter()
        demote_total = 0
        baseline_matched = sum(1 for r in ds if r.get("matched"))
        for r in ds:
            if r.get("tier") != 3 or not r.get("matched"):
                continue
            v = r.get("visual_score")
            if not isinstance(v, (int, float)):
                continue
            b = classify_codepoint(r.get("nom_char") or "")
            thr = T3_BUCKET_THR.get(b, 0.90)
            if v < thr:
                demote_total += 1
                demote_per_bucket[b] += 1
        new_matched = baseline_matched - demote_total
        by_book[book] = {
            "n_total": n,
            "baseline_matched": baseline_matched,
            "baseline_rate_pct": round(100 * baseline_matched / n, 2),
            "demote_total": demote_total,
            "demote_per_bucket": dict(demote_per_bucket),
            "new_matched": new_matched,
            "new_rate_pct": round(100 * new_matched / n, 2),
            "recall_drop_pp": round(
                100 * baseline_matched / n - 100 * new_matched / n, 2
            ),
        }
    return by_book


# ─────────────────────── TEST C ───────────────────────
def test_c_extreme_aspect():
    print("[Test C] V2 sqpad candidate audit — aspect ratio ngoài [0.5, 2.0] ...")
    per_book = {}
    for book in BOOKS:
        ds_path = ROOT / "prepared" / book / "labeled" / "dataset.json"
        if not ds_path.exists():
            continue
        ds = json.load(open(ds_path))
        ext = []
        for r in ds:
            bb = r.get("bbox")
            if not bb or len(bb) != 4:
                continue
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            if h <= 0 or w <= 0:
                continue
            ar = w / h
            if ar < 0.5 or ar > 2.0:
                ext.append({"r": r, "ar": ar})
        per_book[book] = {
            "n_total": len(ds),
            "n_extreme_aspect": len(ext),
            "n_extreme_t3_matched": sum(
                1 for x in ext
                if x["r"].get("tier") == 3 and x["r"].get("matched")
            ),
            "n_extreme_t3_total": sum(
                1 for x in ext if x["r"].get("tier") == 3
            ),
            "ar_range": (
                round(min(x["ar"] for x in ext), 3) if ext else None,
                round(max(x["ar"] for x in ext), 3) if ext else None,
            ),
        }
    return per_book


# ─────────────────────── TEST D ───────────────────────
def test_d_percol_projection():
    """Project per-col ensemble cost based on observed promotion rate."""
    # Observed on 2 debug pages: 12 promotions / 373 records (3.2%)
    obs_promotion_rate = 12 / 373
    obs_api_latency_s = 2.0  # s per col observed

    pages_per_book = {"SachThanhTruyen2": 82, "SachThanhTruyen4": 144,
                      "SachThanhTruyen11": 221}
    n_pages_total = sum(pages_per_book.values())
    n_calls_full = n_pages_total * 9
    n_calls_selective_estimate = int(n_pages_total * 9 * 0.10)  # ~10% cols flagged

    # 84k records total
    n_records = 83564
    projected_promotions = int(n_records * obs_promotion_rate)

    return {
        "pages_per_book": pages_per_book,
        "n_pages_total": n_pages_total,
        "n_records": n_records,
        "full_scope": {
            "api_calls": n_calls_full,
            "est_runtime_min": round(n_calls_full * obs_api_latency_s / 60, 1),
            "projected_promotions": projected_promotions,
            "cost_per_promotion": round(n_calls_full / max(1, projected_promotions), 2),
        },
        "selective_10pct_scope": {
            "api_calls": n_calls_selective_estimate,
            "est_runtime_min": round(
                n_calls_selective_estimate * obs_api_latency_s / 60, 1
            ),
            "note": "Chỉ chạy với cột flag count_ok=False hoặc nhiều ocr_char=null",
        },
        "observed_baseline": {
            "debug_pages": 2,
            "records": 373,
            "promotions": 12,
            "rate_pct": round(100 * obs_promotion_rate, 2),
        },
    }


# ─────────────────────── main ───────────────────────
def main():
    qn_to_nom = load_qn_to_nom(str(QN_DICT))

    res_a = test_a_diacritic_scale(qn_to_nom)
    res_b = test_b_bucket_threshold()
    res_c = test_c_extreme_aspect()
    res_d = test_d_percol_projection()

    summary = {
        "A_diacritic_scale": res_a,
        "B_bucket_threshold_sim": res_b,
        "C_extreme_aspect_audit": res_c,
        "D_percol_ensemble_projection": res_d,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    # ── Markdown report ──
    md = ["# Đánh giá scale 4 đề xuất ưu tiên kế tiếp", "",
          f"Dataset: dataset/all/labels.csv ({res_a['n_total']} records, "
          f"{res_d['n_records']} matched-type) + dataset.json 3 sách.", "",
          "## A — Q1 Diacritic restore (toàn bộ 84k record)", "",
          f"- Records xét: **{res_a['n_with_syllable']}**",
          f"- Syllable đã có trong dict: **{res_a['n_syl_in_dict']}** "
          f"({100*res_a['n_syl_in_dict']/max(1,res_a['n_with_syllable']):.1f}%)",
          f"- Syllable KHÔNG có trong dict: **{res_a['n_syl_not_in_dict']}** "
          f"({100*res_a['n_syl_not_in_dict']/max(1,res_a['n_with_syllable']):.1f}%)",
          f"- Trong số đó, **có thể khôi phục** (cách dấu ≤1): "
          f"**{res_a['n_restorable_within_1_diacritic']}** "
          f"({100*res_a['n_restorable_within_1_diacritic']/max(1,res_a['n_syl_not_in_dict']):.1f}%)",
          f"- Số cặp khôi phục unique: **{res_a['n_unique_restorable_pairs']}**",
          f"- **Promotion T2/T3 → T1** nếu áp Q1: "
          f"**{res_a['n_would_promote_to_T1']}** record",
          "",
          "### Top 20 restoration phổ biến",
          "",
          "| syl từ | syl đến | n |",
          "|---|---|---:|"]
    for (a, b), n in res_a["top_restorations"]:
        md.append(f"| `{a}` | `{b}` | {n} |")

    if res_a["promotion_examples"]:
        md += ["", "### Ví dụ promotion nếu áp Q1 (top 30)", "",
               "| sách | page | syl_từ | syl_đến | ocr | prev_nom | prev_tier |",
               "|------|------|--------|---------|-----|----------|----------:|"]
        for e in res_a["promotion_examples"]:
            md.append(
                f"| {e['source']} | {e['page']} | `{e['syl_from']}` "
                f"| `{e['syl_to']}` | {e['ocr']} | {e['prev_nom']} "
                f"| T{e['prev_tier']} |"
            )

    md += ["", "## B — F2' Bucket-aware T3 threshold simulation",
           "",
           "Ngưỡng: HAN_BASIC ≥ 0.85 · Ext A ≥ 0.87 · Ext B+/PUA ≥ 0.90",
           "",
           "| Book | n | baseline matched | demote | new matched | recall drop |",
           "|---|---:|---:|---:|---:|---:|"]
    for book, s in res_b.items():
        md.append(
            f"| {book} | {s['n_total']} | {s['baseline_matched']} "
            f"({s['baseline_rate_pct']}%) | {s['demote_total']} | "
            f"{s['new_matched']} ({s['new_rate_pct']}%) | "
            f"{s['recall_drop_pp']}pp |"
        )
    md += ["", "Phân bucket demote / sách:", ""]
    for book, s in res_b.items():
        md.append(f"- **{book}**: {dict(s['demote_per_bucket'])}")

    md += ["", "## C — V2 sqpad candidate audit (extreme aspect)", "",
           "Crop có aspect ratio (w/h) ngoài [0.5, 2.0] là candidate cho sqpad rerank.",
           "",
           "| Book | n total | extreme_aspect | extreme T3 matched | extreme T3 total | AR range |",
           "|---|---:|---:|---:|---:|---|"]
    for book, s in res_c.items():
        md.append(
            f"| {book} | {s['n_total']} | {s['n_extreme_aspect']} "
            f"| {s['n_extreme_t3_matched']} | {s['n_extreme_t3_total']} "
            f"| {s['ar_range']} |"
        )

    md += ["", "## D — Per-col ensemble cost/impact projection", ""]
    full = res_d["full_scope"]
    sel = res_d["selective_10pct_scope"]
    obs = res_d["observed_baseline"]
    md += [
        f"**Quan sát baseline** ({obs['debug_pages']} trang debug, "
        f"{obs['records']} record): {obs['promotions']} promotion "
        f"({obs['rate_pct']}%)",
        "",
        "| Scope | API calls | Runtime ước | Promotion dự kiến | Cost/promotion |",
        "|---|---:|---:|---:|---:|",
        f"| **Full** (toàn 9 cột/{res_d['n_pages_total']} trang) | "
        f"**{full['api_calls']:,}** | {full['est_runtime_min']} phút "
        f"| **{full['projected_promotions']:,}** | {full['cost_per_promotion']} call/promotion |",
        f"| **Selective ~10%** (cột flag count_ok=False / null ocr) | "
        f"{sel['api_calls']:,} | {sel['est_runtime_min']} phút | "
        f"~{int(full['projected_promotions']*0.1)} | (gần như free) |",
    ]

    md += ["", "## Tổng kết khuyến nghị", "",
           "| Đề xuất | Verdict | Lý do |",
           "|---|---|---|"]
    # A
    rate_a = res_a["n_would_promote_to_T1"]
    md.append(f"| **A — Q1 Diacritic** | "
              f"{'✅ Đáng làm' if rate_a > 500 else ('🟡 Vừa phải' if rate_a > 50 else '⚪ Tác động nhỏ')} | "
              f"{rate_a} promotion T1 trên 84k = "
              f"{100*rate_a/res_d['n_records']:.2f}% |")
    # B
    total_demote_b = sum(s["demote_total"] for s in res_b.values())
    md.append(f"| **B — F2' bucket threshold** | "
              f"{'⚠️ Cần gold set để chốt' if total_demote_b > 0 else '⚪'} | "
              f"Demote {total_demote_b:,} record (precision↑, recall↓) |")
    # C
    total_ext_c = sum(s["n_extreme_aspect"] for s in res_c.values())
    md.append(f"| **C — V2 sqpad** | "
              f"{'🟡 Đáng implement' if total_ext_c > 100 else '⚪'} | "
              f"{total_ext_c:,} candidate trên 3 sách (cần viết code rerank) |")
    # D
    md.append(f"| **D — Per-col ensemble full** | "
              f"⚠️ Đắt vs benefit | "
              f"{full['api_calls']:,} API call để vớt ~{full['projected_promotions']:,} promotion |")

    (OUT_DIR / "report.md").write_text("\n".join(md))

    # ── Console summary ──
    print("\n=== KẾT QUẢ ===")
    print(f"\n[A] Q1 Diacritic restore @ 84k:")
    print(f"    syl không trong dict: {res_a['n_syl_not_in_dict']:,}")
    print(f"    có thể khôi phục (≤1 dấu): {res_a['n_restorable_within_1_diacritic']:,}")
    print(f"    promotion → T1 dự kiến: {res_a['n_would_promote_to_T1']:,}")
    print(f"\n[B] F2' bucket threshold:")
    for b, s in res_b.items():
        print(f"    {b}: demote {s['demote_total']:>4} "
              f"({s['recall_drop_pp']:+.2f}pp recall)")
    print(f"\n[C] Extreme aspect candidates:")
    for b, s in res_c.items():
        print(f"    {b}: ext_aspect={s['n_extreme_aspect']:>4} "
              f"(T3 matched={s['n_extreme_t3_matched']})")
    print(f"\n[D] Per-col ensemble projection:")
    print(f"    Full scope: {full['api_calls']:,} calls → "
          f"~{full['projected_promotions']:,} promotion ({full['est_runtime_min']} min)")
    print(f"    Selective : {sel['api_calls']:,} calls ({sel['est_runtime_min']} min)")
    print(f"\nOutput: {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()
