"""Chấm điểm verdict người kiểm mù (đầu ra của kiem_nguoi_grid.py).

Đọc <out>/items.csv (khoá: machine_label, tier) + một hay nhiều CSV verdict do index.html
xuất (item_id, verdict, choice_index, reviewer, ts). Tính:

  GOLD      precision = #(verdict == machine_label) / #đã chấm, Wilson 95 % CI
            (bảo thủ: "Không chắc" tính là KHÔNG đúng; báo thêm bản loại "Không chắc").
            Lý do sai tách riêng: chọn ứng viên khác · không có trong danh sách · crop sai
            chữ · không chắc.
  SYLLABLE/ REVIEW  tỷ lệ xác nhận ÂM = chọn một chữ cùng âm trong từ điển (không kể chữ OCR
            ngoài từ điển) + Wilson CI; tách: chọn chữ OCR (lỗ từ điển) · không có · crop sai · không chắc.

Mẫu phân tầng theo trang với phân bổ tỷ lệ → tự trọng số: tỷ lệ mẫu = ước lượng tổng thể.
Nhiều người kiểm cùng chấm: dùng verdict MỚI NHẤT theo ts cho mỗi ô; nếu ≥2 người trùng
≥10 ô thì in thêm % đồng thuận + Cohen κ trên các ô trùng.

Chạy:
  .venv/bin/python pipeline/tools/kiem_nguoi_score.py --items dataset_LucVanTien1883/kiem_nguoi/items.csv \
      --verdicts ~/Downloads/verdicts_A_2026-09-25.csv [--out .../kiem_nguoi/score.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:  # cùng công thức với pipeline.ground_truth (scipy); dự phòng thuần math
    from pipeline.ground_truth.stats import cohens_kappa, wilson_ci
except Exception:  # pragma: no cover
    cohens_kappa = None

    def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
        if n <= 0:
            return (float("nan"), float("nan"))
        z = 1.959963984540054 if abs(conf - 0.95) < 1e-9 else 2.5758293035489
        p = k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return (max(0.0, c - h), min(1.0, c + h))

SPECIAL = ("KHONG_CO", "CROP_SAI", "KHONG_CHAC")


def read_verdicts(paths: list[str]) -> list[dict]:
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files += sorted(pp.glob("verdict*.csv"))
        elif pp.exists():
            files.append(pp)
        else:
            sys.exit(f"không thấy {pp}")
    rows = []
    for f in files:
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("item_id") and r.get("verdict") is not None:
                    r["_file"] = f.name
                    rows.append(r)
    return rows


def classify(item: dict, verdict: str) -> str:
    """Nhãn lý do cho MỘT ô (đã chấm)."""
    if verdict in SPECIAL:
        return {"KHONG_CO": "khong_co_trong_ds", "CROP_SAI": "crop_sai", "KHONG_CHAC": "khong_chac"}[verdict]
    machine = item["machine_label"]
    if item["tier"] == "GOLD":
        return "dung" if verdict == machine else "sai_chon_ung_vien_khac"
    # tier chỉ khẳng định ÂM: ứng viên từ điển cùng âm = xác nhận âm; chữ OCR ngoài từ điển = lỗ từ điển
    cands = item["candidates"].split("|")
    srcs = item["cand_sources"].split("|")
    src = dict(zip(cands, srcs)).get(verdict, "?")
    if src == "dict":
        return "xac_nhan_am"
    if src == "machine":
        return "xac_nhan_am" if item.get("machine_in_dict") == "1" else "chon_chu_ocr_ngoai_tu_dien"
    return "chon_chu_dem"  # similar/random pad — không cùng âm


def summarize(items: dict[str, dict], latest: dict[str, dict]) -> dict:
    out: dict = {}
    by_tier: dict[str, list[str]] = defaultdict(list)
    for iid, it in items.items():
        by_tier[it["tier"]].append(iid)
    for tier, ids in sorted(by_tier.items()):
        reasons = Counter()
        graded = 0
        wrong_pages = Counter()
        for iid in ids:
            v = latest.get(iid)
            if not v:
                reasons["chua_cham"] += 1
                continue
            graded += 1
            c = classify(items[iid], v["verdict"])
            reasons[c] += 1
            if c not in ("dung", "xac_nhan_am"):
                wrong_pages[items[iid]["page"]] += 1
        ok_key = "dung" if tier == "GOLD" else "xac_nhan_am"
        k = reasons[ok_key]
        unsure = reasons["khong_chac"]
        res = {"n_sampled": len(ids), "n_graded": graded, "n_correct": k,
               "reasons": dict(sorted(reasons.items())),
               "wrong_pages_top": wrong_pages.most_common(5)}
        if graded:
            lo, hi = (float(x) for x in wilson_ci(k, graded))
            res["precision" if tier == "GOLD" else "syllable_confirm_rate"] = round(k / graded, 4)
            res["wilson95"] = [round(lo, 4), round(hi, 4)]
            if unsure and graded - unsure > 0:
                lo2, hi2 = (float(x) for x in wilson_ci(k, graded - unsure))
                res["excl_unsure"] = {"n": graded - unsure, "rate": round(k / (graded - unsure), 4),
                                      "wilson95": [round(lo2, 4), round(hi2, 4)]}
        out[tier] = res
    return out


def agreement(verdicts: list[dict], items: dict[str, dict]) -> dict | None:
    by_rev: dict[str, dict[str, str]] = defaultdict(dict)
    for v in verdicts:
        by_rev[v.get("reviewer") or v["_file"]][v["item_id"]] = v["verdict"]
    revs = sorted(by_rev)
    if len(revs) < 2:
        return None
    out = {}
    for i in range(len(revs)):
        for j in range(i + 1, len(revs)):
            common = sorted(set(by_rev[revs[i]]) & set(by_rev[revs[j]]) & set(items))
            if len(common) < 10:
                continue
            pairs = [(by_rev[revs[i]][c], by_rev[revs[j]][c]) for c in common]
            agree = sum(1 for a, b in pairs if a == b) / len(pairs)
            kap = None
            if cohens_kappa is not None:
                try:
                    kap = cohens_kappa(pairs).get("kappa")
                except Exception:
                    kap = None
            out[f"{revs[i]} vs {revs[j]}"] = {"n_common": len(common), "agree": round(agree, 4),
                                              "kappa": None if kap is None else round(float(kap), 4)}
    return out or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--verdicts", nargs="+", required=True, help="CSV verdict (một hay nhiều) hoặc thư mục")
    ap.add_argument("--out", default=None, help="score.json (mặc định cạnh items.csv)")
    a = ap.parse_args(argv)

    items_path = Path(a.items)
    items = {}
    for r in csv.DictReader(open(items_path, encoding="utf-8-sig", newline="")):
        if "machine_in_dict" not in r:  # items.csv đời cũ
            r["machine_in_dict"] = "1" if r["machine_label_kind"] == "label" else "0"
        items[r["item_id"]] = r
    verdicts = read_verdicts(a.verdicts)
    unknown = [v["item_id"] for v in verdicts if v["item_id"] not in items]
    latest: dict[str, dict] = {}
    for v in sorted(verdicts, key=lambda v: v.get("ts", "")):
        if v["item_id"] in items:
            latest[v["item_id"]] = v
    bad = [v for v in latest.values() if v["verdict"] not in SPECIAL
           and v["verdict"] not in items[v["item_id"]]["candidates"].split("|")]

    report = {
        "items_csv": str(items_path), "n_items": len(items), "n_verdict_rows": len(verdicts),
        "n_graded": len(latest), "n_unknown_item_id": len(unknown), "n_verdict_not_a_candidate": len(bad),
        "reviewers": sorted({v.get("reviewer") or v["_file"] for v in verdicts}),
        "tiers": summarize(items, latest),
        "agreement": agreement(verdicts, items),
        "note": "Mẫu phân tầng theo trang, phân bổ tỷ lệ → tự trọng số. GOLD: 'Không chắc' tính là sai (bảo thủ). "
                "Ứng viên chỉ hiện ≤5 chữ: 'không có trong danh sách' = nhãn máy sai nhưng không định vị được chữ đúng.",
    }
    out = Path(a.out) if a.out else items_path.with_name("score.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"items {len(items)} · verdict rows {len(verdicts)} · đã chấm {len(latest)} · id lạ {len(unknown)} · verdict ngoài ứng viên {len(bad)}")
    for tier, r in report["tiers"].items():
        rate_key = "precision" if tier == "GOLD" else "syllable_confirm_rate"
        if r["n_graded"]:
            ex = r.get("excl_unsure", {})
            print(f"[{tier}] {rate_key} = {r['n_correct']}/{r['n_graded']} = {r[rate_key]:.3f}  Wilson95 {r['wilson95']}"
                  + (f"  | loại không-chắc: {ex.get('rate'):.3f} {ex.get('wilson95')}" if ex else ""))
        else:
            print(f"[{tier}] chưa có verdict ({r['n_sampled']} ô)")
        print(f"   lý do: {r['reasons']}")
        if r["wrong_pages_top"]:
            print(f"   trang nhiều lỗi: {r['wrong_pages_top']}")
    if report["agreement"]:
        print("đồng thuận:", report["agreement"])
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
