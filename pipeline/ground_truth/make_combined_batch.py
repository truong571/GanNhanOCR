"""Mẻ audit GỘP — GOLD + SILVER + SYLLABLE trong MỘT dòng chấm duy nhất.

VÌ SAO GỘP
----------
Chấm ba tier thành ba mẻ riêng có hai khuyết điểm không sửa được bằng thống kê:

  1. Người chấm BIẾT mình đang ở tier nào. Kỳ vọng "mẻ này là SILVER nên chắc nhiều lỗi"
     là thiên lệch kỳ vọng cổ điển, và nó tác động đúng vào đại lượng đang đo.
  2. Ba buổi chấm cách nhau nhiều ngày = ba tiêu chí khác nhau. Đây KHÔNG phải giả định:
     kiểm tra lặp 2026-08-04 đo được tỷ lệ lỗi trôi 4,2% -> 16% -> 35% qua ba buổi trên
     cùng một dân số, và precision GOLD dao động 95,8% <-> 84,0% chỉ vì đổi buổi.

Trộn chung rồi xáo trộn thì mọi tier chịu CÙNG một tiêu chí, cùng một trạng thái mệt mỏi,
cùng một buổi — nên chênh lệch precision giữa các tier là chênh lệch THẬT của dữ liệu chứ
không phải chênh lệch của người chấm. Đây chính là điều kiện để bảng so sánh tier trong
bài báo có nghĩa.

THIẾT KẾ MẪU — vì sao KHÔNG "chọn cho đa dạng"
----------------------------------------------
Đa dạng có hai loại, chỉ một loại là hợp lệ:

  * ĐA DẠNG HỢP LỆ = phân tầng có trọng số. Tầng = (tier x sách), phân bổ theo tỷ lệ dân
    số, mỗi ô rút NGẪU NHIÊN, và ghi `design_weight = N_h/n_h`. Bảo đảm cả 3 sách và cả 3
    tier đều có mặt mà ước lượng Horvitz-Thompson vẫn KHÔNG chệch.
  * ĐA DẠNG SAI = cố ý nhặt cho đủ mặt các lớp chữ hiếm / các ca khó. Việc đó làm hỏng
    chính con số precision: mẫu không còn đại diện cho dân số, và Clopper-Pearson tính
    trên nó là vô nghĩa. Muốn soi ca khó thì phải rút một mẻ CHỦ ĐÍCH riêng với
    `design_weight` rỗng — `estimate` tự loại nó khỏi mọi phép tính precision.

Nên mẻ này chỉ dùng loại thứ nhất, và ghi lại độ phủ thực tế (số lớp chữ, số trang, số
sách) vào `plan.json` để bài báo trích được mà không phải bịa.

Ô LẶP TRONG MẺ — đo độ tin cậy mà không cần thêm buổi chấm
----------------------------------------------------------
Một tỷ lệ nhỏ ô được đưa vào HAI LẦN, `item_id` khác nhau, cách nhau tối thiểu `--min-gap`
vị trí trong dòng chấm. Người chấm không thể biết ô nào là ô lặp. So verdict của hai lần
cho ra Cohen's kappa NỘI TẠI ngay trong chính mẻ này — tức bài báo có luôn số độ tin cậy
người chấm mà không phải tổ chức một buổi test-retest riêng.

Ô lặp mang `design_weight` rỗng nên `estimate._split_purposive` tự động LOẠI chúng khỏi
precision. Không có nguy cơ đếm hai lần.

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.make_combined_batch
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit_grid, s3_signals, sampling, stats, suspicion
from .cli import _load_config, _paths
from .make_confusion_batch import audited_images

REPO = Path(__file__).resolve().parents[2]
# Bộ nhãn CÔNG BỐ. Mặc định của make_label_batch là labels_remediated.csv — thế hệ TRƯỚC
# bước sửa nhầm lẫn — và rút mẫu từ đó cho ra con số mô tả một bộ dữ liệu chưa từng phát
# hành. Ở đây cố định vào labels_final.csv.
DEFAULT_LABELS = REPO / "dataset_out" / "labels_final.csv"
GT_DIR = REPO / "dataset_out" / "ground_truth"
DEFAULT_OUT = GT_DIR / "audit_combined"

# n mặc định mỗi tier. GOLD/SYLLABLE nhỏ hơn vì precision kỳ vọng cao (CI hẹp sẵn);
# SILVER lớn hơn vì precision kỳ vọng ~0,75 — vùng phương sai lớn nhất, cần n lớn hơn để
# đạt cùng độ rộng CI (nửa độ rộng +-5% tại p=0,75 cần n=287).
DEFAULT_N = {"GOLD": 250, "SILVER": 300, "SYLLABLE": 250}


def _allocate(sizes: dict[str, int], n_total: int, min_cell: int = 5) -> dict[str, int]:
    """Phân bổ n_total theo tỷ lệ dân số, làm tròn theo phần dư lớn nhất.

    Bảo đảm mỗi ô có ít nhất min_cell (hoặc toàn bộ dân số nếu ô quá nhỏ) để không sách
    nào vắng mặt, và tổng luôn khớp đúng n_total.
    """
    total = sum(sizes.values())
    if total == 0:
        return {k: 0 for k in sizes}
    n_total = min(n_total, total)
    exact = {k: n_total * v / total for k, v in sizes.items()}
    alloc = {k: min(sizes[k], max(min(min_cell, sizes[k]), int(v))) for k, v in exact.items()}

    # cân lại cho tổng khớp: thêm/bớt theo phần dư, không vượt dân số, không xuống dưới sàn
    def _room_up(k):
        return sizes[k] - alloc[k]

    def _room_down(k):
        return alloc[k] - min(min_cell, sizes[k])

    diff = n_total - sum(alloc.values())
    order = sorted(sizes, key=lambda k: (-(exact[k] - int(exact[k])), k))
    while diff > 0:
        moved = False
        for k in order:
            if diff <= 0:
                break
            if _room_up(k) > 0:
                alloc[k] += 1
                diff -= 1
                moved = True
        if not moved:
            break
    while diff < 0:
        moved = False
        for k in reversed(order):
            if diff >= 0:
                break
            if _room_down(k) > 0:
                alloc[k] -= 1
                diff += 1
                moved = True
        if not moved:
            break
    return alloc


def draw_tier(pool: pd.DataFrame, tier: str, n: int, seed: int,
              min_cell: int = 5) -> tuple[pd.DataFrame, list[dict]]:
    """Rút n hàng của một tier, phân tầng theo SÁCH với phân bổ tỷ lệ.

    Trả về (mẫu, mô tả các tầng). Mỗi hàng mang:
      stratum       "<TIER>|<sách>"  -> estimate nhóm theo đây
      design_weight N_h / n_h        -> Horvitz-Thompson quy về dân số
    """
    sizes = pool.groupby("book").size().to_dict()
    alloc = _allocate({str(k): int(v) for k, v in sizes.items()}, n, min_cell)

    parts, cells = [], []
    for book, n_h in sorted(alloc.items()):
        if n_h <= 0:
            continue
        cell = pool[pool["book"].astype(str) == book]
        N_h = len(cell)
        # seed riêng từng ô nhưng dẫn xuất tất định từ seed gốc -> tái lập được
        s = int(hashlib.sha1(f"{seed}:{tier}:{book}".encode()).hexdigest()[:8], 16) % (2**31)
        drawn = sampling.simple_random_sample(cell, n_h, seed=s)
        drawn["stratum"] = f"{tier}|{book}"
        drawn["design_weight"] = float(N_h) / float(n_h)
        # Dân số THẬT của tầng, mang thẳng vào manifest. Suy N_h ngược từ
        # design_weight * n_h là sai khi có ô "không đọc được": n_h co lại thì N_h suy ra
        # cũng co theo, và trọng số giữa các tầng lệch đi. Ghi sẵn thì không suy gì cả.
        drawn["stratum_N"] = int(N_h)
        parts.append(drawn)
        cells.append({"stratum": f"{tier}|{book}", "book": book,
                      "N_h": N_h, "n_h": n_h, "design_weight": round(N_h / n_h, 4)})

    sample = pd.concat(parts, ignore_index=True)
    sample["tier_audit"] = tier
    sample["audit_batch"] = f"combined_{tier.lower()}"
    return sample, cells


def add_repeats(sample: pd.DataFrame, n_repeat: int, seed: int) -> pd.DataFrame:
    """Nhân bản n_repeat ô thành các mục CHẤM LẠI với item_id mới.

    Phân bổ theo tier để kappa phủ cả ba tier. Ô lặp có `design_weight` rỗng nên
    `estimate` tự loại khỏi precision; `repeat_of` trỏ về item_id gốc để ghép cặp.
    """
    if n_repeat <= 0:
        return sample.iloc[0:0].copy()
    per_tier = _allocate(sample.groupby("tier_audit").size().to_dict(), n_repeat, min_cell=1)
    picks = []
    for tier, k in sorted(per_tier.items()):
        g = sample[sample["tier_audit"] == tier]
        s = int(hashlib.sha1(f"{seed}:repeat:{tier}".encode()).hexdigest()[:8], 16) % (2**31)
        idx = np.random.default_rng(s).choice(g.index.to_numpy(), size=min(k, len(g)),
                                              replace=False)
        picks.append(g.loc[idx])
    rep = pd.concat(picks, ignore_index=True).copy()
    rep["repeat_of"] = rep["item_id"]
    rep["item_id"] = rep["item_id"].map(
        lambda v: hashlib.sha1(f"repeat::{v}".encode()).hexdigest()[:16])
    rep["stratum"] = "__repeat__"
    rep["design_weight"] = np.nan          # -> estimate LOẠI khỏi mọi phép tính precision
    rep["audit_batch"] = "repeat"
    return rep


def interleave(sample: pd.DataFrame, repeats: pd.DataFrame, seed: int,
               min_gap: int) -> pd.DataFrame:
    """Trộn bản gốc + ô lặp thành MỘT dòng chấm, ép ô lặp cách bản gốc >= min_gap.

    Xáo trộn bằng khoá băm tất định (cùng seed -> cùng thứ tự). Sau khi xáo, ô lặp nào
    nằm quá gần bản gốc thì hoán đổi với ô xa nhất còn hợp lệ — nếu không, người chấm
    nhận ra ngay là ô trùng và chỉ chép lại lựa chọn vừa bấm, làm kappa vô giá trị.
    """
    allrows = pd.concat([sample, repeats], ignore_index=True)
    key = allrows["item_id"].map(
        lambda k: int(hashlib.sha1(f"{seed}:order:{k}".encode()).hexdigest()[:12], 16))
    allrows = allrows.assign(_k=key).sort_values("_k").drop(columns="_k")
    allrows = allrows.reset_index(drop=True)

    pos = {iid: i for i, iid in enumerate(allrows["item_id"])}
    order = list(allrows["item_id"])
    rep_pairs = [(r.item_id, r.repeat_of) for r in
                 allrows.loc[allrows["stratum"] == "__repeat__"].itertuples()]

    for rid, oid in rep_pairs:
        if oid not in pos:
            continue
        if abs(pos[rid] - pos[oid]) >= min_gap:
            continue
        # tìm vị trí xa cả hai để hoán đổi; quét từ đầu/cuối vào giữa cho tất định
        n = len(order)
        target = None
        for cand in list(range(n - 1, -1, -1)) if pos[oid] < n // 2 else list(range(n)):
            other = order[cand]
            if other in (rid, oid):
                continue
            if abs(cand - pos[oid]) < min_gap:
                continue
            # ô được đổi chỗ không được vô tình phạm khoảng cách của chính nó
            if other in dict(rep_pairs):
                its_orig = dict(rep_pairs)[other]
                if its_orig in pos and abs(pos[rid] - pos[its_orig]) < min_gap:
                    continue
            target = cand
            break
        if target is None:
            continue
        a, b = pos[rid], target
        order[a], order[b] = order[b], order[a]
        pos[order[a]], pos[order[b]] = a, b

    allrows = allrows.set_index("item_id").loc[order].reset_index()
    allrows["audit_order"] = np.arange(len(allrows))
    return allrows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_combined_batch")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-gold", type=int, default=DEFAULT_N["GOLD"])
    ap.add_argument("--n-silver", type=int, default=DEFAULT_N["SILVER"])
    ap.add_argument("--n-syllable", type=int, default=DEFAULT_N["SYLLABLE"])
    ap.add_argument("--n-repeat", type=int, default=60,
                    help="số ô đưa vào hai lần để đo kappa nội tại (0 = tắt)")
    ap.add_argument("--min-gap", type=int, default=200,
                    help="khoảng cách tối thiểu giữa ô lặp và bản gốc trong dòng chấm")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = MỘT file duy nhất; >0 = cắt thành nhiều phần")
    ap.add_argument("--reuse-audited", action="store_true",
                    help="cho phép lấy lại ô đã chấm ở mẻ trước (mặc định: loại)")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels, dtype={"image_md5": str})
    try:
        labels, rep = s3_signals.attach(labels)
        print(f"[gộp] {rep}")
    except FileNotFoundError:
        print("[gộp] không có s3_corpus — mẻ này không cần tín hiệu S3")
    ranked = suspicion.add_suspicion(labels)

    already = set() if args.reuse_audited else audited_images(GT_DIR)
    wanted = {"GOLD": args.n_gold, "SILVER": args.n_silver, "SYLLABLE": args.n_syllable}

    samples, cells, tiers_meta = [], [], []
    for tier, n in wanted.items():
        if n <= 0:
            continue
        pool = ranked[ranked["tier"] == tier]
        free = pool[~pool["image"].astype(str).isin(already)]
        if len(free) < n:
            raise SystemExit(f"{tier}: chỉ còn {len(free)} ô chưa chấm, không đủ {n}")
        s, c = draw_tier(free, tier, n, args.seed)
        print(f"[gộp] {tier}: dân số {len(pool):,} (chưa chấm {len(free):,}) -> n={len(s)} "
              f"trên {len(c)} sách")
        samples.append(s)
        cells.extend(c)
        tiers_meta.append({
            "tier": tier, "N": int(len(pool)), "N_unaudited": int(len(free)),
            "n": int(len(s)),
            "cp_lower_bound": {str(k): stats.cp_lower_bound(len(s) - k, len(s), args.conf)
                               for k in (0, 1, 2, 3, 5, 8, 12) if k <= len(s)},
        })

    sample = pd.concat(samples, ignore_index=True)
    repeats = add_repeats(sample, args.n_repeat, args.seed)
    allrows = interleave(sample, repeats, args.seed, args.min_gap)
    print(f"[gộp] tổng {len(allrows)} ô chấm = {len(sample)} ô mẫu + {len(repeats)} ô lặp")

    # độ phủ thực tế — số liệu mô tả cho bài báo, KHÔNG dùng để chọn mẫu
    lab = sample["label"].dropna().astype(str)
    coverage = {
        "distinct_labels": int(lab[lab.str.len() > 0].nunique()),
        "distinct_pages": int(sample.groupby(["book", "page"]).ngroups),
        "books": sorted(sample["book"].astype(str).unique().tolist()),
        "splits": {str(k): int(v) for k, v in sample["split"].value_counts().items()}
        if "split" in sample.columns else {},
    }
    print(f"[gộp] độ phủ: {coverage['distinct_labels']} lớp chữ · "
          f"{coverage['distinct_pages']} trang · {len(coverage['books'])} sách")

    cfg = _load_config(Path(args.config))
    paths = _paths(cfg)
    qn_dict = None
    try:
        from core.text.dictionary import load_qn_to_nom
        if paths["qn_dict"].exists():
            qn_dict = load_qn_to_nom(str(paths["qn_dict"]))
    except Exception as e:                                   # noqa: BLE001
        print(f"[gộp] bỏ qua từ điển QN ({e})")

    stat = audit_grid.build_audit(
        allrows,
        dataset_dir=paths["dataset_dir"],
        prepared_dir=paths["prepared_dir"],
        fd_dir=paths["fd_dir"],
        out_html=out_dir / "audit.html",
        out_manifest=out_dir / "manifest.jsonl",
        qn_dict=qn_dict,
        font_path=paths["font"],
        with_context=not args.no_context,
        title="Audit nhãn · mẻ gộp",
        batch_size=args.batch_size or None,
        mode="label_only",
    )
    print(f"[gộp] {stat['items']} ô dựng xong (thiếu glyph: "
          f"{stat['missing_reference_glyph']}, thiếu ngữ cảnh: {stat['missing_context']})")

    plan = {
        "design": "stratified (tier x sách), phân bổ tỷ lệ, SRS trong từng ô",
        "labels_source": (str(Path(args.labels).relative_to(REPO))
                          if Path(args.labels).is_absolute()
                          and Path(args.labels).is_relative_to(REPO) else str(args.labels)),
        "mode": "label_only",
        "seed": args.seed, "conf": args.conf,
        "n_sample": int(len(sample)), "n_repeat": int(len(repeats)),
        "n_items_total": int(len(allrows)),
        "min_gap_repeat": args.min_gap,
        "excluded_already_audited": (0 if args.reuse_audited else len(already)),
        "tiers": tiers_meta,
        "strata": cells,
        "coverage": coverage,
        "estimation": ("pipeline.ground_truth.report_combined — precision + CI theo TỪNG "
                       "tier (Wilson/Clopper-Pearson trên các ô của tier đó), tổng hợp "
                       "Horvitz-Thompson có FPC, và kappa nội tại từ các ô lặp."),
        "note": ("Ô lặp mang design_weight rỗng nên estimate._split_purposive tự loại khỏi "
                 "mọi phép tính precision — không có nguy cơ đếm hai lần."),
        "grid": {k: v for k, v in stat.items() if k != "html"},
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(plan, out_dir), encoding="utf-8")
    print(f"[gộp] -> {out_dir}")
    return 0


def _readme(plan: dict, out_dir: Path) -> str:
    rows = []
    for t in plan["tiers"]:
        n = t["n"]
        lines = []
        for k in (0, 1, 2, 3, 5, 8):
            if str(k) not in t["cp_lower_bound"]:
                continue
            lo, hi = stats.clopper_pearson_ci(n - k, n, plan["conf"])
            lines.append(f"| {k} | {(n - k) / n:.1%} | [{lo:.1%} · {hi:.1%}] | "
                         f"{t['cp_lower_bound'][str(k)]:.1%} |")
        rows.append(f"""
### {t['tier']} — n = {n} trên dân số {t['N']:,}

| Số ô sai nhãn | Precision | CI95 (Clopper–Pearson) | Cận dưới một phía |
|--------------:|----------:|------------------------|------------------:|
{chr(10).join(lines)}
""")
    strata = "\n".join(
        f"| {c['stratum']} | {c['N_h']:,} | {c['n_h']} | {c['design_weight']} |"
        for c in plan["strata"])
    rel = out_dir.relative_to(REPO) if out_dir.is_relative_to(REPO) else out_dir
    return f"""# Mẻ audit GỘP — GOLD + SILVER + SYLLABLE, một dòng chấm duy nhất

**{plan['n_items_total']} ô** trong **một file** `audit.html`
= {plan['n_sample']} ô mẫu + {plan['n_repeat']} ô lặp ẩn.
Nguồn nhãn: `{plan['labels_source']}` (bộ **công bố**).

## Chỉ có MỘT câu hỏi

> **Nhãn hiện trên thẻ có đúng là chữ viết trong ô không?**

| Phím | Nghĩa |
|:---:|---|
| **1** | **nhãn ĐÚNG** — chữ trong ô đúng là chữ được gán |
| **2** | **nhãn SAI** — chữ trong ô là một chữ KHÁC |
| **3** | **không đọc được** — không đủ căn cứ, kể cả sau khi xem ảnh ngữ cảnh |

Lựa chọn "sai ảnh" đã bị **bỏ hẳn**. Kiểm tra lặp đo được κ = 0,14 cho phán đoán "crop có
sạch không" — không tái lập được, và chính nó làm precision GOLD nhảy 95,8% ↔ 84,0% giữa
hai buổi. Chất lượng khung cắt nay đo bằng hình học trên toàn bộ 69.440 crop (chỉ 14 ô =
0,02% hỏng kết cấu thật).

**Bốn quy tắc:**

1. Khung cắt xấu **không phải** lỗi nhãn. Dính chút mực chữ bên cạnh mà vẫn đọc ra chữ →
   bấm **1**. Câu hỏi là về **chữ**, không phải về **khung**.
2. Crop khó nhìn thì đọc bằng **ảnh ngữ cảnh** (khung đỏ trên trang scan).
3. Lưỡng lự → bấm **3**, đừng bấm **2**. Ô *không đọc được* bị loại khỏi mẫu số; ô
   *nhãn SAI* bị tính là lỗi. Dữ liệu cũ cho thấy xu hướng **gọi quá tay**: 5/6 lần bấm
   "sai nhãn" thì lần sau chính bạn đảo lại thành đúng.
4. Chấm **hết** theo đúng thứ tự, không bỏ ô khó — bỏ chọn lọc phá tính ngẫu nhiên và
   làm mọi khoảng tin cậy mất hiệu lực.

## Có ô KHÔNG có glyph tham chiếu — đó là tier âm tiết

Một phần ô hiện dấu **?** ở ô chữ lớn và không có ảnh "glyph tham chiếu". Đó là các hàng
mang nhãn **âm tiết Quốc ngữ** chứ không phải một chữ Nôm cụ thể. Với những ô này câu hỏi
đổi thành:

> **Chữ Nôm trong ô này có đọc là âm tiết ‹âm› hiển thị trên thẻ không?**

Dùng dòng **ứng viên** (các chữ Nôm mà từ điển gắn với âm đó) làm căn cứ. Vẫn ba phím như
trên. Đừng bấm **2** chỉ vì không có glyph để so — không đủ căn cứ thì bấm **3**.

## Ba tier trộn chung và xáo trộn — cố ý

Bạn **không** biết ô đang chấm thuộc tier nào, và điều đó là chủ đích: chấm riêng từng
tier thì kỳ vọng "mẻ này chắc nhiều lỗi" trở thành thiên lệch tác động thẳng vào con số
đang đo. Trộn chung còn bảo đảm cả ba tier chịu **cùng một tiêu chí trong cùng một buổi**,
nên bảng so sánh giữa các tier mới có nghĩa.

Có một số ô **xuất hiện hai lần**, cách nhau ít nhất {plan['min_gap_repeat']} vị trí. Đừng
tìm chúng và đừng cố nhớ đã bấm gì — chúng dùng để đo độ ổn định của chính bạn. **Đổi ý là
dữ liệu quý, không phải lỗi.**

## Cách chấm

1. Mở `audit.html` (một file duy nhất, mọi ảnh nhúng sẵn, không cần mạng).
2. Bấm phím **1** / **2** / **3**, hoặc bấm chuột. `←` `→` chuyển ô. Tiến độ tự lưu vào
   trình duyệt — đóng tab rồi mở lại vẫn còn.
3. **Không chấm quá ~150 ô mỗi buổi.** Trôi tiêu chí đã đo được: 4,2% → 16% → 35% qua ba
   buổi liên tiếp. Nghỉ rồi quay lại, tiến độ vẫn giữ.
4. Chấm xong **toàn bộ** → bấm **Xuất verdicts.jsonl** → lưu file vào **chính thư mục này**
   (`{rel}/verdicts.jsonl`).

## Phân tầng (design_weight = N_h / n_h)

| Tầng | N_h | n_h | design_weight |
|---|---:|---:|---:|
{strata}

Độ phủ thực tế của mẫu: **{plan['coverage']['distinct_labels']} lớp chữ phân biệt** ·
**{plan['coverage']['distinct_pages']} trang** · **{len(plan['coverage']['books'])} sách**.

## Mẻ này siết được khoảng tin cậy tới đâu
{''.join(rows)}
## Sau khi chấm

```
.venv/bin/python -m pipeline.ground_truth.report_combined --dir {rel}
```

Sinh ra `report.json` + `BANG_KET_QUA.md` — bảng precision/CI theo từng tier, tổng hợp
Horvitz–Thompson, và κ nội tại từ các ô lặp; dán thẳng vào luận văn.
"""


if __name__ == "__main__":
    raise SystemExit(main())
