"""Mẻ audit NGƯỜI cho tier GOLD — thiết kế HAI TẦNG tách bạch.

BỐI CẢNH
--------
Toàn bộ verdict đang có trong repo (846 hàng, `dataset_out/ground_truth/`) đều mang
`source: "ai_vision"` — tức MÁY chấm MÁY. Dùng chúng để đặt ngưỡng cho chính tín hiệu máy
là đúng lỗi tuần hoàn mà audit nội bộ đã cảnh báo. Mẻ này sinh ra tập verdict NGƯỜI đầu
tiên của dự án.

VÌ SAO PHẢI TÁCH LÀM HAI TẦNG
-----------------------------
Hai mục tiêu xung khắc nhau về mặt thống kê:

  1. ĐO precision GOLD  -> bắt buộc mẫu NGẪU NHIÊN. Mọi cách chọn "khôn" hơn đều làm
     ước lượng chệch lên phía lỗi và KHÔNG được phép báo cáo.
  2. HIỆU CHỈNH ngưỡng S3 -> muốn thật nhiều ca biên. Mẫu ngẫu nhiên gần như toàn hàng
     dễ, rất phí giờ công người chấm.

Trộn hai mục tiêu vào một mẫu là hỏng cả hai. Ở đây chúng được rút RIÊNG, gắn nhãn RIÊNG
bằng cột `audit_batch`, và chỉ tầng `srs` mới được vào công thức ước lượng precision:

  srs               ngẫu nhiên đơn giản từ GOLD; design_weight = N/n
                    -> Clopper–Pearson áp thẳng lên dân số. SỐ BÁO CÁO ĐƯỢC.
  active_lowmargin  các hàng `s3_head_margin` thấp nhất (đầu ArcFace nghi nhãn sai nhất).
                    CHỦ ĐÍCH, không phải mẫu xác suất -> design_weight = NaN.
                    CHỈ dùng để đo AUC / hiệu chỉnh ngưỡng. KHÔNG BAO GIỜ tính precision.

Người chấm KHÔNG phân biệt được hai tầng: `audit_order` được xáo trộn chung, và
`audit_batch` chỉ nằm trong manifest (đã có trong danh sách trường bị giấu của audit_grid).

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.make_gold_batch
    # -> dataset_out/ground_truth/audit_gold_human/audit_001.html + manifest.jsonl
    # chấm xong -> Download JSON -> lưu verdicts_001.jsonl NGAY trong thư mục đó
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

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO / "dataset_out" / "labels_remediated.csv"
DEFAULT_OUT = REPO / "dataset_out" / "ground_truth" / "audit_gold_human"

BATCH_SRS = "srs"
BATCH_ACTIVE = "active_lowmargin"


def _item_id(image: str) -> str:
    return hashlib.sha1(str(image).encode()).hexdigest()[:16]


def build_sample(
    gold: pd.DataFrame, n_srs: int, n_active: int, seed: int = 42
) -> pd.DataFrame:
    """Rút mẫu hai tầng từ khung GOLD đã gắn tín hiệu S3.

    Trả về một khung duy nhất có `audit_batch`, `design_weight`, `item_id`, `audit_order`.
    Hai tầng KHÔNG giao nhau: tầng chủ đích chỉ chọn trong phần còn lại sau khi rút SRS.
    """
    if n_srs <= 0 or n_active < 0:
        raise ValueError("n_srs phải > 0 và n_active >= 0")
    if len(gold) < n_srs + n_active:
        raise ValueError(f"dân số GOLD {len(gold)} nhỏ hơn n_srs+n_active "
                         f"{n_srs + n_active}")

    # --- tầng 1: SRS, mẫu xác suất, dùng để BÁO CÁO ------------------------- #
    srs = sampling.simple_random_sample(gold, n_srs, seed=seed)
    srs["audit_batch"] = BATCH_SRS

    # --- tầng 2: chủ đích theo margin thấp nhất, KHÔNG dùng để báo cáo ------- #
    rest = gold.drop(index=srs["source_row"].to_numpy(), errors="ignore")
    margin = s3_signals.audit_priority(rest)
    pool = rest.loc[margin.notna()].copy()
    if n_active and pool.empty:
        raise ValueError(
            "không có hàng nào mang `s3_head_margin` — chưa gắn tín hiệu S3 corpus-wide. "
            "Chạy `pipeline.consensus_fusion.score_s3 --all` trước.")
    if n_active > len(pool):
        raise ValueError(f"chỉ {len(pool)} hàng có margin, không đủ {n_active}")

    active = pd.DataFrame(columns=srs.columns)
    if n_active:
        # sort ổn định theo (margin, image) để cùng seed luôn ra cùng mẫu
        pool = pool.assign(_m=margin.loc[pool.index])
        pool = pool.sort_values(["_m", "image"], kind="stable").head(n_active)
        active = pool.drop(columns="_m").reset_index(drop=False).rename(
            columns={"index": "source_row"})
        active["item_id"] = active["image"].map(_item_id)
        active["design_weight"] = np.nan       # CHỦ ĐÍCH -> không có trọng số dân số
        active["audit_batch"] = BATCH_ACTIVE

    both = pd.concat([srs, active], ignore_index=True, sort=False)
    if both["item_id"].duplicated().any():
        raise AssertionError("hai tầng bị chồng lấn — item_id trùng")

    # --- LÀM MÙ: xáo trộn thứ tự hiển thị để người chấm không đoán được tầng -- #
    key = both["item_id"].map(lambda k: sampling._stable_key(k, seed))
    both = both.assign(_o=key).sort_values("_o", kind="stable").drop(columns="_o")
    both = both.reset_index(drop=True)
    both["audit_order"] = np.arange(len(both))
    return both


def _readme(n_srs: int, n_active: int, n_pop: int, conf: float) -> str:
    lcb0 = stats.cp_lower_bound(0, n_srs, conf)
    lcb2 = stats.cp_lower_bound(2, n_srs, conf)
    lcb5 = stats.cp_lower_bound(5, n_srs, conf)
    return f"""# Mẻ audit GOLD — verdict NGƯỜI đầu tiên

Tổng **{n_srs + n_active} ô**, chấm một buổi. Dân số GOLD = {n_pop:,} hàng.

## Vì sao mẻ này tồn tại

846 verdict đang có trong repo đều do MÁY chấm (`source: "ai_vision"`). Không thể dùng
chúng để đặt ngưỡng cho chính tín hiệu máy — đó là lập luận vòng tròn. Mẻ này tạo thước đo
độc lập đầu tiên.

## Cách chấm

1. Mở `audit_001.html` (và các phần sau nếu có) trong trình duyệt.
2. Mỗi ô hiện: crop + glyph tham chiếu + ngữ cảnh trang + âm QN + ứng viên từ điển.
   Bấm 1 trong 4:
   - **correct** — nhãn đúng với chữ trong ảnh
   - **wrong_label** — crop cắt đúng một chữ, nhưng nhãn gán sai chữ đó
   - **wrong_image** — crop cắt hỏng: dính 2 chữ, mất nét, hoặc nhầm ô
   - **unsure** — không đủ căn cứ để kết luận
   Tiến độ tự lưu trong trình duyệt.
3. Chấm HẾT rồi bấm **Download JSON**, lưu thành `verdicts_001.jsonl` **ngay trong thư
   mục này**.

Quan trọng: chấm theo đúng thứ tự hiện ra, **đừng bỏ ô khó**. Bỏ chọn lọc sẽ phá tính
ngẫu nhiên của tầng `srs` và làm hỏng ước lượng precision.

## Hai tầng — đọc kỹ trước khi dùng số

| Tầng | n | Cách rút | Được dùng để |
|------|--:|----------|--------------|
| `srs` | {n_srs} | Ngẫu nhiên đơn giản từ GOLD | **Ước lượng precision + CI. Đây là số báo cáo được.** |
| `active_lowmargin` | {n_active} | {n_active} hàng `s3_head_margin` thấp nhất | Đo AUC, hiệu chỉnh ngưỡng S3. **Tuyệt đối không tính precision.** |

Tầng `active_lowmargin` được chọn CHỦ ĐÍCH vì nghi ngờ cao, nên tỷ lệ lỗi trong đó cao hơn
hẳn dân số. Gộp nó vào phép tính precision sẽ cho ra con số thấp giả. `design_weight` của
tầng này để trống chính là để chặn việc gộp nhầm.

Người chấm không phân biệt được hai tầng: thứ tự hiển thị đã xáo trộn chung.

## Mẻ này đo được tới đâu (tầng `srs`, n = {n_srs}, độ tin cậy {conf:.0%})

| Số lỗi tìm thấy | Cận dưới precision |
|----------------:|-------------------:|
| 0 | {lcb0:.4f} |
| 2 | {lcb2:.4f} |
| 5 | {lcb5:.4f} |

Nói thẳng: n={n_srs} **không đủ** để chứng minh mệnh đề "GOLD precision 98%" — kể cả khi
không tìm thấy lỗi nào, cận dưới cũng chỉ đạt {lcb0:.1%}. Mẻ này là thước đo để chỉnh bước
3, **không phải** mẻ nghiệm thu cho luận văn. Mẻ nghiệm thu đầy đủ (n≈450-600) là việc của
giai đoạn audit chính thức.

## Sau khi chấm xong

```
.venv/bin/python -m pipeline.ground_truth estimate \\
    --verdicts {DEFAULT_OUT.relative_to(REPO)} \\
    --manifest {(DEFAULT_OUT / 'manifest.jsonl').relative_to(REPO)} \\
    --design srs --p0 0.97
```
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_gold_batch")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS),
                    help="mặc định labels_remediated.csv = tier GOLD như dataset xuất ra")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-srs", type=int, default=120, dest="n_srs")
    ap.add_argument("--n-active", type=int, default=80, dest="n_active")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels, dtype={"image_md5": str})
    labels, rep = s3_signals.attach(labels)
    print(f"[batch] {rep}")

    ranked = suspicion.add_suspicion(labels)
    gold = ranked[ranked["tier"] == "GOLD"].copy()
    if gold.empty:
        raise SystemExit("không có hàng GOLD nào trong khung")
    print(f"[batch] dân số GOLD = {len(gold):,}")

    sample = build_sample(gold, args.n_srs, args.n_active, seed=args.seed)
    n_srs = int((sample["audit_batch"] == BATCH_SRS).sum())
    n_act = int((sample["audit_batch"] == BATCH_ACTIVE).sum())
    print(f"[batch] mẫu {len(sample)} ô: srs={n_srs}, active_lowmargin={n_act}")

    cfg = _load_config(Path(args.config))
    paths = _paths(cfg)
    qn_dict = None
    qn_path = paths["qn_dict"]
    if qn_path.exists():
        try:
            qn = pd.read_csv(qn_path)
            cols = list(qn.columns)
            qn_dict = (qn.groupby(cols[0])[cols[1]].apply(list).to_dict()
                       if len(cols) >= 2 else None)
        except Exception as e:                       # noqa: BLE001 - chỉ là tiện ích hiển thị
            print(f"[batch] bỏ qua từ điển QN ({e})")

    stat = audit_grid.build_audit(
        sample,
        dataset_dir=paths["dataset_dir"],
        prepared_dir=paths["prepared_dir"],
        fd_dir=paths["fd_dir"],
        out_html=out_dir / "audit.html",
        out_manifest=out_dir / "manifest.jsonl",
        qn_dict=qn_dict,
        font_path=paths["font"],
        with_context=not args.no_context,
        title="Audit GOLD · verdict người",
        batch_size=args.batch_size,
    )
    print(f"[batch] {stat['items']} ô dựng xong "
          f"(thiếu glyph tham chiếu: {stat['missing_reference_glyph']}, "
          f"thiếu ngữ cảnh: {stat['missing_context']})")

    plan = {
        "labels_source": str(Path(args.labels).relative_to(REPO)),
        "population_gold": int(len(gold)),
        "seed": args.seed,
        "conf": args.conf,
        "strata": [
            {"audit_batch": BATCH_SRS, "n": n_srs, "probability_sample": True,
             "design_weight": float(len(gold)) / n_srs if n_srs else None,
             "usable_for": "precision + CI"},
            {"audit_batch": BATCH_ACTIVE, "n": n_act, "probability_sample": False,
             "design_weight": None,
             "usable_for": "AUC / hiệu chỉnh ngưỡng — KHÔNG dùng cho precision"},
        ],
        "cp_lower_bound_srs": {
            str(k): stats.cp_lower_bound(k, n_srs, args.conf) for k in (0, 1, 2, 5)
        },
        "grid": {k: v for k, v in stat.items() if k != "html"},
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        _readme(n_srs, n_act, len(gold), args.conf), encoding="utf-8")

    sample.to_csv(out_dir / "sample.csv", index=False)
    print(f"[batch] -> {out_dir}")
    print(f"[batch] mở {out_dir / 'audit_001.html'} để bắt đầu chấm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
