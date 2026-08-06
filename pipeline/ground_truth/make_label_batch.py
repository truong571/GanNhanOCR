"""Mẻ audit CHỈ HỎI VỀ NHÃN — mẫu SRS để siết khoảng tin cậy của precision nhãn.

VÌ SAO CHỈ HỎI MỘT CÂU
----------------------
Công cụ chấm 4 mức trộn hai câu hỏi khác hẳn nhau vào một lần bấm. Kiểm tra lặp
2026-08-04 (40 ô, `audit_retest/`) tách ra thì thấy rõ hai chiều hành xử ngược nhau:

  chiều NHÃN (nhãn có đúng chữ không)   κ = 0,184 — **0/20 báo động giả**
  chiều CROP (crop có sạch không)       κ = 0,140 — 8 gắn mới / 6 gỡ bỏ trên 40 ô

Chính chiều CROP là nguồn bất ổn, và nó kéo precision GOLD dao động 95,8% ↔ 84,0% giữa
hai buổi chấm trên cùng một dân số. Nay chất lượng crop được ĐO bằng hình học
(`crop_bleed.py`: chỉ 14/69.440 = 0,02% crop hỏng kết cấu) nên **không cần hỏi người nữa**.

Mẻ này bỏ hẳn lựa chọn `wrong_image`. Crop bẩn KHÔNG còn là một verdict — nếu crop khó
đọc thì người chấm dùng ảnh ngữ cảnh (có khung bbox trên trang gốc) để đọc chữ, và chỉ
chọn "không đọc được" khi thật sự không đọc nổi.

THIẾT KẾ
--------
SRS thuần từ GOLD, loại các ô đã chấm ở mẻ trước (tránh nhiễm trí nhớ). design_weight =
N/n nên Clopper–Pearson áp thẳng lên dân số — mẻ này ĐỌC ĐƯỢC như một tuyên bố precision.

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.make_label_batch --n 250
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import audit_grid, s3_signals, sampling, stats, suspicion
from .cli import _load_config, _paths
from .make_confusion_batch import audited_images

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO / "dataset_out" / "labels_remediated.csv"
GT_DIR = REPO / "dataset_out" / "ground_truth"
DEFAULT_OUT = GT_DIR / "audit_label_only"


def _readme(n: int, n_pop: int, conf: float, n_excluded: int) -> str:
    rows = []
    for k in (0, 1, 2, 3, 5, 8):
        lo, hi = stats.clopper_pearson_ci(n - k, n, conf)
        rows.append(f"| {k} | {(n - k) / n:.1%} | [{lo:.1%} · {hi:.1%}] | "
                    f"{stats.cp_lower_bound(n - k, n, conf):.1%} |")
    table = "\n".join(rows)
    return f"""# Mẻ audit NHÃN — chỉ một câu hỏi

**{n} ô**, mẫu ngẫu nhiên đơn giản từ {n_pop:,} hàng GOLD chưa từng được chấm
({n_excluded} ô đã chấm ở các mẻ trước đã bị loại).

## Chỉ có MỘT câu hỏi

> **Nhãn được gán có đúng là chữ viết trong ô này không?**

Ba lựa chọn, không có lựa chọn nào khác:

- **1 · nhãn ĐÚNG** — chữ trong ô đúng là chữ được gán
- **2 · nhãn SAI** — chữ trong ô là một chữ KHÁC
- **3 · không đọc được** — không đủ căn cứ để kết luận

## Điều đã đổi so với các mẻ trước — đọc kỹ

Trước đây có thêm lựa chọn "sai ảnh" cho crop cắt hỏng. **Lựa chọn đó đã bỏ.**

Lý do: kiểm tra lặp cho thấy phán đoán "crop có sạch không" của con người không tái lập
được (κ = 0,14 — chính bạn đảo verdict trên 14/40 ô). Nó cũng chính là thứ làm precision
nhảy từ 95,8% xuống 84,0% giữa hai buổi. Nay chất lượng crop được đo bằng hình học trên
toàn bộ 69.440 crop, và kết quả là chỉ **14 ô (0,02%)** hỏng kết cấu thật.

**Vì vậy: crop bẩn KHÔNG còn là một verdict.**

- Crop dính chút mực của chữ bên cạnh mà vẫn đọc ra chữ → **nhãn ĐÚNG** (nếu nhãn khớp).
- Crop cắt khó nhìn → dùng **ảnh ngữ cảnh** (khung đỏ trên trang gốc) để đọc chữ.
- Chỉ chọn **không đọc được** khi thật sự không đọc nổi chữ đó là gì, kể cả khi nhìn
  ảnh ngữ cảnh.

Đừng phạt một nhãn đúng chỉ vì khung cắt xấu. Câu hỏi ở đây là về **chữ**, không phải về
khung.

## Một lưu ý từ dữ liệu trước

Ở kiểm tra lặp, 5 trên 6 lần bạn bấm "sai nhãn" thì lần sau chính bạn đổi lại thành đúng —
tức xu hướng là **gọi quá tay**. Nếu lưỡng lự giữa "sai nhãn" và "đúng", hãy nhìn kỹ ảnh
ngữ cảnh và bộ ứng viên từ điển trước khi quyết. Vẫn lưỡng lự thì chọn **không đọc được**
chứ đừng chọn "sai nhãn" — ô không đọc được bị loại khỏi phép tính, còn "sai nhãn" thì bị
tính là lỗi.

## Cách chấm

1. Mở `audit_001.html` (và các phần sau).
2. Bấm phím **1** / **2** / **3**, hoặc bấm chuột. Tiến độ tự lưu.
3. Chấm hết MỌI phần rồi bấm **Download JSON** → lưu `verdicts_001.jsonl` ngay trong
   thư mục này.

Chấm hết theo đúng thứ tự, đừng bỏ ô khó — bỏ chọn lọc sẽ phá tính ngẫu nhiên của mẫu.

## Mẻ này siết được CI tới đâu (n = {n}, độ tin cậy {conf:.0%})

| Số ô sai nhãn | Precision | CI95 | Cận dưới một phía |
|--------------:|----------:|------|------------------:|
{table}

So với mẻ trước (n=116, CI [93,9% · 99,8%]), mẻ này thu hẹp khoảng tin cậy khoảng một nửa.

## Sau khi chấm

```
.venv/bin/python -m pipeline.ground_truth --out {DEFAULT_OUT.relative_to(REPO)} estimate \\
    --verdicts {DEFAULT_OUT.relative_to(REPO)} \\
    --manifest {(DEFAULT_OUT / 'manifest.jsonl').relative_to(REPO)} \\
    --design srs --p0 0.97
```
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_label_batch")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tier", default="GOLD")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=125)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels, dtype={"image_md5": str})
    try:
        labels, rep = s3_signals.attach(labels)
        print(f"[label] {rep}")
    except FileNotFoundError:
        print("[label] không có s3_corpus — mẻ này không cần tín hiệu S3")

    ranked = suspicion.add_suspicion(labels)
    pool = ranked[ranked["tier"] == args.tier].copy()

    already = audited_images(GT_DIR)
    free = pool[~pool["image"].astype(str).isin(already)]
    n_excl = len(pool) - len(free)
    print(f"[label] dân số {args.tier} = {len(pool):,}; loại {n_excl} ô đã chấm "
          f"-> còn {len(free):,}")
    if len(free) < args.n:
        raise SystemExit(f"chỉ còn {len(free)} ô chưa chấm, không đủ {args.n}")

    sample = sampling.simple_random_sample(free, args.n, seed=args.seed)
    sample["audit_batch"] = "label_srs"
    print(f"[label] mẫu SRS n={len(sample)}, design_weight="
          f"{sample['design_weight'].iloc[0]:.2f}")

    cfg = _load_config(Path(args.config))
    paths = _paths(cfg)
    qn_dict = None
    try:
        from core.text.dictionary import load_qn_to_nom
        if paths["qn_dict"].exists():
            qn_dict = load_qn_to_nom(str(paths["qn_dict"]))
    except Exception as e:                                   # noqa: BLE001
        print(f"[label] bỏ qua từ điển QN ({e})")

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
        title="Audit NHÃN · chỉ một câu hỏi",
        batch_size=args.batch_size,
        mode="label_only",
    )
    print(f"[label] {stat['items']} ô dựng xong "
          f"(thiếu glyph: {stat['missing_reference_glyph']}, "
          f"thiếu ngữ cảnh: {stat['missing_context']})")

    (out_dir / "plan.json").write_text(json.dumps({
        "tier": args.tier, "n": int(len(sample)),
        "population": int(len(pool)), "population_unaudited": int(len(free)),
        "excluded_already_audited": int(n_excl),
        "design": "srs", "mode": "label_only", "seed": args.seed, "conf": args.conf,
        "design_weight": float(sample["design_weight"].iloc[0]),
        "cp_lower_bound": {str(k): stats.cp_lower_bound(len(sample) - k, len(sample),
                                                        args.conf)
                           for k in (0, 1, 2, 3, 5, 8)},
        "grid": {k: v for k, v in stat.items() if k != "html"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        _readme(len(sample), len(free), args.conf, n_excl), encoding="utf-8")
    sample.to_csv(out_dir / "sample.csv", index=False)
    print(f"[label] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
