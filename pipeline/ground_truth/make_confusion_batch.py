"""Mẻ audit cho MỘT LỚP NHẦM LẪN nghi vấn (một cặp chữ–âm cụ thể).

VÌ SAO
------
Lỗi nhãn trong bộ này đi theo CỤM chứ không rải đều: mẻ audit GOLD đầu tiên tìm được 3
lỗi thì 2 trong số đó là cùng một cặp `奴` / âm "nó". Cặp đó xuất hiện **312 lần** trong
GOLD và là nhãn DUY NHẤT được dùng cho âm "nó" — nên câu hỏi không phải "2 hàng này sai
không" mà "cả 312 hàng có sai không". Một mẻ nhỏ nhắm đúng lớp đó trả lời dứt điểm, trong
khi mọi ngưỡng thị giác đều bó tay (S3 chấm cả 3 lỗi là "rất đáng tin", AUC < 0,5).

CHỐNG HIỆU ỨNG MỎ NEO
---------------------
Một mẻ toàn cùng một chữ rất dễ khiến người chấm rơi vào nếp: quyết định chữ đó sai rồi
đánh sai cả loạt, tạo ra "xác nhận" không có thật. Vì vậy mẻ này LUÔN trộn thêm một nhóm
ĐỐI CHỨNG lấy ngẫu nhiên từ phần GOLD còn lại. Người chấm không phân biệt được nhóm nào,
và chỉ nhóm mục tiêu mới vào ước lượng của lớp.

Hai nhóm là một PHÂN HOẠCH của GOLD (lớp ∪ phần-còn-lại), mỗi nhóm có `design_weight`
riêng, nên `stratum` được đặt bằng tên nhóm để ước lượng phân tầng vẫn đúng. Thứ hạng rủi
ro gốc được giữ lại ở cột `risk_stratum`.

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.make_confusion_batch --label 奴 --syllable nó
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
GT_DIR = REPO / "dataset_out" / "ground_truth"

BATCH_TARGET = "class_target"
BATCH_CONTROL = "control_gold"


def _item_id(image: str) -> str:
    return hashlib.sha1(str(image).encode()).hexdigest()[:16]


def audited_images(gt_dir: Path) -> set[str]:
    """Mọi ảnh đã nằm trong một mẻ audit trước — không lấy lại để khỏi chấm trùng."""
    seen: set[str] = set()
    for mf in gt_dir.rglob("manifest.jsonl"):
        for ln in mf.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                img = json.loads(ln).get("image")
            except json.JSONDecodeError:
                continue
            if img:
                seen.add(str(img))
    return seen


def build_sample(
    gold: pd.DataFrame,
    label: str,
    syllable: str,
    n_target: int,
    n_control: int,
    exclude: set[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Rút (mục tiêu ∪ đối chứng), trộn mù. Trả về (mẫu, thống kê dân số)."""
    exclude = exclude or set()
    in_class = gold["label"].astype(str).eq(label) & gold["syllable"].astype(str).eq(syllable)
    n_class_total = int(in_class.sum())
    if n_class_total == 0:
        raise ValueError(f"không có hàng GOLD nào mang nhãn {label!r} âm {syllable!r}")

    free = ~gold["image"].astype(str).isin(exclude)
    target_pool = gold.loc[in_class & free]
    control_pool = gold.loc[~in_class & free]
    if len(target_pool) < n_target:
        raise ValueError(f"lớp chỉ còn {len(target_pool)} hàng chưa chấm, không đủ {n_target}")
    if len(control_pool) < n_control:
        raise ValueError(f"phần còn lại chỉ có {len(control_pool)} hàng, không đủ {n_control}")

    tgt = sampling.simple_random_sample(target_pool, n_target, seed=seed)
    tgt["audit_batch"] = BATCH_TARGET
    ctl = sampling.simple_random_sample(control_pool, n_control, seed=seed + 1)
    ctl["audit_batch"] = BATCH_CONTROL

    both = pd.concat([tgt, ctl], ignore_index=True, sort=False)
    if both["item_id"].duplicated().any():
        raise AssertionError("mục tiêu và đối chứng chồng lấn")

    # giữ thứ hạng rủi ro gốc, nhưng `stratum` = nhóm để ước lượng phân tầng đúng dân số
    both["risk_stratum"] = both.get("stratum")
    both["stratum"] = both["audit_batch"]

    key = both["item_id"].map(lambda k: sampling._stable_key(k, seed))
    both = both.assign(_o=key).sort_values("_o", kind="stable").drop(columns="_o")
    both = both.reset_index(drop=True)
    both["audit_order"] = np.arange(len(both))

    info = {
        "label": label,
        "syllable": syllable,
        "class_population": n_class_total,
        "class_unaudited": int(len(target_pool)),
        "rest_population": int(len(control_pool)),
        "n_target": n_target,
        "n_control": n_control,
    }
    return both, info


def _readme(info: dict, conf: float) -> str:
    n_t = info["n_target"]
    ub0 = 1.0 - stats.cp_lower_bound(0, n_t, conf)          # cận TRÊN tỷ lệ lỗi khi 0 lỗi
    lb_all = stats.cp_lower_bound(n_t, n_t, conf)
    return f"""# Mẻ audit lớp nhầm lẫn — `{info['label']}` / âm "{info['syllable']}"

**{n_t + info['n_control']} ô**, chấm khoảng nửa giờ.

## Câu hỏi cần trả lời

Mẻ audit GOLD đầu tiên tìm được 3 lỗi, trong đó **2 lỗi là cùng cặp `{info['label']}` /
"{info['syllable']}"**. Cặp này xuất hiện **{info['class_population']} lần** trong GOLD và
là nhãn duy nhất được dùng cho âm đó. Nên câu hỏi thật là:

> {info['class_population']} hàng đó đúng hay sai — toàn bộ?

Hai khả năng, hệ quả khác hẳn nhau:

- **Sai hệ thống** → {info['class_population']} nhãn hỏng (~0,6% GOLD), phải sửa bằng bảng
  nhầm lẫn ở bước 7, và đây là lỗi lớn nhất tìm được cho tới giờ.
- **Hai verdict trước bị nhầm** → precision GOLD thực ra là 116/117 = 99,1%, không có gì
  phải sửa.

## ĐỌC KỸ TRƯỚC KHI CHẤM

Trong mẻ này **nhiều ô cố ý mang cùng một chữ**, xen lẫn các ô chữ khác lấy ngẫu nhiên.
Điều đó là chủ ý của thiết kế.

**Hãy chấm từng ô độc lập.** Đừng vì đã kết luận ở một ô mà đánh y hệt cho các ô sau —
nếu làm vậy, kết quả sẽ chỉ lặp lại phán đoán đầu tiên của bạn chứ không đo được gì. Cũng
đừng cố cho ra kết quả "nhất quán": hoàn toàn có thể một số ô đúng và một số ô sai.

Nếu không đọc ra chữ, chọn **không chắc** — đừng đoán. Ô "không chắc" bị loại khỏi phép
tính chứ không bị tính là sai.

## Cách chấm

1. Mở `audit_001.html` trong trình duyệt.
2. Mỗi ô: crop + glyph tham chiếu + ngữ cảnh trang + âm + ứng viên từ điển. Bấm:
   **correct** / **wrong_label** (crop đúng 1 chữ nhưng gán sai chữ) /
   **wrong_image** (crop cắt hỏng, dính 2 chữ, nhầm ô) / **unsure**.
3. Xong bấm **Download JSON**, lưu thành `verdicts_001.jsonl` ngay trong thư mục này.

## Mẻ này quyết được tới đâu (nhóm mục tiêu, n = {n_t}, {conf:.0%})

| Kết quả | Kết luận |
|---------|----------|
| 0 ô sai | tỷ lệ lỗi của lớp ≤ **{ub0:.1%}** → bác bỏ giả thuyết "sai hệ thống" |
| {n_t} ô sai | tỷ lệ lỗi ≥ **{lb_all:.1%}** → xác nhận sai hệ thống, sửa cả {info['class_population']} hàng |
| lẫn lộn | lỗi phụ thuộc ngữ cảnh, cần soi theo sách/trang |

## Hai nhóm trong mẻ

| Nhóm | n | Dân số | Dùng để |
|------|--:|-------:|---------|
| `{BATCH_TARGET}` | {n_t} | {info['class_unaudited']} hàng chưa chấm của lớp | **tỷ lệ lỗi của lớp** |
| `{BATCH_CONTROL}` | {info['n_control']} | {info['rest_population']:,} hàng GOLD còn lại | phá hiệu ứng mỏ neo; thêm bằng chứng cho precision GOLD |

Các ô đã chấm ở mẻ trước đã được loại, không chấm lại.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_confusion_batch")
    ap.add_argument("--label", required=True, help="chữ Nôm nghi bị gán sai, vd 奴")
    ap.add_argument("--syllable", required=True, help="âm quốc ngữ đi kèm, vd nó")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default="")
    ap.add_argument("--n-target", type=int, default=30, dest="n_target")
    ap.add_argument("--n-control", type=int, default=25, dest="n_control")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else (
        GT_DIR / f"audit_confusion_{args.label}_{args.syllable}")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels, dtype={"image_md5": str})
    try:
        labels, rep = s3_signals.attach(labels)
        print(f"[confusion] {rep}")
    except FileNotFoundError:
        print("[confusion] không có s3_corpus — chạy tiếp, mẻ này không cần tín hiệu S3")

    ranked = suspicion.add_suspicion(labels)
    gold = ranked[ranked["tier"] == "GOLD"].copy()

    already = audited_images(GT_DIR)
    print(f"[confusion] đã có {len(already):,} ảnh từng nằm trong mẻ audit trước -> loại ra")

    sample, info = build_sample(gold, args.label, args.syllable,
                                args.n_target, args.n_control,
                                exclude=already, seed=args.seed)
    print(f"[confusion] lớp {args.label}/{args.syllable}: {info['class_population']} hàng GOLD "
          f"({info['class_unaudited']} chưa chấm)")
    print(f"[confusion] mẫu {len(sample)} ô: mục tiêu={info['n_target']}, "
          f"đối chứng={info['n_control']}")

    cfg = _load_config(Path(args.config))
    paths = _paths(cfg)
    qn_dict = None
    try:
        from core.text.dictionary import load_qn_to_nom
        if paths["qn_dict"].exists():
            qn_dict = load_qn_to_nom(str(paths["qn_dict"]))
    except Exception as e:                                   # noqa: BLE001
        print(f"[confusion] bỏ qua từ điển QN ({e})")

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
        title=f"Audit lớp nhầm lẫn · {args.label}/{args.syllable}",
        batch_size=args.batch_size,
    )
    print(f"[confusion] {stat['items']} ô dựng xong "
          f"(thiếu glyph: {stat['missing_reference_glyph']}, "
          f"thiếu ngữ cảnh: {stat['missing_context']})")

    plan = {
        **info,
        "seed": args.seed,
        "conf": args.conf,
        "design_weight_target": info["class_unaudited"] / info["n_target"],
        "design_weight_control": info["rest_population"] / info["n_control"],
        "class_error_rate_upper_if_zero": 1.0 - stats.cp_lower_bound(
            0, info["n_target"], args.conf),
        "grid": {k: v for k, v in stat.items() if k != "html"},
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(info, args.conf), encoding="utf-8")
    sample.to_csv(out_dir / "sample.csv", index=False)
    print(f"[confusion] -> {out_dir}")
    print(f"[confusion] mở {out_dir / 'audit_001.html'} (hoặc audit.html) để chấm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
