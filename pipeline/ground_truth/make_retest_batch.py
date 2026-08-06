"""Mẻ KIỂM TRA LẶP — đo độ ổn định của chính người chấm (test–retest reliability).

VÌ SAO
------
Hai mẫu NGẪU NHIÊN từ cùng dân số GOLD, chấm cách nhau một ngày, cho tỷ lệ lỗi 4,2%
(n=119) và 16,0% (n=25); riêng `wrong_image` là 0,8% so với 12,0% (Fisher p=0,017).
Dữ liệu không đổi — chỉ tiêu chí chấm đổi. Suy ra precision GOLD dao động **95,8% ↔ 84,0%**
tuỳ buổi chấm, tức chênh 12 điểm phần trăm, lớn hơn mọi thứ đang được tinh chỉnh ở bước 3.

Chừng nào chưa biết con số precision có ổn định không thì mọi ngưỡng, mọi phép hạ cấp,
mọi tuyên bố "GOLD 98%" đều đứng trên cát. Mẻ này đo trực tiếp điều đó: trình bày lại
những ô ĐÃ CHẤM, ẩn danh và xáo trộn, rồi so verdict cũ với verdict mới.

THIẾT KẾ
--------
Mẫu ngẫu nhiên thuần sẽ chỉ chứa ~2-4 ô lỗi trên 40 — quá ít để κ có nghĩa. Nên mẻ này
lấy VƯỢT TỶ LỆ nhóm không-correct, và lấy từ CẢ HAI buổi chấm:

  * lấy vượt tỷ lệ nhóm lỗi  -> ước lượng được cả hai chiều lật:
      P(trước correct -> nay lỗi)   và   P(trước lỗi -> nay correct)
  * lấy từ cả hai buổi       -> biết người chấm nay nghiêng về tiêu chí buổi nào

Vì lấy vượt tỷ lệ nên KHÔNG được đọc mẻ này như một ước lượng precision: `design_weight`
để trống toàn bộ, và `estimate` sẽ cảnh báo nếu ai đó lỡ chạy. Đại lượng cần đọc là các
tỷ lệ đồng thuận CÓ ĐIỀU KIỆN theo verdict cũ, cộng κ tính lại theo tỷ trọng dân số.

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.make_retest_batch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit_grid, sampling, suspicion
from .cli import _load_config, _paths

REPO = Path(__file__).resolve().parents[2]
GT_DIR = REPO / "dataset_out" / "ground_truth"
DEFAULT_OUT = GT_DIR / "audit_retest"
DEFAULT_LABELS = REPO / "dataset_out" / "labels_remediated.csv"

NONCORRECT = ("wrong_label", "wrong_image", "unsure")


def collect_graded(gt_dir: Path) -> pd.DataFrame:
    """Mọi ô ĐÃ CHẤM ở các mẻ trước: (item_id, image, orig_verdict, orig_batch).

    Bỏ qua verdict do MÁY chấm (source=ai_vision) — mẻ này đo độ ổn định của NGƯỜI.
    """
    rows = []
    for mf in sorted(gt_dir.rglob("manifest.jsonl")):
        d = mf.parent
        if d.name == DEFAULT_OUT.name:
            continue                              # không lấy lại chính mẻ retest
        man = {}
        for ln in mf.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                man[str(r["item_id"])] = r
        if not man:
            continue
        for vf in sorted(d.glob("verdicts*.jsonl")):
            for ln in vf.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                v = json.loads(ln)
                src = str(v.get("source") or "human").lower()
                if src != "human":
                    continue
                m = man.get(str(v["item_id"]))
                if not m or not m.get("image"):
                    continue
                rows.append({
                    "item_id": str(v["item_id"]),
                    "image": str(m["image"]),
                    "orig_verdict": str(v["verdict"]),
                    "orig_batch": d.name,
                    "orig_group": str(m.get("audit_batch") or ""),
                })
    if not rows:
        raise ValueError(f"không tìm thấy verdict NGƯỜI nào dưới {gt_dir}")
    df = pd.DataFrame(rows)
    # cùng một ảnh có thể được chấm nhiều lần -> giữ lần CUỐI
    return df.drop_duplicates("item_id", keep="last").reset_index(drop=True)


def build_sample(graded: pd.DataFrame, n_total: int = 40, seed: int = 42) -> pd.DataFrame:
    """Rút mẫu phân tầng theo (buổi chấm × nhóm verdict cũ), rồi xáo trộn mù.

    Chia đôi ngân sách cho nhóm correct / không-correct để cả hai CHIỀU lật đều ước
    lượng được; trong mỗi nhóm lại chia đều cho các buổi chấm để so được tiêu chí.
    """
    if n_total < 4:
        raise ValueError("n_total quá nhỏ để đo độ ổn định")
    g = graded.copy()
    g["cls"] = np.where(g["orig_verdict"].isin(NONCORRECT), "noncorrect", "correct")

    avail = {c: int((g["cls"] == c).sum()) for c in ("correct", "noncorrect")}
    want = {"correct": n_total // 2, "noncorrect": n_total - n_total // 2}
    # Nhóm nào không đủ hàng thì dồn phần thiếu sang nhóm kia, để luôn trả đúng n_total
    # khi tổng thể còn đủ — thay vì âm thầm trả về ít hơn số được yêu cầu.
    for a, b in (("correct", "noncorrect"), ("noncorrect", "correct")):
        short = want[a] - min(want[a], avail[a])
        if short:
            want[a] -= short
            want[b] = min(want[b] + short, avail[b])
    if sum(want.values()) < n_total:
        print(f"[retest] CHỈ rút được {sum(want.values())}/{n_total} ô — "
              f"tổng số ô đã chấm là {len(g)}")

    picks = []
    for cls, budget in want.items():
        pool = g[g["cls"] == cls]
        if pool.empty:
            continue
        batches = sorted(pool["orig_batch"].unique())
        # chia đều cho các buổi, phần dư dồn cho buổi có nhiều hàng hơn
        base = budget // len(batches)
        alloc = {b: min(base, int((pool["orig_batch"] == b).sum())) for b in batches}
        left = budget - sum(alloc.values())
        for b in sorted(batches, key=lambda x: -int((pool["orig_batch"] == x).sum())):
            room = int((pool["orig_batch"] == b).sum()) - alloc[b]
            take = min(left, room)
            alloc[b] += take
            left -= take
        for b, k in alloc.items():
            sub = pool[pool["orig_batch"] == b]
            if k > 0:
                picks.append(sampling.simple_random_sample(sub, k, seed=seed + len(picks)))

    s = pd.concat(picks, ignore_index=True, sort=False)
    s["design_weight"] = np.nan          # CHỦ ĐÍCH -> chặn mọi ước lượng precision
    s["stratum"] = s["cls"] + "|" + s["orig_batch"]
    s["audit_batch"] = "retest"

    key = s["item_id"].map(lambda k: sampling._stable_key(k, seed))
    s = s.assign(_o=key).sort_values("_o", kind="stable").drop(columns="_o")
    s = s.reset_index(drop=True)
    s["audit_order"] = np.arange(len(s))
    return s


def _readme(s: pd.DataFrame) -> str:
    tab = (s.groupby(["orig_batch", "orig_verdict"]).size()
           .reset_index(name="n").to_string(index=False))
    return f"""# Mẻ kiểm tra lặp — đo độ ổn định của người chấm

**{len(s)} ô**, khoảng 20 phút.

## Vì sao có mẻ này

Hai mẫu **ngẫu nhiên từ cùng một dân số GOLD**, bạn chấm cách nhau một ngày:

| | Tổng lỗi | riêng `wrong_image` |
|---|---|---|
| Mẻ 03/08 (n=119) | 4,2% | 0,8% |
| Mẻ 04/08 (n=25) | 16,0% | 12,0% |

Dữ liệu y hệt nhau, chỉ tiêu chí chấm khác. Suy ra precision GOLD là **95,8%** hay
**84,0%** tuỳ buổi — chênh 12 điểm phần trăm. Chừng nào chưa biết con số đó có ổn định
không thì không tuyên bố precision nào đứng vững được.

## ĐỌC KỸ — đây là điểm mấu chốt

**Toàn bộ ô trong mẻ này bạn ĐÃ TỪNG CHẤM.** Tôi nói thẳng để bạn không thấy bị gài.

Nhưng chính vì thế, cách chấm quyết định mẻ này có giá trị hay không:

- **Đừng cố nhớ lần trước bạn đã bấm gì.** Nếu bạn cố nhớ lại, kết quả chỉ đo trí nhớ
  của bạn chứ không đo tiêu chí chấm.
- **Đừng cố tỏ ra nhất quán.** Nếu hôm nay bạn thấy khác hôm qua, hãy bấm theo cái bạn
  thấy HÔM NAY. Chuyện đổi ý là dữ liệu quý, không phải lỗi.
- Nếu lỡ nhớ ra đáp án cũ mà giờ thấy nó sai, **cứ bấm theo cái bạn thấy bây giờ**.

Không đọc ra chữ thì chọn **không chắc** — đừng đoán.

## Câu hỏi cần bạn tự trả lời trước khi bắt đầu

Trước khi chấm, hãy tự chốt trong đầu: **bao nhiêu mực thừa của chữ bên cạnh thì tính là
`wrong_image`?** Một mẩu nét nhỏ ở mép có tính không? Nửa chữ hàng xóm thì sao?

Chênh lệch giữa hai mẻ nằm gần như trọn vẹn ở câu hỏi này. Cứ giữ nguyên một tiêu chí
từ đầu đến cuối mẻ, dù tiêu chí đó là gì.

## Cách chấm

1. Mở `audit_001.html` (hoặc `audit.html`).
2. Bấm: **correct** / **wrong_label** (crop đúng 1 chữ nhưng gán sai chữ) /
   **wrong_image** (crop cắt hỏng, dính chữ khác, nhầm ô) / **unsure**.
3. Xong bấm **Download JSON** → lưu `verdicts_001.jsonl` ngay trong thư mục này.

## Thành phần (bạn KHÔNG cần biết trước khi chấm — để đây cho hồ sơ)

{tab}

Nhóm lỗi được lấy vượt tỷ lệ có chủ đích, để đo được cả hai chiều lật. **Vì vậy mẻ này
KHÔNG phải ước lượng precision** — `design_weight` để trống nhằm chặn việc dùng nhầm.

## Mẻ này quyết được gì

| Kết quả | Kết luận |
|---------|----------|
| đồng thuận cao, κ ≥ 0,8 | tiêu chí ổn định → chênh lệch hai mẻ là do n nhỏ; dùng 95,8% được |
| κ 0,4–0,8 | ổn định vừa → phải viết rubric rõ cho `wrong_image` rồi chấm lại |
| κ < 0,4 | **chưa con số precision nào bảo vệ được** — rubric là việc bắt buộc trước mọi thứ khác |
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_retest_batch")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    graded = collect_graded(GT_DIR)
    print(f"[retest] {len(graded)} ô đã chấm bởi NGƯỜI ở các mẻ trước")
    print(graded.groupby(["orig_batch", "orig_verdict"]).size().to_string())

    s = build_sample(graded, args.n, args.seed)
    # lấy lại metadata hiển thị (nhãn, âm, bbox...) từ labels
    labels = pd.read_csv(args.labels, dtype={"image_md5": str})
    meta = suspicion.add_suspicion(labels)
    keep = [c for c in meta.columns if c not in s.columns]
    s = s.merge(meta[["image", *keep]], on="image", how="left", validate="one_to_one")
    print(f"[retest] mẫu {len(s)} ô")
    print(s.groupby(["orig_batch", "orig_verdict"]).size().to_string())

    cfg = _load_config(Path(args.config))
    paths = _paths(cfg)
    qn_dict = None
    try:
        from core.text.dictionary import load_qn_to_nom
        if paths["qn_dict"].exists():
            qn_dict = load_qn_to_nom(str(paths["qn_dict"]))
    except Exception as e:                                   # noqa: BLE001
        print(f"[retest] bỏ qua từ điển QN ({e})")

    stat = audit_grid.build_audit(
        s,
        dataset_dir=paths["dataset_dir"],
        prepared_dir=paths["prepared_dir"],
        fd_dir=paths["fd_dir"],
        out_html=out_dir / "audit.html",
        out_manifest=out_dir / "manifest.jsonl",
        qn_dict=qn_dict,
        font_path=paths["font"],
        with_context=not args.no_context,
        title="Kiểm tra lặp · độ ổn định người chấm",
        batch_size=args.batch_size,
    )
    print(f"[retest] {stat['items']} ô dựng xong")

    (out_dir / "plan.json").write_text(json.dumps({
        "n": int(len(s)),
        "seed": args.seed,
        "purpose": "test-retest reliability (KHÔNG dùng để ước lượng precision)",
        "composition": {f"{b}|{v}": int(n) for (b, v), n
                        in s.groupby(["orig_batch", "orig_verdict"]).size().items()},
        "grid": {k: v for k, v in stat.items() if k != "html"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(s), encoding="utf-8")
    s.to_csv(out_dir / "sample.csv", index=False)
    print(f"[retest] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
