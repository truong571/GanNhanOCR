"""Mẻ ĐỒNG THUẬN LIÊN NGƯỜI — cùng những ô đó, một người thứ hai chấm độc lập.

VÌ SAO CẦN
----------
Ô lặp ẩn trong mẻ gộp đo được người chấm có **tự nhất quán** không (test–retest, κ nội
tại). Nó KHÔNG trả lời được câu hỏi mà hội đồng chắc chắn hỏi:

    "Người chấm chính là tác giả của bộ dữ liệu. Ai kiểm tra lại?"

Một người tự nhất quán vẫn có thể sai hệ thống theo cùng một hướng ở cả hai lần — κ nội
tại cao KHÔNG loại trừ khả năng đó. Chỉ có người thứ hai, chấm độc lập, mù với verdict của
người thứ nhất, mới tách được "tiêu chí ổn định" khỏi "tiêu chí ổn định nhưng lệch".

THIẾT KẾ — vì sao rút NGẪU NHIÊN THUẦN chứ không lấy vượt tỷ lệ ô lỗi
--------------------------------------------------------------------
Mẻ kiểm tra lặp (`make_retest_batch`) cố ý lấy vượt tỷ lệ nhóm không-`correct`, vì khi lỗi
hiếm thì mẫu ngẫu nhiên gần như không chứa ô nào để hai lần chấm có thể lệch nhau. Cái giá
là κ thu được **có điều kiện theo verdict cũ**, phải hiệu chỉnh lại theo tỷ trọng dân số
mới đọc được — một bước dễ bị chất vấn.

Ở đây không cần cái giá đó: mẻ gộp có tier SILVER (300/800 ô, tỷ lệ lỗi kỳ vọng ~25 %), nên
một mẫu ngẫu nhiên 100 ô đã chứa sẵn khoảng 10–15 ô đủ khả năng gây bất đồng. Rút ngẫu
nhiên thuần cho ra κ **đọc thẳng được**, không cần hiệu chỉnh, không phải giải thích gì
thêm trước hội đồng.

Phân tầng theo tier (phân bổ tỷ lệ) để cả ba tier đều có mặt — vẫn là mẫu xác suất.

MÙ CHO NGƯỜI THỨ HAI
--------------------
`orig_verdict` (verdict của người thứ nhất) đi vào manifest để ghép cặp về sau, nhưng nằm
trong `audit_grid._HIDDEN_FIELDS` nên **không bao giờ** lọt vào HTML. Người thứ hai thấy
đúng những gì người thứ nhất đã thấy, không hơn.

CHẠY (sau khi người thứ nhất chấm xong mẻ gộp)
----------------------------------------------
    .venv/bin/python -m pipeline.ground_truth.make_interrater_batch --n 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit_grid, stats
from .cli import _load_config, _paths

REPO = Path(__file__).resolve().parents[2]
GT_DIR = REPO / "dataset_out" / "ground_truth"
DEFAULT_SOURCE = GT_DIR / "audit_combined"
DEFAULT_OUT = GT_DIR / "audit_interrater"


def load_scored(source: Path) -> pd.DataFrame:
    """Các ô XÁC SUẤT của mẻ nguồn mà người thứ nhất ĐÃ chấm (bỏ ô lặp, bỏ verdict máy)."""
    man = pd.DataFrame(
        [json.loads(ln) for ln in (source / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()])
    rows = []
    for vf in sorted(source.glob("verdicts*.jsonl")):
        for ln in vf.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            v = json.loads(ln)
            if str(v.get("source") or "human").lower() != "human":
                continue
            rows.append({"item_id": str(v["item_id"]), "orig_verdict": str(v["verdict"])})
    if not rows:
        raise SystemExit(
            f"chưa có verdict NGƯỜI nào trong {source} — người thứ nhất phải chấm xong "
            f"(và bấm Xuất verdicts.jsonl) trước khi dựng mẻ liên người")
    v = pd.DataFrame(rows).drop_duplicates("item_id", keep="last")

    j = man.merge(v, on="item_id", how="inner")
    j = j[j["stratum"].astype(str) != "__repeat__"]      # ô lặp không phải mẫu xác suất
    return j.reset_index(drop=True)


def draw(scored: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """SRS phân tầng theo tier, phân bổ tỷ lệ. Giữ nguyên item_id để ghép cặp về sau."""
    scored = scored.copy()
    scored["tier_"] = scored["stratum"].astype(str).str.split("|").str[0]
    sizes = scored.groupby("tier_").size().to_dict()
    total = sum(sizes.values())
    if total < n:
        raise SystemExit(f"chỉ có {total} ô đã chấm, không đủ {n}")

    alloc, rem = {}, n
    for t in sorted(sizes):
        alloc[t] = min(sizes[t], int(n * sizes[t] / total))
        rem -= alloc[t]
    for t in sorted(sizes, key=lambda k: -sizes[k]):
        while rem > 0 and alloc[t] < sizes[t]:
            alloc[t] += 1
            rem -= 1

    parts = []
    for t, k in sorted(alloc.items()):
        if k <= 0:
            continue
        g = scored[scored["tier_"] == t]
        s = int(hashlib.sha1(f"{seed}:ir:{t}".encode()).hexdigest()[:8], 16) % (2**31)
        idx = np.random.default_rng(s).choice(g.index.to_numpy(), size=k, replace=False)
        parts.append(g.loc[idx])
    out = pd.concat(parts, ignore_index=True)

    key = out["item_id"].map(
        lambda i: int(hashlib.sha1(f"{seed}:order:{i}".encode()).hexdigest()[:12], 16))
    out = out.assign(_k=key).sort_values("_k").drop(columns="_k").reset_index(drop=True)
    out["audit_order"] = np.arange(len(out))
    out["audit_batch"] = "interrater"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.make_interrater_batch")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = một file duy nhất")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = ap.parse_args(argv)

    source, out_dir = Path(args.source), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = load_scored(source)
    print(f"[liên người] mẻ nguồn có {len(scored):,} ô xác suất đã chấm")
    sample = draw(scored, args.n, args.seed)
    dist = sample["stratum"].astype(str).str.split("|").str[0].value_counts().to_dict()
    print(f"[liên người] rút {len(sample)} ô: {dist}")

    cfg = _load_config(Path(args.config))
    paths = _paths(cfg)
    qn_dict = None
    try:
        from core.text.dictionary import load_qn_to_nom
        if paths["qn_dict"].exists():
            qn_dict = load_qn_to_nom(str(paths["qn_dict"]))
    except (ImportError, OSError) as exc:
        print(f"[liên người] bỏ qua từ điển QN ({exc})")

    stat = audit_grid.build_audit(
        sample,
        dataset_dir=paths["dataset_dir"], prepared_dir=paths["prepared_dir"],
        fd_dir=paths["fd_dir"],
        out_html=out_dir / "audit.html", out_manifest=out_dir / "manifest.jsonl",
        qn_dict=qn_dict, font_path=paths["font"], with_context=not args.no_context,
        title="Audit nhãn · người chấm độc lập",
        batch_size=args.batch_size or None, mode="label_only",
    )
    print(f"[liên người] {stat['items']} ô dựng xong")

    n = len(sample)
    (out_dir / "plan.json").write_text(json.dumps({
        "purpose": "đồng thuận LIÊN NGƯỜI (inter-rater) — KHÔNG dùng để ước lượng precision",
        "source_batch": source.name, "n": n, "seed": args.seed, "conf": args.conf,
        "design": "SRS phân tầng theo tier, phân bổ tỷ lệ — mẫu xác suất, κ đọc thẳng được",
        "tier_allocation": dist,
        "rater1_verdict_distribution": sample["orig_verdict"].value_counts().to_dict(),
        "kappa_se_reference": {
            "note": "sai số chuẩn xấp xỉ của tỷ lệ đồng thuận thô ở n này",
            "halfwidth_at_agreement_0.90": stats.wilson_ci(int(0.90 * n), n, args.conf),
            "halfwidth_at_agreement_0.95": stats.wilson_ci(int(0.95 * n), n, args.conf),
        },
        "grid": {k: v for k, v in stat.items() if k != "html"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    rel = out_dir.relative_to(REPO) if out_dir.is_relative_to(REPO) else out_dir
    (out_dir / "README.md").write_text(f"""# Mẻ chấm độc lập — dành cho người chấm thứ HAI

**{n} ô**, rút ngẫu nhiên từ mẻ đã được người thứ nhất chấm. Mục đích: đo **độ đồng thuận
giữa hai người**, không phải đo lại chất lượng dữ liệu.

## Dành cho người chấm

Bạn **không** cần biết gì về bộ dữ liệu này, và tốt nhất là không biết. Nhiệm vụ đúng một
câu hỏi cho mỗi ô:

> **Nhãn hiện trên thẻ có đúng là chữ viết trong ô không?**

| Phím | Nghĩa |
|:---:|---|
| **1** | **nhãn ĐÚNG** — chữ trong ô đúng là chữ được gán |
| **2** | **nhãn SAI** — chữ trong ô là một chữ KHÁC |
| **3** | **không đọc được** — không đủ căn cứ, kể cả sau khi xem ảnh ngữ cảnh |

Bốn quy tắc:

1. **Khung cắt xấu không phải lỗi nhãn.** Dính chút mực của chữ bên cạnh mà vẫn đọc ra chữ
   → bấm **1**. Câu hỏi là về **chữ**, không phải về **khung**.
2. Crop khó nhìn thì đọc bằng **ảnh ngữ cảnh** (khung đỏ trên trang scan).
3. Lưỡng lự → bấm **3**, đừng bấm **2**.
4. Chấm **hết** theo đúng thứ tự, không bỏ ô khó.

Một số ô hiện **âm tiết** ở chỗ chữ lớn và không có glyph tham chiếu — với những ô đó câu
hỏi là *"chữ Nôm trong ô có đọc là âm tiết này không?"*, dùng dòng ứng viên từ điển làm
căn cứ.

**Đừng hỏi người thứ nhất đã chấm gì.** Giá trị của mẻ này nằm ở chỗ hai người chấm hoàn
toàn độc lập; biết trước đáp án là làm hỏng phép đo.

## Cách chấm

1. Mở `audit.html`.
2. Bấm **1** / **2** / **3** hoặc bấm chuột; `←` `→` chuyển ô. Tiến độ tự lưu.
3. Chấm hết → **Xuất verdicts.jsonl** → lưu vào chính thư mục này.

## Sau khi chấm

```
.venv/bin/python -m pipeline.ground_truth.report_combined --interrater {rel}
```

→ `KAPPA_LIEN_NGUOI.md`: Cohen's κ, đồng thuận thô, ma trận bất đồng, và phân rã theo tier.
""", encoding="utf-8")
    print(f"[liên người] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
