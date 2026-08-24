"""BÀN THÍ NGHIỆM — chạy MỘT cấu hình, ghi MỘT dòng vào lab/results.csv.

    python -m pipeline.lab.runner --config lab/configs/baseline.yaml
    python -m pipeline.lab.runner --config lab/configs/baseline.yaml --quick

NGUYÊN TẮC
----------
1. KHÔNG đụng `dataset_out/`. Mọi thứ ghi vào `lab/`.
2. TẤT ĐỊNH: cùng cấu hình -> cùng dòng số, byte y hệt. Không dùng thời gian/ngẫu nhiên
   không seed vào phần SỐ (cột `ts` chỉ để tra cứu, không tham gia so sánh).
3. `run_id` = sha256(cấu hình đã chuẩn hoá + VÂN TAY BỘ NHÃN). Phải gồm cả vân tay dữ
   liệu: chỉ băm cấu hình thì chạy cùng cấu hình trên dữ liệu TRƯỚC và SAU một thí
   nghiệm sẽ cho cùng run_id và dòng sau ĐÈ dòng trước.

CẤU HÌNH (YAML) — mọi khoá đều tuỳ chọn, thiếu thì lấy mặc định:

    name: baseline                  # nhãn người đọc
    labels: dataset_out/labels_final.csv
    src_root: dataset_out
    crops: true                     # đo hình học crop (false = bỏ qua, chạy nhanh)
    crop_sample: 0                  # >0 = chỉ đo mẫu đều
    perturb:
      enabled: true
      seeds: 3
      max_columns: 0                # 0 = tất cả
      band_slack: null              # null = giữ nguyên BAND_SLACK của engine
      cost: {}                      # ví dụ {COST_SIMILAR: 0.45}
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "lab"
RESULTS = LAB / "results.csv"

DEFAULTS = {
    "name": "unnamed",
    "labels": "dataset_out/labels_final.csv",
    "src_root": "dataset_out",
    "crops": True,
    "crop_sample": 0,
    "perturb": {"enabled": True, "seeds": 3, "max_columns": 0,
                "band_slack": None, "cost": {}},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def labels_sha(path: Path | str) -> str:
    """sha256 (12 ký tự) của bộ nhãn đầu vào."""
    p = Path(path)
    if not p.exists():
        return "nofile"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:12]


def config_id(cfg: dict, data_sha: str = "") -> str:
    """Vân tay của (cấu hình + TRẠNG THÁI DỮ LIỆU). Bỏ qua `name` — chỉ là nhãn người đọc.

    Phải gồm cả vân tay bộ nhãn: nếu chỉ băm cấu hình thì chạy cùng một cấu hình trên
    dữ liệu TRƯỚC và SAU một thí nghiệm sẽ cho cùng `run_id` và dòng sau ĐÈ dòng trước —
    đúng lỗi đã xảy ra ngày 2026-08-24, làm mất mốc baseline trước T1.
    """
    payload = {k: v for k, v in cfg.items() if k != "name"}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str) + "|" + data_sha
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run(cfg: dict) -> dict:
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import dict_dir, load_qn_to_nom, load_similarity_dict
    from pipeline.lab import metrics, perturb

    cfg = _merge(DEFAULTS, cfg)
    labels = REPO / cfg["labels"]
    dsha = labels_sha(labels)
    rid = config_id(cfg, dsha)
    row: dict = {"run_id": rid, "name": cfg["name"], "labels_sha": dsha,
                 "ts": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "config": json.dumps({k: v for k, v in cfg.items() if k != "name"},
                                      sort_keys=True, ensure_ascii=False)}

    print(f"[lab] {cfg['name']} ({rid}) — thước đo họ D…", flush=True)
    row.update(metrics.collect(labels, REPO / cfg["src_root"],
                               with_crops=bool(cfg["crops"]),
                               crop_sample=int(cfg["crop_sample"] or 0)))

    p = cfg["perturb"]
    if p.get("enabled"):
        print("[lab] chuẩn nhiễu loạn (họ B)…", flush=True)
        cols = perturb.load_columns(labels)
        if p.get("max_columns"):
            cols = cols[: int(p["max_columns"])]
        qn = load_qn_to_nom(str(dict_dir() / "QuocNgu_SinoNom.csv"))
        sim = load_similarity_dict(str(dict_dir() / "SinoNom_Similar.csv"))
        row["perturb_columns"] = len(cols)
        row.update(perturb.run_suite(cols, qn, sim, seeds=int(p.get("seeds", 3)),
                                     band_slack=p.get("band_slack"),
                                     cost=p.get("cost") or {}))
    return row


def append_row(row: dict, results: Path = RESULTS) -> None:
    """Thêm/ghi đè theo `run_id`. Cột mới xuất hiện thì viết lại toàn tệp cho khớp header."""
    results.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if results.exists():
        with open(results, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("run_id") != row["run_id"]]
    rows.append({k: ("" if v is None else v) for k, v in row.items()})
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(results, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.lab.runner")
    ap.add_argument("--config", default=str(LAB / "configs" / "baseline.yaml"))
    ap.add_argument("--quick", action="store_true",
                    help="mẫu 2.000 crop + 300 cột + 1 seed (dùng khi thử tay)")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.quick:
        cfg = _merge(_merge(DEFAULTS, cfg),
                     {"name": (cfg.get("name", "unnamed") + "-quick"),
                      "crop_sample": 2000,
                      "perturb": {"max_columns": 300, "seeds": 1}})
    row = run(cfg)

    print("-" * 64)
    for k in ("run_id", "n_rows", "n_usable", "n_classes_usable", "md5_label_conflicts",
              "syl_in_dict_pct", "syl_out_of_dict", "pages_with_9_cols",
              "anchor_frac_p50", "n_crops", "flag_ok_pct", "flag_bleed_pct",
              "flag_truncated_pct", "anchor_retention", "anchors_tested"):
        if k in row:
            print(f"  {k:22} {row[k]}")
    if not args.no_write:
        append_row(row, Path(args.results))
        print(f"-> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
