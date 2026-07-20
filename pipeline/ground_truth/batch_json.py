"""Xuất JSON máy-đọc-được cho MỖI file audit_XXX.html của một nhóm audit.

audit_grid ghi manifest.jsonl ĐÚNG thứ tự hiển thị (sort audit_order, đã bỏ crop lỗi) rồi
chia HTML theo batch_size (mặc định 150). Module này chunk manifest theo cùng batch_size →
audit_001.json khớp audit_001.html, ... Mỗi JSON là bản metadata của batch (KHÔNG nhúng ảnh
base64 — ảnh đã nằm trong HTML) + ứng viên từ điển của âm, để xử lý/kiểm bằng máy hoặc
audit bằng công cụ khác.

Chạy:
  .venv/bin/python -m pipeline.ground_truth.batch_json --dir dataset_out/ground_truth/audit_SILVER
  .venv/bin/python -m pipeline.ground_truth.batch_json --all      # cả 3 nhóm audit_*
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BATCH_SIZE = 150


def _qn_dict():
    try:
        from core.text.dictionary import load_qn_to_nom
        p = REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"
        return load_qn_to_nom(str(p)) if p.exists() else {}
    except Exception:
        return {}


def export_dir(audit_dir: Path, batch_size: int, qn_dict: dict) -> dict:
    man = audit_dir / "manifest.jsonl"
    if not man.exists():
        raise SystemExit(f"[batch_json] thiếu {man}")
    items = [json.loads(l) for l in man.open(encoding="utf-8") if l.strip()]
    n_html = len(glob.glob(str(audit_dir / "audit_*.html")))
    n_batches = (len(items) + batch_size - 1) // batch_size
    written = []
    for b in range(n_batches):
        chunk = items[b * batch_size:(b + 1) * batch_size]
        enriched = []
        for it in chunk:
            syl = str(it.get("syllable") or "")
            cands = qn_dict.get(syl.lower(), []) if syl else []
            enriched.append({**it, "candidates": cands[:12]})
        out = {
            "batch": b + 1, "of": n_batches, "html": f"audit_{b + 1:03d}.html",
            "tier": (chunk[0].get("tier") if chunk else None),
            "n_items": len(chunk), "items": enriched,
        }
        p = audit_dir / f"audit_{b + 1:03d}.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        written.append(p.name)
    # index tổng cho cả nhóm
    (audit_dir / "audit_index.json").write_text(json.dumps({
        "dir": audit_dir.name, "total_items": len(items), "batch_size": batch_size,
        "n_batches": n_batches, "html_files": n_html, "json_files": written,
        "match_html": n_batches == n_html,
    }, ensure_ascii=False, indent=2))
    status = "OK" if n_batches == n_html else f"CẢNH BÁO: {n_batches} json vs {n_html} html"
    print(f"[batch_json] {audit_dir.name}: {len(items)} item → {n_batches} json ({status})")
    return {"dir": audit_dir.name, "items": len(items), "batches": n_batches,
            "match_html": n_batches == n_html}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.batch_json")
    ap.add_argument("--dir", default="")
    ap.add_argument("--all", action="store_true", help="cả 3 nhóm audit_* trong ground_truth/")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args(argv)
    qn = _qn_dict()
    if args.all:
        base = REPO / "dataset_out" / "ground_truth"
        dirs = sorted(d for d in base.glob("audit_*") if d.is_dir())
        for d in dirs:
            export_dir(d, args.batch_size, qn)
    elif args.dir:
        export_dir(Path(args.dir), args.batch_size, qn)
    else:
        ap.error("cần --dir hoặc --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
