"""python -m pipeline.stt_hai_luot <lệnh> — xem pipeline/stt_hai_luot/__init__.py.

  cache       --books SachThanhTruyen2 … [--force]        dựng cache kim_l1skel_l2 (0 API, bỏ qua trang đã khớp src_sha)
  info        --config config/pipeline.yaml               in biến shell HL_ENABLED / HL_AUX_DIR / HL_WORK / HL_GATE / HL_VDP
  aux-config  --config … --out <aux>/config.yaml          config bản dựng phụ (kim_read := aux_read)
  union       --config … --primary dataset_out            hợp theo ô vào dataset_out/labels_final.csv
  gate        --config … --primary dataset_out --labels dataset_out/labels_final.csv   cổng (sau rescue)
  selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import route as RT

REPO = Path(__file__).resolve().parents[2]


def _cfg(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.stt_hai_luot")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("cache"); a.add_argument("--books", nargs="+", required=True); a.add_argument("--force", action="store_true")
    a = sp.add_parser("info"); a.add_argument("--config", default="config/pipeline.yaml")
    a = sp.add_parser("aux-config"); a.add_argument("--config", default="config/pipeline.yaml"); a.add_argument("--out", required=True)
    a = sp.add_parser("union"); a.add_argument("--config", default="config/pipeline.yaml")
    a.add_argument("--primary", default="dataset_out"); a.add_argument("--workers", type=int, default=4)
    a = sp.add_parser("gate"); a.add_argument("--config", default="config/pipeline.yaml")
    a.add_argument("--primary", default="dataset_out"); a.add_argument("--labels", default="dataset_out/labels_final.csv")
    sp.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        from .selftest import main as st
        return st()
    if a.cmd == "cache":
        from .doc import build_book
        for b in a.books:
            build_book(REPO / "prepared" / b, force=a.force)
        return 0
    cfg = _cfg(a.config)
    h = RT.cfg_of(cfg)
    aux = _abs(h["aux_dir"]); work = aux.parent / "work"
    if a.cmd == "info":
        print(f"HL_ENABLED={int(h['enabled'])}")
        print(f"HL_AUX_DIR={aux}")
        print(f"HL_WORK={work}")
        print(f"HL_GATE={h['gate']}")
        print(f"HL_VDP={int(h['vdp_gate'])}")
        return 0
    if not h["enabled"]:
        print("[hai_luot] stt_hai_luot.enabled = false -> không làm gì", file=sys.stderr)
        return 0
    if a.cmd == "aux-config":
        RT.aux_config(_abs(a.config), _abs(a.out))
        print(f"[hai_luot] config bản dựng phụ ({h['aux_read']}) -> {a.out}")
        return 0
    if a.cmd == "union":
        RT.union(_abs(a.primary), aux, work, a.workers)
        return 0
    if a.cmd == "gate":
        RT.gate(_abs(a.labels), work, _abs(a.primary), aux, h["gate"], h["vdp_gate"])
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
