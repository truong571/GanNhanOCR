"""Selftest B-3 (visual_syl_gate) — bọc để scripts/run_all_selftests.sh gọi `-m` không tham số.

    .venv/bin/python -m pipeline.tools.visual_syl_gate_selftest
"""
from pipeline.tools.visual_syl_gate import _selftest

if __name__ == "__main__":
    raise SystemExit(_selftest())
