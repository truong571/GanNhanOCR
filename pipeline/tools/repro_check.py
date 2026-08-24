"""Chốt chặn TÁI LẬP — bắt các lớp lệch mà T6 phát hiện được.

VÌ SAO CÓ TỆP NÀY
-----------------
`check_evidence.sh` chứng minh "bộ đem đo = bộ đem nộp", `update_bang_so_lieu --check`
chứng minh tài liệu khớp đĩa, `rebuild_proto_index --check` chứng minh chỉ mục đúng thế
hệ. Nhưng cả ba đều KHÔNG bắt được các lớp lệch mà T6 tìm ra:

  R1  CÂY LÀM VIỆC BẨN. `evidence()` ghi commit SHA, nhưng nếu cây bẩn thì mã thực thi
      KHÁC mã ở commit đó — hai lần chạy cùng hash vẫn không chứng minh gì về phiên bản.
  R2  CACHE NGUYÊN MẪU S3 CHẾT. `visual_signal.py:195` ký cache bằng **mtime** chứ không
      bằng băm nội dung: `f"{index.mtime_ns}|{ckpt.mtime_ns}|{PROTO_K}"`. Nên bản
      `s3_proto_cache.pkl` commit trong git (7 MB) có chữ ký KHÔNG BAO GIỜ khớp sau một
      lần clone — mtime là lúc checkout. Nó luôn phải dựng lại, và một lần dựng lại thay
      đổi tier SILVER/SYLLABLE (đo được: index cũ→mới làm SILVER −242, SYLLABLE +158).
  R3  RNG LỌT VÀO ĐƯỜNG BUILD. Tính tất định hiện nay dựa hoàn toàn vào việc KHÔNG CÓ
      nguồn ngẫu nhiên nào trong `align_engine`/`remediation`. Đó là một bất biến không
      ai canh; thêm một `random.shuffle` là mất tất định mà không phép kiểm nào kêu.

    python -m pipeline.tools.repro_check           # in trạng thái
    python -m pipeline.tools.repro_check --check   # exit 1 nếu có lệch
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
INDEX = REPO / "pipeline" / "align_engine" / "data" / "index.csv"
CACHE = REPO / "pipeline" / "align_engine" / "s3_proto_cache.pkl"
CKPT_CANDS = ("nom-embed/best.pt",)
# Đường chạy bước 3-7. RNG ở ngoài các thư mục này (ví dụ lab/, ground_truth/) thì vô hại.
BUILD_PATHS = ("pipeline/align_engine", "pipeline/remediation", "core/align",
               "core/text", "core/image")
RNG = re.compile(r"\b(random\.(random|choice|shuffle|sample|randint|seed)"
                 r"|np\.random|numpy\.random|torch\.rand|\.shuffle\()")

# MIỄN TRỪ CÓ LÝ DO ĐÃ KIỂM CHỨNG. Mỗi mục phải kèm bằng chứng vì sao dòng đó KHÔNG
# tới được đường suy luận. Gỡ miễn trừ phải là hành động CÓ Ý THỨC — đó là điểm của
# việc bắt ghi lý do ở đây.
ALLOW = {
    "pipeline/align_engine/nom_classifier/model.py": (
        "torch.randn trong ArcMargin.__init__ (dau ArcFace, CHI dung khi huan luyen). "
        "Kiem 2026-08-24: grep ArcMargin( chi khop DUNG dong dinh nghia lop, khong noi "
        "nao khoi tao; infer.py:31 chi nap ck['backbone']. Gia tri ngau nhien con bi "
        "nn.init.xavier_uniform_ ghi de ngay dong sau. Khong toi duoc duong suy luan."),
}


def r1_worktree() -> tuple[bool, str]:
    """Cây làm việc có sạch không (bỏ qua submodule: nom-embed luôn bẩn do artefact LFS)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no",
             "--ignore-submodules=all"],
            cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:                                   # pragma: no cover
        return False, f"không chạy được git ({e})"
    if not out:
        return True, "SẠCH — commit trong chuỗi bằng chứng định danh đúng mã đã chạy"
    n = len(out.splitlines())
    head = "; ".join(l.strip() for l in out.splitlines()[:3])
    return False, f"BẨN ({n} tệp) — commit KHÔNG định danh được mã đã chạy: {head}"


def r2_proto_cache() -> tuple[bool, str]:
    """Chữ ký cache nguyên mẫu có khớp mtime hiện tại của index.csv + checkpoint không."""
    if not CACHE.exists():
        return True, "không có cache — sẽ dựng lại, tất định"
    try:
        sig = pickle.load(open(CACHE, "rb")).get("__sig__")
    except Exception as e:
        return False, f"cache hỏng, không đọc được ({type(e).__name__})"
    if not INDEX.exists():
        return False, "không có index.csv để đối chiếu"
    ckpt = next((REPO / c for c in CKPT_CANDS if (REPO / c).exists()), None)
    if ckpt is None:
        return True, "không tìm thấy checkpoint để đối chiếu — bỏ qua"
    want_pref = f"{INDEX.stat().st_mtime_ns}|{ckpt.stat().st_mtime_ns}|"
    if isinstance(sig, str) and sig.startswith(want_pref):
        return True, "chữ ký KHỚP — S3 dùng lại nguyên mẫu, không dựng lại"
    return False, ("chữ ký LỆCH — S3 sẽ DỰNG LẠI nguyên mẫu ở lần build kế tiếp, và "
                   "điều đó đổi tier SILVER/SYLLABLE (đo T6: SILVER −242, SYLLABLE +158). "
                   "Không phải lỗi nếu bạn CỐ Ý đổi index.csv/checkpoint — nhưng phải biết.")


def r3_no_rng() -> tuple[bool, str]:
    """Đường build có nguồn ngẫu nhiên nào lọt vào không."""
    hits: list[str] = []
    for d in BUILD_PATHS:
        base = REPO / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py")):
            if f.name.startswith("selftest") or "/lab/" in str(f):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                if RNG.search(code) and str(f.relative_to(REPO)) not in ALLOW:
                    hits.append(f"{f.relative_to(REPO)}:{i}")
    if not hits:
        return True, (f"0 nguồn ngẫu nhiên tới được đường build "
                      f"({len(BUILD_PATHS)} thư mục, {len(ALLOW)} miễn trừ có lý do)")
    return False, ("RNG LỌT VÀO ĐƯỜNG BUILD — tính tất định không còn được bảo đảm: "
                   + ", ".join(hits[:5]))


CHECKS = (("R1 cây làm việc", r1_worktree),
          ("R2 cache nguyên mẫu S3", r2_proto_cache),
          ("R3 không có RNG trong đường build", r3_no_rng))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.repro_check")
    ap.add_argument("--check", action="store_true", help="exit 1 nếu có lệch")
    args = ap.parse_args(argv)
    bad = 0
    for name, fn in CHECKS:
        ok, msg = fn()
        if not ok:
            bad += 1
        print(f"  {'OK  ' if ok else 'LỆCH'}  {name:34} {msg}")
    if bad:
        print(f"[repro] {bad}/{len(CHECKS)} phép kiểm lệch.", file=sys.stderr)
    return 1 if (bad and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
