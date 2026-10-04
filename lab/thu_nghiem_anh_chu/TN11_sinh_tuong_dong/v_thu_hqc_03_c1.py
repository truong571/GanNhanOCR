"""v_thu_hqc_03_c1.py — nhúng glyph HQC2 ("nền giấy sách", cấu hình chọn ở v_thu_hqc_02_tune, K=3 mảng giấy trung bình embedding) cho
MỌI chữ ứng viên của các ô có sự thật (font) + mọi ảnh FD có trong ứng viên, CPU, chạy tiếp được (lưu theo lô).
Thứ tự ưu tiên để kết quả từng phần dùng được: ký tự của mẫu 1000 ô -> mẫu 3000 ô -> phần còn lại.
  .venv/bin/python .../v_thu_hqc_03_c1.py --variant primary --upto all [--books B34 B18 L16 TK] [--what both|font|fd] [--patches 0 1 2]
  .venv/bin/python .../v_thu_hqc_03_c1.py --variant literal --upto s1000       # công thức nguyên văn: norm=False, K=1, khung glyph
  --variant ctrlwhite   (đối chứng: nền TRẮNG cùng quang học đã chọn, K=1)  --variant paperonly (giấy-chỉ: khung glyph, d=0, σ=0, norm=True, K=3)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = H.OUT


def char_lists(b):
    """(font_chars theo ưu tiên [(tên, [ký tự])], fd_chars) — chữ có glyph font (f_font != NaN) / ảnh FD (f_fd != NaN) trong ứng viên của ô có sự thật.
    Dựng trong TIẾN TRÌNH CON (RAM: bảng ứng viên B18 ~1 GB tạm) rồi cache vào chars_<bộ>.json."""
    f = OUT / f"chars_{b}.json"
    if not f.exists():
        import subprocess
        subprocess.run([sys.executable, __file__, "--make-chars", b], check=True)
    j = json.loads(f.read_text(encoding="utf-8"))
    return [(n, c) for n, c in j["order"]], j["fd"]


def make_chars500(b):
    """ký tự font của 500 ô đầu của hoán vị (s1000[:500]) -> chars500_<bộ>.json (tập con của s1000)."""
    Bk = H.load_eval_book(b, with_E=False)
    F = Bk["F"]
    S = np.load(OUT / f"mau_{b}.npz")["s1000"][:500]
    m = F.i.isin(set(int(x) for x in S)).values & F.f_font.notna().values
    H.jdump(dict(chars=F.c.astype(str)[m].drop_duplicates().tolist()), OUT / f"chars500_{b}.json")


def chars500(b):
    f = OUT / f"chars500_{b}.json"
    if not f.exists():
        import subprocess
        subprocess.run([sys.executable, __file__, "--make-chars500", b], check=True)
    return json.loads(f.read_text(encoding="utf-8"))["chars"]


def make_chars(b):
    Bk = H.load_eval_book(b, with_E=False)
    F = Bk["F"]
    S = np.load(OUT / f"mau_{b}.npz")
    seen = set()
    order = []
    for name, cells in (("s1000", S["s1000"]), ("s3000", S["s3000"]), ("all", Bk["cells"])):
        m = F.i.isin(set(int(x) for x in cells)).values & F.f_font.notna().values
        ch = [c for c in F.c.astype(str)[m].drop_duplicates().tolist() if c not in seen]
        seen |= set(ch); order.append((name, ch))
    fd = sorted(set(F.c.astype(str)[F.f_fd.notna().values]))
    H.jdump(dict(order=order, fd=fd), OUT / f"chars_{b}.json")


def cfg_for(variant, b, tune, kind, pk=0):
    """cấu hình 1 mảng giấy (mảng thứ pk của bộ ba A) — K=3 = trung bình chuẩn hoá của 3 lần chạy pk=0,1,2 (lưu riêng để chạy tiếp/đánh giá K)."""
    pm = H.paper_model(b)
    ids = pm["patch_ids_a"]
    if variant == "primary":
        c = dict(tune[b]["cfg_font" if kind == "font" else "cfg_fd"]); c.update(K=1, patch_ids=[ids[pk]], white_bg=False)
        return c
    if variant == "literal":
        return dict(P=None, d=0, sigma=0.0, norm=False, K=1, patch_ids=ids[:1], white_bg=False)
    if variant == "paperonly":
        return dict(P=None, d=0, sigma=0.0, norm=True, K=1, patch_ids=ids[:1], white_bg=False)
    if variant == "ctrlwhite":
        c = dict(tune[b]["cfg_font" if kind == "font" else "cfg_fd"]); c.update(K=1, patch_ids=ids[:1], white_bg=True)
        return c
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--upto", default="all", choices=["s1000", "s3000", "all"])
    ap.add_argument("--books", nargs="*", default=["B34", "B18", "L16", "TK"])
    ap.add_argument("--patches", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--what", default="both", choices=["both", "font", "fd"])
    ap.add_argument("--make-chars", default=None)
    ap.add_argument("--make-chars500", default=None)
    ap.add_argument("--only500", action="store_true", help="chỉ nhúng ký tự của 500 ô đầu (K=3 trên mẫu nhỏ)")
    a = ap.parse_args()
    if a.make_chars:
        make_chars(a.make_chars); return
    if a.make_chars500:
        make_chars500(a.make_chars500); return
    tune = json.loads((OUT / "tune.json").read_text(encoding="utf-8"))
    pool = H.make_pool()
    if a.only500:
        pks = a.patches if a.variant == "primary" else [0]
        for pk in pks:
            for b in a.books:
                chars = chars500(b)
                cfg = cfg_for(a.variant, b, tune, "font", pk)
                tag = f"{a.variant}" if a.variant != "primary" else f"primary_p{pk}"
                H.log(f"== {b} {tag} (500 ô đầu): {len(chars)} chữ font, cfg {cfg}")
                H.embed_chars(pool, "paper", "font", chars, OUT / "emb" / f"{tag}_{b}_font.npz", book=b, cfg=cfg, tag=f"{tag}.{b}.font")
        pool.close(); pool.join(); H.log("XONG"); return
    lists = {b: char_lists(b) for b in a.books}
    allst = ["s1000", "s3000", "all"]
    stages = allst[: allst.index(a.upto) + 1]
    pks = a.patches if a.variant == "primary" else [0]
    for pk in pks:
        for st in stages:
            for b in a.books:
                order, fd = lists[b]
                chars = [c for name, ch in order if name in stages[: stages.index(st) + 1] for c in ch]
                cfg = cfg_for(a.variant, b, tune, "font", pk)
                tag = f"{a.variant}" if a.variant != "primary" else f"primary_p{pk}"
                if a.what in ("both", "font"):
                    H.log(f"== {b} {tag} giai đoạn {st}: {len(chars)} chữ font, cfg {cfg}")
                    H.embed_chars(pool, "paper", "font", chars, OUT / "emb" / f"{tag}_{b}_font.npz", book=b, cfg=cfg, tag=f"{tag}.{b}.font")
                if a.what in ("both", "fd") and st == stages[0]:
                    cf = cfg_for(a.variant, b, tune, "fd", pk)
                    H.embed_chars(pool, "paper", "fd", fd, OUT / "emb" / f"{tag}_{b}_fd.npz", book=b, cfg=cf, tag=f"{tag}.{b}.fd")
    pool.close(); pool.join()
    H.log("XONG")


if __name__ == "__main__":
    main()
