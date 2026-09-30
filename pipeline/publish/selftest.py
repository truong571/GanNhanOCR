"""Selftest tập công bố theo GOLD chính xác (không pytest).

    .venv/bin/python -m pipeline.publish.selftest

30/09: công cụ công bố cũ (split/metadata/datasheet/export/validate) đã xoá — chỉ còn kiểm `gold_exact_release`.
Exit 0 = mọi phép kiểm PASS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


def test_gold_exact_release():
    """28/09: tập công bố theo GOLD chính xác — tập ẢNH chỉ ô gold_exact = ok (ảnh = crop chuẩn), mọi dòng khác vào tập VĂN BẢN
    kèm lý do; KHÔNG chia tập (30/09, A-10: không cột split/lobo_group, cột split_hint cũ bị bỏ); bộ đánh giá chỉ mang
    evaluation_only; gold_exact.csv lệch labels -> ReleaseError; ghi đĩa + thiếu ảnh -> không ghi."""
    import hashlib
    import tempfile
    from . import gold_exact_release as GR
    rng = np.random.default_rng(7)
    rows, ge = [], []
    for i in range(600):
        bs = ["SachThanhTruyen", "LucVanTien1916", "KimVanKieu1884"][i % 3]
        bk = {"SachThanhTruyen": ["stt2", "stt4"][i % 2], "LucVanTien1916": "lucvantien1916", "KimVanKieu1884": "kimvankieu1884"}[bs]
        tier = "GOLD" if i % 5 else "SYLLABLE"
        u = f"{bs}/{bk}/page_{i // 20:04d}/c{i % 9}/n{i}/s{i}"
        lab = "天地人水火山日月"[int(rng.integers(0, 8))]
        rows.append(dict(cell_uid=u, image=f"crops/{bs}/gold/x{i}.png", book_set=bs, book=bk, page=f"page_{i // 20:04d}",
                         column=str(i % 9), ocr_char=lab, syllable="x", label=lab, unicode="", tier=tier, rule="r", bbox="",
                         image_md5="", evaluation_only="1" if bs == "LucVanTien1916" else "0",
                         split_hint="eval" if bs == "LucVanTien1916" else "train", image_dup="0"))
        if tier == "GOLD":
            st = ["ok", "ok", "text_only", "uncertified", "review"][i % 5 if i % 5 else 0]
            ge.append(dict(cell_uid=u, label=lab, gold_exact=st, reason="" if st == "ok" else f"R_{st}",
                           evidence_level="suy_doan", policy_version="p",
                           crop_chuan=f"crops_chuan/{bs}/{bk}/x{i}.png" if st == "ok" else "",
                           crop_chuan_md5=hashlib.md5(f"x{i}".encode()).hexdigest() if st == "ok" else "",
                           crop_chuan_128=f"crops_chuan_128/{bs}/{bk}/x{i}.png" if st == "ok" else ""))
    L, E = pd.DataFrame(rows), pd.DataFrame(ge)
    I, T, rep = GR.build(L, E, eval_books={"lucvantien1916"})
    okn = int((E.gold_exact == "ok").sum())
    check("release: tập ảnh = đúng ô gold_exact ok", len(I) == okn and set(I.cell_uid) == set(E.cell_uid[E.gold_exact == "ok"]))
    check("release: ảnh = crop chuẩn, image_goc giữ đường gốc", I.image.str.startswith("crops_chuan/").all()
          and I.image_goc.str.startswith("crops/").all())
    check("release: tập văn bản = phần còn lại, không cột ảnh", len(T) == len(L) - okn and "image" not in T.columns)
    ex = rep["loai_khoi_tap_anh"]
    check("release: đếm lý do loại đủ (ồn ào)", sum(ex.values()) == len(T) and "tier:SYLLABLE" in ex
          and any(k.startswith("text_only:") for k in ex), ex)
    check("release: bất biến PASS", all(v is True for k, v in rep.items() if isinstance(v, bool)),
          {k: v for k, v in rep.items() if isinstance(v, bool)})
    ev = L.book_set == "LucVanTien1916"
    check("release: bộ đánh giá -> evaluation_only = 1 (ảnh + văn bản)",
          (I.evaluation_only[I.book_set == "LucVanTien1916"] == "1").all()
          and (T.evaluation_only[T.book_set == "LucVanTien1916"] == "1").all() and ev.any())
    chia = {"split", "split_hint", "lobo_group"}
    check("release: KHÔNG có cột chia tập (kể cả khi labels.csv cũ còn split_hint)",
          not (chia & (set(I.columns) | set(T.columns))) and rep.get("khong_cot_chia_tap") is True
          and not {"split_images", "split_text", "lobo", "classes_train"} & set(rep))
    I2, _, _ = GR.build(L, E, eval_books={"lucvantien1916"})
    check("release: tất định", I2.equals(I))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        L.to_csv(d / "labels.csv", index=False)
        E2 = E.iloc[1:]
        E2.to_csv(d / "gold_exact.csv", index=False)
        try:
            GR.run(d, check_files=False, log=lambda *a: None); check("release: gold_exact.csv lệch tập GOLD -> lỗi", False)
        except GR.ReleaseError:
            check("release: gold_exact.csv lệch tập GOLD -> ReleaseError", True)
        E.to_csv(d / "gold_exact.csv", index=False)
        try:
            GR.run(d, check_files=True, log=lambda *a: None); check("release: thiếu ảnh crop chuẩn -> lỗi, không ghi", False)
        except GR.ReleaseError:
            check("release: thiếu ảnh crop chuẩn -> ReleaseError, không ghi", not (d / GR.OUT_DIRNAME).exists())
        for p, u in zip(I.image, I.cell_uid):
            f = d / p; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(f"x{u.rsplit('/n', 1)[1].split('/')[0]}".encode())
        rec = GR.run(d, check_files=True, log=lambda *a: None)
        out = d / GR.OUT_DIRNAME
        check("release: ghi đủ tệp + md5 khớp", all((out / n).exists() for n in ("images.csv", "text.csv", "EXCLUSIONS.json",
                                                                               "RELEASE.md", "CHECKSUMS.txt"))
              and rec["invariants"]["md5_crop_khop_gold_exact"] is True
              and len(pd.read_csv(out / "images.csv")) == okn)


def test_package_khong_con_chia_tap():
    """Gói publish không còn module/lệnh chia tập (A-10)."""
    from . import cli
    here = Path(__file__).resolve().parent
    check("không còn module của công cụ cũ",
          not [n for n in ("splits", "export", "validate", "metadata", "datasheet", "hashing") if (here / f"{n}.py").exists()])
    subs = next(a for a in cli.build_parser()._actions if a.dest == "cmd").choices
    check("CLI chỉ còn lệnh gold-exact", set(subs) == {"gold-exact"}, set(subs))


def main() -> int:
    print("=" * 64)
    print("PUBLISH SELFTEST (gold-exact, không chia tập)")
    print("=" * 64)
    test_gold_exact_release()
    test_package_khong_con_chia_tap()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
