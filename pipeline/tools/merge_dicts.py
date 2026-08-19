"""Gộp các nguồn từ điển rời vào HAI file chính, có sao lưu và có báo cáo delta.

  Dict/QuocNgu_SinoNom.csv   <- + QuocNgu_SinoNom_TongHop3.xlsx
                                + cột âm Hán-Việt của SinoNom_Similar_HVThiVien.xlsx
  Dict/SinoNom_Similar.csv   <- + SinoNom_Similar_Đạt_v0.xlsx
                                + cột "Nearly similar morphology" của HVThiVien

Quy tắc: HỢP (union), không bao giờ thay thế. Đã đo được rằng
`QuocNgu_SinoNom_TongHop3.xlsx` THIẾU 4.355 cặp so với CSV đang dùng — thay thế thẳng
sẽ mất dữ liệu âm thầm. Chỉ lấy phần bổ sung.

Âm Quốc ngữ mới phải qua `is_plausible_qn_syllable` — cột âm của HVThiVien lẫn pinyin
('ti4', 'tie1', 'tiao3') và những thứ đó tuyệt đối không được vào từ điển âm.

Bản trước khi gộp được ghi ra `Dict/_backup/<tên>.<dấu-thời-gian>.csv`.

    .venv/bin/python -m pipeline.tools.merge_dicts [--apply]     # thiếu --apply = chỉ xem
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import shutil
import unicodedata as ud
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DICT = REPO / "Dict"
QN_CSV = DICT / "QuocNgu_SinoNom.csv"
SIM_CSV = DICT / "SinoNom_Similar.csv"


def _parse_list(v) -> list[str]:
    if not isinstance(v, str) or not v.strip():
        return []
    v = v.strip()
    if v.startswith("["):
        try:
            out = ast.literal_eval(v)
            return [str(x) for x in out] if isinstance(out, list) else []
        except (ValueError, SyntaxError):
            return []
    return [c for c in v.replace(",", " ").split() if c]


def merge_qn(stamp: str, apply: bool) -> dict:
    from core.text.text_utils import is_plausible_qn_syllable
    cur = pd.read_csv(QN_CSV, dtype=str)
    pairs: dict[str, list[str]] = collections.OrderedDict()
    for q, s in zip(cur.iloc[:, 0], cur.iloc[:, 1]):
        q, s = str(q).strip(), str(s).strip()
        if q and s:
            pairs.setdefault(q, [])
            if s not in pairs[q]:
                pairs[q].append(s)
    before = sum(len(v) for v in pairs.values())

    added, rejected = [], []

    def offer(q: str, s: str):
        q = ud.normalize("NFC", str(q).strip().lower())
        s = str(s).strip()
        if not q or not s:
            return
        if not is_plausible_qn_syllable(q):
            rejected.append((q, s))
            return
        if s not in pairs.setdefault(q, []):
            pairs[q].append(s)
            added.append((q, s))

    # bảng tổng hợp bản xlsx — KHÔNG có dòng tiêu đề, dòng đầu đã là dữ liệu
    f = DICT / "QuocNgu_SinoNom_TongHop3.xlsx"
    if f.exists():
        x = pd.read_excel(f, dtype=str, header=None)
        for a, b in zip(x[0], x[1]):
            if isinstance(a, str) and isinstance(b, str):
                offer(a, b)
    # cột âm Hán-Việt của HVThiVien: chữ -> âm
    f = DICT / "SinoNom_Similar_HVThiVien.xlsx"
    if f.exists():
        h = pd.read_excel(f, dtype=str)
        for ch, am in zip(h.iloc[:, 0], h.iloc[:, 1]):
            if isinstance(ch, str) and isinstance(am, str):
                for a in am.replace(",", " ").split():
                    offer(a, ch)

    after = sum(len(v) for v in pairs.values())
    if apply:
        shutil.copy2(QN_CSV, DICT / "_backup" / f"QuocNgu_SinoNom.{stamp}.csv")
        with open(QN_CSV, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["QuocNgu", "SinoNom"])
            for q in sorted(pairs):
                for s in pairs[q]:
                    w.writerow([q, s])
    return {"before": before, "after": after, "added": added, "rejected": rejected}


def merge_sim(stamp: str, apply: bool) -> dict:
    cur = pd.read_csv(SIM_CSV, dtype=str)
    m: dict[str, list[str]] = collections.OrderedDict()
    for a, b in zip(cur.iloc[:, 0], cur.iloc[:, 1]):
        a = str(a).strip()
        if a:
            m.setdefault(a, [])
            for c in _parse_list(b):
                if c not in m[a]:
                    m[a].append(c)
    before_edges = {tuple(sorted((k, v))) for k, vs in m.items() for v in vs if k != v}

    def offer(a: str, b: str):
        a, b = str(a).strip(), str(b).strip()
        if not a or not b or a == b:
            return
        # đối xứng: từ điển gốc lưu một chiều, nhưng "giống nhau" là quan hệ hai chiều
        for x, y in ((a, b), (b, a)):
            if y not in m.setdefault(x, []):
                m[x].append(y)

    for f, col in ((DICT / "SinoNom_Similar_Đạt_v0.xlsx", 1),
                   (DICT / "SinoNom_Similar_HVThiVien.xlsx", 2)):
        if not f.exists():
            continue
        d = pd.read_excel(f, dtype=str)
        for a, b in zip(d.iloc[:, 0], d.iloc[:, col]):
            for c in _parse_list(b):
                offer(a, c)

    after_edges = {tuple(sorted((k, v))) for k, vs in m.items() for v in vs if k != v}
    if apply:
        shutil.copy2(SIM_CSV, DICT / "_backup" / f"SinoNom_Similar.{stamp}.csv")
        with open(SIM_CSV, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Input Character", "Top 20 Similar Characters"])
            for a in sorted(m):
                if m[a]:
                    w.writerow([a, repr(m[a])])
    return {"keys_before": len(cur), "keys_after": len(m),
            "edges_before": len(before_edges), "edges_after": len(after_edges),
            "edges_added": len(after_edges - before_edges)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.merge_dicts")
    ap.add_argument("--apply", action="store_true", help="thiếu cờ này = chỉ in delta")
    ap.add_argument("--stamp", default="pre_merge", help="hậu tố file sao lưu")
    args = ap.parse_args(argv)
    if args.apply:
        (DICT / "_backup").mkdir(parents=True, exist_ok=True)

    q = merge_qn(args.stamp, args.apply)
    print(f"[QN↔Nôm]  {q['before']:,} -> {q['after']:,} cặp (thêm {len(q['added']):,}); "
          f"loại vì âm không hợp lệ: {len(q['rejected'])}")
    for a in q["added"][:10]:
        print(f"           + {a[0]} → {a[1]}")
    for r in q["rejected"][:5]:
        print(f"           - loại: {r[0]} → {r[1]}")

    s = merge_sim(args.stamp, args.apply)
    print(f"[tự dạng] {s['keys_before']:,} -> {s['keys_after']:,} chữ khoá; "
          f"{s['edges_before']:,} -> {s['edges_after']:,} cặp vô hướng "
          f"(thêm {s['edges_added']:,})")
    print("[apply]" if args.apply else "[thử] chưa ghi gì — thêm --apply để ghi thật")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
