"""Kiểm lại quan hệ trùng lặp giữa các file trong Dict/ — chạy được, không phải ghi chú.

Khẳng định trong `Dict/_superseded/README.md` ("là tập con", "chỉ đảo cột", "trùng khít")
là các khẳng định ĐO ĐƯỢC. Nếu ai đó thay một file trong Dict/ mà quan hệ đó không còn
đúng, lệnh này phải báo FAIL — nếu không thì cái README kia chỉ là niềm tin.

    .venv/bin/python -m pipeline.tools.check_dict_duplicates
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DICT = Path(__file__).resolve().parents[2] / "Dict"
OLD = DICT / "_superseded"


def _pairs_qn_sn(df: pd.DataFrame, qn_col: str, sn_col: str) -> set[tuple[str, str]]:
    return {(str(q).strip(), str(s).strip()) for q, s in zip(df[qn_col], df[sn_col])}


def main(argv: list[str] | None = None) -> int:
    checks: list[tuple[str, bool, str]] = []

    tonghop = _pairs_qn_sn(pd.read_csv(DICT / "QuocNgu_SinoNom.csv", dtype=str),
                           "QuocNgu", "SinoNom")
    if (OLD / "QuocNgu_SinoNom_Dic.xlsx").exists():
        a = _pairs_qn_sn(pd.read_excel(OLD / "QuocNgu_SinoNom_Dic.xlsx", dtype=str),
                         "QuocNgu", "SinoNom")
        checks.append(("QuocNgu_SinoNom_Dic.xlsx ⊂ TongHop3", a <= tonghop,
                       f"{len(a):,} cặp, ngoài tập cha: {len(a - tonghop):,}, "
                       f"TongHop3 thêm {len(tonghop - a):,}"))
        if (OLD / "SinoNom_QuocNgu_Dic.xlsx").exists():
            c_df = pd.read_excel(OLD / "SinoNom_QuocNgu_Dic.xlsx", dtype=str)
            c = _pairs_qn_sn(c_df, "QuocNgu", "SinoNom")   # cùng cặp, chỉ khác thứ tự cột
            checks.append(("SinoNom_QuocNgu_Dic.xlsx ≡ QuocNgu_SinoNom_Dic.xlsx (đảo cột)",
                           a == c, f"chỉ ở A: {len(a - c):,}, chỉ ở C: {len(c - a):,}"))

    if (OLD / "SinoNom_Similar_Dic.xlsx").exists():
        # Sau khi gộp thêm Đạt_v0 + HVThiVien (2026-08-19), bản CSV KHÔNG còn trùng khít
        # bản xlsx cũ — nó là tập cha. Bất biến cần giữ là KHÔNG MẤT liên kết nào của bản
        # cũ; kiểm bằng bao hàm chứ không bằng bằng nhau.
        import ast

        def _edges(df) -> set[tuple[str, str]]:
            out = set()
            for a, b in zip(df.iloc[:, 0], df.iloc[:, 1]):
                a = str(a).strip()
                try:
                    lst = ast.literal_eval(str(b).strip())
                except (ValueError, SyntaxError):
                    lst = []
                for c in lst if isinstance(lst, list) else []:
                    if a and c and a != str(c):
                        out.add(tuple(sorted((a, str(c)))))
            return out

        e_old = _edges(pd.read_excel(OLD / "SinoNom_Similar_Dic.xlsx", dtype=str))
        e_new = _edges(pd.read_csv(DICT / "SinoNom_Similar.csv", dtype=str))
        checks.append(("SinoNom_Similar_Dic.xlsx ⊆ SinoNom_Similar.csv (không mất liên kết)",
                       e_old <= e_new,
                       f"cũ {len(e_old):,} cặp, mất {len(e_old - e_new):,}, "
                       f"bản gộp thêm {len(e_new - e_old):,}"))

    if not checks:
        print("không có file cũ nào trong Dict/_superseded — không có gì để kiểm")
        return 0
    ok = True
    for name, passed, detail in checks:
        print(f"[{'OK  ' if passed else 'FAIL'}] {name} — {detail}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
