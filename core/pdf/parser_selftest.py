"""Self-test for QN column parsing — always recover exactly 9 REAL columns (no loss).

    .venv/bin/python -m core.pdf.parser_selftest
Covers the real VietOCR marker corruptions: clean, period-lost, extra-digit (28->8),
mid-line ('Thế 7.'), dash-replaced number, and dropped leading column. Exit 0 = pass.
"""
from __future__ import annotations

from core.pdf.pdf_parser import parse_numbered_lines, build_transcription_columns

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


def cols_of(text):
    return build_transcription_columns(parse_numbered_lines(text))


def nums(c):
    return [x["column"] for x in c]


N = "mot hai ba bon nam"          # 5 "syllables" per column


def test():
    # 1. clean 9
    c = cols_of("\n".join(f"{i}. {N}" for i in range(1, 10)))
    check("clean 9 -> [1..9]", nums(c) == list(range(1, 10)))
    check("clean 9 all non-empty", all(x["num_syllables"] > 0 for x in c))

    # 2. period-lost marker on col 2 ('2 ' not '2.')
    lines = [f"{i}. {N}" for i in range(1, 10)]
    lines[1] = f"2 {N}"
    c = cols_of("\n".join(lines))
    check("period-lost -> [1..9]", nums(c) == list(range(1, 10)))
    check("period-lost col2 has content", c[1]["num_syllables"] == 5)

    # 3. extra-digit marker '28.' for column 8 (n%10 == expected)
    lines = [f"{i}. {N}" for i in range(1, 8)] + ["28. tam tam tam tam tam", f"9. {N}"]
    c = cols_of("\n".join(lines))
    check("extra-digit 28->8 -> [1..9]", nums(c) == list(range(1, 10)))
    check("col8 recovered from 28", c[7]["num_syllables"] == 5)

    # 4. mid-line marker 'word 7.'
    lines = [f"{i}. {N}" for i in range(1, 7)] + [f"The 7. {N}"] + [f"{i}. {N}" for i in (8, 9)]
    c = cols_of("\n".join(lines))
    check("mid-line 'The 7.' -> [1..9]", nums(c) == list(range(1, 10)))

    # 5. spurious reference '17.' inside body (expected != 7) is ignored
    txt = "\n".join(f"{i}. {N}" for i in range(1, 10)).replace(
        f"3. {N}", f"3. {N}\n17. tham chieu")
    c = cols_of(txt)
    check("spurious 17. ignored -> [1..9]", nums(c) == list(range(1, 10)))
    check("no column 17", 17 not in nums(c))

    # 6. dash-replaced markers for cols 6,7 (split at the true dash boundaries)
    lines = [f"{i}. {N}" for i in (1, 2, 3, 4, 5)] + \
            ["- col six words here now", "- col seven words here", f"8. {N}", f"9. {N}"]
    c = cols_of("\n".join(lines))
    check("dash-replaced -> [1..9]", nums(c) == list(range(1, 10)))
    check("dash cols 6,7 have REAL content (not empty)",
          c[5]["num_syllables"] > 0 and c[6]["num_syllables"] > 0)
    check("dash col6 = the six-line", "six" in c[5]["cleaned_text"])
    check("dash col7 = the seven-line", "seven" in c[6]["cleaned_text"])

    # 7. dropped leading column (content before first detected marker) -> col 1, not empty
    c = cols_of("- Phe ro noi dung cot mot\n" + "\n".join(f"{i}. {N}" for i in range(2, 10)))
    check("dropped-leading -> [1..9]", nums(c) == list(range(1, 10)))
    check("col1 keeps leading content (not empty)",
          c[0]["num_syllables"] > 0 and "Phe" in c[0]["cleaned_text"])

    # 8. always exactly 9 columns
    for txt in ("", "no markers", "\n".join(f"{i}. {N}" for i in range(1, 10))):
        check(f"always 9 cols ({txt[:10]!r})", len(cols_of(txt)) == 9)


def main() -> int:
    print("=" * 56)
    print("QN COLUMN PARSER SELFTEST")
    print("=" * 56)
    test()
    print("=" * 56)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 56)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
