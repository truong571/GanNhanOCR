"""List pages that need hand inspection.

Buckets:
  - qn_parse_lt9 : parser_v3 returned < 9 lines.
  - nom_cols_ne_qn : hybrid col count != QN line count.
  - col_underflow : at least one col has cluster.actual < qn.expected.
  - page_seg_low_ink : at least one col has > 25% low-ink bboxes from projection.

Writes out/INSPECT.md with grouped page lists for hand review.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parser_v3 import parse_v3
from probe import load_qn_to_nom
from run_full import nom_cols_hybrid, get_qn_lines


def main():
    repo = Path(__file__).resolve().parents[2]
    qn_to_nom = load_qn_to_nom(str(repo / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"))
    qn_dict = set(qn_to_nom.keys())

    buckets = {
        "qn_parse_lt9": [],
        "nom_cols_ne_qn": [],
        "col_underflow": [],
    }

    for book in ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]:
        book_dir = repo / "prepared" / book
        for af in sorted((book_dir / "aligned").glob("page_*_aligned.json")):
            page = af.stem.replace("_aligned", "")
            ocr_path = book_dir / "detected" / f"{page}_ocr_cache.json"
            if not ocr_path.exists():
                continue
            ocr_data = json.load(open(ocr_path))
            qn_lines, _ = get_qn_lines(book_dir, page, qn_dict)
            cols = nom_cols_hybrid(ocr_data.get("columns", []), min_len=4)
            n_qn = len(qn_lines)
            n_cols = len(cols)

            ref = f"{book}/{page}"
            if n_qn < 9:
                buckets["qn_parse_lt9"].append(f"{ref} (qn={n_qn})")
            elif n_cols != n_qn:
                buckets["nom_cols_ne_qn"].append(f"{ref} (cols={n_cols}/qn={n_qn})")
            else:
                under = []
                for i in range(n_cols):
                    actual = len(cols[i]["chars"])
                    expected = len(qn_lines[sorted(qn_lines.keys())[i]])
                    if actual < expected:
                        under.append(f"col{i+1}:{actual}/{expected}")
                if under:
                    buckets["col_underflow"].append(
                        f"{ref} ({', '.join(under)})")

    out = Path(__file__).parent / "out" / "INSPECT.md"
    md = ["# Pages needing hand inspection\n"]
    for k, v in buckets.items():
        md.append(f"\n## `{k}` — {len(v)} pages\n")
        for line in v:
            md.append(f"- {line}")
    out.write_text("\n".join(md) + "\n")
    print(f"qn_parse_lt9:   {len(buckets['qn_parse_lt9'])}")
    print(f"nom_cols_ne_qn: {len(buckets['nom_cols_ne_qn'])}")
    print(f"col_underflow:  {len(buckets['col_underflow'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
