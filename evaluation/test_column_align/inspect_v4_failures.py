"""List exact v4 failures: parser_v4 < 9 OR hybrid cols != qn."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parser_v4 import parse_v4
from probe import load_qn_to_nom
from run_full import nom_cols_hybrid
from export_dataset_v4 import get_qn_lines_v4


def main():
    repo = Path(__file__).resolve().parents[2]
    qn_to_nom = load_qn_to_nom(str(repo / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"))
    qn_dict = set(qn_to_nom.keys())

    parser_miss = []   # qn parse < 9
    col_mismatch = []  # cols != 9 when qn=9
    underflow = []     # all =9 but underflow per col

    for book in ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]:
        book_dir = repo / "prepared" / book
        for af in sorted((book_dir / "aligned").glob("page_*_aligned.json")):
            page = af.stem.replace("_aligned", "")
            ocr_path = book_dir / "detected" / f"{page}_ocr_cache.json"
            if not ocr_path.exists():
                continue
            ocr_data = json.load(open(ocr_path))
            qn_lines, _ = get_qn_lines_v4(book_dir, page, qn_dict)
            cols = nom_cols_hybrid(ocr_data.get("columns", []), min_len=4)
            n_qn = len(qn_lines)
            n_c = len(cols)
            ref = f"{book}/{page}"
            if n_qn < 9:
                missing = [i for i in range(1, 10) if i not in qn_lines]
                parser_miss.append((ref, n_qn, missing))
            elif n_c != 9:
                col_mismatch.append((ref, n_c, n_qn))
            else:
                under = []
                qkeys = sorted(qn_lines.keys())
                for i in range(9):
                    a = len(cols[i]["chars"])
                    e = len(qn_lines[qkeys[i]])
                    if a < e:
                        under.append((qkeys[i], a, e))
                if under:
                    underflow.append((ref, under))

    print(f"parser_miss: {len(parser_miss)}")
    for r in parser_miss:
        print(f"  {r}")
    print(f"\ncol_mismatch: {len(col_mismatch)}")
    for r in col_mismatch:
        print(f"  {r}")
    print(f"\nunderflow (residual after reseg attempted): {len(underflow)}")
    for r in underflow[:10]:
        print(f"  {r}")


if __name__ == "__main__":
    main()
