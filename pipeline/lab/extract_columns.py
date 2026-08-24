"""Trích ĐẦU VÀO THẬT của bước căn chỉnh (trước align) và cache lại cho T3.

VÌ SAO CẦN
----------
`perturb.load_columns` dựng lại cột từ BỘ NHÃN — tức từ ĐẦU RA của align. Ở đó số chữ
luôn bằng số âm, nên mọi cấu hình chi phí cho cùng một đường chéo:

    yield_del = 0, yield_ins = 0, và đổi COST_NODICT 0,9 -> 0,5 KHÔNG đổi một ô nào.

Chỉ số sản lượng như vậy **không phân biệt được cấu hình** — vô dụng làm trục Pareto.

Đầu vào THẬT (từ `_detect`, tức OCR Nôm + parser QN trước khi align) mới có m ≠ n:
đo trên `stt2/page_0012` là **4/9 cột** lệch độ dài. Đó chính là chỗ ma trận chi phí
phải quyết — và là chỗ duy nhất một cấu hình có thể tốt hơn cấu hình khác.

`_detect` tốn ~0,23s/trang (đọc ảnh + nhị phân hoá + dò cột) nên 445 trang ~100s. Trích
MỘT LẦN rồi cache; sau đó quét 144 cấu hình chỉ còn là chạy `realign_column` trên bộ nhớ.

    python -m pipeline.lab.extract_columns            # sinh lab/columns.pkl
    python -m pipeline.lab.extract_columns --stats    # xem thống kê cache
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "lab" / "columns.pkl"
BOOKS = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4",
         "stt11": "SachThanhTruyen11"}


def extract(verbose: bool = True) -> list[dict]:
    """Mỗi cột: {key, chars, syllables} — ĐẦU VÀO THẬT, chưa qua align."""
    sys.path.insert(0, str(REPO))
    import yaml
    from core.text.dictionary import dict_dir, load_qn_to_nom
    from pipeline.align_engine.align_production import _detect

    qn_keys = set(load_qn_to_nom(str(dict_dir() / "QuocNgu_SinoNom.csv")))
    cfg = yaml.safe_load((REPO / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
    out: list[dict] = []
    for code, book in sorted(BOOKS.items()):
        bd = REPO / "prepared" / book
        pages = sorted(p.stem for p in (bd / "pages").glob("*.png"))
        if verbose:
            print(f"  [{code}] {len(pages)} trang…", flush=True)
        for pg in pages:
            try:
                det = _detect(pg, bd, qn_keys)
            except Exception:
                continue
            if det is None:
                continue
            cols, qn_lines, iter_pairs, _binary, _ok = det
            for nom_idx, line_id in iter_pairs:
                chars = [c.get("ocr_char") or c.get("char") or ""
                         for c in cols[nom_idx].get("chars", [])]
                syls = [str(s).lower() for s in qn_lines.get(line_id, [])]
                if not chars or not syls:
                    continue
                out.append({"key": (code, pg, line_id), "chars": chars, "syllables": syls})
    return out


def load(path: Path | str = CACHE) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"chưa có cache {p} — chạy: python -m pipeline.lab.extract_columns")
    with open(p, "rb") as fh:
        return pickle.load(fh)


def add_anchors(columns: list[dict], qn_to_nom: dict, similar: dict) -> list[dict]:
    """Gắn ô neo = cặp ghép mà align MỐC cho `confirmed=True`.

    Neo phải tính TRÊN CHÍNH đầu vào thật, bằng cấu hình MỐC (chi phí hiện hành). Đó là
    định nghĩa "cặp gần chắc chắn đúng" mà mọi cấu hình khác bị đem ra so.
    """
    sys.path.insert(0, str(REPO))
    from pipeline.align_engine import anchor_align as aa
    for col in columns:
        ops = aa.realign_column([{"ocr_char": c} for c in col["chars"]],
                                col["syllables"], qn_to_nom, similar)
        col["anchor_pairs"] = [(o["nom_idx"], o["syl_idx"]) for o in ops
                               if o["op"] == "match" and o.get("confirmed")]
    return columns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.lab.extract_columns")
    ap.add_argument("--out", default=str(CACHE))
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(argv)

    if args.stats:
        cols = load(args.out)
        ne = sum(1 for c in cols if len(c["chars"]) != len(c["syllables"]))
        na = sum(len(c.get("anchor_pairs", ())) for c in cols)
        print(f"  cột: {len(cols):,}")
        print(f"  cột có m != n (chỗ chi phí thật sự quyết): {ne:,} ({100*ne/max(len(cols),1):.1f}%)")
        print(f"  tổng chữ Nôm: {sum(len(c['chars']) for c in cols):,} | "
              f"âm QN: {sum(len(c['syllables']) for c in cols):,}")
        print(f"  ô neo (align mốc, confirmed): {na:,}")
        return 0

    sys.path.insert(0, str(REPO))
    from core.text.dictionary import dict_dir, load_qn_to_nom, load_similarity_dict
    cols = extract()
    qn = load_qn_to_nom(str(dict_dir() / "QuocNgu_SinoNom.csv"))
    sim = load_similarity_dict(str(dict_dir() / "SinoNom_Similar.csv"))
    cols = add_anchors(cols, qn, sim)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(cols, fh)
    ne = sum(1 for c in cols if len(c["chars"]) != len(c["syllables"]))
    print(f"[extract] {len(cols):,} cột -> {args.out}")
    print(f"[extract] cột có m != n: {ne:,} ({100*ne/max(len(cols),1):.1f}%) "
          f"| ô neo: {sum(len(c['anchor_pairs']) for c in cols):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
