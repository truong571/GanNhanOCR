"""Ghép nghĩa Hán (Unihan `kDefinition`) vào bảng khoảng trống từ điển — tầng 1 của gapcheck.

VÌ SAO NGHĨA HÁN LÀ CĂN CỨ ĐẦU TIÊN
-----------------------------------
Cặp (chữ, âm) mà từ điển ÂM không có phần lớn là **mượn nghĩa**. Nghĩa Hán của chữ là
bằng chứng rẻ nhất, kiểm lại được, và không cần mắt người:

  * 皇 *royal, imperial; ruler*  đọc "thánh"  -> hợp lý, mượn nghĩa thật
  * 而 *and; and then; but*      đọc "làm"    -> vô lý, phải soi lại

`kDefinition` phủ 287/316 cặp SYLLABLE nên dùng được ngay, không phải chờ nguồn nào khác.

BẢNG NÀY KHÔNG SINH NHÃN
------------------------
Cột `ai_flag` là **thứ tự ưu tiên chấm tay**, không phải phán quyết. Quy tắc tất định,
không gọi API, không mô hình:

  no_meaning     Unihan không có nghĩa cho chữ này -> chữ hiếm, người phải xem
  function_word  nghĩa là từ chức năng (particle / and / but / you / final...) -> chữ
                 ngữ pháp KHÔNG mang nghĩa để mượn -> nghi ngờ cao nhất
  lookalike      có ứng viên đúng âm mà gần giống tự dạng -> nghi OCR đọc nhầm chữ
  tone_only      chỉ lệch dấu thanh -> lỗi chuẩn hoá, sửa được
  check_meaning  có nghĩa Hán, không rơi vào nhóm nào -> đọc nghĩa rồi quyết

Đọc `no_meaning` cho đúng: Unihan không có nghĩa vì đó là **chữ Nôm riêng**, không phải
chữ Hán. Nhóm này thường là những cặp ĐÚNG NHẤT (𡖋 đọc "làm", 𦤳 đọc "thánh") — chỉ là
từ điển âm chưa ghi. Xem thêm cột `nom_rival`.

    .venv/bin/python -m pipeline.tools.gapcheck --unihan <Unihan_Readings.txt>

Thiếu `--unihan` thì lấy `Dict/unihan_kdefinition_full.csv` (bản đầy đủ), không có nữa
thì lấy bản cache nhỏ `Dict/_sources/unihan_kdefinition.csv`.
"""
from __future__ import annotations

import argparse
import collections
import unicodedata as ud
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
# Nguồn nghĩa Hán, xét theo thứ tự: cờ --unihan -> bản đầy đủ người dùng đặt trong Dict/
# -> bản cache nhỏ chỉ chứa các chữ của corpus này.
FULL = REPO / "Dict" / "unihan_kdefinition_full.csv"
CACHE = REPO / "Dict" / "_sources" / "unihan_kdefinition.csv"

# Từ khoá trong nghĩa Hán báo hiệu chữ CHỨC NĂNG (hư từ) — loại chữ không có nghĩa thực
# để mượn. 而 "and; but" và 爾 "you; final particle" đều rơi vào đây.
_FUNCTION = ("particle", "final particle", "interjection", "prefix", "suffix",
             "conjunction", "preposition", "pronoun", "Kangxi radical",
             "and; ", "but; ", "you; ", "that, those", "surname")


def load_kdefinition(unihan: Path | None) -> dict[str, str]:
    """Đọc `kDefinition`; ưu tiên file Unihan gốc, không có thì dùng bản cache trong repo."""
    if unihan and unihan.exists():
        out: dict[str, str] = {}
        for ln in unihan.read_text(encoding="utf-8").splitlines():
            if ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) == 3 and p[1] == "kDefinition":
                out[chr(int(p[0][2:], 16))] = p[2]
        return out
    for f in (FULL, CACHE):
        if f.exists():
            d = pd.read_csv(f, dtype=str).fillna("")
            # bản đầy đủ dùng cột 'Character', bản cache dùng 'char'
            col = "Character" if "Character" in d.columns else "char"
            print(f"[gapcheck] nguồn nghĩa: {f.relative_to(REPO)}")
            return dict(zip(d[col], d["kDefinition"]))
    raise SystemExit("không có Unihan_Readings.txt lẫn bản trong Dict/ — truyền --unihan")


def _base(s: str) -> str:
    """Bỏ DẤU THANH, giữ dấu tạo chữ (ă â ê ô ơ ư đ)."""
    keep = "̛̆̂"
    return ud.normalize("NFC", "".join(
        c for c in ud.normalize("NFD", str(s))
        if not ("̀" <= c <= "̣" and c not in keep)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.gapcheck")
    ap.add_argument("--unihan", default="", help="đường dẫn Unihan_Readings.txt")
    ap.add_argument("--gap", nargs="*", default=[
        str(REPO / "dataset_out" / "dict_gap_syllable.csv"),
        str(REPO / "dataset_out" / "dict_gap.csv")])
    args = ap.parse_args(argv)

    kdef = load_kdefinition(Path(args.unihan) if args.unihan else None)
    print(f"[gapcheck] kDefinition: {len(kdef):,} chữ")

    from core.text.dictionary import load_qn_to_nom
    qn = load_qn_to_nom(str(REPO / "Dict" / "QuocNgu_SinoNom.csv"))
    readings: dict[str, set[str]] = collections.defaultdict(set)
    for syl, chars in qn.items():
        for ch in chars:
            readings[ch].add(syl)

    used: dict[str, str] = {}
    for path in args.gap:
        p = Path(path)
        if not p.exists():
            print(f"[gapcheck] bỏ qua (không có): {p}")
            continue
        d = pd.read_csv(p, dtype=str).fillna("")
        d["han_meaning"] = [kdef.get(c, "") for c in d["ocr_char"]]

        flags = []
        for r in d.itertuples():
            m = r.han_meaning
            base_ok = _base(r.syllable) in {_base(x) for x in readings.get(r.ocr_char, ())}
            if not m:
                flags.append("no_meaning")
            elif any(k in m for k in _FUNCTION):
                flags.append("function_word")
            elif r.similar_hit:
                flags.append("lookalike")
            elif base_ok:
                flags.append("tone_only")
            else:
                flags.append("check_meaning")
        d["ai_flag"] = flags

        # ĐỐI THỦ NÔM — bằng chứng mạnh nhất, và nó nằm sẵn trong chính corpus:
        # nếu cùng một âm vừa được viết bằng một chữ NÔM RIÊNG (Unihan không định nghĩa
        # vì đó là chữ chỉ dùng trong văn Nôm) vừa bằng một chữ HÁN có nghĩa, thì nhánh
        # chữ Hán rất đáng ngờ là OCR đọc nhầm — chữ đúng đang nằm ngay cột này.
        # Ví dụ: âm "làm" có 𡖋 (80 ô) đứng cạnh 而 (453 ô) và 白 (113 ô).
        nom_of: dict[str, list[str]] = collections.defaultdict(list)
        for r in d.itertuples():
            if not r.han_meaning:
                nom_of[r.syllable].append(f"{r.ocr_char}({r.n_rows})")
        d["nom_rival"] = [" ".join(nom_of.get(s, ())) if m else ""
                          for s, m in zip(d["syllable"], d["han_meaning"])]
        for c, m in zip(d["ocr_char"], d["han_meaning"]):
            if m:
                used[c] = m

        move = ("han_meaning", "ai_flag", "nom_rival")
        cols = [c for c in d.columns if c not in move]
        i = cols.index("n_rows")
        d = d[cols[:i] + list(move) + cols[i:]]
        d.to_csv(p, index=False, encoding="utf-8-sig")

        n = d["n_rows"].astype(int)
        print(f"[gapcheck] {p.name}: {len(d):,} cặp / {n.sum():,} ô")
        for f, g in d.groupby("ai_flag"):
            print(f"           {f:14} {len(g):5,} cặp / {g['n_rows'].astype(int).sum():6,} ô")

    if used and not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"char": list(used), "kDefinition": list(used.values())}).to_csv(
            CACHE, index=False, encoding="utf-8-sig")
        print(f"[gapcheck] cache -> {CACHE.relative_to(REPO)} ({len(used):,} chữ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
