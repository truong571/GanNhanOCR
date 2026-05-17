"""Apply the LLM-confirmed bigram fixes back to transcription files.

Reads fix_map.json from llm_fixer, scans prepared/<book>/transcriptions/*.json,
and applies each fix in-place. Backs up originals to *.json.pre_fix.

Usage:
    .venv/bin/python evaluation/test_llm_postfix/apply_fixes.py --dry-run
    .venv/bin/python evaluation/test_llm_postfix/apply_fixes.py            # actually apply
    .venv/bin/python evaluation/test_llm_postfix/apply_fixes.py --restore  # revert
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
FIX_MAP_PATH = OUT / "fix_map.json"


def load_fix_map() -> dict[tuple[str, str], str]:
    """Parse fix_map.json -> {(prev, wrong): correct_word}.

    Skips fixes that would change syllable count (e.g. "bốn đao" → "bốn bổn đạo"
    adds a word). Such fixes break Step-2 alignment.
    """
    if not FIX_MAP_PATH.exists():
        return {}
    raw = json.loads(FIX_MAP_PATH.read_text())
    out: dict[tuple[str, str], str] = {}
    skipped: list[tuple[str, str]] = []
    for wrong_bg, correct_bg in raw.items():
        wrong_parts = wrong_bg.split(" ")
        correct_parts = correct_bg.split(" ")
        if len(wrong_parts) != len(correct_parts):
            skipped.append((wrong_bg, correct_bg))
            continue
        if len(wrong_parts) != 2:
            skipped.append((wrong_bg, correct_bg))
            continue
        wp, ww = wrong_parts
        cp, cw = correct_parts
        if wp != cp:
            continue  # prev must match the suspect detection
        out[(wp, ww)] = cw
    if skipped:
        print(f"  [info] skipped {len(skipped)} fixes that change syllable count:")
        for w, c in skipped:
            print(f"          {w!r}  →  {c!r}")
    return out


def apply_to_syllables(syls: list[str],
                       fix_map: dict[tuple[str, str], str]) -> tuple[list[str], int]:
    """Return (new_syllables, n_fixes_applied)."""
    out = list(syls)
    fixes = 0
    for i in range(1, len(out)):
        key = (out[i-1], out[i])
        if key in fix_map:
            out[i] = fix_map[key]
            fixes += 1
    return out, fixes


def process_book(book_dir: Path, fix_map: dict[tuple[str, str], str],
                 dry_run: bool, restore: bool) -> dict:
    trans_dir = book_dir / "transcriptions"
    if not trans_dir.exists():
        return {"pages": 0, "fixes": 0}

    stats = {"pages": 0, "fixes": 0, "books_changed": 0}
    for jf in sorted(trans_dir.glob("page_*.json")):
        backup = jf.with_suffix(".json.pre_fix")
        txt_file = jf.with_suffix(".txt")
        txt_backup = txt_file.with_suffix(".txt.pre_fix")

        if restore:
            if backup.exists():
                shutil.move(str(backup), str(jf))
            if txt_backup.exists():
                shutil.move(str(txt_backup), str(txt_file))
            continue

        data = json.loads(jf.read_text())
        page_fixes = 0
        new_cols = []
        for col in data.get("columns", []):
            new_syls, n = apply_to_syllables(col.get("syllables", []), fix_map)
            page_fixes += n
            col["syllables"] = new_syls
            col["num_syllables"] = len(new_syls)
            new_cols.append(col)
        data["columns"] = new_cols

        if page_fixes > 0:
            stats["pages"] += 1
            stats["fixes"] += page_fixes
            if not dry_run:
                if not backup.exists():
                    shutil.copy2(str(jf), str(backup))
                if txt_file.exists() and not txt_backup.exists():
                    shutil.copy2(str(txt_file), str(txt_backup))
                jf.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                              encoding="utf-8")
                # Also rewrite the .txt mirror
                with open(txt_file, "w", encoding="utf-8") as f:
                    for col in new_cols:
                        f.write(" ".join(col["syllables"]) + "\n")

    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-dir", default=str(REPO / "prepared"))
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would change, don't write")
    p.add_argument("--restore", action="store_true",
                   help="Revert to .pre_fix backups")
    args = p.parse_args()

    if args.restore:
        print("Restoring from .pre_fix backups...")
    else:
        fix_map = load_fix_map()
        if not fix_map:
            print(f"!! {FIX_MAP_PATH} empty or missing. Run llm_fixer.py first.")
            return
        print(f"Loaded {len(fix_map)} confirmed fixes:")
        for (prev, wrong), correct in list(fix_map.items())[:10]:
            print(f"  {prev} {wrong}  →  {prev} {correct}")
        if len(fix_map) > 10:
            print(f"  ... and {len(fix_map)-10} more")
        print()

    total_pages, total_fixes = 0, 0
    for book_dir in sorted(Path(args.prepared_dir).iterdir()):
        if not book_dir.is_dir() or book_dir.name.startswith("_"):
            continue
        s = process_book(book_dir,
                         load_fix_map() if not args.restore else {},
                         args.dry_run, args.restore)
        if args.restore or s["pages"] > 0:
            print(f"  {book_dir.name:25s}  pages: {s['pages']:>4}  fixes: {s['fixes']:>5}")
            total_pages += s["pages"]
            total_fixes += s["fixes"]

    print()
    if args.restore:
        print("Restore complete.")
    elif args.dry_run:
        print(f"DRY RUN: would change {total_pages} pages, apply {total_fixes} fixes")
        print("Run without --dry-run to actually write.")
    else:
        print(f"Applied {total_fixes} fixes across {total_pages} pages.")
        print(f"Originals backed up as *.json.pre_fix / *.txt.pre_fix")
        print(f"To revert: {Path(__file__).name} --restore")


if __name__ == "__main__":
    main()
