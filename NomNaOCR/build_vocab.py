"""Build the EXACT NomNaOCR recognition vocab from the dataset's transcripts (All.txt).

The CRNN's CTC output indices only mean something via this ordered vocab. It is
`Counter(all clean-label chars).most_common()` — the same order loader.DataImporter /
DataHandler.StringLookup use. We only need the TEXT file All.txt (small), not the images.

Get All.txt from the Kaggle dataset `quandang/nomnaocr` (see HUONG_DAN_SU_DUNG.md):
  Datasets/Patches/All.txt   (lines: "<img_name>\t<nom_text>")

Run:
  python build_vocab.py /path/to/All.txt            # -> writes vocab.txt
The script asserts the vocab size == 7479 (model VOCAB_SIZE 7481 = +[PAD] +[UNK]).
"""
import re
import sys
from collections import Counter
from pathlib import Path
from string import printable

# verbatim from ds4v_repo/Text recognition/loader.py :: DataImporter.is_clean_text
NOT_NOM = (r'\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ')
_CLEAN_PAT = re.compile(f'[{NOT_NOM}{re.escape(printable)}]')


def is_clean_nom(text: str) -> bool:
    return not bool(_CLEAN_PAT.search(text.lower()))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python build_vocab.py /path/to/All.txt [out=vocab.txt]")
    all_txt = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "vocab.txt"

    texts, dropped = [], 0
    for line in all_txt.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        _img, text = line.split("\t", 1)
        if len(text) >= 1 and is_clean_nom(text):
            texts.append(text)
        else:
            dropped += 1

    vocab = [ch for ch, _ in Counter("".join(texts)).most_common()]
    out.write_text("\n".join(vocab), encoding="utf-8")

    print(f"clean labels: {len(texts)} (dropped {dropped}) | unique chars: {len(vocab)}")
    print(f"-> {out}")
    if len(vocab) == 7479:
        print("MATCH: 7479 chars (+[PAD]+[UNK] = 7481 = model VOCAB_SIZE). vocab is EXACT. ✅")
    else:
        print(f"WARNING: got {len(vocab)}, expected 7479. Likely a different All.txt version, or "
              f"some patch images were filtered during training. Decoding of RARE chars may be off; "
              f"common chars usually still align. For an exact vocab, run the repo's DataImporter on "
              f"the full dataset (images+All.txt) — see the guide.")


if __name__ == "__main__":
    main()
