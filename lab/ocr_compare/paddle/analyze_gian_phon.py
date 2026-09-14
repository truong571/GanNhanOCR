"""Chữ đọc ra là giản thể hay phồn thể? Ngoài từ vựng Nôm bao nhiêu? (chạy bằng /tmp/venv_paddle, cần opencc)."""
import sys, collections, json
import pandas as pd
from opencc import OpenCC
sys.path.insert(0, "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle")
from score import cjk_chars
s2t, t2s = OpenCC("s2t"), OpenCC("t2s")
vocab = set(open("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle/nom_vocab.txt").read().split("\n"))


def kind(c):
    if s2t.convert(c) != c and t2s.convert(c) == c:
        return "giản thể (có dạng phồn khác)"
    if t2s.convert(c) != c and s2t.convert(c) == c:
        return "phồn thể (có dạng giản khác)"
    return "trung tính (giản = phồn)"


for p in sys.argv[1:]:
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    chars = [c for s in d.pred for c in cjk_chars(s)]
    k = collections.Counter(kind(c) for c in chars)
    n = len(chars)
    out = dict(file=p.split("/")[-1], n_cjk_chars=n, **{a: round(b / n, 4) for a, b in k.items()},
               ngoai_tu_vung_nom=round(sum(c not in vocab for c in chars) / n, 4),
               ngoai_tu_vung_vi_du="".join(sorted({c for c in chars if c not in vocab}))[:40],
               gian_the_vi_du="".join(sorted({c for c in chars if kind(c).startswith("giản")}))[:40])
    print(json.dumps(out, ensure_ascii=False))
