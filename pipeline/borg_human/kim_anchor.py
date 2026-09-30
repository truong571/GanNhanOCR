"""kim_anchor.py — MỐC VỊ TRÍ từ kim (kinhhannom) cho bước gióng chữ người (TN10, 30/09/2026). 0 API: chỉ ĐỌC cache kim.

Chữ kim đọc ra tại một hộp (hộp dòng của kim chia đều theo số chữ) là một bằng chứng vị trí ĐỘC LẬP với encoder. Với mỗi trang:
cặp (đơn vị i, chữ người j) là MỐC khi có một chữ kim có tâm nằm trong hộp đơn vị i (nới `pad` mỗi phía) và trùng chữ người j
theo V1+ (var_eq_plus: trùng hoặc dị thể Unihan/OpenCC). Bước gióng cộng `beta` vào điểm phát xạ Z[i, j] của các cặp mốc
(align.py) — mốc kéo cả đoạn chung quanh về đúng chỗ, nên hậu nghiệm các ô lân cận cũng tăng.

Nguồn kim (params.KIM["src"]):
  lt2      prepared/_auto/<Sách>/kim_raw/<trang>_lt2.json (lượt đọc Nôm, đủ 641 trang)
  lt1x2    prepared/_auto/<Sách>/kim_raw_lt1x2/<trang>.json (lượt đọc Hán trên ảnh phóng ×2; hộp chia lại /2) — khi có cache
Nhiều nguồn: "lt2+lt1x2" (hợp các mốc).
Đơn vị lấy từ WORK/units/units_v1.pkl (cùng thứ tự với align.load_state: đơn vị theo trang, giữ thứ tự trong tệp).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from .geom import human_seq
from .params import REPO

SRC_DIRS = {"lt2": ("kim_raw", "{pn}_lt2.json", 1.0), "lt1x2": ("kim_raw_lt1x2", "{pn}.json", None)}


def kim_chars(book: str, pn: str, src: str):
    """[(chữ, cx, cy)] toạ độ trang gốc từ một nguồn kim; None nếu thiếu cache."""
    from pipeline.tools.ingest_lithograph_book import expand_box_chars
    d_name, pat, scale = SRC_DIRS[src]
    f = REPO / "prepared" / "_auto" / book / d_name / pat.format(pn=pn)
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    sc = scale or float(d.get("scale", 1) or 1)
    out = []
    for b in d.get("boxes") or []:
        for c in expand_box_chars(b):
            x0, y0, x1, y1 = c["bbox"]
            out.append((c["char"], (x0 + x1) / 2 / sc, (y0 + y1) / 2 / sc))
    return out


def build_pairs(work: Path, src: str = "lt2", pad: float = 0.25, log=print) -> Path:
    """-> WORK/kim/pairs_<src>_p<pad>.pkl = {"book/page": [(i_trang, j), ...]} (i = chỉ số đơn vị TRONG trang, như align)."""
    from pipeline.gold_exact.common import Assets, set_lexicon, var_eq_plus
    set_lexicon(Assets())
    out = Path(work) / "kim" / f"pairs_{src.replace('+', '_')}_p{pad:g}.pkl"
    if out.exists():
        return out
    U = pickle.load(open(Path(work) / "units" / "units_v1.pkl", "rb"))["units"]
    by_page: dict = {}
    for u in U:
        by_page.setdefault(f"{u['book']}/{u['page']}", []).append(u)
    memo: dict = {}
    pairs: dict = {}
    n_pages = n_pairs = n_miss = 0
    for key, uu in sorted(by_page.items()):
        book, pn = key.split("/")
        K = []
        for s in src.split("+"):
            k = kim_chars(book, pn, s)
            if k is None:
                n_miss += 1
            else:
                K += k
        if not K:
            continue
        chars = human_seq(book, pn)[0]
        pos = {}
        for j, c in enumerate(chars):
            pos.setdefault(c, []).append(j)
        pp = []
        for i, u in enumerate(uu):
            w, h = u["x2"] - u["x1"], u["y2"] - u["y1"]
            x0, x1 = u["x1"] - pad * w, u["x2"] + pad * w
            y0, y1 = u["y1"] - pad * h, u["y2"] + pad * h
            hit = set()
            for ch, cx, cy in K:
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    for c, js in pos.items():
                        v = memo.get((ch, c))
                        if v is None:
                            v = memo[(ch, c)] = bool(var_eq_plus(ch, c))
                        if v:
                            hit.update(js)
            pp += [(i, j) for j in sorted(hit)]
        pairs[key] = pp
        n_pages += 1
        n_pairs += len(pp)
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(pairs, open(out, "wb"))
    log(f"kim mốc [{src}, pad {pad}]: {n_pages} trang, {n_pairs} cặp (đơn vị, chữ người); thiếu cache {n_miss} lượt trang")
    return out
