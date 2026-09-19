#!/usr/bin/env python3
"""Dựng manifest cho 2 sách IHR-NomDB đã tách theo cuốn: data/LucVanTien1916, data/TruyenKieu1872.

Mỗi thư mục sách giữ nguyên cấu trúc IHR-NomDB: pages/{images,annotation.json,bboxes.json},
patches/, patches_preprocessed/{train,val}, train.json, val.json. Nhãn NomNaOCR (NomNaOCR/All.txt)
chỉ dùng để đối chiếu (cột nomnaocr_label) — đã chứng minh cùng nguồn với IHR.

Chạy từ gốc repo:  python3 scripts/build_ihr_manifest.py
Ra:  data/<Sách>/manifest.tsv + summary.json  (1 dòng = 1 câu thơ = 1 patch)
"""
import csv, json, os, re, sys, unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
NNA_ALL = REPO / "NomNaOCR" / "All.txt"
if not NNA_ALL.exists():
    NNA_ALL = REPO / "data" / "NomNaOCR" / "All.txt"

BOOKS = {  # thư mục sách trong data/ -> (ấn bản, tiền tố nguồn trong NomNaOCR/All.txt)
    "LucVanTien1916": ("nlvnpf-0059 (in mộc bản 1916)", "Luc Van Tien"),
    "TruyenKieu1872": ("Kiều 1872 (in mộc bản)", "Tale of Kieu 1872"),
}

def nfc(s): return unicodedata.normalize("NFC", s)

def load_nna():
    m = {}
    if not NNA_ALL.exists(): return m
    for line in NNA_ALL.read_text(encoding="utf-8").splitlines():
        if "\t" not in line: continue
        p, t = line.split("\t", 1)
        m[p] = nfc(t.strip())
    return m

def split_of(B):
    s = {}
    for name in ("train", "val"):
        for rel in json.load(open(B / f"{name}.json")):
            s[Path(rel).stem] = name   # 'nlvnpf-0059-101.jpg_8_2'
    return s

def column_boxes(B):
    """VoTT: trả về {tên_ảnh: [bbox cột sắp phải→trái]}; chỉ tin khi số cột khớp."""
    bb = json.load(open(B / "pages" / "bboxes.json"))["assets"]
    out = {}
    for a in bb.values():
        regs = [r["boundingBox"] for r in a.get("regions", []) if "Column" in r.get("tags", [])]
        regs.sort(key=lambda b: -b["left"])
        out[a["asset"]["name"]] = [(round(b["left"], 1), round(b["top"], 1), round(b["width"], 1), round(b["height"], 1)) for b in regs]
    return out

def build_book(book, edition, nna_prefix, nna):
    B = DATA / book
    splits = split_of(B); rows = []
    if True:
        boxes = column_boxes(B)
        pages = json.load(open(B / "pages" / "annotation.json"))
        pdir = B / "patches"
        n_bbox_ok = 0; n_verse = 0
        for pg in pages:
            img = os.path.basename(pg["img"])
            verses = pg["annotations"]
            # patch của trang, gom theo (col, part)
            pats = {}
            for j in pdir.glob(f"{img}_*.json"):
                m = re.match(rf"{re.escape(img)}_(\d+)(?:_(\d+))?$", j.stem)
                if m: pats[(int(m.group(1)), int(m.group(2) or 1))] = j
            keys = sorted(pats)
            cols = boxes.get(img, [])
            ncol = max((c for c, _ in keys), default=-1) + 1
            bbox_ok = len(cols) == ncol and ncol > 0
            n_bbox_ok += bbox_ok
            for vi, v in enumerate(verses):
                n_verse += 1
                nom = nfc("".join(v["hn_text"])); qn = nfc(" ".join(v["translation"])).strip()
                key = keys[vi] if vi < len(keys) else None
                pj = pats.get(key) if key else None
                pj_nom = nfc("".join(json.load(open(pj))[0]["hn_text"])) if pj else ""
                if pj and pj_nom != nom:  # patch không khớp thứ tự -> không gán patch
                    pj = None
                stem = pj.stem if pj else ""
                col, part = key if key else ("", "")
                cb = cols[col] if (bbox_ok and key) else None
                nna_key = f"{nna_prefix}/{img[:-4]}_{vi}.jpg"
                nna_txt = nna.get(nna_key, "")
                qn_syl = [w for w in re.split(r"\s+", qn) if w]
                rows.append({
                    "book": book, "edition": edition, "page_id": img[:-4],
                    "page_image": f"data/{book}/pages/images/{img}",
                    "verse_idx_in_page": vi, "col_index": col, "part": part,
                    "patch_image": f"data/{book}/patches/{stem}.jpg" if stem else "",
                    "patch_preprocessed": next((f"data/{book}/patches_preprocessed/{s}/{stem}.jpg" for s in ("train", "val") if stem and (B / "patches_preprocessed" / s / f"{stem}.jpg").exists()), ""),
                    "split": splits.get(stem, ""),
                    "col_bbox_xywh": ",".join(map(str, cb)) if cb else "",
                    "nom_text": nom,
                    "nom_unicode": " ".join(f"U+{ord(c):04X}" for c in nom),
                    "n_nom": len(nom),
                    "qn_verse": qn, "n_qn_syll": len(qn_syl),
                    "len_match": int(len(nom) == len(qn_syl)),
                    "nomnaocr_label": nna_txt,
                    "nna_eq_ihr": int(nna_txt == nom) if nna_txt else "",
                })
    with open(B / "manifest.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
    def cnt(pred): return sum(1 for r in rows if pred(r))
    summ = {
        "book": book, "rows": len(rows), "pages": len(pages), "pages_bbox_ok": n_bbox_ok,
        "with_patch": cnt(lambda r: r["patch_image"]), "with_col_bbox": cnt(lambda r: r["col_bbox_xywh"]),
        "split_train": cnt(lambda r: r["split"] == "train"), "split_val": cnt(lambda r: r["split"] == "val"),
        "len_match": cnt(lambda r: r["len_match"] == 1),
        "has_nomnaocr": cnt(lambda r: r["nomnaocr_label"]), "nna_eq_ihr": cnt(lambda r: r["nna_eq_ihr"] == 1),
        "distinct_nom_chars": len({c for r in rows for c in r["nom_text"]}),
        "pua_chars": len({c for r in rows for c in r["nom_text"] if 0xE000 <= ord(c) <= 0xF8FF or 0xF0000 <= ord(c) <= 0x10FFFD}),
    }
    json.dump(summ, open(B / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))

def main():
    nna = load_nna()
    for book, (edition, nna_prefix) in BOOKS.items():
        build_book(book, edition, nna_prefix, nna)

if __name__ == "__main__":
    main()
