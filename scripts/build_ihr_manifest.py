#!/usr/bin/env python3
"""Gộp IHR-NomDB (data/handwritten) + nhãn NomNaOCR (LVT, Kiều 1872) thành MỘT manifest.

Nguồn gốc duy nhất là data/handwritten (ảnh trang + patch + annotation + bbox cột).
Các bản sao phái sinh (LucVanTien/*.txt, TruyenKieu/*.txt, *_song_ngu.tsv) KHÔNG được
dùng vì TSV lệch câu (LVT 1.084/2.000, Kiều 2.265/3.254 câu sai số âm tiết).

Chạy từ gốc repo:  python3 scripts/build_ihr_manifest.py
Ra:  data/ihr_nomdb_merged/manifest.tsv  (1 dòng = 1 câu thơ = 1 patch)
"""
import csv, json, os, re, sys, unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HW = REPO / "data" / "handwritten"
OUT = REPO / "data" / "ihr_nomdb_merged"
NNA_ALL = REPO / "NomNaOCR" / "All.txt"
if not NNA_ALL.exists():
    NNA_ALL = REPO / "data" / "NomNaOCR" / "All.txt"

BOOKS = {  # thư mục IHR -> (tên sách, ấn bản, tiền tố nguồn trong NomNaOCR/All.txt)
    "Luc-Van-Tien": ("LucVanTien", "nlvnpf-0059 (in mộc bản 1916)", "Luc Van Tien"),
    "tale-of-kieu": ("TruyenKieu", "Kiều 1872 (in mộc bản)", "Tale of Kieu 1872"),
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

def split_of():
    s = {}
    for name in ("train", "val"):
        for rel in json.load(open(HW / "patches" / f"{name}.json")):
            s[Path(rel).stem] = name   # 'nlvnpf-0059-101.jpg_8_2'
    return s

def column_boxes(src):
    """VoTT: trả về {tên_ảnh: [bbox cột sắp phải→trái]}; chỉ tin khi số cột khớp."""
    bb = json.load(open(HW / "pages" / src / "bboxes.json"))["assets"]
    out = {}
    for a in bb.values():
        regs = [r["boundingBox"] for r in a.get("regions", []) if "Column" in r.get("tags", [])]
        regs.sort(key=lambda b: -b["left"])
        out[a["asset"]["name"]] = [(round(b["left"], 1), round(b["top"], 1), round(b["width"], 1), round(b["height"], 1)) for b in regs]
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nna = load_nna(); splits = split_of()
    rows = []; stats = {}
    for src, (book, edition, nna_prefix) in BOOKS.items():
        boxes = column_boxes(src)
        pages = json.load(open(HW / "pages" / src / "annotation.json"))
        pdir = HW / "patches" / src / "patches"
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
                    "page_image": f"data/handwritten/pages/{src}/images/{img}",
                    "verse_idx_in_page": vi, "col_index": col, "part": part,
                    "patch_image": f"data/handwritten/patches/{src}/patches/{stem}.jpg" if stem else "",
                    "patch_preprocessed": next((f"data/handwritten/patches_preprocessed/{s}/{stem}.jpg" for s in ("train", "val") if stem and (HW / "patches_preprocessed" / s / f"{stem}.jpg").exists()), ""),
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
        stats[book] = dict(pages=len(pages), verses=n_verse, pages_bbox_ok=n_bbox_ok)
    with open(OUT / "manifest.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
    # tóm tắt
    def cnt(pred): return sum(1 for r in rows if pred(r))
    summ = {
        "rows": len(rows), **{f"{b}": s for b, s in stats.items()},
        "with_patch": cnt(lambda r: r["patch_image"]), "with_col_bbox": cnt(lambda r: r["col_bbox_xywh"]),
        "split_train": cnt(lambda r: r["split"] == "train"), "split_val": cnt(lambda r: r["split"] == "val"),
        "len_match": cnt(lambda r: r["len_match"] == 1),
        "has_nomnaocr": cnt(lambda r: r["nomnaocr_label"]), "nna_eq_ihr": cnt(lambda r: r["nna_eq_ihr"] == 1),
        "distinct_nom_chars": len({c for r in rows for c in r["nom_text"]}),
        "pua_chars": len({c for r in rows for c in r["nom_text"] if 0xE000 <= ord(c) <= 0xF8FF or 0xF0000 <= ord(c) <= 0x10FFFD}),
    }
    json.dump(summ, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))

if __name__ == "__main__":
    main()
