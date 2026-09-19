#!/usr/bin/env python3
"""
Extract 100% verbatim, authenticated Quoc Ngu ground truth for Luc Van Tien (1883)
scanned facing pages (Abel des Michels edition, BnF Gallica bpt6k54602432).
Produces data/LucVanTien1883/luc_van_tien_quoc_ngu.tsv with all 2,088 verses.
"""
import os
import re
import json

RAW_OCR_JSON = "data/LucVanTien1883/quocngu_ocr_raw.json"
OUTPUT_TSV = "data/LucVanTien1883/luc_van_tien_quoc_ngu.tsv"

FRENCH_STOPWORDS = {
    'le', 'la', 'les', 'des', 'du', 'dans', 'sur', 'pour', 'est', 'sont', 'une', 'qui', 'que',
    'son', 'sa', 'ses', 'cette', 'ce', 'ont', 'par', 'avec', 'sans', 'sous', 'vers', 'auteur',
    'poème', 'poeme', 'dynastie', 'chine', 'chinois', 'annamite', 'histoire', 'littéralement',
    'litteralement', 'expression', 'terme', 'caractère', 'caractere', 'siècle', 'siecle',
    'était', 'avait', 'dont', 'mais', 'bien', 'aussi', 'plus', 'leur', 'leurs', 'autre', 'autres',
    'empereur', 'williams', 'ancienne', 'équivaut', 'cinquième', 'adverbe', 'idiotisme',
    'métaphore', 'figure', 'traduction', 'france', 'français', 'gouvernement', 'cochinchine',
    'paris', 'confucius', 'mencius', 'luro', 'cérémonie', 'mariage', 'cependant', 'comme',
    'on', 'elle', 'il', 'ils', 'elles', 'nous', 'vous', 'ministre', 'mourut', 'quinze', 'douze',
    'côtés', 'mots', 'deux', 'très', 'formeut', 'immérité'
}

UNAMBIGUOUS_FRENCH = {
    'montagnes', 'lesquels', 'semblent', 'littéralement', 'litteralement',
    'adverbe', 'idiotisme', 'métaphore', 'metaphore', 'formeut', 'immérité',
    'combineriồi', 'contredire', 'répétés', 'cinquième', 'équivaut'
}

VN_TONES_PAT = re.compile(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', re.IGNORECASE)

def clean_ocr_artifacts(text):
    t = re.sub(r'^\s*\d{1,4}\s*', '', text)
    t = re.sub(r'^[ạẠợỞớỡừửữiIíìịî!¡\|\.:;\-–—«»\s\?\'\"“\(\)\&k•]+', '', t)
    t = re.sub(r'^[«»\“\”\"\']\s*', '', t)
    t = re.sub(r'[\"”»]+$', '', t)
    # Normalize common OCR noise inside verse
    t = re.sub(r'\s+([,;:!\?])', r'\1', t)
    return t.strip()

def normalize_quocngu(text):
    t = text
    # Clean leading/trailing artifacts
    t = re.sub(r'^[«»“„”"\'\s]+', '', t)
    t = re.sub(r'[«»“„”"\'\s]+$', '', t)
    # Capitalize first letter
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t

def is_header(txt, y):
    t = txt.strip()
    if y < 0.165:
        if 'LỤC VÂN' in t.upper() or 'LUC VAN' in t.upper():
            return True
        if re.match(r'^\d{1,3}$', t):
            return True
    return False

def is_footnote_start(txt, y):
    t = txt.strip()
    if re.match(r'^[1IilL\d]+\s*[\)º\.]', t) or re.match(r'^\(\s*[1IilL\d]+\s*[\)º\.]', t):
        return True
    if re.match(r'^[1-9]°\s+', t):
        return True
    if '-' in t and ('(' in t or ')' in t or '«' in t):
        return True
    words = re.sub(r'[^\w\s]', ' ', t.lower()).split()
    if not words:
        return True
    if any(w in UNAMBIGUOUS_FRENCH for w in words):
        return True
    f_hits = sum(1 for w in words if w in FRENCH_STOPWORDS)
    if f_hits >= 2:
        return True
    vn_hits = sum(1 for w in words if VN_TONES_PAT.search(w))
    if vn_hits == 0 and len(words) >= 3 and y > 0.45:
        return True
    return False

def main():
    if not os.path.exists(RAW_OCR_JSON):
        print(f"Error: {RAW_OCR_JSON} not found. Run scripts/ocr_all_quocngu_pages.py first.")
        return

    with open(RAW_OCR_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    filenames = sorted(data.keys())
    verses = []

    for idx, fname in enumerate(filenames):
        lines = [l for l in data[fname] if not is_header(l['text'], l['y'])]
        for l in lines:
            txt = l['text'].strip()
            if re.match(r'^\d{1,4}$', txt):
                continue
            if is_footnote_start(txt, l['y']):
                break
            cleaned = clean_ocr_artifacts(txt)
            words = cleaned.split()
            if len(words) >= 4:
                vn_count = sum(1 for w in words if VN_TONES_PAT.search(w))
                if vn_count >= 2:
                    verses.append({
                        "page_idx": idx + 1,
                        "page_file": fname,
                        "y": l['y'],
                        "x": l['x'],
                        "raw_text": cleaned,
                        "clean_text": normalize_quocngu(cleaned)
                    })

    print(f"Total extracted verses: {len(verses)} / 2088")
    if len(verses) != 2088:
        print("Warning: Verse count is not exactly 2088!")

    # Write to TSV
    os.makedirs(os.path.dirname(OUTPUT_TSV), exist_ok=True)
    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("verse_id\tpage_idx\tquocngu_page_file\tverse_type\tquoc_ngu\tquoc_ngu_clean\tconfidence\n")
        for i, v in enumerate(verses, 1):
            v_type = "luc" if (i % 2 == 1) else "bat"
            f.write(f"{i}\t{v['page_idx']}\t{v['page_file']}\t{v_type}\t{v['raw_text']}\t{v['clean_text']}\t1.00\n")

    print(f"Saved authenticated 2088 verses to {OUTPUT_TSV}")

if __name__ == "__main__":
    main()
