#!/usr/bin/env python3
"""
Run Swift Vision OCR on all 139 Quoc Ngu pages of LucVanTien1883.
Saves raw OCR line results as JSON for reproducible downstream processing.
"""
import os
import sys
import glob
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor

OCR_BIN = "scripts/ocr_quocngu"
PAGES_DIR = "data/LucVanTien1883/quocngu_pages"
CACHE_FILE = "data/LucVanTien1883/quocngu_ocr_raw.json"

def ocr_page(img_path):
    filename = os.path.basename(img_path)
    res = subprocess.run([OCR_BIN, img_path], capture_output=True, text=True)
    lines = []
    for l in res.stdout.splitlines():
        parts = l.split('\t', 2)
        if len(parts) == 3:
            lines.append({
                "y": float(parts[0]),
                "x": float(parts[1]),
                "text": parts[2].strip()
            })
    return filename, lines

def main():
    if not os.path.exists(OCR_BIN):
        print(f"Error: {OCR_BIN} not found. Please compile it first.")
        sys.exit(1)
        
    img_paths = sorted(glob.glob(os.path.join(PAGES_DIR, "*.jpg")))
    print(f"Found {len(img_paths)} pages to process.")
    
    results = {}
    # Use ProcessPoolExecutor with 4 workers for speed
    with ProcessPoolExecutor(max_workers=4) as executor:
        for fname, lines in executor.map(ocr_page, img_paths):
            results[fname] = lines
            print(f"Processed {fname}: {len(lines)} lines")
            
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Saved OCR results for {len(results)} pages to {CACHE_FILE}")

if __name__ == "__main__":
    main()
