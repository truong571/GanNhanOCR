#!/usr/bin/env python3
"""Download the authentic bilingual Kim Vân Kiều tân truyện (Abel des Michels, Paris, 1884–1885).

Source: Bibliothèque nationale de France (BnF Gallica)
- Tome 1 (bpt6k5439461n): 322 canvases (Quoc Ngu transliteration & French, verses 1-1500)
- Tome 2, Part 1 (bpt6k54394659): 309 canvases (Quoc Ngu transliteration & French, verses 1501-3254)
- Tome 2, Part 2 (bpt6k5453029r): 171 canvases (Complete 3,254 verses in Chữ Nôm lithograph handwriting)

Features:
- Downloads authentic scanned book pages as PNG/JPG
- Preserves 1-to-1 correspondence between Nôm lithograph and Quoc Ngu transliteration
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

VOLUMES = {
    "qn_vol1": {
        "ark": "bpt6k5439461n",
        "desc": "Tome 1: Quoc Ngu + French (Verses 1-1500)",
        "start_canvas": 27,
        "end_canvas": 322,
        "out_dir": "data/KimVanKieu1884/quocngu_pages/vol1",
    },
    "qn_vol2": {
        "ark": "bpt6k54394659",
        "desc": "Tome 2-1: Quoc Ngu + French (Verses 1501-3254)",
        "start_canvas": 1,
        "end_canvas": 309,
        "out_dir": "data/KimVanKieu1884/quocngu_pages/vol2",
    },
    "nom": {
        "ark": "bpt6k5453029r",
        "desc": "Tome 2-2: Nom Lithograph (165 pages, 3,254 verses)",
        "start_canvas": 3,
        "end_canvas": 167,
        "out_dir": "data/KimVanKieu1884/pages",
    },
}

def download_canvas(ark: str, canvas_num: int, out_path: Path, max_retries: int = 3) -> bool:
    if out_path.exists() and out_path.stat().st_size > 10000:
        return True
    
    url = f"https://gallica.bnf.fr/iiif/ark:/12148/{ark}/f{canvas_num}/full/1500,/0/native.jpg"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) > 1000:
                    with open(out_path, "wb") as f:
                        f.write(data)
                    return True
        except Exception:
            time.sleep(1 + attempt)
    return False

def main():
    parser = argparse.ArgumentParser(description="Download Kim Van Kieu 1884 bilingual edition")
    parser.add_argument("--volume", choices=["all", "qn_vol1", "qn_vol2", "nom"], default="all")
    parser.add_argument("--max_pages", type=int, default=None, help="Limit number of pages for testing")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parent.parent
    
    vols_to_dl = VOLUMES.keys() if args.volume == "all" else [args.volume]
    
    for vol_key in vols_to_dl:
        vcfg = VOLUMES[vol_key]
        ark = vcfg["ark"]
        out_dir = root / vcfg["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading {vcfg['desc']} ({ark})...")
        
        canvases = list(range(vcfg["start_canvas"], vcfg["end_canvas"] + 1))
        if args.max_pages:
            canvases = canvases[:args.max_pages]
            
        tasks = []
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            for c in canvases:
                filename = f"canvas_{c:04d}.jpg"
                out_path = out_dir / filename
                tasks.append(pool.submit(download_canvas, ark, c, out_path))
                
            completed = 0
            for fut in as_completed(tasks):
                if fut.result():
                    completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    print(f"  Progress: {completed}/{len(tasks)} pages downloaded")
                    
        print(f"[OK] Completed {vcfg['desc']}: {completed} pages in {out_dir}")

if __name__ == "__main__":
    main()
