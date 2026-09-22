#!/usr/bin/env python3
"""Download the authentic bilingual Kim Vân Kiều tân truyện (Abel des Michels, Paris, 1884–1885).

Source: Bibliothèque nationale de France (BnF Gallica)
- Tome 1 (bpt6k5439461n): 322 canvases (Title, Intro, Quoc Ngu transliteration & French, verses 1-1500)
- Tome 2, Part 1 (bpt6k54394659): 309 canvases (Quoc Ngu transliteration & French, verses 1501-3254)
- Tome 2, Part 2 (bpt6k5453029r): 171 canvases (Complete 3,254 verses in Chữ Nôm lithograph handwriting)

Features:
- Downloads authentic scanned book pages as high-resolution JPG (1500px width)
- Multi-threaded download with retry and resume support
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

VOLUMES = {
    "nom": {
        "ark": "bpt6k5453029r",
        "desc": "Bản Nôm in thạch bản chữ bút lông (Tome 2-2, 171 canvas, 3.254 câu Kiều)",
        "start_canvas": 1,
        "end_canvas": 171,
        "out_dir": "data/KimVanKieu1884/pages",
    },
    "qn_vol1": {
        "ark": "bpt6k5439461n",
        "desc": "Tập 1 Quốc ngữ đối chiếu + Pháp văn (Tome 1, 322 canvas, câu 1-1500)",
        "start_canvas": 1,
        "end_canvas": 322,
        "out_dir": "data/KimVanKieu1884/quocngu_pages/vol1",
    },
    "qn_vol2": {
        "ark": "bpt6k54394659",
        "desc": "Tập 2 Quốc ngữ đối chiếu + Pháp văn (Tome 2-1, 309 canvas, câu 1501-3254)",
        "start_canvas": 1,
        "end_canvas": 309,
        "out_dir": "data/KimVanKieu1884/quocngu_pages/vol2",
    },
}

def download_canvas(ark: str, canvas_num: int, out_path: Path, max_retries: int = 4) -> bool:
    if out_path.exists() and out_path.stat().st_size > 15000:
        return True
    
    url = f"https://gallica.bnf.fr/iiif/ark:/12148/{ark}/f{canvas_num}/full/1500,/0/native.jpg"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) > 1000:
                    with open(out_path, "wb") as f:
                        f.write(data)
                    return True
        except Exception as e:
            time.sleep(1 + attempt * 1.5)
    return False

def main():
    parser = argparse.ArgumentParser(description="Download Kim Van Kieu 1884 bilingual edition")
    parser.add_argument("--volume", choices=["all", "nom", "qn_vol1", "qn_vol2"], default="all")
    parser.add_argument("--max_pages", type=int, default=None, help="Limit number of pages for testing")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parent.parent
    
    vols_to_dl = list(VOLUMES.keys()) if args.volume == "all" else [args.volume]
    
    total_downloaded = 0
    for vol_key in vols_to_dl:
        vcfg = VOLUMES[vol_key]
        ark = vcfg["ark"]
        out_dir = root / vcfg["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=======================================================")
        print(f"Đang tải: {vcfg['desc']}")
        print(f"Mã định danh Ark: {ark} -> {out_dir.relative_to(root)}")
        print(f"=======================================================")
        
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
                if completed % 25 == 0 or completed == len(tasks):
                    print(f"  Tiến độ: {completed}/{len(tasks)} trang đã tải xong")
                    
        total_downloaded += completed
        print(f"[OK] Hoàn thành {vcfg['desc']}: {completed} trang trong {out_dir}")
        
    print(f"\n[HOÀN THÀNH TẤT CẢ] Đã tải thành công trọn bộ {total_downloaded} trang song ngữ Kiều 1884!")

if __name__ == "__main__":
    main()
