#!/usr/bin/env python3
"""Download Chrestomathie cochinchinoise (Abel des Michels, Paris 1872).

Source: Bibliothèque nationale de France (BnF Gallica)
Ark: bpt6k5812244p (173 canvases)
- Canvases 106 to 171: Chữ Nôm in thạch bản chữ bút lông (Chuyện đời xưa, truyện ngụ ngôn, ca dao)
- Canvases 27 to 105: Chữ Quốc ngữ cổ của Pétrus Trương Vĩnh Ký + dịch nghĩa Pháp văn
"""

import argparse
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ARK = "bpt6k5812244p"
TOTAL_CANVASES = 173

def download_canvas(canvas_num: int, out_path: Path, max_retries: int = 4) -> bool:
    if out_path.exists() and out_path.stat().st_size > 15000:
        return True
    
    url = f"https://gallica.bnf.fr/iiif/ark:/12148/{ARK}/f{canvas_num}/full/1500,/0/native.jpg"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) > 1000:
                    with open(out_path, "wb") as f:
                        f.write(data)
                    return True
        except Exception:
            time.sleep(1 + attempt * 1.5)
    return False

def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data/Chrestomathie1872/pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=======================================================")
    print(f"Đang tải Chrestomathie cochinchinoise (Abel des Michels 1872)")
    print(f"Mã Ark: {ARK} -> {out_dir.relative_to(root)} ({TOTAL_CANVASES} canvas)")
    print(f"=======================================================")
    
    tasks = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for c in range(1, TOTAL_CANVASES + 1):
            out_file = out_dir / f"canvas_{c:04d}.jpg"
            tasks.append(pool.submit(download_canvas, c, out_file))
            
        completed = 0
        for fut in as_completed(tasks):
            if fut.result():
                completed += 1
                if completed % 25 == 0 or completed == TOTAL_CANVASES:
                    print(f"  Tiến độ: {completed}/{TOTAL_CANVASES} trang đã tải xong")
                    
    print(f"[OK] Hoàn thành tải {completed}/{TOTAL_CANVASES} trang Chrestomathie 1872.")

if __name__ == "__main__":
    main()
