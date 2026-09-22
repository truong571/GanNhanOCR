#!/usr/bin/env python3
"""Extract all 152 full-color pages from KimVanKieuTanTruyen_1894.pdf (British Library Or. 14844).

Saves lossless extracted page images to:
  data/KimVanKieu1894/pages/page_0001.jpg ... page_0152.jpg
"""

from pathlib import Path
import fitz

def main():
    root = Path(__file__).resolve().parent.parent
    pdf_path = root / "data/KimVanKieu1894/KimVanKieuTanTruyen_1894.pdf"
    out_dir = root / "data/KimVanKieu1894/pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(str(pdf_path))
    print(f"Extracting {len(doc)} pages from {pdf_path.name}...")
    
    count = 0
    for i in range(len(doc)):
        page = doc[i]
        imgs = page.get_images()
        page_num = i + 1
        out_file = out_dir / f"page_{page_num:04d}.jpg"
        
        if imgs:
            xref = imgs[0][0]
            base_img = doc.extract_image(xref)
            img_bytes = base_img["image"]
            ext = base_img["ext"]
            if ext != "jpeg" and ext != "jpg":
                out_file = out_dir / f"page_{page_num:04d}.{ext}"
        else:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("jpeg")
            
        with open(out_file, "wb") as f:
            f.write(img_bytes)
        count += 1
        
        if count % 20 == 0 or count == len(doc):
            print(f"  Extracted {count}/{len(doc)} pages")
            
    print(f"[OK] Successfully extracted all {count} pages to {out_dir}")

if __name__ == "__main__":
    main()
