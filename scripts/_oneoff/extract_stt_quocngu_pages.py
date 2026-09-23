#!/usr/bin/env python3
"""Extract authentic scanned Quoc Ngu translation pages from SachThanhTruyen PDFs.

Saves original high-resolution scanned page images to:
  - data/SachThanhTruyen2/quocngu_pages/
  - data/SachThanhTruyen4/quocngu_pages/
  - data/SachThanhTruyen11/quocngu_pages/

Also generates an INDEX.tsv linking each Quoc Ngu page to its facing Nom page.
"""

import sys
from pathlib import Path
import fitz

BOOKS = [
    ("SachThanhTruyen2", "data/SachThanhTruyen2/STT2.pdf", "data/SachThanhTruyen2/quocngu_pages"),
    ("SachThanhTruyen4", "data/SachThanhTruyen4/STT4.pdf", "data/SachThanhTruyen4/quocngu_pages"),
    ("SachThanhTruyen11", "data/SachThanhTruyen11/STT11.pdf", "data/SachThanhTruyen11/quocngu_pages"),
]

def main():
    root = Path(__file__).resolve().parent.parent
    for book_name, pdf_rel, out_rel in BOOKS:
        pdf_path = root / pdf_rel
        out_dir = root / out_rel
        out_dir.mkdir(parents=True, exist_ok=True)
        
        doc = fitz.open(str(pdf_path))
        print(f"\nProcessing {book_name} ({len(doc)} pages)...")
        
        index_rows = [["pdf_page", "book_page_qn", "facing_nom_page", "qn_image_file"]]
        
        count = 0
        for i in range(len(doc)):
            # Odd index i (1, 3, 5...) is Quoc Ngu page
            if i % 2 == 1:
                qn_page = doc[i]
                nom_page = doc[i - 1]
                
                # Book page numbers
                # By convention in STT: even pages are Nom, odd pages are QN
                # Or extract from text
                imgs = qn_page.get_images()
                if imgs:
                    xref = imgs[0][0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    ext = base_image["ext"]
                else:
                    pix = qn_page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    ext = "png"
                
                qn_filename = f"qn_page_{i+1:04d}.{ext}"
                out_path = out_dir / qn_filename
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                
                index_rows.append([
                    str(i + 1),
                    f"p{i+1}",
                    f"page_{i:04d}",
                    qn_filename,
                ])
                count += 1
                
        # Write INDEX.tsv
        tsv_path = out_dir / "INDEX.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            for row in index_rows:
                f.write("\t".join(row) + "\n")
                
        print(f"  [OK] Extracted {count} scanned Quoc Ngu pages to {out_rel}/")
        print(f"  [OK] Created index at {tsv_path}")

if __name__ == "__main__":
    main()
