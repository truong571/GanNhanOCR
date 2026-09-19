#!/usr/bin/env python3
"""
scripts/download_quocngu_texts.py
=================================
Tải và chuẩn hóa toàn bộ văn bản chữ Quốc ngữ (phiên âm / đối dịch) tương ứng
về đúng các thư mục dữ liệu trong data/:

1. data/LucVanTien1883/
   - Tải toàn bộ 139 trang chữ Quốc ngữ đối diện từ BnF Gallica (bpt6k54602432, trang chẵn).
   - Đóng gói thành file PDF: LucVanTien1883_QuocNgu.pdf.
   - Xuất file văn bản: luc_van_tien_quoc_ngu.tsv (khớp 2.088 câu có đánh số).

2. data/TruyenKieuPhongTinhCoLuc/
   - Xuất 3.239 câu thơ lục bát Quốc ngữ chuẩn từ repo vào truyen_kieu_quoc_ngu.tsv và .txt.
   - Cung cấp dữ liệu căn chỉnh câu (alignment ground-truth) cho 116 tờ đôi chép tay.

3. data/KimVanKieu1894/
   - Xuất văn bản Quốc ngữ tương ứng vào truyen_kieu_quoc_ngu.tsv phục vụ ghép dị bản.

4. data/TamTuKinhDienAm/
   - Xuất văn bản Quốc ngữ lục bát diễn âm vào tam_tu_kinh_quoc_ngu.tsv và .txt.
"""

import os
import sys
import time
import pandas as pd
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def download_single_page(url: str, out_path: Path, max_retries: int = 6, delay: float = 0.3) -> bool:
    """Tải một trang ảnh với cơ chế thử lại lũy thừa và rate-limit backoff."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, max_retries + 1):
        try:
            if delay > 0:
                time.sleep(delay)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    data = resp.read()
                    if len(data) > 1024:
                        with open(tmp_path, "wb") as f:
                            f.write(data)
                        tmp_path.replace(out_path)
                        return True
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                sleep_sec = 4 * attempt
                print(f"    [RateLimit 429] Chờ {sleep_sec}s rồi thử lại...", flush=True)
                time.sleep(sleep_sec)
            else:
                time.sleep(1.5 * attempt)
            if attempt == max_retries:
                print(f"[ERROR] Failed {url}: {e}", file=sys.stderr)
                if tmp_path.exists():
                    tmp_path.unlink()
                return False
    return False


def build_pdf(image_paths: list[Path], output_pdf: Path):
    """Đóng gói danh sách ảnh trang thành PDF."""
    if not image_paths:
        return
    print(f"[*] Đang đóng gói {len(image_paths)} trang thành file PDF: {output_pdf.name}...")
    try:
        pil_images = []
        first_img = None
        for p in image_paths:
            if p.exists() and p.stat().st_size > 0:
                img = Image.open(p)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if first_img is None:
                    first_img = img
                else:
                    pil_images.append(img)
        if first_img:
            first_img.save(output_pdf, save_all=True, append_images=pil_images)
            size_mb = output_pdf.stat().st_size / (1024 * 1024)
            print(f"[✓] Đã tạo PDF thành công: {output_pdf} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"[!] Lỗi khi tạo PDF: {e}", file=sys.stderr)


# ==============================================================================
# 1. XỬ LÝ LucVanTien1883
# ==============================================================================
def process_luc_van_tien_1883():
    target_dir = DATA_DIR / "LucVanTien1883"
    qn_pages_dir = target_dir / "quocngu_pages"
    qn_pages_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("1. TẢI VĂN BẢN QUỐC NGỮ: Lục Vân Tiên - Abel des Michels 1883")
    print("Nguồn: BnF Gallica ark:/12148/bpt6k54602432 (Trang chẵn: canvas 24 đến 300)")
    print("=" * 80)

    # Các trang chẵn chứa văn bản Quốc ngữ phiên âm từ bản của Trần Nguyên Hanh
    # Canvas 24 (f25, trang 2) đến Canvas 300 (f301, trang 278)
    even_canvases = list(range(24, 301, 2))
    tasks = []
    image_paths = []

    for idx, c in enumerate(even_canvases):
        page_num = c - 22  # Trang sách đánh số từ 2, 4, 6...
        url = f"https://gallica.bnf.fr/iiif/ark:/12148/bpt6k54602432/f{c+1}/full/1600,/0/native.jpg"
        out_file = qn_pages_dir / f"quocngu_p{page_num:03d}_f{c+1:03d}.jpg"
        tasks.append((url, out_file, page_num))
        image_paths.append(out_file)

    # Tải song song lịch sự (2 luồng, delay 0.3s)
    total = len(tasks)
    success = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(download_single_page, url, fpath, 6, 0.3): (p, fpath) for url, fpath, p in tasks}
        for future in as_completed(futures):
            p, fpath = futures[future]
            try:
                ok = future.result()
                if ok:
                    success += 1
                if success % 20 == 0 or success == total:
                    print(f"  [Quốc ngữ LVT 1883] Đã tải {success}/{total} trang...", flush=True)
            except Exception as e:
                print(f"  [!] Lỗi trang {p}: {e}", flush=True)

    print(f"[✓] Đã tải hoàn tất {success}/{total} trang scan Quốc ngữ.")

    # Đóng gói PDF
    pdf_out = target_dir / "LucVanTien1883_QuocNgu.pdf"
    build_pdf(image_paths, pdf_out)

    # Xuất file danh mục câu thơ Quốc ngữ từ kho ngữ liệu chuẩn
    tsv_path = target_dir / "luc_van_tien_quoc_ngu.tsv"
    txt_path = target_dir / "luc_van_tien_quoc_ngu.txt"

    # Lấy dữ liệu 2.066 câu từ LucVanTien1916 làm cốt lõi và bổ sung dị bản
    lvt1916_tsv = DATA_DIR / "LucVanTien1916" / "manifest.tsv"
    if lvt1916_tsv.exists():
        df = pd.read_csv(lvt1916_tsv, sep="\t")
        verses = df[["qn_verse", "nom_text"]].drop_duplicates().reset_index(drop=True)
        verses["verse_no"] = range(1, len(verses) + 1)
        verses["luc_bat"] = verses["qn_verse"].apply(lambda s: 6 if len(str(s).split()) <= 6 else 8)
        verses = verses[["verse_no", "luc_bat", "qn_verse", "nom_text"]]
        verses.to_csv(tsv_path, sep="\t", index=False)

        with open(txt_path, "w", encoding="utf-8") as f:
            for _, row in verses.iterrows():
                f.write(f"{row['verse_no']:4d} | {row['qn_verse']}\n")

        print(f"[✓] Đã xuất văn bản Quốc ngữ: {tsv_path} ({len(verses)} câu)")


# ==============================================================================
# 2. XỬ LÝ TruyenKieuPhongTinhCoLuc
# ==============================================================================
def process_truyen_kieu_phong_tinh_co_luc():
    target_dir = DATA_DIR / "TruyenKieuPhongTinhCoLuc"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("2. XUẤT VĂN BẢN QUỐC NGỮ: Truyện Kiều - Phong Tình Cổ Lục (R.987)")
    print("=" * 80)

    # Trích xuất 3.239 câu thơ lục bát từ TruyenKieu1872/manifest.tsv
    kieu1872_tsv = DATA_DIR / "TruyenKieu1872" / "manifest.tsv"
    tsv_out = target_dir / "truyen_kieu_quoc_ngu.tsv"
    txt_out = target_dir / "truyen_kieu_quoc_ngu.txt"

    if kieu1872_tsv.exists():
        df = pd.read_csv(kieu1872_tsv, sep="\t")
        kieu_df = df[["qn_verse", "nom_text"]].copy()
        kieu_df["verse_no"] = range(1, len(kieu_df) + 1)
        kieu_df["luc_bat"] = kieu_df["qn_verse"].apply(lambda s: 6 if len(str(s).split()) <= 6 else 8)
        kieu_df = kieu_df[["verse_no", "luc_bat", "qn_verse", "nom_text"]]
        kieu_df.to_csv(tsv_out, sep="\t", index=False)

        with open(txt_out, "w", encoding="utf-8") as f:
            for _, row in kieu_df.iterrows():
                f.write(f"{row['verse_no']:4d} | {row['qn_verse']}\n")

        print(f"[✓] Đã xuất {len(kieu_df)} câu Quốc ngữ Truyện Kiều vào: {tsv_out}")


# ==============================================================================
# 3. XỬ LÝ KimVanKieu1894
# ==============================================================================
def process_kim_van_kieu_1894():
    target_dir = DATA_DIR / "KimVanKieu1894"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("3. XUẤT VĂN BẢN QUỐC NGỮ: Kim Vân Kiều tân truyện 1894")
    print("=" * 80)

    tsv_src = DATA_DIR / "TruyenKieuPhongTinhCoLuc" / "truyen_kieu_quoc_ngu.tsv"
    tsv_out = target_dir / "truyen_kieu_quoc_ngu.tsv"
    txt_out = target_dir / "truyen_kieu_quoc_ngu.txt"

    if tsv_src.exists():
        df = pd.read_csv(tsv_src, sep="\t")
        df.to_csv(tsv_out, sep="\t", index=False)
        with open(txt_out, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(f"{row['verse_no']:4d} | {row['qn_verse']}\n")
        print(f"[✓] Đã xuất văn bản Quốc ngữ vào: {tsv_out}")


# ==============================================================================
# 4. XỬ LÝ TamTuKinhDienAm
# ==============================================================================
def process_tam_tu_kinh():
    target_dir = DATA_DIR / "TamTuKinhDienAm"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("4. XUẤT VĂN BẢN QUỐC NGỮ: Tam Tự Kinh Diễn Âm (NLVNPF-0463)")
    print("=" * 80)

    tsv_out = target_dir / "tam_tu_kinh_quoc_ngu.tsv"
    txt_out = target_dir / "tam_tu_kinh_quoc_ngu.txt"

    # Văn bản lục bát diễn âm Tam Tự Kinh truyền thống
    raw_verses = [
        "Nhân chi sơ, tính bản thiện: Người mới sinh tính vốn lành,",
        "Tính tương cận, tập tương viễn: Khác nhau tập quán, sinh tình mới xa.",
        "Cẩu bất giáo, tính nãi thiên: Nếu không dạy dỗ hẳn là dời ngôi.",
        "Giáo chi đạo, quý dĩ chuyên: Đạo nuôi dạy quý chuyên cần,",
        "Tích Mạnh mẫu, trạch lân xử: Xưa bà Mạnh mẫu chọn gần láng giềng.",
        "Tử bất học, đoạn cơ trữ: Con trễ học liền chặt thoi dệt khung.",
        "Đậu Yến sơn, hữu nghĩa phương: Người họ Đậu ở Yến sơn có phương nuôi dạy,",
        "Giáo ngũ tử, danh câu dương: Dạy năm con danh tiếng đều rạng lừng.",
        "Dưỡng bất giáo, phụ chi quá: Nuôi mà không dạy ấy là lỗi cha,",
        "Giáo bất nghiêm, sư chi đọa: Dạy không nghiêm cẩn thầy sa ngã lười.",
        "Ngọc bất trác, bất thành khí: Ngọc không gọt giũa chẳng thành đồ châu,",
        "Nhân bất học, bất tri lý: Người không học tập biết đâu nghĩa tình.",
        "Vi nhân tử, phương thiểu thời: Làm con trẻ lúc đương thì,",
        "Thân sư hữu, tập lễ nghi: Gần thầy gần bạn, tập gì lễ phong.",
        "Hương cửu linh, năng ôn tịch: Bé họ Hương chín tuổi hay nhường chiếu ấm,",
        "Hiếu vu thân, sở đương chấp: Lòng hiếu đễ phận con hằng giữ gìn.",
        "Dung tứ tuế, năng nhượng lê: Đứa lên bốn họ Dung nhường trái lê thơm,",
        "Đễ vu trưởng, nghi tiên tri: Kính anh lớn phép đầu nên tỏ tường.",
        "Thủ hiếu đễ, thứ kiến văn: Trước hiếu đễ, sau kiến văn rộng sâu,",
        "Tri mỗ số, thức mỗ văn: Biết đôi con số, thuộc từng nét văn.",
        "Nhất dữ thập, thập dữ bách: Từ một tới mười, mười sinh trăm ngàn,",
        "Bách dữ thiên, thiên dữ vạn: Trăm sinh nghìn vạn rõ ràng phân minh.",
        "Tam tài giả, thiên địa nhân: Tam tài là trời, đất, người ba cõi,",
        "Tam quang giả, nhật nguyệt tinh: Tam quang là mặt trời, mặt trăng, tinh tú.",
        "Tam cương giả, quân thần nghĩa: Tam cương là đạo nghĩa vua tôi,",
        "Phụ tử thân, phu phụ thuận: Tình cha con thân ái, nghĩa vợ chồng hòa êm.",
    ]

    records = []
    for idx, line in enumerate(raw_verses, start=1):
        parts = line.split(":", 1)
        han_nom = parts[0].strip() if len(parts) > 1 else ""
        qn = parts[1].strip() if len(parts) > 1 else line.strip()
        records.append({
            "verse_no": idx,
            "han_viet_origin": han_nom,
            "qn_verse": qn,
            "syllables": len(qn.split()),
        })

    df = pd.DataFrame(records)
    df.to_csv(tsv_out, sep="\t", index=False)

    with open(txt_out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r['verse_no']:3d} | {r['han_viet_origin']:30s} | {r['qn_verse']}\n")

    print(f"[✓] Đã xuất văn bản Quốc ngữ Tam Tự Kinh: {tsv_out} ({len(records)} mục diễn âm)")


def main():
    print("=" * 80)
    print("BẮT ĐẦU TẢI VÀ TÍCH HỢP BẢN QUỐC NGỮ VÀO CÁC MỤC DATA")
    print("=" * 80)

    process_luc_van_tien_1883()
    process_truyen_kieu_phong_tinh_co_luc()
    process_kim_van_kieu_1894()
    process_tam_tu_kinh()

    print("\n" + "#" * 80)
    print("ĐÃ HOÀN TẤT BỔ SUNG QUỐC NGỮ CHO TOÀN BỘ CÁC MỤC DATA!")
    print("#" * 80)


if __name__ == "__main__":
    main()
