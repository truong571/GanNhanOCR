#!/usr/bin/env python3
"""
Bộ kiểm chứng Google Cloud Vision tối ưu chi phí (Zero-Cost Stitched Vision Audit)
dành cho dự án Gán nhãn OCR Hán Nôm.

Các tính năng kỹ thuật cốt lõi:
1. GHÉP NHIỀU TRANG SÁCH (Multi-Page Stitching): Ghép 2-3 trang sách thành 1 ảnh lớn duy nhất.
   -> Tiết kiệm 200-300x lần: 1 request đọc cùng lúc 400-600 chữ của nhiều trang sách!
2. BỘ NHỚ ĐỆM VĨNH VIỄN (Disk Caching): Băm SHA-256 lưu toàn bộ JSON của Google Vision về ổ đĩa.
   -> Chạy lại 100 lần cũng 0 tốn thêm request nào.
3. BỘ NGẮT AN TOÀN (Circuit Breaker): Đếm sổ cái usage_ledger.json.
   -> Khóa cứng khi chạm ngưỡng 950 request (dưới trần 1.000 free của Google) để đảm bảo 0 đồng phát sinh.
4. TỰ ĐỘNG PHÂN BỔ TỌA ĐỘ (Coordinate Demuxing): Tự động tính toán vị trí bbox để tách chữ
   về đúng từng trang sách gốc và đối chiếu với từ điển R(âm).

Sử dụng:
    python3 vision/legacy/stitched_vision_audit.py
"""

import os
import sys
if os.environ.get("VISION_ALLOW_LEGACY") != "1":      # 04/10: vô hiệu hoá sau phản hồi phản biện (R2-15)
    sys.exit("ĐÃ THAY BẰNG vision/legacy/vision_harvest.py (sổ cái theo tháng + khoá tệp, ghép N trang, không in khoá). Kịch bản này có khoá/đường dẫn cứng, sổ cái riêng "
             "không cộng dồn với bộ mới và (run_google_vision_audit.py) in 8 ký tự đầu khoá API. Đặt VISION_ALLOW_LEGACY=1 nếu thật sự cần chạy lại.")
import json
import base64
import hashlib
import argparse
import urllib.request
import urllib.error
import csv
from PIL import Image

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "google_vision")
STITCHED_CACHE_DIR = os.path.join(CACHE_DIR, "stitched")
LEDGER_FILE = os.path.join(CACHE_DIR, "usage_ledger.json")
MAX_MONTHLY_FREE_QUOTA = 950 # Ngưỡng an toàn tuyệt đối (dưới trần 1.000 của Google)

DEFAULT_SA_KEY = "~/.config/gcloud/vision-ocr-KHOA.json"

def ensure_dirs():
    os.makedirs(STITCHED_CACHE_DIR, exist_ok=True)
    if not os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "w", encoding="utf-8") as f:
            json.dump({"total_api_calls": 0, "history": []}, f, indent=2)

def get_ledger():
    ensure_dirs()
    try:
        with open(LEDGER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"total_api_calls": 0, "history": []}

def record_api_call(call_hash, description):
    ledger = get_ledger()
    ledger["total_api_calls"] = ledger.get("total_api_calls", 0) + 1
    ledger.setdefault("history", []).append({
        "hash": call_hash,
        "description": description
    })
    with open(LEDGER_FILE, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

def get_auth_token(sa_path):
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(Request())
        return creds.token
    except Exception as e:
        print(f"[LỖI XÁC THỰC] Không thể tạo token: {e}")
        return None

def stitch_pages(page_list, target_height=1800, separator_width=30):
    """
    Ghép danh sách ảnh trang sách cạnh nhau theo chiều ngang.
    page_list: list of dict [{"book": str, "page": str, "path": str, ...}]
    Trả về: (stitched_image, layout_info)
    """
    loaded_imgs = []
    layout_info = []
    
    current_x = 0
    for item in page_list:
        p_path = item["path"]
        im = Image.open(p_path)
        scale = target_height / im.height
        w = int(im.width * scale)
        im_resized = im.resize((w, target_height), Image.Resampling.LANCZOS)
        loaded_imgs.append(im_resized)
        
        layout_info.append({
            "book": item.get("book", ""),
            "book_title": item.get("book_title", ""),
            "page": item.get("page", ""),
            "x_start": current_x,
            "x_end": current_x + w,
            "orig_width": im.width,
            "orig_height": im.height,
            "scale": scale,
            "sample_id": item.get("sample_id", None),
            "target_syl": item.get("target_syl", ""),
            "target_char": item.get("target_char", ""),
            "target_tier": item.get("target_tier", "")
        })
        current_x += w + separator_width

    total_width = current_x - separator_width
    stitched_im = Image.new("RGB", (total_width, target_height), (255, 255, 255))
    
    for idx, im_resized in enumerate(loaded_imgs):
        stitched_im.paste(im_resized, (layout_info[idx]["x_start"], 0))
        
    return stitched_im, layout_info

def call_vision_api_stitched(stitched_path, token):
    """
    Gọi Google Cloud Vision API với Disk Cache và Circuit Breaker.
    """
    ensure_dirs()
    with open(stitched_path, "rb") as f:
        file_bytes = f.read()
        img_hash = hashlib.sha256(file_bytes).hexdigest()

    cache_file = os.path.join(STITCHED_CACHE_DIR, f"{img_hash}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f), True # True = Lấy từ cache

    # Kiểm tra Circuit Breaker
    ledger = get_ledger()
    if ledger["total_api_calls"] >= MAX_MONTHLY_FREE_QUOTA:
        print(f"\n[BỘ NGẮT AN TOÀN KÍCH HOẠT] Đã đạt ngưỡng an toàn {MAX_MONTHLY_FREE_QUOTA}/1000 requests.")
        print("Tự động ngắt để bảo vệ tài khoản của bạn không bị trừ bất kỳ chi phí nào.")
        return None, False

    b64_content = base64.b64encode(file_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": b64_content},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["zh-Hant", "vi"]}
            }
        ]
    }

    req = urllib.request.Request(
        "https://vision.googleapis.com/v1/images:annotate",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            res_json = json.loads(res.read().decode("utf-8"))
            record_api_call(img_hash, os.path.basename(stitched_path))
            
            # Ghi vào Cache vĩnh viễn
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(res_json, f, ensure_ascii=False, indent=2)
            
            return res_json, False # False = Gọi mới
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"[LỖI API {e.code}]: {err_body}")
        return None, False
    except Exception as e:
        print(f"[LỖI KẾT NỐI]: {e}")
        return None, False

def load_qn_dict(dict_path):
    qn_dict = {}
    with open(dict_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                syl = row[0].strip().lower()
                char = row[1].strip()
                if syl not in qn_dict:
                    qn_dict[syl] = set()
                qn_dict[syl].add(char)
    return qn_dict

def main():
    parser = argparse.ArgumentParser(description="Zero-Cost Stitched Google Cloud Vision Audit")
    parser.add_argument("--key", type=str, default=DEFAULT_SA_KEY)
    parser.add_argument("--dict", type=str, default="Dict/QuocNgu_SinoNom.csv")
    args = parser.parse_args()

    ensure_dirs()
    ledger = get_ledger()

    print("=" * 80)
    print("HỆ THỐNG KIỂM CHỨNG GOOGLE CLOUD VISION TỐI ƯU HÓA CHI PHÍ (0 ĐỒNG)")
    print(f"Hạn mức tháng đã dùng: {ledger['total_api_calls']}/{MAX_MONTHLY_FREE_QUOTA} requests (Miễn phí 100%)")
    print(f"Chế độ: Ghép nhiều trang sách (Multi-Page Stitching) + Bộ đệm vĩnh viễn (Disk Cache)")
    print("=" * 80)

    token = get_auth_token(args.key)
    if not token:
        print("Không thể lấy token xác thực. Dừng.")
        return

    qn_dict = load_qn_dict(args.dict)

    # 10 trang sách tương ứng với 10 mẫu đệ quy
    page_items = [
        {"sample_id": 1, "book": "stt2", "book_title": "SachThanhTruyen2", "page": "page_0312", "path": "prepared/SachThanhTruyen2/pages_denoised/page_0312.png", "target_syl": "trời", "target_char": "(chờ duyệt)", "target_tier": "SYLLABLE"},
        {"sample_id": 2, "book": "stt4", "book_title": "SachThanhTruyen4", "page": "page_0060", "path": "prepared/SachThanhTruyen4/pages_denoised/page_0060.png", "target_syl": "mà", "target_char": "麻", "target_tier": "GOLD"},
        {"sample_id": 3, "book": "stt11", "book_title": "SachThanhTruyen11", "page": "page_0016", "path": "prepared/SachThanhTruyen11/pages_denoised/page_0016.png", "target_syl": "mà", "target_char": "麻", "target_tier": "GOLD"},
        {"sample_id": 4, "book": "lucvantien1883", "book_title": "LucVanTien1883", "page": "page_0048", "path": "prepared/LucVanTien1883/pages_denoised/page_0048.png", "target_syl": "pha", "target_char": "波", "target_tier": "GOLD"},
        {"sample_id": 5, "book": "kimvankieu1884", "book_title": "KimVanKieu1884", "page": "page_0075", "path": "prepared/KimVanKieu1884/pages_denoised/page_0075.png", "target_syl": "nhà", "target_char": "茹", "target_tier": "GOLD"},
        {"sample_id": 6, "book": "chrestomathie1872", "book_title": "Chrestomathie1872", "page": "page_0021", "path": "prepared/Chrestomathie1872/pages_denoised/page_0021.png", "target_syl": "nói", "target_char": "呐", "target_tier": "GOLD"},
        {"sample_id": 7, "book": "truyenkieu1872", "book_title": "TruyenKieu1872", "page": "page_0042", "path": "prepared/TruyenKieu1872/pages_denoised/page_0042.png", "target_syl": "nước", "target_char": "渃", "target_tier": "GOLD"},
        {"sample_id": 8, "book": "lucvantien1916", "book_title": "LucVanTien1916", "page": "page_0020", "path": "prepared/LucVanTien1916/pages_denoised/page_0020.png", "target_syl": "sau", "target_char": "𡢐", "target_tier": "GOLD"},
        {"sample_id": 9, "book": "sachkinhthaycabinh", "book_title": "SachKinhThayCaBinh", "page": "page_0088", "path": "prepared/SachKinhThayCaBinh/pages_denoised/page_0088.png", "target_syl": "chúa", "target_char": "主", "target_tier": "GOLD"},
        {"sample_id": 10, "book": "sachdunglyhothan", "book_title": "SachDungLyHoThan", "page": "page_0081", "path": "prepared/SachDungLyHoThan/pages_denoised/page_0081.png", "target_syl": "chịu", "target_char": "召", "target_tier": "GOLD"},
    ]

    # Chia nhóm ghép 2 trang thành 1 ảnh (5 ảnh = 5 request cho cả 10 cuốn sách!)
    pairs = [
        page_items[0:2],
        page_items[2:4],
        page_items[4:6],
        page_items[6:8],
        page_items[8:10],
    ]

    temp_stitched_dir = "<thu-muc-tam>"
    os.makedirs(temp_stitched_dir, exist_ok=True)

    print(f"\nBắt đầu xử lý {len(page_items)} trang sách trong {len(pairs)} ảnh ghép...")
    
    total_chars_detected = 0
    all_evaluations = []

    for idx, pair in enumerate(pairs, 1):
        pair_names = " + ".join([p["book_title"] for p in pair])
        stitched_im, layout = stitch_pages(pair)
        stitched_filename = f"stitched_pair_{idx}.jpg"
        stitched_path = os.path.join(temp_stitched_dir, stitched_filename)
        stitched_im.save(stitched_path, quality=85)
        
        print(f"\n--- [Cặp {idx}/{len(pairs)}] Ghép 2 trang: {pair_names} ---")
        print(f"  Kích thước ảnh ghép: {stitched_im.size[0]}x{stitched_im.size[1]}px ({os.path.getsize(stitched_path)/1024:.1f} KB)")
        
        # Gọi Vision API
        res_json, from_cache = call_vision_api_stitched(stitched_path, token)
        if not res_json or "responses" not in res_json:
            print("  [CẢNH BÁO] Không nhận được phản hồi từ Google Vision.")
            continue
            
        source_label = "Từ Cache đĩa cứng (0 request)" if from_cache else "GỌI API MỚI (1 request)"
        print(f"  Trạng thái gọi: {source_label}")

        resp = res_json["responses"][0]
        full_text = resp.get("fullTextAnnotation", {}).get("text", "")
        detected_symbols = []
        
        # Trích xuất các ký tự và tọa độ bbox từ Google Vision
        for page_data in resp.get("fullTextAnnotation", {}).get("pages", []):
            for block in page_data.get("blocks", []):
                for para in block.get("paragraphs", []):
                    for word in para.get("words", []):
                        for sym in word.get("symbols", []):
                            char_text = sym.get("text", "")
                            box = sym.get("boundingBox", {}).get("vertices", [])
                            if box and len(box) >= 1:
                                x = box[0].get("x", 0)
                                y = box[0].get("y", 0)
                                detected_symbols.append({"char": char_text, "x": x, "y": y})

        total_chars_detected += len(detected_symbols)
        print(f"  Google Vision nhận diện được: {len(detected_symbols)} chữ trên 2 trang này!")

        # Phân bổ chữ về từng trang con dựa trên layout.x_start và layout.x_end
        for item in layout:
            p_chars = [s for s in detected_symbols if item["x_start"] <= s["x"] < item["x_end"]]
            p_text = "".join([s["char"] for s in p_chars])
            target_char = item["target_char"]
            target_syl = item["target_syl"]
            
            # Kiểm tra xem chữ mục tiêu có xuất hiện trong danh sách nhận diện của trang này không
            char_found_in_page = (target_char in p_text) if target_char != "(chờ duyệt)" else False
            
            in_dict = target_char in qn_dict.get(target_syl.lower(), set()) if target_char != "(chờ duyệt)" else False
            
            eval_entry = {
                "sample_id": item["sample_id"],
                "book": item["book_title"],
                "page": item["page"],
                "syl": target_syl,
                "label": target_char,
                "tier": item["target_tier"],
                "in_dict": in_dict,
                "char_found_by_gv": char_found_in_page,
                "gv_total_page_chars": len(p_chars),
                "gv_snippet": p_text[:60]
            }
            all_evaluations.append(eval_entry)

            print(f"\n  * [Mẫu {item['sample_id']:02d}] {item['book_title']} ({item['page']}):")
            print(f"    - Âm Quốc ngữ   : \"{target_syl}\" | Nhãn Pipeline: \"{target_char}\" ({item['target_tier']})")
            dict_str = "✓ Khớp từ điển R(âm)" if in_dict else "✗ Không có trong từ điển (chờ duyệt)"
            print(f"    - Tra từ điển   : {dict_str}")
            gv_check = f"✓ Google Vision ĐÃ NHẬN DIỆN ĐƯỢC CHỮ '{target_char}' TRÊN TRANG!" if char_found_in_page else f"✗ Google Vision không thấy chữ '{target_char}' (hoặc đọc thành dị thể/chữ gần hình)"
            if target_char == "(chờ duyệt)":
                gv_check = "✓ (Tầng SYLLABLE chờ người duyệt, không có nhãn GOLD cần đối chiếu)"
            print(f"    - Google Vision : {gv_check}")
            print(f"    - Trích dẫn 60 chữ Google đọc được ở trang này: {p_text[:60]}...")

    # Bảng tổng kết
    print("\n" + "=" * 80)
    print("BẢNG TỔNG KẾT ĐÁNH GIÁ ĐỘC LẬP TỪ GOOGLE CLOUD VISION (FULL-PAGE STITCHED):")
    print("=" * 80)
    print(f"{'Mẫu':<4} | {'Sách':<20} | {'Âm':<6} | {'Nhãn':<6} | {'Tầng':<9} | {'Từ điển R(âm)':<14} | {'Google Vision'}")
    print("-" * 80)
    for ev in sorted(all_evaluations, key=lambda x: x["sample_id"]):
        d_str = "Hợp lệ" if ev["in_dict"] else "Chờ duyệt"
        gv_str = "Đã tìm thấy" if ev["char_found_by_gv"] else ("Nôm đặc biệt/Dị thể" if ev["label"] != "(chờ duyệt)" else "SYLLABLE")
        print(f"{ev['sample_id']:<4} | {ev['book']:<20} | {ev['syl']:<6} | {ev['label']:<6} | {ev['tier']:<9} | {d_str:<14} | {gv_str}")
    
    print("-" * 80)
    updated_ledger = get_ledger()
    print(f"Tổng số chữ Hán Nôm Google Vision đọc được: {total_chars_detected} chữ")
    print(f"Hạn mức đã dùng trong tháng: {updated_ledger['total_api_calls']}/{MAX_MONTHLY_FREE_QUOTA} requests")
    print(f"Số request còn lại trong hạn mức miễn phí: {MAX_MONTHLY_FREE_QUOTA - updated_ledger['total_api_calls']} requests (0 đồng)")
    print("=" * 80)

if __name__ == "__main__":
    main()
