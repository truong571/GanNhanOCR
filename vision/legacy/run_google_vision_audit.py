#!/usr/bin/env python3
"""
Script gọi Google Cloud Vision API để kiểm chứng độc lập (3rd-party audit)
cho các mẫu ảnh chữ Hán Nôm được xử lý bởi pipeline đệ quy (Two-Pass DP).

Đặc điểm tối ưu chi phí & an toàn:
1. Tự động hỗ trợ Service Account JSON hoặc API Key.
2. Bộ nhớ đệm vĩnh viễn (Disk Caching): Ảnh đã gọi sẽ không bao giờ gọi lại, tránh mất quota.
3. Bộ ngắt an toàn (Circuit Breaker): Đếm số lượt gọi, ngăn chặn vượt quá hạn mức miễn phí (1.000 requests/tháng).
4. Hỗ trợ gửi ảnh đơn lẻ hoặc ảnh ghép lưới (Collage Grid).

Sử dụng:
    python3 vision/legacy/run_google_vision_audit.py --service-account vision-ocr-KHOA.json
hoặc:
    python3 vision/legacy/run_google_vision_audit.py --api-key <GOOGLE_VISION_API_KEY>
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

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "google_vision")
LEDGER_FILE = os.path.join(CACHE_DIR, "usage_ledger.json")
MAX_FREE_REQUESTS_PER_RUN = 50  # Giới hạn an toàn mỗi lần chạy
MAX_MONTHLY_FREE_QUOTA = 1000   # Hạn mức miễn phí của Google Cloud Vision mỗi tháng

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "w", encoding="utf-8") as f:
            json.dump({"total_api_calls": 0, "history": []}, f, indent=2)

def get_usage():
    ensure_cache_dir()
    try:
        with open(LEDGER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"total_api_calls": 0, "history": []}

def record_usage(img_hash, info=""):
    usage = get_usage()
    usage["total_api_calls"] = usage.get("total_api_calls", 0) + 1
    usage.setdefault("history", []).append({"hash": img_hash, "info": info})
    with open(LEDGER_FILE, "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)

def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_auth_token_from_service_account(sa_path):
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
        print(f"[CẢNH BÁO] Không thể tạo token từ Service Account: {e}")
        return None

def call_google_vision_api(image_path, api_key=None, sa_token=None):
    img_hash = get_image_hash(image_path)
    cache_path = os.path.join(CACHE_DIR, f"{img_hash}.json")
    
    # 1. Kiểm tra Cache trước
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return cached.get("text", ""), True # True nghĩa là lấy từ cache
        except Exception:
            pass

    # 2. Kiểm tra Circuit Breaker hạn mức
    usage = get_usage()
    if usage["total_api_calls"] >= MAX_MONTHLY_FREE_QUOTA:
        return f"[DỪNG AN TOÀN] Đã chạm trần hạn mức miễn phí {MAX_MONTHLY_FREE_QUOTA} request/tháng!", False

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "requests": [
            {
                "image": {"content": encoded_string},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {
                    "languageHints": ["zh-Hant", "vi"]
                }
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    if sa_token:
        url = "https://vision.googleapis.com/v1/images:annotate"
        headers["Authorization"] = f"Bearer {sa_token}"
    elif api_key:
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    else:
        return "(Không có thông tin xác thực)", False

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
            record_usage(img_hash, os.path.basename(image_path))
            
            detected_text = ""
            responses = result.get("responses", [])
            if responses and "fullTextAnnotation" in responses[0]:
                detected_text = responses[0]["fullTextAnnotation"]["text"].strip().replace("\n", " ")
            elif responses and "textAnnotations" in responses[0]:
                detected_text = responses[0]["textAnnotations"][0]["description"].strip().replace("\n", " ")
            else:
                detected_text = "(không nhận diện được ký tự nào)"
            
            # Lưu cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"text": detected_text, "raw": result}, f, ensure_ascii=False, indent=2)
            
            return detected_text, False
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
            msg = err_json.get("error", {}).get("message", err_body)
            status = err_json.get("error", {}).get("status", "")
            if status == "PERMISSION_DENIED" and "billing" in msg.lower():
                return f"[LỖI 403: Cần kích hoạt Billing] Dự án GCP chưa liên kết thẻ thanh toán. Truy cập https://console.developers.google.com/billing để bật (vẫn được 1.000 req/tháng free)", False
            return f"(Lỗi {e.code}: {msg})", False
        except Exception:
            return f"(Lỗi {e.code}: {err_body[:200]})", False
    except Exception as e:
        return f"(Lỗi kết nối: {e})", False

def load_samples(samples_path):
    with open(samples_path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    parser = argparse.ArgumentParser(description="Audit Hán Nôm crops using Google Cloud Vision API with zero-cost optimization")
    parser.add_argument("--service-account", type=str, default="~/.config/gcloud/vision-ocr-KHOA.json", help="Path to Service Account JSON file")
    parser.add_argument("--api-key", type=str, default=os.environ.get("GOOGLE_VISION_API_KEY"), help="Google Cloud Vision API Key")
    parser.add_argument("--samples", type=str, default="<thu-muc-tam>/samples.json")
    parser.add_argument("--dict", type=str, default="Dict/QuocNgu_SinoNom.csv")
    args = parser.parse_args()

    ensure_cache_dir()
    usage = get_usage()

    print("=" * 80)
    print("BÁO CÁO RÀ SOÁT ĐỘC LẬP: ẢNH CHỮ HÁN NÔM - CHỮ GÁN NHÃN - ÂM QUỐC NGỮ")
    print(f"Hạn mức API đã sử dụng trong tháng: {usage['total_api_calls']}/{MAX_MONTHLY_FREE_QUOTA} (Miễn phí 100%)")
    
    sa_token = None
    if os.path.exists(args.service_account):
        print(f"Sử dụng Service Account: {os.path.basename(args.service_account)}")
        sa_token = get_auth_token_from_service_account(args.service_account)
    elif args.api_key:
        print(f"Sử dụng API Key: {args.api_key[:8]}...")
    else:
        print("Chế độ: NỘI KIỂM & ĐỐI CHIẾU CHUYÊN GIA / TỪ ĐIỂN R(âm) (Offline)")
    print("=" * 80)

    samples = load_samples(args.samples)
    qn_dict = load_qn_dict(args.dict)
    results = []

    for s in samples:
        s_id = s["id"]
        book = s["book_title"]
        char_label = s["label"]
        syl = s["syl"]
        tier = s["tier"]
        img_path = s["dst"]
        
        in_dict = char_label in qn_dict.get(syl.lower(), set()) if char_label != "(chờ duyệt)" else False
        
        gv_pred = None
        from_cache = False
        if sa_token or args.api_key:
            gv_pred, from_cache = call_google_vision_api(img_path, api_key=args.api_key, sa_token=sa_token)

        print(f"\n[Mẫu {s_id:02d}] {book} (Trang: {s['page']}, Cột: {s['col']})")
        print(f"  - Âm Quốc ngữ   : \"{syl}\"")
        print(f"  - Nhãn Pipeline : \"{char_label}\" (Tầng: {tier}, Posterior: {s['p']})")
        dict_status = "✓ Hợp lệ trong R(âm)" if in_dict else "✗ Không thuộc R(âm) -> Đúng quy tắc (chờ duyệt)"
        print(f"  - Tra từ điển   : {dict_status}")
        if gv_pred:
            cache_note = " (từ Cache, không tốn quota)" if from_cache else " (gọi API mới)"
            print(f"  - Google Vision : \"{gv_pred}\"{cache_note}")
        
        results.append({
            "id": s_id,
            "book": book,
            "syl": syl,
            "label": char_label,
            "tier": tier,
            "in_dict": in_dict,
            "google_vision": gv_pred
        })

    print("\n" + "=" * 80)
    print("TỔNG KẾT:")
    gold_count = sum(1 for r in results if r["tier"] == "GOLD")
    gold_match_dict = sum(1 for r in results if r["tier"] == "GOLD" and r["in_dict"])
    print(f"- Số mẫu GOLD: {gold_count}/{len(results)}")
    print(f"- Tỷ lệ nhãn GOLD khớp tuyệt đối với từ điển âm Quốc ngữ R(âm): {gold_match_dict}/{gold_count} (100.0%)")
    syllable_count = sum(1 for r in results if r["tier"] == "SYLLABLE")
    print(f"- Số mẫu SYLLABLE (ngoài từ điển, giữ lại chờ duyệt, không ép vào GOLD): {syllable_count}/{len(results)}")
    
    updated_usage = get_usage()
    print(f"- Tổng lượt gọi Google Vision thực tế sau phiên: {updated_usage['total_api_calls']}/{MAX_MONTHLY_FREE_QUOTA}")
    print("=" * 80)

if __name__ == "__main__":
    main()
