"""v_pb_thu_hqc_03_k09_giay.py — PHẢN BIỆN độc lập (04/10): kiểm khẳng định "mẫu giấy của K09 là giấy TỔNG HỢP (nhiễu Gauss) ở B34/L16/TK" bằng cách
gọi ĐÚNG hàm extract_book_paper_model của p05_full_corpus_he_quy_chieu.py rồi so mảng trả về với nhiễu Gauss hạt giống 42 (nhánh dự phòng dòng 112-114),
và đo mức nền/mực trên ẢNH GỐC cùng trang (K09 đọc prepared/<bộ>/pages = ảnh đã xử lý, trong khi crop sản xuất cắt từ ảnh GỐC).  Chỉ đọc."""
import importlib.util, json, sys
from pathlib import Path
import cv2, numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
spec = importlib.util.spec_from_file_location("p05fc", REPO / "lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p05_full_corpus_he_quy_chieu.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from pipeline.align_engine.build_dataset import load_original_page
out = {}
for b in ["B18", "B34", "L16", "TK"]:
    pm = m.extract_book_paper_model(b)
    patch = pm["paper_patch"]
    prep = T.REPO / T.BOOKS[b]["prep"]
    name = pm["sample_page"]
    page = cv2.imread(str(prep / "pages" / name), cv2.IMREAD_GRAYSCALE)
    Tv, _ = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = page.shape; mn = 10**9
    for y in range(20, h - 148, 32):
        for x in range(20, w - 148, 32):
            mn = min(mn, int(np.sum(page[y:y + 128, x:x + 128] < Tv)))
    bg_std = float(np.std(page[page >= Tv]))
    rng = np.random.default_rng(42)
    syn = np.clip(rng.normal(pm["bg_median"], max(bg_std, 4.0), (128, 128)), 0, 255).astype(np.uint8)
    is_syn = bool(patch.shape == syn.shape and np.array_equal(patch, syn))
    # trang gốc cùng tên
    O = load_original_page(prep, Path(name).stem, shape_like=page.shape)
    og = cv2.cvtColor(O, cv2.COLOR_BGR2GRAY)
    To, _ = cv2.threshold(og, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out[b] = dict(sample_page=name, k09_bg=pm["bg_median"], k09_ink=pm["ink_median"], min_ink_window=mn, fallback_threshold=200,
                  patch_is_gauss_seed42=is_syn, patch_mean=round(float(patch.mean()), 1), patch_std=round(float(patch.std()), 1),
                  processed_page_p50=float(np.median(page)), processed_page_p98=float(np.percentile(page, 98)),
                  original_page_bg_otsu_median=float(np.median(og[og >= To])), original_page_p98=float(np.percentile(og, 98)),
                  original_ink_otsu_median=float(np.median(og[og < To])))
    print(b, json.dumps(out[b], ensure_ascii=False))
OUT = REPO / "measure_out/_tn11/verify/pb_thu_hqc"; OUT.mkdir(parents=True, exist_ok=True)
(OUT / "k09_giay.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
