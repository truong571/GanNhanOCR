"""Đánh giá kết quả Enhanced Self-Training v3 (Phương án 1 + 2).

Chạy trên máy Mac:
    .venv/bin/python lab/enhanced_self_training_v3/evaluate_enhanced.py

Đầu vào:
    lab/enhanced_self_training_v3/enhanced_results.zip (hoặc các file đã giải nén)

Đầu ra:
    - enhanced_summary.json: Tóm tắt chỉ số huấn luyện & giải cứu cho luận văn
    - enhanced_preview.html: Báo cáo trực quan kèm ảnh crop chữ viết tay bút lông
"""

from __future__ import annotations

import base64
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

ZIP_PATH = HERE / "enhanced_results.zip"
CSV_PATH = HERE / "enhanced_pseudo_labels.csv"
METRICS_PATH = HERE / "training_metrics.json"
CROPS_PATH = REPO / "KhoiB/v3/crops_v3.npz"
HTML_OUT = HERE / "enhanced_preview.html"
SUMMARY_OUT = HERE / "enhanced_summary.json"


def check_and_unzip():
    if not CSV_PATH.exists() and ZIP_PATH.exists():
        print(f"Đang giải nén kết quả từ: {ZIP_PATH} ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(HERE)
        print("✓ Đã giải nén xong!")


def run_evaluation():
    print("=" * 72)
    print("ĐÁNH GIÁ KẾT QUẢ ENHANCED SELF-TRAINING V3 (PHƯƠNG ÁN 1 + 2)")
    print("=" * 72)

    check_and_unzip()

    if not CSV_PATH.exists():
        print(f"⚠️ Chưa có file kết quả: {CSV_PATH}")
        print("👉 Hãy chạy notebook 'kaggle_run_enhanced_pipeline.ipynb' trên Kaggle GPU,")
        print(f"   tải file 'enhanced_results.zip' về đặt vào: {HERE}")
        print("   sau đó chạy lại lệnh này.")
        return

    df = pd.read_csv(CSV_PATH)
    total_infer = len(df)
    print(f"Tổng số ô suy diễn (chưa xác nhận): {total_infer:,}")

    # Thống kê quyết định
    dec_counts = df["decision"].value_counts().to_dict()
    print("\n1. PHÂN BỐ QUYẾT ĐỊNH:")
    for k, v in dec_counts.items():
        print(f"  - {k}: {v:,} ({v/total_infer:.1%})")

    # Thống kê theo các ngưỡng
    print("\n2. TỶ LỆ GIẢI CỨU THEO NGƯỠNG ĐỘ TIN CẬY:")
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    records = []
    for tau in thresholds:
        m = (df["in_dict"] == True) & (df["prob"] >= tau)
        cnt = m.sum()
        pct = cnt / total_infer
        print(f"  - Ngưỡng tau = {tau:.2f}: {cnt:,} ô ({pct:.1%})")
        records.append({"threshold": tau, "rescued_count": int(cnt), "rescued_pct": round(pct, 4)})

    # Mô phỏng tập GOLD v3 (bộ thuần tự động 49.867 + 2.172 giải cứu Vòng 1 = 52.039)
    gold_v2 = 52039
    n_high = dec_counts.get("RESCUE_V3_HIGH", 0)
    n_med = dec_counts.get("RESCUE_V3_MED", 0)
    gold_v3 = gold_v2 + n_high + n_med

    # Đọc metrics nếu có
    metrics_info = ""
    last_m = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics_data = json.load(f)
        last_m = metrics_data[-1] if metrics_data else {}
        print("\n3. CHỈ SỐ HUẤN LUYỆN CUỐI CÙNG (EPOCH CUỐI):")
        print(f"  - Train Accuracy: {last_m.get('train_acc', 0):.1%}")
        print(f"  - Val Top-1:      {last_m.get('val_top1', 0):.1%}")
        print(f"  - Val Top-5:      {last_m.get('val_top5', 0):.1%}")
        print(f"  - Val Loss:       {last_m.get('val_loss', 0):.4f}")

        metrics_info = f"""
        <div class="metrics-card">
            <h3>📊 Chỉ Số Huấn Luyện EnhancedNomOCRNet (Epoch {last_m.get('epoch', 20)}/20)</h3>
            <div class="grid-stats">
                <div class="stat"><span class="val">{last_m.get('train_acc', 0):.1%}</span><span class="lbl">Train Accuracy</span></div>
                <div class="stat"><span class="val">{last_m.get('val_top1', 0):.1%}</span><span class="lbl">Validation Top-1</span></div>
                <div class="stat"><span class="val">{last_m.get('val_top5', 0):.1%}</span><span class="lbl">Validation Top-5</span></div>
                <div class="stat"><span class="val">{last_m.get('val_loss', 0):.3f}</span><span class="lbl">Val Loss</span></div>
            </div>
        </div>
        """

    summary = {
        "tong_so_suy_dien": total_infer,
        "thong_ke_nguong": records,
        "phan_bo_quyet_dinh": dec_counts,
        "tap_gold_v2": gold_v2,
        "tap_gold_v3_du_kien": gold_v3,
        "tong_so_o_giai_cuu_them": n_high + n_med,
        "chi_so_mo_hinh_cuoi": last_m,
    }

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu tóm tắt số liệu vào: {SUMMARY_OUT}")

    # Sinh HTML
    _generate_html(df, metrics_info)


def _generate_html(df: pd.DataFrame, metrics_info: str):
    crop_lookup = {}
    if CROPS_PATH.exists():
        try:
            print(f"Đang nạp ảnh crops từ: {CROPS_PATH} ...")
            npz = np.load(CROPS_PATH)
            X = npz["X"]
            books = npz["book"]
            pages = npz["page"]
            cols = npz["column"]
            nom_idxs = npz["nom_idx"]
            for i in range(len(X)):
                k = (str(books[i]), str(pages[i]), int(cols[i]), int(nom_idxs[i]))
                crop_lookup[k] = i
            print(f"✓ Đã lập chỉ mục {len(crop_lookup):,} ảnh crop.")
        except Exception as e:
            print(f"Lưu ý: Không tải được crops npz ({e})")

    def get_crop_b64(b, p, c, idx):
        if not crop_lookup:
            return ""
        key = (str(b), str(p), int(c), int(idx))
        if key not in crop_lookup:
            return ""
        arr = X[crop_lookup[key]]
        im = Image.fromarray(arr)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    sample_rescued = df[df["decision"] == "RESCUE_V3_HIGH"].head(60)
    if len(sample_rescued) == 0:
        sample_rescued = df[df["decision"] == "RESCUE_V3_MED"].head(60)

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Báo Cáo Giải Cứu Enhanced OCR (Phương Án 1 + 2)</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; padding: 32px; max-width: 1400px; margin: auto; }",
        "h1 { color: #38bdf8; font-size: 28px; margin-bottom: 8px; }",
        "p.subtitle { color: #94a3b8; font-size: 15px; margin-bottom: 24px; }",
        ".metrics-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 28px; }",
        ".metrics-card h3 { color: #e2e8f0; margin-top: 0; margin-bottom: 16px; font-size: 18px; }",
        ".grid-stats { display: flex; gap: 24px; }",
        ".stat { background: #0f172a; padding: 14px 20px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #334155; }",
        ".stat .val { display: block; font-size: 26px; font-weight: 700; color: #38bdf8; }",
        ".stat .lbl { display: block; font-size: 13px; color: #94a3b8; margin-top: 4px; }",
        "table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 16px; background: #1e293b; border-radius: 10px; overflow: hidden; border: 1px solid #334155; }",
        "th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; vertical-align: middle; }",
        "th { background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }",
        "tr:hover { background: #243248; }",
        ".crop-box { width: 56px; height: 56px; background: #000; border-radius: 6px; border: 1px solid #475569; display: flex; align-items: center; justify-content: center; }",
        ".crop-box img { max-width: 52px; max-height: 52px; border-radius: 4px; filter: contrast(1.15); }",
        ".bad { color: #f87171; text-decoration: line-through; font-size: 1.3em; }",
        ".good { color: #4ade80; font-weight: bold; font-size: 1.6em; }",
        ".prob { color: #fbbf24; font-weight: 700; font-size: 14px; }",
        ".badge { background: #0284c7; color: #e0f2fe; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }",
        "code { background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #a5b4fc; font-family: monospace; }",
        "</style></head><body>",
        "<h1>🎯 Báo Cáo Giải Cứu Enhanced OCR Vòng 3 (Phương Án 1 + 2)</h1>",
        "<p class='subtitle'>Minh chứng thực nghiệm kết hợp: Mô hình <i>EnhancedNomOCRNet</i> (ResNet-SE Attention) huấn luyện trên 54.094 nhãn sạch mở rộng và tích hợp bộ tăng cường nét bút lông chép tay cổ.</p>",
        metrics_info,
        "<table>",
        "<tr><th>STT</th><th>Ảnh Bút Lông</th><th>Vị Trí</th><th>Âm QN</th><th>OCR Cũ (Sai)</th><th>Nhãn Mới (Cứu Được)</th><th>Mã Unicode</th><th>Độ Tin Cậy</th><th>Quyết Định</th></tr>"
    ]

    for idx, (_, r) in enumerate(sample_rescued.iterrows(), 1):
        loc = f"{r['book']}<br><span style='color:#64748b;font-size:12px;'>{r['page']} c{r['column']}</span>"
        b64 = get_crop_b64(r["book"], r["page"], r["column"], r["nom_idx"])
        img_html = f"<div class='crop-box'><img src='{b64}'></div>" if b64 else "<span style='color:#64748b;'>[N/A]</span>"

        html.append(
            f"<tr><td>{idx}</td><td>{img_html}</td><td>{loc}</td><td><b style='font-size:16px;color:#e2e8f0;'>{r['syllable']}</b></td>"
            f"<td class='bad'>{r['ocr_char_old']}</td>"
            f"<td class='good'>{r['predicted_nom']}</td>"
            f"<td><code>{r['unicode']}</code></td>"
            f"<td class='prob'>{r['prob']:.1%}</td>"
            f"<td><span class='badge'>{r['decision']}</span></td></tr>"
        )

    html.append("</table></body></html>")

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"✓ Đã tạo báo cáo HTML trực quan: {HTML_OUT}")


if __name__ == "__main__":
    run_evaluation()
