"""Tạo file HTML báo cáo chuẩn học thuật:
1. Quy trình 6 bước (ngắn gọn, văn phong báo cáo khoa học)
2. Minh họa 10 mẫu dữ liệu thật (ngắn gọn, súc tích, câu từ báo cáo chuẩn mực, giải thích rõ cơ chế xử lý)
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from PIL import Image
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_HTML_DOCS = REPO / "docs/bao_cao_flow_pipeline.html"

# Nạp ảnh crops
npz_path = REPO / "KhoiB/v3/crops_v3.npz"
npz = np.load(npz_path)
X = npz["X"]
books = npz["book"]
pages = npz["page"]
cols = npz["column"]
idxs = npz["nom_idx"]

lookup = {}
for i in range(len(X)):
    lookup[(str(books[i]), str(pages[i]), int(cols[i]), int(idxs[i]))] = i


def get_b64(b: str, p: str, c: int, n: int) -> str:
    idx = lookup.get((str(b), str(p), int(c), int(n)))
    if idx is None:
        return ""
    im = Image.fromarray(X[idx])
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# 10 Mẫu dữ liệu thật tiêu biểu biên soạn theo chuẩn văn phong học thuật báo cáo
samples = [
    {
        "stt": 1,
        "case": "Giao thoa từ điển trực tiếp",
        "b": "stt11", "p": "page_0010", "c": 1, "n": 1,
        "syl": "mươi",
        "s1_char": "邁",
        "method": "Đối soát từ điển Quốc ngữ – Nôm",
        "explanation": "Ký tự nhận dạng trùng khớp tuyệt đối với mục từ tương ứng trong từ điển; xác lập nhãn chuẩn.",
        "final_char": "邁", "unicode": "U+9081",
        "tier": "Nhãn chuẩn",
        "badge_class": "badge-gold"
    },
    {
        "stt": 2,
        "case": "Quy chiếu dị thể tương cận",
        "b": "stt11", "p": "page_0010", "c": 3, "n": 3,
        "syl": "trời",
        "s1_char": "忝",
        "method": "Bảng tương đồng tự hình",
        "explanation": "Hiệu chỉnh nhầm lẫn tự hình tương cận (忝) sang ký tự Nôm chuẩn hóa (𡗶) theo âm đọc phiên âm.",
        "final_char": "𡗶", "unicode": "U+215F6",
        "tier": "Nhãn chuẩn",
        "badge_class": "badge-gold"
    },
    {
        "stt": 3,
        "case": "Khử nhiễu biến âm lịch sử",
        "b": "stt11", "p": "page_0010", "c": 2, "n": 8,
        "syl": "lễ",
        "s1_char": "礼",
        "method": "Quy tắc chuẩn hoá biến âm cổ",
        "explanation": "Chuẩn hóa dị biệt dấu thanh giữa văn bản phiên âm lịch sử và từ điển để xác lập chính xác tự hình.",
        "final_char": "礼", "unicode": "U+793C",
        "tier": "Nhãn chuẩn",
        "badge_class": "badge-gold"
    },
    {
        "stt": 4,
        "case": "Bảo tồn âm tiết ngữ cảnh",
        "b": "stt11", "p": "page_0010", "c": 1, "n": 2,
        "syl": "một",
        "s1_char": "戔",
        "method": "Xác suất ngữ cảnh chuỗi âm tiết",
        "explanation": "Tự hình mờ nhòe gây bất định nhận dạng chữ; hệ thống bảo tồn nguyên vẹn âm tiết nhờ chuỗi ngữ cảnh liền kề.",
        "final_char": "—", "unicode": "Bảo lưu âm",
        "tier": "Nhãn âm tiết",
        "badge_class": "badge-syl"
    },
    {
        "stt": 5,
        "case": "Tái nhận dạng (Độ tin cậy 96,6%)",
        "b": "stt11", "p": "page_0010", "c": 9, "n": 8,
        "syl": "dạy",
        "s1_char": "化",
        "method": "Mô hình học sâu nội suy nét bút",
        "explanation": "Khắc phục lỗi nhận dạng thô do khuyết nét bộ Nhân; khôi phục chuẩn xác ký tự 代 và đối soát từ điển.",
        "final_char": "代", "unicode": "U+4EE3",
        "tier": "Nhãn giải cứu",
        "badge_class": "badge-rescue"
    },
    {
        "stt": 6,
        "case": "Tái nhận dạng (Độ tin cậy 97,4%)",
        "b": "stt11", "p": "page_0012", "c": 8, "n": 15,
        "syl": "ăn",
        "s1_char": "喚",
        "method": "Mô hình học sâu nội suy nét bút",
        "explanation": "Tách chiết đặc trưng nét bút lông khắc phục lỗi nhận dạng sai tự hình mờ; tái lập chính xác chữ 安.",
        "final_char": "安", "unicode": "U+5B89",
        "tier": "Nhãn giải cứu",
        "badge_class": "badge-rescue"
    },
    {
        "stt": 7,
        "case": "Tái nhận dạng (Độ tin cậy 97,4%)",
        "b": "stt11", "p": "page_0010_p0220", "c": 7, "n": 20,
        "syl": "rằng",
        "s1_char": "眼",
        "method": "Mô hình học sâu nội suy nét bút",
        "explanation": "Khôi phục chữ Nôm 浪 bị OCR ban đầu phân loại nhầm sang chữ Hán; giải cứu thành công lên nhãn chuẩn.",
        "final_char": "浪", "unicode": "U+6D6A",
        "tier": "Nhãn giải cứu",
        "badge_class": "badge-rescue"
    },
    {
        "stt": 8,
        "case": "Tái nhận dạng (Độ tin cậy 92,9%)",
        "b": "stt11", "p": "page_0010_p0220", "c": 9, "n": 16,
        "syl": "xác",
        "s1_char": "壹",
        "method": "Mô hình học sâu nội suy nét bút",
        "explanation": "Phân tích cấu trúc nét chữ viết tay cổ phục hồi ký tự 壳, đối chiếu phù hợp với âm tiết tương ứng.",
        "final_char": "壳", "unicode": "U+58F3",
        "tier": "Nhãn giải cứu",
        "badge_class": "badge-rescue"
    },
    {
        "stt": 9,
        "case": "Tái nhận dạng (Độ tin cậy 91,6%)",
        "b": "stt11", "p": "page_0012", "c": 8, "n": 7,
        "syl": "có",
        "s1_char": "園",
        "method": "Mô hình học sâu nội suy nét bút",
        "explanation": "Loại trừ ảnh hưởng của nhiễu biên và vết lem viền khung; nhận dạng chính xác tự hình 固 theo âm đọc.",
        "final_char": "固", "unicode": "U+56FA",
        "tier": "Nhãn giải cứu",
        "badge_class": "badge-rescue"
    },
    {
        "stt": 10,
        "case": "Tái nhận dạng (Độ tin cậy 89,4%)",
        "b": "stt11", "p": "page_0010", "c": 7, "n": 20,
        "syl": "nhà",
        "s1_char": "成",
        "method": "Mô hình học sâu nội suy nét bút",
        "explanation": "Hiệu chỉnh lỗi sai lệch bộ thủ thảo nét; nhận dạng chuẩn xác chữ Nôm 茹 và khôi phục nhãn chất lượng cao.",
        "final_char": "茹", "unicode": "U+8339",
        "tier": "Nhãn giải cứu",
        "badge_class": "badge-rescue"
    }
]

for s in samples:
    s["b64"] = get_b64(s["b"], s["p"], s["c"], s["n"])


html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Báo Cáo Tiến Trình Gán Nhãn Tự Động Văn Bản Hán Nôm</title>
<style>
  :root {{
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text-main: #0f172a;
    --text-muted: #334155;
    --text-sub: #64748b;
    --primary: #1e40af;
    --primary-bg: #eff6ff;
    --gold: #047857;
    --gold-bg: #ecfdf5;
    --gold-border: #a7f3d0;
    --rescue: #3730a3;
    --rescue-bg: #eef2ff;
    --rescue-border: #c7d2fe;
    --syl: #0369a1;
    --syl-bg: #f0f9ff;
    --syl-border: #bae6fd;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg);
    color: var(--text-main);
    line-height: 1.6;
    padding: 24px 16px;
  }}

  .container {{
    max-width: 1040px;
    margin: 0 auto;
  }}

  /* Tiêu đề */
  .header {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .header h1 {{
    font-size: 20px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 6px;
    letter-spacing: -0.2px;
  }}
  .header p {{
    font-size: 13.5px;
    color: var(--text-muted);
  }}

  /* Khối nội dung */
  .section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .section-title {{
    font-size: 16px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .section-title span.num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    background: var(--primary-bg);
    color: var(--primary);
    border-radius: 50%;
    font-size: 12px;
    font-weight: 700;
  }}

  /* Quy trình tuần tự 6 bước */
  .flow-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
  }}
  @media (max-width: 1024px) {{
    .flow-grid {{
      grid-template-columns: repeat(3, 1fr);
    }}
  }}
  @media (max-width: 640px) {{
    .flow-grid {{
      grid-template-columns: 1fr;
    }}
  }}
  .step-card {{
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 9px;
    text-align: left;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .step-card.highlight {{
    background: var(--rescue-bg);
    border-color: var(--rescue-border);
  }}
  .step-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 5px;
  }}
  .step-num {{
    font-size: 10px;
    font-weight: 800;
    color: var(--primary);
    background: var(--primary-bg);
    padding: 1px 5px;
    border-radius: 3px;
    letter-spacing: 0.3px;
  }}
  .step-card.highlight .step-num {{
    color: var(--rescue);
    background: #fed7aa;
  }}
  .step-arrow {{
    font-size: 11px;
    color: var(--text-sub);
    font-weight: bold;
  }}
  .step-name {{
    font-size: 12.5px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 4px;
    line-height: 1.3;
  }}
  .step-card.highlight .step-name {{
    color: var(--rescue);
  }}
  .step-text {{
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.35;
    margin-bottom: 8px;
    flex-grow: 1;
  }}
  .step-ref {{
    font-size: 9.5px;
    color: #475569;
    background: #e2e8f0;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: monospace;
    display: block;
    word-break: break-word;
  }}
  .step-card.highlight .step-ref {{
    background: #ffedd5;
    color: #9a3412;
  }}

  /* Bảng mẫu dữ liệu */
  .table-responsive {{
    overflow-x: auto;
  }}
  table.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    text-align: left;
  }}
  table.data-table th {{
    background: #f1f5f9;
    color: var(--text-main);
    font-weight: 600;
    padding: 9px 10px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    font-size: 12.5px;
  }}
  table.data-table td {{
    padding: 10px 10px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  table.data-table tr:hover td {{
    background: #f8fafc;
  }}

  /* Nhãn phân loại */
  .badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
  }}
  .badge-gold {{
    background: var(--gold-bg);
    color: var(--gold);
    border: 1px solid var(--gold-border);
  }}
  .badge-rescue {{
    background: var(--rescue-bg);
    color: var(--rescue);
    border: 1px solid var(--rescue-border);
  }}
  .badge-syl {{
    background: var(--syl-bg);
    color: var(--syl);
    border: 1px solid var(--syl-border);
  }}

  /* Chữ Nôm & Ảnh */
  .nom-char {{
    font-family: "Noto Serif CJK TC", "Songti SC", "SimSun", serif;
    font-size: 20px;
    font-weight: 600;
    color: #0f172a;
    line-height: 1;
    display: inline-block;
  }}
  .crop-img {{
    width: 42px;
    height: 42px;
    border-radius: 4px;
    border: 1px solid #cbd5e1;
    image-rendering: pixelated;
    display: block;
    background: #ffffff;
  }}

  .case-tag {{
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
    margin-bottom: 3px;
  }}
  .method-tag {{
    font-weight: 700;
    color: #0f172a;
    display: block;
    margin-bottom: 3px;
    font-size: 12.5px;
  }}
  .explain-text {{
    font-size: 12px;
    color: #334155;
    display: block;
    line-height: 1.45;
  }}
</style>
</head>
<body>

<div class="container">

  <!-- Tiêu đề -->
  <div class="header">
    <h1>Báo Cáo Tiến Trình Gán Nhãn Tự Động Văn Bản Hán Nôm</h1>
    <p>Hệ thống tự động hóa hoàn toàn quy trình xử lý và gán nhãn dữ liệu từ thư tịch cổ thế kỷ 19, dựa trên suy diễn thuật toán và mô hình học sâu (0 can thiệp thủ công).</p>
  </div>

  <!-- Phần 1: Quy trình 6 bước -->
  <div class="section">
    <div class="section-title">
      <span class="num">1</span>
      <span>Quy Trình Được Thực Thi Tuần Tự (6 Bước Thuần Tự Động)</span>
    </div>

    <div class="flow-grid">
      <div class="step-card">
        <div>
          <div class="step-header">
            <span class="step-num">BƯỚC 1</span>
            <span class="step-arrow">➔</span>
          </div>
          <div class="step-name">Khởi tạo & Tiền kiểm</div>
          <div class="step-text">Nạp tri thức ngôn ngữ, đồng bộ cấu hình và kiểm định môi trường thực thi tự động.</div>
        </div>
        <div class="step-ref">Data: QuocNgu_SinoNom.csv</div>
      </div>

      <div class="step-card">
        <div>
          <div class="step-header">
            <span class="step-num">BƯỚC 2</span>
            <span class="step-arrow">➔</span>
          </div>
          <div class="step-name">Phân đoạn & Trích xuất</div>
          <div class="step-text">Định vị khung trang, bóc tách chính xác 9 cột chữ Nôm từ ảnh thư tịch cổ thế kỷ 19.</div>
        </div>
        <div class="step-ref">Data: 83.239 ô cắt mộc bản</div>
      </div>

      <div class="step-card">
        <div>
          <div class="step-header">
            <span class="step-num">BƯỚC 3</span>
            <span class="step-arrow">➔</span>
          </div>
          <div class="step-name">Gióng hàng đa tầng</div>
          <div class="step-text">Thuật toán Banded-DP gióng ô chữ với âm; đối soát từ điển cấp nhãn chuẩn GOLD đợt 1.</div>
        </div>
        <div class="step-ref">Data: Banded-DP + Từ điển</div>
      </div>

      <div class="step-card">
        <div>
          <div class="step-header">
            <span class="step-num">BƯỚC 4</span>
            <span class="step-arrow">➔</span>
          </div>
          <div class="step-name">Khử trùng & Hiệu chỉnh</div>
          <div class="step-text">Tự động lọc ô cắt đè; kích hoạt bảng tra dị thể sửa chữa lỗi nhầm nét hệ thống (㝵 ➔ 𠊚).</div>
        </div>
        <div class="step-ref">Data: SinoNom_Similar.csv</div>
      </div>

      <div class="step-card highlight">
        <div>
          <div class="step-header">
            <span class="step-num">BƯỚC 5 · ĐIỂM MỚI</span>
            <span class="step-arrow">➔</span>
          </div>
          <div class="step-name">Tái nhận dạng & Giải cứu</div>
          <div class="step-text">Học sâu SE-ResNet tự học nét bút, đối soát kép khôi phục <strong>2.840 ô</strong> lên GOLD.</div>
        </div>
        <div class="step-ref">Model: SE-ResNet (Self-Train)</div>
      </div>

      <div class="step-card">
        <div>
          <div class="step-header">
            <span class="step-num">BƯỚC 6</span>
            <span class="step-arrow">✓</span>
          </div>
          <div class="step-name">Đóng gói & Xuất bản</div>
          <div class="step-text">Xuất 71.610 ảnh ký tự độc lập, bảng nhãn 12 trường thuộc tính và hồ sơ kiểm định.</div>
        </div>
        <div class="step-ref">Output: labels.csv (52.707 GOLD)</div>
      </div>
    </div>
  </div>

  <!-- Phần 2: Minh họa bằng mẫu dữ liệu thật -->
  <div class="section">
    <div class="section-title">
      <span class="num">2</span>
      <span>Minh Họa Mẫu Dữ Liệu Thực Nghiệm & Cơ Chế Xác Thực</span>
    </div>

    <div class="table-responsive">
      <table class="data-table">
        <thead>
          <tr>
            <th style="width: 32px; text-align: center;">STT</th>
            <th>Ảnh mẫu</th>
            <th>Vị trí ô</th>
            <th>Âm đọc</th>
            <th>OCR ban đầu</th>
            <th style="min-width: 340px;">Cơ chế xử lý & Căn cứ xác thực</th>
            <th>Chữ chuẩn</th>
            <th>Mã chữ</th>
            <th>Phân loại</th>
          </tr>
        </thead>
        <tbody>
"""

for s in samples:
    html_content += f"""
          <tr>
            <td style="text-align: center; color: var(--text-sub); font-weight: 600;">{s['stt']}</td>
            <td><img class="crop-img" src="{s['b64']}" alt="{s['syl']}"></td>
            <td style="color: var(--text-sub); font-family: monospace; font-size: 11px;">{s['b']}_{s['p']}<br>c{s['c']:02d}_{s['n']:03d}</td>
            <td><strong style="font-size: 13.5px;">{s['syl']}</strong></td>
            <td><span class="nom-char">{s['s1_char']}</span></td>
            <td>
              <span class="case-tag">Trường hợp: {s['case']}</span>
              <span class="method-tag">🔹 {s['method']}</span>
              <span class="explain-text">{s['explanation']}</span>
            </td>
            <td><span class="nom-char" style="color: {'#047857' if 'chuẩn' in s['tier'] or 'cứu' in s['tier'] else '#0369a1'};">{s['final_char']}</span></td>
            <td style="font-family: monospace; font-size: 11.5px; color: var(--text-sub);">{s['unicode']}</td>
            <td><span class="badge {s['badge_class']}">{s['tier']}</span></td>
          </tr>
"""

html_content += """
        </tbody>
      </table>
    </div>
  </div>

</div>

</body>
</html>
"""

OUT_HTML_DOCS.write_text(html_content, encoding="utf-8")

print(f"✓ Đã cập nhật thành công bản báo cáo khoa học chuẩn hóa tại: {OUT_HTML_DOCS}")
