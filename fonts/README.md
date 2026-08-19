# Font dự phòng cho glyph tham chiếu

Thư mục này KHÔNG được track (xem `.gitignore`) — các file .ttf ở đây là tài nguyên bên
thứ ba, ~33 MB, tải lại được bằng một lệnh. Chỉ `README.md` này nằm trong git.

## Vì sao cần

`PIL` không báo lỗi khi font thiếu glyph: nó vẽ `.notdef` (ô rỗng) và trả về một ảnh
trông y như glyph thật. Thẻ audit khi đó hiện ô trắng dưới nhãn "GLYPH THAM CHIẾU" và
người chấm so nét chữ trong ảnh scan với... không gì cả.

Đo trên corpus hiện tại (4.696 chữ khác nhau trong `labels_final.csv`):

| chuỗi font | chữ thiếu glyph | số ô ảnh hưởng |
|---|---:|---:|
| `NomNaTong-Regular.ttf` một mình | 568 | **3.326** |
| + Han-nom Minh + HanaMin A/B | 6 | 26 |
| + **Plangothic P1 + P2** | **0** | **0** |

`pipeline/ground_truth/audit_grid.py` tra `cmap` TRƯỚC khi vẽ và đi qua chuỗi font; font
nào vắng mặt thì bỏ qua, không có glyph ở font nào thì thẻ **bỏ hẳn** ô glyph tham chiếu
— thà không có còn hơn có mà sai.

## Tải lại

```bash
cd fonts
for f in PlangothicP1-Regular.ttf PlangothicP2-Regular.ttf; do
  curl -sL -o "$f" \
    "https://github.com/Fitzgerald-Porthmouth-Koenigsegg/Plangothic_Project/releases/download/V2.9.5795/$f"
done
```

Phiên bản đã dùng: **V2.9.5795** (Plangothic Project, giấy phép OFL). Bản mới hơn cũng
được — chuỗi font chỉ cần đúng TÊN FILE, không ghim phiên bản.

## Đừng bỏ font vào `font_diffusion/fonts/`

Thư mục đó thuộc **submodule** `font_diffusion`; thả file vào sẽ làm bẩn cây làm việc của
submodule. `build_font_chain` tìm ở CẢ HAI nơi, nên để font tải thêm ở đây là đúng chỗ.

## Kiểm lại độ phủ

```bash
.venv/bin/python - <<'PY'
import pandas as pd, collections
from pathlib import Path
from pipeline.ground_truth import audit_grid
from pipeline.ground_truth.cli import _load_config, _paths
chain = audit_grid.build_font_chain(_paths(_load_config(Path('config/pipeline.yaml')))['font'])
d = pd.read_csv('dataset_out/labels_final.csv', dtype=str, low_memory=False)
ch = collections.Counter(list(d.ocr_char.dropna()) + list(d.label.dropna()))
miss = [c for c in ch if not any(ord(c[0]) in audit_grid._cmap_of(f) for f in chain)]
print([f.name for f in chain])
print(f'{len(ch)} chữ · THIẾU {len(miss)} chữ / {sum(ch[c] for c in miss)} ô')
PY
```
