"""Dấu vết PUA: nhãn người IHR/NomNaOCR dùng mã riêng (Private Use Area) của phông Nôm Foundation cho chữ chưa có Unicode.
Nếu kim TRẢ RA đúng mã PUA đó ở đúng ô → kim đã học dữ liệu mã hoá theo Nôm Foundation (IHR-NomDB / NomNaOCR …). 0 API.
Đo: (1) trên IHR: số ô nhãn người là PUA, tỉ lệ kim trả ĐÚNG mã đó; (2) mọi bộ: tỉ lệ ký tự PUA trong toàn bộ chữ kim đọc ra.
"""
import json, sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]

def is_pua(ch):
    if not isinstance(ch, str) or len(ch) != 1:
        return False
    o = ord(ch)
    return 0xE000 <= o <= 0xF8FF or 0xF0000 <= o <= 0x10FFFF

out = {}
for b in ("LucVanTien1916", "TruyenKieu1872"):
    C = pd.read_csv(REPO / f"measure_out/{b}/ihr_endtoend/cells.csv", dtype=str, keep_default_na=False)
    g = C[C["gt"].map(is_pua)]
    out[b] = dict(o_nhan_nguoi_PUA=len(g), kim_tra_dung_ma_PUA=int((g.ocr_char == g["gt"]).sum()),
                  kim_tra_PUA_bat_ky=int(g.ocr_char.map(is_pua).sum()),
                  ma_PUA_khac_nhau_nhan=int(g["gt"].nunique()))
src = {"SachThanhTruyen(STT)": REPO / "dataset_out/labels_final.csv"}
for b in ("LucVanTien1883", "KimVanKieu1884", "Chrestomathie1872", "LucVanTien1916", "TruyenKieu1872"):
    src[b] = REPO / f"prepared/{b}/dataset_out/labels_gated.csv"
for b in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
    src[b] = REPO / f"prepared/_auto/{b}/dataset_out/labels_gated.csv"
for b, f in src.items():
    D = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["ocr_char"])
    k = D.ocr_char[D.ocr_char != ""]
    out.setdefault(b, {})["ty_le_chu_kim_la_PUA_%"] = round(k.map(is_pua).mean() * 100, 3)
    out[b]["n_chu_kim"] = int(len(k))
Path(__file__).with_name("kim_pua_signature.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
