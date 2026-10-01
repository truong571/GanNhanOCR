"""Thước đo độ khó chung cho 10 bộ (0 API): tỉ lệ ô mà chữ kim ∈ R(âm Quốc ngữ của ô) (từ điển pipeline) trên bảng mọi tầng
của bản dựng hiện tại, cộng tỉ lệ ô có âm hợp lệ. Bộ có nhãn người ghép thêm độ đúng kim thật (summary.json)."""
import json, sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.tools.ingest_lithograph_book import _nom_to_qn_readings  # noqa: E402
n2q = _nom_to_qn_readings()
src = {}
st = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False, usecols=["book", "ocr_char", "syllable", "rule"])
for k in ("stt2", "stt4", "stt11"):
    src[k] = st[st.book == k]
for b in ("LucVanTien1883", "KimVanKieu1884", "Chrestomathie1872", "LucVanTien1916", "TruyenKieu1872"):
    src[b] = pd.read_csv(REPO / f"prepared/{b}/dataset_out/labels_gated.csv", dtype=str, keep_default_na=False, usecols=["ocr_char", "syllable", "rule"])
for b in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
    src[b] = pd.read_csv(REPO / f"prepared/_auto/{b}/dataset_out/labels_gated.csv", dtype=str, keep_default_na=False, usecols=["ocr_char", "syllable", "rule"])
out = {}
for b, D in src.items():
    has = (D.syllable != "") & (D.ocr_char != "")
    inr = [s in n2q.get(c, ()) for c, s in zip(D.ocr_char[has], D.syllable[has])]
    out[b] = dict(n_o=len(D), kim_khop_am_pct=round(100 * sum(inr) / max(1, len(inr)), 1),
                  am_khong_hop_le_pct=round(100 * (D.rule.str.contains("not_plausible")).mean(), 1))
Path(__file__).with_name("do_kho_10_bo.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for b, v in out.items():
    print(f"{b:20s} {v}")
