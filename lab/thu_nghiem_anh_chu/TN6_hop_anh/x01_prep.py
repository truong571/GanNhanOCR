"""TN6 x01 — chuẩn bị thử nghiệm NGOẠI TUYẾN (nhanh) cho bộ giải mã hộp mới trên Borg (0 API, không mở ảnh bằng LLM).

Đầu vào KHÔNG chứa nhãn người:
  - đơn vị ảnh (hộp detector v1 + ô ảo) + nhúng encoder: measure_out/_borg_human/units (chỉ từ detector + encoder)
  - âm QN của trang + phân cột của adapter: prepared/_auto/<S>/transcriptions (kênh QN của pipeline tự động)
  - chữ kim của cột: prepared/_auto/<S>/detected/*_ocr_cache.json
  - ứng viên R(âm) = Dict/QuocNgu_SinoNom.csv; glyph font NomNaTong/Plangothic + FD (KHÔNG dùng glyph_emb.pkl vì từ vựng
    của nó là chữ NGƯỜI).
Ra: measure_out/_tn6/x01/{seq.pkl, glyph_R.pkl}
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
OUT = REPO / "measure_out/_tn6/x01"; OUT.mkdir(parents=True, exist_ok=True)
from core.text.dictionary import load_qn_to_nom
from core.text.text_utils import normalize_tone_marks
BOOKS = ("SachKinhThayCaBinh", "SachDungLyHoThan")
q2n = load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv"))
seq = {}
voc = set()
for book in BOOKS:
    for f in sorted((REPO / f"prepared/_auto/{book}/transcriptions").glob("page_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        oc = json.loads((REPO / f"prepared/_auto/{book}/detected/{f.stem}_ocr_cache.json").read_text(encoding="utf-8"))
        kcols = oc["columns"]
        items = []
        for ci, c in enumerate(d["columns"]):
            if not (c.get("matched") and c.get("syl_span") and c["syl_span"][0] is not None):
                continue
            j0 = int(c["syl_span"][0])
            kim = [k["char"] for k in kcols[ci]] if ci < len(kcols) else []
            kbox = [k["bbox"] for k in kcols[ci]] if ci < len(kcols) else []
            for si, s in enumerate(c["syllables"]):
                key = normalize_tone_marks(str(s).lower())
                R = q2n.get(key, [])
                voc |= set(R)
                items.append(dict(col=int(c["column"]), si=si, J=j0 + si, syl=s, R=R))
            voc |= set(kim)
        seq[f"{book}/{f.stem}"] = dict(items=items, cols={int(c["column"]): dict(x_range=c.get("x_range"), y_range=c.get("y_range"),
                                    kim=[k["char"] for k in kcols[i]] if i < len(kcols) else [],
                                    kbox=[k["bbox"] for k in kcols[i]] if i < len(kcols) else [])
                                    for i, c in enumerate(d["columns"])})
pickle.dump(seq, open(OUT / "seq.pkl", "wb"))
print("trang", len(seq), "từ vựng R ∪ kim", len(voc))
t0 = time.time()
from pipeline.borg_human.encoders import Glyphs, MultiEnc, device
G = Glyphs(); enc = MultiEnc(["v1", "v2"], device("auto"), True)
voc = sorted(voc)
fonts, fds, fchars, dchars = [], [], [], []
for c in voc:
    im = G.font(c)
    if im is not None:
        fonts.append(im); fchars.append(c)
    im2 = G.fdimg(c)
    if im2 is not None:
        fds.append(im2); dchars.append(c)
Ef = enc.embed(fonts) if fonts else np.zeros((0, 512)); Ed = enc.embed(fds) if fds else np.zeros((0, 512))
pickle.dump(dict(font=dict(zip(fchars, Ef.astype(np.float32))), fd=dict(zip(dchars, Ed.astype(np.float32))), voc=voc),
            open(OUT / "glyph_R.pkl", "wb"))
print(f"glyph: font {len(fchars)} fd {len(dchars)} ({time.time()-t0:.0f} s)")
