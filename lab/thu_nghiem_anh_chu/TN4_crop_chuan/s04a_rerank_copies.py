#!/usr/bin/env python
"""s04a_rerank_copies.py — chép r02_candidates.py, r02b_self.py, r03_model.py (kim_bottleneck/rerank) sang rerank_new/,
CHỈ đổi đường dẫn: OUT -> measure_out/_thu_nghiem_anh_chu/TN4/rerank_new, SCR -> scratchpad tuyệt đối, nguồn nguyên mẫu 'oth'
(E_crops của m_ocr) -> E_crops mới (crop chuẩn). Ghi rerank_new/diff.json (số dòng đổi) làm invariant."""
import difflib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ver import BIG, VER  # noqa
SP = '/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad'
SRC = Path(SP) / 'kim_bottleneck/rerank'
BIGO = str(BIG / 'rerank_new')
DST = HERE / f'rerank_{VER}'; DST.mkdir(exist_ok=True)
REP = [('SCR = HERE.parent.parent', f"SCR = Path('{SP}')"),
       ('HERE = Path(__file__).resolve().parent; SCR = HERE.parent.parent', f"HERE = Path(__file__).resolve().parent; SCR = Path('{SP}')"),
       ('OUT = HERE / "out"', f'OUT = Path("{BIGO}")'),
       ('SCR / "gold_img_audit/m_ocr/out/E_crops.f16.npy"', 'OUT / "E_crops_new.f16.npy"')]
D = {}
for f in ('r02_candidates.py', 'r02b_self.py', 'r03_model.py'):
    s = (SRC / f).read_text()
    t = s
    for a, b in REP:
        t = t.replace(a, b)
    (DST / f).write_text(t)
    D[f] = sum(1 for l in difflib.unified_diff(s.splitlines(), t.splitlines(), lineterm='', n=0) if l.startswith('+') and not l.startswith('+++'))
json.dump(D, open(DST / 'diff.json', 'w'), indent=1)
print(D)
