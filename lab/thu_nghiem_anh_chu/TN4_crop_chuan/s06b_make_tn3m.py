#!/usr/bin/env python
"""s06b_make_tn3m.py — sinh s06c_tn3m.py = BẢN CHÉP t02_matrix.py của TN3 (bộ ước lượng độ chính xác hai vế cho 8 bộ), CHỈ đổi:
(1) nơi ghi (lab/.../out_<ver>/tn3m_*, measure_out/.../TN4/<ver>/tn3m/); (2) thay mặt nạ H1..Rv3 bằng mặt nạ crop chuẩn
(masks_new.pkl của s06a) và khe IHR bằng khe của crop mới. Mọi đầu vào ước lượng khác (gate_v2, dị bản, bể STT...) GIỮ NGUYÊN.
Ghi s06c_diff.json (các dòng đổi) làm invariant."""
import difflib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'TN3_tat_ca_bo/t02_matrix.py'
s = SRC.read_text()
t = s
REP = [
 ("BIG = REPO / 'measure_out/_thu_nghiem_anh_chu/TN3'; BIG.mkdir(parents=True, exist_ok=True)",
  "BIG = REPO / 'measure_out/_thu_nghiem_anh_chu/TN3'\nsys.path.insert(0, str(HERE))\nfrom ver import BIG as TN4BIG, LABOUT, VER  # noqa\nBIGO = TN4BIG / 'tn3m'; BIGO.mkdir(parents=True, exist_ok=True)"),
 ("M['H1g7'] = ~(A0 | B0 | CNT | BC_L)          # độ nhạy: G7 nguyên văn (trên L16 là chọn trong mẫu)",
  "M['H1g7'] = ~(A0 | B0 | CNT | BC_L)          # độ nhạy: G7 nguyên văn (trên L16 là chọn trong mẫu)\n"
  "# === TN4: mặt nạ crop chuẩn (s06a)\n"
  "_MN = pd.read_pickle(TN4BIG / 'masks_new.pkl'); assert (_MN.cell_uid.values == X.cell_uid.values).all()\n"
  "import os as _os\n_SIMGC = _os.environ.get('TN4_SIMG', 'SIMGn'); _SUFM = '' if _SIMGC == 'SIMGn' else '_lai'\n"
  "H1 = _MN.H1n.values; SIMG = _MN[_SIMGC].values\n"
  "M['H1'] = H1; M['H2'] = H1 & SIMG; M['H3'] = H1 & TA_OK; M['H4'] = H1 & SIMG & TA_OK; M['H5'] = H1 | FIXOK\n"
  "M['H6'] = M['H4'] | (FIXOK & TA_OK); M['Rv3'] = M['H4'] & ~(AINT | MOCR); M['H6nt'] = M['H2'] | FIXOK\n"
  "M['Rv3nt'] = M['H2'] & ~(AINT | MOCR); M['Rv3nc'] = M['H3'] & ~(AINT | MOCR); M['H1g7'] = H1"),
 ("slot = X.slot_ok.fillna('').astype(str).values", "slot = _MN.slot_new.astype(str).values   # TN4: khe của crop mới"),
 ("MX.to_csv(HERE / 'matrix.csv', index=False)", "MX.to_csv(LABOUT / f'tn3m_matrix{_SUFM}.csv', index=False)"),
 ("np.save(BIG / 'boot_img_sums.npy', BI)", "np.save(BIGO / 'boot_img_sums.npy', BI)"),
 ("CM.to_pickle(BIG / 'cells_masks.pkl')", "CM.to_pickle(BIGO / 'cells_masks.pkl')"),
 ("json.load(open(HERE / 't01_summary.json'))['theta_used']", "json.load(open(HERE.parent / 'TN3_tat_ca_bo/t01_summary.json'))['theta_used']"),
 ("json.dump(SUM, open(HERE / 't02_summary.json', 'w')", "json.dump(SUM, open(LABOUT / f'tn3m_summary{_SUFM}.json', 'w')"),
]
miss = []
for a, b in REP:
    if a not in t:
        miss.append(a[:60])
    t = t.replace(a, b)
assert not miss, miss
(HERE / 's06c_tn3m.py').write_text(t)
D = [l for l in difflib.unified_diff(s.splitlines(), t.splitlines(), lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
json.dump(dict(n_changed=len(D), lines=D), open(HERE / 's06c_diff.json', 'w'), ensure_ascii=False, indent=1)
print(len(D))
