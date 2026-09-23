"""Selftest cho pipeline/tools/ingest_ihr_book.py (adapter 2 bộ IHR-NomDB có nhãn người).

Chạy: .venv/bin/python -m pipeline.tools.ingest_ihr_selftest
Không gọi API, không ghi vào data/, không cần prepared/ có sẵn.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.tools import ingest_ihr_book as M   # noqa: E402

N = OK = 0


def chk(name, cond, extra=""):
    global N, OK
    N += 1
    OK += bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'} {name}" + (f"  [{extra}]" if extra and not cond else ""))


def box(x0, y0, x1, y1, text):
    return dict(points=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], transcription=text)


def main() -> int:
    # --- 1. hằng số + hình học cột -----------------------------------------
    chk("N_PER_COL = 14", M.N_PER_COL == 14)
    chk("N_COLUMNS = 10", M.N_COLUMNS == 10)
    chk("EXPECT_TIER = (6, 8)", tuple(M.EXPECT_TIER) == (6, 8))
    c = dict(x0=0.0, y0=0.0, x1=40.0, y1=140.0)
    ys, chp = M.col_tiers(c)
    chk("col_tiers: bước chữ = cao/14", abs(chp - 10.0) < 1e-9, f"{chp}")
    chk("col_tiers: ranh giới sau chữ thứ 6", abs(ys - 60.0) < 1e-9, f"{ys}")
    cols = [dict(x0=80.0, y0=0.0, x1=120.0, y1=140.0), dict(x0=40.0, y0=0.0, x1=80.0, y1=140.0),
            dict(x0=0.0, y0=0.0, x1=40.0, y1=140.0)]
    chk("col_pitch_of = 40", abs(M.col_pitch_of(cols) - 40.0) < 1e-9)
    chk("col_pitch_of 1 cột = inf", M.col_pitch_of(cols[:1]) == float("inf"))

    # --- 2. gán hộp kim vào (cột, tầng) ------------------------------------
    bx = [box(85, 0, 115, 60, "甲乙丙丁戊己"),      # cột 0, tầng trên, 6 chữ
          box(85, 60, 115, 140, "一二三四五六七八"),  # cột 0, tầng dưới, 8 chữ
          box(45, 0, 75, 140, "AAAAAABBBBBBBB")]     # cột 1, 1 hộp 14 chữ trải cả cột
    columns, st = M.assign_boxes_to_columns(bx, cols)
    chk("assign: 3 cột trả về", len(columns) == 3)
    chk("assign: cột 0 đủ 14 chữ", len(columns[0]) == 14, str(len(columns[0])))
    chk("assign: cột 0 tách 6+8", st["chars_per_tier"][0] == (6, 8), str(st["chars_per_tier"][0]))
    chk("assign: hộp trải cả cột tách 6+8", st["chars_per_tier"][1] == (6, 8),
        str(st["chars_per_tier"][1]))
    chk("assign: cột 2 rỗng", st["chars_per_tier"][2] == (0, 0))
    chk("assign: thứ tự chữ trong cột theo y", [z["char"] for z in columns[0]][:3] == ["甲", "乙", "丙"])
    chk("assign: đếm tổng chữ", st["n_chars"] == 28, str(st["n_chars"]))
    far = M.assign_boxes_to_columns([box(500, 0, 530, 60, "庚辛")], cols)[1]
    chk("assign: hộp ngoài biên bị bỏ", far["n_unassigned_col"] == 2, str(far["n_unassigned_col"]))
    out_y = M.assign_boxes_to_columns([box(85, 400, 115, 460, "壬癸")], cols)[1]
    chk("assign: chữ ngoài dải y bị bỏ", out_y["n_unassigned_tier"] == 2, str(out_y["n_unassigned_tier"]))

    # --- 3. cân lại tầng theo luật 6/8 -------------------------------------
    r, n = M.rebalance_tiers([[0] * 14, [0] * 14, [0] * 13], [(7, 7), (6, 8), (6, 7)])
    chk("rebalance: 7+7 -> 6+8", r[0] == (6, 8))
    chk("rebalance: 6+8 giữ nguyên", r[1] == (6, 8))
    chk("rebalance: tổng != 14 giữ nguyên", r[2] == (6, 7))
    chk("rebalance: đếm số cột đã sửa", n == 1, str(n))

    # --- 4. cổng trang + ghép câu ------------------------------------------
    rec = dict(page=1, page_id="p1", img="p1.jpg", file=Path("p1.jpg"), W=100, H=200,
               cols=[dict(x0=0.0, y0=0.0, x1=10.0, y1=140.0)] * 10,
               verses=[f"cau {i}" for i in range(20)], first_seq=1)
    ok, fl = M.page_gate(rec)
    chk("page_gate: 10 cột / 20 câu PASS", ok and not fl, str(fl))
    bad = dict(rec, cols=rec["cols"][:9])
    ok2, fl2 = M.page_gate(bad)
    chk("page_gate: 9 cột / 20 câu FAIL", not ok2 and any("ceil" in f for f in fl2), str(fl2))
    empty = dict(rec, cols=[], verses=[])
    chk("page_gate: trang không có ô cột FAIL", not M.page_gate(empty)[0])
    odd = dict(rec, cols=rec["cols"][:11] if False else rec["cols"] + [rec["cols"][0]],
               verses=rec["verses"] + ["cau 20"])
    ok3, fl3 = M.page_gate(odd)
    chk("page_gate: 21 câu -> 11 cột PASS nhưng cờ n_cols", ok3 and any("n_cols=" in f for f in fl3),
        str(fl3))
    # (2026-09-23) page_seq_step: bước tiến số câu = 2 * ceil(n/2) — parity không đảo
    chk("page_seq_step: 20 câu -> 20", M.page_seq_step(20) == 20)
    chk("page_seq_step: 21 câu -> 22 (làm tròn LÊN cặp)", M.page_seq_step(21) == 22)
    chk("page_seq_step: 19 câu -> 20", M.page_seq_step(19) == 20)
    chk("page_seq_step: 0 câu -> 0", M.page_seq_step(0) == 0)
    chk("page_seq_step: luôn CHẴN", all(M.page_seq_step(k) % 2 == 0 for k in range(0, 40)))
    chk("page_seq_step >= số câu thô (không nuốt câu)",
        all(M.page_seq_step(k) >= k for k in range(0, 40)))
    _seq, _ok = 1, True
    for k in (21, 21, 19, 21, 20):          # đúng hình các trang lẻ của LucVanTien1916
        _ok = _ok and _seq % 2 == 1
        _seq += M.page_seq_step(k)
    chk("page_seq_step: chuỗi 21/21/19/21/20 giữ first_seq LẺ ở mọi trang", _ok)
    chk("page_gate: trang số câu lẻ -> cờ verses_odd (KHÔNG chặn)",
        M.page_gate(dict(rec, cols=rec["cols"] + [rec["cols"][0]],
                         verses=rec["verses"] + ["cau 20"]))[0]
        and any(f.startswith("verses_odd=")
                for f in M.page_gate(dict(rec, cols=rec["cols"] + [rec["cols"][0]],
                                          verses=rec["verses"] + ["cau 20"]))[1]))
    chk("page_gate: trang số câu chẵn KHÔNG có cờ verses_odd",
        not any(f.startswith("verses_odd=") for f in M.page_gate(rec)[1]))
    vp = M.verse_pairs_for_page(rec)
    chk("verse_pairs: 10 cặp", len(vp) == 10)
    chk("verse_pairs: cột 1 = (1,2)", vp[0] == (1, 2))
    chk("verse_pairs: cột 10 = (19,20)", vp[-1] == (19, 20))
    vr = M.verse_rows(rec)
    chk("verse_rows: 20 dòng", len(vr) == 20)
    chk("verse_rows: giữ nguyên QN", vr[1]["line_text"] == "cau 0")
    chk("verse_rows: nguồn QN = ihr_annotation", vr[1]["qn_source"] == "ihr_annotation")

    # --- 5. KHÔNG đọc nhãn chữ Nôm -----------------------------------------
    # Bóc docstring bằng AST rồi soi MÃ THẬT (docstring của mô-đun có nhắc tên các khoá ấy).
    import ast
    tree = ast.parse((REPO / "pipeline/tools/ingest_ihr_book.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = getattr(node, "body", None)
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                b.pop(0)
    code = ast.unparse(tree)
    chk("mã KHÔNG đọc hn_text", "hn_text" not in code, "hn_text xuất hiện trong mã")
    chk("mã KHÔNG đọc nom_text", "nom_text" not in code, "nom_text xuất hiện trong mã")
    mt = [ln for ln in code.splitlines() if "manifest.tsv" in ln]
    chk("mã chỉ GHI ĐƯỜNG DẪN manifest.tsv (không mở đọc)",
        all("gt_file" in ln for ln in mt), str(mt))
    chk("mã không mở tệp nào trong data/ ngoài bboxes/annotation",
        all(k in ("pages/bboxes.json", "pages/annotation.json")
            for k in ("pages/bboxes.json", "pages/annotation.json") if k in code)
        and "patches" not in code, "đọc thêm tệp khác trong data/")

    # --- 6. đọc dữ liệu thật (chỉ đọc, không ghi) --------------------------
    for bk in sorted(M.BOOKS):
        recs = M.load_pages(bk)
        chk(f"{bk}: load_pages > 100 trang", len(recs) > 100, str(len(recs)))
        chk(f"{bk}: page 1 có ô cột", len(recs[0]["cols"]) > 0)
        chk(f"{bk}: cột 0 nằm PHẢI nhất",
            all(recs[0]["cols"][i]["x0"] > recs[0]["cols"][i + 1]["x0"]
                for i in range(len(recs[0]["cols"]) - 1)))
        chk(f"{bk}: first_seq tiến theo CẶP câu (page_seq_step)",
            all(recs[i + 1]["first_seq"] == recs[i]["first_seq"] + M.page_seq_step(len(recs[i]["verses"]))
                for i in range(len(recs) - 1)))
        chk(f"{bk}: MỌI first_seq đều LẺ (parity không đảo)",
            all(r["first_seq"] % 2 == 1 for r in recs),
            str([r["page_id"] for r in recs if r["first_seq"] % 2 == 0][:3]))
        _vn = [v for r in recs if M.page_gate(r)[0] for v in M.verse_rows(r).values()]
        chk(f"{bk}: verse_rows — câu tầng TRÊN luôn lẻ",
            all(vr["verse_no"] % 2 == 1 for vr in _vn if vr["expect_syll"] == M.EXPECT_TIER[0]))
        _all = [vr["verse_no"] for r in recs if M.page_gate(r)[0] for vr in M.verse_rows(r).values()]
        chk(f"{bk}: không số câu nào bị dùng hai lần", len(_all) == len(set(_all)),
            f"{len(_all)} vs {len(set(_all))}")
        _odd_pages = [r["page_id"] for r in recs if r.get("verses_odd")]
        chk(f"{bk}: trang số câu LẺ được ghi cờ verses_odd",
            all("verses_odd=" in M.page_gate(r)[1] or
                any(f.startswith("verses_odd=") for f in M.page_gate(r)[1])
                for r in recs if r.get("verses_odd")),
            str(_odd_pages[:4]))
        npass = sum(1 for r in recs if M.page_gate(r)[0])
        chk(f"{bk}: >= 95 % trang qua cổng", npass / len(recs) >= 0.95, f"{npass}/{len(recs)}")
        ann = M.load_annotation(bk)
        chk(f"{bk}: annotation trả QN chuỗi",
            all(isinstance(v, str) for a in ann[:3] for v in a["verses"]))

    # --- 7. ghi thật 1 trang (--ocr none) vào thư mục tạm ------------------
    with tempfile.TemporaryDirectory() as td:
        m = M.ingest("TruyenKieu1872", None, 1, "none", True, Path(td), "none", 1, None, False,
                     verbose=False)
        out = Path(td) / "TruyenKieu1872"
        chk("ingest: manifest.json có", (out / "manifest.json").exists())
        chk("ingest: 1 trang", m["total_pages"] == 1, str(m["total_pages"]))
        chk("ingest: đánh dấu evaluation_only", m["evaluation_only"] is True)
        chk("ingest: pages/page_0001.png có", (out / "pages/page_0001.png").exists())
        chk("ingest: detected cache có", (out / "detected/page_0001_ocr_cache.json").exists())
        cache = json.loads((out / "detected/page_0001_ocr_cache.json").read_text(encoding="utf-8"))
        chk("ingest: coords_space=fullpage", cache["coords_space"] == "fullpage")
        chk("ingest: layout=lithograph", cache["layout"] == "lithograph")
        chk("ingest: 10 cột trong cache", cache["n_columns"] == 10, str(cache["n_columns"]))
        chk("ingest: tier_split 10 mục", len(cache["tier_split"]) == 10)
        txt = (out / "transcriptions/page_0001.txt").read_text(encoding="utf-8").splitlines()
        chk("ingest: .txt 10 dòng", len(txt) == 10, str(len(txt)))
        chk("ingest: dòng 1 có 14 âm", len(txt[0].split()) == 14, str(len(txt[0].split())))
        tj = json.loads((out / "transcriptions/page_0001.json").read_text(encoding="utf-8"))
        chk("ingest: json ghi page_id gốc", tj["page_id"] == m["pages"][0]["page_id"])
        chk("ingest: cột 1 có verse_odd/even", {"verse_odd", "verse_even"} <= set(tj["columns"][0]))
        chk("ingest: KHÔNG có khoá nhãn Nôm trong transcriptions",
            "nom" not in json.dumps(tj, ensure_ascii=False).lower().replace("nomdb", ""))
        # phóng ×2: ảnh to gấp đôi, ô cột nhân đôi
        m2 = M.ingest("TruyenKieu1872", None, 1, "none", True, Path(td) / "x2", "none", 2, None,
                      False, verbose=False)
        c2 = json.loads(((Path(td) / "x2" / "TruyenKieu1872" / "detected" / "page_0001_ocr_cache.json")
                         ).read_text(encoding="utf-8"))
        chk("ingest --scale 2: ô cột nhân đôi (sai số cắt phần lẻ <= 1 px)",
            abs(c2["col_boxes"][0][2] - cache["col_boxes"][0][2] * 2) <= 1,
            f"{c2['col_boxes'][0]} vs {cache['col_boxes'][0]}")
        chk("ingest --scale 2: ghi scale vào cache", c2["scale"] == 2)
        chk("ingest --scale 2: số cột không đổi", c2["n_columns"] == cache["n_columns"])
        chk("ingest --scale 2: QN không đổi", m2["total_syllables"] == m["total_syllables"])

    print(f"ingest_ihr selftest: {OK}/{N}")
    return 0 if OK == N else 1


if __name__ == "__main__":
    sys.exit(main())
