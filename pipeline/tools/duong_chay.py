"""duong_chay.py — bảng "ĐƯỜNG CHẠY" in ở ĐẦU mỗi lượt run_pipeline.sh (28/09): pipeline dùng GÌ cho từng bộ.

Đọc (0 API, không ghi gì): config/pipeline*.yaml của bộ (theo `run_config:` như book_profile của run_pipeline.sh) qua
pipeline.align_engine.book_layout (giá trị ĐÃ GIẢI mặc định), config/gold_exact.yaml (profile printed/handwriting, second_read)
và cache lt2 của STT (pipeline.gold_exact.signals_lt2.activation — cùng hàm bước gold_exact dùng để bật/tắt).

    .venv/bin/python -m pipeline.tools.duong_chay --books SachThanhTruyen2 LucVanTien1883 … [--gold-exact on|off]
        [--publish on|off] [--merge on|off]
    .venv/bin/python -m pipeline.tools.duong_chay --selftest

Cột: bộ · config · layout · box_decoder (hộp ảnh) · kim lang_type (1 = Hán, 2 = Nôm) · crop_source (ảnh crop của labels.csv) ·
chon_chu (30/09, bước 4b chọn chữ bằng ảnh — config/chon_chu.yaml: BẬT họ/đòn bẩy/mô hình | TẮT) ·
gold_exact profile (printed | handwriting) · second_read (lt2 STT: BẬT khi đủ cache, TẮT + lý do) · ảnh giao ô ok (crop chuẩn v2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STT = {"SachThanhTruyen2": "stt2", "SachThanhTruyen4": "stt4", "SachThanhTruyen11": "stt11"}
SET8 = {**STT, "Chrestomathie1872": "Chr", "LucVanTien1883": "L83", "KimVanKieu1884": "KVK", "LucVanTien1916": "L16",
        "TruyenKieu1872": "TK", "SachKinhThayCaBinh": "B18", "SachDungLyHoThan": "B34"}
ALL10 = list(SET8)


def book_cfg(book: str):
    """(đường config, dict books[] của sách) — STT: config/pipeline.yaml; sách khác: config/pipeline_<Book>.yaml (+ run_config)."""
    p = REPO / ("config/pipeline.yaml" if book in STT else f"config/pipeline_{book}.yaml")
    if not p.exists():
        raise FileNotFoundError(f"không thấy {p.relative_to(REPO)}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if cfg.get("run_config"):
        p = REPO / str(cfg["run_config"])
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    b = next((x for x in (cfg.get("books") or []) if str(x.get("name", "")).lower() == book.lower()), None)
    if b is None:
        raise KeyError(f"{p.relative_to(REPO)}: books[] không có {book}")
    return p, b


def chon_chu_route(book: str, ccfg: dict | None) -> str:
    """Ô 'chon_chu' của bảng: TẮT | BẬT (họ: đòn bẩy; mô hình) — đọc config/chon_chu.yaml qua pipeline.chon_chu.policy."""
    if ccfg is None:
        return "— (không có config/chon_chu.yaml)"
    from pipeline.chon_chu import policy as CP
    try:
        b = CP.book_cfg(ccfg, book)
    except ValueError as e:
        return f"LỖI config: {e}"
    if b is None:
        return "— (không khai)"
    if not b["enabled"]:
        return "TẮT"
    lv = "L1/L2/L2b/L4+L5" if b["family"] == "hand" else "L1+confusion_fix"
    lv += "".join(f"+{k}" for k in ("qn_geo", "np_geo") if b.get(k))      # luật TN9 sách in (kim_geo)
    return f"BẬT {b['family']}: {lv}; {b['model']}"


def routes(books, gold_exact=True):
    from pipeline.align_engine.book_layout import book_layout
    from pipeline.gold_exact import policy as POL
    from pipeline.gold_exact import signals_lt2 as SL
    from pipeline.gold_exact.common import load_cfg
    gcfg = load_cfg()
    prof = POL.profile_hw(gcfg)
    hw = set(prof["sets"]) if prof else set()
    s8s = [SET8.get(b, b) for b in books]
    try:
        _, lt2, _ = SL.activation([s for s in s8s if s in SL.STT_SETS], gcfg)
    except Exception as e:  # noqa: BLE001
        lt2 = {s: dict(active=False, why=f"lỗi đọc cache lt2: {type(e).__name__}") for s in s8s if s in SL.STT_SETS}
    cc_path = REPO / "config/chon_chu.yaml"
    ccfg = yaml.safe_load(cc_path.read_text(encoding="utf-8")) if cc_path.exists() else None
    rows = []
    for b, s8 in zip(books, s8s):
        try:
            p, bd = book_cfg(b)
            L = book_layout(bd)
            layout, dec, lang, src = L.layout, L.box_decoder, str(L.kim_lang_type), L.crop_source
            cfgs = str(p.relative_to(REPO))
        except Exception as e:  # noqa: BLE001
            layout = dec = lang = src = "?"; cfgs = f"LỖI: {e}"
        pf = "handwriting" if s8 in hw else "printed"
        if not gold_exact:
            sr = "— (gold_exact off)"
        elif s8 in lt2:
            r = lt2[s8]
            sr = (f"BẬT ({r.get('pages_lt2')}/{r.get('pages_lt1')} trang)" if r["active"]
                  else f"TẮT ({r.get('pages_lt2')}/{r.get('pages_lt1')} trang lt2)")
        else:
            sr = "—"
        rows.append((b, cfgs, layout, dec, lang, src, chon_chu_route(b, ccfg), pf if gold_exact else "— (off)", sr))
    return rows, gcfg, prof


def render(rows, gcfg, prof, gold_exact=True, publish=False, merge=True) -> str:
    hdr = ("bộ", "config", "layout", "box_decoder", "kim lang_type", "crop_source", "chon_chu (4b)", "gold_exact profile",
           "second_read (lt2)")
    tb = [hdr] + [tuple(str(x) for x in r) for r in rows]
    w = [max(len(r[i]) for r in tb) for i in range(len(hdr))]
    out = ["ĐƯỜNG CHẠY (config hiện hành; đổi config = đổi đường):"]
    for k, r in enumerate(tb):
        out.append("  | " + " | ".join(r[i].ljust(w[i]) for i in range(len(hdr))) + " |")
        if k == 0:
            out.append("  |" + "|".join("-" * (w[i] + 2) for i in range(len(hdr))) + "|")
    sr = ((gcfg.get("profiles") or {}).get("handwriting") or {}).get("second_read") or {}
    chain = ["các bộ (… cổng -> chon_chu 4b -> export)"] + (["gộp dataset/_ALL"] if merge else [])
    if merge and gold_exact:
        chain.append(f"gold_exact (policy {gcfg.get('version')}; ô ok -> crop chuẩn v2; lt2 {sr.get('mode', 'off') if isinstance(sr, dict) else sr})")
    if merge and gold_exact and publish:
        chain.append("publish dataset/_ALL/cong_bo (tập ảnh = ô ok)")
    chain.append("nghiệm thu (nếu bật)")
    out.append("  chuỗi: " + " -> ".join(chain))
    if prof:
        out.append(f"  profile handwriting = {', '.join(prof['sets'])} (bỏ cổng {prof['drop']}, cổng khe {prof['slot']}); "
                   "bộ khác = printed. second_read chỉ áp STT; TẮT = y hệt không có tín hiệu.")
    return "\n".join(out)


def selftest() -> int:
    ok, fail = 0, []

    def chk(n, c):
        nonlocal ok
        if c:
            ok += 1
        else:
            fail.append(n)
    rows, gcfg, prof = routes(ALL10)
    by = {r[0]: r for r in rows}
    chk("du_10_bo_doc_duoc_config", all("LỖI" not in r[1] for r in rows) and len(rows) == 10)
    chk("stt_kim_lang_type_1", all(by[b][4] == "1" for b in STT))
    chk("sach_moi_kim_lang_type_2", all(by[b][4] == "2" for b in ALL10 if b not in STT))
    chk("profile_handwriting_stt_borg", all(by[b][7] == ("handwriting" if SET8[b] in (prof or {}).get("sets", []) else "printed")
                                            for b in ALL10))
    chk("second_read_chi_stt", all((by[b][8] != "—") == (b in STT) for b in ALL10))
    chk("chon_chu_stt_tat", all(by[b][6] == "TẮT" for b in STT))
    chk("chon_chu_borg_viet_tay_lobo", by["SachKinhThayCaBinh"][6].startswith("BẬT hand") and "hand_B34" in by["SachKinhThayCaBinh"][6]
        and "hand_B18" in by["SachDungLyHoThan"][6])
    chk("chon_chu_sach_in_chi_L1", all(by[b][6].startswith("BẬT print: L1+confusion_fix")
                                        for b in ("Chrestomathie1872", "LucVanTien1883", "KimVanKieu1884", "LucVanTien1916", "TruyenKieu1872")))
    chk("chon_chu_tn9_geo", "+qn_geo+np_geo" in by["KimVanKieu1884"][6] and "+qn_geo" in by["LucVanTien1883"][6]
        and "np_geo" not in by["LucVanTien1883"][6] and "qn_geo" not in by["TruyenKieu1872"][6])
    txt = render(rows, gcfg, prof, publish=True)
    chk("render_co_chuoi", "ĐƯỜNG CHẠY" in txt and "cong_bo" in txt and "gold_exact" in txt)
    print(f"RESULT: {ok} passed, {len(fail)} failed" + (f" {fail}" if fail else ""))
    return 0 if not fail else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.tools.duong_chay")
    ap.add_argument("--books", nargs="*", default=None, help=f"mặc định 10 bộ: {' '.join(ALL10)}")
    ap.add_argument("--gold-exact", choices=("on", "off"), default="on")
    ap.add_argument("--publish", choices=("on", "off"), default="off")
    ap.add_argument("--merge", choices=("on", "off"), default="on")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    books = [b for b in (a.books or ALL10) if b]
    rows, gcfg, prof = routes(books, a.gold_exact == "on")
    print(render(rows, gcfg, prof, a.gold_exact == "on", a.publish == "on", a.merge == "on"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
