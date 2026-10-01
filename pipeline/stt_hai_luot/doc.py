"""doc.py — cache kim "l1skel_l2" cho STT (TN9, 01/10): KHUNG lt1 + CHỮ lt2 gióng theo chuỗi. 0 API, tất định.

Nguồn (chỉ ĐỌC): prepared/<Sách>/detected/<trang>_ocr_cache.json (kim lang_type 1 = Hán, lượt 1, đủ số chữ) và
prepared/<Sách>/kim_raw_lt2/<trang>_lt2.json (kim lang_type 2 = Nôm, lượt 2; pipeline/tools/stt_reocr_lt2.py).
Ghi: prepared/<Sách>/kim_l1skel_l2/<trang>_ocr_cache.json — CÙNG lược đồ cache lt1, đọc bởi align_production._detect khi
books[].kim_read = l1skel_l2.

Phép ghép (bản chép NGUYÊN lab/thu_nghiem_kim/TN9_stt_in/t02_build_stt.py::merged_cache — con số TN9 đo trên nó):
  với mỗi dòng kim lt1 (hộp dòng + chuỗi), dòng lt2 có IoU hộp dòng ≥ 0,5 (best_line), căn Needleman–Wunsch có dải
  (khớp V1+ +2, lệch −1, khoảng −1; = scripts/measure/stt_lt2_eval.align = pipeline.gold_exact.signals_lt2.align);
  vị trí lt1 gióng được -> chữ lt2, không gióng -> giữ chữ lt1. HÌNH HỌC (số dòng, số chữ, hộp chia đều, cột) GIỮ NGUYÊN
  lt1 từng điểm: cột = cột cache lt1, chỉ thay `char` theo thứ tự boxes_to_columns của khung mới (kiểm hình dạng trùng
  khớp; lệch -> giữ nguyên lt1, đếm `fallback`).
Thêm vào mỗi chữ của cột: `lt1` (chữ lượt 1 tại vị trí) và `lt2` (chữ lượt 2 gióng được, '' nếu không) — để bước gold_exact
(signals_lt2: định vị trong dòng lt1 bằng chữ lt1) và cổng hai lượt truy được hai lần đọc.
KHÔNG dùng lab/…/tn7_kim_tham_so.merge_two_pass (ghép HÌNH HỌC): TN9 §1 đo nó kém (đổi 54–58 % vị trí, khe DP ×2–4).
Cache bất biến theo nội dung: `src_sha` = sha256(bytes lt1 ‖ bytes lt2 ‖ MERGE_VERSION) — trùng thì không ghi lại.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
L1L2_DIR = "kim_l1skel_l2"
LT1_DIR = "detected"
LT2_DIR = "kim_raw_lt2"
MERGE_VERSION = "tn9_l1skel_l2_v1"
SENTINEL = "\x00"            # đánh dấu vị trí KHÔNG gióng được trong khung song song (không phải khoảng trắng)

_LEX = {}


def _lexicon():
    """V1+ (gold_exact.common.var_eq_plus) cần bảng biến thể — nạp một lần (như stt_lt2_eval.lexicon)."""
    if not _LEX:
        from pipeline.gold_exact import common as C
        C.set_lexicon(C.Assets(verify=True))
        _LEX["ok"] = True


def _sl():
    _lexicon()
    from pipeline.gold_exact import signals_lt2 as SL
    return SL


def valid_lt2(c1: dict, c2: dict | None) -> bool:
    """lt2 dùng được: cùng ảnh (image_hash) với lt1 và toạ độ fullpage (như t02 + signals_lt2.load_pages)."""
    return bool(c2) and c2.get("image_hash") == c1.get("image_hash") and c2.get("coords_space") == "fullpage"


def merged_cache(c1: dict, c2: dict | None) -> tuple[dict, dict]:
    """Khung lt1 + chữ lt2 (vị trí gióng được). Trả (cache mới, thống kê)."""
    from core.ocr.ocr_api import boxes_to_columns
    st = dict(lines=0, lines_matched=0, pos=0, pos_l2=0, pos_changed=0, fallback=0, no_lt2=0)
    out = copy.deepcopy(c1)
    for col in out.get("columns") or []:
        for x in col:
            x["lt1"] = x.get("char", ""); x["lt2"] = ""
    out["kim_read"] = "l1skel_l2"; out["merge_version"] = MERGE_VERSION
    if not valid_lt2(c1, c2):
        st["no_lt2"] = 1
        return out, st
    SL = _sl()
    R1, R2 = SL.raw_lines(c1.get("boxes_raw")), SL.raw_lines(c2.get("boxes_raw"))
    new_raw = copy.deepcopy(c1.get("boxes_raw") or [])
    par_raw = copy.deepcopy(c1.get("boxes_raw") or [])          # khung song song: chữ lt2 hoặc SENTINEL
    valid_idx = [i for i, b in enumerate(c1.get("boxes_raw") or [])
                 if [ch for ch in (b.get("transcription") or "").strip() if ch.strip()]]
    for li, L in enumerate(R1):
        st["lines"] += 1
        st["pos"] += len(L["s"])
        s, par = list(L["s"]), [SENTINEL] * len(L["s"])
        oi = SL.best_line(L, R2)
        if oi is not None:
            st["lines_matched"] += 1
            m = SL.align(L["s"], R2[oi]["s"])
            for k, j in enumerate(m):
                if j is not None:
                    st["pos_l2"] += 1
                    if R2[oi]["s"][j] != s[k]:
                        st["pos_changed"] += 1
                    s[k] = R2[oi]["s"][j]
                    par[k] = R2[oi]["s"][j]
        new_raw[valid_idx[li]]["transcription"] = "".join(s)
        par_raw[valid_idx[li]]["transcription"] = "".join(par)
    nc = boxes_to_columns(new_raw)
    pc = boxes_to_columns(par_raw)
    oc = boxes_to_columns(c1.get("boxes_raw"))
    shape = [len(c) for c in nc]
    if (shape != [len(c) for c in oc] or shape != [len(c) for c in c1["columns"]] or shape != [len(c) for c in pc]
            or [[x["char"] for x in c] for c in oc] != [[x["char"] for x in c] for c in c1["columns"]]):
        st["fallback"] = 1
        return out, st
    for col_out, col_new, col_par in zip(out["columns"], nc, pc):
        for x, y, z in zip(col_out, col_new, col_par):
            x["char"] = y["char"]
            x["lt2"] = "" if z["char"] == SENTINEL else z["char"]
    out["boxes_raw"] = new_raw
    return out, st


def src_sha(f1: Path, f2: Path | None) -> str:
    h = hashlib.sha256()
    h.update(Path(f1).read_bytes())
    h.update(b"\x01")
    if f2 is not None and Path(f2).exists():
        h.update(Path(f2).read_bytes())
    h.update(MERGE_VERSION.encode())
    return h.hexdigest()


def build_book(book_dir: Path, force: bool = False, log=print) -> dict:
    """Dựng/cập nhật cache l1skel_l2 của MỘT sách (prepared/<Sách>). Trả thống kê (trang ghi / bỏ qua / thiếu lt2 / fallback)."""
    book_dir = Path(book_dir)
    od = book_dir / L1L2_DIR
    od.mkdir(parents=True, exist_ok=True)
    agg = dict(pages=0, written=0, unchanged=0, no_lt2=0, fallback=0, pos=0, pos_l2=0, pos_changed=0)
    for f1 in sorted((book_dir / LT1_DIR).glob("page_*_ocr_cache.json")):
        page = f1.name[: -len("_ocr_cache.json")]
        f2 = book_dir / LT2_DIR / f"{page}_lt2.json"
        fo = od / f1.name
        sha = src_sha(f1, f2 if f2.exists() else None)
        agg["pages"] += 1
        if not force and fo.exists():
            try:
                if json.loads(fo.read_text(encoding="utf-8")).get("src_sha") == sha:
                    agg["unchanged"] += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
        c1 = json.loads(f1.read_text(encoding="utf-8"))
        c2 = json.loads(f2.read_text(encoding="utf-8")) if f2.exists() else None
        c, st = merged_cache(c1, c2)
        c["src_sha"] = sha
        for k in ("no_lt2", "fallback", "pos", "pos_l2", "pos_changed"):
            agg[k] += st[k]
        tmp = fo.with_suffix(".tmp")
        tmp.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
        tmp.replace(fo)
        agg["written"] += 1
    log(f"[hai_luot] cache {book_dir.name}/{L1L2_DIR}: {agg}")
    return agg
