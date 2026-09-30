"""TN7 (30/09) — kim đang được gọi đúng tham số chưa? Thử font_type "viết tay" / "tự động" + phóng ảnh trên sách CHÉP TAY.

Phát hiện: mọi bộ gọi kim với font_type = 1 (IN), kể cả STT ×3 và Borg ×2 là chữ VIẾT TAY; STT còn dùng lang_type = 1 (Hán).
Borg chỉ rộng 720 px. Thử nghiệm này đo xem đổi tham số có làm kim đọc đúng hơn không — TRƯỚC khi đổi pipeline.

Biến thể (mỗi trang mẫu = 1 lượt upload + recognize):
  Borg (ảnh JPG gốc như adapter ingest_borg_book; đường hiện tại = lt2 + font 1, cache prepared/_auto/<S>/kim_raw):
    lt2_viettay      lang_type 2, font_type 2
    lt2_tudong       lang_type 2, font_type 0
    lt2_viettay_x2   ảnh phóng ×2 (LANCZOS) + lang_type 2, font_type 2
  STT (ảnh cắt khung pad 12 như ocr_page; đường hiện tại = lt1 + font 1 (detected/*_ocr_cache.json), lần đọc thứ hai lt2 + font 1
  (kim_raw_lt2/)):
    lt2_viettay      lang_type 2, font_type 2
    lt1_viettay      lang_type 1, font_type 2

Chấm (0 API, `--eval`) trên CÙNG các trang, so với đường hiện tại:
  * Borg — SỰ THẬT: chuỗi chữ kim cả trang căn LCS với chữ Nôm NGƯỜI phiên (V1+ = trùng hoặc dị thể) -> precision (chữ kim
    đúng) / recall (chữ người được đọc ra). Cùng phép đo với scripts/measure/borg_endtoend_eval.py (kim_block).
  * STT + Borg — GIÁN TIẾP (không cần nhãn người): LCS chữ kim với âm Quốc ngữ của trang, khớp khi chữ ∈ R(âm) của
    Dict/QuocNgu_SinoNom.csv -> "hợp âm" (precision) / "phủ âm" (recall, thấp = đọc sót). Borg có cả hai thước -> kiểm được
    thước gián tiếp có xếp hạng biến thể giống sự thật không, rồi mới dùng nó cho STT.
  CI 95 % bootstrap theo trang cho CHÊNH LỆCH (biến thể − hiện tại), ghép cặp trên cùng trang.

Lệnh:
  .venv/bin/python lab/thu_nghiem_kim/tn7_kim_tham_so.py --dry-run      # in kế hoạch, 0 API
  .venv/bin/python lab/thu_nghiem_kim/tn7_kim_tham_so.py --run          # gọi API (≈180 trang, ≈20 phút); chạy lại = bỏ qua trang đã có
  .venv/bin/python lab/thu_nghiem_kim/tn7_kim_tham_so.py --status
  .venv/bin/python lab/thu_nghiem_kim/tn7_kim_tham_so.py --eval         # 0 API -> measure_out/_tn7_kim/KET_QUA.md + summary.json

KHÔNG đụng cache OCR của pipeline (chỉ ĐỌC); ghi riêng measure_out/_tn7_kim/<bộ>/<biến thể>/<trang>.json + calls.jsonl.
Dừng ngay khi tài khoản lỗi / chế độ khách / 3 trang lỗi liên tiếp.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "measure"))

OUT = REPO / "measure_out" / "_tn7_kim"
LEDGER = OUT / "calls.jsonl"
SEED = 20260930
BORG = {"SachKinhThayCaBinh": 30, "SachDungLyHoThan": 10}          # số trang mẫu
STT = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}
STT_N = 10                                                           # trang mẫu mỗi quyển
BORG_VARIANTS = {"lt2_viettay": dict(lang_type=2, font_type=2, scale=1),
                 "lt2_tudong": dict(lang_type=2, font_type=0, scale=1),
                 "lt2_viettay_x2": dict(lang_type=2, font_type=2, scale=2)}
STT_VARIANTS = {"lt2_viettay": dict(lang_type=2, font_type=2, scale=1),
                "lt1_viettay": dict(lang_type=1, font_type=2, scale=1)}
FRAME_PAD = 12
MAX_CONSEC_FAIL = 3


# ------------------------------------------------------------------------------------------------ mẫu trang
def borg_pages(book: str, n: int) -> list[dict]:
    """Trang có ảnh + chữ người + cache kim hiện tại; lấy đều theo thứ tự trang (tất định)."""
    from borg_endtoend_eval import human_pages, kim_page_chars
    from pipeline.tools.ingest_borg_book import load_pages
    H = human_pages(book)
    recs = {r["page_name"]: r for r in load_pages(book)}
    ok = [p for p in sorted(H) if H[p]["exists"] and p in recs and kim_page_chars(book, p) is not None
          and sum(len(s["nom"]) for s in H[p]["sents"]) >= 20]
    step = max(1, len(ok) // n)
    pick = ok[::step][:n]
    return [dict(book=book, page=p, image=str(recs[p]["file"])) for p in pick]


def stt_pages(key: str, n: int) -> list[dict]:
    from pipeline.tools.stt_reocr_lt2 import out_path
    d = REPO / "prepared" / STT[key] / "detected"
    cands = []
    for cf in sorted(d.glob("*_ocr_cache.json")):
        page = cf.name[: -len("_ocr_cache.json")]
        if not out_path(key, page).exists() or not (REPO / "prepared" / STT[key] / "transcriptions" / f"{page}.json").exists():
            continue
        cands.append((page, cf))
    rnd = random.Random(f"{SEED}/{key}")
    pick = sorted(rnd.sample(cands, min(n, len(cands))))
    out = []
    for page, cf in pick:
        lt1 = json.loads(cf.read_text(encoding="utf-8"))
        img, why = _resolve(lt1)
        if img is None:
            print(f"  [bỏ] {key} {page}: {why}")
            continue
        out.append(dict(book=key, page=page, image=str(img)))
    return out


def _resolve(lt1: dict):
    from pipeline.tools.stt_reocr_lt2 import md5_file
    img = Path(lt1.get("image", ""))
    img = img if img.is_absolute() else REPO / img
    if not img.exists():
        return None, f"thiếu ảnh {img}"
    if lt1.get("image_hash") and md5_file(img) != lt1["image_hash"]:
        return None, "ảnh đã đổi so với lúc OCR lt1"
    return img, "ok"


def plan() -> list[dict]:
    jobs = []
    for b, n in BORG.items():
        for p in borg_pages(b, n):
            for v, prm in BORG_VARIANTS.items():
                jobs.append(dict(p, kind="borg", variant=v, **prm))
    for k in STT:
        for p in stt_pages(k, STT_N):
            for v, prm in STT_VARIANTS.items():
                jobs.append(dict(p, kind="stt", variant=v, **prm))
    return jobs


def out_file(j: dict) -> Path:
    return OUT / j["book"] / j["variant"] / f"{j['page']}.json"


# ------------------------------------------------------------------------------------------------ gọi API
def prepare_image(j: dict) -> tuple[str, str | None]:
    """-> (đường ảnh gửi, tệp tạm cần xoá). Borg: JPG gốc (×scale). STT: cắt khung pad 12 như ocr_page."""
    if j["kind"] == "stt":
        import cv2
        from core.image.frame_detector import crop_to_frame
        bgr = cv2.imread(j["image"])
        crop = crop_to_frame(bgr, pad=FRAME_PAD)
        t = tempfile.NamedTemporaryFile(suffix=".png", delete=False); t.close()
        cv2.imwrite(t.name, crop)
        return t.name, t.name
    if j["scale"] == 1:
        return j["image"], None
    from PIL import Image
    im = Image.open(j["image"]).convert("RGB")
    im = im.resize((im.width * j["scale"], im.height * j["scale"]), Image.LANCZOS)
    t = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False); t.close()
    im.save(t.name, quality=95)
    return t.name, t.name


def run(budget: int | None) -> int:
    from core.ocr import ocr_api as A
    jobs = [j for j in plan() if not out_file(j).exists()]
    if budget is not None:
        jobs = jobs[:budget]
    print(f"[tn7] cần gọi {len(jobs)} trang (đã có bỏ qua) · ước ≈ {len(jobs) * 6.5 / 60:.0f} phút")
    OUT.mkdir(parents=True, exist_ok=True)
    fails, done, t0 = 0, 0, time.time()
    for i, j in enumerate(jobs, 1):
        path, tmp = prepare_image(j)
        err = ""
        try:
            fname = A.upload_image(path)
            boxes = A.recognize(fname, lang_type=j["lang_type"], font_type=j["font_type"]) if fname else None
        except Exception as e:  # noqa: BLE001
            boxes, err = None, f"{type(e).__name__}: {e}"
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        guest = bool(getattr(A, "is_guest_mode", lambda: False)())
        rec = dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), book=j["book"], page=j["page"], variant=j["variant"],
                   ok=boxes is not None and not guest, err=err or ("guest_mode" if guest else ""))
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if guest:
            print("[tn7] DỪNG: API chuyển chế độ khách (tài khoản/token lỗi) — kiểm tra .env rồi chạy lại lệnh --run.")
            return 2
        if boxes is None:
            fails += 1
            print(f"  [{i}/{len(jobs)}] LỖI {j['book']} {j['page']} {j['variant']} {err}")
            if fails >= MAX_CONSEC_FAIL:
                print(f"[tn7] DỪNG: {MAX_CONSEC_FAIL} trang lỗi liên tiếp — chạy lại --run sau (trang đã xong được giữ).")
                return 3
            continue
        fails = 0
        of = out_file(j); of.parent.mkdir(parents=True, exist_ok=True)
        tmpf = of.with_suffix(".tmp")
        tmpf.write_text(json.dumps(dict(book=j["book"], page=j["page"], variant=j["variant"], image=j["image"],
                                        kim_params=dict(lang_type=j["lang_type"], font_type=j["font_type"],
                                                        ocr_id=1, reading_direction=1),
                                        scale=j["scale"], framed=j["kind"] == "stt", boxes=boxes),
                                   ensure_ascii=False), encoding="utf-8")
        tmpf.replace(of)
        done += 1
        if i % 10 == 0 or i == len(jobs):
            el = time.time() - t0
            print(f"  [{i}/{len(jobs)}] xong {done} · {el / 60:.1f} phút · còn ≈ {(len(jobs) - i) * el / i / 60:.0f} phút")
    print(f"[tn7] XONG {done} trang. Chấm: .venv/bin/python lab/thu_nghiem_kim/tn7_kim_tham_so.py --eval")
    return 0


def status() -> int:
    jobs = plan()
    have = defaultdict(lambda: [0, 0])
    for j in jobs:
        k = (j["book"], j["variant"])
        have[k][1] += 1
        have[k][0] += out_file(j).exists()
    for (b, v), (h, n) in sorted(have.items()):
        print(f"  {b:20s} {v:16s} {h:3d}/{n}")
    print(f"  tổng {sum(h for h, _ in have.values())}/{len(jobs)}")
    return 0


# ------------------------------------------------------------------------------------------------ chấm (0 API)
def chars_of_boxes(boxes: list[dict]) -> list[str]:
    from pipeline.tools.ingest_borg_book import cluster_columns
    cols, _ = cluster_columns(boxes or [])
    return [c["char"] for col in cols for c in col]


def stt_baseline_chars(key: str, page: str, which: str) -> list[str] | None:
    if which == "hien_tai_lt1_in":
        f = REPO / "prepared" / STT[key] / "detected" / f"{page}_ocr_cache.json"
    else:
        from pipeline.tools.stt_reocr_lt2 import out_path
        f = out_path(key, page)
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    return chars_of_boxes(d.get("boxes_raw") or [])


def stt_syllables(key: str, page: str) -> list[str]:
    import unicodedata
    d = json.loads((REPO / "prepared" / STT[key] / "transcriptions" / f"{page}.json").read_text(encoding="utf-8"))
    return [unicodedata.normalize("NFC", s.lower()) for c in d.get("columns", []) for s in (c.get("syllables") or [])]


def evaluate() -> int:
    import numpy as np
    from borg_endtoend_eval import Eq, human_pages, kim_page_chars, levenshtein_pairs, page_syllables
    from pipeline.tools.ingest_lithograph_book import _nom_to_qn_readings
    E, n2q = Eq(), _nom_to_qn_readings()

    def hop_am(ch, syl):
        return bool(ch) and syl in n2q.get(ch, ())

    def lcs(a, b, eq):
        return len(levenshtein_pairs(a, b, eq)) if a and b else 0

    rows = []      # 1 dòng / (bộ, biến thể, trang)
    jobs = plan()
    pages_by = defaultdict(set)
    for j in jobs:
        pages_by[(j["kind"], j["book"])].add(j["page"])
    Hc = {}
    for (kind, book), pages in sorted(pages_by.items()):
        if kind == "borg":
            H = Hc.setdefault(book, human_pages(book))
            variants = ["hien_tai_lt2_in"] + list(BORG_VARIANTS)
        else:
            variants = ["hien_tai_lt1_in", "hien_tai_lt2_in"] + list(STT_VARIANTS)
        for pg in sorted(pages):
            for v in variants:
                if v.startswith("hien_tai"):
                    kc = kim_page_chars(book, pg) if kind == "borg" else stt_baseline_chars(book, pg, v)
                else:
                    f = OUT / book / v / f"{pg}.json"
                    kc = chars_of_boxes(json.loads(f.read_text(encoding="utf-8"))["boxes"]) if f.exists() else None
                if kc is None:
                    continue
                syl = page_syllables(H[pg]) if kind == "borg" else stt_syllables(book, pg)
                r = dict(kind=kind, book=book, page=pg, variant=v, n_kim=len(kc), n_syl=len(syl),
                         m_am=lcs(kc, syl, hop_am))
                if kind == "borg":
                    hc = [c for s in H[pg]["sents"] for c in s["nom"]]
                    r.update(n_human=len(hc), m_v1p=lcs(kc, hc, E.v1p))
                rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def agg(rs, num, den):
        return sum(r[num] for r in rs) / max(1, sum(r[den] for r in rs))

    def boot_diff(a_rows, b_rows, num, den, B=2000):
        """CI 95 % của (tỉ lệ b − tỉ lệ a) ghép cặp theo trang."""
        pa = {r["page"]: r for r in a_rows}; pb = {r["page"]: r for r in b_rows}
        pages = sorted(set(pa) & set(pb))
        if len(pages) < 3:
            return None, None, len(pages)
        rng = np.random.default_rng(SEED)
        d = []
        for _ in range(B):
            s = [pages[i] for i in rng.integers(0, len(pages), len(pages))]
            d.append(agg([pb[p] for p in s], num, den) - agg([pa[p] for p in s], num, den))
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(lo), float(hi), len(pages)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["book"], r["variant"])].append(r)
    books = sorted({(r["kind"], r["book"]) for r in rows})
    metrics = [("m_v1p", "n_kim", "ĐÚNG/chữ kim (precision, sự thật)"), ("m_v1p", "n_human", "ĐỌC ĐƯỢC/chữ người (recall, sự thật)"),
               ("m_am", "n_kim", "hợp âm/chữ kim (gián tiếp)"), ("m_am", "n_syl", "phủ âm/âm QN (gián tiếp, thấp = sót)"),
               ("n_kim", "n_syl", "số chữ kim / số âm QN")]
    L = ["# TN7 — tham số kim trên sách chép tay (30/09)", "",
         "Cùng các trang mẫu; chênh lệch = biến thể − đường HIỆN TẠI, CI 95 % bootstrap ghép cặp theo trang. "
         "Borg có sự thật (chữ Nôm người phiên); STT chỉ có thước gián tiếp (chữ kim ∈ R(âm QN)).", ""]
    summ = {}
    for kind, book in books:
        base = "hien_tai_lt2_in" if kind == "borg" else "hien_tai_lt1_in"
        vs = sorted({v for (k, b, v) in groups if k == kind and b == book}, key=lambda v: (not v.startswith("hien_tai"), v))
        L += [f"## {book}", "", "| biến thể | trang | " + " | ".join(m[2] for m in metrics if kind == "borg" or not m[0] == "m_v1p") + " |",
              "|---|---:|" + "---:|" * sum(1 for m in metrics if kind == "borg" or m[0] != "m_v1p")]
        for v in vs:
            rs = groups[(kind, book, v)]
            cells = []
            for num, den, name in metrics:
                if kind != "borg" and num == "m_v1p":
                    continue
                val = agg(rs, num, den)
                txt = f"{val * 100:.1f} %" if num != "n_kim" else f"{val:.3f}"
                if v != base and num != "n_kim":
                    lo, hi, npg = boot_diff(groups[(kind, book, base)], rs, num, den)
                    if lo is not None:
                        txt += f" ({(val - agg([r for r in groups[(kind, book, base)] if r['page'] in {x['page'] for x in rs}], num, den)) * 100:+.1f} [{lo * 100:+.1f}; {hi * 100:+.1f}])"
                cells.append(txt)
                summ.setdefault(book, {}).setdefault(v, {})[f"{num}/{den}"] = round(val, 5)
            L.append(f"| {v}{' (HIỆN TẠI)' if v == base else ''} | {len(rs)} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Đọc bảng: chênh lệch có CI 95 % KHÔNG chứa 0 mới tính là khác thật. Trên Borg so hai thước: nếu thước gián tiếp xếp "
          "hạng biến thể giống thước sự thật thì mới tin nó cho STT.", ""]
    (OUT / "KET_QUA.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), n_rows=len(rows),
                                                      by_book=summ), ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    print(f"[tn7] -> {OUT / 'KET_QUA.md'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--eval", action="store_true")
    ap.add_argument("--budget", type=int, default=None, help="tối đa số trang gọi trong lượt này")
    a = ap.parse_args(argv)
    if a.dry_run:
        jobs = plan()
        per = defaultdict(int)
        for j in jobs:
            per[(j["book"], j["variant"])] += 1
        for (b, v), n in sorted(per.items()):
            print(f"  {b:20s} {v:16s} {n:3d} trang")
        todo = sum(1 for j in jobs if not out_file(j).exists())
        print(f"[tn7] tổng {len(jobs)} lượt (còn {todo}) · ≈ {todo * 6.5 / 60:.0f} phút · ghi {OUT}")
        return 0
    if a.status:
        return status()
    if a.eval:
        return evaluate()
    return run(a.budget)


if __name__ == "__main__":
    raise SystemExit(main())
