"""w_quy_mo_va_ngan_sach_lib.py — TN11 / hướng "QUY MÔ DANH SÁCH CHỮ + NGÂN SÁCH GPU" cho gói Kaggle sinh ảnh theo sách (0 API, chỉ CPU).

Dựng từ nhãn MỚI NHẤT của 10 bộ (prepared/_auto/<Sách>/dataset_out/labels_gated.csv; 3 STT: dataset_out/labels_final.csv, tách theo cột book)
và từ điển R(âm) (Dict/QuocNgu_SinoNom.csv qua pipeline.gold_exact.common.R_of):
  S0     = chữ ở cột label (≠ '') ∪ chữ kim (ocr_char)
  S1(k)  = S0 ∪ top-k ứng viên R(âm) CỦA MỖI ÂM trong cuốn, xếp theo TIÊN NGHIỆM (k = 3, 5, 8)
  S2     = hợp R(âm) của mọi âm xuất hiện trong cuốn (cận trên)
Hai bản S0: "literal" (đúng định nghĩa đề bài: cột label cuối cùng) và "sach" (cột label TRƯỚC bước 4b chon_chu ∪ kim). Với 8/10 bộ hai bản
trùng; ở B18/B34 bộ chọn chữ chon_chu đã đổi nhãn bằng mô hình học (LOBO) trên nhãn NGƯỜI của sách Borg kia ⇒ bản "sach" không chịu ảnh
hưởng gián tiếp nào của nhãn người ⇒ bản "sach" là bản DÙNG ĐỂ DỰNG GÓI; bản "literal" chỉ để đối chiếu.

TIÊN NGHIỆM (không nhãn người): đếm (âm, chữ) trên ô NEO TỰ ĐỘNG (rule bắt đầu s1_inter_s2_direct, nhãn = chữ kim ∈ R(âm); nhãn lấy bản trước
chon_chu): điểm(âm, chữ) = 3·n_cùng_cuốn + 1·n_cuốn_khác; hoà thì theo tần suất toàn cục của chữ (số ô neo của chữ ở mọi cuốn, với cuốn đang xét chỉ tính
phần đang dùng), rồi theo mã Unicode (tất định). Mô-đun này KHÔNG nạp mô-đun nhãn người (t00_base, tn6lib, borg_endtoend_eval, eval_ihr).

Chạy với PYTHONDONTWRITEBYTECODE=1 (không để lại .pyc ngoài thư mục đầu ra). Đầu ra: measure_out/_tn11/full/quy_mo_va_ngan_sach/.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TN8 = REPO / "lab" / "thu_nghiem_kim" / "TN8_chon_chu"
for _p in (str(REPO), str(TN8)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tn8lib as T  # noqa: E402  (chỉ lấy danh mục bộ, đường dẫn nhãn, lex(); KHÔNG nạp mô-đun nhãn người ở đây)

OUT = REPO / "measure_out" / "_tn11" / "full" / "quy_mo_va_ngan_sach"
ORDER = list(T.ORDER)                       # stt2 stt4 stt11 Chr L83 KVK L16 TK B18 B34
HUMAN = ("B18", "B34", "L16", "TK")         # có nhãn người — CHỈ để chấm đường cong phủ
KS = (3, 5, 8)
KS_ALL = tuple(range(1, 13))                # bảng chi phí theo k (đồ thị chi phí)
W_SELF, W_OTH = 3, 1
ANCHOR_PREFIX = "s1_inter_s2_direct"
INF = 10 ** 9
NS = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 16000, 20000, 25000, 32000]

# ----- giả định ngân sách (đề bài + log Kaggle đo được)
SEC_PER_IMG = 2.7            # giây/ảnh/GPU (log_B34: 2,63–2,74; log_B18: 2,67–2,75)
N_GPU = 2                    # Kaggle T4 x2, hai tiến trình song song
SHIFT_H = 9.0                # trần một ca (giờ, đồng hồ tường)
SHIFT_OVERHEAD_H = 0.25      # nạp mô hình + pip + nén kết quả (giả định, chưa đo)
SESSION_MAX_H = 12.0         # trần phiên Kaggle (đề bài: phổ biến, đang xác minh ở hướng khác)
WEEK_QUOTA_H = 30.0          # hạn mức GPU/tuần (đề bài: khoảng, đang xác minh ở hướng khác)
FT_SEC_PER_STEP = 3.44       # tinh chỉnh (lô 8 x tích luỹ 2): log_B34 10 327 s / 3000 bước

# ----------------------------------------------------------------------------------------------------------------------
# Unicode
# ----------------------------------------------------------------------------------------------------------------------


def block_of(c: str) -> str:
    o = ord(c)
    if 0x4E00 <= o <= 0x9FFF:
        return "URO"
    if 0x3400 <= o <= 0x4DBF:
        return "ExtA"
    if 0x20000 <= o <= 0x2A6DF:
        return "ExtB"
    if 0x2A700 <= o <= 0x2EBEF:
        return "ExtC-F"
    if 0x30000 <= o <= 0x323AF:
        return "ExtG-H"
    if 0xF900 <= o <= 0xFAFF or 0x2F800 <= o <= 0x2FA1F:
        return "Compat"
    if 0x2E80 <= o <= 0x2FDF:
        return "Kangxi"
    if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
        return "Kana"
    if 0xE000 <= o <= 0xF8FF:
        return "PUA-BMP"
    if 0xF0000 <= o <= 0x10FFFF:
        return "PUA-sup"
    return "other"


def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Nhãn mới nhất của 10 bộ (CHỈ các cột kế hoạch; không đọc cột nào của nhãn người)
# ----------------------------------------------------------------------------------------------------------------------
PLAN_COLS = ["book", "page", "column", "syllable", "ocr_char", "label", "tier", "rule", "chon_chu", "chon_chu_truoc"]


def labels_path(b: str) -> Path:
    return REPO / T.BOOKS[b]["labels"]


def load_cells(b: str) -> pd.DataFrame:
    cfg = T.BOOKS[b]
    L = pd.read_csv(labels_path(b), dtype=str, keep_default_na=False, usecols=lambda c: c in PLAN_COLS)
    if cfg.get("book_filter"):
        L = L[L.book == cfg["book_filter"]]
    L = L.reset_index(drop=True)
    for c in ("chon_chu", "chon_chu_truoc"):
        if c not in L.columns:
            L[c] = ""
    return L


def label_truoc_chon_chu(L: pd.DataFrame) -> np.ndarray:
    """Nhãn TRƯỚC bước 4b chon_chu. chon_chu_truoc = '<tầng>|<nhãn>[|âm:...]' (vd 'REVIEW|名', 'SYLLABLE|', 'REVIEW||âm:g2ø')."""
    lab = L.label.to_numpy(dtype=object).copy()
    m = (L.chon_chu != "").to_numpy()
    if m.any():
        parts = L.chon_chu_truoc[m].str.split("|", n=2, expand=True)
        if parts.shape[1] > 1:
            lab[m] = parts[1].fillna("").to_numpy(dtype=object)
        else:
            lab[m] = ""
    return lab


def prepare_cells(b: str, C) -> pd.DataFrame:
    L = load_cells(b)
    L["rk"] = [C.R_key(s) for s in L.syllable]
    L["lab_pre"] = label_truoc_chon_chu(L)
    return L


# ----------------------------------------------------------------------------------------------------------------------
# Tiên nghiệm + danh sách chữ
# ----------------------------------------------------------------------------------------------------------------------


def anchors_counter(L: pd.DataFrame, Rm: dict, mask: np.ndarray | None = None) -> Counter:
    """Ô NEO tự động: rule bắt đầu s1_inter_s2_direct, nhãn (trước chon_chu) = chữ kim ∈ R(âm). Đếm (rk, chữ)."""
    lp = L.lab_pre.to_numpy(dtype=object)
    kim = L.ocr_char.to_numpy(dtype=object)
    m = L.rule.str.startswith(ANCHOR_PREFIX).to_numpy() & (lp != "") & (lp == kim)
    if mask is not None:
        m &= mask
    cnt: Counter = Counter()
    for rk, c in zip(L.rk.to_numpy()[m], lp[m]):
        if c in Rm[rk]:
            cnt[(rk, c)] += 1
    return cnt


def build_world(books=ORDER, log=print) -> dict:
    C = T.lex()
    cells, Rm = {}, {}
    for b in books:
        L = prepare_cells(b, C)
        cells[b] = L
        for rk in set(L.rk):
            if rk not in Rm:
                Rm[rk] = frozenset(C.R_of(rk))
        log(f"[lib] {b}: {len(L)} ô, {L.rk.nunique()} âm, nhãn {labels_path(b).relative_to(REPO)}")
    n_self = {b: anchors_counter(cells[b], Rm) for b in books}
    n_all: Counter = Counter()
    for b in books:
        n_all.update(n_self[b])
    n_oth = {b: n_all - n_self[b] for b in books}
    g_oth = {}
    for b in books:
        g = Counter()
        for (rk, c), n in n_oth[b].items():
            g[c] += n
        g_oth[b] = g
    return dict(C=C, cells=cells, Rm=Rm, n_self=n_self, n_oth=n_oth, g_oth=g_oth)


def plan_book(L: pd.DataFrame, Rm: dict, n_self_b: Counter, n_oth_b: Counter, g_oth_b: Counter, mode: str,
              mask: np.ndarray | None = None) -> dict:
    """Danh sách chữ của MỘT cuốn. mode = 'literal' (cột label cuối) | 'sach' (nhãn trước chon_chu).
    mask: chỉ các ô này góp bằng chứng chữ (S0, ô neo) — dùng cho kiểm giữ-ra theo khối trang; âm của MỌI ô vẫn biết (QN có sẵn)."""
    labs = L.label.to_numpy(dtype=object) if mode == "literal" else L.lab_pre.to_numpy(dtype=object)
    kims = L.ocr_char.to_numpy(dtype=object)
    sel = np.ones(len(L), bool) if mask is None else mask
    e: Counter = Counter()
    for a, k in zip(labs[sel], kims[sel]):
        if a and k and a != k:
            e[a] += 1
            e[k] += 1
        elif a or k:
            e[a or k] += 1
    S0 = set(e)
    g: Counter = Counter(g_oth_b)
    for (rk, c), n in n_self_b.items():
        g[c] += n
    rk_counts = Counter(L.rk.tolist())
    best: dict = {}
    demand: Counter = Counter()
    for rk, ncell in rk_counts.items():
        R = Rm[rk]
        if not R:
            continue
        sc = sorted(R, key=lambda c: (-(W_SELF * n_self_b.get((rk, c), 0) + W_OTH * n_oth_b.get((rk, c), 0)), -g.get(c, 0), ord(c)))
        for i, c in enumerate(sc, 1):
            b0 = best.get(c)
            if b0 is None or i < b0:
                best[c] = i
                demand[c] = ncell
            elif i == b0:
                demand[c] += ncell
    S2 = set(best)
    S1 = {k: S0 | {c for c, r in best.items() if r <= k} for k in KS_ALL}
    r0 = sorted(S0, key=lambda c: (-e[c], -g.get(c, 0), ord(c)))
    r1 = sorted((c for c in S2 if c not in S0), key=lambda c: (best[c], -demand[c], -g.get(c, 0), ord(c)))
    return dict(S0=S0, S1=S1, S2=S2, ranked=r0 + r1, n0=len(r0), best=best, e=e, demand=demand, rk_counts=rk_counts,
                n_cells=int(len(L)), n_am=int(len(rk_counts)), n_am_co_R=int(sum(1 for rk in rk_counts if Rm[rk])),
                n_o_am_co_R=int(sum(n for rk, n in rk_counts.items() if Rm[rk])))


def filtered(plan: dict, avail) -> dict:
    """Lọc mọi tập/danh sách theo vị từ avail(c) (chữ có phông). Trả bản mới cùng khoá."""
    f = dict(plan)
    f["S0"] = {c for c in plan["S0"] if avail(c)}
    f["S1"] = {k: {c for c in v if avail(c)} for k, v in plan["S1"].items()}
    f["S2"] = {c for c in plan["S2"] if avail(c)}
    f["ranked"] = [c for c in plan["ranked"] if avail(c)]
    f["n0"] = sum(1 for c in plan["ranked"][: plan["n0"]] if avail(c))
    return f


def positions(ranked: list) -> dict:
    return {c: i + 1 for i, c in enumerate(ranked)}


def cand_coverage(Rm: dict, rk_counts: Counter, pos: dict, Ns) -> tuple:
    """Phủ ỨNG VIÊN (không cần sự thật): với mỗi ô, tỉ lệ R(âm) của ô thuộc N chữ đầu (trung bình theo ô) và tỉ lệ ô có TOÀN BỘ R(âm) thuộc N chữ đầu."""
    Ns = np.asarray(Ns)
    acc = np.zeros(len(Ns)); full = np.zeros(len(Ns)); tot = 0
    for rk, n in rk_counts.items():
        R = Rm[rk]
        if not R:
            continue
        p = np.sort(np.array([pos.get(c, INF) for c in R]))
        cnt = np.searchsorted(p, Ns, side="right")
        acc += n * cnt / len(R)
        full += n * (cnt == len(R))
        tot += n
    return acc / max(tot, 1), full / max(tot, 1), tot


# ----------------------------------------------------------------------------------------------------------------------
# Phông
# ----------------------------------------------------------------------------------------------------------------------
FONT_FILES = {
    "NomNaTong-Regular.ttf [font_diffusion]": "font_diffusion/fonts/NomNaTong-Regular.ttf",
    "NomNaTong-Regular.ttf [fonts]": "fonts/NomNaTong-Regular.ttf",
    "NomNaTong-Regular.otf": "font_diffusion/fonts/NomNaTong-Regular.otf",
    "NomNaTong-Regular2.otf": "font_diffusion/fonts/NomNaTong-Regular2.otf",
    "NomNaTongLight.ttf": "font_diffusion/fonts/NomNaTongLight.ttf",
    "NomNaTongLight2.ttf": "font_diffusion/fonts/NomNaTongLight2.ttf",
    "Han-nom Minh 1.42.otf": "font_diffusion/fonts/Han-nom Minh 1.42.otf",
    "HAN NOM A.ttf": "font_diffusion/fonts/HAN NOM A.ttf",
    "HAN NOM B.ttf": "font_diffusion/fonts/HAN NOM B.ttf",
    "HanaMinA.ttf": "font_diffusion/fonts/HanaMinA.ttf",
    "HanaMinA.otf": "font_diffusion/fonts/HanaMinA.otf",
    "HanaMinB.ttf": "font_diffusion/fonts/HanaMinB.ttf",
    "HanaMinB.otf": "font_diffusion/fonts/HanaMinB.otf",
    "HanaMinC.otf": "font_diffusion/fonts/HanaMinC.otf",
    "Han-Nom Kai 1.00.otf": "font_diffusion/fonts/Han-Nom Kai 1.00.otf",
    "Han-Nom-Khai-Regular-300623.ttf": "font_diffusion/fonts/Han-Nom-Khai-Regular-300623.ttf",
    "PlangothicP1-Regular.ttf": "fonts/PlangothicP1-Regular.ttf",
    "PlangothicP2-Regular.ttf": "fonts/PlangothicP2-Regular.ttf",
}
BASE_FONT = "NomNaTong-Regular.ttf [font_diffusion]"      # = config/pipeline.yaml paths.font_path = phông của gói TN11 (p06) và của core/ranking/fontdiffusion_gen.py
# Họ kiểu chữ theo TÊN phông/giấy phép (chưa kiểm hình bằng mắt): serif/Song-Minh · Khải (thư pháp) · Gothic (không chân)
FONT_FAMILY = {k: ("Gothic" if k.startswith("Plangothic") else "Khải" if "Kai" in k or "Khai" in k else "Song-Minh") for k in FONT_FILES}


def cmap_of(path: str | Path) -> set:
    """Hợp mọi bảng con cmap (đúng cách font_diffusion/src/tools/utils.is_char_in_font và pipeline/ground_truth/audit_grid._cmap_of)."""
    from fontTools.ttLib import TTFont
    f = TTFont(str(REPO / path), fontNumber=0, lazy=True)
    s: set = set()
    for tb in f["cmap"].tables:
        s |= set(tb.cmap)
    f.close()
    return s


def all_cmaps() -> dict:
    return {k: cmap_of(p) for k, p in FONT_FILES.items()}


# Mã PUA (vùng dùng riêng) mang nghĩa THEO TỪNG PHÔNG: cùng một mã có thể là chữ khác ở phông khác. Kim và từ điển R(âm) xuất mã PUA theo bảng của họ NomNaTong,
# nên chữ PUA chỉ được vẽ bằng phông họ NomNaTong; có glyph ở HAN NOM A/B, HanaMin, Plangothic CHƯA đủ để tin (chưa xác minh cùng bảng mã) ⇒ coi là không có phông.
PUA_SAFE = tuple(k for k in FONT_FILES if k.startswith("NomNaTong"))


def is_pua_char(c: str) -> bool:
    return block_of(c).startswith("PUA")


def assign_fonts(chars, order, cmaps, bad: set | None = None, pua_safe=None) -> dict:
    """chữ -> phông ĐẦU TIÊN trong order có cmap chứa chữ (và chưa bị đánh dấu 'vẽ rỗng'/'trùng .notdef'); None nếu không phông nào có.
    pua_safe (tuỳ chọn): với chữ PUA chỉ xét các phông thuộc tập này."""
    out = {}
    for c in chars:
        cp = ord(c)
        cand = order if (pua_safe is None or not is_pua_char(c)) else [f for f in order if f in pua_safe]
        for f in cand:
            if cp in cmaps[f] and not (bad and (f, c) in bad):
                out[c] = f
                break
        else:
            out[c] = None
    return out


def render_check(assign: dict, size: int = 64) -> tuple:
    """Vẽ từng chữ bằng phông được gán (PIL/FreeType, cùng thư viện nền với pygame.freetype của FontDiffuser). Trả (rỗng, trùng_notdef) dạng set((phông, chữ)).
    rỗng = ảnh không có điểm mực; trùng_notdef = ảnh giống hệt ô .notdef của phông (vẽ một mã không có trong cmap)."""
    from PIL import Image, ImageDraw, ImageFont
    by = defaultdict(list)
    for c, f in assign.items():
        if f is not None:
            by[f].append(c)
    blank, notdef = set(), set()
    W = size * 3
    for f, chars in by.items():
        font = ImageFont.truetype(str(REPO / FONT_FILES[f]), size)
        cm = cmap_of(FONT_FILES[f])

        def draw(ch):
            im = Image.new("L", (W, W), 0)
            ImageDraw.Draw(im).text((size // 2, size // 2), ch, font=font, fill=255)
            return im
        sigs = set()
        for cp in (0xFFFF, 0x2FFFE, 0x10FFFE):
            if cp not in cm:
                im = draw(chr(cp))
                if im.getbbox() is not None:
                    sigs.add(hashlib.md5(im.tobytes()).hexdigest())
        for c in chars:
            im = draw(c)
            if im.getbbox() is None:
                blank.add((f, c))
            elif hashlib.md5(im.tobytes()).hexdigest() in sigs:
                notdef.add((f, c))
    return blank, notdef


def robust_assign(chars, order, cmaps, max_iter: int = 4, pua_safe=None) -> tuple:
    """Gán phông có kiểm vẽ: chữ vẽ rỗng/trùng .notdef ở phông gán ⇒ chuyển sang phông kế tiếp. Trả (assign, bad_set, nhật ký)."""
    bad: set = set()
    hist = []
    for it in range(max_iter):
        a = assign_fonts(chars, order, cmaps, bad, pua_safe)
        blank, notdef = render_check(a)
        new = (blank | notdef) - bad
        hist.append(dict(vong=it, ro=len(blank), trung_notdef=len(notdef)))
        if not new:
            return a, bad, hist
        bad |= new
    return assign_fonts(chars, order, cmaps, bad, pua_safe), bad, hist


# ----------------------------------------------------------------------------------------------------------------------
# Đường cong phủ
# ----------------------------------------------------------------------------------------------------------------------


def curve(p: np.ndarray, Ns) -> list:
    """p = vị trí (1-based, INF nếu vắng) của chữ đúng từng ô; trả tỉ lệ ô có p ≤ N."""
    p = np.sort(np.asarray(p, dtype=np.int64))
    n = len(p)
    return [float(np.searchsorted(p, N, side="right")) / n if n else float("nan") for N in Ns]


def truth_positions(truth, pos: dict) -> np.ndarray:
    return np.array([pos.get(t, INF) for t in truth], dtype=np.int64)


def variant_index(C, chars) -> dict:
    """Chỉ mục tương đương V1+ (= gold_exact.common.var_eq_plus(truth, c)) trên một tập chữ: truth -> các c trong tập tương đương."""
    chars = list(chars)
    simp_inv, nfc_inv = defaultdict(list), defaultdict(list)
    for c in chars:
        simp_inv[C.simp(c)].append(c)
        nfc_inv[unicodedata.normalize("NFC", c)].append(c)
    jp = defaultdict(set)
    for a, c in C._LEX["JP"]:
        jp[a].add(c)
    present = set(chars)

    def neighbors(g: str) -> set:
        out = {g} if g in present else set()
        if not C.is_pua(g):
            out |= {x for x in C.variants_of(g) if x in present and not C.is_pua(x)}
            out |= {x for x in simp_inv.get(C.simp(g), ()) if not C.is_pua(x)}
        out |= {x for x in jp.get(g, ()) if x in present}
        ng = unicodedata.normalize("NFC", g)
        out |= set(nfc_inv.get(ng, ()))
        return out
    return neighbors


def variant_positions(truth, pos: dict, neighbors) -> np.ndarray:
    memo = {}
    out = np.empty(len(truth), dtype=np.int64)
    for i, g in enumerate(truth):
        v = memo.get(g)
        if v is None:
            v = min([pos.get(x, INF) for x in neighbors(g)] or [INF])
            memo[g] = v
        out[i] = v
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Ngân sách / ca
# ----------------------------------------------------------------------------------------------------------------------


def gpu_hours(n_images: float, sec: float = SEC_PER_IMG) -> float:
    return n_images * sec / 3600.0


def wall_hours(n_images: float, n_gpu: int = N_GPU, sec: float = SEC_PER_IMG) -> float:
    return n_images * sec / 3600.0 / n_gpu


def knapsack_caps(curves: dict, grid: int, total_units: int, styles: dict, weight: dict | None = None) -> tuple:
    """Quy hoạch động nhóm-ba-lô: chọn N_b (bội của grid) mỗi cuốn để TỐI ĐA hoá tổng trọng số phủ, ràng buộc Σ styles_b·N_b ≤ total_units·grid.
    curves[b] = hàm phủ ở bội số grid: mảng c[j] = phủ khi lấy j·grid chữ (j = 0..J_b). Trả (N_b, mục tiêu)."""
    books = list(curves)
    w = weight or {b: 1.0 for b in books}
    NEG = -1e18
    dp = np.full(total_units + 1, NEG); dp[0] = 0.0
    choice = []
    for b in books:
        c = np.asarray(curves[b]); m = styles[b]
        new = np.full(total_units + 1, NEG); ch = np.zeros(total_units + 1, dtype=np.int64)
        for j in range(len(c)):
            cost = j * m
            if cost > total_units:
                break
            cand = dp[: total_units + 1 - cost] + w[b] * c[j]
            tgt = new[cost:]
            better = cand > tgt
            tgt[better] = cand[better]
            ch[cost:][better] = j
        dp = new
        choice.append(ch)
    u = int(np.argmax(dp)); obj = float(dp[u])
    caps = {}
    for b, ch, m in zip(reversed(books), reversed(choice), reversed([styles[x] for x in books])):
        j = int(ch[u]); caps[b] = j * grid; u -= j * m
    return caps, obj


def pack_shifts(units: list, cap_h: float, overhead_h: float, n_gpu: int = N_GPU, sec: float = SEC_PER_IMG) -> list:
    """units = [(khóa, số_ảnh)] theo thứ tự ưu tiên; xếp tuần tự vào ca: mỗi ca có n_gpu làn, đơn vị kế tiếp vào làn ÍT việc nhất; khi làn ít việc
    nhất đã đầy (cap_h − overhead_h) thì mọi làn đầy ⇒ đóng ca, mở ca mới. Một đơn vị có thể bị cắt ở ranh giới ca. Trả list ca:
    dict(lan=[[(khóa, ảnh), ...] ...], n=[ảnh mỗi làn])."""
    lim = int((cap_h - overhead_h) * 3600.0 / sec)       # ảnh tối đa mỗi làn trong một ca
    shifts, cur = [], dict(lan=[[] for _ in range(n_gpu)], n=[0] * n_gpu)
    for key, n in units:
        rem = int(n)
        while rem > 0:
            k = int(np.argmin(cur["n"]))
            room = lim - cur["n"][k]
            if room <= 0:
                shifts.append(cur)
                cur = dict(lan=[[] for _ in range(n_gpu)], n=[0] * n_gpu)
                continue
            take = min(rem, room)
            cur["lan"][k].append((key, take)); cur["n"][k] += take; rem -= take
    if any(cur["n"]):
        shifts.append(cur)
    return shifts
