"""common.py — đường dẫn, bảng nhãn, tài sản (kiểm sha256), cache đặc trưng, từ điển V1+ dùng chung cho gold_exact.

Từ điển / biến thể chữ (R_of, var_eq, V1+) là bản chép NGUYÊN harness_lib.py + vlib.var_eq_plus của thư mục thử nghiệm
(kim_bottleneck/harness): R(âm) = core.text.dictionary.load_qn_to_nom trên Dict/QuocNgu_SinoNom.csv với khoá
normalize_tone_marks(lower); biến thể = Unihan_Variants (5 trường, cạnh trực tiếp) ∪ OpenCC t2s; V1+ thêm
kJapanese(Old)Variant và NFC. Bảng biến thể đã phân tích sẵn nằm trong tài sản lexicon.pt (không đọc thư mục thử nghiệm).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
MODELS = REPO / "models" / "gold_exact"
CONFIG = REPO / "config" / "gold_exact.yaml"

SETS8 = ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK"]
AB = {"Chrestomathie1872": "Chr", "LucVanTien1883": "L83", "KimVanKieu1884": "KVK",
      "LucVanTien1916": "L16", "TruyenKieu1872": "TK"}
BOOK_SETS = ["SachThanhTruyen", "Chrestomathie1872", "LucVanTien1883", "KimVanKieu1884", "LucVanTien1916",
             "TruyenKieu1872"]
PREP_DIR = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}
# 5 sách khai crop_source: original (ảnh giao = điểm ảnh trang GỐC); STT: trang đã xử lý như pipeline
ORIGINAL = {"LucVanTien1883", "KimVanKieu1884", "Chrestomathie1872", "LucVanTien1916", "TruyenKieu1872"}
IHR = ("LucVanTien1916", "TruyenKieu1872")
EVIDENCE = {"L16": "do_tren_nhan_nguoi", "TK": "do_tren_nhan_nguoi", "KVK": "uoc_luong", "L83": "uoc_luong",
            "stt2": "suy_doan", "stt4": "suy_doan", "stt11": "suy_doan", "Chr": "suy_doan"}


def rd(path, **k) -> pd.DataFrame:
    """Đọc CSV giữ nguyên chuỗi: `nan` là một âm tiếng Việt (難), ô rỗng giữ ''."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, **k)


def load_cfg(path: Path | str = CONFIG) -> dict:
    import yaml
    return yaml.safe_load(open(path, encoding="utf-8"))


def set8_of(book_set, book) -> np.ndarray:
    bs = np.asarray(book_set, dtype=object); bk = np.asarray(book, dtype=object)
    return np.where(bs == "SachThanhTruyen", bk, pd.Series(bs).map(AB).fillna("").values).astype(object)


def uid_path(uid: str) -> str:
    """cell_uid -> đường dẫn tương đối của crop (quy ước TN4): <book_set>/<book>/<page>/c.._n.._s...png"""
    a = uid.split("/")
    return "/".join(a[:3]) + "/" + "_".join(a[3:]) + ".png"


def page_dir(book_set: str, book: str) -> Path:
    return REPO / "prepared" / PREP_DIR.get(book, book_set)


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def sha256_file(p: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Log:
    def __init__(self, quiet=False):
        self.t0 = time.time(); self.quiet = quiet

    def __call__(self, msg):
        if not self.quiet:
            print(f"[gold_exact {time.time() - self.t0:6.0f}s] {msg}", flush=True)


def load_gold(all_dir: Path) -> pd.DataFrame:
    """Mọi ô GOLD của bộ giao nộp (thứ tự hàng giữ như labels.csv)."""
    L = rd(Path(all_dir) / "labels.csv")
    G = L[L.tier == "GOLD"].reset_index(drop=True)
    G["set8"] = set8_of(G.book_set.values, G.book.values)
    x = G.cell_uid.str.extract(r"/c(\d+)/n(-?\d+)/s(-?\d+)$")
    G["nom_i"] = pd.to_numeric(x[1], errors="coerce").fillna(-1).astype(int)
    G["syl_idx"] = x[2].fillna("")
    return G


# ------------------------------------------------------------------------------------------------ tài sản
class AssetError(RuntimeError):
    pass


class Assets:
    """models/gold_exact + MANIFEST.json. Mọi tệp được kiểm sha256 TRƯỚC khi nạp; thiếu/sai -> AssetError rõ ràng.
    `external` = tệp nặng có sẵn trong repo (encoder nom-embed/ArcFace, font) — kiểm sha tại chỗ, không chép."""

    def __init__(self, root: Path = MODELS, verify: bool = True):
        self.root = Path(root)
        mf = self.root / "MANIFEST.json"
        if not mf.exists():
            raise AssetError(f"Thiếu {mf}. Dựng tài sản: .venv/bin/python -m pipeline.gold_exact.export_assets --sp <thư mục thử nghiệm>")
        self.man = json.load(open(mf, encoding="utf-8"))
        self.verify = verify
        self._ok = {}

    def _check(self, key, path: Path, ent: dict):
        if key in self._ok:
            return path
        if not path.exists():
            raise AssetError(f"Thiếu tài sản {path} (MANIFEST: {ent.get('desc', '')}). Dựng lại bằng export_assets.")
        if ent.get("kind") == "dir_digest":
            d = dir_digest(path, ent.get("pattern", "U+*.png"), ent.get("min_bytes", 1024))
            if self.verify and d != ent["digest"]:
                raise AssetError(f"Thư mục {path} đã đổi (digest {d[:12]} ≠ {ent['digest'][:12]}).")
        elif self.verify:
            if path.stat().st_size != ent["bytes"]:
                raise AssetError(f"Tài sản {path} sai kích thước ({path.stat().st_size} ≠ {ent['bytes']}).")
            s = sha256_file(path)
            if s != ent["sha256"]:
                raise AssetError(f"Tài sản {path} sai sha256 ({s[:12]} ≠ {ent['sha256'][:12]}).")
        self._ok[key] = True
        return path

    def path(self, name: str) -> Path:
        ent = self.man["files"].get(name)
        if ent is None:
            raise AssetError(f"MANIFEST không có tài sản '{name}'.")
        return self._check(name, self.root / name, ent)

    def ext(self, rel: str) -> Path:
        ent = self.man["external"].get(rel)
        if ent is None:
            raise AssetError(f"MANIFEST không có tệp ngoài '{rel}'.")
        return self._check("ext:" + rel, REPO / rel, ent)

    def sha(self, name: str) -> str:
        ent = self.man["files"].get(name) or self.man["external"].get(name)
        return ent.get("sha256") or ent.get("digest")

    def load(self, name: str):
        import torch
        return torch.load(self.path(name), map_location="cpu", weights_only=False)


def dir_digest(root: Path, pattern="U+*.png", min_bytes=1024) -> str:
    """Băm (đường dẫn tương đối, kích thước) của các tệp glyph đủ lớn (con trỏ git-LFS < 1 KB bị bỏ như sv_lib)."""
    items = sorted((str(p.relative_to(root)), p.stat().st_size) for p in root.rglob(pattern) if p.stat().st_size >= min_bytes)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ cache
class EmbCache:
    """Cache vector đặc trưng theo khoá (md5 ảnh / md5 mảng) và sha mô hình: <cache>/<tag>_<sha16>.npz."""

    def __init__(self, cache_dir: Path | None, tag: str, model_sha: str, dim: int | None = None, recompute=False):
        self.f = (Path(cache_dir) / f"{tag}_{model_sha[:16]}.npz") if cache_dir else None
        self.d = {}
        if self.f is not None and self.f.exists() and not recompute:
            z = np.load(self.f, allow_pickle=False)
            for k, v in zip(z["keys"], z["E"]):
                self.d[str(k)] = v
        self.dirty = False

    def missing(self, keys):
        return [k for k in dict.fromkeys(keys) if k not in self.d]

    def put(self, keys, E):
        for k, v in zip(keys, E):
            self.d[k] = np.asarray(v, np.float32)
        self.dirty = True

    def get(self, keys) -> np.ndarray:
        return np.stack([self.d[k] for k in keys]).astype(np.float32)

    def save(self):
        if self.f is None or not self.dirty:
            return
        self.f.parent.mkdir(parents=True, exist_ok=True)
        ks = list(self.d)
        tmp = self.f.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(ks), E=np.stack([self.d[k] for k in ks]).astype(np.float32))
        tmp.replace(self.f)
        self.dirty = False


def f16(x) -> np.ndarray:
    """Làm tròn qua float16 như các bước gốc (emb_*.f16.npy) — cần để tái lập đúng điểm."""
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


# ------------------------------------------------------------------------------------------------ từ điển V1+
@lru_cache(maxsize=1)
def _dicts():
    from core.text.dictionary import load_qn_to_nom, load_similarity_dict
    from core.text.text_utils import normalize_tone_marks
    q2n = load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv"))
    Rset = {k: set(v) for k, v in q2n.items()}
    SIM = {k: set(v) for k, v in load_similarity_dict(str(REPO / "Dict/SinoNom_Similar.csv")).items()}
    return Rset, SIM, normalize_tone_marks


def R_key(syllable: str) -> str:
    return _dicts()[2]((syllable or "").strip().lower())


def R_of(syllable: str) -> set:
    return _dicts()[0].get(R_key(syllable), set())


def is_sim(a: str, b: str) -> bool:
    S = _dicts()[1]
    return bool(a and b) and (b in S.get(a, ()) or a in S.get(b, ()))


_LEX = {}


def set_lexicon(assets: Assets):
    """Nạp bảng biến thể (Unihan + OpenCC + kJapanese) từ tài sản lexicon.pt."""
    if not _LEX:
        z = assets.load("lexicon.pt")
        _LEX["V"] = {a: set(b) for a, b in z["V"].items()}
        _LEX["SIMP"] = dict(z["SIMP"])
        _LEX["JP"] = set(map(tuple, z["JP"]))
    return _LEX


def is_pua(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch[0])
    return 0xE000 <= o <= 0xF8FF or 0xF0000 <= o <= 0x10FFFD


def simp(c: str) -> str:
    return _LEX["SIMP"].get(c, c)


def variants_of(c: str) -> set:
    return _LEX["V"].get(c, set())


def var_eq(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if is_pua(a) or is_pua(b):
        return False
    if b in variants_of(a):
        return True
    return simp(a) == simp(b)


def var_in(c: str, S) -> bool:
    if not c:
        return False
    if c in S:
        return True
    return any(var_eq(c, x) for x in S)


def var_eq_plus(a: str, c: str) -> bool:
    """V1+ (= r4 p03_apply.var_eq_plus)."""
    if not a or not c:
        return False
    JP = _LEX["JP"]
    if var_eq(a, c) or (a, c) in JP:
        return True
    na, nc = unicodedata.normalize("NFC", a), unicodedata.normalize("NFC", c)
    return na == nc or var_eq(na, nc) or (na, nc) in JP


def wilson(k: int, n: int, z: float = 1.96):
    import math
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))
