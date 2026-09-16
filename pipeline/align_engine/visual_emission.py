"""B-2 · PHÁT XẠ ẢNH TRONG DP (DANH_MUC_SUA_DOI_CUOI_2026-09-16 §1B B-2, giả thuyết H1).

Cho một cột: hộp chữ Nôm (hộp OCR thô của `cluster["chars"]` — có sẵn LÚC DP, trước
khi gán hộp 3 nhánh) × âm tiết QN → ma trận logP (n_box × n_syl) với
    logP[i, j] = log P(âm_j | crop hộp_i)        (CNN âm tiết 5-fold OUT-OF-FOLD, Khối B-1')
    logP[i, j] = log(NEUTRAL_P) = log 0,5         khi âm_j ∉ lớp mô hình, hoặc hộp i không cắt được
rồi chi phí DP (CALIB + ảnh) — CHÉP ĐÚNG công thức đã đo ở lab/tham_dinh_2026-09-16/dp_vis5.py:
    cost(i, j)  = cost_text(c_i, s_j) + λ · min(−logP[i, j], CAP)       (dp_vis5.py:70, :101-104)
    khe (del/ins) = COST_DEL/INS + λ · GAP_VIS                         (dp_vis5.py:63)
với λ = 0,25 (CALIB tốt nhất; λ ≥ 1 hại — benchmark 523 cột), CAP = 12, GAP_VIS = 8.
Posterior p_register dùng CÙNG chi phí (posterior_matches(cost_ij=…, cost_del=…, cost_ins=…)).

Mô hình theo trang: fold = int(md5(f"{book}|{page}").hexdigest(), 16) % 5 (KhoiB/v3/train_oof_cnn_v3.py
FOLD_FORMULA) → dùng models/fold{fold}.pt = mô hình KHÔNG học trang đó (out-of-fold thật; mô hình fold k
huấn luyện trên fold ≠ k). Kiến trúc/tiền xử lý/cắt ảnh chép y hệt KhoiB/v3/train_oof_cnn_v3.py và
lab/gan_nhan_2026-09-13/thi_giac_am_tiet.py:cut() (pad 0,08, tighten_box, vuông 64×64, mực = 1 − x/255)
để crop lúc DP cùng phân bố với crop lúc huấn luyện (crops_v3.npz). Selftest đối chiếu byte với lab.

Thiếu mô hình (chưa chạy Kaggle B-1') → `VisualEmitter.available == False` + `reason` rõ ràng;
build_dataset in cảnh báo và chạy văn bản thuần (không crash) — trừ khi --strict.

CHỈ ĐO / SIDECAR: mô-đun này không đổi tier; p_visual_syl ghi vào labels_trace (cột thêm khi bật cờ).

    .venv/bin/python -m pipeline.align_engine.visual_emission --selftest
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

from pipeline.align_engine import anchor_align as aa
from pipeline.align_engine.bbox_fix import tighten_box

REPO = Path(__file__).resolve().parents[2]

# --- hằng số đã đo (dp_vis5.py:20, :63, :101; DANH_MUC B-2) --------------------------
LAMBDA = 0.25          # trọng số kênh ảnh (λ ≥ 1 hại)
CAP = 12.0             # trần −logP một cặp (nat)
GAP_VIS = 8.0          # khe ảnh cộng vào COST_DEL/COST_INS (× λ)
NEUTRAL_P = 0.5        # âm ∉ lớp / hộp không cắt được -> logP = log 0,5 (trung tính)
N_FOLDS = 5
SZ = 64
CUT_PAD = 0.08         # thi_giac_am_tiet.cut(pad=0.08) — KHÔNG phải crop_pad_frac 0,12 của PASS 2
FOLD_FORMULA = 'int(hashlib.md5(f"{book}|{page}".encode()).hexdigest(), 16) % 5'
# thứ tự tìm mô hình: README KhoiB/v3 (kết quả Kaggle giải nén) rồi đường ghi trong DANH_MUC
DEFAULT_MODEL_DIRS = (REPO / "KhoiB/v3/p_visual_oof_v3_results/models",
                      REPO / "KhoiB/v3/models")


def page_fold(book: str, page: str) -> int:
    """fold của trang — y hệt KhoiB/v3/train_oof_cnn_v3.py:page_fold (md5 "book|page")."""
    return int(hashlib.md5(f"{book}|{page}".encode()).hexdigest(), 16) % N_FOLDS


def find_models_dir(explicit=None) -> Path | None:
    """Thư mục có đủ fold0..4.pt; None nếu chưa có (chưa chạy Kaggle B-1')."""
    cands = [Path(explicit)] if explicit else list(DEFAULT_MODEL_DIRS)
    for d in cands:
        if d.is_dir() and all((d / f"fold{k}.pt").exists() for k in range(N_FOLDS)):
            return d
    return None


# --------------------------------------------------------------------------- cắt ảnh (== lab cut())
def cut_box(gray: np.ndarray, bbox, pad: float = CUT_PAD):
    """Chép y hệt lab/gan_nhan_2026-09-13/thi_giac_am_tiet.py:cut() (crops_v3.npz cắt bằng hàm
    ấy). None nếu hộp rỗng. Trả uint8 (SZ, SZ), nền 255."""
    import cv2
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    a, b = max(0, x1 - pw), max(0, y1 - ph)
    c, d = min(W, x2 + pw), min(H, y2 + ph)
    g = gray[b:d, a:c]
    if g.size == 0:
        return None
    tb = tighten_box(g)
    if tb is not None:
        xa, ya, xb, yb = tb
        g = g[ya:yb, xa:xb]
    hh, ww = g.shape
    s = max(hh, ww)
    canvas = np.full((s, s), 255, np.uint8)
    canvas[(s - hh) // 2:(s - hh) // 2 + hh, (s - ww) // 2:(s - ww) // 2 + ww] = g
    return cv2.resize(canvas, (SZ, SZ), interpolation=cv2.INTER_AREA)


def load_page_gray(page_png) -> np.ndarray | None:
    import cv2
    return cv2.imread(str(page_png), cv2.IMREAD_GRAYSCALE)


# --------------------------------------------------------------------------- mô hình (== train_oof_cnn_v3)
def build_net(n_cls: int, emb: int = 256):
    """Y hệt KhoiB/v3/train_oof_cnn_v3.py:build_net (state_dict tương thích)."""
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.e = nn.Sequential(nn.Linear(256, emb), nn.BatchNorm1d(emb))
            self.c = nn.Linear(emb, n_cls)

        def forward(self, x):
            z = self.e(self.f(x))
            return self.c(z), z

    return Net()


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prep(X: np.ndarray, dev):
    import torch
    x = X.astype(np.float32) / 255.0
    return torch.from_numpy(1.0 - x).unsqueeze(1).to(dev)      # mực = 1, nền = 0


class VisualEmitter:
    """Nạp lười 5 mô hình fold; phát xạ logP cho hộp bất kỳ của một trang."""

    def __init__(self, models_dir=None, device=None, lam: float = LAMBDA, cap: float = CAP,
                 gap_vis: float = GAP_VIS, neutral_p: float = NEUTRAL_P, strict: bool = False):
        self.lam, self.cap, self.gap_vis = float(lam), float(cap), float(gap_vis)
        self.neutral_logp = math.log(float(neutral_p))
        self.models_dir = find_models_dir(models_dir)
        self._nets: dict[int, tuple] = {}
        self._dev = device
        self.reason = ""
        self.n_boxes = self.n_cut_fail = 0
        if self.models_dir is None:
            where = [str(Path(models_dir))] if models_dir else [str(d) for d in DEFAULT_MODEL_DIRS]
            self.reason = ("chưa có models/fold0..4.pt (B-1' chưa chạy Kaggle — xem KhoiB/v3/README.md §3); "
                           "đã tìm: " + " | ".join(where))
            if strict:
                raise FileNotFoundError(self.reason)
            return
        try:
            import torch  # noqa: F401
        except Exception as e:                                # pragma: no cover
            self.models_dir = None
            self.reason = f"thiếu torch ({type(e).__name__}: {e})"
            if strict:
                raise
            return
        self.reason = f"mô hình: {self.models_dir}"

    # ----- trạng thái
    @property
    def available(self) -> bool:
        return self.models_dir is not None

    @property
    def dev(self):
        if self._dev is None:
            self._dev = get_device()
        return self._dev

    def model_for_page(self, book: str, page: str):
        """(net.eval(), classes, cid, fold) — mô hình fold của trang = mô hình KHÔNG học trang ấy."""
        if not self.available:
            raise RuntimeError(f"visual_emission không sẵn sàng: {self.reason}")
        fold = page_fold(book, page)
        if fold not in self._nets:
            import torch
            ck = torch.load(self.models_dir / f"fold{fold}.pt", map_location=self.dev)
            ff = ck.get("fold_formula")
            if ff is not None and ff != FOLD_FORMULA:
                raise RuntimeError(f"fold{fold}.pt ghi fold_formula khác mô-đun: {ff!r} ≠ {FOLD_FORMULA!r}")
            if ck.get("fold") is not None and int(ck["fold"]) != fold:
                raise RuntimeError(f"fold{fold}.pt ghi fold={ck['fold']} ≠ {fold}")
            classes = list(ck["classes"])
            net = build_net(len(classes)).to(self.dev)
            net.load_state_dict(ck["state"])
            net.eval()
            self._nets[fold] = (net, classes, {s: i for i, s in enumerate(classes)}, fold)
        return self._nets[fold]

    # ----- phát xạ
    def logprobs(self, gray: np.ndarray, boxes, book: str, page: str, bs: int = 512):
        """log-softmax (n_box × C) float32 của mô hình fold trang; hàng NaN khi hộp không cắt được.
        Trả (LP, classes, cid, fold)."""
        import torch
        import torch.nn.functional as F
        net, classes, cid, fold = self.model_for_page(book, page)
        n = len(boxes)
        LP = np.full((n, len(classes)), np.nan, np.float32)
        X = np.zeros((n, SZ, SZ), np.uint8)
        okm = np.zeros(n, bool)
        for i, bb in enumerate(boxes):
            g = cut_box(gray, bb) if (bb is not None and gray is not None) else None
            if g is not None:
                X[i] = g
                okm[i] = True
        self.n_boxes += n
        self.n_cut_fail += int((~okm).sum())
        idx = np.where(okm)[0]
        with torch.no_grad():
            for k in range(0, len(idx), bs):
                sel = idx[k:k + bs]
                lg, _ = net(_prep(X[sel], self.dev))
                LP[sel] = F.log_softmax(lg.float(), 1).cpu().numpy()
        return LP, classes, cid, fold

    def emission_from_LP(self, LP: np.ndarray, cid: dict, syllables, rows=None) -> np.ndarray:
        """logP (n_box × n_syl) từ LP đã tính (n_box_all × C): logP[i, j] = LP[rows[i], cid[s_j.lower()]];
        trung tính log(NEUTRAL_P) khi âm ∉ lớp hoặc hàng LP NaN (hộp không cắt được). rows = None ->
        mọi hàng theo thứ tự (benchmark rụng chữ truyền rows = chỉ số hộp còn lại)."""
        rows = np.arange(len(LP)) if rows is None else np.asarray(rows, int)
        n, m = len(rows), len(syllables)
        logP = np.full((n, m), self.neutral_logp, np.float32)
        cols = np.array([cid.get(str(s or "").lower(), -1) for s in syllables], int)
        has = ~np.isnan(LP[rows, 0]) if (LP.ndim == 2 and LP.shape[1]) else np.zeros(n, bool)
        jj = np.where(cols >= 0)[0]
        ii = np.where(has)[0]
        if len(jj) and len(ii):
            logP[np.ix_(ii, jj)] = LP[np.ix_(rows[ii], cols[jj])]
        return logP

    def emission(self, gray: np.ndarray, boxes, syllables, book: str, page: str):
        """Ma trận logP (n_box × n_syl): logP[i, j] = LP[i, cid[s_j.lower()]]; trung tính
        log(NEUTRAL_P) khi âm ∉ lớp hoặc hộp i không cắt được. Trả (logP, info) với
        info = {fold, classes, cid, LP, argmax (n_box, chuỗi), max_p (n_box)}."""
        LP, classes, cid, fold = self.logprobs(gray, boxes, book, page)
        n = len(boxes)
        logP = self.emission_from_LP(LP, cid, syllables)
        cols = np.array([cid.get(str(s or "").lower(), -1) for s in syllables], int)
        has = ~np.isnan(LP[:, 0]) if LP.shape[1] else np.zeros(n, bool)
        argmax = np.array([classes[int(LP[i].argmax())] if has[i] else "" for i in range(n)], object)
        max_p = np.array([float(np.exp(LP[i].max())) if has[i] else float("nan") for i in range(n)], np.float32)
        return logP, {"fold": fold, "classes": classes, "cid": cid, "LP": LP,
                      "argmax": argmax, "max_p": max_p, "syl_in_classes": cols >= 0}

    # ----- chi phí DP (dp_vis5.py:63, :101-104)
    def cost_matrix(self, logP: np.ndarray) -> np.ndarray:
        """λ · min(−logP, CAP) — phần ảnh của chi phí thay thế."""
        return self.lam * np.minimum(-logP, self.cap)

    def make_cost_ij(self, text_cost_fn, logP: np.ndarray):
        """cost_ij(i, j, c, s) = cost_text(c, s) + λ·min(−logP[i, j], CAP) cho realign_column/posterior_matches."""
        V = self.cost_matrix(logP)

        def cost_ij(i, j, c, s):
            return text_cost_fn(c, s) + float(V[i, j])
        return cost_ij

    def gap_costs(self) -> tuple[float, float]:
        """(cost_del, cost_ins) = COST_DEL/INS + λ·GAP_VIS (đọc hằng engine lúc gọi)."""
        return aa.COST_DEL + self.lam * self.gap_vis, aa.COST_INS + self.lam * self.gap_vis

    def describe(self) -> str:
        return (f"λ={self.lam} cap={self.cap} gap_vis={self.gap_vis} neutral_p={math.exp(self.neutral_logp):.2f} "
                f"| {self.reason}")


# --------------------------------------------------------------------------- selftest
def _selftest() -> int:
    from pipeline.align_engine.visual_emission_selftest import selftest
    return selftest()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    em = VisualEmitter(models_dir=next((a for a in sys.argv[1:] if not a.startswith("-")), None))
    print(f"visual_emission: available={em.available} | {em.describe()}")
