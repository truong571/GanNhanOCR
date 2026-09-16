"""KÊNH THỊ GIÁC GIÁM SÁT BẰNG ÂM TIẾT — thí nghiệm hướng tấn công mới (2026-09-14).

Khác mọi mẻ trước: mô hình KHÔNG học "chữ nào trong 4.700 lớp" từ nhãn OCR (nhiễm thiên lệch URO),
mà học P(ÂM TIẾT | crop) từ nhãn âm do căn chỉnh sinh ra (~98% đúng, độc lập với OCR). Rồi dùng
embedding của nó cho ba việc chỉ ảnh mới làm được:

    crops     cắt 82.780 ô từ trang gốc theo bbox (đồng nhất, 64×64)          -> crops.npz
    train     CNN nhỏ, lớp = âm tiết (≥5 ô train), chia theo TRANG           -> model.pt
    eval      E1: top-1/top-5 âm tiết trên trang test (ảnh nói được âm gì?)
    gap       E2: (a) hộp thiếu: DP hộp↔âm bằng phát xạ thị giác; (b) thanh ghi trượt: P(âm|crop) bắt ô sai
    cluster   E3: 128 lớp (sách, âm, nhãn thiểu số nhìn giống đa số): một cụm hay hai cụm?
    propagate E4: ô chưa có mã chữ có "hàng xóm" GOLD-trực-tiếp cùng (sách, âm) đủ gần không
                  (kiểm chứng trên ô cầu tự dạng, giữ lại theo trang)
    cluster_all E3 mở rộng: MỌI lớp (sách, âm, nhãn thiểu số ≥8 ô) + gom cụm 'người' theo sách

    .venv/bin/python lab/gan_nhan_2026-09-13/thi_giac_am_tiet.py all
"""
from __future__ import annotations

import json
import math
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from core.text.dictionary import load_qn_to_nom, load_similarity_dict   # noqa: E402
from pipeline.align_engine.bbox_fix import tighten_box                   # noqa: E402

BOOKDIR = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}
SZ = 64
CROPS = HERE / "crops.npz"
MODEL = HERE / "model.pt"
EMB = HERE / "emb.npz"


def h(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72, flush=True)


# --------------------------------------------------------------------------- crops
def cut(img, bbox, pad=0.08):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    a, b = max(0, x1 - pw), max(0, y1 - ph)
    c, d = min(W, x2 + pw), min(H, y2 + ph)
    g = img[b:d, a:c]
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


def cmd_crops():
    df = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
    df["bb"] = df.bbox.apply(json.loads)
    X = np.zeros((len(df), SZ, SZ), np.uint8)
    ok = np.zeros(len(df), bool)
    t0 = time.time()
    for (book, page), g in df.groupby(["book", "page"]):
        img = cv2.imread(str(REPO / "prepared" / BOOKDIR[book] / "pages" / f"{page}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for idx, bb in zip(g.index, g.bb):
            c = cut(img, bb)
            if c is not None:
                X[idx] = c
                ok[idx] = True
    np.savez_compressed(CROPS, X=X, ok=ok)
    print(f"crops: {ok.sum()}/{len(df)} ô cắt được, {time.time()-t0:.0f}s -> {CROPS}")


def load_all():
    df = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
    # A-10 (16/09): bộ nhãn v3 không còn cột split — tự chia theo TRANG bằng đúng công thức
    # cũ của build_dataset (≤25/08) để E1/E2 tái lập nguyên xi trên cả hai thế hệ.
    if "split" not in df.columns:
        import hashlib
        key = df["book"] + "|" + df["page"]
        hv = key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 100)
        df["split"] = np.where(hv < 80, "train", np.where(hv < 90, "val", "test"))
    z = np.load(CROPS)
    return df, z["X"], z["ok"]


# --------------------------------------------------------------------------- model
def build_net(n_cls, emb=256):
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.e = nn.Sequential(nn.Linear(256, emb), nn.BatchNorm1d(emb))
            self.c = nn.Linear(emb, n_cls)

        def forward(self, x):
            z = self.e(self.f(x))
            return self.c(z), z
    return Net()


def _prep_batch(X, idx, dev, aug=False):
    import torch
    x = X[idx].astype(np.float32) / 255.0
    if aug:
        # nhiễu hình học nhẹ: dịch ±3 px, co giãn ±8% — chữ viết tay, không lật
        out = np.empty_like(x)
        for k in range(len(idx)):
            s = 1 + random.uniform(-0.08, 0.08)
            M = np.float32([[s, 0, random.uniform(-3, 3) + (1 - s) * SZ / 2], [0, s, random.uniform(-3, 3) + (1 - s) * SZ / 2]])
            out[k] = cv2.warpAffine(x[k], M, (SZ, SZ), borderValue=1.0)
        x = out
    return torch.from_numpy(1.0 - x).unsqueeze(1).to(dev)      # mực = 1, nền = 0


def cmd_train(epochs=18, bs=256):
    import torch
    import torch.nn.functional as F
    df, X, ok = load_all()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # nhãn = âm tiết chuẩn hoá; huấn luyện trên GOLD+SYLLABLE (âm tin được), trang train
    tr = df[(df.split == "train") & df.tier.isin(["GOLD", "SYLLABLE"]) & ok].copy()
    cnt = Counter(tr.syllable)
    classes = sorted(s for s, n in cnt.items() if n >= 5)
    cid = {s: i for i, s in enumerate(classes)}
    tr = tr[tr.syllable.isin(cid)]
    y = np.array([cid[s] for s in tr.syllable])
    idx_all = tr.index.to_numpy()
    print(f"train: {len(tr)} ô · {len(classes)} âm tiết (≥5 ô) · device {dev}")
    net = build_net(len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    # tần suất lớp lệch mạnh -> label smoothing + lấy mẫu cân bằng nhẹ (sqrt)
    w = np.array([1 / math.sqrt(cnt[s]) for s in tr.syllable]); w /= w.sum()
    va = df[(df.split == "val") & (df.tier == "GOLD") & ok & df.syllable.isin(cid)]
    va_idx = va.index.to_numpy(); va_y = np.array([cid[s] for s in va.syllable])
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        order = np.random.default_rng(ep).choice(len(tr), size=len(tr), replace=True, p=w)
        tot = 0.0
        for k in range(0, len(order), bs):
            sel = order[k:k + bs]
            xb = _prep_batch(X, idx_all[sel], dev, aug=True)
            yb = torch.from_numpy(y[sel]).to(dev)
            logits, _ = net(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(sel)
        net.eval(); correct = 0
        with torch.no_grad():
            for k in range(0, len(va_idx), 1024):
                lg, _ = net(_prep_batch(X, va_idx[k:k + 1024], dev))
                correct += (lg.argmax(1).cpu().numpy() == va_y[k:k + 1024]).sum()
        print(f"  epoch {ep+1:2d}/{epochs} loss {tot/len(order):.3f} · val top-1 {correct/len(va_idx):.3f} · {time.time()-t0:.0f}s", flush=True)
    torch.save({"state": net.state_dict(), "classes": classes}, MODEL)
    print("->", MODEL)


def load_model():
    import torch
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(MODEL, map_location=dev)
    net = build_net(len(ck["classes"])).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    return net, ck["classes"], dev


def cmd_embed():
    """logits (log-softmax) + embedding cho MỌI ô -> emb.npz"""
    import torch
    df, X, ok = load_all()
    net, classes, dev = load_model()
    N = len(df)
    Z = np.zeros((N, 256), np.float32); LP = np.zeros((N, len(classes)), np.float16)
    with torch.no_grad():
        for k in range(0, N, 1024):
            idx = np.arange(k, min(N, k + 1024))
            lg, z = net(_prep_batch(X, idx, dev))
            Z[idx] = torch.nn.functional.normalize(z, dim=1).cpu().numpy()
            LP[idx] = torch.log_softmax(lg, 1).cpu().numpy().astype(np.float16)
    np.savez_compressed(EMB, Z=Z, LP=LP, classes=np.array(classes))
    print("->", EMB)


def load_emb():
    z = np.load(EMB, allow_pickle=True)
    return z["Z"], z["LP"].astype(np.float32), list(z["classes"])


# --------------------------------------------------------------------------- E1
def cmd_eval():
    df, X, ok = load_all()
    Z, LP, classes = load_emb()
    cid = {s: i for i, s in enumerate(classes)}
    h("E1 · ẢNH NÓI ĐƯỢC ÂM GÌ — top-k âm tiết trên trang TEST (chia theo trang)")
    for name, mask in [("GOLD-trực-tiếp", (df.tier == "GOLD") & (df.rule == "s1_inter_s2_direct")),
                       ("GOLD-cầu tự dạng", (df.tier == "GOLD") & df.rule.str.startswith("s1_inter_s2_similar")),
                       ("SYLLABLE", df.tier == "SYLLABLE"), ("REVIEW", df.tier == "REVIEW")]:
        m = mask & (df.split == "test") & ok & df.syllable.isin(cid)
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        y = np.array([cid[s] for s in df.syllable.iloc[idx]])
        top = np.argsort(-LP[idx], axis=1)[:, :5]
        t1 = (top[:, 0] == y).mean(); t5 = (top == y[:, None]).any(1).mean()
        rank = (np.argsort(np.argsort(-LP[idx], axis=1), axis=1)[np.arange(len(idx)), y] + 1)
        print(f"  {name:18s} n={len(idx):5d}  top-1 {t1:.3f}  top-5 {t5:.3f}  hạng trung vị {int(np.median(rank))}")
    # đối chứng: "âm phổ biến nhất" vô nghĩa ở đây (mọi âm đều là lớp); ghi thêm entropy
    print("  (REVIEW: âm do căn chỉnh gán có thể sai ở cột lệch — top-1 thấp hơn là kỳ vọng)")


# --------------------------------------------------------------------------- E2
def _visual_dp(LPcol, n_syl, gap_pen):
    """DP hộp(M) ↔ âm(N) với phát xạ log P(âm_j | crop_i); trả về tập âm KHÔNG có hộp."""
    M = LPcol.shape[0]; N = n_syl
    NEG = -1e18
    dp = np.full((M + 1, N + 1), NEG); bt = np.zeros((M + 1, N + 1), np.int8)
    dp[0, 0] = 0.0
    for i in range(M + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                continue
            best, op = NEG, 0
            if i > 0 and j > 0 and dp[i - 1, j - 1] + LPcol[i - 1, j - 1] > best:
                best, op = dp[i - 1, j - 1] + LPcol[i - 1, j - 1], 1
            if j > 0 and dp[i, j - 1] - gap_pen > best:          # âm j-1 không có hộp
                best, op = dp[i, j - 1] - gap_pen, 2
            if i > 0 and dp[i - 1, j] - gap_pen > best:          # hộp i-1 không có âm (hộp giả)
                best, op = dp[i - 1, j] - gap_pen, 3
            dp[i, j], bt[i, j] = best, op
    i, j = M, N; missing = []
    while i > 0 or j > 0:
        op = bt[i, j]
        if op == 1: i -= 1; j -= 1
        elif op == 2: j -= 1; missing.append(j)
        else: i -= 1
    return missing


def cmd_gap():
    """E2a · hộp bị THIẾU (detector bỏ sót 1 glyph): DP hộp↔âm bằng phát xạ thị giác định vị được âm
    không có hộp không?  E2b · thanh ghi TRƯỢT (từ vị trí k, ô nhận âm của ô kế): P(âm gán | crop)
    có bắt được các ô ghép sai không?"""
    from thuc_nghiem import load_cols
    df, X, ok = load_all()
    Z, LP, classes = load_emb()
    cid = {s: i for i, s in enumerate(classes)}
    cols = load_cols()
    rng = random.Random(5)
    test_pages = set(map(tuple, df[df.split == "test"][["book", "page"]].drop_duplicates().itertuples(index=False)))
    eq = [c for c in cols if len(c["chars"]) == len(c["syl"]) >= 10 and c["rows"] is not None
          and len(c["rows"]) == len(c["syl"]) and all(o["op"] == "match" for o in c["ops"])
          and (c["book"], c["page"]) in test_pages]
    h(f"E2 · THỊ GIÁC CHO THANH GHI — {len(eq)} cột khớp số trên trang TEST (chưa từng dùng để huấn luyện)")
    n = 0; hit = Counter(); chance = 0.0
    aucs = []; det = Counter()
    for col in eq:
        syl, rows = col["syl"], col["rows"]
        if not all(ok[r] for r in rows) or not all(s.lower() in cid for s in syl):
            continue
        n += 1; N = len(syl)
        L = np.array([[LP[rows[i], cid[syl[j].lower()]] for j in range(N)] for i in range(N)])   # crop i × âm j
        # E2a: bỏ hộp k
        k = rng.randrange(1, N - 1)
        Lm = np.delete(L, k, axis=0)
        for pen in (4.0, 8.0, 12.0, 16.0, 24.0):
            miss = _visual_dp(Lm, N, pen)
            hit[pen] += (miss == [k])
        chance += 1.0 / N
        # E2b: trượt thanh ghi từ k: ô i>=k nhận âm syl[i+1] (ô cuối bỏ)
        k2 = rng.randrange(3, N - 3)
        p_right = [float(np.exp(L[i, i])) for i in range(0, k2)]
        p_wrong = [float(np.exp(L[i, i + 1])) for i in range(k2, N - 1)]
        for a in p_wrong:
            det["wrong"] += 1; det["wrong_flag"] += a < 0.05
        for a in p_right:
            det["right"] += 1; det["right_flag"] += a < 0.05
        pr = np.array(p_right); pw = np.array(p_wrong)
        aucs.append(np.mean([1.0 if w < r else 0.5 if w == r else 0.0 for w in pw for r in pr]))
    print(f"  E2a · hộp thiếu 1: DP thị giác chỉ đúng âm không có hộp:  "
          + " · ".join(f"phạt {pen}: {hit[pen]/n:.1%}" for pen in (4.0, 8.0, 12.0, 16.0, 24.0)) + f"   (đoán mò {chance/n:.1%})")
    print(f"  E2b · thanh ghi trượt: AUC bắt ô ghép sai bằng P(âm gán | crop) = {np.mean(aucs):.3f}; "
          f"ngưỡng p<0,05: bắt {det['wrong_flag']/det['wrong']:.0%} ô sai · oan {det['right_flag']/det['right']:.1%} ô đúng")
    print("  (ô 'đúng' ở đây là cặp của bộ hiện hành — bản thân nó có ~1–2% sai, nên 'oan' là cận trên)")




def _hog(imgs, cell=8, bins=9):
    """HOG thô, không học: gradient 4 hướng có dấu -> 9 bin, ô 8×8, chuẩn hoá L2 toàn vector."""
    out = []
    for im in imgs:
        g = im.astype(np.float32) / 255.0
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy); ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi) * bins
        b = np.floor(ang).astype(int) % bins
        feat = np.zeros((SZ // cell, SZ // cell, bins), np.float32)
        for i in range(SZ // cell):
            for j in range(SZ // cell):
                m = mag[i*cell:(i+1)*cell, j*cell:(j+1)*cell]; bb = b[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
                feat[i, j] = np.bincount(bb.ravel(), weights=m.ravel(), minlength=bins)
        v = feat.ravel(); v = np.sqrt(v / (v.sum() + 1e-6))
        out.append(v / (np.linalg.norm(v) + 1e-6))
    return np.array(out)

# --------------------------------------------------------------------------- E3
def cmd_cluster():
    df, X, ok = load_all()
    Z, LP, classes = load_emb()
    sim = load_similarity_dict(str(REPO / "dict/SinoNom_Similar.csv"))
    chk_path = REPO / "re-dataset/check/labels.xlsx"
    chk = pd.read_excel(chk_path, sheet_name="Nhan", dtype=str)[["image", "dung_sai"]] if chk_path.exists() else None
    if chk is not None:
        df = df.merge(chk, on="image", how="left")
        df["nghi_sai"] = df.dung_sai.astype(str).str.lower().isin(["0", "0.0", "false"])
    h("E3 · CÙNG (sách, âm): nhãn thiểu số nhìn giống nhãn đa số — MỘT hình hay HAI hình?")
    print("  phép đo: k-NN (k=5, loại-một-ô) trên embedding giữa hai nhóm nhãn; 'tách được' = độ chính xác k-NN "
          "vượt xa mức đoán theo tỉ lệ; 'một hình' = k-NN không hơn đoán -> OCR đọc nhầm cùng một glyph.")
    g = df[(df.tier == "GOLD") & ok]
    rows = []
    for (b, s), grp in g.groupby(["book", "syllable"]):
        c = Counter(grp.label)
        if len(c) < 2:
            continue
        M = c.most_common(1)[0][0]
        for L, n in c.items():
            if L == M or n < 8:
                continue
            if not ((L in sim.get(M, [])) or (M in sim.get(L, []))):
                continue
            A = grp[grp.label == M].index.to_numpy(); B = grp[grp.label == L].index.to_numpy()
            if len(A) > 200:
                A = np.random.default_rng(0).choice(A, 200, replace=False)
            idx = np.concatenate([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
            E = Z[idx]; S = E @ E.T; np.fill_diagonal(S, -9)
            nn_ = np.argsort(-S, axis=1)[:, :5]
            pred = (y[nn_].mean(1) > 0.5).astype(float)
            # balanced accuracy (hai nhóm lệch cỡ)
            bacc = 0.5 * ((pred[y == 0] == 0).mean() + (pred[y == 1] == 1).mean())
            # ý kiến thứ hai: HOG KHÔNG giám sát (không thể bị "ép" bởi nhãn âm)
            Hg = _hog(X[idx]); Sh = Hg @ Hg.T; np.fill_diagonal(Sh, -9)
            nnh = np.argsort(-Sh, axis=1)[:, :5]
            ph = (y[nnh].mean(1) > 0.5).astype(float)
            hbacc = 0.5 * ((ph[y == 0] == 0).mean() + (ph[y == 1] == 1).mean())
            # đối chứng hoán vị nhãn
            perm = []
            for r in range(20):
                yp = np.random.default_rng(r).permutation(y)
                pp = (yp[nn_].mean(1) > 0.5).astype(float)
                perm.append(0.5 * ((pp[yp == 0] == 0).mean() + (pp[yp == 1] == 1).mean()))
            chk_s = grp[grp.label == L].nghi_sai.mean() if "nghi_sai" in grp else float("nan")
            rows.append(dict(book=b, syllable=s, major=M, nA=len(A), minor=L, nB=len(B), knn_bacc=round(bacc, 2), hog_bacc=round(hbacc, 2),
                             perm_bacc=round(float(np.mean(perm)), 2), z=round((bacc - np.mean(perm)) / (np.std(perm) + 1e-6), 1),
                             ext_chk=round(chk_s, 2)))
    R = pd.DataFrame(rows).sort_values("nB", ascending=False)
    R["verdict"] = np.where((R.knn_bacc >= 0.75) | (R.hog_bacc >= 0.75), "HAI HÌNH (dị thể thật)",
                            np.where((R.knn_bacc <= 0.6) & (R.hog_bacc <= 0.6), "MỘT HÌNH (OCR nhầm)", "mờ"))
    print(R.to_string(index=False))
    print("\n  tổng kết:", R.verdict.value_counts().to_dict())
    R.to_csv(HERE / "e3_clusters.csv", index=False)
    # 㝵/𠊚
    ng = df[(df.syllable == "người") & ok]
    print(f"\n  'người': {len(ng)} ô có crop · nhãn: {Counter(ng.label).most_common(5)} · ocr_char: {Counter(ng.ocr_char).most_common(5)}")


# --------------------------------------------------------------------------- E4
def cmd_propagate():
    df, X, ok = load_all()
    Z, LP, classes = load_emb()
    qn = load_qn_to_nom(str(REPO / "dict/QuocNgu_SinoNom.csv"))
    h("E4 · LAN NHÃN THEO CỤM cùng (sách, âm) — kiểm trên ô CẦU TỰ DẠNG (OCR đọc sai, có nhãn độc lập)")
    direct = df[(df.tier == "GOLD") & (df.rule == "s1_inter_s2_direct") & ok]
    proto = {}
    for (b, s), grp in direct.groupby(["book", "syllable"]):
        proto[(b, s)] = (grp.index.to_numpy(), grp.label.to_numpy())
    # kiểm chứng: ô cầu tự dạng trên trang test; nhãn cầu làm "sự thật thay thế" (bản thân 18,7% nghi sai)
    tests = df[(df.tier == "GOLD") & df.rule.str.startswith("s1_inter_s2_similar") & ok & (df.split == "test")]
    res = Counter(); margins = []
    for i, r in tests.iterrows():
        key = (r.book, r.syllable)
        if key not in proto:
            res["no_proto"] += 1; continue
        pidx, plab = proto[key]
        keep = df.page.iloc[pidx].to_numpy() != r.page          # loại cùng trang
        pidx, plab = pidx[keep], plab[keep]
        if len(pidx) < 3:
            res["no_proto"] += 1; continue
        sims = Z[pidx] @ Z[i]
        order = np.argsort(-sims)[:5]
        votes = Counter(plab[order]); lab, v = votes.most_common(1)[0]
        top = float(sims[order[0]])
        res["n"] += 1
        res["in_pool"] += r.label in set(plab)
        res["hit"] += (lab == r.label)
        res["hit_conf"] += (lab == r.label) and (v >= 4) and (top >= 0.6)
        res["fire_conf"] += (v >= 4) and (top >= 0.6)
        res["freq_hit"] += (Counter(plab).most_common(1)[0][0] == r.label)
    n = res["n"]
    print(f"  ô kiểm: {n} (+{res['no_proto']} không có nguyên mẫu cùng sách-âm)")
    print(f"  nhãn cầu có trong bể nguyên mẫu: {res['in_pool']/n:.1%}   (trần tồn kho)")
    print(f"  k-NN(5) đúng nhãn cầu: {res['hit']/n:.1%}   · đối chứng 'chữ phổ biến nhất của (sách, âm)': {res['freq_hit']/n:.1%}")
    print(f"  cổng tin cậy (≥4/5 phiếu & cos≥0,6): bắn {res['fire_conf']/n:.1%} · đúng {res['hit_conf']/max(1,res['fire_conf']):.1%}")
    # bao nhiêu ô KHÔNG có mã chữ có hàng xóm tin cậy
    pool = df[df.tier.isin(["REVIEW", "SYLLABLE", "SILVER_uncalibrated"]) & ok]
    fire = 0; tot = 0
    for i, r in pool.iterrows():
        key = (r.book, r.syllable)
        if key not in proto:
            continue
        pidx, plab = proto[key]
        keep = df.page.iloc[pidx].to_numpy() != r.page
        pidx, plab = pidx[keep], plab[keep]
        if len(pidx) < 3:
            continue
        tot += 1
        sims = Z[pidx] @ Z[i]; order = np.argsort(-sims)[:5]
        v = Counter(plab[order]).most_common(1)[0][1]
        fire += (v >= 4) and (float(sims[order[0]]) >= 0.6)
    print(f"\n  ô chưa có mã chữ có nguyên mẫu cùng (sách, âm) ≥3 ở trang khác: {tot} · qua cổng tin cậy: {fire} ({fire/max(1,tot):.1%})")



# --------------------------------------------------------------------------- E3 mở rộng
def cmd_cluster_all():
    df, X, ok = load_all()
    Z, LP, classes = load_emb()
    sim = load_similarity_dict(str(REPO / "dict/SinoNom_Similar.csv"))
    h("E3 MỞ RỘNG · mọi lớp (sách, âm, nhãn thiểu số ≥8 ô) — một hình hay hai hình?")
    g = df[(df.tier == "GOLD") & ok]
    rows = []
    for (b, s), grp in g.groupby(["book", "syllable"]):
        c = Counter(grp.label)
        if len(c) < 2:
            continue
        M = c.most_common(1)[0][0]
        for L, n in c.items():
            if L == M or n < 8:
                continue
            A = grp[grp.label == M].index.to_numpy(); B = grp[grp.label == L].index.to_numpy()
            if len(A) > 200:
                A = np.random.default_rng(0).choice(A, 200, replace=False)
            idx = np.concatenate([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]

            def bacc(F):
                S = F @ F.T; np.fill_diagonal(S, -9); nn_ = np.argsort(-S, axis=1)[:, :5]
                p = (y[nn_].mean(1) > 0.5)
                return 0.5 * ((p[y == 0] == 0).mean() + (p[y == 1] == 1).mean())
            kb, hb = bacc(Z[idx]), bacc(_hog(X[idx]))
            look = (L in sim.get(M, [])) or (M in sim.get(L, []))
            v = "HAI" if (kb >= 0.75 or hb >= 0.75) else ("MOT" if (kb <= 0.6 and hb <= 0.6) else "mo")
            rows.append(dict(book=b, syl=s, M=M, L=L, nB=n, look=look, knn=round(kb, 2), hog=round(hb, 2), v=v))
    R = pd.DataFrame(rows)
    print(f"lớp: {len(R)} · ô: {R.nB.sum()}")
    print(R.groupby(["look", "v"]).agg(lop=("nB", "size"), o=("nB", "sum")))
    print("\nMỘT HÌNH nhưng KHÔNG có trong SinoNom_Similar (từ điển tự dạng thiếu cặp):")
    print(R[(~R.look) & (R.v == "MOT")].sort_values("nB", ascending=False).head(8).to_string(index=False))
    R.to_csv(HERE / "e3_all_classes.csv", index=False)

    h("'người' — gom cụm 2.281 crop: cụm đi theo SÁCH hay theo chữ OCR?")
    ng = df[(df.syllable == "người") & ok]
    E = Z[ng.index.to_numpy()]
    rng = np.random.default_rng(0)
    for k in (2, 3, 4):
        C = E[rng.choice(len(E), k, replace=False)]
        for _ in range(30):
            a = np.argmax(E @ C.T, 1)
            C = np.array([E[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            C /= np.linalg.norm(C, axis=1, keepdims=True)
        print(f"k={k}:")
        for j in range(k):
            m = a == j
            print(f"   cụm {j}: {m.sum():5d} ô · ocr_char {Counter(ng.ocr_char[m]).most_common(3)} · sách {Counter(ng.book[m]).most_common(3)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"crops": cmd_crops, "train": cmd_train, "embed": cmd_embed, "eval": cmd_eval,
             "gap": cmd_gap, "cluster": cmd_cluster, "propagate": cmd_propagate, "cluster_all": cmd_cluster_all}
    if cmd == "all":
        for k in ("crops", "train", "embed", "eval", "gap", "cluster", "cluster_all", "propagate"):
            steps[k]()
    else:
        steps[cmd]()
