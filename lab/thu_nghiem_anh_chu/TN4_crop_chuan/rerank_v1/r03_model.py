#!/usr/bin/env python
"""r03_model.py — xếp hạng lại TRONG TẬP ĐÓNG R(âm) ∪ {kim} bằng ảnh, học ngưỡng/trọng số theo LOBO.

Mô hình: logit có điều kiện (softmax trên các ứng viên của MỘT ô), tuyến tính, L2, học bằng torch/CPU.
Nhãn học = ứng viên var_eq GT người (chỉ ô IHR slot_ok=1 có GT ∈ ứng viên). KHÔNG dùng nhãn pipeline làm truth.
Biến thể (đặc trưng):
  vis0   : KHÔNG học — điểm = s_faug (glyph font có biến dạng); p = softmax(s/τ) với τ=0,05 cố định
  vis    : s_font, s_faug, s_fontB, s_faugB, s_oth(+cờ), s_head(+cờ), cờ có-font, thứ hạng tương đối trong ô
  visp   : vis + tiên nghiệm tần suất log1p(n_oth) (tần suất chữ trong GOLD các bộ KHÁC — không nhìn sách đang chấm)
  full   : visp + is_kim + kim∉R           (kênh nhiễu: P(ảnh|x)·P(kim|x))
  fullh  : full + s_hum (mẫu crop NHÃN NGƯỜI; sách tune: rời-trang cùng sách; sách test: sách IHR kia)
LOBO: hướng LVT1916→TK1872 và TK1872→LVT1916. Sách tune: dự đoán out-of-fold 5 phần theo trang (để chọn ngưỡng);
sách test: mô hình học trên toàn sách tune. LVT1883/KVK1884: mô hình học trên cả hai sách IHR.
Ra: out/pred_<variant>_<dir>.pkl (1 hàng / ô: top1, p_top1, top2, p_top2, p_kim, …), out/r03_meta.json
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
SCR = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
sys.path.insert(0, str(SCR / "kim_bottleneck" / "harness"))
from harness_lib import var_eq  # noqa
OUT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/measure_out/_thu_nghiem_anh_chu/TN4/v1/rerank_new")
IHR = ("LucVanTien1916", "TruyenKieu1872")
torch.set_num_threads(3)

VARIANTS = {
    "vis": ["s_font", "s_faug", "s_fontB", "s_faugB", "s_oth0", "has_oth", "s_head0", "has_head", "has_font",
            "d_faug", "d_faugB", "d_oth", "r1_faug"],
    "visp": None, "full": None, "fullh": None}
VARIANTS["visp"] = VARIANTS["vis"] + ["lp_oth"]
VARIANTS["full"] = VARIANTS["visp"] + ["is_kim", "kim_outR"]
VARIANTS["fullh"] = VARIANTS["full"] + ["s_hum0", "has_hum", "d_hum"]
SELF = ["s_self0", "has_self", "d_self", "lp_self"]
VARIANTS["viss"] = VARIANTS["visp"] + SELF                     # + mẫu cùng sách rời trang (nhãn kim trang khác)
VARIANTS["vissh"] = VARIANTS["viss"] + ["s_hum0", "has_hum", "d_hum"]
VARIANTS["fullsh"] = VARIANTS["fullh"] + SELF


def build(cand, d, EA):
    c = cand[cand.diag == 0].copy()
    c["has_font"] = c.s_font.notna().astype(float)
    for k in ("font", "faug", "fontB", "faugB"):
        c[f"s_{k}"] = c[f"s_{k}"].fillna(c.groupby("i")[f"s_{k}"].transform("min")).fillna(0.0)
    c["has_oth"] = c.s_oth.notna().astype(float); c["s_oth0"] = c.s_oth.fillna(0.0)
    c["has_head"] = c.s_head.notna().astype(float); c["s_head0"] = c.s_head.fillna(0.0)
    c["lp_oth"] = np.log1p(c.n_oth.fillna(0).astype(float))
    # chênh so với ứng viên TỐT NHẤT KHÁC trong ô (đặc trưng tương đối)
    for k, col in (("faug", "s_faug"), ("faugB", "s_faugB"), ("oth", "s_oth")):
        v = c[col].fillna(-1.0).values
        g = c.i.values
        mx = pd.Series(v).groupby(g).transform("max").values
        # max thứ hai
        s2 = pd.Series(v).groupby(g).transform(lambda x: np.sort(x.values)[-2] if len(x) > 1 else -1.0).values
        other = np.where(v >= mx, s2, mx)
        dd = v - other
        dd[np.isnan(c[col].values)] = 0.0
        c[f"d_{k}"] = np.clip(dd, -0.5, 0.5)
    c["has_self"] = c.s_self.notna().astype(float); c["s_self0"] = c.s_self.fillna(0.0)
    c["lp_self"] = np.log1p(c.n_self.astype(float))
    v = c.s_self.fillna(-1.0).values; g = c.i.values
    mx = pd.Series(v).groupby(g).transform("max").values
    s2 = pd.Series(v).groupby(g).transform(lambda x: np.sort(x.values)[-2] if len(x) > 1 else -1.0).values
    dd = v - np.where(v >= mx, s2, mx); dd[np.isnan(c.s_self.values)] = 0.0
    c["d_self"] = np.clip(dd, -0.5, 0.5)
    c["r1_faug"] = (c.groupby("i").s_faug.rank(ascending=False, method="min") == 1).astype(float)
    c["is_kim"] = c.is_kim.astype(float)
    c["kim_outR"] = ((c.is_kim == 1) & (c.inR == 0)).astype(float)
    # truth
    gt = d.gt_char.values; slot = d.slot_ok.values
    c["y"] = [1.0 if (gt[i] and var_eq(x, gt[i])) else 0.0 for i, x in zip(c.i.values, c.ch.values)]
    c["train_ok"] = [(slot[i] == "1") for i in c.i.values]
    return c


def add_hum(c, d, EA, mode_book):
    """s_hum0/has_hum/d_hum: mẫu người. mode_book: dict book -> ('self_lpo') hoặc ('cross', [books])."""
    gt = d.gt_char.values; pages = d.page.values; books = d.book.values
    ok = (d.slot_ok == "1").values & (d.gt_char.str.len() == 1).values & (d.gt_img == d.gt_char).values
    S = {}   # (book, page, ch) -> (sum, n)
    for b in IHR:
        m = ok & (books == b)
        for i in np.nonzero(m)[0]:
            k = (b, pages[i], gt[i])
            s, n = S.get(k, (0, 0)); S[k] = (s + EA[i], n + 1)
    T = {}
    for (b, p, ch), (s, n) in S.items():
        s0, n0 = T.get((b, ch), (0, 0)); T[(b, ch)] = (s0 + s, n0 + n)
    val = np.full(len(c), np.nan)
    ii = c.i.values; chs = c.ch.values
    for r in range(len(c)):
        i = ii[r]; b = books[i]; ch = chs[r]
        how = mode_book.get(b)
        if how is None:
            continue
        if how[0] == "self_lpo":
            s, n = T.get((b, ch), (0, 0))
            sp, npg = S.get((b, pages[i], ch), (0, 0))
            s, n = s - sp, n - npg
            if n < 2:
                continue
        else:
            s, n = 0, 0
            for bb in how[1]:
                s1, n1 = T.get((bb, ch), (0, 0)); s, n = s + s1, n + n1
            if n < 2:
                continue
        v = s / (np.linalg.norm(s) + 1e-9)
        val[r] = float(EA[i] @ v)
    c = c.copy()
    c["has_hum"] = (~np.isnan(val)).astype(float); c["s_hum0"] = np.nan_to_num(val)
    v = np.where(np.isnan(val), -1.0, val)
    g = c.i.values
    mx = pd.Series(v).groupby(g).transform("max").values
    s2 = pd.Series(v).groupby(g).transform(lambda x: np.sort(x.values)[-2] if len(x) > 1 else -1.0).values
    dd = v - np.where(v >= mx, s2, mx); dd[np.isnan(val)] = 0.0
    c["d_hum"] = np.clip(dd, -0.5, 0.5)
    return c


def seg_logsoftmax(logit, seg, nseg):
    mx = torch.full((nseg,), -1e9).scatter_reduce(0, seg, logit, "amax")
    e = torch.exp(logit - mx[seg])
    z = torch.zeros(nseg).index_add(0, seg, e)
    return logit - mx[seg] - torch.log(z[seg])


def fit(c, feats, l2=1e-3, steps=400):
    c = c[c.train_ok]
    pos = c.groupby("i").y.transform("sum")
    c = c[pos > 0]
    X = torch.tensor(c[feats].values, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    cells, seg = np.unique(c.i.values, return_inverse=True)
    seg = torch.tensor(seg); ns = len(cells)
    y = torch.tensor(c.y.values, dtype=torch.float32)
    y = y / torch.zeros(ns).index_add(0, seg, y)[seg]
    w = torch.zeros(len(feats), requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lp = seg_logsoftmax(X @ w, seg, ns)
        loss = -(y * lp).sum() / ns + l2 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return dict(w=w.detach(), mu=mu, sd=sd, feats=feats, n_cells=ns)


def predict(m, c):
    X = (torch.tensor(c[m["feats"]].values, dtype=torch.float32) - m["mu"]) / m["sd"]
    cells, seg = np.unique(c.i.values, return_inverse=True)
    lp = seg_logsoftmax(X @ m["w"], torch.tensor(seg), len(cells))
    return np.exp(lp.detach().numpy())


def vis0_prob(c, tau=0.05):
    cells, seg = np.unique(c.i.values, return_inverse=True)
    x = torch.tensor(c.s_faug.values / tau, dtype=torch.float32)
    return np.exp(seg_logsoftmax(x, torch.tensor(seg), len(cells)).numpy())


def summarize(c, p, d):
    c = c.assign(p=p)
    # gộp ứng viên CÙNG HÌNH glyph (md5 render trùng) với kim: coi như kim
    kim = d.ocr_char.values
    kmd5 = c[c.is_kim == 1].set_index("i").gmd5.to_dict()
    same = np.array([(g != "" and kmd5.get(i, "") == g) for i, g in zip(c.i.values, c.gmd5.values)])
    c = c.assign(same_kim=same | (c.is_kim.values == 1))
    c = c.sort_values(["i", "p"], ascending=[True, False])
    rows = []
    for i, g in c.groupby("i", sort=False):
        chs = g.ch.values; ps = g.p.values; sk = g.same_kim.values
        pk = float(ps[sk].sum())
        # ứng viên khác kim tốt nhất
        o = np.nonzero(~sk)[0]
        alt = chs[o[0]] if len(o) else ""; palt = float(ps[o[0]]) if len(o) else 0.0
        top1 = kim[i] if sk[0] else chs[0]
        ptop = pk if sk[0] else float(ps[0])
        rows.append((i, top1, ptop, pk, alt, palt, len(chs), int(sk[0])))
    return pd.DataFrame(rows, columns=["i", "top1", "p_top1", "p_kim", "alt", "p_alt", "n_cand", "top_is_kim"])


def main():
    T0 = time.time()
    d = pd.read_csv(SCR / "kim_bottleneck/harness/cells_eval.csv", dtype=str, keep_default_na=False)
    EA = np.load(OUT / "E_A.f16.npy").astype(np.float32)
    cand = pd.read_pickle(OUT / "cand.pkl")
    cs_ = pd.read_pickle(OUT / "cand_self.pkl"); cand["s_self"] = cs_.s_self.values; cand["n_self"] = cs_.n_self.values
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    c = build(cand, d, EA)
    print(f"[build] {len(c)} hàng {time.time()-T0:.0f}s", flush=True)
    book = d.book.values
    c["book"] = book[c.i.values]
    page = d.page.values
    meta = {"weights": {}}
    dirs = [("LucVanTien1916", "TruyenKieu1872"), ("TruyenKieu1872", "LucVanTien1916"), ("IHR2", "LITHO")]
    for tune, test in dirs:
        if tune == "IHR2":
            ctr = c[c.book.isin(IHR)]; cte = c[c.book.isin(["LucVanTien1883", "KimVanKieu1884"])]
            hum_tr = {b: ("self_lpo",) for b in IHR}; hum_te = {b: ("cross", list(IHR)) for b in ["LucVanTien1883", "KimVanKieu1884"]}
        else:
            ctr = c[c.book == tune]; cte = c[c.book == test]
            hum_tr = {tune: ("self_lpo",)}; hum_te = {test: ("cross", [tune])}
        ctr = add_hum(ctr, d, EA, hum_tr); cte = add_hum(cte, d, EA, hum_te)
        # gấp theo trang cho OOF trên sách tune
        pg = pd.Series(page[ctr.i.values])
        upg = sorted(pg.unique()); fold_of = {p: k % 5 for k, p in enumerate(upg)}
        fold = pg.map(fold_of).values
        for v in ["vis0"] + list(VARIANTS):
            if only and v not in only:
                continue
            if v == "vis0":
                p_te = vis0_prob(cte); p_tr = vis0_prob(ctr)
            else:
                f = VARIANTS[v]
                m = fit(ctr, f)
                meta["weights"][f"{v}|{tune}"] = dict(zip(f, [round(float(x), 3) for x in (m["w"] / m["sd"])]))
                p_te = predict(m, cte)
                p_tr = np.zeros(len(ctr))
                for k in range(5):
                    mk = fit(ctr[fold != k], f)
                    sel = fold == k
                    p_tr[sel] = predict(mk, ctr[sel])
            s_te = summarize(cte, p_te, d); s_tr = summarize(ctr, p_tr, d)
            s_te["role"] = "test"; s_tr["role"] = "tune_oof"
            s = pd.concat([s_tr, s_te], ignore_index=True)
            s["cell_uid"] = d.cell_uid.values[s.i.values]
            s["book"] = book[s.i.values]
            tag = "LITHO" if tune == "IHR2" else f"{tune[:3]}2{test[:3]}"
            s.to_pickle(OUT / f"pred_{v}_{tag}.pkl")
            print(f"[{v} {tag}] {time.time()-T0:.0f}s", flush=True)
    meta["t_total"] = round(time.time() - T0)
    mf = OUT / "r03_meta.json"
    if mf.exists():
        old = json.load(open(mf)); old["weights"].update(meta["weights"]); meta["weights"] = old["weights"]
    json.dump(meta, open(OUT / "r03_meta.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
