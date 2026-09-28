"""visual_dp — bộ giải mã HỘP ẢNH theo TRANG bằng quy hoạch động + phát xạ THỊ GIÁC (box_decoder: visual_dp, TN6 2026-09-28).

Bật qua khoá config `books[].box_decoder: visual_dp` (mặc định "legacy"; sách không khai KHÔNG đổi một byte).
Tài liệu + số đo: docs/HOP_ANH_TN6_2026-09-28.md; thử nghiệm: lab/thu_nghiem_anh_chu/TN6_hop_anh/.

Vì sao: `pitch` ép N hộp theo BƯỚC trong khung tầng dựng từ hộp chữ kim (kim chia đều trong dòng). Trên chữ viết tay
(Borg) kim bỏ sót ~23 % chữ và khung dòng kim hụt/đứt ⇒ hộp trượt ±1 ô (đúng vị trí so với hộp người chỉ 61–74 %).
`pipeline.borg_human` đã chứng minh (trên chữ NGƯỜI) rằng gióng CHUỖI CHỮ vào ĐƠN VỊ ẢNH (hộp detector + ô ảo, cột
phải→trái) bằng DP đơn điệu có phát xạ thị giác cho hộp đúng (Paddle độc lập chọn cùng hộp 97,3 %). Ở đây chuỗi là
ÂM QN của trang (kênh QN của pipeline, KHÔNG dùng chữ Nôm người) và tham chiếu thị giác của một âm là TẬP ỨNG VIÊN
R(âm) = Dict/QuocNgu_SinoNom.csv (glyph font NomNaTong/Plangothic + ảnh FD) — không phụ thuộc chữ kim (sai ~50 % trên
chữ viết tay).

Cách làm (mỗi sách, sau PASS 1 của build_dataset, trước PASS 1b):
  1. Đơn vị trang = hộp detector (ứng viên ≥ 0,05 của chính lượt pitch) gom cột phải→trái + ô ảo ở khe lớn
     (pipeline.borg_human.geom.units_for_page); nhúng crop bằng encoder [nom-embed v1, ArcFace v2]/√2 (MultiEnc).
  2. Chuỗi mục = âm QN các cột của trang (cột xếp phải→trái theo tâm x của cột kim), mỗi mục mang dải x cột kim.
  3. Phát xạ z(u, j) = chuẩn hoá theo hàng của max_{c ∈ R(âm_j)} cos(emb(u), glyph(c)); tiên nghiệm hình học MỀM:
     đơn vị ngoài dải x cột kim ± col_margin·w_med bị trừ lam_col (thang z) — kim đúng về PHẠM VI dòng.
  4. DP đơn điệu Viterbi + forward-backward (chép pipeline.borg_human.align): ghép hộp / ô ảo / gộp 2 hộp liền / bỏ hộp /
     bỏ mục; hậu nghiệm ghép = độ tin.
  5. Lượt nguyên mẫu (tự học, không nhãn): nguyên mẫu (khối trang f, âm) = trung bình nhúng đơn vị THẬT hậu nghiệm ≥ 0,9
     của lượt trước, CHỈ từ 4 khối trang KHÁC (5 khối liền nhau/sách) ⇒ phát xạ = max(R-glyph, nguyên mẫu). 1 lượt.
  6. Ghi lại vào trạng thái cột: G = hộp theo syl_idx (len n_qn), box_rule 'pitch' (PASS 1b gán lại theo syl_idx của ops
     lượt 2 bằng assign_boxes_pitch), count_source 'visual_dp'. box_source theo ô:
        vdp_det      đơn vị thật (hậu nghiệm ≥ LOW_POST)          vdp_merge  gộp 2 hộp (hậu nghiệm ≥ LOW_POST)
        vdp_agree    hậu nghiệm < LOW_POST hoặc ô ảo, nhưng hộp TRÙNG hộp pitch của cùng âm (IoU ≥ 0,5) -> tin
        vdp_low      hậu nghiệm < LOW_POST (0,95), lệch hộp pitch      vdp_virtual  ô ảo, lệch hộp pitch
        vdp_fallback mục bị DP bỏ -> hộp pitch/PASS 1 của cùng âm
     Ba giá trị cuối là "hộp không tự tin" cho cổng (a') của mechanism_gates (như ink_cut/detector_low của pitch).
Tham số chọn trên Borg (TN6 x02: chọn trên Kinh báo DungLy và ngược lại — cùng một cấu hình thắng cả hai).
"""
from __future__ import annotations

import time

import numpy as np

# ---- tham số ĐÓNG BĂNG (TN6 x02/catcol3q): DP = pipeline.borg_human.params.DP; enc cat; lam_col 3; 1 lượt nguyên mẫu
DP = dict(b_real=1.5, b_virt=0.2, b_merge=0.3, skipc=-2.5, skipv=-0.05, skipr0=-0.3, skipr1=-3.0, merge_max=1.7, alpha=1.0)
PARAMS = dict(encoders=("v1", "v2"), lam_col=3.0, col_margin=0.5, proto_passes=1, n_folds=5, post_min=0.9, n_min=2,
              proto_w=1.0)
LOW_POST = 0.95          # TN6: hậu nghiệm < 0,95 -> đúng vị trí 67–93 % (≥ 0,95: 97–99,6 %) — chọn trên Kinh, kiểm DungLy
SRC_DET, SRC_MERGE, SRC_LOW, SRC_VIRT, SRC_FALLBACK = "vdp_det", "vdp_merge", "vdp_low", "vdp_virtual", "vdp_fallback"
SRC_AGREE = "vdp_agree"   # hậu nghiệm thấp / ô ảo NHƯNG trùng hộp pitch (IoU ≥ AGREE_IOU): hai bộ giải độc lập đồng ý
AGREE_IOU = 0.5
# (biến thể LAI, box_decoder: visual_dp_hybrid) ô hậu nghiệm thấp/ảo LỆCH pitch và mục bị DP bỏ -> dùng HỘP PITCH, box_source
# = "vdp_pitch_" + nguồn pitch (detector | detector_low | ink_cut; đường 3 nhánh: split | midpoint) -> cổng giữ nghĩa của pitch.
PITCH_PREFIX = "vdp_pitch_"
PITCH_LOW = tuple(PITCH_PREFIX + x for x in ("detector_low", "ink_cut", "split", "midpoint"))
BOX_LOW_CONF = (SRC_LOW, SRC_VIRT, SRC_FALLBACK) + PITCH_LOW
COUNT_SOURCE = "visual_dp"

_ENC = {}
_GLYPH = {}


def _nrm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def encoder():
    """MultiEnc [v1, v2] (nạp 1 lần / tiến trình; chỉ ĐỌC checkpoint có sẵn)."""
    if "enc" not in _ENC:
        from pipeline.borg_human.encoders import MultiEnc, device
        _ENC["enc"] = MultiEnc(list(PARAMS["encoders"]), device("auto"), True)
    return _ENC["enc"]


# --------------------------------------------------------------------------- PASS 1: đơn vị + nhúng theo trang
def page_units(page_boxes_low, gray) -> dict | None:
    """Đơn vị ảnh của trang (hộp detector gom cột phải→trái + ô ảo) + nhúng float16. None nếu không có hộp."""
    from pipeline.borg_human.geom import crop_gray, units_for_page
    bx = [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])] for b in (page_boxes_low or [])]
    if not bx or gray is None:
        return None
    U, meta = units_for_page(bx)
    if not U:
        return None
    E = encoder().embed([crop_gray(gray, u) for u in U]).astype(np.float16)
    keep = ("col", "row", "x1", "y1", "x2", "y2", "score", "virtual")
    return dict(U=[{k: u[k] for k in keep} for u in U], E=E, meta=meta)


# --------------------------------------------------------------------------- glyph R(âm)
def glyph_bank(chars) -> dict:
    """{chữ: vector chuẩn hoá} = trung bình glyph font (NomNaTong/Plangothic) và ảnh FD; cache trong tiến trình."""
    need = sorted(set(c for c in chars if c and c not in _GLYPH))
    if need:
        from pipeline.borg_human.encoders import Glyphs
        G = _GLYPH.get("__G__") or Glyphs()
        _GLYPH["__G__"] = G
        imgs, keys = [], []
        for c in need:
            for kind, im in (("font", G.font(c)), ("fd", G.fdimg(c))):
                if im is not None:
                    imgs.append(im); keys.append(c)
        if imgs:
            E = _nrm(encoder().embed(imgs).astype(np.float32))
            acc: dict = {}
            for c, v in zip(keys, E):
                acc.setdefault(c, []).append(v)
            for c, vs in acc.items():
                _GLYPH[c] = _nrm(np.mean(vs, 0)).astype(np.float32)
        for c in need:
            _GLYPH.setdefault(c, None)
    return _GLYPH


# --------------------------------------------------------------------------- DP
def dp_align(uu, Z, pitch, P=DP):
    """DP đơn điệu (hành vi pipeline.borg_human.align.align_page). Trả list theo mục: None (bỏ) | (kind, i, box, post)."""
    M, N = Z.shape
    if M == 0 or N == 0:
        return [None] * N
    virt = np.array([u["virtual"] for u in uu]); sc = np.array([u["score"] for u in uu])
    colid = np.array([u["col"] for u in uu])
    Zp = Z * P["alpha"]
    mb = np.where(virt == 1, P["b_virt"], P["b_real"])[:, None] + Zp
    sb = np.where(virt == 1, P["skipv"], P["skipr0"] + P["skipr1"] * sc)
    can = np.zeros(M, bool)
    for i in range(M - 1):
        if virt[i] == 0 and virt[i + 1] == 0 and colid[i] == colid[i + 1] and \
                (uu[i + 1]["y2"] - uu[i]["y1"]) <= P["merge_max"] * pitch:
            can[i] = True
    mg = np.full((M, N), -1e9)
    if M > 1:
        mg[:-1] = np.where(can[:-1, None], P["b_merge"] + np.maximum(Zp[:-1], Zp[1:]), -1e9)
    scc = P["skipc"]; NEG = -1e18
    ar = np.arange(N + 1)
    V = np.full((M + 1, N + 1), NEG); V[0] = scc * ar
    BP = np.zeros((M + 1, N + 1), np.int8)      # 1 ghép, 2 bỏ hộp, 3 bỏ mục, 4 gộp
    for i in range(1, M + 1):
        cs = V[i - 1] + sb[i - 1]
        cm = np.full(N + 1, NEG); cm[1:] = V[i - 1, :-1] + mb[i - 1]
        cg = np.full(N + 1, NEG)
        if i >= 2:
            cg[1:] = V[i - 2, :-1] + mg[i - 2]
        T = np.maximum(np.maximum(cs, cm), cg)
        tb = np.where(T == cm, 1, np.where(T == cg, 4, 2))
        row = np.maximum.accumulate(T - scc * ar) + scc * ar
        V[i] = row; BP[i] = np.where(row > T + 1e-9, 3, tb)
    i, j = M, N
    asg = {}
    while i > 0 or j > 0:
        t = BP[i, j] if i > 0 else 3
        if t == 1:
            asg[j - 1] = ("m", i - 1); i -= 1; j -= 1
        elif t == 4:
            asg[j - 1] = ("g", i - 2); i -= 2; j -= 1
        elif t == 2:
            i -= 1
        else:
            j -= 1

    def fwd(mb_, mg_, sb_):
        F = np.full((M + 1, N + 1), NEG); F[0] = scc * ar
        for k in range(1, M + 1):
            a = F[k - 1] + sb_[k - 1]
            b = np.full(N + 1, NEG); b[1:] = F[k - 1, :-1] + mb_[k - 1]
            g = np.full(N + 1, NEG)
            if k >= 2:
                g[1:] = F[k - 2, :-1] + mg_[k - 2]
            T = np.logaddexp(np.logaddexp(a, b), g)
            F[k] = np.logaddexp.accumulate(T - scc * ar) + scc * ar
        return F
    F = fwd(mb, mg, sb)
    Bk = fwd(mb[::-1, ::-1], np.vstack([mg[:-1][::-1, ::-1], mg[-1:]]) if M > 1 else mg, sb[::-1])
    Zt = F[M, N]
    out = []
    for j in range(N):
        a = asg.get(j)
        if a is None:
            out.append(None); continue
        t, i = a
        u = uu[i]
        if t == "g":
            u2 = uu[i + 1]
            box = [min(u["x1"], u2["x1"]), u["y1"], max(u["x2"], u2["x2"]), u2["y2"]]
            p = float(np.exp(F[i, j] + mg[i, j] + Bk[M - i - 2, N - j - 1] - Zt)); kind = "merge"
        else:
            box = [u["x1"], u["y1"], u["x2"], u["y2"]]
            p = float(np.exp(F[i, j] + mb[i, j] + Bk[M - i - 1, N - j - 1] - Zt))
            kind = "virtual" if u["virtual"] else "real"
        out.append((kind, i, box, min(1.0, p)))
    return out


def emission(Eu, items, G, proto=None, fold=None, P=PARAMS):
    """z(u, j): chuẩn hoá theo hàng của max_{c∈R(âm_j)} cos(u, glyph c) (∨ nguyên mẫu âm của khối khác)."""
    M, N = len(Eu), len(items)
    S = np.full((M, N), np.nan, np.float32)
    cache: dict = {}
    for j, it in enumerate(items):
        ck = (tuple(it["R"]), it["key"])
        if ck not in cache:
            refs = [G[c] for c in it["R"] if G.get(c) is not None]
            s = (Eu @ np.stack(refs).T).max(1) if refs else np.full(M, np.nan, np.float32)
            if proto is not None:
                pv = proto.get((fold, it["key"]))
                if pv is not None:
                    sp = Eu @ pv
                    s = np.where(np.isnan(s), P["proto_w"] * sp, np.maximum(s, P["proto_w"] * sp))
            cache[ck] = s
        S[:, j] = cache[ck]
    if N:
        rowm = np.nanmean(np.where(np.isnan(S).all(1, keepdims=True), 0.0, S), 1, keepdims=True)
        S = np.where(np.isnan(S), rowm, S)
    mu = S.mean(1, keepdims=True); sd = S.std(1, keepdims=True) + 1e-6
    return (S - mu) / sd


def col_prior(uu, items, wmed, P=PARAMS):
    cx = np.array([(u["x1"] + u["x2"]) / 2 for u in uu])
    pr = np.zeros((len(uu), len(items)), np.float32)
    m = P["col_margin"] * wmed
    for j, it in enumerate(items):
        xr = it.get("x_range")
        if xr:
            pr[(cx < xr[0] - m) | (cx > xr[1] + m), j] = -P["lam_col"]
    return pr


def page_items(col_states, qn_to_nom) -> list[dict]:
    """Mục = âm QN của các cột (cột xếp phải→trái theo tâm x cột kim; trong cột theo syl_idx)."""
    from core.text.text_utils import normalize_tone_marks
    order = sorted(range(len(col_states)),
                   key=lambda k: -((col_states[k]["cluster"].get("x_range") or [0, 0])[0]
                                   + (col_states[k]["cluster"].get("x_range") or [0, 0])[1]))
    items = []
    for k in order:
        cs = col_states[k]
        xr = cs["cluster"].get("x_range")
        for si, s in enumerate(cs["syllables"]):
            key = normalize_tone_marks(str(s).lower())
            items.append(dict(cs=k, si=si, key=key, R=list(qn_to_nom.get(key, [])), x_range=xr))
    return items


def _align_page(vp, items, G, proto=None, fold=None):
    Eu = _nrm(vp["E"].astype(np.float32))
    Z = emission(Eu, items, G, proto, fold) + col_prior(vp["U"], items, vp["meta"]["wmed"])
    return dp_align(vp["U"], Z, vp["meta"]["pitch"])


def _iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _fallback_box(cs, si):
    """Hộp dự phòng cho âm si khi DP bỏ mục: hộp pitch theo syl_idx; hoặc hộp PASS 1 của chữ ghép âm si (ops lượt 1)."""
    G, Gs = cs.get("G"), cs.get("G_src")
    if cs.get("box_rule") == "pitch" and G and Gs and len(G) == cs["n_qn"] and si < len(G):
        return [int(v) for v in G[si][:4]]
    rb = cs.get("reseg_boxes")
    for o in cs.get("ops1") or []:
        if o.get("op") == "match" and o.get("syl_idx") == si and rb and 0 <= o["nom_idx"] < len(rb) and rb[o["nom_idx"]]:
            return [int(v) for v in rb[o["nom_idx"]][:4]]
    chars = cs["cluster"].get("chars") or []
    if chars:
        c = chars[min(si, len(chars) - 1)].get("bbox")
        if c:
            return [int(v) for v in c[:4]]
    return None


def _fallback_src(cs, si) -> str:
    """Nguồn hộp pitch/PASS 1 của âm si ('' nếu không tra được)."""
    G, Gs = cs.get("G"), cs.get("G_src")
    if cs.get("box_rule") == "pitch" and G and Gs and len(G) == cs["n_qn"] and si < len(Gs):
        return str(Gs[si] or "")
    bs = cs.get("box_source")
    for o in cs.get("ops1") or []:
        if o.get("op") == "match" and o.get("syl_idx") == si and bs and 0 <= o["nom_idx"] < len(bs):
            return str(bs[o["nom_idx"]] or "")
    return ""


def apply_book(recs: list, qn_to_nom: dict, log=print, P=PARAMS, hybrid: bool = False) -> dict:
    """recs: bản ghi trang (align_page, box_decoder visual_dp) của MỘT sách, có rec['vdp_page'] + rec['col_states'].
    Ghi đè trạng thái cột (G, G_src, box_rule 'pitch', count_source 'visual_dp', reseg_boxes/box_source theo ops lượt 1,
    vdp_post) rồi xoá rec['vdp_page']. Trả thống kê."""
    from pipeline.align_engine.align_production import assign_boxes_pitch
    t0 = time.time()
    pages = [r for r in recs if r.get("col_states")]
    pages.sort(key=lambda r: r["page"])
    nf = P["n_folds"]
    fold = {r["page"]: (k * nf // max(1, len(pages))) for k, r in enumerate(pages)}
    items_by = {r["page"]: page_items(r["col_states"], qn_to_nom) for r in pages}
    G = glyph_bank({c for its in items_by.values() for it in its for c in it["R"]})
    t_g = time.time() - t0
    res = {}
    for r in pages:
        vp = r.get("vdp_page")
        res[r["page"]] = _align_page(vp, items_by[r["page"]], G) if vp else [None] * len(items_by[r["page"]])
    for _ in range(P["proto_passes"]):
        sums: dict = {}; cnt: dict = {}
        for r in pages:
            vp = r.get("vdp_page")
            if not vp:
                continue
            Eu = _nrm(vp["E"].astype(np.float32)); f = fold[r["page"]]
            for it, a in zip(items_by[r["page"]], res[r["page"]]):
                if a is None or a[0] != "real" or a[3] < P["post_min"]:
                    continue
                k = (f, it["key"])
                sums[k] = sums.get(k, 0) + Eu[a[1]]; cnt[k] = cnt.get(k, 0) + 1
        keys = {k for (_, k) in sums}
        proto = {}
        for f in range(nf):
            for s in keys:
                n = sum(cnt.get((g, s), 0) for g in range(nf) if g != f)
                if n >= P["n_min"]:
                    v = sum(sums[(g, s)] for g in range(nf) if g != f and (g, s) in sums)
                    proto[(f, s)] = _nrm(v / n).astype(np.float32)
        for r in pages:
            vp = r.get("vdp_page")
            if vp:
                res[r["page"]] = _align_page(vp, items_by[r["page"]], G, proto, fold[r["page"]])
    st = dict(pages=len(pages), items=0, kinds={}, fallback=0, low=0, seconds_glyph=round(t_g, 1))
    for r in pages:
        cstates = r["col_states"]
        per_cs: dict = {}
        for it, a in zip(items_by[r["page"]], res[r["page"]]):
            per_cs.setdefault(it["cs"], {})[it["si"]] = a
        for k, cs in enumerate(cstates):
            n_qn = cs["n_qn"]
            if n_qn <= 0:
                continue
            got = per_cs.get(k, {})
            Gv, Gs, post = [], [], []
            for si in range(n_qn):
                a = got.get(si)
                st["items"] += 1
                if a is None:
                    fb = _fallback_box(cs, si)
                    fsrc = _fallback_src(cs, si) if (hybrid and fb) else ""
                    Gv.append([*(fb or [0, 0, 1, 1]), 0.0])
                    Gs.append(PITCH_PREFIX + fsrc if fsrc else SRC_FALLBACK); post.append(0.0)
                    st["fallback"] += 1
                    continue
                kind, _i, box, p = a
                st["kinds"][kind] = st["kinds"].get(kind, 0) + 1
                src = SRC_VIRT if kind == "virtual" else (SRC_LOW if p < LOW_POST else
                                                          (SRC_MERGE if kind == "merge" else SRC_DET))
                if src in (SRC_VIRT, SRC_LOW) and _iou(box, _fallback_box(cs, si)) >= AGREE_IOU:
                    src = SRC_AGREE                      # hai bộ giải (DP thị giác, pitch) cùng một hộp
                    st["agree"] = st.get("agree", 0) + 1
                if hybrid and src in (SRC_VIRT, SRC_LOW):
                    fb, fsrc = _fallback_box(cs, si), _fallback_src(cs, si)
                    if fb and fsrc:                      # LAI: hộp pitch thay hộp DP không tự tin
                        box, src = fb, PITCH_PREFIX + fsrc
                        st["hybrid_pitch"] = st.get("hybrid_pitch", 0) + 1
                st["low"] += int(src == SRC_LOW)
                Gv.append([int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3])), round(p, 4)])
                Gs.append(src); post.append(round(p, 4))
            cs["G_pass1"] = cs.get("G")
            cs["G"], cs["G_src"], cs["vdp_post"] = Gv, Gs, post
            cs["box_rule"] = "pitch"                     # PASS 1b: assign_boxes_pitch theo syl_idx
            rb, bs, csrc = assign_boxes_pitch(Gv, Gs, cs.get("ops1") or [], cs["n_ocr"], n_qn, mode=COUNT_SOURCE)
            if rb is not None:
                cs["reseg_boxes"], cs["box_source"], cs["count_source"] = rb, bs, csrc
        r.pop("vdp_page", None)
    st["seconds"] = round(time.time() - t0, 1)
    log(f"  [visual_dp] {st['pages']} trang, {st['items']:,} âm: {st['kinds']} | dự phòng {st['fallback']:,} | "
        f"hậu nghiệm < {LOW_POST} lệch pitch: {st['low']:,} | thấp/ảo nhưng trùng pitch: {st.get('agree', 0):,} "
        + (f"| LAI -> hộp pitch: {st.get('hybrid_pitch', 0):,} " if hybrid else "") +
        f"({st['seconds']} s, glyph {st['seconds_glyph']} s)")
    return st


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    """DP + phát xạ trên dữ liệu tổng hợp (không cần model): hai cột, khung kim hụt, một hộp thừa."""
    ok = True
    cnt = [0, 0]

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra and not cond else ""))
        ok &= bool(cond)
        cnt[0 if cond else 1] += 1
    rng = np.random.default_rng(0)
    D = 16
    V = _nrm(rng.normal(size=(6, D)).astype(np.float32))          # 6 "chữ" a..f
    # đơn vị: cột phải (x 100) 4 chữ a b c d + 1 hộp rác; cột trái (x 50) 2 chữ e f
    seq_units = [(0, 100, 0, "a"), (0, 100, 1, "b"), (0, 100, 2, None), (0, 100, 3, "c"), (0, 100, 4, "d"),
                 (1, 50, 0, "e"), (1, 50, 1, "f")]
    U, E = [], []
    for col, x, r, ch in seq_units:
        U.append(dict(col=col, row=r, x1=x - 10, y1=r * 30, x2=x + 10, y2=r * 30 + 25, score=0.6 if ch else 0.25, virtual=0))
        E.append(V["abcdef".index(ch)] + 0.05 * rng.normal(size=D) if ch else rng.normal(size=D) * 0.3)
    E = _nrm(np.array(E, np.float32))
    G = {c: V[i] for i, c in enumerate("abcdef")}
    # kim chia cột SAI: cột phải chỉ nhận 3 âm (a b c), cột trái nhận d e f (lỗi phân cột ở ranh giới)
    items = [dict(cs=0, si=i, key=c, R=[c], x_range=[90, 110]) for i, c in enumerate("abc")] + \
            [dict(cs=1, si=i, key=c, R=[c], x_range=[40, 60]) for i, c in enumerate("def")]
    vp = dict(U=U, E=E.astype(np.float16), meta=dict(pitch=30.0, wmed=20.0))
    res = _align_page(vp, items, G)
    got = ["".join(seq_units[a[1]][3] or "?") if a else "-" for a in res]
    check("DP gióng âm vào đúng hộp dù kim chia cột lệch ở ranh giới (d nằm cột phải) và bỏ hộp rác",
          got == list("abcdef"), got)
    check("hậu nghiệm ghép ∈ [0, 1] và cao khi phát xạ rõ", all(a and 0.5 < a[3] <= 1.0 for a in res), [a[3] for a in res if a])
    # không phát xạ (glyph thiếu) -> không vỡ
    res0 = _align_page(vp, [dict(cs=0, si=0, key="zz", R=["zz"], x_range=None)], {})
    check("âm không có glyph -> vẫn trả kết quả (phát xạ trung tính)", len(res0) == 1)
    check("trang rỗng -> [None]*N", dp_align([], np.zeros((0, 2)), 30.0) == [None, None])
    # apply_book: ghi trạng thái cột (không cần encoder — glyph lấy từ cache _GLYPH)
    _GLYPH.update(G)
    cs0 = dict(cluster=dict(x_range=[90, 110], chars=[dict(bbox=[90, 0, 110, 25])] * 3), syllables=list("abc"), n_qn=3, n_ocr=3,
               ops1=[dict(op="match", nom_idx=i, syl_idx=i) for i in range(3)], G=None, box_rule="syl_index", reseg_boxes=None)
    cs1 = dict(cluster=dict(x_range=[40, 60], chars=[dict(bbox=[40, 0, 60, 25])] * 3), syllables=list("def"), n_qn=3, n_ocr=3,
               ops1=[dict(op="match", nom_idx=i, syl_idx=i) for i in range(3)], G=None, box_rule="syl_index", reseg_boxes=None)
    rec = dict(page="p1", col_states=[cs0, cs1], vdp_page=vp)
    st = apply_book([rec], {c: [c] for c in "abcdef"}, log=lambda *a: None, P=dict(PARAMS, proto_passes=0))
    check("apply_book: count_source visual_dp, box_rule pitch, G theo syl_idx (len n_qn)",
          cs0["count_source"] == COUNT_SOURCE and cs0["box_rule"] == "pitch" and len(cs0["G"]) == 3 and len(cs1["G"]) == 3)
    check("apply_book: âm d (cột kim trái) nhận hộp ở cột PHẢI (x≈100)", abs((cs1["G"][0][0] + cs1["G"][0][2]) / 2 - 100) < 1,
          cs1["G"][0])
    check("apply_book: box_source ∈ {vdp_det, vdp_merge, vdp_agree, vdp_low, vdp_virtual, vdp_fallback}",
          set(cs0["box_source"]) | set(cs1["box_source"]) <= {SRC_DET, SRC_MERGE, SRC_AGREE, SRC_LOW, SRC_VIRT, SRC_FALLBACK})
    check("_iou: trùng = 1, rời = 0, thiếu = 0", _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
          and _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0 and _iou(None, [0, 0, 1, 1]) == 0.0)
    # hậu nghiệm thấp nhưng trùng hộp pitch -> vdp_agree (không bị coi là hộp không tự tin)
    cs2 = dict(cluster=dict(x_range=[90, 110], chars=[]), syllables=["a"], n_qn=1, n_ocr=1, box_rule="pitch",
               G=[[90, 0, 110, 25, 0.5]], G_src=["detector"], ops1=[dict(op="match", nom_idx=0, syl_idx=0)])
    check("_fallback_box: hộp pitch theo syl_idx", _fallback_box(cs2, 0) == [90, 0, 110, 25])
    check("_fallback_src: nguồn pitch theo syl_idx", _fallback_src(cs2, 0) == "detector")
    # LAI: mục có phát xạ vô nghĩa + tiên nghiệm lệch -> hậu nghiệm thấp, lệch pitch -> hộp pitch, nguồn vdp_pitch_detector
    U2 = [dict(col=0, row=r, x1=90, y1=r * 30, x2=110, y2=r * 30 + 25, score=0.6, virtual=0) for r in range(2)]
    E2 = _nrm(np.stack([V[0], V[0]]).astype(np.float32))       # hai hộp giống hệt nhau -> DP không phân biệt
    cs3 = dict(cluster=dict(x_range=[90, 110], chars=[]), syllables=["a"], n_qn=1, n_ocr=1, box_rule="pitch",
               G=[[90, 30, 110, 55, 0.5]], G_src=["detector"], ops1=[dict(op="match", nom_idx=0, syl_idx=0)])
    rec3 = dict(page="p3", col_states=[cs3], vdp_page=dict(U=U2, E=E2.astype(np.float16), meta=dict(pitch=30.0, wmed=20.0)))
    apply_book([rec3], {"a": ["a"]}, log=lambda *a: None, P=dict(PARAMS, proto_passes=0), hybrid=True)
    check("LAI: hộp DP không tự tin & lệch pitch -> hộp pitch + box_source vdp_pitch_detector (hoặc DP trùng pitch)",
          (cs3["G_src"][0] == PITCH_PREFIX + "detector" and cs3["G"][0][:4] == [90, 30, 110, 55])
          or cs3["G_src"][0] in (SRC_DET, SRC_AGREE), (cs3["G_src"], cs3["G"]))
    check("BOX_LOW_CONF gồm vdp_pitch_ink_cut/detector_low, KHÔNG gồm vdp_pitch_detector",
          PITCH_PREFIX + "ink_cut" in BOX_LOW_CONF and PITCH_PREFIX + "detector" not in BOX_LOW_CONF)
    check("apply_book: xoá vdp_page (giải phóng bộ nhớ)", "vdp_page" not in rec and st["items"] == 6)
    print(f"visual_dp selftest: {'PASS' if ok else 'FAIL'}")
    print(f"RESULT: {cnt[0]} passed, {cnt[1]} failed")
    return 0 if ok else 1


if __name__ == "__main__":
    # `python -m pipeline.align_engine.visual_dp` (không đối số, như scripts/run_all_selftests.sh) = chạy selftest
    import sys
    sys.exit(selftest())
