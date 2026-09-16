"""B-4 · H2-lite — HÀNG ĐỢI QUYẾT ĐỊNH THEO LỚP cho người ký (máy KHÔNG gán nhãn).

Lớp = (sách, âm, mã đa số, mã thiểu số) trên GOLD v3 (dataset_out_v3/labels.csv, tier == GOLD).
Ba nguồn, gộp theo khoá (sách, âm, {mã đa số, mã thiểu số}):
  E3   lab/gan_nhan_2026-09-13/e3_all_classes.csv  (112 lớp; verdict MOT/HAI/mo, k-NN & HOG bacc, đo trên bộ CŨ)
  TN   thuc_nghiem.py classes: (sách, âm) GOLD có nhãn thiểu số ≥ 3 ô, nhìn giống đa số theo SinoNom_Similar
       — tính LẠI trên GOLD v3 (bản cũ 128 lớp là trên labels_final 64.525)
  CS   re-dataset/check/labels.xlsx sheet Cap_sai (286 cặp (label, syllable) bộ kiểm độc lập, có sua_thanh)
       — chỉ lấy cặp mà label là mã THIỂU SỐ của (sách, âm) trong GOLD v3, ≥ 3 ô
Loại: cặp đã ký trong config/decisions.yaml (di_the da_ky), (chữ, âm) corpus_readings da_ky, âm 'người'
(QĐ-01), lớp E3 HAI (38 lớp hai hình — giữ nguyên, không hỏi).
Ưu tiên: 1 = E3 MOT · 2 = E3 mờ · 3 = TN nhìn giống / CS có sua_thanh (không có verdict E3) · 4 = còn lại;
trong mỗi nhóm xếp theo số ô GOLD v3 của mã thiểu số giảm dần.

Với mỗi lớp: 12 crop mỗi mã (GOLD v3, cùng sách), chọn bằng k-means trên HOG (chỉ để SẮP cho đa dạng,
không để quyết), HOG bacc tính lại trên v3 (không học, ý kiến thứ hai), chữ trong R(âm) từ điển,
số ô các sách khác, gợi ý Cap_sai.

Đầu ra (KhoiB/h2/):
  lop_review.html                 giao diện chọn: dung / nham:<mã đúng> / di_the:<chuẩn> / mo → xuất CSV
  lop_decisions_template.csv      id_lop,book,syllable,ma_da_so,ma_thieu_so,n_da_so,n_thieu_so,e3_verdict,quyet,nguoi_ky,ngay,xuat_xu
  lop_queue_info.csv              mọi trường đo được của từng lớp (nguồn, ưu tiên, HOG v3, R(âm), Cap_sai…)
  lop_queue_summary.json          số lớp/ô theo nhóm + ước lượng ô ảnh hưởng nếu ký toàn bộ theo gợi ý E3

    .venv/bin/python KhoiB/h2/xay_hang_doi_lop.py [--max-crop 12] [--out KhoiB/h2]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from core.text.dictionary import load_qn_to_nom, load_similarity_dict          # noqa: E402
from pipeline.decisions import nfc, uni                                          # noqa: E402

NA = dict(dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
BOOKS = ("stt2", "stt4", "stt11")
MIN_TN = 3            # thuc_nghiem: nhãn thiểu số ≥ 3 ô
SZ = 64               # crops_v3.npz 64×64
THUMB_H = 72          # chiều cao thumbnail trong HTML
TEMPLATE_COLS = ["id_lop", "book", "syllable", "ma_da_so", "ma_thieu_so", "n_da_so", "n_thieu_so",
                 "e3_verdict", "quyet", "nguoi_ky", "ngay", "xuat_xu"]


# --------------------------------------------------------------------------- HOG (y hệt lab _hog)
def hog(imgs, cell=8, bins=9):
    import cv2
    out = []
    for im in imgs:
        g = im.astype(np.float32) / 255.0
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi) * bins
        b = np.floor(ang).astype(int) % bins
        feat = np.zeros((SZ // cell, SZ // cell, bins), np.float32)
        for i in range(SZ // cell):
            for j in range(SZ // cell):
                m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
                bb = b[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
                feat[i, j] = np.bincount(bb.ravel(), weights=m.ravel(), minlength=bins)
        v = feat.ravel()
        v = np.sqrt(v / (v.sum() + 1e-6))
        out.append(v / (np.linalg.norm(v) + 1e-6))
    return np.array(out)


def hog_bacc(FA, FB):
    """k-NN k=5 loại-một-ô, cân bằng — như E3 (thi_giac_am_tiet.py:466-470)."""
    F = np.concatenate([FA, FB])
    y = np.r_[np.zeros(len(FA)), np.ones(len(FB))]
    S = F @ F.T
    np.fill_diagonal(S, -9)
    k = min(5, len(F) - 1)
    nn_ = np.argsort(-S, axis=1)[:, :k]
    p = (y[nn_].mean(1) > 0.5)
    return float(0.5 * ((p[y == 0] == 0).mean() + (p[y == 1] == 1).mean()))


def kmeans_medoids(F, k, seed=0, iters=20):
    """k-means đơn giản (numpy) -> chỉ số medoid từng cụm, xếp theo cỡ cụm giảm dần. CHỈ để sắp crop."""
    n = len(F)
    if n <= k:
        return list(range(n))
    rng = np.random.default_rng(seed)
    C = F[rng.choice(n, k, replace=False)].copy()
    for _ in range(iters):
        D = ((F[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = D.argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = F[a == j].mean(0)
    D = ((F[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    a = D.argmin(1)
    out = []
    for j in sorted(range(k), key=lambda j: -(a == j).sum()):
        m = np.where(a == j)[0]
        if len(m):
            out.append(int(m[D[m, j].argmin()]))
    return out


def thumb_b64(path: Path) -> str:
    im = Image.open(path).convert("L")
    w, h = im.size
    im = im.resize((max(1, round(w * THUMB_H / h)), THUMB_H), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- nguồn lớp
def load_sources(repo: Path):
    labels = pd.read_csv(repo / "dataset_out_v3" / "labels.csv", **NA)
    labels["nom_idx_i"] = labels["nom_idx"].astype(int)
    gold = labels[(labels["tier"] == "GOLD") & (labels["image"] != "")]
    e3 = pd.read_csv(repo / "lab" / "gan_nhan_2026-09-13" / "e3_all_classes.csv", dtype=str, keep_default_na=False)
    e3["nB"] = e3["nB"].astype(int)
    dec = yaml.safe_load(open(repo / "config" / "decisions.yaml", encoding="utf-8"))
    cs = None
    xl = repo / "re-dataset" / "check" / "labels.xlsx"
    if xl.exists():
        try:
            cs = pd.read_excel(xl, sheet_name="Cap_sai", dtype=str).fillna("")
        except Exception as ex:                      # openpyxl thiếu / sheet đổi tên
            print(f"[warn] không đọc được Cap_sai: {ex}", file=sys.stderr)
    return labels, gold, e3, dec, cs


def build_classes(labels, gold, e3, dec, cs, qn, sim):
    cnt = Counter(zip(gold["book"], gold["syllable"], gold["label"]))
    cnt_all = Counter(zip(labels[labels["label"] != ""]["book"], labels[labels["label"] != ""]["syllable"],
                          labels[labels["label"] != ""]["label"]))
    per_bs: dict[tuple, Counter] = defaultdict(Counter)
    for (b, s, l), n in cnt.items():
        per_bs[(b, s)][l] += n

    # --- loại trừ theo decisions.yaml ---
    dt_signed = {frozenset(x["quan_sat"]): x["id"] for x in dec["di_the"] if x.get("trang_thai") == "da_ky"}
    dt_cho_ky = {frozenset(x["quan_sat"]): x["id"] for x in dec["di_the"] if x.get("trang_thai") != "da_ky"}
    cr_signed = {(x["ocr_char"], x["syllable_raw"]) for x in dec["corpus_readings"] if x.get("trang_thai") == "da_ky"}
    lop_nham_ids = {(x["syllable"], x["ocr"]): x["id"] for x in dec["lop_nham"]}

    classes: dict[tuple, dict] = {}          # key (book, syl, frozenset{a,b})

    def key_of(b, s, a, c):
        return (b, s, frozenset([a, c]))

    def get(b, s, a, c):
        k = key_of(b, s, a, c)
        if k not in classes:
            na, nc = cnt.get((b, s, a), 0), cnt.get((b, s, c), 0)
            M, L = (a, c) if na >= nc else (c, a)
            classes[k] = dict(book=b, syllable=s, ma_da_so=M, ma_thieu_so=L,
                              n_da_so=cnt.get((b, s, M), 0), n_thieu_so=cnt.get((b, s, L), 0),
                              nguon=set(), e3_verdict="", e3_M="", e3_L="", e3_nB="", e3_knn="", e3_hog="", e3_look="",
                              tn_look="", cs_ket_luan="", cs_sua_thanh="", cs_so_dong="", cs_ung_vien="")
        return classes[k]

    # E3
    for r in e3.itertuples():
        c = get(r.book, r.syl, r.M, r.L)
        c["nguon"].add("E3")
        c.update(e3_verdict=r.v, e3_M=r.M, e3_L=r.L, e3_nB=int(r.nB), e3_knn=r.knn, e3_hog=r.hog, e3_look=r.look)
    # TN (thuc_nghiem classes, tính lại trên GOLD v3)
    n_tn = n_tn_look = 0
    for (b, s), cc in per_bs.items():
        if len(cc) < 2:
            continue
        M = cc.most_common(1)[0][0]
        for L, n in cc.items():
            if L == M or n < MIN_TN:
                continue
            n_tn += 1
            look = (L in sim.get(M, [])) or (M in sim.get(L, []))
            n_tn_look += look
            c = get(b, s, M, L)
            c["nguon"].add("TN")
            c["tn_look"] = look
    # CS (Cap_sai)
    n_cs = 0
    if cs is not None:
        for r in cs.itertuples():
            lab, syl = nfc(r.label), nfc(r.syllable).lower()
            for b in BOOKS:
                n = cnt.get((b, syl, lab), 0)
                cc = per_bs.get((b, syl))
                if n < MIN_TN or not cc:
                    continue
                M = cc.most_common(1)[0][0]
                if M == lab:
                    continue                    # Cap_sai chê mã ĐA SỐ -> không phải lớp đa số/thiểu số (corpus_reading?)
                n_cs += 1
                c = get(b, syl, M, lab)
                c["nguon"].add("CS")
                c.update(cs_ket_luan=str(r.ket_luan), cs_sua_thanh=nfc(r.sua_thanh), cs_so_dong=str(r.so_dong),
                         cs_ung_vien=str(r.ung_vien_sua))

    # --- loại + ưu tiên ---
    out, ngoai, loai = [], [], Counter()
    for k, c in classes.items():
        pair = k[2]
        if c["syllable"] == "người":
            loai["qd01_nguoi"] += 1
            continue
        if pair in dt_signed:
            loai[f"di_the_da_ky"] += 1
            continue
        if any((x, c["syllable"]) in cr_signed for x in pair):
            loai["corpus_reading_da_ky"] += 1
            continue
        if c["e3_verdict"] == "HAI":
            loai["e3_HAI_giu_nguyen"] += 1
            continue
        if c["n_thieu_so"] == 0:
            loai["khong_con_o_gold_v3"] += 1
            continue
        v = c["e3_verdict"]
        if v == "MOT":
            uu = 1
        elif v == "mo":
            uu = 2
        elif c["tn_look"] is True or c["cs_sua_thanh"]:
            uu = 3
        else:
            # TN không nhìn giống theo SinoNom_Similar và Cap_sai không gợi ý sửa: ngoài 3 nguồn
            # B-4 quy định (TN chỉ lấy lớp nhìn giống) -> KHÔNG vào hàng đợi, chỉ đếm + ghi dự phòng
            loai["tn_khong_nhin_giong_ngoai_hang_doi"] += 1
            ngoai.append(c)
            continue
        c["uu_tien"] = uu
        c["da_co_yaml_id"] = dt_cho_ky.get(pair, "") or lop_nham_ids.get((c["syllable"], c["ma_thieu_so"]), "")
        c["nguon"] = "+".join(sorted(c["nguon"]))
        s = c["syllable"]
        R = qn.get(s, [])
        c["R_am"] = "".join(R)
        c["da_so_trong_R"] = c["ma_da_so"] in R
        c["thieu_so_trong_R"] = c["ma_thieu_so"] in R
        c["nhan_khac_cung_sach"] = " ".join(f"{l}:{n}" for l, n in per_bs[(c['book'], s)].most_common()
                                            if l not in pair)
        c["sach_khac"] = "; ".join(
            f"{b}: {c['ma_da_so']} {cnt.get((b, s, c['ma_da_so']), 0)} / {c['ma_thieu_so']} {cnt.get((b, s, c['ma_thieu_so']), 0)}"
            for b in BOOKS if b != c["book"])
        c["n_da_so_moi_tier"] = cnt_all.get((c["book"], s, c["ma_da_so"]), 0)
        c["n_thieu_so_moi_tier"] = cnt_all.get((c["book"], s, c["ma_thieu_so"]), 0)
        c["u_da_so"], c["u_thieu_so"] = uni(c["ma_da_so"]), uni(c["ma_thieu_so"])
        out.append(c)
    out.sort(key=lambda c: (c["uu_tien"], -c["n_thieu_so"], c["book"], c["syllable"]))
    for i, c in enumerate(out, 1):
        c["id_lop"] = f"L{i:03d}_{c['book']}_{c['syllable']}_{c['u_da_so'][2:].lower()}_{c['u_thieu_so'][2:].lower()}"
    tk = dict(n_e3=len(e3), n_tn_v3=n_tn, n_tn_look_v3=n_tn_look, n_cs_hit=n_cs,
              n_gop=len(classes), loai=dict(loai), n_hang_doi=len(out),
              n_ngoai_hang_doi=len(ngoai), o_ngoai_hang_doi=sum(c["n_thieu_so"] for c in ngoai))
    for c in ngoai:
        c["nguon"] = "+".join(sorted(c["nguon"])) if isinstance(c["nguon"], set) else c["nguon"]
    return out, ngoai, tk


# --------------------------------------------------------------------------- crop + HOG
def attach_crops(out, gold, X, repo: Path, max_crop: int):
    idx_by = defaultdict(list)
    for i, b, s, l in zip(gold.index, gold["book"], gold["syllable"], gold["label"]):
        idx_by[(b, s, l)].append(i)
    rng = np.random.default_rng(0)
    for c in out:
        b, s = c["book"], c["syllable"]
        crops = {}
        feats = {}
        for code in (c["ma_da_so"], c["ma_thieu_so"]):
            ii = np.array(idx_by[(b, s, code)])
            F = hog(X[ii])
            feats[code] = (ii, F)
            # k-means chỉ để chọn mẫu đa dạng
            sel = kmeans_medoids(F, max_crop) if len(ii) > max_crop else list(range(len(ii)))
            crops[code] = [int(ii[j]) for j in sel]
        A, B = feats[c["ma_da_so"]], feats[c["ma_thieu_so"]]
        ia, Fa = A
        if len(ia) > 200:
            pick = rng.choice(len(ia), 200, replace=False)
            Fa = Fa[pick]
        c["hog_bacc_v3"] = round(hog_bacc(Fa, B[1]), 2) if len(B[1]) >= 2 and len(Fa) >= 2 else ""
        c["_crops"] = crops
    return out


# --------------------------------------------------------------------------- HTML
CSS = """
:root{--bg:#faf9f6;--fg:#1c1b19;--mut:#6b6860;--line:#ddd8cc;--acc:#8a4b1f;--ok:#2d6a4f;--warn:#b5541b;--card:#fff}
*{box-sizing:border-box}body{margin:0;padding:16px;font:14px/1.45 -apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px;border-bottom:2px solid var(--line);padding-bottom:4px}
.top{position:sticky;top:0;background:var(--bg);z-index:5;padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.top input{padding:4px 6px;border:1px solid var(--line);border-radius:4px}
.top button{padding:6px 12px;border:1px solid var(--acc);background:var(--acc);color:#fff;border-radius:4px;cursor:pointer}
.top button.sec{background:#fff;color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}
.card.done{border-color:var(--ok);box-shadow:inset 4px 0 0 var(--ok)}
.hd{display:flex;gap:16px;flex-wrap:wrap;align-items:baseline}
.hd .id{font-family:ui-monospace,Menlo,monospace;color:var(--mut);font-size:12px}
.hd .syl{font-size:22px;font-weight:600}.hd .book{color:var(--acc);font-weight:600}
.meta{color:var(--mut);font-size:12.5px;margin:4px 0 8px}.meta b{color:var(--fg)}
.codes{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:700px){.codes{grid-template-columns:1fr}}
.code{border:1px solid var(--line);border-radius:6px;padding:8px}
.code .ch{font-size:34px;line-height:1;font-family:"Nom Na Tong","Han-Nom Gothic","Songti SC","STSong","Noto Serif CJK SC",serif}
.code .n{color:var(--mut);font-size:12px;margin-left:8px}
.strip{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.strip a img{height:72px;border:1px solid var(--line);background:#fff;image-rendering:auto}
.strip a:hover img{border-color:var(--acc)}
.ch-inline{font-family:"Nom Na Tong","Han-Nom Gothic","Songti SC","STSong","Noto Serif CJK SC",serif;font-size:16px}
.opts{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:10px;align-items:center}
.opts label{cursor:pointer;padding:3px 8px;border:1px solid var(--line);border-radius:14px;background:#fff}
.opts label:has(input:checked){border-color:var(--acc);background:#f6ebe1}
.opts input[type=text]{width:4em;padding:2px 4px;border:1px solid var(--line);border-radius:4px;font-size:16px}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;background:#eee;color:#333;margin-right:4px}
.tag.mot{background:#fde2e2;color:#8a1c1c}.tag.mo{background:#fff1cc;color:#7a5200}.tag.p3{background:#e3ecf7;color:#1f3f6b}.tag.p4{background:#eee}
.warn{color:var(--warn)}.ok{color:var(--ok)}
textarea{width:100%;height:160px;font:12px ui-monospace,Menlo,monospace}
.small{font-size:12px;color:var(--mut)}
"""

JS = r"""
const KEY='h2_lop_decisions_v1';
function load(){try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return{}}}
function save(st){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}}
function val(card){
  const r=card.querySelector('input[type=radio]:checked'); if(!r) return '';
  let v=r.value;
  if(v==='nham:__'){const t=card.querySelector('input.nham_khac').value.trim(); return t?('nham:'+t):''}
  return v;
}
function refresh(){
  const st=load(); let n=0;
  document.querySelectorAll('.card').forEach(c=>{const v=val(c); if(v){st[c.dataset.id]=v;n++;c.classList.add('done')}else{delete st[c.dataset.id];c.classList.remove('done')}});
  st.__nguoi_ky=document.getElementById('nguoi_ky').value; st.__ngay=document.getElementById('ngay').value; st.__xuat_xu=document.getElementById('xuat_xu').value;
  save(st); document.getElementById('cnt').textContent=n+'/'+document.querySelectorAll('.card').length;
}
function restore(){
  const st=load();
  ['nguoi_ky','ngay','xuat_xu'].forEach(k=>{if(st['__'+k]) document.getElementById(k).value=st['__'+k]});
  document.querySelectorAll('.card').forEach(c=>{const v=st[c.dataset.id]; if(!v) return;
    let r=c.querySelector('input[type=radio][value="'+v.replace(/"/g,'&quot;')+'"]');
    if(!r && v.startsWith('nham:')){r=c.querySelector('input[type=radio][value="nham:__"]'); c.querySelector('input.nham_khac').value=v.slice(5)}
    if(r) r.checked=true;});
  refresh();
}
function csvq(s){s=String(s==null?'':s); return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function buildCSV(){
  const nk=document.getElementById('nguoi_ky').value.trim(), ng=document.getElementById('ngay').value.trim(), xx=document.getElementById('xuat_xu').value.trim();
  const cols=['id_lop','book','syllable','ma_da_so','ma_thieu_so','n_da_so','n_thieu_so','e3_verdict','quyet','nguoi_ky','ngay','xuat_xu'];
  const rows=[cols.join(',')];
  document.querySelectorAll('.card').forEach(c=>{const d=c.dataset; const v=val(c);
    rows.push([d.id,d.book,d.syl,d.m,d.l,d.nm,d.nl,d.e3,v,v?nk:'',v?ng:'',v?xx:''].map(csvq).join(','))});
  return '﻿'+rows.join('\n')+'\n';
}
function exportCSV(){
  const txt=buildCSV(); document.getElementById('csvout').value=txt;
  try{const b=new Blob([txt],{type:'text/csv;charset=utf-8'}); const a=document.createElement('a'); a.href=URL.createObjectURL(b); a.download='lop_decisions_filled.csv'; document.body.appendChild(a); a.click(); a.remove();}catch(e){}
}
function clearAll(){ if(confirm('Xoá mọi lựa chọn đã lưu trong trình duyệt?')){localStorage.removeItem(KEY); document.querySelectorAll('input[type=radio]').forEach(r=>r.checked=false); refresh();} }
document.addEventListener('change',refresh); document.addEventListener('input',e=>{if(e.target.matches('input.nham_khac,#nguoi_ky,#ngay,#xuat_xu')) refresh()});
window.addEventListener('load',restore);
"""


def render_html(out, gold, repo: Path, out_dir: Path, tk: dict, thumbs: dict) -> str:
    rel = Path("../..") / "dataset_out_v3"        # KhoiB/h2/ -> repo/dataset_out_v3
    esc = html.escape
    parts = [f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>H2-lite · hàng đợi lớp</title><style>{CSS}</style></head><body>"]
    parts.append("<h1>B-4 · H2-lite — hàng đợi quyết định theo LỚP (người ký)</h1>"
                 "<p class='small'>Máy không gán nhãn. Mỗi thẻ = một lớp (sách, âm, mã đa số / mã thiểu số) trên GOLD v3. "
                 "12 crop mỗi mã chọn bằng k-means HOG chỉ để đa dạng. Chọn: <b>dung</b> = giữ cả hai mã (hai hình thật) · "
                 "<b>nham:&lt;mã đúng&gt;</b> = mã còn lại là OCR nhầm, ghi đè theo sách · <b>di_the:&lt;chuẩn&gt;</b> = một hình, hai mã Unicode → "
                 "<code>label_canonical</code> · <b>mo</b> = chưa quyết, ô mã thiểu số → REVIEW. "
                 "Lựa chọn lưu tạm trong trình duyệt; bấm <b>Xuất CSV</b> rồi chạy "
                 "<code>pipeline/tools/apply_lop_decisions.py</code>.</p>")
    parts.append("<div class='top'>Người ký <input id='nguoi_ky' placeholder='tên'> Ngày <input id='ngay' placeholder='2026-09-..'> "
                 "Xuất xứ <input id='xuat_xu' size='34' placeholder='docs/NGUOI_CHAM_QUYET_DINH.md#h2'> "
                 "<span>đã chọn <b id='cnt'>0</b></span> <button onclick='exportCSV()'>Xuất CSV</button> "
                 "<button class='sec' onclick='clearAll()'>Xoá lựa chọn</button></div>")
    nhom = {1: "Nhóm 1 · E3 MỘT HÌNH — nhầm khác chữ hoặc dị thể chưa ký", 2: "Nhóm 2 · E3 mờ",
            3: "Nhóm 3 · nhìn giống theo SinoNom_Similar (TN v3) / Cap_sai có gợi ý sửa — chưa có verdict E3"}
    cur = None
    for c in out:
        if c["uu_tien"] != cur:
            cur = c["uu_tien"]
            n_o = sum(x["n_thieu_so"] for x in out if x["uu_tien"] == cur)
            n_l = sum(1 for x in out if x["uu_tien"] == cur)
            parts.append(f"<h2>{esc(nhom[cur])} — {n_l} lớp · {n_o} ô thiểu số</h2>")
        M, L = c["ma_da_so"], c["ma_thieu_so"]
        tag = {"MOT": "<span class='tag mot'>E3 MỘT</span>", "mo": "<span class='tag mo'>E3 mờ</span>"}.get(
            c["e3_verdict"], f"<span class='tag p{c['uu_tien']}'>{esc(c['nguon'])}</span>")
        e3s = (f"E3 (bộ cũ): k-NN <b>{c['e3_knn']}</b> · HOG <b>{c['e3_hog']}</b> · nB {c['e3_nB']}"
               + (f" · E3 ghi {esc(c['e3_M'])}/{esc(c['e3_L'])}" if c["e3_M"] and (c["e3_M"], c["e3_L"]) != (M, L) else "")
               if c["e3_verdict"] else "E3: —")
        hv = c["hog_bacc_v3"]
        hvs = f"HOG v3 (không học) <b>{hv}</b>" + (" <span class='small'>(≤0,6 một hình · ≥0,75 hai hình)</span>" if hv != "" else "")
        rs = (f"R(<i>{esc(c['syllable'])}</i>) từ điển: <span class='ch-inline'>{esc(c['R_am'][:40])}</span>"
              + (f"… ({len(c['R_am'])})" if len(c["R_am"]) > 40 else f" ({len(c['R_am'])})")
              + f" — {M} {'∈' if c['da_so_trong_R'] else '∉'} R, {L} {'∈' if c['thieu_so_trong_R'] else '∉'} R")
        css = ""
        if c["cs_sua_thanh"] or c["cs_ket_luan"]:
            css = (f" · Cap_sai: {esc(c['cs_ket_luan'][:28])} sửa→<span class='ch-inline'>{esc(c['cs_sua_thanh'] or '—')}</span>"
                   f" ({esc(c['cs_ung_vien'][:40])})")
        yid = f" · <span class='warn'>đã có mục {esc(c['da_co_yaml_id'])} (cho_ky) trong decisions.yaml</span>" if c["da_co_yaml_id"] else ""
        parts.append(f"<div class='card' data-id='{esc(c['id_lop'])}' data-book='{esc(c['book'])}' data-syl='{esc(c['syllable'])}' "
                     f"data-m='{esc(M)}' data-l='{esc(L)}' data-nm='{c['n_da_so']}' data-nl='{c['n_thieu_so']}' data-e3='{esc(c['e3_verdict'])}'>")
        parts.append(f"<div class='hd'><span class='id'>{esc(c['id_lop'])}</span> {tag} <span class='book'>{esc(c['book'])}</span> "
                     f"<span class='syl'>{esc(c['syllable'])}</span> <span class='ch-inline' style='font-size:22px'>{esc(M)} / {esc(L)}</span></div>")
        parts.append(f"<div class='meta'>{e3s} · {hvs}<br>{rs}{css}{yid}<br>"
                     f"Sách khác: {esc(c['sach_khac'])}"
                     + (f" · nhãn khác cùng (sách, âm): <span class='ch-inline'>{esc(c['nhan_khac_cung_sach'])}</span>" if c["nhan_khac_cung_sach"] else "")
                     + f" · mọi tier có label: {M} {c['n_da_so_moi_tier']} / {L} {c['n_thieu_so_moi_tier']}</div>")
        parts.append("<div class='codes'>")
        for code, n, lab in ((M, c["n_da_so"], "đa số"), (L, c["n_thieu_so"], "thiểu số")):
            parts.append(f"<div class='code'><span class='ch'>{esc(code)}</span><span class='n'>{uni(code)} · {lab} · <b>{n}</b> ô GOLD "
                         f"({len(c['_crops'][code])} crop)</span><div class='strip'>")
            for i in c["_crops"][code]:
                r = gold.loc[i]
                p = rel / r["image"]
                title = f"{r['book']} {r['page']} c{r['column']} nom_idx {r['nom_idx']} · {r['rule']}"
                parts.append(f"<a href='{esc(p.as_posix())}' target='_blank' title='{esc(title)}'><img src='{thumbs[i]}' alt='{esc(code)}'></a>")
            parts.append("</div></div>")
        parts.append("</div>")
        nm = "q_" + c["id_lop"]
        parts.append("<div class='opts'>"
                     f"<label><input type='radio' name='{nm}' value='dung'> dung — giữ cả hai</label>"
                     f"<label><input type='radio' name='{nm}' value='nham:{esc(M)}'> nham:{esc(M)} — {esc(L)} là OCR nhầm → {esc(M)}</label>"
                     f"<label><input type='radio' name='{nm}' value='nham:{esc(L)}'> nham:{esc(L)} — {esc(M)} nhầm → {esc(L)}</label>"
                     f"<label><input type='radio' name='{nm}' value='nham:__'> nham: mã khác <input type='text' class='nham_khac' placeholder='mã'></label>"
                     f"<label><input type='radio' name='{nm}' value='di_the:{esc(M)}'> di_the:{esc(M)}</label>"
                     f"<label><input type='radio' name='{nm}' value='di_the:{esc(L)}'> di_the:{esc(L)}</label>"
                     f"<label><input type='radio' name='{nm}' value='mo'> mo — để REVIEW</label>"
                     "</div></div>")
    parts.append("<h2>CSV đã xuất (sao chép nếu trình duyệt chặn tải)</h2><textarea id='csvout' readonly></textarea>")
    parts.append(f"<p class='small'>Nguồn: E3 {tk['n_e3']} lớp · TN v3 {tk['n_tn_v3']} lớp (nhìn giống {tk['n_tn_look_v3']}) · Cap_sai khớp {tk['n_cs_hit']} · "
                 f"gộp {tk['n_gop']} · loại {esc(json.dumps(tk['loai'], ensure_ascii=False))} · hàng đợi {tk['n_hang_doi']} · "
                 f"ngoài hàng đợi (TN không nhìn giống, xem lop_ngoai_hang_doi.csv) {tk['n_ngoai_hang_doi']} lớp / {tk['o_ngoai_hang_doi']} ô.</p>")
    parts.append(f"<script>{JS}</script></body></html>")
    return "".join(parts)


# --------------------------------------------------------------------------- ước lượng ô ảnh hưởng
def uoc_luong(out, labels):
    """Nếu người ký TOÀN BỘ theo gợi ý E3 (MOT -> nham:đa số, mờ -> mo, nhóm 3 -> nham:đa số):
    ô label == mã thiểu số, cùng (sách, âm), ngoài khoá QĐ-01 (mọi tier có label). Cận trên."""
    lab = labels[(labels["label"] != "") & (labels["qd01_locked"] != "1")]
    cnt = Counter(zip(lab["book"], lab["syllable"], lab["label"]))
    cnt_g = Counter(zip(lab[lab["tier"] == "GOLD"]["book"], lab[lab["tier"] == "GOLD"]["syllable"], lab[lab["tier"] == "GOLD"]["label"]))
    r = defaultdict(lambda: Counter())
    for c in out:
        k = (c["book"], c["syllable"], c["ma_thieu_so"])
        grp = {1: "nhom1_MOT_nham", 2: "nhom2_mo_review", 3: "nhom3_nham"}[c["uu_tien"]]
        r[grp]["lop"] += 1
        r[grp]["o_gold"] += cnt_g.get(k, 0)
        r[grp]["o_moi_tier_ngoai_khoa"] += cnt.get(k, 0)
    tot = Counter()
    for v in r.values():
        tot.update(v)
    r["tong"] = tot
    return {k: dict(v) for k, v in r.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--max-crop", type=int, default=12)
    ap.add_argument("--crops", default=str(REPO / "KhoiB" / "v3" / "crops_v3.npz"))
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    qn = load_qn_to_nom(str(REPO / "dict/QuocNgu_SinoNom.csv"))
    sim = load_similarity_dict(str(REPO / "dict/SinoNom_Similar.csv"))
    labels, gold, e3, dec, cs = load_sources(REPO)
    print(f"[nguon] labels v3 {len(labels)} · GOLD có ảnh {len(gold)} · E3 {len(e3)} · Cap_sai {0 if cs is None else len(cs)}")

    out, ngoai, tk = build_classes(labels, gold, e3, dec, cs, qn, sim)
    print(f"[lop] gộp {tk['n_gop']} → hàng đợi {tk['n_hang_doi']} · loại {tk['loai']}")
    pd.DataFrame([{k: c.get(k, "") for k in ("book", "syllable", "ma_da_so", "ma_thieu_so", "n_da_so", "n_thieu_so", "nguon")}
                  for c in ngoai]).sort_values("n_thieu_so", ascending=False).to_csv(
        out_dir / "lop_ngoai_hang_doi.csv", index=False, encoding="utf-8-sig")
    for u in (1, 2, 3):
        xs = [c for c in out if c["uu_tien"] == u]
        print(f"   nhóm {u}: {len(xs)} lớp · {sum(c['n_thieu_so'] for c in xs)} ô thiểu số GOLD v3 · {sum(c['n_da_so'] for c in xs)} ô đa số")

    z = np.load(a.crops)
    assert (z["book"] == labels["book"].values).all() and (z["nom_idx"].astype(int) == labels["nom_idx_i"].values).all(), \
        "crops_v3.npz không cùng thứ tự labels.csv"
    X = z["X"]
    out = attach_crops(out, gold, X, REPO, a.max_crop)

    # thumbnails
    need = sorted({i for c in out for v in c["_crops"].values() for i in v})
    thumbs = {i: thumb_b64(REPO / "dataset_out_v3" / gold.loc[i, "image"]) for i in need}
    print(f"[crop] {len(need)} thumbnail · {sum(len(v) for v in thumbs.values()) / 1e6:.1f} MB base64")

    page = render_html(out, gold, REPO, out_dir, tk, thumbs)
    (out_dir / "lop_review.html").write_text(page, encoding="utf-8")

    # template + info
    tpl = pd.DataFrame([{k: c.get(k, "") for k in TEMPLATE_COLS} for c in out], columns=TEMPLATE_COLS)
    tpl.to_csv(out_dir / "lop_decisions_template.csv", index=False, encoding="utf-8-sig")
    info_cols = ["id_lop", "uu_tien", "nguon", "book", "syllable", "ma_da_so", "u_da_so", "ma_thieu_so", "u_thieu_so",
                 "n_da_so", "n_thieu_so", "n_da_so_moi_tier", "n_thieu_so_moi_tier", "e3_verdict", "e3_nB", "e3_knn", "e3_hog",
                 "e3_look", "e3_M", "e3_L", "hog_bacc_v3", "tn_look", "da_so_trong_R", "thieu_so_trong_R", "R_am",
                 "nhan_khac_cung_sach", "sach_khac", "cs_ket_luan", "cs_sua_thanh", "cs_ung_vien", "cs_so_dong", "da_co_yaml_id"]
    info = pd.DataFrame([{k: c.get(k, "") for k in info_cols} for c in out], columns=info_cols)
    info.to_csv(out_dir / "lop_queue_info.csv", index=False, encoding="utf-8-sig")

    ul = uoc_luong(out, labels)
    summ = dict(nguon=tk, nhom={u: dict(lop=sum(1 for c in out if c["uu_tien"] == u),
                                       o_thieu_so_gold=sum(c["n_thieu_so"] for c in out if c["uu_tien"] == u),
                                       o_da_so_gold=sum(c["n_da_so"] for c in out if c["uu_tien"] == u)) for u in (1, 2, 3)},
                uoc_luong_ky_toan_bo=ul, n_thumb=len(need), max_crop=a.max_crop,
                labels_csv=str(REPO / "dataset_out_v3" / "labels.csv"), crops=a.crops)
    (out_dir / "lop_queue_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[uoc_luong] ký toàn bộ theo gợi ý: {ul['tong']}")
    print(f"[out] {out_dir}/lop_review.html ({(out_dir / 'lop_review.html').stat().st_size / 1e6:.1f} MB), "
          f"lop_decisions_template.csv ({len(tpl)} dòng), lop_queue_info.csv, lop_queue_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
