"""Biến thể: (a) AMB — nghiệm nào cho DP 0 khe / nhiều direct nhất; (b) NOFILL — tách nới k<=gap+1 rồi để DP bù."""
import sys, itertools, collections
from pathlib import Path
REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"lab/gan_nhan_2026-09-13"))
import thuc_nghiem as tn
from thuc_nghiem import realign_column, qn, sim, set_matrix, PROD, Corpus, tier_v3
from core.text.text_utils import is_plausible_qn_syllable as plaus
cols = tn.load_cols(); qs = tn.qs; lex = {s for s in qs if plaus(s)}
set_matrix(PROD); C = Corpus(cols)
def splits(tok, k):
    tok = tok.lower(); out = []
    def rec(i, acc):
        if len(acc) == k:
            if i == len(tok): out.append(list(acc))
            return
        for j in range(i+1, len(tok)+1):
            p = tok[i:j]
            if p in lex: rec(j, acc+[p])
    rec(0, []); return out
def apply(c, oov, combo):
    repl = {i: combo[k] for k, (i, _) in enumerate(oov)}
    new = []
    for i, s in enumerate(c["syl"]):
        new.extend(repl[i]) if i in repl else new.append(s)
    return new
def score(c, new):
    chars_d = [{"char": ch, "ocr_char": ch} for ch in c["chars"]]
    ops = realign_column(chars_d, new, qn, sim); pg = (c["book"], c["page"])
    gaps = sum(1 for o in ops if o["op"] != "match")
    us = 0; direct = 0
    for o in ops:
        if o["op"] != "match": continue
        f = C.feats(c["chars"], new, o["nom_idx"], o["syl_idx"], pg)
        t = tier_v3(f, 1.0); us += t != "REVIEW"; direct += f["direct"]
    return gaps, us, direct, len(new)
tot_amb = 0; tot_nofill = 0
for c in cols:
    m, n = len(c["chars"]), len(c["syl"])
    if m <= n or (c["book"] == "stt4" and c["page"] == "page_0146"): continue
    gap = m-n
    oov = [(i, s) for i, s in enumerate(c["syl"]) if len(s) >= 5 and s.lower() not in qs]
    if not oov: continue
    # nghiệm CHẶT (= gap+1)
    per = []
    for _, s in oov:
        opts = {k: splits(s, k) for k in range(1, min(len(s), gap+1)+1)}
        per.append({k: v for k, v in opts.items() if v})
    strict = []
    for ks in itertools.product(*[list(o.keys()) for o in per]):
        if sum(k-1 for k in ks) == gap:
            strict.extend(itertools.product(*[per[i][k] for i, k in enumerate(ks)]))
    base = score(c, c["syl"])
    if len(strict) > 1:
        res = [(combo, score(c, apply(c, oov, combo))) for combo in strict]
        clean = [r for r in res if r[1][0] == 0]
        best = max(res, key=lambda r: (r[1][1], -r[1][0]))
        print(f"AMB    {c['book']} {c['page']} c{c['column']} gap={gap} {[s for _,s in oov]} | gốc (khe,usable,direct)={base[:3]} | " +
              " ; ".join(f"{'|'.join('+'.join(p) for p in r[0])}→{r[1][:3]}" for r in res) + f" | sạch khe: {len(clean)}")
        tot_amb += max(0, best[1][1] - base[1]) if len(clean) == 1 else 0
    elif not strict:
        # nới: mọi tổ hợp k_i in 2..gap+1, tổng mảnh thêm <= gap, mảnh ∈ lex
        relaxed = []
        for ks in itertools.product(*[[k for k in o.keys() if k >= 2] for o in per]):
            if 0 < sum(k-1 for k in ks) <= gap:
                relaxed.extend(itertools.product(*[per[i][k] for i, k in enumerate(ks)]))
        if not relaxed:
            print(f"NOFILL {c['book']} {c['page']} c{c['column']} gap={gap} {[s for _,s in oov]} | không có cách tách nào ∈ từ điển (kể cả nới)")
            continue
        res = [(combo, score(c, apply(c, oov, combo))) for combo in relaxed]
        best = max(res, key=lambda r: (r[1][1], -r[1][0]))
        print(f"NOFILL {c['book']} {c['page']} c{c['column']} gap={gap} {[s for _,s in oov]} | gốc={base[:3]} | nới {len(relaxed)} nghiệm, tốt nhất {'|'.join('+'.join(p) for p in best[0])}→{best[1][:3]}")
        tot_nofill += max(0, best[1][1] - base[1]) if len(relaxed) == 1 else 0
print(f"\nΔusable(p=1) nếu chọn nghiệm DUY NHẤT sạch khe: AMB={tot_amb}; NOFILL nới duy nhất={tot_nofill}")
