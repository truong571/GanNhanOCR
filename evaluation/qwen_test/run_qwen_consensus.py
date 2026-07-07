"""Qwen3-VL-Flash vs Kinhhannom — consensus feasibility test on N pages.

Design (chốt với anh Truong):
  • Kinhhannom = coordinate BASE (per-char bbox). Qwen votes on the CHARACTER only.
  • Qwen reads the full page -> flat char sequence -> aligned (Levenshtein) to kim's
    char sequence. No bbox is taken from Qwen.
  • BONUS oracle: at every kim!=qwen disagreement we crop kim's bbox and ask the
    trained NomEncoder ArcFace head "does this pixel look more like kim's char or
    qwen's char?" -> estimates whether Qwen FIXES kim or just adds noise.

Run:
  .venv/bin/python -m evaluation.qwen_test.run_qwen_consensus --book SachThanhTruyen4 --n 10
Outputs: evaluation/qwen_test/cache/*.json (raw Qwen, reused), out/<page>.txt (side by
side), summary.csv, and a printed report. Re-runs are free (Qwen cached).
"""
from __future__ import annotations
import argparse, base64, csv, io, json, sys, urllib.request, urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from core.ocr.ocr_api import load_columns_fullpage           # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"; OUT = HERE / "out"
CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

QWEN_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen3-vl-flash"

PROMPT = (
    "Đây là một trang văn bản chữ Hán-Nôm khắc gỗ, viết theo CỘT DỌC, đọc từ PHẢI "
    "sang TRÁI, mỗi cột từ TRÊN xuống DƯỚI. Hãy phiên chính xác TẤT CẢ các chữ "
    "Hán-Nôm trong ảnh theo đúng thứ tự đọc đó.\n"
    "QUY TẮC XUẤT:\n"
    "- MỖI CỘT MỘT DÒNG (cột phải nhất là dòng đầu tiên).\n"
    "- CHỈ xuất ký tự Hán-Nôm. TUYỆT ĐỐI không phiên âm Quốc Ngữ, không dấu câu, "
    "không số, không giải thích, không thêm bất kỳ chữ nào khác.\n"
    "- Không bỏ sót và không bịa thêm chữ."
)


def _env(k: str) -> str:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith(k + "=") or line.startswith(k + " ="):
            return line.partition("=")[2].strip().strip("'\"")
    return ""


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2EBEF or 0x2F800 <= o <= 0x2FA1F)


def call_qwen(image_path: Path, cache_path: Path, key: str) -> str:
    if cache_path.exists():
        return json.loads(cache_path.read_text())["text"]
    from PIL import Image
    im = Image.open(image_path).convert("RGB")
    if max(im.size) > 2048:                       # cap tokens; native is usually smaller
        r = 2048 / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = json.dumps({
        "model": QWEN_MODEL, "temperature": 0, "max_tokens": 2048,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    }).encode()
    req = urllib.request.Request(QWEN_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    cache_path.write_text(json.dumps({"text": text, "usage": usage}, ensure_ascii=False, indent=1))
    return text


def parse_qwen_cols(text: str) -> list[list[str]]:
    cols = []
    for line in text.splitlines():
        chars = [c for c in line if is_cjk(c)]
        if chars:
            cols.append(chars)
    return cols


def align(a: list[str], b: list[str]):
    """Global Levenshtein backtrace. Returns list of (op, a_char, b_char).
    op in {match, sub, del(a only), ins(b only)}."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            ops.append(("match" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, b[j - 1])); j -= 1
    ops.reverse()
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen4")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pages", default="", help="comma list of page stems e.g. page_0012,page_0014")
    ap.add_argument("--no-head", action="store_true", help="skip the visual-head oracle")
    args = ap.parse_args()
    key = _env("Qwen3-VL-Flash")
    assert key, "no Qwen3-VL-Flash key in .env"

    pg_dir = REPO / "prepared" / args.book / "pages_denoised"
    det_dir = REPO / "prepared" / args.book / "detected"
    if args.pages:
        stems = args.pages.split(",")
    else:
        stems = [f.stem for f in sorted(pg_dir.glob("page_*.png"))
                 if (det_dir / f"{f.stem}_ocr_cache.json").exists()][:args.n]

    # optional visual-head oracle
    enc = None; lab2idx = {}; tighten = None
    if not args.no_head:
        try:
            from pipeline.align_engine.nom_classifier.infer import NomEncoder
            from pipeline.align_engine.bbox_fix import tighten_box as tighten
            import numpy as np                                        # noqa
            enc = NomEncoder(ckpt=str(REPO / "nom-embed" / "best.pt"))
            lab2idx = {v: k for k, v in enc.classes.items()} if enc.classes else {}
            print(f"[head] loaded on {enc.device}, {len(lab2idx)} classes")
        except Exception as e:
            print(f"[head] disabled: {type(e).__name__}: {e}")

    def head_favor(page_im, bbox, kc, qc):
        """Return 'kim'|'qwen'|'neither'|'oov' — which char the pixel looks like."""
        if enc is None:
            return "off"
        import numpy as np
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return "oov"
        gray = np.asarray(page_im.crop((x1, y1, x2, y2)).convert("L"))
        tb = tighten(gray) if tighten else None
        if tb is not None:
            a, c, b, d = tb
            if b - a >= 8 and d - c >= 8:
                gray = gray[c:d, a:b]
        lg = enc.logits(enc.embed_gray(gray))
        ik, iq = lab2idx.get(kc), lab2idx.get(qc)
        if ik is None and iq is None:
            return "oov"
        ck = float(lg[ik]) if ik is not None else -9
        cq = float(lg[iq]) if iq is not None else -9
        return "kim" if ck > cq else "qwen"

    rows = []
    tot_kim = tot_qwen = tot_match = tot_sub = tot_del = tot_ins = 0
    fav = {"kim": 0, "qwen": 0, "oov": 0, "off": 0}
    from PIL import Image
    print("kCol = kim columns (MUST be 9, authority) | qSeg = qwen's own line count (info only)")
    print(f"\n{'page':14} {'kimC':>5} {'qwnC':>5} {'kCol':>5} {'qSeg':>5} "
          f"{'match%':>7} {'sub':>4} {'del':>4} {'ins':>4}")
    print("-" * 72)
    for stem in stems:
        img = pg_dir / f"{stem}.png"
        cache = det_dir / f"{stem}_ocr_cache.json"
        if not img.exists() or not cache.exists():
            print(f"{stem:14} MISSING"); continue
        kim_cols = load_columns_fullpage(str(cache), str(img))
        n_kcol = len(kim_cols)                              # kim = 9-column AUTHORITY
        kim_flat = [(c["char"], c["bbox"], ci) for ci, col in enumerate(kim_cols) for c in col]
        kim_chars = [c for c, _, _ in kim_flat]
        try:
            qtext = call_qwen(img, CACHE / f"{args.book}_{stem}.json", key)
        except urllib.error.HTTPError as e:
            print(f"{stem:14} QWEN HTTP {e.code}: {e.read().decode()[:120]}"); continue
        qcols = parse_qwen_cols(qtext)
        qchars = [c for col in qcols for c in col]
        ops = align(kim_chars, qchars)
        # Project every model onto KIM's 9-column grid (kim = authority). Qwen's own
        # line count (8-10) is IGNORED for structure — only a segmentation-quality flag.
        percol = [[0, 0, 0] for _ in range(n_kcol)]        # [match, sub, del] per kim column
        page_favors = []
        qim = None
        ki = 0
        for o in ops:
            if o[0] == "ins":                              # qwen-only extra char: no kim slot
                continue
            col = kim_flat[ki][2] if ki < len(kim_flat) else n_kcol - 1
            if o[0] == "match":
                percol[col][0] += 1
            elif o[0] == "sub":
                percol[col][1] += 1
                if enc is not None:
                    if qim is None:
                        qim = Image.open(img).convert("RGB")
                    f = head_favor(qim, kim_flat[ki][1], o[1], o[2])
                    fav[f if f in fav else "oov"] += 1
                    page_favors.append((o[1], o[2], f))
            else:                                          # del (kim-only)
                percol[col][2] += 1
            ki += 1
        match = sum(p[0] for p in percol)
        sub = sum(p[1] for p in percol)
        dele = sum(p[2] for p in percol)
        ins = sum(1 for o in ops if o[0] == "ins")
        denom = match + sub or 1
        colpct = " ".join(f"{100 * p[0] // (p[0] + p[1] or 1):02d}" for p in percol)
        flag = "" if n_kcol == 9 else f"  <<WARN kim={n_kcol}!=9"
        print(f"{stem:14} {len(kim_chars):5} {len(qchars):5} {n_kcol:5} "
              f"{len(qcols):5} {100 * match / denom:6.1f}% {sub:4} {dele:4} {ins:4}{flag}")
        print(f"   cột1-9 khớp%: {colpct}")
        # side-by-side dump
        lines = [f"# {stem}  kim={len(kim_chars)} qwen={len(qchars)} "
                 f"| kim_cols={n_kcol}{'' if n_kcol == 9 else ' WARN!=9'} qwen_seg={len(qcols)}",
                 f"# cột1-9 khớp%: {colpct}", ""]
        for o in ops:
            tag = {"match": "  ", "sub": "XX", "del": "k-", "ins": "-q"}[o[0]]
            lines.append(f"{tag}  kim={o[1] or '·'}  qwen={o[2] or '·'}")
        if page_favors:
            lines += ["", "# head oracle at disagreements (kim vs qwen):"]
            lines += [f"  kim={k} qwen={q} -> pixel looks like: {f}" for k, q, f in page_favors]
        (OUT / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        rows.append([stem, len(kim_chars), len(qchars), len(kim_cols), len(qcols),
                     round(100 * match / denom, 1), sub, dele, ins])
        tot_kim += len(kim_chars); tot_qwen += len(qchars); tot_match += match
        tot_sub += sub; tot_del += dele; tot_ins += ins

    with open(HERE / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["page", "kim_chars", "qwen_chars", "kim_cols",
                                       "qwen_cols", "match_pct", "sub", "del", "ins"])
        w.writerows(rows)

    d = tot_match + tot_sub or 1
    print("-" * 72)
    print(f"TOTAL kim_chars={tot_kim} qwen_chars={tot_qwen} | aligned match={tot_match} "
          f"({100*tot_match/d:.1f}%) sub={tot_sub} del(kim-only)={tot_del} ins(qwen-only)={tot_ins}")
    if enc is not None:
        fk, fq, fo = fav["kim"], fav["qwen"], fav["oov"]
        fd = fk + fq or 1
        print(f"\nHEAD ORACLE at {fk+fq+fo} kim!=qwen disagreements (which char the PIXEL matches):")
        print(f"  pixel looks like KIM  : {fk} ({100*fk/fd:.0f}% of decidable) -> Qwen WRONG here")
        print(f"  pixel looks like QWEN : {fq} ({100*fq/fd:.0f}% of decidable) -> Qwen FIXED a kim error")
        print(f"  undecidable (OOV head): {fo}")
        print(f"  => of kim/qwen disagreements, Qwen is right ~{100*fq/fd:.0f}%, kim right ~{100*fk/fd:.0f}%")
    print(f"\nSide-by-side per page -> {OUT}/  | summary -> {HERE/'summary.csv'}")


if __name__ == "__main__":
    main()
