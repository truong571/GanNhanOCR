"""Sửa DẤU THANH do VietOCR đọc sai cột Quốc ngữ — chỉ nhóm chắc chắn, không đoán.

PHẠM VI: HẸP CÓ CHỦ Ý
---------------------
Toàn corpus có 1.851 ô mà âm Quốc ngữ lệch dấu thanh so với từ điển. Tuyệt đại đa số
KHÔNG được đụng vào, vì "lệch thanh" và "sai chữ" là hai giả thuyết ngang nhau:

    異 đọc "là"  — "là" là từ có thật, nên hoặc VietOCR rơi dấu nặng (lạ -> là),
                   hoặc chữ trong ô không phải 異. Sửa thanh tự động sẽ CHE mất
                   khả năng thứ hai, tức là giấu đúng lớp lỗi đang cần tìm.

Chỉ sửa khi âm đọc ra **không phải một từ tiếng Việt có thật** (không có trong từ điển
với tư cách khoá). Lúc đó không còn giả thuyết nào khác ngoài lỗi dấu:

    礼 đọc "trấy" — "trấy" không tồn tại, chữ 礼 đọc "trẩy" -> sửa
    孛 đọc "but"  — "but" không tồn tại, chữ 孛 đọc "bụt"  -> sửa

CHỌN ĐÍCH: HAI QUY TẮC, KHÔNG CÓ QUY TẮC THỨ BA
-----------------------------------------------
  1. bỏ dấu thanh xong khớp ĐÚNG MỘT âm của chính chữ đó       -> lấy âm ấy
  2. khớp nhiều âm, nhưng chỉ MỘT âm từng được từ điển xác nhận
     cho chữ đó ở nơi khác TRONG CHÍNH CORPUS NÀY               -> lấy âm ấy

Quy tắc 2 không phải suy đoán: nó dùng bằng chứng đã có sẵn trong dữ liệu. 噲 đọc "goi"
có hai đích (gọi/gỏi), nhưng 噲 xuất hiện 17 lần với "gọi" và 0 lần với "gỏi".

Còn nhập nhằng thì ĐỂ NGUYÊN và ghi vào báo cáo. Thà còn 15 ô chờ người xem còn hơn
sửa bừa 15 ô rồi không ai biết chỗ nào đã bị đoán.

KHÔNG GHI ĐÈ BỘ CÔNG BỐ
-----------------------
Ghi ra file riêng, giữ cột `syllable_raw`. Chỗ đúng của phép sửa này là bước normalize
TRƯỚC build (xem docs/FLOW_CAP_NHAT_2026-08-19.md); file ở đây là bằng chứng đo được
rằng phép sửa chạy đúng, không phải bản thay thế.

    .venv/bin/python -m pipeline.tools.fix_tone [--apply]
"""
from __future__ import annotations

import argparse
import collections
import json
import unicodedata as ud
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO / "dataset_out" / "labels_final.csv"
DEFAULT_OUT = REPO / "dataset_out" / "labels_tonefix.csv"
DEFAULT_REPORT = REPO / "dataset_out" / "tonefix_report.json"


def strip_tone(s: str) -> str:
    """Bỏ dấu THANH, giữ dấu tạo chữ (ă â ê ô ơ ư đ).

    Bỏ sạch mọi dấu thì "vừa" và "vua" thành một, và phép sửa sẽ nối hai từ khác hẳn
    nhau. Giữ lại dấu tạo chữ nên chỉ có sáu thanh bị gộp — đúng cái VietOCR đọc sai.
    """
    keep = "̛̆̂"
    return ud.normalize("NFC", "".join(
        c for c in ud.normalize("NFD", str(s))
        if not ("̀" <= c <= "̣" and c not in keep)))


def plan(labels: pd.DataFrame, qn: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    readings: dict[str, set[str]] = collections.defaultdict(set)
    for syl, chars in qn.items():
        for ch in chars:
            readings[ch].add(syl)
    keys = set(qn)

    rows = labels[labels["ocr_char"].notna() & labels["syllable"].notna()]
    syl_l = rows["syllable"].str.lower()

    # Âm mà từ điển ĐÃ xác nhận cho chính chữ đó, đếm trên chính corpus này.
    attested: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for ch, s in zip(rows["ocr_char"], syl_l):
        if s in readings[ch]:
            attested[ch][s] += 1

    fixes, ambiguous = [], []
    for idx, ch, s in zip(rows.index, rows["ocr_char"], syl_l):
        if s in readings[ch] or s in keys:
            continue                       # đã khớp, hoặc là từ có thật -> không đụng
        cand = sorted(x for x in readings[ch] if strip_tone(x) == strip_tone(s))
        if not cand:
            continue
        if len(cand) == 1:
            fixes.append((idx, ch, s, cand[0], "unique"))
            continue
        seen = [c for c in cand if attested[ch].get(c)]
        if len(seen) == 1:
            fixes.append((idx, ch, s, seen[0], f"corpus:{attested[ch][seen[0]]}"))
        else:
            ambiguous.append((ch, s, "/".join(cand)))

    f = pd.DataFrame(fixes, columns=["idx", "ocr_char", "syllable_raw", "syllable", "rule"])
    amb = collections.Counter(ambiguous)
    return f, {"ambiguous": [{"ocr_char": a, "syllable_raw": b, "candidates": c, "n": n}
                             for (a, b, c), n in amb.most_common()]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.tools.fix_tone")
    ap.add_argument("--labels", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--apply", action="store_true", help="thiếu cờ này = chỉ in, không ghi")
    args = ap.parse_args(argv)

    from core.text.dictionary import load_qn_to_nom
    qn = load_qn_to_nom(str(REPO / "Dict" / "QuocNgu_SinoNom.csv"))
    labels = pd.read_csv(args.labels, dtype=str, low_memory=False)

    fixes, extra = plan(labels, qn)
    by_rule = collections.Counter(r.split(":")[0] for r in fixes["rule"])
    print(f"[thanh] sửa được {len(fixes)} ô "
          f"(ứng viên duy nhất {by_rule['unique']}, corpus chốt {by_rule['corpus']})")
    print(f"[thanh] còn nhập nhằng {sum(a['n'] for a in extra['ambiguous'])} ô "
          f"trên {len(extra['ambiguous'])} cặp -> để nguyên")
    top = fixes.groupby(["ocr_char", "syllable_raw", "syllable"]).size().sort_values(
        ascending=False)
    for (ch, raw, new), n in top.head(8).items():
        print(f'          {ch} "{raw}" x{n:3} -> "{new}"')

    out = labels.copy()
    if "syllable_raw" not in out.columns:
        out.insert(out.columns.get_loc("syllable") + 1, "syllable_raw", "")
    out.loc[fixes["idx"], "syllable_raw"] = fixes["syllable_raw"].values
    out.loc[fixes["idx"], "syllable"] = fixes["syllable"].values

    # KIỂM CHỨNG: sau khi sửa, mọi ô vừa sửa phải khớp từ điển ĐÚNG THANH.
    readings: dict[str, set[str]] = collections.defaultdict(set)
    for syl, chars in qn.items():
        for ch in chars:
            readings[ch].add(syl)
    fixed = out.loc[fixes["idx"]]
    ok = sum(1 for ch, s in zip(fixed["ocr_char"], fixed["syllable"].str.lower())
             if s in readings[ch])
    print(f"[thanh] kiểm chứng: {ok}/{len(fixes)} ô đã sửa nay khớp từ điển đúng thanh")
    if ok != len(fixes):
        raise SystemExit("phép sửa không tự nhất quán — dừng, không ghi")

    tiers = collections.Counter(fixed["tier"])
    print(f"[thanh] tier của các ô đã sửa: {dict(tiers)}")

    report = {
        "labels_source": Path(args.labels).name,
        "n_fixed": int(len(fixes)),
        "by_rule": dict(by_rule),
        "verified_match_after_fix": int(ok),
        "tiers_of_fixed": {k: int(v) for k, v in tiers.items()},
        "pairs": [{"ocr_char": ch, "syllable_raw": raw, "syllable": new, "n": int(n)}
                  for (ch, raw, new), n in top.items()],
        **extra,
        "note": ("Chỉ sửa khi âm đọc ra không phải từ tiếng Việt có thật. Trường hợp âm "
                 "là từ có thật (異 'là' vs 'lạ') KHÔNG đụng vào — đó có thể là sai chữ "
                 "chứ không phải sai thanh, phải để người chấm."),
    }
    if args.apply:
        out.to_csv(args.out, index=False, encoding="utf-8")
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[thanh] -> {args.out}")
        print(f"[thanh] -> {args.report}")
    else:
        print("[thử] chưa ghi gì — thêm --apply để ghi thật")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
