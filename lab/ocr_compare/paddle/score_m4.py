"""Chấm M4 (cột dọc): số ký tự CJK đọc ra so với kinhhannom; LCS ký tự với chuỗi kinhhannom;
số hộp detector tìm được; hộp có phải hộp dọc (h/w≥1.5, bị xoay 90° trước rec) không."""
import sys, json
import pandas as pd
sys.path.insert(0, "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle")
from score import cjk_chars


def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x == y else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def summarize(path):
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for cfg in ("pipe", "pipe_ori", "pipe_rot90ccw", "pipe_up2", "rec_rot90ccw", "rec_rot90cw"):
        col = f"{cfg}_text"
        if col not in d:
            continue
        texts = d[col].map(lambda s: cjk_chars(s.replace("|", "")))
        kin = d.kinh_text
        n_read = texts.map(len); n_kin = kin.map(len)
        l = [lcs(a, b) for a, b in zip(texts, kin)]
        r = dict(n_cols=len(d), n_read_mean=float(n_read.mean()), n_kinh_mean=float(n_kin.mean()),
                 ratio_read_over_kinh=float((n_read / n_kin).mean()),
                 cols_with_any_cjk=int((n_read > 0).sum()),
                 cols_within_pm2_of_kinh=int(((n_read - n_kin).abs() <= 2).sum()),
                 lcs_total=int(sum(l)), lcs_over_kinh=float(sum(l) / n_kin.sum()),
                 sec_mean=float(pd.to_numeric(d.get(f"{cfg}_sec", pd.Series([0] * len(d))), errors="coerce").mean()))
        if f"{cfg}_n_boxes" in d:
            nb = pd.to_numeric(d[f"{cfg}_n_boxes"])
            r["n_boxes_mean"] = float(nb.mean())
            # hộp dọc?
            tall = 0; total = 0
            for bj in d[f"{cfg}_boxes"]:
                for poly in json.loads(bj):
                    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                    w = max(xs) - min(xs); h = max(ys) - min(ys)
                    total += 1; tall += (h / max(w, 1)) >= 1.5
            r["boxes_total"] = total; r["boxes_tall_rotated"] = tall
        out[cfg] = r
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p); print(json.dumps(summarize(p), ensure_ascii=False, indent=1))
