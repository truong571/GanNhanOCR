# Giai đoạn 1 — Sửa lỗi đã chứng minh

Tầng **pipeline chính** sửa các lỗi đã được chứng minh trên **dataset đã commit**, theo
**đúng thứ tự bắt buộc** (P0-C), deterministic, có assert, không chạy lại pipeline.

Có hai phần:

1. **`pipeline/remediation/`** (đây) — dọn dữ liệu đã commit (post-hoc).
2. **Vá mã nguồn chống tái phát** — nằm cạnh code gây lỗi (không ở đây):
   `align_engine._pick_reseg` (gán monotonic 1-1 + guard), lowercase SYLLABLE gate,
   OCR retry/backoff, cờ `--strict` cho S3.

## Ba bước, đúng thứ tự

```
1. QUARANTINE  lỗi crop trùng AE-1 ∪ F1 (một ảnh, >1 nhãn)
                 nhóm xung đột nhãn -> quarantine TẤT CẢ (không bản nào tin được)
                 nhóm trùng cùng nhãn -> giữ 1 đại diện, quarantine phần dư
2. DEMOTE      GOLD similar-bridge có s3_cosine < 0.62 -> REVIEW
                 (bằng chứng thị giác đã mâu thuẫn với ký tự look-alike)
3. DEDUP-SPLIT ép mỗi image_md5 về MỘT split, đóng rò train/test (P0-D)
                 rồi ASSERT không md5 nào span >1 split
```

Thứ tự là bắt buộc: quarantine trước nên một dòng vừa là dup-defect vừa là similar-bridge
cosine-thấp sẽ bị **quarantine** (lỗi mạnh hơn) chứ không chỉ demote; và rò split là tập
con của F1 nên quarantine đóng luôn nó (bước 3 chỉ là lưới an toàn + assert).

Mọi dòng đổi được **retag tại chỗ** (`tier -> QUARANTINE` hoặc `GOLD -> REVIEW`, rule
gắn hậu tố `|quarantine_dup` / `|demoted_lowcos_s3`) và **giữ trong file** để auditable —
không xoá âm thầm.

## Lệnh (từ gốc repo)

```bash
PY=.venv/bin/python
$PY -m pipeline.remediation census        # chỉ báo cáo census
$PY -m pipeline.remediation apply         # ghi labels_remediated.csv + remediation_report.json
```

## Kết quả đo trên dataset hiện tại

```
census : AE-1 701 rows/328 grp · F1 1,686/820 · UNION 2,321/1,116 md5-grp ·
         conflicting 1,094 · provably-wrong 1,177 (GOLD 955)
apply  : quarantined 2,299 (conflict 2,277, dup 22, kept 22 đại diện) ·
         demoted 748 similar-bridge (72 dòng còn lại đã bị quarantine trước) ·
         split-leak (original 288 md5 -> final 0) ·
         usable 68,076 -> 65,029
```

`labels_remediated.csv` là dataset đã sạch lỗi đã chứng minh; đưa nó vào train/công bố
thay cho `labels.csv`. Idempotent: chạy lại trên output không đổi gì thêm.

## Bất biến được assert

- Không dòng usable (GOLD/SILVER/SYLLABLE) nào còn nằm trong nhóm crop-trùng **xung đột
  nhãn**.
- Không `image_md5` nào span >1 giá trị `split` trong tập usable.

Nếu một trong hai vỡ, `remediate()` raise `AssertionError` (fail loud) — không xuất
dataset lỗi.

## Test

```bash
.venv/bin/python -m pipeline.remediation.selftest   # 33 assertions, exit 0 = pass
```

Kiểm: census trên frame tổng hợp có lỗi biết trước (đếm chính xác) + trên labels.csv thật
(dup_bbox=701, cross_col=1686, union=2321); remediate đúng phân loại conflict/duplicate,
demote đúng ngưỡng, đóng split-leak, idempotent, không mutate input.

## Phụ thuộc

`pandas`, `numpy`. Đầu ra: `dataset_out/labels_remediated.csv` +
`dataset_out/remediation_report.json`.
