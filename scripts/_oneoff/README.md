# scripts/_oneoff/ — script CHẠY MỘT LẦN (giữ làm hồ sơ xuất xứ dữ liệu)

Đây là các script đã dùng để **tải / dựng dữ liệu nguồn** trong `data/` và để đo một lần.
Không nhánh nào của `run_pipeline.sh`, `scripts/measure/measure.py` hay các selftest gọi chúng
(kiểm bằng `scripts/maintenance/dep_graph.py --check`). Giữ lại vì chúng là **chuỗi tái lập**
của phần dữ liệu trong luận văn (NLVNPF / BnF / IHR-NomDB) — xoá là mất dấu vết xuất xứ.

Dời về đây ngày 23/09 (trước đó nằm rải ở gốc `scripts/`, riêng `ra_soat_2026-09-13_do_lai.py`
nằm nhầm trong `docs/`). Đường dẫn cũ → mới:

| Cũ | Mới |
|----|-----|
| `scripts/<tên>.py` (12 tệp dưới đây) | `scripts/_oneoff/<tên>.py` |
| `docs/ra_soat_2026-09-13_do_lai.py` | `scripts/_oneoff/ra_soat_2026-09-13_do_lai.py` |

- Tải dữ liệu: `download_chrestomathie1872.py`, `download_handwritten_manuscripts.py`,
  `download_kieu1884_bilingual.py`, `download_quocngu_texts.py`
- Dựng/bóc dữ liệu: `build_ihr_manifest.py`, `build_lucvantien1883_dataset.py`,
  `extract_kieu1894_pages.py`, `extract_stt_quocngu_pages.py`, `ocr_all_quocngu_pages.py`,
  `delete_pua_codepoints.py`
- Kiểm/báo cáo một lần: `check_ocr_token.py`, `make_flow_report_html.py`,
  `ra_soat_2026-09-13_do_lai.py` (số liệu của `docs/RA_SOAT_TOAN_BO_2026-09-13.md`)

Khác `_retired/`: `_retired/` là phương án **đã thử và loại bỏ** (bằng chứng phản chứng);
`_oneoff/` là việc **đã làm xong một lần**, còn dùng lại được nếu phải dựng lại dữ liệu.
