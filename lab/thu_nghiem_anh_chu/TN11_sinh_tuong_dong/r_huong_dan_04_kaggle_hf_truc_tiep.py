"""r_huong_dan_04_kaggle_hf_truc_tiep.py — REVIEW hướng "huong_dan" (4/5): kiểm LẠI TRỰC TIẾP (hôm nay) các khẳng định về Kaggle/HF trong
docs/KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md từ nguồn chính thức, dùng lại bộ lấy tin của w_runner_gioi_han_kaggle.py (docs Kaggle qua điểm cuối
CMS công khai, bài nhân viên Kaggle qua điểm cuối diễn đàn, mã JS giao diện, docs HF) và bổ sung: NGÀY của bài nhân viên, chuỗi giao diện
(hộp "Accelerator", "Requires phone verification", "Save & Run All with an accelerator", Secrets/Attach, môi trường). Cần mạng, KHÔNG đăng nhập,
0 GPU/0 API trả phí. KHÔNG ghi đè gioi_han_kaggle.json của gói (ghi vào thư mục review).

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_huong_dan_04_kaggle_hf_truc_tiep.py
Ra: measure_out/_tn11/full/review/huong_dan/04_kaggle_hf_truc_tiep.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import html
import importlib.util
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL = REPO / "measure_out" / "_tn11" / "full"
OUT = FULL / "review" / "huong_dan"
spec = importlib.util.spec_from_file_location("G", HERE / "w_runner_gioi_han_kaggle.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "invariants": {}, "lan_chay_truoc": None, "kiem_lai_45": {}, "bo_sung": {}}


def inv(name, ok, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:300], flush=True)


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def staff_date(k, tid: int, rx: str):
    """Ngày đăng (postDate) của bình luận NHÂN VIÊN khớp rx trong bài tid."""
    r = k.call("discussions.DiscussionsService/GetForumTopicById", {"forumTopicId": tid, "includeComments": True})["forumTopic"]
    hits = []

    def walk(cs):
        for c in cs:
            a = c.get("author") or {}
            t = c.get("rawMarkdown") or ""
            if a.get("tier") == "STAFF" and re.search(rx, flat(t)):
                hits.append(dict(ngay=c.get("postDate"), trich=flat(t)[:200]))
            walk(c.get("replies") or [])
    walk(r.get("comments", []))
    first = r.get("firstMessage") or {}
    return dict(tieu_de=r.get("name"), ngay_bai=r.get("postDate"), binh_luan_nhan_vien_khop=hits[:2], so_binh_luan=r.get("totalMessages"),
                ngay_binh_luan_cuoi=max([c.get("postDate") or "" for c in r.get("comments", [])] or [""]))


def main():
    prev = json.loads((FULL / "runner" / "gioi_han_kaggle.json").read_text(encoding="utf-8"))
    R["lan_chay_truoc"] = prev["tao"]
    k = G.Kaggle()

    # ---------------------------------------------------------------- A. chạy lại 45 phép kiểm của gói (đọc nguồn HÔM NAY)
    cache = {}
    ok_now = {}
    for key, kind, ref, rx, url in G.CHECKS:
        try:
            if kind == "kdoc":
                text = k.doc(ref)
            elif kind == "ktopic":
                tid, who = ref
                text = " ".join(t for w, t in k.topic(tid) if who == "any" or w == who)
            elif kind == "kjs":
                text = k.js(ref)
            elif kind == "gh":
                u = f"https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/{ref}"
                text = cache.setdefault(u, G.fetch_text(u))
            else:
                u = f"https://huggingface.co/docs/{'hub' if kind == 'hf' else 'huggingface_hub/guides'}/{ref}"
                raw = cache.setdefault(u, G.fetch_text(u))
                text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))
            text = re.sub(r"\s+", " ", text) if kind != "kjs" else text
            m = re.search(rx, text, re.S)
            ok_now[key] = bool(m)
            R["kiem_lai_45"][key] = dict(tim_thay_hom_nay=bool(m), tim_thay_lan_truoc=prev["checks"].get(key, {}).get("found"))
        except Exception as e:  # noqa: BLE001
            ok_now[key] = False
            R["kiem_lai_45"][key] = dict(tim_thay_hom_nay=False, loi=f"{type(e).__name__}: {str(e)[:100]}")
    inv("A1.45_phep_kiem_cua_goi_con_hieu_luc_hom_nay", all(ok_now.values()) and len(ok_now) == 45, dict(tim_thay=sum(ok_now.values()), tong=len(ok_now), truot=[x for x, v in ok_now.items() if not v]))

    # ---------------------------------------------------------------- B. NGÀY của các bài nhân viên mà hướng dẫn dựa vào
    d361 = staff_date(k, 361104, r"2xT4 setup only consumes 1 hour of your GPU quota")
    R["bo_sung"]["bai_361104_T4x2_1gio_quota"] = d361
    inv("B1.bai_nhan_vien_T4x2_1_gio_quota_dang_10_2022", bool(d361["binh_luan_nhan_vien_khop"]) and d361["binh_luan_nhan_vien_khop"][0]["ngay"].startswith("2022-10"), d361["binh_luan_nhan_vien_khop"][:1])
    R["bo_sung"]["bai_361104_binh_luan_moi_nhat"] = d361["ngay_binh_luan_cuoi"]
    d173 = staff_date(k, 173129, r"Quota is enforced at the beginning of a new session")
    R["bo_sung"]["bai_173129_quota"] = d173
    inv("B2.bai_nhan_vien_ve_quota_co_ngay", bool(d173["binh_luan_nhan_vien_khop"]), d173["binh_luan_nhan_vien_khop"][:1])

    # ---------------------------------------------------------------- C. tài liệu Kaggle hiện hành (không phụ thuộc bài cũ)
    nb = flat(k.doc(33335))
    ef = flat(k.doc(63991))
    inv("C1.docs_hien_hanh_khong_noi_he_so_T4x2_chi_noi_quota_GPU_30h", "30 hours or sometimes higher" in ef and "T4" not in ef, "efficient-gpu-usage: không nhắc T4; nói 'quota GPU 30 giờ/tuần hoặc cao hơn'")
    m = re.search(r"make sure \"Internet\" is enabled in the Settings pane \(([^)]*)\)", nb)
    R["bo_sung"]["docs_internet"] = m.group(0) if m else None
    inv("C2.docs_Internet_mac_dinh_bat_voi_notebook_moi", bool(m) and "by default" in m.group(1), m.group(1) if m else None)
    m2 = re.search(r"Save & Run All is identical to the .{0,10}Commit.{0,10} behavior", nb)
    inv("C3.Save_Run_All_chinh_la_Commit", bool(m2), m2.group(0) if m2 else None)
    m3 = re.search(r"reuse that data in any future Notebook:.{0,200}Add Input.{0,120}Notebook with output files", nb)
    inv("C4.dau_ra_notebook_dung_lam_dau_vao_notebook_khac_qua_Add_Input", bool(m3), None)

    # ---------------------------------------------------------------- D. chuỗi giao diện trong mã JS KernelEditor (bundle đang chạy trên kaggle.com)
    js = k.js("KernelEditor")
    R["bo_sung"]["js_kernel_editor_ky_tu"] = len(js)
    S = {
        "hop_Accelerator_hours_this_week_Quota_resets_weekly": r'" hours this week\. Quota resets weekly with a minimum "',
        "hop_Accelerator_You_have_N_hours_remaining": r'" You have ",d," hours remaining\."',
        "Requires_phone_verification_thay_o_chon_Accelerator": r'"Requires phone verification"',
        "Want_more_power_GPU_TPU_hoac_internet_can_xac_minh_dien_thoai": r"Want more power\? Access GPU/TPU at no cost or turn on an internet connection\.",
        "Save_Run_All_with_an_accelerator": r'label:"Save & Run All with an accelerator"',
        "tuy_chon_gpu_once": r'label:"Run with GPU for this session",value:"gpu_once"',
        "mac_dinh_batch_theo_phien_tuong_tac": r'function C\(\)\{return n&&n\.batchSessionDefaultAccelerator&&\(0,tK\.GY\)\(n\.batchSessionDefaultAccelerator\)\?"gpu_always":\(0,tK\.GY\)\(r\)\?"gpu_once":\(0,tK\.To\)\(r\)\?"tpu_once":"none"\}',
        "Pin_to_original_environment": r'"Pin to original environment"',
        "Always_use_latest_environment": r'"Always use latest environment"',
        "Add_ons_Secrets_trong_menu_File": r'"Add-ons" > "Secrets" in the File menu',
        "nut_Attach_secret": r'acceptLabel:"Attach"',
        "menu_Add_input": r'text:"Add input"',
        "quota_GPU_va_TPU_hai_bo_dem_rieng": r"gpu:t,tpu:n,quotaRefreshTime",
        "tin_quota_con_lai_ban_dau": r'" hours remaining on your "\)\.concat\(n," quota this week\."',
    }
    found = {}
    for name, rx in S.items():
        found[name] = bool(re.search(rx, js))
    R["bo_sung"]["chuoi_giao_dien"] = found
    inv("D1.chuoi_giao_dien_quan_trong_con_trong_bundle_hom_nay", all(found.values()), {k_: v for k_, v in found.items() if not v})
    # trích bản gốc ngắn để dẫn chứng
    for name, rx in (("hop_Accelerator", r"Availability is limited to .{0,700}hours remaining\."), ("phone", r'to\.createElement\(tT\.Yq,\{href:"/account/phone/number\?returnUrl="\+encodeURI\(window\.location\.pathname\)\},"Requires phone verification"\)'),
                     ("mac_dinh_batch", r'function C\(\)\{return n&&n\.batchSessionDefaultAccelerator.{0,200}')):
        mm = re.search(rx, js)
        R["bo_sung"][f"trich_js_{name}"] = mm.group(0)[:360] if mm else None

    # ---------------------------------------------------------------- E. HF: CLI + token
    def hf_text(url):
        raw = G.fetch_text(url)
        return flat(html.unescape(re.sub(r"<[^>]+>", " ", raw)))
    try:
        cli = hf_text("https://huggingface.co/docs/huggingface_hub/guides/cli")
        R["bo_sung"]["hf_cli_trang"] = dict(co_hf_download=bool(re.search(r"hf download", cli)), co_local_dir=bool(re.search(r"--local-dir", cli)), co_repo_type=bool(re.search(r"--repo-type", cli)),
                                          co_hf_auth_login=bool(re.search(r"hf auth login", cli)), co_HF_TOKEN=bool(re.search(r"HF_TOKEN", cli)))
        inv("E1.docs_HF_CLI_co_hf_download_local_dir_repo_type_va_hf_auth_login", all(R["bo_sung"]["hf_cli_trang"].values()), R["bo_sung"]["hf_cli_trang"])
    except Exception as e:  # noqa: BLE001
        inv("E1.docs_HF_CLI", False, f"{type(e).__name__}: {e}")
    try:
        tok = hf_text("https://huggingface.co/docs/hub/security-tokens")
        mm = re.search(r"(Write|write)[^.]{0,200}(create|upload|push)[^.]{0,120}", tok)
        R["bo_sung"]["hf_token_trang"] = dict(nhac_write=bool(re.search(r"\bwrite\b", tok, re.I)), nhac_fine_grained=bool(re.search(r"fine-grained", tok, re.I)), trich=mm.group(0)[:240] if mm else None)
        inv("E2.docs_HF_token_co_loai_write_va_fine_grained", R["bo_sung"]["hf_token_trang"]["nhac_write"] and R["bo_sung"]["hf_token_trang"]["nhac_fine_grained"], R["bo_sung"]["hf_token_trang"])
    except Exception as e:  # noqa: BLE001
        inv("E2.docs_HF_token", False, f"{type(e).__name__}: {e}")
    try:
        up = hf_text("https://huggingface.co/docs/huggingface_hub/guides/upload")
        mm = re.search(r"upload_folder.{0,400}", up)
        inv("E3.docs_HF_upload_folder_va_allow_patterns", "allow_patterns" in up and "upload_folder" in up, mm.group(0)[:200] if mm else None)
        mm2 = re.search(r"(too many|many) (files|commits).{0,300}|upload_large_folder.{0,300}", up)
        R["bo_sung"]["hf_upload_ghi_chu_nhieu_tep"] = mm2.group(0)[:300] if mm2 else None
    except Exception as e:  # noqa: BLE001
        inv("E3.docs_HF_upload", False, f"{type(e).__name__}: {e}")

    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]),
                     fail_ten=[x for x, v in R["invariants"].items() if not v["ok"]])
    (OUT / "04_kaggle_hf_truc_tiep.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
