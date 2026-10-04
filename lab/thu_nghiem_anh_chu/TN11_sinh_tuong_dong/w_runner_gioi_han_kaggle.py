"""TN11 "full" — hướng RUNNER/KAGGLE: tái kiểm bằng SCRIPT các giới hạn Kaggle/HF từ NGUỒN CHÍNH THỨC (cần mạng; 0 GPU, 0 API trả phí, không đăng nhập).

Vì sao có script: trang docs Kaggle dựng bằng JavaScript nên WebFetch chỉ thấy khung rỗng. Script gọi chính các điểm cuối công khai mà trang Kaggle dùng
(cms.LegacyCmsService/GetPage cho docs; discussions.DiscussionsService/GetForumTopicById cho bài của nhân viên Kaggle; mã JS giao diện),
tệp docs trên GitHub Kaggle/kaggle-cli và trang docs HF. Mỗi mục: mẫu regex -> trích đoạn nguyên văn -> found true/false. Số liệu nghi ngờ/trôi => chạy lại.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_runner_gioi_han_kaggle.py
Ra: measure_out/_tn11/full/runner/gioi_han_kaggle.json
"""
from __future__ import annotations

import html
import http.cookiejar
import json
import re
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "measure_out" / "_tn11" / "full" / "runner"
KG = "https://www.kaggle.com"
UA = {"User-Agent": "Mozilla/5.0"}

# (khoá, loại, tham chiếu, regex, URL báo cáo)
CHECKS = [
    ("gpu_quota_tuan", "kdoc", 63991, r"The quota resets weekly and is 30 hours or sometimes higher depending on demand and resources", f"{KG}/docs/efficient-gpu-usage"),
    ("idle_timeout_60p_trang_GPU", "kdoc", 63991, r"60 minute idle timeout limit", f"{KG}/docs/efficient-gpu-usage"),
    ("phien_toi_da_12h_CPU_GPU_9h_TPU", "kdoc", 33335, r"12 hours execution time for CPU and GPU notebook sessions and 9 hours for TPU notebook sessions", f"{KG}/docs/notebooks"),
    ("working_20GB_tu_luu", "kdoc", 33335, r"20 Gigabytes of auto-saved disk space \(/kaggle/working\)", f"{KG}/docs/notebooks"),
    ("scratch_ngoai_working_khong_luu", "kdoc", 33335, r"scratchpad disk space \(outside /kaggle/working\) that will not be saved outside of the current session", f"{KG}/docs/notebooks"),
    ("T4x2_2GPU_4CPU_29GB_RAM", "kdoc", 33335, r"T4 x2 GPU Specifications.{0,120}2 Nvidia Tesla T4 GPUs.{0,60}4 CPU cores.{0,60}29 Gigabytes of RAM", f"{KG}/docs/notebooks"),
    ("idle_tuong_tac_20p", "kdoc", 33335, r"20 minutes of idle time for your interactive session", f"{KG}/docs/notebooks"),
    ("save_run_all_phai_xong_12h", "kdoc", 33335, r"the entire Notebook must execute within 12 hours \(9 hours for TPU notebooks\)", f"{KG}/docs/notebooks"),
    ("save_run_all_phien_sach", "kdoc", 33335, r"Save & Run All creates a new session with a completely clean state", f"{KG}/docs/notebooks"),
    ("dau_ra_20GB_lam_dau_vao_phien_sau", "kdoc", 33335, r"Up to 20 GBs of output from a Notebook may be saved to disk in /kaggle/working.{0,260}Add Input", f"{KG}/docs/notebooks"),
    ("internet_bat_trong_Settings", "kdoc", 33335, r"make sure \"Internet\" is enabled in the Settings pane", f"{KG}/docs/notebooks"),
    ("anh_docker_cap_nhat_khoang_2_tuan", "kdoc", 33335, r"We update the images about every two weeks", f"{KG}/docs/notebooks"),
    ("co_the_ghim_moi_truong_goc", "kdoc", 33335, r"select the original environment that the Notebook was created with", f"{KG}/docs/notebooks"),
    ("dataset_200GB_moi_dataset", "kdoc", 33661, r"200GB per dataset limit", f"{KG}/docs/datasets"),
    ("dataset_200GB_rieng_tu", "kdoc", 33661, r"200GB max private datasets", f"{KG}/docs/datasets"),
    ("dataset_toi_da_50_tep_cap_mot", "kdoc", 33661, r"A max of 50 top-level files", f"{KG}/docs/datasets"),
    ("dataset_zip_duoc_giai_nen", "kdoc", 33661, r"Any archives \(e\.g\., ZIP files\) that you upload are uncompressed", f"{KG}/docs/datasets"),
    ("quota_noi_(floating)_hon_30h", "ktopic", (173129, "first"), r"more than 30 hours of quota in a given week", f"{KG}/product-feedback/173129"),
    ("quota_reset_thu_bay_0h_UTC", "ktopic", (173129, "first"), r"reset weekly on Saturday morning \(midnight UTC\)", f"{KG}/product-feedback/173129"),
    ("quota_tinh_luc_bat_dau_phien_va_phan_dat_truoc", "ktopic", (173129, "STAFF"), r"Quota is enforced at the beginning of a new session.{0,330}", f"{KG}/product-feedback/173129"),
    ("T4x2_tinh_1_gio_quota_cho_1_gio_chay", "ktopic", (361104, "STAFF"), r"1 hour of Notebook runtime in the 2xT4 setup only consumes 1 hour of your GPU quota", f"{KG}/product-feedback/361104"),
    ("T4_va_P100_chung_mot_quota", "ktopic", (361104, "first"), r"(?i)both GPU environments will count towards the same GPU quota", f"{KG}/product-feedback/361104"),
    ("quota_chi_tinh_thoi_gian_jupyter_chay", "ktopic", (168365, "STAFF"), r"We actually do only count time a machine is serving jupyter notebooks as quota time", f"{KG}/product-feedback/168365"),
    ("12h_cat_phien_va_giu_dau_ra", "ktopic", (317907, "STAFF"), r"12h is the current limit.{0,200}", f"{KG}/product-feedback/317907"),
    ("phien_CPU_GPU_tang_9h_len_12h", "ktopic", (302908, "first"), r"increasing session runtime limits from 9 hours to 12 hours for CPU and GPU notebooks", f"{KG}/product-feedback/302908"),
    ("working_20GiB_scratch_60GiB", "ktopic", (372506, "STAFF"), r"/kaggle/working is limited to 20GiBs.{0,260}", f"{KG}/discussions/product-feedback/372506"),
    ("persistence_chi_tuong_tac_khong_ap_dung_Save_Version", "ktopic", (355440, "first"), r"This \*only\* affects \*interactive\* Notebooks, not \*Save Versions\*", f"{KG}/product-feedback/355440"),
    ("persistence_best_effort", "ktopic", (355440, "first"), r"All persistence modes are best-effort", f"{KG}/product-feedback/355440"),
    ("secrets_gan_vao_notebook", "ktopic", (114053, "first"), r"attach any secrets you would like to be used by the current Notebook", f"{KG}/product-feedback/114053"),
    ("secrets_nhan_vien_khuyen_dung_khi_commit", "ktopic", (209530, "STAFF"), r"we recommend using the Kaggle .{0,6}Secrets", f"{KG}/discussions/general/209530"),
    ("gpu_dong_thoi_2019_1_tuong_tac_2_commit", "ktopic", (105509, "first"), r"1 interactive GPU session and 2 commit sessions", f"{KG}/general/105509"),
    ("tep_dau_ra_500_nguoi_dung_bao_cao", "ktopic", (181143, "any"), r"capped at 500 files|500 file limit", f"{KG}/product-feedback/181143"),
    ("tep_dau_ra_500_nhan_vien_khong_co_y", "ktopic", (420538, "STAFF"), r"there's no intentional 500 file limit", f"{KG}/product-feedback/420538"),
    ("ui_quota_tuan_toi_thieu", "kjs", "KernelEditor", r"Availability is limited to .{0,40}hours this week\. Quota resets weekly with a minimum", f"{KG}/static/assets (KernelEditor.*.js)"),
    ("ui_loi_phien_khong_khoi_dong_do_du_phong_quota", "kjs", "KernelEditor", r"Session cannot start because currently running sessions are projected to exceed maximum weekly", f"{KG}/static/assets (KernelEditor.*.js)"),
    ("cli_T4x2_la_GPU_mac_dinh", "gh", "docs/kernels.md", r"NvidiaTeslaT4\W{0,3}\(GPU T4 ×2, default GPU\)", "https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md"),
    ("cli_danh_sach_accelerator_09_2026", "gh", "docs/kernels.md", r"Accelerators available as of Sep 2026.{0,250}", "https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md"),
    ("cli_push_accelerator_va_timeout", "gh", "docs/kernels.md", r"--accelerator <ACCELERATOR_ID>.{0,500}-t, --timeout <SECONDS>`: Maximum run time in seconds", "https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md"),
    ("cli_machine_shape_trong_metadata", "gh", "docs/kernels_metadata.md", r"machine_shape`: The accelerator/GPU type to use.{0,120}", "https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md"),
    ("hf_toi_da_100k_tep_moi_repo", "hf", "repositories-recommendations", r"Files per repo <100k", "https://huggingface.co/docs/hub/repositories-recommendations"),
    ("hf_toi_da_10k_muc_moi_thu_muc", "hf", "repositories-recommendations", r"Entries per folder <10k", "https://huggingface.co/docs/hub/repositories-recommendations"),
    ("hf_moi_commit_duoi_100_tep", "hf", "repositories-recommendations", r"Commit size <100 files", "https://huggingface.co/docs/hub/repositories-recommendations"),
    ("hf_rieng_tu_mien_phi_100GB", "hf", "repositories-recommendations", r"Free user or org Best-effort\* 100GB", "https://huggingface.co/docs/hub/repositories-recommendations"),
    ("hf_gioi_han_commit_khong_cong_bo", "hf", "rate-limits", r"We don.t currently document the rate limits for those specific actions", "https://huggingface.co/docs/hub/rate-limits"),
    ("hf_upload_tiep_tuc_duoc", "hf_guides", "upload", r"Resumable.{0,60}re-run the same call", "https://huggingface.co/docs/huggingface_hub/guides/upload"),
]


class Kaggle:
    def __init__(self):
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.op.addheaders = list(UA.items())
        page = self.op.open(f"{KG}/docs/notebooks", timeout=60).read().decode("utf-8", "replace")
        self.runtime = re.search(r"/static/assets/runtime\.js\?v=[0-9a-f]+", page).group(0)
        xs = [c.value for c in self.cj if c.name.upper() == "XSRF-TOKEN"][0]
        self.hdr = {"Content-Type": "application/json", "X-XSRF-TOKEN": xs, "Accept": "application/json", **UA}
        self._doc, self._topic, self._js = {}, {}, {}

    def call(self, ep, body):
        req = urllib.request.Request(f"{KG}/api/i/{ep}", data=json.dumps(body).encode(), headers=self.hdr, method="POST")
        return json.loads(self.op.open(req, timeout=90).read().decode("utf-8", "replace"))

    @staticmethod
    def plain(s):
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()

    def doc(self, pid):
        if pid not in self._doc:
            self._doc[pid] = self.plain(self.call("cms.LegacyCmsService/GetPage", {"pageId": pid})["pageContent"])
        return self._doc[pid]

    def topic(self, tid):
        if tid not in self._topic:
            ft = self.call("discussions.DiscussionsService/GetForumTopicById", {"forumTopicId": tid, "includeComments": True})["forumTopic"]
            msgs = [("first", (ft.get("firstMessage") or {}).get("rawMarkdown", ""))]

            def walk(cs):
                for c in cs:
                    a = c.get("author") or {}
                    msgs.append(("STAFF" if a.get("tier") == "STAFF" else "user", c.get("rawMarkdown", "") or ""))
                    walk(c.get("replies") or [])
            walk(ft.get("comments", []))
            self._topic[tid] = msgs
        return self._topic[tid]

    def js(self, name):
        if name not in self._js:
            rt = self.op.open(KG + self.runtime, timeout=60).read().decode("utf-8", "replace")
            i = rt.find("b.u=e=>")
            seg = rt[i:]
            m = re.search(r"\(\{(.*?)\}\)\[e\]\|\|e\)\+\"\.\"\+\(\{(.*?)\}\)\[e\]\+\"\.js\"", seg, re.S)
            names = dict(re.findall(r'(\d+):"([^"]+)"', m.group(1)))
            hashes = dict(re.findall(r'(\d+):"([0-9a-f]+)"', m.group(2)))
            cid = next(k for k, v in names.items() if v == name)
            self._js[name] = self.op.open(f"{KG}/static/assets/{name}.{hashes[cid]}.js", timeout=120).read().decode("utf-8", "replace")
        return self._js[name]


def fetch_text(url):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}, "loi": {}}
    k = Kaggle()
    cache = {}
    for key, kind, ref, rx, url in CHECKS:
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
                text = cache.setdefault(u, fetch_text(u))
            elif kind in ("hf", "hf_guides"):
                u = f"https://huggingface.co/docs/{'hub' if kind == 'hf' else 'huggingface_hub/guides'}/{ref}"
                raw = cache.setdefault(u, fetch_text(u))
                text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))
            else:
                raise ValueError(kind)
            text = re.sub(r"\s+", " ", text) if kind != "kjs" else text
            m = re.search(rx, text, re.S)
            R["checks"][key] = dict(found=bool(m), nguon=url, quote=(re.sub(r"\s+", " ", m.group(0))[:420] if m else None))
        except Exception as e:  # noqa: BLE001
            R["checks"][key] = dict(found=False, nguon=url, quote=None)
            R["loi"][key] = f"{type(e).__name__}: {str(e)[:150]}"
    R["tong"] = dict(tim_thay=sum(v["found"] for v in R["checks"].values()), tong=len(R["checks"]), loi=len(R["loi"]))
    (OUT / "gioi_han_kaggle.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    for key, v in R["checks"].items():
        print(("OK  " if v["found"] else "MISS"), key, "|", (v["quote"] or "")[:140])
    print("TỔNG:", R["tong"], R["loi"])


if __name__ == "__main__":
    main()
