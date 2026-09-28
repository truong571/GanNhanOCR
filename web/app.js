/**
 * web/app.js — GanNhanOCR: giao diện trình diễn luận văn Thạc sĩ.
 * Hai chế độ: (1) máy chủ web/server.py (API trực tiếp trên dữ liệu thật); (2) mở index.html trực tiếp -> sample_data.json.
 * Mọi con số hiển thị lấy từ API / sample_data.json — không gõ cứng.
 */

const LS = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* bỏ qua */ } },
};

const AppState = {
  theme: LS.get("gannhan_theme", "light"),
  activeTab: "overview",
  isLiveServer: false,
  stats: null,
  books: [],
  pipelineFlow: [],
  benchmarks: null,
  sampleData: null,
  sampleFormat: null,
  inspector: {
    book: "LucVanTien1883",
    page: "page_0002",
    zoom: 1.0,
    chars: [],
    pageData: null,
    selectedChar: null,
    colorMode: LS.get("gannhan_color", "tier"),
    hidden: { tier: new Set(), gold_exact: new Set() },
  },
  gallery: { query: "", book: "all", tier: "all", gx: "all", results: [], loaded: false },
};

const NA = "chưa có";

// Danh mục màu: theo tầng nhãn và theo GOLD chính xác
const TIER_CATS = {
  GOLD: { label: "Mức 1 · GOLD", cls: "cat-gold" },
  SYLLABLE: { label: "Mức 2 · SYLLABLE", cls: "cat-syl" },
  GOLD_text_only: { label: "Mức 3 · text_only", cls: "cat-txt" },
  SILVER: { label: "SILVER", cls: "cat-silver" },
  REVIEW: { label: "REVIEW", cls: "cat-reviewtier" },
  QUARANTINE: { label: "QUARANTINE", cls: "cat-quarantine" },
  keep_v5: { label: "Nhãn người · keep_v5", cls: "cat-keepv5" },
  keep: { label: "Nhãn người · keep", cls: "cat-keep" },
  keep_high: { label: "Nhãn người · keep_high", cls: "cat-keephigh" },
  khong: { label: "Nhãn người · không giữ", cls: "cat-khong" },
  OTHER: { label: "Khác", cls: "cat-other" },
};
const GX_CATS = {
  ok: { label: "ok — GOLD chính xác", cls: "gx-ok" },
  text_only: { label: "text_only — chỉ tin chữ", cls: "gx-text_only" },
  uncertified: { label: "uncertified — chưa chứng nhận", cls: "gx-uncertified" },
  review: { label: "review — cần xem lại", cls: "gx-review" },
  chua_co: { label: "GOLD chưa có gold_exact", cls: "gx-chua_co" },
  khong_ap_dung: { label: "Không phải ô GOLD", cls: "gx-na" },
};
const GX_SHORT = { ok: "ok", text_only: "text_only", uncertified: "uncertified", review: "review", chua_co: "chưa có" };
const EV_CLASS = { do_tren_nhan_nguoi: "ev-do", do: "ev-do", uoc_luong: "ev-uoc", suy_doan: "ev-suy", do_tu_dong: "ev-auto" };
const ROLE_GROUPS = [
  { role: "giao_nop", label: "Giao nộp" },
  { role: "danh_gia_ihr", label: "Đánh giá — IHR-NomDB (nhãn người)" },
  { role: "danh_gia_borg", label: "Đánh giá — Borg, chép tay (nhãn người)" },
  { role: "nhan_nguoi", label: "Borg — bộ crop nhãn người" },
];

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initInspectorControls();
  initGalleryControls();
  initReloadButton();
  loadData();
});

/* ==========================================================================
   0. Tiện ích
   ========================================================================== */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmtInt(x) {
  if (x === null || x === undefined || x === "" || Number.isNaN(Number(x))) return NA;
  return Number(x).toLocaleString("vi-VN");
}
function fmtPct1(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return NA;
  return `${Number(x).toLocaleString("vi-VN", { maximumFractionDigits: 1, minimumFractionDigits: 1 })} %`;
}
function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}
function roleLabel(role) {
  const g = ROLE_GROUPS.find((x) => x.role === role);
  return g ? g.label : role || "";
}
function evBadge(level, text) {
  const cls = EV_CLASS[level] || "ev-auto";
  return `<span class="ev-badge ${cls}">${esc(text || level || "")}</span>`;
}

/* ==========================================================================
   1. Giao diện Sáng / Tối
   ========================================================================== */
function initTheme() {
  document.documentElement.setAttribute("data-theme", AppState.theme);
  document.getElementById("themeToggleBtn").addEventListener("click", () => {
    AppState.theme = AppState.theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", AppState.theme);
    LS.set("gannhan_theme", AppState.theme);
  });
}

/* ==========================================================================
   2. Điều hướng Tab
   ========================================================================== */
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab")));
  });
}

function switchTab(tabId) {
  AppState.activeTab = tabId;
  document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId));
  document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.toggle("active", pane.id === `tab-${tabId}`));
  if (tabId === "inspector") {
    loadInspectorPage();
  } else if (tabId === "gallery" && !AppState.gallery.loaded) {
    executeSearch();
  }
}

/* ==========================================================================
   3. Nạp dữ liệu (ưu tiên API -> dự phòng sample_data.json)
   ========================================================================== */
async function loadData() {
  const badge = document.getElementById("connectionBadge");
  const badgeText = badge.querySelector(".status-text");
  if (location.protocol !== "file:") try {
    const signal = (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") ? AbortSignal.timeout(15000) : undefined;
    const testResp = await fetch("/api/stats", signal ? { signal, cache: "no-store" } : { cache: "no-store" });
    if (testResp.ok) {
      AppState.isLiveServer = true;
      AppState.stats = await testResp.json();
      const [booksResp, flowResp, benchResp] = await Promise.all([
        fetch("/api/books", { cache: "no-store" }).then((r) => r.json()),
        fetch("/api/pipeline_flow", { cache: "no-store" }).then((r) => r.json()),
        fetch("/api/benchmarks", { cache: "no-store" }).then((r) => r.json()),
      ]);
      AppState.books = booksResp || [];
      AppState.pipelineFlow = flowResp || [];
      AppState.benchmarks = benchResp;
      badge.classList.add("connected");
      badgeText.textContent = "Máy chủ API: Trực tuyến";
      document.getElementById("reloadDataBtn").classList.remove("hidden");
      renderAllComponents();
      return;
    }
  } catch (err) {
    console.warn("[GanNhanOCR] API không phản hồi, chuyển sang dữ liệu mẫu:", err);
  }

  const sample = await loadSampleData();
  if (sample) {
    AppState.sampleData = sample;
    badge.classList.remove("connected");
    badgeText.textContent = "Chế độ dữ liệu mẫu";
    adaptSampleData();
    renderAllComponents();
    return;
  }
  console.error("[GanNhanOCR] Không nạp được cả API lẫn sample_data.json / sample_data.js");
  badgeText.textContent = "Ngoại tuyến (không có dữ liệu)";
}

// sample_data.json qua fetch (khi web/ được phục vụ bằng HTTP); mở index.html bằng file:// thì Chrome chặn fetch ->
// dùng sample_data.js (cùng nội dung, gán window.GANNHANOCR_SAMPLE) nạp bằng thẻ <script>.
async function loadSampleData() {
  if (window.GANNHANOCR_SAMPLE) return window.GANNHANOCR_SAMPLE;
  try {
    const r = await fetch("sample_data.json");
    if (r.ok) return await r.json();
  } catch (e) { /* file:// -> thử sample_data.js */ }
  return await new Promise((resolve) => {
    const s = document.createElement("script");
    s.src = "sample_data.js";
    s.onload = () => resolve(window.GANNHANOCR_SAMPLE || null);
    s.onerror = () => resolve(null);
    document.head.appendChild(s);
  });
}

function initReloadButton() {
  const btn = document.getElementById("reloadDataBtn");
  btn.addEventListener("click", async () => {
    if (!AppState.isLiveServer) return;
    btn.disabled = true;
    btn.textContent = "Đang nạp lại...";
    try {
      const r = await fetch("/api/reload", { cache: "no-store" });
      const d = await r.json();
      if (!d.ok) alert(d.message || "Không nạp lại được");
      AppState.gallery.loaded = false;
      await loadData();
      if (AppState.activeTab === "gallery") executeSearch();
    } catch (e) {
      alert("Lỗi khi nạp lại: " + e);
    } finally {
      btn.disabled = false;
      btn.textContent = "Nạp lại dữ liệu";
    }
  });
}

// Bước quy trình dự phòng cho sample_data.json đời cũ (không kèm số đo)
const FALLBACK_FLOW = [
  { step: 1, name: "Tiền xử lý ảnh & OCR trang", tag: "Chuẩn hoá ảnh · OCR có bộ đệm", input: "Ảnh quét trang và bản phiên âm Quốc ngữ.", model: "Kéo giãn tương phản / Otsu; OCR chữ Nôm (kim) và OCR Quốc ngữ có bộ đệm.", process: "Chuẩn hoá nền giấy, tách biên trang, dựng bộ nạp theo loại sách.", output: "prepared/<Bộ>/pages, detected/, transcriptions/", evidence: "Chạy lại toàn bộ bằng bộ đệm: không gọi API." },
  { step: 2, name: "Phát hiện vị trí ký tự", tag: "CenterNet & Pitch Decoding", input: "Ảnh cột chữ dọc.", model: "CenterNet (ResNet-34) + giải mã nhịp ký tự.", process: "Dự đoán tâm chữ, tách chữ dính bằng giải mã nhịp đều.", output: "Hộp bao cho từng chữ.", evidence: "Số đo: xem tab Kết quả khi chạy với máy chủ." },
  { step: 3, name: "Gióng hàng song ngữ", tag: "Banded Dynamic Programming", input: "Chuỗi hộp chữ và chuỗi âm Quốc ngữ.", model: "Banded DP + từ điển Quốc ngữ ↔ Hán Nôm.", process: "Ghép tối ưu hộp ↔ âm tiết, ràng buộc nhịp thơ 6/8.", output: "Nhãn sơ bộ + mã luật.", evidence: "Gán nhãn hoàn toàn theo luật." },
  { step: 4, name: "Kiểm kê & hiệu chỉnh lỗi", tag: "Rà soát nhầm lẫn hệ thống", input: "Bảng nhãn sơ bộ.", model: "Kiểm kê + bảng sửa nhầm lẫn.", process: "Xếp tầng GOLD / SYLLABLE / REVIEW.", output: "Bảng nhãn chuẩn hoá.", evidence: "Lưu vết sha256." },
  { step: 5, name: "Cổng cơ chế & đối soát dị bản", tag: "Cổng (a') · dị bản", input: "Ô nghi vấn.", model: "Cổng cơ chế + so chéo dị bản người.", process: "Ô chắc chữ nhưng không chắc ảnh -> GOLD_text_only.", output: "labels_gated.csv", evidence: "Tỉ lệ khớp dị bản là cận dưới." },
  { step: 6, name: "Đóng gói bộ theo sách", tag: "dataset/<Bộ>/", input: "Bảng nhãn đã qua cổng.", model: "export_final_dataset.", process: "Xuất crop, labels.csv 12 cột; bộ IHR/Borg gắn evaluation_only.", output: "dataset/<Bộ>/", evidence: "—" },
  { step: 7, name: "Gộp bộ dataset/_ALL", tag: "cell_uid · evaluation_only", input: "Các bộ dataset/<Bộ>/.", model: "merge_datasets.", process: "Gộp, gắn khoá cell_uid và cờ đánh giá.", output: "dataset/_ALL/", evidence: "—" },
  { step: 8, name: "GOLD chính xác (B8)", tag: "Ảnh + chữ · 4 trạng thái", input: "Mọi ô GOLD.", model: "Luật review → text_only → uncertified → ok; crop chuẩn; profile chữ viết tay.", process: "Gắn trạng thái, lý do, mức chứng cứ cho từng ô GOLD.", output: "dataset/_ALL/gold_exact.csv, crops_chuan/", evidence: "Độ chính xác chỉ ĐO được trên bộ có nhãn người." },
];

function adaptSampleData() {
  const s = AppState.sampleData;
  if (!s) return;
  if (s.format === "gannhanocr-web-v2") {
    AppState.sampleFormat = "v2";
    AppState.stats = s.stats || null;
    AppState.books = s.books || [];
    AppState.pipelineFlow = s.pipeline_flow || FALLBACK_FLOW;
    AppState.benchmarks = s.benchmarks || null;
    AppState.gallery.results = s.gallery || [];
    return;
  }
  // sample_data.json đời cũ (trước 28/09)
  AppState.sampleFormat = "legacy";
  AppState.stats = { impact_metrics: s.stats || {}, books: s.books || {}, legacy: true };
  AppState.books = s.books ? Object.values(s.books).map((b) => ({
    id: b.id, title: b.title, subtitle: b.subtitle, layout: b.layout, role: "giao_nop", status: "ok",
    total_chars: b.total, total: b.total, available_pages: [b.sample_page], sample_pages: [b.sample_page], default_page: b.sample_page,
  })) : [];
  AppState.pipelineFlow = FALLBACK_FLOW;
  AppState.benchmarks = null;
  AppState.gallery.results = s.gallery || [];
}

function renderAllComponents() {
  renderHeaderAndKpis();
  renderOverview();
  renderPipelineFlow();
  renderBenchmarks();
  populateBookSelects();
  if (AppState.activeTab === "inspector") loadInspectorPage();
}

/* ==========================================================================
   4. Tổng quan + KPI
   ========================================================================== */
function renderHeaderAndKpis() {
  const st = AppState.stats || {};
  const im = st.impact_metrics || {};
  setText("hdrTotalChars", fmtInt(im.total_characters));
  setText("hdrBooksCount", im.books_count ?? NA);
  setText("kpiTotalChars", fmtInt(im.total_characters));
  if (im.deliver_characters !== undefined) {
    setText("kpiTotalDesc", `Giao nộp ${fmtInt(im.deliver_characters)} · tập đánh giá ${fmtInt(im.eval_characters)}`);
  }
  const gr = im.gold_rate ?? im.gold_rate_overall;
  setText("kpiGoldRate", gr === null || gr === undefined ? NA : fmtPct1(gr));
  if (im.total_gold !== undefined) setText("kpiGoldDesc", `${fmtInt(im.total_gold)} ô GOLD trên tổng số ô`);
  setText("kpiSylChars", fmtInt(im.total_syllable));
  if (im.gold_exact_available) {
    setText("kpiGoldExactOk", fmtInt(im.gold_exact_ok));
    setText("kpiGoldExactDesc", `${fmtPct1(im.gold_exact_ok_pct)} của ${fmtInt(im.gold_exact_gold)} ô GOLD · ảnh + chữ cùng qua mọi cổng`);
  } else {
    setText("kpiGoldExactOk", NA);
    setText("kpiGoldExactDesc", "Chưa có dataset/_ALL/gold_exact.csv (bước B8)");
  }
  if (im.books_count !== undefined) {
    setText("kpiBooksCount", `${im.books_count} bộ`);
    const nWith = im.books_with_data ?? im.books_count;
    setText("kpiBooksDesc", nWith < im.books_count ? `${nWith}/${im.books_count} bộ đã có dữ liệu (còn lại: đang dựng lại)` : "Giao nộp + tập đánh giá có nhãn người");
  }
  if (im.invariants_available) {
    setText("kpiInvariants", `${fmtInt(im.invariants_pass)} PASS`);
    setText("kpiInvDesc", `${fmtInt(im.invariants_fail)} FAIL cứng · ${im.invariants_generated_at || ""}`);
  } else {
    setText("kpiInvariants", im.code_invariants || NA);
    setText("kpiInvDesc", "measure_out/SUMMARY.json chưa có");
  }
  setText("footerLoadedAt", st.loaded_at || (AppState.sampleData?.generated_at ? `${AppState.sampleData.generated_at} (dữ liệu mẫu)` : "—"));
  setText("footerRoot", st.data_root ? `Gốc dữ liệu: ${st.data_root}` : (AppState.sampleFormat === "legacy" ? "sample_data.json đời cũ — sinh lại: .venv/bin/python web/build_sample_data.py" : ""));
}

function renderOverview() {
  const st = AppState.stats;
  if (!st) return;
  const im = st.impact_metrics || {};
  const books = st.books || {};
  const ids = Object.keys(books);
  setText("overviewBooksBadge", `${ids.length} bộ`);
  setText("ovBooksCount", im.books_count ?? ids.length);
  setText("ovDictEntries", im.dict_entries ? fmtInt(im.dict_entries) : NA);

  const tot = im.total_characters || 0;
  const pct = (n) => (tot ? ` (${fmtPct1((n / tot) * 100)})` : "");
  setText("tierCountGold", im.total_gold !== undefined ? `${fmtInt(im.total_gold)} nhãn${pct(im.total_gold)}` : NA);
  setText("tierCountSyl", im.total_syllable !== undefined ? `${fmtInt(im.total_syllable)} nhãn${pct(im.total_syllable)}` : NA);
  setText("tierCountTxt", im.total_gold_text_only !== undefined ? `${fmtInt(im.total_gold_text_only)} nhãn${pct(im.total_gold_text_only)}` : NA);

  // Danh mục sách theo vai trò
  const listEl = document.getElementById("overviewBooksList");
  listEl.innerHTML = "";
  const byRole = {};
  Object.values(books).forEach((b) => { (byRole[b.role || "giao_nop"] = byRole[b.role || "giao_nop"] || []).push(b); });
  ROLE_GROUPS.forEach((g) => {
    const arr = byRole[g.role];
    if (!arr || !arr.length) return;
    const h = document.createElement("div");
    h.className = "book-group-title";
    h.textContent = g.label;
    listEl.appendChild(h);
    arr.forEach((b) => listEl.appendChild(bookStatItem(b)));
  });
  const human = st.human_books || {};
  if (Object.keys(human).length) {
    const h = document.createElement("div");
    h.className = "book-group-title";
    h.textContent = "Borg — bộ crop nhãn người (tách khỏi GOLD tự động)";
    listEl.appendChild(h);
    Object.values(human).forEach((b) => {
      const item = document.createElement("div");
      item.className = "book-stat-item";
      const lv = b.keep_levels || {};
      item.innerHTML = b.total
        ? `<div class="book-stat-top"><span class="book-stat-title">${esc(b.title)}</span><span class="book-stat-count">${fmtInt(b.total)} chữ người</span></div>
           <div class="book-stat-sub">keep_v5 ${fmtInt(lv.keep_v5 || 0)} · keep ${fmtInt(lv.keep || 0)} · keep_high ${fmtInt(lv.keep_high || 0)} · có ảnh ${fmtInt(b.with_image || 0)}</div>`
        : `<div class="book-stat-top"><span class="book-stat-title">${esc(b.title)}</span><span class="status-pill st-chua_co">chưa có</span></div>
           <div class="book-stat-sub">${esc(b.status_vi || "")}</div>`;
      listEl.appendChild(item);
    });
  }

  // GOLD chính xác
  const gx = st.gold_exact || {};
  setText("gxPolicyBadge", gx.policy_version ? `chính sách ${gx.policy_version}` : (gx.available ? "gold_exact.csv" : "chưa có"));
  const grid = document.getElementById("gxCardsGrid");
  grid.innerHTML = "";
  const states = gx.states || {
    ok: { label: GX_CATS.ok.label, desc: "" }, text_only: { label: GX_CATS.text_only.label, desc: "" },
    uncertified: { label: GX_CATS.uncertified.label, desc: "" }, review: { label: GX_CATS.review.label, desc: "" },
  };
  const t = gx.totals || {};
  ["ok", "text_only", "uncertified", "review"].forEach((k) => {
    const c = document.createElement("div");
    c.className = `gx-card gx-card-${k}`;
    const n = gx.available ? t[k] : null;
    const share = gx.available && t.gold ? ` (${fmtPct1((t[k] / t.gold) * 100)})` : "";
    c.innerHTML = `<div class="tier-header"><span class="gx-pill gx-pill-${k}">${esc(GX_SHORT[k])}</span><span class="tier-count">${n === null ? NA : fmtInt(n) + share}</span></div>
      <h4>${esc(states[k]?.label || k)}</h4><p>${esc(states[k]?.desc || "")}</p>`;
    grid.appendChild(c);
  });
  if (!gx.available) {
    const note = document.createElement("p");
    note.className = "table-note";
    note.textContent = "Chưa có dataset/_ALL/gold_exact.csv — bước B8 chạy sau khi gộp (./run_pipeline.sh --book all --yes).";
    grid.appendChild(note);
  }
  const evEl = document.getElementById("evidenceLegend");
  const evLevels = st.evidence_levels || { do_tren_nhan_nguoi: "ĐO trên nhãn người", uoc_luong: "ƯỚC LƯỢNG", suy_doan: "SUY ĐOÁN" };
  const evBooks = {};
  Object.values(books).forEach((b) => { if (b.evidence_level) (evBooks[b.evidence_level] = evBooks[b.evidence_level] || []).push(b.title); });
  evEl.innerHTML = `<div class="box-title">Mức chứng cứ về độ chính xác của ô ok</div>` + Object.entries(evLevels).map(([k, v]) =>
    `<div class="ev-row">${evBadge(k, v)} <span>${esc((evBooks[k] || []).join(" · ") || "—")}</span></div>`).join("");
}

function bookStatItem(b) {
  const item = document.createElement("div");
  item.className = "book-stat-item";
  const total = b.total ?? b.total_chars ?? 0;
  const gx = b.gold_exact || {};
  const hasGx = b.gold_exact_available && gx.ok !== null && gx.ok !== undefined;
  const status = b.status || "ok";
  const statusPill = status === "ok" ? "" : `<span class="status-pill st-${esc(status)}" title="${esc(b.note || b.status_vi || "")}">${status === "ban_cu" ? "bản cũ" : status === "chua_co" ? "chưa có" : esc(status)}</span>`;
  if (!total) {
    item.innerHTML = `<div class="book-stat-top"><span class="book-stat-title">${esc(b.title)}</span>${statusPill || '<span class="status-pill st-chua_co">chưa có</span>'}</div>
      <div class="book-stat-sub">${esc(b.status_vi || "Chưa có dữ liệu")}</div>`;
    return item;
  }
  item.innerHTML = `
    <div class="book-stat-top">
      <span class="book-stat-title">${esc(b.title)} ${statusPill}</span>
      <span class="book-stat-count">${fmtInt(total)} ký tự</span>
    </div>
    <div class="book-stat-sub">${esc(b.layout || b.subtitle || "")} · ${fmtInt(b.gold)} GOLD (${fmtPct1(b.gold_pct)})${hasGx ? ` · ok ${fmtInt(gx.ok)}` : ""}</div>
    <div class="progress-bar-wrap" title="Tỉ lệ GOLD${hasGx ? " / GOLD chính xác ok" : ""}">
      <div class="progress-bar-fill" style="width: ${Math.max(0, Math.min(100, Number(b.gold_pct) || 0))}%"></div>
      ${hasGx && total ? `<div class="progress-bar-gx" style="width: ${Math.max(0, Math.min(100, (gx.ok / total) * 100))}%"></div>` : ""}
    </div>`;
  return item;
}

/* ==========================================================================
   5. Quy trình xử lý
   ========================================================================== */
let currentFlowStep = 1;

function renderPipelineFlow() {
  const steps = AppState.pipelineFlow || [];
  if (!steps.length) return;
  setText("flowStepCount", steps.length);
  const navEl = document.getElementById("stepperNav");
  navEl.innerHTML = "";
  if (!steps.some((s) => s.step === currentFlowStep)) currentFlowStep = steps[0].step;
  steps.forEach((s) => {
    const btn = document.createElement("button");
    btn.className = `step-nav-btn ${s.step === currentFlowStep ? "active" : ""}`;
    btn.innerHTML = `<div class="step-top"><span class="step-index-pill">Giai đoạn ${esc(s.step)}</span></div>
      <div class="step-title-text">${esc(s.name)}</div><div class="step-tag-text">${esc(s.tag)}</div>`;
    btn.addEventListener("click", () => {
      currentFlowStep = s.step;
      document.querySelectorAll(".step-nav-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderStepDetail(s, steps.length);
    });
    navEl.appendChild(btn);
  });
  renderStepDetail(steps.find((s) => s.step === currentFlowStep) || steps[0], steps.length);
}

function renderStepDetail(s, total) {
  const metrics = (s.metrics || []).map((m) => `
    <li><span class="metric-label">${esc(m.label)}:</span> <strong>${esc(m.value)}</strong>
      ${m.level_vi ? `<span class="metric-level">${esc(m.level_vi)}</span>` : ""}
      ${m.source ? `<span class="metric-src">${esc(m.source)}</span>` : ""}</li>`).join("");
  const details = (s.details || []).map((d) => `<li>${esc(d)}</li>`).join("");
  document.getElementById("stepDetailContainer").innerHTML = `
    <div class="step-detail-head">
      <div class="step-main-title"><span>GIAI ĐOẠN ${esc(s.step)} / ${esc(total)}</span><h3>${esc(s.name)} (${esc(s.tag)})</h3></div>
      <span class="badge badge-neutral">Tự động theo quy tắc</span>
    </div>
    <div class="step-grid-info">
      <div class="info-box"><h4>Dữ liệu đầu vào</h4><p>${esc(s.input)}</p></div>
      <div class="info-box"><h4>Phương pháp & Mô hình áp dụng</h4><p><strong>${esc(s.model)}</strong></p></div>
      <div class="info-box span-2"><h4>Nội dung và nguyên lý thực hiện</h4><p style="white-space: pre-line;">${esc(s.process)}</p></div>
      <div class="info-box"><h4>Kết quả đầu ra</h4><p>${esc(s.output)}</p></div>
      <div class="info-box"><h4>Chỉ tiêu kiểm soát & Đánh giá</h4><p>${esc(s.evidence)}</p>
        ${metrics ? `<ul class="metric-list">${metrics}</ul>` : ""}</div>
      ${details ? `<div class="info-box span-2 info-box-accent"><h4>${esc(s.details_title || "Chi tiết")}</h4><ul class="academic-bullet-list">${details}</ul></div>` : ""}
    </div>`;
}

/* ==========================================================================
   6. Trình soi bản quét
   ========================================================================== */
function bookOptionsHtml(includeGroups) {
  let html = "";
  if (includeGroups) {
    html += `<option value="all">Tất cả 10 bộ (tự động)</option><option value="giao_nop">Nhóm: giao nộp</option>
      <option value="danh_gia">Nhóm: tập đánh giá (IHR + Borg)</option><option value="nhan_nguoi">Nhóm: Borg nhãn người</option>`;
  }
  ROLE_GROUPS.forEach((g) => {
    const arr = AppState.books.filter((b) => (b.role || "giao_nop") === g.role);
    if (!arr.length) return;
    html += `<optgroup label="${esc(g.label)}">` + arr.map((b) => {
      const empty = !(b.total_chars || b.total) ? " — chưa có dữ liệu" : (b.status === "ban_cu" ? " — bản cũ" : "");
      return `<option value="${esc(b.id)}">${esc(b.title)}${empty}</option>`;
    }).join("") + "</optgroup>";
  });
  return html;
}

function populateBookSelects() {
  const bookSelect = document.getElementById("inspectorBookSelect");
  if (bookSelect && AppState.books.length) {
    bookSelect.innerHTML = bookOptionsHtml(false);
    const pick = AppState.books.find((b) => b.id === AppState.inspector.book && (b.total_chars || b.total))
      || AppState.books.find((b) => b.id === "LucVanTien1883" && (b.total_chars || b.total))
      || AppState.books.find((b) => b.total_chars || b.total) || AppState.books[0];
    AppState.inspector.book = pick.id;
    bookSelect.value = pick.id;
    updatePageSelectOptions(true);
  }
  const gf = document.getElementById("galleryBookFilter");
  if (gf && AppState.books.length) {
    const cur = gf.value || "all";
    gf.innerHTML = bookOptionsHtml(true);
    gf.value = [...gf.options].some((o) => o.value === cur) ? cur : "all";
  }
}

function currentBook() {
  return AppState.books.find((b) => b.id === AppState.inspector.book);
}

function updatePageSelectOptions(preferDefault) {
  const pageSelect = document.getElementById("inspectorPageSelect");
  const b = currentBook();
  const pages = b?.available_pages?.length ? b.available_pages : (b?.sample_pages || []);
  pageSelect.innerHTML = pages.length
    ? pages.map((p) => `<option value="${esc(p)}">Trang ${esc(String(p).replace("page_", ""))}</option>`).join("")
    : `<option value="">(không có trang)</option>`;
  if (!preferDefault && pages.includes(AppState.inspector.page)) {
    pageSelect.value = AppState.inspector.page;
  } else if (b?.default_page && pages.includes(b.default_page)) {
    AppState.inspector.page = b.default_page;
    pageSelect.value = b.default_page;
  } else if (pages.length) {
    AppState.inspector.page = pages[0];
    pageSelect.value = pages[0];
  } else {
    AppState.inspector.page = "";
  }
  const stEl = document.getElementById("inspectorBookStatus");
  if (stEl) {
    const bits = [];
    if (b) {
      if (b.role) bits.push(roleLabel(b.role));
      if (b.status && b.status !== "ok") bits.push(b.status_vi || b.status);
      if (b.note) bits.push(b.note);
    }
    stEl.textContent = bits.join(" · ");
  }
  // Bộ nhãn người không có gold_exact -> khoá chế độ tô theo GOLD chính xác
  const human = b?.kind === "human";
  document.querySelectorAll('input[name="colorMode"]').forEach((r) => {
    if (r.value === "gold_exact") r.disabled = human;
  });
  if (human && AppState.inspector.colorMode === "gold_exact") setColorMode("tier");
}

function setColorMode(mode) {
  AppState.inspector.colorMode = mode;
  LS.set("gannhan_color", mode);
  document.querySelectorAll('input[name="colorMode"]').forEach((r) => { r.checked = r.value === mode; });
  renderBoundingBoxes(AppState.inspector.chars, true);
}

function initInspectorControls() {
  const bookSelect = document.getElementById("inspectorBookSelect");
  const pageSelect = document.getElementById("inspectorPageSelect");
  bookSelect.addEventListener("change", (e) => {
    AppState.inspector.book = e.target.value;
    updatePageSelectOptions(true);
    loadInspectorPage();
  });
  pageSelect.addEventListener("change", (e) => {
    AppState.inspector.page = e.target.value;
    loadInspectorPage();
  });
  const step = (d) => {
    const opts = [...pageSelect.options].map((o) => o.value).filter(Boolean);
    const i = opts.indexOf(AppState.inspector.page);
    const j = i + d;
    if (j >= 0 && j < opts.length) {
      AppState.inspector.page = opts[j];
      pageSelect.value = opts[j];
      loadInspectorPage();
    }
  };
  document.getElementById("btnPrevPage").addEventListener("click", () => step(-1));
  document.getElementById("btnNextPage").addEventListener("click", () => step(1));
  document.querySelectorAll('input[name="colorMode"]').forEach((r) => {
    r.checked = r.value === AppState.inspector.colorMode;
    r.addEventListener("change", (e) => { if (e.target.checked) setColorMode(e.target.value); });
  });
  document.getElementById("btnZoomIn").addEventListener("click", () => { AppState.inspector.zoom = Math.min(AppState.inspector.zoom + 0.2, 2.4); applyZoom(); });
  document.getElementById("btnZoomOut").addEventListener("click", () => { AppState.inspector.zoom = Math.max(AppState.inspector.zoom - 0.2, 0.6); applyZoom(); });
  document.getElementById("btnZoomReset").addEventListener("click", () => { AppState.inspector.zoom = 1.0; applyZoom(); });
}

function applyZoom() {
  document.getElementById("scanContainer").style.transform = `scale(${AppState.inspector.zoom})`;
}

function samplePage(book) {
  const s = AppState.sampleData;
  if (!s) return null;
  if (AppState.sampleFormat === "v2") return s.pages?.[book] || null;
  const sp = s.sample_pages?.[book];
  if (!sp) return null;
  return {
    book, page: sp.page, has_scan: false, scan_url: null, dimensions: sp.dimensions, status: "ok",
    characters: (sp.characters || []).map((c) => ({ ...c, crop_url: c.crop_url || (c.crop_rel ? `../dataset/${c.crop_rel}` : null) })),
  };
}

async function loadInspectorPage() {
  const { book, page } = AppState.inspector;
  const overlay = document.getElementById("bboxOverlay");
  const scanImg = document.getElementById("pageScanImage");
  const b = currentBook();
  setText("viewerTitle", `${b?.title || book}${page ? " — Trang " + String(page).replace("page_", "") : ""}`);
  overlay.innerHTML = "";

  let pageData = null;
  if (AppState.isLiveServer) {
    try {
      const resp = await fetch(`/api/page?book=${encodeURIComponent(book)}&page=${encodeURIComponent(page || "")}`);
      if (resp.ok) pageData = await resp.json();
    } catch (e) {
      console.warn("Lỗi khi tải trang:", e);
    }
  }
  if (!pageData) pageData = samplePage(book);
  if (AppState.inspector.book !== book || AppState.inspector.page !== page) return; // người dùng đã chuyển trang khác

  const msgEl = document.getElementById("pageStatusMsg");
  if (!pageData) {
    AppState.inspector.chars = [];
    overlay.innerHTML = "";
    scanImg.src = createPageCanvasPlaceholder(1896, 3212, "Chưa có dữ liệu trang");
    msgEl.textContent = `Chưa có dữ liệu cho ${b?.title || book}${page ? " / " + page : ""} trong chế độ này.`;
    msgEl.classList.remove("hidden");
    ["pgTotalChars", "pgGoldChars", "pgSylChars", "pgGxOk"].forEach((id) => setText(id, "0"));
    renderBoundingBoxes([], false);
    return;
  }
  AppState.inspector.pageData = pageData;
  AppState.inspector.chars = pageData.characters || [];
  const chars = AppState.inspector.chars;
  const msg = pageData.message || "";
  msgEl.textContent = msg;
  msgEl.classList.toggle("hidden", !msg);

  setText("pgTotalChars", chars.length);
  setText("pgGoldChars", chars.filter((c) => c.tier === "GOLD").length);
  setText("pgSylChars", chars.filter((c) => c.tier === "SYLLABLE").length);
  setText("pgGxOk", chars.filter((c) => c.gold_exact === "ok").length);

  const W = pageData.dimensions?.width || 1896;
  const H = pageData.dimensions?.height || 3212;
  scanImg.onerror = () => {
    scanImg.onerror = null;
    scanImg.src = createPageCanvasPlaceholder(W, H, "Không tải được ảnh trang");
  };
  scanImg.src = pageData.scan_url || createPageCanvasPlaceholder(W, H, pageData.has_scan === false ? "Không có ảnh trang (chỉ có hộp chữ)" : "Bản quét trang sách cổ");
  renderBoundingBoxes(chars, false);
}

function createPageCanvasPlaceholder(w, h, text) {
  const canvas = document.createElement("canvas");
  canvas.width = 600;
  canvas.height = Math.max(200, Math.round((h / w) * 600));
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#1e1b18";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#443d35";
  ctx.lineWidth = 4;
  ctx.strokeRect(30, 30, canvas.width - 60, canvas.height - 60);
  ctx.fillStyle = "#8c8273";
  ctx.font = "16px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text || "Bản quét trang sách cổ", canvas.width / 2, canvas.height / 2);
  return canvas.toDataURL();
}

function categoryOf(c, mode) {
  if (mode === "gold_exact") return c.gold_exact || "khong_ap_dung";
  return TIER_CATS[c.tier] ? c.tier : "OTHER";
}
function catDef(cat, mode) {
  return (mode === "gold_exact" ? GX_CATS : TIER_CATS)[cat] || { label: cat, cls: "cat-other" };
}

function renderBoundingBoxes(chars, keepSelection) {
  const overlay = document.getElementById("bboxOverlay");
  const mode = AppState.inspector.colorMode;
  overlay.innerHTML = "";
  chars.forEach((c) => {
    const cat = categoryOf(c, mode);
    const box = document.createElement("div");
    box.className = `char-bbox bbox-rect ${catDef(cat, mode).cls}`;
    box.id = `bbox-${c.index}`;
    box.setAttribute("data-cat", cat);
    box.setAttribute("data-tier", c.tier || "");
    box.style.left = `${c.rect.left_pct}%`;
    box.style.top = `${c.rect.top_pct}%`;
    box.style.width = `${c.rect.width_pct}%`;
    box.style.height = `${c.rect.height_pct}%`;
    const tip = document.createElement("div");
    tip.className = "bbox-tooltip";
    tip.textContent = `${c.label || c.ocr_char || "字"} · ${c.syllable || ""}${c.gold_exact ? " · " + (GX_SHORT[c.gold_exact] || c.gold_exact) : ""}`;
    box.appendChild(tip);
    box.addEventListener("mouseenter", () => selectCharacter(c, box));
    box.addEventListener("click", () => selectCharacter(c, box));
    overlay.appendChild(box);
  });
  renderLegend(chars);
  filterBboxes();
  const sel = AppState.inspector.selectedChar;
  const again = keepSelection && sel ? chars.find((c) => c.index === sel.index && c.cell_uid === sel.cell_uid) : null;
  if (again) selectCharacter(again, document.getElementById(`bbox-${again.index}`));
  else if (chars.length) selectCharacter(chars[0], document.getElementById(`bbox-${chars[0].index}`));
  else {
    AppState.inspector.selectedChar = null;
    document.getElementById("previewEmpty").classList.remove("hidden");
    document.getElementById("previewActive").classList.add("hidden");
  }
}

function renderLegend(chars) {
  const el = document.getElementById("legendFilters");
  const mode = AppState.inspector.colorMode;
  const counts = {};
  chars.forEach((c) => { const k = categoryOf(c, mode); counts[k] = (counts[k] || 0) + 1; });
  const order = Object.keys(mode === "gold_exact" ? GX_CATS : TIER_CATS);
  const cats = order.filter((k) => counts[k]);
  const hidden = AppState.inspector.hidden[mode];
  if (!cats.length) {
    el.innerHTML = `<div class="legend-empty">Trang không có ô chữ.</div>`;
    return;
  }
  el.innerHTML = cats.map((k) => {
    const d = catDef(k, mode);
    return `<label class="cb-label legend-item"><input type="checkbox" data-cat="${esc(k)}" ${hidden.has(k) ? "" : "checked"}>
      <span class="cb-box legend-swatch ${d.cls}"></span> <span class="legend-text">${esc(d.label)}</span> <span class="legend-count">${fmtInt(counts[k])}</span></label>`;
  }).join("");
  el.querySelectorAll("input[type=checkbox]").forEach((cb) => {
    cb.addEventListener("change", (e) => {
      const k = e.target.getAttribute("data-cat");
      if (e.target.checked) hidden.delete(k); else hidden.add(k);
      filterBboxes();
    });
  });
}

function filterBboxes() {
  const hidden = AppState.inspector.hidden[AppState.inspector.colorMode];
  document.querySelectorAll("#bboxOverlay .char-bbox").forEach((el) => {
    el.classList.toggle("hidden", hidden.has(el.getAttribute("data-cat")));
  });
}

function setImg(imgEl, url, figEl) {
  if (url) {
    imgEl.style.display = "";
    imgEl.onerror = () => { imgEl.style.display = "none"; };
    imgEl.src = url;
    if (figEl) figEl.classList.remove("hidden");
  } else {
    imgEl.removeAttribute("src");
    imgEl.style.display = "none";
    if (figEl) figEl.classList.add("hidden");
  }
}

function selectCharacter(c, boxEl) {
  AppState.inspector.selectedChar = c;
  document.querySelectorAll("#bboxOverlay .char-bbox").forEach((b) => b.classList.remove("selected"));
  if (boxEl) boxEl.classList.add("selected");
  document.getElementById("previewEmpty").classList.add("hidden");
  document.getElementById("previewActive").classList.remove("hidden");

  setText("previewNomChar", c.label || c.ocr_char || "字");
  setText("previewSylText", c.syllable ? `/${c.syllable}/` : "(âm dị bản)");
  setText("previewUnicode", c.unicode || "—");
  setText("previewOcrChar", c.ocr_char || "—");
  setText("previewColOrder", `Cột ${c.column ?? "?"} · Ô thứ ${c.index ?? "?"}`);
  setText("previewRule", c.rule || "—");
  setText("previewBbox", JSON.stringify((c.bbox || []).map((v) => Math.round(v))));
  setText("previewUid", c.cell_uid || "—");

  const tierBadge = document.getElementById("previewTierBadge");
  tierBadge.textContent = c.tier || "—";
  tierBadge.className = `preview-tier-badge ${catDef(TIER_CATS[c.tier] ? c.tier : "OTHER", "tier").cls}`;

  setImg(document.getElementById("previewCropImg"), c.crop_url || (c.crop_rel ? `../dataset/${c.crop_rel}` : null), null);
  setImg(document.getElementById("previewCropChuanImg"), c.crop_chuan_url, document.getElementById("previewChuanFig"));

  const gxBadge = document.getElementById("previewGxBadge");
  const gxBox = document.getElementById("previewGxBox");
  if (c.keep_level !== undefined) {
    gxBadge.className = "gx-pill hidden";
    gxBox.classList.remove("hidden");
    setText("previewGxStatus", `Nhãn người · ${c.keep_vi || c.keep_level || "—"}`);
    setText("previewGxReason", `Độ tin căn hộp: ${c.align_conf || "—"} · one_char_ok: ${c.one_char_ok || "—"} · split: ${c.split_hint || "—"}${c.folio ? " · tờ " + c.folio : ""}`);
    document.getElementById("previewEvidence").outerHTML = `<span id="previewEvidence">${evBadge("do_tren_nhan_nguoi", "Nhãn do người phiên")}</span>`;
  } else if (c.gold_exact) {
    gxBadge.className = `gx-pill gx-pill-${c.gold_exact}`;
    gxBadge.textContent = GX_SHORT[c.gold_exact] || c.gold_exact;
    gxBox.classList.remove("hidden");
    setText("previewGxStatus", c.gold_exact_vi || c.gold_exact);
    setText("previewGxReason", c.ge_reason_vi || "—");
    document.getElementById("previewEvidence").outerHTML = `<span id="previewEvidence">${c.evidence_level ? evBadge(c.evidence_level, c.evidence_vi || c.evidence_level) : "—"}</span>`;
  } else {
    gxBadge.className = "gx-pill hidden";
    gxBox.classList.add("hidden");
  }
}

/* ==========================================================================
   7. Tra cứu ký tự
   ========================================================================== */
function initGalleryControls() {
  const searchInput = document.getElementById("gallerySearchInput");
  let debounceTimer = null;
  searchInput.addEventListener("input", (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { AppState.gallery.query = e.target.value.trim(); executeSearch(); }, 250);
  });
  document.getElementById("galleryBookFilter").addEventListener("change", (e) => { AppState.gallery.book = e.target.value; executeSearch(); });
  document.getElementById("galleryTierFilter").addEventListener("change", (e) => { AppState.gallery.tier = e.target.value; executeSearch(); });
  document.getElementById("galleryGxFilter").addEventListener("change", (e) => { AppState.gallery.gx = e.target.value; executeSearch(); });
}

const GROUP_ROLES = { giao_nop: ["giao_nop"], danh_gia: ["danh_gia_ihr", "danh_gia_borg"], nhan_nguoi: ["nhan_nguoi"] };

async function executeSearch() {
  const { query, book, tier, gx } = AppState.gallery;
  const gridEl = document.getElementById("galleryCardsGrid");
  const countEl = document.getElementById("galleryCountText");
  gridEl.innerHTML = `<div class="skeleton-loader" style="grid-column: 1 / -1;">Đang tra cứu cơ sở dữ liệu...</div>`;
  let results = [];
  let total = null;
  if (AppState.isLiveServer) {
    try {
      const url = `/api/search?q=${encodeURIComponent(query)}&book=${encodeURIComponent(book)}&tier=${encodeURIComponent(tier)}&gx=${encodeURIComponent(gx)}&limit=96`;
      const resp = await fetch(url);
      if (resp.ok) {
        const data = await resp.json();
        results = data.results || [];
        total = data.total_matches;
      }
    } catch (e) {
      console.warn("Lỗi khi tìm kiếm qua API:", e);
    }
  } else if (AppState.sampleData) {
    const roleOf = {};
    AppState.books.forEach((b) => { roleOf[b.id] = b.role || "giao_nop"; });
    const q = query.toLowerCase();
    const qn = stripAccents(q);
    results = (AppState.gallery.results || []).filter((item) => {
      if (book === "all") { if (roleOf[item.book] === "nhan_nguoi") return false; }
      else if (GROUP_ROLES[book]) { if (!GROUP_ROLES[book].includes(roleOf[item.book] || "giao_nop")) return false; }
      else if (item.book !== book) return false;
      if (tier !== "all" && item.tier !== tier) return false;
      if (gx !== "all" && (item.gold_exact || "") !== gx) return false;
      if (q) {
        const syl = (item.syllable || "").toLowerCase();
        const ok = syl.includes(q) || stripAccents(syl).includes(qn) || (item.label || "").includes(query) || (item.unicode || "").toLowerCase().includes(q);
        if (!ok) return false;
      }
      return true;
    });
    total = results.length;
  }
  AppState.gallery.loaded = true;
  countEl.textContent = total !== null && total > results.length
    ? `Hiển thị ${fmtInt(results.length)} / ${fmtInt(total)} mẫu phù hợp (rải đều giữa các bộ):`
    : `Tìm thấy ${fmtInt(results.length)} mẫu ký tự phù hợp${AppState.isLiveServer ? "" : " trong dữ liệu mẫu"}:`;
  gridEl.innerHTML = "";
  if (!results.length) {
    gridEl.innerHTML = `<div class="gallery-empty">Không tìm thấy mẫu ký tự nào${query ? ` khớp với "<strong>${esc(query)}</strong>"` : ""} với bộ lọc hiện tại.</div>`;
    return;
  }
  results.forEach((item) => {
    const card = document.createElement("div");
    card.className = "char-card";
    const tierCls = catDef(TIER_CATS[item.tier] ? item.tier : "OTHER", "tier").cls;
    const img = item.crop_chuan_url || item.crop_url || (item.crop_rel ? `../dataset/${item.crop_rel}` : "");
    const imgKind = item.crop_chuan_url ? "crop chuẩn" : (img ? "crop gốc" : "");
    card.innerHTML = `
      <div class="char-card-top">
        <span class="char-tier-pill ${tierCls}">${esc(item.tier)}</span>
        ${item.gold_exact ? `<span class="gx-pill gx-pill-${esc(item.gold_exact)}" title="${esc(item.ge_reason_vi || "")}">${esc(GX_SHORT[item.gold_exact] || item.gold_exact)}</span>` : ""}
      </div>
      <div class="char-book-tag" title="${esc(item.book_title || item.book)}">${esc(item.book_title || item.book)}</div>
      <div class="char-card-visual">
        ${img ? `<img class="char-crop-img" src="${esc(img)}" alt="crop" loading="lazy" onerror="this.style.display='none'">` : ""}
        <div class="char-nom-display">${esc(item.label || item.ocr_char || "字")}</div>
      </div>
      <div class="char-card-info">
        <div class="char-syl-main">${esc(item.syllable || "(âm dị thể)")}</div>
        <div class="char-uni-code">${esc(item.unicode || "—")}${imgKind ? ` · <span class="img-kind">${imgKind}</span>` : ""}</div>
        <div class="char-page-loc">${esc(item.page || "")} · Cột ${esc(item.column ?? "")}</div>
      </div>`;
    card.addEventListener("click", () => {
      AppState.inspector.book = item.book;
      AppState.inspector.page = item.page || "";
      const bookSel = document.getElementById("inspectorBookSelect");
      if (bookSel) bookSel.value = item.book;
      updatePageSelectOptions(false);
      const pageSel = document.getElementById("inspectorPageSelect");
      if (pageSel && item.page && [...pageSel.options].some((o) => o.value === item.page)) {
        pageSel.value = item.page;
        AppState.inspector.page = item.page;
      }
      if (item.gold_exact && AppState.inspector.colorMode !== "gold_exact" && currentBook()?.kind !== "human") setColorMode("gold_exact");
      switchTab("inspector");
    });
    gridEl.appendChild(card);
  });
}

function stripAccents(s) {
  return String(s || "").replace(/đ/g, "d").replace(/Đ/g, "D").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/* ==========================================================================
   8. Kết quả & đánh giá
   ========================================================================== */
function accRowHtml(r, lastCol) {
  const val = r.available ? `<strong>${esc(r.value_text)}</strong>` : `<span class="td-dim">${esc(r.value_text || NA)}</span>`;
  return `<tr>
    <td class="metric-name">${esc(r.book_title || r.book)}</td>
    <td>${evBadge(r.level, r.level_vi)}</td>
    <td>${esc(r.metric)}${r.note && lastCol !== "note" ? `<div class="td-sub">${esc(r.note)}</div>` : ""}</td>
    <td class="td-num">${val}</td>
    <td class="td-num td-dim">${esc(r.ci_text || "")}</td>
    <td class="td-num">${esc(r.n_text || "")}</td>
    <td class="td-src">${esc(lastCol === "note" ? (r.note || "") : (r.source || ""))}</td></tr>`;
}

function renderBenchmarks() {
  const bm = AppState.benchmarks;
  const accBody = document.getElementById("accTableBody");
  const gxBody = document.getElementById("gxTableBody");
  const kimBody = document.getElementById("kimTableBody");
  const cmpBody = document.getElementById("benchmarkTableBody");
  if (!bm) {
    const msg = `<tr><td colspan="9" class="td-dim">Chưa có số đo trong chế độ này — chạy máy chủ (.venv/bin/python web/server.py) hoặc sinh lại sample_data.json (.venv/bin/python web/build_sample_data.py).</td></tr>`;
    [accBody, gxBody, kimBody, cmpBody].forEach((el) => { if (el) el.innerHTML = msg; });
    document.getElementById("invCardBody").innerHTML = `<p class="table-note">${NA}</p>`;
    document.getElementById("borgHumanBody").innerHTML = `<p class="table-note">${NA}</p>`;
    renderDatasetTree();
    return;
  }
  accBody.innerHTML = (bm.accuracy || []).map((r) => accRowHtml(r)).join("") || `<tr><td colspan="7" class="td-dim">${NA}</td></tr>`;
  setText("accNote", bm.accuracy_note || "");

  const gt = bm.gold_exact_table || {};
  setText("gxTableBadge", gt.policy_version ? `chính sách ${gt.policy_version}` : (gt.available ? "gold_exact.csv" : NA));
  gxBody.innerHTML = (gt.rows || []).map((r) => `<tr>
      <td class="metric-name">${esc(r.book_title)}</td><td>${esc(r.role_vi)}</td>
      <td class="td-num">${fmtInt(r.gold)}</td><td class="td-num gx-num-ok">${fmtInt(r.ok)}</td><td class="td-num">${fmtInt(r.text_only)}</td>
      <td class="td-num">${fmtInt(r.uncertified)}</td><td class="td-num">${fmtInt(r.review)}</td>
      <td class="td-num">${r.ok_pct === null || r.ok_pct === undefined ? NA : fmtPct1(r.ok_pct * 100)}</td>
      <td>${evBadge(r.evidence, r.evidence_vi)}</td></tr>`).join("")
    + (gt.available ? `<tr class="tr-total"><td class="metric-name">Tổng</td><td></td><td class="td-num">${fmtInt(gt.totals?.gold)}</td>
      <td class="td-num gx-num-ok">${fmtInt(gt.totals?.ok)}</td><td class="td-num">${fmtInt(gt.totals?.text_only)}</td>
      <td class="td-num">${fmtInt(gt.totals?.uncertified)}</td><td class="td-num">${fmtInt(gt.totals?.review)}</td>
      <td class="td-num">${gt.totals?.ok_pct === null || gt.totals?.ok_pct === undefined ? NA : fmtPct1(gt.totals.ok_pct * 100)}</td><td></td></tr>` : "");
  const prof = gt.profile_handwriting;
  setText("gxTableNote", [
    gt.source ? `Nguồn: ${gt.source}${gt.config_sha16 ? " · config " + gt.config_sha16 : ""}.` : "Chưa có gold_exact.csv.",
    prof ? `Profile chữ viết tay áp cho ${(prof.sets || []).join(", ")}: bỏ cổng ${(prof.drop || []).join(", ") || "—"}, cổng khe ${prof.slot ?? "—"}, q = ${prof.hand_q ?? "—"}, ≥ ${prof.n_hum_min ?? "—"} mẫu chữ người.` : "",
    "Số ô ok phụ thuộc ngưỡng đã đăng ký trước; chỉ ở bộ có nhãn người mới ĐO được độ chính xác.",
  ].filter(Boolean).join(" "));

  cmpBody.innerHTML = (bm.comparisons || []).map((c) => `<tr>
      <td class="metric-name"><strong>${esc(c.metric)}</strong></td>
      <td class="td-dim">${esc(c.baseline)}</td>
      <td class="td-proposed">${esc(c.proposed)}</td>
      <td>${c.improvement && c.improvement !== "—" ? `<span class="improvement-badge">${esc(c.improvement)}</span>` : "—"}</td>
      <td class="td-sub-cell">${esc(c.impact)}${c.level_vi ? `<div class="td-sub">${esc(c.level_vi)}${c.source ? " · " + esc(c.source) : ""}</div>` : ""}</td></tr>`).join("");

  kimBody.innerHTML = (bm.kim || []).map((r) => accRowHtml(r, "note")).join("") || `<tr><td colspan="7" class="td-dim">${NA} (cần measure_out/&lt;IHR|Borg&gt;/…/summary.json)</td></tr>`;

  // Bất biến
  const inv = bm.invariants || {};
  setText("invBadge", inv.available ? inv.text : NA);
  const evRows = (inv.evals || []).map((e) => `<tr><td>${esc(e.name)}</td><td class="td-num">${e.available ? `${fmtInt(e.n_pass)}/${fmtInt(e.total)}` : NA}</td>
      <td class="td-num ${e.n_fail ? "td-bad" : ""}">${e.available ? fmtInt(e.n_fail) : ""}</td></tr>`).join("");
  const t = inv.totals || {};
  document.getElementById("invCardBody").innerHTML = `
    ${inv.available ? `<p class="table-note">measure_out/SUMMARY.json (${esc(inv.generated_at || "")}): <strong>${fmtInt(t.inv_pass)} PASS</strong> · ${fmtInt(t.inv_fail)} FAIL · ${fmtInt(t.inv_soft_fail)} FAIL mềm · ${fmtInt(t.inv_skip)} SKIP trên ${fmtInt(t.steps)} bước.</p>`
      : `<p class="table-note">measure_out/SUMMARY.json: ${NA} — chạy <code>.venv/bin/python scripts/measure/measure.py --all</code>.</p>`}
    <div class="table-wrap"><table class="benchmark-table compact"><thead><tr><th>Phép đo</th><th>PASS</th><th>FAIL</th></tr></thead><tbody>${evRows}</tbody></table></div>`;

  // Borg nhãn người
  const bh = bm.borg_human || {};
  const per = bh.per_book || {};
  const lines = Object.entries(per).map(([k, v]) => `<li><strong>${esc(k)}</strong>: ${fmtInt(v.cells)} chữ người · ${fmtInt(v.pages)} trang · keep_v5 ${fmtInt(v.levels?.keep_v5 || 0)} · keep ${fmtInt(v.levels?.keep || 0)} · keep_high ${fmtInt(v.levels?.keep_high || 0)}</li>`).join("");
  const cum = bh.cumulative || {};
  document.getElementById("borgHumanBody").innerHTML = bh.available ? `
    <p class="table-note">Hộp + chữ do <strong>người</strong> phiên (Excel), tách khỏi GOLD tự động; dùng để đo và hiệu chuẩn bộ kiểm chữ viết tay.</p>
    <ul class="academic-checklist mt-8">${lines || "<li>Chưa nạp được labels.csv của bộ nhãn người.</li>"}
      ${bh.n_cells !== undefined ? `<li>Tổng ${fmtInt(bh.n_cells)} ô · luỹ kế keep_v5 ${fmtInt(cum.keep_v5)} ⊂ keep ${fmtInt(cum.keep)} ⊂ keep_high ${fmtInt(cum.keep_high)} · có ảnh ${fmtInt(bh.with_image)} · một chữ ${fmtInt(bh.one_char_ok)}</li>` : ""}
      ${bh.theta ? `<li>${esc(bh.theta.meaning)}: <strong>${esc(bh.theta.text)}</strong> ${esc(bh.theta.ci_text || "")} (n ${fmtInt(bh.theta.n)}) ${evBadge("uoc_luong", bh.theta.level_vi)}</li>` : ""}
    </ul>` : `<p class="table-note">${NA} — dựng bằng <code>.venv/bin/python -m pipeline.borg_human --stage all</code>.</p>`;

  const miss = bm.missing_sources || [];
  const card = document.getElementById("missingSourcesCard");
  card.classList.toggle("hidden", !miss.length);
  document.getElementById("missingSourcesList").innerHTML = miss.map((m) => `<li><code>${esc(m)}</code></li>`).join("");
  renderDatasetTree();
}

function renderDatasetTree() {
  const el = document.getElementById("datasetTree");
  const st = AppState.stats;
  if (!el || !st) return;
  const books = st.books || {};
  const lines = ["dataset/"];
  const entries = [];
  const pad = (s) => (s + " ".repeat(Math.max(1, 24 - s.length)));
  const stt = Object.values(books).filter((b) => b.book_set === "SachThanhTruyen");
  const sttNew = stt.filter((b) => b.status === "ok");
  if (sttNew.length) {
    const tot = sttNew.reduce((a, b) => a + (b.total || 0), 0);
    entries.push([`SachThanhTruyen/`, `# ${fmtInt(tot)} nhãn (${sttNew.map((b) => b.set8 + " " + fmtInt(b.total)).join(", ")})`]);
  }
  stt.filter((b) => b.status === "ban_cu").forEach((b) => entries.push([`${b.id}/`, `# bản cũ 23/09: ${fmtInt(b.total)} nhãn`]));
  Object.values(books).filter((b) => b.book_set !== "SachThanhTruyen").forEach((b) => {
    const tag = b.role === "giao_nop" ? "" : " · tập đánh giá";
    entries.push([`${b.book_set || b.id}/`, b.total ? `# ${fmtInt(b.total)} nhãn${tag}` : `# ${b.status_vi || NA}`]);
  });
  const all = st.all_dataset;
  const gx = st.gold_exact || {};
  entries.push(["_ALL/", all ? `# bộ gộp ${fmtInt(all.n_dong)} dòng · ${fmtInt(all.eval_only_dong)} evaluation_only` : `# bộ gộp: ${NA}`]);
  entries.push(["  gold_exact.csv", gx.available ? `# ${fmtInt(gx.totals?.gold)} ô GOLD · ok ${fmtInt(gx.totals?.ok)}` : `# ${NA}`]);
  entries.push(["  crops_chuan/", gx.available ? "# crop chuẩn của ô ok" : `# ${NA}`]);
  const human = Object.values(st.human_books || {});
  const hTot = human.reduce((a, b) => a + (b.total || 0), 0);
  if (human.length) entries.push(["_BORG_NHAN_NGUOI/", hTot ? `# ${fmtInt(hTot)} chữ người (Borg.18 + Borg.34)` : `# ${NA}`]);
  entries.forEach(([a, b], i) => {
    const last = i === entries.length - 1;
    const pre = a.startsWith("  ") ? "│   " + (a.trim() === "crops_chuan/" ? "└── " : "├── ") : (last ? "└── " : "├── ");
    lines.push(pre + pad(a.trim()) + b);
  });
  el.textContent = lines.join("\n");
}
