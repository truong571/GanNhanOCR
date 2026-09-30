/**
 * web/app.js — GanNhanOCR: giao diện trình diễn luận văn Thạc sĩ (chỉ trình bày TẦNG NHÃN).
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
  inspector: {
    book: "LucVanTien1883",
    page: "page_0002",
    zoom: 1.0,
    chars: [],
    selectedChar: null,
    hidden: new Set(),
  },
  gallery: { query: "", book: "all", tier: "all", results: [], loaded: false },
};

const NA = "chưa có";

// Màu + nhãn theo tầng
const TIER_CATS = {
  GOLD: { label: "Mức 1 · GOLD", cls: "cat-gold" },
  SYLLABLE: { label: "Mức 2 · SYLLABLE", cls: "cat-syl" },
  GOLD_text_only: { label: "Mức 3 · GOLD_text_only", cls: "cat-txt" },
  REVIEW: { label: "REVIEW", cls: "cat-reviewtier" },
  QUARANTINE: { label: "QUARANTINE", cls: "cat-quarantine" },
  OTHER: { label: "Khác", cls: "cat-other" },
};
const ROLE_GROUPS = [
  { role: "giao_nop", label: "Giao nộp" },
  { role: "danh_gia_ihr", label: "Đánh giá — IHR-NomDB" },
  { role: "danh_gia_borg", label: "Đánh giá — Borg, chép tay" },
];
const GROUP_ROLES = { giao_nop: ["giao_nop"], danh_gia: ["danh_gia_ihr", "danh_gia_borg"] };

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
function tierKey(t) {
  return TIER_CATS[t] && t !== "OTHER" ? t : "OTHER";
}
function tierDef(t) {
  return TIER_CATS[tierKey(t)];
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
    AppState.stats = sample.stats || null;
    AppState.books = sample.books || [];
    AppState.pipelineFlow = sample.pipeline_flow || [];
    AppState.benchmarks = sample.benchmarks || null;
    AppState.gallery.results = sample.gallery || [];
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
  const tot = im.total_characters || 0;
  const share = (n) => (tot && n !== undefined && n !== null ? `${fmtPct1((n / tot) * 100)} tổng số ô` : "");
  setText("hdrTotalChars", fmtInt(im.total_characters));
  setText("hdrBooksCount", im.books_count ?? NA);
  setText("kpiTotalChars", fmtInt(im.total_characters));
  if (im.deliver_characters !== undefined) {
    setText("kpiTotalDesc", `Giao nộp ${fmtInt(im.deliver_characters)} · tập đánh giá ${fmtInt(im.eval_characters)}`);
  }
  setText("kpiGold", fmtInt(im.total_gold));
  setText("kpiGoldDesc", share(im.total_gold));
  setText("kpiSyl", fmtInt(im.total_syllable));
  setText("kpiSylDesc", share(im.total_syllable));
  setText("kpiReview", fmtInt(im.total_review));
  setText("kpiReviewDesc", share(im.total_review));
  setText("kpiQuarantine", fmtInt(im.total_quarantine));
  setText("kpiQuarantineDesc", share(im.total_quarantine));
  if (im.books_count !== undefined) {
    setText("kpiBooksCount", `${im.books_count} bộ`);
    const nWith = im.books_with_data ?? im.books_count;
    setText("kpiBooksDesc", nWith < im.books_count ? `${nWith}/${im.books_count} bộ đã có dữ liệu (còn lại: đang dựng lại)` : "Giao nộp + tập đánh giá");
  }
  const invEl = document.getElementById("summaryInvariants");
  if (invEl) {
    invEl.textContent = im.invariants_available && im.invariants_text ? im.invariants_text : "";
    invEl.classList.toggle("hidden", !(im.invariants_available && im.invariants_text));
  }
  setText("footerLoadedAt", st.loaded_at || (AppState.sampleData?.generated_at ? `${AppState.sampleData.generated_at} (dữ liệu mẫu)` : "—"));
  setText("footerRoot", st.data_root ? `Gốc dữ liệu: ${st.data_root}` : "");
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

  // Thẻ quy ước tầng nhãn: số ô đọc từ dữ liệu; GOLD_text_only chỉ hiện khi có ô
  const tot = im.total_characters || 0;
  const cnt = (n) => (n === undefined || n === null ? NA : `${fmtInt(n)} nhãn${tot ? ` (${fmtPct1((n / tot) * 100)})` : ""}`);
  setText("tierCountGold", cnt(im.total_gold));
  setText("tierCountSyl", cnt(im.total_syllable));
  setText("tierCountTxt", cnt(im.total_gold_text_only));
  setText("tierCountReview", cnt(im.total_review));
  setText("tierCountQuarantine", cnt(im.total_quarantine));
  const txtCard = document.getElementById("tierCardTxt");
  if (txtCard) txtCard.classList.toggle("hidden", !im.total_gold_text_only);
  setText("tierCardsBadge", `${(im.tier_columns || []).length || 4} tầng nhãn`);

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
}

function bookStatItem(b) {
  const item = document.createElement("div");
  item.className = "book-stat-item";
  const total = b.total ?? b.total_chars ?? 0;
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
    <div class="book-stat-sub">${esc(b.layout || b.subtitle || "")} · ${fmtInt(b.gold)} GOLD (${fmtPct1(b.gold_pct)}) · ${fmtInt(b.syllable)} SYLLABLE</div>
    <div class="progress-bar-wrap" title="Tỉ lệ GOLD">
      <div class="progress-bar-fill" style="width: ${Math.max(0, Math.min(100, Number(b.gold_pct) || 0))}%"></div>
    </div>`;
  return item;
}

/* ==========================================================================
   5. Quy trình xử lý
   ========================================================================== */
let currentFlowStep = 1;

function renderPipelineFlow() {
  const steps = AppState.pipelineFlow || [];
  const navEl = document.getElementById("stepperNav");
  if (!steps.length) {
    setText("flowStepCount", "—");
    navEl.innerHTML = "";
    document.getElementById("stepDetailContainer").innerHTML = `<p class="table-note">Chưa có mô tả quy trình trong chế độ này.</p>`;
    return;
  }
  setText("flowStepCount", steps.length);
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
      ${m.source ? `<span class="metric-src">${esc(m.source)}</span>` : ""}</li>`).join("");
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
      <div class="info-box"><h4>Chỉ tiêu kiểm soát & Đánh giá</h4><p>${esc(s.control)}</p>
        ${metrics ? `<ul class="metric-list">${metrics}</ul>` : ""}</div>
    </div>`;
}

/* ==========================================================================
   6. Trình soi bản quét
   ========================================================================== */
function bookOptionsHtml(includeGroups) {
  let html = "";
  if (includeGroups) {
    html += `<option value="all">Tất cả ${AppState.books.length || 10} bộ</option><option value="giao_nop">Nhóm: giao nộp</option>
      <option value="danh_gia">Nhóm: tập đánh giá (IHR + Borg)</option>`;
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
  document.getElementById("btnZoomIn").addEventListener("click", () => { AppState.inspector.zoom = Math.min(AppState.inspector.zoom + 0.2, 2.4); applyZoom(); });
  document.getElementById("btnZoomOut").addEventListener("click", () => { AppState.inspector.zoom = Math.max(AppState.inspector.zoom - 0.2, 0.6); applyZoom(); });
  document.getElementById("btnZoomReset").addEventListener("click", () => { AppState.inspector.zoom = 1.0; applyZoom(); });
}

function applyZoom() {
  document.getElementById("scanContainer").style.transform = `scale(${AppState.inspector.zoom})`;
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
  if (!pageData) pageData = AppState.sampleData?.pages?.[book] || null;
  if (AppState.inspector.book !== book || AppState.inspector.page !== page) return; // người dùng đã chuyển trang khác

  const msgEl = document.getElementById("pageStatusMsg");
  if (!pageData) {
    AppState.inspector.chars = [];
    scanImg.src = createPageCanvasPlaceholder(1896, 3212, "Chưa có dữ liệu trang");
    msgEl.textContent = `Chưa có dữ liệu cho ${b?.title || book}${page ? " / " + page : ""} trong chế độ này.`;
    msgEl.classList.remove("hidden");
    ["pgTotalChars", "pgGoldChars", "pgSylChars"].forEach((id) => setText(id, "0"));
    renderBoundingBoxes([], false);
    return;
  }
  AppState.inspector.chars = pageData.characters || [];
  const chars = AppState.inspector.chars;
  const msg = pageData.message || "";
  msgEl.textContent = msg;
  msgEl.classList.toggle("hidden", !msg);

  setText("pgTotalChars", chars.length);
  setText("pgGoldChars", chars.filter((c) => c.tier === "GOLD").length);
  setText("pgSylChars", chars.filter((c) => c.tier === "SYLLABLE").length);

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

function renderBoundingBoxes(chars, keepSelection) {
  const overlay = document.getElementById("bboxOverlay");
  overlay.innerHTML = "";
  chars.forEach((c) => {
    const cat = tierKey(c.tier);
    const box = document.createElement("div");
    box.className = `char-bbox bbox-rect ${TIER_CATS[cat].cls}`;
    box.id = `bbox-${c.index}`;
    box.setAttribute("data-cat", cat);
    box.style.left = `${c.rect.left_pct}%`;
    box.style.top = `${c.rect.top_pct}%`;
    box.style.width = `${c.rect.width_pct}%`;
    box.style.height = `${c.rect.height_pct}%`;
    const tip = document.createElement("div");
    tip.className = "bbox-tooltip";
    tip.textContent = `${c.label || c.ocr_char || "字"} · ${c.syllable || ""} · ${c.tier || ""}`;
    box.appendChild(tip);
    box.addEventListener("mouseenter", () => selectCharacter(c, box));
    box.addEventListener("click", () => selectCharacter(c, box));
    overlay.appendChild(box);
  });
  renderLegend(chars);
  filterBboxes();
  const sel = AppState.inspector.selectedChar;
  const again = keepSelection && sel ? chars.find((c) => c.index === sel.index && c.image_rel === sel.image_rel) : null;
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
  const counts = {};
  chars.forEach((c) => { const k = tierKey(c.tier); counts[k] = (counts[k] || 0) + 1; });
  const cats = Object.keys(TIER_CATS).filter((k) => counts[k]);
  const hidden = AppState.inspector.hidden;
  if (!cats.length) {
    el.innerHTML = `<div class="legend-empty">Trang không có ô chữ.</div>`;
    return;
  }
  el.innerHTML = cats.map((k) => {
    const d = TIER_CATS[k];
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
  const hidden = AppState.inspector.hidden;
  document.querySelectorAll("#bboxOverlay .char-bbox").forEach((el) => {
    el.classList.toggle("hidden", hidden.has(el.getAttribute("data-cat")));
  });
}

function setImg(imgEl, url) {
  if (url) {
    imgEl.style.display = "";
    imgEl.onerror = () => { imgEl.style.display = "none"; };
    imgEl.src = url;
  } else {
    imgEl.removeAttribute("src");
    imgEl.style.display = "none";
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

  const tierBadge = document.getElementById("previewTierBadge");
  tierBadge.textContent = c.tier || "—";
  tierBadge.className = `preview-tier-badge ${tierDef(c.tier).cls}`;
  setImg(document.getElementById("previewCropImg"), c.crop_url);
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
}

async function executeSearch() {
  const { query, book, tier } = AppState.gallery;
  const gridEl = document.getElementById("galleryCardsGrid");
  const countEl = document.getElementById("galleryCountText");
  gridEl.innerHTML = `<div class="skeleton-loader" style="grid-column: 1 / -1;">Đang tra cứu cơ sở dữ liệu...</div>`;
  let results = [];
  let total = null;
  if (AppState.isLiveServer) {
    try {
      const url = `/api/search?q=${encodeURIComponent(query)}&book=${encodeURIComponent(book)}&tier=${encodeURIComponent(tier)}&limit=96`;
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
      if (GROUP_ROLES[book]) { if (!GROUP_ROLES[book].includes(roleOf[item.book] || "giao_nop")) return false; }
      else if (book !== "all" && item.book !== book) return false;
      if (tier !== "all" && item.tier !== tier) return false;
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
    const img = item.crop_url || "";
    card.innerHTML = `
      <div class="char-card-top">
        <span class="char-tier-pill ${tierDef(item.tier).cls}">${esc(item.tier)}</span>
      </div>
      <div class="char-book-tag" title="${esc(item.book_title || item.book)}">${esc(item.book_title || item.book)}</div>
      <div class="char-card-visual">
        ${img ? `<img class="char-crop-img" src="${esc(img)}" alt="crop" loading="lazy" onerror="this.style.display='none'">` : ""}
        <div class="char-nom-display">${esc(item.label || item.ocr_char || "字")}</div>
      </div>
      <div class="char-card-info">
        <div class="char-syl-main">${esc(item.syllable || "(âm dị thể)")}</div>
        <div class="char-uni-code">${esc(item.unicode || "—")}</div>
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
      switchTab("inspector");
    });
    gridEl.appendChild(card);
  });
}

function stripAccents(s) {
  return String(s || "").replace(/đ/g, "d").replace(/Đ/g, "D").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/* ==========================================================================
   8. Kết quả: bảng tầng nhãn theo bộ + cây thư mục dataset/
   ========================================================================== */
function renderBenchmarks() {
  const tt = AppState.benchmarks?.tier_table;
  const head = document.getElementById("tierTableHead");
  const body = document.getElementById("tierTableBody");
  const note = document.getElementById("tierTableNote");
  if (!tt) {
    head.innerHTML = "";
    body.innerHTML = `<tr><td class="td-dim">Chưa có dữ liệu trong chế độ này — chạy máy chủ (.venv/bin/python web/server.py) hoặc sinh lại sample_data.json (.venv/bin/python web/build_sample_data.py).</td></tr>`;
    note.textContent = "";
    renderDatasetTree();
    return;
  }
  const cols = tt.columns || [];
  const lbl = (t) => (tt.labels || {})[t] || t;
  head.innerHTML = `<tr><th>Bộ</th><th class="td-num">Tổng ô</th>${cols.map((t) => `<th class="td-num"><span class="char-tier-pill ${tierDef(t).cls}">${esc(t)}</span></th>`).join("")}<th class="td-num">% GOLD</th></tr>`;
  const cells = (counts) => cols.map((t) => `<td class="td-num">${fmtInt((counts || {})[t] || 0)}</td>`).join("");
  let lastRole = null;
  const rows = (tt.rows || []).map((r) => {
    const group = r.role !== lastRole ? `<tr class="tr-group"><td colspan="${cols.length + 3}">${esc(roleLabel(r.role))}</td></tr>` : "";
    lastRole = r.role;
    if (!r.total) {
      return `${group}<tr><td class="metric-name">${esc(r.book_title)}<div class="td-sub">${esc(r.status_vi || "")}</div></td>
        <td class="td-num td-dim">—</td>${cols.map(() => `<td class="td-num td-dim">—</td>`).join("")}<td class="td-num td-dim">—</td></tr>`;
    }
    return `${group}<tr><td class="metric-name">${esc(r.book_title)}${r.status && r.status !== "ok" ? `<div class="td-sub">${esc(r.status_vi || r.status)}</div>` : ""}</td>
      <td class="td-num"><strong>${fmtInt(r.total)}</strong></td>${cells(r.counts)}<td class="td-num">${fmtPct1(r.gold_pct)}</td></tr>`;
  }).join("");
  const t = tt.totals || {};
  body.innerHTML = rows + `<tr class="tr-total"><td class="metric-name">Tổng</td><td class="td-num">${fmtInt(t.total)}</td>${cells(t.counts)}<td class="td-num">${fmtPct1(t.gold_pct)}</td></tr>`;
  const empty = tt.empty_tiers || [];
  note.textContent = [
    "GOLD / SYLLABLE / GOLD_text_only đếm trên labels.csv đã đóng gói của từng bộ; REVIEW / QUARANTINE (không giao ảnh) đếm trên bảng mọi tầng của bản dựng (dataset_out/labels_final.csv của STT, labels_gated.csv của sách khác).",
    empty.length ? `Tầng ${empty.join(", ")}: không có ô nào.` : "",
  ].filter(Boolean).join(" ");
  renderDatasetTree();
}

function renderDatasetTree() {
  const el = document.getElementById("datasetTree");
  const card = document.getElementById("datasetTreeCard");
  const st = AppState.stats;
  const books = Object.values(st?.books || {}).filter((b) => b.total);
  if (!el || !books.length) {
    if (card) card.classList.add("hidden");
    return;
  }
  if (card) card.classList.remove("hidden");
  const pad = (s) => (s + " ".repeat(Math.max(1, 24 - s.length)));
  const entries = [];
  const stt = books.filter((b) => b.book_set === "SachThanhTruyen");
  const sttNew = stt.filter((b) => b.status === "ok");
  if (sttNew.length) {
    const tot = sttNew.reduce((a, b) => a + (b.total || 0), 0);
    entries.push(["SachThanhTruyen/", `# ${fmtInt(tot)} nhãn (${sttNew.map((b) => b.set8 + " " + fmtInt(b.total)).join(", ")})`]);
  }
  stt.filter((b) => b.status === "ban_cu").forEach((b) => entries.push([`${b.id}/`, `# bản cũ: ${fmtInt(b.total)} nhãn`]));
  books.filter((b) => b.book_set !== "SachThanhTruyen").forEach((b) => {
    const tag = b.role === "giao_nop" ? "" : " · tập đánh giá";
    entries.push([`${b.book_set || b.id}/`, `# ${fmtInt(b.total)} nhãn${tag}`]);
  });
  const all = st.all_dataset;
  if (all) entries.push(["_ALL/", `# bộ gộp ${fmtInt(all.n_dong)} dòng · ${fmtInt(all.eval_only_dong)} evaluation_only`]);
  const lines = ["dataset/"];
  entries.forEach(([a, b], i) => lines.push((i === entries.length - 1 ? "└── " : "├── ") + pad(a) + b));
  el.textContent = lines.join("\n");
}
