/**
 * web/app.js — GanNhanOCR: trang minh hoạ bộ dữ liệu (chỉ trình bày TẦNG NHÃN).
 * Hai chế độ: (1) máy chủ web/server.py (API trên dữ liệu thật); (2) mở index.html trực tiếp -> sample_data.js/.json.
 * Mọi con số lấy từ API / dữ liệu mẫu — không gõ cứng.
 */

const AppState = {
  activeTab: "overview",
  isLiveServer: false,
  stats: null,
  books: [],
  pipelineFlow: [],
  benchmarks: null,
  sampleData: null,
  inspector: { book: "LucVanTien1883", page: "", zoom: 1, chars: [], selected: null, hidden: new Set() },
  gallery: { query: "", book: "all", tier: "all", sample: [], loaded: false },
};

const NA = "—";
const TIERS = ["GOLD", "SYLLABLE", "GOLD_text_only", "REVIEW", "QUARANTINE"];
const ROLE_GROUPS = [
  { role: "giao_nop", label: "Giao nộp" },
  { role: "danh_gia_ihr", label: "Đánh giá (IHR-NomDB, có nhãn người)" },
  { role: "danh_gia_borg", label: "Đánh giá (Borg, chép tay, có nhãn người)" },
];
const GROUP_ROLES = { giao_nop: ["giao_nop"], danh_gia: ["danh_gia_ihr", "danh_gia_borg"] };
const TAB_HASH = { overview: "gioi-thieu", flow: "quy-trinh", inspector: "ban-quet", gallery: "tra-cuu", benchmarks: "so-lieu" };

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initInspectorControls();
  initGalleryControls();
  initReloadButton();
  loadData();
});

/* ---------------- Tiện ích ---------------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmtInt(x) {
  if (x === null || x === undefined || x === "" || Number.isNaN(Number(x))) return NA;
  return Number(x).toLocaleString("vi-VN");
}
function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return NA;
  return `${Number(x).toLocaleString("vi-VN", { maximumFractionDigits: 1, minimumFractionDigits: 1 })} %`;
}
function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}
function tierKey(t) { return TIERS.includes(t) ? t : "OTHER"; }
function pageNo(p) { return String(p || "").replace("page_", "").replace(/^0+(?=\d)/, ""); }
function roleOfBook(id) { return (AppState.books.find((b) => b.id === id) || {}).role || "giao_nop"; }
function stripAccents(s) {
  return String(s || "").replace(/đ/g, "d").replace(/Đ/g, "D").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/* ---------------- Mục (tab) ---------------- */
function initTabs() {
  document.querySelectorAll(".site-nav a").forEach((a) => {
    a.addEventListener("click", (e) => { e.preventDefault(); switchTab(a.dataset.tab, true); });
  });
  const fromHash = Object.keys(TAB_HASH).find((k) => `#${TAB_HASH[k]}` === location.hash);
  switchTab(fromHash || "overview", false);
  window.addEventListener("hashchange", () => {
    const k = Object.keys(TAB_HASH).find((x) => `#${TAB_HASH[x]}` === location.hash);
    if (k && k !== AppState.activeTab) switchTab(k, false);
  });
}

function switchTab(tab, pushHash) {
  AppState.activeTab = tab;
  document.querySelectorAll(".site-nav a").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
  document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));
  if (pushHash && location.hash !== `#${TAB_HASH[tab]}`) {
    try { history.replaceState(null, "", `#${TAB_HASH[tab]}`); } catch (e) { location.hash = TAB_HASH[tab]; }
  }
  if (!AppState.stats) return;
  if (tab === "inspector") loadInspectorPage();
  else if (tab === "gallery" && !AppState.gallery.loaded) executeSearch();
}

/* ---------------- Nạp dữ liệu: API -> dự phòng dữ liệu mẫu ---------------- */
async function loadData() {
  if (location.protocol !== "file:") {
    try {
      const signal = (typeof AbortSignal !== "undefined" && AbortSignal.timeout) ? AbortSignal.timeout(15000) : undefined;
      const r = await fetch("/api/stats", { cache: "no-store", signal });
      if (r.ok) {
        AppState.isLiveServer = true;
        AppState.stats = await r.json();
        const [books, flow, bench] = await Promise.all(
          ["/api/books", "/api/pipeline_flow", "/api/benchmarks"].map((u) => fetch(u, { cache: "no-store" }).then((x) => x.json())));
        AppState.books = books || [];
        AppState.pipelineFlow = flow || [];
        AppState.benchmarks = bench;
        setText("connectionText", `Dữ liệu từ máy chủ, nạp lúc ${AppState.stats.loaded_at || NA}.`);
        document.getElementById("reloadDataBtn").classList.remove("hidden");
        renderAll();
        return;
      }
    } catch (err) {
      console.warn("[GanNhanOCR] API không phản hồi, dùng dữ liệu mẫu:", err);
    }
  }
  const sample = await loadSampleData();
  if (!sample) {
    setText("connectionText", "Không có dữ liệu (không có máy chủ và không có sample_data.js).");
    return;
  }
  AppState.sampleData = sample;
  AppState.stats = sample.stats || null;
  AppState.books = sample.books || [];
  AppState.pipelineFlow = sample.pipeline_flow || [];
  AppState.benchmarks = sample.benchmarks || null;
  AppState.gallery.sample = sample.gallery || [];
  setText("connectionText", `Dữ liệu mẫu (1 trang mỗi bộ), sinh lúc ${sample.generated_at || NA}.`);
  renderAll();
}

// file:// chặn fetch() -> dùng sample_data.js (gán window.GANNHANOCR_SAMPLE) nạp bằng thẻ <script>.
async function loadSampleData() {
  if (window.GANNHANOCR_SAMPLE) return window.GANNHANOCR_SAMPLE;
  if (location.protocol !== "file:") {
    try {
      const r = await fetch("sample_data.json");
      if (r.ok) return await r.json();
    } catch (e) { /* thử sample_data.js */ }
  }
  return new Promise((resolve) => {
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
    btn.textContent = "Đang nạp lại…";
    try {
      const d = await (await fetch("/api/reload", { cache: "no-store" })).json();
      if (!d.ok) alert(d.message || "Không nạp lại được");
      AppState.gallery.loaded = false;
      await loadData();
    } catch (e) {
      alert("Lỗi khi nạp lại: " + e);
    } finally {
      btn.disabled = false;
      btn.textContent = "Nạp lại dữ liệu";
    }
  });
}

function renderAll() {
  renderHeader();
  renderOverview();
  renderFlow();
  renderNumbers();
  populateBookSelects();
  if (AppState.activeTab === "inspector") loadInspectorPage();
  else if (AppState.activeTab === "gallery") executeSearch();
}

/* ---------------- Đầu trang ---------------- */
function renderHeader() {
  const im = AppState.stats?.impact_metrics || {};
  setText("hdrSummary", `${im.books_count ?? NA} bộ, ${fmtInt(im.total_characters)} ô`);
  setText("footerRoot", AppState.stats?.data_root ? `Gốc dữ liệu: ${AppState.stats.data_root}` : "");
}

/* ---------------- 1. Giới thiệu ---------------- */
function groupedRows(items, roleOf, colspan, rowHtml) {
  let html = "";
  ROLE_GROUPS.forEach((g) => {
    const arr = items.filter((x) => (roleOf(x) || "giao_nop") === g.role);
    if (!arr.length) return;
    html += `<tr class="group"><td colspan="${colspan}">${esc(g.label)}</td></tr>` + arr.map(rowHtml).join("");
  });
  return html;
}

function bookKind(b) {
  const s = b.script === "viết tay" ? "chép tay" : (b.script || "");
  return [s, (b.genre || "").toLowerCase()].filter(Boolean).join(", ");
}

function statusNote(b) {
  if (!b.status || b.status === "ok") return "";
  if (b.status === "ban_cu") return " (bản cũ)";
  if (b.status === "chua_co") return " (chưa có dữ liệu)";
  return ` (${b.status_vi || b.status})`;
}

function renderOverview() {
  const st = AppState.stats;
  if (!st) return;
  const im = st.impact_metrics || {};
  const books = Object.values(st.books || {});
  setText("ovBooksCount", im.books_count ?? books.length);
  const row = (b) => `<tr><td>${esc(b.title)}<span class="dim">${esc(statusNote(b))}</span></td><td class="hide-narrow">${esc(bookKind(b))}</td>
    <td class="num">${b.total ? fmtInt(b.total) : NA}</td><td class="num">${b.total ? fmtInt(b.gold) : NA}</td>
    <td class="num">${b.total ? fmtInt(b.review) : NA}</td></tr>`;
  document.getElementById("bookTableBody").innerHTML = groupedRows(books, (b) => b.role, 5, row)
    + `<tr class="total"><td>Tổng</td><td class="hide-narrow"></td><td class="num">${fmtInt(im.total_characters)}</td>
       <td class="num">${fmtInt(im.total_gold)}</td><td class="num">${fmtInt(im.total_review)}</td></tr>`;
  const hasTxt = !!im.total_gold_text_only;
  document.getElementById("dtTextOnly").classList.toggle("hidden", !hasTxt);
  document.getElementById("ddTextOnly").classList.toggle("hidden", !hasTxt);
}

/* ---------------- 2. Quy trình ---------------- */
function renderFlow() {
  const steps = AppState.pipelineFlow || [];
  const body = document.getElementById("flowTableBody");
  if (!steps.length) {
    body.innerHTML = `<tr><td colspan="5" class="dim">Chưa có mô tả quy trình.</td></tr>`;
    document.getElementById("flowMetricsBlock").classList.add("hidden");
    return;
  }
  body.innerHTML = steps.map((s) => `<tr><td class="num">${esc(s.step)}</td><td>${esc(s.name)}</td>
    <td>${esc(s.input)}</td><td>${esc(s.method ?? s.model ?? "")}</td><td>${/[/.]/.test(s.output || "") ? `<code>${esc(s.output)}</code>` : esc(s.output)}</td></tr>`).join("");
  const mrows = [];
  steps.forEach((s) => (s.metrics || []).forEach((m) => mrows.push(
    `<tr><td>${esc(s.step)}</td><td title="${esc(m.source || "")}">${esc(m.label)}</td><td class="num">${esc(m.value)}</td></tr>`)));
  document.getElementById("flowMetricsBody").innerHTML = mrows.join("");
  document.getElementById("flowMetricsBlock").classList.toggle("hidden", !mrows.length);
}

/* ---------------- 3. Bản quét ---------------- */
function bookOptionsHtml(withGroups) {
  let html = withGroups ? `<option value="all">Tất cả</option><option value="giao_nop">Nhóm giao nộp</option><option value="danh_gia">Nhóm đánh giá</option>` : "";
  ROLE_GROUPS.forEach((g) => {
    const arr = AppState.books.filter((b) => (b.role || "giao_nop") === g.role);
    if (!arr.length) return;
    html += `<optgroup label="${esc(g.label)}">` + arr.map((b) =>
      `<option value="${esc(b.id)}">${esc(b.title)}${esc(statusNote(b))}</option>`).join("") + "</optgroup>";
  });
  return html;
}

function hasData(b) { return !!(b && (b.total_chars || b.total)); }
function currentBook() { return AppState.books.find((b) => b.id === AppState.inspector.book); }

function populateBookSelects() {
  const sel = document.getElementById("inspectorBookSelect");
  if (AppState.books.length) {
    sel.innerHTML = bookOptionsHtml(false);
    const pick = AppState.books.find((b) => b.id === AppState.inspector.book && hasData(b))
      || AppState.books.find(hasData) || AppState.books[0];
    AppState.inspector.book = pick.id;
    sel.value = pick.id;
    updatePageOptions(!AppState.inspector.page);
  }
  const gf = document.getElementById("galleryBookFilter");
  if (AppState.books.length) {
    const cur = gf.value || "all";
    gf.innerHTML = bookOptionsHtml(true);
    gf.value = [...gf.options].some((o) => o.value === cur) ? cur : "all";
  }
}

function updatePageOptions(preferDefault) {
  const sel = document.getElementById("inspectorPageSelect");
  const b = currentBook();
  const pages = b?.available_pages?.length ? b.available_pages : (b?.sample_pages || []);
  sel.innerHTML = pages.length ? pages.map((p) => `<option value="${esc(p)}">${esc(pageNo(p))}</option>`).join("") : `<option value="">—</option>`;
  const ins = AppState.inspector;
  if (!preferDefault && pages.includes(ins.page)) { /* giữ trang đang chọn */ }
  else if (b?.default_page && pages.includes(b.default_page)) ins.page = b.default_page;
  else ins.page = pages[0] || "";
  sel.value = ins.page;
  const bits = [];
  if (b?.status && b.status !== "ok") bits.push(b.status_vi || b.status);
  if (b?.note) bits.push(b.note);
  if (!AppState.isLiveServer && pages.length) bits.push("Chế độ dữ liệu mẫu: chỉ có 1 trang cho mỗi bộ.");
  setText("inspectorBookStatus", bits.join(" · "));
}

function initInspectorControls() {
  const bookSel = document.getElementById("inspectorBookSelect");
  const pageSel = document.getElementById("inspectorPageSelect");
  bookSel.addEventListener("change", (e) => { AppState.inspector.book = e.target.value; updatePageOptions(true); loadInspectorPage(); });
  pageSel.addEventListener("change", (e) => { AppState.inspector.page = e.target.value; loadInspectorPage(); });
  const step = (d) => {
    const opts = [...pageSel.options].map((o) => o.value).filter(Boolean);
    const j = opts.indexOf(AppState.inspector.page) + d;
    if (j >= 0 && j < opts.length) { AppState.inspector.page = opts[j]; pageSel.value = opts[j]; loadInspectorPage(); }
  };
  document.getElementById("btnPrevPage").addEventListener("click", () => step(-1));
  document.getElementById("btnNextPage").addEventListener("click", () => step(1));
  const zoom = (z) => { AppState.inspector.zoom = z; applyZoom(); };
  document.getElementById("btnZoomIn").addEventListener("click", () => zoom(Math.min(AppState.inspector.zoom + 0.25, 3)));
  document.getElementById("btnZoomOut").addEventListener("click", () => zoom(Math.max(AppState.inspector.zoom - 0.25, 0.5)));
  document.getElementById("btnZoomReset").addEventListener("click", () => zoom(1));
}

function applyZoom() {
  const z = AppState.inspector.zoom;
  document.getElementById("scanContainer").style.width = `${Math.round(z * 100)}%`;
  setText("btnZoomReset", `${Math.round(z * 100)}%`);
}

async function loadInspectorPage() {
  const { book, page } = AppState.inspector;
  const b = currentBook();
  const img = document.getElementById("pageScanImage");
  document.getElementById("bboxOverlay").innerHTML = "";
  let data = null;
  if (AppState.isLiveServer) {
    try {
      const r = await fetch(`/api/page?book=${encodeURIComponent(book)}&page=${encodeURIComponent(page || "")}`);
      if (r.ok) data = await r.json();
    } catch (e) { console.warn("Lỗi khi tải trang:", e); }
  } else {
    const p = AppState.sampleData?.pages?.[book];
    data = p && (!page || p.page === page) ? p : null;
  }
  if (AppState.inspector.book !== book || AppState.inspector.page !== page) return; // đã chuyển trang khác
  applyZoom();
  const msgEl = document.getElementById("pageStatusMsg");
  const title = b?.title || book;
  if (!data) {
    AppState.inspector.chars = [];
    img.src = placeholder(1000, 1400, "Chưa có dữ liệu trang");
    setText("scanCaption", `Hình 1. ${title}: chưa có dữ liệu trang.`);
    msgEl.classList.add("hidden");
    renderBoxes([]);
    return;
  }
  const chars = data.characters || [];
  AppState.inspector.chars = chars;
  msgEl.textContent = data.message || "";
  msgEl.classList.toggle("hidden", !data.message);
  const counts = {};
  chars.forEach((c) => { const k = tierKey(c.tier); counts[k] = (counts[k] || 0) + 1; });
  const parts = [...TIERS, "OTHER"].filter((t) => counts[t]).map((t) => `${t === "OTHER" ? "khác" : t} ${fmtInt(counts[t])}`);
  setText("scanCaption", `Hình 1. ${title}, trang ${pageNo(data.page || page)}: ${fmtInt(chars.length)} ô${parts.length ? ` (${parts.join(", ")})` : ""}. Viền hộp tô theo tầng nhãn.`);
  const W = data.dimensions?.width || 1000;
  const H = data.dimensions?.height || 1400;
  img.onerror = () => { img.onerror = null; img.src = placeholder(W, H, "Không tải được ảnh trang"); };
  img.src = data.scan_url || placeholder(W, H, "Không có ảnh trang");
  renderBoxes(chars);
}

function placeholder(w, h, text) {
  const c = document.createElement("canvas");
  c.width = 600;
  c.height = Math.max(200, Math.round((h / w) * 600));
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#f5f5f3";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#5c5c5c";
  ctx.font = "16px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, c.width / 2, c.height / 2);
  return c.toDataURL();
}

function renderBoxes(chars) {
  const overlay = document.getElementById("bboxOverlay");
  overlay.innerHTML = "";
  chars.forEach((c) => {
    const k = tierKey(c.tier);
    const el = document.createElement("div");
    el.className = `bbox t-${k}`;
    el.id = `bbox-${c.index}`;
    el.dataset.cat = k;
    Object.assign(el.style, { left: `${c.rect.left_pct}%`, top: `${c.rect.top_pct}%`, width: `${c.rect.width_pct}%`, height: `${c.rect.height_pct}%` });
    el.title = `${c.label || c.ocr_char || ""} ${c.syllable || ""} · ${c.tier || ""}`;
    el.addEventListener("click", () => selectChar(c, el));
    overlay.appendChild(el);
  });
  renderLegend(chars);
  applyHidden();
  const first = chars.find((c) => c.tier === "GOLD") || chars[0];
  if (first) selectChar(first, document.getElementById(`bbox-${first.index}`));
  else {
    AppState.inspector.selected = null;
    document.getElementById("detailBody").innerHTML = `<tr><td class="dim">Trang không có ô chữ.</td></tr>`;
  }
}

function renderLegend(chars) {
  const el = document.getElementById("legendFilters");
  const counts = {};
  chars.forEach((c) => { const k = tierKey(c.tier); counts[k] = (counts[k] || 0) + 1; });
  const cats = [...TIERS, "OTHER"].filter((k) => counts[k]);
  const hidden = AppState.inspector.hidden;
  el.innerHTML = cats.length ? "Hiện: " + cats.map((k) => `<label class="t-${k}"><input type="checkbox" data-cat="${k}" ${hidden.has(k) ? "" : "checked"}>`
    + `<span class="lt">${k === "OTHER" ? "khác" : k}</span> <span class="n">(${fmtInt(counts[k])})</span></label>`).join("") : "";
  el.querySelectorAll("input").forEach((cb) => cb.addEventListener("change", (e) => {
    const k = e.target.dataset.cat;
    if (e.target.checked) hidden.delete(k); else hidden.add(k);
    applyHidden();
  }));
}

function applyHidden() {
  const hidden = AppState.inspector.hidden;
  document.querySelectorAll("#bboxOverlay .bbox").forEach((el) => el.classList.toggle("hidden", hidden.has(el.dataset.cat)));
}

function selectChar(c, el) {
  AppState.inspector.selected = c;
  document.querySelectorAll("#bboxOverlay .bbox.selected").forEach((x) => x.classList.remove("selected"));
  if (el) el.classList.add("selected");
  const k = tierKey(c.tier);
  const rows = [
    ["Ảnh cắt", c.crop_url ? `<img src="${esc(c.crop_url)}" alt="crop" onerror="this.replaceWith(document.createTextNode('—'))">` : `<span class="dim">không có</span>`],
    ["Chữ", `<span class="glyph">${esc(c.label || NA)}</span>`],
    ["Âm", esc(c.syllable || NA)],
    ["Tầng", `<span class="sw t-${k}"></span>${esc(c.tier || NA)}`],
    ["Luật gán", `<span class="mono">${esc(c.rule || NA)}</span>`],
    ["Unicode", `<span class="mono">${esc(c.unicode || NA)}</span>`],
    ["OCR", `<span class="glyph" style="font-size:1.25rem">${esc(c.ocr_char || NA)}</span>`],
    ["Cột · ô", `${esc(c.column ?? "?")} · ${esc(c.index ?? "?")}`],
    ["Hộp (px)", `<span class="mono">${esc((c.bbox || []).map((v) => Math.round(v)).join(", "))}</span>`],
  ];
  document.getElementById("detailBody").innerHTML = rows.map(([a, b]) => `<tr><th>${a}</th><td>${b}</td></tr>`).join("");
}

/* ---------------- 4. Tra cứu ---------------- */
function initGalleryControls() {
  let t = null;
  document.getElementById("gallerySearchInput").addEventListener("input", (e) => {
    clearTimeout(t);
    t = setTimeout(() => { AppState.gallery.query = e.target.value.trim(); executeSearch(); }, 250);
  });
  document.getElementById("galleryBookFilter").addEventListener("change", (e) => { AppState.gallery.book = e.target.value; executeSearch(); });
  document.getElementById("galleryTierFilter").addEventListener("change", (e) => { AppState.gallery.tier = e.target.value; executeSearch(); });
}

async function executeSearch() {
  const { query, book, tier } = AppState.gallery;
  const body = document.getElementById("galleryBody");
  body.innerHTML = `<tr><td colspan="7" class="dim">Đang tìm…</td></tr>`;
  let results = [];
  let total = null;
  if (AppState.isLiveServer) {
    try {
      const r = await fetch(`/api/search?q=${encodeURIComponent(query)}&book=${encodeURIComponent(book)}&tier=${encodeURIComponent(tier)}&limit=60`);
      if (r.ok) { const d = await r.json(); results = d.results || []; total = d.total_matches; }
    } catch (e) { console.warn("Lỗi tìm kiếm:", e); }
  } else {
    const q = query.toLowerCase();
    const qn = stripAccents(q);
    results = AppState.gallery.sample.filter((it) => {
      if (GROUP_ROLES[book]) { if (!GROUP_ROLES[book].includes(roleOfBook(it.book))) return false; }
      else if (book !== "all" && it.book !== book) return false;
      if (tier !== "all" && it.tier !== tier) return false;
      if (!q) return true;
      const syl = (it.syllable || "").toLowerCase();
      return syl.includes(q) || stripAccents(syl).includes(qn) || (it.label || "").includes(query) || (it.unicode || "").toLowerCase().includes(q);
    });
    total = results.length;
  }
  AppState.gallery.loaded = true;
  setText("galleryCountText", total !== null && total > results.length
    ? `${fmtInt(total)} ô khớp; hiện ${fmtInt(results.length)} (rải đều giữa các bộ).`
    : `${fmtInt(results.length)} ô khớp${AppState.isLiveServer ? "" : " trong dữ liệu mẫu"}.`);
  if (!results.length) {
    body.innerHTML = `<tr><td colspan="7" class="dim">Không có kết quả.</td></tr>`;
    return;
  }
  body.innerHTML = "";
  results.forEach((it) => {
    const tr = document.createElement("tr");
    const k = tierKey(it.tier);
    tr.innerHTML = `<td>${it.crop_url ? `<img src="${esc(it.crop_url)}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">` : ""}</td>
      <td class="glyph">${esc(it.label || it.ocr_char || NA)}</td><td>${esc(it.syllable || NA)}</td>
      <td class="tier"><span class="sw t-${k}"></span>${esc(it.tier || NA)}</td><td class="mono hide-narrow">${esc(it.unicode || NA)}</td>
      <td>${esc(it.book_title || it.book)}</td><td class="num">${esc(pageNo(it.page))} · ${esc(it.column ?? "")}</td>`;
    tr.addEventListener("click", () => openInScan(it));
    body.appendChild(tr);
  });
}

function openInScan(it) {
  AppState.inspector.book = it.book;
  AppState.inspector.page = it.page || "";
  document.getElementById("inspectorBookSelect").value = it.book;
  updatePageOptions(false);
  switchTab("inspector", true);
  window.scrollTo(0, 0);
}

/* ---------------- 5. Số liệu ---------------- */
function renderNumbers() {
  const tt = AppState.benchmarks?.tier_table;
  const head = document.getElementById("tierTableHead");
  const body = document.getElementById("tierTableBody");
  const note = document.getElementById("tierTableNote");
  if (!tt) {
    head.innerHTML = "";
    body.innerHTML = `<tr><td class="dim">Chưa có dữ liệu.</td></tr>`;
    note.textContent = "";
  } else {
    const cols = tt.columns || [];
    const n = cols.length + 3;
    head.innerHTML = `<tr><th>Bộ</th><th class="num">Tổng</th>${cols.map((t) => `<th class="num">${esc(t)}</th>`).join("")}<th class="num">% GOLD</th></tr>`;
    const cells = (c) => cols.map((t) => `<td class="num">${fmtInt((c || {})[t] || 0)}</td>`).join("");
    const row = (r) => (r.total
      ? `<tr><td>${esc(r.book_title)}<span class="dim">${esc(statusNote(r))}</span></td><td class="num">${fmtInt(r.total)}</td>${cells(r.counts)}<td class="num">${fmtPct(r.gold_pct)}</td></tr>`
      : `<tr><td>${esc(r.book_title)} <span class="dim">(chưa có dữ liệu)</span></td>${"<td class=\"num dim\">—</td>".repeat(n - 1)}</tr>`);
    const t = tt.totals || {};
    body.innerHTML = groupedRows(tt.rows || [], (r) => r.role, n, row)
      + `<tr class="total"><td>Tổng</td><td class="num">${fmtInt(t.total)}</td>${cells(t.counts)}<td class="num">${fmtPct(t.gold_pct)}</td></tr>`;
    const empty = tt.empty_tiers || [];
    note.textContent = "GOLD, SYLLABLE, GOLD_text_only: đếm trên dataset/<Bộ>/labels.csv. REVIEW, QUARANTINE (không đóng gói ảnh): đếm trên bảng mọi tầng của bản dựng."
      + (empty.length ? ` Tầng không có ô: ${empty.join(", ")}.` : "");
  }
  const im = AppState.stats?.impact_metrics || {};
  const inv = document.getElementById("invariantsLine");
  inv.textContent = im.invariants_available && im.invariants_text
    ? `${im.invariants_text} (measure_out/SUMMARY.json${im.invariants_generated_at ? `, ${im.invariants_generated_at.replace("T", " ")}` : ""}).` : "";
  inv.classList.toggle("hidden", !inv.textContent);
  renderDatasetTree();
}

function renderDatasetTree() {
  const card = document.getElementById("datasetTreeCard");
  const st = AppState.stats;
  const books = Object.values(st?.books || {}).filter((b) => b.total);
  if (!books.length) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  const entries = [];
  const stt = books.filter((b) => b.book_set === "SachThanhTruyen");
  const sttNew = stt.filter((b) => b.status === "ok");
  if (sttNew.length) {
    entries.push(["SachThanhTruyen/", `${fmtInt(sttNew.reduce((a, b) => a + (b.total || 0), 0))} ô (${sttNew.map((b) => `${b.set8} ${fmtInt(b.total)}`).join(", ")})`]);
  }
  stt.filter((b) => b.status === "ban_cu").forEach((b) => entries.push([`${b.id}/`, `bản cũ, ${fmtInt(b.total)} ô`]));
  books.filter((b) => b.book_set !== "SachThanhTruyen").forEach((b) =>
    entries.push([`${b.book_set || b.id}/`, `${fmtInt(b.total)} ô${b.role === "giao_nop" ? "" : ", evaluation_only"}`]));
  if (st.all_dataset) entries.push(["_ALL/", `gộp: ${fmtInt(st.all_dataset.n_dong)} dòng, ${fmtInt(st.all_dataset.eval_only_dong)} evaluation_only`]);
  const w = Math.max(...entries.map(([a]) => a.length)) + 2;
  document.getElementById("datasetTree").textContent = ["dataset/"].concat(entries.map(([a, b], i) =>
    `${i === entries.length - 1 ? "└── " : "├── "}${a.padEnd(w)}# ${b}`)).join("\n");
}
