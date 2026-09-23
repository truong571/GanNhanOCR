/**
 * web/app.js — GanNhanOCR Web Presentation Client Logic
 * Điều khiển giao diện tương tác trình diễn đề tài luận văn Thạc sĩ
 */

// Trạng thái toàn cục của ứng dụng
const AppState = {
  theme: localStorage.getItem("gannhan_theme") || "light",
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
    filterGold: true,
    filterSyl: true,
    filterTxt: true,
  },
  gallery: {
    query: "",
    book: "all",
    tier: "all",
    results: [],
  },
};

// Khởi tạo khi DOM sẵn sàng
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initInspectorControls();
  initGalleryControls();
  loadData();
});

/* ==========================================================================
   1. Theme Management (Sáng / Tối)
   ========================================================================== */
function initTheme() {
  document.documentElement.setAttribute("data-theme", AppState.theme);
  const themeBtn = document.getElementById("themeToggleBtn");
  themeBtn.addEventListener("click", () => {
    AppState.theme = AppState.theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", AppState.theme);
    localStorage.setItem("gannhan_theme", AppState.theme);
  });
}

/* ==========================================================================
   2. Điều Hướng Tab (SPA Navigation)
   ========================================================================== */
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  AppState.activeTab = tabId;

  // Cập nhật nút tab
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
  });

  // Cập nhật nội dung tab
  document.querySelectorAll(".tab-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.id === `tab-${tabId}`);
  });

  // Khởi chạy tác vụ đặc thù theo từng tab nếu cần
  if (tabId === "inspector") {
    loadInspectorPage();
  } else if (tabId === "gallery" && AppState.gallery.results.length === 0) {
    executeSearch();
  }
}

/* ==========================================================================
   3. Nạp Dữ Liệu (Hybrid: Ưu tiên Live API -> Fallback Sample Data)
   ========================================================================== */
async function loadData() {
  const badge = document.getElementById("connectionBadge");
  const badgeText = badge.querySelector(".status-text");

  try {
    // Thử gọi API server nội bộ
    const testResp = await fetch("/api/stats", { signal: AbortSignal.timeout(2000) });
    if (testResp.ok) {
      AppState.isLiveServer = true;
      AppState.stats = await testResp.json();

      // Nạp song song các tài nguyên API
      const [booksResp, flowResp, benchResp] = await Promise.all([
        fetch("/api/books").then((r) => r.json()),
        fetch("/api/pipeline_flow").then((r) => r.json()),
        fetch("/api/benchmarks").then((r) => r.json()),
      ]);

      AppState.books = booksResp;
      AppState.pipelineFlow = flowResp;
      AppState.benchmarks = benchResp;

      badge.classList.add("connected");
      badgeText.textContent = "Máy chủ API: Trực tuyến";
      renderAllComponents();
      return;
    }
  } catch (err) {
    console.warn("[GanNhanOCR] Server API không phản hồi, chuyển sang chế độ dữ liệu mẫu:", err);
  }

  // Chế độ Standalone: Đọc từ sample_data.json
  try {
    const localResp = await fetch("sample_data.json");
    if (localResp.ok) {
      AppState.sampleData = await localResp.json();
      badge.classList.remove("connected");
      badgeText.textContent = "Chế độ dữ liệu mẫu";

      adaptSampleData();
      renderAllComponents();
    }
  } catch (err) {
    console.error("[GanNhanOCR] Không nạp được cả API lẫn sample_data.json", err);
    badgeText.textContent = "Ngoại tuyến";
  }
}

function adaptSampleData() {
  if (!AppState.sampleData) return;
  const s = AppState.sampleData;

  AppState.stats = {
    impact_metrics: s.stats,
    books: s.books,
  };

  AppState.books = Object.values(s.books).map((b) => ({
    id: b.id,
    title: b.title,
    subtitle: b.subtitle,
    layout: b.layout,
    total_chars: b.total,
    sample_pages: [b.sample_page],
    default_page: b.sample_page,
  }));

  AppState.pipelineFlow = [
    {
      step: 1,
      name: "Tiền xử lý ảnh tài liệu",
      tag: "Phân đoạn & Khử nhiễu",
      input: "Tệp ảnh quét tài liệu gốc (300 DPI) và bản phiên âm Quốc ngữ đối ứng",
      model: "Thuật toán nhị phân hóa thích nghi Sauvola kết hợp Otsu",
      process: "Khử nhiễu nền giấy ố vàng, tách biên trang, phân đoạn cột văn bản dọc (10 cột với thơ, 7 cột với văn xuôi) và khởi tạo nhận dạng ký tự sơ bộ.",
      output: "Tập ảnh trang đã chuẩn hóa, tọa độ phân đoạn cột và dữ liệu nhận dạng ban đầu",
      evidence: "Đảm bảo tính độc lập và khả năng tái lập kết quả phân đoạn cột.",
    },
    {
      step: 2,
      name: "Phát hiện vị trí ký tự",
      tag: "CenterNet & Pitch Decoding",
      input: "Ảnh các cột chữ dọc bóc tách từ trang tài liệu",
      model: "Mạng CenterNet (Backbone ResNet-34) kết hợp giải mã nhịp ký tự (Pitch Decoding)",
      process: "Dự đoán tâm ký tự Nôm qua bản đồ nhiệt (heatmap), phân tách các vị trí dính chữ bằng giải mã khoảng cách nhịp đều, triệt tiêu hộp rỗng và cắt phạm nét.",
      output: "Tập hợp tọa độ hộp bao ký tự [xmin, ymin, xmax, ymax] cho từng cột chữ",
      evidence: "Tỷ lệ cắt phạm thân chữ giảm từ 10.6% xuống 2.6%; đếm đúng số chữ trên cột đạt >78%.",
    },
    {
      step: 3,
      name: "Gióng hàng song ngữ",
      tag: "Banded Dynamic Programming",
      input: "Tập hộp ký tự phát hiện được và chuỗi âm Quốc ngữ đối ứng",
      model: "Quy hoạch động dải hẹp (Banded DP) kết hợp từ điển Hán Nôm Quốc ngữ (104.177 mục từ)",
      process: "Tìm đường đi tối ưu giữa chuỗi hộp ảnh và chuỗi âm tiết văn bản. Áp dụng ràng buộc cấu trúc nhịp thơ lục bát (câu lục 6 chữ, câu bát 8 chữ) để ngăn lệch vị trí xuyên dòng.",
      output: "Bảng nhãn sơ bộ cho từng hộp ký tự kèm xác suất hậu nghiệm và mã quy tắc liên kết",
      evidence: "Quy trình gán nhãn vận hành hoàn toàn theo quy tắc thuật toán, không cần can thiệp thủ công.",
    },
    {
      step: 4,
      name: "Kiểm kê & Hiệu chỉnh lỗi",
      tag: "Rà soát nhầm lẫn dị tự",
      input: "Bảng nhãn sơ bộ sau giai đoạn gióng hàng",
      model: "Kiểm kê tần suất ngữ cảnh và bảng tri thức sửa lỗi nhầm lẫn có tính hệ thống",
      process: "Phát hiện các chữ bị nhầm lẫn phổ biến (đồng âm khác nghĩa, tự dạng gần giống nhau). Phân loại nhãn theo 3 bậc chất lượng: GOLD, SYLLABLE, và nhãn cần cách ly.",
      output: "Bảng nhãn đã được chuẩn hóa và hiệu chỉnh",
      evidence: "Lưu vết kiểm tra toàn vẹn bằng chuỗi mã băm SHA-256 sau mỗi bước biến đổi dữ liệu.",
    },
    {
      step: 5,
      name: "Kiểm soát biên & Cứu nhãn",
      tag: "Cổng cơ chế & Đối soát dị bản",
      input: "Các vị trí ký tự nghi vấn hoặc lệch số lượng",
      model: "Mô hình nhận dạng nội vùng SE-ResNet kết hợp 4 cổng kiểm soát điều kiện biên",
      process: "Đối chiếu độc lập qua 4 cổng điều kiện (số lượng chữ trên cột, nhịp pitch, ranh giới hộp bao, và so sánh chéo với các bản khắc độc lập 1871/1916) để nâng bậc nhãn an toàn hoặc phân loại nhãn văn bản thuần.",
      output: "Bảng nhãn công bố hoàn thiện",
      evidence: "Nâng tỷ lệ nhãn đạt chuẩn chất lượng cao mà không làm tăng độ nhiễu của bộ ngữ liệu.",
    },
    {
      step: 6,
      name: "Đóng gói tập ngữ liệu",
      tag: "Xuất bản bộ dữ liệu chuẩn",
      input: "Bảng nhãn hoàn thiện và tập ảnh trích xuất từ các giai đoạn trước",
      model: "Quy trình đóng gói tự chứa chuẩn hóa",
      process: "Trích xuất ảnh cắt ký tự theo từng mức tin cậy (thư mục gold/, syllable/), xây dựng bảng chỉ mục chuẩn 12 trường thông tin (labels.csv, labels.xlsx) và tự động tạo tài liệu mô tả xuất xứ thư tịch.",
      output: "Thư mục dataset/ tự chứa hoàn chỉnh: labels.csv, gold/, syllable/, tài liệu kỹ thuật",
      evidence: "Tập dữ liệu độc lập hoàn toàn, sẵn sàng phục vụ huấn luyện và đánh giá các mô hình OCR.",
    },
  ];

  AppState.benchmarks = {
    comparisons: [
      {
        metric: "Tỷ lệ đếm đúng số chữ trên cột (n_det == N)",
        baseline: "59.2% (Ngưỡng thô 0.20)",
        proposed: "78.4% - 90.0% (CenterNet + Pitch Decoder)",
        improvement: "+19.2% đến +30.8%",
        impact: "Triệt tiêu hiện tượng dính chữ và mất chữ trên các cột thạch bản nét mảnh.",
      },
      {
        metric: "Tỷ lệ cắt phạm vào thân chữ (Ink Cut Rate)",
        baseline: "10.6% (Bổ đôi hộp đều tuyến tính)",
        proposed: "2.6% (Pitch Decoding thích ứng)",
        improvement: "Giảm 4 lần (giảm 8.0%)",
        impact: "Bảo toàn nguyên vẹn cấu trúc nét của từng chữ Nôm trong ảnh crop.",
      },
      {
        metric: "Độ chính xác đối soát dị bản (Cross-Edition Agreement)",
        baseline: "64.4% (OCR đơn kênh Hán thông thường)",
        proposed: "86.3% (OCR kênh Nôm chuyên biệt + Cổng cơ chế)",
        improvement: "+21.9%",
        impact: "Khớp chính xác với các bản khắc độc lập thời Nguyễn.",
      },
      {
        metric: "Tính Tự Động Hoá (Human-in-the-loop Rules)",
        baseline: "Cần can thiệp người gán nhãn thủ công",
        proposed: "0 ô can thiệp thủ công (quyet_dinh_nguoi = 0)",
        improvement: "Tự động 100%",
        impact: "Đảm bảo tính khách quan tuyệt đối và khả năng nhân rộng trên hàng trăm cuốn sách cổ.",
      },
    ],
  };

  AppState.gallery.results = s.gallery || [];
}

function renderAllComponents() {
  renderOverview();
  renderPipelineFlow();
  renderBenchmarks();
  populateBookSelects();
  loadInspectorPage();
}

/* ==========================================================================
   4. Render Tab 1: Tổng Quan (Overview)
   ========================================================================== */
function renderOverview() {
  if (!AppState.stats) return;
  const im = AppState.stats.impact_metrics;

  document.getElementById("kpiTotalChars").textContent = (im.total_characters || 111525).toLocaleString();
  document.getElementById("kpiGoldRate").textContent = `${im.gold_rate_overall || 80.6}%`;

  // Render danh sách 4 sách
  const listEl = document.getElementById("overviewBooksList");
  listEl.innerHTML = "";

  const books = AppState.stats.books || {};
  for (const [key, b] of Object.entries(books)) {
    const item = document.createElement("div");
    item.className = "book-stat-item";
    item.innerHTML = `
      <div class="book-stat-top">
        <span class="book-stat-title">${b.title}</span>
        <span class="book-stat-count">${b.total.toLocaleString()} ký tự</span>
      </div>
      <div class="book-stat-sub">${b.subtitle || b.layout} · ${b.gold.toLocaleString()} nhãn GOLD (${b.gold_pct}%)</div>
      <div class="progress-bar-wrap">
        <div class="progress-bar-fill" style="width: ${b.gold_pct}%"></div>
      </div>
    `;
    listEl.appendChild(item);
  }
}

/* ==========================================================================
   5. Render Tab 2: Quy Trình Xử Lý (Pipeline Flow)
   ========================================================================== */
let currentFlowStep = 1;

function renderPipelineFlow() {
  const steps = AppState.pipelineFlow;
  if (!steps || steps.length === 0) return;

  const navEl = document.getElementById("stepperNav");
  navEl.innerHTML = "";

  steps.forEach((s) => {
    const btn = document.createElement("button");
    btn.className = `step-nav-btn ${s.step === currentFlowStep ? "active" : ""}`;
    btn.innerHTML = `
      <div class="step-top">
        <span class="step-index-pill">Giai đoạn ${s.step}</span>
      </div>
      <div class="step-title-text">${s.name}</div>
      <div class="step-tag-text">${s.tag}</div>
    `;
    btn.addEventListener("click", () => {
      currentFlowStep = s.step;
      document.querySelectorAll(".step-nav-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderStepDetail(s);
    });
    navEl.appendChild(btn);
  });

  // Hiển thị chi tiết bước đầu tiên
  renderStepDetail(steps[currentFlowStep - 1]);
}

function renderStepDetail(s) {
  const detailEl = document.getElementById("stepDetailContainer");
  detailEl.innerHTML = `
    <div class="step-detail-head">
      <div class="step-main-title">
        <span>GIAI ĐOẠN ${s.step} / 6</span>
        <h3>${s.name} (${s.tag})</h3>
      </div>
      <span class="badge badge-neutral">Tự động theo quy tắc</span>
    </div>
    <div class="step-grid-info">
      <div class="info-box">
        <h4>Dữ liệu đầu vào</h4>
        <p>${s.input}</p>
      </div>
      <div class="info-box">
        <h4>Phương pháp & Mô hình áp dụng</h4>
        <p><strong>${s.model}</strong></p>
      </div>
      <div class="info-box" style="grid-column: span 2;">
        <h4>Nội dung và nguyên lý thực hiện</h4>
        <p style="white-space: pre-line;">${s.process}</p>
      </div>
      <div class="info-box">
        <h4>Kết quả đầu ra</h4>
        <p>${s.output}</p>
      </div>
      <div class="info-box">
        <h4>Chỉ tiêu kiểm soát & Đánh giá</h4>
        <p>${s.evidence}</p>
      </div>
    </div>
  `;
}

/* ==========================================================================
   6. Render Tab 3: Trình Soi Bản Thảo Trực Tiếp (Manuscript Inspector)
   ========================================================================== */
function populateBookSelects() {
  const bookSelect = document.getElementById("inspectorBookSelect");
  if (!bookSelect) return;

  bookSelect.innerHTML = "";
  AppState.books.forEach((b) => {
    const opt = document.createElement("option");
    opt.value = b.id;
    opt.textContent = `${b.title} (${b.layout === "prose" ? "Văn xuôi" : "Thơ"})`;
    bookSelect.appendChild(opt);
  });

  bookSelect.value = AppState.inspector.book;
  updatePageSelectOptions();
}

function updatePageSelectOptions() {
  const pageSelect = document.getElementById("inspectorPageSelect");
  const bookId = AppState.inspector.book;
  const currentBook = AppState.books.find((b) => b.id === bookId);

  pageSelect.innerHTML = "";
  const pages = currentBook?.available_pages || currentBook?.sample_pages || ["page_0002", "page_0003", "page_0004"];

  pages.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = `Trang ${p.replace("page_", "")}`;
    pageSelect.appendChild(opt);
  });

  if (pages.includes(AppState.inspector.page)) {
    pageSelect.value = AppState.inspector.page;
  } else if (pages.length > 0) {
    AppState.inspector.page = pages[0];
    pageSelect.value = pages[0];
  }
}

function initInspectorControls() {
  const bookSelect = document.getElementById("inspectorBookSelect");
  const pageSelect = document.getElementById("inspectorPageSelect");

  bookSelect.addEventListener("change", (e) => {
    AppState.inspector.book = e.target.value;
    updatePageSelectOptions();
    loadInspectorPage();
  });

  pageSelect.addEventListener("change", (e) => {
    AppState.inspector.page = e.target.value;
    loadInspectorPage();
  });

  // Bật/tắt filter bounding box
  document.getElementById("cbFilterGold").addEventListener("change", (e) => {
    AppState.inspector.filterGold = e.target.checked;
    filterBboxes();
  });
  document.getElementById("cbFilterSyl").addEventListener("change", (e) => {
    AppState.inspector.filterSyl = e.target.checked;
    filterBboxes();
  });
  document.getElementById("cbFilterTxt").addEventListener("change", (e) => {
    AppState.inspector.filterTxt = e.target.checked;
    filterBboxes();
  });

  // Zoom controls
  document.getElementById("btnZoomIn").addEventListener("click", () => {
    AppState.inspector.zoom = Math.min(AppState.inspector.zoom + 0.2, 2.4);
    applyZoom();
  });
  document.getElementById("btnZoomOut").addEventListener("click", () => {
    AppState.inspector.zoom = Math.max(AppState.inspector.zoom - 0.2, 0.6);
    applyZoom();
  });
  document.getElementById("btnZoomReset").addEventListener("click", () => {
    AppState.inspector.zoom = 1.0;
    applyZoom();
  });
}

function applyZoom() {
  const container = document.getElementById("scanContainer");
  container.style.transform = `scale(${AppState.inspector.zoom})`;
}

async function loadInspectorPage() {
  const { book, page } = AppState.inspector;
  const overlay = document.getElementById("bboxOverlay");
  const scanImg = document.getElementById("pageScanImage");
  const bookCfg = AppState.books?.find((b) => b.id === book);
  const bookTitle = bookCfg?.title || book;
  titleEl.textContent = `${bookTitle} — Trang ${page.replace("page_", "")}`;
  overlay.innerHTML = "";

  let pageData = null;

  if (AppState.isLiveServer) {
    try {
      const resp = await fetch(`/api/page?book=${book}&page=${page}`);
      if (resp.ok) {
        pageData = await resp.json();
      }
    } catch (e) {
      console.warn("Lỗi khi tải trang từ live API:", e);
    }
  }

  // Fallback sang sample_data nếu không có kết nối live
  if (!pageData && AppState.sampleData) {
    const sp = AppState.sampleData.sample_pages?.[book];
    if (sp) {
      pageData = {
        book: book,
        page: sp.page,
        has_scan: false,
        scan_url: null,
        dimensions: sp.dimensions,
        characters: sp.characters,
        tier_counts: {
          GOLD: sp.characters.filter((c) => c.tier === "GOLD").length,
          SYLLABLE: sp.characters.filter((c) => c.tier === "SYLLABLE").length,
          GOLD_text_only: sp.characters.filter((c) => c.tier === "GOLD_text_only").length,
        },
      };
    }
  }

  if (!pageData) {
    overlay.innerHTML = `<div class="skeleton-loader">Chưa có dữ liệu trang cho ${book} / ${page}</div>`;
    return;
  }

  AppState.inspector.chars = pageData.characters || [];

  // Cập nhật thống kê trang
  const tc = pageData.tier_counts || {};
  document.getElementById("pgTotalChars").textContent = pageData.characters.length;
  document.getElementById("pgGoldChars").textContent = tc.GOLD || 0;
  document.getElementById("pgSylChars").textContent = tc.SYLLABLE || 0;

  // Cập nhật ảnh scan
  if (pageData.scan_url) {
    scanImg.src = pageData.scan_url;
  } else {
    // Vẽ placeholder canvas giả lập trang scan nếu không có file ảnh thật
    scanImg.src = createPageCanvasPlaceholder(pageData.dimensions?.width || 1896, pageData.dimensions?.height || 3212);
  }

  // Render các Bounding Box
  renderBoundingBoxes(pageData.characters);
}

function createPageCanvasPlaceholder(w, h) {
  const canvas = document.createElement("canvas");
  canvas.width = 600;
  canvas.height = Math.round((h / w) * 600);
  const ctx = canvas.getContext("2d");

  // Nền giấy cổ
  ctx.fillStyle = "#1e1b18";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Đường viền khung trang
  ctx.strokeStyle = "#443d35";
  ctx.lineWidth = 4;
  ctx.strokeRect(30, 30, canvas.width - 60, canvas.height - 60);

  ctx.fillStyle = "#8c8273";
  ctx.font = "16px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Bản Quét Trang Sách Cổ", canvas.width / 2, canvas.height / 2);
  return canvas.toDataURL();
}

function renderBoundingBoxes(chars) {
  const overlay = document.getElementById("bboxOverlay");
  overlay.innerHTML = "";

  chars.forEach((c) => {
    const box = document.createElement("div");
    const tClass = c.tier.toLowerCase();
    box.className = `bbox-rect tier-${tClass}`;
    box.id = `bbox-${c.index}`;
    box.setAttribute("data-tier", c.tier);

    box.style.left = `${c.rect.left_pct}%`;
    box.style.top = `${c.rect.top_pct}%`;
    box.style.width = `${c.rect.width_pct}%`;
    box.style.height = `${c.rect.height_pct}%`;

    // Tooltip nổi khi di chuột
    const tip = document.createElement("div");
    tip.className = "bbox-tooltip";
    tip.textContent = `${c.label || c.ocr_char || "字"} · ${c.syllable || ""}`;
    box.appendChild(tip);

    // Sự kiện tương tác
    box.addEventListener("mouseenter", () => {
      selectCharacter(c, box);
    });

    box.addEventListener("click", () => {
      selectCharacter(c, box);
    });

    overlay.appendChild(box);
  });

  filterBboxes();

  // Mặc định chọn ký tự đầu tiên
  if (chars.length > 0) {
    selectCharacter(chars[0], document.getElementById(`bbox-${chars[0].index}`));
  }
}

function filterBboxes() {
  const { filterGold, filterSyl, filterTxt } = AppState.inspector;
  document.querySelectorAll(".bbox-rect").forEach((el) => {
    const tier = el.getAttribute("data-tier");
    let show = true;
    if (tier === "GOLD" && !filterGold) show = false;
    if (tier === "SYLLABLE" && !filterSyl) show = false;
    if (tier === "GOLD_text_only" && !filterTxt) show = false;
    el.classList.toggle("hidden", !show);
  });
}

function selectCharacter(c, boxEl) {
  AppState.inspector.selectedChar = c;

  // Cập nhật selected trên overlay
  document.querySelectorAll(".bbox-rect").forEach((b) => b.classList.remove("selected"));
  if (boxEl) boxEl.classList.add("selected");

  // Cập nhật sidebar chi tiết
  document.getElementById("previewEmpty").classList.add("hidden");
  document.getElementById("previewActive").classList.remove("hidden");

  document.getElementById("previewNomChar").textContent = c.label || c.ocr_char || "字";
  document.getElementById("previewSylText").textContent = c.syllable ? `/${c.syllable}/` : "(âm dị bản)";
  document.getElementById("previewUnicode").textContent = c.unicode || "—";
  document.getElementById("previewOcrChar").textContent = c.ocr_char || "—";
  document.getElementById("previewColOrder").textContent = `Cột ${c.column} · Ô thứ ${c.index}`;
  document.getElementById("previewRule").textContent = c.rule || "s1_inter_s2_direct";
  document.getElementById("previewBbox").textContent = JSON.stringify(c.bbox);

  const badge = document.getElementById("previewTierBadge");
  badge.textContent = c.tier;
  badge.className = `preview-tier-badge tier-${c.tier.toLowerCase()}`;

  // Ảnh crop
  const cropImg = document.getElementById("previewCropImg");
  if (c.crop_url) {
    cropImg.src = c.crop_url;
  } else if (c.crop_rel) {
    // Phục vụ đường dẫn tương đối trong dataset
    cropImg.src = `../dataset/${c.crop_rel}`;
  } else {
    cropImg.src = "";
  }
}

/* ==========================================================================
   7. Render Tab 4: Tra Cứu Ký Tự (Gallery & Search)
   ========================================================================== */
function initGalleryControls() {
  const searchInput = document.getElementById("gallerySearchInput");
  const bookFilter = document.getElementById("galleryBookFilter");
  const tierFilter = document.getElementById("galleryTierFilter");

  let debounceTimer = null;
  searchInput.addEventListener("input", (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      AppState.gallery.query = e.target.value.trim();
      executeSearch();
    }, 250);
  });

  bookFilter.addEventListener("change", (e) => {
    AppState.gallery.book = e.target.value;
    executeSearch();
  });

  tierFilter.addEventListener("change", (e) => {
    AppState.gallery.tier = e.target.value;
    executeSearch();
  });
}

async function executeSearch() {
  const { query, book, tier } = AppState.gallery;
  const gridEl = document.getElementById("galleryCardsGrid");
  const countEl = document.getElementById("galleryCountText");

  gridEl.innerHTML = `<div class="skeleton-loader" style="grid-column: 1 / -1;">Đang tra cứu cơ sở dữ liệu...</div>`;

  let results = [];

  if (AppState.isLiveServer) {
    try {
      const url = `/api/search?q=${encodeURIComponent(query)}&book=${book}&tier=${tier}&limit=80`;
      const resp = await fetch(url);
      if (resp.ok) {
        const data = await resp.json();
        results = data.results || [];
      }
    } catch (e) {
      console.warn("Lỗi khi tìm kiếm qua API:", e);
    }
  }

  // Fallback sang mẫu trong sampleData
  if (results.length === 0 && AppState.sampleData) {
    const list = AppState.sampleData.gallery || [];
    results = list.filter((item) => {
      if (book !== "all" && item.book !== book) return false;
      if (tier !== "all" && item.tier !== tier) return false;
      if (query) {
        const q = query.toLowerCase();
        const matchSyl = item.syllable?.toLowerCase().includes(q);
        const matchNom = item.label?.toLowerCase().includes(q);
        const matchUni = item.unicode?.toLowerCase().includes(q);
        if (!matchSyl && !matchNom && !matchUni) return false;
      }
      return true;
    });
  }

  countEl.textContent = `Tìm thấy ${results.length} mẫu ký tự phù hợp trong bộ ngữ liệu:`;
  gridEl.innerHTML = "";

  if (results.length === 0) {
    gridEl.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--text-dim);">
        Không tìm thấy mẫu ký tự nào khớp với từ khóa "<strong>${query}</strong>".
      </div>
    `;
    return;
  }

  results.forEach((item) => {
    const card = document.createElement("div");
    card.className = "char-card";
    const tClass = item.tier.toLowerCase();
    card.innerHTML = `
      <div class="char-card-top">
        <span class="char-tier-pill tier-${tClass}">${item.tier}</span>
        <span class="char-book-tag">${item.book_title || item.book}</span>
      </div>
      <div class="char-card-visual">
        <img class="char-crop-img" src="${item.crop_url || `../dataset/${item.crop_rel}`}" alt="crop" onerror="this.style.display='none'">
        <div class="char-nom-display">${item.label || item.ocr_char || "字"}</div>
      </div>
      <div class="char-card-info">
        <div class="char-syl-main">${item.syllable || "(âm dị thể)"}</div>
        <div class="char-uni-code">${item.unicode || "—"}</div>
        <div class="char-page-loc">${item.page || ""} · Cột ${item.column || ""}</div>
      </div>
    `;

    // Nhấp vào thẻ chuyển sang màn hình soi trang
    card.addEventListener("click", () => {
      AppState.inspector.book = item.book;
      AppState.inspector.page = item.page || "page_0002";
      switchTab("inspector");
      document.getElementById("inspectorBookSelect").value = item.book;
      updatePageSelectOptions();
      document.getElementById("inspectorPageSelect").value = item.page;
    });

    gridEl.appendChild(card);
  });
}

/* ==========================================================================
   8. Render Tab 5: Luận Chứng Khoa Học (Benchmarks)
   ========================================================================== */
function renderBenchmarks() {
  const bm = AppState.benchmarks;
  if (!bm || !bm.comparisons) return;

  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = "";

  bm.comparisons.forEach((c) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="metric-name"><strong>${c.metric}</strong></td>
      <td style="color: var(--text-dim);">${c.baseline}</td>
      <td style="color: var(--gold); font-weight: 700;">${c.proposed}</td>
      <td><span class="improvement-badge">${c.improvement}</span></td>
      <td style="font-size: 12.5px; color: var(--text-muted);">${c.impact}</td>
    `;
    tbody.appendChild(tr);
  });
}
