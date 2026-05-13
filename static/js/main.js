/**
 * JobGuard AI — main.js
 * Client-side logic for the Fake Job Description Detector
 */

(function () {
  "use strict";

  // ── DOM refs ──────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const form = $("#predict-form");
  const btnAnalyze = $("#btn-analyze");
  const resultPanel = $("#result-panel");
  const resultPlaceholder = $("#result-placeholder");
  const resultContent = $("#result-content");
  const scanOverlay = $("#scan-overlay");
  const typewriterEl = $("#typewriter");

  // Batch
  const dropZone = $("#drop-zone");
  const fileInput = $("#file-input");
  const fileNameEl = $("#file-name");
  const btnBatch = $("#btn-batch-analyze");
  const batchWrapper = $("#batch-results-wrapper");
  const batchTbody = $("#batch-tbody");
  const paginationEl = $("#pagination");

  // ── Typewriter ────────────────────────────────────────────
  const TAGLINE = "Detecting fraudulent job postings with AI";
  let twIdx = 0;
  function typewrite() {
    if (twIdx <= TAGLINE.length) {
      typewriterEl.textContent = TAGLINE.slice(0, twIdx);
      twIdx++;
      setTimeout(typewrite, 55);
    } else {
      $("#cursor").style.animation = "blink 0.8s step-end infinite";
    }
  }
  typewrite();

  // ── Tabs ──────────────────────────────────────────────────
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.remove("active"));
      $$(".tab-content").forEach((c) => c.classList.remove("active"));
      tab.classList.add("active");
      $(`#content-${tab.dataset.tab}`).classList.add("active");
    });
  });

  // ── Character counters ────────────────────────────────────
  $$("textarea[data-counter]").forEach((ta) => {
    const id = ta.id.replace("input-", "");
    const counter = $(`#counter-${id}`);
    if (!counter) return;
    ta.addEventListener("input", () => {
      counter.textContent = ta.value.length;
    });
  });

  // ── Gauge helpers ─────────────────────────────────────────
  const ARC_LENGTH = 251.2; // circumference of the SVG arc
  function setGauge(pct, isFake) {
    const fill = $("#gauge-fill");
    const valEl = $("#gauge-value");
    const offset = ARC_LENGTH - (ARC_LENGTH * pct) / 100;

    fill.classList.toggle("danger", isFake);
    fill.style.strokeDashoffset = ARC_LENGTH; // reset
    valEl.textContent = "0%";

    requestAnimationFrame(() => {
      setTimeout(() => {
        fill.style.strokeDashoffset = offset;
        animateNumber(valEl, 0, Math.round(pct), 600);
      }, 50);
    });
  }

  function animateNumber(el, from, to, duration) {
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const val = Math.round(from + (to - from) * easeOut(progress));
      el.textContent = val + "%";
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }
  function easeOut(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  // ── Probability bars ──────────────────────────────────────
  function setProbBars(xgb, cb) {
    const barXgb = $("#prob-bar-xgb");
    const barCb = $("#prob-bar-cb");
    const valXgb = $("#prob-val-xgb");
    const valCb = $("#prob-val-cb");
    barXgb.style.width = "0";
    barCb.style.width = "0";
    valXgb.textContent = "0%";
    valCb.textContent = "0%";
    requestAnimationFrame(() => {
      setTimeout(() => {
        barXgb.style.width = (xgb * 100).toFixed(1) + "%";
        barCb.style.width = (cb * 100).toFixed(1) + "%";
        valXgb.textContent = (xgb * 100).toFixed(1) + "%";
        valCb.textContent = (cb * 100).toFixed(1) + "%";
      }, 100);
    });
  }

  // ── Render Result ─────────────────────────────────────────
  function renderResult(data) {
    const isFake = data.prediction === "FAKE";

    // Verdict
    const badge = $("#verdict-badge");
    badge.className = "verdict-badge " + (isFake ? "fake" : "real");
    $("#verdict-icon").className = "fas " + (isFake ? "fa-skull-crossbones" : "fa-shield-halved") + " verdict-icon";
    $("#verdict-text").textContent = data.prediction;

    // Panel glow
    resultPanel.classList.remove("glow-cyan", "glow-red");
    resultPanel.classList.add(isFake ? "glow-red" : "glow-cyan");

    // Gauge
    setGauge(data.confidence, isFake);

    // Prob bars
    setProbBars(data.xgb_prob, data.cb_prob);

    // Risk
    const riskBadge = $("#risk-badge");
    riskBadge.className = "risk-badge " + data.risk_level.toLowerCase();
    $("#risk-text").textContent = data.risk_level;

    // Red flags
    const flagsSec = $("#flags-section");
    const flagsList = $("#flags-list");
    flagsList.innerHTML = "";
    if (data.red_flags && data.red_flags.length) {
      flagsSec.style.display = "block";
      data.red_flags.forEach((f) => {
        const li = document.createElement("li");
        li.innerHTML = `<i class="fas fa-triangle-exclamation"></i> ${escapeHtml(f)}`;
        flagsList.appendChild(li);
      });
    } else {
      flagsSec.style.display = "none";
    }

    // Footer
    $("#model-used-text").textContent = "Analyzed with: " + data.model_used;

    // Show
    resultPlaceholder.style.display = "none";
    resultContent.style.display = "block";
  }

  // ── Predict (single) ─────────────────────────────────────
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const desc = $("#input-description").value.trim();
    if (!desc) {
      $("#input-description").focus();
      $("#input-description").style.borderColor = "var(--red)";
      setTimeout(() => ($("#input-description").style.borderColor = ""), 1500);
      return;
    }

    const payload = {
      title: $("#input-title").value,
      company_profile: $("#input-company").value,
      description: desc,
      requirements: $("#input-requirements").value,
      benefits: $("#input-benefits").value,
      employment_type: $("#select-employment").value,
      required_experience: $("#select-experience").value,
      required_education: $("#select-education").value,
      has_company_logo: $("#toggle-logo").checked,
      has_questions: $("#toggle-questions").checked,
      model_choice: document.querySelector('input[name="model"]:checked').value,
    };

    // UI: scanning
    btnAnalyze.classList.add("scanning");
    btnAnalyze.disabled = true;
    scanOverlay.classList.add("active");
    resultPlaceholder.style.display = "none";
    resultContent.style.display = "none";

    console.log("[JobGuard] Sending prediction request:", payload);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      console.log("[JobGuard] Prediction response:", data);

      if (data.error) {
        alert("Error: " + data.error);
        return;
      }

      // Short delay for scanning effect
      setTimeout(() => {
        scanOverlay.classList.remove("active");
        renderResult(data);
      }, 800);
    } catch (err) {
      console.error("[JobGuard] Request failed:", err);
      alert("Request failed. Is the server running?");
      scanOverlay.classList.remove("active");
    } finally {
      setTimeout(() => {
        btnAnalyze.classList.remove("scanning");
        btnAnalyze.disabled = false;
      }, 900);
    }
  });

  // ── Batch Upload ──────────────────────────────────────────
  let batchFile = null;

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.name.endsWith(".csv")) {
      alert("Please upload a .csv file");
      return;
    }
    batchFile = file;
    fileNameEl.textContent = file.name;
    btnBatch.disabled = false;
    console.log("[JobGuard] File selected:", file.name);
  }

  let batchResults = [];
  const ROWS_PER_PAGE = 10;
  let currentPage = 1;

  btnBatch.addEventListener("click", async () => {
    if (!batchFile) return;
    btnBatch.classList.add("scanning");
    btnBatch.disabled = true;

    const fd = new FormData();
    fd.append("file", batchFile);

    console.log("[JobGuard] Uploading batch file:", batchFile.name);

    try {
      const res = await fetch("/batch", { method: "POST", body: fd });
      const data = await res.json();
      console.log("[JobGuard] Batch response:", data);

      if (data.error) {
        alert("Error: " + data.error);
        return;
      }

      batchResults = data;
      currentPage = 1;
      renderBatchTable();
      batchWrapper.style.display = "block";
    } catch (err) {
      console.error("[JobGuard] Batch request failed:", err);
      alert("Batch request failed. Is the server running?");
    } finally {
      btnBatch.classList.remove("scanning");
      btnBatch.disabled = false;
    }
  });

  function renderBatchTable() {
    const totalPages = Math.ceil(batchResults.length / ROWS_PER_PAGE);
    const start = (currentPage - 1) * ROWS_PER_PAGE;
    const page = batchResults.slice(start, start + ROWS_PER_PAGE);

    batchTbody.innerHTML = "";
    page.forEach((r) => {
      const isFake = r.prediction === "FAKE";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.row_index}</td>
        <td>${escapeHtml((r.title || "—").substring(0, 40))}</td>
        <td style="color:${isFake ? "var(--red)" : "var(--cyan)"}; font-weight:700">${r.prediction || "ERR"}</td>
        <td>${r.confidence != null ? r.confidence + "%" : "—"}</td>
        <td><span class="risk-badge ${(r.risk_level || "").toLowerCase()}">${r.risk_level || "—"}</span></td>
        <td>${r.red_flags ? r.red_flags.length : 0}</td>
      `;
      batchTbody.appendChild(tr);
    });

    paginationEl.innerHTML = "";
    for (let i = 1; i <= totalPages; i++) {
      const btn = document.createElement("button");
      btn.className = "page-btn" + (i === currentPage ? " active" : "");
      btn.textContent = i;
      btn.addEventListener("click", () => { currentPage = i; renderBatchTable(); });
      paginationEl.appendChild(btn);
    }
  }

  // ── Utils ─────────────────────────────────────────────────
  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  console.log("[JobGuard] Frontend initialized.");
})();
