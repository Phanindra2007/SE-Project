/* ==========================================================================
   Banking Requirements & SDLC Explorer — Single-Page Controller
   ========================================================================== */

(function () {
  "use strict";

  const PRESET_VALUES = new Set([
    "Domestic Instant Fund Transfer",
    "Customer Notification and Alert Preferences",
    "Beneficiary Management and Step-Up Verification",
    "Account Statement and Audit Export",
  ]);

  const state = {
    overview: null,
    llm: null,
    documents: [],
    step1: null,
    step1Factors: null,
    step2: null,
    step2Markdown: "",
    isRunning: false,
    runningStage: null,
    timerInterval: null,
    startedAt: 0,
    reqFilter: "all",
    reqSearch: "",
    activeArtifact: "step1_json",
    artifactViewMode: "formatted",
    lastDurations: { step1Ms: null, step2Ms: null },
  };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatPct(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return "0%";
    return `${Number(num.toFixed(2))}%`;
  }

  function formatSignalList(rawSignals, maxItems = 3) {
    const items = String(rawSignals || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!items.length) return "";
    if (items.length <= maxItems) {
      return items.length === 1
        ? items[0]
        : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
    }
    const shown = items.slice(0, maxItems).join(", ");
    return `${shown}, and ${items.length - maxItems} more signals`;
  }

  function countReqIds(text) {
    const matches = String(text || "").match(/\bREQ-(?:NF|F)-\d{3}\b/g);
    return matches ? new Set(matches).size : 0;
  }

  function summarizeRationale(rawText) {
    const text = String(rawText || "").trim();
    if (!text) return "";

    const fitMatch = text.match(/Its fit in (.+?) is supported by ([^.]+)\./i);
    const tradeoffMatch = text.match(
      /Its trade-offs are (.+?)(?:, also present in ([^.]+))?\./i
    );
    const assumptionMatch = text.match(/Validate assumptions in ([^.]+)\./i);

    if (fitMatch) {
      const parts = [];
      const fitSignals = formatSignalList(fitMatch[1], 3);
      const fitCount = countReqIds(fitMatch[2]);
      parts.push(
        `Fits ${fitSignals}${fitCount ? ` across ${fitCount} requirement${fitCount === 1 ? "" : "s"}` : ""}.`
      );

      if (tradeoffMatch) {
        const tradeoffSignals = formatSignalList(tradeoffMatch[1], 3);
        parts.push(`Key trade-offs involve ${tradeoffSignals}.`);
      } else {
        parts.push("No major lifecycle trade-offs flagged for the detected signals.");
      }

      if (assumptionMatch) {
        const assumptionCount = countReqIds(assumptionMatch[1]);
        parts.push(
          assumptionCount
            ? `Requires validating assumptions in ${assumptionCount} requirement${assumptionCount === 1 ? "" : "s"}.`
            : "Requires validating flagged requirement assumptions."
        );
      }

      return parts.slice(0, 3).join(" ");
    }

    // Fallback for free-form LLM rationales: keep first 2-3 complete sentences
    const sentences = text.split(/(?<=[.!?])\s+(?=[A-Z0-9])/).filter(Boolean);
    if (sentences.length <= 3 && text.length <= 280) {
      return text;
    }
    const condensed = sentences.slice(0, 2).join(" ");
    return condensed.length > 280 ? `${condensed.slice(0, 277).trim()}...` : condensed;
  }

  function highlightRequirementIds(text) {
    return escapeHtml(text).replace(
      /\b(REQ-(?:NF|F)-\d{3})\b/g,
      '<span class="req-id-inline">$1</span>'
    );
  }

  function levelPillClass(level) {
    const norm = String(level || "").toLowerCase();
    if (norm === "high") return "pill-danger";
    if (norm === "medium") return "pill-warning";
    if (norm === "low") return "pill-success";
    return "pill-muted";
  }

  // ========================================================================
  // Alerts & Loading
  // ========================================================================

  function showAlert(type, title, message) {
    const el = $("#global-alert");
    el.className = `global-alert alert-${type}`;
    $("#global-alert-icon").textContent =
      type === "error" ? "!" : type === "warning" ? "⚠" : "✓";
    $("#global-alert-title").textContent = title;
    $("#global-alert-message").textContent = message;
  }

  function hideAlert() {
    $("#global-alert").classList.add("hidden");
  }

  function setLoading(isLoading, stageName = null) {
    state.isRunning = isLoading;
    state.runningStage = stageName;

    clearInterval(state.timerInterval);
    state.timerInterval = null;

    $("#step1-loading-row").classList.toggle("hidden", stageName !== "stage1");
    $("#step2-loading-row").classList.toggle("hidden", stageName !== "stage2");
    $("#card-step1").classList.toggle("is-running", stageName === "stage1");
    $("#card-step2").classList.toggle("is-running", stageName === "stage2");

    if (isLoading) {
      state.startedAt = performance.now();
      const timerEl = stageName === "stage1" ? $("#step1-timer") : $("#step2-timer");
      timerEl.textContent = "0.0s";
      state.timerInterval = setInterval(() => {
        const elapsed = ((performance.now() - state.startedAt) / 1000).toFixed(1);
        timerEl.textContent = `${elapsed}s`;
      }, 100);
    }

    syncWorkflowCards();
  }

  // ========================================================================
  // Functionality Selector Sync
  // ========================================================================

  function setFunctionalityValue(value) {
    const clean = String(value ?? "");
    $("#functionality-input").value = clean;
    $("#step1-target-preview").textContent = clean.trim() || "Enter target above";
    const select = $("#functionality-preset-select");
    select.value = PRESET_VALUES.has(clean.trim()) ? clean.trim() : "__custom__";
  }

  // ========================================================================
  // API Requests
  // ========================================================================

  async function apiRequest(url, options = {}) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      const detail =
        payload && (payload.detail || payload.message)
          ? payload.detail || payload.message
          : `Request failed with status ${response.status}`;
      throw new Error(detail);
    }
    return payload;
  }

  async function loadOverview() {
    try {
      const data = await apiRequest("/api/overview");
      state.overview = data.project;
      state.llm = data.llm;
      state.documents = data.documents || [];

      if (data.saved && data.saved.step1) {
        state.step1 = data.saved.step1;
        state.step1Factors = data.saved.step1_factors;
        const savedTarget =
          state.step1.functionality_profile?.target_functionality || "";
        if (savedTarget) {
          setFunctionalityValue(savedTarget);
        }
      } else {
        setFunctionalityValue($("#functionality-preset-select").value);
      }

      if (data.saved && data.saved.step2) {
        state.step2 = data.saved.step2;
        state.step2Markdown = data.saved.step2_markdown || "";
      }

      renderAll();
    } catch (err) {
      showAlert("error", "Failed to load application state", err.message);
    }
  }

  async function refreshHealth() {
    try {
      const data = await apiRequest("/api/health");
      state.llm = data.llm;
      renderLlmStatus();
    } catch (err) {
      showAlert("error", "Health check failed", err.message);
    }
  }

  async function handleRunStep1() {
    hideAlert();
    const functionality = $("#functionality-input").value.trim();
    if (!functionality) {
      showAlert(
        "warning",
        "Banking functionality required",
        "Select a preset or enter a target banking functionality before running Step 1."
      );
      $("#functionality-input").focus();
      return;
    }

    setLoading(true, "stage1");
    try {
      const data = await apiRequest("/api/step1", {
        method: "POST",
        body: JSON.stringify({ functionality }),
      });

      state.step1 = data.result;
      state.step1Factors = data.factors;
      state.step2 = null;
      state.step2Markdown = "";
      state.lastDurations.step1Ms = data.duration_ms;
      state.lastDurations.step2Ms = null;

      renderAll();

      const reqCount = data.result?.requirements?.length || 0;
      const warnText =
        data.warnings && data.warnings.length
          ? ` Note: ${data.warnings.join(" ")}`
          : "";
      showAlert(
        "success",
        "Step 1 Complete",
        `Generated ${reqCount} grounded requirements in ${(data.duration_ms / 1000).toFixed(1)}s.${warnText}`
      );
      $("#step1-results-block").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      showAlert("error", "Step 1 Failed", err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleRunStep2() {
    hideAlert();
    if (!state.step1) {
      showAlert(
        "warning",
        "Step 1 required first",
        "Run Step 1 (Generate Requirements) before selecting an SDLC model."
      );
      return;
    }

    const projectContext = $("#step2-context-input").value.trim() || null;
    setLoading(true, "stage2");

    try {
      const data = await apiRequest("/api/step2", {
        method: "POST",
        body: JSON.stringify({ project_context: projectContext }),
      });

      state.step2 = data.result;
      state.step1Factors = data.factors;
      state.step2Markdown = data.markdown || "";
      state.lastDurations.step2Ms = data.duration_ms;

      renderAll();

      showAlert(
        "success",
        "Step 2 Complete",
        `Recommended SDLC: ${data.result.recommended_sdlc} (completed in ${(data.duration_ms / 1000).toFixed(1)}s).`
      );
      $("#step2-results-block").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      showAlert("error", "Step 2 Failed", err.message);
    } finally {
      setLoading(false);
    }
  }

  // ========================================================================
  // Rendering
  // ========================================================================

  function renderAll() {
    renderLlmStatus();
    renderHeaderMeta();
    renderReferenceDocuments();
    syncWorkflowCards();
    renderResults();
  }

  function renderLlmStatus() {
    const llm = state.llm;
    if (!llm) return;

    const connected = Boolean(llm.reachable);
    const pill = $("#llm-status-pill");
    pill.classList.toggle("connected", connected);
    pill.classList.toggle("disconnected", !connected);
    $("#llm-status-text").textContent = connected
      ? `${llm.model} · Connected`
      : `${llm.model} · Offline`;
  }

  function renderHeaderMeta() {
    if (state.llm) {
      $("#meta-model-chip").textContent = `Model: ${state.llm.model} (temp ${state.llm.temperature})`;
    }
    const docs = state.documents || [];
    const totalRules = docs.reduce((sum, d) => sum + (d.rule_count || 0), 0);
    $("#meta-docs-chip").textContent = `Specs: ${docs.length} files · ${totalRules} rules`;

    const candidateCount = state.overview?.candidate_models?.length || 0;
    $("#meta-candidates-chip").textContent = `SDLC Models: ${candidateCount} candidates`;
  }

  function renderReferenceDocuments() {
    const container = $("#reference-docs-container");
    const docs = state.documents || [];
    if (!docs.length) {
      container.innerHTML = '<p class="muted-small">No Markdown specs found in data/banking/.</p>';
      return;
    }

    container.innerHTML = docs
      .map((doc) => {
        const rulesHtml = (doc.rules || [])
          .map((rule) => `<li>${escapeHtml(rule)}</li>`)
          .join("");
        return `
          <article class="doc-card">
            <div class="doc-card-head">
              <span class="doc-filename">${escapeHtml(doc.filename)}</span>
              <span class="pill pill-info">${doc.rule_count} rules</span>
            </div>
            <ul class="doc-rules-list">${rulesHtml}</ul>
          </article>
        `;
      })
      .join("");
  }

  function syncWorkflowCards() {
    const hasStep1 = Boolean(state.step1 && Array.isArray(state.step1.requirements));
    const hasStep2 = Boolean(state.step2);
    const reqCount = state.step1?.requirements?.length || 0;

    $("#btn-run-step1").disabled = state.isRunning;
    $("#btn-run-step2").disabled = state.isRunning || !hasStep1;

    const s1Pill = $("#step1-status-pill");
    if (state.runningStage === "stage1") {
      s1Pill.textContent = "Running";
      s1Pill.className = "pill pill-purple";
    } else if (hasStep1) {
      s1Pill.textContent = `Completed · ${reqCount} Reqs`;
      s1Pill.className = "pill pill-success";
    } else {
      s1Pill.textContent = "Ready";
      s1Pill.className = "pill pill-muted";
    }

    const s2Pill = $("#step2-status-pill");
    if (state.runningStage === "stage2") {
      s2Pill.textContent = "Running";
      s2Pill.className = "pill pill-purple";
    } else if (hasStep2) {
      s2Pill.textContent = `Selected · ${state.step2.recommended_sdlc}`;
      s2Pill.className = "pill pill-success";
    } else if (hasStep1) {
      s2Pill.textContent = "Ready";
      s2Pill.className = "pill pill-info";
    } else {
      s2Pill.textContent = "Requires Step 1";
      s2Pill.className = "pill pill-muted";
    }
  }

  function renderResults() {
    const hasStep1 = Boolean(state.step1);
    const hasStep2 = Boolean(state.step2);

    $("#results-empty-state").classList.toggle("hidden", hasStep1 || hasStep2);
    $("#step1-results-block").classList.toggle("hidden", !hasStep1);
    $("#step2-results-block").classList.toggle("hidden", !hasStep2);
    $("#artifacts-viewer-card").classList.toggle("hidden", !hasStep1 && !hasStep2);

    if (hasStep1) renderStep1Results();
    if (hasStep2) renderStep2Results();
    if (hasStep1 || hasStep2) renderArtifactViewer();
  }

  function renderStep1Results() {
    const data = state.step1;
    const profile = data.functionality_profile || {};
    const reqs = data.requirements || [];
    const funcReqs = reqs.filter((r) => r.type === "Functional");
    const nonFuncReqs = reqs.filter((r) => r.type === "Non-Functional");
    const assumptionReqs = reqs.filter((r) =>
      String(r.reasoning || "").toLowerCase().includes("[assumption]")
    );

    $("#profile-target-title").textContent = profile.target_functionality || "—";

    const critEl = $("#profile-criticality");
    critEl.textContent = profile.estimated_criticality || "—";
    critEl.className = `pill ${levelPillClass(profile.estimated_criticality)}`;

    const freqEl = $("#profile-change-freq");
    freqEl.textContent = profile.expected_change_frequency || "—";
    freqEl.className = `pill ${levelPillClass(profile.expected_change_frequency)}`;

    $("#profile-func-count").textContent = String(funcReqs.length);
    $("#profile-nonfunc-count").textContent = String(nonFuncReqs.length);
    $("#profile-assumption-count").textContent = String(assumptionReqs.length);

    $("#step1-timing-badge").textContent = state.lastDurations.step1Ms
      ? `Generated in ${(state.lastDurations.step1Ms / 1000).toFixed(1)}s`
      : "Loaded from outputs/";

    $("#count-all").textContent = String(reqs.length);
    $("#count-func").textContent = String(funcReqs.length);
    $("#count-nonfunc").textContent = String(nonFuncReqs.length);
    $("#count-assumption").textContent = String(assumptionReqs.length);

    renderFilteredRequirements(reqs);
  }

  function renderFilteredRequirements(reqs) {
    const query = state.reqSearch.trim().toLowerCase();
    const filtered = reqs.filter((item) => {
      if (state.reqFilter === "Functional" && item.type !== "Functional") {
        return false;
      }
      if (state.reqFilter === "Non-Functional" && item.type !== "Non-Functional") {
        return false;
      }
      if (
        state.reqFilter === "assumption" &&
        !String(item.reasoning || "").toLowerCase().includes("[assumption]")
      ) {
        return false;
      }
      if (query) {
        const hay = `${item.id} ${item.title} ${item.requirement} ${item.reasoning}`.toLowerCase();
        if (!hay.includes(query)) return false;
      }
      return true;
    });

    const container = $("#requirements-groups-container");
    if (!filtered.length) {
      container.innerHTML =
        '<div class="card empty-state"><p>No requirements match the current filter.</p></div>';
      return;
    }

    const functionalGroup = filtered.filter((r) => r.type === "Functional");
    const nonFunctionalGroup = filtered.filter((r) => r.type === "Non-Functional");

    const buildCard = (item) => {
      const hasAssumption = String(item.reasoning || "")
        .toLowerCase()
        .includes("[assumption]");
      const reasoningHtml = escapeHtml(item.reasoning).replace(
        /\[Assumption\]/gi,
        '<span class="assumption-tag">[Assumption]</span>'
      );
      const refsHtml = (item.source_references || [])
        .map((ref) => `<span class="ref-chip">${escapeHtml(ref)}</span>`)
        .join("");
      const typePill = item.type === "Functional" ? "pill-purple" : "pill-info";

      return `
        <article class="req-card">
          <div class="req-card-top">
            <span class="req-id">${escapeHtml(item.id)}</span>
            <span class="pill ${typePill}">${escapeHtml(item.type)}</span>
          </div>
          <h4 class="req-title">${escapeHtml(item.title)}</h4>
          <p class="req-statement">${escapeHtml(item.requirement)}</p>
          <div class="req-reasoning ${hasAssumption ? "has-assumption" : ""}">
            ${reasoningHtml}
          </div>
          <div class="req-refs">${refsHtml}</div>
        </article>
      `;
    };

    const groupsHtml = [];
    if (functionalGroup.length) {
      groupsHtml.push(`
        <div>
          <h3 class="req-group-title">
            Functional Requirements
            <span class="pill pill-purple">${functionalGroup.length}</span>
          </h3>
          <div class="req-grid">${functionalGroup.map(buildCard).join("")}</div>
        </div>
      `);
    }
    if (nonFunctionalGroup.length) {
      groupsHtml.push(`
        <div>
          <h3 class="req-group-title">
            Non-Functional Requirements
            <span class="pill pill-info">${nonFunctionalGroup.length}</span>
          </h3>
          <div class="req-grid">${nonFunctionalGroup.map(buildCard).join("")}</div>
        </div>
      `);
    }

    container.innerHTML = groupsHtml.join("");
  }

  function renderStep2Results() {
    const data = state.step2;
    const ranked = data.ranked_sdlc_options || [];
    const top = ranked[0] || {};
    const topPct = Math.min(100, Math.max(0, Number(top.percentage || 0)));
    const profile = state.step1?.functionality_profile || {};

    $("#step2-timing-badge").textContent = state.lastDurations.step2Ms
      ? `Evaluated in ${(state.lastDurations.step2Ms / 1000).toFixed(1)}s`
      : "Loaded from outputs/";

    $("#sdlc-recommended-name").textContent = data.recommended_sdlc || "—";
    $("#sdlc-recommended-sub").textContent = profile.target_functionality
      ? `Recommended for ${profile.target_functionality} (${profile.estimated_criticality} criticality · ${profile.expected_change_frequency} change frequency)`
      : "Highest-ranked lifecycle fit based on current requirements.";
    $("#sdlc-recommended-score").textContent = formatPct(topPct);
    $("#sdlc-recommended-bar").style.width = `${topPct}%`;

    $("#sdlc-justification-text").innerHTML = highlightRequirementIds(
      data.se_justification || ""
    );

    // SDLC Fit Chart
    $("#chart-sdlc-bars").innerHTML = ranked
      .map((opt, idx) => {
        const pct = Math.min(100, Math.max(0, Number(opt.percentage || 0)));
        return `
          <div class="bar-row">
            <div class="bar-row-head">
              <span class="bar-label">#${escapeHtml(opt.rank)} ${escapeHtml(opt.sdlc_model)}</span>
              <span class="bar-value">${formatPct(pct)}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill ${idx === 0 ? "top" : ""}" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
      })
      .join("");

    // Signal Breakdown Chart
    const counts = state.step1Factors?.category_counts || {};
    const signalIds = state.step1Factors?.signal_requirement_ids || {};
    const entries = Object.entries(counts)
      .filter(([, count]) => count > 0)
      .sort((a, b) => b[1] - a[1]);

    if (entries.length) {
      const maxCount = Math.max(...entries.map(([, c]) => c), 1);
      $("#chart-signals-bars").innerHTML = entries
        .map(([signal, count]) => {
          const width = Math.round((count / maxCount) * 100);
          const ids = (signalIds[signal] || []).join(", ");
          const label = signal.replace(/_/g, " ");
          return `
            <div class="bar-row" title="${escapeHtml(ids)}">
              <div class="bar-row-head">
                <span class="bar-label" style="text-transform: capitalize;">${escapeHtml(label)}</span>
                <span class="bar-value">${count} req${count === 1 ? "" : "s"}</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" style="width: ${width}%"></div>
              </div>
            </div>
          `;
        })
        .join("");
    } else {
      $("#chart-signals-bars").innerHTML =
        '<p class="muted-small">No requirement signal counts detected.</p>';
    }

    // Ranked Table with concise summary + expandable full rationale
    $("#ranked-count-badge").textContent = `${ranked.length} Models`;
    $("#sdlc-ranked-tbody").innerHTML = ranked
      .map((opt) => {
        const isTop = Number(opt.rank) === 1;
        const pct = Math.min(100, Math.max(0, Number(opt.percentage || 0)));
        const fullRationale = String(opt.rationale || "");
        const conciseRationale = summarizeRationale(fullRationale);
        return `
          <tr class="${isTop ? "rank-one-row" : ""}">
            <td><span class="rank-badge ${isTop ? "top" : ""}">${escapeHtml(opt.rank)}</span></td>
            <td><strong>${escapeHtml(opt.sdlc_model)}</strong></td>
            <td>
              <div class="fit-cell">
                <span>${formatPct(pct)}</span>
                <div class="progress-track">
                  <div class="progress-fill" style="width: ${pct}%"></div>
                </div>
              </div>
            </td>
            <td>
              <p class="rationale-summary">${escapeHtml(conciseRationale)}</p>
              <details class="rationale-details">
                <summary class="rationale-toggle">View Full Rationale</summary>
                <div class="rationale-full">${highlightRequirementIds(fullRationale)}</div>
              </details>
            </td>
          </tr>
        `;
      })
      .join("");

    // Risks & Mitigations
    const risks = data.key_risks_and_mitigations || [];
    $("#sdlc-risks-grid").innerHTML = risks
      .map(
        (item) => `
          <article class="risk-card">
            <div class="risk-block">
              <span class="risk-label risk">Risk</span>
              <div>${highlightRequirementIds(item.risk || "")}</div>
            </div>
            <div class="mitigation-block">
              <span class="risk-label mitigation">Mitigation</span>
              <div>${highlightRequirementIds(item.mitigation || "")}</div>
            </div>
          </article>
        `
      )
      .join("");
  }

  // ========================================================================
  // Artifact Viewer: Interactive JSON Tree & Markdown Report Renderer
  // ========================================================================

  function renderJsonPrimitive(value) {
    if (value === null) return '<span class="json-null">null</span>';
    if (typeof value === "boolean") {
      return `<span class="json-boolean">${value}</span>`;
    }
    if (typeof value === "number") {
      return `<span class="json-number">${value}</span>`;
    }
    return `<span class="json-string">"${escapeHtml(String(value))}"</span>`;
  }

  function renderJsonNode(value, keyLabel = null, isLast = true, depth = 0) {
    const comma = isLast ? "" : '<span class="json-punct">,</span>';
    const keyHtml =
      keyLabel !== null
        ? `<span class="json-key">"${escapeHtml(keyLabel)}"</span><span class="json-punct">: </span>`
        : "";

    if (value === null || typeof value !== "object") {
      return `<div class="json-leaf">${keyHtml}${renderJsonPrimitive(value)}${comma}</div>`;
    }

    const isArray = Array.isArray(value);
    const entries = isArray
      ? value.map((item, idx) => [null, item])
      : Object.entries(value);
    const openBrace = isArray ? "[" : "{";
    const closeBrace = isArray ? "]" : "}";

    if (entries.length === 0) {
      return `<div class="json-leaf">${keyHtml}<span class="json-punct">${openBrace}${closeBrace}</span>${comma}</div>`;
    }

    const countText = isArray
      ? `${entries.length} item${entries.length === 1 ? "" : "s"}`
      : `${entries.length} key${entries.length === 1 ? "" : "s"}`;
    const openAttr = depth <= 1 ? " open" : "";

    const childrenHtml = entries
      .map(([childKey, childVal], index) =>
        renderJsonNode(
          childVal,
          isArray ? null : childKey,
          index === entries.length - 1,
          depth + 1
        )
      )
      .join("");

    return `
      <details class="json-node"${openAttr}>
        <summary class="json-summary">
          <span class="json-caret" aria-hidden="true">▸</span>
          ${keyHtml}<span class="json-punct">${openBrace}</span>
          <span class="json-meta">${countText}</span>
        </summary>
        <div class="json-children">${childrenHtml}</div>
        <div class="json-leaf"><span class="json-punct">${closeBrace}</span>${comma}</div>
      </details>
    `;
  }

  function splitMarkdownTableRow(rowLine) {
    let line = rowLine.trim();
    if (line.startsWith("|")) line = line.slice(1);
    if (line.endsWith("|") && !line.endsWith("\\|")) line = line.slice(0, -1);

    const cells = [];
    let current = "";
    for (let i = 0; i < line.length; i += 1) {
      if (line[i] === "\\" && line[i + 1] === "|") {
        current += "|";
        i += 1;
      } else if (line[i] === "|") {
        cells.push(current.trim());
        current = "";
      } else {
        current += line[i];
      }
    }
    cells.push(current.trim());
    return cells;
  }

  function formatMarkdownInline(text) {
    const codeTokens = [];
    let working = String(text || "").replace(/`([^`]+)`/g, (_, code) => {
      const token = `\u0000CODE${codeTokens.length}\u0000`;
      codeTokens.push(`<code>${escapeHtml(code)}</code>`);
      return token;
    });

    working = escapeHtml(working);
    working = working.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    working = working.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    working = working.replace(
      /\b(REQ-(?:NF|F)-\d{3})\b/g,
      '<span class="req-id-inline">$1</span>'
    );

    return working.replace(/\u0000CODE(\d+)\u0000/g, (_, idx) => codeTokens[Number(idx)]);
  }

  function renderMarkdownDocument(mdText) {
    const lines = String(mdText || "").replace(/\r\n/g, "\n").split("\n");
    const blocks = [];
    let i = 0;

    const isTableSeparator = (line) =>
      /^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$/.test(line.trim());

    while (i < lines.length) {
      const line = lines[i];
      const trimmed = line.trim();

      if (!trimmed) {
        i += 1;
        continue;
      }

      // Fenced code block
      if (trimmed.startsWith("```")) {
        const codeLines = [];
        i += 1;
        while (i < lines.length && !lines[i].trim().startsWith("```")) {
          codeLines.push(lines[i]);
          i += 1;
        }
        i += 1;
        blocks.push(
          `<pre class="md-code-block"><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`
        );
        continue;
      }

      // Horizontal rule
      if (/^(?:-{3,}|\*{3,})$/.test(trimmed)) {
        blocks.push("<hr />");
        i += 1;
        continue;
      }

      // Headings
      const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
      if (headingMatch) {
        const level = headingMatch[1].length;
        blocks.push(`<h${level}>${formatMarkdownInline(headingMatch[2])}</h${level}>`);
        i += 1;
        continue;
      }

      // GFM Pipe Table
      if (
        trimmed.includes("|") &&
        i + 1 < lines.length &&
        isTableSeparator(lines[i + 1])
      ) {
        const headers = splitMarkdownTableRow(lines[i]);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim().includes("|") && lines[i].trim() !== "") {
          rows.push(splitMarkdownTableRow(lines[i]));
          i += 1;
        }
        const thead = `<thead><tr>${headers
          .map((h) => `<th>${formatMarkdownInline(h)}</th>`)
          .join("")}</tr></thead>`;
        const tbody = `<tbody>${rows
          .map(
            (row) =>
              `<tr>${headers
                .map((_, colIdx) => `<td>${formatMarkdownInline(row[colIdx] || "")}</td>`)
                .join("")}</tr>`
          )
          .join("")}</tbody>`;
        blocks.push(`<div class="md-table-wrap"><table class="md-table">${thead}${tbody}</table></div>`);
        continue;
      }

      // Blockquote
      if (trimmed.startsWith("> ")) {
        const quoteLines = [];
        while (i < lines.length && lines[i].trim().startsWith("> ")) {
          quoteLines.push(lines[i].trim().slice(2));
          i += 1;
        }
        blocks.push(`<blockquote>${formatMarkdownInline(quoteLines.join(" "))}</blockquote>`);
        continue;
      }

      // Unordered or Ordered List (with nested sub-bullets)
      const ulMatch = line.match(/^(\s*)[-*]\s+(.+)$/);
      const olMatch = line.match(/^(\s*)\d+\.\s+(.+)$/);
      if (ulMatch || olMatch) {
        const isOrdered = Boolean(olMatch);
        const tag = isOrdered ? "ol" : "ul";
        const baseIndent = (ulMatch || olMatch)[1].length;
        const items = [];

        while (i < lines.length) {
          const cur = lines[i];
          const curUl = cur.match(/^(\s*)[-*]\s+(.+)$/);
          const curOl = cur.match(/^(\s*)\d+\.\s+(.+)$/);
          const match = isOrdered ? curOl : curUl;
          if (!match) break;

          const indent = match[1].length;
          if (indent > baseIndent && items.length > 0) {
            const subItems = [];
            while (i < lines.length) {
              const subCur = lines[i];
              const subMatch = subCur.match(/^(\s*)(?:[-*]|\d+\.)\s+(.+)$/);
              if (!subMatch || subMatch[1].length <= baseIndent) break;
              subItems.push(`<li>${formatMarkdownInline(subMatch[2])}</li>`);
              i += 1;
            }
            items[items.length - 1] += `<ul>${subItems.join("")}</ul>`;
          } else if (indent < baseIndent) {
            break;
          } else {
            items.push(`<li>${formatMarkdownInline(match[2])}</li>`);
            i += 1;
          }
        }
        blocks.push(`<${tag}>${items.map((li) => `${li}`).join("")}</${tag}>`);
        continue;
      }

      // Paragraph
      const paraLines = [];
      while (
        i < lines.length &&
        lines[i].trim() !== "" &&
        !lines[i].trim().startsWith("#") &&
        !lines[i].trim().startsWith("```") &&
        !lines[i].trim().startsWith("> ") &&
        !/^(?:-{3,}|\*{3,})$/.test(lines[i].trim()) &&
        !(lines[i].trim().includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) &&
        !/^(\s*)(?:[-*]|\d+\.)\s+/.test(lines[i])
      ) {
        paraLines.push(lines[i].trim());
        i += 1;
      }
      blocks.push(`<p>${formatMarkdownInline(paraLines.join(" "))}</p>`);
    }

    return `<article class="markdown-report">${blocks.join("")}</article>`;
  }

  function getActiveArtifactContent() {
    if (state.activeArtifact === "step1_json") {
      return {
        kind: "json",
        filename: "step1_requirements.json",
        data: state.step1,
        text: state.step1
          ? JSON.stringify(state.step1, null, 2)
          : "// Step 1 has not been executed yet.",
        available: Boolean(state.step1),
      };
    }
    if (state.activeArtifact === "step2_json") {
      return {
        kind: "json",
        filename: "step2_sdlc_justification.json",
        data: state.step2,
        text: state.step2
          ? JSON.stringify(state.step2, null, 2)
          : "// Step 2 has not been executed yet.",
        available: Boolean(state.step2),
      };
    }
    return {
      kind: "markdown",
      filename: "step2_sdlc_justification.md",
      data: state.step2Markdown,
      text: state.step2Markdown
        ? state.step2Markdown
        : "# Step 2 Markdown report has not been generated yet.",
      available: Boolean(state.step2Markdown),
    };
  }

  function renderArtifactSummaryStrip(artifactKey, data) {
    const strip = $("#artifact-summary-strip");
    if (!data || state.artifactViewMode !== "formatted") {
      strip.classList.add("hidden");
      strip.innerHTML = "";
      return;
    }

    let items = [];
    if (artifactKey === "step1_json" && typeof data === "object") {
      const profile = data.functionality_profile || {};
      const reqs = Array.isArray(data.requirements) ? data.requirements : [];
      items = [
        { label: "Target Functionality", value: profile.target_functionality || "—" },
        { label: "Criticality", value: profile.estimated_criticality || "—" },
        { label: "Change Frequency", value: profile.expected_change_frequency || "—" },
        { label: "Requirements", value: `${reqs.length} items` },
      ];
    } else if (artifactKey === "step2_json" && typeof data === "object") {
      const ranked = Array.isArray(data.ranked_sdlc_options) ? data.ranked_sdlc_options : [];
      const top = ranked[0] || {};
      const risks = Array.isArray(data.key_risks_and_mitigations) ? data.key_risks_and_mitigations : [];
      items = [
        { label: "Recommended SDLC", value: data.recommended_sdlc || "—" },
        { label: "Top Fit Score", value: top.percentage != null ? formatPct(top.percentage) : "—" },
        { label: "Candidates Ranked", value: `${ranked.length} models` },
        { label: "Risks & Mitigations", value: `${risks.length} items` },
      ];
    }

    if (!items.length) {
      strip.classList.add("hidden");
      strip.innerHTML = "";
      return;
    }

    strip.classList.remove("hidden");
    strip.innerHTML = items
      .map(
        (item) => `
          <div class="av-summary-item">
            <span class="av-summary-label">${escapeHtml(item.label)}</span>
            <span class="av-summary-val" title="${escapeHtml(item.value)}">${escapeHtml(item.value)}</span>
          </div>
        `
      )
      .join("");
  }

  function renderArtifactViewer() {
    // Update tab active/availability state
    const availMap = {
      step1_json: Boolean(state.step1),
      step2_json: Boolean(state.step2),
      step2_md: Boolean(state.step2Markdown),
    };
    $$(".artifact-tab").forEach((tab) => {
      const key = tab.dataset.artifact;
      tab.classList.toggle("active", key === state.activeArtifact);
      tab.classList.toggle("unavailable", !availMap[key]);
    });

    const artifact = getActiveArtifactContent();
    const isJson = artifact.kind === "json";
    const isFormatted = state.artifactViewMode === "formatted";

    $("#json-tree-controls").classList.toggle(
      "hidden",
      !(isJson && isFormatted && artifact.available)
    );
    $("#copy-artifact-btn").textContent = isJson ? "Copy JSON" : "Copy Markdown";

    const dlBtn = $("#download-artifact-btn");
    if (artifact.available) {
      dlBtn.href = `/api/results/download/${artifact.filename}`;
      dlBtn.classList.remove("hidden");
    } else {
      dlBtn.href = "#";
      dlBtn.classList.add("hidden");
    }

    const formattedEl = $("#artifact-formatted-view");
    const rawEl = $("#artifact-raw-view");
    rawEl.querySelector("code").textContent = artifact.text;

    if (!isFormatted || !artifact.available) {
      renderArtifactSummaryStrip(state.activeArtifact, null);
      formattedEl.classList.add("hidden");
      rawEl.classList.remove("hidden");
      return;
    }

    rawEl.classList.add("hidden");
    formattedEl.classList.remove("hidden");

    if (isJson) {
      renderArtifactSummaryStrip(state.activeArtifact, artifact.data);
      formattedEl.innerHTML = `<div class="json-tree">${renderJsonNode(artifact.data)}</div>`;
    } else {
      renderArtifactSummaryStrip(state.activeArtifact, null);
      formattedEl.innerHTML = renderMarkdownDocument(artifact.text);
    }
  }

  // ========================================================================
  // Event Wiring
  // ========================================================================

  function bindEvents() {
    $("#global-alert-close").addEventListener("click", hideAlert);
    $("#refresh-health-btn").addEventListener("click", refreshHealth);

    $("#functionality-preset-select").addEventListener("change", (e) => {
      const val = e.target.value;
      if (val === "__custom__") {
        $("#functionality-input").value = "";
        $("#step1-target-preview").textContent = "Enter target above";
        $("#functionality-input").focus();
      } else {
        setFunctionalityValue(val);
      }
    });

    $("#functionality-input").addEventListener("input", (e) => {
      setFunctionalityValue(e.target.value);
    });

    $("#btn-run-step1").addEventListener("click", handleRunStep1);
    $("#btn-run-step2").addEventListener("click", handleRunStep2);

    $$(".filter-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        $$(".filter-pill").forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        state.reqFilter = pill.dataset.filter || "all";
        if (state.step1?.requirements) {
          renderFilteredRequirements(state.step1.requirements);
        }
      });
    });

    $("#req-search-input").addEventListener("input", (e) => {
      state.reqSearch = e.target.value || "";
      if (state.step1?.requirements) {
        renderFilteredRequirements(state.step1.requirements);
      }
    });

    $$(".artifact-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        state.activeArtifact = tab.dataset.artifact || "step1_json";
        renderArtifactViewer();
      });
    });

    $$(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.artifactViewMode = btn.dataset.viewMode || "formatted";
        renderArtifactViewer();
      });
    });

    $("#json-expand-all-btn").addEventListener("click", () => {
      $$("#artifact-formatted-view .json-node").forEach((node) => {
        node.open = true;
      });
    });

    $("#json-collapse-all-btn").addEventListener("click", () => {
      $$("#artifact-formatted-view .json-node").forEach((node, idx) => {
        node.open = idx === 0;
      });
    });

    $("#copy-artifact-btn").addEventListener("click", async () => {
      const { kind, text } = getActiveArtifactContent();
      const btn = $("#copy-artifact-btn");
      const defaultLabel = kind === "json" ? "Copy JSON" : "Copy Markdown";
      let copied = false;

      if (navigator.clipboard && navigator.clipboard.writeText) {
        try {
          await navigator.clipboard.writeText(text);
          copied = true;
        } catch {
          copied = false;
        }
      }

      if (!copied) {
        const helper = document.createElement("textarea");
        helper.value = text;
        helper.setAttribute("readonly", "");
        helper.style.position = "fixed";
        helper.style.opacity = "0";
        document.body.appendChild(helper);
        helper.select();
        try {
          copied = document.execCommand("copy");
        } catch {
          copied = false;
        }
        document.body.removeChild(helper);
      }

      if (copied) {
        btn.textContent = "Copied!";
        setTimeout(() => {
          btn.textContent = defaultLabel;
        }, 1400);
      } else {
        showAlert("warning", "Copy failed", "Clipboard access is unavailable.");
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    loadOverview();
  });
})();
