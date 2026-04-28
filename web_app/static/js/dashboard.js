// ======================================================================
// DASHBOARD PAGE (ES MODULE VERSION)
// Aligns with monitor.js (SocketCore + GlobalState)
// Uses: ./core/socket_core.js, ./core/utils.js, ./core/ui_core.js
// ======================================================================

import SocketCore from "./core/socket_core.js";
import { smoothNumber, fadeIn, playAlertTone } from "./core/utils.js";
import { showError, showSuccess } from "./core/ui_core.js";

console.log("%c[DASHBOARD] Loaded (ES Module)", "color:#3b8cf0;font-weight:bold;");

/**
 * INIDS Demo Platform - Dashboard Controller
 * Handles module cards + modal loading + demo/refresh controls
 * AND (new): real-time socket wiring consistent with monitor.js
 */
class DashboardController {
  constructor() {
    // ---- Module system (preserved) ----------------------------------
    this.moduleModal = null;
    this.currentModule = null;

    this.moduleRegistry = {
      "real-time-detection": {
        title: "Real-Time Detection Panel",
        route: "/modules/real-time-detection",
        description: "Live event stream showing real-time threat detection",
      },
      "multi-engine": {
        title: "Multi-Engine Voting System",
        route: "/modules/multi-engine",
        description: "Five detection engines voting on verdict consensus",
      },
      "risk-score": {
        title: "Risk Score Visualizer",
        route: "/modules/risk-score",
        description: "Multi-factor risk calculation with animated gauges",
      },
      "auto-blocking": {
        title: "Automated Blocking",
        route: "/modules/auto-blocking",
        description: "Detection to firewall block execution timeline",
      },
      "approval-workflow": {
        title: "Approval Workflow",
        route: "/modules/approval-workflow",
        description: "Human-in-the-loop review process",
      },
      "false-positive": {
        title: "False Positive Learning",
        route: "/modules/false-positive",
        description: "Feedback-driven system learning",
      },
      "threat-intel": {
        title: "Threat Intelligence Enrichment",
        route: "/modules/threat-intel",
        description: "External reputation checks and badging",
      },
      "anomaly-learning": {
        title: "Anomaly Learning Engine",
        route: "/modules/anomaly-learning",
        description: "Behavioral baseline and deviation detection",
      },
      analytics: {
        title: "Analytics Dashboard",
        route: "/modules/analytics",
        description: "Security posture metrics and insights",
      },
      escalation: {
        title: "Escalation State Machine",
        route: "/modules/escalation",
        description: "Per-IP escalation progression",
      },
      "pipeline-monitor": {
        title: "Pipeline Monitor",
        route: "/modules/pipeline-monitor",
        description: "Ingestion throughput and health metrics",
      },
      "policy-tuning": {
        title: "Policy Tuning Simulator",
        route: "/modules/policy-tuning",
        description: "Interactive policy parameter adjustment",
      },
      "alert-lifecycle": {
        title: "Alert Lifecycle Manager",
        route: "/modules/alert-lifecycle",
        description: "SOC workflow in Kanban board",
      },
      "engine-playground": {
        title: "Engine Toggle Playground",
        route: "/modules/engine-playground",
        description: "Disable engines to see coverage impact",
      },
      "pattern-detector": {
        title: "Behavioral Pattern Detector",
        route: "/modules/pattern-detector",
        description: "Network graph attack pattern visualization",
      },
    };

    // ---- Controls (preserved) ---------------------------------------
    this.demoMode = false;

    // ---- Init --------------------------------------------------------
    this.cacheDOM();
    this.setupModal();
    this.attachCardListeners();
    this.attachControlListeners();

    // (new) State + socket wiring
    this.subscribeToState();
    this.attachRealtimeHandlers();
    this.wireBlockButtons();

    // Preserve original behavior
    this.loadInitialMetrics();
    this.animateSparklines();
  }

  // ====================================================================
  // DOM CACHE (new)
  // - Uses resilient selectors to avoid fragile nth-child dependencies
  // ====================================================================
  cacheDOM() {
    const q = (sel) => document.querySelector(sel);

    this.el = {
      // status strip
      statusStrip: q(".status-strip"),
      ingestedValue: q(".status-strip .status-cell:nth-child(1) .status-cell-value"),
      processedValue: q(".status-strip .status-cell:nth-child(2) .status-cell-value"),
      alertsValue: q(".status-strip .status-cell.alerts .status-cell-value"),
      blockedValue: q(".status-strip .status-cell.blocked .status-cell-value"),

      // global badge (NORMAL / UNDER ATTACK)
      globalStatusBadge: q(".panel-tag"),

      // topbar
      liveDot: q(".pulse-dot"),
      topbarTimestamp: q(".topbar-meta span:nth-child(2)"),

      // threat intelligence panels
      alertsPanel: q(".alert-panel-active"),
      alertsTableBody: q(".alert-panel-active .data-table tbody"),
      alertsTag: q(".alert-panel-active .panel-tag"),

      // prevention actions table (best-effort)
      actionsTableBody: null,

      // operations: active blocks + audit
      blocksTableBody: null,
      blocksTag: null,
      auditTableBody: null,

      // timeline
      timelineWrap: null,

      // reconcile strip
      recDbActive: q(".reconcile-strip .rec-cell:nth-child(1) .rec-val"),
      recFirewallRules: q(".reconcile-strip .rec-cell:nth-child(2) .rec-val"),
      recMissingFW: q(".reconcile-strip .rec-cell:nth-child(3) .rec-val"),
      recOrphanFW: q(".reconcile-strip .rec-cell:nth-child(4) .rec-val"),

      // sidebar counters
      sidebarRequests: null,
      sidebarAlerts: null,
      sidebarActions: null,
      sidebarRateLtd: null,
      sidebarUnauth: null,
    };

    // Resolve table bodies by panel titles inside grids (dashboard.html structure)
    const grids = document.querySelectorAll(".grid-2");
    if (grids[0]) {
      // Threat Intelligence grid: Recent Alerts + Prevention Actions
      grids[0].querySelectorAll(".panel").forEach((panel) => {
        const title = panel.querySelector(".panel-title")?.textContent || "";
        if (title.includes("Prevention") || title.includes("Actions")) {
          this.el.actionsTableBody = panel.querySelector(".data-table tbody");
        }
      });
    }
    if (grids[1]) {
      // Operations grid: Active Blocks + Audit Log
      grids[1].querySelectorAll(".panel").forEach((panel) => {
        const title = panel.querySelector(".panel-title")?.textContent || "";
        if (title.includes("Active Blocks") || title.includes("Blocks")) {
          this.el.blocksTableBody = panel.querySelector(".data-table tbody");
          this.el.blocksTag = panel.querySelector(".panel-tag");
        }
        if (title.includes("Audit")) {
          this.el.auditTableBody = panel.querySelector(".data-table tbody");
        }
      });
    }

    // Timeline panel
    document.querySelectorAll(".panel").forEach((panel) => {
      const title = panel.querySelector(".panel-title")?.textContent || "";
      if (title.includes("Timeline")) {
        this.el.timelineWrap = panel.querySelector(".table-wrap") || panel.querySelector(".panel-body");
      }
    });

    // Sidebar event counters
    const sidebarSections = document.querySelectorAll(".sidebar-section");
    const eventSection = sidebarSections[sidebarSections.length - 1];
    if (eventSection) {
      const statMinis = eventSection.querySelectorAll(".stat-mini");
      if (statMinis.length >= 5) {
        this.el.sidebarRequests = statMinis[0].querySelector(".stat-mini-val");
        this.el.sidebarAlerts = statMinis[1].querySelector(".stat-mini-val");
        this.el.sidebarActions = statMinis[2].querySelector(".stat-mini-val");
        this.el.sidebarRateLtd = statMinis[3].querySelector(".stat-mini-val");
        this.el.sidebarUnauth = statMinis[4].querySelector(".stat-mini-val");
      }
    }
  }

  // ====================================================================
  // SOC THREAT STATE SYSTEM (preserved behavior)
  // ====================================================================
  syncThreatState(alertCount) {
    if (typeof window.updateThreatState === "function") {
      window.updateThreatState(alertCount);
    }
    this.el.statusStrip?.classList.toggle("threat-active", alertCount > 0);
  }

  updateGlobalStatus(alertCount) {
    const statusTag = this.el.globalStatusBadge;
    if (!statusTag) return;

    if (alertCount > 0) {
      statusTag.textContent = "🚨 UNDER ATTACK";
      statusTag.classList.remove("tag-green");
      statusTag.classList.add("tag-red", "pulse-soft");
    } else {
      statusTag.textContent = "✅ NORMAL";
      statusTag.classList.remove("tag-red", "pulse-soft");
      statusTag.classList.add("tag-green");
    }
  }

  // ====================================================================
  // STATE SUBSCRIPTION (upgraded)
  // - Uses window.GlobalState (same as monitor.js)
  // - Supports BOTH legacy flat fields and new nested fields
  // ====================================================================
  subscribeToState() {
    window.GlobalState.subscribe((state) => {
      if (!state || typeof state !== "object") return;

      // ---- Status ----------------------------------------------------
      if (state.status) {
        const s = String(state.status).toLowerCase();
        const safe = s === "safe" || s === "normal";
        this.updateGlobalStatus(safe ? 0 : 1);
        if (!safe) playAlertTone("high");
      }

      // ---- Nested metrics (preferred) --------------------------------
      if (state.metrics) {
        const m = state.metrics;
        if (this.el.ingestedValue) smoothNumber(this.el.ingestedValue, m.ingested_total || 0);
        if (this.el.processedValue) smoothNumber(this.el.processedValue, m.processed_ingestion_total || 0);
        if (this.el.alertsValue) smoothNumber(this.el.alertsValue, m.alerts_total || 0);
        if (this.el.blockedValue) smoothNumber(this.el.blockedValue, m.prevention_actions_total || 0);

        // Sidebar counters
        if (this.el.sidebarRequests) smoothNumber(this.el.sidebarRequests, m.requests_total || 0);
        if (this.el.sidebarAlerts) {
          smoothNumber(this.el.sidebarAlerts, m.alerts_total || 0);
          this.el.sidebarAlerts.classList.toggle("pulse-soft", (m.alerts_total || 0) > 0);
        }
        if (this.el.sidebarActions) smoothNumber(this.el.sidebarActions, m.prevention_actions_total || 0);
        if (this.el.sidebarRateLtd) smoothNumber(this.el.sidebarRateLtd, m.rate_limited_total || 0);
        if (this.el.sidebarUnauth) smoothNumber(this.el.sidebarUnauth, m.unauthorized_total || 0);

        // Threat state system (alert count)
        this.syncThreatState(m.alerts_total || 0);
      }

      // ---- Nested reconciliation -------------------------------------
      if (state.reconciliation) {
        const r = state.reconciliation;
        if (this.el.recDbActive) smoothNumber(this.el.recDbActive, r.db_active || 0);
        if (this.el.recFirewallRules) smoothNumber(this.el.recFirewallRules, r.firewall_rules || 0);
        if (this.el.recMissingFW) {
          smoothNumber(this.el.recMissingFW, r.missing_in_firewall || 0);
          this.el.recMissingFW.className = r.missing_in_firewall > 0 ? "rec-val val-red" : "rec-val val-green";
        }
        if (this.el.recOrphanFW) {
          smoothNumber(this.el.recOrphanFW, r.orphan_firewall_rules || 0);
          this.el.recOrphanFW.className = r.orphan_firewall_rules > 0 ? "rec-val val-amber" : "rec-val val-green";
        }
      }

      // ---- Legacy flat fields (backwards compatible) -----------------
      if (state.active_alerts !== undefined) {
        this.syncThreatState(state.active_alerts || 0);
        this.updateGlobalStatus(state.active_alerts > 0 ? 1 : 0);
      }

      // ---- Topbar timestamp ------------------------------------------
      if (this.el.topbarTimestamp) {
        this.el.topbarTimestamp.textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
      }
    });
  }

  // ====================================================================
  // REAL-TIME SOCKET HANDLERS (new, aligned with monitor.js)
  // ====================================================================
  attachRealtimeHandlers() {
    // ---- Connection events ------------------------------------------
    SocketCore.on("connect", () => {
      console.log("%c[DASHBOARD] Connected to WebSocket", "color:#4caf50;font-weight:bold;");
      if (this.el.liveDot) this.el.liveDot.style.background = "var(--green)";
      showSuccess("Real-time feed connected");
    });

    SocketCore.on("disconnect", () => {
      console.warn("[DASHBOARD] Socket disconnected — data may be stale");
      if (this.el.liveDot) this.el.liveDot.style.background = "var(--red)";
      showError("Real-time feed lost — data may be stale");
    });

    SocketCore.on("reconnect", () => {
      console.log("%c[DASHBOARD] Reconnected", "color:#4caf50;font-weight:bold;");
      if (this.el.liveDot) this.el.liveDot.style.background = "var(--green)";
      showSuccess("Real-time feed restored");
    });

    // ---- Mirror monitor.js approval request event -------------------
    SocketCore.on("approval_request", (item) => {
      // Dashboard doesn't render the approval UI by default, but we
      // react with sound and optional toast to keep parity with monitor.
      playAlertTone(item?.severity || "medium");
    });

    // ---- New alert: prepend into Recent Alerts table ----------------
    SocketCore.on("new_alert", (alert) => {
      if (!this.el.alertsTableBody) return;

      this._removeEmptyState(this.el.alertsPanel);

      const tr = document.createElement("tr");
      tr.className = this._sevRowClass(alert?.severity);

      const sev = (alert?.severity || "low").toLowerCase();
      const pred = alert?.prediction || alert?.attack_type || "Unknown";
      const conf = alert?.confidence ?? 0;

      tr.innerHTML = `
        <td class="mono-val">${alert?.id ?? "—"}</td>
        <td>
          <span class="${this._sevClass(sev)}" style="font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.5px">
            ${sev}
          </span>
        </td>
        <td>${pred}</td>
        <td><span class="mono-val">${conf}%</span></td>
        <td><button class="btn-danger" data-ip="${alert?.ip || alert?.target || ""}">Block</button></td>
      `;

      fadeIn(tr);
      this.el.alertsTableBody.prepend(tr);

      // Update "ACTIVE" tag count
      if (this.el.alertsTag) {
        const count = this.el.alertsTableBody.querySelectorAll("tr").length;
        this.el.alertsTag.className = "panel-tag tag-red pulse-soft";
        this.el.alertsTag.textContent = `${count} ACTIVE`;
      }

      playAlertTone(sev);
    });

    // ---- New prevention action: prepend to Prevention Actions table --
    SocketCore.on("new_action", (action) => {
      if (!this.el.actionsTableBody) return;

      this._removeEmptyState(this.el.actionsTableBody.closest(".panel"));

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="panel-tag tag-red" style="white-space:nowrap">${String(action?.action || "BLOCK").toUpperCase()}</span></td>
        <td class="mono-val">${action?.target || "—"}</td>
        <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-muted)">${action?.reason || "—"}</td>
      `;
      fadeIn(tr);
      this.el.actionsTableBody.prepend(tr);
    });

    // ---- Block update: prepend to Active Blocks table ---------------
    SocketCore.on("block_update", (block) => {
      if (!this.el.blocksTableBody) return;

      this._removeEmptyState(this.el.blocksTableBody.closest(".panel"));

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono-val">${block?.target || "—"}</td>
        <td><span class="panel-tag tag-red">${String(block?.status || "ACTIVE").toUpperCase()}</span></td>
        <td class="mono-val" style="color:var(--text-muted)">${block?.expires_at || "—"}</td>
      `;
      fadeIn(tr);
      this.el.blocksTableBody.prepend(tr);

      if (this.el.blocksTag) {
        const count = this.el.blocksTableBody.querySelectorAll("tr").length;
        this.el.blocksTag.textContent = `${count} BLOCKED`;
      }
    });

    // ---- Audit event: prepend to Audit Log table --------------------
    SocketCore.on("audit_event", (audit) => {
      if (!this.el.auditTableBody) return;

      this._removeEmptyState(this.el.auditTableBody.closest(".panel"));

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono-val" style="white-space:nowrap;color:var(--cyan)">${audit?.event_type || "SYSTEM"}</td>
        <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-muted)">${audit?.message || "—"}</td>
      `;
      fadeIn(tr);
      this.el.auditTableBody.prepend(tr);
    });

    // ---- Timeline event: prepend to Action Timeline -----------------
    SocketCore.on("timeline_event", (item) => {
      if (!this.el.timelineWrap) return;

      this._removeEmptyState(this.el.timelineWrap.closest(".panel"));

      const dotColor = item?.type === "alert" ? "var(--red)" : item?.type === "block" ? "var(--amber)" : "var(--cyan)";
      const when = this._shortTime(item?.when);

      const div = document.createElement("div");
      div.className = "timeline-item";
      div.innerHTML = `
        <div class="timeline-when">${when}</div>
        <div class="timeline-dot ${item?.type === "alert" ? "alert pulse-soft" : ""}" style="background:${dotColor}"></div>
        <div class="timeline-content">
          <div class="timeline-type">${String(item?.type || "EVENT").toUpperCase()}</div>
          <div class="timeline-msg">${item?.message || "—"}</div>
        </div>
      `;
      fadeIn(div);
      this.el.timelineWrap.prepend(div);
    });
  }

  // ====================================================================
  // BLOCK BUTTON DELEGATION (new)
  // - Uses same event name as monitor.js bridge: "approval_response"
  // ====================================================================
  wireBlockButtons() {
    document.addEventListener("click", (e) => {
      const btn = e.target.closest(".btn-danger");
      if (!btn) return;

      const row = btn.closest("tr");
      const ip = btn.dataset.ip || row?.querySelector(".mono-val")?.textContent?.trim();
      if (!ip) return;

      SocketCore.emit("approval_response", { ip, action: "block" });
      console.log(`%c[DASHBOARD] approval_response: BLOCK → ${ip}`, "color:#e8413a;font-weight:bold;");

      btn.disabled = true;
      btn.textContent = "Blocked";
      btn.style.opacity = "0.6";
      btn.style.cursor = "default";

      showSuccess(`Blocked ${ip}`);
    });
  }

  // ====================================================================
  // MODAL SYSTEM (preserved)
  // ====================================================================
  setupModal() {
    const modalElement = document.getElementById("moduleModal");
    if (modalElement && typeof bootstrap !== "undefined") {
      this.moduleModal = new bootstrap.Modal(modalElement);
    }
  }

  attachCardListeners() {
    const cards = document.querySelectorAll(".capability-card");
    cards.forEach((card) => {
      card.setAttribute("tabindex", "0");
      card.setAttribute("role", "button");

      card.addEventListener("click", (e) => {
        e.preventDefault();
        this.openModule(card.dataset.module);
      });

      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          this.openModule(card.dataset.module);
        }
      });
    });
  }

  attachControlListeners() {
    const demoModeToggle = document.getElementById("demoModeToggle");
    if (demoModeToggle) {
      demoModeToggle.addEventListener("click", () => this.toggleDemoMode());
    }

    const refreshBtn = document.getElementById("refreshBtn");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", () => this.refreshAllData());
    }

    const settingsBtn = document.getElementById("settingsBtn");
    if (settingsBtn) {
      settingsBtn.addEventListener("click", () => this.openSettings());
    }
  }

  openModule(moduleKey) {
    const moduleConfig = this.moduleRegistry[moduleKey];
    if (!moduleConfig) {
      console.error(`Unknown module: ${moduleKey}`);
      return;
    }

    this.currentModule = moduleKey;
    const modalTitle = document.getElementById("moduleTitle");
    const modalContent = document.getElementById("moduleContent");

    if (modalTitle) {
      modalTitle.textContent = moduleConfig.title;
    }

    if (modalContent) {
      modalContent.innerHTML = `
        <div class="text-center py-5">
          <div class="spinner-border text-info" role="status">
            <span class="visually-hidden">Loading...</span>
          </div>
          <p class="mt-2 text-muted">Loading ${moduleConfig.title}...</p>
        </div>
      `;
    }

    if (this.moduleModal) {
      this.moduleModal.show();
    }

    this.loadModuleContent(moduleConfig);
  }

  loadModuleContent(config, retryCount = 0) {
    const maxRetries = 2;
    const timeoutMs = 15000;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    fetch(config.route, {
      method: "GET",
      headers: { "X-Requested-With": "XMLHttpRequest", Accept: "text/html" },
      signal: controller.signal,
    })
      .then((response) => {
        clearTimeout(timeout);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.text();
      })
      .then((html) => {
        const modalContent = document.getElementById("moduleContent");
        if (modalContent) {
          modalContent.innerHTML = html;
          this.initializeModuleScripts();
        }
      })
      .catch((error) => {
        clearTimeout(timeout);
        console.error(`Error loading ${config.title}:`, error.message);

        const modalContent = document.getElementById("moduleContent");
        if (modalContent) {
          const isNetworkError = error.name === "AbortError" || !navigator.onLine;

          let retryButton = "";
          if (retryCount < maxRetries) {
            retryButton = `
              <button class="btn btn-sm btn-primary mt-3"
                      onclick="window.dashboard.loadModuleContent(${JSON.stringify(config)}, ${retryCount + 1})">
                Retry (${maxRetries - retryCount} left)
              </button>
            `;
          }

          modalContent.innerHTML = `
            <div class="alert alert-danger" role="alert">
              <h4 class="alert-heading">⚠️ ${isNetworkError ? "Connection Error" : "Module Not Available"}</h4>
              <p>${isNetworkError
                ? "Could not connect to the server. Please check your connection."
                : `The ${config.title} module is still in development.`
              }</p>
              <hr>
              <p class="mb-0 text-muted" style="font-size:0.9em">Error: ${error.message}</p>
              ${retryButton}
            </div>
          `;
        }
      });
  }

  initializeModuleScripts() {
    console.log("[DASHBOARD] Module scripts initialized");
  }

  // ====================================================================
  // DEMO MODE (preserved)
  // ====================================================================
  toggleDemoMode() {
    this.demoMode = !this.demoMode;
    const demoModeText = document.getElementById("demoModeText");
    const demoModeToggle = document.getElementById("demoModeToggle");

    if (demoModeText) {
      demoModeText.textContent = this.demoMode ? "Demo Mode: ON" : "Demo Mode: OFF";
    }

    if (demoModeToggle) {
      demoModeToggle.classList.toggle("btn-outline-warning", !this.demoMode);
      demoModeToggle.classList.toggle("btn-warning", this.demoMode);
    }

    if (this.demoMode) this.startDemoTraffic();
    else this.stopDemoTraffic();
  }

  startDemoTraffic() {
    console.log("[DASHBOARD] Starting demo traffic simulation...");
    fetch("/api/demo/start", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        console.log("Demo started:", data);
        this.showNotification("Demo Mode Activated", "Generating synthetic attack traffic...", "info");
      })
      .catch((error) => {
        console.error("Error starting demo:", error);
        this.showNotification("Demo Mode Error", "Could not start demo traffic", "danger");
      });
  }

  stopDemoTraffic() {
    console.log("[DASHBOARD] Stopping demo traffic simulation...");
    fetch("/api/demo/stop", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        console.log("Demo stopped:", data);
        this.showNotification("Demo Mode Deactivated", "Demo traffic stopped", "info");
      })
      .catch((error) => {
        console.error("Error stopping demo:", error);
      });
  }

  // ====================================================================
  // REFRESH (upgraded)
  // - Uses SocketCore.hydrate() if available (instead of INIDSSocketManager)
  // ====================================================================
  refreshAllData() {
    console.log("[DASHBOARD] Refreshing all data...");

    const refreshBtn = document.getElementById("refreshBtn");
    if (refreshBtn) {
      refreshBtn.disabled = true;
      refreshBtn.innerHTML = "⟳ Refreshing...";
      refreshBtn.classList.add("pulse-soft");
    }

    fetch("/api/dashboard/refresh", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        window.GlobalState.set({ lastRefresh: data.timestamp || new Date().toISOString() });

        if (typeof SocketCore.hydrate === "function") {
          SocketCore.hydrate().catch((err) => console.error("[DASHBOARD] Hydrate failed:", err));
        }

        this.showNotification("Refresh Complete", "All data updated", "success");
      })
      .catch((error) => {
        console.error("Error refreshing data:", error);
        this.showNotification("Refresh Failed", "Could not refresh data", "danger");
      })
      .finally(() => {
        if (refreshBtn) {
          refreshBtn.disabled = false;
          refreshBtn.innerHTML = "⟳ Refresh";
          refreshBtn.classList.remove("pulse-soft");
        }
      });
  }

  loadInitialMetrics() {
    // Use existing state immediately
    if (window.GlobalState.data && Object.keys(window.GlobalState.data).length > 0) {
      return;
    }

    // Prefer SocketCore hydration
    if (typeof SocketCore.hydrate === "function") {
      SocketCore.hydrate().catch((error) => {
        console.error("[DASHBOARD] Error loading shared dashboard state:", error);
        this.useMockMetrics();
      });
      return;
    }

    this.useMockMetrics();
  }

  // Legacy-compatible update path (kept for dev mode)
  useMockMetrics() {
    const mockData = {
      status: "suspicious",
      metrics: {
        ingested_total: 4200,
        processed_ingestion_total: 4100,
        alerts_total: 12,
        prevention_actions_total: 5,
        requests_total: 12000,
        rate_limited_total: 23,
        unauthorized_total: 3,
      },
      reconciliation: {
        db_active: 5,
        firewall_rules: 5,
        missing_in_firewall: 0,
        orphan_firewall_rules: 1,
      },
    };
    window.GlobalState.set(mockData);
  }

  openSettings() {
    this.showNotification("Settings", "Settings panel coming soon", "info");
  }

  // Toast notifications (preserved)
  showNotification(title, message, type = "info") {
    const toastHtml = `
      <div class="toast align-items-center text-white bg-${this.mapAlertType(type)} border-0" role="alert" aria-live="assertive" aria-atomic="true">
        <div class="d-flex">
          <div class="toast-body"><strong>${title}:</strong> ${message}</div>
          <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
      </div>
    `;

    let toastContainer = document.getElementById("toastContainer");
    if (!toastContainer) {
      toastContainer = document.createElement("div");
      toastContainer.id = "toastContainer";
      toastContainer.className = "toast-container position-fixed bottom-0 end-0 p-3";
      document.body.appendChild(toastContainer);
    }

    const wrapper = document.createElement("div");
    wrapper.innerHTML = toastHtml;
    toastContainer.appendChild(wrapper.firstElementChild);

    if (typeof bootstrap !== "undefined") {
      const toast = new bootstrap.Toast(wrapper.firstElementChild);
      toast.show();
      wrapper.firstElementChild.addEventListener("hidden.bs.toast", () => {
        wrapper.firstElementChild.remove();
      });
    }
  }

  mapAlertType(type) {
    const typeMap = { info: "info", success: "success", warning: "warning", danger: "danger" };
    return typeMap[type] || "info";
  }

  // Optional sparklines (preserved)
  animateSparklines() {
    document.querySelectorAll(".spark-bar").forEach((bar, i) => {
      const h = 10 + Math.random() * 90;
      setTimeout(() => {
        bar.style.height = h + "%";
        if (h > 70) bar.classList.add("hi");
      }, i * 60);
    });
  }

  // Helpers
  _sevClass(sev) {
    switch (String(sev || "").toLowerCase()) {
      case "critical":
        return "sev-critical";
      case "high":
        return "sev-high";
      case "medium":
        return "sev-medium";
      default:
        return "sev-low";
    }
  }

  _sevRowClass(sev) {
    switch (String(sev || "").toLowerCase()) {
      case "critical":
        return "sev-critical-row pulse-soft";
      case "high":
        return "sev-high-row";
      default:
        return "";
    }
  }

  _shortTime(dateStr) {
    try {
      const d = new Date(dateStr);
      if (Number.isNaN(+d)) throw new Error();
      return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
    } catch {
      return new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
    }
  }

  _removeEmptyState(container) {
    if (!container) return;
    const empty = container.querySelector(".empty-state");
    if (empty) empty.remove();
  }
}

// ======================================================================
// BOOTSTRAP
// - Expose on window so inline retry buttons work
// ======================================================================

window.dashboard = new DashboardController();
console.log("%c[DASHBOARD] Initialized", "color:#3b8cf0;font-weight:bold;");
