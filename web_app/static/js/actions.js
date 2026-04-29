// ======================================================================
// ACTIONS PAGE (ES MODULE VERSION)
// Aligned with monitor.js + dashboard.js architecture
// Uses: core/socket_core.js, core/utils.js, core/ui_core.js
// ======================================================================

import SocketCore from "./core/socket_core.js";
import { fadeIn, playAlertTone } from "./core/utils.js";
import { showError as coreShowError, showSuccess as coreShowSuccess } from "./core/ui_core.js";

console.log("%c[ACTIONS] Loaded (ES Module)", "color:#f0a500;font-weight:bold;");


// ======================================================================
// STATE
// ======================================================================

let currentAction = null;
let approvalModal = null;
let allActions = [];
let pendingActions = [];


// ======================================================================
// INIT
// ======================================================================

(function init() {
    const modalEl = document.getElementById("approvalModal");
    if (modalEl && typeof bootstrap !== "undefined") {
        approvalModal = new bootstrap.Modal(modalEl);
    }

    // Initial data load via REST API
    loadAllActions();
    loadPendingActions();

    // Confirm-approve button
    const confirmBtn = document.getElementById("btn-confirm-approve");
    if (confirmBtn) {
        confirmBtn.addEventListener("click", approveAction);
    }

    // Subscribe to GlobalState (same pattern as dashboard.js)
    if (window.GlobalState) {
        window.GlobalState.subscribe((state) => {
            if (!state) return;

            // If metrics include action counts, update badge reactively
            if (state.metrics && state.metrics.prevention_actions_total !== undefined) {
                const badge = document.getElementById("pending-badge");
                if (badge && pendingActions.length > 0) {
                    badge.innerHTML = `<span class="badge bg-warning">${pendingActions.length}</span>`;
                }
            }
        });
    }

    // Wire up real-time socket events
    attachSocketHandlers();

    // Fallback polling (60s instead of 30s — socket handles most updates now)
    setInterval(() => {
        loadAllActions();
        loadPendingActions();
    }, 60000);
})();


// ======================================================================
// SOCKET REAL-TIME HANDLERS (new — aligned with monitor.js + dashboard.js)
// ======================================================================

function attachSocketHandlers() {

    // ---- New action created by the pipeline -------------------------
    SocketCore.on("new_action", (action) => {
        console.log("%c[ACTIONS] new_action received", "color:#f0a500;", action);

        // Add to local arrays and re-render
        if (action) {
            allActions.unshift(action);
            renderActions("all-actions-list", allActions);

            const status = (action.status || "").toLowerCase();
            if (["pending", "pending_approval"].includes(status)) {
                pendingActions.unshift(action);
                renderActions("pending-actions-list", pendingActions, true);
                updatePendingBadge();
                playAlertTone("medium");
            }
        }
    });

    // ---- Approval request (same event monitor.js listens to) --------
    SocketCore.on("approval_request", (item) => {
        console.log("%c[ACTIONS] approval_request received", "color:#ff9800;", item);
        playAlertTone(item?.severity || "medium");

        // Refresh pending list to pick up the new approval
        loadPendingActions();
    });

    // ---- Block update (action was executed) -------------------------
    SocketCore.on("block_update", (block) => {
        console.log("%c[ACTIONS] block_update received", "color:#e8413a;", block);
        // Refresh all actions to reflect status changes
        loadAllActions();
        loadPendingActions();
    });

    // ---- Action status changed (approved/executed/failed) -----------
    SocketCore.on("action_status_change", (data) => {
        console.log("%c[ACTIONS] action_status_change received", "color:#27c97a;", data);
        loadAllActions();
        loadPendingActions();
    });

    // ---- Connection events ------------------------------------------
    SocketCore.on("connect", () => {
        console.log("%c[ACTIONS] Connected to WebSocket", "color:#4caf50;font-weight:bold;");
        coreShowSuccess("Actions: real-time feed connected");
        // Refresh data on reconnect to catch missed events
        loadAllActions();
        loadPendingActions();
    });

    SocketCore.on("disconnect", () => {
        console.warn("[ACTIONS] Socket disconnected \u2014 falling back to polling");
        coreShowError("Actions: real-time feed lost \u2014 using fallback polling");
    });

    SocketCore.on("reconnect", () => {
        console.log("%c[ACTIONS] Reconnected", "color:#4caf50;");
        coreShowSuccess("Actions: real-time feed restored");
        loadAllActions();
        loadPendingActions();
    });
}


// ======================================================================
// TAB SWITCHING (preserved — exposed on window for onclick= in HTML)
// ======================================================================

function switchTab(tabName, evt) {
    document.querySelectorAll(".tab-content").forEach((tab) => {
        tab.classList.remove("active");
    });
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.classList.remove("active");
    });

    document.getElementById(`tab-${tabName}`).classList.add("active");
    evt.target.classList.add("active");

    // Auto-load executed/failed tabs from allActions
    if (tabName === "executed") {
        renderActions("executed-actions-list", allActions);
        hideLoader("loading-executed");
    } else if (tabName === "failed") {
        renderActions("failed-actions-list", allActions);
        hideLoader("loading-failed");
    }
}

// Expose on window (HTML uses onclick="switchTab(...)")
window.switchTab = switchTab;


// ======================================================================
// API: LOAD ALL ACTIONS (preserved + enhanced)
// ======================================================================

async function loadAllActions() {
    try {
        const response = await fetch("/api/actions?limit=200");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        allActions = data.actions || [];

        renderActions("all-actions-list", allActions);
        hideLoader("loading-all");

        // Also refresh executed/failed if their tabs are visible
        const executedTab = document.getElementById("tab-executed");
        if (executedTab?.classList.contains("active")) {
            renderActions("executed-actions-list", allActions);
            hideLoader("loading-executed");
        }
        const failedTab = document.getElementById("tab-failed");
        if (failedTab?.classList.contains("active")) {
            renderActions("failed-actions-list", allActions);
            hideLoader("loading-failed");
        }

    } catch (error) {
        console.error("[ACTIONS] Error loading actions:", error);
        coreShowError("Failed to load actions");
    }
}


// ======================================================================
// API: LOAD PENDING ACTIONS (preserved + enhanced)
// ======================================================================

async function loadPendingActions() {
    try {
        const response = await fetch("/api/actions/pending");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        pendingActions = data.actions || [];

        updatePendingBadge();
        renderActions("pending-actions-list", pendingActions, true);
        hideLoader("loading-pending");

    } catch (error) {
        console.error("[ACTIONS] Error loading pending actions:", error);
        coreShowError("Failed to load pending actions");
    }
}


// ======================================================================
// PENDING BADGE HELPER (new)
// ======================================================================

function updatePendingBadge() {
    const badge = document.getElementById("pending-badge");
    if (!badge) return;
    if (pendingActions.length > 0) {
        badge.innerHTML = `<span class="badge bg-warning">${pendingActions.length}</span>`;
    } else {
        badge.innerHTML = "";
    }
}


// ======================================================================
// FILTER BY STATUS (preserved)
// ======================================================================

function filterActionsByStatus(actions, status) {
    if (!status) return actions;
    return actions.filter((a) => (a.status || "").toLowerCase() === status.toLowerCase());
}


// ======================================================================
// RENDER ACTIONS (preserved — card-based layout)
// ======================================================================

function renderActions(containerId, actions, showApproveButton = false) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!actions || actions.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">\uD83D\uDCED</div>
                <h4>No Actions</h4>
                <p>No actions to display</p>
            </div>
        `;
        return;
    }

    // Filter further by tab
    let filtered = actions;
    const tabId = containerId.split("-")[0];
    if (tabId === "executed") {
        filtered = filterActionsByStatus(actions, "executed");
    } else if (tabId === "failed") {
        filtered = filterActionsByStatus(actions, "failed");
    }

    if (filtered.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">\uD83D\uDCED</div>
                <h4>No Actions</h4>
                <p>No actions in this category</p>
            </div>
        `;
        return;
    }

    let html = "";

    filtered.forEach((action) => {
        const status = (action.status || "pending").toLowerCase();
        const isPending = ["pending", "pending_approval"].includes(status);
        const actionType = action.action_type || action.action || "unknown";

        html += `
            <div class="action-card">
                <div class="action-header">
                    <span class="action-type">${escapeHtml(actionType)}</span>
                    <span class="status-badge status-${status}">${status}</span>
                </div>
                <div class="action-timestamp">${formatTimestamp(action.created_at)}</div>
                <div class="action-target">\uD83C\uDFAF ${escapeHtml(action.target || "N/A")}</div>
                <div class="action-details">
                    <strong>ID:</strong> <code>${escapeHtml((action.id || "N/A").substring(0, 16))}</code>
                </div>
                <div class="action-details">
                    <strong>Alert:</strong> <code>${escapeHtml((action.alert_id || "N/A").substring(0, 16))}</code>
                </div>
                ${action.reason ? `<div class="action-details"><strong>Reason:</strong> ${escapeHtml(action.reason)}</div>` : ""}
                ${action.created_by ? `<div class="action-details"><strong>Created by:</strong> ${escapeHtml(action.created_by)}</div>` : ""}
                ${action.approved_at ? `<div class="action-details"><strong>Approved:</strong> ${formatTimestamp(action.approved_at)}</div>` : ""}

                ${isPending && showApproveButton ? `
                    <button class="btn btn-sm btn-approve mt-2" onclick="openApprovalModal('${escapeHtml(JSON.stringify(action).replace(/'/g, "\\\\'"))}')">
                        \u2714\uFE0F Review & Approve
                    </button>
                ` : ""}
            </div>
        `;
    });

    container.innerHTML = html;
}


// ======================================================================
// APPROVAL MODAL (preserved)
// ======================================================================

function openApprovalModal(actionJson) {
    try {
        currentAction = JSON.parse(actionJson);

        const detailsHtml = `
            <div class="detail-row">
                <div class="detail-label">Action ID</div>
                <div class="detail-value code-block">${escapeHtml(currentAction.id || "N/A")}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Action Type</div>
                <div class="detail-value"><span class="action-type-badge">${escapeHtml(currentAction.action_type || "unknown")}</span></div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Target</div>
                <div class="detail-value code-block">${escapeHtml(currentAction.target || "N/A")}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Related Alert</div>
                <div class="detail-value code-block">${escapeHtml(currentAction.alert_id || "N/A")}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Reason</div>
                <div class="detail-value">${escapeHtml(currentAction.reason || "No reason specified")}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Created</div>
                <div class="detail-value">${formatTimestamp(currentAction.created_at)}</div>
            </div>
            ${currentAction.created_by ? `
            <div class="detail-row">
                <div class="detail-label">Created By</div>
                <div class="detail-value">${escapeHtml(currentAction.created_by)}</div>
            </div>
            ` : ""}
            <div class="approval-notes">
                <strong>\u2139\uFE0F Note:</strong><br>
                Approving this action will immediately execute the ${escapeHtml(currentAction.action_type || "action")}
                on target <code>${escapeHtml(currentAction.target || "unknown")}</code>.
            </div>
        `;

        document.getElementById("approval-details").innerHTML = detailsHtml;
        approvalModal.show();

    } catch (error) {
        console.error("[ACTIONS] Error parsing action:", error);
        coreShowError("Error loading action details");
    }
}

// Expose on window (HTML uses onclick="openApprovalModal(...)")
window.openApprovalModal = openApprovalModal;


// ======================================================================
// APPROVE ACTION (upgraded — REST API + SocketCore.emit)
// ======================================================================

async function approveAction() {
    if (!currentAction) return;

    try {
        // 1. REST API call (existing backend endpoint)
        const response = await fetch(
            `/api/actions/${encodeURIComponent(currentAction.id)}/approve`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ notes: "Approved via UI" }),
            }
        );

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        // 2. Also emit via SocketCore (so monitor.js + dashboard.js react instantly)
        SocketCore.emit("approval_response", {
            ip: currentAction.target,
            action: "block",
            action_id: currentAction.id,
            source: "actions_page",
        });

        console.log(
            `%c[ACTIONS] Approved: ${currentAction.id} \u2192 ${currentAction.target}`,
            "color:#27c97a;font-weight:bold;"
        );

        approvalModal.hide();
        coreShowSuccess("Action approved and executed successfully");

        // 3. Reload data
        await loadAllActions();
        await loadPendingActions();

    } catch (error) {
        console.error("[ACTIONS] Error approving action:", error);
        coreShowError("Failed to approve action: " + error.message);
    }
}


// ======================================================================
// HELPERS (preserved)
// ======================================================================

function hideLoader(loaderId) {
    const loader = document.getElementById(loaderId);
    if (loader) loader.style.display = "none";
}

function formatTimestamp(ts) {
    if (!ts) return "N/A";
    try {
        const date = new Date(ts);
        return date.toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    } catch {
        return ts;
    }
}

function escapeHtml(text) {
    if (!text) return "";
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
}
