// ======================================================================
// ALLOWLIST PAGE (ES MODULE VERSION)
// Aligned with alerts.js, actions.js, dashboard.js, monitor.js
// Uses: core/socket_core.js, core/utils.js, core/ui_core.js
// ======================================================================

import SocketCore from "./core/socket_core.js";
import {
  showError as coreShowError,
  showSuccess as coreShowSuccess,
} from "./core/ui_core.js";

console.log(
  "%c[ALLOWLIST] Loaded (ES Module)",
  "color:#10b981;font-weight:bold;"
);

// ======================================================================
// STATE
// ======================================================================

let allAllowlist = [];
let filteredAllowlist = [];

let currentPage = 1;
const itemsPerPage = 10;

let currentDetailId = null;
let deleteId = null;

// ======================================================================
// INIT
// ======================================================================

(function init() {
  loadAllowlist(false);

  document.getElementById("searchBox")?.addEventListener("keyup", filterAllowlist);
  document.getElementById("typeFilter")?.addEventListener("change", filterAllowlist);
  document.getElementById("reasonFilter")?.addEventListener("change", filterAllowlist);

  if (window.GlobalState) {
    window.GlobalState.subscribe((state) => {
      if (!state) return;
      if (Array.isArray(state.allowlist)) {
        syncAllowlistFromState(state.allowlist);
      }
    });
  }

  attachSocketHandlers();
  setInterval(() => loadAllowlist(false), 30000);
})();

// ======================================================================
// SOCKET REAL-TIME HANDLERS
// ======================================================================

function attachSocketHandlers() {
  SocketCore.on("allowlist_update", (payload) => {
    console.log("%c[ALLOWLIST] allowlist_update", "color:#3b82f6;", payload);
    loadAllowlist(true);
  });

  SocketCore.on("block_update", () => loadAllowlist(true));
  SocketCore.on("action_status_change", () => loadAllowlist(true));

  SocketCore.on("connect", () => {
    console.log("%c[ALLOWLIST] Socket connected", "color:#10b981;");
    coreShowSuccess("Allowlist connected to live updates");
    loadAllowlist(true);
  });

  SocketCore.on("disconnect", () => {
    coreShowError("Allowlist connection lost — using cached data");
  });

  SocketCore.on("reconnect", () => {
    coreShowSuccess("Allowlist reconnected");
    loadAllowlist(true);
  });
}

// ======================================================================
// LOAD + STATE SYNC
// ======================================================================

async function loadAllowlist(force = false) {
  if (
    !force &&
    window.GlobalState &&
    Array.isArray(window.GlobalState.data?.allowlist)
  ) {
    syncAllowlistFromState(window.GlobalState.data.allowlist);
    return;
  }

  try {
    const res = await fetch("/api/allowlist");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    const entries = data.entries || [];

    if (window.GlobalState) {
      window.GlobalState.set({ allowlist: entries });
    }

    syncAllowlistFromState(entries);
  } catch (err) {
    console.error("[ALLOWLIST] Load failed:", err);
    coreShowError("Failed to load allowlist");
  }
}

function syncAllowlistFromState(entries) {
  allAllowlist = entries;
  filteredAllowlist = [...entries];
  currentPage = 1;

  updateStats();
  renderAllowlist();
}

// ======================================================================
// FILTERING
// ======================================================================

function filterAllowlist() {
  const search = document.getElementById("searchBox")?.value.toLowerCase() || "";
  const typeFilter = document.getElementById("typeFilter")?.value || "";
  const reasonFilter = document.getElementById("reasonFilter")?.value || "";

  filteredAllowlist = allAllowlist.filter((item) => {
    const matchSearch = item.entry.toLowerCase().includes(search);

    const matchType =
      !typeFilter ||
      (typeFilter === "ip" && isIP(item.entry)) ||
      (typeFilter === "domain" && !isIP(item.entry));

    const matchReason = !reasonFilter || item.reason === reasonFilter;

    return matchSearch && matchType && matchReason;
  });

  currentPage = 1;
  renderAllowlist();
}

// ======================================================================
// RENDERING
// ======================================================================

function renderAllowlist() {
  const tbody = document.getElementById("allowlistTable");
  const pagination = document.getElementById("pagination");

  const start = (currentPage - 1) * itemsPerPage;
  const end = start + itemsPerPage;
  const pageItems = filteredAllowlist.slice(start, end);

  if (!pageItems.length) {
    tbody.innerHTML =
      '<tr><td colspan="6" style="text-align:center;padding:24px;color:rgba(255,255,255,0.4);">No entries found</td></tr>';
    pagination.style.display = "none";
    return;
  }

  tbody.innerHTML = pageItems
    .map(
      (item) => `
    <tr onclick="showDetails('${escapeHtml(item.entry)}')" style="cursor:pointer;">
      <td>
        <strong>${escapeHtml(item.entry)}</strong><br>
        <small class="text-muted">${isIP(item.entry) ? "\uD83D\uDCCD IP Address" : "\uD83C\uDF10 Domain"}</small>
      </td>
      <td><span class="badge ${getTypeBadge(item.entry)}">${getTypeLabel(item.entry)}</span></td>
      <td>${escapeHtml(item.reason || "N/A")}</td>
      <td>${escapeHtml(item.added_by || "system")}</td>
      <td><small>${new Date(item.added_at).toLocaleString()}</small></td>
      <td>
        <button class="btn btn-sm btn-outline-danger"
          onclick="event.stopPropagation(); openDeleteModal('${escapeHtml(item.entry)}')"
          title="Remove">
          \uD83D\uDDD1\uFE0F
        </button>
      </td>
    </tr>
  `
    )
    .join("");

  updatePagination();
}

// ======================================================================
// PAGINATION
// ======================================================================

function updatePagination() {
  const pagination = document.getElementById("pagination");
  const totalPages = Math.ceil(filteredAllowlist.length / itemsPerPage);

  if (totalPages <= 1) {
    pagination.style.display = "none";
    return;
  }

  pagination.style.display = "block";
  document.getElementById("pageInfo").textContent = `Page ${currentPage} of ${totalPages}`;
  document.getElementById("prevBtn").classList.toggle("disabled", currentPage === 1);
  document.getElementById("nextBtn").classList.toggle("disabled", currentPage === totalPages);
}

function previousPage() {
  if (currentPage > 1) {
    currentPage--;
    renderAllowlist();
    window.scrollTo(0, 0);
  }
}

function nextPage() {
  const totalPages = Math.ceil(filteredAllowlist.length / itemsPerPage);
  if (currentPage < totalPages) {
    currentPage++;
    renderAllowlist();
    window.scrollTo(0, 0);
  }
}

// ======================================================================
// ADD / DELETE
// ======================================================================

function openAddModal() {
  document.getElementById("addForm")?.reset();
  new bootstrap.Modal(document.getElementById("addModal")).show();
}

async function saveEntry() {
  const entry = document.getElementById("entryInput").value.trim();
  const reason = document.getElementById("reasonSelect").value;
  const notes = document.getElementById("notesInput").value.trim();

  if (!entry || !reason) {
    coreShowError("Please fill in all required fields");
    return;
  }

  if (!isValidEntry(entry)) {
    coreShowError("Invalid IP or domain format");
    return;
  }

  try {
    const res = await fetch("/api/allowlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entry, reason, notes }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    SocketCore.emit("allowlist_update", { action: "add", entry });
    bootstrap.Modal.getInstance(document.getElementById("addModal")).hide();
    coreShowSuccess("Entry added successfully");
    await loadAllowlist(true);
  } catch (err) {
    console.error("[ALLOWLIST] Save failed:", err);
    coreShowError("Failed to add entry: " + err.message);
  }
}

async function confirmDelete() {
  if (!deleteId) return;

  try {
    const res = await fetch(`/api/allowlist/${encodeURIComponent(deleteId)}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    SocketCore.emit("allowlist_update", { action: "delete", entry: deleteId });

    bootstrap.Modal.getInstance(document.getElementById("deleteModal")).hide();
    const detailsModalEl = document.getElementById("detailsModal");
    const detailsInst = bootstrap.Modal.getInstance(detailsModalEl);
    if (detailsInst) detailsInst.hide();

    coreShowSuccess("Entry removed successfully");
    await loadAllowlist(true);
  } catch (err) {
    console.error("[ALLOWLIST] Delete failed:", err);
    coreShowError("Failed to remove entry: " + err.message);
  }
}

// ======================================================================
// DETAILS MODAL
// ======================================================================

function showDetails(entry) {
  const item = allAllowlist.find((a) => a.entry === entry);
  if (!item) return;

  currentDetailId = entry;
  document.getElementById("detailEntry").textContent = item.entry;
  document.getElementById("detailType").textContent = getTypeLabel(item.entry);
  document.getElementById("detailReason").textContent = item.reason || "N/A";
  document.getElementById("detailAddedBy").textContent = item.added_by || "system";
  document.getElementById("detailAddedDate").textContent = new Date(item.added_at).toLocaleString();
  document.getElementById("detailNotes").textContent = item.notes || "No notes";

  new bootstrap.Modal(document.getElementById("detailsModal")).show();
}

function openDeleteModal(entry) {
  deleteId = entry;
  const item = allAllowlist.find((a) => a.entry === entry);
  if (!item) return;

  document.getElementById("deleteItemText").textContent = item.entry;
  new bootstrap.Modal(document.getElementById("deleteModal")).show();
}

// ======================================================================
// UTILITIES
// ======================================================================

function updateStats() {
  const el = (id) => document.getElementById(id);
  el("totalCount").textContent = allAllowlist.length;
  el("ipCount").textContent = allAllowlist.filter((a) => isIP(a.entry)).length;
  el("domainCount").textContent = allAllowlist.filter((a) => !isIP(a.entry)).length;
  el("lastUpdated").textContent = new Date().toLocaleTimeString();
}

function isIP(entry) {
  return /^(\d{1,3}\.){3}\d{1,3}(\/\d{1,2})?$/.test(entry);
}

function isValidEntry(entry) {
  const ipMatch = /^(\d{1,3}\.){3}\d{1,3}(\/(8|16|24|32))?$/.test(entry);
  const domainMatch = /^([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9]?\.)+[a-zA-Z]{2,}$/.test(entry);
  return ipMatch || domainMatch;
}

function getTypeLabel(entry) {
  return isIP(entry) ? "IP" : "Domain";
}

function getTypeBadge(entry) {
  return isIP(entry) ? "bg-info" : "bg-success";
}

function escapeHtml(text) {
  if (text == null) return "";
  const map = { "&": "&", "<": "<", ">": ">", '"': """, "'": "'" };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

// ======================================================================
// EXPOSE FOR INLINE HTML (onclick= handlers in allowlist.html)
// ======================================================================

window.loadAllowlist = loadAllowlist;
window.filterAllowlist = filterAllowlist;
window.openAddModal = openAddModal;
window.saveEntry = saveEntry;
window.showDetails = showDetails;
window.openDeleteModal = openDeleteModal;
window.confirmDelete = confirmDelete;
window.previousPage = previousPage;
window.nextPage = nextPage;