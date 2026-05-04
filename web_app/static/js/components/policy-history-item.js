/**
 * PolicyHistoryItem Component - Display Policy Change History
 * Shows audit trail of policy modifications with before/after comparison
 * 
 * Usage:
 *   PolicyHistoryItem(policyChangeData)
 */

import { UIBadge } from "./ui_badge.js";

export function PolicyHistoryItem(item) {
    if (!item) return document.createElement("div");

    const {
        id = null,
        timestamp = null,
        user = "System",
        action = "modified", // created, modified, deleted, reverted
        field = null,
        oldValue = null,
        newValue = null,
        policyName = "Policy",
        reason = "",
        status = "applied", // applied, pending, reverted
    } = item;

    // Create container
    const container = document.createElement("div");
    container.className = `
        bg-[#0f1117] border border-[#1a1f2e] rounded-lg p-4
        hover:border-[#2a2f3e] transition-all duration-200
    `;

    if (id) {
        container.setAttribute("data-history-id", id);
    }

    // Header row
    const header = document.createElement("div");
    header.className = "flex items-start justify-between gap-3 mb-3";

    // Left side: action badge and user info
    const leftSide = document.createElement("div");
    leftSide.className = "flex items-center gap-2";

    const actionBadge = UIBadge.enum(action, {
        created: { label: "Created", variant: "safe", icon: "plus-circle" },
        modified: { label: "Modified", variant: "info", icon: "pencil" },
        deleted: { label: "Deleted", variant: "threat", icon: "trash" },
        reverted: { label: "Reverted", variant: "warn", icon: "arrow-counterclockwise" },
    });
    leftSide.appendChild(actionBadge);

    // User and timestamp
    const userInfo = document.createElement("div");
    userInfo.className = "flex items-center gap-2 text-xs text-gray-400";
    userInfo.innerHTML = `
        <i class="bi bi-person-fill"></i>
        <span>${user}</span>
    `;
    leftSide.appendChild(userInfo);

    // Status badge on right
    const statusBadge = UIBadge.status(status);

    header.appendChild(leftSide);
    header.appendChild(statusBadge);
    container.appendChild(header);

    // Policy and field info
    const infoRow = document.createElement("div");
    infoRow.className = "flex items-center gap-3 text-sm mb-3 pb-3 border-b border-gray-700";

    const policyEl = document.createElement("span");
    policyEl.className = "font-semibold text-white";
    policyEl.textContent = policyName;
    infoRow.appendChild(policyEl);

    if (field) {
        const fieldSeparator = document.createElement("span");
        fieldSeparator.className = "text-gray-600";
        fieldSeparator.textContent = "•";
        infoRow.appendChild(fieldSeparator);

        const fieldEl = document.createElement("code");
        fieldEl.className = "text-gray-400 text-xs bg-gray-900/50 px-2 py-1 rounded";
        fieldEl.textContent = field;
        infoRow.appendChild(fieldEl);
    }

    container.appendChild(infoRow);

    // Before/After comparison (if available)
    if (oldValue !== null && newValue !== null) {
        const comparisonEl = document.createElement("div");
        comparisonEl.className = "grid grid-cols-2 gap-3 mb-3 text-xs";

        // Old value
        const oldEl = document.createElement("div");
        oldEl.className = "bg-red-900/20 border border-red-700/30 rounded p-2";
        oldEl.innerHTML = `
            <p class="text-red-400 font-semibold mb-1">Previous</p>
            <code class="text-red-300 text-xs break-all">${JSON.stringify(oldValue)}</code>
        `;
        comparisonEl.appendChild(oldEl);

        // New value
        const newEl = document.createElement("div");
        newEl.className = "bg-green-900/20 border border-green-700/30 rounded p-2";
        newEl.innerHTML = `
            <p class="text-green-400 font-semibold mb-1">Current</p>
            <code class="text-green-300 text-xs break-all">${JSON.stringify(newValue)}</code>
        `;
        comparisonEl.appendChild(newEl);

        container.appendChild(comparisonEl);
    }

    // Reason
    if (reason) {
        const reasonEl = document.createElement("div");
        reasonEl.className = "bg-blue-900/20 border-l-2 border-blue-600 px-3 py-2 rounded text-xs text-blue-300 mb-3";
        reasonEl.innerHTML = `<i class="bi bi-info-circle"></i> ${reason}`;
        container.appendChild(reasonEl);
    }

    // Timestamp footer
    const footer = document.createElement("div");
    footer.className = "flex items-center justify-between text-xs text-gray-500 pt-3 border-t border-gray-700";

    const timeEl = document.createElement("span");
    if (timestamp) {
        const date = new Date(timestamp);
        timeEl.textContent = date.toLocaleString([], {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
    } else {
        timeEl.textContent = "No timestamp";
    }
    footer.appendChild(timeEl);

    // Action indicator
    const actionIndicator = document.createElement("span");
    actionIndicator.className = "text-gray-600";
    actionIndicator.innerHTML = `<i class="bi bi-check-all"></i> ${action}`;
    footer.appendChild(actionIndicator);

    container.appendChild(footer);

    return container;
}

/**
 * Helper to create policy history timeline
 */
export function createPolicyHistoryTimeline(items = [], limit = null) {
    const timeline = document.createElement("div");
    timeline.className = "space-y-2";

    const itemsToShow = limit ? items.slice(0, limit) : items;

    itemsToShow.forEach((item) => {
        timeline.appendChild(PolicyHistoryItem(item));
    });

    // Show "Load more" button if items were limited
    if (limit && items.length > limit) {
        const loadMoreBtn = document.createElement("button");
        loadMoreBtn.className =
            "w-full py-2 text-sm text-blue-400 hover:text-blue-300 border-t border-gray-700 transition-colors";
        loadMoreBtn.textContent = `Load more (${items.length - limit} remaining)`;
        loadMoreBtn.onclick = () => {
            timeline.innerHTML = "";
            items.forEach((item) => {
                timeline.appendChild(PolicyHistoryItem(item));
            });
            loadMoreBtn.remove();
        };
        timeline.appendChild(loadMoreBtn);
    }

    return timeline;
}

/**
 * Helper to display policy comparison view
 */
export function createPolicyComparison(oldPolicy, newPolicy) {
    const container = document.createElement("div");
    container.className = "grid grid-cols-2 gap-4";

    // Old policy
    const oldEl = document.createElement("div");
    oldEl.className = "bg-red-900/10 border border-red-700/30 rounded-lg p-4";

    const oldTitle = document.createElement("h4");
    oldTitle.className = "text-red-400 font-semibold mb-3";
    oldTitle.textContent = "Previous Configuration";
    oldEl.appendChild(oldTitle);

    const oldContent = document.createElement("pre");
    oldContent.className = "text-xs text-red-300 overflow-auto max-h-96 font-mono";
    oldContent.textContent = JSON.stringify(oldPolicy, null, 2);
    oldEl.appendChild(oldContent);

    // New policy
    const newEl = document.createElement("div");
    newEl.className = "bg-green-900/10 border border-green-700/30 rounded-lg p-4";

    const newTitle = document.createElement("h4");
    newTitle.className = "text-green-400 font-semibold mb-3";
    newTitle.textContent = "New Configuration";
    newEl.appendChild(newTitle);

    const newContent = document.createElement("pre");
    newContent.className = "text-xs text-green-300 overflow-auto max-h-96 font-mono";
    newContent.textContent = JSON.stringify(newPolicy, null, 2);
    newEl.appendChild(newContent);

    container.appendChild(oldEl);
    container.appendChild(newEl);

    return container;
}
