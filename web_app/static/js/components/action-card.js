/**
 * ActionCard Component - Display Action History Entries
 * Shows individual security action records (blocks, rate limits, etc.)
 * 
 * Usage:
 *   ActionCard(actionData)
 */

import { UICard } from "./ui-card.js";
import { UIBadge } from "./ui-badge.js";

export function ActionCard(action) {
    if (!action) return document.createElement("div");

    const {
        id = null,
        type = "unknown", // block, rate_limit, temp_block, alert, investigate
        target = "N/A",
        status = "executed", // pending, executed, failed, rolled_back
        reason = "",
        timestamp = null,
        duration = null,
        executor = "System",
        result = null,
    } = action;

    // Build content
    const contentEl = document.createElement("div");
    contentEl.className = "space-y-2 text-sm";

    // Action type badge
    const typeEl = document.createElement("div");
    typeEl.className = "flex items-center gap-2";

    const typeLabel = document.createElement("span");
    typeLabel.className = "text-gray-400";
    typeLabel.textContent = "Action:";

    const typeBadge = UIBadge.enum(type, {
        block: { label: "Block", variant: "threat", icon: "shield-x" },
        rate_limit: { label: "Rate Limit", variant: "warn", icon: "hourglass-split" },
        temp_block: { label: "Temporary Block", variant: "warn", icon: "clock" },
        alert: { label: "Alert", variant: "info", icon: "bell" },
        investigate: { label: "Investigate", variant: "info", icon: "search" },
    });

    typeEl.appendChild(typeLabel);
    typeEl.appendChild(typeBadge);
    contentEl.appendChild(typeEl);

    // Target
    const targetEl = document.createElement("div");
    targetEl.className = "flex items-start justify-between";

    const targetLabel = document.createElement("span");
    targetLabel.className = "text-gray-400";
    targetLabel.textContent = "Target:";

    const targetValue = document.createElement("span");
    targetValue.className = "text-gray-300 font-mono text-xs text-right";
    targetValue.textContent = target;

    targetEl.appendChild(targetLabel);
    targetEl.appendChild(targetValue);
    contentEl.appendChild(targetEl);

    // Status badge
    const statusEl = document.createElement("div");
    statusEl.className = "flex items-center gap-2";

    const statusLabel = document.createElement("span");
    statusLabel.className = "text-gray-400";
    statusLabel.textContent = "Status:";

    const statusBadge = UIBadge.status(status);
    statusEl.appendChild(statusLabel);
    statusEl.appendChild(statusBadge);
    contentEl.appendChild(statusEl);

    // Reason (if provided)
    if (reason) {
        const reasonEl = document.createElement("div");
        reasonEl.className = "bg-gray-800/50 border-l-2 border-blue-600 px-2 py-1 rounded text-xs text-gray-400";
        reasonEl.textContent = reason;
        contentEl.appendChild(reasonEl);
    }

    // Duration (if applicable)
    if (duration) {
        const durationEl = document.createElement("p");
        durationEl.className = "text-xs text-gray-500";

        let durationText = "";
        if (typeof duration === "number") {
            const seconds = duration;
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;

            if (hours > 0) {
                durationText = `${hours}h ${minutes}m`;
            } else if (minutes > 0) {
                durationText = `${minutes}m ${secs}s`;
            } else {
                durationText = `${secs}s`;
            }
        } else {
            durationText = String(duration);
        }

        durationEl.textContent = `Duration: ${durationText}`;
        contentEl.appendChild(durationEl);
    }

    // Result (if available)
    if (result) {
        const resultEl = document.createElement("div");
        resultEl.className = "text-xs text-gray-500 space-y-1";

        if (typeof result === "object") {
            Object.entries(result).forEach(([key, value]) => {
                const row = document.createElement("p");
                row.textContent = `${key}: ${value}`;
                resultEl.appendChild(row);
            });
        } else {
            resultEl.textContent = String(result);
        }

        contentEl.appendChild(resultEl);
    }

    // Executor info
    if (executor) {
        const executorEl = document.createElement("p");
        executorEl.className = "text-xs text-gray-600 pt-2 border-t border-gray-700";
        executorEl.innerHTML = `<i class="bi bi-person-check text-gray-500"></i> ${executor}`;
        contentEl.appendChild(executorEl);
    }

    // Build footer with timestamp
    let footerText = "";
    if (timestamp) {
        const date = new Date(timestamp);
        footerText = date.toLocaleString([], {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    }

    // Create card
    const card = UICard.create({
        icon: "shield-check",
        title: `${type.toUpperCase()} - ${target}`,
        content: contentEl,
        footer: footerText ? `<span class="text-gray-500">${footerText}</span>` : "",
        className: "hover:shadow-md transition-all",
    });

    // Add action ID attribute
    if (id) {
        card.setAttribute("data-action-id", id);
    }

    return card;
}

/**
 * Helper to create action timeline view
 */
export function createActionTimeline(actions = []) {
    const timeline = document.createElement("div");
    timeline.className = "space-y-2";

    actions.forEach((action, index) => {
        timeline.appendChild(ActionCard(action));
    });

    return timeline;
}

