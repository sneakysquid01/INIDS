/**
 * AlertCard Component - Display Individual Alerts
 * Shows alert details with severity badge, timestamps, and actions
 * 
 * Includes block IP action (fixes INT-002: broken block action)
 * 
 * Usage:
 *   AlertCard(alertData)
 *   container.appendChild(AlertCard(alert))
 */

import { UICard } from "./ui-card.js";
import { UIBadge } from "./ui-badge.js";
import { UIButton } from "./ui-button.js";
import { AppToast } from "./app-toast.js";
import { HttpClient_Instance as HttpClient } from "../core/http-client.js";
import { GlobalState } from "../core/global-state.js";

export function AlertCard(alert) {
    if (!alert) return document.createElement("div");

    // Helper to format timestamp
    const formatTime = (timestamp) => {
        if (!timestamp) return "N/A";
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    // Build alert details content
    const detailsEl = document.createElement("div");
    detailsEl.className = "space-y-3 text-sm";

    // Source/Destination
    if (alert.src_ip || alert.dst_ip) {
        const flowEl = document.createElement("div");
        flowEl.className = "flex items-center gap-2 text-gray-400";
        flowEl.innerHTML = `
            <span class="font-mono text-xs">${alert.src_ip || "??"}</span>
            <i class="bi bi-arrow-right text-blue-400"></i>
            <span class="font-mono text-xs">${alert.dst_ip || "??"}</span>
        `;
        detailsEl.appendChild(flowEl);
    }

    // Alert type/reason
    if (alert.alert_type || alert.reason) {
        const typeEl = document.createElement("p");
        typeEl.className = "text-gray-300";
        typeEl.textContent = `Type: ${alert.alert_type || alert.reason || "Unknown"}`;
        detailsEl.appendChild(typeEl);
    }

    // Confidence/Score
    if (alert.confidence || alert.score) {
        const scoreEl = document.createElement("div");
        scoreEl.className = "flex items-center gap-2";
        const scoreValue = alert.confidence || alert.score || 0;
        scoreEl.innerHTML = `
            <span class="text-gray-400 text-xs">Confidence:</span>
            <div class="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden max-w-[150px]">
                <div class="h-full bg-gradient-to-r from-blue-600 to-blue-400" 
                     style="width: ${scoreValue}%"></div>
            </div>
            <span class="text-gray-300 text-xs font-mono">${scoreValue}%</span>
        `;
        detailsEl.appendChild(scoreEl);
    }

    // Detection method
    if (alert.detection_method || alert.engine) {
        const methodEl = document.createElement("p");
        methodEl.className = "text-gray-400 text-xs";
        methodEl.textContent = `Detected by: ${alert.detection_method || alert.engine || "Unknown"}`;
        detailsEl.appendChild(methodEl);
    }

    // Additional context
    if (alert.context || alert.details) {
        const contextEl = document.createElement("p");
        contextEl.className = "text-gray-400 text-xs border-l-2 border-blue-600 pl-2";
        contextEl.textContent = alert.context || alert.details;
        detailsEl.appendChild(contextEl);
    }

    // Action buttons
    const handleBlockIP = async () => {
        try {
            AppToast.show("Blocking IP address...", {
                type: "info",
                duration: false,
            });

            // POST /api/actions with block action (FIXES INT-002)
            const response = await HttpClient.post("/api/actions", {
                alert_id: alert.id || alert.alert_id,
                type: "block",
                target_ip: alert.src_ip,
            });

            if (response.ok || response.status === 201) {
                AppToast.success(`Blocked IP: ${alert.src_ip}`);

                // Update GlobalState with new action
                GlobalState.push("actions", {
                    id: response.data?.id,
                    type: "block",
                    target: alert.src_ip,
                    timestamp: new Date().toISOString(),
                    status: "executed",
                });

                // Disable button after successful block
                blockBtn.disabled = true;
                blockBtn.textContent = "✓ Blocked";
            }
        } catch (error) {
            console.error("[AlertCard] Block action failed:", error);
            AppToast.error(`Failed to block IP: ${error.message}`);
        }
    };

    const handleDismiss = () => {
        // Remove card from DOM
        const card = document.querySelector(`[data-alert-id="${alert.id}"]`);
        if (card) {
            card.style.opacity = "0";
            setTimeout(() => card.remove(), 200);
        }
    };

    const blockBtn = UIButton.danger("Block IP", handleBlockIP, {
        size: "sm",
        icon: "shield-x",
    });

    const dismissBtn = UIButton.ghost("Dismiss", handleDismiss, {
        size: "sm",
    });

    // Build footer with timestamp
    const footerEl = document.createElement("div");
    footerEl.className = "flex items-center justify-between text-xs text-gray-400";
    footerEl.innerHTML = `<span>${formatTime(alert.timestamp || alert.created_at)}</span>`;
    footerEl.appendChild(dismissBtn);
    footerEl.appendChild(blockBtn);

    // Severity level
    const severity = alert.severity?.toLowerCase() || "info";

    // Create card with UICard component
    const card = UICard.create({
        icon: "exclamation-triangle",
        title: alert.title || alert.alert_type || "Security Alert",
        content: detailsEl,
        footer: footerEl,
        className: "transition-all duration-200 hover:shadow-lg",
    });

    // Add severity badge to header
    const header = card.querySelector("div");
    if (header) {
        const badge = UIBadge.severity(severity);
        const titleContainer = header.querySelector("div:first-child");
        if (titleContainer) {
            titleContainer.appendChild(badge);
        }
    }

    // Store alert ID for reference
    card.setAttribute("data-alert-id", alert.id || alert.alert_id || "unknown");

    return card;
}

