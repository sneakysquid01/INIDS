/**
 * ModuleCard Component - Display Capability Modules
 * Shows module status, health, and module-specific data
 * Includes error boundary to catch module loading failures (fixes STRUCT-001)
 * 
 * Usage:
 *   ModuleCard(moduleId, moduleConfig)
 */

import { UICard } from "./ui-card.js";
import { UIBadge } from "./ui-badge.js";
import { AppToast } from "./app-toast.js";
import { HttpClient_Instance as HttpClient } from "../core/http-client.js";
import { GlobalState } from "../core/global-state.js";
import { LoadingSpinner } from "./loading-spinner.js";

export function ModuleCard(moduleId, config = {}) {
    const {
        title = moduleId,
        description = "",
        endpoint = null,
        refreshInterval = 5000,
    } = config;

    // Build content container
    const contentEl = document.createElement("div");
    contentEl.className = "space-y-3";

    // Description
    if (description) {
        const descEl = document.createElement("p");
        descEl.className = "text-sm text-gray-400";
        descEl.textContent = description;
        contentEl.appendChild(descEl);
    }

    // Module status badge
    const statusEl = document.createElement("div");
    statusEl.className = "flex items-center gap-2";
    statusEl.appendChild(UIBadge.status("loading"));
    contentEl.appendChild(statusEl);

    // Data container (populated by fetchModuleData)
    const dataEl = document.createElement("div");
    dataEl.className = "space-y-2 text-sm";
    dataEl.setAttribute("data-module-data", "true");
    contentEl.appendChild(dataEl);

    // Refresh button
    const refreshBtn = document.createElement("button");
    refreshBtn.className =
        "text-xs text-blue-400 hover:text-blue-300 transition-colors px-2 py-1 hover:bg-gray-800 rounded";
    refreshBtn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Refresh';
    refreshBtn.onclick = async () => {
        refreshBtn.disabled = true;
        await loadModuleData();
        refreshBtn.disabled = false;
    };

    // Action buttons
    const actions = [refreshBtn];

    // Create card
    const card = UICard.create({
        title,
        icon: "plugin2",
        content: contentEl,
        actions,
        className: "hover:shadow-md transition-all",
    });

    // Error boundary wrapper
    let isErrored = false;

    /**
     * Load module data with error boundary
     */
    const loadModuleData = async () => {
        if (isErrored) return;

        try {
            statusEl.innerHTML = "";
            statusEl.appendChild(UIBadge.status("loading"));

            if (endpoint) {
                // Fetch module-specific data
                const response = await HttpClient.get(endpoint, {
                    timeout: 5000,
                    attempts: 1,
                });

                if (response.ok || response.status === 200) {
                    const data = response.data || response;

                    // Update module data display
                    dataEl.innerHTML = "";

                    if (typeof data === "object" && data !== null) {
                        Object.entries(data).forEach(([key, value]) => {
                            const row = document.createElement("div");
                            row.className = "flex justify-between";

                            const keyEl = document.createElement("span");
                            keyEl.className = "text-gray-400";
                            keyEl.textContent = key;

                            const valueEl = document.createElement("span");
                            valueEl.className = "text-white font-mono text-xs";
                            valueEl.textContent =
                                typeof value === "object"
                                    ? JSON.stringify(value)
                                    : String(value);

                            row.appendChild(keyEl);
                            row.appendChild(valueEl);
                            dataEl.appendChild(row);
                        });
                    } else {
                        dataEl.textContent = String(data);
                    }

                    statusEl.innerHTML = "";
                    statusEl.appendChild(UIBadge.status("active"));
                }
            } else {
                // No endpoint - subscribe to GlobalState slice
                const slice = moduleId.replace(/-/g, "_");
                GlobalState.subscribe(slice, (data) => {
                    dataEl.innerHTML = "";

                    if (typeof data === "object" && data !== null) {
                        Object.entries(data).forEach(([key, value]) => {
                            const row = document.createElement("div");
                            row.className = "flex justify-between";

                            const keyEl = document.createElement("span");
                            keyEl.className = "text-gray-400";
                            keyEl.textContent = key;

                            const valueEl = document.createElement("span");
                            valueEl.className = "text-white font-mono text-xs";
                            valueEl.textContent =
                                typeof value === "object"
                                    ? JSON.stringify(value)
                                    : String(value);

                            row.appendChild(keyEl);
                            row.appendChild(valueEl);
                            dataEl.appendChild(row);
                        });
                    }
                });

                statusEl.innerHTML = "";
                statusEl.appendChild(UIBadge.status("active"));
            }
        } catch (error) {
            // Error boundary: catch and display error gracefully
            isErrored = true;
            console.error(`[ModuleCard] Error loading ${moduleId}:`, error);

            statusEl.innerHTML = "";
            statusEl.appendChild(UIBadge.status("failed"));

            dataEl.innerHTML = `
                <div class="text-red-400 text-xs p-2 bg-red-900/20 rounded border border-red-700/30">
                    <p class="font-semibold mb-1">Error loading module</p>
                    <p>${error.message}</p>
                </div>
            `;

            // Log to AppToast with option to retry
            console.warn(
                `[ModuleCard] ${moduleId} failed - user can manually refresh or check console`
            );
        }
    };

    // Load module data when card is mounted
    setTimeout(() => loadModuleData(), 100);

    // Set up auto-refresh if interval provided
    if (refreshInterval > 0) {
        const refreshTimer = setInterval(() => {
            if (!isErrored) {
                loadModuleData();
            }
        }, refreshInterval);

        // Cleanup timer when card is removed
        card.addEventListener("remove", () => {
            clearInterval(refreshTimer);
        });
    }

    // Store reference to error state
    card.isErrored = () => isErrored;
    card.reload = loadModuleData;

    return card;
}

/**
 * Helper to create module grid
 */
export function createModuleGrid(modules = {}, columns = 3) {
    const grid = document.createElement("div");
    grid.className = `grid grid-cols-${columns} gap-4`;

    Object.entries(modules).forEach(([moduleId, config]) => {
        grid.appendChild(ModuleCard(moduleId, config));
    });

    return grid;
}

