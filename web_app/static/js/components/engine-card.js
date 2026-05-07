/**
 * EngineCard Component - Display Detection Engines
 * Shows engine status, load, and performance metrics
 * 
 * Usage:
 *   EngineCard({ name: "DL-Engine", status: "active", load: 45 })
 */

import { UICard } from "./ui-card.js";
import { UIBadge } from "./ui-badge.js";

export function EngineCard(engine) {
    if (!engine) return document.createElement("div");

    const {
        name = "Engine",
        id = null,
        status = "unknown",
        load = 0,
        accuracy = null,
        model = null,
        version = null,
        lastUpdate = null,
        detections = 0,
    } = engine;

    // Build content
    const contentEl = document.createElement("div");
    contentEl.className = "space-y-3";

    // Status row
    const statusRow = document.createElement("div");
    statusRow.className = "flex items-center justify-between";

    const statusLabel = document.createElement("span");
    statusLabel.className = "text-gray-400 text-sm";
    statusLabel.textContent = "Status:";

    const statusBadge = UIBadge.status(status);
    statusRow.appendChild(statusLabel);
    statusRow.appendChild(statusBadge);
    contentEl.appendChild(statusRow);

    // Load indicator
    if (load !== null && load !== undefined) {
        const loadRow = document.createElement("div");
        loadRow.className = "space-y-1";

        const loadLabel = document.createElement("p");
        loadLabel.className = "text-gray-400 text-xs";
        loadLabel.textContent = "Load";

        const loadContainer = document.createElement("div");
        loadContainer.className = "flex items-center gap-2";

        const loadBar = document.createElement("div");
        loadBar.className = "flex-1 h-2 bg-gray-700 rounded-full overflow-hidden";

        const loadFill = document.createElement("div");
        const loadColor =
            load > 80
                ? "bg-red-500"
                : load > 60
                ? "bg-amber-500"
                : load > 40
                ? "bg-yellow-500"
                : "bg-green-500";
        loadFill.className = `h-full ${loadColor} transition-all duration-300`;
        loadFill.style.width = `${Math.min(load, 100)}%`;
        loadBar.appendChild(loadFill);

        const loadText = document.createElement("span");
        loadText.className = "text-gray-300 text-xs font-mono w-10 text-right";
        loadText.textContent = `${load}%`;

        loadContainer.appendChild(loadBar);
        loadContainer.appendChild(loadText);

        loadRow.appendChild(loadLabel);
        loadRow.appendChild(loadContainer);
        contentEl.appendChild(loadRow);
    }

    // Accuracy (if available)
    if (accuracy !== null && accuracy !== undefined) {
        const accuracyRow = document.createElement("div");
        accuracyRow.className = "flex items-center justify-between text-sm";

        const accuracyLabel = document.createElement("span");
        accuracyLabel.className = "text-gray-400";
        accuracyLabel.textContent = "Accuracy:";

        const accuracyValue = document.createElement("span");
        const accuracyColor =
            accuracy > 95
                ? "text-green-400"
                : accuracy > 90
                ? "text-yellow-400"
                : "text-red-400";
        accuracyValue.className = `font-mono ${accuracyColor}`;
        accuracyValue.textContent = `${accuracy.toFixed(1)}%`;

        accuracyRow.appendChild(accuracyLabel);
        accuracyRow.appendChild(accuracyValue);
        contentEl.appendChild(accuracyRow);
    }

    // Model info
    if (model || version) {
        const modelRow = document.createElement("div");
        modelRow.className = "text-xs text-gray-500 space-y-1";

        if (model) {
            const modelEl = document.createElement("p");
            modelEl.textContent = `Model: ${model}`;
            modelRow.appendChild(modelEl);
        }

        if (version) {
            const versionEl = document.createElement("p");
            versionEl.textContent = `Version: ${version}`;
            modelRow.appendChild(versionEl);
        }

        contentEl.appendChild(modelRow);
    }

    // Last update
    if (lastUpdate) {
        const updateRow = document.createElement("p");
        updateRow.className = "text-xs text-gray-500";
        const date = new Date(lastUpdate);
        updateRow.textContent = `Updated: ${date.toLocaleString()}`;
        contentEl.appendChild(updateRow);
    }

    // Detections counter (if available)
    if (detections > 0) {
        const detectRow = document.createElement("div");
        detectRow.className = "flex items-center gap-2 pt-2 border-t border-gray-700";

        const detectIcon = document.createElement("i");
        detectIcon.className = "bi bi-shield-check text-blue-400";

        const detectText = document.createElement("span");
        detectText.className = "text-sm text-gray-300";
        detectText.textContent = `${detections.toLocaleString()} detections`;

        detectRow.appendChild(detectIcon);
        detectRow.appendChild(detectText);
        contentEl.appendChild(detectRow);
    }

    // Create card
    const card = UICard.create({
        title: name,
        icon: "cpu",
        content: contentEl,
        className: "hover:shadow-md transition-all",
    });

    // Add ID attribute for reference
    if (id) {
        card.setAttribute("data-engine-id", id);
    }

    return card;
}

/**
 * Helper to create engine grid
 */
export function createEngineGrid(engines = [], columns = 3) {
    const grid = document.createElement("div");
    grid.className = `grid grid-cols-${columns} gap-4`;

    engines.forEach((engine) => {
        grid.appendChild(EngineCard(engine));
    });

    return grid;
}

