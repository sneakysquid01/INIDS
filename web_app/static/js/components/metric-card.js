/**
 * MetricCard Component - Display System Metrics
 * Shows metric values with sparklines, status indicators, and comparisons
 * 
 * Usage:
 *   MetricCard({ label: "Alerts/min", value: 42, max: 100, trend: 5 })
 */

import { UICard } from "./ui-card.js";

export function MetricCard(metric) {
    if (!metric) return document.createElement("div");

    const {
        label = "Metric",
        value = 0,
        unit = "",
        max = 100,
        threshold = null,
        trend = null,
        sparkline = [],
        status = "normal", // normal, warning, critical
    } = metric;

    // Build content
    const contentEl = document.createElement("div");
    contentEl.className = "space-y-2";

    // Value display
    const valueContainer = document.createElement("div");
    valueContainer.className = "flex items-baseline gap-1";

    const valueEl = document.createElement("span");
    valueEl.className = "text-3xl font-bold text-white";
    valueEl.textContent = typeof value === "number" ? value.toLocaleString() : value;
    valueContainer.appendChild(valueEl);

    if (unit) {
        const unitEl = document.createElement("span");
        unitEl.className = "text-gray-400 text-sm";
        unitEl.textContent = unit;
        valueContainer.appendChild(unitEl);
    }

    contentEl.appendChild(valueContainer);

    // Status bar (if max provided)
    if (max && typeof max === "number") {
        const barContainer = document.createElement("div");
        barContainer.className = "w-full h-2 bg-gray-700 rounded-full overflow-hidden";

        const bar = document.createElement("div");
        const percent = Math.min((value / max) * 100, 100);

        // Determine color based on status
        const colors = {
            normal: "bg-blue-500",
            warning: "bg-amber-500",
            critical: "bg-red-500",
        };

        bar.className = `h-full ${colors[status]} transition-all duration-300`;
        bar.style.width = `${percent}%`;
        barContainer.appendChild(bar);
        contentEl.appendChild(barContainer);

        // Percentage text
        const percentText = document.createElement("p");
        percentText.className = "text-xs text-gray-400";
        percentText.textContent = `${Math.round(percent)}% of ${max}${unit}`;
        contentEl.appendChild(percentText);
    }

    // Trend indicator
    if (trend !== null && trend !== undefined) {
        const trendEl = document.createElement("div");
        const trendDir = trend > 0 ? "↑" : trend < 0 ? "↓" : "→";
        const trendColor =
            trend > 0 ? "text-red-400" : trend < 0 ? "text-green-400" : "text-gray-400";

        trendEl.className = `flex items-center gap-1 ${trendColor} text-sm font-medium`;
        trendEl.innerHTML = `<span>${trendDir}</span> <span>${Math.abs(trend)}% vs last period</span>`;
        contentEl.appendChild(trendEl);
    }

    // Sparkline (simple bar chart)
    if (Array.isArray(sparkline) && sparkline.length > 0) {
        const sparklineContainer = document.createElement("div");
        sparklineContainer.className = "flex items-end gap-1 h-12";

        const maxSparkValue = Math.max(...sparkline);
        sparkline.forEach((val) => {
            const bar = document.createElement("div");
            const height = (val / maxSparkValue) * 100;
            bar.className = "flex-1 bg-blue-600/50 hover:bg-blue-500 rounded-t transition-colors cursor-pointer";
            bar.style.height = `${height}%`;
            bar.title = `${val}`;
            sparklineContainer.appendChild(bar);
        });

        contentEl.appendChild(sparklineContainer);
    }

    // Threshold indicator
    if (threshold !== null && threshold !== undefined) {
        const thresholdEl = document.createElement("p");
        const isAbove = value > threshold;
        thresholdEl.className = `text-xs ${isAbove ? "text-red-400" : "text-green-400"}`;
        thresholdEl.textContent = `Threshold: ${threshold}${unit} ${isAbove ? "⚠️ EXCEEDED" : "✓ OK"}`;
        contentEl.appendChild(thresholdEl);
    }

    // Create card
    const card = UICard.create({
        title: label,
        icon: "graph-up",
        content: contentEl,
        className: `
            hover:shadow-md transition-shadow
            ${status === "critical" ? "border-red-700/50" : ""}
            ${status === "warning" ? "border-amber-700/50" : ""}
        `,
    });

    return card;
}

/**
 * Helper to create metric data from raw data
 */
export function createMetricData(source, mapping = {}) {
    return {
        label: mapping.label || "Metric",
        value: source[mapping.value_key || "value"],
        unit: mapping.unit || "",
        max: source[mapping.max_key] || 100,
        threshold: source[mapping.threshold_key],
        trend: source[mapping.trend_key],
        sparkline: source[mapping.sparkline_key] || [],
        status: source[mapping.status_key] || "normal",
    };
}
