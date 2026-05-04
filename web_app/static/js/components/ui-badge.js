/**
 * UIBadge Component - Status/Severity Badges
 * Small labels for status indicators, severity levels, tags
 * 
 * Variants: threat (red), warn (amber), safe (green), info (blue), neutral (gray)
 * Sizes: sm, md, lg
 * 
 * Usage:
 *   UIBadge.create("Critical", { variant: "threat" })
 *   UIBadge.severity("high")
 *   UIBadge.status("active")
 */

export const UIBadge = {
    /**
     * Create a badge element
     */
    create(label, options = {}) {
        const {
            variant = "neutral",
            size = "md",
            icon = null,
            rounded = true,
            className = "",
        } = options;

        const badge = document.createElement("span");

        // Variant classes
        const variants = {
            threat: "bg-red-900/40 text-red-300 border border-red-700/50",
            warn: "bg-amber-900/40 text-amber-300 border border-amber-700/50",
            safe: "bg-green-900/40 text-green-300 border border-green-700/50",
            info: "bg-blue-900/40 text-blue-300 border border-blue-700/50",
            neutral: "bg-gray-700/40 text-gray-300 border border-gray-600/50",
        };

        // Size classes
        const sizes = {
            sm: "px-2 py-0.5 text-xs font-medium",
            md: "px-2.5 py-1 text-sm font-medium",
            lg: "px-3 py-1.5 text-base font-medium",
        };

        // Build class string
        const roundClass = rounded ? "rounded-full" : "rounded";
        badge.className = `inline-flex items-center gap-1 ${variants[variant]} ${sizes[size]} ${roundClass} ${className}`;

        // Add icon if provided
        if (icon) {
            const iconEl = document.createElement("i");
            iconEl.className = `bi bi-${icon} text-xs`;
            badge.appendChild(iconEl);
        }

        // Add label
        const labelEl = document.createElement("span");
        labelEl.textContent = label;
        badge.appendChild(labelEl);

        return badge;
    },

    /**
     * Create severity badge
     * Levels: critical, high, medium, low, info
     */
    severity(level) {
        const config = {
            critical: { variant: "threat", icon: "exclamation-octagon", label: "Critical" },
            high: { variant: "threat", icon: "exclamation-circle", label: "High" },
            medium: { variant: "warn", icon: "exclamation-triangle", label: "Medium" },
            low: { variant: "info", icon: "info-circle", label: "Low" },
            info: { variant: "info", icon: "info-circle", label: "Info" },
        };

        const config_item = config[level] || config.info;
        return this.create(config_item.label, {
            variant: config_item.variant,
            icon: config_item.icon,
            size: "sm",
        });
    },

    /**
     * Create status badge
     * Statuses: active, inactive, pending, completed, failed, blocked
     */
    status(status) {
        const config = {
            active: { variant: "safe", icon: "check-circle", label: "Active" },
            inactive: { variant: "neutral", icon: "dash-circle", label: "Inactive" },
            pending: { variant: "warn", icon: "hourglass-split", label: "Pending" },
            completed: { variant: "safe", icon: "check-circle", label: "Completed" },
            failed: { variant: "threat", icon: "x-circle", label: "Failed" },
            blocked: { variant: "threat", icon: "shield-x", label: "Blocked" },
            running: { variant: "info", icon: "play-circle", label: "Running" },
            paused: { variant: "warn", icon: "pause-circle", label: "Paused" },
        };

        const config_item = config[status] || config.inactive;
        return this.create(config_item.label, {
            variant: config_item.variant,
            icon: config_item.icon,
            size: "sm",
        });
    },

    /**
     * Create count badge (pill-style)
     */
    count(count, options = {}) {
        const { variant = "info", max = 99 } = options;

        const displayCount = count > max ? `${max}+` : count;
        return this.create(displayCount, {
            variant,
            size: "sm",
            rounded: true,
            className: "min-w-[24px] justify-center",
        });
    },

    /**
     * Create tag badge
     */
    tag(label, options = {}) {
        const { variant = "neutral", removable = false, onRemove = null } = options;

        const badge = this.create(label, {
            variant,
            size: "sm",
            className: "cursor-default",
        });

        if (removable && onRemove) {
            const closeBtn = document.createElement("button");
            closeBtn.className = "ml-1 hover:opacity-70 transition-opacity flex-shrink-0";
            closeBtn.innerHTML = '<i class="bi bi-x-lg text-xs"></i>';
            closeBtn.onclick = (e) => {
                e.stopPropagation();
                onRemove();
            };
            badge.appendChild(closeBtn);
        }

        return badge;
    },

    /**
     * Create variant badges (all together for reference)
     */
    allVariants(label = "Label") {
        const container = document.createElement("div");
        container.className = "flex flex-wrap gap-2";

        ["threat", "warn", "safe", "info", "neutral"].forEach((variant) => {
            container.appendChild(this.create(label, { variant, size: "sm" }));
        });

        return container;
    },

    /**
     * Create enum-based badge (maps value to label/variant)
     */
    enum(value, mapping = {}) {
        const config = mapping[value] || {
            label: String(value).toUpperCase(),
            variant: "neutral",
        };

        return this.create(config.label, {
            variant: config.variant,
            icon: config.icon,
            size: "sm",
        });
    },

    /**
     * Create threat level badge (0-100 scale)
     */
    threatLevel(score) {
        let variant, label;
        if (score >= 80) {
            variant = "threat";
            label = "Critical";
        } else if (score >= 60) {
            variant = "threat";
            label = "High";
        } else if (score >= 40) {
            variant = "warn";
            label = "Medium";
        } else if (score >= 20) {
            variant = "info";
            label = "Low";
        } else {
            variant = "safe";
            label = "Safe";
        }

        return this.create(`${label} (${score}%)`, {
            variant,
            size: "sm",
        });
    },
};
