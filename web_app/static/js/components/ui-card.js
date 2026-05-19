/**
 * UICard Component - Reusable Card Container
 * Base component for building panels, data displays, and sections
 * 
 * Features: header, content, footer, hover effects, clickable, collapsible
 * 
 * Usage:
 *   UICard.create({ title: "Title", content: "Content" })
 *   UICard.panel("Section Title")
 */

export const UICard = {
    /**
     * Create a card element
     */
    create(options = {}) {
        const {
            title = "",
            content = "",
            footer = "",
            icon = null,
            actions = [],
            clickable = false,
            onClick = null,
            collapsible = false,
            collapsed = false,
            className = "",
            headerClass = "",
        } = options;

        const card = document.createElement("div");
        card.className = `
            bg-[#151922] border border-[#1a1f2e]
            rounded-lg overflow-hidden
            transition-all duration-200
            ${clickable ? "cursor-pointer hover:border-[#2a2f3e] hover:shadow-lg" : ""}
            ${className}
        `;

        // Header
        if (title || icon || actions.length > 0) {
            const header = document.createElement("div");
            header.className = `
                bg-[#0f1117] border-b border-[#1a1f2e]
                px-4 py-3 flex items-center justify-between
                ${headerClass}
            `;

            // Title with icon
            const titleContainer = document.createElement("div");
            titleContainer.className = "flex items-center gap-2 flex-1";

            if (icon) {
                const iconEl = document.createElement("i");
                iconEl.className = `bi bi-${icon} text-lg text-blue-400`;
                titleContainer.appendChild(iconEl);
            }

            if (title) {
                const titleEl = document.createElement("h3");
                titleEl.className = "text-sm font-semibold text-white";
                titleEl.textContent = title;
                titleContainer.appendChild(titleEl);
            }

            header.appendChild(titleContainer);

            // Actions/Buttons
            if (actions.length > 0) {
                const actionsContainer = document.createElement("div");
                actionsContainer.className = "flex items-center gap-2";
                actions.forEach((action) => {
                    actionsContainer.appendChild(action);
                });
                header.appendChild(actionsContainer);
            }

            // Collapse button
            if (collapsible) {
                const collapseBtn = document.createElement("button");
                collapseBtn.className =
                    "text-gray-400 hover:text-white transition-colors p-1 ml-2";
                collapseBtn.innerHTML = `<i class="bi bi-chevron-down transition-transform duration-200"></i>`;

                const toggleCollapse = () => {
                    const isCollapsed = content.classList.toggle("hidden");
                    const icon = collapseBtn.querySelector("i");
                    if (isCollapsed) {
                        icon.style.transform = "rotate(-90deg)";
                    } else {
                        icon.style.transform = "rotate(0deg)";
                    }
                };

                collapseBtn.onclick = toggleCollapse;
                header.appendChild(collapseBtn);
            }

            card.appendChild(header);
        }

        // Content
        const contentEl = document.createElement("div");
        contentEl.className = "px-4 py-4 text-gray-300";
        if (typeof content === "string") {
            contentEl.innerHTML = content;
        } else if (content instanceof HTMLElement) {
            contentEl.appendChild(content);
        }

        if (collapsed) {
            contentEl.classList.add("hidden");
        }

        card.appendChild(contentEl);

        // Footer
        if (footer) {
            const footerEl = document.createElement("div");
            footerEl.className = "bg-[#0f1117] border-t border-[#1a1f2e] px-4 py-3 text-gray-400 text-sm";
            if (typeof footer === "string") {
                footerEl.innerHTML = footer;
            } else if (footer instanceof HTMLElement) {
                footerEl.appendChild(footer);
            }
            card.appendChild(footerEl);
        }

        // Click handler
        if (clickable && onClick) {
            card.addEventListener("click", onClick);
        }

        card.contentEl = contentEl; // Expose for updates
        return card;
    },

    /**
     * Create a simple panel (card with just title and content)
     */
    panel(title, content, options = {}) {
        return this.create({
            title,
            content,
            icon: options.icon,
            ...options,
        });
    },

    /**
     * Create a stat card (for displaying metrics)
     */
    stat(label, value, options = {}) {
        const { unit = "", trend = null, icon = "graph-up", trendClass = "" } = options;

        const content = document.createElement("div");
        content.className = "text-center";

        const labelEl = document.createElement("p");
        labelEl.className = "text-gray-400 text-sm mb-2";
        labelEl.textContent = label;
        content.appendChild(labelEl);

        const valueContainer = document.createElement("div");
        valueContainer.className = "flex items-baseline justify-center gap-2";

        const valueEl = document.createElement("p");
        valueEl.className = "text-2xl font-bold text-white";
        valueEl.textContent = typeof value === "number" ? value.toLocaleString() : value;
        valueContainer.appendChild(valueEl);

        if (unit) {
            const unitEl = document.createElement("span");
            unitEl.className = "text-gray-400 text-sm";
            unitEl.textContent = unit;
            valueContainer.appendChild(unitEl);
        }

        if (trend) {
            const trendEl = document.createElement("span");
            const trendDir = trend > 0 ? "↑" : trend < 0 ? "↓" : "→";
            const trendColor =
                trend > 0
                    ? "text-green-400"
                    : trend < 0
                    ? "text-red-400"
                    : "text-gray-400";
            trendEl.className = `ml-2 ${trendColor} text-sm font-semibold`;
            trendEl.textContent = `${trendDir} ${Math.abs(trend)}%`;
            valueContainer.appendChild(trendEl);
        }

        content.appendChild(valueContainer);

        return this.create({
            icon,
            content,
            ...options,
        });
    },

    /**
     * Create a horizontal divider card
     */
    divider(label = "") {
        const card = document.createElement("div");
        card.className = "px-4 py-3 text-center text-gray-500 text-xs";

        if (label) {
            card.innerHTML = `<span class="bg-surface-100 px-2">${label}</span>`;
            card.className += " ui-card-divider";
        } else {
            card.className = "h-px bg-[#1a1f2e]";
        }

        return card;
    },

    /**
     * Create a grid of cards
     */
    grid(cards, columns = 3) {
        const grid = document.createElement("div");
        grid.className = `grid grid-cols-${columns} gap-4`;
        cards.forEach((card) => grid.appendChild(card));
        return grid;
    },

    /**
     * Create a list item card (for use in card-based lists)
     */
    listItem(options = {}) {
        const {
            primary = "",
            secondary = "",
            icon = null,
            avatar = null,
            badge = null,
            actions = [],
            onClick = null,
        } = options;

        const item = document.createElement("div");
        item.className = `
            flex items-center gap-3 p-3 border-b border-[#1a1f2e] last:border-b-0
            hover:bg-[#1a1f2e]/50 transition-colors
            ${onClick ? "cursor-pointer" : ""}
        `;

        if (onClick) {
            item.addEventListener("click", onClick);
        }

        // Avatar/Icon
        if (avatar || icon) {
            const avatarContainer = document.createElement("div");
            if (avatar) {
                avatarContainer.innerHTML = `<img src="${avatar}" class="w-8 h-8 rounded-full" />`;
            } else {
                avatarContainer.innerHTML = `<i class="bi bi-${icon} text-lg text-blue-400"></i>`;
            }
            item.appendChild(avatarContainer);
        }

        // Content
        const content = document.createElement("div");
        content.className = "flex-1";

        const primaryEl = document.createElement("p");
        primaryEl.className = "text-sm font-medium text-white";
        primaryEl.textContent = primary;
        content.appendChild(primaryEl);

        if (secondary) {
            const secondaryEl = document.createElement("p");
            secondaryEl.className = "text-xs text-gray-400";
            secondaryEl.textContent = secondary;
            content.appendChild(secondaryEl);
        }

        item.appendChild(content);

        // Badge
        if (badge) {
            item.appendChild(badge);
        }

        // Actions
        if (actions.length > 0) {
            const actionsContainer = document.createElement("div");
            actionsContainer.className = "flex items-center gap-2";
            actions.forEach((action) => {
                actionsContainer.appendChild(action);
            });
            item.appendChild(actionsContainer);
        }

        return item;
    },

    /**
     * Update card content dynamically
     */
    updateContent(card, newContent) {
        if (card.contentEl) {
            if (typeof newContent === "string") {
                card.contentEl.innerHTML = newContent;
            } else if (newContent instanceof HTMLElement) {
                card.contentEl.innerHTML = "";
                card.contentEl.appendChild(newContent);
            }
        }
    },

    /**
     * Toggle loading state
     */
    setLoading(card, isLoading) {
        const contentEl = card.contentEl;
        if (!contentEl) return;

        if (isLoading) {
            contentEl.innerHTML = `
                <div class="flex justify-center py-8">
                    <i class="bi bi-hourglass-split animate-spin text-blue-400 text-2xl"></i>
                </div>
            `;
        }
    },

    /**
     * Show error state
     */
    setError(card, errorMessage) {
        const contentEl = card.contentEl;
        if (!contentEl) return;

        contentEl.innerHTML = `
            <div class="text-center text-red-400 py-4">
                <i class="bi bi-exclamation-circle text-2xl mb-2 block"></i>
                <p class="text-sm">${errorMessage}</p>
            </div>
        `;
    },
};
