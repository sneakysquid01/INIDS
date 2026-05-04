/**
 * UIButton Component - Reusable Button Variants
 * Factory function for creating styled buttons with various variants
 * 
 * Variants: primary, secondary, danger, success, warning, outline, ghost
 * Sizes: sm, md, lg
 * States: normal, loading, disabled
 * 
 * Usage:
 *   UIButton.create("Click Me", { variant: "primary", size: "lg" })
 *   UIButton.create("Delete", { variant: "danger", onClick: handleDelete })
 */

export const UIButton = {
    /**
     * Create a button element
     */
    create(label, options = {}) {
        const {
            variant = "secondary",
            size = "md",
            icon = null,
            onClick = null,
            disabled = false,
            loading = false,
            className = "",
        } = options;

        const btn = document.createElement("button");

        // Base classes
        const baseClass =
            "font-medium rounded transition-all duration-200 flex items-center gap-2 justify-center";

        // Variant classes
        const variants = {
            primary:
                "bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-blue-500/50 disabled:bg-blue-900 disabled:cursor-not-allowed",
            secondary:
                "bg-gray-700 hover:bg-gray-600 text-white disabled:bg-gray-900 disabled:cursor-not-allowed",
            danger:
                "bg-red-600 hover:bg-red-700 text-white shadow-lg hover:shadow-red-500/50 disabled:bg-red-900 disabled:cursor-not-allowed",
            success:
                "bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-green-500/50 disabled:bg-green-900 disabled:cursor-not-allowed",
            warning:
                "bg-amber-600 hover:bg-amber-700 text-white shadow-lg hover:shadow-amber-500/50 disabled:bg-amber-900 disabled:cursor-not-allowed",
            outline:
                "border-2 border-gray-500 hover:border-gray-300 text-gray-300 hover:bg-gray-900/50 disabled:border-gray-700 disabled:text-gray-600 disabled:cursor-not-allowed",
            ghost:
                "text-gray-300 hover:bg-gray-800 hover:text-white disabled:text-gray-600 disabled:cursor-not-allowed",
        };

        // Size classes
        const sizes = {
            sm: "px-3 py-1.5 text-sm",
            md: "px-4 py-2 text-base",
            lg: "px-6 py-3 text-lg",
        };

        // Build final class string
        btn.className = `${baseClass} ${variants[variant]} ${sizes[size]} ${className}`;

        // Set disabled state
        if (disabled || loading) {
            btn.disabled = true;
        }

        // Add icon if provided
        if (icon) {
            const iconEl = document.createElement("i");
            iconEl.className = `bi bi-${icon}`;
            btn.appendChild(iconEl);
        }

        // Add label
        const labelEl = document.createElement("span");
        labelEl.textContent = label;
        btn.appendChild(labelEl);

        // Add loading spinner if loading
        if (loading) {
            const spinner = document.createElement("i");
            spinner.className = "bi bi-hourglass-split animate-spin ml-1";
            btn.appendChild(spinner);
        }

        // Add click handler
        if (onClick && !disabled && !loading) {
            btn.addEventListener("click", onClick);
        }

        return btn;
    },

    /**
     * Create a primary button (commonly used)
     */
    primary(label, onClick, options = {}) {
        return this.create(label, {
            variant: "primary",
            onClick,
            ...options,
        });
    },

    /**
     * Create a secondary button (commonly used)
     */
    secondary(label, onClick, options = {}) {
        return this.create(label, {
            variant: "secondary",
            onClick,
            ...options,
        });
    },

    /**
     * Create a danger button (commonly used)
     */
    danger(label, onClick, options = {}) {
        return this.create(label, {
            variant: "danger",
            onClick,
            ...options,
        });
    },

    /**
     * Create a success button (commonly used)
     */
    success(label, onClick, options = {}) {
        return this.create(label, {
            variant: "success",
            onClick,
            ...options,
        });
    },

    /**
     * Create an outline button
     */
    outline(label, onClick, options = {}) {
        return this.create(label, {
            variant: "outline",
            onClick,
            ...options,
        });
    },

    /**
     * Create a ghost button
     */
    ghost(label, onClick, options = {}) {
        return this.create(label, {
            variant: "ghost",
            onClick,
            ...options,
        });
    },

    /**
     * Update button loading state
     */
    setLoading(btn, isLoading) {
        const labelEl = btn.querySelector("span");
        if (isLoading) {
            btn.disabled = true;
            const spinner = document.createElement("i");
            spinner.className = "bi bi-hourglass-split animate-spin ml-1";
            btn.appendChild(spinner);
        } else {
            btn.disabled = false;
            const spinner = btn.querySelector(".animate-spin");
            if (spinner) spinner.remove();
        }
    },

    /**
     * Update button disabled state
     */
    setDisabled(btn, isDisabled) {
        btn.disabled = isDisabled;
    },

    /**
     * Create a button group container
     */
    group(buttons, options = {}) {
        const { direction = "horizontal", align = "left" } = options;

        const group = document.createElement("div");
        const dirClass = direction === "vertical" ? "flex flex-col" : "flex flex-row";
        const alignClass =
            align === "center"
                ? "justify-center"
                : align === "right"
                ? "justify-end"
                : "justify-start";

        group.className = `${dirClass} ${alignClass} gap-2`;
        buttons.forEach((btn) => group.appendChild(btn));

        return group;
    },

    /**
     * Create icon-only button
     */
    icon(iconName, onClick, options = {}) {
        const btn = document.createElement("button");
        const { size = "md", variant = "ghost" } = options;

        const sizes = {
            sm: "p-1.5 text-sm",
            md: "p-2 text-base",
            lg: "p-3 text-lg",
        };

        const variants = {
            ghost:
                "text-gray-400 hover:text-white hover:bg-gray-800 transition-colors rounded",
            secondary:
                "bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors",
            outline:
                "border border-gray-600 text-gray-300 hover:bg-gray-800 rounded transition-colors",
        };

        btn.className = `${sizes[size]} ${variants[variant]}`;
        btn.innerHTML = `<i class="bi bi-${iconName}"></i>`;
        btn.onclick = onClick;

        return btn;
    },
};
