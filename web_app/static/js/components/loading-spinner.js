/**
 * LoadingSpinner Component - Loading State Indicators
 * Various spinner styles for different use cases
 * 
 * Styles: spin, pulse, bounce, wave
 * Sizes: sm, md, lg
 * 
 * Usage:
 *   LoadingSpinner.create()
 *   LoadingSpinner.withText("Loading data...")
 */

export const LoadingSpinner = {
    /**
     * Create a spinning loader
     */
    create(options = {}) {
        const { size = "md", style = "spin" } = options;

        const container = document.createElement("div");
        container.className = "flex items-center justify-center";

        const spinner = document.createElement("div");

        const sizes = {
            sm: "w-4 h-4",
            md: "w-8 h-8",
            lg: "w-12 h-12",
        };

        const styles = {
            spin: `
                ${sizes[size]} border-2 border-gray-600 border-t-blue-400
                rounded-full animate-spin
            `,
            pulse: `
                ${sizes[size]} bg-blue-500 rounded-full
                animate-pulse opacity-75
            `,
            bounce: `
                ${sizes[size]} bg-blue-400 rounded-full
                animate-bounce
            `,
            wave: `
                flex items-center justify-center gap-1 h-8
            `,
        };

        spinner.className = `${styles[style]}`;

        if (style === "wave") {
            for (let i = 0; i < 3; i++) {
                const dot = document.createElement("div");
                dot.className = `w-2 h-2 bg-blue-400 rounded-full animate-bounce`;
                dot.style.animationDelay = `${i * 0.1}s`;
                spinner.appendChild(dot);
            }
        }

        container.appendChild(spinner);
        return container;
    },

    /**
     * Create spinner with text
     */
    withText(text, options = {}) {
        const container = document.createElement("div");
        container.className = "flex flex-col items-center justify-center gap-3 py-8";

        const spinner = this.create(options);
        container.appendChild(spinner);

        const textEl = document.createElement("p");
        textEl.className = "text-gray-400 text-sm";
        textEl.textContent = text;
        container.appendChild(textEl);

        return container;
    },

    /**
     * Create inline spinner (for buttons, etc.)
     */
    inline(options = {}) {
        const spinner = document.createElement("i");
        spinner.className = "bi bi-hourglass-split animate-spin text-sm";
        return spinner;
    },

    /**
     * Create centered full-screen loader
     */
    fullscreen(message = "Loading...", options = {}) {
        const overlay = document.createElement("div");
        overlay.className = `
            fixed inset-0 bg-black/30 flex items-center justify-center z-50
            opacity-0 transition-opacity duration-200
        `;
        overlay.id = "fullscreen-loader";

        const card = document.createElement("div");
        card.className = `
            bg-[#151922] border border-[#1a1f2e] rounded-lg p-8
            text-center transform scale-95 transition-all duration-200
        `;

        const spinner = this.create({ size: "lg" });
        card.appendChild(spinner);

        if (message) {
            const text = document.createElement("p");
            text.className = "mt-4 text-gray-300";
            text.textContent = message;
            card.appendChild(text);
        }

        overlay.appendChild(card);
        document.body.appendChild(overlay);

        // Trigger animation
        requestAnimationFrame(() => {
            overlay.classList.remove("opacity-0");
            card.classList.remove("scale-95");
        });

        return {
            element: overlay,
            close() {
                overlay.classList.add("opacity-0");
                setTimeout(() => overlay.remove(), 200);
            },
            setMessage(newMessage) {
                const textEl = card.querySelector("p");
                if (textEl) textEl.textContent = newMessage;
            },
        };
    },

    /**
     * Create skeleton loading placeholder
     */
    skeleton(lines = 3, options = {}) {
        const { width = "w-full", height = "h-4" } = options;

        const container = document.createElement("div");
        container.className = "space-y-2";

        for (let i = 0; i < lines; i++) {
            const line = document.createElement("div");
            line.className = `
                ${width} ${height} bg-gray-700 rounded
                animate-pulse
            `;
            if (i === lines - 1) {
                line.style.width = "75%"; // Last line shorter
            }
            container.appendChild(line);
        }

        return container;
    },

    /**
     * Create progress bar
     */
    progress(current = 0, total = 100, options = {}) {
        const { animated = true, showPercent = true } = options;

        const container = document.createElement("div");
        container.className = "w-full";

        const barContainer = document.createElement("div");
        barContainer.className = "w-full h-2 bg-gray-700 rounded-full overflow-hidden";

        const bar = document.createElement("div");
        const percent = (current / total) * 100;
        bar.className = `h-full bg-blue-500 transition-all duration-300 ${
            animated ? "animate-pulse" : ""
        }`;
        bar.style.width = `${percent}%`;
        barContainer.appendChild(bar);
        container.appendChild(barContainer);

        if (showPercent) {
            const text = document.createElement("p");
            text.className = "text-xs text-gray-400 mt-1";
            text.textContent = `${Math.round(percent)}%`;
            container.appendChild(text);
        }

        return container;
    },

    /**
     * Create activity indicator (dots)
     */
    dots(options = {}) {
        const { count = 3, size = "md" } = options;

        const container = document.createElement("div");
        container.className = "flex items-center justify-center gap-1";

        const sizes = {
            sm: "w-2 h-2",
            md: "w-3 h-3",
            lg: "w-4 h-4",
        };

        for (let i = 0; i < count; i++) {
            const dot = document.createElement("div");
            dot.className = `${sizes[size]} bg-blue-400 rounded-full animate-bounce`;
            dot.style.animationDelay = `${i * 0.15}s`;
            container.appendChild(dot);
        }

        return container;
    },

    /**
     * Create shimmer effect
     */
    shimmer(options = {}) {
        const { width = "w-full", height = "h-12" } = options;

        const container = document.createElement("div");
        container.className = `
            ${width} ${height} bg-gradient-to-r
            from-gray-700 via-gray-600 to-gray-700
            rounded animate-pulse
            bg-[length:200%_100%]
        `;

        return container;
    },
};
