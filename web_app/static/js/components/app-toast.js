/**
 * AppToast Component - Toast Notification System
 * Displays temporary messages (success, error, warning, info)
 * Auto-dismisses after configured duration, supports stacking
 * 
 * Usage:
 *   AppToast.show("Operation successful", { type: "success" });
 *   AppToast.show("An error occurred", { type: "error", duration: 5000 });
 */

export class AppToast {
    static toasts = [];
    static container = null;

    /**
     * Initialize container if not exists
     */
    static initContainer() {
        if (this.container) return;
        this.container = document.getElementById("app-toast-root");
        if (!this.container) {
            this.container = document.createElement("div");
            this.container.id = "app-toast-root";
            this.container.className = "fixed bottom-4 right-4 flex flex-col gap-2 z-40 pointer-events-none";
            document.body.appendChild(this.container);
        }
    }

    /**
     * Get icon based on toast type
     */
    static getIcon(type) {
        const icons = {
            success: "bi-check-circle-fill text-green-500",
            error: "bi-exclamation-circle-fill text-red-500",
            warning: "bi-exclamation-triangle-fill text-amber-500",
            info: "bi-info-circle-fill text-blue-500",
        };
        return icons[type] || icons.info;
    }

    /**
     * Get background color based on type
     */
    static getBgClass(type) {
        const colors = {
            success: "bg-green-900/30 border-green-700/50",
            error: "bg-red-900/30 border-red-700/50",
            warning: "bg-amber-900/30 border-amber-700/50",
            info: "bg-blue-900/30 border-blue-700/50",
        };
        return colors[type] || colors.info;
    }

    /**
     * Get text color based on type
     */
    static getTextClass(type) {
        const colors = {
            success: "text-green-300",
            error: "text-red-300",
            warning: "text-amber-300",
            info: "text-blue-300",
        };
        return colors[type] || colors.info;
    }

    /**
     * Show a toast notification
     */
    static show(message, options = {}) {
        this.initContainer();

        const type = options.type || "info"; // success, error, warning, info
        const duration = options.duration !== false ? options.duration || 3500 : null;
        const dismissible = options.dismissible !== false;

        // Create toast element
        const toast = document.createElement("div");
        const toastId = `toast-${Date.now()}-${Math.random()}`;
        toast.id = toastId;
        toast.className = `
            pointer-events-auto
            bg-[#0f1117] border ${this.getBgClass(type)}
            rounded-lg px-4 py-3 shadow-xl
            flex items-center gap-3 min-w-[300px] max-w-[400px]
            opacity-0 translate-x-8 transform transition-all duration-200
        `;

        // Icon
        const icon = document.createElement("i");
        icon.className = `bi ${this.getIcon(type)} text-xl flex-shrink-0`;

        // Content
        const content = document.createElement("div");
        content.className = `flex-1 ${this.getTextClass(type)}`;
        content.textContent = message;

        toast.appendChild(icon);
        toast.appendChild(content);

        // Close button
        if (dismissible) {
            const closeBtn = document.createElement("button");
            closeBtn.className = `${this.getTextClass(type)} hover:opacity-70 transition-opacity flex-shrink-0 p-1`;
            closeBtn.innerHTML = '<i class="bi bi-x-lg"></i>';
            closeBtn.onclick = () => this.dismiss(toastId);
            toast.appendChild(closeBtn);
        }

        this.container.appendChild(toast);
        this.toasts.push(toastId);

        // Trigger entrance animation
        requestAnimationFrame(() => {
            toast.classList.remove("opacity-0", "translate-x-8");
        });

        // Auto-dismiss
        if (duration) {
            setTimeout(() => this.dismiss(toastId), duration);
        }

        return toastId;
    }

    /**
     * Show success toast
     */
    static success(message, duration = 3000) {
        return this.show(message, { type: "success", duration });
    }

    /**
     * Show error toast
     */
    static error(message, duration = 4000) {
        return this.show(message, { type: "error", duration });
    }

    /**
     * Show warning toast
     */
    static warning(message, duration = 3500) {
        return this.show(message, { type: "warning", duration });
    }

    /**
     * Show info toast
     */
    static info(message, duration = 3000) {
        return this.show(message, { type: "info", duration });
    }

    /**
     * Dismiss a specific toast
     */
    static dismiss(toastId) {
        const toast = document.getElementById(toastId);
        if (!toast) return;

        toast.classList.add("opacity-0", "translate-x-8");

        setTimeout(() => {
            toast.remove();
            this.toasts = this.toasts.filter((id) => id !== toastId);
        }, 200);
    }

    /**
     * Dismiss all toasts
     */
    static dismissAll() {
        this.toasts.forEach((id) => this.dismiss(id));
    }

    /**
     * Show loading toast (no auto-dismiss by default)
     */
    static loading(message) {
        return this.show(message, { type: "info", duration: false });
    }

    /**
     * Update existing toast content
     */
    static update(toastId, message, options = {}) {
        const toast = document.getElementById(toastId);
        if (!toast) return;

        const type = options.type;
        const content = toast.querySelector("div");

        if (type) {
            toast.className = toast.className.replace(
                /bg-\[.*?\]\/30 border-\[.*?\]\/50/,
                `${this.getBgClass(type)}`
            );
            content.className = content.className.replace(
                /text-\w+-\d+/,
                this.getTextClass(type).split(" ")[0]
            );

            const icon = toast.querySelector("i");
            icon.className = `bi ${this.getIcon(type)} text-xl flex-shrink-0`;
        }

        if (message) {
            content.textContent = message;
        }
    }
}

// Auto-initialize container on DOM ready
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => AppToast.initContainer());
} else {
    AppToast.initContainer();
}
