/**
 * AppModal Component - Reusable Modal Dialog System
 * Handles alerts, forms, confirmations, and content modals
 * 
 * Usage:
 *   const modal = new AppModal({ title: "Confirm Action", content: "Are you sure?" });
 *   modal.open();
 *   modal.close();
 */

export class AppModal {
    constructor(options = {}) {
        this.title = options.title || "";
        this.content = options.content || "";
        this._contentIsHtml = options.contentIsHtml === true;
        this.size = options.size || "lg"; // sm, md, lg, xl
        this.onClose = options.onClose || (() => {});
        this.closeable = options.closeable !== false;
        this.backdrop = options.backdrop !== false;
        this.escapeKey = options.escapeKey !== false;

        this.isOpen = false;
        this.el = null;
        this.backdropEl = null;
    }

    /**
     * Generate size classes for modal width
     */
    getSizeClass() {
        const sizes = {
            sm: "max-w-sm",
            md: "max-w-md",
            lg: "max-w-lg",
            xl: "max-w-2xl",
            full: "max-w-4xl",
        };
        return sizes[this.size] || sizes.lg;
    }

    /**
     * Render modal DOM structure
     */
    render() {
        // Create backdrop
        if (this.backdrop) {
            this.backdropEl = document.createElement("div");
            this.backdropEl.className =
                "fixed inset-0 bg-black/40 z-40 opacity-0 transition-opacity duration-200";
            this.backdropEl.onclick = () => this.backdrop && this.close();
        }

        // Create modal container
        this.el = document.createElement("div");
        this.el.className = `fixed inset-0 flex items-center justify-center z-50 pointer-events-none`;

        // Create modal panel
        const panel = document.createElement("div");
        panel.className = `
            bg-[#151922] border border-[#1a1f2e] rounded-lg shadow-2xl
            w-full mx-4 ${this.getSizeClass()}
            opacity-0 scale-95 transform transition-all duration-200
            pointer-events-auto overflow-hidden
        `;
        panel.setAttribute("data-modal-panel", "true");

        // Create header
        const header = document.createElement("div");
        header.className = "bg-[#0f1117] border-b border-[#1a1f2e] px-6 py-4 flex items-center justify-between";

        const titleEl = document.createElement("h2");
        titleEl.className = "text-lg font-semibold text-white flex-1";
        titleEl.textContent = this.title;
        header.appendChild(titleEl);

        if (this.closeable) {
            const closeBtn = document.createElement("button");
            closeBtn.className =
                "text-gray-400 hover:text-white transition-colors p-1 -mr-2";
            closeBtn.innerHTML = '<i class="bi bi-x-lg text-xl"></i>';
            closeBtn.onclick = () => this.close();
            header.appendChild(closeBtn);
        }

        // Create content
        const content = document.createElement("div");
        content.className = "px-6 py-4 max-h-[60vh] overflow-auto text-gray-300";
        content.setAttribute("data-modal-content", "true");
        this._applyContent(content, this.content, this._contentIsHtml);

        // Create footer with default close button
        const footer = document.createElement("div");
        footer.className = "bg-[#0f1117] border-t border-[#1a1f2e] px-6 py-4 text-right";
        footer.setAttribute("data-modal-footer", "true");

        const closeBtn = document.createElement("button");
        closeBtn.className =
            "px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors";
        closeBtn.textContent = "Close";
        closeBtn.onclick = () => this.close();
        footer.appendChild(closeBtn);

        panel.appendChild(header);
        panel.appendChild(content);
        panel.appendChild(footer);
        this.el.appendChild(panel);
    }

    /**
     * Open modal with animation
     */
    open() {
        if (this.isOpen) return;

        this.render();

        if (this.backdropEl) {
            document.body.appendChild(this.backdropEl);
            // Trigger animation
            requestAnimationFrame(() => {
                this.backdropEl.classList.remove("opacity-0");
            });
        }

        document.body.appendChild(this.el);

        // Trigger animation
        requestAnimationFrame(() => {
            const panel = this.el.querySelector("[data-modal-panel]");
            if (panel) {
                panel.classList.remove("opacity-0", "scale-95");
            }
        });

        this.isOpen = true;

        // Handle escape key
        if (this.escapeKey) {
            this.handleEscape = (e) => {
                if (e.key === "Escape") this.close();
            };
            document.addEventListener("keydown", this.handleEscape);
        }

        // Prevent body scroll
        document.body.style.overflow = "hidden";
    }

    /**
     * Close modal with animation
     */
    close() {
        if (!this.isOpen) return;

        const panel = this.el?.querySelector("[data-modal-panel]");
        if (panel) {
            panel.classList.add("opacity-0", "scale-95");
        }

        if (this.backdropEl) {
            this.backdropEl.classList.add("opacity-0");
        }

        // Wait for animation
        setTimeout(() => {
            this.el?.remove();
            this.backdropEl?.remove();
            this.isOpen = false;
            document.body.style.overflow = "";

            if (this.handleEscape) {
                document.removeEventListener("keydown", this.handleEscape);
            }

            this.onClose();
        }, 200);
    }

    /**
     * Apply content to a DOM element safely.
     * Strings default to textContent; pass { html: true } to opt in to innerHTML.
     */
    _applyContent(el, content, isHtml = false) {
        if (content instanceof Node) {
            el.replaceChildren(content);
        } else if (typeof content === "string") {
            if (isHtml) {
                el.innerHTML = content;
            } else {
                el.textContent = content;
            }
        } else {
            el.replaceChildren();
        }
    }

    /**
     * Update modal content dynamically.
     * Pass { html: true } to render raw HTML (caller is responsible for sanitization).
     */
    setContent(content, options = {}) {
        const { html = false } = options;
        if (this.isOpen) {
            const el = this.el?.querySelector("[data-modal-content]");
            if (el) this._applyContent(el, content, html);
        } else {
            this.content = content;
            this._contentIsHtml = html;
        }
    }

    /**
     * Update modal title
     */
    setTitle(title) {
        this.title = title;
        if (this.isOpen) {
            const titleEl = this.el?.querySelector("h2");
            if (titleEl) {
                titleEl.textContent = title;
            }
        }
    }

    /**
     * Add custom button to footer
     */
    addButton(label, callback, options = {}) {
        if (!this.isOpen) return;

        const footer = this.el?.querySelector("[data-modal-footer]");
        if (!footer) return;

        const btn = document.createElement("button");
        const variant = options.variant || "secondary"; // primary, secondary, danger
        const variantClass = {
            primary: "bg-blue-600 hover:bg-blue-700",
            secondary: "bg-gray-700 hover:bg-gray-600",
            danger: "bg-red-600 hover:bg-red-700",
        }[variant];

        btn.className = `px-4 py-2 ${variantClass} text-white rounded transition-colors mr-2`;
        btn.textContent = label;
        btn.onclick = () => callback(this);

        footer.insertBefore(btn, footer.firstChild);
    }

    /**
     * Static helper to show confirmation dialog
     */
    static confirm(title, message, onConfirm, onCancel) {
        const msgEl = document.createElement("p");
        msgEl.className = "text-gray-300";
        msgEl.textContent = message;
        const modal = new AppModal({
            title,
            content: msgEl,
            size: "md",
        });

        modal.open();

        const footer = modal.el.querySelector("[data-modal-footer]");
        footer.innerHTML = "";

        const cancelBtn = document.createElement("button");
        cancelBtn.className =
            "px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors mr-2";
        cancelBtn.textContent = "Cancel";
        cancelBtn.onclick = () => {
            modal.close();
            onCancel?.();
        };
        footer.appendChild(cancelBtn);

        const confirmBtn = document.createElement("button");
        confirmBtn.className =
            "px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors";
        confirmBtn.textContent = "Confirm";
        confirmBtn.onclick = () => {
            modal.close();
            onConfirm?.();
        };
        footer.appendChild(confirmBtn);

        return modal;
    }

    /**
     * Static helper to show alert
     */
    static alert(title, message, onClose) {
        const msgEl = document.createElement("p");
        msgEl.className = "text-gray-300";
        msgEl.textContent = message;
        const modal = new AppModal({
            title,
            content: msgEl,
            size: "md",
            onClose,
        });

        modal.open();
        return modal;
    }
}
