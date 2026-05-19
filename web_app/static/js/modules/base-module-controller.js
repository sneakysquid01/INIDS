class BaseModuleController {
    constructor(moduleElement, config = {}) {
        this.element = moduleElement;
        this.config = config;
        this.state = {
            loading: true,
            error: null,
            empty: false,
            lastUpdate: null
        };
        this.socket = null;
        this.pollInterval = null;
        this.useWebSocket = config.useWebSocket !== false;
        this.setupElements();
        this.attachListeners();
        this.initializeConnection();
    }

    initializeConnection() {
        if (this.useWebSocket && typeof io !== 'undefined') {
            try {
                this.socket = io();
                this.socket.on('connect', () => {
                    console.log('WebSocket connected');
                    if (this.config.moduleId) {
                        this.socket.emit('subscribe_module', { module_id: this.config.moduleId });
                    }
                    this.onSocketReady();
                });
                this.socket.on('module_update', (data) => {
                    if (data.module_id === this.config.moduleId) {
                        this.onDataUpdate(data.data);
                    }
                });
                this.socket.on('disconnect', () => {
                    console.warn('WebSocket disconnected, falling back to polling');
                    this.fallbackToPolling();
                });
            } catch (e) {
                console.warn('WebSocket unavailable, using polling:', e);
                this.fallbackToPolling();
            }
        } else {
            this.fallbackToPolling();
        }
    }

    fallbackToPolling() {
        this.socket = null;
    }

    onSocketReady() {}

    onDataUpdate(data) {}

    broadcastViaSocket(event, data) {
        if (this.socket && this.socket.connected) {
            this.socket.emit(event, data);
        }
    }

    setupElements() {
        this.loading = this.element.querySelector('#moduleLoading');
        this.main = this.element.querySelector('#moduleMain');
        this.error = this.element.querySelector('#moduleError');
        this.empty = this.element.querySelector('#moduleEmpty');
        this.contextBar = this.element.querySelector('#contextBar');
        this.errorMessage = this.element.querySelector('#errorMessage');
        this.statusDot = this.element.querySelector('.status-dot');
        this.statusText = this.element.querySelector('.status-text');
    }

    attachListeners() {
        this.element.querySelector('#moduleRefresh')?.addEventListener('click', () => this.refresh());
        this.element.querySelector('#moduleFullscreen')?.addEventListener('click', () => this.toggleFullscreen());
        this.element.querySelector('#moduleSettings')?.addEventListener('click', () => this.openSettings());
        this.element.querySelector('#errorRetry')?.addEventListener('click', () => this.retry());
    }

    showLoading(message = 'Loading module...') {
        this.setState({ loading: true, error: null, empty: false });
        this.loading.style.display = 'flex';
        this.main.style.display = 'none';
        this.error.style.display = 'none';
        this.empty.style.display = 'none';
    }

    showContent(content) {
        this.setState({ loading: false, error: null, empty: false });
        if (typeof content === 'string') {
            this.main.innerHTML = content;
        } else {
            this.main.appendChild(content);
        }
        this.loading.style.display = 'none';
        this.main.style.display = 'flex';
        this.error.style.display = 'none';
        this.empty.style.display = 'none';
        this.setStatus('ready');
    }

    showError(message) {
        this.setState({ loading: false, error: message });
        this.errorMessage.textContent = message;
        this.loading.style.display = 'none';
        this.main.style.display = 'none';
        this.error.style.display = 'flex';
        this.empty.style.display = 'none';
        this.setStatus('error');
    }

    showEmpty(message = 'No data available') {
        this.setState({ loading: false, error: null, empty: true });
        this.element.querySelector('.empty-message').textContent = message;
        this.loading.style.display = 'none';
        this.main.style.display = 'none';
        this.error.style.display = 'none';
        this.empty.style.display = 'flex';
    }

    setState(newState) {
        this.state = { ...this.state, ...newState, lastUpdate: new Date().toISOString() };
        this.updateContextBar();
    }

    updateContextBar() {
        if (this.contextBar) {
            this.contextBar.style.display = 'flex';
            const status = this.state.loading ? 'loading' : this.state.error ? 'error' : 'active';
            this.element.querySelector('#contextStatus').textContent = status.toUpperCase();
            this.element.querySelector('#contextUpdated').textContent = new Date(this.state.lastUpdate).toLocaleTimeString();
        }
    }

    setStatus(status) {
        this.statusDot?.classList.remove('status-loading', 'status-error');
        this.statusText?.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        if (status === 'loading') {
            this.statusDot?.classList.add('status-loading');
        } else if (status === 'error') {
            this.statusDot?.classList.add('status-error');
        }
    }

    async refresh() {
        this.showLoading('Refreshing...');
        try {
            await this.loadData();
            this.showContent(this.renderContent());
        } catch (error) {
            this.showError('Refresh failed: ' + error.message);
        }
    }

    retry() {
        this.refresh();
    }

    toggleFullscreen() {
        document.documentElement.classList.toggle('module-fullscreen');
    }

    openSettings() {
        if (typeof AppModal !== 'undefined' && AppModal && AppModal.alert) {
            AppModal.alert('Module settings', 'Module settings coming soon.');
        } else if (typeof AppToast !== 'undefined' && AppToast && AppToast.info) {
            AppToast.info('Module settings coming soon');
        } else {
            console.warn('[base-module-controller] Module settings UI not available');
        }
    }

    async loadData() {
        throw new Error('loadData() must be implemented');
    }

    renderContent() {
        throw new Error('renderContent() must be implemented');
    }

    cleanup() {
        if (this.pollInterval) clearInterval(this.pollInterval);
        if (this.socket) {
            this.socket.emit('unsubscribe_module', { module_id: this.config.moduleId });
            this.socket = null;
        }
    }

    onClose() {
        this.cleanup();
    }
}

if (typeof window !== 'undefined') {
    window.BaseModuleController = BaseModuleController;
}
