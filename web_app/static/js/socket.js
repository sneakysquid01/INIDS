(function (window) {
    if (window.INIDSSocketManager) {
        return;
    }

    const GlobalState = window.GlobalState;

    if (!GlobalState || typeof window.io !== 'function') {
        console.error('Shared state or Socket.IO is unavailable.');
        return;
    }

    let fallbackTimer = null;
    const socket = io('/events', { transports: ['websocket', 'polling'] });

    function normalizeMetricsPayload(payload) {
        const data = payload && typeof payload === 'object' ? payload : {};
        const pulse = data.pulse && typeof data.pulse === 'object' ? data.pulse : {};
        const current = data.current || pulse.current || {};
        const rollingAverages = data.rolling_averages || pulse.rolling_averages || {};

        return {
            ...data,
            lastAlert: null,
            pulse: pulse.current ? pulse : {
                current,
                rolling_averages: rollingAverages,
                status: data.status,
                pulse_strength: data.pulse_strength,
                timestamp: data.timestamp
            },
            current,
            rolling_averages: rollingAverages,
            status: data.status || pulse.status || 'SAFE'
        };
    }

    function buildRealtimeAlert(payload) {
        const wrapper = payload && typeof payload === 'object' ? payload : {};
        const raw = wrapper.data && typeof wrapper.data === 'object' ? wrapper.data : wrapper;
        const timestamp = raw.timestamp || wrapper.timestamp || new Date().toISOString();
        const sourceIp = raw.source_ip || raw.source || '';
        const baseId = raw.id || `rt_${sourceIp || 'alert'}_${timestamp}`;

        return {
            id: String(baseId).replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64),
            timestamp,
            severity: String(raw.severity || 'low').toLowerCase(),
            prediction: raw.prediction || 'unknown',
            confidence: Number.isFinite(Number(raw.confidence)) ? Number(raw.confidence) : 0,
            status: raw.status || 'open',
            profile: raw.profile || 'N/A',
            source_ip: sourceIp,
            attack_type: raw.attack_type || '',
            reason: raw.reason || ''
        };
    }

    function upsertAlert(alert) {
        const alerts = Array.isArray(GlobalState.data.alerts) ? GlobalState.data.alerts.slice() : [];
        const existingIndex = alerts.findIndex(item => item.id === alert.id);

        if (existingIndex >= 0) {
            alerts.splice(existingIndex, 1);
        }

        alerts.unshift(alert);
        return alerts.slice(0, 200);
    }

    async function parseJson(response) {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return response.json();
    }

    async function hydrateFromApi() {
        const [metricsResult, alertsResult, pulseResult] = await Promise.allSettled([
            fetch('/api/dashboard/metrics').then(parseJson),
            fetch('/api/alerts?limit=200').then(parseJson),
            fetch('/api/perception/pulse').then(parseJson)
        ]);

        const nextState = {};

        if (metricsResult.status === 'fulfilled') {
            Object.assign(nextState, metricsResult.value);
        }

        if (alertsResult.status === 'fulfilled') {
            nextState.alerts = alertsResult.value.alerts || [];
            nextState.alertsCount = alertsResult.value.count || nextState.alerts.length;

            if (typeof nextState.active_alerts !== 'number') {
                nextState.active_alerts = Math.min(nextState.alerts.length, 99);
            }
        }

        if (pulseResult.status === 'fulfilled') {
            Object.assign(nextState, normalizeMetricsPayload({
                pulse: pulseResult.value,
                current: pulseResult.value.current || {},
                rolling_averages: pulseResult.value.rolling_averages || {},
                status: pulseResult.value.status,
                pulse_strength: pulseResult.value.pulse_strength,
                timestamp: pulseResult.value.timestamp
            }));
        }

        if (Object.keys(nextState).length > 0) {
            GlobalState.set(nextState);
        }
    }

    function startFallback() {
        if (fallbackTimer !== null) {
            return;
        }

        hydrateFromApi().catch(error => {
            console.error('Failed to hydrate fallback state:', error);
        });

        fallbackTimer = window.setInterval(() => {
            fetch('/api/perception/pulse')
                .then(r => r.json())
                .then(data => GlobalState.set(normalizeMetricsPayload({
                    pulse: data,
                    current: data.current || {},
                    rolling_averages: data.rolling_averages || {},
                    status: data.status,
                    pulse_strength: data.pulse_strength,
                    timestamp: data.timestamp
                })))
                .catch(error => {
                    console.error('Fallback pulse update failed:', error);
                });
        }, 5000);
    }

    function stopFallback() {
        if (fallbackTimer !== null) {
            window.clearInterval(fallbackTimer);
            fallbackTimer = null;
        }
    }

    socket.on('connect', () => {
        stopFallback();
        GlobalState.set({ socketConnected: true });
        socket.emit('subscribe_metrics');
        socket.emit('subscribe_alerts');
    });

    socket.on('disconnect', () => {
        GlobalState.set({ socketConnected: false });
        startFallback();
    });

    socket.on('connect_error', () => {
        GlobalState.set({ socketConnected: false });
        startFallback();
    });

    socket.on('metrics.update', data => {
        stopFallback();
        GlobalState.set(normalizeMetricsPayload(data));
    });

    socket.on('alert.new', alert => {
        const normalizedAlert = buildRealtimeAlert(alert);
        const alerts = upsertAlert(normalizedAlert);

        GlobalState.set({
            lastAlert: normalizedAlert,
            alerts,
            alertsCount: alerts.length,
            active_alerts: typeof GlobalState.data.active_alerts === 'number'
                ? Math.max(GlobalState.data.active_alerts, Math.min(alerts.length, 99))
                : Math.min(alerts.length, 99)
        });
    });

    window.INIDSSocketManager = {
        socket,
        hydrate: hydrateFromApi,
        startFallback,
        stopFallback
    };
})(window);
