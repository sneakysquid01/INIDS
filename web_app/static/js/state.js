(function (window) {
    function getAlertCount(data) {
        if (typeof data.active_alerts === 'number') {
            return data.active_alerts;
        }

        if (typeof data.alertsCount === 'number') {
            return data.alertsCount;
        }

        if (Array.isArray(data.alerts)) {
            return data.alerts.length;
        }

        return 0;
    }

    function updateThreatState(alerts) {
        if (alerts > 0) {
            document.body.classList.add('under-attack');
        } else {
            document.body.classList.remove('under-attack');
        }
    }

    const GlobalState = {
        data: {},

        listeners: [],

        set(newData) {
            this.data = { ...this.data, ...newData };
            updateThreatState(getAlertCount(this.data));
            this.listeners.forEach(fn => fn(this.data));
        },

        subscribe(fn) {
            this.listeners.push(fn);
        }
    };

    window.GlobalState = GlobalState;
    window.updateThreatState = updateThreatState;
})(window);
