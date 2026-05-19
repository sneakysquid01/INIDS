// Sidebar toggle
const sidebar = document.getElementById('sidebar');
const toggleBtn = document.getElementById('sidebar-toggle');
toggleBtn?.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    document.getElementById('topbar').style.left =
        sidebar.classList.contains('collapsed')
            ? 'var(--sidebar-collapsed-w)'
            : 'var(--sidebar-w)';
    document.getElementById('page-wrapper').style.marginLeft =
        sidebar.classList.contains('collapsed')
            ? 'var(--sidebar-collapsed-w)'
            : 'var(--sidebar-w)';
});

// Live clock — IST (Asia/Kolkata, UTC+5:30)
function updateClock() {
    const el = document.getElementById('live-clock');
    if (!el) return;
    const now = new Date();
    const parts = new Intl.DateTimeFormat('en-IN', {
        timeZone: 'Asia/Kolkata',
        weekday: 'short', day: '2-digit', month: 'short',
        year: 'numeric', hour: '2-digit', minute: '2-digit',
        second: '2-digit', hour12: false
    }).formatToParts(now);
    const get = type => parts.find(p => p.type === type)?.value ?? '';
    el.textContent = `${get('weekday')}, ${get('day')} ${get('month')} ${get('year')} ${get('hour')}:${get('minute')}:${get('second')} IST`;
}
updateClock();
setInterval(updateClock, 1000);

// Chart.js global defaults for dark theme
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = 'rgba(255,255,255,0.45)';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
    Chart.defaults.font.family = "'JetBrains Mono', monospace";
    Chart.defaults.font.size = 11;
}
