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

// Live clock
function updateClock() {
    const el = document.getElementById('live-clock');
    if (el) {
        const now = new Date();
        el.textContent = now.toUTCString().replace('GMT', 'UTC');
    }
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
