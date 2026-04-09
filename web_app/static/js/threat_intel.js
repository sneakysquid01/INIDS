let recentSearches = [];
const MAX_RECENT = 5;

// Load on page load
document.addEventListener('DOMContentLoaded', () => {
    loadRecentSearches();
    document.getElementById('noResultsSection').style.display = 'block';
});

function handleQueryKeyup() {
    if (event.key === 'Enter') {
        performLookup();
    }
}

function isValidIPv4(ip) {
    const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
    return ipv4Regex.test(ip);
}

function isValidIPv6(ip) {
    const ipv6Regex = /^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4})$/;
    return ipv6Regex.test(ip);
}

function isValidDomain(domain) {
    const domainRegex = /^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$/;
    return domainRegex.test(domain);
}

function getQueryType(query) {
    if (isValidIPv4(query)) return 'ipv4';
    if (isValidIPv6(query)) return 'ipv6';
    if (isValidDomain(query)) return 'domain';
    return 'unknown';
}

async function performLookup() {
    const query = document.getElementById('queryInput').value.trim();
    
    if (!query) {
        showError('Please enter an IP address or domain');
        return;
    }

    const type = getQueryType(query);
    if (type === 'unknown') {
        showError('Invalid IP address or domain format');
        return;
    }

    // Show loading
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('noResultsSection').style.display = 'none';

    try {
        const response = await fetch(`/api/threat-intel/lookup?query=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error('Lookup failed');
        
        const data = await response.json();
        
        // Add to recent searches
        addToRecentSearches(query);
        
        // Display results
        displayResults(query, type, data);
        
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('noResultsSection').style.display = 'none';
        
        showSuccess('Threat intelligence lookup completed');
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('noResultsSection').style.display = 'block';
        showError('Failed to lookup threat intelligence: ' + error.message);
    }
}

function displayResults(query, type, data) {
    // Summary
    document.getElementById('resultQuery').textContent = escapeHtml(query);
    document.getElementById('resultType').innerHTML = `<span class="badge ${getTypeBadgeClass(type)}">${type.toUpperCase()}</span>`;
    
    const reputation = data.reputation || {};
    const riskScore = reputation.risk_score || 0;
    const threatLevel = getThreatLevel(riskScore);
    
    document.getElementById('reputationScore').textContent = Math.round(riskScore) + '%';
    document.getElementById('reputationProgress').style.width = riskScore + '%';
    document.getElementById('reputationProgress').className = `progress-bar ${getReputationColorClass(riskScore)}`;
    document.getElementById('threatLevel').textContent = threatLevel;
    document.getElementById('threatLevel').className = `fs-6 fw-bold text-${getReputationColorClass(riskScore).replace('bg-', '')}`;
    
    // Reputation Tab
    document.getElementById('rep-lastSeen').textContent = reputation.last_seen || '--';
    document.getElementById('rep-firstSeen').textContent = reputation.first_seen || '--';
    document.getElementById('rep-confidence').style.width = (reputation.confidence || 0) + '%';
    document.getElementById('rep-riskScore').textContent = Math.round(riskScore) + '%';
    
    // Threat categories
    const categories = reputation.threats || [];
    document.getElementById('threatCategories').innerHTML = categories
        .map(cat => `<span class="badge bg-warning">${escapeHtml(cat)}</span>`)
        .join('');
    
    // Threats Tab
    const threats = data.threats || [];
    if (threats.length > 0) {
        document.getElementById('threatsList').innerHTML = threats
            .map(threat => `
                <div class="alert alert-${getThreatSeverityClass(threat.severity)} mb-2">
                    <h6 class="mb-1">${escapeHtml(threat.name)}</h6>
                    <p class="mb-1 small">${escapeHtml(threat.description)}</p>
                    <small>
                        <strong>Severity:</strong> ${escapeHtml(threat.severity)} | 
                        <strong>Last Seen:</strong> ${escapeHtml(threat.last_seen)}
                    </small>
                </div>
            `).join('');
    } else {
        document.getElementById('threatsList').innerHTML = '<p class="text-muted">No threats found</p>';
    }
    
    // Feeds Tab
    const feeds = data.threat_feeds || [];
    if (feeds.length > 0) {
        document.getElementById('feedsList').innerHTML = feeds
            .map(feed => `
                <div class="card mb-2">
                    <div class="card-body py-2">
                        <h6 class="card-title mb-1">${escapeHtml(feed.name)}</h6>
                        <small class="text-muted">${escapeHtml(feed.source)}</small>
                        <br>
                        <span class="badge bg-secondary">${escapeHtml(feed.type)}</span>
                        <span class="badge bg-info">Listed: ${escapeHtml(feed.list_date)}</span>
                    </div>
                </div>
            `).join('');
    } else {
        document.getElementById('feedsList').innerHTML = '<p class="text-muted">Not listed in any threat feed</p>';
    }
    
    // Incidents Tab
    const incidents = data.incidents || [];
    if (incidents.length > 0) {
        document.getElementById('incidentsList').innerHTML = incidents
            .map(incident => `
                <div class="card mb-2 border-${getIncidentSeverityClass(incident.severity)}">
                    <div class="card-body py-2">
                        <h6 class="card-title mb-1">${escapeHtml(incident.title)}</h6>
                        <p class="mb-1 small">${escapeHtml(incident.description)}</p>
                        <small class="text-muted">
                            <strong>Date:</strong> ${escapeHtml(incident.date)} | 
                            <strong>Severity:</strong> ${escapeHtml(incident.severity)}
                        </small>
                    </div>
                </div>
            `).join('');
    } else {
        document.getElementById('incidentsList').innerHTML = '<p class="text-muted">No related incidents found</p>';
    }
    
    // Geolocation Tab
    const geo = data.geolocation || {};
    document.getElementById('geo-country').textContent = geo.country || '--';
    document.getElementById('geo-city').textContent = geo.city || '--';
    document.getElementById('geo-coords').textContent = geo.latitude && geo.longitude 
        ? `${geo.latitude}, ${geo.longitude}` 
        : '--';
    document.getElementById('geo-timezone').textContent = geo.timezone || '--';
    document.getElementById('geo-asn').textContent = geo.asn || '--';
}

function getTypeBadgeClass(type) {
    switch(type) {
        case 'ipv4': return 'bg-primary';
        case 'ipv6': return 'bg-info';
        case 'domain': return 'bg-success';
        default: return 'bg-secondary';
    }
}

function getReputationColorClass(score) {
    if (score >= 80) return 'bg-danger';
    if (score >= 60) return 'bg-warning';
    if (score >= 40) return 'bg-info';
    return 'bg-success';
}

function getThreatSeverityClass(severity) {
    severity = String(severity).toLowerCase();
    if (severity === 'critical') return 'danger';
    if (severity === 'high') return 'warning';
    if (severity === 'medium') return 'info';
    return 'secondary';
}

function getIncidentSeverityClass(severity) {
    severity = String(severity).toLowerCase();
    if (severity === 'critical') return 'danger';
    if (severity === 'high') return 'warning';
    if (severity === 'medium') return 'info';
    return 'secondary';
}

function getThreatLevel(score) {
    if (score >= 80) return '🔴 Critical';
    if (score >= 60) return '🟠 High';
    if (score >= 40) return '🟡 Medium';
    if (score >= 20) return '🔵 Low';
    return '🟢 Clean';
}

function addToRecentSearches(query) {
    // Remove if already exists
    recentSearches = recentSearches.filter(q => q !== query);
    // Add to front
    recentSearches.unshift(query);
    // Limit to max
    recentSearches = recentSearches.slice(0, MAX_RECENT);
    // Save to localStorage
    localStorage.setItem('threatIntelRecentSearches', JSON.stringify(recentSearches));
    updateRecentSearchesDisplay();
}

function loadRecentSearches() {
    const saved = localStorage.getItem('threatIntelRecentSearches');
    if (saved) {
        recentSearches = JSON.parse(saved);
        updateRecentSearchesDisplay();
    }
}

function updateRecentSearchesDisplay() {
    if (recentSearches.length === 0) {
        document.getElementById('recentSearchesSection').style.display = 'none';
        return;
    }
    
    document.getElementById('recentSearchesSection').style.display = 'block';
    document.getElementById('recentSearches').innerHTML = recentSearches
        .map(query => `
            <button class="btn btn-outline-secondary btn-sm" onclick="quickSearch('${escapeQuote(query)}')">
                ${escapeHtml(query)}
            </button>
        `).join('');
}

function quickSearch(query) {
    document.getElementById('queryInput').value = query;
    performLookup();
}

function clearResults() {
    document.getElementById('queryInput').value = '';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('noResultsSection').style.display = 'block';
    document.getElementById('queryInput').focus();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeQuote(text) {
    return text.replace(/'/g, "\\'");
}

function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show position-fixed';
    alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alert.innerHTML = message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';
    document.body.appendChild(alert);
    setTimeout(() => alert.remove(), 3000);
}

function showError(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show position-fixed';
    alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alert.innerHTML = message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';
    document.body.appendChild(alert);
}
