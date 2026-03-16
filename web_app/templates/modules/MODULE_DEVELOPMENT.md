# INIDS Module Development Guide

## Overview

This directory contains reusable templates and utilities for building the 15 demo capability modules for the INIDS platform. Each module is a self-contained interactive demonstration of a specific security capability.

## File Structure

```
modules/
├── base_module.html          # Template base class (inheritance system)
├── MODULE_DEVELOPMENT.md     # This file
└── [module-specific files]
    ├── real_time_detection/
    │   ├── template.html
    │   ├── script.js
    │   └── README.md
    └── [15 total modules...]
```

## Quick Start

### 1. Creating a New Module

Each module extends `BaseModuleController` and follows this pattern:

```html
<!-- web_app/templates/modules/my_module/template.html -->

{% extends "modules/base_module.html" %}

{% block module_config %}
{
    "moduleId": "my-module",
    "moduleName": "My Demo Module",
    "moduleCategory": "Detection",
    "emptyMessage": "No data available"
}
{% endblock %}

{% block module_content %}
<!-- Your module's custom content -->
<div class="my-module-content">
    <!-- Chart, list, form, etc. -->
</div>
{% endblock %}

<script>
class MyModuleController extends BaseModuleController {
    async loadData() {
        // Fetch data from /api/modules/my-module
        const response = await fetch('/api/modules/my-module');
        if (!response.ok) throw new Error('Failed to load data');
        return response.json();
    }

    renderContent() {
        // Return HTML or DOM element
        return `<div>Custom module content</div>`;
    }
}

// Auto-initialize when module is loaded
document.addEventListener('DOMContentLoaded', () => {
    const moduleElement = document.querySelector('[data-module-id="my-module"]');
    window.myModuleController = new MyModuleController(moduleElement);
    window.myModuleController.loadData().then(data => {
        window.myModuleController.showContent(data);
    }).catch(error => {
        window.myModuleController.showError(error.message);
    });
});
</script>
```

### 2. Module State Management

The `BaseModuleController` manages 4 states:

- **Loading**: Spinner animation, message "Loading module..."
- **Content**: Main data display
- **Error**: Error icon + message with retry button
- **Empty**: Empty state with helpful message

```javascript
// Show state methods (mutually exclusive)
controller.showLoading()          // Spinner
controller.showContent(html)      // Display data
controller.showError(message)     // Error state
controller.showEmpty(message)     // Empty state
controller.setStatus(status)      // Update footer indicator
```

### 3. Module Configuration

Required fields when extending base:

```javascript
const config = {
    moduleId:       // CSS data attribute, kebab-case
    moduleName:     // Display heading
    moduleCategory: // Dashboard category (Detection, Prevention, Learning, etc.)
    emptyMessage:   // Message when no data available
}
```

## Base Module Features

### Header Actions
- **⟳ Refresh** - Calls `refresh()` method
- **⛶ Fullscreen** - Toggles fullscreen mode
- **⚙️ Settings** - Opens module settings

### Context Bar (Optional)
Shows meta information:
- Status: Active/Loading/Error
- Last Updated: ISO timestamp
- Data Points: Custom metric

To enable:
```javascript
controller.contextBar.style.display = 'flex';
document.getElementById('contextStatus').textContent = 'ACTIVE';
```

### Footer Status
- Green dot (pulsing) = Loading
- Green dot (solid) = Ready
- Red dot = Error

```javascript
controller.setStatus('ready')    // Green dot
controller.setStatus('loading')  // Pulsing yellow
controller.setStatus('error')    // Red dot
```

## API Integration Pattern

### Backend Endpoint Structure

```python
# web_app/app.py
@app.route("/api/modules/<module_id>", methods=["GET"])
def api_module_data(module_id):
    """Return data for module."""
    try:
        data = fetch_module_data(module_id)
        return jsonify({
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### Frontend Fetch Pattern

```javascript
async loadData() {
    const response = await fetch(`/api/modules/${this.config.moduleId}`);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
}
```

## CSS Convention

All modules use CSS variables from dashboard.css:

```css
--primary-dark: #0a0e27
--secondary-dark: #1a1f3a
--tertiary-dark: #252d45
--accent-blue: #00d4ff
--accent-red: #ff4444
--accent-green: #44ff44
--accent-yellow: #ffaa00
--accent-purple: #dd44ff
--text-light: #e0e0e0
--text-muted: #808080
```

Example module styling:

```css
.my-module-content {
    background: var(--secondary-dark);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 8px;
    padding: 16px;
}

.my-module-item {
    color: var(--text-light);
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
}

.my-module-item.highlight {
    background: rgba(0, 212, 255, 0.05);
}
```

## Data Visualization Libraries

Pre-integrated and available globally:

- **Chart.js** - Line, bar, pie, radar charts
- **Bootstrap 5** - UI components, grid system
- **D3.js** (optional) - Network graphs, custom visualizations
- **Font Awesome** (optional) - Icons

### Chart.js Example

```javascript
renderContent() {
    const canvas = document.createElement('canvas');
    new Chart(canvas, {
        type: 'line',
        data: {
            labels: ['6am', '9am', '12pm'],
            datasets: [{
                label: 'Attacks',
                data: [65, 120, 85],
                borderColor: 'var(--accent-red)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
    return canvas;
}
```

## Module Routes

### Dashboard Integration

Modules are loaded into modals when cards are clicked:

1. User clicks card on dashboard
2. Dashboard calls `/modules/[module-id]`
3. Module HTML is loaded into modal
4. Module's JS initializes automatically

### URL Pattern

```
GET /modules/[module-id]
```

### Route Handler (Sample)

```python
@app.route("/modules/<module_id>")
def module_view(module_id):
    modules = {
        'real-time-detection': 'modules/real_time_detection/template.html',
        'multi-engine': 'modules/multi_engine/template.html',
        # ... 15 total
    }
    template = modules.get(module_id)
    if not template:
        return "Module not found", 404
    return render_template(template)
```

## 15 Modules to Implement

### Detection Tier
1. **Real-Time Detection** - Live event stream
2. **Multi-Engine Voting** - Consensus engine voting
3. **Risk Score Visualizer** - Multi-factor risk gauge

### Prevention Tier
4. **Auto-Blocking** - Detection → Block timeline
5. **Approval Workflow** - HITL review system
6. **Escalation Machine** - Per-IP escalation progression

### Learning Tier
7. **False Positive Learning** - Feedback suppression
8. **Anomaly Learning** - Baseline self-training
9. **Threat Intelligence** - External reputation enrichment

### Analytics Tier
10. **Analytics Dashboard** - Health metrics & reports
11. **Pipeline Monitor** - Throughput & latency
12. **Policy Tuning** - Interactive policy simulator

### Advanced Tier
13. **Alert Lifecycle Manager** - Kanban workflow board
14. **Engine Playground** - Toggle engine visibility
15. **Pattern Detector** - Network graph visualization

## Testing a Module

### 1. Manual Testing

Run Flask app:
```bash
python -m web_app.app
```

Navigate to:
```
http://localhost:5000/dashboard/main
```

Click a module card → Should load in modal without errors

### 2. Unit Testing

Add to `tests/test_modules.py`:

```python
import pytest
from web_app import app

def test_module_my_module_loads():
    """Test module renders without error."""
    with app.test_client() as client:
        response = client.get('/modules/my-module')
        assert response.status_code == 200
        assert 'my-module-content' in response.text
```

### 3. Integration Testing

Test with demo data:

```python
def test_module_api_data():
    """Test module API returns valid JSON."""
    with app.test_client() as client:
        response = client.get('/api/modules/my-module')
        assert response.status_code == 200
        data = response.json
        assert 'data' in data
        assert 'timestamp' in data
```

## Module Lifecycle Hooks

The base controller provides hooks for custom logic:

```javascript
class MyModuleController extends BaseModuleController {
    // Override lifecycle methods
    async loadData() { }      // Fetch data
    renderContent() { }       // Return HTML/DOM
    refresh() { }             // Custom refresh logic
    onFullscreen() { }        // Entering fullscreen
    onClose() { }             // Modal closing
}
```

## Performance Considerations

1. **Limit API calls**: Cache data, avoid burst requests
2. **Chart rendering**: Use `responsive: true` and `maintainAspectRatio: false`
3. **Large datasets**: Implement pagination or filtering
4. **Memory**: Clean up charts on refresh with `.destroy()`

Example:
```javascript
refresh() {
    if (this.chart) this.chart.destroy();  // Clean up old chart
    super.refresh();
}
```

## Common Patterns

### Pattern 1: Real-Time Updates

```javascript
async loadData() {
    const data = await fetch('/api/modules/my-module').then(r => r.json());
    this.startLiveUpdates(data);
    return data;
}

startLiveUpdates(data) {
    this.interval = setInterval(async () => {
        const updated = await fetch('/api/modules/my-module').then(r => r.json());
        this.updateContent(updated);
    }, 2000);  // Every 2 seconds
}

onClose() {
    clearInterval(this.interval);
}
```

### Pattern 2: Interactive Forms

```javascript
renderContent() {
    const form = document.createElement('form');
    form.innerHTML = `
        <div class="form-group">
            <label>Setting</label>
            <input type="text" id="setting" />
        </div>
        <button type="submit">Apply</button>
    `;
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        this.applySetting(form.querySelector('#setting').value);
    });
    return form;
}

async applySetting(value) {
    const response = await fetch('/api/modules/my-module', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ setting: value })
    });
    // Update UI based on response
}
```

### Pattern 3: Multi-Tab Content

```javascript
renderContent() {
    return `
        <ul class="nav nav-tabs">
            <li class="nav-item">
                <a class="nav-link active" data-tab="overview">Overview</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-tab="details">Details</a>
            </li>
        </ul>
        <div id="overview" class="tab-pane active">...</div>
        <div id="details" class="tab-pane">...</div>
    `;
}
```

## Troubleshooting

### Module doesn't load
- Check browser console (F12)
- Verify template file exists
- Check Flask route returns 200 OK
- Ensure module ID matches dashboard registry

### Data not showing
- Check API endpoint returns valid JSON
- Verify `loadData()` doesn't throw
- Check `renderContent()` returns valid HTML
- Use `controller.showError()` for debugging

### Styling issues
- Verify CSS variables are imported via bundle
- Check specificity conflicts with Bootstrap
- Use `!important` sparingly, override via class names

## Best Practices

1. **Always provide fallback UI**: Show empty state, not blank
2. **Error handling**: Catch and display errors gracefully
3. **Responsive design**: Test on mobile (resize to 320px width)
4. **Loading states**: Always show spinner on initial load
5. **Cleanup**: Remove event listeners, clear intervals on destroy
6. **Accessibility**: Use semantic HTML, ARIA labels where needed
7. **Dark theme**: Use CSS variables, avoid hard-coded colors

## Resources

- [Chart.js Documentation](https://www.chartjs.org/docs/latest/)
- [Bootstrap 5 Components](https://getbootstrap.com/docs/5.0/components/)
- [Fetch API MDN](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
- [INIDS Detection Service API](../../src/detection/README.md)
- [INIDS Prevention Service API](../../src/prevention/README.md)

## Questions?

Refer to existing module implementations or check dashboard.js for integration patterns.
