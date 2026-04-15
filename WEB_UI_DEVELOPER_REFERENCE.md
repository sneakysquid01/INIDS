# INIDS Web UI - Developer Reference

## Quick Reference for Extending or Modifying the UI

---

## 🏗️ Adding a New UI Page

### Step 1: Create HTML Template
**File**: `web_app/templates/mypage.html`

```html
{% extends "base.html" %}

{% block title %}My Feature{% endblock %}

{% block content %}
<div class="container mt-4">
    <h1>My Feature</h1>
    <!-- Your content here -->
</div>
{% endblock %}

{% block scripts %}
<script src="{{ url_for('static', filename='js/mypage.js') }}"></script>
{% endblock %}
```

### Step 2: Create JavaScript Logic
**File**: `web_app/static/js/mypage.js`

```javascript
async function loadMyData() {
    try {
        const response = await fetch('/api/my-endpoint');
        if (!response.ok) throw new Error('Failed to load');
        const data = await response.json();
        renderMyData(data);
    } catch (error) {
        showError('Error loading data: ' + error.message);
    }
}

function renderMyData(data) {
    // Render your data here
}

// Load on page load
document.addEventListener('DOMContentLoaded', loadMyData);
```

### Step 3: Add Flask Route
**File**: `web_app/app.py`

```python
@app.route('/mypage')
@require_role('analyst')
def mypage():
    return render_template('mypage.html')
```

### Step 4: Add Navigation Link
**File**: `web_app/templates/index_main.html`

Add to feature cards section:
```html
<div class="col-md-6 col-lg-4 mb-3">
    <a href="/mypage" class="card-link">
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">My Feature</h5>
                <p class="card-text">Description of my feature</p>
            </div>
        </div>
    </a>
</div>
```

---

## 📡 API Integration Patterns

### Pattern 1: Simple GET with Rendering

```javascript
async function loadData() {
    const response = await fetch('/api/data');
    const data = await response.json();
    
    data.forEach(item => {
        // Process each item
    });
}
```

### Pattern 2: POST with Error Handling

```javascript
async function updateData(id, payload) {
    try {
        const response = await fetch(`/api/data/${id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error(response.statusText);
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to update');
    }
}
```

### Pattern 3: Query Parameters

```javascript
async function filterData(severity, status) {
    const params = new URLSearchParams({
        severity: severity,
        status: status,
        limit: 100
    });
    
    const response = await fetch(`/api/alerts?${params}`);
    return await response.json();
}
```

### Pattern 4: File Download

```javascript
async function downloadReport() {
    const response = await fetch('/api/export');
    const blob = await response.blob();
    
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'report.csv';
    a.click();
}
```

---

## 🎨 UI Component Patterns

### Alert/Toast Notification

```javascript
function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show';
    alert.innerHTML = message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';
    document.body.insertBefore(alert, document.body.firstChild);
    setTimeout(() => alert.remove(), 3000);
}

function showError(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.innerHTML = message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';
    document.body.insertBefore(alert, document.body.firstChild);
}
```

### Modal Dialog

```html
<div class="modal fade" id="myModal">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Title</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body">
                <!-- Content here -->
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                <button type="button" class="btn btn-primary" onclick="handleAction()">Action</button>
            </div>
        </div>
    </div>
</div>
```

```javascript
function openModal() {
    new bootstrap.Modal(document.getElementById('myModal')).show();
}
```

### Table with Data

```html
<table class="table">
    <thead>
        <tr>
            <th>Column 1</th>
            <th>Column 2</th>
        </tr>
    </thead>
    <tbody id="tableBody">
        <!-- Filled by JavaScript -->
    </tbody>
</table>
```

```javascript
function renderTable(data) {
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = data.map(row => `
        <tr>
            <td>${row.col1}</td>
            <td>${row.col2}</td>
        </tr>
    `).join('');
}
```

### Form Input Example

```html
<form id="myForm">
    <div class="mb-3">
        <label for="input1" class="form-label">Field 1</label>
        <input type="text" class="form-control" id="input1" required>
    </div>
    <div class="mb-3">
        <label for="input2" class="form-label">Field 2</label>
        <select class="form-select" id="input2">
            <option value="1">Option 1</option>
            <option value="2">Option 2</option>
        </select>
    </div>
    <button type="submit" class="btn btn-primary">Submit</button>
</form>
```

```javascript
document.getElementById('myForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        field1: document.getElementById('input1').value,
        field2: document.getElementById('input2').value
    };
    
    await updateData(formData);
});
```

### Toggle Button with State

```html
<button class="btn btn-outline-primary" id="toggleBtn" onclick="toggle()">OFF</button>
```

```javascript
let state = false;

function toggle() {
    state = !state;
    const btn = document.getElementById('toggleBtn');
    
    if (state) {
        btn.classList.remove('btn-outline-primary');
        btn.classList.add('btn-primary');
        btn.textContent = 'ON';
    } else {
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-outline-primary');
        btn.textContent = 'OFF';
    }
}
```

---

## 🔒 Security Best Practices

### XSS Prevention

```javascript
// ❌ BAD: Risk of XSS injection
element.innerHTML = userInput;

// ✅ GOOD: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
element.innerHTML = escapeHtml(userInput);

// ✅ BETTER: Use textContent for text-only
element.textContent = userInput;
```

### CSRF Token Usage

```javascript
// For POST requests, include CSRF token
async function makeRequest(url, method, data) {
    const response = await fetch(url, {
        method: method,
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').content
        },
        body: JSON.stringify(data)
    });
}
```

### Input Validation

```javascript
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function validateIP(ip) {
    const re = /^(\d{1,3}\.){3}\d{1,3}$/;
    return re.test(ip);
}
```

---

## ⚙️ Debugging Tips

### Browser Console Debugging

```javascript
// Log with timestamp
console.log('[' + new Date().toISOString() + '] Message:', data);

// Group related logs
console.group('API Call');
console.log('URL:', url);
console.log('Method:', method);
console.log('Response:', response);
console.groupEnd();

// Conditional debug logging
const DEBUG = true;
if (DEBUG) console.log('Debug:', data);
```

### Network Tab Analysis

- **Open DevTools**: F12
- **Network Tab**: See all API calls
- **Right-click request**: Copy as cURL
- **Check headers**: Verify Content-Type, Auth
- **Check response**: View JSON prettified

### Common Issues

```javascript
// Issue: CORS error
// Solution: Ensure backend allows requests from localhost:5000

// Issue: 401 Unauthorized
// Solution: Check role permissions, verify session

// Issue: 404 Not Found
// Solution: Check API endpoint URL, verify route exists

// Issue: Data not updating
// Solution: Check auto-refresh interval, verify API response
```

---

## 📊 Working with Chart Data

### Rendering Simple Chart (HTML5 Canvas)

```html
<canvas id="myChart"></canvas>
```

```javascript
function renderChart(data) {
    const canvas = document.getElementById('myChart');
    const ctx = canvas.getContext('2d');
    
    // Simple bar chart
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const barWidth = canvas.width / data.length;
    data.forEach((value, index) => {
        ctx.fillStyle = '#0066cc';
        ctx.fillRect(
            index * barWidth,
            canvas.height - (value * 10),
            barWidth - 1,
            value * 10
        );
    });
}
```

---

## 🎯 Common Tasks

### Reload Page Every 30 Seconds

```javascript
setInterval(() => {
    location.reload();
}, 30000);
```

### Add Loading Spinner

```html
<div id="spinner" style="display:none;">
    <div class="spinner-border" role="status">
        <span class="visually-hidden">Loading...</span>
    </div>
</div>
```

```javascript
function showSpinner() {
    document.getElementById('spinner').style.display = 'block';
}

function hideSpinner() {
    document.getElementById('spinner').style.display = 'none';
}
```

### Format Timestamp

```javascript
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}
```

### Color Code Status

```javascript
function getStatusClass(status) {
    switch(status) {
        case 'critical': return 'danger';
        case 'high': return 'warning';
        case 'medium': return 'info';
        case 'low': return 'success';
        default: return 'secondary';
    }
}
```

---

## 📚 File Structure Guide

```
web_app/
├── app.py                          # Main Flask app
├── templates/
│   ├── base.html                   # Base template
│   ├── index_main.html             # Landing page
│   ├── alerts.html                 # Alerts page
│   ├── actions.html                # Actions page
│   ├── detection.html              # Detection page
│   ├── engines.html                # Engines page
│   ├── policy.html                 # Policy page
│   └── ...
├── static/
│   ├── js/
│   │   ├── alerts.js               # Alerts logic
│   │   ├── actions.js              # Actions logic
│   │   ├── detection.js            # Detection logic
│   │   ├── engines.js              # Engines logic
│   │   ├── policy.js               # Policy logic
│   │   └── ...
│   ├── css/
│   │   └── style.css               # Custom styles (minimal)
│   └── bootstrap-5/                # Bootstrap files
└── uploads/                        # User uploads (if any)
```

---

## 🚀 Performance Optimization

### Reduce API Calls

```javascript
// ❌ BAD: Call API for each item
items.forEach(item => {
    fetch(`/api/detail/${item.id}`).then(r => r.json());
});

// ✅ GOOD: Get all at once
fetch(`/api/details?ids=${itemIds.join(',')}`).then(r => r.json());
```

### Lazy Loading

```javascript
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            loadMoreData();
        }
    });
});

observer.observe(document.getElementById('sentinel'));
```

### Caching

```javascript
const cache = {};

async function getCachedData(key) {
    if (cache[key]) {
        return cache[key];
    }
    
    const data = await fetch(`/api/${key}`).then(r => r.json());
    cache[key] = data;
    return data;
}
```

---

## 💡 Best Practices

1. **Always validate inputs** - Check length, format, allowed values
2. **Handle errors gracefully** - Show user-friendly messages
3. **Use proper HTTP methods** - GET for retrieval, POST for creation, etc.
4. **Cache when appropriate** - Reduce unnecessary API calls
5. **Test in browser** - Use DevTools to verify behavior
6. **Keep code simple** - Avoid complex nesting and abstractions
7. **Comment confusing code** - Make it maintainable
8. **Use semantic HTML** - Improve accessibility
9. **Mobile-first design** - Works on all screen sizes
10. **Monitor performance** - Use Lighthouse and DevTools

---

## 📖 Learning Resources

- [MDN Web Docs](https://developer.mozilla.org/)
- [Bootstrap Documentation](https://getbootstrap.com/docs/)
- [Fetch API Guide](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
- [JavaScript ES6 Features](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Happy coding!** ✨
