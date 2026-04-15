# Integration Guide: New Security Modules

This guide explains how to use the three new security modules added in Phase 4 of the INIDS hardening effort.

---

## 1. Input Sanitization Module (`src/input_sanitizer.py`)

### Purpose
Comprehensive input validation and sanitization to prevent injection attacks, malformed data, and resource exhaustion.

### Quick Start

```python
from src.input_sanitizer import (
    sanitize_id,
    sanitize_ip_address,
    sanitize_port,
    sanitize_severity,
    SanitizationError
)

@app.route('/api/alert/<alert_id>', methods=['GET'])
def get_alert(alert_id):
    try:
        alert_id = sanitize_id(alert_id)
    except SanitizationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Use sanitized alert_id
    alert = ops_store.get_alert(alert_id)
    return jsonify(alert)
```

### Available Functions

#### String Sanitization
```python
from src.input_sanitizer import sanitize_string

# Basic string with length limit
clean_string = sanitize_string(user_input, max_length=100)

# No special characters allowed
clean_id = sanitize_string(user_input, allow_special_chars=False)

# Convert to lowercase and allow spaces
clean_name = sanitize_string(user_input, lowercase=True, allow_spaces=True)
```

#### ID Fields
```python
from src.input_sanitizer import sanitize_id

# Validate IDs (alert_id, engine_id, action_id, etc.)
alert_id = sanitize_id(request.args.get('id'))
```

#### IP Addresses
```python
from src.input_sanitizer import sanitize_ip_address

# Validates both IPv4 and IPv6
source_ip = sanitize_ip_address(request.remote_addr)
```

#### Port Numbers
```python
from src.input_sanitizer import sanitize_port

# Ensures port is between 1 and 65535
port = sanitize_port(request.json.get('port'))
```

#### Severity Levels
```python
from src.input_sanitizer import sanitize_severity

# Validates: low, medium, high, critical
severity = sanitize_severity(request.json.get('severity'))
```

#### Numeric Values
```python
from src.input_sanitizer import sanitize_integer, sanitize_float

# Integer with bounds
limit = sanitize_integer(request.args.get('limit'), min_value=1, max_value=1000)

# Float with bounds
confidence = sanitize_float(request.json.get('confidence'), min_value=0.0, max_value=1.0)
```

### Error Handling

```python
from src.input_sanitizer import SanitizationError

try:
    alert_id = sanitize_id(user_input)
except SanitizationError as e:
    logger.warning("Invalid input: %s", e)
    return jsonify({"error": "Invalid input format"}), 400
```

---

## 2. Correlation Tracing Module (`src/correlation_tracing.py`)

### Purpose
Enables distributed request tracing across system components for better debugging and monitoring.

### Automatic Setup (Already Registered)

The middleware is automatically registered in `web_app/app.py`:

```python
from src.correlation_tracing import correlation_id_middleware
correlation_id_middleware(app)
```

This automatically:
- Generates/retrieves correlation IDs from request headers
- Attaches correlation ID to response headers
- Tracks request duration

### Using Correlation IDs in Code

```python
from src.correlation_tracing import get_correlation_id

@app.route('/api/predict', methods=['POST'])
def predict():
    correlation_id = get_correlation_id()
    logger.info(f"Processing prediction {correlation_id}")
    # All logs include the correlation ID in context
```

### Accessing Correlation ID

```python
from src.correlation_tracing import get_correlation_id

def background_worker():
    correlation_id = get_correlation_id()
    # Use for tracing
    print(f"Worker processing: {correlation_id}")
```

### Context Manager Usage

```python
from src.correlation_tracing import CorrelationContextManager

with CorrelationContextManager('custom-id') as cid:
    # Code here has custom correlation ID
    do_work()
    # Correlation ID automatically restored on exit
```

### Creating Correlation-Aware Loggers

```python
from src.correlation_tracing import create_correlation_logger

logger = create_correlation_logger(__name__)

# Automatically includes correlation ID in all logs
logger.info("This message includes correlation ID")
```

### Request Headers

Outgoing requests should include:
```
X-Correlation-ID: req_abc123def456
```

Response headers will include:
```
X-Correlation-ID: req_abc123def456
X-Request-Duration-MS: 125.45
```

---

## 3. CSRF Protection Module (`src/csrf_protection.py`)

### Purpose
Prevent Cross-Site Request Forgery attacks on sensitive endpoints.

### Automatic Setup (Already Registered)

The middleware is automatically registered in `web_app/app.py`:

```python
from src.csrf_protection import csrf_protect_middleware
csrf_protect_middleware(app)
```

This automatically:
- Generates CSRF tokens for sessions
- Injects tokens into template context
- Validates tokens on protected endpoints

### Protecting Endpoints

#### For HTML Forms

```python
@app.route('/alert/update', methods=['GET', 'POST'])
def update_alert():
    if request.method == 'POST':
        # Protection applied automatically by decorator
        return handle_update()
    
    # Render form with CSRF token
    return render_template('alert_form.html')
```

Template: `alert_form.html`
```html
<form method="POST">
    {{ csrf_token() }}
    <input type="text" name="alert_id" />
    <button type="submit">Update</button>
</form>
```

#### For API Endpoints (JSON)

```python
from src.csrf_protection import require_csrf_token

@app.route('/api/alert/update', methods=['POST'])
@require_csrf_token
def api_update_alert():
    data = request.get_json()
    # CSRF token validated before this executes
    return handle_update(data)
```

Client code:
```javascript
// Send token in header
fetch('/api/alert/update', {
    method: 'POST',
    headers: {
        'X-CSRF-Token': document.querySelector('#csrf_token').value,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({alert_id: '123'})
});
```

#### For AJAX Requests

```javascript
// Include token in request headers
$.ajax({
    url: '/api/alert/update',
    type: 'POST',
    headers: {
        'X-CSRF-Token': getCsrfToken()  // Get from meta tag or response
    },
    data: JSON.stringify({alert_id: '123'}),
    contentType: 'application/json'
});
```

### Exempting Endpoints

For webhooks or external integrations:

```python
from src.csrf_protection import exempt_from_csrf

@app.route('/webhook/alerts', methods=['POST'])
@exempt_from_csrf
def handle_webhook():
    # CSRF protection skipped
    return process_webhook()
```

### JSON Response Helper

```python
from src.csrf_protection import CSRFProtectedJsonResponse

@app.route('/api/config', methods=['POST'])
@require_csrf_token
def save_config():
    response = CSRFProtectedJsonResponse()
    try:
        # Process request
        response.set_data({'status': 'saved', 'id': config_id})
        response.add_csrf_token()
    except Exception as e:
        response.add_error('Failed to save config', 'save_error')
    
    return jsonify(response.get_data())
```

### Manual Token Access

```python
from src.csrf_protection import get_csrf_token, create_csrf_token_field

# Get token value
token = get_csrf_token()

# Create HTML field
html_field = create_csrf_token_field('csrf_token')

# Use in template
@app.route('/form')
def show_form():
    return render_template('form.html', csrf_field=create_csrf_token_field())
```

---

## Integration Examples

### Complete Endpoint with All Security

```python
from src.input_sanitizer import sanitize_id, sanitize_ip_address, SanitizationError
from src.correlation_tracing import get_correlation_id
from src.csrf_protection import require_csrf_token

@app.route('/api/alert/<alert_id>', methods=['POST'])
@require_csrf_token
def update_alert(alert_id):
    # Input validation
    try:
        alert_id = sanitize_id(alert_id)
        source_ip = sanitize_ip_address(request.remote_addr)
    except SanitizationError as e:
        return jsonify({"error": str(e)}), 400
    
    # Get correlation ID for tracing
    correlation_id = get_correlation_id()
    logger.info(f"Updating alert {alert_id} from {source_ip} [{correlation_id}]")
    
    # Process request
    try:
        alert = ops_store.get_alert(alert_id)
        if not alert:
            return jsonify({"error": "Alert not found"}), 404
        
        # Update alert
        new_status = request.json.get('status')
        ops_store.update_alert(alert_id, status=new_status)
        
        return jsonify({
            "status": "updated",
            "alert_id": alert_id,
            "_csrf_token": get_csrf_token()
        })
    except Exception as e:
        logger.exception(f"Failed to update alert: {e}")
        return jsonify({"error": "Internal error"}), 500
```

### Batch Operation with Sanitization

```python
@app.route('/api/alerts/batch-update', methods=['POST'])
@require_csrf_token
def batch_update_alerts():
    try:
        payload = request.get_json(force=True)
        alert_ids = payload.get('ids', [])
        
        # Sanitize all IDs
        sanitized_ids = []
        for alert_id in alert_ids:
            try:
                sanitized_ids.append(sanitize_id(alert_id))
            except SanitizationError:
                return jsonify({"error": "Invalid alert ID"}), 400
        
        # Process
        correlation_id = get_correlation_id()
        logger.info(f"Batch updating {len(sanitized_ids)} alerts [{correlation_id}]")
        
        updated = 0
        for alert_id in sanitized_ids:
            ops_store.update_alert(alert_id, status='processed')
            updated += 1
        
        return jsonify({"updated": updated})
    
    except SanitizationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Batch update failed")
        return jsonify({"error": "Internal error"}), 500
```

---

## Migration Checklist

When adding these modules to existing endpoints:

- [ ] Add input sanitization to user-supplied parameters
- [ ] Use correlation IDs for debugging and tracing
- [ ] Add CSRF protection to state-changing endpoints
- [ ] Test with invalid inputs
- [ ] Test request tracing across services
- [ ] Verify CSRF tokens in forms and AJAX
- [ ] Update API documentation with new headers
- [ ] Add unit tests for sanitization functions

---

## Performance Considerations

### Input Sanitization
- **Cost**: ~1-5ms per sanitization (regex-based)
- **Recommendation**: Use for all user inputs
- **Optimization**: Cache compiled patterns for high-throughput endpoints

### Correlation Tracing
- **Cost**: ~0.1ms per request (context variable operations)
- **Recommendation**: Enabled by default (no overhead)
- **Optimization**: Correlation IDs are inherited in async calls

### CSRF Protection
- **Cost**: ~1-3ms per request (token validation)
- **Recommendation**: Use for forms and sensitive APIs
- **Optimization**: Consider caching CSRF tokens in Redis for distributed systems

---

## Troubleshooting

### CSRF Token Not Validating

1. Ensure middleware is registered:
```python
csrf_protect_middleware(app)
```

2. Check token is in correct location:
- Form: `<input name="csrf_token">`
- Header: `X-CSRF-Token`
- JSON: `{"csrf_token": "..."}`

3. Verify session is enabled and persistent

### Correlation ID Not Appearing in Logs

1. Check middleware registration:
```python
correlation_id_middleware(app)
```

2. Use `create_correlation_logger()` for existing loggers:
```python
logger = create_correlation_logger(__name__)
```

3. Verify request headers don't override correlation ID

### Input Sanitization Too Strict

1. Review sanitization requirements
2. Use appropriate sanitization function:
   - `sanitize_string()` for flexible input
   - `sanitize_id()` for strict ID validation
3. Add custom patterns if needed

---

## Support & Questions

For issues or questions:
1. Check exception message for specific validation error
2. Review SanitizationError details in logs
3. Check correlation IDs for request tracing
4. Reference examples above

---

**Document Version**: 1.0  
**Last Updated**: April 16, 2026
