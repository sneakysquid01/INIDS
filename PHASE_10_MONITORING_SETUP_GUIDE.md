# Phase 10.3: Monitoring & Alerting Setup Guide

**Version**: 1.0  
**Date**: April 16, 2026  
**Status**: Production Configuration  

---

## 1. Monitoring Architecture

### Components
```
┌─────────────────────────────────────────────────────┐
│                   Metrics Sources                    │
├─────────────────────────────────────────────────────┤
│  API Server  │  Workers  │  Database  │   Cache    │
└────────────┬─────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│              Metrics Collectors                      │
├─────────────────────────────────────────────────────┤
│  Prometheus │ Telegraf │ Datadog Agent │ StatsD    │
└────────────┬─────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│           Metrics Time Series Database               │
├─────────────────────────────────────────────────────┤
│  Prometheus  │  InfluxDB  │  CloudWatch             │
└────────────┬─────────────────────────────────────────┘
             │
             ├─────────┬──────────┬──────────┐
             ▼         ▼          ▼          ▼
          Grafana  AlertManager Datadog  Custom
          (Dashboard) (Alerting)  (APM)    (API)
```

---

## 2. Metrics Collection Setup

### 2.1 Prometheus Configuration

**File**: `/etc/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'
    environment: 'prod'

scrape_configs:
  - job_name: 'inids-api'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance

  - job_name: 'inids-worker'
    static_configs:
      - targets: ['localhost:5001']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:5432']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:6379']

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

**Enable Prometheus**:
```bash
systemctl enable prometheus
systemctl start prometheus
curl http://localhost:9090/-/healthy
```

### 2.2 Application Metrics Instrumentation

**File**: `src/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Request metrics
request_count = Counter(
    'inids_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'inids_request_duration_seconds',
    'Request latency',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Detection metrics
alerts_processed = Counter(
    'inids_alerts_processed_total',
    'Alerts processed',
    ['detector', 'result']
)

model_inference_time = Histogram(
    'inids_model_inference_seconds',
    'ML model inference time',
    ['model_name']
)

# Resource metrics
active_connections = Gauge(
    'inids_active_connections',
    'Active database connections'
)

cache_hit_rate = Gauge(
    'inids_cache_hit_rate',
    'Redis cache hit rate'
)
```

**Integrate in Flask**:
```python
from flask import Flask
from metrics import request_count, request_duration

app = Flask(__name__)

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    request_count.labels(
        method=request.method,
        endpoint=request.path,
        status=response.status_code
    ).inc()
    request_duration.labels(
        method=request.method,
        endpoint=request.path
    ).observe(duration)
    return response
```

### 2.3 Database Metrics

**PostgreSQL Exporter**:
```bash
# Install postgres_exporter
wget https://github.com/prometheus-community/postgres_exporter/releases/download/v0.11.1/postgres_exporter-0.11.1.linux-amd64.tar.gz
tar xvfz postgres_exporter-0.11.1.linux-amd64.tar.gz
sudo mv postgres_exporter /usr/local/bin/

# Configure
export DATA_SOURCE_NAME="postgresql://user:password@localhost:5432/inids?sslmode=disable"
postgres_exporter &

# Verify
curl http://localhost:9187/metrics
```

### 2.4 Redis Metrics

**Redis Exporter**:
```bash
# Install redis_exporter
wget https://github.com/oliver006/redis_exporter/releases/download/v1.45.0/redis_exporter-v1.45.0.linux-amd64.tar.gz
tar xvfz redis_exporter-v1.45.0.linux-amd64.tar.gz
sudo mv redis_exporter /usr/local/bin/

# Run
redis_exporter -redis.addr localhost:6379 &
```

---

## 3. Alert Definitions

### 3.1 AlertManager Configuration

**File**: `/etc/alertmanager/config.yml`

```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

route:
  receiver: 'default'
  group_by: ['alertname', 'cluster']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  routes:
    - match:
        severity: 'critical'
      receiver: 'critical-team'
      repeat_interval: 5m
    - match:
        severity: 'warning'
      receiver: 'ops-team'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'
        title: 'Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'critical-team'
    slack_configs:
      - channel: '#critical-alerts'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
      
  - name: 'ops-team'
    slack_configs:
      - channel: '#ops-alerts'
```

### 3.2 Alert Rules

**File**: `/etc/prometheus/rules/inids-alerts.yml`

```yaml
groups:
  - name: inids-alerts
    interval: 30s
    rules:

      # API Alerts
      - alert: APIServerDown
        expr: up{job="inids-api"} == 0
        for: 2m
        annotations:
          severity: critical
          summary: "API server is down"
          description: "API server {{ $labels.instance }} has been down for 2 minutes"

      - alert: HighAPILatency
        expr: histogram_quantile(0.95, inids_request_duration_seconds) > 1
        for: 5m
        annotations:
          severity: warning
          summary: "API latency is high"
          description: "95th percentile latency is {{ $value }}s (threshold: 1s)"

      - alert: HighErrorRate
        expr: (rate(inids_requests_total{status=~"5.."}[5m]) / rate(inids_requests_total[5m])) > 0.05
        for: 5m
        annotations:
          severity: warning
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}% (threshold: 5%)"

      # Database Alerts
      - alert: DatabaseDown
        expr: pg_up == 0
        for: 2m
        annotations:
          severity: critical
          summary: "Database is down"

      - alert: HighDatabaseConnections
        expr: sum(pg_stat_activity_count) > 80
        for: 5m
        annotations:
          severity: warning
          summary: "Database connection pool nearly exhausted"
          description: "{{ $value }} connections active (max: 100)"

      - alert: DatabaseDiskFull
        expr: (pg_database_size_bytes / pg_settings_max_wal_size_bytes) > 0.9
        for: 5m
        annotations:
          severity: critical
          summary: "Database disk space low"

      # Cache Alerts
      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 2m
        annotations:
          severity: critical
          summary: "Redis cache is down"

      - alert: RedisMemoryHigh
        expr: (redis_memory_used_bytes / redis_memory_max_bytes) > 0.9
        for: 5m
        annotations:
          severity: warning
          summary: "Redis memory usage is high"
          description: "Memory usage is {{ $value }}% (threshold: 90%)"

      # Detection Pipeline Alerts
      - alert: PipelineSlowed
        expr: rate(inids_alerts_processed_total[5m]) < 10
        for: 10m
        annotations:
          severity: warning
          summary: "Detection pipeline processing rate is slow"
          description: "Processing rate {{ $value }} alerts/sec (target: >10)"

      - alert: ModelInferenceSlow
        expr: histogram_quantile(0.95, inids_model_inference_seconds) > 2
        for: 5m
        annotations:
          severity: warning
          summary: "ML model inference is slow"
          description: "P95 inference time is {{ $value }}s (threshold: 2s)"

      # System Alerts
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) > 0.8
        for: 5m
        annotations:
          severity: warning
          summary: "High CPU usage detected"

      - alert: HighMemoryUsage
        expr: (process_resident_memory_bytes / 4294967296) > 0.8
        for: 5m
        annotations:
          severity: warning
          summary: "High memory usage detected"
          description: "Memory usage is {{ humanize $value }}GB (threshold: 3.2GB)"

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 5m
        annotations:
          severity: warning
          summary: "Disk space is low"
          description: "{{ $value }}% disk free (threshold: 10%)"

      # Security Alerts
      - alert: CSRFTokenFailureSpike
        expr: rate(inids_csrf_validation_failures_total[5m]) > 1
        for: 2m
        annotations:
          severity: warning
          summary: "CSRF token validation failure spike"
          description: "{{ $value }} failures/sec detected"

      - alert: SanitizationRejectionSpike
        expr: rate(inids_sanitization_rejections_total[5m]) > 10
        for: 2m
        annotations:
          severity: warning
          summary: "Input sanitization rejection spike"
          description: "{{ $value }} rejections/sec detected"

      - alert: UnauthorizedAccessAttempt
        expr: rate(inids_unauthorized_access_total[5m]) > 5
        for: 1m
        annotations:
          severity: critical
          summary: "Unauthorized access attempts detected"
          description: "{{ $value }} attempts/sec from potential attacker"
```

---

## 4. Grafana Dashboards

### 4.1 Dashboard Setup

**URL**: `http://localhost:3000`  
**Default Credentials**: `admin/admin`

### 4.2 System Overview Dashboard

Key metrics to display:
- API request rate (req/sec)
- API latency (p50, p95, p99)
- Error rate (%)
- Active users
- Detection pipeline throughput (alerts/min)
- Model inference time (ms)
- Database connections
- Cache hit rate (%)
- CPU usage (%)
- Memory usage (%)
- Disk space (%)

### 4.3 Security Monitoring Dashboard

Key metrics to display:
- CSRF token validations (total/success/fail)
- Input sanitization operations (accepted/rejected)
- Correlation ID generation rate
- Failed security validations (spike detection)
- Unauthorized access attempts
- Security module latencies
- Authentication failures

### 4.4 Database Performance Dashboard

Key metrics to display:
- Query latency distribution
- Connection pool usage
- Slow query count
- Transaction rate
- Replication lag (if applicable)
- Cache hit rate
- Disk I/O

---

## 5. Logging Aggregation

### 5.1 Log Collection Setup

**Using ELK Stack** (Elasticsearch, Logstash, Kibana)

**Filebeat Configuration** (`/etc/filebeat/filebeat.yml`):
```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/inids/*.log

processors:
  - add_fields:
      target: ''
      fields:
        environment: production
        app: inids

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "inids-%{+yyyy.MM.dd}"

setup.kibana:
  host: "localhost:5601"
```

**Enable Filebeat**:
```bash
systemctl enable filebeat
systemctl start filebeat
```

### 5.2 Log Retention Policy

- **Application Logs**: 30 days (hot), 90 days (warm), delete after 1 year
- **Security Logs**: 90 days (hot), 1 year (archive)
- **Audit Logs**: 2 years (cold storage)
- **Debug Logs**: 7 days

### 5.3 Log Indexing

**Important fields**:
```json
{
  "timestamp": "2026-04-16T10:30:45Z",
  "level": "ERROR",
  "service": "api",
  "correlation_id": "abc-123-def",
  "user_id": "user-456",
  "endpoint": "/api/detect",
  "method": "POST",
  "status": 500,
  "duration_ms": 1250,
  "error": "Database connection timeout",
  "stacktrace": "..."
}
```

---

## 6. Performance Baselines

### 6.1 Baseline Metrics

```
API Performance:
├── Request Latency (p50): 50ms ✅
├── Request Latency (p95): 200ms ✅
├── Request Latency (p99): 500ms ✅
├── Error Rate: <0.1% ✅
└── Throughput: >1000 req/sec ✅

Database Performance:
├── Query Latency (p50): 10ms ✅
├── Query Latency (p95): 50ms ✅
├── Connection Pool Usage: <50% ✅
└── Replication Lag: <1s ✅

Cache Performance:
├── Cache Hit Rate: >85% ✅
├── Cache Latency: <5ms ✅
└── Eviction Rate: <1% ✅

Detection Pipeline:
├── Throughput: >100 alerts/sec ✅
├── Latency (p95): <200ms ✅
└── Model Inference: <500ms ✅

Resource Usage:
├── CPU Usage: <50% ✅
├── Memory Usage: <60% ✅
├── Disk I/O: <70% ✅
└── Network I/O: <50% ✅
```

### 6.2 Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| API Latency (p95) | >500ms | >2000ms |
| Error Rate | >1% | >5% |
| Database Connections | >70 | >90 |
| Cache Hit Rate | <70% | <50% |
| Pipeline Throughput | <50 alerts/sec | <10 alerts/sec |
| CPU Usage | >70% | >90% |
| Memory Usage | >70% | >85% |
| Disk Space | <20% free | <10% free |

---

## 7. Custom Metrics

### 7.1 Security Module Metrics

```python
# Input Sanitization
sanitization_operations = Counter(
    'inids_sanitization_operations_total',
    'Sanitization operations',
    ['operation', 'result']
)

sanitization_latency = Histogram(
    'inids_sanitization_duration_seconds',
    'Sanitization operation latency'
)

# Correlation Tracing
correlation_id_generated = Counter(
    'inids_correlation_ids_generated_total',
    'Correlation IDs generated'
)

correlation_context_usage = Gauge(
    'inids_correlation_context_active',
    'Active correlation contexts'
)

# CSRF Protection
csrf_token_validations = Counter(
    'inids_csrf_token_validations_total',
    'CSRF token validations',
    ['result']
)

csrf_validation_latency = Histogram(
    'inids_csrf_validation_duration_seconds',
    'CSRF token validation latency'
)
```

---

## 8. Maintenance

### 8.1 Metrics Cleanup

```bash
# Archive old metrics
find /prometheus/wal -mtime +30 -delete

# Check storage size
du -sh /prometheus/

# Retention policy (prometheus.yml)
global:
  retention: 15d
```

### 8.2 Dashboard Maintenance

- Weekly: Review alert firing patterns
- Monthly: Update threshold baselines
- Quarterly: Dashboard refresh and optimization

### 8.3 Log Cleanup

```bash
# Archive old logs
tar -czf logs-$(date +%Y%m%d).tar.gz /var/log/inids/*.log
mv logs-*.tar.gz /archive/

# Clean logs
find /var/log/inids -mtime +30 -delete
```

---

## 9. Verification Checklist

- [ ] Prometheus running and scraping metrics
- [ ] AlertManager configured and routing alerts
- [ ] Grafana dashboards created and populated
- [ ] Log aggregation pipeline active
- [ ] All alert rules defined and tested
- [ ] Baseline metrics established
- [ ] Custom metrics instrumented
- [ ] Retention policies configured
- [ ] Backup monitoring in place
- [ ] On-call rotation configured

---

**Setup Completed**: April 16, 2026  
**Next Review**: April 23, 2026
