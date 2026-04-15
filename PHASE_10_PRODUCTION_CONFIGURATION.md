# Phase 10.5: Production Configuration

**Version**: 1.0  
**Date**: April 16, 2026  
**Environment**: Production  

---

## 1. Environment Configuration

### 1.1 Production Environment Variables

**File**: `/etc/inids/production.env`

```bash
# ============================================
# INIDS Production Configuration
# ============================================

# Application
APP_ENV=production
APP_DEBUG=false
APP_LOG_LEVEL=INFO
SECRET_KEY=${PRODUCTION_SECRET_KEY}  # Injected via secrets manager

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
WORKERS=4
WORKER_CLASS=sync
WORKER_TIMEOUT=30
KEEPALIVE=5

# Database
DATABASE_URL=postgresql://inids:${DB_PASSWORD}@db.prod.internal:5432/inids
DB_POOL_SIZE=100
DB_POOL_RECYCLE=3600
DB_ECHO=false
DB_SSL_MODE=require

# Cache
REDIS_URL=redis://:${REDIS_PASSWORD}@cache.prod.internal:6379/0
REDIS_SENTINEL_URLS=redis://sentinel1:26379,redis://sentinel2:26379,redis://sentinel3:26379
REDIS_SENTINEL_SERVICE=inids-cache
REDIS_SSL=true
CACHE_TTL=3600
CACHE_MAX_CONNECTIONS=50

# Security
CORS_ORIGINS=https://inids.example.com,https://api.inids.example.com
CSRF_TOKEN_ENABLED=true
SESSION_SECURE=true
SESSION_HTTPONLY=true
SESSION_SAMESITE=Strict
SSL_REDIRECT=true
HSTS_MAX_AGE=31536000

# Logging
LOG_FORMAT=json
LOG_OUTPUT=stdout
LOG_FILE=/var/log/inids/application.log
LOG_ROTATION_SIZE=104857600  # 100MB
LOG_RETENTION_DAYS=30
SYSLOG_ENABLED=true
SYSLOG_HOST=logs.prod.internal
SYSLOG_PORT=514

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
METRICS_ENABLED=true
METRICS_EXPORT_INTERVAL=60
HEALTH_CHECK_PATH=/api/health
HEALTH_CHECK_INTERVAL=30

# Model Management
ML_MODEL_PATH=/var/lib/inids/models
ML_MODEL_VERSION=latest
ML_BATCH_SIZE=128
ML_INFERENCE_TIMEOUT=2000  # milliseconds

# Detection Pipeline
DETECTION_ENABLED=true
ALERT_BUFFER_SIZE=10000
ALERT_FLUSH_INTERVAL=5
RISK_SCORING_ENABLED=true
ANOMALY_DETECTION_ENABLED=true

# Event Bus
EVENT_BUS_TYPE=kafka
EVENT_BUS_BROKERS=kafka1:9092,kafka2:9092,kafka3:9092
EVENT_BUS_TOPIC_PREFIX=inids
EVENT_RETENTION_DAYS=7
CONSUMER_GROUP=inids-prod-group

# Backup
BACKUP_ENABLED=true
BACKUP_FREQUENCY=hourly
BACKUP_RETENTION_DAYS=30
BACKUP_PATH=/mnt/backups/inids
BACKUP_ENCRYPTION_KEY=${BACKUP_ENCRYPTION_KEY}

# External Services
GEOIP_DB_PATH=/var/lib/inids/geoip/GeoLite2-City.mmdb
TLS_CERT_CHECK_ENABLED=true
HTTP_CLIENT_TIMEOUT=10000

# Performance Tuning
QUERY_TIMEOUT=5000
CONNECTION_POOL_TIMEOUT=10
MAX_CONNECTIONS=500
REQUEST_SIZE_LIMIT=10485760  # 10MB
RESPONSE_TIMEOUT=30

# Feature Flags
FEATURE_ADVANCED_ANALYTICS=true
FEATURE_CUSTOM_RULES=true
FEATURE_POLICY_ENFORCEMENT=true
FEATURE_REMEDIATION=true

# Compliance
GDPR_ENABLED=true
AUDIT_LOGGING_ENABLED=true
PII_MASKING_ENABLED=true
DATA_RETENTION_DAYS=90
```

### 1.2 Secrets Management

**Using HashiCorp Vault**:

```bash
# Initialize Vault
vault operator init -key-shares=5 -key-threshold=3

# Unseal Vault
vault operator unseal <key1>
vault operator unseal <key2>
vault operator unseal <key3>

# Configure authentication
vault auth enable kubernetes

# Create secret
vault kv put secret/inids/prod/db \
  username=inids \
  password=<strong-password> \
  host=db.prod.internal \
  database=inids

# Create policy
cat <<EOF > /tmp/inids-policy.hcl
path "secret/inids/prod/*" {
  capabilities = ["read", "list"]
}
EOF

vault policy write inids /tmp/inids-policy.hcl

# Create Kubernetes auth role
vault write auth/kubernetes/role/inids \
  bound_service_account_names=inids \
  bound_service_account_namespaces=production \
  policies=inids \
  ttl=24h
```

**Retrieve Secrets in Application**:

```python
import hvac

def get_secrets():
    client = hvac.Client(url='http://vault.prod.internal:8200')
    client.auth.kubernetes.login(role='inids')
    
    db_secret = client.secrets.kv.read_secret_version(
        path='inids/prod/db'
    )
    
    return {
        'db_user': db_secret['data']['data']['username'],
        'db_pass': db_secret['data']['data']['password'],
        'db_host': db_secret['data']['data']['host'],
    }
```

---

## 2. Docker Compose Production Setup

**File**: `deploy/docker-compose.prod.yml`

```yaml
version: '3.9'

services:
  # API Server
  api:
    image: inids:latest-prod
    container_name: inids-api
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgresql://inids:${DB_PASSWORD}@postgres:5432/inids
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
    ports:
      - "5000:5000"
    volumes:
      - /var/log/inids:/var/log/inids
      - /var/lib/inids/models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    depends_on:
      - postgres
      - redis
    restart: always
    networks:
      - inids-network

  # Detection Workers
  worker:
    image: inids:latest-prod
    container_name: inids-worker
    command: python -m inids.worker
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgresql://inids:${DB_PASSWORD}@postgres:5432/inids
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
    volumes:
      - /var/log/inids:/var/log/inids
      - /var/lib/inids/models:/app/models:ro
    depends_on:
      - postgres
      - redis
    restart: always
    networks:
      - inids-network

  # Database
  postgres:
    image: postgres:15-alpine
    container_name: inids-postgres
    environment:
      - POSTGRES_USER=inids
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=inids
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - /backups/db:/backups:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U inids"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: always
    networks:
      - inids-network

  # Cache
  redis:
    image: redis:7-alpine
    container_name: inids-redis
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: always
    networks:
      - inids-network

volumes:
  postgres-data:
    driver: local
  redis-data:
    driver: local

networks:
  inids-network:
    driver: bridge
```

---

## 3. Kubernetes Configuration

### 3.1 Deployment Manifest

**File**: `deploy/kubernetes/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inids-api
  namespace: production
  labels:
    app: inids
    component: api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: inids
      component: api
  template:
    metadata:
      labels:
        app: inids
        component: api
    spec:
      serviceAccountName: inids
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      containers:
      - name: api
        image: inids:latest-prod
        imagePullPolicy: IfNotPresent
        
        ports:
        - containerPort: 5000
          name: http
        - containerPort: 9090
          name: metrics
        
        env:
        - name: APP_ENV
          value: production
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: inids-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: inids-secrets
              key: redis-url
        
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 2Gi
        
        livenessProbe:
          httpGet:
            path: /api/health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /api/health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: logs
          mountPath: /var/log/inids
      
      volumes:
      - name: tmp
        emptyDir: {}
      - name: logs
        emptyDir: {}
      
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - inids
              topologyKey: kubernetes.io/hostname
```

### 3.2 Service Configuration

**File**: `deploy/kubernetes/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: inids-api
  namespace: production
spec:
  type: ClusterIP
  selector:
    app: inids
    component: api
  ports:
  - name: http
    port: 80
    targetPort: 5000
  - name: metrics
    port: 9090
    targetPort: 9090
```

### 3.3 Horizontal Pod Autoscaler

**File**: `deploy/kubernetes/hpa.yaml`

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inids-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inids-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

---

## 4. Deployment Scripts

### 4.1 Deploy Script

**File**: `scripts/deploy.sh`

```bash
#!/bin/bash
set -e

ENVIRONMENT=${1:-production}
IMAGE_VERSION=${2:-latest}

echo "🚀 Deploying INIDS to $ENVIRONMENT..."

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."
./scripts/pre-deploy-checks.sh $ENVIRONMENT

# Build image
echo "🔨 Building Docker image..."
docker build -t inids:$IMAGE_VERSION -f Dockerfile.prod .

# Push to registry
echo "📤 Pushing to registry..."
docker push registry.inids.io/inids:$IMAGE_VERSION

# Deploy using Kubernetes
echo "🔄 Deploying to Kubernetes..."
kubectl set image deployment/inids-api \
  inids-api=registry.inids.io/inids:$IMAGE_VERSION \
  --namespace=production \
  --record

# Wait for rollout
echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/inids-api -n production --timeout=5m

# Post-deployment checks
echo "✅ Running post-deployment checks..."
./scripts/post-deploy-checks.sh $ENVIRONMENT

echo "✅ Deployment completed successfully!"
```

### 4.2 Health Check Script

**File**: `scripts/health_check.sh`

```bash
#!/bin/bash

CHECKS_PASSED=0
CHECKS_FAILED=0

echo "🏥 Running health checks..."
echo ""

# Check 1: API Server
echo -n "API Server: "
if curl -s http://localhost:5000/api/health > /dev/null; then
    echo "✅ OK"
    ((CHECKS_PASSED++))
else
    echo "❌ FAILED"
    ((CHECKS_FAILED++))
fi

# Check 2: Database
echo -n "Database: "
if psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; then
    echo "✅ OK"
    ((CHECKS_PASSED++))
else
    echo "❌ FAILED"
    ((CHECKS_FAILED++))
fi

# Check 3: Redis
echo -n "Redis Cache: "
if redis-cli -h $REDIS_HOST ping > /dev/null; then
    echo "✅ OK"
    ((CHECKS_PASSED++))
else
    echo "❌ FAILED"
    ((CHECKS_FAILED++))
fi

# Check 4: Disk Space
echo -n "Disk Space: "
FREE_SPACE=$(df / | tail -1 | awk '{print $4}')
if [ $FREE_SPACE -gt 10485760 ]; then  # 10GB
    echo "✅ OK ($(($FREE_SPACE / 1048576))GB free)"
    ((CHECKS_PASSED++))
else
    echo "❌ FAILED ($(($FREE_SPACE / 1048576))GB free, need 10GB)"
    ((CHECKS_FAILED++))
fi

# Summary
echo ""
echo "================================"
echo "Health Check Summary"
echo "Passed: $CHECKS_PASSED"
echo "Failed: $CHECKS_FAILED"
echo "================================"

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✅ All checks passed!"
    exit 0
else
    echo "❌ Some checks failed!"
    exit 1
fi
```

### 4.3 Backup Script

**File**: `scripts/backup.sh`

```bash
#!/bin/bash

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/backups/inids/$BACKUP_DATE
RETENTION_DAYS=30

echo "💾 Starting backup process..."

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
echo "📦 Backing up database..."
pg_dump inids | gzip > $BACKUP_DIR/db-$BACKUP_DATE.sql.gz

# Backup configuration
echo "📦 Backing up configuration..."
tar -czf $BACKUP_DIR/config-$BACKUP_DATE.tar.gz /etc/inids/

# Verify backup
echo "✓ Backup complete: $BACKUP_DIR"

# Cleanup old backups
echo "🧹 Cleaning up old backups..."
find /backups/inids -mtime +$RETENTION_DAYS -delete

echo "✅ Backup process completed!"
```

---

## 5. Configuration Validation

### 5.1 Validation Tests

**File**: `tests/test_production_config.py`

```python
import os
import pytest
from src.web_app import app

class TestProductionConfig:
    
    def test_debug_disabled(self):
        """DEBUG mode should be disabled in production"""
        assert app.config['DEBUG'] is False
    
    def test_secret_key_set(self):
        """SECRET_KEY must be configured"""
        assert app.config['SECRET_KEY']
        assert len(app.config['SECRET_KEY']) > 32
    
    def test_database_configured(self):
        """Database must be configured"""
        assert os.environ.get('DATABASE_URL')
        assert 'postgresql' in os.environ.get('DATABASE_URL')
    
    def test_redis_configured(self):
        """Redis must be configured"""
        assert os.environ.get('REDIS_URL')
    
    def test_security_headers(self):
        """Security headers should be set"""
        with app.test_client() as client:
            response = client.get('/api/health')
            assert 'X-Content-Type-Options' in response.headers
            assert response.headers['X-Content-Type-Options'] == 'nosniff'
    
    def test_cors_configured(self):
        """CORS should be properly configured"""
        assert app.config.get('CORS_ORIGINS')
```

**Run Validation**:
```bash
pytest tests/test_production_config.py -v
```

### 5.2 Configuration Audit

**File**: `scripts/audit-config.sh`

```bash
#!/bin/bash

echo "🔍 Production Configuration Audit"
echo ""

ISSUES_FOUND=0

# Check 1: SECRET_KEY
if [ -z "$SECRET_KEY" ]; then
    echo "❌ SECRET_KEY not set"
    ((ISSUES_FOUND++))
else
    echo "✅ SECRET_KEY configured"
fi

# Check 2: DEBUG mode
if [ "$APP_DEBUG" = "true" ]; then
    echo "⚠️  DEBUG mode enabled (should be false)"
    ((ISSUES_FOUND++))
else
    echo "✅ DEBUG mode disabled"
fi

# Check 3: SSL
if ! grep -q "ssl_mode.*require" <<< "$DATABASE_URL"; then
    echo "⚠️  Database SSL not enforced"
    ((ISSUES_FOUND++))
else
    echo "✅ Database SSL enforced"
fi

# Summary
echo ""
if [ $ISSUES_FOUND -eq 0 ]; then
    echo "✅ Configuration audit passed!"
else
    echo "❌ $ISSUES_FOUND issue(s) found!"
fi
```

---

## 6. Pre-Deployment Checklist

- [ ] All environment variables set
- [ ] Secrets configured in Vault
- [ ] Database backup created
- [ ] Health checks pass
- [ ] Security tests pass
- [ ] Configuration validation passes
- [ ] Incident response team briefed
- [ ] Monitoring and alerting active
- [ ] Load testing completed
- [ ] Disaster recovery tested

---

## 7. Post-Deployment Checklist

- [ ] API responding to requests
- [ ] Database connections healthy
- [ ] Cache warm
- [ ] Monitoring metrics flowing
- [ ] Alerts configured and firing (test)
- [ ] Logging aggregation active
- [ ] Performance baselines established
- [ ] Error rate < 0.1%
- [ ] Latency p95 < 500ms
- [ ] Customer notifications sent

---

**Configuration Version**: 1.0  
**Last Updated**: April 16, 2026  
**Next Review**: July 16, 2026
