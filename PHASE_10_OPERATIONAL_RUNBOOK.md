# Phase 10.2: Operational Runbook

**Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: April 16, 2026  

---

## Table of Contents

1. [Deployment Procedures](#deployment-procedures)
2. [Health Check Procedures](#health-check-procedures)
3. [Scaling Guidelines](#scaling-guidelines)
4. [Maintenance Schedule](#maintenance-schedule)
5. [Emergency Procedures](#emergency-procedures)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Deployment Procedures

### Pre-Deployment Checklist

- [ ] Environment variables configured
- [ ] Database initialized and tested
- [ ] Redis connection verified
- [ ] SSL certificates valid (>30 days)
- [ ] Backup systems operational
- [ ] Monitoring agents installed
- [ ] Logging aggregation active
- [ ] All security modules loaded
- [ ] Performance baselines established
- [ ] Incident response team briefed

### Deployment Steps

#### Step 1: Prepare Environment
```bash
# Activate virtual environment
source /opt/inids/venv/bin/activate

# Load environment variables
export $(cat /etc/inids/production.env | xargs)

# Verify configuration
python -c "from src.web_app import app; print('Config OK')"
```

#### Step 2: Pre-Flight Checks
```bash
# Run health checks
./scripts/health_check.sh

# Expected output:
# ✅ Database connection: OK
# ✅ Redis connection: OK
# ✅ Disk space: OK (>10GB)
# ✅ Memory available: OK (>4GB)
# ✅ Python modules: OK
# ✅ Security modules: OK
```

#### Step 3: Initialize/Migrate Database
```bash
# Apply migrations
python -m alembic upgrade head

# Seed baseline data if needed
python scripts/seed_production_data.py

# Verify data integrity
python scripts/verify_data.py
```

#### Step 4: Start Services
```bash
# Using Docker Compose
docker-compose -f deploy/docker-compose.prod.yml up -d

# Using Systemd (alternative)
systemctl start inids-api inids-worker inids-scheduler

# Verify service startup (wait 10 seconds)
sleep 10
./scripts/health_check.sh
```

#### Step 5: Verify Deployment
```bash
# Check API endpoints
curl -s http://localhost:5000/api/health | jq .

# Check worker status
curl -s http://localhost:5000/api/admin/workers | jq .

# Check pipeline status
curl -s http://localhost:5000/api/admin/pipeline | jq .
```

#### Step 6: Enable Monitoring & Alerts
```bash
# Start monitoring collection
systemctl start prometheus-agent telegraf

# Verify metrics flow
curl -s http://localhost:9090/metrics | head -20

# Enable alert rules
curl -X POST http://localhost:9090/-/reload
```

**Deployment Status**: ✅ COMPLETE

---

## Health Check Procedures

### Quick Health Check (30 seconds)
```bash
./scripts/health_check.sh
```

**Expected Output**:
```
✅ API Service: UP (response time: 45ms)
✅ Database: Connected (1200ms)
✅ Redis Cache: Connected (3ms)
✅ Detection Pipeline: Running (45 alerts/min)
✅ ML Model: Loaded (inference: 120ms)
✅ Event Bus: Active (2,100 events/hour)
✅ Security Modules: Loaded
  ├── Input Sanitizer: OK
  ├── Correlation Tracing: OK
  └── CSRF Protection: OK
```

### Detailed Health Check (5 minutes)
```bash
./scripts/health_check.sh --detailed
```

Checks include:
- API response times (p99 < 500ms)
- Database query performance
- Redis memory usage
- Memory leaks detection
- Certificate expiry (>30 days)
- Disk space usage (<90%)
- Network latency
- Backup status
- Log file sizes

### Dashboard Health View

**URL**: `https://inids.example.com/admin/health`

Metrics displayed:
- Request throughput (req/sec)
- Response time distribution
- Error rate
- CPU usage
- Memory usage
- Network I/O
- Database connections
- Cache hit rate
- Model inference time

---

## Scaling Guidelines

### Horizontal Scaling

#### Adding API Instances
```bash
# Scale API servers from 3 to 5 instances
kubectl scale deployment inids-api --replicas=5

# Verify scaling
kubectl get pods -l app=inids-api

# Monitor scaling progress
watch kubectl get pods -l app=inids-api
```

**Load Balancer Behavior**:
- Automatically detects new instances
- Health checks every 30 seconds
- Connection draining on scale-down
- No data loss guaranteed

#### Adding Worker Instances
```bash
# Scale workers from 2 to 4
kubectl scale deployment inids-worker --replicas=4

# Distribute tasks automatically
# Workers pick up tasks from event queue
```

**Worker Scaling Limits**:
- Max 10 instances per deployment
- Each worker needs 2GB RAM, 2 CPU cores
- Task distribution is automatic
- Monitor queue depth: `curl /api/admin/queue-depth`

### Vertical Scaling

#### Increase Resource Limits
```bash
# Edit deployment resource limits
kubectl edit deployment inids-api

# Modify:
# resources:
#   limits:
#     memory: "4Gi"    # was 2Gi
#     cpu: "2000m"     # was 1000m
#   requests:
#     memory: "2Gi"
#     cpu: "1000m"

# Apply changes
kubectl apply -f deployment.yaml
```

### Auto-Scaling Policy

```yaml
# HorizontalPodAutoscaler configuration
minReplicas: 3
maxReplicas: 10
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80

# Scaling behavior
scale-down:
  stabilizationWindowSeconds: 300
  policies:
    - type: Percent
      value: 50
      periodSeconds: 60

scale-up:
  stabilizationWindowSeconds: 30
  policies:
    - type: Percent
      value: 100
      periodSeconds: 30
```

---

## Maintenance Schedule

### Daily Tasks
- [ ] 02:00 UTC - Database backup
- [ ] 03:00 UTC - Log rotation
- [ ] 04:00 UTC - Cache cleanup
- [ ] 05:00 UTC - Health metrics review

### Weekly Tasks
- [ ] Monday 00:00 - Security patch review
- [ ] Wednesday 14:00 - Performance analysis
- [ ] Friday 18:00 - Backup verification
- [ ] Sunday 20:00 - Capacity planning review

### Monthly Tasks
- [ ] 1st - Security audit
- [ ] 8th - Dependency updates
- [ ] 15th - Database optimization
- [ ] 25th - Disaster recovery drill

### Quarterly Tasks
- [ ] Q1/Q2/Q3/Q4 - Full system penetration test
- [ ] Compliance audit
- [ ] Load testing
- [ ] Disaster recovery validation

### Maintenance Windows

**Preferred Maintenance Windows**:
- Tuesday-Thursday: 02:00-06:00 UTC
- Saturday-Sunday: 10:00-14:00 UTC

**Emergency Maintenance**:
- Can be performed anytime with 1 hour notice
- Automated failover activated
- Zero-downtime deployment where possible

---

## Emergency Procedures

### Service Outage Response

#### 1. Immediate Response (0-5 minutes)
```bash
# Check service status
systemctl status inids-api inids-worker

# View recent logs
tail -50 /var/log/inids/api.log
tail -50 /var/log/inids/worker.log

# Check resource usage
free -h
df -h
top -bn1 | head -20
```

#### 2. Investigation (5-15 minutes)
```bash
# Check database connectivity
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1"

# Check Redis connectivity
redis-cli -h $REDIS_HOST ping

# Check API response
curl -v http://localhost:5000/api/health

# Review error logs
grep ERROR /var/log/inids/*.log | tail -100
```

#### 3. Recovery (15-30 minutes)

**Option A: Service Restart**
```bash
# Stop service gracefully
systemctl stop inids-api

# Wait for graceful shutdown (max 30s)
sleep 30

# Restart service
systemctl start inids-api

# Verify recovery
./scripts/health_check.sh
```

**Option B: Failover to Backup**
```bash
# Activate standby instance
curl -X POST http://backup-server:5000/api/admin/activate-primary

# Verify primary is active
curl http://primary-server:5000/api/health

# Monitor failover completion
watch curl http://primary-server:5000/api/health
```

**Option C: Full Rollback**
```bash
# Rollback to previous deployment
kubectl rollout undo deployment/inids-api

# Verify rollback
kubectl get deployment inids-api -o jsonpath='{.status.conditions[]}'

# Monitor service recovery
watch ./scripts/health_check.sh
```

### Database Corruption Recovery

#### 1. Detect Corruption
```sql
-- Run integrity check
PRAGMA integrity_check;  -- SQLite
DBCC CHECKDB;           -- SQL Server
CHECK TABLE schema_name;  -- MySQL
```

#### 2. Recover from Backup
```bash
# Stop API service
systemctl stop inids-api

# Restore from backup
psql -h $DB_HOST -U $DB_USER $DB_NAME < backups/latest.sql

# Verify data
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT COUNT(*) FROM alerts"

# Restart service
systemctl start inids-api
```

### Memory Leak Response

#### 1. Detect Leak
```bash
# Monitor memory usage
watch -n 5 'ps aux | grep inids'

# Check for steady increase over 24 hours
./scripts/memory_trend.sh
```

#### 2. Mitigate
```bash
# Increase memory limit temporarily
# Edit /etc/systemd/system/inids-api.service
# MemoryLimit=8G  # was 4G

# Reload systemd
systemctl daemon-reload

# Restart service
systemctl restart inids-api
```

#### 3. Investigate
```bash
# Check for pending transactions
ps aux | grep -E "BEGIN|LOCK"

# Kill stuck processes if needed
pkill -f "stuck_query"

# Analyze memory using profiler
python -m memory_profiler src/web_app.py
```

### Security Incident Response

#### 1. Detect Anomaly
- Log spike in failed CSRF validations
- Spike in input sanitization rejections
- Unauthorized correlation ID usage

#### 2. Immediate Actions
```bash
# Enable enhanced logging
sed -i 's/level=INFO/level=DEBUG/' /etc/inids/logging.conf
systemctl restart inids-api

# Isolate affected service
iptables -A INPUT -s $ATTACKER_IP -j DROP

# Notify security team
curl -X POST https://slack.com/api/chat.postMessage \
  -H 'Content-Type: application/json' \
  -d '{"channel":"#security","text":"Security incident detected"}'
```

#### 3. Investigation
```bash
# Collect logs for analysis
tar -czf incident-$(date +%s).tar.gz /var/log/inids/

# Review security audit log
grep SECURITY /var/log/inids/*.log | head -200

# Check for data exfiltration
grep "SELECT.*FROM" /var/log/inids/db.log | wc -l
```

---

## Troubleshooting Guide

### Problem: High API Response Time (>500ms)

**Root Causes** (in order of likelihood):
1. Database query slow
2. Memory pressure / GC pauses
3. Network latency
4. ML model inference slow
5. External service latency

**Diagnosis Steps**:
```bash
# 1. Check database
psql -x -c "SELECT pid, query, wait_event FROM pg_stat_activity WHERE state != 'idle'"

# 2. Check memory
free -h
# If free < 500MB, restart service

# 3. Check network
ping -c 5 database.example.com
mtr -r -c 100 database.example.com

# 4. Check model inference
curl http://localhost:5000/api/admin/model-stats

# 5. Check external services
curl -w "@curl-format.txt" https://external-api.com/
```

**Solutions**:
- Restart affected service: `systemctl restart inids-api`
- Increase database connections
- Add API instance: `kubectl scale deployment inids-api --replicas=5`
- Optimize slow queries
- Clear cache: `redis-cli FLUSHALL`

### Problem: High Memory Usage (>80%)

**Root Causes**:
1. Memory leak in application
2. Large cache accumulation
3. Model loaded in memory
4. Query result set too large

**Diagnosis**:
```bash
# Check process memory
ps aux --sort=-%mem | head -5

# Check cache size
redis-cli INFO memory

# Check model size
du -sh /opt/inids/models/

# Check for leak
python -m memory_profiler --help
```

**Solutions**:
- Clear Redis cache: `redis-cli FLUSHALL`
- Reduce batch size for queries
- Increase VM memory limit
- Restart service to clear leaks

### Problem: High CPU Usage (>90%)

**Root Causes**:
1. CPU-intensive task (ML inference)
2. Unoptimized query
3. Busy-wait loop
4. High concurrency

**Diagnosis**:
```bash
# Check CPU usage by process
top -p $(pidof python)

# Check for hot functions
python -m cProfile -s cumtime src/detection.py

# Check system-wide load
uptime
```

**Solutions**:
- Add worker instances
- Reduce batch size
- Optimize algorithms
- Scale horizontally

### Problem: Database Connection Errors

**Root Causes**:
1. Database server down
2. Connection pool exhausted
3. Network connectivity issue
4. Authentication failure

**Diagnosis**:
```bash
# Test connection
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1"

# Check connection pool
curl http://localhost:5000/api/admin/db-connections

# Check network
ping $DB_HOST
traceroute $DB_HOST

# Check logs
tail -50 /var/log/inids/api.log | grep -i database
```

**Solutions**:
- Restart database server
- Increase connection pool size
- Check firewall rules
- Verify credentials

---

## Contact & Escalation

### Support Contacts

| Level | Contact | Response Time |
|-------|---------|---|
| Tier 1 | oncall@inids-team.com | 15 minutes |
| Tier 2 | security-team@inids-team.com | 30 minutes |
| Tier 3 | engineering-lead@inids-team.com | 1 hour |
| Critical | +1-XXX-XXX-XXXX | Immediate |

### Escalation Criteria

- **Critical**: Production data loss, security breach, complete outage
- **High**: >5% error rate, >1000ms latency, partial outage
- **Medium**: >1% error rate, >500ms latency, degraded performance
- **Low**: Informational alerts, <1% error rate

---

**Runbook Version**: 1.0  
**Last Updated**: April 16, 2026  
**Next Review**: April 16, 2027
