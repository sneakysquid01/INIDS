# Phase 10.4: Incident Response Playbook

**Version**: 1.0  
**Date**: April 16, 2026  
**Classification**: Internal Use Only  

---

## Table of Contents

1. [Incident Classification](#incident-classification)
2. [Response Procedures](#response-procedures)
3. [Escalation Matrix](#escalation-matrix)
4. [Post-Incident Review](#post-incident-review)
5. [Communication Plan](#communication-plan)
6. [Recovery Procedures](#recovery-procedures)

---

## Incident Classification

### Severity Levels

| Level | Impact | Response Time | Duration |
|-------|--------|---|---|
| **SEV-1 (Critical)** | Complete outage, data loss risk | 5 minutes | 4 hour limit |
| **SEV-2 (High)** | Partial outage, degraded service | 30 minutes | 8 hour limit |
| **SEV-3 (Medium)** | Single feature down, workaround available | 2 hours | 24 hour limit |
| **SEV-4 (Low)** | Minor issue, no impact on users | 1 business day | Backlog |

### Incident Types

#### Type A: Service Outage
- API server down
- Database unavailable
- Cache failure
- Complete loss of service

**RTO**: <15 minutes  
**RPO**: <1 hour

#### Type B: Performance Degradation
- Latency >2000ms p95
- Error rate >5%
- Throughput <50% baseline
- Timeout spikes

**RTO**: <1 hour  
**RPO**: <4 hours

#### Type C: Security Incident
- Unauthorized access detected
- Data exfiltration suspected
- Attack pattern detected
- Credential compromise

**RTO**: <30 minutes  
**RPO**: <1 hour

#### Type D: Data Integrity Issue
- Database corruption
- Message queue loss
- Cache inconsistency
- Backup failure

**RTO**: <2 hours  
**RPO**: <15 minutes

#### Type E: Dependency Failure
- External API down
- Third-party service unavailable
- Network connectivity issue
- DNS resolution failure

**RTO**: <30 minutes  
**RPO**: <1 hour

---

## Response Procedures

### Step 1: Detection & Initial Assessment (0-5 minutes)

**Trigger Sources**:
- Automated alerting system (primary)
- User/customer reports (secondary)
- Manual monitoring dashboard review (tertiary)

**Initial Actions**:
```
1. Check alert details
   - Severity level
   - Affected component
   - Time of incident
   - Alert history

2. Verify alert validity
   - Check dashboard health
   - Confirm with manual tests
   - Rule out false positives

3. Declare incident
   - Assign incident ID (INC-YYYYMMDD-XXX)
   - Log in incident tracker
   - Start response timer

4. Notify on-call team
   - Page primary on-call
   - Notify Slack #incidents
   - Start war room if SEV-1
```

**Example Alert Message**:
```
🚨 CRITICAL ALERT 🚨
Incident ID: INC-20260416-001
Service: API Server
Severity: SEV-1
Status: Active response

Alert: APIServerDown
Affected Instance: prod-api-01
Time: 2026-04-16T10:30:45Z
Last OK: 2026-04-16T10:28:30Z

Action: Immediate investigation started
```

### Step 2: Impact Assessment (5-10 minutes)

**Data to Collect**:
```
□ Number of users affected
□ Business impact
□ Downstream dependencies
□ Customer notification required?
□ SLA violation risk?
□ Data loss risk?
```

**Assessment Questions**:
1. Is this incident ongoing or intermittent?
2. What percentage of users are affected? (0%, <1%, 1-5%, 5-25%, 25-100%)
3. Can users work around the issue?
4. Are other systems depending on this one?
5. Is there data loss risk?

**Severity Confirmation**:
- If customer-impacting + ongoing = **SEV-1**
- If partial impact + ongoing = **SEV-2**
- If minor impact or workaround = **SEV-3**

### Step 3: Root Cause Investigation (10-30 minutes)

**Investigation Checklist**:
```
□ Check recent deployments
  - What was deployed in last hour?
  - Any configuration changes?
  - Database migrations?

□ Check system logs
  - API error logs (ERROR, CRITICAL level)
  - Database logs for connection errors
  - Network logs for connectivity issues

□ Check metrics
  - CPU usage spike?
  - Memory usage spike?
  - Disk space issue?
  - Network latency?

□ Check dependencies
  - Database responding?
  - Redis responding?
  - External services up?
  - DNS resolving?

□ Check resource limits
  - Connection pool exhausted?
  - Thread pool exhausted?
  - File descriptor limit?
  - Memory limit exceeded?
```

**Common Root Causes & Quick Fixes**:

| Symptom | Likely Cause | Quick Fix |
|---------|---|---|
| API timeout errors | Database slow | Restart DB service, check slow queries |
| 503 Service Unavailable | Workers crashed | Restart worker service |
| High latency | Memory pressure | Increase memory limit, restart service |
| Connection refused | Service not running | `systemctl start service-name` |
| Auth failures | Secret/key issue | Verify environment variables |
| Data missing | Backup not running | Check backup job, restore if needed |

### Step 4: Mitigation & Recovery (30-60 minutes)

#### Option 1: Restart Service
```bash
# Graceful restart with connection draining
systemctl restart inids-api

# Verify recovery
./scripts/health_check.sh

# Monitor for 5 minutes
watch -n 5 curl http://localhost:5000/api/health
```

#### Option 2: Failover
```bash
# Switch to standby instance
curl -X POST http://backup:5000/api/admin/activate-primary

# Verify new primary is active
curl http://backup:5000/api/health

# Monitor for data sync
watch curl http://backup:5000/api/admin/replication-status
```

#### Option 3: Rollback
```bash
# Identify deployment causing issue
kubectl get deployment -o wide

# Rollback to previous version
kubectl rollout undo deployment/inids-api

# Verify rollback
kubectl get rs -o wide
./scripts/health_check.sh
```

#### Option 4: Scale Down & Up
```bash
# Remove affected instance
kubectl delete pod inids-api-pod-name

# Kubernetes auto-restarts pod
kubectl get pods

# Monitor for recovery
./scripts/health_check.sh
```

### Step 5: Stabilization (60-120 minutes)

**Stabilization Checklist**:
```
□ Service is responding to requests
□ Error rate < 1%
□ Latency returned to baseline
□ No new alerts firing
□ Database integrity verified
□ Cache consistency verified
□ All replicas healthy
□ Backups running successfully
```

**Validation Commands**:
```bash
# API health
curl -s http://localhost:5000/api/health | jq .

# Error rate
curl -s http://localhost:5000/api/admin/metrics | grep error_rate

# Database integrity
psql -c "PRAGMA integrity_check;"

# Cache sync
redis-cli INFO replication
```

### Step 6: Post-Incident Review (24-48 hours later)

See [Post-Incident Review](#post-incident-review) section.

---

## Escalation Matrix

### Initial Response

**On-Call Primary**:
- Primary: inids-primary-oncall@company.com
- Backup: inids-backup-oncall@company.com

**Response Time**: 5 minutes
**Authority**: Incident commander until escalation

### Level 1: Technical Escalation (30 minutes no progress)

**Escalate to**:
- Engineering Lead: engineering-lead@company.com
- Database DBA: database-dba@company.com
- Infrastructure Team: infrastructure@company.com

**Action**:
- Call war room: +1-555-0123
- Invite stakeholders
- Update severity if needed

### Level 2: Management Escalation (60 minutes for SEV-1/2)

**Escalate to**:
- VP Engineering: vp-engineering@company.com
- Product Manager: product@company.com
- Customer Success: cs@company.com

**Action**:
- Prepare customer communication
- Consider compensation/credit
- Set expectations for resolution

### Level 3: Executive Escalation (continued impact >2 hours)

**Escalate to**:
- CTO: cto@company.com
- CEO: ceo@company.com
- Chief Legal Officer: legal@company.com

**Action**:
- Executive war room
- Customer notification strategy
- Board communication plan

---

## Communication Plan

### Internal Communication

**During Incident**:
- **#incidents Slack channel**: Status updates every 15 minutes
- **War room**: Real-time technical discussion
- **Email**: Executive updates (SEV-1 only)

**Message Template**:
```
🚨 Incident Update - INC-20260416-001

Time: 2026-04-16T10:45:00Z
Status: [Investigating|Mitigating|Recovered]
Severity: SEV-2

Details:
- Affected Service: API Server
- Users Impacted: ~5% of customer base
- Root Cause: [Identified|Under Investigation|Unknown]
- Current Action: [Action taken]
- ETA to Recovery: [Time estimate]

Previous Updates:
- 10:30 - Initial alert
- 10:35 - Issue confirmed
- 10:40 - Root cause identified
```

### External Communication

**Decision Matrix**:

| Severity | Customer Impact | Communication |
|----------|---|---|
| SEV-1 | >25% | Immediate public status page update |
| SEV-2 | >5% | Within 15 minutes to status page |
| SEV-3 | <5% | Within 1 hour to status page |
| SEV-4 | None | Post-incident only (optional) |

**Customer Notification Template**:
```
Subject: [URGENT] Partial Service Disruption - INIDS Platform

Dear Customers,

We are currently experiencing a partial service disruption affecting approximately 10% of our users.

Current Status: Investigating root cause
Time Started: 2026-04-16T10:30:00Z
Expected Update: 2026-04-16T11:00:00Z

Workaround:
[Provide workaround if available]

Impact:
- Alert creation: AFFECTED
- Alert viewing: AVAILABLE
- API access: AFFECTED

We apologize for the inconvenience and are actively working to restore full service.

Updates: https://status.inids.io
```

**Post-Recovery Communication**:
```
Subject: ✅ Service Restored - INIDS Platform

Dear Customers,

Full service has been restored.

Issue Summary:
- Duration: 30 minutes
- Impact: 10% of customer base
- Root Cause: Database connection pool exhaustion

Resolution:
- Increased connection pool limit
- Deployed improved connection handling

Compensation:
- All affected customers receive 1-day credit

We appreciate your patience.
```

---

## Post-Incident Review

### Timing
- **SEV-1**: Within 24 hours of recovery
- **SEV-2**: Within 48 hours of recovery
- **SEV-3**: Within 1 week of recovery
- **SEV-4**: Optional

### PIR Attendees
- Incident Commander
- Engineers involved in response
- Product Manager
- Customer Success Manager (if customer-impacting)
- CTO (if SEV-1)

### PIR Agenda

**1. Incident Timeline (10 minutes)**
```
10:30 UTC - Alert fired
10:35 UTC - On-call engineer acknowledged
10:40 UTC - Root cause identified (connection pool exhaustion)
10:45 UTC - Mitigation started (increase pool from 50 to 100)
11:00 UTC - Service recovered
```

**2. Root Cause Analysis (15 minutes)**
```
Primary Cause:
- New API endpoint added without connection pool tuning
- Load testing didn't simulate peak concurrency
- Monitoring didn't alert on connection pool approaching limit

Contributing Factors:
- No pre-deployment review of resource requirements
- Connection pool settings not documented
- Capacity planning gap identified
```

**3. Impact Assessment (10 minutes)**
```
- Duration: 30 minutes
- Users Affected: 12,000 (10% of customer base)
- Requests Failed: 45,000
- Customer Complaints: 23
- Revenue Impact: $2,500 estimated
- Reputation Impact: Medium (public incident)
```

**4. Response Evaluation (10 minutes)**
```
What went well:
✅ Alert fired within 30 seconds of threshold breach
✅ On-call response time: 2 minutes (target: 5 minutes)
✅ Root cause identified quickly (10 minutes)
✅ Recovery executed cleanly

What could improve:
⚠️ Alerting threshold could be lower (80% vs 100%)
⚠️ Documentation of connection pool settings lacking
⚠️ No pre-deployment connection pool review process
⚠️ Load testing didn't include peak concurrency scenario
```

**5. Remediation Action Items (15 minutes)**

| Action | Owner | Priority | Deadline |
|--------|-------|----------|----------|
| Implement pre-deployment resource review | DevOps Lead | P0 | 1 week |
| Add connection pool usage to dashboard | Monitoring Team | P1 | 2 weeks |
| Update load testing to include peak scenarios | QA Lead | P1 | 2 weeks |
| Document connection pool tuning guidelines | Tech Lead | P2 | 1 month |
| Implement connection pool auto-scaling | Backend Lead | P2 | 1 quarter |

**6. Follow-up Actions**
```
□ Write root cause summary email
□ Share PIR recording with team
□ Update runbook with lessons learned
□ Implement action items per timeline
□ Schedule follow-up in 1 week
```

### PIR Template

**File**: `docs/incident-pir-template.md`

```markdown
# Post-Incident Review - INC-20260416-001

**Incident**: API Server Outage
**Date**: April 16, 2026
**Duration**: 30 minutes (10:30 UTC - 11:00 UTC)
**Severity**: SEV-2

## Incident Timeline
[Detailed timeline with timestamps]

## Root Cause
[Primary and contributing causes]

## Impact Summary
- Users Affected: 12,000
- Requests Failed: 45,000
- Revenue Impact: $2,500
- Recovery Time: 30 minutes

## What We Learned
1. [Learning point 1]
2. [Learning point 2]
3. [Learning point 3]

## Action Items
| Item | Owner | Deadline | Status |
|------|-------|----------|--------|
| [Action] | [Person] | [Date] | Not Started |

## Next Steps
- [ ] Implement action items
- [ ] Follow-up PIR in 1 week
- [ ] Monitor related metrics
```

---

## Recovery Procedures

### Database Recovery

**Steps**:
1. Stop affected service
2. Restore from latest backup
3. Apply transaction logs to recover to point-of-failure
4. Verify data integrity
5. Restart service
6. Monitor for 30 minutes

**Verify Commands**:
```bash
# Check backup integrity
tar -tzf /backups/db-20260416.tar.gz

# Restore
tar -xzf /backups/db-20260416.tar.gz -C /tmp/
psql -d inids < /tmp/db-backup.sql

# Verify
psql -c "SELECT COUNT(*) FROM alerts"
psql -c "SELECT MAX(created_at) FROM alerts"
```

### Cache Recovery

**Steps**:
1. Flush Redis cache
2. Warm cache with frequent keys
3. Monitor cache hit rate recovery

```bash
# Flush
redis-cli FLUSHALL

# Warm cache
python scripts/warm-cache.py

# Verify
redis-cli INFO stats
```

### Service Recovery

**Steps**:
1. Restart service
2. Wait for readiness probe to pass
3. Health check passes
4. Monitor error rates drop
5. Alert team when recovered

```bash
# Restart
systemctl restart inids-api

# Monitor
journalctl -u inids-api -f

# Health
curl http://localhost:5000/api/health
```

---

## On-Call Escalation

### On-Call Schedule
- Primary: Week 1-2
- Backup: Week 3-4
- Rotation period: 4 weeks

### Handoff Procedure
- Friday 5pm: Outgoing brief incoming
- Review incidents from past 4 weeks
- Document any known issues
- Confirm contact information

---

## Training & Drills

### Quarterly Incident Drills

**Drill 1: Simulated Outage**
- Randomly disable service
- Team follows response procedures
- Measure response time and recovery
- Debrief and lessons learned

**Drill 2: Simulated Security Incident**
- Simulate unauthorized access
- Execute security response procedures
- Test communication plan
- Review incident response effectiveness

### Annual Incident Response Training
- Review OWASP incident response guidelines
- Practice technical troubleshooting
- Communication skills workshop
- Tabletop exercises

---

**Playbook Version**: 1.0  
**Last Updated**: April 16, 2026  
**Next Review**: April 16, 2027
