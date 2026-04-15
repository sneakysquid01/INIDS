"""
Three-Layer Alert Filtering Engine
Implements sophisticated alert filtering with three layers:

1. EXCLUDE Layer: Completely block alerts matching patterns (no alerts generated)
2. IGNORE Layer: Deprioritize alerts without blocking (reduce severity, suppress notifications)
3. MERGE Layer: Combine similar alerts from same source (deduplicate, group)

This prevents alert fatigue and focuses on high-value threats.
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import re

logger = logging.getLogger(__name__)


class FilterAction(Enum):
    """Filter action result."""
    EXCLUDE = "exclude"  # Block alert completely
    IGNORE = "ignore"  # Deprioritize alert
    MERGE = "merge"  # Combine with recent similar alert
    PASS = "pass"  # Allow alert through


@dataclass
class FilterRule:
    """Base filter rule."""
    rule_id: str
    name: str
    description: str = ""
    enabled: bool = True
    conditions: dict[str, Any] = field(default_factory=dict)  # Key-value conditions to match
    action: FilterAction = FilterAction.PASS
    priority: int = 0  # Higher = evaluated first
    
    def matches(self, alert: dict[str, Any]) -> bool:
        """Check if alert matches all conditions."""
        for key, value in self.conditions.items():
            if key not in alert:
                return False
            
            alert_value = alert[key]
            
            # Support different matching types
            if isinstance(value, str) and value.startswith("regex:"):
                # Regex match
                pattern = value[6:]
                if not isinstance(alert_value, str) or not re.search(pattern, alert_value):
                    return False
            elif isinstance(value, str) and value.startswith("contains:"):
                # Substring match
                substring = value[9:]
                if substring not in str(alert_value):
                    return False
            elif isinstance(value, (list, tuple)):
                # One-of match
                if alert_value not in value:
                    return False
            else:
                # Exact match
                if alert_value != value:
                    return False
        
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "conditions": self.conditions,
            "action": self.action.value,
            "priority": self.priority,
        }


@dataclass
class ExcludeRule(FilterRule):
    """Rule that completely blocks alerts."""
    action: FilterAction = FilterAction.EXCLUDE
    reason: str = "Manually excluded"


@dataclass
class IgnoreRule(FilterRule):
    """Rule that deprioritizes alerts."""
    action: FilterAction = FilterAction.IGNORE
    severity_reduction: int = 1  # Reduce severity by N levels
    suppress_notifications: bool = True


@dataclass
class MergeRule(FilterRule):
    """Rule that merges similar alerts."""
    action: FilterAction = FilterAction.MERGE
    merge_window_seconds: int = 300  # 5 minutes default
    merge_key: str = "source_ip"  # Merge alerts grouped by this field
    similarity_fields: list[str] = field(default_factory=lambda: ["attack_type", "source_ip"])


@dataclass
class AlertFilterResult:
    """Result of filtering an alert."""
    original_alert_id: str
    action: FilterAction
    rule_applied: Optional[FilterRule] = None
    merged_with_alert_id: Optional[str] = None
    modified_severity: Optional[int] = None
    suppressed_notifications: bool = False
    reason: str = ""


class ThreeLayerAlertFilter:
    """Three-layer alert filtering system."""
    
    def __init__(self, ops_store=None):
        """
        Initialize filter engine.
        
        Args:
            ops_store: OpsStore instance for alert lookup and persistence
        """
        self.ops_store = ops_store
        self.exclude_rules: list[ExcludeRule] = []
        self.ignore_rules: list[IgnoreRule] = []
        self.merge_rules: list[MergeRule] = []
        self.recent_alerts: dict[str, dict] = {}  # {alert_id: alert_data}
        self.merge_groups: dict[str, list[str]] = {}  # {merge_key: [alert_ids]}
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Create backing table for persisted filter rules when a store is provided."""
        if not self.ops_store:
            return
        self.ops_store._execute(
            """
            CREATE TABLE IF NOT EXISTS alert_filter_rules (
                rule_id TEXT PRIMARY KEY,
                layer TEXT NOT NULL,
                rule_config TEXT NOT NULL
            )
            """
        )
    
    def filter_alert(self, alert: dict[str, Any]) -> AlertFilterResult:
        """
        Apply three-layer filtering to an alert.
        
        Returns:
            AlertFilterResult with action and modified alert if applicable
        """
        alert_id = alert.get("id", "unknown")
        
        # Layer 1: EXCLUDE - Block alerts completely
        for rule in sorted(self.exclude_rules, key=lambda r: r.priority, reverse=True):
            if not rule.enabled:
                continue
            if rule.matches(alert):
                logger.info(f"Alert {alert_id} excluded by rule {rule.rule_id}: {rule.name}")
                return AlertFilterResult(
                    original_alert_id=alert_id,
                    action=FilterAction.EXCLUDE,
                    rule_applied=rule,
                    reason=f"Excluded by rule: {rule.name}"
                )
        
        # Layer 2: IGNORE - Deprioritize alerts
        for rule in sorted(self.ignore_rules, key=lambda r: r.priority, reverse=True):
            if not rule.enabled:
                continue
            if rule.matches(alert):
                logger.info(f"Alert {alert_id} ignored by rule {rule.rule_id}: {rule.name}")
                return AlertFilterResult(
                    original_alert_id=alert_id,
                    action=FilterAction.IGNORE,
                    rule_applied=rule,
                    modified_severity=max(0, alert.get("severity", 0) - rule.severity_reduction),
                    suppressed_notifications=rule.suppress_notifications,
                    reason=f"Deprioritized by rule: {rule.name}"
                )
        
        # Layer 3: MERGE - Combine similar alerts
        for rule in sorted(self.merge_rules, key=lambda r: r.priority, reverse=True):
            if not rule.enabled:
                continue
            if rule.matches(alert):
                similar_alert = self._find_similar_alert(alert, rule)
                if similar_alert:
                    logger.info(f"Alert {alert_id} merged with {similar_alert['id']} by rule {rule.rule_id}")
                    return AlertFilterResult(
                        original_alert_id=alert_id,
                        action=FilterAction.MERGE,
                        rule_applied=rule,
                        merged_with_alert_id=similar_alert["id"],
                        reason=f"Merged with similar alert by rule: {rule.name}"
                    )
        
        # No filtering applied
        return AlertFilterResult(
            original_alert_id=alert_id,
            action=FilterAction.PASS,
            reason="Alert passed all filters"
        )
    
    def _find_similar_alert(self, alert: dict[str, Any], rule: MergeRule) -> Optional[dict]:
        """Find a recent similar alert to merge with."""
        current_time = datetime.now().astimezone()
        merge_window = timedelta(seconds=rule.merge_window_seconds)
        
        # Check recent alerts
        for recent_id, recent_alert in list(self.recent_alerts.items()):
            alert_time = datetime.fromisoformat(
                recent_alert.get("timestamp", datetime.now().astimezone().isoformat())
            )
            if alert_time.tzinfo is None:
                alert_time = alert_time.replace(tzinfo=current_time.tzinfo)
            
            # Check if within merge window
            if current_time - alert_time > merge_window:
                del self.recent_alerts[recent_id]
                continue
            
            # Check if similarity fields match
            matches_fields = all(
                alert.get(field) == recent_alert.get(field)
                for field in rule.similarity_fields
            )
            
            if matches_fields:
                return recent_alert
        
        return None
    
    def track_alert(self, alert: dict[str, Any]) -> None:
        """Track alert for future merge operations."""
        alert_id = alert.get("id", "unknown")
        self.recent_alerts[alert_id] = alert
        
        # Cleanup old alerts
        current_time = datetime.now().astimezone()
        for alert_id_to_check in list(self.recent_alerts.keys()):
            alert_data = self.recent_alerts[alert_id_to_check]
            alert_time = datetime.fromisoformat(
                alert_data.get("timestamp", datetime.now().astimezone().isoformat())
            )
            if alert_time.tzinfo is None:
                alert_time = alert_time.replace(tzinfo=current_time.tzinfo)
            if current_time - alert_time > timedelta(hours=1):
                del self.recent_alerts[alert_id_to_check]
    
    # ============ Rule Management ============
    
    def add_exclude_rule(self, rule: ExcludeRule) -> bool:
        """Add an exclude rule."""
        if any(r.rule_id == rule.rule_id for r in self.exclude_rules):
            logger.warning(f"Exclude rule {rule.rule_id} already exists")
            return False
        self.exclude_rules.append(rule)
        self._persist_rule("exclude", rule)
        logger.info(f"Added exclude rule: {rule.rule_id}")
        return True
    
    def add_ignore_rule(self, rule: IgnoreRule) -> bool:
        """Add an ignore rule."""
        if any(r.rule_id == rule.rule_id for r in self.ignore_rules):
            logger.warning(f"Ignore rule {rule.rule_id} already exists")
            return False
        self.ignore_rules.append(rule)
        self._persist_rule("ignore", rule)
        logger.info(f"Added ignore rule: {rule.rule_id}")
        return True
    
    def add_merge_rule(self, rule: MergeRule) -> bool:
        """Add a merge rule."""
        if any(r.rule_id == rule.rule_id for r in self.merge_rules):
            logger.warning(f"Merge rule {rule.rule_id} already exists")
            return False
        self.merge_rules.append(rule)
        self._persist_rule("merge", rule)
        logger.info(f"Added merge rule: {rule.rule_id}")
        return True
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        for rules_list in [self.exclude_rules, self.ignore_rules, self.merge_rules]:
            original_len = len(rules_list)
            rules_list[:] = [r for r in rules_list if r.rule_id != rule_id]
            if len(rules_list) < original_len:
                logger.info(f"Removed rule: {rule_id}")
                return True
        return False
    
    def update_rule(self, rule_id: str, updates: dict) -> bool:
        """Update rule properties."""
        for rules_list in [self.exclude_rules, self.ignore_rules, self.merge_rules]:
            for rule in rules_list:
                if rule.rule_id == rule_id:
                    for key, value in updates.items():
                        if hasattr(rule, key):
                            setattr(rule, key, value)
                    self._persist_rule(None, rule)
                    logger.info(f"Updated rule: {rule_id}")
                    return True
        return False
    
    def get_all_rules(self) -> dict:
        """Get all rules organized by layer."""
        return {
            "exclude": [r.to_dict() for r in self.exclude_rules],
            "ignore": [r.to_dict() for r in self.ignore_rules],
            "merge": [r.to_dict() for r in self.merge_rules],
        }
    
    def get_rule_stats(self) -> dict:
        """Get statistics on rules and filtering."""
        return {
            "exclude_rules_count": len(self.exclude_rules),
            "ignore_rules_count": len(self.ignore_rules),
            "merge_rules_count": len(self.merge_rules),
            "recent_alerts_tracked": len(self.recent_alerts),
            "merge_groups_active": len(self.merge_groups),
        }
    
    def _persist_rule(self, layer: Optional[str], rule: FilterRule) -> None:
        """Persist rule to storage."""
        if self.ops_store:
            try:
                rule_data = {
                    "rule_id": rule.rule_id,
                    "layer": layer,
                    "rule_config": json.dumps(rule.to_dict()),
                }
                self.ops_store._execute(
                    """
                    INSERT OR REPLACE INTO alert_filter_rules (rule_id, layer, rule_config)
                    VALUES (:rule_id, :layer, :rule_config)
                    """,
                    rule_data,
                )
            except Exception as e:
                logger.warning(f"Failed to persist rule {rule.rule_id}: {e}")
    
    def load_rules_from_storage(self) -> None:
        """Load persisted rules from storage."""
        if not self.ops_store:
            return
        
        try:
            rows = self.ops_store._fetchall("SELECT rule_id, layer, rule_config FROM alert_filter_rules")
            for row in rows:
                try:
                    config = json.loads(row.get("rule_config", "{}"))
                    layer = row.get("layer")
                    
                    if layer == "exclude":
                        rule = ExcludeRule(**config)
                        self.exclude_rules.append(rule)
                    elif layer == "ignore":
                        rule = IgnoreRule(**config)
                        self.ignore_rules.append(rule)
                    elif layer == "merge":
                        rule = MergeRule(**config)
                        self.merge_rules.append(rule)
                except Exception as e:
                    logger.warning(f"Failed to load rule {row.get('rule_id')}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load rules from storage: {e}")


def create_default_rules() -> tuple[list[ExcludeRule], list[IgnoreRule], list[MergeRule]]:
    """Create default recommended rules."""
    
    exclude_rules = [
        ExcludeRule(
            rule_id="exclude_localhost",
            name="Exclude localhost attacks",
            description="Ignore alerts from localhost (127.0.0.1)",
            conditions={"source_ip": "127.0.0.1"},
            priority=100,
        ),
        ExcludeRule(
            rule_id="exclude_gateway",
            name="Exclude gateway scanning",
            description="Ignore basic network scanning from internal gateways",
            conditions={"attack_type": "network_scan", "source_ip": "regex:^192\\.168\\.1\\."},
            priority=50,
        ),
    ]
    
    ignore_rules = [
        IgnoreRule(
            rule_id="ignore_low_confidence",
            name="Deprioritize low-confidence alerts",
            description="Reduce severity for alerts with < 0.5 confidence",
            conditions={"confidence": "< 0.5"},  # Custom logic in matches()
            severity_reduction=2,
            suppress_notifications=True,
            priority=90,
        ),
        IgnoreRule(
            rule_id="ignore_internal_scans",
            name="Deprioritize internal network scans",
            description="Reduce severity for internal network reconnaissance",
            conditions={"attack_type": "network_scan", "source_ip": "regex:^10\\."},
            severity_reduction=1,
            suppress_notifications=False,
            priority=70,
        ),
    ]
    
    merge_rules = [
        MergeRule(
            rule_id="merge_consecutive_brute_force",
            name="Merge consecutive brute force attempts",
            description="Group brute force attacks from same IP",
            conditions={"attack_type": "brute_force"},
            merge_window_seconds=600,
            merge_key="source_ip",
            similarity_fields=["attack_type", "source_ip"],
            priority=80,
        ),
        MergeRule(
            rule_id="merge_port_scan_attempts",
            name="Merge port scan attempts",
            description="Group port scans within 5-minute window",
            conditions={"attack_type": "network_scan"},
            merge_window_seconds=300,
            merge_key="source_ip",
            similarity_fields=["attack_type", "source_ip", "destination"],
            priority=70,
        ),
    ]
    
    return exclude_rules, ignore_rules, merge_rules
