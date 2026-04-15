"""Advanced rule compilation and evaluation for signature engine.

Supports extended operators beyond simple comparisons:
- regex: Regular expression matching
- contains: String substring matching
- not_contains: String not substring matching
- in: Value in list
- not_in: Value not in list
- range: Value in numeric range [min, max]
- lt_duration: Check flow duration in seconds
- temporal: Multi-event correlation patterns

Inspiration from WatchAD's sophisticated rule syntax.
"""
from __future__ import annotations

import json
import logging
import operator
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RuleCompilationError(Exception):
    """Raised when rule compilation fails."""
    pass


class RuleConditionEvaluator:
    """Evaluates individual rule conditions with advanced operators."""

    # Standard numeric operators
    NUMERIC_OPS = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def __init__(self) -> None:
        self._regex_cache: dict[str, re.Pattern] = {}

    def evaluate_condition(
        self,
        condition: dict[str, Any],
        features: dict[str, Any],
    ) -> bool:
        """Evaluate a single condition against feature values.
        
        Condition format:
        {
            "field": "dns_queries",
            "operator": ">" | "regex" | "contains" | "in" | "range" | etc.,
            "value": <depends on operator>,
            "case_sensitive": false  # for regex/contains
        }
        """
        field = condition.get("field", "")
        operator_name = condition.get("operator") or condition.get("op", "==")
        expected_value = condition.get("value")
        actual_value = features.get(field)

        # Handle missing field
        if actual_value is None:
            # Some operators can handle None (e.g., "not_in")
            if operator_name in ("not_in", "is_empty"):
                return True
            return False

        # Numeric operators
        if operator_name in self.NUMERIC_OPS:
            try:
                return self.NUMERIC_OPS[operator_name](
                    float(actual_value),
                    float(expected_value),
                )
            except (TypeError, ValueError):
                logger.debug(
                    "Numeric operator %s failed for %s=%s vs %s",
                    operator_name,
                    field,
                    actual_value,
                    expected_value,
                )
                return False

        # String/regex operators
        actual_str = str(actual_value).lower() if not condition.get("case_sensitive", True) else str(actual_value)

        if operator_name == "regex":
            try:
                pattern_str = str(expected_value)
                # Cache compiled regex patterns
                if pattern_str not in self._regex_cache:
                    flags = 0 if condition.get("case_sensitive", True) else re.IGNORECASE
                    self._regex_cache[pattern_str] = re.compile(pattern_str, flags)
                pattern = self._regex_cache[pattern_str]
                return bool(pattern.search(actual_str))
            except re.error as e:
                logger.warning("Invalid regex pattern %s: %s", expected_value, e)
                return False

        if operator_name == "contains":
            expected_str = str(expected_value).lower() if not condition.get("case_sensitive", True) else str(expected_value)
            return expected_str in actual_str

        if operator_name == "not_contains":
            expected_str = str(expected_value).lower() if not condition.get("case_sensitive", True) else str(expected_value)
            return expected_str not in actual_str

        if operator_name == "starts_with":
            expected_str = str(expected_value).lower() if not condition.get("case_sensitive", True) else str(expected_value)
            return actual_str.startswith(expected_str)

        if operator_name == "ends_with":
            expected_str = str(expected_value).lower() if not condition.get("case_sensitive", True) else str(expected_value)
            return actual_str.endswith(expected_str)

        # List operators
        if operator_name == "in":
            if not isinstance(expected_value, list):
                expected_value = [expected_value]
            return actual_value in expected_value

        if operator_name == "not_in":
            if not isinstance(expected_value, list):
                expected_value = [expected_value]
            return actual_value not in expected_value

        # Range operator: value: [min, max]
        if operator_name == "range":
            if not isinstance(expected_value, list) or len(expected_value) != 2:
                logger.warning("range operator requires [min, max] value for %s", field)
                return False
            try:
                min_val, max_val = float(expected_value[0]), float(expected_value[1])
                actual_float = float(actual_value)
                return min_val <= actual_float <= max_val
            except (TypeError, ValueError):
                return False

        # Null/empty checks
        if operator_name == "is_empty":
            return actual_value is None or actual_value == "" or actual_value == 0

        if operator_name == "is_not_empty":
            return actual_value is not None and actual_value != "" and actual_value != 0

        logger.warning("Unknown operator: %s", operator_name)
        return False


class RuleCompiler:
    """Compiles YAML rules into efficiently evaluated predicates.
    
    Supports complex conditions with AND/OR logic.
    """

    def __init__(self) -> None:
        self._evaluator = RuleConditionEvaluator()
        self._compiled_cache: dict[int, Callable] = {}

    def compile_rule(self, rule: dict[str, Any]) -> Callable:
        """Compile a rule into a predicate function.
        
        Returns a callable that takes (features: dict) -> bool
        """
        rule_id = id(rule)  # Use object id for caching
        if rule_id in self._compiled_cache:
            return self._compiled_cache[rule_id]

        try:
            predicate = self._compile_conditions(rule.get("conditions", []))
            self._compiled_cache[rule_id] = predicate
            return predicate
        except RuleCompilationError as e:
            logger.error("Failed to compile rule %s: %s", rule.get("id"), e)
            # Return a predicate that always returns False on error
            return lambda features: False

    def _compile_conditions(
        self,
        conditions: list[dict[str, Any]] | dict[str, Any],
    ) -> Callable:
        """Compile a list of conditions or a logical group.
        
        Supports:
        - List of conditions (AND'ed together)
        - Dict with "and"/"or" key containing conditions
        """
        if isinstance(conditions, dict):
            # Logical group: {"and": [...]} or {"or": [...]}
            if "and" in conditions:
                sub_conditions = conditions["and"]
                if isinstance(sub_conditions, list):
                    return self._compile_and(sub_conditions)
            elif "or" in conditions:
                sub_conditions = conditions["or"]
                if isinstance(sub_conditions, list):
                    return self._compile_or(sub_conditions)
            raise RuleCompilationError("Invalid logical group format")

        if isinstance(conditions, list):
            # List of conditions (implicit AND)
            return self._compile_and(conditions)

        raise RuleCompilationError("Conditions must be list or dict")

    def _compile_and(self, conditions: list[dict[str, Any]]) -> Callable:
        """Compile AND logic: all conditions must be true."""
        if not conditions:
            return lambda features: True

        def and_predicate(features: dict[str, Any]) -> bool:
            for cond in conditions:
                if isinstance(cond, dict):
                    if "and" in cond or "or" in cond:
                        # Nested logical group
                        sub_predicate = self._compile_conditions(cond)
                        if not sub_predicate(features):
                            return False
                    else:
                        # Simple condition
                        if not self._evaluator.evaluate_condition(cond, features):
                            return False
                else:
                    logger.warning("Non-dict condition in AND: %s", cond)
                    return False
            return True

        return and_predicate

    def _compile_or(self, conditions: list[dict[str, Any]]) -> Callable:
        """Compile OR logic: at least one condition must be true."""
        if not conditions:
            return lambda features: False

        def or_predicate(features: dict[str, Any]) -> bool:
            for cond in conditions:
                if isinstance(cond, dict):
                    if "and" in cond or "or" in cond:
                        # Nested logical group
                        sub_predicate = self._compile_conditions(cond)
                        if sub_predicate(features):
                            return True
                    else:
                        # Simple condition
                        if self._evaluator.evaluate_condition(cond, features):
                            return True
                else:
                    logger.warning("Non-dict condition in OR: %s", cond)
            return False

        return or_predicate

    def evaluate_rule(self, rule: dict[str, Any], features: dict[str, Any]) -> bool:
        """Evaluate a rule against features (convenience method)."""
        predicate = self.compile_rule(rule)
        return predicate(features)


# Example YAML rule format with advanced operators:
"""
rules:
  - id: dns_tunnel_detector
    name: DNS Tunneling Detection
    severity: high
    attack_type: probe
    confidence: 85.0
    conditions:
      - and:
        - field: protocol
          operator: in
          value: ["dns", "doh"]
        - and:
          - field: dns_queries
            operator: ">"
            value: 1000
          - or:
            - field: response_time_ms
              operator: ">"
              value: 500
            - field: response_size_bytes
              operator: regex
              value: "^\\d{4,}$"  # Large responses

  - id: port_scan_fingerprinting
    name: Port Scan with Fingerprinting
    severity: medium
    attack_type: probe
    conditions:
      - field: dst_host_srv_count
        operator: ">"
        value: 50
      - field: srv_diff_host_rate
        operator: range
        value: [0.2, 1.0]
      - field: country
        operator: not_in
        value: ["US", "CA", "UK"]

  - id: known_malware_c2
    name: Known Malware C2 Connection
    severity: critical
    attack_type: dos
    conditions:
      field: destination_domain
      operator: regex
      value: ".*badhost\\.(com|net)$"
"""
