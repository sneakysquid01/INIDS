"""Signature-based Detection Engine — matches flows against YAML rule definitions.

Supports advanced operators: regex, contains, in, range, and logical combinations (AND/OR).
"""
from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Any

import yaml

from src.detection.engine_base import DetectionEngine, EngineResult
from src.detection.rule_compiler import RuleCompiler

logger = logging.getLogger(__name__)

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


class SignatureEngine(DetectionEngine):
    """Evaluate traffic features against a set of YAML-defined signature rules.

    Supports advanced operators: regex, contains, in, range, AND/OR logic.
    
    Each rule has the form::

        - id: SIG-001
          name: SYN Flood Indicator
          severity: high
          attack_type: dos
          conditions:
            - and:
              - field: serror_rate
                operator: ">"
                value: 0.8
              - field: count
                operator: ">"
                value: 200
    """

    def __init__(self, rules_path: str | Path | None = None, *, engine_id: str = "signature", fp_manager=None) -> None:
        self._engine_id = engine_id
        self._rules: list[dict[str, Any]] = []
        self._fp_manager = fp_manager
        self._rule_compiler = RuleCompiler()
        if rules_path is not None:
            self.load_rules(rules_path)

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def load_rules(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            logger.warning("Signature rules file not found: %s", path)
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            self._rules = data.get("rules", [])
        elif isinstance(data, list):
            self._rules = data
        else:
            self._rules = []
        logger.info("Loaded %d signature rules from %s", len(self._rules), path)

    def add_rule(self, rule: dict[str, Any]) -> None:
        self._rules.append(rule)

    # ------------------------------------------------------------------
    # DetectionEngine interface
    # ------------------------------------------------------------------

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def engine_type(self) -> str:
        return "signature"

    def is_ready(self) -> bool:
        return len(self._rules) > 0

    def evaluate(self, features: dict[str, Any]) -> EngineResult:
        matched_rule: dict[str, Any] | None = None
        for rule in self._rules:
            if self._match_rule(rule, features):
                rule_id = rule.get("id", "unknown")
                # Skip rules that have been suppressed by analyst FP feedback.
                if self._fp_manager is not None and self._fp_manager.is_suppressed(self._engine_id, rule_id):
                    logger.debug("Suppressed signature rule %s skipped", rule_id)
                    continue
                # First non-suppressed matching rule wins (rules ordered by priority in YAML).
                matched_rule = rule
                break

        if matched_rule is None:
            return EngineResult(
                engine_id=self._engine_id,
                engine_type=self.engine_type,
                verdict="normal",
                confidence=100.0,
                severity="low",
                attack_type="normal",
            )

        return EngineResult(
            engine_id=self._engine_id,
            engine_type=self.engine_type,
            verdict="attack",
            confidence=float(matched_rule.get("confidence", 95.0)),
            severity=matched_rule.get("severity", "high"),
            attack_type=matched_rule.get("attack_type", "unknown"),
            rule_id=matched_rule.get("id", "unknown"),
            metadata={"rule_name": matched_rule.get("name", "")},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_rule(self, rule: dict[str, Any], features: dict[str, Any]) -> bool:
        """Match rule using advanced rule compiler with backward compatibility."""
        conditions = rule.get("conditions", [])
        if not conditions:
            return False
        
        # Try to use the advanced rule compiler first
        try:
            predicate = self._rule_compiler.compile_rule(rule)
            return predicate(features)
        except Exception as e:
            logger.debug("Advanced rule compilation failed for %s, falling back to legacy: %s", 
                        rule.get("id"), e)
            # Fall back to legacy _match_rule_legacy
            return self._match_rule_legacy(rule, features)

    @staticmethod
    def _match_rule_legacy(rule: dict[str, Any], features: dict[str, Any]) -> bool:
        """Legacy rule matching (backward compatibility)."""
        conditions = rule.get("conditions", [])
        if not conditions:
            return False
        for cond in conditions:
            field = cond.get("field", "")
            op_str = cond.get("op", "==")
            expected = cond.get("value")
            actual = features.get(field)
            if actual is None:
                return False
            cmp = _OPS.get(op_str)
            if cmp is None:
                logger.warning("Unknown operator '%s' in rule %s", op_str, rule.get("id"))
                return False
            try:
                if not cmp(float(actual), float(expected)):
                    return False
            except (TypeError, ValueError):
                if not cmp(str(actual), str(expected)):
                    return False
        return True
