"""
Configuration Schema Validation

Validates:
- Detection rules format
- API request/response schemas
- Configuration files
"""

import json
import logging
from typing import Tuple, Optional, Dict, Any
from jsonschema import Draft7Validator, ValidationError

logger = logging.getLogger(__name__)


# JSON Schema for detection rules
RULE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["name", "pattern", "severity"],
    "properties": {
        "id": {
            "type": "string",
            "description": "Rule ID (e.g., SIG-001)"
        },
        "name": {
            "type": "string",
            "minLength": 3,
            "maxLength": 256,
            "pattern": "^[a-zA-Z0-9_\\-\\s]+$",
            "description": "Rule name"
        },
        "description": {
            "type": "string",
            "maxLength": 1024,
            "description": "Detailed description"
        },
        "pattern": {
            "type": "string",
            "minLength": 1,
            "maxLength": 4096,
            "description": "Regex pattern or rule logic"
        },
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "Alert severity level"
        },
        "enabled": {
            "type": "boolean",
            "default": True,
            "description": "Whether rule is active"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Base confidence score (0-1)"
        },
        "timeout_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 3600,
            "description": "Rule timeout"
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tags for categorization"
        },
        "correlation": {
            "type": "object",
            "properties": {
                "window_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 86400
                },
                "threshold": {
                    "type": "integer",
                    "minimum": 1
                },
                "method": {
                    "type": "string",
                    "enum": ["count", "rate", "pattern"]
                }
            },
            "description": "Correlation settings"
        },
        "active_response": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "action": {"type": "string", "enum": ["block", "rate_limit", "alert"]},
                "ttl_seconds": {"type": "integer", "minimum": 1}
            },
            "description": "Active response configuration"
        }
    }
}

# JSON Schema for API requests
PREDICT_REQUEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["features"],
    "properties": {
        "features": {
            "type": "object",
            "description": "Feature vector for prediction"
        },
        "profile": {
            "type": "string",
            "enum": ["strict", "balanced", "lenient"],
            "default": "balanced",
            "description": "Detection profile"
        },
        "source_ip": {
            "type": "string",
            "description": "Source IP address"
        },
        "source_port": {
            "type": "integer",
            "minimum": 1,
            "maximum": 65535
        },
        "attack_type": {
            "type": ["string", "null"],
            "description": "Known attack type"
        }
    }
}

DETECT_REQUEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["features"],
    "properties": {
        "features": {
            "type": "object",
            "description": "Feature vector for detection"
        }
    }
}

POLICY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["monitor", "auto_block"],
            "default": "monitor"
        },
        "alert_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "default": 0.4
        },
        "rate_limit_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "default": 0.6
        },
        "temp_block_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "default": 0.75
        },
        "block_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "default": 0.85
        },
        "block_ttl_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 86400,
            "default": 300
        },
        "dry_run": {
            "type": "boolean",
            "default": False
        }
    }
}


class ConfigValidator:
    """Validate configurations against schemas."""
    
    def __init__(self):
        self.validators = {
            'rule': Draft7Validator(RULE_SCHEMA),
            'predict_request': Draft7Validator(PREDICT_REQUEST_SCHEMA),
            'detect_request': Draft7Validator(DETECT_REQUEST_SCHEMA),
            'policy': Draft7Validator(POLICY_SCHEMA),
        }
    
    def validate_rule(self, rule: Dict[str, Any]) -> Tuple[bool, Optional[str], list]:
        """Validate detection rule.
        
        Returns:
            (is_valid, error_message, error_details)
        """
        try:
            validator = self.validators['rule']
            errors = list(validator.iter_errors(rule))
            
            if errors:
                error_details = [
                    {
                        'path': list(e.path),
                        'message': e.message,
                        'schema_path': list(e.schema_path)
                    }
                    for e in errors
                ]
                return False, f"Rule validation failed: {errors[0].message}", error_details
            
            return True, None, []
        
        except Exception as e:
            logger.error(f"Rule validation error: {str(e)}")
            return False, f"Validation error: {str(e)}", []
    
    def validate_predict_request(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate predict request."""
        try:
            validator = self.validators['predict_request']
            errors = list(validator.iter_errors(data))
            
            if errors:
                return False, errors[0].message
            
            return True, None
        
        except Exception as e:
            return False, str(e)
    
    def validate_detect_request(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate detect request."""
        try:
            validator = self.validators['detect_request']
            errors = list(validator.iter_errors(data))
            
            if errors:
                return False, errors[0].message
            
            return True, None
        
        except Exception as e:
            return False, str(e)
    
    def validate_policy(self, policy: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate policy."""
        try:
            validator = self.validators['policy']
            errors = list(validator.iter_errors(policy))
            
            if errors:
                return False, errors[0].message
            
            return True, None
        
        except Exception as e:
            return False, str(e)


# Global validator instance
_validator = ConfigValidator()


def get_validator() -> ConfigValidator:
    """Get global validator instance."""
    return _validator


def validate_rule(rule: Dict[str, Any]) -> Tuple[bool, Optional[str], list]:
    """Validate a rule."""
    return _validator.validate_rule(rule)


def validate_predict_request(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate predict request."""
    return _validator.validate_predict_request(data)


def validate_detect_request(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate detect request."""
    return _validator.validate_detect_request(data)


def validate_policy(policy: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate policy."""
    return _validator.validate_policy(policy)
