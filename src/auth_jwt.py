"""
JWT Authentication & Run-As Impersonation

Provides:
- JWT token generation with ECC signing
- Token validation and claims extraction
- Run-as impersonation for admin workflows
- Complete audit trail for run-as actions
"""

import jwt
import json
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from dataclasses import dataclass
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from functools import wraps

from flask import request, jsonify

logger = logging.getLogger(__name__)


@dataclass
class JWTClaims:
    """JWT token claims."""
    sub: str  # Subject (username)
    user_id: str
    roles: list
    iat: int  # Issued at
    exp: int  # Expiration
    aud: str  # Audience
    run_as_admin: Optional[str] = None  # If using run-as
    context_hash: Optional[str] = None  # Hash for audit trail


class JWTAuthManager:
    """Manage JWT token generation and validation."""

    @staticmethod
    def _claims_from_payload(payload: dict) -> JWTClaims:
        return JWTClaims(
            sub=payload['sub'],
            user_id=payload['user_id'],
            roles=payload['roles'],
            iat=payload['iat'],
            exp=payload['exp'],
            aud=payload['aud'],
            run_as_admin=payload.get('run_as_admin'),
            context_hash=payload.get('context_hash')
        )

    def __init__(self, secret_key: str = None):
        """Initialize with ECC keypair."""
        self.algorithm = 'ES256'
        self.audience = 'INIDS-API'
        
        # Generate or use provided key
        if secret_key:
            # For simple deployments, allow symmetric key
            self.private_key = secret_key.encode()
            self.public_key = secret_key.encode()
            self.algorithm = 'HS256'
        else:
            # Generate ECC keypair
            self.private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            self.public_key = self.private_key.public_key()
    
    def create_token(self, user_id: str, username: str, roles: list,
                    expires_in: int = 86400,  # 24 hours
                    run_as_admin: str = None,
                    context_hash: str = None) -> str:
        """Create JWT token.
        
        Args:
            user_id: User ID
            username: Username
            roles: List of roles
            expires_in: Token expiration in seconds
            run_as_admin: If set, this is a run-as token (admin user)
            context_hash: Hash of run-as context for audit trail
        
        Returns:
            JWT token string
        """
        now = datetime.now(timezone.utc)
        exp = now + timedelta(seconds=expires_in)
        
        payload = {
            'sub': username,
            'user_id': user_id,
            'roles': roles,
            'iat': int(now.timestamp()),
            'exp': int(exp.timestamp()),
            'aud': self.audience,
        }
        
        if run_as_admin:
            payload['run_as_admin'] = run_as_admin
            payload['context_hash'] = context_hash
        
        if self.algorithm == 'HS256':
            token = jwt.encode(payload, self.private_key, algorithm=self.algorithm)
        else:
            token = jwt.encode(payload, self.private_key, algorithm=self.algorithm)
        
        logger.info(f"JWT token created for user {username}")
        return token
    
    def verify_token(
        self,
        token: str,
        *,
        allow_expired: bool = False,
    ) -> Tuple[bool, Optional[JWTClaims], Optional[str]]:
        """Verify JWT token.
        
        Returns:
            (is_valid, claims, error_message)
        """
        try:
            decode_kwargs = {
                "algorithms": [self.algorithm],
                "audience": self.audience,
            }
            if allow_expired:
                decode_kwargs["options"] = {"verify_exp": False}

            if self.algorithm == 'HS256':
                payload = jwt.decode(token, self.private_key, **decode_kwargs)
            else:
                payload = jwt.decode(token, self.public_key, **decode_kwargs)

            return True, self._claims_from_payload(payload), None
        
        except jwt.ExpiredSignatureError:
            return False, None, "Token expired"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid token: {str(e)}"
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            return False, None, "Token verification failed"
    
    def extract_token_from_request(self) -> Optional[str]:
        """Extract JWT from Authorization header."""
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return None
        
        return auth_header[7:]  # Remove 'Bearer ' prefix
    
    def get_current_user(self) -> Optional[JWTClaims]:
        """Get current user claims from request."""
        token = self.extract_token_from_request()
        
        if not token:
            return None
        
        is_valid, claims, error = self.verify_token(token)
        
        if not is_valid:
            logger.warning(f"Invalid token: {error}")
            return None
        
        return claims


class RunAsManager:
    """Manage admin run-as functionality with audit trail."""

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def __init__(self, auth_manager: JWTAuthManager, audit_store=None):
        """Initialize.
        
        Args:
            auth_manager: JWTAuthManager instance
            audit_store: OpsStore for audit logging
        """
        self.auth_manager = auth_manager
        self.audit_store = audit_store
    
    def create_runas_token(self, admin_user: str, target_user: str,
                          admin_roles: list = None,
                          target_roles: list = None,
                          reason: str = None) -> Tuple[bool, str, str]:
        """Create run-as token.
        
        Args:
            admin_user: Admin user performing run-as
            target_user: User to impersonate
            admin_roles: Admin's roles
            target_roles: Target user's roles
            reason: Reason for run-as (for audit trail)
        
        Returns:
            (success, token_or_error, context_hash)
        """
        # Create context for audit trail
        context = {
            'admin': admin_user,
            'target': target_user,
            'timestamp': self._utc_now_iso(),
            'reason': reason or 'No reason provided'
        }
        
        # Create hash of context
        context_str = json.dumps(context, sort_keys=True)
        context_hash = hashlib.blake2b(
            context_str.encode(),
            digest_size=16
        ).hexdigest()
        
        try:
            # Create token with run-as flag
            token = self.auth_manager.create_token(
                user_id=target_user,
                username=target_user,
                roles=target_roles or [],
                expires_in=3600,  # 1 hour for run-as
                run_as_admin=admin_user,
                context_hash=context_hash
            )
            
            # Log run-as creation
            self._audit_runas_created(
                admin_user=admin_user,
                target_user=target_user,
                context_hash=context_hash,
                reason=reason
            )
            
            logger.info(f"Run-as token created: {admin_user} → {target_user}")
            return True, token, context_hash
        
        except Exception as e:
            error_msg = f"Failed to create run-as token: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def audit_runas_action(self, admin_user: str, target_user: str, context_hash: str,
                          action: str, result: str, status: str = 'success') -> None:
        """Audit an action performed with run-as.
        
        Args:
            admin_user: Admin performing action
            target_user: User being impersonated
            context_hash: Hash of run-as context
            action: Action performed
            result: Result of action
            status: Status (success/error)
        """
        if not self.audit_store:
            return
        
        audit_entry = {
            'timestamp': self._utc_now_iso(),
            'type': 'run_as_action',
            'admin_user': admin_user,
            'target_user': target_user,
            'context_hash': context_hash,
            'action': action,
            'result': result,
            'status': status
        }
        
        try:
            self.audit_store.add_audit(
                'run_as_action',
                json.dumps(audit_entry, separators=(",", ":"), sort_keys=True),
                self._utc_now_iso(),
            )
            logger.info(f"Run-as action logged: {admin_user} as {target_user} → {action}")
        except Exception as e:
            logger.error(f"Failed to log run-as action: {str(e)}")
    
    def _audit_runas_created(self, admin_user: str, target_user: str,
                            context_hash: str, reason: str = None) -> None:
        """Log run-as token creation."""
        if not self.audit_store:
            return
        
        audit_entry = {
            'timestamp': self._utc_now_iso(),
            'type': 'run_as_created',
            'admin': admin_user,
            'target': target_user,
            'context_hash': context_hash,
            'reason': reason
        }
        
        try:
            self.audit_store.add_audit(
                'run_as_created',
                json.dumps(audit_entry, separators=(",", ":"), sort_keys=True),
                self._utc_now_iso(),
            )
        except Exception as e:
            logger.error(f"Failed to log run-as creation: {str(e)}")


def require_auth(f):
    """Decorator: Require valid JWT token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({'error': 'Missing authorization token'}), 401
        
        # This will be set by app initialization
        if not hasattr(request, 'jwt_manager'):
            return jsonify({'error': 'Auth not configured'}), 500
        
        is_valid, claims, error = request.jwt_manager.verify_token(token)
        if not is_valid:
            return jsonify({'error': f'Invalid token: {error}'}), 401
        
        # Store claims in request for handler access
        request.claims = claims
        
        return f(*args, **kwargs)
    
    return decorated


def require_role(*required_roles):
    """Decorator: Require specific role."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(request, 'claims'):
                return jsonify({'error': 'Not authenticated'}), 401
            
            user_roles = request.claims.roles or []
            
            if not any(role in user_roles for role in required_roles):
                return jsonify({
                    'error': 'Insufficient permissions',
                    'required_roles': list(required_roles),
                    'user_roles': user_roles
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated
    
    return decorator


def require_admin(f):
    """Decorator: Require admin role."""
    return require_role('admin')(f)


def inject_current_user(f):
    """Decorator: Inject current user into request."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if token and hasattr(request, 'jwt_manager'):
            is_valid, claims, _ = request.jwt_manager.verify_token(token)
            if is_valid:
                request.current_user = claims.sub
                request.current_user_id = claims.user_id
                request.current_user_roles = claims.roles
                request.run_as_admin = claims.run_as_admin
                request.context_hash = claims.context_hash
        
        return f(*args, **kwargs)
    
    return decorated
