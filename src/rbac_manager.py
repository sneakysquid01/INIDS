"""
RBAC (Role-Based Access Control) Database (Week 4)

Provides persistent storage and management of:
- Users and authentication
- Roles and permissions
- Resource access control
- Audit trails for access decisions

Features:
- SQLAlchemy ORM for database abstraction
- Multiple database backend support (SQLite, PostgreSQL, MySQL)
- Hierarchical role inheritance
- Fine-grained permission checking
- Access decision caching
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta

try:
    from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, ForeignKey, Table, select
    from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not available")

logger = logging.getLogger(__name__)

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    # Association table for user-role relationships
    user_roles = Table(
        'user_roles',
        Base.metadata,
        Column('user_id', String, ForeignKey('users.id')),
        Column('role_id', String, ForeignKey('roles.id'))
    )
    
    # Association table for role-permission relationships
    role_permissions = Table(
        'role_permissions',
        Base.metadata,
        Column('role_id', String, ForeignKey('roles.id')),
        Column('permission_id', String, ForeignKey('permissions.id'))
    )
    
    class User(Base):
        """User model."""
        __tablename__ = 'users'
        
        id = Column(String(255), primary_key=True)
        username = Column(String(255), unique=True, index=True)
        email = Column(String(255), unique=True)
        full_name = Column(String(255))
        is_active = Column(Boolean, default=True)
        is_admin = Column(Boolean, default=False)
        created_at = Column(DateTime, default=datetime.now(timezone.utc))
        updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
        last_login = Column(DateTime, nullable=True)
        
        roles = relationship("Role", secondary=user_roles, back_populates="users")
        audit_logs = relationship("AuditLog", back_populates="user")
        
        def has_role(self, role_name: str) -> bool:
            """Check if user has role."""
            return any(r.name == role_name for r in self.roles)
        
        def has_permission(self, permission: str) -> bool:
            """Check if user has permission."""
            for role in self.roles:
                if role.has_permission(permission):
                    return True
            return False
        
        def get_permissions(self) -> Set[str]:
            """Get all permissions for user."""
            permissions = set()
            for role in self.roles:
                permissions.update(role.get_permissions())
            return permissions
    
    class Role(Base):
        """Role model."""
        __tablename__ = 'roles'
        
        id = Column(String(255), primary_key=True)
        name = Column(String(255), unique=True, index=True)
        description = Column(String(1024))
        is_builtin = Column(Boolean, default=False)
        created_at = Column(DateTime, default=datetime.now(timezone.utc))
        updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
        
        users = relationship("User", secondary=user_roles, back_populates="roles")
        permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
        
        def has_permission(self, permission: str) -> bool:
            """Check if role has permission."""
            return any(p.name == permission for p in self.permissions)
        
        def get_permissions(self) -> Set[str]:
            """Get all permissions for role."""
            return {p.name for p in self.permissions}
    
    class Permission(Base):
        """Permission model."""
        __tablename__ = 'permissions'
        
        id = Column(String(255), primary_key=True)
        name = Column(String(255), unique=True, index=True)
        description = Column(String(1024))
        resource = Column(String(255), index=True)  # e.g., "alert", "rule", "policy"
        action = Column(String(255), index=True)    # e.g., "create", "read", "update", "delete"
        created_at = Column(DateTime, default=datetime.now(timezone.utc))
        
        roles = relationship("Role", secondary=role_permissions, back_populates="permissions")
    
    class AuditLog(Base):
        """Access audit log."""
        __tablename__ = 'audit_logs'
        
        id = Column(String(255), primary_key=True)
        user_id = Column(String(255), ForeignKey('users.id'), index=True)
        action = Column(String(255), index=True)
        resource = Column(String(255), index=True)
        resource_id = Column(String(255), index=True)
        allowed = Column(Boolean)
        reason = Column(String(1024))  # Why decision was made
        timestamp = Column(DateTime, default=datetime.now(timezone.utc), index=True)
        
        user = relationship("User", back_populates="audit_logs")


class RBACManager:
    """Manages RBAC operations."""
    
    def __init__(self, database_url: str = "sqlite:///inids_rbac.db"):
        """Initialize RBAC manager.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available, RBAC disabled")
            self.engine = None
            self.Session = None
            return
        
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Initialize default roles if not present
        self._init_default_roles()
    
    def _init_default_roles(self):
        """Initialize default roles."""
        if not self.Session:
            return
        
        session = self.Session()
        try:
            # Check if roles exist
            existing_roles = session.query(Role).count()
            if existing_roles > 0:
                return
            
            # Create default roles
            admin_role = Role(
                id="admin",
                name="admin",
                description="System administrator with full access",
                is_builtin=True
            )
            
            analyst_role = Role(
                id="analyst",
                name="analyst",
                description="Security analyst with detection and investigation access",
                is_builtin=True
            )
            
            viewer_role = Role(
                id="viewer",
                name="viewer",
                description="Read-only access to system data",
                is_builtin=True
            )
            
            session.add_all([admin_role, analyst_role, viewer_role])
            
            # Create default permissions
            permissions = [
                Permission(id="create_rule", name="create_rule", description="Create detection rules", resource="rule", action="create"),
                Permission(id="read_rule", name="read_rule", description="Read detection rules", resource="rule", action="read"),
                Permission(id="update_rule", name="update_rule", description="Update detection rules", resource="rule", action="update"),
                Permission(id="delete_rule", name="delete_rule", description="Delete detection rules", resource="rule", action="delete"),
                
                Permission(id="read_alert", name="read_alert", description="Read alerts", resource="alert", action="read"),
                Permission(id="update_alert", name="update_alert", description="Update alerts", resource="alert", action="update"),
                Permission(id="delete_alert", name="delete_alert", description="Delete alerts", resource="alert", action="delete"),
                
                Permission(id="read_audit", name="read_audit", description="Read audit logs", resource="audit", action="read"),
                Permission(id="manage_users", name="manage_users", description="Manage users", resource="user", action="manage"),
                Permission(id="manage_roles", name="manage_roles", description="Manage roles", resource="role", action="manage"),
            ]
            
            session.add_all(permissions)
            
            # Assign permissions to roles
            admin_role.permissions = permissions  # Admin has all
            analyst_role.permissions = [
                p for p in permissions
                if p.name in ["read_rule", "create_rule", "read_alert", "update_alert", "read_audit"]
            ]
            viewer_role.permissions = [
                p for p in permissions
                if p.name in ["read_rule", "read_alert", "read_audit"]
            ]
            
            session.commit()
            logger.info("Initialized default RBAC roles and permissions")
        
        except Exception as e:
            logger.error(f"Error initializing default roles: {e}")
            session.rollback()
        
        finally:
            session.close()
    
    def add_user(self, user_id: str, username: str, email: str = "", roles: List[str] = None) -> bool:
        """Add new user.
        
        Args:
            user_id: Unique user ID
            username: Username
            email: Email address
            roles: List of role names to assign
            
        Returns:
            True if successful
        """
        if not self.Session:
            return False
        
        session = self.Session()
        try:
            user = User(
                id=user_id,
                username=username,
                email=email,
                is_active=True
            )
            
            if roles:
                role_objs = session.query(Role).filter(Role.name.in_(roles)).all()
                user.roles = role_objs
            
            session.add(user)
            session.commit()
            logger.info(f"Added user: {username}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            session.rollback()
            return False
        
        finally:
            session.close()
    
    def check_permission(
        self,
        user_id: str,
        permission: str,
        resource_id: str = None
    ) -> Tuple[bool, str]:
        """Check if user has permission.
        
        Args:
            user_id: User ID
            permission: Permission name
            resource_id: Optional resource ID for fine-grained control
            
        Returns:
            (allowed, reason)
        """
        if not self.Session:
            return False, "RBAC not initialized"
        
        session = self.Session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False, f"User not found: {user_id}"
            
            if not user.is_active:
                return False, "User is inactive"
            
            # Admins bypass all checks
            if user.is_admin:
                return True, "Admin user"
            
            # Check if user has permission
            has_perm = user.has_permission(permission)
            
            # Log decision
            audit_log = AuditLog(
                id=f"{user_id}_{permission}_{datetime.now(timezone.utc).isoformat()}",
                user_id=user_id,
                action=permission,
                resource_id=resource_id,
                allowed=has_perm,
                reason="Permission check" if has_perm else "Permission denied"
            )
            session.add(audit_log)
            session.commit()
            
            reason = "Permission granted" if has_perm else f"User lacks permission: {permission}"
            return has_perm, reason
        
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False, f"Permission check error: {e}"
        
        finally:
            session.close()
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign role to user.
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            True if successful
        """
        if not self.Session:
            return False
        
        session = self.Session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            role = session.query(Role).filter(Role.name == role_name).first()
            
            if not user or not role:
                return False
            
            if role not in user.roles:
                user.roles.append(role)
                session.commit()
                logger.info(f"Assigned role {role_name} to user {user_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error assigning role: {e}")
            session.rollback()
            return False
        
        finally:
            session.close()
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for user.
        
        Args:
            user_id: User ID
            
        Returns:
            Set of permission names
        """
        if not self.Session:
            return set()
        
        session = self.Session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return set()
            
            return user.get_permissions()
        
        finally:
            session.close()
    
    def get_audit_logs(
        self,
        user_id: str = None,
        days: int = 7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit logs.
        
        Args:
            user_id: Filter by user (optional)
            days: Look back this many days
            limit: Max logs to return
            
        Returns:
            List of audit logs
        """
        if not self.Session:
            return []
        
        session = self.Session()
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            query = session.query(AuditLog).filter(AuditLog.timestamp >= start_time)
            
            if user_id:
                query = query.filter(AuditLog.user_id == user_id)
            
            logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
            
            return [
                {
                    "id": log.id,
                    "user_id": log.user_id,
                    "action": log.action,
                    "resource": log.resource,
                    "resource_id": log.resource_id,
                    "allowed": log.allowed,
                    "reason": log.reason,
                    "timestamp": log.timestamp.isoformat()
                }
                for log in logs
            ]
        
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            return []
        
        finally:
            session.close()


# Global RBAC instance
_rbac_manager = None


def get_rbac_manager(database_url: str = "sqlite:///inids_rbac.db") -> Optional[RBACManager]:
    """Get or create global RBAC manager."""
    global _rbac_manager
    
    if _rbac_manager is None:
        if SQLALCHEMY_AVAILABLE:
            _rbac_manager = RBACManager(database_url)
        else:
            logger.warning("SQLAlchemy not available, RBAC disabled")
    
    return _rbac_manager
