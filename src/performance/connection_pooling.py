"""
INIDS Performance Optimization Module: Connection Pooling

Connection pools for Redis, HTTP, and other persistent connections.
Reduces connection overhead and improves throughput.
"""

import threading
import time
import logging
from collections import deque
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field


@dataclass
class ConnectionPoolStats:
    """Connection pool statistics"""
    total_created: int = 0
    current_idle: int = 0
    current_active: int = 0
    reused_count: int = 0
    created_count: int = 0
    failed_connections: int = 0
    total_wait_time: float = 0.0
    
    def avg_wait_time_ms(self) -> float:
        """Average wait time in milliseconds"""
        if self.reused_count + self.created_count == 0:
            return 0.0
        return self.total_wait_time / (self.reused_count + self.created_count) * 1000


class Connection:
    """
    Pooled connection wrapper.
    
    Tracks connection state and usage.
    """
    
    def __init__(self, connection_obj: Any, connection_id: int):
        """
        Initialize pooled connection.
        
        Args:
            connection_obj: Underlying connection object
            connection_id: Unique connection ID
        """
        self.connection = connection_obj
        self.connection_id = connection_id
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.is_valid = True
    
    def mark_used(self):
        """Mark connection as used"""
        self.last_used = time.time()
        self.use_count += 1
    
    def age_seconds(self) -> float:
        """Age of connection in seconds"""
        return time.time() - self.created_at
    
    def idle_seconds(self) -> float:
        """Idle time in seconds"""
        return time.time() - self.last_used


class ConnectionPool:
    """
    Generic connection pool for managing persistent connections.
    
    Features:
    - Connection reuse
    - Health checking
    - Timeout handling
    - Statistics tracking
    """
    
    def __init__(
        self,
        connection_factory: Callable,
        initial_size: int = 5,
        max_size: int = 20,
        max_age_seconds: int = 3600,
        max_idle_seconds: int = 300,
        health_check_interval: int = 60,
        health_check_func: Optional[Callable] = None,
    ):
        """
        Initialize connection pool.
        
        Args:
            connection_factory: Callable that creates new connections
            initial_size: Initial pool size
            max_size: Maximum pool size
            max_age_seconds: Max connection age before retirement
            max_idle_seconds: Max idle time before closing
            health_check_interval: Health check interval
            health_check_func: Optional function to validate connections
        """
        self.connection_factory = connection_factory
        self.max_size = max_size
        self.max_age = max_age_seconds
        self.max_idle = max_idle_seconds
        self.health_check_interval = health_check_interval
        self.health_check_func = health_check_func
        
        self.idle_connections: deque = deque()
        self.active_connections: Dict[int, Connection] = {}
        self.lock = threading.Lock()
        
        self.stats = ConnectionPoolStats()
        self.connection_counter = 0
        self.last_health_check = time.time()
        
        self.logger = logging.getLogger("INIDS.Performance.ConnectionPool")
        
        # Create initial connections
        for _ in range(initial_size):
            try:
                conn_obj = connection_factory()
                conn = Connection(conn_obj, self.connection_counter)
                self.connection_counter += 1
                self.idle_connections.append(conn)
                self.stats.total_created += 1
            except Exception as e:
                self.logger.error(f"Failed to create initial connection: {e}")
                self.stats.failed_connections += 1
    
    def acquire(self, timeout: float = 5.0) -> Optional[Connection]:
        """
        Acquire connection from pool.
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Connection or None if timeout/failed
        """
        start_time = time.time()
        
        with self.lock:
            # Health check if needed
            if time.time() - self.last_health_check > self.health_check_interval:
                self._health_check_idle_connections()
                self.last_health_check = time.time()
            
            # Try to get idle connection
            while self.idle_connections:
                conn = self.idle_connections.popleft()
                
                # Check connection validity
                if self._is_connection_valid(conn):
                    conn.mark_used()
                    self.active_connections[conn.connection_id] = conn
                    self.stats.reused_count += 1
                    self.stats.current_idle = len(self.idle_connections)
                    
                    wait_time = time.time() - start_time
                    self.stats.total_wait_time += wait_time
                    
                    return conn
                else:
                    # Connection invalid, close it
                    self._close_connection(conn)
            
            # No idle connections, create new if possible
            if self.stats.total_created < self.max_size:
                try:
                    conn_obj = self.connection_factory()
                    conn = Connection(conn_obj, self.connection_counter)
                    self.connection_counter += 1
                    conn.mark_used()
                    self.active_connections[conn.connection_id] = conn
                    self.stats.total_created += 1
                    self.stats.created_count += 1
                    
                    wait_time = time.time() - start_time
                    self.stats.total_wait_time += wait_time
                    
                    return conn
                except Exception as e:
                    self.logger.error(f"Failed to create connection: {e}")
                    self.stats.failed_connections += 1
            
            # Wait for available connection (with timeout)
            while time.time() - start_time < timeout:
                self.lock.release()
                time.sleep(0.1)
                self.lock.acquire()
                
                if self.idle_connections:
                    conn = self.idle_connections.popleft()
                    if self._is_connection_valid(conn):
                        conn.mark_used()
                        self.active_connections[conn.connection_id] = conn
                        self.stats.reused_count += 1
                        
                        wait_time = time.time() - start_time
                        self.stats.total_wait_time += wait_time
                        
                        return conn
                    else:
                        self._close_connection(conn)
        
        self.logger.warning(f"Failed to acquire connection within {timeout}s")
        return None
    
    def release(self, conn: Connection) -> None:
        """
        Release connection back to pool.
        
        Args:
            conn: Connection to release
        """
        with self.lock:
            if conn.connection_id in self.active_connections:
                del self.active_connections[conn.connection_id]
            
            if self._is_connection_valid(conn):
                self.idle_connections.appendleft(conn)
                self.stats.current_idle = len(self.idle_connections)
            else:
                self._close_connection(conn)
    
    def _is_connection_valid(self, conn: Connection) -> bool:
        """
        Check if connection is still valid.
        
        Args:
            conn: Connection to validate
        
        Returns:
            True if connection is valid
        """
        # Check age
        if conn.age_seconds() > self.max_age:
            return False
        
        # Check idle time
        if conn.idle_seconds() > self.max_idle:
            return False
        
        # Check health function if provided
        if self.health_check_func:
            try:
                return self.health_check_func(conn.connection)
            except:
                return False
        
        return conn.is_valid
    
    def _health_check_idle_connections(self) -> None:
        """
        Health check idle connections and remove invalid ones.
        Internal method, must be called with lock held.
        """
        valid_connections = deque()
        
        while self.idle_connections:
            conn = self.idle_connections.popleft()
            
            if self._is_connection_valid(conn):
                valid_connections.append(conn)
            else:
                self._close_connection(conn)
        
        self.idle_connections = valid_connections
    
    def _close_connection(self, conn: Connection) -> None:
        """
        Close connection safely.
        Internal method.
        
        Args:
            conn: Connection to close
        """
        try:
            if hasattr(conn.connection, 'close'):
                conn.connection.close()
        except:
            pass
        
        conn.is_valid = False
    
    def get_stats(self) -> ConnectionPoolStats:
        """Get pool statistics"""
        with self.lock:
            stats = ConnectionPoolStats(
                total_created=self.stats.total_created,
                current_idle=len(self.idle_connections),
                current_active=len(self.active_connections),
                reused_count=self.stats.reused_count,
                created_count=self.stats.created_count,
                failed_connections=self.stats.failed_connections,
                total_wait_time=self.stats.total_wait_time,
            )
            return stats
    
    def close_all(self) -> None:
        """Close all connections"""
        with self.lock:
            # Close idle connections
            while self.idle_connections:
                conn = self.idle_connections.popleft()
                self._close_connection(conn)
            
            # Close active connections
            for conn in self.active_connections.values():
                self._close_connection(conn)
            
            self.active_connections.clear()


class RedisConnectionPool:
    """
    Redis-specific connection pool.
    
    Manages Redis connections with built-in health checks.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        initial_size: int = 5,
        max_size: int = 20,
    ):
        """
        Initialize Redis connection pool.
        
        Args:
            host: Redis host
            port: Redis port
            db: Database number
            password: Password (optional)
            initial_size: Initial pool size
            max_size: Maximum pool size
        """
        def create_redis_connection():
            import redis
            return redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
        
        def health_check(redis_conn):
            try:
                redis_conn.ping()
                return True
            except:
                return False
        
        self.pool = ConnectionPool(
            connection_factory=create_redis_connection,
            initial_size=initial_size,
            max_size=max_size,
            health_check_func=health_check,
        )
    
    def acquire(self) -> Optional[Any]:
        """Acquire Redis connection"""
        conn = self.pool.acquire()
        return conn.connection if conn else None
    
    def release(self, redis_conn: Any) -> None:
        """Release Redis connection"""
        # Find and release connection wrapper
        # Note: This is simplified - real implementation would track it
        pass
    
    def get_stats(self) -> ConnectionPoolStats:
        """Get pool statistics"""
        return self.pool.get_stats()


class HTTPConnectionPool:
    """
    HTTP connection pool for webhooks and REST APIs.
    
    Reuses HTTP sessions for better performance.
    """
    
    def __init__(
        self,
        initial_size: int = 5,
        max_size: int = 20,
        timeout: int = 5,
    ):
        """
        Initialize HTTP connection pool.
        
        Args:
            initial_size: Initial pool size
            max_size: Maximum pool size
            timeout: Connection timeout
        """
        def create_http_session():
            import requests
            session = requests.Session()
            session.timeout = timeout
            return session
        
        def health_check(session):
            try:
                # Session doesn't need health check (auto-reconnects)
                return True
            except:
                return False
        
        self.pool = ConnectionPool(
            connection_factory=create_http_session,
            initial_size=initial_size,
            max_size=max_size,
            health_check_func=health_check,
        )
    
    def acquire(self) -> Optional[Any]:
        """Acquire HTTP session"""
        conn = self.pool.acquire()
        return conn.connection if conn else None
    
    def get_stats(self) -> ConnectionPoolStats:
        """Get pool statistics"""
        return self.pool.get_stats()


# Global instances
_redis_pool: Optional[RedisConnectionPool] = None
_http_pool: Optional[HTTPConnectionPool] = None


def get_redis_pool(
    host: str = "localhost",
    port: int = 6379,
) -> Optional[RedisConnectionPool]:
    """Get or create global Redis connection pool"""
    global _redis_pool
    try:
        if _redis_pool is None:
            _redis_pool = RedisConnectionPool(host=host, port=port)
        return _redis_pool
    except ImportError:
        return None


def get_http_pool() -> HTTPConnectionPool:
    """Get or create global HTTP connection pool"""
    global _http_pool
    if _http_pool is None:
        _http_pool = HTTPConnectionPool()
    return _http_pool
