"""
INIDS Output Backends Module

Multiple output destinations for EVE JSON events:
- FileBackend: Rotating log files
- SyslogBackend: System logging (UDP/TCP)
- RedisBackend: Redis queue for real-time processing
- WebhookBackend: HTTP POST to external services

Each backend is thread-safe and non-blocking.
"""

import json
import socket
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Callable
from queue import Queue, Empty
from dataclasses import dataclass

from .eve_json import EVEEvent


@dataclass
class BackendStats:
    """Statistics for backend operation"""
    events_sent: int = 0
    events_failed: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def add_error(self, error: str):
        """Record an error"""
        self.errors.append(error)
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]  # Keep last 100
    
    def clear_errors(self):
        """Clear error list"""
        self.errors = []


class OutputBackend(ABC):
    """Base class for output backends"""
    
    def __init__(self, name: str):
        self.name = name
        self.stats = BackendStats()
        self.logger = logging.getLogger(f"INIDS.Output.{name}")
    
    @abstractmethod
    def send(self, event: EVEEvent) -> bool:
        """
        Send event to backend.
        
        Args:
            event: EVE JSON event
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close backend connection (cleanup)"""
        pass
    
    def get_stats(self) -> BackendStats:
        """Get backend statistics"""
        return self.stats


class FileBackend(OutputBackend):
    """
    File-based backend with log rotation.
    
    Writes EVE JSON events to file, one per line.
    Supports rotation by size or daily.
    """
    
    def __init__(
        self,
        filepath: str = "/var/log/inids/alerts.json",
        max_size_mb: int = 100,
        backup_count: int = 10,
    ):
        super().__init__("File")
        self.filepath = Path(filepath)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self.current_size = 0
        self.lock = threading.Lock()
        
        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current file size if exists
        if self.filepath.exists():
            self.current_size = self.filepath.stat().st_size
        
        self.file_handle = None
        self._open_file()
    
    def _open_file(self):
        """Open or reopen file"""
        try:
            self.file_handle = open(self.filepath, 'a', buffering=1)
        except IOError as e:
            self.logger.error(f"Failed to open file: {e}")
            self.stats.add_error(str(e))
    
    def _rotate_file(self):
        """Rotate log file if it exceeds max size"""
        if self.current_size >= self.max_size_bytes:
            # Close current file
            if self.file_handle:
                self.file_handle.close()
            
            # Rotate backups
            for i in range(self.backup_count - 1, 0, -1):
                old_path = self.filepath.with_suffix(f'.{i}.json')
                new_path = self.filepath.with_suffix(f'.{i+1}.json')
                if old_path.exists():
                    old_path.rename(new_path)
            
            # Rename current to .1
            backup_path = self.filepath.with_suffix('.1.json')
            if self.filepath.exists():
                self.filepath.rename(backup_path)
            
            # Reset size and open new file
            self.current_size = 0
            self._open_file()
    
    def send(self, event: EVEEvent) -> bool:
        """Write event to file"""
        with self.lock:
            try:
                # Check if rotation needed
                self._rotate_file()
                
                # Write event as JSON line
                json_line = event.to_json()
                self.file_handle.write(json_line + '\n')
                self.file_handle.flush()
                
                self.current_size += len(json_line) + 1
                self.stats.events_sent += 1
                return True
            
            except Exception as e:
                self.logger.error(f"Failed to write event: {e}")
                self.stats.events_failed += 1
                self.stats.add_error(str(e))
                return False
    
    def close(self):
        """Close file handle"""
        with self.lock:
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None


class SyslogBackend(OutputBackend):
    """
    Syslog output backend (UDP or TCP).
    
    Sends EVE JSON events to syslog server (e.g., rsyslog, syslog-ng).
    Supports both UDP (faster) and TCP (reliable).
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 514,
        protocol: str = "udp",
        facility: int = 16,  # local0
        severity: int = 6,   # info
    ):
        super().__init__("Syslog")
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.facility = facility
        self.severity = severity
        self.socket = None
        self.lock = threading.Lock()
        
        if self.protocol not in ["udp", "tcp"]:
            raise ValueError(f"Invalid protocol: {protocol}")
        
        self._connect()
    
    def _connect(self):
        """Connect to syslog server"""
        try:
            if self.protocol == "udp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            else:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
            
            self.logger.info(f"Connected to syslog {self.host}:{self.port} ({self.protocol})")
        
        except Exception as e:
            self.logger.error(f"Failed to connect to syslog: {e}")
            self.stats.add_error(str(e))
            self.socket = None
    
    def send(self, event: EVEEvent) -> bool:
        """Send event to syslog"""
        with self.lock:
            if not self.socket:
                self._connect()
                if not self.socket:
                    self.stats.events_failed += 1
                    return False
            
            try:
                # Calculate priority = facility * 8 + severity
                priority = self.facility * 8 + self.severity
                
                # Format syslog message
                json_str = event.to_json()
                timestamp = datetime.now().strftime("%b %d %H:%M:%S")
                syslog_msg = f"<{priority}> {timestamp} inids[{id(event)}]: {json_str}"
                
                # Send message
                if self.protocol == "udp":
                    self.socket.sendto(syslog_msg.encode(), (self.host, self.port))
                else:
                    self.socket.send(syslog_msg.encode() + b'\n')
                
                self.stats.events_sent += 1
                return True
            
            except Exception as e:
                self.logger.error(f"Failed to send syslog: {e}")
                self.stats.events_failed += 1
                self.stats.add_error(str(e))
                self.socket = None
                return False
    
    def close(self):
        """Close syslog connection"""
        with self.lock:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None


class RedisBackend(OutputBackend):
    """
    Redis output backend.
    
    Pushes EVE JSON events to Redis queue for real-time processing
    by external tools (e.g., Logstash, custom consumers).
    
    Supports:
    - List (LPUSH)
    - PubSub (PUBLISH)
    - Stream (XADD)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        key: str = "inids:alerts",
        mode: str = "list",  # list, pubsub, stream
        password: Optional[str] = None,
    ):
        super().__init__("Redis")
        self.host = host
        self.port = port
        self.db = db
        self.key = key
        self.mode = mode
        self.password = password
        self.redis = None
        
        # Try to import redis
        try:
            import redis
            self.redis_module = redis
            self._connect()
        except ImportError:
            self.logger.error("redis-py not installed: pip install redis")
            self.stats.add_error("redis-py not installed")
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self.redis = self.redis_module.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            # Test connection
            self.redis.ping()
            self.logger.info(f"Connected to Redis {self.host}:{self.port}")
        
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.stats.add_error(str(e))
            self.redis = None
    
    def send(self, event: EVEEvent) -> bool:
        """Send event to Redis"""
        if not self.redis:
            self.stats.events_failed += 1
            return False
        
        try:
            json_str = event.to_json()
            
            if self.mode == "list":
                # LPUSH to list
                self.redis.lpush(self.key, json_str)
            
            elif self.mode == "pubsub":
                # PUBLISH to channel
                self.redis.publish(self.key, json_str)
            
            elif self.mode == "stream":
                # XADD to stream
                self.redis.xadd(self.key, {"data": json_str})
            
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            
            self.stats.events_sent += 1
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to send to Redis: {e}")
            self.stats.events_failed += 1
            self.stats.add_error(str(e))
            return False
    
    def close(self):
        """Close Redis connection"""
        if self.redis:
            try:
                self.redis.close()
            except:
                pass
            self.redis = None


class WebhookBackend(OutputBackend):
    """
    Webhook output backend.
    
    Sends EVE JSON events to external HTTP(S) endpoint via POST.
    Non-blocking with background worker thread.
    """
    
    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        batch_size: int = 1,
        max_queue_size: int = 1000,
    ):
        super().__init__("Webhook")
        self.url = url
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.event_queue = Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.running = False
        
        # Try to import requests
        try:
            import requests
            self.requests_module = requests
            self._start_worker()
        except ImportError:
            self.logger.error("requests not installed: pip install requests")
            self.stats.add_error("requests not installed")
    
    def _start_worker(self):
        """Start background worker thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Webhook worker thread started")
    
    def _worker_loop(self):
        """Background worker thread loop"""
        batch = []
        
        while self.running:
            try:
                # Collect events for batch
                while len(batch) < self.batch_size:
                    try:
                        event = self.event_queue.get(timeout=1.0)
                        batch.append(event)
                    except Empty:
                        break
                
                # Send batch if we have events
                if batch:
                    self._send_batch(batch)
                    batch = []
            
            except Exception as e:
                self.logger.error(f"Worker thread error: {e}")
    
    def _send_batch(self, events: List[EVEEvent]) -> bool:
        """Send batch of events to webhook"""
        try:
            # Prepare payload
            payload = [event.to_dict() for event in events]
            json_data = json.dumps(payload)
            
            # POST to webhook
            response = self.requests_module.post(
                self.url,
                data=json_data,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                self.stats.events_sent += len(events)
                return True
            else:
                self.logger.error(f"Webhook returned {response.status_code}")
                self.stats.events_failed += len(events)
                self.stats.add_error(f"HTTP {response.status_code}")
                return False
        
        except Exception as e:
            self.logger.error(f"Failed to send webhook: {e}")
            self.stats.events_failed += len(events)
            self.stats.add_error(str(e))
            return False
    
    def send(self, event: EVEEvent) -> bool:
        """Queue event for webhook delivery (non-blocking)"""
        try:
            self.event_queue.put_nowait(event)
            return True
        except:
            self.logger.error("Webhook queue full")
            self.stats.events_failed += 1
            return False
    
    def close(self):
        """Close webhook backend and wait for worker"""
        self.running = False
        
        # Flush remaining events
        remaining = []
        while True:
            try:
                remaining.append(self.event_queue.get_nowait())
            except Empty:
                break
        
        if remaining:
            self._send_batch(remaining)
        
        # Wait for worker thread
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)


class OutputAggregator:
    """
    Manages multiple output backends.
    
    Routes EVE JSON events to all configured backends concurrently.
    Thread-safe and non-blocking.
    """
    
    def __init__(self):
        self.backends: List[OutputBackend] = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger("INIDS.Output.Aggregator")
    
    def add_backend(self, backend: OutputBackend) -> None:
        """Add output backend"""
        with self.lock:
            self.backends.append(backend)
            self.logger.info(f"Added backend: {backend.name}")
    
    def send_event(self, event: EVEEvent) -> bool:
        """
        Send event to all backends.
        
        Returns:
            True if at least one backend succeeded
        """
        with self.lock:
            if not self.backends:
                return False
            
            succeeded = False
            for backend in self.backends:
                try:
                    if backend.send(event):
                        succeeded = True
                except Exception as e:
                    self.logger.error(f"Backend {backend.name} error: {e}")
            
            return succeeded
    
    def send_events(self, events: List[EVEEvent]) -> int:
        """
        Send multiple events to all backends.
        
        Returns:
            Number of events successfully sent to at least one backend
        """
        count = 0
        for event in events:
            if self.send_event(event):
                count += 1
        return count
    
    def get_stats(self) -> dict:
        """Get statistics from all backends"""
        with self.lock:
            return {
                backend.name: {
                    "events_sent": backend.stats.events_sent,
                    "events_failed": backend.stats.events_failed,
                    "recent_errors": backend.stats.errors[-5:],
                }
                for backend in self.backends
            }
    
    def close_all(self) -> None:
        """Close all backends"""
        with self.lock:
            for backend in self.backends:
                try:
                    backend.close()
                    self.logger.info(f"Closed backend: {backend.name}")
                except Exception as e:
                    self.logger.error(f"Error closing {backend.name}: {e}")
            self.backends.clear()
