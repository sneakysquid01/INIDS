"""
Distributed State Store - Abstraction for Redis/Etcd

Provides unified interface for distributed state management:
- Key-value operations
- Distributed locking
- Event streams
- Hash operations

Supports:
- Redis: In-memory data store (easy to deploy)
- Etcd: Consistent distributed configuration (HA ready)
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import timedelta

logger = logging.getLogger(__name__)


class DistributedStateStore(ABC):
    """Abstract base class for distributed state stores."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to state store."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from state store."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set key-value pair."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    async def set_hash(self, hash_key: str, field: str, value: Dict, ttl: int = None) -> bool:
        """Set hash field."""
        pass
    
    @abstractmethod
    async def get_hash(self, hash_key: str) -> Dict[str, Any]:
        """Get all hash fields."""
        pass
    
    @abstractmethod
    async def acquire_lock(self, lock_key: str, owner: str, ttl: int = 10) -> bool:
        """Acquire distributed lock."""
        pass
    
    @abstractmethod
    async def renew_lock(self, lock_key: str, owner: str, ttl: int = 10) -> bool:
        """Renew distributed lock."""
        pass
    
    @abstractmethod
    async def release_lock(self, lock_key: str, owner: str) -> bool:
        """Release distributed lock."""
        pass
    
    @abstractmethod
    async def push_to_stream(self, stream_key: str, data: Dict) -> str:
        """Push message to stream, returns message ID."""
        pass
    
    @abstractmethod
    async def read_stream(self, stream_key: str, count: int = 10) -> List[Dict]:
        """Read messages from stream."""
        pass


class RedisStateStore(DistributedStateStore):
    """Redis-backed distributed state store."""
    
    def __init__(self, url: str = "redis://localhost:6379"):
        """Initialize Redis state store.
        
        Args:
            url: Redis URL (redis://host:port/db)
        """
        self.url = url
        self.redis = None
    
    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            import aioredis
            self.redis = await aioredis.create_redis_pool(self.url)
            logger.info(f"Connected to Redis: {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
    
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set key-value pair."""
        try:
            if ttl:
                await self.redis.setex(key, ttl, value)
            else:
                await self.redis.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        try:
            value = await self.redis.get(key)
            return value.decode() if value else None
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    async def set_hash(self, hash_key: str, field: str, value: Dict, ttl: int = None) -> bool:
        """Set hash field."""
        try:
            value_json = json.dumps(value)
            await self.redis.hset(hash_key, field, value_json)
            if ttl:
                await self.redis.expire(hash_key, ttl)
            return True
        except Exception as e:
            logger.error(f"Error setting hash {hash_key}:{field}: {e}")
            return False
    
    async def get_hash(self, hash_key: str) -> Dict[str, Any]:
        """Get all hash fields."""
        try:
            hash_data = await self.redis.hgetall(hash_key)
            result = {}
            for key, value in hash_data.items():
                try:
                    result[key.decode()] = json.loads(value.decode())
                except:
                    result[key.decode()] = value.decode()
            return result
        except Exception as e:
            logger.error(f"Error getting hash {hash_key}: {e}")
            return {}
    
    async def acquire_lock(self, lock_key: str, owner: str, ttl: int = 10) -> bool:
        """Acquire distributed lock."""
        try:
            # SET key value EX ttl NX (only set if not exists)
            result = await self.redis.set(lock_key, owner, expire=ttl, exist=False)
            return result == b"OK" or result == True
        except Exception as e:
            logger.error(f"Error acquiring lock {lock_key}: {e}")
            return False
    
    async def renew_lock(self, lock_key: str, owner: str, ttl: int = 10) -> bool:
        """Renew distributed lock."""
        try:
            # Check current owner
            current_owner = await self.get(lock_key)
            if current_owner == owner:
                await self.redis.expire(lock_key, ttl)
                return True
            return False
        except Exception as e:
            logger.error(f"Error renewing lock {lock_key}: {e}")
            return False
    
    async def release_lock(self, lock_key: str, owner: str) -> bool:
        """Release distributed lock."""
        try:
            current_owner = await self.get(lock_key)
            if current_owner == owner:
                await self.delete(lock_key)
                return True
            return False
        except Exception as e:
            logger.error(f"Error releasing lock {lock_key}: {e}")
            return False
    
    async def push_to_stream(self, stream_key: str, data: Dict) -> str:
        """Push message to stream."""
        try:
            data_json = json.dumps(data)
            msg_id = await self.redis.xadd(stream_key, {"data": data_json})
            return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
        except Exception as e:
            logger.error(f"Error pushing to stream {stream_key}: {e}")
            return ""
    
    async def read_stream(self, stream_key: str, count: int = 10) -> List[Dict]:
        """Read messages from stream."""
        try:
            # Read last N messages
            messages = await self.redis.xrevrange(stream_key, count=count)
            result = []
            for msg_id, fields in messages:
                data_str = fields.get(b"data", b"{}").decode()
                result.append(json.loads(data_str))
            return result
        except Exception as e:
            logger.error(f"Error reading stream {stream_key}: {e}")
            return []


class EtcdStateStore(DistributedStateStore):
    """Etcd-backed distributed state store."""
    
    def __init__(self, url: str = "etcd://localhost:2379"):
        """Initialize Etcd state store.
        
        Args:
            url: Etcd URL (etcd://host:port)
        """
        self.url = url
        self.etcd = None
        self.prefix = "inids"
    
    async def connect(self) -> bool:
        """Connect to Etcd."""
        try:
            import aioetcd3
            host, port = self.url.replace("etcd://", "").split(":")
            self.etcd = await aioetcd3.client(host=host, port=int(port))
            logger.info(f"Connected to Etcd: {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Etcd: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Etcd."""
        if self.etcd:
            await self.etcd.close()
    
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set key-value pair."""
        try:
            full_key = f"{self.prefix}/{key}"
            lease = None
            if ttl:
                lease = await self.etcd.grant_lease(ttl)
            await self.etcd.put(full_key, value, lease=lease)
            return True
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        try:
            full_key = f"{self.prefix}/{key}"
            value, _ = await self.etcd.get(full_key)
            return value.decode() if value else None
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            full_key = f"{self.prefix}/{key}"
            await self.etcd.delete(full_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    async def set_hash(self, hash_key: str, field: str, value: Dict, ttl: int = None) -> bool:
        """Set hash field."""
        try:
            full_key = f"{self.prefix}/{hash_key}/{field}"
            value_json = json.dumps(value)
            lease = None
            if ttl:
                lease = await self.etcd.grant_lease(ttl)
            await self.etcd.put(full_key, value_json, lease=lease)
            return True
        except Exception as e:
            logger.error(f"Error setting hash {hash_key}:{field}: {e}")
            return False
    
    async def get_hash(self, hash_key: str) -> Dict[str, Any]:
        """Get all hash fields."""
        try:
            prefix = f"{self.prefix}/{hash_key}/"
            result = {}
            async for value, metadata in self.etcd.get_prefix(prefix):
                try:
                    field_name = metadata.key.decode().replace(prefix, "")
                    result[field_name] = json.loads(value.decode())
                except:
                    pass
            return result
        except Exception as e:
            logger.error(f"Error getting hash {hash_key}: {e}")
            return {}
    
    async def acquire_lock(self, lock_key: str, owner: str, ttl: int = 10) -> bool:
        """Acquire distributed lock using Etcd lease."""
        try:
            full_key = f"{self.prefix}/lock/{lock_key}"
            
            # Try to acquire
            lease = await self.etcd.grant_lease(ttl)
            txn = self.etcd.transaction()
            txn.If(self.etcd.transactions.key(full_key).exists() == False)
            txn.Then(self.etcd.transactions.put(full_key, owner, lease=lease))
            txn.Else(self.etcd.transactions.get(full_key))
            
            result = await txn.commit()
            return result.succeeded
        
        except Exception as e:
            logger.error(f"Error acquiring lock {lock_key}: {e}")
            return False
    
    async def renew_lock(self, lock_key: str, owner: str, ttl: int = 10) -> bool:
        """Renew distributed lock."""
        try:
            # Etcd leases auto-renew, just check ownership
            current_owner = await self.get(f"lock/{lock_key}")
            return current_owner == owner
        except Exception as e:
            logger.error(f"Error renewing lock {lock_key}: {e}")
            return False
    
    async def release_lock(self, lock_key: str, owner: str) -> bool:
        """Release distributed lock."""
        try:
            full_key = f"{self.prefix}/lock/{lock_key}"
            current_owner = await self.get(f"lock/{lock_key}")
            if current_owner == owner:
                await self.etcd.delete(full_key)
                return True
            return False
        except Exception as e:
            logger.error(f"Error releasing lock {lock_key}: {e}")
            return False
    
    async def push_to_stream(self, stream_key: str, data: Dict) -> str:
        """Push message to stream."""
        try:
            full_key = f"{self.prefix}/stream/{stream_key}/{datetime.now().isoformat()}"
            data_json = json.dumps(data)
            await self.etcd.put(full_key, data_json)
            return full_key
        except Exception as e:
            logger.error(f"Error pushing to stream {stream_key}: {e}")
            return ""
    
    async def read_stream(self, stream_key: str, count: int = 10) -> List[Dict]:
        """Read messages from stream."""
        try:
            prefix = f"{self.prefix}/stream/{stream_key}/"
            result = []
            async for value, metadata in self.etcd.get_prefix(prefix):
                try:
                    result.append(json.loads(value.decode()))
                    if len(result) >= count:
                        break
                except:
                    pass
            return list(reversed(result))[:count]  # Return newest first
        except Exception as e:
            logger.error(f"Error reading stream {stream_key}: {e}")
            return []


def create_state_store(store_type: str = "redis", url: str = None) -> DistributedStateStore:
    """Factory function to create state store."""
    if store_type == "redis":
        return RedisStateStore(url or "redis://localhost:6379")
    elif store_type == "etcd":
        return EtcdStateStore(url or "etcd://localhost:2379")
    else:
        raise ValueError(f"Unknown state store type: {store_type}")
