"""
Async Utilities for INIDS

Provides utilities for transitioning from synchronous Flask to async pipelines.
Enables non-blocking I/O for Elasticsearch, external API calls, and detection processing.

Features:
- Async context manager helpers
- Thread pool executors for sync-to-async bridges
- Batch processing utilities
- Rate limiting helpers
"""

import asyncio
import logging
from typing import Callable, Any, List, Dict, TypeVar, Generic, Optional, Coroutine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, partial
from datetime import datetime, timezone, timedelta
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncExecutor:
    """Manages thread and process pools for async execution."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize executor pools.
        
        Args:
            max_workers: Max workers per pool
        """
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers // 2)
    
    async def run_in_thread(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Run blocking function in thread pool.
        
        Args:
            func: Blocking function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            partial(func, *args, **kwargs)
        )
    
    async def run_in_process(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Run function in process pool (for CPU-intensive work).
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_pool,
            partial(func, *args, **kwargs)
        )
    
    def shutdown(self):
        """Shutdown executor pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class AsyncBatchProcessor(Generic[T]):
    """Process items in batches asynchronously.
    
    Collects items and processes them in configurable batch sizes
    with optional delay for rate limiting.
    """
    
    def __init__(
        self,
        process_func: Callable[[List[T]], Coroutine],
        batch_size: int = 100,
        max_wait_seconds: float = 5.0,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """Initialize batch processor.
        
        Args:
            process_func: Async function to process batch
            batch_size: Items per batch
            max_wait_seconds: Max time to wait before processing incomplete batch
            on_error: Error handler callback
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds
        self.on_error = on_error or (lambda e: logger.error(f"Batch processing error: {e}"))
        
        self.batch = []
        self.last_process_time = datetime.now(timezone.utc)
        self.lock = asyncio.Lock()
        self.processing = False
    
    async def add(self, item: T) -> bool:
        """Add item to batch.
        
        Processes batch immediately if it reaches batch_size.
        
        Args:
            item: Item to add
            
        Returns:
            True if added successfully
        """
        async with self.lock:
            self.batch.append(item)
            
            if len(self.batch) >= self.batch_size:
                asyncio.create_task(self._process_batch())
            else:
                # Schedule processing after max_wait_seconds if not full
                asyncio.create_task(self._schedule_process())
        
        return True
    
    async def add_many(self, items: List[T]) -> bool:
        """Add multiple items to batch.
        
        Args:
            items: Items to add
            
        Returns:
            True if all added successfully
        """
        for item in items:
            await self.add(item)
        return True
    
    async def _schedule_process(self):
        """Schedule processing if batch has been waiting."""
        if len(self.batch) > 0 and not self.processing:
            time_since_last = (
                datetime.now(timezone.utc) - self.last_process_time
            ).total_seconds()
            
            if time_since_last >= self.max_wait_seconds:
                await self._process_batch()
            else:
                # Schedule check in remaining time
                remaining = self.max_wait_seconds - time_since_last + 0.1
                await asyncio.sleep(remaining)
                await self._schedule_process()
    
    async def _process_batch(self):
        """Process current batch."""
        async with self.lock:
            if len(self.batch) == 0 or self.processing:
                return
            
            self.processing = True
            batch_to_process = self.batch[:]
            self.batch = []
            self.last_process_time = datetime.now(timezone.utc)
        
        try:
            await self.process_func(batch_to_process)
        except Exception as e:
            self.on_error(e)
        finally:
            self.processing = False
    
    async def flush(self):
        """Process remaining items in batch."""
        async with self.lock:
            if len(self.batch) > 0 and not self.processing:
                await self._process_batch()


class RateLimiter:
    """Async rate limiter for API calls and operations.
    
    Implements token bucket algorithm.
    """
    
    def __init__(
        self,
        rate: float,  # Operations per second
        burst_size: int = 10
    ):
        """Initialize rate limiter.
        
        Args:
            rate: Operations per second
            burst_size: Max burst size
        """
        self.rate = rate
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = datetime.now(timezone.utc)
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1):
        """Acquire tokens from rate limiter.
        
        Waits if insufficient tokens available.
        
        Args:
            tokens: Number of tokens to acquire
        """
        async with self.lock:
            while self.tokens < tokens:
                # Calculate time to wait
                now = datetime.now(timezone.utc)
                elapsed = (now - self.last_update).total_seconds()
                self.tokens = min(
                    self.burst_size,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now
                
                if self.tokens < tokens:
                    # Wait for tokens to be available
                    wait_time = (tokens - self.tokens) / self.rate
                    await asyncio.sleep(wait_time)
                    now = datetime.now(timezone.utc)
                    elapsed = (now - self.last_update).total_seconds()
                    self.tokens = min(
                        self.burst_size,
                        self.tokens + elapsed * self.rate
                    )
                    self.last_update = now
            
            self.tokens -= tokens


async def gather_with_limit(
    *coros,
    limit: int = 10,
    return_exceptions: bool = False
) -> List[Any]:
    """Run multiple coroutines with concurrency limit.
    
    Args:
        *coros: Coroutines to run
        limit: Max concurrent coroutines
        return_exceptions: Whether to return exceptions
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(
        *[bounded_coro(coro) for coro in coros],
        return_exceptions=return_exceptions
    )


async def retry_async(
    func: Callable[..., Coroutine],
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Any:
    """Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Positional arguments
        max_retries: Max retry attempts
        initial_delay: Initial delay between retries
        backoff_factor: Multiplier for delay
        **kwargs: Keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_retries} attempts failed")
    
    raise last_exception


def async_to_sync(async_func: Callable[..., Coroutine]) -> Callable:
    """Convert async function to sync function.
    
    Useful for integrating async code into sync Flask routes.
    
    Args:
        async_func: Async function to wrap
        
    Returns:
        Sync wrapper function
    """
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # In async context, create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    async_func(*args, **kwargs)
                )
                return future.result()
        else:
            # In sync context, run normally
            return loop.run_until_complete(
                async_func(*args, **kwargs)
            )
    
    return wrapper


# Global executor instance
_executor = None


def get_async_executor(max_workers: int = 4) -> AsyncExecutor:
    """Get or create global async executor."""
    global _executor
    
    if _executor is None:
        _executor = AsyncExecutor(max_workers=max_workers)
    
    return _executor


def shutdown_async_executor():
    """Shutdown global async executor."""
    global _executor
    
    if _executor:
        _executor.shutdown()
        _executor = None
