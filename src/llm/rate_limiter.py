import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Any
from src.utils.logger import setup_logger
from src.config.settings import settings

logger = setup_logger(__name__)

class RateLimitRouter:
    """Rate limiter for LLM API calls with load balancing"""
    
    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        self.lock = asyncio.Lock()
        self.total_calls = 0
        logger.info(f"RateLimitRouter initialized with {max_requests_per_minute} requests/minute")
    
    async def call_with_limit(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with rate limiting"""
        async with self.lock:
            now = datetime.now()
            
            # Remove requests older than 1 minute
            while self.request_times and self.request_times[0] < now - timedelta(minutes=1):
                self.request_times.popleft()
            
            # Check if we need to wait
            if len(self.request_times) >= self.max_requests:
                sleep_time = 60 - (now - self.request_times[0]).total_seconds()
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                now = datetime.now()
            
            # Record this request
            self.request_times.append(now)
            self.total_calls += 1
            
        # Execute the function
        logger.debug(f"Executing LLM call #{self.total_calls}")
        
        # Handle both sync and async functions
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        return {
            "total_calls": self.total_calls,
            "requests_in_last_minute": len(self.request_times),
            "max_requests_per_minute": self.max_requests
        }

# Global rate limiter instance
rate_limiter = RateLimitRouter(max_requests_per_minute=settings.MAX_REQUESTS_PER_MINUTE)
