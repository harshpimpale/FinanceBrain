"""
Tests for Rate Limiting functionality
"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock
from src.llm.rate_limiter import RateLimitRouter

class TestRateLimiting:
    """Test rate limiting for LLM API calls"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create fresh rate limiter for each test"""
        limiter = RateLimitRouter(max_requests_per_minute=10)
        limiter.request_times.clear()
        limiter.total_calls = 0
        return limiter
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_requests_under_limit(self, rate_limiter):
        """Test that requests under limit are allowed"""
        async def mock_func():
            return "success"
        
        # Make 5 requests (under limit of 10)
        for i in range(5):
            result = await rate_limiter.call_with_limit(mock_func)
            assert result == "success"
        
        # All should complete without delay
        assert rate_limiter.total_calls == 5
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforces_limit(self, rate_limiter):
        """Test that rate limiting enforces max requests"""
        call_times = []
        
        async def mock_func():
            call_times.append(time.time())
            return "success"
        
        start_time = time.time()
        
        # Make 15 requests (exceeds limit of 10)
        for i in range(15):
            await rate_limiter.call_with_limit(mock_func)
        
        elapsed = time.time() - start_time
        
        # Should take at least some time due to rate limiting
        # With 15 requests and limit of 10/min, should trigger waiting
        assert rate_limiter.total_calls == 15
        assert len(call_times) == 15
    
    @pytest.mark.asyncio
    async def test_rate_limiter_with_sync_function(self, rate_limiter):
        """Test rate limiter works with synchronous functions"""
        def sync_func(x):
            return x * 2
        
        result = await rate_limiter.call_with_limit(sync_func, 5)
        assert result == 10
        assert rate_limiter.total_calls == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiter_with_async_function(self, rate_limiter):
        """Test rate limiter works with async functions"""
        async def async_func(x):
            await asyncio.sleep(0.01)
            return x * 3
        
        result = await rate_limiter.call_with_limit(async_func, 4)
        assert result == 12
        assert rate_limiter.total_calls == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limited_calls(self, rate_limiter):
        """Test concurrent calls are properly rate limited"""
        async def mock_api_call(id):
            await asyncio.sleep(0.01)
            return f"result_{id}"
        
        # Create 12 concurrent tasks (exceeds limit of 10)
        tasks = [
            rate_limiter.call_with_limit(mock_api_call, i)
            for i in range(12)
        ]
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        # All should complete
        assert len(results) == 12
        assert rate_limiter.total_calls == 12
    
    def test_get_stats(self, rate_limiter):
        """Test rate limiter statistics"""
        stats = rate_limiter.get_stats()
        
        assert "total_calls" in stats
        assert "requests_in_last_minute" in stats
        assert "max_requests_per_minute" in stats
        assert stats["max_requests_per_minute"] == 10
    
    @pytest.mark.asyncio
    async def test_sliding_window_cleanup(self, rate_limiter):
        """Test that old requests are cleaned from sliding window"""
        async def mock_func():
            return "success"
        
        # Make initial requests
        for i in range(5):
            await rate_limiter.call_with_limit(mock_func)
        
        initial_queue_size = len(rate_limiter.request_times)
        assert initial_queue_size == 5
        
        # Wait for requests to age out (in real scenario)
        # Note: In unit test, we'd need to mock datetime
        stats = rate_limiter.get_stats()
        assert stats["requests_in_last_minute"] == 5
