"""
Cache Strategies for AI Options Trading System
Implements different caching patterns for optimal performance
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategy types"""
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"
    REFRESH_AHEAD = "refresh_ahead"

class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def average_latency_ms(self) -> float:
        total_ops = self.hits + self.misses + self.writes
        return (self.total_latency_ms / total_ops) if total_ops > 0 else 0.0

class CacheStrategyManager:
    """Manages different caching strategies for the trading system"""
    
    def __init__(self, redis_manager, database_manager=None):
        """Initialize cache strategy manager"""
        self.redis_manager = redis_manager
        self.database_manager = database_manager
        self.metrics = CacheMetrics()
        self.refresh_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache-refresh")
        self.write_behind_queue = asyncio.Queue() if asyncio.get_event_loop().is_running() else None
        self.refresh_tasks = {}
        
    def _record_hit(self, latency_ms: float = 0):
        """Record cache hit"""
        self.metrics.hits += 1
        self.metrics.total_latency_ms += latency_ms
    
    def _record_miss(self, latency_ms: float = 0):
        """Record cache miss"""
        self.metrics.misses += 1
        self.metrics.total_latency_ms += latency_ms
    
    def _record_write(self, latency_ms: float = 0):
        """Record cache write"""
        self.metrics.writes += 1
        self.metrics.total_latency_ms += latency_ms
    
    def _record_error(self):
        """Record cache error"""
        self.metrics.errors += 1

class StockPriceCacheStrategy(CacheStrategyManager):
    """Caching strategy for stock prices - high frequency, real-time data"""
    
    def __init__(self, redis_manager, database_manager=None):
        super().__init__(redis_manager, database_manager)
        self.strategy = CacheStrategy.WRITE_THROUGH
        self.ttl = 300  # 5 minutes
        
    def get_price(self, symbol: str) -> Optional[Dict]:
        """Get stock price with cache-aside pattern"""
        start_time = time.time()
        
        try:
            # Try cache first
            cached_data = self.redis_manager.get_stock_price(symbol)
            
            if cached_data:
                self._record_hit((time.time() - start_time) * 1000)
                return cached_data
            
            # Cache miss - fetch from database
            self._record_miss((time.time() - start_time) * 1000)
            
            if self.database_manager:
                db_data = self.database_manager.get_latest_stock_price(symbol)
                if db_data:
                    # Cache the data
                    self.redis_manager.cache_stock_price(symbol, db_data)
                    return db_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting stock price for {symbol}: {e}")
            self._record_error()
            return None
    
    def set_price(self, symbol: str, price_data: Dict) -> bool:
        """Set stock price with write-through pattern"""
        start_time = time.time()
        
        try:
            # Write to cache
            cache_success = self.redis_manager.cache_stock_price(symbol, price_data)
            
            # Write to database (write-through)
            db_success = True
            if self.database_manager:
                db_success = self.database_manager.insert_stock_price(price_data)
            
            self._record_write((time.time() - start_time) * 1000)
            return cache_success and db_success
            
        except Exception as e:
            logger.error(f"Error setting stock price for {symbol}: {e}")
            self._record_error()
            return False
    
    def bulk_update_prices(self, price_updates: List[Dict]) -> int:
        """Bulk update stock prices efficiently"""
        success_count = 0
        
        try:
            # Batch cache operations
            pipe = self.redis_manager.redis_client.pipeline()
            
            for price_data in price_updates:
                symbol = price_data.get("symbol")
                if symbol:
                    serialized_data = self.redis_manager._serialize_data(price_data)
                    key = self.redis_manager._generate_key("stock_price", symbol)
                    pipe.setex(key, self.ttl, serialized_data)
                    
                    # Update latest prices hash
                    pipe.hset("latest_prices", symbol, price_data.get("close_price", 0))
            
            # Execute batch
            results = pipe.execute()
            success_count = sum(1 for result in results if result)
            
            # Async database write (write-behind pattern for bulk operations)
            if self.database_manager and self.write_behind_queue:
                asyncio.create_task(self._write_behind_bulk_prices(price_updates))
            
            return success_count
            
        except Exception as e:
            logger.error(f"Error in bulk price update: {e}")
            self._record_error()
            return 0
    
    async def _write_behind_bulk_prices(self, price_updates: List[Dict]):
        """Write bulk price updates to database asynchronously"""
        try:
            if self.database_manager:
                await asyncio.get_event_loop().run_in_executor(
                    self.refresh_executor,
                    self.database_manager.bulk_insert_stock_prices,
                    price_updates
                )
        except Exception as e:
            logger.error(f"Error in write-behind bulk prices: {e}")

class OptionsCacheStrategy(CacheStrategyManager):
    """Caching strategy for options data - complex, structured data"""
    
    def __init__(self, redis_manager, database_manager=None):
        super().__init__(redis_manager, database_manager)
        self.strategy = CacheStrategy.CACHE_ASIDE
        self.ttl = 300  # 5 minutes
        
    def get_options_chain(self, symbol: str, expiry: str) -> Optional[List[Dict]]:
        """Get options chain with cache-aside pattern"""
        start_time = time.time()
        
        try:
            # Try cache first
            cached_data = self.redis_manager.get_options_chain(symbol, expiry)
            
            if cached_data:
                self._record_hit((time.time() - start_time) * 1000)
                return cached_data
            
            # Cache miss - fetch from database
            self._record_miss((time.time() - start_time) * 1000)
            
            if self.database_manager:
                db_data = self.database_manager.get_options_chain(symbol, expiry)
                if db_data:
                    # Cache the data
                    self.redis_manager.cache_options_chain(symbol, expiry, db_data)
                    return db_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            self._record_error()
            return None
    
    def set_options_chain(self, symbol: str, expiry: str, options_data: List[Dict]) -> bool:
        """Set options chain with write-around pattern (cache only)"""
        start_time = time.time()
        
        try:
            # Write to cache only (write-around for frequently changing data)
            cache_success = self.redis_manager.cache_options_chain(symbol, expiry, options_data)
            
            # Database writes are handled separately by data ingestion service
            self._record_write((time.time() - start_time) * 1000)
            return cache_success
            
        except Exception as e:
            logger.error(f"Error setting options chain for {symbol}: {e}")
            self._record_error()
            return False
    
    def get_options_summary(self, symbol: str) -> Optional[Dict]:
        """Get options summary with refresh-ahead pattern"""
        start_time = time.time()
        
        try:
            key = f"options_summary:{symbol}"
            cached_data = self.redis_manager.get_analytics(key)
            
            if cached_data:
                self._record_hit((time.time() - start_time) * 1000)
                
                # Check if refresh is needed (refresh-ahead)
                updated_at = cached_data.get("updated_at")
                if updated_at:
                    update_time = datetime.fromisoformat(updated_at)
                    if datetime.now() - update_time > timedelta(minutes=2):
                        # Trigger background refresh
                        self._schedule_refresh(symbol, "options_summary")
                
                return cached_data
            
            # Cache miss - fetch and cache
            self._record_miss((time.time() - start_time) * 1000)
            return self._refresh_options_summary(symbol)
            
        except Exception as e:
            logger.error(f"Error getting options summary for {symbol}: {e}")
            self._record_error()
            return None
    
    def _refresh_options_summary(self, symbol: str) -> Optional[Dict]:
        """Refresh options summary from database"""
        try:
            if self.database_manager:
                summary_data = self.database_manager.get_options_summary(symbol)
                if summary_data:
                    key = f"options_summary:{symbol}"
                    self.redis_manager.cache_analytics(key, summary_data, self.ttl)
                    return summary_data
            return None
            
        except Exception as e:
            logger.error(f"Error refreshing options summary for {symbol}: {e}")
            return None
    
    def _schedule_refresh(self, symbol: str, data_type: str):
        """Schedule background refresh of data"""
        task_key = f"{data_type}:{symbol}"
        
        # Avoid duplicate refresh tasks
        if task_key not in self.refresh_tasks:
            future = self.refresh_executor.submit(self._background_refresh, symbol, data_type)
            self.refresh_tasks[task_key] = future
            
            # Clean up completed tasks
            def cleanup(fut):
                self.refresh_tasks.pop(task_key, None)
            
            future.add_done_callback(cleanup)
    
    def _background_refresh(self, symbol: str, data_type: str):
        """Background refresh of cached data"""
        try:
            if data_type == "options_summary":
                self._refresh_options_summary(symbol)
            
            logger.debug(f"Background refresh completed for {data_type}:{symbol}")
            
        except Exception as e:
            logger.error(f"Error in background refresh for {data_type}:{symbol}: {e}")

class SignalCacheStrategy(CacheStrategyManager):
    """Caching strategy for trading signals - medium frequency, important data"""
    
    def __init__(self, redis_manager, database_manager=None):
        super().__init__(redis_manager, database_manager)
        self.strategy = CacheStrategy.WRITE_THROUGH
        self.ttl = 1800  # 30 minutes
        
    def get_signal(self, signal_id: str) -> Optional[Dict]:
        """Get signal with read-through pattern"""
        start_time = time.time()
        
        try:
            # Try cache first
            cached_data = self.redis_manager.get_signal(signal_id)
            
            if cached_data:
                self._record_hit((time.time() - start_time) * 1000)
                return cached_data
            
            # Cache miss - fetch from database (read-through)
            self._record_miss((time.time() - start_time) * 1000)
            
            if self.database_manager:
                db_data = self.database_manager.get_signal(signal_id)
                if db_data:
                    # Cache the data
                    self.redis_manager.cache_signal(signal_id, db_data)
                    return db_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting signal {signal_id}: {e}")
            self._record_error()
            return None
    
    def set_signal(self, signal_id: str, signal_data: Dict) -> bool:
        """Set signal with write-through pattern"""
        start_time = time.time()
        
        try:
            # Write to cache
            cache_success = self.redis_manager.cache_signal(signal_id, signal_data)
            
            # Write to database (write-through)
            db_success = True
            if self.database_manager:
                db_success = self.database_manager.insert_signal(signal_data)
            
            self._record_write((time.time() - start_time) * 1000)
            return cache_success and db_success
            
        except Exception as e:
            logger.error(f"Error setting signal {signal_id}: {e}")
            self._record_error()
            return False
    
    def get_active_signals(self, symbol: str = None) -> List[Dict]:
        """Get active signals with smart caching"""
        try:
            if symbol:
                # Get signals for specific symbol
                return self.redis_manager.get_signals_for_symbol(symbol, limit=20)
            else:
                # Get all active signals (cached aggregate)
                cached_data = self.redis_manager.get_analytics("active_signals")
                
                if cached_data:
                    return cached_data.get("signals", [])
                
                # Fetch from database and cache
                if self.database_manager:
                    active_signals = self.database_manager.get_active_signals()
                    self.redis_manager.cache_analytics(
                        "active_signals",
                        {"signals": active_signals, "updated_at": datetime.now().isoformat()},
                        ttl=300  # 5 minutes for active signals
                    )
                    return active_signals
                
                return []
                
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            self._record_error()
            return []

class SessionCacheStrategy(CacheStrategyManager):
    """Caching strategy for user sessions - security-sensitive data"""
    
    def __init__(self, redis_manager):
        super().__init__(redis_manager)
        self.strategy = CacheStrategy.CACHE_ASIDE
        self.ttl = 3600  # 1 hour
        
    def create_session(self, user_id: str, session_data: Dict) -> Optional[str]:
        """Create session with cache-only storage"""
        try:
            session_id = self.redis_manager.create_session(user_id, session_data)
            if session_id:
                self._record_write()
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session for user {user_id}: {e}")
            self._record_error()
            return None
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session with automatic refresh"""
        start_time = time.time()
        
        try:
            session_data = self.redis_manager.get_session(session_id)
            
            if session_data:
                self._record_hit((time.time() - start_time) * 1000)
                return session_data
            else:
                self._record_miss((time.time() - start_time) * 1000)
                return None
                
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            self._record_error()
            return None
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user"""
        try:
            pattern = f"session:*"
            return self.redis_manager.invalidate_pattern(pattern)
            
        except Exception as e:
            logger.error(f"Error invalidating sessions for user {user_id}: {e}")
            self._record_error()
            return 0

class AnalyticsCacheStrategy(CacheStrategyManager):
    """Caching strategy for analytics data - computed, expensive data"""
    
    def __init__(self, redis_manager, database_manager=None):
        super().__init__(redis_manager, database_manager)
        self.strategy = CacheStrategy.REFRESH_AHEAD
        self.ttl = 900  # 15 minutes
        
    def get_analytics(self, key: str, compute_func: Callable = None) -> Optional[Dict]:
        """Get analytics with refresh-ahead pattern"""
        start_time = time.time()
        
        try:
            cached_data = self.redis_manager.get_analytics(key)
            
            if cached_data:
                self._record_hit((time.time() - start_time) * 1000)
                
                # Check if refresh is needed
                updated_at = cached_data.get("updated_at")
                if updated_at:
                    update_time = datetime.fromisoformat(updated_at)
                    if datetime.now() - update_time > timedelta(minutes=10):
                        # Trigger background refresh
                        if compute_func:
                            self._schedule_analytics_refresh(key, compute_func)
                
                return cached_data
            
            # Cache miss - compute and cache
            self._record_miss((time.time() - start_time) * 1000)
            
            if compute_func:
                computed_data = compute_func()
                if computed_data:
                    computed_data["updated_at"] = datetime.now().isoformat()
                    self.redis_manager.cache_analytics(key, computed_data, self.ttl)
                    return computed_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting analytics {key}: {e}")
            self._record_error()
            return None
    
    def _schedule_analytics_refresh(self, key: str, compute_func: Callable):
        """Schedule background refresh of analytics"""
        task_key = f"analytics:{key}"
        
        if task_key not in self.refresh_tasks:
            future = self.refresh_executor.submit(self._background_analytics_refresh, key, compute_func)
            self.refresh_tasks[task_key] = future
            
            def cleanup(fut):
                self.refresh_tasks.pop(task_key, None)
            
            future.add_done_callback(cleanup)
    
    def _background_analytics_refresh(self, key: str, compute_func: Callable):
        """Background refresh of analytics data"""
        try:
            computed_data = compute_func()
            if computed_data:
                computed_data["updated_at"] = datetime.now().isoformat()
                self.redis_manager.cache_analytics(key, computed_data, self.ttl)
            
            logger.debug(f"Background analytics refresh completed for {key}")
            
        except Exception as e:
            logger.error(f"Error in background analytics refresh for {key}: {e}")

class CacheStrategyFactory:
    """Factory for creating cache strategies"""
    
    @staticmethod
    def create_stock_price_strategy(redis_manager, database_manager=None):
        """Create stock price caching strategy"""
        return StockPriceCacheStrategy(redis_manager, database_manager)
    
    @staticmethod
    def create_options_strategy(redis_manager, database_manager=None):
        """Create options caching strategy"""
        return OptionsCacheStrategy(redis_manager, database_manager)
    
    @staticmethod
    def create_signal_strategy(redis_manager, database_manager=None):
        """Create signal caching strategy"""
        return SignalCacheStrategy(redis_manager, database_manager)
    
    @staticmethod
    def create_session_strategy(redis_manager):
        """Create session caching strategy"""
        return SessionCacheStrategy(redis_manager)
    
    @staticmethod
    def create_analytics_strategy(redis_manager, database_manager=None):
        """Create analytics caching strategy"""
        return AnalyticsCacheStrategy(redis_manager, database_manager)

# Unified cache manager
class UnifiedCacheManager:
    """Unified cache manager that combines all strategies"""
    
    def __init__(self, redis_manager, database_manager=None):
        """Initialize unified cache manager"""
        self.redis_manager = redis_manager
        self.database_manager = database_manager
        
        # Initialize strategies
        self.stock_prices = CacheStrategyFactory.create_stock_price_strategy(
            redis_manager, database_manager
        )
        self.options = CacheStrategyFactory.create_options_strategy(
            redis_manager, database_manager
        )
        self.signals = CacheStrategyFactory.create_signal_strategy(
            redis_manager, database_manager
        )
        self.sessions = CacheStrategyFactory.create_session_strategy(redis_manager)
        self.analytics = CacheStrategyFactory.create_analytics_strategy(
            redis_manager, database_manager
        )
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall cache metrics"""
        strategies = [
            ("stock_prices", self.stock_prices),
            ("options", self.options),
            ("signals", self.signals),
            ("sessions", self.sessions),
            ("analytics", self.analytics)
        ]
        
        total_metrics = CacheMetrics()
        strategy_metrics = {}
        
        for name, strategy in strategies:
            metrics = strategy.metrics
            strategy_metrics[name] = {
                "hits": metrics.hits,
                "misses": metrics.misses,
                "writes": metrics.writes,
                "errors": metrics.errors,
                "hit_ratio": metrics.hit_ratio,
                "average_latency_ms": metrics.average_latency_ms
            }
            
            # Aggregate totals
            total_metrics.hits += metrics.hits
            total_metrics.misses += metrics.misses
            total_metrics.writes += metrics.writes
            total_metrics.errors += metrics.errors
            total_metrics.total_latency_ms += metrics.total_latency_ms
        
        return {
            "overall": {
                "total_hits": total_metrics.hits,
                "total_misses": total_metrics.misses,
                "total_writes": total_metrics.writes,
                "total_errors": total_metrics.errors,
                "overall_hit_ratio": total_metrics.hit_ratio,
                "overall_average_latency_ms": total_metrics.average_latency_ms
            },
            "by_strategy": strategy_metrics,
            "redis_stats": self.redis_manager.get_cache_stats()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        redis_health = self.redis_manager.health_check()
        cache_metrics = self.get_overall_metrics()
        
        return {
            "redis_health": redis_health,
            "cache_metrics": cache_metrics,
            "status": "healthy" if redis_health.get("status") == "healthy" else "degraded"
        }

