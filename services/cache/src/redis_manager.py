"""
Redis Cache Manager for AI Options Trading System
Handles caching of real-time market data, signals, and session management
"""

import redis
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from decimal import Decimal
import hashlib
import pickle
import gzip

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for Redis cache"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    max_connections: int = 100
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # TTL settings (in seconds)
    ttl_stock_prices: int = 300      # 5 minutes
    ttl_options_data: int = 300      # 5 minutes
    ttl_signals: int = 1800          # 30 minutes
    ttl_user_sessions: int = 3600    # 1 hour
    ttl_rate_limits: int = 3600      # 1 hour
    ttl_analytics: int = 900         # 15 minutes
    ttl_market_status: int = 60      # 1 minute
    
    # Compression settings
    compress_threshold: int = 1024   # Compress data larger than 1KB
    compression_level: int = 6       # gzip compression level

class RedisManager:
    """High-performance Redis cache manager for trading system"""
    
    def __init__(self, config: CacheConfig):
        """Initialize Redis manager with configuration"""
        self.config = config
        self.redis_client = None
        self.connection_pool = None
        self._setup_connection()
        
    def _setup_connection(self):
        """Set up Redis connection with connection pooling"""
        try:
            # Create connection pool
            self.connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                ssl=self.config.ssl,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression"""
        try:
            # Handle Decimal objects
            if isinstance(data, dict):
                data = self._convert_decimals(data)
            elif isinstance(data, list):
                data = [self._convert_decimals(item) if isinstance(item, dict) else item for item in data]
            
            # Serialize to JSON first
            json_data = json.dumps(data, default=str).encode('utf-8')
            
            # Compress if data is large enough
            if len(json_data) > self.config.compress_threshold:
                compressed_data = gzip.compress(json_data, compresslevel=self.config.compression_level)
                # Add compression marker
                return b'GZIP:' + compressed_data
            
            return json_data
            
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            # Fallback to pickle for complex objects
            return b'PICKLE:' + pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with decompression support"""
        try:
            if data.startswith(b'GZIP:'):
                # Decompress gzipped data
                compressed_data = data[5:]  # Remove 'GZIP:' prefix
                json_data = gzip.decompress(compressed_data)
                return json.loads(json_data.decode('utf-8'))
            elif data.startswith(b'PICKLE:'):
                # Unpickle data
                pickled_data = data[7:]  # Remove 'PICKLE:' prefix
                return pickle.loads(pickled_data)
            else:
                # Regular JSON data
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            return None
    
    def _convert_decimals(self, obj: Dict) -> Dict:
        """Convert Decimal objects to float for JSON serialization"""
        if isinstance(obj, dict):
            return {k: float(v) if isinstance(v, Decimal) else v for k, v in obj.items()}
        return obj
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate cache key with consistent format"""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis health and return status"""
        try:
            start_time = time.time()
            
            # Test basic operations
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.now().isoformat()}
            
            # Set and get test
            self.redis_client.setex(test_key, 10, json.dumps(test_value))
            retrieved = self.redis_client.get(test_key)
            
            if retrieved:
                self.redis_client.delete(test_key)
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Get Redis info
            info = self.redis_client.info()
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_ratio": self._calculate_hit_ratio(info),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _calculate_hit_ratio(self, info: Dict) -> float:
        """Calculate cache hit ratio"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return round((hits / total) * 100, 2)
    
    # Stock Price Caching
    def cache_stock_price(self, symbol: str, price_data: Dict) -> bool:
        """Cache latest stock price data"""
        try:
            key = self._generate_key("stock_price", symbol)
            serialized_data = self._serialize_data(price_data)
            
            result = self.redis_client.setex(
                key, 
                self.config.ttl_stock_prices, 
                serialized_data
            )
            
            # Also update the latest prices hash
            self.redis_client.hset(
                "latest_prices", 
                symbol, 
                price_data.get("close_price", 0)
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to cache stock price for {symbol}: {e}")
            return False
    
    def get_stock_price(self, symbol: str) -> Optional[Dict]:
        """Get cached stock price data"""
        try:
            key = self._generate_key("stock_price", symbol)
            data = self.redis_client.get(key)
            
            if data:
                return self._deserialize_data(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get stock price for {symbol}: {e}")
            return None
    
    def get_latest_prices(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get latest prices for multiple symbols"""
        try:
            if symbols:
                prices = self.redis_client.hmget("latest_prices", *symbols)
                return {
                    symbol: float(price) if price else None 
                    for symbol, price in zip(symbols, prices)
                }
            else:
                all_prices = self.redis_client.hgetall("latest_prices")
                return {
                    symbol.decode(): float(price) 
                    for symbol, price in all_prices.items()
                }
                
        except Exception as e:
            logger.error(f"Failed to get latest prices: {e}")
            return {}
    
    # Options Data Caching
    def cache_options_chain(self, symbol: str, expiry: str, options_data: List[Dict]) -> bool:
        """Cache options chain data"""
        try:
            key = self._generate_key("options_chain", symbol, expiry)
            serialized_data = self._serialize_data(options_data)
            
            result = self.redis_client.setex(
                key,
                self.config.ttl_options_data,
                serialized_data
            )
            
            # Update options summary
            self._update_options_summary(symbol, options_data)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to cache options chain for {symbol}: {e}")
            return False
    
    def get_options_chain(self, symbol: str, expiry: str) -> Optional[List[Dict]]:
        """Get cached options chain data"""
        try:
            key = self._generate_key("options_chain", symbol, expiry)
            data = self.redis_client.get(key)
            
            if data:
                return self._deserialize_data(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get options chain for {symbol}: {e}")
            return None
    
    def _update_options_summary(self, symbol: str, options_data: List[Dict]):
        """Update options summary statistics"""
        try:
            call_volume = sum(opt.get("volume", 0) for opt in options_data if opt.get("option_type") == "CALL")
            put_volume = sum(opt.get("volume", 0) for opt in options_data if opt.get("option_type") == "PUT")
            
            put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
            
            summary = {
                "call_volume": call_volume,
                "put_volume": put_volume,
                "put_call_ratio": put_call_ratio,
                "total_volume": call_volume + put_volume,
                "updated_at": datetime.now().isoformat()
            }
            
            key = self._generate_key("options_summary", symbol)
            self.redis_client.setex(
                key,
                self.config.ttl_options_data,
                self._serialize_data(summary)
            )
            
        except Exception as e:
            logger.error(f"Failed to update options summary for {symbol}: {e}")
    
    # Signal Caching
    def cache_signal(self, signal_id: str, signal_data: Dict) -> bool:
        """Cache trading signal"""
        try:
            key = self._generate_key("signal", signal_id)
            serialized_data = self._serialize_data(signal_data)
            
            result = self.redis_client.setex(
                key,
                self.config.ttl_signals,
                serialized_data
            )
            
            # Add to symbol-specific signal list
            symbol = signal_data.get("symbol")
            if symbol:
                symbol_key = self._generate_key("signals", symbol)
                self.redis_client.lpush(symbol_key, signal_id)
                self.redis_client.ltrim(symbol_key, 0, 99)  # Keep last 100 signals
                self.redis_client.expire(symbol_key, self.config.ttl_signals)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to cache signal {signal_id}: {e}")
            return False
    
    def get_signal(self, signal_id: str) -> Optional[Dict]:
        """Get cached signal"""
        try:
            key = self._generate_key("signal", signal_id)
            data = self.redis_client.get(key)
            
            if data:
                return self._deserialize_data(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get signal {signal_id}: {e}")
            return None
    
    def get_signals_for_symbol(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent signals for a symbol"""
        try:
            symbol_key = self._generate_key("signals", symbol)
            signal_ids = self.redis_client.lrange(symbol_key, 0, limit - 1)
            
            signals = []
            for signal_id in signal_ids:
                signal_data = self.get_signal(signal_id.decode())
                if signal_data:
                    signals.append(signal_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get signals for {symbol}: {e}")
            return []
    
    # Session Management
    def create_session(self, user_id: str, session_data: Dict) -> str:
        """Create user session"""
        try:
            session_id = hashlib.sha256(f"{user_id}:{time.time()}".encode()).hexdigest()
            key = self._generate_key("session", session_id)
            
            session_data.update({
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            })
            
            serialized_data = self._serialize_data(session_data)
            
            result = self.redis_client.setex(
                key,
                self.config.ttl_user_sessions,
                serialized_data
            )
            
            if result:
                # Add to user sessions set
                user_sessions_key = self._generate_key("user_sessions", user_id)
                self.redis_client.sadd(user_sessions_key, session_id)
                self.redis_client.expire(user_sessions_key, self.config.ttl_user_sessions)
                
                return session_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        try:
            key = self._generate_key("session", session_id)
            data = self.redis_client.get(key)
            
            if data:
                session_data = self._deserialize_data(data)
                
                # Update last activity
                session_data["last_activity"] = datetime.now().isoformat()
                self.redis_client.setex(
                    key,
                    self.config.ttl_user_sessions,
                    self._serialize_data(session_data)
                )
                
                return session_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        try:
            key = self._generate_key("session", session_id)
            result = self.redis_client.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    # Rate Limiting
    def check_rate_limit(self, identifier: str, limit: int, window: int) -> Dict[str, Any]:
        """Check rate limit using sliding window"""
        try:
            key = self._generate_key("rate_limit", identifier)
            current_time = int(time.time())
            
            # Use sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, current_time - window)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(key, window)
            
            results = pipe.execute()
            current_count = results[1]
            
            remaining = max(0, limit - current_count)
            reset_time = current_time + window
            
            return {
                "allowed": current_count < limit,
                "limit": limit,
                "remaining": remaining,
                "reset_time": reset_time,
                "current_count": current_count
            }
            
        except Exception as e:
            logger.error(f"Failed to check rate limit for {identifier}: {e}")
            return {
                "allowed": True,  # Allow on error
                "limit": limit,
                "remaining": limit,
                "reset_time": int(time.time()) + window,
                "current_count": 0
            }
    
    # Analytics Caching
    def cache_analytics(self, key_suffix: str, data: Dict, ttl: int = None) -> bool:
        """Cache analytics data"""
        try:
            key = self._generate_key("analytics", key_suffix)
            serialized_data = self._serialize_data(data)
            
            ttl = ttl or self.config.ttl_analytics
            
            result = self.redis_client.setex(key, ttl, serialized_data)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to cache analytics {key_suffix}: {e}")
            return False
    
    def get_analytics(self, key_suffix: str) -> Optional[Dict]:
        """Get cached analytics data"""
        try:
            key = self._generate_key("analytics", key_suffix)
            data = self.redis_client.get(key)
            
            if data:
                return self._deserialize_data(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get analytics {key_suffix}: {e}")
            return None
    
    # Market Status
    def set_market_status(self, status: Dict) -> bool:
        """Set market status"""
        try:
            key = "market_status"
            serialized_data = self._serialize_data(status)
            
            result = self.redis_client.setex(
                key,
                self.config.ttl_market_status,
                serialized_data
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set market status: {e}")
            return False
    
    def get_market_status(self) -> Optional[Dict]:
        """Get market status"""
        try:
            data = self.redis_client.get("market_status")
            
            if data:
                return self._deserialize_data(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return None
    
    # Utility Methods
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to invalidate pattern {pattern}: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.redis_client.info()
            
            # Count keys by prefix
            key_counts = {}
            for prefix in ["stock_price", "options_chain", "signal", "session", "analytics"]:
                pattern = f"{prefix}:*"
                keys = self.redis_client.keys(pattern)
                key_counts[prefix] = len(keys)
            
            return {
                "total_keys": info.get("db0", {}).get("keys", 0),
                "memory_usage": info.get("used_memory_human", "unknown"),
                "hit_ratio": self._calculate_hit_ratio(info),
                "connected_clients": info.get("connected_clients", 0),
                "key_counts": key_counts,
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def flush_cache(self, confirm: bool = False) -> bool:
        """Flush all cache data (use with caution)"""
        if not confirm:
            logger.warning("Cache flush requires confirmation")
            return False
        
        try:
            result = self.redis_client.flushdb()
            logger.warning("Cache flushed successfully")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")
            return False
    
    def close(self):
        """Close Redis connection"""
        try:
            if self.connection_pool:
                self.connection_pool.disconnect()
            logger.info("Redis connection closed")
            
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

# Factory function for easy initialization
def create_redis_manager(
    host: str = "localhost",
    port: int = 6379,
    password: str = None,
    **kwargs
) -> RedisManager:
    """Create Redis manager with default configuration"""
    config = CacheConfig(
        host=host,
        port=port,
        password=password,
        **kwargs
    )
    
    return RedisManager(config)

