"""
Cache Service Configuration for AI Options Trading System
Handles environment-based configuration for Redis and caching strategies
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class RedisConfig:
    """Redis connection configuration"""
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    ssl: bool = field(default_factory=lambda: os.getenv("REDIS_SSL", "false").lower() == "true")
    socket_timeout: int = field(default_factory=lambda: int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")))
    socket_connect_timeout: int = field(default_factory=lambda: int(os.getenv("REDIS_CONNECT_TIMEOUT", "5")))
    max_connections: int = field(default_factory=lambda: int(os.getenv("REDIS_MAX_CONNECTIONS", "100")))
    retry_on_timeout: bool = field(default_factory=lambda: os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true")
    health_check_interval: int = field(default_factory=lambda: int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")))
    
    # TTL settings (in seconds)
    ttl_stock_prices: int = field(default_factory=lambda: int(os.getenv("TTL_STOCK_PRICES", "300")))
    ttl_options_data: int = field(default_factory=lambda: int(os.getenv("TTL_OPTIONS_DATA", "300")))
    ttl_signals: int = field(default_factory=lambda: int(os.getenv("TTL_SIGNALS", "1800")))
    ttl_user_sessions: int = field(default_factory=lambda: int(os.getenv("TTL_USER_SESSIONS", "3600")))
    ttl_rate_limits: int = field(default_factory=lambda: int(os.getenv("TTL_RATE_LIMITS", "3600")))
    ttl_analytics: int = field(default_factory=lambda: int(os.getenv("TTL_ANALYTICS", "900")))
    ttl_market_status: int = field(default_factory=lambda: int(os.getenv("TTL_MARKET_STATUS", "60")))
    
    # Compression settings
    compress_threshold: int = field(default_factory=lambda: int(os.getenv("REDIS_COMPRESS_THRESHOLD", "1024")))
    compression_level: int = field(default_factory=lambda: int(os.getenv("REDIS_COMPRESSION_LEVEL", "6")))

@dataclass
class ServiceConfig:
    """Cache service configuration"""
    host: str = field(default_factory=lambda: os.getenv("CACHE_SERVICE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("CACHE_SERVICE_PORT", "8001")))
    workers: int = field(default_factory=lambda: int(os.getenv("CACHE_SERVICE_WORKERS", "4")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # API settings
    api_prefix: str = field(default_factory=lambda: os.getenv("API_PREFIX", "/api/v1"))
    enable_cors: bool = field(default_factory=lambda: os.getenv("ENABLE_CORS", "true").lower() == "true")
    cors_origins: str = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*"))
    
    # Rate limiting
    rate_limit_enabled: bool = field(default_factory=lambda: os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true")
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "1000")))
    rate_limit_window: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "3600")))
    
    # Monitoring
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true")
    health_check_enabled: bool = field(default_factory=lambda: os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true")

@dataclass
class DatabaseConfig:
    """Database configuration for cache service"""
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "trading_db"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "trading_admin"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    ssl_mode: str = field(default_factory=lambda: os.getenv("DB_SSL_MODE", "prefer"))
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "10")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "20")))
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.ssl_mode}"

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    file_enabled: bool = field(default_factory=lambda: os.getenv("LOG_FILE_ENABLED", "true").lower() == "true")
    file_path: str = field(default_factory=lambda: os.getenv("LOG_FILE_PATH", "/var/log/ai-ots/cache-service.log"))
    file_max_bytes: int = field(default_factory=lambda: int(os.getenv("LOG_FILE_MAX_BYTES", "10485760")))  # 10MB
    file_backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_FILE_BACKUP_COUNT", "5")))
    
    # Structured logging
    json_enabled: bool = field(default_factory=lambda: os.getenv("LOG_JSON_ENABLED", "false").lower() == "true")
    
    # External logging
    syslog_enabled: bool = field(default_factory=lambda: os.getenv("LOG_SYSLOG_ENABLED", "false").lower() == "true")
    syslog_host: str = field(default_factory=lambda: os.getenv("LOG_SYSLOG_HOST", "localhost"))
    syslog_port: int = field(default_factory=lambda: int(os.getenv("LOG_SYSLOG_PORT", "514")))

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    prometheus_enabled: bool = field(default_factory=lambda: os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true")
    prometheus_port: int = field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9090")))
    
    # Health check settings
    health_check_interval: int = field(default_factory=lambda: int(os.getenv("HEALTH_CHECK_INTERVAL", "30")))
    health_check_timeout: int = field(default_factory=lambda: int(os.getenv("HEALTH_CHECK_TIMEOUT", "5")))
    
    # Alerting
    alert_enabled: bool = field(default_factory=lambda: os.getenv("ALERT_ENABLED", "false").lower() == "true")
    alert_webhook_url: str = field(default_factory=lambda: os.getenv("ALERT_WEBHOOK_URL", ""))
    
    # Performance thresholds
    latency_threshold_ms: float = field(default_factory=lambda: float(os.getenv("LATENCY_THRESHOLD_MS", "100.0")))
    error_rate_threshold: float = field(default_factory=lambda: float(os.getenv("ERROR_RATE_THRESHOLD", "5.0")))
    memory_threshold_mb: int = field(default_factory=lambda: int(os.getenv("MEMORY_THRESHOLD_MB", "512")))

class CacheServiceConfig:
    """Main configuration class for cache service"""
    
    def __init__(self, environment: Environment = None):
        """Initialize configuration based on environment"""
        self.environment = environment or self._detect_environment()
        
        # Load configurations
        self.redis = RedisConfig()
        self.service = ServiceConfig()
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()
        self.monitoring = MonitoringConfig()
        
        # Apply environment-specific overrides
        self._apply_environment_overrides()
        
        # Validate configuration
        self._validate_config()
    
    def _detect_environment(self) -> Environment:
        """Detect environment from environment variables"""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            logging.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.DEVELOPMENT:
            self.service.debug = True
            self.logging.level = "DEBUG"
            self.redis.ttl_stock_prices = 60  # Shorter TTL for development
            
        elif self.environment == Environment.TESTING:
            self.service.debug = True
            self.logging.level = "DEBUG"
            self.redis.db = 1  # Use different Redis DB for testing
            
        elif self.environment == Environment.STAGING:
            self.service.debug = False
            self.logging.level = "INFO"
            self.monitoring.alert_enabled = True
            
        elif self.environment == Environment.PRODUCTION:
            self.service.debug = False
            self.logging.level = "WARNING"
            self.monitoring.alert_enabled = True
            self.redis.ssl = True  # Force SSL in production
            self.service.enable_cors = False  # Disable CORS in production
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate Redis configuration
        if not self.redis.host:
            errors.append("Redis host is required")
        
        if self.redis.port < 1 or self.redis.port > 65535:
            errors.append("Redis port must be between 1 and 65535")
        
        # Validate service configuration
        if self.service.port < 1 or self.service.port > 65535:
            errors.append("Service port must be between 1 and 65535")
        
        if self.service.workers < 1:
            errors.append("Number of workers must be at least 1")
        
        # Validate database configuration
        if not self.database.host:
            errors.append("Database host is required")
        
        if not self.database.name:
            errors.append("Database name is required")
        
        if not self.database.user:
            errors.append("Database user is required")
        
        # Validate TTL values
        if self.redis.ttl_stock_prices < 1:
            errors.append("Stock prices TTL must be at least 1 second")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "ssl": self.redis.ssl,
                "max_connections": self.redis.max_connections,
                "ttl_stock_prices": self.redis.ttl_stock_prices,
                "ttl_options_data": self.redis.ttl_options_data,
                "ttl_signals": self.redis.ttl_signals,
                "compress_threshold": self.redis.compress_threshold
            },
            "service": {
                "host": self.service.host,
                "port": self.service.port,
                "workers": self.service.workers,
                "debug": self.service.debug,
                "api_prefix": self.service.api_prefix,
                "enable_cors": self.service.enable_cors,
                "rate_limit_enabled": self.service.rate_limit_enabled
            },
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
                "pool_size": self.database.pool_size
            },
            "logging": {
                "level": self.logging.level,
                "file_enabled": self.logging.file_enabled,
                "json_enabled": self.logging.json_enabled
            },
            "monitoring": {
                "prometheus_enabled": self.monitoring.prometheus_enabled,
                "health_check_interval": self.monitoring.health_check_interval,
                "alert_enabled": self.monitoring.alert_enabled
            }
        }
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        import logging.handlers
        import json
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.logging.level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        if self.logging.json_enabled:
            # JSON formatter for structured logging
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    log_entry = {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno
                    }
                    
                    if record.exc_info:
                        log_entry["exception"] = self.formatException(record.exc_info)
                    
                    return json.dumps(log_entry)
            
            console_handler.setFormatter(JsonFormatter())
        else:
            # Standard formatter
            formatter = logging.Formatter(self.logging.format)
            console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # File handler
        if self.logging.file_enabled:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(self.logging.file_path)
            os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.file_max_bytes,
                backupCount=self.logging.file_backup_count
            )
            
            if self.logging.json_enabled:
                file_handler.setFormatter(JsonFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(self.logging.format))
            
            logger.addHandler(file_handler)
        
        # Syslog handler
        if self.logging.syslog_enabled:
            syslog_handler = logging.handlers.SysLogHandler(
                address=(self.logging.syslog_host, self.logging.syslog_port)
            )
            syslog_handler.setFormatter(logging.Formatter(
                "ai-ots-cache: %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(syslog_handler)
        
        logging.info(f"Logging configured for {self.environment.value} environment")

# Global configuration instance
config = None

def get_config() -> CacheServiceConfig:
    """Get global configuration instance"""
    global config
    if config is None:
        config = CacheServiceConfig()
    return config

def init_config(environment: Environment = None) -> CacheServiceConfig:
    """Initialize global configuration"""
    global config
    config = CacheServiceConfig(environment)
    config.setup_logging()
    return config

# Environment-specific configuration presets
DEVELOPMENT_CONFIG = {
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_DB": "0",
    "TTL_STOCK_PRICES": "60",
    "TTL_OPTIONS_DATA": "60",
    "DEBUG": "true",
    "LOG_LEVEL": "DEBUG"
}

TESTING_CONFIG = {
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_DB": "1",
    "TTL_STOCK_PRICES": "30",
    "TTL_OPTIONS_DATA": "30",
    "DEBUG": "true",
    "LOG_LEVEL": "DEBUG",
    "RATE_LIMIT_ENABLED": "false"
}

PRODUCTION_CONFIG = {
    "REDIS_SSL": "true",
    "DEBUG": "false",
    "LOG_LEVEL": "WARNING",
    "ENABLE_CORS": "false",
    "PROMETHEUS_ENABLED": "true",
    "ALERT_ENABLED": "true",
    "LOG_JSON_ENABLED": "true"
}

