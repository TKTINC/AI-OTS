"""
Centralized Logging Configuration for AI Options Trading System
Provides structured logging with different levels and outputs
"""

import logging
import logging.config
import os
import json
from datetime import datetime
from typing import Dict, Any
import sys

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d] %(funcName)s(): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(funcName)s %(message)s"
        },
        "access": {
            "format": "%(asctime)s [ACCESS] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "audit": {
            "format": "%(asctime)s [AUDIT] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "console_detailed": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        },
        "file_info": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": "/var/log/ai-ots/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "file_error": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "/var/log/ai-ots/error.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "file_json": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "/var/log/ai-ots/app.json",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "file_access": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "access",
            "filename": "/var/log/ai-ots/access.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "encoding": "utf8"
        },
        "file_audit": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "audit",
            "filename": "/var/log/ai-ots/audit.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "encoding": "utf8"
        },
        "syslog": {
            "class": "logging.handlers.SysLogHandler",
            "level": "INFO",
            "formatter": "standard",
            "address": ("localhost", 514),
            "facility": "local0"
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file_info", "file_error"],
            "level": "INFO",
            "propagate": False
        },
        "ai_ots": {
            "handlers": ["console", "file_info", "file_error", "file_json"],
            "level": "INFO",
            "propagate": False
        },
        "ai_ots.api_gateway": {
            "handlers": ["console", "file_info", "file_error"],
            "level": "INFO",
            "propagate": False
        },
        "ai_ots.cache": {
            "handlers": ["console", "file_info", "file_error"],
            "level": "INFO",
            "propagate": False
        },
        "ai_ots.data_ingestion": {
            "handlers": ["console", "file_info", "file_error"],
            "level": "INFO",
            "propagate": False
        },
        "ai_ots.analytics": {
            "handlers": ["console", "file_info", "file_error"],
            "level": "INFO",
            "propagate": False
        },
        "ai_ots.access": {
            "handlers": ["file_access"],
            "level": "INFO",
            "propagate": False
        },
        "ai_ots.audit": {
            "handlers": ["file_audit", "syslog"],
            "level": "INFO",
            "propagate": False
        },
        "werkzeug": {
            "handlers": ["file_access"],
            "level": "INFO",
            "propagate": False
        },
        "urllib3": {
            "handlers": ["file_info"],
            "level": "WARNING",
            "propagate": False
        },
        "requests": {
            "handlers": ["file_info"],
            "level": "WARNING",
            "propagate": False
        }
    }
}

class StructuredLogger:
    """Structured logger with additional context"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set logging context"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context"""
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

class AccessLogger:
    """HTTP access logger"""
    
    def __init__(self):
        self.logger = logging.getLogger("ai_ots.access")
    
    def log_request(self, request_data: Dict[str, Any]):
        """Log HTTP request"""
        message = (
            f"{request_data.get('remote_addr', '-')} "
            f"{request_data.get('method', '-')} "
            f"{request_data.get('path', '-')} "
            f"{request_data.get('status_code', '-')} "
            f"{request_data.get('response_size', '-')} "
            f"{request_data.get('response_time', '-')}ms "
            f"\"{request_data.get('user_agent', '-')}\""
        )
        self.logger.info(message)

class AuditLogger:
    """Security and compliance audit logger"""
    
    def __init__(self):
        self.logger = logging.getLogger("ai_ots.audit")
    
    def log_event(self, event_type: str, user_id: str = None, details: Dict[str, Any] = None):
        """Log audit event"""
        audit_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details or {}
        }
        
        message = json.dumps(audit_data)
        self.logger.info(message)
    
    def log_authentication(self, user_id: str, success: bool, ip_address: str = None):
        """Log authentication attempt"""
        self.log_event(
            "authentication",
            user_id=user_id,
            details={
                "success": success,
                "ip_address": ip_address
            }
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str):
        """Log data access"""
        self.log_event(
            "data_access",
            user_id=user_id,
            details={
                "resource": resource,
                "action": action
            }
        )
    
    def log_configuration_change(self, user_id: str, component: str, changes: Dict[str, Any]):
        """Log configuration changes"""
        self.log_event(
            "configuration_change",
            user_id=user_id,
            details={
                "component": component,
                "changes": changes
            }
        )

class PerformanceLogger:
    """Performance metrics logger"""
    
    def __init__(self):
        self.logger = logging.getLogger("ai_ots.performance")
    
    def log_timing(self, operation: str, duration: float, **kwargs):
        """Log operation timing"""
        self.logger.info(
            f"TIMING: {operation} completed in {duration:.3f}s",
            extra={"operation": operation, "duration": duration, **kwargs}
        )
    
    def log_throughput(self, operation: str, count: int, duration: float):
        """Log throughput metrics"""
        rate = count / duration if duration > 0 else 0
        self.logger.info(
            f"THROUGHPUT: {operation} processed {count} items in {duration:.3f}s ({rate:.2f}/s)",
            extra={"operation": operation, "count": count, "duration": duration, "rate": rate}
        )

def setup_logging(config_override: Dict[str, Any] = None, log_level: str = None):
    """Setup logging configuration"""
    
    # Create log directory if it doesn't exist
    log_dir = "/var/log/ai-ots"
    os.makedirs(log_dir, exist_ok=True)
    
    # Apply configuration overrides
    config = LOGGING_CONFIG.copy()
    if config_override:
        config.update(config_override)
    
    # Override log level if specified
    if log_level:
        for logger_config in config["loggers"].values():
            logger_config["level"] = log_level.upper()
    
    # Apply environment-specific settings
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "development":
        # In development, use detailed console logging
        config["loggers"][""]["handlers"] = ["console_detailed", "file_info"]
    elif environment == "production":
        # In production, use structured logging
        config["loggers"][""]["handlers"] = ["console", "file_json", "file_error", "syslog"]
    
    # Configure logging
    logging.config.dictConfig(config)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.info(f"Logging configured for environment: {environment}")

def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(f"ai_ots.{name}")

def get_access_logger() -> AccessLogger:
    """Get access logger instance"""
    return AccessLogger()

def get_audit_logger() -> AuditLogger:
    """Get audit logger instance"""
    return AuditLogger()

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance"""
    return PerformanceLogger()

# Context managers for logging
class LoggingContext:
    """Context manager for adding logging context"""
    
    def __init__(self, logger: StructuredLogger, **context):
        self.logger = logger
        self.context = context
        self.original_context = {}
    
    def __enter__(self):
        self.original_context = self.logger.context.copy()
        self.logger.set_context(**self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.context = self.original_context

class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, logger: StructuredLogger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting {self.operation}", **self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {duration:.3f}s",
                duration=duration,
                **self.kwargs
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration:.3f}s: {exc_val}",
                duration=duration,
                error=str(exc_val),
                **self.kwargs
            )

# Utility functions
def log_function_call(logger: StructuredLogger):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimingContext(logger, f"{func.__name__}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def log_exception(logger: StructuredLogger, message: str = "Exception occurred"):
    """Log exception with traceback"""
    import traceback
    logger.error(f"{message}: {traceback.format_exc()}")

# Initialize logging on import
if __name__ != "__main__":
    setup_logging()

# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="DEBUG")
    
    # Get loggers
    app_logger = get_logger("example")
    access_logger = get_access_logger()
    audit_logger = get_audit_logger()
    perf_logger = get_performance_logger()
    
    # Example logging
    app_logger.info("Application started")
    
    with LoggingContext(app_logger, user_id="test_user", session_id="test_session"):
        app_logger.info("Processing request")
        app_logger.warning("This is a warning with context")
    
    # Access logging
    access_logger.log_request({
        "remote_addr": "127.0.0.1",
        "method": "GET",
        "path": "/api/v1/health",
        "status_code": 200,
        "response_size": 1024,
        "response_time": 50,
        "user_agent": "curl/7.68.0"
    })
    
    # Audit logging
    audit_logger.log_authentication("test_user", True, "127.0.0.1")
    audit_logger.log_data_access("test_user", "/api/v1/data", "read")
    
    # Performance logging
    perf_logger.log_timing("database_query", 0.125)
    perf_logger.log_throughput("data_processing", 1000, 2.5)
    
    print("Logging examples completed. Check log files in /var/log/ai-ots/")

