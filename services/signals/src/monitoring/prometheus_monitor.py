"""
Prometheus Monitoring for Signal Generation Service
Custom metrics, health monitoring, and performance tracking
"""

import time
import logging
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from functools import wraps
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Define Prometheus metrics
SIGNAL_GENERATION_COUNTER = Counter(
    'signals_generated_total',
    'Total number of signals generated',
    ['symbol', 'strategy', 'signal_type']
)

SIGNAL_EXECUTION_COUNTER = Counter(
    'signals_executed_total',
    'Total number of signals executed',
    ['symbol', 'strategy', 'outcome']
)

SIGNAL_GENERATION_DURATION = Histogram(
    'signal_generation_duration_seconds',
    'Time spent generating signals',
    ['strategy']
)

SIGNAL_CONFIDENCE_HISTOGRAM = Histogram(
    'signal_confidence_distribution',
    'Distribution of signal confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

SIGNAL_SCORE_HISTOGRAM = Histogram(
    'signal_score_distribution',
    'Distribution of signal quality scores',
    buckets=[50, 60, 70, 80, 90, 100]
)

ACTIVE_SIGNALS_GAUGE = Gauge(
    'active_signals_count',
    'Number of currently active signals',
    ['symbol', 'strategy']
)

SERVICE_HEALTH_GAUGE = Gauge(
    'service_health_status',
    'Health status of integrated services',
    ['service_name']
)

API_REQUEST_COUNTER = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

API_REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'Time spent processing API requests',
    ['method', 'endpoint']
)

WEBSOCKET_CONNECTIONS_GAUGE = Gauge(
    'websocket_connections_active',
    'Number of active WebSocket connections'
)

NOTIFICATION_COUNTER = Counter(
    'notifications_sent_total',
    'Total number of notifications sent',
    ['channel', 'status']
)

CACHE_OPERATIONS_COUNTER = Counter(
    'cache_operations_total',
    'Total number of cache operations',
    ['operation', 'status']
)

CIRCUIT_BREAKER_STATE_GAUGE = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['service_name']
)

PERFORMANCE_METRICS_GAUGE = Gauge(
    'trading_performance_metrics',
    'Trading performance metrics',
    ['metric_type', 'user_id']
)

# Service info
SERVICE_INFO = Info(
    'signal_service_info',
    'Information about the signal generation service'
)

class PrometheusMonitor:
    """Prometheus monitoring for signal generation service"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.monitoring_thread = None
        self.running = False
        
        # Set service info
        SERVICE_INFO.info({
            'version': '2.0.0',
            'service': 'signal-generation',
            'build_date': datetime.now(timezone.utc).isoformat()
        })
    
    def start_monitoring(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port)
            self.running = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Error starting Prometheus metrics server: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        logger.info("Prometheus monitoring stopped")
    
    def record_signal_generation(self, signal_data: Dict[str, Any]):
        """Record signal generation metrics"""
        try:
            symbol = signal_data.get('symbol', 'unknown')
            strategy = signal_data.get('strategy_name', 'unknown')
            signal_type = signal_data.get('signal_type', 'unknown')
            confidence = signal_data.get('confidence', 0.0)
            score = signal_data.get('score', {}).get('overall_score', 0.0)
            
            # Increment generation counter
            SIGNAL_GENERATION_COUNTER.labels(
                symbol=symbol,
                strategy=strategy,
                signal_type=signal_type
            ).inc()
            
            # Record confidence distribution
            SIGNAL_CONFIDENCE_HISTOGRAM.observe(confidence)
            
            # Record score distribution
            SIGNAL_SCORE_HISTOGRAM.observe(score)
            
            logger.debug(f"Recorded signal generation metrics for {symbol}")
            
        except Exception as e:
            logger.error(f"Error recording signal generation metrics: {e}")
    
    def record_signal_execution(self, execution_data: Dict[str, Any]):
        """Record signal execution metrics"""
        try:
            symbol = execution_data.get('symbol', 'unknown')
            strategy = execution_data.get('strategy_name', 'unknown')
            outcome = execution_data.get('outcome', 'unknown')
            
            # Increment execution counter
            SIGNAL_EXECUTION_COUNTER.labels(
                symbol=symbol,
                strategy=strategy,
                outcome=outcome
            ).inc()
            
            logger.debug(f"Recorded signal execution metrics for {symbol}")
            
        except Exception as e:
            logger.error(f"Error recording signal execution metrics: {e}")
    
    def update_active_signals(self, active_signals: Dict[str, Dict[str, int]]):
        """Update active signals gauge"""
        try:
            # Clear existing gauges
            ACTIVE_SIGNALS_GAUGE.clear()
            
            # Set new values
            for symbol, strategies in active_signals.items():
                for strategy, count in strategies.items():
                    ACTIVE_SIGNALS_GAUGE.labels(
                        symbol=symbol,
                        strategy=strategy
                    ).set(count)
            
        except Exception as e:
            logger.error(f"Error updating active signals metrics: {e}")
    
    def update_service_health(self, health_status: Dict[str, Any]):
        """Update service health metrics"""
        try:
            services = health_status.get('services', {})
            
            for service_name, status in services.items():
                health_value = 1 if status.get('status') == 'healthy' else 0
                SERVICE_HEALTH_GAUGE.labels(service_name=service_name).set(health_value)
            
        except Exception as e:
            logger.error(f"Error updating service health metrics: {e}")
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        try:
            # Increment request counter
            API_REQUEST_COUNTER.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            
            # Record request duration
            API_REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Error recording API request metrics: {e}")
    
    def update_websocket_connections(self, count: int):
        """Update WebSocket connections count"""
        try:
            WEBSOCKET_CONNECTIONS_GAUGE.set(count)
        except Exception as e:
            logger.error(f"Error updating WebSocket connections metrics: {e}")
    
    def record_notification(self, channel: str, status: str):
        """Record notification metrics"""
        try:
            NOTIFICATION_COUNTER.labels(
                channel=channel,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error recording notification metrics: {e}")
    
    def record_cache_operation(self, operation: str, status: str):
        """Record cache operation metrics"""
        try:
            CACHE_OPERATIONS_COUNTER.labels(
                operation=operation,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Error recording cache operation metrics: {e}")
    
    def update_circuit_breaker_state(self, service_name: str, state: str):
        """Update circuit breaker state"""
        try:
            state_value = {"CLOSED": 0, "OPEN": 1, "HALF_OPEN": 2}.get(state, 0)
            CIRCUIT_BREAKER_STATE_GAUGE.labels(service_name=service_name).set(state_value)
        except Exception as e:
            logger.error(f"Error updating circuit breaker state metrics: {e}")
    
    def update_performance_metrics(self, user_id: str, metrics: Dict[str, float]):
        """Update trading performance metrics"""
        try:
            for metric_type, value in metrics.items():
                PERFORMANCE_METRICS_GAUGE.labels(
                    metric_type=metric_type,
                    user_id=user_id
                ).set(value)
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

# Decorator for timing functions
def monitor_duration(metric_name: str, labels: Dict[str, str] = None):
    """Decorator to monitor function duration"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                
                if metric_name == 'signal_generation':
                    strategy = labels.get('strategy', 'unknown') if labels else 'unknown'
                    SIGNAL_GENERATION_DURATION.labels(strategy=strategy).observe(duration)
                elif metric_name == 'api_request':
                    method = labels.get('method', 'unknown') if labels else 'unknown'
                    endpoint = labels.get('endpoint', 'unknown') if labels else 'unknown'
                    API_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
        return wrapper
    return decorator

# Global monitor instance
prometheus_monitor = PrometheusMonitor()

# Flask middleware for API monitoring
def monitor_flask_requests(app):
    """Add monitoring to Flask app"""
    
    @app.before_request
    def before_request():
        request.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        try:
            if hasattr(request, 'start_time'):
                duration = time.time() - request.start_time
                
                prometheus_monitor.record_api_request(
                    method=request.method,
                    endpoint=request.endpoint or 'unknown',
                    status_code=response.status_code,
                    duration=duration
                )
        except Exception as e:
            logger.error(f"Error in request monitoring: {e}")
        
        return response

# Example usage functions
def example_signal_generation_monitoring():
    """Example of how to use signal generation monitoring"""
    
    @monitor_duration('signal_generation', {'strategy': 'momentum_breakout'})
    def generate_momentum_signal():
        # Simulate signal generation
        time.sleep(0.1)
        
        signal_data = {
            'symbol': 'AAPL',
            'strategy_name': 'momentum_breakout',
            'signal_type': 'BUY_CALL',
            'confidence': 0.75,
            'score': {'overall_score': 85.0}
        }
        
        # Record metrics
        prometheus_monitor.record_signal_generation(signal_data)
        
        return signal_data
    
    return generate_momentum_signal()

def example_service_health_monitoring():
    """Example of service health monitoring"""
    health_status = {
        'services': {
            'data_ingestion': {'status': 'healthy'},
            'analytics': {'status': 'healthy'},
            'cache': {'status': 'unhealthy'}
        }
    }
    
    prometheus_monitor.update_service_health(health_status)

if __name__ == "__main__":
    # Start monitoring server
    prometheus_monitor.start_monitoring()
    
    # Example usage
    example_signal_generation_monitoring()
    example_service_health_monitoring()
    
    logger.info("Prometheus monitoring examples completed")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        prometheus_monitor.stop_monitoring()

