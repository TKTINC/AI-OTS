"""
Prometheus Monitoring for Risk Management Service
Custom metrics collection and monitoring for risk management operations

This module provides comprehensive monitoring for:
- Risk calculation performance
- Position limit violations
- Drawdown protection events
- Stress testing results
- Alert generation and delivery
- Compliance violations
- Dashboard usage

Author: Manus AI
Version: 4.0.0
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, Summary, Info, start_http_server
import redis
import psycopg2

logger = logging.getLogger(__name__)

class RiskPrometheusMonitor:
    """Prometheus monitoring for Risk Management Service"""
    
    def __init__(self, port: int = 8007):
        """Initialize Prometheus monitoring"""
        self.port = port
        self.setup_metrics()
        
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        
        # Risk Calculation Metrics
        self.risk_calculations_total = Counter(
            'risk_calculations_total',
            'Total number of risk calculations performed',
            ['portfolio_id', 'calculation_type']
        )
        
        self.risk_calculation_duration = Histogram(
            'risk_calculation_duration_seconds',
            'Time spent calculating risk metrics',
            ['portfolio_id', 'calculation_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.portfolio_var_95 = Gauge(
            'portfolio_var_95_dollars',
            'Portfolio Value at Risk 95%',
            ['portfolio_id']
        )
        
        self.portfolio_var_99 = Gauge(
            'portfolio_var_99_dollars',
            'Portfolio Value at Risk 99%',
            ['portfolio_id']
        )
        
        self.portfolio_expected_shortfall = Gauge(
            'portfolio_expected_shortfall_dollars',
            'Portfolio Expected Shortfall',
            ['portfolio_id']
        )
        
        self.portfolio_volatility = Gauge(
            'portfolio_volatility_ratio',
            'Portfolio Volatility (annualized)',
            ['portfolio_id']
        )
        
        self.portfolio_sharpe_ratio = Gauge(
            'portfolio_sharpe_ratio',
            'Portfolio Sharpe Ratio',
            ['portfolio_id']
        )
        
        self.portfolio_max_drawdown = Gauge(
            'portfolio_max_drawdown_ratio',
            'Portfolio Maximum Drawdown',
            ['portfolio_id']
        )
        
        self.portfolio_current_drawdown = Gauge(
            'portfolio_current_drawdown_ratio',
            'Portfolio Current Drawdown',
            ['portfolio_id']
        )
        
        # Position Limit Metrics
        self.position_limit_violations_total = Counter(
            'position_limit_violations_total',
            'Total number of position limit violations',
            ['portfolio_id', 'limit_type', 'scope']
        )
        
        self.position_limit_checks_total = Counter(
            'position_limit_checks_total',
            'Total number of position limit checks',
            ['portfolio_id']
        )
        
        self.enforcement_actions_total = Counter(
            'enforcement_actions_total',
            'Total number of enforcement actions taken',
            ['portfolio_id', 'action_type', 'scope']
        )
        
        self.active_position_limits = Gauge(
            'active_position_limits',
            'Number of active position limits',
            ['portfolio_id', 'limit_type']
        )
        
        # Drawdown Protection Metrics
        self.drawdown_events_total = Counter(
            'drawdown_events_total',
            'Total number of drawdown events',
            ['portfolio_id', 'drawdown_level']
        )
        
        self.protection_actions_total = Counter(
            'protection_actions_total',
            'Total number of protection actions taken',
            ['portfolio_id', 'action_type']
        )
        
        self.drawdown_duration_days = Gauge(
            'drawdown_duration_days',
            'Current drawdown duration in days',
            ['portfolio_id']
        )
        
        # Stress Testing Metrics
        self.stress_tests_total = Counter(
            'stress_tests_total',
            'Total number of stress tests performed',
            ['portfolio_id', 'scenario_type', 'stress_level']
        )
        
        self.stress_test_duration = Histogram(
            'stress_test_duration_seconds',
            'Time spent running stress tests',
            ['scenario_type', 'stress_level'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.stress_test_portfolio_loss = Gauge(
            'stress_test_portfolio_loss_ratio',
            'Portfolio loss from latest stress test',
            ['portfolio_id', 'scenario_type', 'stress_level']
        )
        
        self.monte_carlo_simulations_total = Counter(
            'monte_carlo_simulations_total',
            'Total number of Monte Carlo simulations',
            ['portfolio_id']
        )
        
        # Alert Metrics
        self.risk_alerts_total = Counter(
            'risk_alerts_total',
            'Total number of risk alerts generated',
            ['portfolio_id', 'alert_type', 'severity']
        )
        
        self.alert_notifications_total = Counter(
            'alert_notifications_total',
            'Total number of alert notifications sent',
            ['channel', 'status']
        )
        
        self.alert_acknowledgments_total = Counter(
            'alert_acknowledgments_total',
            'Total number of alert acknowledgments',
            ['portfolio_id', 'alert_type']
        )
        
        self.active_alerts = Gauge(
            'active_alerts',
            'Number of active (unacknowledged) alerts',
            ['portfolio_id', 'severity']
        )
        
        self.alert_response_time = Histogram(
            'alert_response_time_seconds',
            'Time from alert generation to acknowledgment',
            ['alert_type', 'severity'],
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400]  # 1min to 4hrs
        )
        
        # Compliance Metrics
        self.compliance_checks_total = Counter(
            'compliance_checks_total',
            'Total number of compliance checks performed',
            ['portfolio_id', 'framework']
        )
        
        self.compliance_violations_total = Counter(
            'compliance_violations_total',
            'Total number of compliance violations',
            ['portfolio_id', 'framework', 'violation_type']
        )
        
        self.compliance_score = Gauge(
            'compliance_score_ratio',
            'Portfolio compliance score (0-1)',
            ['portfolio_id', 'framework']
        )
        
        self.audit_events_total = Counter(
            'audit_events_total',
            'Total number of audit events logged',
            ['event_type', 'compliance_relevant']
        )
        
        self.regulatory_reports_total = Counter(
            'regulatory_reports_total',
            'Total number of regulatory reports generated',
            ['framework', 'report_type']
        )
        
        # Dashboard Metrics
        self.dashboard_views_total = Counter(
            'dashboard_views_total',
            'Total number of dashboard views',
            ['dashboard_type', 'user_id']
        )
        
        self.dashboard_updates_total = Counter(
            'dashboard_updates_total',
            'Total number of dashboard updates',
            ['dashboard_id', 'update_type']
        )
        
        self.active_dashboards = Gauge(
            'active_dashboards',
            'Number of active dashboards',
            ['dashboard_type']
        )
        
        self.dashboard_update_duration = Histogram(
            'dashboard_update_duration_seconds',
            'Time spent updating dashboard data',
            ['dashboard_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # System Performance Metrics
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.database_connections = Gauge(
            'database_connections',
            'Number of active database connections'
        )
        
        self.redis_connections = Gauge(
            'redis_connections',
            'Number of active Redis connections'
        )
        
        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage_ratio = Gauge(
            'cpu_usage_ratio',
            'CPU usage ratio (0-1)'
        )
        
        # Service Health Metrics
        self.service_uptime_seconds = Gauge(
            'service_uptime_seconds',
            'Service uptime in seconds'
        )
        
        self.service_health_score = Gauge(
            'service_health_score',
            'Overall service health score (0-1)'
        )
        
        self.background_tasks_active = Gauge(
            'background_tasks_active',
            'Number of active background tasks',
            ['task_type']
        )
        
        # Business Metrics
        self.portfolios_monitored = Gauge(
            'portfolios_monitored',
            'Number of portfolios being monitored'
        )
        
        self.total_portfolio_value = Gauge(
            'total_portfolio_value_dollars',
            'Total value of all monitored portfolios'
        )
        
        self.risk_budget_utilization = Gauge(
            'risk_budget_utilization_ratio',
            'Risk budget utilization (0-1)',
            ['portfolio_id', 'risk_type']
        )
        
        # Service Information
        self.service_info = Info(
            'risk_service_info',
            'Risk Management Service information'
        )
        
        # Set service info
        self.service_info.info({
            'version': '4.0.0',
            'service': 'risk-management',
            'build_date': datetime.now().isoformat(),
            'python_version': '3.11'
        })
        
        logger.info("Prometheus metrics initialized")
    
    def start_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def record_risk_calculation(self, portfolio_id: str, calculation_type: str, duration: float):
        """Record risk calculation metrics"""
        self.risk_calculations_total.labels(
            portfolio_id=portfolio_id,
            calculation_type=calculation_type
        ).inc()
        
        self.risk_calculation_duration.labels(
            portfolio_id=portfolio_id,
            calculation_type=calculation_type
        ).observe(duration)
    
    def update_risk_metrics(self, portfolio_id: str, risk_metrics: Dict[str, float]):
        """Update risk metrics gauges"""
        if 'var_95' in risk_metrics:
            self.portfolio_var_95.labels(portfolio_id=portfolio_id).set(risk_metrics['var_95'])
        
        if 'var_99' in risk_metrics:
            self.portfolio_var_99.labels(portfolio_id=portfolio_id).set(risk_metrics['var_99'])
        
        if 'expected_shortfall' in risk_metrics:
            self.portfolio_expected_shortfall.labels(portfolio_id=portfolio_id).set(risk_metrics['expected_shortfall'])
        
        if 'volatility' in risk_metrics:
            self.portfolio_volatility.labels(portfolio_id=portfolio_id).set(risk_metrics['volatility'])
        
        if 'sharpe_ratio' in risk_metrics:
            self.portfolio_sharpe_ratio.labels(portfolio_id=portfolio_id).set(risk_metrics['sharpe_ratio'])
        
        if 'max_drawdown' in risk_metrics:
            self.portfolio_max_drawdown.labels(portfolio_id=portfolio_id).set(risk_metrics['max_drawdown'])
        
        if 'current_drawdown' in risk_metrics:
            self.portfolio_current_drawdown.labels(portfolio_id=portfolio_id).set(risk_metrics['current_drawdown'])
    
    def record_position_limit_violation(self, portfolio_id: str, limit_type: str, scope: str):
        """Record position limit violation"""
        self.position_limit_violations_total.labels(
            portfolio_id=portfolio_id,
            limit_type=limit_type,
            scope=scope
        ).inc()
    
    def record_enforcement_action(self, portfolio_id: str, action_type: str, scope: str):
        """Record enforcement action"""
        self.enforcement_actions_total.labels(
            portfolio_id=portfolio_id,
            action_type=action_type,
            scope=scope
        ).inc()
    
    def record_drawdown_event(self, portfolio_id: str, drawdown_level: str):
        """Record drawdown event"""
        self.drawdown_events_total.labels(
            portfolio_id=portfolio_id,
            drawdown_level=drawdown_level
        ).inc()
    
    def record_protection_action(self, portfolio_id: str, action_type: str):
        """Record protection action"""
        self.protection_actions_total.labels(
            portfolio_id=portfolio_id,
            action_type=action_type
        ).inc()
    
    def record_stress_test(self, portfolio_id: str, scenario_type: str, stress_level: str, 
                          duration: float, portfolio_loss: float):
        """Record stress test metrics"""
        self.stress_tests_total.labels(
            portfolio_id=portfolio_id,
            scenario_type=scenario_type,
            stress_level=stress_level
        ).inc()
        
        self.stress_test_duration.labels(
            scenario_type=scenario_type,
            stress_level=stress_level
        ).observe(duration)
        
        self.stress_test_portfolio_loss.labels(
            portfolio_id=portfolio_id,
            scenario_type=scenario_type,
            stress_level=stress_level
        ).set(portfolio_loss)
    
    def record_risk_alert(self, portfolio_id: str, alert_type: str, severity: str):
        """Record risk alert"""
        self.risk_alerts_total.labels(
            portfolio_id=portfolio_id,
            alert_type=alert_type,
            severity=severity
        ).inc()
    
    def record_alert_notification(self, channel: str, status: str):
        """Record alert notification"""
        self.alert_notifications_total.labels(
            channel=channel,
            status=status
        ).inc()
    
    def record_alert_acknowledgment(self, portfolio_id: str, alert_type: str, response_time: float):
        """Record alert acknowledgment"""
        self.alert_acknowledgments_total.labels(
            portfolio_id=portfolio_id,
            alert_type=alert_type
        ).inc()
        
        self.alert_response_time.labels(
            alert_type=alert_type,
            severity='unknown'  # Would need to be passed in
        ).observe(response_time)
    
    def record_compliance_check(self, portfolio_id: str, framework: str):
        """Record compliance check"""
        self.compliance_checks_total.labels(
            portfolio_id=portfolio_id,
            framework=framework
        ).inc()
    
    def record_compliance_violation(self, portfolio_id: str, framework: str, violation_type: str):
        """Record compliance violation"""
        self.compliance_violations_total.labels(
            portfolio_id=portfolio_id,
            framework=framework,
            violation_type=violation_type
        ).inc()
    
    def update_compliance_score(self, portfolio_id: str, framework: str, score: float):
        """Update compliance score"""
        self.compliance_score.labels(
            portfolio_id=portfolio_id,
            framework=framework
        ).set(score)
    
    def record_dashboard_view(self, dashboard_type: str, user_id: str):
        """Record dashboard view"""
        self.dashboard_views_total.labels(
            dashboard_type=dashboard_type,
            user_id=user_id
        ).inc()
    
    def record_dashboard_update(self, dashboard_id: str, update_type: str, duration: float):
        """Record dashboard update"""
        self.dashboard_updates_total.labels(
            dashboard_id=dashboard_id,
            update_type=update_type
        ).inc()
        
        self.dashboard_update_duration.labels(
            dashboard_type=update_type  # Simplified
        ).observe(duration)
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        self.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def update_system_metrics(self, db_connections: int, redis_connections: int, 
                            memory_usage: int, cpu_usage: float):
        """Update system performance metrics"""
        self.database_connections.set(db_connections)
        self.redis_connections.set(redis_connections)
        self.memory_usage_bytes.set(memory_usage)
        self.cpu_usage_ratio.set(cpu_usage)
    
    def update_business_metrics(self, portfolios_count: int, total_value: float):
        """Update business metrics"""
        self.portfolios_monitored.set(portfolios_count)
        self.total_portfolio_value.set(total_value)
    
    def update_service_health(self, uptime: float, health_score: float):
        """Update service health metrics"""
        self.service_uptime_seconds.set(uptime)
        self.service_health_score.set(health_score)

def monitor_risk_calculation(monitor: RiskPrometheusMonitor):
    """Decorator to monitor risk calculations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            portfolio_id = kwargs.get('portfolio_id', args[0] if args else 'unknown')
            calculation_type = func.__name__
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.record_risk_calculation(portfolio_id, calculation_type, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_risk_calculation(portfolio_id, f"{calculation_type}_error", duration)
                raise
        return wrapper
    return decorator

def monitor_api_request(monitor: RiskPrometheusMonitor):
    """Decorator to monitor API requests"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract request info (simplified)
                method = getattr(func, '_method', 'GET')
                endpoint = func.__name__
                status_code = 200
                
                monitor.record_api_request(method, endpoint, status_code, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                method = getattr(func, '_method', 'GET')
                endpoint = func.__name__
                status_code = 500
                
                monitor.record_api_request(method, endpoint, status_code, duration)
                raise
        return wrapper
    return decorator

# Global monitor instance
risk_monitor = RiskPrometheusMonitor()

def start_monitoring(port: int = 8007):
    """Start Prometheus monitoring server"""
    global risk_monitor
    risk_monitor = RiskPrometheusMonitor(port)
    risk_monitor.start_server()
    return risk_monitor

if __name__ == '__main__':
    # Start monitoring server for testing
    monitor = start_monitoring()
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Monitoring server stopped")

